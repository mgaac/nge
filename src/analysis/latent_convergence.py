"""Latent convergence analysis for NGE execution.

This script probes per-step processor latents and computes distance between
successive latents to test convergence after execution completes.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import mlx.core as mx
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_dataset
from src.model import AggregationFn, NGE
from src.utils import CheckpointManager, ExperimentConfig, load_config, validate_config


DistanceFn = Callable[[np.ndarray, np.ndarray], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze latent convergence by measuring distances between successive latents."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory with config_resolved.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory or file to load (defaults to latest in run-dir).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to use when --dataset is not provided.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path (npz).",
    )
    parser.add_argument(
        "--graph-index",
        type=int,
        default=None,
        help="Graph index for single-graph analysis.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Limit number of graphs for dataset statistics (uses first N).",
    )
    parser.add_argument(
        "--extra-steps",
        type=int,
        default=0,
        help="Extra steps to run after termination using final algorithm state.",
    )
    parser.add_argument(
        "--latent",
        type=str,
        choices=["processed", "encoded"],
        default="processed",
        help="Which latent representation to probe.",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="l2",
        choices=["l2", "l1", "mse", "cosine", "mean_l2"],
        help="Built-in distance metric.",
    )
    parser.add_argument(
        "--distance-fn",
        type=str,
        default=None,
        help="Custom distance function as module:function (overrides --distance).",
    )
    parser.add_argument(
        "--distance-input",
        type=str,
        choices=["numpy", "mx"],
        default="numpy",
        help="Input type passed to custom distance function.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs (plots + JSON).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title override.",
    )
    return parser.parse_args()


def create_model(config: ExperimentConfig) -> NGE:
    agg_fn_map = {
        "SUM": AggregationFn.SUM,
        "AVG": AggregationFn.AVG,
        "MIN": AggregationFn.MIN,
        "MAX": AggregationFn.MAX,
    }
    agg_fn = agg_fn_map[config.model.agg_fn]
    return NGE(
        embed_dim=config.model.embed_dim,
        residual_connections=config.model.residual_connections,
        agg_fn=agg_fn,
        num_mp_layers=config.model.num_mp_layers,
        dropout=config.model.dropout,
    )


def resolve_config(args: argparse.Namespace) -> Tuple[ExperimentConfig, Path | None]:
    run_dir = Path(args.run_dir) if args.run_dir else None
    config_path = None
    if run_dir is not None:
        config_path = run_dir / "config_resolved.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config_resolved.yaml in run dir: {run_dir}")
    elif args.config:
        config_path = Path(args.config)
    else:
        raise ValueError("Provide --config or --run-dir.")

    config = load_config(config_path)
    validate_config(config)
    return config, run_dir


def resolve_dataset_path(args: argparse.Namespace, config: ExperimentConfig) -> Path:
    if args.dataset:
        return Path(args.dataset)

    if args.split == "train":
        return Path(config.data.train_path)
    if args.split == "val":
        return Path(config.data.val_path)
    return Path(config.data.test_path)


def resolve_checkpoint_path(args: argparse.Namespace, run_dir: Path | None) -> Path | None:
    if args.checkpoint is None and run_dir is None:
        return None

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.is_file():
            ckpt_path = ckpt_path.parent
        return ckpt_path

    if run_dir is not None:
        return run_dir / "checkpoints"

    return None


def load_model_from_checkpoint(
    config: ExperimentConfig, checkpoint_path: Path | None, run_dir: Path | None
) -> Tuple[NGE, int | None]:
    model = create_model(config)
    model.eval()

    if checkpoint_path is None:
        return model, None

    if checkpoint_path.name == "checkpoints":
        manager = CheckpointManager(checkpoint_path)
        model, _, step = manager.load(model, optimizer=None, checkpoint_path=None)
        return model, step

    checkpoint_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    manager = CheckpointManager(checkpoint_dir)
    model, _, step = manager.load(model, optimizer=None, checkpoint_path=checkpoint_path)
    return model, step


def built_in_distances() -> dict[str, DistanceFn]:
    def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.sqrt(np.sum(diff * diff)))

    def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(np.abs(a - b)))

    def mse_distance(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.mean(diff * diff))

    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-8
        if denom == 0.0:
            return 0.0
        cosine_sim = float(np.dot(a_flat, b_flat) / denom)
        return 1.0 - cosine_sim

    def mean_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        if diff.ndim == 1:
            return float(np.linalg.norm(diff))
        per_node = np.linalg.norm(diff, axis=1)
        return float(np.mean(per_node))

    return {
        "l2": l2_distance,
        "l1": l1_distance,
        "mse": mse_distance,
        "cosine": cosine_distance,
        "mean_l2": mean_l2_distance,
    }


def load_custom_distance_fn(path: str) -> DistanceFn:
    if ":" not in path:
        raise ValueError("Custom distance must be in module:function format.")
    module_path, fn_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise ValueError(f"Custom distance {path} is not callable.")
    return fn


def iter_execution_inputs(
    graph_data: dict, extra_steps: int
) -> Iterable[Tuple[mx.array, mx.array]]:
    bf_steps = graph_data["bf_distance_targets"]
    bfs_steps = graph_data["bfs_state_targets"]
    num_bf_steps = len(bf_steps)
    num_bfs_steps = len(bfs_steps)
    num_steps = max(num_bf_steps, num_bfs_steps)

    for i in range(num_steps):
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        if not (bf_sample_exists or bfs_sample_exists):
            continue
        true_bfs_state = bfs_steps[i] if bfs_sample_exists else bfs_steps[-1]
        true_distance_bf = bf_steps[i] if bf_sample_exists else bf_steps[-1]
        yield true_bfs_state, true_distance_bf

    if extra_steps > 0:
        final_bfs = bfs_steps[-1]
        final_bf = bf_steps[-1]
        for _ in range(extra_steps):
            yield final_bfs, final_bf


def compute_encoded_embeddings(model: NGE, input_embeddings):
    bfs_encoded = model.bfs_encoder(input_embeddings)
    bf_encoded = model.bf_encoder(input_embeddings)
    encoded = mx.concatenate([bfs_encoded, bf_encoded], axis=1)
    return model.ln(encoded)


def compute_distance_sequence(
    model: NGE,
    graph_data: dict,
    embed_dim: int,
    latent_kind: str,
    distance_fn: Callable,
    distance_input: str,
    extra_steps: int,
) -> List[float]:
    num_nodes = graph_data["num_nodes"]
    previous_step_hidden_states = mx.zeros([num_nodes, 2 * embed_dim])
    distances: List[float] = []
    previous_latent = None

    for true_bfs_state, true_distance_bf in iter_execution_inputs(graph_data, extra_steps):
        node_algo_features = mx.stack([true_bfs_state, true_distance_bf], axis=1)
        input_embeddings = mx.concatenate(
            [previous_step_hidden_states, node_algo_features], axis=1
        )

        _, _, _, processed_embeddings = model((input_embeddings, graph_data["edge_matrix"]))

        if latent_kind == "processed":
            latent = processed_embeddings
        elif latent_kind == "encoded":
            latent = compute_encoded_embeddings(model, input_embeddings)
        else:
            raise ValueError(f"Unknown latent kind: {latent_kind}")

        if previous_latent is not None:
            if distance_input == "mx":
                value = distance_fn(previous_latent, latent)
            else:
                prev_np = np.array(previous_latent, copy=False)
                curr_np = np.array(latent, copy=False)
                value = distance_fn(prev_np, curr_np)
            if hasattr(value, "item"):
                value = value.item()
            value = float(value)
            distances.append(value)

        previous_latent = latent
        previous_step_hidden_states = processed_embeddings

    return distances


def aggregate_distance_series(series_list: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not series_list:
        return np.array([]), np.array([]), np.array([])

    max_len = max(len(series) for series in series_list)
    sums = np.zeros(max_len, dtype=np.float64)
    sums_sq = np.zeros(max_len, dtype=np.float64)
    counts = np.zeros(max_len, dtype=np.float64)

    for series in series_list:
        if not series:
            continue
        arr = np.array(series, dtype=np.float64)
        length = len(arr)
        sums[:length] += arr
        sums_sq[:length] += arr * arr
        counts[:length] += 1

    mean = sums / np.maximum(counts, 1.0)
    var = (sums_sq / np.maximum(counts, 1.0)) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std, counts


def plot_single_series(
    distances: List[float],
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    steps = np.arange(1, len(distances) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, distances, linewidth=2)
    ax.set_xlabel("Execution step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_stats(
    mean: np.ndarray,
    std: np.ndarray,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    steps = np.arange(1, len(mean) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mean, linewidth=2, label="mean")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25, label="std")
    ax.set_xlabel("Execution step")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    config, run_dir = resolve_config(args)

    dataset_path = resolve_dataset_path(args, config)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    checkpoint_path = resolve_checkpoint_path(args, run_dir)
    model, step = load_model_from_checkpoint(config, checkpoint_path, run_dir)
    model.eval()

    dataset = load_dataset(dataset_path)

    if args.graph_index is not None:
        if args.graph_index < 0 or args.graph_index >= len(dataset):
            raise IndexError(
                f"graph-index {args.graph_index} out of range (0..{len(dataset) - 1})"
            )
        graphs = [dataset[args.graph_index]]
    else:
        graphs = dataset
        if args.max_graphs is not None:
            graphs = dataset[: args.max_graphs]

    if args.distance_fn:
        distance_fn = load_custom_distance_fn(args.distance_fn)
        distance_label = args.distance_fn
        distance_input = args.distance_input
    else:
        distance_fn = built_in_distances()[args.distance]
        distance_label = args.distance
        distance_input = "numpy"

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (run_dir / "analysis" / "latent_convergence" if run_dir else Path("analysis/latent_convergence"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    distance_series = [
        compute_distance_sequence(
            model=model,
            graph_data=graph,
            embed_dim=config.model.embed_dim,
            latent_kind=args.latent,
            distance_fn=distance_fn,
            distance_input=distance_input,
            extra_steps=args.extra_steps,
        )
        for graph in graphs
    ]

    metadata = {
        "config_name": config.name,
        "checkpoint_step": step,
        "latent": args.latent,
        "distance": distance_label,
        "distance_input": args.distance_input,
        "extra_steps": args.extra_steps,
        "dataset": str(dataset_path),
        "split": args.split if args.dataset is None else None,
        "num_graphs": len(graphs),
    }

    if args.graph_index is not None:
        distances = distance_series[0]
        title = args.title or f"Latent distance over time (graph {args.graph_index})"
        plot_path = output_dir / f"graph_{args.graph_index}_distance.png"
        plot_single_series(
            distances, plot_path, title=title, y_label=f"distance ({distance_label})"
        )
        write_json(
            output_dir / f"graph_{args.graph_index}_distances.json",
            {"distances": distances, "metadata": metadata},
        )
    else:
        mean, std, counts = aggregate_distance_series(distance_series)
        title = args.title or "Latent distance over time (dataset)"
        plot_path = output_dir / "dataset_distance.png"
        plot_dataset_stats(
            mean, std, plot_path, title=title, y_label=f"distance ({distance_label})"
        )
        write_json(
            output_dir / "dataset_stats.json",
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "counts": counts.tolist(),
                "metadata": metadata,
            },
        )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
