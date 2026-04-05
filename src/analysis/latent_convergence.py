"""Latent convergence analysis for NGE execution.

This script probes per-step processor latents and computes distance between
successive latents to test convergence after execution completes.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable, List, Tuple

import mlx.core as mx
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.common import (
    compute_forward_latents,
    iter_execution_feature_values,
    load_analysis_dataset,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
    resolve_config,
    resolve_dataset_path,
)
from src.model import NGE
from src.utils.task_specs import (
    ANALYSIS_LATENT_CHOICES,
    build_node_algo_features,
    normalize_algorithm_order,
)


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
        choices=ANALYSIS_LATENT_CHOICES,
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
        "--mode",
        type=str,
        choices=["successive", "to_final"],
        default="successive",
        help="Distance mode: successive step changes or distance to final step.",
    )
    parser.add_argument(
        "--converge-threshold",
        type=float,
        default=None,
        help="Optional distance threshold to define convergence.",
    )
    parser.add_argument(
        "--converge-patience",
        type=int,
        default=1,
        help="Number of consecutive steps below threshold to mark convergence.",
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


def compute_latent_sequence(
    model: NGE,
    graph_data: dict,
    embed_dim: int,
    latent_kind: str,
    extra_steps: int,
) -> List[mx.array]:
    num_nodes = graph_data["num_nodes"]
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])
    latents: List[mx.array] = []

    for feature_values in iter_execution_feature_values(
        graph_data, extra_steps, model.algorithms
    ):
        node_algo_features = build_node_algo_features(feature_values, model.algorithms)
        input_embeddings = mx.concatenate(
            [previous_step_hidden_states, node_algo_features], axis=1
        )

        if latent_kind == "processed":
            processed_embeddings, _, _ = compute_forward_latents(
                model, input_embeddings, graph_data["edge_matrix"]
            )
            latent = processed_embeddings
        elif latent_kind == "encoded":
            processed_embeddings, encoded, _ = compute_forward_latents(
                model, input_embeddings, graph_data["edge_matrix"]
            )
            latent = encoded
        elif latent_kind.startswith("encoded_"):
            processed_embeddings, _, encoded_by_algorithm = compute_forward_latents(
                model, input_embeddings, graph_data["edge_matrix"]
            )
            algorithm = latent_kind[len("encoded_") :]
            if algorithm not in encoded_by_algorithm:
                raise ValueError(
                    f"Latent '{latent_kind}' is unavailable for algorithms {model.algorithms}."
                )
            latent = encoded_by_algorithm[algorithm]
        elif latent_kind.startswith("processed_zero_") and latent_kind.endswith("_input"):
            algorithm = latent_kind[len("processed_zero_") : -len("_input")]
            if algorithm not in model.algorithms:
                raise ValueError(
                    f"Latent '{latent_kind}' is unavailable for algorithms {model.algorithms}."
                )
            processed_embeddings, _, _ = compute_forward_latents(
                model,
                input_embeddings,
                graph_data["edge_matrix"],
                zero_input_algorithms=(algorithm,),
            )
            latent = processed_embeddings
        else:
            raise ValueError(f"Unknown latent kind: {latent_kind}")

        latents.append(latent)
        previous_step_hidden_states = processed_embeddings

    return latents


def compute_distance_sequence(
    latents: List[mx.array],
    distance_fn: Callable,
    distance_input: str,
    mode: str,
) -> List[float]:
    if not latents:
        return []

    distances: List[float] = []

    if mode == "successive":
        previous_latent = None
        for latent in latents:
            if previous_latent is not None:
                if distance_input == "mx":
                    value = distance_fn(previous_latent, latent)
                else:
                    prev_np = np.array(previous_latent, copy=False)
                    curr_np = np.array(latent, copy=False)
                    value = distance_fn(prev_np, curr_np)
                if hasattr(value, "item"):
                    value = value.item()
                distances.append(float(value))
            previous_latent = latent
        return distances

    if mode == "to_final":
        final_latent = latents[-1]
        for latent in latents:
            if distance_input == "mx":
                value = distance_fn(latent, final_latent)
            else:
                curr_np = np.array(latent, copy=False)
                final_np = np.array(final_latent, copy=False)
                value = distance_fn(curr_np, final_np)
            if hasattr(value, "item"):
                value = value.item()
            distances.append(float(value))
        return distances

    raise ValueError(f"Unknown distance mode: {mode}")


def compute_pair_distance(
    a: mx.array,
    b: mx.array,
    distance_fn: Callable,
    distance_input: str,
) -> float:
    if distance_input == "mx":
        value = distance_fn(a, b)
    else:
        a_np = np.array(a, copy=False)
        b_np = np.array(b, copy=False)
        value = distance_fn(a_np, b_np)
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def first_convergence_step(
    distances: List[float], threshold: float, patience: int
) -> int | None:
    if threshold is None or not distances:
        return None
    if patience <= 0:
        patience = 1
    consecutive = 0
    for idx, value in enumerate(distances, start=1):
        if value <= threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= patience:
            return idx - patience + 1
    return None


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
    probe_distance: float | None = None,
) -> None:
    steps = np.arange(1, len(distances) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, distances, linewidth=2)
    if probe_distance is not None:
        probe_step = len(distances) + 1
        if len(distances) > 0:
            ax.plot(
                [len(distances), probe_step],
                [distances[-1], probe_distance],
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                color="#d62728",
            )
        ax.scatter(
            [probe_step],
            [probe_distance],
            marker="X",
            s=70,
            color="#d62728",
            edgecolors="black",
            linewidths=0.5,
            label="terminal probe",
        )
        ax.legend(loc="best")
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
    probe_mean: float | None = None,
    probe_std: float | None = None,
) -> None:
    steps = np.arange(1, len(mean) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mean, linewidth=2, label="mean")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25, label="std")
    if probe_mean is not None:
        probe_step = len(mean) + 1
        if len(mean) > 0:
            ax.plot(
                [len(mean), probe_step],
                [mean[-1], probe_mean],
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                color="#d62728",
            )
        yerr = probe_std if probe_std is not None else 0.0
        ax.errorbar(
            [probe_step],
            [probe_mean],
            yerr=[yerr],
            fmt="X",
            markersize=8,
            color="#d62728",
            ecolor="#d62728",
            elinewidth=1.0,
            capsize=3,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="terminal probe",
        )
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
    config, run_dir = resolve_config(args.config, args.run_dir)

    algorithm_order = normalize_algorithm_order(config.model.algorithms)
    dataset_path = resolve_dataset_path(
        args.dataset, args.split, config, algorithm_order=algorithm_order
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, run_dir)
    model, step = load_model_from_checkpoint(config, checkpoint_path, run_dir)
    model.eval()

    dataset = load_analysis_dataset(dataset_path, algorithm_order)

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

    distance_series = []
    convergence_steps: List[int | None] = []
    terminal_probe_distances: List[float | None] = []
    for graph in graphs:
        latents = compute_latent_sequence(
            model=model,
            graph_data=graph,
            embed_dim=config.model.embed_dim,
            latent_kind=args.latent,
            extra_steps=args.extra_steps,
        )
        distances = compute_distance_sequence(
            latents=latents,
            distance_fn=distance_fn,
            distance_input=distance_input,
            mode=args.mode,
        )
        distance_series.append(distances)
        convergence_steps.append(
            first_convergence_step(
                distances, args.converge_threshold, args.converge_patience
            )
        )
        probe_latents = compute_latent_sequence(
            model=model,
            graph_data=graph,
            embed_dim=config.model.embed_dim,
            latent_kind=args.latent,
            extra_steps=args.extra_steps + 1,
        )
        probe_distance = None
        if latents and len(probe_latents) > len(latents):
            probe_distance = compute_pair_distance(
                latents[-1], probe_latents[-1], distance_fn, distance_input
            )
        terminal_probe_distances.append(probe_distance)

    metadata = {
        "config_name": config.name,
        "checkpoint_step": step,
        "latent": args.latent,
        "distance": distance_label,
        "distance_input": args.distance_input,
        "mode": args.mode,
        "extra_steps": args.extra_steps,
        "converge_threshold": args.converge_threshold,
        "converge_patience": args.converge_patience,
        "dataset": str(dataset_path),
        "split": args.split if args.dataset is None else None,
        "num_graphs": len(graphs),
        "terminal_probe_definition": "distance from final analyzed latent to one extra terminal-target forward pass",
    }

    if args.graph_index is not None:
        distances = distance_series[0]
        title = args.title or f"Latent distance over time (graph {args.graph_index})"
        plot_path = output_dir / f"graph_{args.graph_index}_distance.png"
        plot_single_series(
            distances,
            plot_path,
            title=title,
            y_label=f"distance ({distance_label})",
            probe_distance=terminal_probe_distances[0],
        )
        write_json(
            output_dir / f"graph_{args.graph_index}_distances.json",
            {
                "distances": distances,
                "terminal_probe_distance": terminal_probe_distances[0],
                "convergence_step": convergence_steps[0],
                "metadata": metadata,
            },
        )
    else:
        mean, std, counts = aggregate_distance_series(distance_series)
        probe_values = [x for x in terminal_probe_distances if x is not None]
        probe_mean = float(np.mean(probe_values)) if probe_values else None
        probe_std = float(np.std(probe_values)) if probe_values else None
        title = args.title or "Latent distance over time (dataset)"
        plot_path = output_dir / "dataset_distance.png"
        plot_dataset_stats(
            mean,
            std,
            plot_path,
            title=title,
            y_label=f"distance ({distance_label})",
            probe_mean=probe_mean,
            probe_std=probe_std,
        )
        write_json(
            output_dir / "dataset_stats.json",
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "counts": counts.tolist(),
                "terminal_probe_distances": terminal_probe_distances,
                "terminal_probe_mean": probe_mean,
                "terminal_probe_std": probe_std,
                "convergence_steps": convergence_steps,
                "metadata": metadata,
            },
        )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
