"""Extract embedding trajectories and PCA views for NGE executions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

from src.analysis.common import (
    compute_encoded_embeddings,
    iter_execution_inputs,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
    resolve_config,
    resolve_dataset_path,
)
from src.data import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract latent embedding trajectories and compute PCA views."
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
        "--max-graphs",
        type=int,
        default=None,
        help="Limit number of graphs before step filtering (uses first N).",
    )
    parser.add_argument(
        "--latent",
        type=str,
        choices=["processed", "encoded"],
        default="processed",
        help="Which latent representation to collect.",
    )
    parser.add_argument(
        "--node-agg",
        type=str,
        choices=["max", "min", "mean"],
        default="max",
        help="Aggregation over node dimension to form per-step embeddings.",
    )
    parser.add_argument(
        "--step-policy",
        type=str,
        choices=["common", "fixed", "min", "max"],
        default="common",
        help="How to choose a shared execution length across graphs.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Fixed step count to use when --step-policy=fixed.",
    )
    parser.add_argument(
        "--extra-steps",
        type=int,
        default=0,
        help="Extra steps to run after termination using final algorithm state.",
    )
    parser.add_argument(
        "--pca",
        type=str,
        choices=["none", "step", "trajectory", "both"],
        default="both",
        help="Which PCA views to compute.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=2,
        help="Number of PCA components to retain.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save 2D scatter plots for PCA results (first two components).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs (npz + json + plots).",
    )
    return parser.parse_args()


def count_execution_steps(graph_data: dict, extra_steps: int) -> int:
    num_bf_steps = len(graph_data["bf_distance_targets"])
    num_bfs_steps = len(graph_data["bfs_state_targets"])
    base_steps = max(num_bf_steps, num_bfs_steps) - 1
    if base_steps < 0:
        base_steps = 0
    return base_steps + max(extra_steps, 0)


def choose_step_count(step_counts: List[int], policy: str, fixed_steps: int | None) -> int:
    if not step_counts:
        raise ValueError("No step counts available to choose from.")

    if policy == "fixed":
        if fixed_steps is None:
            raise ValueError("--steps is required when --step-policy=fixed.")
        return fixed_steps
    if policy == "min":
        return min(step_counts)
    if policy == "max":
        return max(step_counts)
    if policy == "common":
        counts = Counter(step_counts)
        most_common = counts.most_common()
        top_freq = most_common[0][1]
        candidates = [steps for steps, freq in most_common if freq == top_freq]
        return max(candidates)

    raise ValueError(f"Unknown step policy: {policy}")


def aggregate_nodes(latent: np.ndarray, node_agg: str) -> np.ndarray:
    if node_agg == "max":
        return latent.max(axis=0)
    if node_agg == "min":
        return latent.min(axis=0)
    if node_agg == "mean":
        return latent.mean(axis=0)
    raise ValueError(f"Unknown node aggregation: {node_agg}")


def collect_graph_trajectory(
    model,
    graph_data: dict,
    embed_dim: int,
    latent_kind: str,
    node_agg: str,
    extra_steps: int,
) -> np.ndarray:
    num_nodes = graph_data["num_nodes"]
    previous_step_hidden_states = mx.zeros([num_nodes, 2 * embed_dim])
    step_embeddings: List[np.ndarray] = []

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

        latent_np = np.array(latent, copy=False)
        reduced = aggregate_nodes(latent_np, node_agg)
        step_embeddings.append(reduced.astype(np.float32, copy=False))
        previous_step_hidden_states = processed_embeddings

    if not step_embeddings:
        return np.empty((0, 2 * embed_dim), dtype=np.float32)

    return np.stack(step_embeddings, axis=0)


def build_stepwise_indices(
    selected_graph_indices: List[int], num_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    graph_indices = np.repeat(np.array(selected_graph_indices, dtype=np.int32), num_steps)
    step_indices = np.tile(np.arange(num_steps, dtype=np.int32), len(selected_graph_indices))
    return graph_indices, step_indices


def pca_fit_transform(
    data: np.ndarray, n_components: int
) -> Tuple[Dict[str, np.ndarray], int]:
    if data.ndim != 2:
        raise ValueError("PCA expects a 2D array.")
    num_samples, num_features = data.shape
    if num_samples == 0 or num_features == 0:
        raise ValueError("Empty data passed to PCA.")

    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    max_components = min(Vt.shape[0], n_components)

    components = Vt[:max_components]
    projected = centered @ components.T

    if num_samples > 1:
        explained_variance = (S * S) / (num_samples - 1)
    else:
        explained_variance = S * S

    total_variance = explained_variance.sum()
    if total_variance > 0:
        explained_ratio = explained_variance[:max_components] / total_variance
    else:
        explained_ratio = np.zeros(max_components, dtype=np.float64)

    payload = {
        "projected": projected.astype(np.float32, copy=False),
        "components": components.astype(np.float32, copy=False),
        "mean": mean.squeeze(axis=0).astype(np.float32, copy=False),
        "explained_variance": explained_variance[:max_components].astype(
            np.float32, copy=False
        ),
        "explained_variance_ratio": explained_ratio.astype(np.float32, copy=False),
    }
    return payload, max_components


def plot_scatter(
    points: np.ndarray, colors: np.ndarray, output_path: Path, title: str, color_label: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=12,
        cmap="viridis",
        alpha=0.8,
        linewidths=0.0,
    )
    fig.colorbar(scatter, ax=ax, label=color_label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
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

    dataset_path = resolve_dataset_path(args.dataset, args.split, config)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, run_dir)
    model, step = load_model_from_checkpoint(config, checkpoint_path, run_dir)
    model.eval()

    dataset = load_dataset(dataset_path)
    graphs = dataset
    if args.max_graphs is not None:
        graphs = dataset[: args.max_graphs]

    step_counts = [count_execution_steps(graph, args.extra_steps) for graph in graphs]
    target_steps = choose_step_count(step_counts, args.step_policy, args.steps)
    selected_indices = [
        index for index, steps in enumerate(step_counts) if steps == target_steps
    ]

    if not selected_indices:
        raise ValueError(
            f"No graphs matched step count {target_steps} under policy {args.step_policy}."
        )

    selected_graphs = [graphs[i] for i in selected_indices]

    trajectories = [
        collect_graph_trajectory(
            model=model,
            graph_data=graph,
            embed_dim=config.model.embed_dim,
            latent_kind=args.latent,
            node_agg=args.node_agg,
            extra_steps=args.extra_steps,
        )
        for graph in selected_graphs
    ]

    trajectories_tensor = np.stack(trajectories, axis=0)
    num_graphs, num_steps, latent_dim = trajectories_tensor.shape

    trajectory_matrix = trajectories_tensor.reshape(num_graphs, num_steps * latent_dim)
    step_matrix = trajectories_tensor.reshape(num_graphs * num_steps, latent_dim)
    graph_indices, step_indices = build_stepwise_indices(selected_indices, num_steps)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            run_dir / "analysis" / "embedding_trajectories"
            if run_dir
            else Path("analysis/embedding_trajectories")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "trajectories.npz",
        trajectories=trajectories_tensor.astype(np.float32, copy=False),
        trajectory_matrix=trajectory_matrix.astype(np.float32, copy=False),
        step_matrix=step_matrix.astype(np.float32, copy=False),
        graph_indices=graph_indices,
        step_indices=step_indices,
        selected_graph_indices=np.array(selected_indices, dtype=np.int32),
    )

    metadata = {
        "config_name": config.name,
        "checkpoint_step": step,
        "latent": args.latent,
        "node_agg": args.node_agg,
        "step_policy": args.step_policy,
        "target_steps": target_steps,
        "extra_steps": args.extra_steps,
        "latent_dim": latent_dim,
        "dataset": str(dataset_path),
        "split": args.split if args.dataset is None else None,
        "num_graphs": num_graphs,
        "selected_graph_indices": selected_indices,
        "trajectory_shape": list(trajectories_tensor.shape),
        "trajectory_matrix_shape": list(trajectory_matrix.shape),
        "step_matrix_shape": list(step_matrix.shape),
    }
    write_json(output_dir / "metadata.json", metadata)

    if args.pca != "none":
        if args.pca in ("trajectory", "both"):
            payload, used_components = pca_fit_transform(
                trajectory_matrix, args.pca_components
            )
            np.savez(output_dir / "pca_trajectory.npz", **payload)
            if args.plot and used_components >= 2:
                plot_scatter(
                    payload["projected"],
                    np.array(selected_indices, dtype=np.int32),
                    output_dir / "pca_trajectory.png",
                    "Trajectory-wise PCA",
                    "graph",
                )

        if args.pca in ("step", "both"):
            payload, used_components = pca_fit_transform(step_matrix, args.pca_components)
            np.savez(output_dir / "pca_step.npz", **payload)
            if args.plot and used_components >= 2:
                plot_scatter(
                    payload["projected"],
                    step_indices,
                    output_dir / "pca_step.png",
                    "Step-wise PCA",
                    "step",
                )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
