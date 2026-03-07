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
    compute_forward_latents,
    iter_execution_inputs,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
    resolve_config,
    resolve_dataset_path,
)
from src.data import load_dataset
from src.utils.task_specs import ANALYSIS_LATENT_CHOICES, build_node_algo_features


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
        "--graph-index",
        type=int,
        default=None,
        help="Analyze a single graph by index (bypasses step-policy filtering).",
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
        choices=ANALYSIS_LATENT_CHOICES,
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
        help=(
            "Fake-continue execution for N additional steps after algorithm "
            "termination by reusing the final algorithm states as inputs."
        ),
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
    num_prim_steps = len(graph_data["prim_key_targets"])
    base_steps = max(num_bf_steps, num_bfs_steps, num_prim_steps) - 1
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
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])
    step_embeddings: List[np.ndarray] = []

    for (
        true_bfs_state,
        true_distance_bf,
        true_prim_state,
        true_prim_key,
    ) in iter_execution_inputs(graph_data, extra_steps):
        node_algo_features = build_node_algo_features(
            true_bfs_state,
            true_distance_bf,
            true_prim_state,
            true_prim_key,
        )
        input_embeddings = mx.concatenate(
            [previous_step_hidden_states, node_algo_features], axis=1
        )
        if latent_kind == "processed":
            processed_embeddings, _, _, _, _ = compute_forward_latents(
                model, input_embeddings, graph_data["edge_matrix"]
            )
            latent = processed_embeddings
        elif latent_kind == "encoded":
            processed_embeddings, encoded, _, _, _ = compute_forward_latents(
                model, input_embeddings, graph_data["edge_matrix"]
            )
            latent = encoded
        elif latent_kind == "encoded_bfs":
            processed_embeddings, _, bfs_encoded, _, _ = compute_forward_latents(
                model, input_embeddings, graph_data["edge_matrix"]
            )
            latent = bfs_encoded
        elif latent_kind == "encoded_bf":
            processed_embeddings, _, _, bf_encoded, _ = compute_forward_latents(
                model, input_embeddings, graph_data["edge_matrix"]
            )
            latent = bf_encoded
        elif latent_kind == "encoded_prim":
            processed_embeddings, _, _, _, prim_encoded = compute_forward_latents(
                model, input_embeddings, graph_data["edge_matrix"]
            )
            latent = prim_encoded
        elif latent_kind == "processed_zero_bfs_input":
            processed_embeddings, _, _, _, _ = compute_forward_latents(
                model,
                input_embeddings,
                graph_data["edge_matrix"],
                zero_bfs_input=True,
                zero_bf_input=False,
                zero_prim_input=False,
            )
            latent = processed_embeddings
        elif latent_kind == "processed_zero_bf_input":
            processed_embeddings, _, _, _, _ = compute_forward_latents(
                model,
                input_embeddings,
                graph_data["edge_matrix"],
                zero_bfs_input=False,
                zero_bf_input=True,
                zero_prim_input=False,
            )
            latent = processed_embeddings
        elif latent_kind == "processed_zero_prim_input":
            processed_embeddings, _, _, _, _ = compute_forward_latents(
                model,
                input_embeddings,
                graph_data["edge_matrix"],
                zero_bfs_input=False,
                zero_bf_input=False,
                zero_prim_input=True,
            )
            latent = processed_embeddings
        else:
            raise ValueError(f"Unknown latent kind: {latent_kind}")

        latent_np = np.array(latent, copy=False)
        reduced = aggregate_nodes(latent_np, node_agg)
        step_embeddings.append(reduced.astype(np.float32, copy=False))
        previous_step_hidden_states = processed_embeddings

    if not step_embeddings:
        return np.empty((0, model.processor_embed_dim), dtype=np.float32)

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


def pca_project(data: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    centered = data - mean.reshape(1, -1)
    return centered @ components.T


def step_pca_mean_coordinates(
    projected: np.ndarray, step_indices: np.ndarray, num_steps: int
) -> List[dict]:
    means: List[dict] = []
    for step in range(num_steps):
        mask = step_indices == step
        if not np.any(mask):
            continue
        mean_coord = projected[mask].mean(axis=0)
        means.append(
            {
                "step": int(step),
                "mean_coordinate": mean_coord.astype(np.float64, copy=False).tolist(),
            }
        )
    return means


def _format_title_with_variance(title: str, explained_ratio: np.ndarray) -> str:
    if explained_ratio.size >= 2:
        pc1 = explained_ratio[0] * 100.0
        pc2 = explained_ratio[1] * 100.0
        return f"{title} (PC1 {pc1:.1f}%, PC2 {pc2:.1f}%)"
    return title


def _build_discrete_color_map(labels: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    import matplotlib

    unique_labels = np.unique(labels)
    base = np.array(matplotlib.colormaps["tab20"](np.linspace(0.0, 1.0, 20)))
    color_map: dict[int, np.ndarray] = {}
    for idx, label in enumerate(unique_labels):
        # Reuse a qualitative palette first, then fall back to a sampled hue palette.
        if idx < len(base):
            color = np.array(base[idx], dtype=np.float32)
        else:
            frac = (idx - len(base)) / max(len(unique_labels) - len(base), 1)
            color = np.array(matplotlib.colormaps["hsv"](frac), dtype=np.float32)
        color_map[int(label)] = color
    return unique_labels, color_map


def plot_trajectory_scatter(
    points: np.ndarray,
    graph_labels: np.ndarray,
    output_path: Path,
    title: str,
    explained_ratio: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    unique_graphs = np.unique(graph_labels)
    norm = plt.Normalize(vmin=float(unique_graphs.min()), vmax=float(unique_graphs.max()))
    cmap = matplotlib.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=graph_labels,
        cmap=cmap,
        norm=norm,
        s=30,
        alpha=0.9,
        linewidths=0.2,
        edgecolors="black",
    )
    cbar = fig.colorbar(scatter, ax=ax, label="graph index")
    cbar.ax.tick_params(labelsize=8)

    mid_graph = unique_graphs[len(unique_graphs) // 2]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap(norm(float(unique_graphs[0]))),
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=6,
            label=f"graph {int(unique_graphs[0])}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap(norm(float(mid_graph))),
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=6,
            label=f"graph {int(mid_graph)}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap(norm(float(unique_graphs[-1]))),
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=6,
            label=f"graph {int(unique_graphs[-1])}",
        ),
    ]
    ax.legend(handles=legend_handles, title="Gradient anchors", loc="best", fontsize=8)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(_format_title_with_variance(title, explained_ratio))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_step_trajectories(
    points: np.ndarray,
    graph_labels: np.ndarray,
    step_indices: np.ndarray,
    output_path: Path,
    title: str,
    explained_ratio: np.ndarray,
    completion_probe_points: np.ndarray | None = None,
    completion_probe_mean: np.ndarray | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    unique_steps, step_color_map = _build_discrete_color_map(step_indices)

    fig, ax = plt.subplots(figsize=(7, 6))
    for graph_label in np.unique(graph_labels):
        graph_mask = graph_labels == graph_label
        graph_points = points[graph_mask]
        graph_steps = step_indices[graph_mask]
        order = np.argsort(graph_steps)
        graph_points = graph_points[order]
        graph_steps = graph_steps[order]

        # Keep trajectory lines subtle so step colors remain visually dominant.
        ax.plot(
            graph_points[:, 0],
            graph_points[:, 1],
            color="#5f6368",
            linewidth=0.45,
            alpha=0.14,
            zorder=1,
        )
        point_colors = np.array([step_color_map[int(step)] for step in graph_steps])
        ax.scatter(
            graph_points[:, 0],
            graph_points[:, 1],
            c=point_colors,
            s=22,
            alpha=0.9,
            linewidths=0.2,
            edgecolors="black",
            zorder=2,
        )

    if completion_probe_points is not None and completion_probe_points.size > 0:
        ax.scatter(
            completion_probe_points[:, 0],
            completion_probe_points[:, 1],
            marker="X",
            s=46,
            color="#d62728",
            alpha=0.8,
            linewidths=0.4,
            edgecolors="black",
            zorder=3,
        )
    if completion_probe_mean is not None and completion_probe_mean.size >= 2:
        ax.scatter(
            [completion_probe_mean[0]],
            [completion_probe_mean[1]],
            marker="*",
            s=170,
            color="#d62728",
            alpha=0.95,
            linewidths=0.6,
            edgecolors="black",
            zorder=4,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=step_color_map[int(step)],
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=5,
            label=f"step {int(step)}",
        )
        for step in unique_steps
    ]
    if completion_probe_points is not None and completion_probe_points.size > 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color="none",
                markerfacecolor="#d62728",
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=6,
                label="terminal probe",
            )
        )
    if completion_probe_mean is not None and completion_probe_mean.size >= 2:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#d62728",
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=8,
                label="terminal probe mean",
            )
        )
    if legend_handles:
        ncols = 1 if len(legend_handles) <= 12 else 2 if len(legend_handles) <= 24 else 3
        ax.legend(
            handles=legend_handles,
            title="Execution step",
            loc="best",
            fontsize=7,
            ncol=ncols,
            framealpha=0.9,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(_format_title_with_variance(title, explained_ratio))
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
    if args.extra_steps < 0:
        raise ValueError("--extra-steps must be non-negative.")
    if args.pca_components <= 0:
        raise ValueError("--pca-components must be positive.")
    config, run_dir = resolve_config(args.config, args.run_dir)

    dataset_path = resolve_dataset_path(args.dataset, args.split, config)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, run_dir)
    model, step = load_model_from_checkpoint(config, checkpoint_path, run_dir)
    model.eval()

    dataset = load_dataset(dataset_path)
    if args.graph_index is not None:
        if args.graph_index < 0 or args.graph_index >= len(dataset):
            raise IndexError(
                f"--graph-index out of range: {args.graph_index} (dataset size={len(dataset)})"
            )
        selected_indices = [args.graph_index]
        selected_graphs = [dataset[args.graph_index]]
        target_steps = count_execution_steps(selected_graphs[0], args.extra_steps)
    else:
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
    pca_fit_excludes_extra_steps = args.extra_steps > 0 and num_steps > args.extra_steps
    pca_fit_steps = num_steps - args.extra_steps if pca_fit_excludes_extra_steps else num_steps
    pca_fit_tensor = trajectories_tensor[:, :pca_fit_steps, :]

    completion_probe_vectors: List[np.ndarray] = []
    completion_probe_graph_indices: List[int] = []
    for graph_index, graph in zip(selected_indices, selected_graphs):
        probe_trajectory = collect_graph_trajectory(
            model=model,
            graph_data=graph,
            embed_dim=config.model.embed_dim,
            latent_kind=args.latent,
            node_agg=args.node_agg,
            extra_steps=args.extra_steps + 1,
        )
        if probe_trajectory.shape[0] == 0:
            continue
        completion_probe_vectors.append(probe_trajectory[-1].astype(np.float32, copy=False))
        completion_probe_graph_indices.append(int(graph_index))

    if completion_probe_vectors:
        completion_probes = np.stack(completion_probe_vectors, axis=0).astype(
            np.float32, copy=False
        )
        completion_probe_graph_indices_arr = np.array(
            completion_probe_graph_indices, dtype=np.int32
        )
    else:
        completion_probes = np.empty((0, latent_dim), dtype=np.float32)
        completion_probe_graph_indices_arr = np.empty((0,), dtype=np.int32)

    trajectory_matrix = trajectories_tensor.reshape(num_graphs, num_steps * latent_dim)
    trajectory_pca_matrix = pca_fit_tensor.reshape(num_graphs, pca_fit_steps * latent_dim)
    step_matrix = trajectories_tensor.reshape(num_graphs * num_steps, latent_dim)
    step_pca_fit_matrix = pca_fit_tensor.reshape(num_graphs * pca_fit_steps, latent_dim)
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
        completion_probes=completion_probes,
        completion_probe_graph_indices=completion_probe_graph_indices_arr,
    )

    metadata = {
        "config_name": config.name,
        "checkpoint_step": step,
        "latent": args.latent,
        "node_agg": args.node_agg,
        "step_policy": args.step_policy,
        "graph_index": args.graph_index,
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
        "pca_fit_excludes_extra_steps": bool(pca_fit_excludes_extra_steps),
        "pca_fit_steps": int(pca_fit_steps),
        "pca_fit_step_matrix_shape": list(step_pca_fit_matrix.shape),
        "pca_fit_trajectory_matrix_shape": list(trajectory_pca_matrix.shape),
        "step_pca_mean_coordinates": None,
        "completion_probe_shape": list(completion_probes.shape),
        "completion_probe_graph_indices": completion_probe_graph_indices,
        "step_pca_completion_probe_mean_coordinate": None,
        "step_pca_completion_probe_count": int(completion_probes.shape[0]),
    }

    if args.pca != "none":
        if args.pca in ("trajectory", "both"):
            payload, used_components = pca_fit_transform(
                trajectory_pca_matrix, args.pca_components
            )
            np.savez(output_dir / "pca_trajectory.npz", **payload)
            if args.plot and used_components >= 2:
                plot_trajectory_scatter(
                    payload["projected"],
                    np.array(selected_indices, dtype=np.int32),
                    output_dir / "pca_trajectory.png",
                    "Trajectory-wise PCA",
                    payload["explained_variance_ratio"],
                )

        if args.pca in ("step", "both"):
            fit_payload, used_components = pca_fit_transform(
                step_pca_fit_matrix, args.pca_components
            )
            fit_mean = np.asarray(fit_payload["mean"], dtype=np.float64)
            fit_components = np.asarray(fit_payload["components"], dtype=np.float64)
            step_projected_all = pca_project(
                step_matrix.astype(np.float64),
                fit_components,
                fit_mean,
            ).astype(np.float32, copy=False)
            payload = dict(fit_payload)
            payload["projected"] = step_projected_all
            np.savez(output_dir / "pca_step.npz", **payload)
            metadata["step_pca_mean_coordinates"] = step_pca_mean_coordinates(
                projected=step_projected_all,
                step_indices=step_indices,
                num_steps=num_steps,
            )
            completion_probe_projected = np.empty((0, used_components), dtype=np.float32)
            completion_probe_mean = None
            if completion_probes.shape[0] > 0:
                completion_probe_projected = pca_project(
                    completion_probes.astype(np.float64),
                    fit_components,
                    fit_mean,
                ).astype(np.float32, copy=False)
                completion_probe_mean = completion_probe_projected.mean(axis=0)
                metadata["step_pca_completion_probe_mean_coordinate"] = (
                    completion_probe_mean.astype(np.float64).tolist()
                )
                np.savez(
                    output_dir / "pca_step_completion_probes.npz",
                    projected=completion_probe_projected,
                    graph_indices=completion_probe_graph_indices_arr,
                )
            if args.plot and used_components >= 2:
                plot_step_trajectories(
                    step_projected_all,
                    graph_indices,
                    step_indices,
                    output_dir / "pca_step.png",
                    "Step-wise PCA",
                    payload["explained_variance_ratio"],
                    completion_probe_points=completion_probe_projected,
                    completion_probe_mean=completion_probe_mean,
                )

    write_json(output_dir / "metadata.json", metadata)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
