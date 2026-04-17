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
    count_execution_steps as shared_count_execution_steps,
    iter_execution_feature_values,
    load_analysis_dataset,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
    resolve_config,
    resolve_dataset_path,
)
from src.utils.task_specs import (
    ANALYSIS_LATENT_CHOICES,
    build_node_algo_features,
    execution_step_counts,
    feature_values_for_step,
    input_feature_names,
    normalize_algorithm_order,
    supported_algorithms,
    zero_feature_values,
)


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
        "--isolate-input-algorithm",
        type=str,
        choices=supported_algorithms(),
        default=None,
        help=(
            "If provided, keep recurrent inputs only for this algorithm and zero "
            "all other algorithm recurrent inputs at every step."
        ),
    )
    parser.add_argument(
        "--isolate-other-inputs-mode",
        type=str,
        choices=["zero_others", "final_others"],
        default="zero_others",
        help=(
            "Behavior for non-selected algorithms when --isolate-input-algorithm is set: "
            "'zero_others' keeps them at zero; 'final_others' pins them to their final state."
        ),
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
        "--execution-window",
        type=str,
        choices=["all", *supported_algorithms()],
        default="all",
        help=(
            "Restrict plotted/PCA trajectory steps to the prefix where the selected "
            "algorithm is still executing. For example, --execution-window bfs keeps "
            "only steps < max_bfs_execution_steps across the selected graphs."
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


def count_execution_steps(
    graph_data: dict,
    extra_steps: int,
    algorithm_order: tuple[str, ...],
) -> int:
    return shared_count_execution_steps(graph_data, extra_steps, algorithm_order)


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


def execution_window_steps(
    selected_graphs: List[dict],
    execution_window: str,
    algorithm_order: tuple[str, ...],
) -> int | None:
    if execution_window == "all":
        return None
    if execution_window not in algorithm_order:
        raise ValueError(
            f"Execution window '{execution_window}' is unavailable for algorithms {algorithm_order}."
        )

    active_steps = [
        execution_step_counts(graph, algorithm_order)[execution_window]
        for graph in selected_graphs
    ]
    if not active_steps:
        return None

    max_active_steps = int(max(active_steps))
    if max_active_steps <= 0:
        raise ValueError(
            f"No executable steps found for execution window '{execution_window}'."
        )
    return max_active_steps


def execution_window_label(execution_window: str) -> str | None:
    if execution_window == "all":
        return None
    return f"{execution_window.upper()}-active"


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
    isolate_input_algorithm: str | None = None,
    isolate_other_inputs_mode: str = "zero_others",
) -> np.ndarray:
    num_nodes = graph_data["num_nodes"]
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])
    step_embeddings: List[np.ndarray] = []
    final_feature_values: Dict[str, mx.array] | None = None
    isolate_feature_names: tuple[str, ...] = ()
    if isolate_input_algorithm is not None:
        isolate_feature_names = input_feature_names((isolate_input_algorithm,))
        if isolate_other_inputs_mode == "final_others":
            final_step = max(execution_step_counts(graph_data, model.algorithms).values(), default=0)
            final_feature_values = feature_values_for_step(
                graph_data, final_step, model.algorithms
            )
        elif isolate_other_inputs_mode != "zero_others":
            raise ValueError(
                f"Unknown isolate_other_inputs_mode: {isolate_other_inputs_mode}"
            )

    for feature_values in iter_execution_feature_values(
        graph_data, extra_steps, model.algorithms
    ):
        model_feature_values = feature_values
        if isolate_input_algorithm is not None:
            if isolate_other_inputs_mode == "zero_others":
                model_feature_values = zero_feature_values(num_nodes, model.algorithms)
            else:
                assert final_feature_values is not None
                model_feature_values = dict(final_feature_values)
            for feature_name in isolate_feature_names:
                model_feature_values[feature_name] = feature_values[feature_name]

        node_algo_features = build_node_algo_features(model_feature_values, model.algorithms)
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


def algorithm_reference_chain(
    projected: np.ndarray,
    step_indices: np.ndarray,
    selected_graphs: List[dict],
    num_steps: int,
    algorithm_order: tuple[str, ...],
) -> List[dict]:
    references: List[dict] = []
    if projected.ndim != 2:
        raise ValueError("Projected PCA coordinates must be a 2D array.")

    for algorithm in algorithm_order:
        end_steps = [
            float(execution_step_counts(graph, algorithm_order)[algorithm])
            for graph in selected_graphs
        ]
        if not end_steps:
            continue
        mean_end_step = float(np.mean(np.array(end_steps, dtype=np.float64)))
        rounded_step = int(np.rint(mean_end_step))
        if num_steps <= 0:
            continue
        clamped_step = min(max(rounded_step, 0), num_steps - 1)
        mask = step_indices == clamped_step
        if not np.any(mask):
            continue
        mean_coord = projected[mask].mean(axis=0)
        references.append(
            {
                "algorithm": algorithm,
                "mean_end_step": mean_end_step,
                "reference_step": clamped_step,
                "reference_step_raw": rounded_step,
                "mean_coordinate": mean_coord.astype(np.float64, copy=False).tolist(),
            }
        )
    return references


def _format_title_with_variance(title: str, explained_ratio: np.ndarray) -> str:
    if explained_ratio.size >= 3:
        pc1 = explained_ratio[0] * 100.0
        pc2 = explained_ratio[1] * 100.0
        pc3 = explained_ratio[2] * 100.0
        return f"{title} (PC1 {pc1:.1f}%, PC2 {pc2:.1f}%, PC3 {pc3:.1f}%)"
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


def _plot_dimensions(points: np.ndarray) -> int:
    if points.ndim != 2:
        raise ValueError("Plotting expects a 2D array of projected points.")
    if points.shape[1] >= 3:
        return 3
    if points.shape[1] >= 2:
        return 2
    raise ValueError("Need at least 2 PCA components to plot trajectories.")


def _axis_labels(num_components: int) -> tuple[str, ...]:
    if num_components == 3:
        return ("PC1", "PC2", "PC3")
    return ("PC1", "PC2")


def _pairwise_component_views(num_components: int) -> list[tuple[int, int]]:
    if num_components < 3:
        return []
    return [(0, 1), (0, 2), (1, 2)]


def _component_view_suffix(axes: tuple[int, int]) -> str:
    return f"_pc{axes[0] + 1}{axes[1] + 1}"


def _reference_segments(
    algorithm_reference_points: List[dict] | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not algorithm_reference_points:
        return []
    point_map = {
        entry["algorithm"]: np.array(entry["mean_coordinate"], dtype=np.float64)
        for entry in algorithm_reference_points
    }
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    if "bf" in point_map and "bfs" in point_map:
        segments.append((point_map["bf"], point_map["bfs"]))
    if "bf" in point_map and "prim" in point_map:
        segments.append((point_map["bf"], point_map["prim"]))
    return segments


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

    num_components = _plot_dimensions(points)
    unique_graphs = np.unique(graph_labels)
    norm = plt.Normalize(vmin=float(unique_graphs.min()), vmax=float(unique_graphs.max()))
    cmap = matplotlib.colormaps["viridis"]
    fig = plt.figure(figsize=(7, 6) if num_components == 3 else (6, 5))
    if num_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=graph_labels,
            cmap=cmap,
            norm=norm,
            s=30,
            alpha=0.9,
            linewidths=0.2,
            edgecolors="black",
        )
    else:
        ax = fig.add_subplot(111)
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

    labels = _axis_labels(num_components)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if num_components == 3:
        ax.set_zlabel(labels[2])
    ax.set_title(_format_title_with_variance(title, explained_ratio))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_scatter_pairwise(
    points: np.ndarray,
    graph_labels: np.ndarray,
    output_path: Path,
    title: str,
    explained_ratio: np.ndarray,
    axes: tuple[int, int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if points.ndim != 2 or points.shape[1] <= max(axes):
        raise ValueError("Pairwise trajectory plotting requires the requested PCA axes.")

    unique_graphs = np.unique(graph_labels)
    norm = plt.Normalize(vmin=float(unique_graphs.min()), vmax=float(unique_graphs.max()))
    cmap = matplotlib.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        points[:, axes[0]],
        points[:, axes[1]],
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

    ax.set_xlabel(f"PC{axes[0] + 1}")
    ax.set_ylabel(f"PC{axes[1] + 1}")
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
    algorithm_reference_points: List[dict] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    num_components = _plot_dimensions(points)
    cmap = matplotlib.colormaps["viridis"]
    step_min = int(np.min(step_indices))
    step_max = int(np.max(step_indices))
    norm = mcolors.Normalize(
        vmin=step_min,
        vmax=step_max if step_max > step_min else step_min + 1,
    )

    fig = plt.figure(figsize=(8, 7) if num_components == 3 else (7, 6))
    if num_components == 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    for graph_label in np.unique(graph_labels):
        graph_mask = graph_labels == graph_label
        graph_points = points[graph_mask]
        graph_steps = step_indices[graph_mask]
        order = np.argsort(graph_steps)
        graph_points = graph_points[order]
        graph_steps = graph_steps[order]

        # Keep trajectory lines subtle so step colors remain visually dominant.
        if num_components == 3:
            ax.plot(
                graph_points[:, 0],
                graph_points[:, 1],
                graph_points[:, 2],
                color="#5f6368",
                linewidth=0.45,
                alpha=0.14,
                zorder=1,
            )
        else:
            ax.plot(
                graph_points[:, 0],
                graph_points[:, 1],
                color="#5f6368",
                linewidth=0.45,
                alpha=0.14,
                zorder=1,
            )
        point_colors = cmap(norm(graph_steps))
        if num_components == 3:
            ax.scatter(
                graph_points[:, 0],
                graph_points[:, 1],
                graph_points[:, 2],
                c=point_colors,
                s=22,
                alpha=0.9,
                linewidths=0.2,
                edgecolors="black",
                zorder=2,
            )
        else:
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
        if num_components == 3:
            ax.scatter(
                completion_probe_points[:, 0],
                completion_probe_points[:, 1],
                completion_probe_points[:, 2],
                marker="X",
                s=46,
                color="#d62728",
                alpha=0.8,
                linewidths=0.4,
                edgecolors="black",
                zorder=3,
            )
        else:
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
    if completion_probe_mean is not None and completion_probe_mean.size >= num_components:
        if num_components == 3:
            ax.scatter(
                [completion_probe_mean[0]],
                [completion_probe_mean[1]],
                [completion_probe_mean[2]],
                marker="*",
                s=170,
                color="#d62728",
                alpha=0.95,
                linewidths=0.6,
                edgecolors="black",
                zorder=4,
            )
        else:
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

    reference_handles: List[Line2D] = []
    if algorithm_reference_points:
        ref_color = "#111111"
        ref_coords = np.array(
            [entry["mean_coordinate"][:num_components] for entry in algorithm_reference_points],
            dtype=np.float64,
        )
        for start, end in _reference_segments(algorithm_reference_points):
            if num_components == 3:
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=ref_color,
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=5,
                )
            else:
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=ref_color,
                    linewidth=1.4,
                    alpha=0.9,
                    zorder=5,
                )
        for entry, coord in zip(algorithm_reference_points, ref_coords):
            label = (
                f"{entry['algorithm']} @ step {entry['reference_step']}"
            )
            if num_components == 3:
                ax.scatter(
                    [coord[0]],
                    [coord[1]],
                    [coord[2]],
                    marker="D",
                    s=56,
                    color=ref_color,
                    alpha=0.95,
                    linewidths=0.4,
                    edgecolors="white",
                    zorder=6,
                )
                ax.text(
                    coord[0],
                    coord[1],
                    coord[2],
                    f" {entry['algorithm'].upper()}",
                    color=ref_color,
                    fontsize=8,
                    zorder=7,
                )
            else:
                ax.scatter(
                    [coord[0]],
                    [coord[1]],
                    marker="D",
                    s=56,
                    color=ref_color,
                    alpha=0.95,
                    linewidths=0.4,
                    edgecolors="white",
                    zorder=6,
                )
                ax.text(
                    coord[0],
                    coord[1],
                    f" {entry['algorithm'].upper()}",
                    color=ref_color,
                    fontsize=8,
                    zorder=7,
                )
            reference_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color=ref_color,
                    markerfacecolor=ref_color,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    linewidth=1.4,
                    markersize=6,
                    label=label,
                )
            )

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("execution step")
    colorbar.ax.tick_params(labelsize=8)

    legend_handles: List[Line2D] = []
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
    legend_handles.extend(reference_handles)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Markers",
            loc="upper right",
            fontsize=7,
            ncol=1,
            framealpha=0.9,
        )

    labels = _axis_labels(num_components)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if num_components == 3:
        ax.set_zlabel(labels[2])
    ax.set_title(_format_title_with_variance(title, explained_ratio))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_step_trajectories_pairwise(
    points: np.ndarray,
    graph_labels: np.ndarray,
    step_indices: np.ndarray,
    output_path: Path,
    title: str,
    explained_ratio: np.ndarray,
    axes: tuple[int, int],
    completion_probe_points: np.ndarray | None = None,
    completion_probe_mean: np.ndarray | None = None,
    algorithm_reference_points: List[dict] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    if points.ndim != 2 or points.shape[1] <= max(axes):
        raise ValueError("Pairwise step plotting requires the requested PCA axes.")

    cmap = matplotlib.colormaps["viridis"]
    step_min = int(np.min(step_indices))
    step_max = int(np.max(step_indices))
    norm = mcolors.Normalize(
        vmin=step_min,
        vmax=step_max if step_max > step_min else step_min + 1,
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for graph_label in np.unique(graph_labels):
        graph_mask = graph_labels == graph_label
        graph_points = points[graph_mask]
        graph_steps = step_indices[graph_mask]
        order = np.argsort(graph_steps)
        graph_points = graph_points[order]
        graph_steps = graph_steps[order]

        ax.plot(
            graph_points[:, axes[0]],
            graph_points[:, axes[1]],
            color="#5f6368",
            linewidth=0.45,
            alpha=0.14,
            zorder=1,
        )
        point_colors = cmap(norm(graph_steps))
        ax.scatter(
            graph_points[:, axes[0]],
            graph_points[:, axes[1]],
            c=point_colors,
            s=22,
            alpha=0.9,
            linewidths=0.2,
            edgecolors="black",
            zorder=2,
        )

    if completion_probe_points is not None and completion_probe_points.size > 0:
        ax.scatter(
            completion_probe_points[:, axes[0]],
            completion_probe_points[:, axes[1]],
            marker="X",
            s=46,
            color="#d62728",
            alpha=0.8,
            linewidths=0.4,
            edgecolors="black",
            zorder=3,
        )
    if completion_probe_mean is not None and completion_probe_mean.size > max(axes):
        ax.scatter(
            [completion_probe_mean[axes[0]]],
            [completion_probe_mean[axes[1]]],
            marker="*",
            s=170,
            color="#d62728",
            alpha=0.95,
            linewidths=0.6,
            edgecolors="black",
            zorder=4,
        )

    reference_handles: List[Line2D] = []
    if algorithm_reference_points:
        ref_color = "#111111"
        ref_coords = np.array(
            [
                [entry["mean_coordinate"][axes[0]], entry["mean_coordinate"][axes[1]]]
                for entry in algorithm_reference_points
            ],
            dtype=np.float64,
        )
        for start, end in _reference_segments(algorithm_reference_points):
            ax.plot(
                [start[axes[0]], end[axes[0]]],
                [start[axes[1]], end[axes[1]]],
                color=ref_color,
                linewidth=1.4,
                alpha=0.9,
                zorder=5,
            )
        for entry, coord in zip(algorithm_reference_points, ref_coords):
            label = f"{entry['algorithm']} @ step {entry['reference_step']}"
            ax.scatter(
                [coord[0]],
                [coord[1]],
                marker="D",
                s=56,
                color=ref_color,
                alpha=0.95,
                linewidths=0.4,
                edgecolors="white",
                zorder=6,
            )
            ax.text(
                coord[0],
                coord[1],
                f" {entry['algorithm'].upper()}",
                color=ref_color,
                fontsize=8,
                zorder=7,
            )
            reference_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color=ref_color,
                    markerfacecolor=ref_color,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    linewidth=1.4,
                    markersize=6,
                    label=label,
                )
            )

    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    colorbar.set_label("execution step")
    colorbar.ax.tick_params(labelsize=8)

    legend_handles: List[Line2D] = []
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
    if completion_probe_mean is not None and completion_probe_mean.size > max(axes):
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
    legend_handles.extend(reference_handles)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Markers",
            loc="upper right",
            fontsize=7,
            ncol=1,
            framealpha=0.9,
        )

    ax.set_xlabel(f"PC{axes[0] + 1}")
    ax.set_ylabel(f"PC{axes[1] + 1}")
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
    algorithm_order = normalize_algorithm_order(config.model.algorithms)
    if (
        args.isolate_input_algorithm is not None
        and args.isolate_input_algorithm not in algorithm_order
    ):
        raise ValueError(
            f"--isolate-input-algorithm={args.isolate_input_algorithm} is not present in "
            f"the configured model algorithms: {algorithm_order}"
        )

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
                f"--graph-index out of range: {args.graph_index} (dataset size={len(dataset)})"
            )
        selected_indices = [args.graph_index]
        selected_graphs = [dataset[args.graph_index]]
        target_steps = count_execution_steps(
            selected_graphs[0], args.extra_steps, algorithm_order
        )
    else:
        graphs = dataset
        if args.max_graphs is not None:
            graphs = dataset[: args.max_graphs]

        step_counts = [
            count_execution_steps(graph, args.extra_steps, algorithm_order)
            for graph in graphs
        ]
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
            isolate_input_algorithm=args.isolate_input_algorithm,
            isolate_other_inputs_mode=args.isolate_other_inputs_mode,
        )
        for graph in selected_graphs
    ]

    trajectories_tensor = np.stack(trajectories, axis=0)
    raw_num_steps = int(trajectories_tensor.shape[1])
    window_steps = execution_window_steps(
        selected_graphs, args.execution_window, algorithm_order
    )
    if window_steps is not None:
        trajectories_tensor = trajectories_tensor[:, :window_steps, :]
    num_graphs, num_steps, latent_dim = trajectories_tensor.shape

    base_execution_steps = max(raw_num_steps - max(args.extra_steps, 0), 0)
    pca_fit_steps = min(num_steps, base_execution_steps) if args.extra_steps > 0 else num_steps
    pca_fit_excludes_extra_steps = args.extra_steps > 0 and num_steps > pca_fit_steps
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
            isolate_input_algorithm=args.isolate_input_algorithm,
            isolate_other_inputs_mode=args.isolate_other_inputs_mode,
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

    output_artifact_name = (
        f"embedding_trajectories_{Path(args.dataset).stem}"
        if args.dataset is not None
        else "embedding_trajectories"
    )
    if args.isolate_input_algorithm is not None:
        output_artifact_name = f"{output_artifact_name}_iso_{args.isolate_input_algorithm}"
        if args.isolate_other_inputs_mode == "final_others":
            output_artifact_name = f"{output_artifact_name}_final_others"

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            run_dir / "analysis" / output_artifact_name
            if run_dir
            else Path("analysis") / output_artifact_name
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
        "algorithms": list(algorithm_order),
        "isolate_input_algorithm": args.isolate_input_algorithm,
        "isolate_other_inputs_mode": args.isolate_other_inputs_mode,
        "latent": args.latent,
        "node_agg": args.node_agg,
        "step_policy": args.step_policy,
        "graph_index": args.graph_index,
        "target_steps": num_steps,
        "original_target_steps": int(target_steps),
        "extra_steps": args.extra_steps,
        "execution_window": args.execution_window,
        "execution_window_label": execution_window_label(args.execution_window),
        "execution_window_steps": int(num_steps),
        "latent_dim": latent_dim,
        "pca_components_requested": int(args.pca_components),
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
        "trajectory_pca_components_used": None,
        "step_pca_components_used": None,
        "trajectory_pca_pairwise_views": [],
        "step_pca_pairwise_views": [],
        "step_pca_algorithm_reference_chain": None,
    }

    if args.pca != "none":
        window_suffix = execution_window_label(args.execution_window)
        trajectory_title = "Trajectory-wise PCA"
        step_title = "Step-wise PCA"
        if window_suffix is not None:
            trajectory_title = f"{trajectory_title} [{window_suffix}]"
            step_title = f"{step_title} [{window_suffix}]"

        if args.pca in ("trajectory", "both"):
            payload, used_components = pca_fit_transform(
                trajectory_pca_matrix, args.pca_components
            )
            metadata["trajectory_pca_components_used"] = int(used_components)
            np.savez(output_dir / "pca_trajectory.npz", **payload)
            if args.plot and used_components >= 2:
                plot_trajectory_scatter(
                    payload["projected"],
                    np.array(selected_indices, dtype=np.int32),
                    output_dir / "pca_trajectory.png",
                    trajectory_title,
                    payload["explained_variance_ratio"],
                )
                if used_components >= 3:
                    for axes in _pairwise_component_views(used_components):
                        pairwise_path = (
                            output_dir
                            / f"pca_trajectory{_component_view_suffix(axes)}.png"
                        )
                        plot_trajectory_scatter_pairwise(
                            payload["projected"],
                            np.array(selected_indices, dtype=np.int32),
                            pairwise_path,
                            f"{trajectory_title} (PC{axes[0] + 1}-PC{axes[1] + 1})",
                            payload["explained_variance_ratio"],
                            axes,
                        )
                        metadata["trajectory_pca_pairwise_views"].append(pairwise_path.name)

        if args.pca in ("step", "both"):
            fit_payload, used_components = pca_fit_transform(
                step_pca_fit_matrix, args.pca_components
            )
            metadata["step_pca_components_used"] = int(used_components)
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
            algorithm_reference_points = algorithm_reference_chain(
                projected=step_projected_all,
                step_indices=step_indices,
                selected_graphs=selected_graphs,
                num_steps=num_steps,
                algorithm_order=algorithm_order,
            )
            metadata["step_pca_algorithm_reference_chain"] = algorithm_reference_points
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
                    step_title,
                    payload["explained_variance_ratio"],
                    completion_probe_points=completion_probe_projected,
                    completion_probe_mean=completion_probe_mean,
                    algorithm_reference_points=algorithm_reference_points,
                )
                if used_components >= 3:
                    for axes in _pairwise_component_views(used_components):
                        pairwise_path = (
                            output_dir / f"pca_step{_component_view_suffix(axes)}.png"
                        )
                        plot_step_trajectories_pairwise(
                            step_projected_all,
                            graph_indices,
                            step_indices,
                            pairwise_path,
                            f"{step_title} (PC{axes[0] + 1}-PC{axes[1] + 1})",
                            payload["explained_variance_ratio"],
                            axes,
                            completion_probe_points=completion_probe_projected,
                            completion_probe_mean=completion_probe_mean,
                            algorithm_reference_points=algorithm_reference_points,
                        )
                        metadata["step_pca_pairwise_views"].append(pairwise_path.name)

    write_json(output_dir / "metadata.json", metadata)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
