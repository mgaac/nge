"""Analyze monotonic relations between sequential latent distances and PCA axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.common import (
    load_analysis_dataset,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
    resolve_config,
    resolve_dataset_path,
)
from src.analysis.embedding_trajectories import (
    choose_step_count,
    collect_graph_trajectory,
    count_execution_steps,
    pca_fit_transform,
    pca_project,
)
from src.utils.task_specs import (
    ANALYSIS_LATENT_CHOICES,
    SELECT_TASK_CHOICES,
    algorithm_display_name,
    execution_step_counts,
    normalize_algorithm_order,
    resolve_selected_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether sequential latent-change magnitudes are monotonic with "
            "respect to PCA coordinates near task-specific termination steps."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
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
        help="Override dataset path (.npz).",
    )
    parser.add_argument(
        "--graph-index",
        type=int,
        default=None,
        help="Analyze a single graph by index.",
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
        help="Fixed step count when --step-policy=fixed.",
    )
    parser.add_argument(
        "--extra-steps",
        type=int,
        default=0,
        help="Fake-continue execution for N additional steps after termination.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Number of PCA components to fit.",
    )
    parser.add_argument(
        "--pc-index",
        type=int,
        default=None,
        help="Optional 1-based PCA component to isolate (default: analyze all fitted PCs).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        choices=SELECT_TASK_CHOICES,
        help="Tasks to analyze: all, bf, bfs, or prim.",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="l2",
        choices=["l2", "l1", "mse", "cosine"],
        help="Distance used between consecutive aggregated embeddings.",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="Optional absolute task end step to filter graphs on.",
    )
    parser.add_argument(
        "--end-step-tol",
        type=int,
        default=0,
        help="Allowed absolute deviation from --end-step.",
    )
    parser.add_argument(
        "--relative-start",
        type=int,
        default=-3,
        help="Start of relative step window around task end step.",
    )
    parser.add_argument(
        "--relative-end",
        type=int,
        default=0,
        help="End of relative step window around task end step.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs.",
    )
    return parser.parse_args()


def compute_step_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    diff = b - a
    if metric == "l2":
        return float(np.linalg.norm(diff))
    if metric == "l1":
        return float(np.sum(np.abs(diff)))
    if metric == "mse":
        return float(np.mean(diff * diff))
    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        if denom == 0.0:
            return 0.0
        return float(1.0 - np.dot(a, b) / denom)
    raise ValueError(f"Unknown distance metric: {metric}")


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    x_rank = average_ranks(x)
    y_rank = average_ranks(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denom <= 1e-12:
        return None
    return float(np.dot(x_centered, y_centered) / denom)


def monotonic_violation_rate(x: np.ndarray, y: np.ndarray, rho: float | None) -> float | None:
    if x.size < 2 or y.size < 2 or rho is None:
        return None
    order = np.argsort(x, kind="mergesort")
    diffs = np.diff(y[order])
    if diffs.size == 0:
        return None
    if rho >= 0:
        violations = diffs < 0
    else:
        violations = diffs > 0
    return float(np.mean(violations.astype(np.float64)))


def summarize_pc_relation(x: np.ndarray, y: np.ndarray) -> dict:
    rho = spearman_corr(x, y)
    violation_rate = monotonic_violation_rate(x, y, rho)
    direction = None
    if rho is not None:
        direction = "increasing" if rho >= 0 else "decreasing"
    return {
        "count": int(x.size),
        "spearman_rho": rho,
        "monotonic_direction": direction,
        "monotonic_violation_rate": violation_rate,
        "pc_mean": float(np.mean(x)) if x.size else None,
        "pc_std": float(np.std(x)) if x.size else None,
        "distance_mean": float(np.mean(y)) if y.size else None,
        "distance_std": float(np.std(y)) if y.size else None,
    }


def build_selection(
    dataset: List[dict],
    algorithm_order: tuple[str, ...],
    graph_index: int | None,
    max_graphs: int | None,
    extra_steps: int,
    step_policy: str,
    fixed_steps: int | None,
) -> tuple[List[int], List[dict], int]:
    if graph_index is not None:
        if graph_index < 0 or graph_index >= len(dataset):
            raise IndexError(
                f"--graph-index out of range: {graph_index} (dataset size={len(dataset)})"
            )
        graph = dataset[graph_index]
        return [graph_index], [graph], count_execution_steps(graph, extra_steps, algorithm_order)

    graphs = dataset[: max_graphs] if max_graphs is not None else dataset
    step_counts = [
        count_execution_steps(graph, extra_steps, algorithm_order) for graph in graphs
    ]
    target_steps = choose_step_count(step_counts, step_policy, fixed_steps)
    selected_indices = [idx for idx, steps in enumerate(step_counts) if steps == target_steps]
    if not selected_indices:
        raise ValueError(
            f"No graphs matched step count {target_steps} under policy {step_policy}."
        )
    return selected_indices, [graphs[i] for i in selected_indices], target_steps


def plot_task_scatter_grid(
    task: str,
    points: List[dict],
    pc_indices: List[int],
    output_path: Path,
) -> None:
    if not points:
        return

    relative_steps = np.array([point["relative_step"] for point in points], dtype=np.int32)
    distances = np.array([point["distance"] for point in points], dtype=np.float64)
    num_cols = min(len(pc_indices), 3)
    num_rows = int(np.ceil(len(pc_indices) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))
    axes = np.atleast_1d(axes).reshape(num_rows, num_cols)

    for axis in axes.flat[len(pc_indices):]:
        axis.axis("off")

    for axis, pc_idx in zip(axes.flat, pc_indices):
        coords = np.array([point["pcs"][pc_idx] for point in points], dtype=np.float64)
        stats = summarize_pc_relation(coords, distances)
        scatter = axis.scatter(
            coords,
            distances,
            c=relative_steps,
            cmap="coolwarm",
            alpha=0.85,
            s=28,
            linewidths=0.2,
            edgecolors="black",
        )
        axis.set_xlabel(f"PC{pc_idx + 1}")
        axis.set_ylabel("Sequential distance")
        axis.set_title(
            f"{algorithm_display_name(task)} vs PC{pc_idx + 1}\n"
            f"rho={stats['spearman_rho'] if stats['spearman_rho'] is not None else float('nan'):.3f}"
        )
        axis.grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=axis, label="relative step to task end")

    fig.suptitle(
        f"Sequential distance monotonicity near {algorithm_display_name(task)} termination"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spearman_heatmap(
    results: Dict[str, dict],
    pc_indices: List[int],
    output_path: Path,
) -> None:
    tasks = list(results.keys())
    if not tasks or not pc_indices:
        return

    matrix = np.full((len(tasks), len(pc_indices)), np.nan, dtype=np.float64)
    for row, task in enumerate(tasks):
        for col, pc_idx in enumerate(pc_indices):
            key = f"pc{pc_idx + 1}"
            rho = results[task]["overall"].get(key, {}).get("spearman_rho")
            if rho is not None:
                matrix[row, col] = rho

    fig, ax = plt.subplots(figsize=(1.8 * len(pc_indices) + 2.5, 1.4 * len(tasks) + 2.0))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(pc_indices)), [f"PC{idx + 1}" for idx in pc_indices])
    ax.set_yticks(
        np.arange(len(tasks)),
        [algorithm_display_name(task) for task in tasks],
    )
    ax.set_title("Spearman rho: sequential distance vs PCA coordinate")
    for row in range(len(tasks)):
        for col in range(len(pc_indices)):
            value = matrix[row, col]
            label = "nan" if np.isnan(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.extra_steps < 0:
        raise ValueError("--extra-steps must be non-negative.")
    if args.pca_components <= 0:
        raise ValueError("--pca-components must be positive.")
    if args.end_step is not None and args.end_step < 0:
        raise ValueError("--end-step must be non-negative.")
    if args.end_step_tol < 0:
        raise ValueError("--end-step-tol must be non-negative.")
    if args.relative_start > args.relative_end:
        raise ValueError("--relative-start must be <= --relative-end.")

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
    selected_indices, selected_graphs, target_steps = build_selection(
        dataset=dataset,
        algorithm_order=algorithm_order,
        graph_index=args.graph_index,
        max_graphs=args.max_graphs,
        extra_steps=args.extra_steps,
        step_policy=args.step_policy,
        fixed_steps=args.steps,
    )

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

    trajectory_tensor = np.stack(trajectories, axis=0)
    num_graphs, num_steps, latent_dim = trajectory_tensor.shape
    pca_fit_steps = num_steps - args.extra_steps if args.extra_steps > 0 and num_steps > args.extra_steps else num_steps
    pca_fit_matrix = trajectory_tensor[:, :pca_fit_steps, :].reshape(num_graphs * pca_fit_steps, latent_dim)
    fit_payload, used_components = pca_fit_transform(pca_fit_matrix, args.pca_components)
    projected = pca_project(
        trajectory_tensor.reshape(num_graphs * num_steps, latent_dim).astype(np.float64),
        np.asarray(fit_payload["components"], dtype=np.float64),
        np.asarray(fit_payload["mean"], dtype=np.float64),
    ).astype(np.float32, copy=False).reshape(num_graphs, num_steps, used_components)

    pc_indices = list(range(used_components))
    if args.pc_index is not None:
        if args.pc_index <= 0 or args.pc_index > used_components:
            raise ValueError(
                f"--pc-index must be in 1..{used_components}, got {args.pc_index}"
            )
        pc_indices = [args.pc_index - 1]

    sequential_distances = np.zeros((num_graphs, max(num_steps - 1, 0)), dtype=np.float64)
    for graph_idx in range(num_graphs):
        for step_idx in range(1, num_steps):
            sequential_distances[graph_idx, step_idx - 1] = compute_step_distance(
                trajectory_tensor[graph_idx, step_idx - 1],
                trajectory_tensor[graph_idx, step_idx],
                args.distance,
            )

    selected_tasks = resolve_selected_tasks(args.tasks, algorithm_order)
    task_results: Dict[str, dict] = {}
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            run_dir / "analysis" / "pc_monotonicity"
            if run_dir
            else Path("analysis/pc_monotonicity")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in algorithm_order:
        if not selected_tasks.get(task, False):
            continue

        points: List[dict] = []
        matched_graph_indices: List[int] = []
        end_steps_for_task: List[int] = []
        for local_idx, graph in enumerate(selected_graphs):
            end_step = execution_step_counts(graph, algorithm_order)[task]
            if args.end_step is not None and abs(end_step - args.end_step) > args.end_step_tol:
                continue
            matched_graph_indices.append(int(selected_indices[local_idx]))
            end_steps_for_task.append(int(end_step))

            step_start = max(1, end_step + args.relative_start)
            step_end = min(num_steps - 1, end_step + args.relative_end)
            if step_start > step_end:
                continue
            for step_idx in range(step_start, step_end + 1):
                points.append(
                    {
                        "graph_index": int(selected_indices[local_idx]),
                        "step": int(step_idx),
                        "relative_step": int(step_idx - end_step),
                        "end_step": int(end_step),
                        "distance": float(sequential_distances[local_idx, step_idx - 1]),
                        "pcs": projected[local_idx, step_idx].astype(np.float64).tolist(),
                    }
                )

        overall: Dict[str, dict] = {}
        relative_stats: Dict[str, dict] = {}
        for pc_idx in pc_indices:
            coords = np.array([point["pcs"][pc_idx] for point in points], dtype=np.float64)
            distances = np.array([point["distance"] for point in points], dtype=np.float64)
            overall[f"pc{pc_idx + 1}"] = summarize_pc_relation(coords, distances)

        for relative_step in range(args.relative_start, args.relative_end + 1):
            subset = [point for point in points if point["relative_step"] == relative_step]
            relative_stats[str(relative_step)] = {
                f"pc{pc_idx + 1}": summarize_pc_relation(
                    np.array([point["pcs"][pc_idx] for point in subset], dtype=np.float64),
                    np.array([point["distance"] for point in subset], dtype=np.float64),
                )
                for pc_idx in pc_indices
            }
            relative_stats[str(relative_step)]["count"] = int(len(subset))

        task_results[task] = {
            "num_points": int(len(points)),
            "num_graphs": int(len(matched_graph_indices)),
            "graph_indices": matched_graph_indices,
            "end_step_summary": {
                "min": min(end_steps_for_task) if end_steps_for_task else None,
                "max": max(end_steps_for_task) if end_steps_for_task else None,
                "mean": float(np.mean(end_steps_for_task)) if end_steps_for_task else None,
            },
            "overall": overall,
            "relative_step_stats": relative_stats,
        }

        if points:
            plot_task_scatter_grid(
                task=task,
                points=points,
                pc_indices=pc_indices,
                output_path=output_dir / f"{task}_scatter.png",
            )

    if task_results:
        plot_spearman_heatmap(
            results=task_results,
            pc_indices=pc_indices,
            output_path=output_dir / "spearman_heatmap.png",
        )

    payload = {
        "config_name": config.name,
        "checkpoint_step": step,
        "dataset": str(dataset_path),
        "split": args.split if args.dataset is None else None,
        "latent": args.latent,
        "node_agg": args.node_agg,
        "distance": args.distance,
        "pca_components_requested": int(args.pca_components),
        "pca_components_used": int(used_components),
        "pc_indices_analyzed": [int(idx + 1) for idx in pc_indices],
        "algorithms": list(algorithm_order),
        "selected_tasks": [
            task for task in algorithm_order if selected_tasks.get(task, False)
        ],
        "graph_index": args.graph_index,
        "max_graphs": args.max_graphs,
        "target_steps": int(target_steps),
        "extra_steps": int(args.extra_steps),
        "end_step_filter": args.end_step,
        "end_step_tolerance": int(args.end_step_tol),
        "relative_window": [int(args.relative_start), int(args.relative_end)],
        "selected_graph_indices": [int(idx) for idx in selected_indices],
        "task_results": task_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
