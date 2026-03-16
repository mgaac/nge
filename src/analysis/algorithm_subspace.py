"""Analyze whether latent delta directions are algorithm-specific subspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGORITHMS = ("bf", "bfs", "prim")
SELECT_TASK_CHOICES = ("all", *ALGORITHMS)

SEQUENCE_KEYS = {
    "bf": "bf_distance_targets",
    "bfs": "bfs_state_targets",
    "prim": "prim_key_targets",
}


def resolve_selected_tasks(tasks_arg: str) -> Dict[str, bool]:
    if tasks_arg == "all":
        return {algorithm: True for algorithm in ALGORITHMS}
    if tasks_arg not in ALGORITHMS:
        raise ValueError(f"Unknown tasks selection: {tasks_arg}")
    return {algorithm: algorithm == tasks_arg for algorithm in ALGORITHMS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether latent delta directions concentrate in algorithm-specific "
            "subspaces using an existing embedding_trajectories artifact."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory used to resolve default trajectories/output paths.",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default=None,
        help=(
            "Directory containing trajectories.npz + metadata.json from "
            "src.analysis.embedding_trajectories. Defaults to "
            "<run-dir>/analysis/embedding_trajectories."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path (.npz). Defaults to metadata dataset.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        choices=SELECT_TASK_CHOICES,
        help="Algorithms to analyze: all, bf, bfs, or prim.",
    )
    parser.add_argument(
        "--membership-mode",
        type=str,
        default="active",
        choices=["active", "exclusive"],
        help=(
            "Use all deltas while an algorithm is active, or only deltas where it is "
            "the sole active algorithm."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Number of PCA components to fit per algorithm delta set.",
    )
    parser.add_argument(
        "--subspace-dim",
        type=int,
        default=None,
        help=(
            "Subspace dimension used for overlap/explained-variance comparisons. "
            "Defaults to min(fitted components across analyzed algorithms)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs. Default: <run-dir>/analysis/algorithm_subspace",
    )
    return parser.parse_args()


def resolve_trajectories_dir(args: argparse.Namespace) -> Path:
    if args.trajectories_dir is not None:
        return Path(args.trajectories_dir)
    if args.run_dir is not None:
        return Path(args.run_dir) / "analysis" / "embedding_trajectories"
    raise ValueError("Provide --trajectories-dir or --run-dir.")


def execution_step_counts_raw(dataset: np.lib.npyio.NpzFile, graph_index: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for algorithm, key in SEQUENCE_KEYS.items():
        array_key = f"{key}_{graph_index}"
        if array_key not in dataset:
            counts[algorithm] = 0
            continue
        counts[algorithm] = max(int(dataset[array_key].shape[0]) - 1, 0)
    return counts


def load_trajectory_source(
    trajectories_dir: Path, dataset_override: str | None
) -> tuple[dict, np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    metadata_path = trajectories_dir / "metadata.json"
    trajectories_path = trajectories_dir / "trajectories.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    if not trajectories_path.exists():
        raise FileNotFoundError(f"Missing trajectories.npz: {trajectories_path}")

    metadata = json.loads(metadata_path.read_text())
    dataset_path = Path(dataset_override) if dataset_override else Path(metadata["dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    trajectories_payload = np.load(trajectories_path)
    dataset_payload = np.load(dataset_path, allow_pickle=True)
    metadata["dataset"] = str(dataset_path)
    return metadata, trajectories_payload, dataset_payload


def aligned_completion_probes(
    trajectories_payload: np.lib.npyio.NpzFile,
    selected_graph_indices: np.ndarray,
) -> np.ndarray | None:
    if "completion_probes" not in trajectories_payload:
        return None
    probes = np.asarray(trajectories_payload["completion_probes"], dtype=np.float32)
    probe_indices = np.asarray(
        trajectories_payload.get("completion_probe_graph_indices", np.empty((0,), dtype=np.int32)),
        dtype=np.int32,
    )
    if probes.shape[0] == 0 or probe_indices.shape[0] != probes.shape[0]:
        return None

    probe_map = {int(index): probes[i] for i, index in enumerate(probe_indices.tolist())}
    aligned = []
    for graph_index in selected_graph_indices.tolist():
        if int(graph_index) not in probe_map:
            return None
        aligned.append(probe_map[int(graph_index)])
    return np.stack(aligned, axis=0).astype(np.float32, copy=False)


def extend_trajectories_with_terminal_probe(
    trajectories: np.ndarray,
    terminal_probes: np.ndarray | None,
) -> np.ndarray:
    if terminal_probes is None:
        return trajectories
    return np.concatenate([trajectories, terminal_probes[:, None, :]], axis=1)


def pca_fit_transform(data: np.ndarray, n_components: int) -> tuple[dict, int]:
    if data.ndim != 2:
        raise ValueError("PCA expects a 2D array.")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("Empty array passed to PCA.")

    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    used_components = min(vh.shape[0], n_components)
    components = vh[:used_components]
    projected = centered @ components.T
    if data.shape[0] > 1:
        explained_variance = (singular_values * singular_values) / (data.shape[0] - 1)
    else:
        explained_variance = singular_values * singular_values
    total_variance = explained_variance.sum()
    explained_ratio = (
        explained_variance[:used_components] / total_variance
        if total_variance > 0
        else np.zeros(used_components, dtype=np.float64)
    )
    return {
        "projected": projected.astype(np.float32, copy=False),
        "components": components.astype(np.float32, copy=False),
        "mean": mean.squeeze(0).astype(np.float32, copy=False),
        "explained_variance": explained_variance[:used_components].astype(np.float32, copy=False),
        "explained_variance_ratio": explained_ratio.astype(np.float32, copy=False),
    }, used_components


def pca_project(data: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return (data - mean.reshape(1, -1)) @ components.T


def format_title(title: str, explained_ratio: np.ndarray) -> str:
    if explained_ratio.size >= 2:
        return f"{title} (PC1 {explained_ratio[0] * 100.0:.1f}%, PC2 {explained_ratio[1] * 100.0:.1f}%)"
    return title


def build_discrete_color_map(labels: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    unique_labels = np.unique(labels)
    base = np.array(matplotlib.colormaps["tab20"](np.linspace(0.0, 1.0, 20)))
    color_map: dict[int, np.ndarray] = {}
    for idx, label in enumerate(unique_labels):
        if idx < len(base):
            color = np.array(base[idx], dtype=np.float32)
        else:
            frac = (idx - len(base)) / max(len(unique_labels) - len(base), 1)
            color = np.array(matplotlib.colormaps["hsv"](frac), dtype=np.float32)
        color_map[int(label)] = color
    return unique_labels, color_map


def plot_algorithm_delta_pca(
    projected: np.ndarray,
    step_indices: np.ndarray,
    output_path: Path,
    title: str,
    explained_ratio: np.ndarray,
) -> None:
    if projected.shape[1] < 2:
        return
    unique_steps, step_color_map = build_discrete_color_map(step_indices)
    colors = np.array([step_color_map[int(step)] for step in step_indices])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=colors,
        s=24,
        alpha=0.88,
        linewidths=0.2,
        edgecolors="black",
    )
    legend_handles = [
        plt.Line2D(
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
    ncols = 1 if len(legend_handles) <= 12 else 2
    ax.legend(handles=legend_handles, title="Execution step", loc="best", fontsize=7, ncol=ncols)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(format_title(title, explained_ratio))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    cbar_label: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Evaluated deltas")
    ax.set_ylabel("Basis algorithm")
    ax.set_title(title)
    threshold = float(np.nanmax(matrix)) * 0.55 if matrix.size > 0 else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color=text_color)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_active_vs_inactive_explained_variance(
    labels: List[str],
    active_scores: np.ndarray,
    inactive_scores: np.ndarray,
    output_path: Path,
) -> None:
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - width / 2.0, active_scores, width=width, color="#1f77b4", label="active")
    ax.bar(x + width / 2.0, inactive_scores, width=width, color="#ff7f0e", label="inactive")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Own subspace: active vs inactive delta variance capture")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def principal_correlation_summary(basis_a: np.ndarray, basis_b: np.ndarray) -> dict:
    singular_values = np.linalg.svd(basis_a @ basis_b.T, compute_uv=False)
    clipped = np.clip(singular_values, -1.0, 1.0)
    return {
        "canonical_correlations": clipped.astype(np.float64).tolist(),
        "mean_canonical_correlation": float(np.mean(clipped)),
        "min_canonical_correlation": float(np.min(clipped)),
        "principal_angles_deg": np.degrees(np.arccos(clipped)).astype(np.float64).tolist(),
    }


def explained_variance_ratio_in_subspace(data: np.ndarray, basis: np.ndarray) -> float:
    centered = data - data.mean(axis=0, keepdims=True)
    total = float(np.sum(centered * centered))
    if total <= 1e-12:
        return 0.0
    projected = centered @ basis.T
    reconstructed = projected @ basis
    residual = centered - reconstructed
    residual_energy = float(np.sum(residual * residual))
    return float(max(0.0, 1.0 - residual_energy / total))


def build_algorithm_delta_sets(
    trajectories: np.ndarray,
    selected_graph_indices: np.ndarray,
    dataset_payload: np.lib.npyio.NpzFile,
    selected_tasks: Dict[str, bool],
    membership_mode: str,
) -> Dict[str, dict]:
    algorithm_sets: Dict[str, dict] = {}
    deltas = np.diff(trajectories.astype(np.float64), axis=1)
    total_steps = int(deltas.shape[1])

    for algorithm in ALGORITHMS:
        if not selected_tasks.get(algorithm, False):
            continue
        vectors: List[np.ndarray] = []
        graph_labels: List[int] = []
        step_labels: List[int] = []

        for graph_row, graph_index in enumerate(selected_graph_indices.tolist()):
            step_counts = execution_step_counts_raw(dataset_payload, int(graph_index))
            for step in range(total_steps):
                is_active = step < step_counts[algorithm]
                if membership_mode == "exclusive":
                    is_active = is_active and all(
                        step >= step_counts[other]
                        for other in ALGORITHMS
                        if other != algorithm
                    )
                if not is_active:
                    continue
                vectors.append(deltas[graph_row, step])
                graph_labels.append(int(graph_index))
                step_labels.append(int(step))

        if not vectors:
            continue
        algorithm_sets[algorithm] = {
            "vectors": np.stack(vectors, axis=0).astype(np.float32, copy=False),
            "graph_labels": np.array(graph_labels, dtype=np.int32),
            "step_labels": np.array(step_labels, dtype=np.int32),
        }
    return algorithm_sets


def inactive_vectors_for_algorithm(
    deltas: np.ndarray,
    selected_graph_indices: np.ndarray,
    dataset_payload: np.lib.npyio.NpzFile,
    algorithm: str,
    membership_mode: str,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total_steps = int(deltas.shape[1])
    for graph_row, graph_index in enumerate(selected_graph_indices.tolist()):
        step_counts = execution_step_counts_raw(dataset_payload, int(graph_index))
        for step in range(total_steps):
            is_active = step < step_counts[algorithm]
            if membership_mode == "exclusive":
                is_active = is_active and all(
                    step >= step_counts[other]
                    for other in ALGORITHMS
                    if other != algorithm
                )
            if is_active:
                continue
            vectors.append(deltas[graph_row, step])
    if not vectors:
        return np.empty((0, deltas.shape[2]), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    trajectories_dir = resolve_trajectories_dir(args)
    metadata, trajectories_payload, dataset_payload = load_trajectory_source(
        trajectories_dir=trajectories_dir,
        dataset_override=args.dataset,
    )

    selected_tasks = resolve_selected_tasks(args.tasks)
    selected_graph_indices = np.asarray(
        trajectories_payload["selected_graph_indices"], dtype=np.int32
    )
    trajectories = np.asarray(trajectories_payload["trajectories"], dtype=np.float32)
    terminal_probes = aligned_completion_probes(trajectories_payload, selected_graph_indices)
    extended_trajectories = extend_trajectories_with_terminal_probe(trajectories, terminal_probes)
    deltas = np.diff(extended_trajectories.astype(np.float64), axis=1)

    algorithm_sets = build_algorithm_delta_sets(
        trajectories=extended_trajectories,
        selected_graph_indices=selected_graph_indices,
        dataset_payload=dataset_payload,
        selected_tasks=selected_tasks,
        membership_mode=args.membership_mode,
    )
    if not algorithm_sets:
        raise ValueError("No algorithm deltas matched the requested selection.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (
            Path(args.run_dir) / "analysis" / "algorithm_subspace"
            if args.run_dir is not None
            else Path("analysis/algorithm_subspace")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labels: List[str] = []
    active_scores: List[float] = []
    inactive_scores: List[float] = []
    algorithm_summaries: Dict[str, dict] = {}
    fitted_payloads: Dict[str, dict] = {}
    fitted_components = []

    for algorithm in ALGORITHMS:
        if algorithm not in algorithm_sets:
            continue
        entry = algorithm_sets[algorithm]
        payload, used_components = pca_fit_transform(
            entry["vectors"].astype(np.float64),
            args.pca_components,
        )
        fitted_payloads[algorithm] = payload
        fitted_components.append(payload["components"])
        labels.append(algorithm.upper())

        np.savez(output_dir / f"{algorithm}_delta_pca.npz", **payload)
        plot_algorithm_delta_pca(
            projected=payload["projected"],
            step_indices=entry["step_labels"],
            output_path=output_dir / f"{algorithm}_delta_pca.png",
            title=f"{algorithm.upper()} delta PCA [{args.membership_mode}]",
            explained_ratio=payload["explained_variance_ratio"],
        )

        basis = np.asarray(payload["components"], dtype=np.float64)
        active_score = explained_variance_ratio_in_subspace(
            entry["vectors"].astype(np.float64),
            basis,
        )
        inactive_vectors = inactive_vectors_for_algorithm(
            deltas=deltas,
            selected_graph_indices=selected_graph_indices,
            dataset_payload=dataset_payload,
            algorithm=algorithm,
            membership_mode=args.membership_mode,
        )
        inactive_score = explained_variance_ratio_in_subspace(
            inactive_vectors.astype(np.float64),
            basis,
        ) if inactive_vectors.shape[0] > 0 else 0.0
        active_scores.append(active_score)
        inactive_scores.append(inactive_score)

        unique_steps, counts = np.unique(entry["step_labels"], return_counts=True)
        algorithm_summaries[algorithm] = {
            "count": int(entry["vectors"].shape[0]),
            "step_histogram": {
                str(int(step)): int(count)
                for step, count in zip(unique_steps.tolist(), counts.tolist())
            },
            "pca_components_used": int(used_components),
            "explained_variance_ratio": payload["explained_variance_ratio"].astype(np.float64).tolist(),
            "self_explained_variance_ratio": active_score,
            "inactive_explained_variance_ratio": inactive_score,
        }

    common_subspace_dim = min(arr.shape[0] for arr in fitted_components)
    if args.subspace_dim is not None:
        common_subspace_dim = min(common_subspace_dim, int(args.subspace_dim))
    if common_subspace_dim <= 0:
        raise ValueError("Subspace comparison requires at least one PCA component.")

    overlap_matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    explained_matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    pairwise_details: Dict[str, dict] = {}

    for row, basis_algorithm in enumerate([label.lower() for label in labels]):
        basis = np.asarray(fitted_payloads[basis_algorithm]["components"], dtype=np.float64)[
            :common_subspace_dim
        ]
        for col, eval_algorithm in enumerate([label.lower() for label in labels]):
            eval_vectors = algorithm_sets[eval_algorithm]["vectors"].astype(np.float64)
            explained_matrix[row, col] = explained_variance_ratio_in_subspace(
                eval_vectors,
                basis,
            )
            eval_basis = np.asarray(fitted_payloads[eval_algorithm]["components"], dtype=np.float64)[
                :common_subspace_dim
            ]
            pair_summary = principal_correlation_summary(basis, eval_basis)
            overlap_matrix[row, col] = pair_summary["mean_canonical_correlation"]
            pairwise_details[f"{basis_algorithm}_vs_{eval_algorithm}"] = pair_summary

    np.savetxt(output_dir / "subspace_overlap_mean_canonical_corr.csv", overlap_matrix, delimiter=",", fmt="%.6f")
    np.savetxt(output_dir / "cross_explained_variance.csv", explained_matrix, delimiter=",", fmt="%.6f")
    plot_heatmap(
        matrix=overlap_matrix,
        labels=labels,
        title=f"Subspace overlap [{args.membership_mode}]",
        cbar_label="mean canonical correlation",
        output_path=output_dir / "subspace_overlap_mean_canonical_corr.png",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap(
        matrix=explained_matrix,
        labels=labels,
        title=f"Cross explained variance [{args.membership_mode}]",
        cbar_label="explained variance ratio",
        output_path=output_dir / "cross_explained_variance.png",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    plot_active_vs_inactive_explained_variance(
        labels=labels,
        active_scores=np.array(active_scores, dtype=np.float64),
        inactive_scores=np.array(inactive_scores, dtype=np.float64),
        output_path=output_dir / "active_vs_inactive_explained_variance.png",
    )

    summary = {
        "source_trajectories_dir": str(trajectories_dir),
        "dataset": metadata["dataset"],
        "latent": metadata.get("latent"),
        "node_agg": metadata.get("node_agg"),
        "num_graphs": int(selected_graph_indices.shape[0]),
        "uses_terminal_probe_extension": terminal_probes is not None,
        "membership_mode": args.membership_mode,
        "tasks": args.tasks,
        "labels": labels,
        "common_subspace_dim": int(common_subspace_dim),
        "algorithm_summaries": algorithm_summaries,
        "subspace_overlap_mean_canonical_corr": overlap_matrix.tolist(),
        "cross_explained_variance": explained_matrix.tolist(),
        "pairwise_details": pairwise_details,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
