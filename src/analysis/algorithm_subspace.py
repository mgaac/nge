"""Analyze whether latent delta directions are algorithm-specific subspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.task_specs import (
    SELECT_TASK_CHOICES,
    algorithm_display_name,
    normalize_algorithm_order,
    primary_target_key,
    resolve_selected_tasks,
    supported_algorithms,
)


def parse_algorithm_order_arg(raw_value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError(
            "--algorithm-order must contain at least one comma-separated algorithm."
        )

    valid = set(supported_algorithms())
    invalid = [algorithm for algorithm in items if algorithm not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            "Unknown algorithms in --algorithm-order: " + ", ".join(invalid)
        )

    duplicates = []
    seen = set()
    for algorithm in items:
        if algorithm in seen and algorithm not in duplicates:
            duplicates.append(algorithm)
        seen.add(algorithm)
    if duplicates:
        raise argparse.ArgumentTypeError(
            "Algorithms in --algorithm-order must be unique. Duplicates: "
            + ", ".join(duplicates)
        )
    return items


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
        action="append",
        default=None,
        help=(
            "Directory containing trajectories.npz + metadata.json from "
            "src.analysis.embedding_trajectories. Repeat this flag to merge multiple "
            "artifacts. If omitted, the tool scans <run-dir>/analysis for "
            "embedding_trajectories* directories."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Override dataset path (.npz). Only valid when using a single "
            "embedding_trajectories artifact."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        choices=SELECT_TASK_CHOICES,
        help="Algorithms to analyze: all, bf, bfs, or prim.",
    )
    parser.add_argument(
        "--algorithm-order",
        type=parse_algorithm_order_arg,
        default=None,
        help=(
            "Optional comma-separated order for algorithms in the output matrices "
            "and plots, for example: bf,bfs,prim,dijkstra,dag_shortest_paths. "
            "Must include every algorithm present in the provided sources."
        ),
    )
    parser.add_argument(
        "--membership-mode",
        type=str,
        default="active",
        choices=["active", "exclusive", "isolated_inputs"],
        help=(
            "Use all deltas while an algorithm is active, or only deltas where it is "
            "the sole active algorithm. Use 'isolated_inputs' to consume per-algorithm "
            "embedding_trajectories artifacts generated with "
            "--isolate-input-algorithm and build each basis from its own isolated run."
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


def resolve_trajectories_dirs(args: argparse.Namespace) -> list[Path]:
    if args.trajectories_dir is not None:
        return [Path(path) for path in args.trajectories_dir]
    if args.run_dir is None:
        raise ValueError("Provide --trajectories-dir or --run-dir.")

    analysis_dir = Path(args.run_dir) / "analysis"
    candidates = []
    for path in sorted(analysis_dir.glob("embedding_trajectories*")):
        if not path.is_dir():
            continue
        metadata_path = path / "metadata.json"
        trajectories_path = path / "trajectories.npz"
        if not metadata_path.exists() or not trajectories_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("execution_window", "all") != "all":
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No embedding_trajectories artifacts found under: {analysis_dir}"
        )
    return candidates


def execution_step_counts_raw(
    dataset: np.lib.npyio.NpzFile,
    graph_index: int,
    algorithm_order: tuple[str, ...],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for algorithm in algorithm_order:
        key = primary_target_key(algorithm)
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


def load_trajectory_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    trajectories_dirs = resolve_trajectories_dirs(args)
    if args.dataset is not None and len(trajectories_dirs) > 1:
        raise ValueError(
            "--dataset can only be used with a single embedding_trajectories artifact."
        )

    sources: list[dict[str, Any]] = []
    for trajectories_dir in trajectories_dirs:
        metadata, trajectories_payload, dataset_payload = load_trajectory_source(
            trajectories_dir=trajectories_dir,
            dataset_override=args.dataset,
        )
        algorithm_order = normalize_algorithm_order(metadata.get("algorithms"))
        selected_graph_indices = np.asarray(
            trajectories_payload["selected_graph_indices"], dtype=np.int32
        )
        trajectories = np.asarray(trajectories_payload["trajectories"], dtype=np.float32)
        terminal_probes = aligned_completion_probes(trajectories_payload, selected_graph_indices)
        extended_trajectories = extend_trajectories_with_terminal_probe(
            trajectories, terminal_probes
        )
        sources.append(
            {
                "dir": trajectories_dir,
                "metadata": metadata,
                "trajectories_payload": trajectories_payload,
                "dataset_payload": dataset_payload,
                "algorithm_order": algorithm_order,
                "selected_graph_indices": selected_graph_indices,
                "extended_trajectories": extended_trajectories,
                "terminal_probes": terminal_probes,
            }
        )

    latent_values = {str(source["metadata"].get("latent")) for source in sources}
    node_agg_values = {str(source["metadata"].get("node_agg")) for source in sources}
    if len(latent_values) > 1 or len(node_agg_values) > 1:
        raise ValueError(
            "All merged embedding_trajectories artifacts must use the same latent and node_agg. "
            f"Found latents={sorted(latent_values)}, node_aggs={sorted(node_agg_values)}."
        )
    return sources


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


def source_deltas(source: dict[str, Any]) -> np.ndarray:
    return np.diff(source["extended_trajectories"].astype(np.float64), axis=1)


def source_isolated_algorithm(source: dict[str, Any]) -> str | None:
    isolated_algorithm = source["metadata"].get("isolate_input_algorithm")
    if isolated_algorithm is None:
        return None
    return str(isolated_algorithm)


def build_algorithm_delta_sets(
    sources: list[dict[str, Any]],
    selected_tasks: Dict[str, bool],
    membership_mode: str,
    algorithm_order: tuple[str, ...],
) -> Dict[str, dict]:
    if membership_mode == "isolated_inputs":
        algorithm_sets: Dict[str, dict] = {}
        for algorithm in algorithm_order:
            if not selected_tasks.get(algorithm, False):
                continue
            vectors: List[np.ndarray] = []
            graph_labels: List[int] = []
            step_labels: List[int] = []
            graph_offset = 0
            matching_sources = [
                source
                for source in sources
                if source_isolated_algorithm(source) == algorithm
            ]
            for source in matching_sources:
                source_algorithm_order = tuple(source["algorithm_order"])
                deltas = source_deltas(source)
                total_steps = int(deltas.shape[1])
                selected_graph_indices = np.asarray(
                    source["selected_graph_indices"], dtype=np.int32
                )
                dataset_payload = source["dataset_payload"]
                for graph_row, graph_index in enumerate(selected_graph_indices.tolist()):
                    step_counts = execution_step_counts_raw(
                        dataset_payload, int(graph_index), source_algorithm_order
                    )
                    active_steps = min(total_steps, step_counts.get(algorithm, 0))
                    for step in range(active_steps):
                        vectors.append(deltas[graph_row, step])
                        graph_labels.append(graph_offset + int(graph_index))
                        step_labels.append(int(step))
                graph_offset += int(selected_graph_indices.shape[0]) + 100000

            if not vectors:
                continue
            algorithm_sets[algorithm] = {
                "vectors": np.stack(vectors, axis=0).astype(np.float32, copy=False),
                "graph_labels": np.array(graph_labels, dtype=np.int32),
                "step_labels": np.array(step_labels, dtype=np.int32),
            }
        return algorithm_sets

    algorithm_sets: Dict[str, dict] = {}
    for algorithm in algorithm_order:
        if not selected_tasks.get(algorithm, False):
            continue
        vectors: List[np.ndarray] = []
        graph_labels: List[int] = []
        step_labels: List[int] = []
        graph_offset = 0
        for source in sources:
            source_algorithm_order = tuple(source["algorithm_order"])
            deltas = source_deltas(source)
            total_steps = int(deltas.shape[1])
            selected_graph_indices = np.asarray(source["selected_graph_indices"], dtype=np.int32)
            dataset_payload = source["dataset_payload"]
            for graph_row, graph_index in enumerate(selected_graph_indices.tolist()):
                step_counts = execution_step_counts_raw(
                    dataset_payload, int(graph_index), source_algorithm_order
                )
                for step in range(total_steps):
                    is_active = step < step_counts.get(algorithm, 0)
                    if membership_mode == "exclusive":
                        is_active = is_active and all(
                            step >= step_counts.get(other, 0)
                            for other in source_algorithm_order
                            if other != algorithm
                        )
                    if not is_active:
                        continue
                    vectors.append(deltas[graph_row, step])
                    graph_labels.append(graph_offset + int(graph_index))
                    step_labels.append(int(step))
            graph_offset += int(selected_graph_indices.shape[0]) + 100000

        if not vectors:
            continue
        algorithm_sets[algorithm] = {
            "vectors": np.stack(vectors, axis=0).astype(np.float32, copy=False),
            "graph_labels": np.array(graph_labels, dtype=np.int32),
            "step_labels": np.array(step_labels, dtype=np.int32),
        }
    return algorithm_sets


def inactive_vectors_for_algorithm(
    sources: list[dict[str, Any]],
    algorithm: str,
    membership_mode: str,
    algorithm_order: tuple[str, ...],
) -> np.ndarray:
    if membership_mode == "isolated_inputs":
        latent_dim = int(sources[0]["extended_trajectories"].shape[2])
        return np.empty((0, latent_dim), dtype=np.float32)

    vectors: List[np.ndarray] = []
    for source in sources:
        source_algorithm_order = tuple(source["algorithm_order"])
        deltas = source_deltas(source)
        total_steps = int(deltas.shape[1])
        selected_graph_indices = np.asarray(source["selected_graph_indices"], dtype=np.int32)
        dataset_payload = source["dataset_payload"]
        for graph_row, graph_index in enumerate(selected_graph_indices.tolist()):
            step_counts = execution_step_counts_raw(
                dataset_payload, int(graph_index), source_algorithm_order
            )
            for step in range(total_steps):
                is_active = step < step_counts.get(algorithm, 0)
                if membership_mode == "exclusive":
                    is_active = is_active and all(
                        step >= step_counts.get(other, 0)
                        for other in source_algorithm_order
                        if other != algorithm
                    )
                if is_active:
                    continue
                vectors.append(deltas[graph_row, step])
    if not vectors:
        latent_dim = int(sources[0]["extended_trajectories"].shape[2])
        return np.empty((0, latent_dim), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32, copy=False)


def validate_isolated_input_sources(
    sources: list[dict[str, Any]],
    requested_algorithms: list[str],
) -> None:
    missing_metadata = [
        str(source["dir"])
        for source in sources
        if source_isolated_algorithm(source) is None
    ]
    if missing_metadata:
        raise ValueError(
            "membership_mode=isolated_inputs requires every source artifact to include "
            "metadata.isolate_input_algorithm. Missing in: "
            + ", ".join(missing_metadata)
        )

    available_algorithms = {
        source_isolated_algorithm(source)
        for source in sources
        if source_isolated_algorithm(source) is not None
    }
    missing_algorithms = [
        algorithm
        for algorithm in requested_algorithms
        if algorithm not in available_algorithms
    ]
    if missing_algorithms:
        raise ValueError(
            "membership_mode=isolated_inputs is missing sources for: "
            + ", ".join(missing_algorithms)
        )


def main() -> None:
    args = parse_args()
    sources = load_trajectory_sources(args)
    source_dirs = [Path(source["dir"]) for source in sources]
    present_algorithms = [
        algorithm
        for algorithm in supported_algorithms()
        if any(algorithm in source["algorithm_order"] for source in sources)
    ]
    algorithm_order = normalize_algorithm_order(present_algorithms)
    if args.algorithm_order is not None:
        missing_present = [
            algorithm for algorithm in present_algorithms if algorithm not in args.algorithm_order
        ]
        if missing_present:
            raise ValueError(
                "--algorithm-order must include every algorithm present in the provided "
                "sources. Missing: " + ", ".join(missing_present)
            )
        algorithm_order = tuple(
            algorithm for algorithm in args.algorithm_order if algorithm in present_algorithms
        )
    selected_tasks = resolve_selected_tasks(args.tasks, algorithm_order)
    requested_algorithms = [
        algorithm for algorithm in algorithm_order if selected_tasks.get(algorithm, False)
    ]
    if args.membership_mode == "isolated_inputs":
        validate_isolated_input_sources(sources, requested_algorithms)

    algorithm_sets = build_algorithm_delta_sets(
        sources=sources,
        selected_tasks=selected_tasks,
        membership_mode=args.membership_mode,
        algorithm_order=algorithm_order,
    )
    omitted_algorithms = [
        algorithm for algorithm in requested_algorithms if algorithm not in algorithm_sets
    ]
    if not algorithm_sets:
        raise ValueError(
            "No algorithm deltas matched the requested selection. "
            "This usually means the embedding_trajectories artifact was generated on a "
            "dataset that does not contain the requested algorithms. "
            f"Requested: {requested_algorithms}. "
            f"Sources: {[str(path) for path in source_dirs]}."
        )
    if omitted_algorithms:
        print(
            "Skipping algorithms with no active deltas in the source trajectories: "
            + ", ".join(omitted_algorithms)
        )

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

    for algorithm in algorithm_order:
        if algorithm not in algorithm_sets:
            continue
        entry = algorithm_sets[algorithm]
        payload, used_components = pca_fit_transform(
            entry["vectors"].astype(np.float64),
            args.pca_components,
        )
        fitted_payloads[algorithm] = payload
        fitted_components.append(payload["components"])
        labels.append(algorithm_display_name(algorithm))

        np.savez(output_dir / f"{algorithm}_delta_pca.npz", **payload)
        plot_algorithm_delta_pca(
            projected=payload["projected"],
            step_indices=entry["step_labels"],
            output_path=output_dir / f"{algorithm}_delta_pca.png",
            title=f"{algorithm_display_name(algorithm)} delta PCA [{args.membership_mode}]",
            explained_ratio=payload["explained_variance_ratio"],
        )

        basis = np.asarray(payload["components"], dtype=np.float64)
        active_score = explained_variance_ratio_in_subspace(
            entry["vectors"].astype(np.float64),
            basis,
        )
        inactive_vectors = inactive_vectors_for_algorithm(
            sources=sources,
            algorithm=algorithm,
            membership_mode=args.membership_mode,
            algorithm_order=algorithm_order,
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

    active_algorithms = [algorithm for algorithm in algorithm_order if algorithm in algorithm_sets]
    for row, basis_algorithm in enumerate(active_algorithms):
        basis = np.asarray(fitted_payloads[basis_algorithm]["components"], dtype=np.float64)[
            :common_subspace_dim
        ]
        for col, eval_algorithm in enumerate(active_algorithms):
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

    if args.membership_mode == "isolated_inputs":
        unique_graph_refs = {
            (str(source["metadata"]["dataset"]), int(graph_index))
            for source in sources
            for graph_index in np.asarray(source["selected_graph_indices"], dtype=np.int32).tolist()
        }
        summary_num_graphs = int(len(unique_graph_refs))
    else:
        summary_num_graphs = int(
            sum(int(np.asarray(source["selected_graph_indices"]).shape[0]) for source in sources)
        )

    summary = {
        "source_trajectories_dirs": [str(path) for path in source_dirs],
        "source_datasets": [source["metadata"]["dataset"] for source in sources],
        "algorithms": list(algorithm_order),
        "latent": sources[0]["metadata"].get("latent"),
        "node_agg": sources[0]["metadata"].get("node_agg"),
        "num_graphs": summary_num_graphs,
        "uses_terminal_probe_extension": any(
            source["terminal_probes"] is not None for source in sources
        ),
        "membership_mode": args.membership_mode,
        "source_isolate_input_algorithms": [
            source_isolated_algorithm(source) for source in sources
        ],
        "tasks": args.tasks,
        "algorithm_order": list(algorithm_order),
        "requested_algorithms": requested_algorithms,
        "omitted_algorithms": omitted_algorithms,
        "num_sources": len(sources),
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
