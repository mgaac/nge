"""Step-wise geometric diagnostics for embedding trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.analysis.common import load_analysis_dataset
from src.utils.task_specs import execution_step_counts, normalize_algorithm_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run step-wise geometry diagnostics over an embedding_trajectories artifact: "
            "global PCA projection, per-step covariance spectra, per-algorithm spectra, "
            "and consecutive-step subspace drift via principal angles."
        )
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        required=True,
        help="Directory containing trajectories.npz and metadata.json.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Override dataset path; defaults to metadata.dataset.",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=3,
        help="Number of leading components/eigen-directions to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory; defaults to "
            "<trajectories-dir>/step_dynamics_diagnostics_k<components>."
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def pca_from_centered(centered: np.ndarray, components: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return principal axes and explained-variance ratios from centered data."""
    num_samples = centered.shape[0]
    if num_samples < 2:
        return np.zeros((0, centered.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.float64)

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(vh.shape[0], components)
    if rank <= 0:
        return np.zeros((0, centered.shape[1]), dtype=np.float64), np.zeros(0, dtype=np.float64)
    eigenvalues = (singular_values * singular_values) / (num_samples - 1)
    total = float(np.sum(eigenvalues))
    if total > 0.0:
        ratios = eigenvalues[:rank] / total
    else:
        ratios = np.zeros(rank, dtype=np.float64)
    return vh[:rank], ratios


def covariance_spectrum_topk(data: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return top-k covariance eigenvalues and cumulative explained-variance ratios."""
    if data.ndim != 2:
        raise ValueError("Expected 2D matrix for covariance spectrum.")
    num_samples, dim = data.shape
    k = min(topk, dim)
    if num_samples < 2 or k <= 0:
        return np.full(k, np.nan, dtype=np.float64), np.full(k, np.nan, dtype=np.float64)

    centered = data - data.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / float(num_samples - 1)
    evals = np.linalg.eigvalsh(covariance)[::-1]
    total = float(np.sum(evals))
    top_evals = evals[:k]
    if total > 0.0:
        cumulative = np.cumsum(top_evals / total)
    else:
        cumulative = np.zeros(k, dtype=np.float64)
    return top_evals, cumulative


def principal_angles_degrees(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Compute principal angles (in degrees) between two orthonormal subspaces."""
    if basis_a.ndim != 2 or basis_b.ndim != 2:
        raise ValueError("Bases must be 2D.")
    if basis_a.shape[1] != basis_b.shape[1]:
        raise ValueError("Bases must share ambient dimension.")
    if basis_a.shape[0] == 0 or basis_b.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    overlap = basis_a @ basis_b.T
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    angles = np.degrees(np.arccos(singular_values))
    return angles.astype(np.float64, copy=False)


def plot_cumulative_variance(
    steps: np.ndarray,
    cumulative: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, cumulative[:, 0], marker="o", linewidth=1.8, label=r"$\lambda_1$")
    if cumulative.shape[1] >= 2:
        ax.plot(
            steps,
            cumulative[:, 1],
            marker="o",
            linewidth=1.8,
            label=r"$\lambda_1 + \lambda_2$",
        )
    if cumulative.shape[1] >= 3:
        ax.plot(
            steps,
            cumulative[:, 2],
            marker="o",
            linewidth=1.8,
            label=r"$\lambda_1 + \lambda_2 + \lambda_3$",
        )
    ax.set_xlabel("Execution step t")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("Per-step covariance cumulative variance over time")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_global_basis_cumulative_variance(
    steps: np.ndarray,
    cumulative: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, cumulative[:, 0], marker="o", linewidth=1.8, label=r"global PC1")
    if cumulative.shape[1] >= 2:
        ax.plot(
            steps,
            cumulative[:, 1],
            marker="o",
            linewidth=1.8,
            label=r"global PC1+PC2",
        )
    if cumulative.shape[1] >= 3:
        ax.plot(
            steps,
            cumulative[:, 2],
            marker="o",
            linewidth=1.8,
            label=r"global PC1+PC2+PC3",
        )
    ax.set_xlabel("Execution step t")
    ax.set_ylabel("Cumulative variance ratio in fixed global basis")
    ax.set_title("Step clouds projected into one global PCA basis")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_principal_angle_drift(
    transition_steps: np.ndarray,
    angles: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in range(angles.shape[1]):
        ax.plot(
            transition_steps,
            angles[:, idx],
            marker="o",
            linewidth=1.4,
            alpha=0.9,
            label=f"angle {idx + 1}",
        )
    ax.plot(
        transition_steps,
        angles.mean(axis=1),
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="mean angle",
    )
    ax.set_xlabel("Transition step t→t+1")
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title("Consecutive-step subspace drift")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_algorithm_cumvar3(
    steps: np.ndarray,
    algorithm_order: tuple[str, ...],
    cumulative_by_algo: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for algo_index, algorithm in enumerate(algorithm_order):
        series = cumulative_by_algo[algo_index, :, -1]
        ax.plot(
            steps,
            series,
            marker="o",
            linewidth=1.6,
            markersize=3.2,
            alpha=0.95,
            label=algorithm,
        )
    ax.set_xlabel("Execution step t")
    ax.set_ylabel("Cumulative explained variance ratio (top-3)")
    ax.set_title("Per-algorithm active-set covariance concentration")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.components <= 0:
        raise ValueError("--components must be positive.")

    trajectories_dir = args.trajectories_dir
    trajectories_path = trajectories_dir / "trajectories.npz"
    metadata_path = trajectories_dir / "metadata.json"
    if not trajectories_path.exists():
        raise FileNotFoundError(f"Missing trajectories file: {trajectories_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path) as handle:
        metadata = json.load(handle)
    algorithm_order = normalize_algorithm_order(metadata.get("algorithms"))

    dataset_path = args.dataset
    if dataset_path is None:
        dataset_field = metadata.get("dataset")
        if not dataset_field:
            raise ValueError("Dataset path missing in metadata; provide --dataset.")
        dataset_path = Path(dataset_field)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else trajectories_dir / f"step_dynamics_diagnostics_k{args.components}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_payload = np.load(trajectories_path)
    trajectories = np.asarray(trajectory_payload["trajectories"], dtype=np.float64)
    if trajectories.ndim != 3:
        raise ValueError(
            "Expected trajectories to have shape [num_graphs, num_steps, latent_dim], "
            f"found {trajectories.shape}."
        )
    num_graphs, num_steps, latent_dim = trajectories.shape
    components = min(args.components, num_graphs, latent_dim)
    if components <= 0:
        raise ValueError("Not enough samples/dimensions to compute requested components.")

    selected_graph_indices = np.asarray(
        trajectory_payload["selected_graph_indices"], dtype=np.int32
    )
    if selected_graph_indices.shape[0] != num_graphs:
        raise ValueError(
            "Mismatch between selected_graph_indices and trajectory graph dimension."
        )

    dataset = load_analysis_dataset(dataset_path, algorithm_order)
    selected_graphs = [dataset[int(idx)] for idx in selected_graph_indices.tolist()]
    execution_counts = np.array(
        [
            [execution_step_counts(graph, algorithm_order)[algorithm] for algorithm in algorithm_order]
            for graph in selected_graphs
        ],
        dtype=np.int32,
    )

    # 1) Global PCA over all graphs and all steps.
    global_matrix = trajectories.reshape(num_graphs * num_steps, latent_dim)
    global_mean = global_matrix.mean(axis=0, keepdims=True)
    global_centered = global_matrix - global_mean
    global_basis, global_evr = pca_from_centered(global_centered, components)
    global_projected = (global_centered @ global_basis.T).reshape(num_graphs, num_steps, components)

    # Variance of each step-cloud along fixed global basis directions.
    global_step_axis_variance = np.zeros((num_steps, components), dtype=np.float64)
    global_step_axis_ratio = np.zeros((num_steps, components), dtype=np.float64)
    for step in range(num_steps):
        step_centered_global = trajectories[:, step, :] - global_mean
        axis_scores = step_centered_global @ global_basis.T
        axis_var = axis_scores.var(axis=0, ddof=1) if num_graphs > 1 else np.zeros(components)
        total_var = float(np.var(step_centered_global, axis=0, ddof=1).sum()) if num_graphs > 1 else 0.0
        global_step_axis_variance[step] = axis_var
        if total_var > 0.0:
            global_step_axis_ratio[step] = axis_var / total_var
        else:
            global_step_axis_ratio[step] = 0.0
    global_step_axis_cumulative = np.cumsum(global_step_axis_ratio, axis=1)

    # 2) Per-algorithm covariance spectra at each step, using active-graph subsets.
    num_algorithms = len(algorithm_order)
    per_algorithm_top_eigvals = np.full(
        (num_algorithms, num_steps, components), np.nan, dtype=np.float64
    )
    per_algorithm_cumulative = np.full(
        (num_algorithms, num_steps, components), np.nan, dtype=np.float64
    )
    per_algorithm_active_counts = np.zeros((num_algorithms, num_steps), dtype=np.int32)

    # 3) Step-local covariance cumulative variance (all graphs).
    step_local_top_eigvals = np.zeros((num_steps, components), dtype=np.float64)
    step_local_cumulative = np.zeros((num_steps, components), dtype=np.float64)
    step_local_bases = np.zeros((num_steps, components, latent_dim), dtype=np.float64)
    step_local_basis_rank = np.zeros(num_steps, dtype=np.int32)

    for step in range(num_steps):
        step_data = trajectories[:, step, :]
        eigvals, cumulative = covariance_spectrum_topk(step_data, components)
        step_local_top_eigvals[step] = eigvals
        step_local_cumulative[step] = cumulative

        step_centered = step_data - step_data.mean(axis=0, keepdims=True)
        basis, _ = pca_from_centered(step_centered, components)
        rank = basis.shape[0]
        step_local_basis_rank[step] = rank
        if rank > 0:
            step_local_bases[step, :rank, :] = basis

        for algorithm_index, algorithm in enumerate(algorithm_order):
            active_mask = step < execution_counts[:, algorithm_index]
            active_count = int(np.sum(active_mask))
            per_algorithm_active_counts[algorithm_index, step] = active_count
            if active_count < 2:
                continue
            algo_data = step_data[active_mask]
            algo_eigvals, algo_cumulative = covariance_spectrum_topk(algo_data, components)
            per_algorithm_top_eigvals[algorithm_index, step] = algo_eigvals
            per_algorithm_cumulative[algorithm_index, step] = algo_cumulative

    # 4) Subspace drift via principal angles between consecutive steps.
    principal_angles = np.full((num_steps - 1, components), np.nan, dtype=np.float64)
    for step in range(num_steps - 1):
        rank_left = int(step_local_basis_rank[step])
        rank_right = int(step_local_basis_rank[step + 1])
        rank = min(rank_left, rank_right, components)
        if rank <= 0:
            continue
        basis_left = step_local_bases[step, :rank, :]
        basis_right = step_local_bases[step + 1, :rank, :]
        angles = principal_angles_degrees(basis_left, basis_right)
        principal_angles[step, : angles.shape[0]] = angles

    np.savez(
        output_dir / "diagnostics.npz",
        global_basis=global_basis.astype(np.float32, copy=False),
        global_mean=global_mean.squeeze(0).astype(np.float32, copy=False),
        global_explained_variance_ratio=global_evr.astype(np.float32, copy=False),
        global_projected=global_projected.astype(np.float32, copy=False),
        global_step_axis_variance=global_step_axis_variance.astype(np.float32, copy=False),
        global_step_axis_ratio=global_step_axis_ratio.astype(np.float32, copy=False),
        global_step_axis_cumulative=global_step_axis_cumulative.astype(np.float32, copy=False),
        step_local_top_eigvals=step_local_top_eigvals.astype(np.float32, copy=False),
        step_local_cumulative=step_local_cumulative.astype(np.float32, copy=False),
        per_algorithm_top_eigvals=per_algorithm_top_eigvals.astype(np.float32, copy=False),
        per_algorithm_cumulative=per_algorithm_cumulative.astype(np.float32, copy=False),
        per_algorithm_active_counts=per_algorithm_active_counts,
        principal_angles=principal_angles.astype(np.float32, copy=False),
    )

    steps = np.arange(num_steps, dtype=np.int32)
    transition_steps = np.arange(num_steps - 1, dtype=np.int32)
    plot_cumulative_variance(
        steps=steps,
        cumulative=step_local_cumulative,
        output_path=output_dir / "cumulative_variance_over_time.png",
    )
    plot_global_basis_cumulative_variance(
        steps=steps,
        cumulative=global_step_axis_cumulative,
        output_path=output_dir / "global_basis_cumulative_variance_over_time.png",
    )
    plot_principal_angle_drift(
        transition_steps=transition_steps,
        angles=principal_angles,
        output_path=output_dir / "principal_angle_drift.png",
    )
    plot_per_algorithm_cumvar3(
        steps=steps,
        algorithm_order=algorithm_order,
        cumulative_by_algo=per_algorithm_cumulative,
        output_path=output_dir / "per_algorithm_cumulative_variance_top3.png",
    )

    summary = {
        "source_trajectories_dir": str(trajectories_dir),
        "source_dataset": str(dataset_path),
        "algorithm_order": list(algorithm_order),
        "num_graphs": int(num_graphs),
        "num_steps": int(num_steps),
        "latent_dim": int(latent_dim),
        "components_requested": int(args.components),
        "components_used": int(components),
        "global_explained_variance_ratio": global_evr.astype(np.float64, copy=False).tolist(),
        "global_step_axis_cumulative_variance": global_step_axis_cumulative.astype(
            np.float64, copy=False
        ).tolist(),
        "step_local_cumulative_variance": step_local_cumulative.astype(
            np.float64, copy=False
        ).tolist(),
        "per_algorithm_active_counts": {
            algorithm: per_algorithm_active_counts[index].tolist()
            for index, algorithm in enumerate(algorithm_order)
        },
        "per_algorithm_cumulative_variance_top3": {
            algorithm: per_algorithm_cumulative[index].astype(np.float64, copy=False).tolist()
            for index, algorithm in enumerate(algorithm_order)
        },
        "principal_angles_degrees": principal_angles.astype(np.float64, copy=False).tolist(),
        "files": {
            "diagnostics_npz": "diagnostics.npz",
            "cumulative_variance_plot": "cumulative_variance_over_time.png",
            "global_basis_cumulative_variance_plot": "global_basis_cumulative_variance_over_time.png",
            "principal_angle_drift_plot": "principal_angle_drift.png",
            "per_algorithm_cumvar_plot": "per_algorithm_cumulative_variance_top3.png",
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Saved step dynamics diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
