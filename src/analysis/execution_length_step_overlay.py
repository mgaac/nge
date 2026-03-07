"""Overlay average step PCA trajectories across fixed execution lengths.

This utility runs `src.analysis.embedding_trajectories` for each fixed step
count in a range, then overlays only the average step coordinates per run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.common import (
    load_model_from_checkpoint,
    resolve_checkpoint_path,
    resolve_config,
    resolve_dataset_path,
)
from src.analysis.embedding_trajectories import collect_graph_trajectory, count_execution_steps
from src.data import load_dataset
from src.utils.task_specs import ANALYSIS_LATENT_CHOICES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run embedding_trajectories for a step range and overlay average step "
            "coordinates per run."
        )
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset override path (.npz).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Split to use when --dataset is not provided.",
    )
    parser.add_argument(
        "--latent",
        type=str,
        default="processed",
        choices=ANALYSIS_LATENT_CHOICES,
        help="Latent type forwarded to embedding_trajectories.",
    )
    parser.add_argument(
        "--node-agg",
        type=str,
        default="max",
        choices=["max", "min", "mean"],
        help="Node aggregation for embedding_trajectories.",
    )
    parser.add_argument(
        "--steps-min",
        type=int,
        default=0,
        help="Minimum fixed execution length to evaluate.",
    )
    parser.add_argument(
        "--steps-max",
        type=int,
        default=8,
        help="Maximum fixed execution length to evaluate.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=2,
        help="Number of PCA components for step-wise PCA.",
    )
    parser.add_argument(
        "--extra-steps",
        type=int,
        default=0,
        help=(
            "Forwarded to embedding_trajectories. Fake-continue each graph for "
            "N additional steps after termination."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for sweep artifacts and overlay. "
            "Default: <run-dir>/analysis/execution_length_step_overlay"
        ),
    )
    parser.add_argument(
        "--keep-step-plots",
        action="store_true",
        help="Forward --plot to per-step embedding_trajectories runs.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail if a step length has no matching graphs.",
    )
    parser.add_argument(
        "--probe-heatmaps",
        action="store_true",
        help=(
            "Also compute/save pairwise terminal-probe closeness heatmaps "
            "(L2 distance and cosine similarity)."
        ),
    )
    parser.add_argument(
        "--pca-alignment-heatmaps",
        action="store_true",
        help=(
            "Also compute/save pairwise PCA-direction alignment heatmaps "
            "for corresponding principal components."
        ),
    )
    return parser.parse_args()


def run_single_step(args: argparse.Namespace, base_step_count: int, output_dir: Path) -> bool:
    total_step_count = base_step_count + args.extra_steps
    cmd = [
        sys.executable,
        "-m",
        "src.analysis.embedding_trajectories",
        "--run-dir",
        args.run_dir,
        "--split",
        args.split,
        "--latent",
        args.latent,
        "--node-agg",
        args.node_agg,
        "--step-policy",
        "fixed",
        "--steps",
        str(total_step_count),
        "--pca",
        "step",
        "--pca-components",
        str(args.pca_components),
        "--extra-steps",
        str(args.extra_steps),
        "--output-dir",
        str(output_dir),
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.keep_step_plots:
        cmd.append("--plot")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[ok] base_steps={base_step_count} total_steps={total_step_count}")
        return True

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    combined = f"{stdout}\n{stderr}"
    missing_msg = "No graphs matched step count"
    if missing_msg in combined and not args.fail_on_missing:
        print(
            f"[skip] base_steps={base_step_count} total_steps={total_step_count} "
            "(no matching graphs)"
        )
        return False

    raise RuntimeError(
        f"embedding_trajectories failed for base_steps={base_step_count} "
        f"(total_steps={total_step_count})\n"
        f"command: {' '.join(cmd)}\n"
        f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
    )


def load_step_means(step_dir: Path) -> Dict[str, Any]:
    metadata_path = step_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    means = metadata.get("step_pca_mean_coordinates")
    if means is None:
        raise ValueError(
            "step_pca_mean_coordinates missing. Ensure embedding_trajectories ran with --pca step|both."
        )
    return metadata


def completion_average_in_step_pca(
    *,
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    step_dir: Path,
    model: Any,
    dataset: list[dict],
    embed_dim: int,
) -> List[float] | None:
    selected_indices = metadata.get("selected_graph_indices")
    if not selected_indices:
        return None

    completion_vectors: List[np.ndarray] = []
    expected_base_steps = int(metadata["target_steps"]) - int(args.extra_steps)

    for graph_index in selected_indices:
        graph = dataset[int(graph_index)]
        base_steps = count_execution_steps(graph, extra_steps=0)
        if base_steps != expected_base_steps:
            continue

        # Probe one extra transition after true termination inputs.
        completion_trajectory = collect_graph_trajectory(
            model=model,
            graph_data=graph,
            embed_dim=embed_dim,
            latent_kind=args.latent,
            node_agg=args.node_agg,
            extra_steps=1,
        )
        if completion_trajectory.shape[0] == 0:
            continue
        completion_vectors.append(completion_trajectory[-1])

    if not completion_vectors:
        return None

    completion_matrix = np.asarray(np.stack(completion_vectors, axis=0), dtype=np.float64)
    pca_payload = np.load(step_dir / "pca_step.npz")
    components = np.asarray(pca_payload["components"], dtype=np.float64)
    mean = np.asarray(pca_payload["mean"], dtype=np.float64)
    projected = (completion_matrix - mean) @ components.T
    avg_coord = projected.mean(axis=0)
    return [float(v) for v in avg_coord.tolist()]


def plot_overlay(series: List[Dict[str, Any]], output_path: Path, extra_steps: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    if not series:
        raise ValueError("No step series to plot.")

    step_values = [item["base_steps"] for item in series]
    min_step = min(step_values)
    max_step = max(step_values)
    denom = max(max_step - min_step, 1)

    cmap = plt.get_cmap("viridis")

    for item in sorted(series, key=lambda x: x["base_steps"]):
        coords = np.array(
            [entry["mean_coordinate"] for entry in item["step_pca_mean_coordinates"]],
            dtype=np.float64,
        )
        if coords.shape[0] == 0:
            continue
        color = cmap((item["base_steps"] - min_step) / denom)
        if extra_steps > 0:
            label = (
                f"base={item['base_steps']}, total={item['total_steps']} "
                f"(n={item['num_graphs']})"
            )
        else:
            label = f"steps={item['base_steps']} (n={item['num_graphs']})"
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            marker="o",
            linewidth=1.6,
            markersize=4,
            alpha=0.95,
            color=color,
            label=label,
        )
        completion_coord = item.get("completion_avg_coordinate")
        if completion_coord is not None:
            completion = np.array(completion_coord, dtype=np.float64)
            ax.plot(
                [coords[-1, 0], completion[0]],
                [coords[-1, 1], completion[1]],
                linestyle="--",
                linewidth=0.8,
                alpha=0.55,
                color=color,
            )
            ax.scatter(
                [completion[0]],
                [completion[1]],
                marker="X",
                s=88,
                color=color,
                edgecolors="black",
                linewidths=0.7,
                alpha=0.98,
                label=f"terminal-probe base={item['base_steps']}",
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if extra_steps > 0:
        ax.set_title(
            "Average Step Coordinates by Base Execution Length "
            f"(+{extra_steps} extra steps)"
        )
    else:
        ax.set_title("Average Step Coordinates by Execution Length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _extract_probe_series(series: List[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    valid = [item for item in series if item.get("completion_avg_coordinate") is not None]
    if not valid:
        return np.empty((0,), dtype=np.int32), np.empty((0, 2), dtype=np.float64)
    base_steps = np.array([int(item["base_steps"]) for item in valid], dtype=np.int32)
    coords = np.array(
        [item["completion_avg_coordinate"] for item in valid], dtype=np.float64
    )
    return base_steps, coords


def _pairwise_l2(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = coords - coords[i]
        out[i] = np.linalg.norm(diff, axis=1)
    return out


def _pairwise_cosine_similarity(coords: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    denom = np.maximum(norms @ norms.T, 1e-12)
    sim = (coords @ coords.T) / denom
    return np.clip(sim, -1.0, 1.0)


def _plot_matrix_heatmap(
    matrix: np.ndarray,
    base_steps: np.ndarray,
    title: str,
    cbar_label: str,
    output_path: Path,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    tick_labels = [str(int(step)) for step in base_steps]
    ax.set_xticks(np.arange(len(base_steps)))
    ax.set_yticks(np.arange(len(base_steps)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Base step")
    ax.set_ylabel("Base step")
    ax.set_title(title)

    threshold = float(np.nanmax(matrix)) * 0.55 if matrix.size > 0 else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value > threshold else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_probe_matrices(
    series: List[Dict[str, Any]], root_output: Path
) -> Dict[str, Any] | None:
    base_steps, coords = _extract_probe_series(series)
    if coords.shape[0] == 0:
        return None

    l2_matrix = _pairwise_l2(coords)
    cosine_matrix = _pairwise_cosine_similarity(coords)

    l2_csv = root_output / "terminal_probe_pairwise_l2_matrix.csv"
    cosine_csv = root_output / "terminal_probe_pairwise_cosine_similarity_matrix.csv"
    l2_heatmap = root_output / "terminal_probe_pairwise_l2_matrix_heatmap.png"
    cosine_heatmap = (
        root_output / "terminal_probe_pairwise_cosine_similarity_matrix_heatmap.png"
    )
    meta_json = root_output / "terminal_probe_pairwise_matrix_meta.json"

    np.savetxt(l2_csv, l2_matrix, delimiter=",", fmt="%.6f")
    np.savetxt(cosine_csv, cosine_matrix, delimiter=",", fmt="%.6f")

    _plot_matrix_heatmap(
        matrix=l2_matrix,
        base_steps=base_steps,
        title="Terminal-probe pairwise L2 distance",
        cbar_label="L2 distance",
        output_path=l2_heatmap,
        cmap="magma",
    )
    _plot_matrix_heatmap(
        matrix=cosine_matrix,
        base_steps=base_steps,
        title="Terminal-probe pairwise cosine similarity",
        cbar_label="cosine similarity",
        output_path=cosine_heatmap,
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )

    meta_payload = {
        "base_steps": [int(step) for step in base_steps.tolist()],
        "num_probes": int(coords.shape[0]),
        "l2_matrix_csv": str(l2_csv),
        "l2_heatmap_png": str(l2_heatmap),
        "cosine_similarity_matrix_csv": str(cosine_csv),
        "cosine_similarity_heatmap_png": str(cosine_heatmap),
    }
    meta_json.write_text(json.dumps(meta_payload, indent=2))
    return meta_payload


def _extract_pca_component_series(series: List[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    valid: List[tuple[int, np.ndarray]] = []
    for item in series:
        pca_step_path = Path(item["pca_step_path"])
        if not pca_step_path.exists():
            continue
        payload = np.load(pca_step_path)
        components = np.asarray(payload["components"], dtype=np.float64)
        valid.append((int(item["base_steps"]), components))

    if not valid:
        return np.empty((0,), dtype=np.int32), np.empty((0, 0, 0), dtype=np.float64)

    component_counts = {components.shape[0] for _, components in valid}
    if len(component_counts) != 1:
        raise ValueError(
            "Per-run PCA components disagree in count; cannot compute alignment matrices."
        )

    ordered = sorted(valid, key=lambda x: x[0])
    base_steps = np.array([step for step, _ in ordered], dtype=np.int32)
    stacked = np.stack([components for _, components in ordered], axis=0)
    return base_steps, stacked


def _pairwise_corresponding_component_metrics(
    stacked_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    runs, component_count, _ = stacked_components.shape
    l2 = np.zeros((component_count, runs, runs), dtype=np.float64)
    cosine = np.zeros((component_count, runs, runs), dtype=np.float64)

    for component_idx in range(component_count):
        vectors = stacked_components[:, component_idx, :]
        norms = np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)
        for i in range(runs):
            ai = vectors[i]
            for j in range(runs):
                bj = vectors[j]
                raw_cos = float(np.dot(ai, bj) / (norms[i] * norms[j]))
                # PCA directions are sign-indeterminate; align by absolute cosine.
                cosine[component_idx, i, j] = abs(max(min(raw_cos, 1.0), -1.0))
                l2[component_idx, i, j] = min(
                    float(np.linalg.norm(ai - bj)),
                    float(np.linalg.norm(ai + bj)),
                )

    return l2, cosine


def save_pca_alignment_matrices(
    series: List[Dict[str, Any]], root_output: Path
) -> Dict[str, Any] | None:
    base_steps, stacked_components = _extract_pca_component_series(series)
    if stacked_components.size == 0:
        return None

    l2_by_component, cosine_by_component = _pairwise_corresponding_component_metrics(
        stacked_components
    )
    mean_l2 = l2_by_component.mean(axis=0)
    mean_cosine = cosine_by_component.mean(axis=0)

    mean_l2_csv = root_output / "pca_alignment_pairwise_mean_l2_matrix.csv"
    mean_cosine_csv = root_output / "pca_alignment_pairwise_mean_cosine_similarity_matrix.csv"
    mean_l2_heatmap = root_output / "pca_alignment_pairwise_mean_l2_heatmap.png"
    mean_cosine_heatmap = (
        root_output / "pca_alignment_pairwise_mean_cosine_similarity_heatmap.png"
    )
    meta_json = root_output / "pca_alignment_pairwise_matrix_meta.json"

    np.savetxt(mean_l2_csv, mean_l2, delimiter=",", fmt="%.6f")
    np.savetxt(mean_cosine_csv, mean_cosine, delimiter=",", fmt="%.6f")
    _plot_matrix_heatmap(
        matrix=mean_l2,
        base_steps=base_steps,
        title="PCA alignment (mean over PCs): pairwise L2",
        cbar_label="sign-invariant L2",
        output_path=mean_l2_heatmap,
        cmap="magma",
    )
    _plot_matrix_heatmap(
        matrix=mean_cosine,
        base_steps=base_steps,
        title="PCA alignment (mean over PCs): pairwise cosine similarity",
        cbar_label="|cosine similarity|",
        output_path=mean_cosine_heatmap,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    component_payloads: List[Dict[str, Any]] = []
    for component_idx in range(l2_by_component.shape[0]):
        pc_one_based = component_idx + 1
        l2_matrix = l2_by_component[component_idx]
        cosine_matrix = cosine_by_component[component_idx]
        l2_csv = root_output / f"pca_alignment_pc{pc_one_based:02d}_pairwise_l2_matrix.csv"
        cosine_csv = (
            root_output
            / f"pca_alignment_pc{pc_one_based:02d}_pairwise_cosine_similarity_matrix.csv"
        )
        l2_heatmap = (
            root_output / f"pca_alignment_pc{pc_one_based:02d}_pairwise_l2_heatmap.png"
        )
        cosine_heatmap = (
            root_output
            / f"pca_alignment_pc{pc_one_based:02d}_pairwise_cosine_similarity_heatmap.png"
        )

        np.savetxt(l2_csv, l2_matrix, delimiter=",", fmt="%.6f")
        np.savetxt(cosine_csv, cosine_matrix, delimiter=",", fmt="%.6f")
        _plot_matrix_heatmap(
            matrix=l2_matrix,
            base_steps=base_steps,
            title=f"PCA alignment PC{pc_one_based}: pairwise L2",
            cbar_label="sign-invariant L2",
            output_path=l2_heatmap,
            cmap="magma",
        )
        _plot_matrix_heatmap(
            matrix=cosine_matrix,
            base_steps=base_steps,
            title=f"PCA alignment PC{pc_one_based}: pairwise cosine similarity",
            cbar_label="|cosine similarity|",
            output_path=cosine_heatmap,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        component_payloads.append(
            {
                "component": int(pc_one_based),
                "l2_matrix_csv": str(l2_csv),
                "l2_heatmap_png": str(l2_heatmap),
                "cosine_similarity_matrix_csv": str(cosine_csv),
                "cosine_similarity_heatmap_png": str(cosine_heatmap),
            }
        )

    meta_payload = {
        "base_steps": [int(step) for step in base_steps.tolist()],
        "num_runs": int(stacked_components.shape[0]),
        "num_components": int(stacked_components.shape[1]),
        "sign_invariant": True,
        "mean_l2_matrix_csv": str(mean_l2_csv),
        "mean_l2_heatmap_png": str(mean_l2_heatmap),
        "mean_cosine_similarity_matrix_csv": str(mean_cosine_csv),
        "mean_cosine_similarity_heatmap_png": str(mean_cosine_heatmap),
        "per_component": component_payloads,
    }
    meta_json.write_text(json.dumps(meta_payload, indent=2))
    return meta_payload


def main() -> None:
    args = parse_args()
    if args.steps_min > args.steps_max:
        raise ValueError("--steps-min must be <= --steps-max.")
    if args.extra_steps < 0:
        raise ValueError("--extra-steps must be non-negative.")
    config, run_dir = resolve_config(config_path=None, run_dir=args.run_dir)
    dataset_path = resolve_dataset_path(args.dataset, args.split, config)
    dataset = load_dataset(dataset_path)
    checkpoint_path = resolve_checkpoint_path(checkpoint=None, run_dir=run_dir)
    model, _ = load_model_from_checkpoint(config, checkpoint_path, run_dir)
    model.eval()

    root_output = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.run_dir) / "analysis" / "execution_length_step_overlay"
    )
    sweep_root = root_output / "per_step_runs"
    sweep_root.mkdir(parents=True, exist_ok=True)

    collected: List[Dict[str, Any]] = []
    for base_step_count in range(args.steps_min, args.steps_max + 1):
        step_dir = sweep_root / f"steps_{base_step_count:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        ok = run_single_step(args, base_step_count, step_dir)
        if not ok:
            continue
        metadata = load_step_means(step_dir)
        total_steps = int(metadata["target_steps"])
        expected_total_steps = base_step_count + args.extra_steps
        if total_steps != expected_total_steps:
            raise ValueError(
                "Unexpected target_steps returned by embedding_trajectories: "
                f"got {total_steps}, expected {expected_total_steps} "
                f"(base_steps={base_step_count}, extra_steps={args.extra_steps})."
            )
        collected.append(
            {
                "base_steps": int(base_step_count),
                "total_steps": total_steps,
                "num_graphs": int(metadata["num_graphs"]),
                "step_pca_mean_coordinates": metadata["step_pca_mean_coordinates"],
                "completion_avg_coordinate": completion_average_in_step_pca(
                    args=args,
                    metadata=metadata,
                    step_dir=step_dir,
                    model=model,
                    dataset=dataset,
                    embed_dim=config.model.embed_dim,
                ),
                "pca_step_path": str(step_dir / "pca_step.npz"),
                "metadata_path": str(step_dir / "metadata.json"),
            }
        )

    if not collected:
        raise ValueError("No runs succeeded. Nothing to overlay.")

    overlay_path = root_output / "avg_step_coordinate_overlay.png"
    plot_overlay(collected, overlay_path, args.extra_steps)
    probe_matrices = None
    if args.probe_heatmaps:
        probe_matrices = save_probe_matrices(collected, root_output)
    pca_alignment_matrices = None
    if args.pca_alignment_heatmaps:
        pca_alignment_matrices = save_pca_alignment_matrices(collected, root_output)

    summary = {
        "run_dir": args.run_dir,
        "dataset": str(dataset_path),
        "split": args.split,
        "latent": args.latent,
        "node_agg": args.node_agg,
        "steps_min_base": args.steps_min,
        "steps_max_base": args.steps_max,
        "extra_steps": args.extra_steps,
        "series": sorted(collected, key=lambda x: x["base_steps"]),
        "overlay_plot": str(overlay_path),
        "probe_heatmaps": probe_matrices,
        "pca_alignment_heatmaps": pca_alignment_matrices,
    }
    summary_path = root_output / "avg_step_coordinate_overlay.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved overlay plot to: {overlay_path}")
    if args.probe_heatmaps:
        if probe_matrices is None:
            print("Skipped probe heatmaps: no completion probes available.")
        else:
            print(
                "Saved probe heatmaps to: "
                f"{probe_matrices['l2_heatmap_png']} and "
                f"{probe_matrices['cosine_similarity_heatmap_png']}"
            )
    if args.pca_alignment_heatmaps:
        if pca_alignment_matrices is None:
            print("Skipped PCA alignment heatmaps: no PCA payloads available.")
        else:
            print(
                "Saved PCA alignment heatmaps to: "
                f"{pca_alignment_matrices['mean_l2_heatmap_png']} and "
                f"{pca_alignment_matrices['mean_cosine_similarity_heatmap_png']}"
            )
    print(f"Saved overlay summary to: {summary_path}")


if __name__ == "__main__":
    main()
