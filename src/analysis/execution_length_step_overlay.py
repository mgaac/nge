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
        choices=[
            "processed",
            "encoded",
            "encoded_bfs",
            "encoded_bf",
            "processed_zero_bfs_input",
            "processed_zero_bf_input",
        ],
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
    return parser.parse_args()


def run_single_step(args: argparse.Namespace, step_count: int, output_dir: Path) -> bool:
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
        str(step_count),
        "--pca",
        "step",
        "--pca-components",
        str(args.pca_components),
        "--output-dir",
        str(output_dir),
    ]
    if args.dataset:
        cmd.extend(["--dataset", args.dataset])
    if args.keep_step_plots:
        cmd.append("--plot")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[ok] steps={step_count}")
        return True

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    combined = f"{stdout}\n{stderr}"
    missing_msg = "No graphs matched step count"
    if missing_msg in combined and not args.fail_on_missing:
        print(f"[skip] steps={step_count} (no matching graphs)")
        return False

    raise RuntimeError(
        f"embedding_trajectories failed for steps={step_count}\n"
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


def plot_overlay(series: List[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    if not series:
        raise ValueError("No step series to plot.")

    step_values = [item["target_steps"] for item in series]
    min_step = min(step_values)
    max_step = max(step_values)
    denom = max(max_step - min_step, 1)

    cmap = plt.get_cmap("viridis")

    for item in sorted(series, key=lambda x: x["target_steps"]):
        coords = np.array(
            [entry["mean_coordinate"] for entry in item["step_pca_mean_coordinates"]],
            dtype=np.float64,
        )
        if coords.shape[0] == 0:
            continue
        color = cmap((item["target_steps"] - min_step) / denom)
        ax.plot(
            coords[:, 0],
            coords[:, 1],
            marker="o",
            linewidth=1.6,
            markersize=4,
            alpha=0.95,
            color=color,
            label=f"steps={item['target_steps']} (n={item['num_graphs']})",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Average Step Coordinates by Execution Length")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.steps_min > args.steps_max:
        raise ValueError("--steps-min must be <= --steps-max.")

    root_output = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.run_dir) / "analysis" / "execution_length_step_overlay"
    )
    sweep_root = root_output / "per_step_runs"
    sweep_root.mkdir(parents=True, exist_ok=True)

    collected: List[Dict[str, Any]] = []
    for step_count in range(args.steps_min, args.steps_max + 1):
        step_dir = sweep_root / f"steps_{step_count:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        ok = run_single_step(args, step_count, step_dir)
        if not ok:
            continue
        metadata = load_step_means(step_dir)
        collected.append(
            {
                "target_steps": int(metadata["target_steps"]),
                "num_graphs": int(metadata["num_graphs"]),
                "step_pca_mean_coordinates": metadata["step_pca_mean_coordinates"],
                "metadata_path": str(step_dir / "metadata.json"),
            }
        )

    if not collected:
        raise ValueError("No runs succeeded. Nothing to overlay.")

    overlay_path = root_output / "avg_step_coordinate_overlay.png"
    plot_overlay(collected, overlay_path)

    summary = {
        "run_dir": args.run_dir,
        "dataset": args.dataset,
        "split": args.split,
        "latent": args.latent,
        "node_agg": args.node_agg,
        "steps_min": args.steps_min,
        "steps_max": args.steps_max,
        "series": sorted(collected, key=lambda x: x["target_steps"]),
        "overlay_plot": str(overlay_path),
    }
    summary_path = root_output / "avg_step_coordinate_overlay.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved overlay plot to: {overlay_path}")
    print(f"Saved overlay summary to: {summary_path}")


if __name__ == "__main__":
    main()

