"""Sweep termination thresholds and plot test-set termination accuracies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from src.analysis.common import (
    load_analysis_dataset,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
    resolve_config,
    resolve_dataset_path,
)
from src.train import evaluate_model
from src.utils.task_specs import (
    SELECT_TASK_CHOICES,
    TERMINATION_LATENT_CHOICES,
    algorithm_display_name,
    metric_index,
    normalize_algorithm_order,
    resolve_selected_tasks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep termination thresholds and plot per-algorithm termination accuracy. "
            "By default, sweep evaluation uses distance mode so legacy head-trained runs are supported."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory containing config_resolved.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory or file (defaults to latest in run-dir).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate when --dataset is not provided.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset path override (.npz).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        choices=SELECT_TASK_CHOICES,
        default="all",
        help="Tasks to evaluate.",
    )
    parser.add_argument(
        "--termination-mode",
        type=str,
        choices=["head", "distance"],
        default="distance",
        help=(
            "Termination mode used during the sweep. Default is distance, which "
            "enables threshold sweeps on legacy head-mode runs."
        ),
    )
    parser.add_argument(
        "--termination-latent",
        type=str,
        default=None,
        choices=TERMINATION_LATENT_CHOICES,
        help="Optional override for termination_distance_latent.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated threshold list. Overrides min/max/num.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.0,
        help="Minimum threshold for linspace sweep.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.005,
        help=(
            "Step size for threshold sweep. Used when --thresholds is not provided. "
            "Set to <=0 to fall back to --num-thresholds linspace."
        ),
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=0.2,
        help="Maximum threshold for linspace sweep.",
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=21,
        help="Number of thresholds in linspace sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plot and JSON.",
    )
    return parser.parse_args()

def parse_thresholds(args: argparse.Namespace) -> List[float]:
    if args.thresholds:
        values = [v.strip() for v in args.thresholds.split(",") if v.strip()]
        if not values:
            raise ValueError("--thresholds provided but no valid numeric values found.")
        thresholds = [float(v) for v in values]
    else:
        if not np.isfinite(args.threshold_min) or not np.isfinite(args.threshold_max):
            raise ValueError("--threshold-min and --threshold-max must be finite.")
        if args.threshold_max < args.threshold_min:
            raise ValueError("--threshold-max must be >= --threshold-min.")
        if args.threshold_step > 0:
            if not np.isfinite(args.threshold_step):
                raise ValueError("--threshold-step must be finite.")
            num = int(np.floor((args.threshold_max - args.threshold_min) / args.threshold_step)) + 1
            if num > 5000:
                raise ValueError(
                    f"Threshold sweep would create {num} points (>5000). "
                    "Increase --threshold-step or set explicit --thresholds."
                )
            thresholds = (
                args.threshold_min + np.arange(num, dtype=np.float64) * args.threshold_step
            ).tolist()
            if thresholds and thresholds[-1] < args.threshold_max:
                thresholds.append(float(args.threshold_max))
            thresholds = [min(float(t), float(args.threshold_max)) for t in thresholds]
        else:
            if args.num_thresholds <= 0:
                raise ValueError("--num-thresholds must be positive.")
            thresholds = np.linspace(
                args.threshold_min, args.threshold_max, num=args.num_thresholds
            ).tolist()
            if len(thresholds) > 5000:
                raise ValueError(
                    f"Threshold sweep would create {len(thresholds)} points (>5000). "
                    "Lower --num-thresholds or use --threshold-step."
                )

    if any(t < 0 for t in thresholds):
        raise ValueError("All thresholds must be non-negative.")
    return thresholds


def save_plot(
    thresholds: List[float],
    accuracy_series: dict[str, List[float]],
    split: str,
    latent: str,
    distance: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render threshold sweep plots. "
            "Install it in your active environment."
        ) from exc

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    markers = ["o", "s", "^", "D", "P"]
    for index, (algorithm, values) in enumerate(accuracy_series.items()):
        ax.plot(
            thresholds,
            values,
            marker=markers[index % len(markers)],
            linewidth=1.8,
            markersize=4,
            label=f"{algorithm_display_name(algorithm)} termination acc",
            color=colors[index % len(colors)],
        )
    ax.set_xlabel("Termination threshold")
    ax.set_ylabel("Accuracy")
    series_values = [np.asarray(values, dtype=np.float64) for values in accuracy_series.values() if values]
    if series_values:
        first_points = np.array([values[0] for values in series_values], dtype=np.float64)
        all_points = np.concatenate(series_values, axis=0)

        # Start with a local zoom around the first threshold point.
        y_min = float(np.min(first_points) - 0.05)
        y_max = float(np.max(first_points) + 0.05)
        if (y_max - y_min) < 0.08:
            center = float(np.mean(first_points))
            y_min = center - 0.04
            y_max = center + 0.04

        # Expand bounds to guarantee every datapoint is visible.
        global_min = float(np.min(all_points))
        global_max = float(np.max(all_points))
        y_min = min(y_min, global_min - 0.01)
        y_max = max(y_max, global_max + 0.01)

        y_min = max(0.0, y_min)
        y_max = min(1.0, y_max)
        if y_max <= y_min:
            y_min, y_max = 0.0, 1.0
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(
        f"Termination Threshold Sweep ({split})\n"
        f"latent={latent}, distance={distance}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config, run_dir = resolve_config(args.config, args.run_dir)
    algorithm_order = normalize_algorithm_order(config.model.algorithms)
    original_mode = config.model.termination_mode
    config.model.termination_mode = args.termination_mode
    if original_mode != config.model.termination_mode:
        print(
            f"Overriding termination mode for sweep: {original_mode} -> {config.model.termination_mode}"
        )
    if args.termination_latent is not None:
        config.model.termination_distance_latent = args.termination_latent

    dataset_path = resolve_dataset_path(
        args.dataset, args.split, config, algorithm_order=algorithm_order
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, run_dir)
    model, step = load_model_from_checkpoint(config, checkpoint_path, run_dir)
    model.eval()

    dataset = load_analysis_dataset(dataset_path, algorithm_order)
    selected_tasks = resolve_selected_tasks(args.tasks, algorithm_order)
    thresholds = parse_thresholds(args)
    metric_indices = metric_index(algorithm_order)

    termination_accuracy = {
        algorithm: [] for algorithm in algorithm_order if selected_tasks.get(algorithm, False)
    }
    losses: List[float] = []

    for threshold in thresholds:
        idx = len(losses) + 1
        print(f"[{idx}/{len(thresholds)}] threshold={threshold:.8f}")
        config.model.termination_distance_threshold = float(threshold)
        _, loss, accuracies = evaluate_model(
            model=model,
            dataset=dataset,
            embed_dim=config.model.embed_dim,
            termination_cfg=config.model,
            selected_tasks=selected_tasks,
        )
        losses.append(float(loss))
        for algorithm in termination_accuracy:
            metric_name = f"{algorithm}_termination"
            termination_accuracy[algorithm].append(float(accuracies[metric_indices[metric_name]]))

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            run_dir / "analysis" / "termination_threshold_sweep"
            if run_dir
            else Path("analysis/termination_threshold_sweep")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f"{args.split}_threshold_vs_termination_accuracy.png"
    save_plot(
        thresholds=thresholds,
        accuracy_series=termination_accuracy,
        split=args.split,
        latent=config.model.termination_distance_latent,
        distance=config.model.termination_distance,
        output_path=plot_path,
    )

    payload = {
        "checkpoint_step": step,
        "split": args.split,
        "dataset": str(dataset_path),
        "tasks": args.tasks,
        "termination": {
            "mode": config.model.termination_mode,
            "original_mode": original_mode,
            "distance": config.model.termination_distance,
            "latent": config.model.termination_distance_latent,
        },
        "thresholds": thresholds,
        "loss": losses,
        "termination_accuracy": termination_accuracy,
        "plot": str(plot_path),
    }
    with open(output_dir / f"{args.split}_threshold_sweep.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved plot to: {plot_path}")
    print(f"Saved metrics to: {output_dir / f'{args.split}_threshold_sweep.json'}")


if __name__ == "__main__":
    main()
