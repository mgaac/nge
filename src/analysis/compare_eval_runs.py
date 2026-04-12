"""Compare eval-only outputs across multiple runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.transfer_matrix import primary_metric_for_algorithm
from src.utils.task_specs import supported_algorithms

BASELINE_TASK_NONTERMINATION_MEANS = {
    "bfs": 0.679,
    "bf": (0.128 + 0.453) / 2.0,
    "prim": (0.401 + 0.118 + 0.422) / 3.0,
    "dijkstra": (0.043 + 0.278) / 2.0,
    "dag_shortest_paths": (0.008 + 0.511) / 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare eval-only run outputs.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories whose analysis/<eval-file-name> payloads will be compared.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for the provided run dirs.",
    )
    parser.add_argument(
        "--eval-file-name",
        type=str,
        default="eval_only.json",
        help="Filename inside each run's analysis directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split payload to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for CSV/JSON/plot outputs.",
    )
    return parser.parse_args()


def latest_checkpoint_step(run_dir: Path) -> int | None:
    latest_file = run_dir / "checkpoints" / "latest.json"
    if not latest_file.exists():
        return None
    with open(latest_file, "r") as handle:
        latest_info = json.load(handle)
    step = latest_info.get("step")
    return None if step is None else int(step)


def load_eval_payload(run_dir: Path, eval_file_name: str, split: str) -> dict:
    eval_path = run_dir / "analysis" / eval_file_name
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval artifact not found: {eval_path}")
    with open(eval_path, "r") as handle:
        payload = json.load(handle)
    latest_step = latest_checkpoint_step(run_dir)
    eval_step = payload.get("checkpoint_step")
    if latest_step is not None and eval_step is not None and int(eval_step) != int(latest_step):
        raise ValueError(
            f"Stale eval-only artifact detected for {run_dir}: "
            f"{eval_file_name} used checkpoint_step={eval_step}, "
            f"latest available is {latest_step}."
        )
    split_payload = payload.get(split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Missing split '{split}' in {eval_path}")
    return {
        "checkpoint_step": eval_step,
        "latest_checkpoint_step": latest_step,
        "split_payload": split_payload,
    }


def task_nontermination_means(payload: dict, algorithms: list[str]) -> dict[str, float]:
    """Average non-termination accuracy metrics per algorithm."""
    means: dict[str, float] = {}
    for algorithm in algorithms:
        prefix = f"{algorithm}_"
        values = [
            float(payload[f"acc/{metric}"])
            for metric in (
                key.removeprefix("acc/") for key in payload.keys() if key.startswith("acc/")
            )
            if metric.startswith(prefix) and not metric.endswith("_termination")
        ]
        if values:
            means[algorithm] = float(np.mean(values))
    return means


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in args.run_dirs]
    labels = (
        args.labels
        if args.labels is not None
        else [run_dir.name for run_dir in run_dirs]
    )
    if len(labels) != len(run_dirs):
        raise ValueError("--labels must match the number of --run-dirs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload_entries = [
        load_eval_payload(run_dir, args.eval_file_name, args.split)
        for run_dir in run_dirs
    ]
    payloads = [entry["split_payload"] for entry in payload_entries]

    metric_names = sorted(
        {
            key.removeprefix("acc/")
            for payload in payloads
            for key in payload
            if key.startswith("acc/")
        }
    )

    csv_path = output_dir / f"{args.split}_accuracy_comparison.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + labels)
        for metric in metric_names:
            row = [metric]
            key = f"acc/{metric}"
            for payload in payloads:
                value = payload.get(key)
                row.append("" if value is None else f"{float(value):.6f}")
            writer.writerow(row)

    available_algorithms = [
        algorithm
        for algorithm in supported_algorithms()
        if any(f"acc/{primary_metric_for_algorithm(algorithm)}" in payload for payload in payloads)
    ]
    primary_metrics = [primary_metric_for_algorithm(algorithm) for algorithm in available_algorithms]
    task_mean_algorithms = [
        algorithm
        for algorithm in supported_algorithms()
        if any(
            f"acc/{metric}".startswith(f"acc/{algorithm}_") and not metric.endswith("_termination")
            for payload in payloads
            for metric in (
                key.removeprefix("acc/") for key in payload.keys() if key.startswith("acc/")
            )
        )
    ]
    primary_summary = []
    for label, run_dir, payload, payload_entry in zip(labels, run_dirs, payloads, payload_entries):
        values = [
            float(payload[f"acc/{metric}"])
            for metric in primary_metrics
            if f"acc/{metric}" in payload
        ]
        task_means = task_nontermination_means(payload, task_mean_algorithms)
        primary_summary.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "checkpoint_step": payload_entry["checkpoint_step"],
                "latest_checkpoint_step": payload_entry["latest_checkpoint_step"],
                "primary_metric_values": {
                    metric: float(payload[f"acc/{metric}"])
                    for metric in primary_metrics
                    if f"acc/{metric}" in payload
                },
                "primary_metric_mean": float(np.mean(values)) if values else None,
                "all_accuracy_mean": float(
                    np.mean([float(payload[f"acc/{metric}"]) for metric in metric_names if f"acc/{metric}" in payload])
                )
                if metric_names
                else None,
                "task_nontermination_metric_values": task_means,
                "task_nontermination_metric_mean": float(np.mean(list(task_means.values())))
                if task_means
                else None,
            }
        )

    summary = {
        "split": args.split,
        "eval_file_name": args.eval_file_name,
        "runs": primary_summary,
        "primary_metrics": primary_metrics,
        "task_nontermination_metrics": task_mean_algorithms,
        "baseline_task_nontermination_metric_values": {
            algorithm: float(BASELINE_TASK_NONTERMINATION_MEANS[algorithm])
            for algorithm in task_mean_algorithms
            if algorithm in BASELINE_TASK_NONTERMINATION_MEANS
        },
        "csv_path": str(csv_path),
    }
    summary_path = output_dir / f"{args.split}_accuracy_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    if task_mean_algorithms:
        fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(task_mean_algorithms) + 2.5), 4.5))
        x = np.arange(len(task_mean_algorithms))
        width = 0.8 / max(len(labels), 1)
        colors = ["#0f766e", "#b45309", "#1d4ed8", "#be123c", "#4d7c0f"]
        for index, (label, payload) in enumerate(zip(labels, payloads)):
            task_means = task_nontermination_means(payload, task_mean_algorithms)
            values = [
                float(task_means.get(algorithm, np.nan))
                for algorithm in task_mean_algorithms
            ]
            ax.bar(
                x + (index - (len(labels) - 1) / 2) * width,
                values,
                width,
                label=label,
                color=colors[index % len(colors)],
            )
        baseline_values = [
            float(BASELINE_TASK_NONTERMINATION_MEANS.get(algorithm, np.nan))
            for algorithm in task_mean_algorithms
        ]
        ax.plot(
            x,
            baseline_values,
            color="#111827",
            linestyle="--",
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label="baseline",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(task_mean_algorithms, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("accuracy")
        ax.set_title(f"{args.split.capitalize()} task mean accuracy (non-termination)")
        ax.legend()
        ax.grid(alpha=0.18, linewidth=0.6)
        fig.tight_layout()
        plot_path = output_dir / f"{args.split}_primary_metric_comparison.png"
        fig.savefig(plot_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        summary["plot_path"] = str(plot_path)
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)

    print(f"Saved comparison CSV to: {csv_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
