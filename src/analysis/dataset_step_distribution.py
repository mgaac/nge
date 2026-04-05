"""Visualize execution-step distributions for the algorithms present in a dataset."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.data import load_dataset
from src.utils.task_specs import (
    algorithm_display_name,
    algorithm_target_keys,
    supported_algorithms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot execution-step count distributions for each algorithm in a dataset."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (.npz).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: analysis/dataset_step_distribution).",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap on number of graphs to include.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title override.",
    )
    return parser.parse_args()


def detect_algorithms(dataset: list[dict]) -> tuple[str, ...]:
    present = []
    for algorithm in supported_algorithms():
        keys = algorithm_target_keys(algorithm)
        if any(all(key in graph for key in keys) for graph in dataset):
            present.append(algorithm)
    if not present:
        raise ValueError("Could not infer any supported algorithms from the dataset schema.")
    return tuple(present)


def execution_steps(graph: dict, algorithms: tuple[str, ...]) -> dict[str, int]:
    return {
        algorithm: max(len(graph[algorithm_target_keys(algorithm)[0]]) - 1, 0)
        for algorithm in algorithms
    }


def integer_distribution(values: list[int]) -> dict[int, int]:
    if not values:
        return {}
    unique, counts = np.unique(np.array(values, dtype=np.int32), return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def summarize(values: list[int]) -> dict:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None, "std": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def save_plot(
    distributions: dict[str, dict[int, int]],
    num_graphs: int,
    output_path: Path,
    title: str | None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render dataset step distributions."
        ) from exc

    algorithms = tuple(distributions.keys())
    fig, axes = plt.subplots(1, len(algorithms), figsize=(5.5 * len(algorithms), 5), sharey=True)
    if len(algorithms) == 1:
        axes = [axes]

    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    for axis, algorithm, color in zip(axes, algorithms, palette, strict=False):
        dist = distributions[algorithm]
        steps = sorted(dist.keys())
        counts = [dist[step] for step in steps]
        bars = axis.bar(steps, counts, color=color, alpha=0.9, width=0.8)
        axis.set_title(f"{algorithm_display_name(algorithm)} execution steps")
        axis.set_xlabel("Execution steps")
        axis.set_xticks(steps)
        axis.grid(axis="y", alpha=0.3)
        if counts:
            axis.bar_label(bars, labels=[str(count) for count in counts], padding=2, fontsize=8)

    axes[0].set_ylabel("Graph count")
    fig.suptitle(title if title else f"Execution-step distributions ({num_graphs} graphs)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pairwise_relation_counts(step_series: dict[str, list[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    algorithms = tuple(step_series.keys())
    for left, right in combinations(algorithms, 2):
        left_values = np.array(step_series[left], dtype=np.int32)
        right_values = np.array(step_series[right], dtype=np.int32)
        counts[f"{left}_eq_{right}"] = int(np.sum(left_values == right_values))
        counts[f"{left}_gt_{right}"] = int(np.sum(left_values > right_values))
        counts[f"{right}_gt_{left}"] = int(np.sum(right_values > left_values))
    return counts


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if args.max_graphs is not None and args.max_graphs <= 0:
        raise ValueError("--max-graphs must be positive.")

    dataset = load_dataset(dataset_path)
    if args.max_graphs is not None:
        dataset = dataset[: args.max_graphs]
    if not dataset:
        raise ValueError("Dataset is empty after applying --max-graphs.")

    algorithms = detect_algorithms(dataset)
    step_series = {algorithm: [] for algorithm in algorithms}
    for graph in dataset:
        counts = execution_steps(graph, algorithms)
        for algorithm in algorithms:
            step_series[algorithm].append(counts[algorithm])

    distributions = {
        algorithm: integer_distribution(values) for algorithm, values in step_series.items()
    }

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("analysis") / "dataset_step_distribution"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / "execution_step_distribution.png"
    save_plot(
        distributions=distributions,
        num_graphs=len(dataset),
        output_path=plot_path,
        title=args.title,
    )

    payload = {
        "dataset": str(dataset_path),
        "num_graphs": len(dataset),
        "algorithms": list(algorithms),
        "step_distributions": {
            algorithm: {
                "distribution": distributions[algorithm],
                "summary": summarize(step_series[algorithm]),
            }
            for algorithm in algorithms
        },
        "relation_counts": pairwise_relation_counts(step_series),
        "plot": str(plot_path),
    }
    summary_path = output_dir / "execution_step_distribution.json"
    summary_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
