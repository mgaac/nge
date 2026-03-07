"""Visualize BF/BFS/Prim execution-step distributions for a dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot execution-step count distributions for BF and BFS."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (.npz).",
    )
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


def execution_steps(graph: dict) -> tuple[int, int, int]:
    bf_steps = max(len(graph["bf_distance_targets"]) - 1, 0)
    bfs_steps = max(len(graph["bfs_state_targets"]) - 1, 0)
    prim_steps = max(len(graph["prim_key_targets"]) - 1, 0)
    return bf_steps, bfs_steps, prim_steps


def integer_distribution(values: list[int]) -> dict[int, int]:
    if not values:
        return {}
    unique, counts = np.unique(np.array(values, dtype=np.int32), return_counts=True)
    return {int(u): int(c) for u, c in zip(unique, counts)}


def summarize(values: list[int]) -> dict:
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def save_plot(
    bf_dist: dict[int, int],
    bfs_dist: dict[int, int],
    prim_dist: dict[int, int],
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    series = [
        ("BF execution steps", bf_dist, "#1f77b4"),
        ("BFS execution steps", bfs_dist, "#ff7f0e"),
        ("Prim execution steps", prim_dist, "#2ca02c"),
    ]

    for ax, (subplot_title, dist, color) in zip(axes, series):
        steps = sorted(dist.keys())
        counts = [dist[s] for s in steps]
        bars = ax.bar(steps, counts, color=color, alpha=0.9, width=0.8)
        ax.set_title(subplot_title)
        ax.set_xlabel("Execution steps")
        ax.set_xticks(steps)
        ax.grid(axis="y", alpha=0.3)
        if counts:
            ax.bar_label(bars, labels=[str(c) for c in counts], padding=2, fontsize=8)

    axes[0].set_ylabel("Graph count")
    total_title = title if title else f"Execution-step distributions ({num_graphs} graphs)"
    fig.suptitle(total_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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

    bf_steps: list[int] = []
    bfs_steps: list[int] = []
    prim_steps: list[int] = []
    equal_count = 0
    bf_gt_count = 0
    bfs_gt_count = 0
    bf_gt_prim_count = 0
    prim_gt_bf_count = 0
    bfs_gt_prim_count = 0
    prim_gt_bfs_count = 0

    for graph in dataset:
        bf, bfs, prim = execution_steps(graph)
        bf_steps.append(bf)
        bfs_steps.append(bfs)
        prim_steps.append(prim)
        if bf == bfs == prim:
            equal_count += 1
        if bf > bfs:
            bf_gt_count += 1
        elif bfs > bf:
            bfs_gt_count += 1
        if bf > prim:
            bf_gt_prim_count += 1
        elif prim > bf:
            prim_gt_bf_count += 1
        if bfs > prim:
            bfs_gt_prim_count += 1
        elif prim > bfs:
            prim_gt_bfs_count += 1

    bf_dist = integer_distribution(bf_steps)
    bfs_dist = integer_distribution(bfs_steps)
    prim_dist = integer_distribution(prim_steps)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("analysis") / "dataset_step_distribution"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / "execution_step_distribution.png"
    save_plot(
        bf_dist=bf_dist,
        bfs_dist=bfs_dist,
        prim_dist=prim_dist,
        num_graphs=len(dataset),
        output_path=plot_path,
        title=args.title,
    )

    payload = {
        "dataset": str(dataset_path),
        "num_graphs": len(dataset),
        "bf_steps": {
            "distribution": bf_dist,
            "summary": summarize(bf_steps),
        },
        "bfs_steps": {
            "distribution": bfs_dist,
            "summary": summarize(bfs_steps),
        },
        "prim_steps": {
            "distribution": prim_dist,
            "summary": summarize(prim_steps),
        },
        "relation_counts": {
            "all_equal": equal_count,
            "bf_gt_bfs": bf_gt_count,
            "bfs_gt_bf": bfs_gt_count,
            "bf_gt_prim": bf_gt_prim_count,
            "prim_gt_bf": prim_gt_bf_count,
            "bfs_gt_prim": bfs_gt_prim_count,
            "prim_gt_bfs": prim_gt_bfs_count,
        },
        "plot": str(plot_path),
    }
    summary_path = output_dir / "execution_step_distribution.json"
    summary_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved plot to: {plot_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
