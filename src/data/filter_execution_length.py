"""Filter dataset graphs by BF/BFS execution-length relationship.

Example:
    python -m src.data.filter_execution_length \
        --input data/train_dataset.npz \
        --output data/train_dataset_unequal_exec.npz \
        --relation unequal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a dataset by BF/BFS execution-length relationship."
    )
    parser.add_argument("--input", type=str, required=True, help="Input dataset (.npz).")
    parser.add_argument("--output", type=str, required=True, help="Output dataset (.npz).")
    parser.add_argument(
        "--relation",
        type=str,
        default="unequal",
        choices=["unequal", "equal", "bf_gt_bfs", "bfs_gt_bf"],
        help="Execution-length relation to keep.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap on filtered graphs.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle filtered graphs before applying --max-graphs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is enabled.",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional path for a summary JSON (default: <output>.summary.json).",
    )
    return parser.parse_args()


def execution_steps(graph: dict) -> tuple[int, int]:
    bf_steps = max(len(graph["bf_distance_targets"]) - 1, 0)
    bfs_steps = max(len(graph["bfs_state_targets"]) - 1, 0)
    return bf_steps, bfs_steps


def relation_predicate(relation: str) -> Callable[[int, int], bool]:
    if relation == "unequal":
        return lambda bf, bfs: bf != bfs
    if relation == "equal":
        return lambda bf, bfs: bf == bfs
    if relation == "bf_gt_bfs":
        return lambda bf, bfs: bf > bfs
    if relation == "bfs_gt_bf":
        return lambda bf, bfs: bfs > bf
    raise ValueError(f"Unknown relation: {relation}")


def summarize(dataset: list[dict]) -> dict:
    same = 0
    bf_gt = 0
    bfs_gt = 0
    bf_steps: list[int] = []
    bfs_steps: list[int] = []
    for graph in dataset:
        bf, bfs = execution_steps(graph)
        bf_steps.append(bf)
        bfs_steps.append(bfs)
        if bf == bfs:
            same += 1
        elif bf > bfs:
            bf_gt += 1
        else:
            bfs_gt += 1
    return {
        "num_graphs": len(dataset),
        "length_relation_counts": {
            "equal": same,
            "bf_gt_bfs": bf_gt,
            "bfs_gt_bf": bfs_gt,
        },
        "bf_steps": {
            "min": int(min(bf_steps)) if bf_steps else None,
            "max": int(max(bf_steps)) if bf_steps else None,
            "mean": float(np.mean(np.array(bf_steps))) if bf_steps else None,
        },
        "bfs_steps": {
            "min": int(min(bfs_steps)) if bfs_steps else None,
            "max": int(max(bfs_steps)) if bfs_steps else None,
            "mean": float(np.mean(np.array(bfs_steps))) if bfs_steps else None,
        },
    }


def load_dataset_numpy(path: Path) -> list[dict]:
    loaded = np.load(path, allow_pickle=False)
    num_graphs = int(loaded["num_graphs"])
    dataset = []
    for i in range(num_graphs):
        dataset.append(
            {
                "num_nodes": int(loaded[f"num_nodes_{i}"]),
                "edge_matrix": loaded[f"edge_matrix_{i}"],
                "source_node": int(loaded[f"source_node_{i}"]),
                "bf_distance_targets": loaded[f"bf_distance_targets_{i}"],
                "bf_predecessor_targets": loaded[f"bf_predecessor_targets_{i}"],
                "bfs_state_targets": loaded[f"bfs_state_targets_{i}"],
            }
        )
    return dataset


def save_dataset_numpy(dataset: list[dict], path: Path) -> None:
    save_dict: dict[str, object] = {"num_graphs": len(dataset)}
    for i, graph in enumerate(dataset):
        save_dict[f"num_nodes_{i}"] = int(graph["num_nodes"])
        save_dict[f"edge_matrix_{i}"] = np.array(graph["edge_matrix"])
        save_dict[f"source_node_{i}"] = int(graph["source_node"])
        save_dict[f"bf_distance_targets_{i}"] = np.array(graph["bf_distance_targets"])
        save_dict[f"bf_predecessor_targets_{i}"] = np.array(graph["bf_predecessor_targets"])
        save_dict[f"bfs_state_targets_{i}"] = np.array(graph["bfs_state_targets"])
    np.savez_compressed(path, **save_dict)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = (
        Path(args.summary_json)
        if args.summary_json
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    dataset = load_dataset_numpy(input_path)
    predicate = relation_predicate(args.relation)

    filtered = []
    selected_indices: list[int] = []
    for idx, graph in enumerate(dataset):
        bf_steps, bfs_steps = execution_steps(graph)
        if predicate(bf_steps, bfs_steps):
            filtered.append(graph)
            selected_indices.append(idx)

    if args.shuffle and filtered:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(filtered))
        filtered = [filtered[int(i)] for i in perm]
        selected_indices = [selected_indices[int(i)] for i in perm]

    if args.max_graphs is not None:
        if args.max_graphs <= 0:
            raise ValueError("--max-graphs must be positive.")
        filtered = filtered[: args.max_graphs]
        selected_indices = selected_indices[: args.max_graphs]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_numpy(filtered, output_path)

    source_summary = summarize(dataset)
    filtered_summary = summarize(filtered)
    payload = {
        "input": str(input_path),
        "output": str(output_path),
        "relation": args.relation,
        "selected_indices": selected_indices,
        "source_summary": source_summary,
        "filtered_summary": filtered_summary,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved filtered dataset to: {output_path}")
    print(f"Saved summary to: {summary_path}")
    print(
        "Filtered counts: "
        f"equal={filtered_summary['length_relation_counts']['equal']}, "
        f"bf_gt_bfs={filtered_summary['length_relation_counts']['bf_gt_bfs']}, "
        f"bfs_gt_bf={filtered_summary['length_relation_counts']['bfs_gt_bf']}"
    )


if __name__ == "__main__":
    main()
