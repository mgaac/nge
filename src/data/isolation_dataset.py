"""Build counterfactual datasets that isolate execution to a chosen algorithm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEQUENCE_GROUPS = {
    "bf": ("bf_distance_targets", "bf_predecessor_targets"),
    "bfs": ("bfs_state_targets",),
    "prim": ("prim_state_targets", "prim_key_targets", "prim_predecessor_targets"),
}
ALL_SEQUENCE_KEYS = tuple(
    key for keys in SEQUENCE_GROUPS.values() for key in keys
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dataset where only one algorithm evolves and the others remain "
            "frozen at their initial target state."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Input dataset (.npz).")
    parser.add_argument("--output", type=str, required=True, help="Output dataset (.npz).")
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["bf", "bfs", "prim", "balanced"],
        help=(
            "Which isolated variants to emit. 'balanced' concatenates BF-only, "
            "BFS-only, and Prim-only variants for each selected source graph."
        ),
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap on source graphs before variants are generated.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle source graphs before applying --max-graphs.",
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
        help="Optional path for summary JSON (default: <output>.summary.json).",
    )
    return parser.parse_args()


def load_dataset_numpy(path: Path) -> list[dict]:
    loaded = np.load(path, allow_pickle=False)
    num_graphs = int(loaded["num_graphs"])
    dataset: list[dict] = []
    for i in range(num_graphs):
        graph = {
            "num_nodes": int(loaded[f"num_nodes_{i}"]),
            "edge_matrix": loaded[f"edge_matrix_{i}"],
            "source_node": int(loaded[f"source_node_{i}"]),
        }
        for key in ALL_SEQUENCE_KEYS:
            graph[key] = loaded[f"{key}_{i}"]
        dataset.append(graph)
    return dataset


def save_dataset_numpy(dataset: list[dict], path: Path) -> None:
    save_dict: dict[str, object] = {"num_graphs": len(dataset)}
    for i, graph in enumerate(dataset):
        save_dict[f"num_nodes_{i}"] = int(graph["num_nodes"])
        save_dict[f"edge_matrix_{i}"] = np.array(graph["edge_matrix"])
        save_dict[f"source_node_{i}"] = int(graph["source_node"])
        for key in ALL_SEQUENCE_KEYS:
            save_dict[f"{key}_{i}"] = np.array(graph[key])
    np.savez_compressed(path, **save_dict)


def collapse_sequence_to_initial_state(array: np.ndarray) -> np.ndarray:
    if array.ndim == 0:
        return array.reshape(1)
    return np.array(array[:1], copy=True)


def isolate_graph(graph: dict, active_algorithm: str) -> dict:
    isolated = {
        "num_nodes": int(graph["num_nodes"]),
        "edge_matrix": np.array(graph["edge_matrix"], copy=True),
        "source_node": int(graph["source_node"]),
    }
    for algorithm, keys in SEQUENCE_GROUPS.items():
        for key in keys:
            value = np.array(graph[key], copy=True)
            if algorithm == active_algorithm:
                isolated[key] = value
            else:
                isolated[key] = collapse_sequence_to_initial_state(value)
    return isolated


def execution_steps(graph: dict) -> dict[str, int]:
    return {
        algorithm: max(int(graph[keys[0]].shape[0]) - 1, 0)
        for algorithm, keys in SEQUENCE_GROUPS.items()
    }


def variants_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "balanced":
        return ("bf", "bfs", "prim")
    return (mode,)


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
    source_indices = list(range(len(dataset)))
    if args.shuffle and source_indices:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(source_indices)
    if args.max_graphs is not None:
        if args.max_graphs <= 0:
            raise ValueError("--max-graphs must be positive.")
        source_indices = source_indices[: args.max_graphs]

    active_algorithms = variants_for_mode(args.mode)
    isolated_dataset: list[dict] = []
    variant_records: list[dict] = []
    for source_index in source_indices:
        source_graph = dataset[int(source_index)]
        source_steps = execution_steps(source_graph)
        for active_algorithm in active_algorithms:
            isolated_graph = isolate_graph(source_graph, active_algorithm)
            isolated_dataset.append(isolated_graph)
            variant_records.append(
                {
                    "source_index": int(source_index),
                    "active_algorithm": active_algorithm,
                    "source_execution_steps": source_steps,
                    "isolated_execution_steps": execution_steps(isolated_graph),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_numpy(isolated_dataset, output_path)

    by_algorithm = {
        algorithm: sum(
            1 for record in variant_records if record["active_algorithm"] == algorithm
        )
        for algorithm in active_algorithms
    }
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "mode": args.mode,
        "num_source_graphs": len(source_indices),
        "num_output_graphs": len(isolated_dataset),
        "active_algorithm_counts": by_algorithm,
        "variants": variant_records,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved isolated dataset to: {output_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
