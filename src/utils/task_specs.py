"""Shared task schema utilities for multi-algorithm execution."""

from __future__ import annotations

from typing import Any, Dict, Mapping


SELECT_TASK_CHOICES = ("all", "bf", "bfs", "prim")
ALGORITHMS = ("bf", "bfs", "prim")
TERMINATION_TASKS = ALGORITHMS

INPUT_FEATURE_NAMES = ("bfs_state", "bf_distance", "prim_state", "prim_key")
INPUT_FEATURE_DIM = len(INPUT_FEATURE_NAMES)

SEQUENCE_KEYS = {
    "bf": "bf_distance_targets",
    "bfs": "bfs_state_targets",
    "prim": "prim_key_targets",
}

METRIC_SPECS = (
    ("bf_distance", "bf"),
    ("bf_predecessor", "bf"),
    ("bfs_state", "bfs"),
    ("prim_state", "prim"),
    ("prim_key", "prim"),
    ("prim_predecessor", "prim"),
    ("bf_termination", "bf"),
    ("bfs_termination", "bfs"),
    ("prim_termination", "prim"),
)
METRIC_NAMES = tuple(name for name, _ in METRIC_SPECS)
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}

TERMINATION_LATENT_CHOICES = (
    "processed",
    "encoded",
    "encoded_bfs",
    "encoded_bf",
    "encoded_prim",
)
ANALYSIS_LATENT_CHOICES = TERMINATION_LATENT_CHOICES + (
    "processed_zero_bfs_input",
    "processed_zero_bf_input",
    "processed_zero_prim_input",
)


def resolve_selected_tasks(tasks_arg: str) -> Dict[str, bool]:
    """Resolve CLI task selection into per-algorithm enable flags."""
    if tasks_arg == "all":
        return {algorithm: True for algorithm in ALGORITHMS}
    if tasks_arg not in ALGORITHMS:
        raise ValueError(f"Unknown tasks selection: {tasks_arg}")
    return {algorithm: algorithm == tasks_arg for algorithm in ALGORITHMS}


def sequence_lengths(graph_data: Mapping[str, object]) -> Dict[str, int]:
    """Return raw sequence lengths for each algorithm trace."""
    return {algorithm: len(graph_data[key]) for algorithm, key in SEQUENCE_KEYS.items()}


def execution_step_counts(graph_data: Mapping[str, object]) -> Dict[str, int]:
    """Return executable step counts for each algorithm trace."""
    return {
        algorithm: max(length - 1, 0)
        for algorithm, length in sequence_lengths(graph_data).items()
    }


def effective_step_count(
    step_counts: Mapping[str, int], selected_tasks: Mapping[str, bool]
) -> int:
    """Choose the averaging denominator from the enabled algorithms."""
    active_counts = [
        int(step_counts[algorithm])
        for algorithm in ALGORITHMS
        if bool(selected_tasks.get(algorithm, False))
    ]
    return max(active_counts + [1])


def metric_mask(selected_tasks: Mapping[str, bool]) -> Any:
    """Return a mask over the metric vector for the enabled algorithms."""
    import mlx.core as mx

    values = [
        1.0 if bool(selected_tasks.get(algorithm, False)) else 0.0
        for _, algorithm in METRIC_SPECS
    ]
    return mx.array(values, dtype=mx.float32)


def metric_counters(step_counts: Mapping[str, int]) -> Any:
    """Return per-metric divisors derived from algorithm step counts."""
    import mlx.core as mx

    values = [max(int(step_counts[algorithm]), 1) for _, algorithm in METRIC_SPECS]
    return mx.array(values, dtype=mx.float32)


def metric_dict(prefix: str, values: Any) -> Dict[str, float]:
    """Convert a metric vector into a named dictionary."""
    return {
        f"{prefix}/{metric_name}": float(values[index])
        for index, metric_name in enumerate(METRIC_NAMES)
    }


def build_node_algo_features(
    bfs_state: Any,
    bf_distance: Any,
    prim_state: Any,
    prim_key: Any,
) -> Any:
    """Pack recurrent algorithm state features for a graph step."""
    import mlx.core as mx

    return mx.stack([bfs_state, bf_distance, prim_state, prim_key], axis=1)
