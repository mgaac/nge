"""Shared task schema utilities for multi-algorithm execution."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, Mapping, Sequence


SUPPORTED_ALGORITHM_SPECS = OrderedDict(
    [
        (
            "bfs",
            {
                "family": "state_mask",
                "display_name": "BFS",
                "input_features": ("bfs_state",),
                "target_keys": ("bfs_state_targets",),
                "metric_names": ("bfs_state", "bfs_termination"),
            },
        ),
        (
            "bf",
            {
                "family": "shortest_path",
                "display_name": "Bellman-Ford",
                "input_features": ("bf_distance",),
                "target_keys": ("bf_distance_targets", "bf_predecessor_targets"),
                "metric_names": ("bf_distance", "bf_predecessor", "bf_termination"),
            },
        ),
        (
            "dijkstra",
            {
                "family": "shortest_path",
                "display_name": "Dijkstra",
                "input_features": ("dijkstra_distance",),
                "target_keys": (
                    "dijkstra_distance_targets",
                    "dijkstra_predecessor_targets",
                ),
                "metric_names": (
                    "dijkstra_distance",
                    "dijkstra_predecessor",
                    "dijkstra_termination",
                ),
            },
        ),
        (
            "dag_shortest_paths",
            {
                "family": "shortest_path",
                "display_name": "DAG Shortest Paths",
                "input_features": ("dag_shortest_paths_distance",),
                "target_keys": (
                    "dag_shortest_paths_distance_targets",
                    "dag_shortest_paths_predecessor_targets",
                ),
                "metric_names": (
                    "dag_shortest_paths_distance",
                    "dag_shortest_paths_predecessor",
                    "dag_shortest_paths_termination",
                ),
            },
        ),
        (
            "prim",
            {
                "family": "mst",
                "display_name": "Prim",
                "input_features": ("prim_state", "prim_key"),
                "target_keys": (
                    "prim_state_targets",
                    "prim_key_targets",
                    "prim_predecessor_targets",
                ),
                "metric_names": (
                    "prim_state",
                    "prim_key",
                    "prim_predecessor",
                    "prim_termination",
                ),
            },
        ),
    ]
)

DEFAULT_ALGORITHMS = ("bf", "bfs", "prim")
LEGACY_PROCESSOR_ALGORITHMS = ("bfs", "bf", "prim")
ALGORITHMS = DEFAULT_ALGORITHMS
SELECT_TASK_CHOICES = ("all", *SUPPORTED_ALGORITHM_SPECS.keys())
TERMINATION_TASKS = tuple(SUPPORTED_ALGORITHM_SPECS.keys())


def supported_algorithms() -> tuple[str, ...]:
    """Return all algorithm identifiers supported by the codebase."""
    return tuple(SUPPORTED_ALGORITHM_SPECS.keys())


def normalize_algorithm_order(algorithms: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and normalize algorithm order declarations."""
    if algorithms is None:
        return DEFAULT_ALGORITHMS
    normalized = tuple(str(algorithm) for algorithm in algorithms)
    if not normalized:
        raise ValueError("At least one algorithm must be configured.")
    invalid = [algorithm for algorithm in normalized if algorithm not in SUPPORTED_ALGORITHM_SPECS]
    if invalid:
        raise ValueError(
            "Unknown algorithms: " + ", ".join(invalid)
        )
    duplicates = []
    seen = set()
    for algorithm in normalized:
        if algorithm in seen and algorithm not in duplicates:
            duplicates.append(algorithm)
        seen.add(algorithm)
    if duplicates:
        raise ValueError(
            "Algorithms must be unique. Duplicates: " + ", ".join(duplicates)
        )
    return normalized


def algorithm_family(algorithm: str) -> str:
    return str(SUPPORTED_ALGORITHM_SPECS[algorithm]["family"])


def algorithm_display_name(algorithm: str) -> str:
    return str(SUPPORTED_ALGORITHM_SPECS[algorithm]["display_name"])


def processor_algorithm_order(
    algorithm_order: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return the latent concatenation order used by the processor."""
    normalized = normalize_algorithm_order(algorithm_order)
    if normalized == DEFAULT_ALGORITHMS:
        return LEGACY_PROCESSOR_ALGORITHMS
    return normalized


def input_feature_names(algorithm_order: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return ordered recurrent input feature names for the configured algorithms."""
    features = []
    for algorithm in processor_algorithm_order(algorithm_order):
        features.extend(SUPPORTED_ALGORITHM_SPECS[algorithm]["input_features"])
    return tuple(features)


INPUT_FEATURE_NAMES = input_feature_names(DEFAULT_ALGORITHMS)
INPUT_FEATURE_DIM = len(INPUT_FEATURE_NAMES)


def input_feature_dim(algorithm_order: Sequence[str] | None = None) -> int:
    return len(input_feature_names(algorithm_order))


def termination_latent_choices() -> tuple[str, ...]:
    base = ["processed", "encoded"]
    base.extend(f"encoded_{algorithm}" for algorithm in supported_algorithms())
    return tuple(base)


TERMINATION_LATENT_CHOICES = termination_latent_choices()
ANALYSIS_LATENT_CHOICES = TERMINATION_LATENT_CHOICES + tuple(
    f"processed_zero_{algorithm}_input" for algorithm in supported_algorithms()
)


def sequence_keys(algorithm_order: Sequence[str] | None = None) -> Dict[str, tuple[str, ...]]:
    return {
        algorithm: tuple(SUPPORTED_ALGORITHM_SPECS[algorithm]["target_keys"])
        for algorithm in normalize_algorithm_order(algorithm_order)
    }


SEQUENCE_KEYS = {
    algorithm: keys[0] for algorithm, keys in sequence_keys(DEFAULT_ALGORITHMS).items()
}


def primary_target_key(algorithm: str) -> str:
    return str(SUPPORTED_ALGORITHM_SPECS[algorithm]["target_keys"][0])


def metric_specs(algorithm_order: Sequence[str] | None = None) -> tuple[tuple[str, str], ...]:
    specs = []
    for algorithm in normalize_algorithm_order(algorithm_order):
        for metric_name in SUPPORTED_ALGORITHM_SPECS[algorithm]["metric_names"]:
            specs.append((str(metric_name), algorithm))
    return tuple(specs)


METRIC_SPECS = metric_specs(DEFAULT_ALGORITHMS)
METRIC_NAMES = tuple(name for name, _ in METRIC_SPECS)
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_NAMES)}


def metric_names(algorithm_order: Sequence[str] | None = None) -> tuple[str, ...]:
    return tuple(name for name, _ in metric_specs(algorithm_order))


def metric_index(algorithm_order: Sequence[str] | None = None) -> Dict[str, int]:
    return {name: index for index, name in enumerate(metric_names(algorithm_order))}


def resolve_selected_tasks(
    tasks_arg: str,
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, bool]:
    """Resolve CLI task selection into per-algorithm enable flags."""
    normalized = normalize_algorithm_order(algorithm_order)
    if tasks_arg == "all":
        return {algorithm: True for algorithm in normalized}
    if tasks_arg not in normalized:
        raise ValueError(
            f"Unknown tasks selection: {tasks_arg}. Available: {', '.join(normalized)}"
        )
    return {algorithm: algorithm == tasks_arg for algorithm in normalized}


def selected_tasks_for_graph(
    graph_data: Mapping[str, object],
    selected_tasks: Mapping[str, bool],
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, bool]:
    """Resolve per-graph task masking, honoring graph_data['active_task'] when present."""
    normalized = normalize_algorithm_order(algorithm_order)
    active_task = graph_data.get("active_task")
    if active_task is None:
        return {
            algorithm: bool(selected_tasks.get(algorithm, False)) for algorithm in normalized
        }
    if active_task not in normalized:
        raise ValueError(f"Unknown active_task on graph sample: {active_task}")
    return {
        algorithm: bool(selected_tasks.get(algorithm, False)) and algorithm == active_task
        for algorithm in normalized
    }


def sequence_lengths(
    graph_data: Mapping[str, object], algorithm_order: Sequence[str] | None = None
) -> Dict[str, int]:
    """Return raw sequence lengths for each algorithm trace."""
    return {
        algorithm: len(graph_data[primary_target_key(algorithm)])
        for algorithm in normalize_algorithm_order(algorithm_order)
    }


def execution_step_counts(
    graph_data: Mapping[str, object], algorithm_order: Sequence[str] | None = None
) -> Dict[str, int]:
    """Return executable step counts for each algorithm trace."""
    return {
        algorithm: max(length - 1, 0)
        for algorithm, length in sequence_lengths(graph_data, algorithm_order).items()
    }


def effective_step_count(
    step_counts: Mapping[str, int], selected_tasks: Mapping[str, bool]
) -> int:
    """Choose the averaging denominator from the enabled algorithms."""
    active_counts = [
        int(step_counts[algorithm])
        for algorithm in step_counts
        if bool(selected_tasks.get(algorithm, False))
    ]
    return max(active_counts + [1])


def metric_mask(
    selected_tasks: Mapping[str, bool],
    algorithm_order: Sequence[str] | None = None,
) -> Any:
    """Return a mask over the metric vector for the enabled algorithms."""
    import mlx.core as mx

    values = [
        1.0 if bool(selected_tasks.get(algorithm, False)) else 0.0
        for _, algorithm in metric_specs(algorithm_order)
    ]
    return mx.array(values, dtype=mx.float32)


def metric_counters(
    step_counts: Mapping[str, int],
    algorithm_order: Sequence[str] | None = None,
) -> Any:
    """Return per-metric divisors derived from algorithm step counts."""
    import mlx.core as mx

    values = [
        max(int(step_counts[algorithm]), 1)
        for _, algorithm in metric_specs(algorithm_order)
    ]
    return mx.array(values, dtype=mx.float32)


def metric_dict(
    prefix: str,
    values: Any,
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Convert a metric vector into a named dictionary."""
    return {
        f"{prefix}/{metric_name}": float(values[index])
        for index, metric_name in enumerate(metric_names(algorithm_order))
    }


def build_node_algo_features(
    feature_values: Mapping[str, Any] | Any,
    algorithm_order: Sequence[str] | Any | None = None,
    *legacy_values: Any,
) -> Any:
    """Pack recurrent algorithm state features for a graph step.

    Supports the new mapping-based call signature and the legacy
    positional BF/BFS/Prim signature used by older analysis tools.
    """
    import mlx.core as mx

    if isinstance(feature_values, Mapping):
        normalized_algorithm_order = algorithm_order
        ordered_values = [feature_values[name] for name in input_feature_names(normalized_algorithm_order)]
        return mx.stack(ordered_values, axis=1)

    legacy_payload = (feature_values, algorithm_order, *legacy_values)
    if len(legacy_payload) != len(INPUT_FEATURE_NAMES):
        raise ValueError(
            "Legacy build_node_algo_features calls require exactly "
            f"{len(INPUT_FEATURE_NAMES)} positional feature tensors."
        )

    return mx.stack(list(legacy_payload), axis=1)


def zero_feature_values(
    num_nodes: int,
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Create zero recurrent inputs for every configured algorithm feature."""
    import mlx.core as mx

    return {
        feature_name: mx.zeros([num_nodes], dtype=mx.float32)
        for feature_name in input_feature_names(algorithm_order)
    }


def algorithm_target_keys(algorithm: str) -> tuple[str, ...]:
    return tuple(SUPPORTED_ALGORITHM_SPECS[algorithm]["target_keys"])


def algorithm_metric_names(algorithm: str) -> tuple[str, ...]:
    return tuple(SUPPORTED_ALGORITHM_SPECS[algorithm]["metric_names"])


def algorithm_target_lengths(
    graph_data: Mapping[str, object],
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, int]:
    return {
        algorithm: len(graph_data[primary_target_key(algorithm)])
        for algorithm in normalize_algorithm_order(algorithm_order)
    }


def step_sample_exists(
    step_index: int,
    graph_data: Mapping[str, object],
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, bool]:
    lengths = algorithm_target_lengths(graph_data, algorithm_order)
    return {
        algorithm: (step_index + 1) < length for algorithm, length in lengths.items()
    }


def feature_values_for_step(
    graph_data: Mapping[str, object],
    step_index: int,
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    sample_exists = step_sample_exists(step_index, graph_data, algorithm_order)
    for algorithm in normalize_algorithm_order(algorithm_order):
        family = algorithm_family(algorithm)
        if family == "state_mask":
            key = primary_target_key(algorithm)
            values[input_feature_names((algorithm,))[0]] = (
                graph_data[key][step_index]
                if sample_exists[algorithm]
                else graph_data[key][-1]
            )
        elif family == "shortest_path":
            distance_key = algorithm_target_keys(algorithm)[0]
            values[input_feature_names((algorithm,))[0]] = (
                graph_data[distance_key][step_index]
                if sample_exists[algorithm]
                else graph_data[distance_key][-1]
            )
        elif family == "mst":
            state_key, key_key, _ = algorithm_target_keys(algorithm)
            values[f"{algorithm}_state"] = (
                graph_data[state_key][step_index]
                if sample_exists[algorithm]
                else graph_data[state_key][-1]
            )
            values[f"{algorithm}_key"] = (
                graph_data[key_key][step_index]
                if sample_exists[algorithm]
                else graph_data[key_key][-1]
            )
        else:
            raise ValueError(f"Unsupported algorithm family: {family}")
    return values


def targets_for_step(
    graph_data: Mapping[str, object],
    step_index: int,
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    targets: Dict[str, Dict[str, Any]] = {}
    sample_exists = step_sample_exists(step_index, graph_data, algorithm_order)
    for algorithm in normalize_algorithm_order(algorithm_order):
        family = algorithm_family(algorithm)
        keys = algorithm_target_keys(algorithm)
        if family == "state_mask":
            state_key = keys[0]
            targets[algorithm] = {
                "state": graph_data[state_key][step_index + 1]
                if sample_exists[algorithm]
                else graph_data[state_key][-1]
            }
        elif family == "shortest_path":
            distance_key, predecessor_key = keys
            targets[algorithm] = {
                "distance": graph_data[distance_key][step_index + 1]
                if sample_exists[algorithm]
                else graph_data[distance_key][-1],
                "predecessor": graph_data[predecessor_key][step_index + 1]
                if sample_exists[algorithm]
                else graph_data[predecessor_key][-1],
            }
        elif family == "mst":
            state_key, key_key, predecessor_key = keys
            targets[algorithm] = {
                "state": graph_data[state_key][step_index + 1]
                if sample_exists[algorithm]
                else graph_data[state_key][-1],
                "key": graph_data[key_key][step_index + 1]
                if sample_exists[algorithm]
                else graph_data[key_key][-1],
                "predecessor": graph_data[predecessor_key][step_index + 1]
                if sample_exists[algorithm]
                else graph_data[predecessor_key][-1],
            }
        else:
            raise ValueError(f"Unsupported algorithm family: {family}")
    return targets


def termination_targets_for_step(
    graph_data: Mapping[str, object],
    step_index: int,
    algorithm_order: Sequence[str] | None = None,
) -> Dict[str, Any]:
    import mlx.core as mx

    lengths = algorithm_target_lengths(graph_data, algorithm_order)
    return {
        algorithm: mx.array(1.0 if (step_index + 1) == (length - 1) else 0.0)
        for algorithm, length in lengths.items()
    }


def iter_algorithm_order(
    algorithm_order: Sequence[str] | None = None,
) -> Iterable[str]:
    return normalize_algorithm_order(algorithm_order)
