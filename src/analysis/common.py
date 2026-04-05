"""Shared analysis helpers for NGE research scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import mlx.core as mx

from src.data import load_dataset, materialize_graph_sample
from src.model import AggregationFn, NGE
from src.utils import CheckpointManager, ExperimentConfig, load_config, validate_config
from src.utils.task_specs import (
    DEFAULT_ALGORITHMS,
    algorithm_target_keys,
    execution_step_counts,
    feature_values_for_step,
    normalize_algorithm_order,
    processor_algorithm_order,
)


def create_model(config: ExperimentConfig) -> NGE:
    agg_fn_map = {
        "SUM": AggregationFn.SUM,
        "AVG": AggregationFn.AVG,
        "MIN": AggregationFn.MIN,
        "MAX": AggregationFn.MAX,
    }
    agg_fn = agg_fn_map[config.model.agg_fn]
    return NGE(
        embed_dim=config.model.embed_dim,
        residual_connections=config.model.residual_connections,
        agg_fn=agg_fn,
        num_mp_layers=config.model.num_mp_layers,
        dropout=config.model.dropout,
        algorithms=tuple(config.model.algorithms),
    )


def resolve_config(
    config_path: str | None, run_dir: str | None
) -> Tuple[ExperimentConfig, Path | None]:
    run_dir_path = Path(run_dir) if run_dir else None
    resolved_config_path = None
    if run_dir_path is not None:
        resolved_config_path = run_dir_path / "config_resolved.yaml"
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Missing config_resolved.yaml in run dir: {run_dir_path}")
    elif config_path:
        resolved_config_path = Path(config_path)
    else:
        raise ValueError("Provide --config or --run-dir.")

    config = load_config(resolved_config_path)
    validate_config(config)
    return config, run_dir_path


def resolve_dataset_path(
    dataset_path: str | None,
    split: str,
    config: ExperimentConfig,
    algorithm_order: Sequence[str] | None = None,
) -> Path:
    def task_split_path(task_entry: Any) -> str:
        split_key = f"{split}_path"
        if isinstance(task_entry, Mapping):
            return str(task_entry[split_key])
        return str(getattr(task_entry, split_key))

    if dataset_path:
        return Path(dataset_path)

    if config.data.task_paths:
        algorithms = normalize_algorithm_order(
            algorithm_order if algorithm_order is not None else config.model.algorithms
        )
        candidate_paths = {
            task_split_path(config.data.task_paths[algorithm])
            for algorithm in algorithms
            if algorithm in config.data.task_paths
        }
        if len(candidate_paths) == 1:
            return Path(next(iter(candidate_paths)))
        legacy_paths = {
            task_split_path(config.data.task_paths[algorithm])
            for algorithm in DEFAULT_ALGORITHMS
            if algorithm in config.data.task_paths
        }
        if set(DEFAULT_ALGORITHMS).issubset(set(config.data.task_paths)) and len(legacy_paths) == 1:
            return Path(next(iter(legacy_paths)))
        raise ValueError(
            "This config uses task-specific datasets. Provide --dataset explicitly for "
            "analysis, or restrict the analysis to algorithms that share one dataset path."
        )

    if split == "train":
        return Path(config.data.train_path)
    if split == "val":
        return Path(config.data.val_path)
    if split == "test":
        return Path(config.data.test_path)
    raise ValueError(f"Unknown split: {split}")


def resolve_checkpoint_path(checkpoint: str | None, run_dir: Path | None) -> Path | None:
    if checkpoint is None and run_dir is None:
        return None

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_file():
            ckpt_path = ckpt_path.parent
        return ckpt_path

    if run_dir is not None:
        return run_dir / "checkpoints"

    return None


def load_model_from_checkpoint(
    config: ExperimentConfig, checkpoint_path: Path | None, run_dir: Path | None
) -> Tuple[NGE, int | None]:
    model = create_model(config)
    model.eval()

    if checkpoint_path is None:
        return model, None

    if checkpoint_path.name == "checkpoints":
        manager = CheckpointManager(checkpoint_path)
        model, _, step = manager.load(model, optimizer=None, checkpoint_path=None)
        return model, step

    checkpoint_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    manager = CheckpointManager(checkpoint_dir)
    model, _, step = manager.load(model, optimizer=None, checkpoint_path=checkpoint_path)
    return model, step


def infer_active_task(
    graph_data: Mapping[str, Any],
    algorithm_order: Sequence[str] | None,
) -> str | None:
    if "active_task" in graph_data:
        active_task = graph_data["active_task"]
        return str(active_task) if active_task is not None else None

    algorithms = normalize_algorithm_order(algorithm_order)
    present = [
        algorithm
        for algorithm in algorithms
        if all(key in graph_data for key in algorithm_target_keys(algorithm))
    ]
    if len(present) == 1:
        return present[0]
    return None


def materialize_analysis_graph(
    graph_data: Mapping[str, Any],
    algorithm_order: Sequence[str] | None,
) -> dict[str, Any]:
    algorithms = normalize_algorithm_order(algorithm_order)
    graph_dict = dict(graph_data)
    active_task = infer_active_task(graph_dict, algorithms)
    return materialize_graph_sample(graph_dict, algorithms, active_task=active_task)


def load_analysis_dataset(
    dataset_path: str | Path,
    algorithm_order: Sequence[str] | None,
) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_path)
    return [materialize_analysis_graph(graph, algorithm_order) for graph in dataset]


def count_execution_steps(
    graph_data: Mapping[str, Any],
    extra_steps: int,
    algorithm_order: Sequence[str] | None,
) -> int:
    base_steps = max(execution_step_counts(graph_data, algorithm_order).values(), default=0)
    return base_steps + max(extra_steps, 0)


def iter_execution_feature_values(
    graph_data: Mapping[str, Any],
    extra_steps: int,
    algorithm_order: Sequence[str] | None,
) -> Iterable[dict[str, Any]]:
    algorithms = normalize_algorithm_order(algorithm_order)
    base_steps = max(execution_step_counts(graph_data, algorithms).values(), default=0)
    for step_index in range(base_steps):
        yield feature_values_for_step(graph_data, step_index, algorithms)

    if extra_steps <= 0:
        return

    final_values = feature_values_for_step(graph_data, base_steps, algorithms)
    for _ in range(extra_steps):
        yield final_values


def compute_forward_latents(
    model: NGE,
    input_embeddings: mx.array,
    edge_matrix: mx.array,
    zero_input_algorithms: Sequence[str] | None = None,
) -> Tuple[mx.array, mx.array, Dict[str, mx.array]]:
    zero_input_set = set(zero_input_algorithms or [])
    encoded_by_algorithm: Dict[str, mx.array] = {}

    for algorithm in model.algorithms:
        encoder = getattr(model, f"{algorithm}_encoder")
        encoder_input = (
            mx.zeros_like(input_embeddings) if algorithm in zero_input_set else input_embeddings
        )
        encoded_by_algorithm[algorithm] = encoder(encoder_input)

    encoded = mx.concatenate(
        [encoded_by_algorithm[algorithm] for algorithm in processor_algorithm_order(model.algorithms)],
        axis=1,
    )
    encoded = model.ln(encoded)

    processed = model.processor((encoded, edge_matrix))
    return processed, encoded, encoded_by_algorithm
