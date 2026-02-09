"""Shared analysis helpers for NGE research scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import mlx.core as mx

from src.model import AggregationFn, NGE
from src.utils import CheckpointManager, ExperimentConfig, load_config, validate_config


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
    dataset_path: str | None, split: str, config: ExperimentConfig
) -> Path:
    if dataset_path:
        return Path(dataset_path)

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


def iter_execution_inputs(
    graph_data: dict, extra_steps: int
) -> Iterable[Tuple[mx.array, mx.array]]:
    bf_steps = graph_data["bf_distance_targets"]
    bfs_steps = graph_data["bfs_state_targets"]
    num_bf_steps = len(bf_steps)
    num_bfs_steps = len(bfs_steps)
    num_steps = max(num_bf_steps, num_bfs_steps)

    for i in range(num_steps):
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        if not (bf_sample_exists or bfs_sample_exists):
            continue
        true_bfs_state = bfs_steps[i] if bfs_sample_exists else bfs_steps[-1]
        true_distance_bf = bf_steps[i] if bf_sample_exists else bf_steps[-1]
        yield true_bfs_state, true_distance_bf

    if extra_steps > 0:
        final_bfs = bfs_steps[-1]
        final_bf = bf_steps[-1]
        for _ in range(extra_steps):
            yield final_bfs, final_bf


def compute_encoded_embeddings(model: NGE, input_embeddings: mx.array) -> mx.array:
    bfs_encoded = model.bfs_encoder(input_embeddings)
    bf_encoded = model.bf_encoder(input_embeddings)
    encoded = mx.concatenate([bfs_encoded, bf_encoded], axis=1)
    return model.ln(encoded)


def compute_forward_latents(
    model: NGE,
    input_embeddings: mx.array,
    edge_matrix: mx.array,
    zero_bfs_input: bool = False,
    zero_bf_input: bool = False,
) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
    bfs_input = mx.zeros_like(input_embeddings) if zero_bfs_input else input_embeddings
    bf_input = mx.zeros_like(input_embeddings) if zero_bf_input else input_embeddings

    bfs_encoded = model.bfs_encoder(bfs_input)
    bf_encoded = model.bf_encoder(bf_input)

    encoded = mx.concatenate([bfs_encoded, bf_encoded], axis=1)
    encoded = model.ln(encoded)

    processed = model.processor((encoded, edge_matrix))
    return processed, encoded, bfs_encoded, bf_encoded
