"""Main training script for NGE model with research workflow.

Usage:
    python -m src.train --config configs/prims_bf_bfs.yaml
    python -m src.train --config configs/prims_bf_bfs.yaml --resume
"""

import argparse
import contextlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils
import mlx.optimizers as optim

from src.model import NGE, AggregationFn
from src.data import load_dataset, materialize_graph_sample
from src.utils import (
    ExperimentConfig,
    load_config,
    save_config,
    validate_config,
    set_seed,
    get_git_info,
    create_run_metadata,
    save_run_metadata,
    generate_run_name,
    CheckpointManager,
    MetricsLogger,
    analyze_failure_modes,
    calculate_accuracies,
    calculate_losses_and_accuracies,
    extract_per_head_magnitude_grads,
    print_execution_details,
)
from src.utils.termination import (
    compute_distance_termination_logits,
    get_distance_latent,
    needs_aux_latents,
    resolve_termination_settings,
)
from src.utils.task_specs import (
    SELECT_TASK_CHOICES,
    TERMINATION_LATENT_CHOICES,
    algorithm_display_name,
    algorithm_family,
    build_node_algo_features,
    effective_step_count,
    execution_step_counts,
    feature_values_for_step,
    metric_counters,
    metric_dict,
    metric_names,
    metric_mask,
    resolve_selected_tasks,
    selected_tasks_for_graph,
    targets_for_step,
    termination_targets_for_step,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train NGE model with research workflow')
    parser.add_argument('--config', type=str, required=False,
                        help='Path to YAML config file (e.g., configs/prims_bf_bfs.yaml)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint in existing run')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Specific run directory to resume from (only with --resume)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint directory or file to load (for --eval-only)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training and run evaluation only')
    parser.add_argument('--tasks', type=str, default=None, choices=SELECT_TASK_CHOICES,
                        help='Tasks to optimize/evaluate. Overrides training.tasks from config.')
    parser.add_argument('--termination-threshold', type=float, default=None,
                        help='Override termination_distance_threshold (useful in --eval-only)')
    parser.add_argument(
        '--termination-mode',
        type=str,
        default=None,
        choices=['head', 'distance'],
        help='Override termination mode (head or distance).',
    )
    parser.add_argument(
        '--termination-latent',
        type=str,
        default=None,
        choices=TERMINATION_LATENT_CHOICES,
        help='Override termination_distance_latent (useful in --eval-only)',
    )
    parser.add_argument('--disable-distance-termination-signal', action='store_true',
                        help='Disable termination BCE supervision when termination_mode=distance')
    parser.add_argument(
        '--accuracies-only',
        action='store_true',
        help='In --eval-only mode, compute and save accuracies only (skip losses).',
    )
    parser.add_argument(
        '--analyze-failures',
        action='store_true',
        help='In --eval-only mode, save per-graph failure analysis JSON.',
    )
    parser.add_argument(
        '--failure-split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to analyze when --analyze-failures is enabled.',
    )
    parser.add_argument(
        '--failure-max-graphs',
        type=int,
        default=None,
        help='Optional max number of graphs to inspect for failure analysis.',
    )
    parser.add_argument(
        '--failure-max-records',
        type=int,
        default=200,
        help='Maximum number of failed graph records written to JSON.',
    )
    parser.add_argument(
        '--failure-include-step-details',
        action='store_true',
        help='Include per-step mismatch counts in failure analysis output.',
    )
    parser.add_argument(
        '--failure-debug-graphs',
        type=str,
        default=None,
        help='Comma-separated graph indices for detailed debug dumps.',
    )
    parser.add_argument(
        '--failure-debug-top-k',
        type=int,
        default=0,
        help='Also dump debug traces for top-K failed graphs.',
    )
    return parser.parse_args()


def parse_graph_indices(indices_arg: str | None) -> list[int]:
    """Parse comma-separated graph indices from CLI."""
    if not indices_arg:
        return []
    values = [chunk.strip() for chunk in indices_arg.split(",") if chunk.strip()]
    indices = []
    for value in values:
        try:
            index = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid graph index: {value}") from exc
        if index < 0:
            raise ValueError(f"Graph indices must be non-negative, got {index}")
        indices.append(index)
    return sorted(set(indices))


def resolve_task_selection(args, config: ExperimentConfig) -> str:
    """Resolve task selection with CLI override precedence over config."""
    return args.tasks if args.tasks is not None else config.training.tasks


def format_accuracy_summary(values, algorithm_order) -> str:
    """Format a metric vector into a compact evaluation summary line."""
    names = metric_names(algorithm_order)
    parts = []
    for index, name in enumerate(names):
        parts.append(f"{name}={float(values[index]):.3f}")
    return ", ".join(parts)


def normalize_model_forward(
    model_output,
    algorithm_order,
    return_latents: bool = False,
):
    """Normalize legacy and dynamic model outputs into dictionaries."""
    if return_latents:
        if isinstance(model_output[0], dict):
            return model_output
        bfs_output, bf_output, prim_output, termination_probs, processed_embeddings, aux = model_output
        outputs = {"bfs": bfs_output, "bf": bf_output, "prim": prim_output}
        return outputs, termination_probs, processed_embeddings, aux

    if isinstance(model_output[0], dict):
        return model_output
    bfs_output, bf_output, prim_output, termination_probs, processed_embeddings = model_output
    outputs = {"bfs": bfs_output, "bf": bf_output, "prim": prim_output}
    return outputs, termination_probs, processed_embeddings


def resolve_module_path(root, module_path: str):
    """Resolve a dotted module path against a model/module tree."""
    module = root
    for part in module_path.split("."):
        if not hasattr(module, part):
            raise ValueError(f"Unknown module path: {module_path}")
        module = getattr(module, part)
    return module


def flatten_parameter_shapes(tree, prefix: str = "") -> dict[str, tuple[int, ...]]:
    """Flatten a parameter pytree into path -> shape mappings."""
    if hasattr(tree, "shape"):
        return {prefix: tuple(int(dim) for dim in tree.shape)}
    if isinstance(tree, dict):
        shapes = {}
        for key, value in tree.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            shapes.update(flatten_parameter_shapes(value, child_prefix))
        return shapes
    if isinstance(tree, (list, tuple)):
        shapes = {}
        for index, value in enumerate(tree):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            shapes.update(flatten_parameter_shapes(value, child_prefix))
        return shapes
    raise TypeError(
        f"Unsupported parameter tree leaf for prefix '{prefix}': {type(tree).__name__}"
    )


def resolve_checkpoint_directory(checkpoint_ref: str | Path) -> Path:
    """Resolve a checkpoint reference into a concrete checkpoint directory."""
    checkpoint_path = Path(checkpoint_ref)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initialization checkpoint not found: {checkpoint_path}")

    if checkpoint_path.is_file():
        checkpoint_path = checkpoint_path.parent

    if (checkpoint_path / "checkpoint.json").exists():
        return checkpoint_path

    if checkpoint_path.name == "checkpoints":
        manager = CheckpointManager(checkpoint_path)
        latest_step = manager.get_latest_step()
        if latest_step is None:
            raise FileNotFoundError(
                f"No checkpoints found under initialization directory: {checkpoint_path}"
            )
        return checkpoint_path / f"step_{latest_step:08d}"

    checkpoints_dir = checkpoint_path / "checkpoints"
    if checkpoints_dir.exists():
        manager = CheckpointManager(checkpoints_dir)
        latest_step = manager.get_latest_step()
        if latest_step is None:
            raise FileNotFoundError(
                f"No checkpoints found under run directory: {checkpoint_path}"
            )
        return checkpoints_dir / f"step_{latest_step:08d}"

    raise FileNotFoundError(
        "Initialization checkpoint must point to a run directory, a checkpoints "
        f"directory, or a concrete checkpoint directory. Got {checkpoint_ref}"
    )


def resolve_run_directory_from_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Resolve the owning run directory for a checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.name.startswith("step_") and checkpoint_dir.parent.name == "checkpoints":
        return checkpoint_dir.parents[1]
    if checkpoint_dir.name == "checkpoints":
        return checkpoint_dir.parent
    if (checkpoint_dir / "config_resolved.yaml").exists():
        return checkpoint_dir
    raise FileNotFoundError(
        "Could not locate config_resolved.yaml for initialization checkpoint "
        f"{checkpoint_dir}"
    )


def initialize_selected_modules_from_checkpoint(
    model,
    config: ExperimentConfig,
    module_paths: list[str],
) -> int:
    """Copy selected modules from a source checkpoint into the current model."""
    checkpoint_dir = resolve_checkpoint_directory(config.training.init_checkpoint)
    source_run_dir = resolve_run_directory_from_checkpoint(checkpoint_dir)
    source_config = load_config(source_run_dir / "config_resolved.yaml")
    source_model = create_model(source_config)
    manager = CheckpointManager(checkpoint_dir.parent)
    source_model, _, step = manager.load(source_model, optimizer=None, checkpoint_path=checkpoint_dir)

    for module_path in module_paths:
        target_module = resolve_module_path(model, module_path)
        source_module = resolve_module_path(source_model, module_path)
        source_params = flatten_parameter_shapes(source_module.parameters())
        target_params = flatten_parameter_shapes(target_module.parameters())
        if set(source_params.keys()) != set(target_params.keys()):
            raise ValueError(
                f"Incompatible initialization for module '{module_path}': source parameter "
                f"keys {sorted(source_params.keys())} do not match target keys "
                f"{sorted(target_params.keys())}. Source algorithms={source_model.algorithms}, "
                f"target algorithms={model.algorithms}."
            )
        for key in sorted(source_params.keys()):
            source_shape = source_params[key]
            target_shape = target_params[key]
            if source_shape != target_shape:
                raise ValueError(
                    f"Incompatible initialization for module '{module_path}.{key}': source "
                    f"shape {source_shape} does not match target shape {target_shape}. "
                    f"Source algorithms={source_model.algorithms}, target algorithms={model.algorithms}. "
                    "This source checkpoint is not processor-compatible with the target config."
                )
        target_module.update(source_module.parameters())

    return int(step)


def initialize_from_checkpoint_if_needed(model, config: ExperimentConfig) -> int | None:
    """Load model weights from an initialization checkpoint when configured."""
    init_checkpoint = config.training.init_checkpoint
    if not init_checkpoint:
        return None

    if config.training.init_checkpoint_modules:
        return initialize_selected_modules_from_checkpoint(
            model, config, config.training.init_checkpoint_modules
        )

    checkpoint_dir = resolve_checkpoint_directory(init_checkpoint)
    manager = CheckpointManager(checkpoint_dir)
    model, _, step = manager.load(model, optimizer=None, checkpoint_path=checkpoint_dir)
    return int(step)


def reset_selected_modules(model, reference_model, module_paths: list[str]) -> None:
    """Reset selected modules by copying parameters from a fresh reference model."""
    for module_path in module_paths:
        target_module = resolve_module_path(model, module_path)
        source_module = resolve_module_path(reference_model, module_path)
        target_module.update(source_module.parameters())


def zero_frozen_gradients(grads, frozen_module_paths: tuple[str, ...]):
    """Zero gradients for parameter subtrees under the configured module prefixes."""
    if not frozen_module_paths:
        return grads

    def maybe_zero(path, value):
        if any(path == prefix or path.startswith(f"{prefix}.") for prefix in frozen_module_paths):
            return mx.zeros_like(value)
        return value

    return utils.tree_map_with_path(maybe_zero, grads)


def _eval_trees(*trees):
    """Force evaluation of MLX arrays nested inside pytrees."""
    leaves = []
    for tree in trees:
        if tree is None:
            continue
        leaves.extend(value for _, value in utils.tree_flatten(tree))
    if leaves:
        mx.eval(*leaves)


def setup_run_directory(config: ExperimentConfig, resume: bool = False, run_dir: str = None) -> Path:
    """Setup or resume run directory with all required artifacts.
    
    Args:
        config: Experiment configuration
        resume: Whether to resume from existing run
        run_dir: Specific run directory path (for resume)
        
    Returns:
        Path to run directory
    """
    runs_root = Path("runs")
    runs_root.mkdir(exist_ok=True)
    
    if resume:
        if run_dir is not None:
            # Resume from specific directory
            run_path = Path(run_dir)
            if not run_path.exists():
                raise ValueError(f"Run directory does not exist: {run_dir}")
        else:
            # Resume from latest run
            existing_runs = sorted(runs_root.glob("*"))
            if not existing_runs:
                raise ValueError("No existing runs found to resume from")
            run_path = existing_runs[-1]
        
        print(f"Resuming from: {run_path}")
        return run_path
    else:
        # Create new run directory
        git_info = get_git_info()
        run_name = generate_run_name(config.name, git_info)
        run_path = runs_root / run_name
        run_path.mkdir(parents=True, exist_ok=False)
        
        # Create subdirectories
        (run_path / "checkpoints").mkdir(exist_ok=True)
        
        # Save resolved config
        save_config(config, run_path / "config_resolved.yaml")
        
        # Create and save metadata
        metadata = create_run_metadata(config.to_dict(), config.training.seed)
        save_run_metadata(metadata, run_path / "meta.json")
        
        print(f"Created run directory: {run_path}")
        return run_path


def create_model(config: ExperimentConfig) -> NGE:
    """Create model from configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Initialized NGE model
    """
    # Map string aggregation function to enum
    agg_fn_map = {
        'SUM': AggregationFn.SUM,
        'AVG': AggregationFn.AVG,
        'MIN': AggregationFn.MIN,
        'MAX': AggregationFn.MAX,
    }
    
    agg_fn = agg_fn_map[config.model.agg_fn]
    
    model = NGE(
        embed_dim=config.model.embed_dim,
        residual_connections=config.model.residual_connections,
        agg_fn=agg_fn,
        num_mp_layers=config.model.num_mp_layers,
        dropout=config.model.dropout,
        algorithms=tuple(config.model.algorithms),
    )
    
    return model


def load_split_dataset(
    config: ExperimentConfig,
    split: str,
    selected_tasks: dict[str, bool],
):
    """Load either a legacy single dataset or a CLRS-style task mixture."""
    if config.data.task_paths is None:
        path = {
            "train": config.data.train_path,
            "val": config.data.val_path,
            "test": config.data.test_path,
        }[split]
        return load_dataset(path)

    split_key = f"{split}_path"
    dataset = []
    algorithm_order = tuple(config.model.algorithms)
    for algorithm in algorithm_order:
        if not selected_tasks.get(algorithm, False):
            continue
        raw_dataset = load_dataset(config.data.task_paths[algorithm][split_key])
        dataset.extend(
            materialize_graph_sample(graph, algorithm_order, active_task=algorithm)
            for graph in raw_dataset
        )
    return dataset


def graph_execution_loss_fn(model, graph_data, embed_dim, termination_cfg, selected_tasks):
    """Compute loss for graph execution task.
    
    Args:
        model: NGE model
        graph_data: Graph data dictionary
        embed_dim: Embedding dimension
        
    Returns:
        Tuple of (average_loss, per_task_losses)
    """
    algorithm_order = tuple(model.algorithms)
    current_metric_names = metric_names(algorithm_order)
    current_metric_index = {name: index for index, name in enumerate(current_metric_names)}
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([len(current_metric_names)])

    num_nodes = graph_data["num_nodes"]
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])

    step_counts = execution_step_counts(graph_data, algorithm_order)
    num_steps = max(step_counts.values(), default=0) + 1
    termination_settings = resolve_termination_settings(termination_cfg)
    previous_distance_latent = None
    sample_selected_tasks = selected_tasks_for_graph(
        graph_data, selected_tasks, algorithm_order
    )
    loss_mask = metric_mask(sample_selected_tasks, algorithm_order)

    for i in range(num_steps):
        sample_exists = {algorithm: i < step_counts[algorithm] for algorithm in algorithm_order}
        if not any(sample_exists.values()):
            continue

        feature_values = feature_values_for_step(graph_data, i, algorithm_order)
        targets = targets_for_step(graph_data, i, algorithm_order)
        termination_targets = termination_targets_for_step(graph_data, i, algorithm_order)

        node_algo_features = build_node_algo_features(feature_values, algorithm_order)
        input_embeddings = mx.concatenate(
            [previous_step_hidden_states, node_algo_features], axis=1
        )
        model_input = (input_embeddings, graph_data["edge_matrix"])

        need_aux = needs_aux_latents(termination_settings)
        if need_aux:
            algorithm_outputs, termination_probs, processed_embeddings, aux = normalize_model_forward(
                model(model_input, return_latents=True),
                algorithm_order,
                return_latents=True,
            )
        else:
            algorithm_outputs, termination_probs, processed_embeddings = normalize_model_forward(
                model(model_input),
                algorithm_order,
                return_latents=False,
            )
            aux = None

        if termination_settings["mode"] == "distance":
            current_latent = get_distance_latent(
                termination_settings, processed_embeddings, aux
            )
            termination_logits = compute_distance_termination_logits(
                settings=termination_settings,
                prev_latent=previous_distance_latent,
                current_latent=current_latent,
                algorithms=algorithm_order,
            )
            previous_distance_latent = current_latent
        else:
            termination_logits = termination_probs

        raw_losses = mx.zeros([len(current_metric_names)])
        for algorithm in algorithm_order:
            if not sample_exists[algorithm] or not sample_selected_tasks.get(algorithm, False):
                continue

            family = algorithm_family(algorithm)
            target = targets[algorithm]
            output = algorithm_outputs[algorithm]

            if family == "state_mask":
                state_loss = nn.losses.binary_cross_entropy(
                    output,
                    target["state"],
                    reduction="mean",
                    with_logits=True,
                )
                raw_losses = raw_losses.at[current_metric_index[f"{algorithm}_state"]].add(
                    state_loss
                )
            elif family == "shortest_path":
                distance_predictions, predecessor_predictions = output
                distance_loss = nn.losses.mse_loss(
                    distance_predictions,
                    target["distance"],
                    reduction="mean",
                )
                valid_mask = target["predecessor"] != -1
                safe_targets = mx.where(
                    valid_mask,
                    target["predecessor"],
                    mx.zeros_like(target["predecessor"]),
                )
                per_node_ce = nn.losses.cross_entropy(
                    predecessor_predictions,
                    safe_targets,
                    reduction="none",
                )
                valid_mask_f = valid_mask.astype(mx.float32)
                denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
                predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
                raw_losses = raw_losses.at[
                    current_metric_index[f"{algorithm}_distance"]
                ].add(distance_loss)
                raw_losses = raw_losses.at[
                    current_metric_index[f"{algorithm}_predecessor"]
                ].add(predecessor_loss)
            elif family == "mst":
                state_predictions, key_predictions, predecessor_predictions = output
                state_loss = nn.losses.binary_cross_entropy(
                    state_predictions,
                    target["state"],
                    reduction="mean",
                    with_logits=True,
                )
                key_loss = nn.losses.mse_loss(
                    key_predictions,
                    target["key"],
                    reduction="mean",
                )
                valid_mask = target["predecessor"] != -1
                safe_targets = mx.where(
                    valid_mask,
                    target["predecessor"],
                    mx.zeros_like(target["predecessor"]),
                )
                per_node_ce = nn.losses.cross_entropy(
                    predecessor_predictions,
                    safe_targets,
                    reduction="none",
                )
                valid_mask_f = valid_mask.astype(mx.float32)
                denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
                predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
                raw_losses = raw_losses.at[current_metric_index[f"{algorithm}_state"]].add(
                    state_loss
                )
                raw_losses = raw_losses.at[current_metric_index[f"{algorithm}_key"]].add(
                    key_loss
                )
                raw_losses = raw_losses.at[
                    current_metric_index[f"{algorithm}_predecessor"]
                ].add(predecessor_loss)
            else:
                raise ValueError(f"Unsupported algorithm family: {family}")

            termination_loss = nn.losses.binary_cross_entropy(
                termination_logits[algorithm],
                termination_targets[algorithm],
                reduction="mean",
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                termination_loss = mx.array(0.0)
            raw_losses = raw_losses.at[
                current_metric_index[f"{algorithm}_termination"]
            ].add(termination_loss)

        raw_losses = raw_losses * loss_mask
        total_step_loss = mx.sum(raw_losses)

        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss
        accumulated_aux_losses += raw_losses

    effective_steps = effective_step_count(step_counts, sample_selected_tasks)
    average_loss = accumulated_loss / effective_steps
    per_task_counter = metric_counters(step_counts, algorithm_order)
    avg_aux_losses = accumulated_aux_losses / per_task_counter

    return average_loss, avg_aux_losses


def evaluate_model(model, dataset, embed_dim, termination_cfg, selected_tasks):
    """Evaluate model on a dataset.
    
    Args:
        model: NGE model
        dataset: List of graph data dictionaries
        embed_dim: Embedding dimension
        termination_cfg: ModelConfig controlling termination behavior
        selected_tasks: Dict with task enable flags for bf/bfs
        
    Returns:
        Tuple of (avg_aux_losses, avg_loss, avg_accuracies)
    """
    model.eval()
    current_metric_names = metric_names(model.algorithms)

    accumulated_epoch_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([len(current_metric_names)])
    accumulated_accuracies = mx.zeros([len(current_metric_names)])
    accumulated_metric_presence = mx.zeros([len(current_metric_names)])

    for graph_data in dataset:
        sample_selected_tasks = selected_tasks_for_graph(
            graph_data, selected_tasks, model.algorithms
        )
        aux_losses, loss, accuracies = calculate_losses_and_accuracies(
            model, graph_data, embed_dim, termination_cfg, selected_tasks
        )
        accumulated_epoch_loss += loss
        sample_mask = metric_mask(sample_selected_tasks, model.algorithms)
        accumulated_aux_losses += aux_losses * sample_mask
        accumulated_accuracies += accuracies * sample_mask
        accumulated_metric_presence += sample_mask
        _eval_trees(
            aux_losses,
            loss,
            accuracies,
            accumulated_epoch_loss,
            accumulated_aux_losses,
            accumulated_accuracies,
            accumulated_metric_presence,
        )

    avg_epoch_loss = accumulated_epoch_loss / len(dataset)
    safe_presence = mx.maximum(accumulated_metric_presence, mx.ones_like(accumulated_metric_presence))
    avg_aux_losses = accumulated_aux_losses / safe_presence
    avg_accuracies = accumulated_accuracies / safe_presence
    _eval_trees(avg_epoch_loss, avg_aux_losses, avg_accuracies)
    
    model.train()
    return avg_aux_losses, avg_epoch_loss, avg_accuracies


def evaluate_model_accuracies_only(model, dataset, embed_dim, termination_cfg, selected_tasks):
    """Evaluate model and return only mean accuracies over the dataset."""
    model.eval()
    current_metric_names = metric_names(model.algorithms)
    accumulated_accuracies = mx.zeros([len(current_metric_names)])
    accumulated_metric_presence = mx.zeros([len(current_metric_names)])
    for graph_data in dataset:
        sample_selected_tasks = selected_tasks_for_graph(
            graph_data, selected_tasks, model.algorithms
        )
        accuracies = calculate_accuracies(
            model=model,
            graph_data=graph_data,
            embedding_dim=embed_dim,
            termination_cfg=termination_cfg,
            selected_tasks=selected_tasks,
        )
        sample_mask = metric_mask(sample_selected_tasks, model.algorithms)
        accumulated_accuracies += accuracies * sample_mask
        accumulated_metric_presence += sample_mask
        _eval_trees(accuracies, accumulated_accuracies, accumulated_metric_presence)
    safe_presence = mx.maximum(accumulated_metric_presence, mx.ones_like(accumulated_metric_presence))
    avg_accuracies = accumulated_accuracies / safe_presence
    _eval_trees(avg_accuracies)
    model.train()
    return avg_accuracies


def save_termination_mispredict_distribution_plot(
    distribution: dict,
    output_path: Path,
    split: str,
) -> None:
    """Plot termination mispredict counts across execution steps."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render failure-distribution plots."
        ) from exc

    algorithms = list(distribution.keys())
    if not algorithms:
        raise ValueError("Termination mispredict distribution is empty.")

    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(5.5 * len(algorithms), 5),
        sharey=True,
    )
    if len(algorithms) == 1:
        axes = [axes]

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#17becf"]
    for ax, algorithm, color in zip(axes, algorithms, palette * ((len(algorithms) // len(palette)) + 1)):
        title = f"{algorithm_display_name(algorithm)} termination mispredicts"
        payload = distribution.get(algorithm, {})
        counts = payload.get("counts_by_step", {})
        totals = payload.get("totals_by_step", {})
        steps = sorted({int(s) for s in totals.keys()} | {int(s) for s in counts.keys()})
        y = [int(counts.get(str(step), counts.get(step, 0))) for step in steps]
        bars = ax.bar(steps, y, color=color, alpha=0.9, width=0.8)
        labels = []
        for step in steps:
            total = int(totals.get(str(step), totals.get(step, 0)))
            errors = int(counts.get(str(step), counts.get(step, 0)))
            labels.append(f"{errors}/{total}")
        if labels:
            ax.bar_label(bars, labels=labels, padding=2, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Execution step")
        ax.set_xticks(steps)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Termination mispredict count")
    fig.suptitle(f"Termination mispredict distribution by step ({split})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def train_epoch(
    model,
    dataset,
    optimizer,
    embed_dim,
    batch_size,
    max_grad_norm,
    logger: MetricsLogger,
    epoch: int,
    termination_cfg,
    selected_tasks,
    frozen_module_paths: tuple[str, ...] = (),
    log_interval: int = 1,
):
    """Train for one epoch.
    
    Args:
        model: NGE model
        dataset: Training dataset
        optimizer: Optimizer
        embed_dim: Embedding dimension
        batch_size: Batch size for gradient accumulation
        max_grad_norm: Maximum gradient norm for clipping
        logger: Metrics logger
        epoch: Current epoch number
        termination_cfg: ModelConfig controlling termination behavior
        selected_tasks: Dict with task enable flags for bf/bfs
        frozen_module_paths: Dotted module prefixes whose gradients are zeroed
        log_interval: How often to log metrics
        
    Returns:
        Tuple of (avg_loss, avg_aux_losses)
    """
    model.train()
    algorithm_order = tuple(model.algorithms)
    
    loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)
    
    accumulated_epoch_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([len(metric_names(algorithm_order))])
    accumulated_metric_presence = mx.zeros([len(metric_names(algorithm_order))])
    accumulated_per_head_grads = {}
    
    permutation = mx.random.permutation(len(dataset))
    
    acc_batch_grads = None
    bucket_count = 0
    
    for idx_in_epoch, idx in enumerate(permutation):
        graph_data = dataset[int(idx.item())]
        
        (loss, aux_losses), grads = loss_and_grad_fn(
            model, graph_data, embed_dim, termination_cfg, selected_tasks
        )
        grads = zero_frozen_gradients(grads, frozen_module_paths)

        per_head_magnitude_grads = extract_per_head_magnitude_grads(grads)
        _eval_trees(loss, aux_losses, grads, per_head_magnitude_grads)
        
        # Accumulate per-head gradients
        for head_name, grad_value in per_head_magnitude_grads.items():
            if head_name not in accumulated_per_head_grads:
                accumulated_per_head_grads[head_name] = grad_value
            else:
                accumulated_per_head_grads[head_name] += grad_value
        
        # Gradient accumulation
        if acc_batch_grads is None:
            acc_batch_grads = grads
        else:
            acc_batch_grads = utils.tree_map(lambda a, b: a + b, acc_batch_grads, grads)
        bucket_count += 1

        # Keep the lazy graph bounded while preserving gradient accumulation.
        accumulated_epoch_loss += loss
        sample_selected_tasks = selected_tasks_for_graph(
            graph_data, selected_tasks, algorithm_order
        )
        sample_mask = metric_mask(sample_selected_tasks, algorithm_order)
        accumulated_aux_losses += aux_losses * sample_mask
        accumulated_metric_presence += sample_mask
        _eval_trees(
            acc_batch_grads,
            accumulated_epoch_loss,
            accumulated_aux_losses,
            accumulated_metric_presence,
            accumulated_per_head_grads,
        )
        
        end_of_bucket = (bucket_count == batch_size)
        end_of_epoch  = (idx_in_epoch + 1 == len(permutation))
        if end_of_bucket or end_of_epoch:
            # Average gradients
            avg_grads = utils.tree_map(lambda x: x / bucket_count, acc_batch_grads)
            
            # Clip gradients
            avg_grads, norm = optim.clip_grad_norm(avg_grads, max_norm=max_grad_norm)
            
            # Update model
            optimizer.update(model, avg_grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # Reset for next bucket
            acc_batch_grads = None
            bucket_count = 0
    
    avg_epoch_loss = accumulated_epoch_loss / len(dataset)
    safe_presence = mx.maximum(accumulated_metric_presence, mx.ones_like(accumulated_metric_presence))
    avg_aux_losses = accumulated_aux_losses / safe_presence
    
    # Compute average per-head gradients
    avg_per_head_grads = {
        head_name: grad_value / len(dataset)
        for head_name, grad_value in accumulated_per_head_grads.items()
    }
    
    # Log metrics
    if epoch % log_interval == 0:
        metrics = {
            "loss": float(avg_epoch_loss),
            "lr": float(optimizer.learning_rate),
        }
        metrics.update(metric_dict("losses", avg_aux_losses, algorithm_order))
        
        # Add gradient norms
        for head_name, grad_value in avg_per_head_grads.items():
            metrics[f"grad_avg/{head_name}"] = float(grad_value)
        
        logger.log(epoch, metrics, split="train")
        print(f"Epoch {epoch}: loss = {avg_epoch_loss:.6f}")
    
    return avg_epoch_loss, avg_aux_losses


def main():
    """Main training function."""
    args = parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    config_path = None
    if args.eval_only:
        # In eval-only mode, explicit --config takes precedence over --run-dir config.
        if args.config:
            config_path = Path(args.config)
        elif run_dir is not None:
            config_path = run_dir / "config_resolved.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config_resolved.yaml in run dir: {run_dir}")
        else:
            raise ValueError("Provide --run-dir or --config for --eval-only.")
    else:
        if args.config:
            config_path = Path(args.config)
        elif args.resume and run_dir is not None:
            config_path = run_dir / "config_resolved.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config_resolved.yaml in run dir: {run_dir}")
        else:
            raise ValueError("--config is required unless --eval-only with --run-dir.")

    config = load_config(config_path)
    validate_config(config)
    if args.termination_mode is not None:
        config.model.termination_mode = args.termination_mode
    if args.termination_threshold is not None:
        if args.termination_threshold < 0:
            raise ValueError("--termination-threshold must be non-negative.")
        config.model.termination_distance_threshold = float(args.termination_threshold)
    if args.termination_latent is not None:
        config.model.termination_distance_latent = args.termination_latent
    if args.disable_distance_termination_signal:
        config.model.termination_distance_signal = False
    if args.failure_max_graphs is not None and args.failure_max_graphs < 0:
        raise ValueError("--failure-max-graphs must be non-negative.")
    if args.failure_max_records is not None and args.failure_max_records < 0:
        raise ValueError("--failure-max-records must be non-negative.")
    if args.failure_debug_top_k < 0:
        raise ValueError("--failure-debug-top-k must be non-negative.")
    if args.resume and config.training.init_checkpoint:
        raise ValueError("Use either --resume or training.init_checkpoint, not both.")
    tasks_selection = resolve_task_selection(args, config)
    selected_tasks = resolve_selected_tasks(tasks_selection, config.model.algorithms)

    print("=" * 80)
    print(f"Experiment: {config.name}")
    print(f"Config source: {config_path}")
    print(f"Selected tasks: {tasks_selection}")
    print(
        "Data mode: "
        + ("task-mixture" if config.data.task_paths is not None else "single-dataset")
    )
    print(
        "Termination settings: "
        f"mode={config.model.termination_mode}, "
        f"distance={config.model.termination_distance}, "
        f"latent={config.model.termination_distance_latent}, "
        f"threshold={config.model.termination_distance_threshold}, "
        f"distance_signal={config.model.termination_distance_signal}"
    )
    if config.training.init_checkpoint:
        print(f"Init checkpoint: {config.training.init_checkpoint}")
    if config.training.init_checkpoint_modules:
        print(
            "Init checkpoint modules: "
            + ", ".join(config.training.init_checkpoint_modules)
        )
    if config.training.freeze_modules:
        print(f"Frozen modules: {', '.join(config.training.freeze_modules)}")
    if config.training.reset_modules:
        print(f"Reset modules: {', '.join(config.training.reset_modules)}")
    if args.termination_threshold is not None and config.model.termination_mode != "distance":
        print(
            "Note: --termination-threshold is set but termination_mode is not 'distance'; "
            "threshold does not affect termination logits in head mode."
        )
    if args.termination_latent is not None and config.model.termination_mode != "distance":
        print(
            "Note: --termination-latent is set but termination_mode is not 'distance'; "
            "latent selection does not affect termination logits in head mode."
        )
    if args.disable_distance_termination_signal and config.model.termination_mode != "distance":
        print(
            "Note: --disable-distance-termination-signal is set but termination_mode is not "
            "'distance'; this flag has no effect in head mode."
        )
    if args.accuracies_only and not args.eval_only:
        print("Note: --accuracies-only only affects --eval-only mode.")
    if args.analyze_failures and not args.eval_only:
        print("Note: --analyze-failures only affects --eval-only mode.")
    print("=" * 80)

    if args.eval_only:
        set_seed(config.training.seed)
        model = create_model(config)
        model.eval()

        checkpoint_path = None
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        elif run_dir is not None:
            checkpoint_path = run_dir / "checkpoints"

        if checkpoint_path is None:
            raise ValueError("Provide --checkpoint or --run-dir for --eval-only.")

        if checkpoint_path.is_file():
            checkpoint_dir = checkpoint_path.parent
        else:
            checkpoint_dir = checkpoint_path
        manager = CheckpointManager(checkpoint_dir)
        if checkpoint_path.name == "checkpoints":
            model, _, step = manager.load(model, optimizer=None, checkpoint_path=None)
        else:
            model, _, step = manager.load(model, optimizer=None, checkpoint_path=checkpoint_path)

        print("\nLoading datasets...")
        train_dataset = load_split_dataset(config, "train", selected_tasks)
        val_dataset = load_split_dataset(config, "val", selected_tasks)
        test_dataset = load_split_dataset(config, "test", selected_tasks)
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        output_dir = run_dir / "analysis" if run_dir else Path("analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\nRunning evaluation...")
        if args.accuracies_only:
            val_accuracies = evaluate_model_accuracies_only(
                model, val_dataset, config.model.embed_dim, config.model, selected_tasks
            )
            test_accuracies = evaluate_model_accuracies_only(
                model, test_dataset, config.model.embed_dim, config.model, selected_tasks
            )
            val_aux_losses = None
            test_aux_losses = None
            val_loss = None
            test_loss = None
        else:
            val_aux_losses, val_loss, val_accuracies = evaluate_model(
                model, val_dataset, config.model.embed_dim, config.model, selected_tasks
            )
            test_aux_losses, test_loss, test_accuracies = evaluate_model(
                model, test_dataset, config.model.embed_dim, config.model, selected_tasks
            )

        val_payload = metric_dict("acc", val_accuracies, model.algorithms)
        test_payload = metric_dict("acc", test_accuracies, model.algorithms)
        if not args.accuracies_only:
            val_payload["loss"] = float(val_loss)
            val_payload.update(metric_dict("losses", val_aux_losses, model.algorithms))
            test_payload["loss"] = float(test_loss)
            test_payload.update(metric_dict("losses", test_aux_losses, model.algorithms))

        results = {
            "checkpoint_step": step,
            "eval_mode": "accuracies_only" if args.accuracies_only else "full",
            "selected_tasks": tasks_selection,
            "termination": {
                "mode": config.model.termination_mode,
                "distance": config.model.termination_distance,
                "latent": config.model.termination_distance_latent,
                "threshold": float(config.model.termination_distance_threshold),
                "distance_signal": bool(config.model.termination_distance_signal),
            },
            "val": val_payload,
            "test": test_payload,
        }

        if not args.accuracies_only:
            print(f"Val loss: {val_loss:.6f}")
        print("Val accuracies: " + format_accuracy_summary(val_accuracies, model.algorithms))
        if not args.accuracies_only:
            print(f"Test loss: {test_loss:.6f}")
        print("Test accuracies: " + format_accuracy_summary(test_accuracies, model.algorithms))

        with open(output_dir / "eval_only.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved eval-only results to: {output_dir / 'eval_only.json'}")

        if args.analyze_failures:
            split_to_dataset = {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
            }
            failure_dataset = split_to_dataset[args.failure_split]
            failure_report = analyze_failure_modes(
                model=model,
                dataset=failure_dataset,
                embedding_dim=config.model.embed_dim,
                termination_cfg=config.model,
                selected_tasks=selected_tasks,
                max_graphs=args.failure_max_graphs,
                max_failure_records=args.failure_max_records,
                include_step_details=args.failure_include_step_details,
            )
            failure_report["split"] = args.failure_split

            debug_indices = parse_graph_indices(args.failure_debug_graphs)
            if args.failure_debug_top_k > 0:
                debug_indices = sorted(
                    set(
                        debug_indices
                        + failure_report["ranked_failed_graph_indices"][: args.failure_debug_top_k]
                    )
                )
            debug_paths = []
            if debug_indices:
                debug_dir = output_dir / "failure_debug" / args.failure_split
                debug_dir.mkdir(parents=True, exist_ok=True)
                for graph_index in debug_indices:
                    if graph_index >= len(failure_dataset):
                        print(
                            f"Skipping debug dump for graph {graph_index}: out of range "
                            f"for split '{args.failure_split}' ({len(failure_dataset)} graphs)."
                        )
                        continue
                    debug_path = debug_dir / f"graph_{graph_index:04d}.txt"
                    with open(debug_path, "w") as f:
                        with contextlib.redirect_stdout(f):
                            print_execution_details(
                                model=model,
                                graph_data=failure_dataset[graph_index],
                                embedding_dim=config.model.embed_dim,
                                termination_cfg=config.model,
                            )
                    debug_paths.append(str(debug_path))
            failure_report["debug_dump_paths"] = debug_paths

            term_dist = failure_report.get("termination_mispredict_distribution")
            if term_dist is not None:
                dist_plot_path = (
                    output_dir
                    / f"failure_termination_mispredict_distribution_{args.failure_split}.png"
                )
                save_termination_mispredict_distribution_plot(
                    distribution=term_dist,
                    output_path=dist_plot_path,
                    split=args.failure_split,
                )
                failure_report["termination_mispredict_plot"] = str(dist_plot_path)

            failure_path = output_dir / f"failure_modes_{args.failure_split}.json"
            with open(failure_path, "w") as f:
                json.dump(failure_report, f, indent=2)
            print(f"Saved failure analysis to: {failure_path}")
            if term_dist is not None:
                print(
                    "Saved termination mispredict distribution plot to: "
                    f"{failure_report['termination_mispredict_plot']}"
                )
            if debug_paths:
                print(f"Saved {len(debug_paths)} debug execution traces under: {output_dir / 'failure_debug'}")
        return

    # Setup run directory
    run_dir = setup_run_directory(config, resume=args.resume, run_dir=args.run_dir)

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Initialize logger
    logger = MetricsLogger(
        log_file=run_dir / "metrics.jsonl",
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
        wandb_entity=config.logging.wandb_entity if config.logging.wandb_entity else None,
        wandb_config=config.to_dict(),
    )

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = load_split_dataset(config, "train", selected_tasks)
    val_dataset = load_split_dataset(config, "val", selected_tasks)
    test_dataset = load_split_dataset(config, "test", selected_tasks)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    init_step = initialize_from_checkpoint_if_needed(model, config)
    for module_path in config.training.freeze_modules:
        resolve_module_path(model, module_path)
    if config.training.reset_modules:
        fresh_model = create_model(config)
        reset_selected_modules(model, fresh_model, config.training.reset_modules)
    if init_step is not None:
        print(f"Loaded initialization checkpoint at step {init_step}.")
    model.train()

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config.training.learning_rate)

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(run_dir / "checkpoints")

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        try:
            latest_step = checkpoint_manager.get_latest_step()
            if latest_step is not None:
                print(f"\nResuming from step {latest_step}...")
                model, optimizer, loaded_epoch = checkpoint_manager.load(model, optimizer)
                # Checkpoints are saved with the current epoch index.
                # Resume must continue from the next epoch to avoid repeating work.
                start_epoch = int(loaded_epoch) + 1
                print(
                    f"Loaded checkpoint epoch {loaded_epoch}; "
                    f"continuing at epoch {start_epoch}"
                )
            else:
                print("\nNo checkpoint found, starting from scratch")
        except Exception as e:
            print(f"\nWarning: Could not load checkpoint: {e}")
            print("Starting from scratch")

    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    print(f"Configured epochs: {config.training.epochs}")

    if start_epoch >= config.training.epochs:
        print(
            f"\nConfig epochs ({config.training.epochs}) already reached "
            f"by checkpoint epoch {start_epoch}. Skipping training loop."
        )
        start_epoch = config.training.epochs

    for epoch in range(start_epoch, config.training.epochs):
        # Train for one epoch
        train_loss, train_aux_losses = train_epoch(
            model=model,
            dataset=train_dataset,
            optimizer=optimizer,
            embed_dim=config.model.embed_dim,
            batch_size=config.training.batch_size,
            max_grad_norm=config.training.max_grad_norm,
            logger=logger,
            epoch=epoch,
            termination_cfg=config.model,
            selected_tasks=selected_tasks,
            frozen_module_paths=tuple(config.training.freeze_modules),
            log_interval=config.logging.log_interval,
        )

        # Evaluation
        if (epoch + 1) % config.training.eval_interval == 0:
            print(f"\nEvaluating at epoch {epoch}...")

            # Validation
            val_aux_losses, val_loss, val_accuracies = evaluate_model(
                model, val_dataset, config.model.embed_dim, config.model, selected_tasks
            )
            _eval_trees(val_aux_losses, val_loss, val_accuracies)

            # Train subsample (for fair comparison with validation)
            train_subsample_size = len(val_dataset)
            train_subsample_indices = mx.random.permutation(len(train_dataset))[:train_subsample_size]
            train_subsample = [train_dataset[int(idx.item())] for idx in train_subsample_indices]
            _, _, train_accuracies = evaluate_model(
                model, train_subsample, config.model.embed_dim, config.model, selected_tasks
            )
            _eval_trees(train_accuracies)

            # Log validation metrics
            val_metrics = {"loss": float(val_loss)}
            val_metrics.update(metric_dict("acc", val_accuracies, model.algorithms))
            val_metrics.update(metric_dict("losses", val_aux_losses, model.algorithms))
            logger.log(epoch, val_metrics, split="val")

            # Log train accuracies
            train_acc_metrics = metric_dict("acc", train_accuracies, model.algorithms)
            logger.log(epoch, train_acc_metrics, split="train_eval")

            print(f"Val loss: {val_loss:.6f}")
            print("Val accuracies: " + format_accuracy_summary(val_accuracies, model.algorithms))

        # Checkpointing
        if config.logging.save_checkpoints and (epoch + 1) % config.logging.checkpoint_interval == 0:
            print(f"Saving checkpoint at epoch {epoch}...")
            checkpoint_manager.save(model, optimizer, epoch, metadata={'epoch': epoch})

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final evaluation on test set...")
    test_aux_losses, test_loss, test_accuracies = evaluate_model(
        model, test_dataset, config.model.embed_dim, config.model, selected_tasks
    )

    test_metrics = {"loss": float(test_loss)}
    test_metrics.update(metric_dict("acc", test_accuracies, model.algorithms))
    logger.log(config.training.epochs, test_metrics, split="test")
    logger.log_summary({"final_" + k: v for k, v in test_metrics.items()})

    print("\nTest Results:")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  Accuracies: {format_accuracy_summary(test_accuracies, model.algorithms)}")

    # Final checkpoint
    if config.logging.save_checkpoints:
        print("\nSaving final checkpoint...")
        checkpoint_manager.save(
            model, optimizer, config.training.epochs,
            metadata={'epoch': config.training.epochs, 'final': True}
        )

    # Cleanup old checkpoints according to retention policy.
    if config.logging.save_checkpoints:
        checkpoint_manager.cleanup_old_checkpoints(
            keep_last_n=config.logging.checkpoint_keep_last
        )

    logger.finish()
    print("\n" + "=" * 80)
    print(f"Training complete! Results saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
