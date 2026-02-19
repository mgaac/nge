"""Main training script for NGE model with research workflow.

Usage:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/baseline.yaml --resume
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils
import mlx.optimizers as optim

from src.model import NGE, AggregationFn
from src.data import load_dataset
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
    calculate_losses_and_accuracies,
    extract_per_head_magnitude_grads,
)
from src.utils.termination import (
    compute_distance_termination_logits,
    get_distance_latent,
    needs_aux_latents,
    resolve_termination_settings,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train NGE model with research workflow')
    parser.add_argument('--config', type=str, required=False,
                        help='Path to YAML config file (e.g., configs/baseline.yaml)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint in existing run')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Specific run directory to resume from (only with --resume)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint directory or file to load (for --eval-only)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training and run evaluation only')
    parser.add_argument('--tasks', type=str, default='all', choices=['all', 'bf', 'bfs'],
                        help='Tasks to optimize/evaluate: all, bf, or bfs')
    parser.add_argument('--termination-threshold', type=float, default=None,
                        help='Override termination_distance_threshold (useful in --eval-only)')
    parser.add_argument(
        '--termination-latent',
        type=str,
        default=None,
        choices=['processed', 'encoded', 'encoded_bfs', 'encoded_bf'],
        help='Override termination_distance_latent (useful in --eval-only)',
    )
    parser.add_argument('--disable-distance-termination-signal', action='store_true',
                        help='Disable termination BCE supervision when termination_mode=distance')
    return parser.parse_args()


def resolve_selected_tasks(tasks_arg: str) -> dict[str, bool]:
    """Resolve CLI task selection into task enable flags."""
    if tasks_arg == "all":
        return {"bf": True, "bfs": True}
    if tasks_arg == "bf":
        return {"bf": True, "bfs": False}
    if tasks_arg == "bfs":
        return {"bf": False, "bfs": True}
    raise ValueError(f"Unknown tasks selection: {tasks_arg}")


def effective_step_count(bf_steps: int, bfs_steps: int, selected_tasks: dict[str, bool]) -> int:
    """Choose the averaging denominator based on selected tasks."""
    if selected_tasks["bf"] and selected_tasks["bfs"]:
        return max(bf_steps, bfs_steps, 1)
    if selected_tasks["bf"]:
        return max(bf_steps, 1)
    if selected_tasks["bfs"]:
        return max(bfs_steps, 1)
    return 1


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
    )
    
    return model


def graph_execution_loss_fn(model, graph_data, embed_dim, termination_cfg, selected_tasks):
    """Compute loss for graph execution task.
    
    Args:
        model: NGE model
        graph_data: Graph data dictionary
        embed_dim: Embedding dimension
        
    Returns:
        Tuple of (average_loss, per_task_losses)
    """
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, embed_dim * 2])

    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])

    num_steps = max(num_bf_steps, num_bfs_steps)
    termination_settings = resolve_termination_settings(termination_cfg)
    previous_distance_latent = None
    
    loss_mask = mx.array(
        [
            1.0 if selected_tasks["bf"] else 0.0,
            1.0 if selected_tasks["bf"] else 0.0,
            1.0 if selected_tasks["bfs"] else 0.0,
            1.0 if selected_tasks["bf"] else 0.0,
            1.0 if selected_tasks["bfs"] else 0.0,
        ],
        dtype=mx.float32,
    )

    for i in range(num_steps):
        # Check if samples exist
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps

        if not (bf_sample_exists or bfs_sample_exists):
            continue

        # Prepare data for current step
        if bfs_sample_exists:
            true_bfs_state = graph_data['bfs_state_targets'][i]
            target_bfs_state = graph_data['bfs_state_targets'][i+1]
        else:
            true_bfs_state = graph_data['bfs_state_targets'][-1]
            target_bfs_state = graph_data['bfs_state_targets'][-1]

        if bf_sample_exists:
            true_distance_bf = graph_data['bf_distance_targets'][i]
            target_distance_bf = graph_data['bf_distance_targets'][i+1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][i+1]
        else:
            true_distance_bf = graph_data['bf_distance_targets'][-1]
            target_distance_bf = graph_data['bf_distance_targets'][-1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][-1]

        # Generate termination targets
        is_last_bf_step = (i + 1) == (num_bf_steps - 1)
        is_last_bfs_step = (i + 1) == (num_bfs_steps - 1)
        termination_targets = {
            'bf': mx.array(1.0 if is_last_bf_step else 0.0),
            'bfs': mx.array(1.0 if is_last_bfs_step else 0.0)
        }

        # Prepare model inputs
        node_algo_features = mx.stack([true_bfs_state, true_distance_bf], axis=1)
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])

        need_aux = needs_aux_latents(termination_settings)
        if need_aux:
            bfs_output, bf_output, termination_probs, processed_embeddings, aux = model(
                model_input, return_latents=True
            )
        else:
            bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)
            aux = None

        if termination_settings["mode"] == "distance":
            current_latent = get_distance_latent(termination_settings, processed_embeddings, aux)
            termination_logits = compute_distance_termination_logits(
                settings=termination_settings,
                prev_latent=previous_distance_latent,
                current_latent=current_latent,
            )
            previous_distance_latent = current_latent
        else:
            termination_logits = termination_probs

        # Compute losses
        if bf_sample_exists and selected_tasks["bf"]:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')

            # Convert invalid, denoted by -1, to a valid class, 0.
            valid_mask = (target_predecessor_bf != -1)
            safe_targets = mx.where(valid_mask, target_predecessor_bf,
                                    mx.zeros_like(target_predecessor_bf))  
            
            per_node_ce = nn.losses.cross_entropy(bf_predecessor_predictions, safe_targets, reduction='none')
            
            # Only consider loss over valid nodes
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom

            bf_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bf'],
                termination_targets['bf'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bf_termination_loss = mx.array(0.0)
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)

        if bfs_sample_exists and selected_tasks["bfs"]:
            bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean', with_logits=True)
            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bfs'],
                termination_targets['bfs'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bfs_termination_loss = mx.array(0.0)
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)

        raw_losses = mx.array(
            [bf_distance_loss, bf_predecessor_loss, bfs_state_loss, bf_termination_loss, bfs_termination_loss]
        )
        raw_losses = raw_losses * loss_mask
        total_step_loss = mx.sum(raw_losses)

        # Update for next step
        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss
        accumulated_aux_losses += raw_losses

    # Compute averages
    bf_steps  = max(num_bf_steps  - 1, 0)
    bfs_steps = max(num_bfs_steps - 1, 0)
    effective_steps = effective_step_count(bf_steps, bfs_steps, selected_tasks)

    average_loss = accumulated_loss / effective_steps
    per_task_counter = mx.array([
        max(bf_steps, 1),
        max(bf_steps, 1),
        max(bfs_steps, 1),
        max(bf_steps, 1),
        max(bfs_steps, 1),
    ], dtype=mx.float32)
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
    
    accumulated_epoch_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])
    accumulated_accuracies = mx.zeros([5])

    for graph_data in dataset:
        aux_losses, loss, accuracies = calculate_losses_and_accuracies(
            model, graph_data, embed_dim, termination_cfg, selected_tasks
        )
        accumulated_epoch_loss += loss
        accumulated_aux_losses += aux_losses
        accumulated_accuracies += accuracies

    avg_epoch_loss = accumulated_epoch_loss / len(dataset)
    avg_aux_losses = accumulated_aux_losses / len(dataset)
    avg_accuracies = accumulated_accuracies / len(dataset)
    
    model.train()
    return avg_aux_losses, avg_epoch_loss, avg_accuracies


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
        log_interval: How often to log metrics
        
    Returns:
        Tuple of (avg_loss, avg_aux_losses)
    """
    model.train()
    
    loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)
    
    accumulated_epoch_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])
    accumulated_per_head_grads = {}
    
    permutation = mx.random.permutation(len(dataset))
    
    acc_batch_grads = None
    bucket_count = 0
    
    for idx_in_epoch, idx in enumerate(permutation):
        graph_data = dataset[int(idx.item())]
        
        (loss, aux_losses), grads = loss_and_grad_fn(
            model, graph_data, embed_dim, termination_cfg, selected_tasks
        )
        
        per_head_magnitude_grads = extract_per_head_magnitude_grads(grads)
        
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
        
        # Book-keeping
        accumulated_epoch_loss += loss
        accumulated_aux_losses += aux_losses
    
    avg_epoch_loss = accumulated_epoch_loss / len(dataset)
    avg_aux_losses = accumulated_aux_losses / len(dataset)
    
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
            "losses/bf_distance": float(avg_aux_losses[0]),
            "losses/bf_predecessor": float(avg_aux_losses[1]),
            "losses/bfs_state": float(avg_aux_losses[2]),
            "losses/bf_termination": float(avg_aux_losses[3]),
            "losses/bfs_termination": float(avg_aux_losses[4]),
        }
        
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
    if args.termination_threshold is not None:
        if args.termination_threshold < 0:
            raise ValueError("--termination-threshold must be non-negative.")
        config.model.termination_distance_threshold = float(args.termination_threshold)
    if args.termination_latent is not None:
        config.model.termination_distance_latent = args.termination_latent
    if args.disable_distance_termination_signal:
        config.model.termination_distance_signal = False
    selected_tasks = resolve_selected_tasks(args.tasks)

    print("=" * 80)
    print(f"Experiment: {config.name}")
    print(f"Config source: {config_path}")
    print(f"Selected tasks: {args.tasks}")
    print(
        "Termination settings: "
        f"mode={config.model.termination_mode}, "
        f"distance={config.model.termination_distance}, "
        f"latent={config.model.termination_distance_latent}, "
        f"threshold={config.model.termination_distance_threshold}, "
        f"distance_signal={config.model.termination_distance_signal}"
    )
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
        train_dataset = load_dataset(config.data.train_path)
        val_dataset = load_dataset(config.data.val_path)
        test_dataset = load_dataset(config.data.test_path)
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        print("\nRunning evaluation...")
        val_aux_losses, val_loss, val_accuracies = evaluate_model(
            model, val_dataset, config.model.embed_dim, config.model, selected_tasks
        )
        test_aux_losses, test_loss, test_accuracies = evaluate_model(
            model, test_dataset, config.model.embed_dim, config.model, selected_tasks
        )

        results = {
            "checkpoint_step": step,
            "selected_tasks": args.tasks,
            "termination": {
                "mode": config.model.termination_mode,
                "distance": config.model.termination_distance,
                "latent": config.model.termination_distance_latent,
                "threshold": float(config.model.termination_distance_threshold),
                "distance_signal": bool(config.model.termination_distance_signal),
            },
            "val": {
                "loss": float(val_loss),
                "acc/bf_distance": float(val_accuracies[0]),
                "acc/bf_predecessor": float(val_accuracies[1]),
                "acc/bfs_state": float(val_accuracies[2]),
                "acc/bf_termination": float(val_accuracies[3]),
                "acc/bfs_termination": float(val_accuracies[4]),
                "losses/bf_distance": float(val_aux_losses[0]),
                "losses/bf_predecessor": float(val_aux_losses[1]),
                "losses/bfs_state": float(val_aux_losses[2]),
                "losses/bf_termination": float(val_aux_losses[3]),
                "losses/bfs_termination": float(val_aux_losses[4]),
            },
            "test": {
                "loss": float(test_loss),
                "acc/bf_distance": float(test_accuracies[0]),
                "acc/bf_predecessor": float(test_accuracies[1]),
                "acc/bfs_state": float(test_accuracies[2]),
                "acc/bf_termination": float(test_accuracies[3]),
                "acc/bfs_termination": float(test_accuracies[4]),
                "losses/bf_distance": float(test_aux_losses[0]),
                "losses/bf_predecessor": float(test_aux_losses[1]),
                "losses/bfs_state": float(test_aux_losses[2]),
                "losses/bf_termination": float(test_aux_losses[3]),
                "losses/bfs_termination": float(test_aux_losses[4]),
            },
        }

        print(f"Val loss: {val_loss:.6f}")
        print(
            f"Val accuracies: BF_dist={val_accuracies[0]:.3f}, "
            f"BF_pred={val_accuracies[1]:.3f}, BFS={val_accuracies[2]:.3f}, "
            f"BF_term={val_accuracies[3]:.3f}, BFS_term={val_accuracies[4]:.3f}"
        )
        print(f"Test loss: {test_loss:.6f}")
        print(
            f"Test accuracies: BF_dist={test_accuracies[0]:.3f}, "
            f"BF_pred={test_accuracies[1]:.3f}, BFS={test_accuracies[2]:.3f}, "
            f"BF_term={test_accuracies[3]:.3f}, BFS_term={test_accuracies[4]:.3f}"
        )

        output_dir = run_dir / "analysis" if run_dir else Path("analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "eval_only.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved eval-only results to: {output_dir / 'eval_only.json'}")
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
    train_dataset = load_dataset(config.data.train_path)
    val_dataset = load_dataset(config.data.val_path)
    test_dataset = load_dataset(config.data.test_path)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_model(config)
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
            log_interval=config.logging.log_interval,
        )

        # Evaluation
        if (epoch + 1) % config.training.eval_interval == 0:
            print(f"\nEvaluating at epoch {epoch}...")

            # Validation
            val_aux_losses, val_loss, val_accuracies = evaluate_model(
                model, val_dataset, config.model.embed_dim, config.model, selected_tasks
            )

            # Train subsample (for fair comparison with validation)
            train_subsample_size = len(val_dataset)
            train_subsample_indices = mx.random.permutation(len(train_dataset))[:train_subsample_size]
            train_subsample = [train_dataset[int(idx.item())] for idx in train_subsample_indices]
            _, _, train_accuracies = evaluate_model(
                model, train_subsample, config.model.embed_dim, config.model, selected_tasks
            )

            # Log validation metrics
            val_metrics = {
                "loss": float(val_loss),
                "acc/bf_distance": float(val_accuracies[0]),
                "acc/bf_predecessor": float(val_accuracies[1]),
                "acc/bfs_state": float(val_accuracies[2]),
                "acc/bf_termination": float(val_accuracies[3]),
                "acc/bfs_termination": float(val_accuracies[4]),
                "losses/bf_distance": float(val_aux_losses[0]),
                "losses/bf_predecessor": float(val_aux_losses[1]),
                "losses/bfs_state": float(val_aux_losses[2]),
                "losses/bf_termination": float(val_aux_losses[3]),
                "losses/bfs_termination": float(val_aux_losses[4]),
            }
            logger.log(epoch, val_metrics, split="val")

            # Log train accuracies
            train_acc_metrics = {
                "acc/bf_distance": float(train_accuracies[0]),
                "acc/bf_predecessor": float(train_accuracies[1]),
                "acc/bfs_state": float(train_accuracies[2]),
                "acc/bf_termination": float(train_accuracies[3]),
                "acc/bfs_termination": float(train_accuracies[4]),
            }
            logger.log(epoch, train_acc_metrics, split="train_eval")

            print(f"Val loss: {val_loss:.6f}")
            print(
                f"Val accuracies: BF_dist={val_accuracies[0]:.3f}, "
                f"BF_pred={val_accuracies[1]:.3f}, BFS={val_accuracies[2]:.3f}"
            )

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

    test_metrics = {
        "loss": float(test_loss),
        "acc/bf_distance": float(test_accuracies[0]),
        "acc/bf_predecessor": float(test_accuracies[1]),
        "acc/bfs_state": float(test_accuracies[2]),
        "acc/bf_termination": float(test_accuracies[3]),
        "acc/bfs_termination": float(test_accuracies[4]),
    }
    logger.log(config.training.epochs, test_metrics, split="test")
    logger.log_summary({"final_" + k: v for k, v in test_metrics.items()})

    print("\nTest Results:")
    print(f"  Loss: {test_loss:.6f}")
    print(f"  BF Distance Acc: {test_accuracies[0]:.3f}")
    print(f"  BF Predecessor Acc: {test_accuracies[1]:.3f}")
    print(f"  BFS State Acc: {test_accuracies[2]:.3f}")
    print(f"  BF Termination Acc: {test_accuracies[3]:.3f}")
    print(f"  BFS Termination Acc: {test_accuracies[4]:.3f}")

    # Final checkpoint
    if config.logging.save_checkpoints:
        print("\nSaving final checkpoint...")
        checkpoint_manager.save(
            model, optimizer, config.training.epochs,
            metadata={'epoch': config.training.epochs, 'final': True}
        )

    # Cleanup old checkpoints (keep last 5)
    if config.logging.save_checkpoints:
        checkpoint_manager.cleanup_old_checkpoints(keep_last_n=5)

    logger.finish()
    print("\n" + "=" * 80)
    print(f"Training complete! Results saved to: {run_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
