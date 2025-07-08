import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils

import mlx.optimizers as optim

import numpy as np

from model import nge, aggregation_fn
from gg_lean import load_dataset
from logger import (
    log_graph_execution_details, log_step_details, log_model_inputs,
    log_model_outputs, log_targets, log_losses, log_execution_summary,
    log_epoch_loss, set_verbose,
    # WandB functions
    init_wandb, log_wandb_epoch, finish_wandb
)

MODEL_CONFIG = {
    'embedding_dim': 128,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 2
}

# Initialize WandB with model configuration
wandb_logger = init_wandb("nge-training", enabled=True)
if wandb_logger and wandb_logger.enabled:
    # Log model configuration
    config_dict = {
        'embedding_dim': MODEL_CONFIG['embedding_dim'],
        'skip_connections': MODEL_CONFIG['skip_connections'],
        'aggregation_fn': MODEL_CONFIG['aggregation_fn'].name,
        'num_mp_layers': MODEL_CONFIG['num_mp_layers']
    }
    wandb_logger.log_config(config_dict)

model = nge(**MODEL_CONFIG)

dataset = load_dataset('dataset.npz')  # or use your custom filename

def bf_loss_fn(bf_output, bf_distance_targets, bf_predecessor_targets):

    bf_distance_predictions, bf_predecessor_predictions = bf_output

    bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, bf_distance_targets, reduction='mean')
    bf_distance_loss = bf_distance_loss * 0.1  # Scale down distance loss

    bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, bf_predecessor_targets, reduction='mean')

    return bf_distance_loss


def bfs_loss_fn(bfs_output, bfs_state_targets):

    bfs_state_predictions = bfs_output
    bfs_state_loss = nn.losses.binary_cross_entropy(bfs_state_predictions, bfs_state_targets, reduction='mean', with_logits=True)

    return bfs_state_loss


def step_loss_fn(bfs_output, bf_output, termination_probs, target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets):

    bf_loss_value = bf_loss_fn(bf_output, target_distance_bf, target_predecessor_bf)
    bfs_loss_value = bfs_loss_fn(bfs_output, target_bfs_state)

    bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean', with_logits=False)
    bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean', with_logits=False)

    total_step_loss = bf_loss_value + bfs_loss_value + bf_termination_loss + bfs_termination_loss

    return total_step_loss


def graph_execution_loss_fn(model, graph_data):
    accumulated_loss = mx.array(0.0)
    
    # Initialize component loss accumulators
    accumulated_bf_distance_loss = mx.array(0.0)
    accumulated_bf_predecessor_loss = mx.array(0.0)
    accumulated_bfs_state_loss = mx.array(0.0)
    accumulated_bf_termination_loss = mx.array(0.0)
    accumulated_bfs_termination_loss = mx.array(0.0)

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, MODEL_CONFIG['embedding_dim']])

    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])

    num_steps = max(num_bf_steps, num_bfs_steps)
    
    # Log graph execution details
    log_graph_execution_details(graph_data, num_steps, MODEL_CONFIG)
    
    for i in range(num_steps):
        # Check if samples exist
        bf_sample_exists = i < num_bf_steps and (i + 1) < num_bf_steps
        bfs_sample_exists = i < num_bfs_steps and (i + 1) < num_bfs_steps

        # Log step details
        log_step_details(i, bf_sample_exists, bfs_sample_exists)

        # If neither sample exists, skip this step
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
            true_predecessor_bf = graph_data['bf_predecessor_targets'][i]
            target_distance_bf = graph_data['bf_distance_targets'][i+1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][i+1]
        else:
            true_distance_bf = graph_data['bf_distance_targets'][-1]
            true_predecessor_bf = graph_data['bf_predecessor_targets'][-1]
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
        node_algo_features = mx.concatenate([true_bfs_state, true_distance_bf, true_predecessor_bf]).reshape([-1, 3])
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])

        # Log model inputs
        log_model_inputs(input_embeddings, graph_data['edge_matrix'], previous_step_hidden_states, 
                        node_algo_features, true_bfs_state, true_distance_bf, true_predecessor_bf, num_nodes)

        # Forward pass
        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)

        # Log model outputs
        log_model_outputs(bfs_output, bf_output, termination_probs, processed_embeddings, num_nodes)

        # Log targets
        log_targets(target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets, num_nodes)

        # Compute losses
        if bf_sample_exists:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')
            bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, target_predecessor_bf, reduction='mean')
            bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)

        if bfs_sample_exists:
            bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean', with_logits=True)
            bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean', with_logits=True)
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)

        total_step_loss = bf_distance_loss + bf_predecessor_loss + bfs_state_loss + bf_termination_loss + bfs_termination_loss

        # Log losses
        log_losses(bf_distance_loss, bf_predecessor_loss, bfs_state_loss, bf_termination_loss, bfs_termination_loss, total_step_loss)

        # Accumulate component losses
        accumulated_bf_distance_loss += bf_distance_loss
        accumulated_bf_predecessor_loss += bf_predecessor_loss
        accumulated_bfs_state_loss += bfs_state_loss
        accumulated_bf_termination_loss += bf_termination_loss
        accumulated_bfs_termination_loss += bfs_termination_loss

        # Update for next step
        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss

    average_loss = accumulated_loss / (num_steps - 1)
    
    # Calculate average component losses
    avg_bf_distance_loss = accumulated_bf_distance_loss / (num_steps - 1)
    avg_bf_predecessor_loss = accumulated_bf_predecessor_loss / (num_steps - 1)
    avg_bfs_state_loss = accumulated_bfs_state_loss / (num_steps - 1)
    avg_bf_termination_loss = accumulated_bf_termination_loss / (num_steps - 1)
    avg_bfs_termination_loss = accumulated_bfs_termination_loss / (num_steps - 1)
    
    # Log execution summary
    log_execution_summary(average_loss)

    return average_loss, {
        'bf_distance': avg_bf_distance_loss,
        'bf_predecessor': avg_bf_predecessor_loss,
        'bfs_state': avg_bfs_state_loss,
        'bf_termination': avg_bf_termination_loss,
        'bfs_termination': avg_bfs_termination_loss
    }


loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)

mx.eval(model.parameters())


def compute_first_graph_metrics(model, graph_data, component_losses):
    """Compute detailed metrics for the first graph for monitoring"""
    try:
        # Run a forward pass to get predictions
        num_nodes = graph_data['num_nodes']
        previous_step_hidden_states = mx.zeros([num_nodes, MODEL_CONFIG['embedding_dim']])
        
        # Use first step data for metrics computation
        true_bfs_state = graph_data['bfs_state_targets'][0]
        true_distance_bf = graph_data['bf_distance_targets'][0]
        true_predecessor_bf = graph_data['bf_predecessor_targets'][0]
        
        # Get next step targets
        target_bfs_state = graph_data['bfs_state_targets'][1]
        target_distance_bf = graph_data['bf_distance_targets'][1]
        target_predecessor_bf = graph_data['bf_predecessor_targets'][1]
        
        # Prepare model input
        node_algo_features = mx.concatenate([true_bfs_state, true_distance_bf, true_predecessor_bf]).reshape([-1, 3])
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])
        
        # Forward pass
        bfs_output, bf_output, termination_probs, _ = model(model_input)
        bf_distance_predictions, bf_predecessor_predictions = bf_output
        
        # Compute accuracies
        metrics = {}
        
        # BFS accuracy
        bfs_predictions = mx.sigmoid(bfs_output) > 0.5
        bfs_accuracy = float(mx.mean((bfs_predictions == target_bfs_state).astype(mx.float32)))
        metrics['bfs_accuracy'] = bfs_accuracy
        
        # BF distance accuracy (within tolerance)
        distance_diff = mx.abs(bf_distance_predictions - target_distance_bf)
        bf_distance_accuracy = float(mx.mean((distance_diff < 0.1).astype(mx.float32)))
        metrics['bf_distance_accuracy'] = bf_distance_accuracy
        
        # BF predecessor accuracy
        pred_predecessors = mx.argmax(bf_predecessor_predictions, axis=1)
        bf_predecessor_accuracy = float(mx.mean((pred_predecessors == target_predecessor_bf).astype(mx.float32)))
        metrics['bf_predecessor_accuracy'] = bf_predecessor_accuracy
        
        # Termination accuracies
        bfs_term_target = 1.0 if len(graph_data['bfs_state_targets']) <= 2 else 0.0
        bf_term_target = 1.0 if len(graph_data['bf_distance_targets']) <= 2 else 0.0
        
        bfs_term_pred = termination_probs['bfs'] > 0.5
        bf_term_pred = termination_probs['bf'] > 0.5
        
        metrics['bfs_term_accuracy'] = float(bfs_term_pred == bfs_term_target)
        metrics['bf_term_accuracy'] = float(bf_term_pred == bf_term_target)
        
        # Include component losses
        for key, value in component_losses.items():
            metrics[f"{key}_loss"] = float(value)
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Could not compute first graph metrics: {e}")
        return {}


def train_model(model, dataset, optimizer, epochs, print_last_epoch_only=False):

    for epoch in range(epochs):
        
        # Set verbosity based on print_last_epoch_only flag
        if print_last_epoch_only:
            set_verbose(epoch == epochs - 1)  # Only verbose on last epoch
        else:
            set_verbose(True)  #  verbose

        total_epoch_loss = 0
        
        # Initialize epoch component loss accumulators
        epoch_component_losses = {
            'bf_distance': 0.0,
            'bf_predecessor': 0.0,
            'bfs_state': 0.0,
            'bf_termination': 0.0,
            'bfs_termination': 0.0
        }

        # Track first graph metrics for detailed monitoring
        first_graph_metrics = {}

        for i, graph_data in enumerate(dataset):

            (loss, component_losses), grads = loss_and_grad_fn(model, graph_data)

            optimizer.update(model, grads)

            mx.eval(model.parameters(), optimizer.state)

            total_epoch_loss += loss
            
            # Accumulate component losses
            for key in epoch_component_losses:
                epoch_component_losses[key] += float(component_losses[key])

            # Compute detailed metrics for first graph only (for monitoring)
            if i == 0:
                first_graph_metrics = compute_first_graph_metrics(model, graph_data, component_losses)

        avg_epoch_loss = float(total_epoch_loss) / len(dataset)
        
        # Calculate average component losses for the epoch
        avg_component_losses = {
            key: value / len(dataset) for key, value in epoch_component_losses.items()
        }
        
        # Log to console
        log_epoch_loss(epoch, avg_epoch_loss, avg_component_losses)
        
        # Log to WandB
        wandb_metrics = {
            'loss': avg_epoch_loss,
            **avg_component_losses
        }
        log_wandb_epoch(epoch, wandb_metrics)
        
        # Log first graph detailed metrics
        if first_graph_metrics:
            from logger import log_wandb_graph
            log_wandb_graph(0, first_graph_metrics, epoch)


optimizer = optim.Adam(learning_rate=1e-5)
    
train_model(model, dataset, optimizer, epochs=500, print_last_epoch_only=True)

# Clean up WandB
finish_wandb()
