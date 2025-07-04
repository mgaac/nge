import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils

import mlx.optimizers as optim

import numpy as np

import wandb

from model import nge, aggregation_fn
from generate_dataset import load_dataset

MODEL_CONFIG = {
    'embedding_dim': 128,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 2
}

# Initialize wandb
wandb.init(
    project="graph-algorithm-latent-execution",
    config=MODEL_CONFIG,
    name=None,
    notes="Training NGE model with BFS and Bellman-Ford supervision"
)

model = nge(**MODEL_CONFIG)

dataset = load_dataset('graph_dataset.npz')  # or use your custom filename

def bf_loss_fn(bf_output, bf_distance_targets, bf_predecessor_targets):

    bf_distance_predictions, bf_predecessor_predictions = bf_output

    bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, bf_distance_targets, reduction='mean')
    bf_distance_loss = bf_distance_loss * 0.1  # Scale down distance loss

    bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, bf_predecessor_targets, reduction='mean')

    return bf_distance_loss


def bfs_loss_fn(bfs_output, bfs_state_targets):

    bfs_state_predictions = bfs_output
    bfs_state_loss = nn.losses.binary_cross_entropy(bfs_state_predictions, bfs_state_targets, reduction='mean')

    return bfs_state_loss


def step_loss_fn(bfs_output, bf_output, termination_probs, target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets):

    bf_loss_value = bf_loss_fn(bf_output, target_distance_bf, target_predecessor_bf)
    bfs_loss_value = bfs_loss_fn(bfs_output, target_bfs_state)

    bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
    bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean')

    total_step_loss = bf_loss_value + bfs_loss_value + bf_termination_loss + bfs_termination_loss

    return total_step_loss


def calculate_step_metrics(bf_output, bfs_output, termination_probs, target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets):
    """Calculate accuracy metrics for a single step."""
    bf_distance_predictions, bf_predecessor_predictions = bf_output
    
    # Calculate accuracies
    bfs_pred_binary = (bfs_output > 0.5).astype(mx.float32)
    bfs_accuracy = mx.mean(bfs_pred_binary == target_bfs_state)
    
    distance_tolerance = 1.0  # Accept predictions within 1.0 of target
    distance_within_tolerance = mx.abs(bf_distance_predictions - target_distance_bf) <= distance_tolerance
    bf_distance_accuracy = mx.mean(distance_within_tolerance)
    
    bf_pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
    bf_predecessor_accuracy = mx.mean(bf_pred_argmax == target_predecessor_bf)
    
    bf_term_pred = (termination_probs['bf'] > 0.5).astype(mx.float32)
    bf_term_accuracy = float(bf_term_pred == termination_targets['bf'])
    
    bfs_term_pred = (termination_probs['bfs'] > 0.5).astype(mx.float32)
    bfs_term_accuracy = float(bfs_term_pred == termination_targets['bfs'])
    
    return {
        "bfs_accuracy": float(bfs_accuracy),
        "bf_distance_accuracy": float(bf_distance_accuracy),
        "bf_predecessor_accuracy": float(bf_predecessor_accuracy),
        "bf_term_accuracy": bf_term_accuracy,
        "bfs_term_accuracy": bfs_term_accuracy,
    }


def print_step_details(step_idx, bf_output, bfs_output, termination_probs, target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets, losses):
    """Print detailed information for a single step."""
    bf_distance_predictions, bf_predecessor_predictions = bf_output
    
    print(f"\n--- Step {step_idx} ---")
    print(f"  Losses: bf_dist={float(losses['bf_distance_loss']):.4f}, bf_pred={float(losses['bf_predecessor_loss']):.4f}, "
          f"bfs_state={float(losses['bfs_state_loss']):.4f}, bf_term={float(losses['bf_termination_loss']):.4f}, "
          f"bfs_term={float(losses['bfs_termination_loss']):.4f}, total={float(losses['total_step_loss']):.4f}")
    
    # Calculate accuracies for printing
    metrics = calculate_step_metrics(bf_output, bfs_output, termination_probs, target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets)
    distance_tolerance = 1.0
    print(f"  Accuracies: bfs_state={metrics['bfs_accuracy']:.3f}, bf_dist(±{distance_tolerance})={metrics['bf_distance_accuracy']:.3f}, "
          f"bf_pred={metrics['bf_predecessor_accuracy']:.3f}, bf_term={metrics['bf_term_accuracy']:.3f}, bfs_term={metrics['bfs_term_accuracy']:.3f}")
    
    # Show full vectors for visible outputs
    print(f"  Full vectors:")
    print(f"    BFS state: pred={np.array(bfs_output).round(3)}, target={np.array(target_bfs_state)}")
    print(f"    BF distance: pred={np.array(bf_distance_predictions).round(3)}, target={np.array(target_distance_bf)}")
    
    bf_pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
    print(f"    BF predecessor: argmax={np.array(bf_pred_argmax)}, target={np.array(target_predecessor_bf)}")
    print(f"    Termination: bf={np.array(termination_probs['bf']).round(3)}, bfs={np.array(termination_probs['bfs']).round(3)}")
    
    # Show magnitudes for hidden states (predecessor predictions before argmax)
    bf_pred_magnitudes = np.linalg.norm(np.array(bf_predecessor_predictions), axis=-1)
    print(f"    BF predecessor magnitudes: {bf_pred_magnitudes.round(3)}")


def graph_execution_loss_fn(model, graph_data, verbose=False, wandb_log_dict=None):
    accumulated_loss = mx.array(0.0)
    previous_step_hidden_states = mx.zeros([*graph_data['node_embeddings'].shape])

    num_steps = len(graph_data['steps'])

    # For wandb logging: accumulate per-step metrics
    step_metrics = {
        "bf_distance_loss": [],
        "bf_predecessor_loss": [],
        "bfs_state_loss": [],
        "bf_termination_loss": [],
        "bfs_termination_loss": [],
        "total_step_loss": [],
        "bfs_accuracy": [],
        "bf_distance_accuracy": [],
        "bf_predecessor_accuracy": [],
        "bf_term_accuracy": [],
        "bfs_term_accuracy": [],
    }

    if num_steps <= 1:
        return accumulated_loss

    for i, step_data in enumerate(graph_data['steps'][:-1]):
        true_bfs_state = step_data['bfs_state_targets']
        true_distance_bf = step_data['bf_distance_targets']
        true_predecessor_bf = step_data['bf_predecessor_targets']

        target_bfs_state = graph_data['steps'][i+1]['bfs_state_targets']
        target_distance_bf = graph_data['steps'][i+1]['bf_distance_targets']
        target_predecessor_bf = graph_data['steps'][i+1]['bf_predecessor_targets']
        termination_targets = graph_data['steps'][i+1]['termination_targets']

        node_algo_features = mx.concatenate([true_bfs_state, true_distance_bf, true_predecessor_bf]).reshape([-1, 3])
        input_embeddings = mx.concatenate([graph_data['node_embeddings'] + previous_step_hidden_states, node_algo_features], axis=1)

        model_input = (input_embeddings, graph_data['connection_matrix'])

        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)

        # Compute each loss component separately
        bf_distance_predictions, bf_predecessor_predictions = bf_output
        bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')
        bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, target_predecessor_bf, reduction='mean')
        bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean')
        bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
        bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean')

        total_step_loss = bf_distance_loss + bf_predecessor_loss + bfs_state_loss + bf_termination_loss + bfs_termination_loss

        # Calculate accuracy measures for wandb logging
        accuracy_metrics = calculate_step_metrics(bf_output, bfs_output, termination_probs, target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets)

        # Accumulate for wandb logging
        if wandb_log_dict is not None:
            step_metrics["bf_distance_loss"].append(float(bf_distance_loss))
            step_metrics["bf_predecessor_loss"].append(float(bf_predecessor_loss))
            step_metrics["bfs_state_loss"].append(float(bfs_state_loss))
            step_metrics["bf_termination_loss"].append(float(bf_termination_loss))
            step_metrics["bfs_termination_loss"].append(float(bfs_termination_loss))
            step_metrics["total_step_loss"].append(float(total_step_loss))
            step_metrics["bfs_accuracy"].append(accuracy_metrics["bfs_accuracy"])
            step_metrics["bf_distance_accuracy"].append(accuracy_metrics["bf_distance_accuracy"])
            step_metrics["bf_predecessor_accuracy"].append(accuracy_metrics["bf_predecessor_accuracy"])
            step_metrics["bf_term_accuracy"].append(accuracy_metrics["bf_term_accuracy"])
            step_metrics["bfs_term_accuracy"].append(accuracy_metrics["bfs_term_accuracy"])

        if verbose:
            losses = {
                'bf_distance_loss': bf_distance_loss,
                'bf_predecessor_loss': bf_predecessor_loss,
                'bfs_state_loss': bfs_state_loss,
                'bf_termination_loss': bf_termination_loss,
                'bfs_termination_loss': bfs_termination_loss,
                'total_step_loss': total_step_loss
            }
            print_step_details(i, bf_output, bfs_output, termination_probs, target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets, losses)

        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss

    average_loss = accumulated_loss / (num_steps - 1)

    # Log average metrics to wandb if requested
    if wandb_log_dict is not None and (num_steps > 1):
        # Compute mean of each metric over steps
        for k in step_metrics:
            if len(step_metrics[k]) > 0:
                wandb_log_dict[k] = float(np.mean(step_metrics[k]))

    return average_loss


loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)

mx.eval(model.parameters())


def train_model(model, dataset, optimizer, epochs):

    for epoch in range(epochs):

        total_epoch_loss = 0
        wandb_epoch_metrics = {}

        # Only print verbose output for the first graph in each epoch, but only every 100 epochs
        print_this_epoch = (epoch % 100 == 0)
        for i, graph_data in enumerate(dataset):
            verbose = print_this_epoch and (i == 0)
            wandb_log_dict = {} if i == 0 else None  # Only log detailed metrics for first graph

            # Pass wandb_log_dict to loss fn for logging
            loss, grads = loss_and_grad_fn(model, graph_data, verbose=verbose, wandb_log_dict=wandb_log_dict)
            cliped_grads, total_norm = optim.clip_grad_norm(grads, max_norm=2.0)

            optimizer.update(model, cliped_grads)

            mx.eval(model.parameters(), optimizer.state)

            total_epoch_loss += loss

            # Log per-graph metrics for the first graph in the epoch
            if i == 0 and wandb_log_dict is not None:
                for k, v in wandb_log_dict.items():
                    wandb_epoch_metrics[f"first_graph/{k}"] = v

        avg_epoch_loss = float(total_epoch_loss) / len(dataset)
        wandb_epoch_metrics["epoch"] = epoch
        wandb_epoch_metrics["loss"] = avg_epoch_loss

        wandb.log(wandb_epoch_metrics, step=epoch)

        if print_this_epoch:
            print(f"Epoch {epoch} loss: {avg_epoch_loss}")


optimizer = optim.Adam(learning_rate=1e-5)
    
train_model(model, dataset, optimizer, epochs=10000)
