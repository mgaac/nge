import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils

import mlx.optimizers as optim

import numpy as np

from model import nge, aggregation_fn
from generate_dataset import load_dataset

MODEL_CONFIG = {
    'embedding_dim': 128,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 2
}

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


def graph_execution_loss_fn(model, graph_data, verbose=False):
    accumulated_loss = mx.array(0.0)
    previous_step_hidden_states = mx.zeros([*graph_data['node_embeddings'].shape])

    num_steps = len(graph_data['steps'])

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

        if verbose:
            print(f"\n--- Step {i} ---")
            
            # Calculate accuracy measures
            # BFS state accuracy (binary)
            bfs_pred_binary = (bfs_output > 0.5).astype(mx.float32)
            bfs_accuracy = mx.mean(bfs_pred_binary == target_bfs_state)
            
            # BF distance accuracy (within tolerance)
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            distance_tolerance = 1.0  # Accept predictions within 1.0 of target
            distance_within_tolerance = mx.abs(bf_distance_predictions - target_distance_bf) <= distance_tolerance
            bf_distance_accuracy = mx.mean(distance_within_tolerance)
            
            # BF predecessor accuracy (categorical)
            bf_pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            bf_predecessor_accuracy = mx.mean(bf_pred_argmax == target_predecessor_bf)
            
            # Termination accuracies (binary)
            bf_term_pred = (termination_probs['bf'] > 0.5).astype(mx.float32)
            bf_term_accuracy = float(bf_term_pred == termination_targets['bf'])
            
            bfs_term_pred = (termination_probs['bfs'] > 0.5).astype(mx.float32)
            bfs_term_accuracy = float(bfs_term_pred == termination_targets['bfs'])

        # Compute each loss component separately for printing or not
        bf_distance_predictions, bf_predecessor_predictions = bf_output
        bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')
        bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, target_predecessor_bf, reduction='mean')
        bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean')
        bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
        bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean')

        total_step_loss = bf_predecessor_loss + bfs_state_loss + bf_termination_loss + bfs_termination_loss

        if verbose:
            # Compact output with losses and accuracies
            print(f"  Losses: bf_dist={float(bf_distance_loss):.4f}, bf_pred={float(bf_predecessor_loss):.4f}, "
                  f"bfs_state={float(bfs_state_loss):.4f}, bf_term={float(bf_termination_loss):.4f}, "
                  f"bfs_term={float(bfs_termination_loss):.4f}, total={float(total_step_loss):.4f}")
            
            print(f"  Accuracies: bfs_state={float(bfs_accuracy):.3f}, bf_dist(±{distance_tolerance})={float(bf_distance_accuracy):.3f}, "
                  f"bf_pred={float(bf_predecessor_accuracy):.3f}, bf_term={bf_term_accuracy:.3f}, bfs_term={bfs_term_accuracy:.3f}")
            
            # Sample predictions vs targets for debugging
            sample_size = min(5, len(target_bfs_state))
            print(f"  Sample predictions (first {sample_size}):")
            print(f"    BFS: pred={np.array(bfs_output[:sample_size]).round(3)}, target={np.array(target_bfs_state[:sample_size])}")
            print(f"    BF dist: pred={np.array(bf_distance_predictions[:sample_size]).round(3)}, target={np.array(target_distance_bf[:sample_size])}")
            print(f"    BF pred: argmax={np.array(bf_pred_argmax[:sample_size])}, target={np.array(target_predecessor_bf[:sample_size])}")

        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss

    average_loss = accumulated_loss / (num_steps - 1)

    return average_loss


loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)

mx.eval(model.parameters())


def train_model(model, dataset, optimizer, epochs):

    for epoch in range(epochs):

        total_epoch_loss = 0

        # Only print verbose output for the first graph in each epoch, but only every 10 epochs
        print_this_epoch = (epoch % 10 == 0)
        for i, graph_data in enumerate(dataset):
            verbose = print_this_epoch and (i == 0)
            
            loss, grads = loss_and_grad_fn(model, graph_data, verbose=verbose)
            cliped_grads, total_norm = optim.clip_grad_norm(grads, max_norm=1.0)

            optimizer.update(model, cliped_grads)

            mx.eval(model.parameters(), optimizer.state)

            total_epoch_loss += loss

        if print_this_epoch:
            print(f"Epoch {epoch} loss: {total_epoch_loss / len(dataset)}")


optimizer = optim.Adam(learning_rate=1e-5)
    
train_model(model, dataset, optimizer, epochs=1000)
