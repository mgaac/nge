import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers

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
    bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, bf_predecessor_targets, reduction='mean')

    return bf_distance_loss + bf_predecessor_loss


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


def graph_execution_loss_fn(model, graph_data):
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

        # Print model input nicely formatted
        print(f"\n--- Step {i} ---")
        print("Model input:")
        print("  input_embeddings shape:", input_embeddings.shape)
        print("  input_embeddings (first 2 rows):\n", np.array(input_embeddings)[:2])
        print("  connection_matrix shape:", graph_data['connection_matrix'].shape)
        print("  connection_matrix (first 2 columns):\n", np.array(graph_data['connection_matrix'])[:, :2])

        # Print targets
        print("Targets:")
        print("  target_bfs_state shape:", np.array(target_bfs_state).shape)
        print("  target_bfs_state (first 5):", np.array(target_bfs_state)[:5])
        print("  target_distance_bf shape:", np.array(target_distance_bf).shape)
        print("  target_distance_bf (first 5):", np.array(target_distance_bf)[:5])
        print("  target_predecessor_bf shape:", np.array(target_predecessor_bf).shape)
        # Handle both 1D and 2D target_predecessor_bf for printing
        np_target_predecessor_bf = np.array(target_predecessor_bf)
        if np_target_predecessor_bf.ndim == 2:
            print("  target_predecessor_bf (first 2 rows):\n", np_target_predecessor_bf[:2, :5])
        else:
            print("  target_predecessor_bf (first 5):", np_target_predecessor_bf[:5])
        print("  termination_targets:")
        for k, v in termination_targets.items():
            print(f"    {k}: shape {np.array(v).shape}, value {np.array(v)}")

        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)

        # Print model outputs nicely formatted
        print("Model outputs:")
        print("  bfs_output shape:", np.array(bfs_output).shape)
        print("  bfs_output (first 5):", np.array(bfs_output)[:5])
        if isinstance(bf_output, tuple) and len(bf_output) == 2:
            bf_distance_pred, bf_pred_pred = bf_output
            print("  bf_output[0] (distance) shape:", np.array(bf_distance_pred).shape)
            print("  bf_output[0] (distance, first 5):", np.array(bf_distance_pred))
            print("  bf_output[1] (predecessor) shape:", np.array(bf_pred_pred).shape)
            np_bf_pred_pred = np.array(bf_pred_pred)
            if np_bf_pred_pred.ndim == 2:
                print("  bf_output[1] (predecessor, first 2 rows):\n", np_bf_pred_pred[:2, :5])
            else:
                print("  bf_output[1] (predecessor, first 5):", np_bf_pred_pred[:5])
        else:
            print("  bf_output:", bf_output)
        print("  termination_probs:")
        for k, v in termination_probs.items():
            print(f"    {k}: shape {np.array(v).shape}, value {np.array(v)}")
        print("  processed_embeddings shape:", np.array(processed_embeddings).shape)
        print("  processed_embeddings (first 2 rows):\n", np.array(processed_embeddings)[:2])

        # Compute each loss component separately for printing
        bf_distance_predictions, bf_predecessor_predictions = bf_output
        bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')
        bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, target_predecessor_bf, reduction='mean')
        bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean')
        bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
        bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean')

        total_step_loss = bf_distance_loss + bf_predecessor_loss + bfs_state_loss + bf_termination_loss + bfs_termination_loss

        # Print all loss components compactly
        print(
            f"  Step {i} loss: "
            f"bf_dist={float(bf_distance_loss):.6f}, "
            f"bf_pred={float(bf_predecessor_loss):.6f}, "
            f"bfs_state={float(bfs_state_loss):.6f}, "
            f"bf_term={float(bf_termination_loss):.6f}, "
            f"bfs_term={float(bfs_termination_loss):.6f}, "
            f"total={float(total_step_loss):.6f}"
        )

        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss

    average_loss = accumulated_loss / (num_steps - 1)

    return average_loss


loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)
mx.eval(model.parameters())

def train_model(model, dataset, optimizer, epochs):

    for epoch in range(epochs):

        total_epoch_loss = 0

        for graph_data in dataset:

            loss, grads = loss_and_grad_fn(model, graph_data)

            optimizer.update(model, grads)

            mx.eval(model.parameters(), optimizer.state)

            total_epoch_loss += loss

        print(f"Epoch {epoch} loss: {total_epoch_loss / len(dataset)}")


optimizer = optimizers.Adam(learning_rate=1e-5)
    
train_model(model, dataset, optimizer, epochs=20)

    