import mlx.core as mx
import mlx.nn as nn

import numpy as np

from model import nge, aggregation_fn
from generate_dataset import load_dataset

MODEL_CONFIG = {
    'embedding_dim': 128,
    'skip_connections': True,
    'aggregation_fn': aggregation_fn.MAX,
    'num_mp_layers': 3
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


def step_loss_fn(bfs_output, bf_output, termination_probs, step_data):

    bfs_state_targets = step_data['bfs_state_targets']
    bf_distance_targets = step_data['bf_distance_targets']
    bf_predecessor_targets = step_data['bf_predecessor_targets']
    termination_targets = step_data['termination_targets']

    bf_loss_value = bf_loss_fn(bf_output, bf_distance_targets, bf_predecessor_targets)
    bfs_loss_value = bfs_loss_fn(bfs_output, bfs_state_targets)

    bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
    bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean')

    total_step_loss = bf_loss_value + bfs_loss_value + bf_termination_loss + bfs_termination_loss

    return total_step_loss


def graph_execution_loss_fn(model, graph_data):

    accumulated_loss = 0
    previous_step_hidden_states = mx.zeros([*graph_data['node_embeddings'].shape])

    for step_data in graph_data['steps']: 

        true_bfs_state = step_data['bfs_state_targets']
        true_distance_bf = step_data['bf_distance_targets']
        true_predecessor_bf = step_data['bf_predecessor_targets']

        input_embeddings = mx.concatenate([graph_data['node_embeddings'] + previous_step_hidden_states, true_bfs_state, true_distance_bf, true_predecessor_bf], axis=1)

        model_input = (input_embeddings, graph_data['connection_matrix'])
        
        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)

        loss = step_loss_fn(bfs_output, bf_output, termination_probs, step_data)

        previous_step_hidden_states = processed_embeddings
        accumulated_loss += loss

    average_loss = accumulated_loss / len(graph_data['steps'])

    return average_loss

loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)

def train_dataset(model, dataset, optimizer):

    for graph_data in dataset:

        loss, grads = loss_and_grad_fn(model, graph_data)

        optimizer.update(model, grads)


    




# print(dataset[0]['node_embeddings'].shape)
# print(dataset[0]['connection_matrix'].shape)
# print(len(dataset[0]['steps']))


#dict_keys(['bfs_state_targets', 'bf_distance_targets', 'bf_predecessor_targets', 'termination_targets'])

# z=20

# for x in range(len(dataset[z]['steps'])):
#     print("bfs_state_targets:")
#     print(np.array(dataset[z]['steps'][x]['bfs_state_targets']))
#     print("bf_distance_targets:")
#     print(np.array(dataset[z]['steps'][x]['bf_distance_targets']))
#     print("bf_predecessor_targets:")
#     print(np.array(dataset[z]['steps'][x]['bf_predecessor_targets']))

#     print("termination_targets:")
#     #bfs, bf
#     print(np.array(dataset[z]['steps'][x]['termination_targets']['bfs']))
#     print(np.array(dataset[z]['steps'][x]['termination_targets']['bf']))
#     print("--------------------------------")


# print(len(dataset))