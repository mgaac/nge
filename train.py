import mlx.core as mx
import mlx.nn as nn

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

def bf_loss(bf_output, bf_distance_targets, bf_predecessor_targets):

    bf_distance_predictions, bf_predecessor_predictions = bf_output

    bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, bf_distance_targets, reduction='mean')
    bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, bf_predecessor_targets, reduction='mean')

    return bf_distance_loss + bf_predecessor_loss

def bfs_loss(bfs_output, bfs_state_targets):

    bfs_state_predictions = bfs_output
    bfs_state_loss = nn.losses.binary_cross_entropy(bfs_state_predictions, bfs_state_targets, reduction='mean')

    return bfs_state_loss

def step_loss(bf_output, bfs_output, bfs_state_targets, bf_distance_targets, bf_predecessor_targets, termination_probs,termination_targets):

    bf_loss_value = bf_loss(bf_output, bf_distance_targets, bf_predecessor_targets)
    bfs_loss_value = bfs_loss(bfs_output, bfs_state_targets)

    bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
    bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean')


    total_step_loss = bf_loss_value + bfs_loss_value + bf_termination_loss + bfs_termination_loss

    return total_step_loss