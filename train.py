import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils

import mlx.optimizers as optim

import numpy as np

import wandb

from model import nge, aggregation_fn
from data.data import load_dataset
from utils import print_execution_details

MODEL_CONFIG = {
    'embed_dim': 128,
    'residual_connections': True,
    'agg_fn': aggregation_fn.MAX,
    'num_mp_layers': 2
}

HYPERPARAMETERS = {
    'epochs': 500,
    'lr': 1e-5,
}

model = nge(**MODEL_CONFIG)

train_dataset = load_dataset('data/train_dataset.npz')
val_dataset = load_dataset('data/val_dataset.npz')
test_dataset = load_dataset('data/test_dataset.npz')

@mx.compile
def bf_loss_fn(bf_output, bf_distance_targets, bf_predecessor_targets):

    bf_distance_predictions, bf_predecessor_predictions = bf_output

    bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, bf_distance_targets, reduction='mean')
    bf_distance_loss = bf_distance_loss * 0.1  # Scale down distance loss

    bf_predecessor_loss = nn.losses.cross_entropy(bf_predecessor_predictions, bf_predecessor_targets, reduction='mean')

    return bf_distance_loss

@mx.compile
def bfs_loss_fn(bfs_output, bfs_state_targets):

    bfs_state_predictions = bfs_output
    bfs_state_loss = nn.losses.binary_cross_entropy(bfs_state_predictions, bfs_state_targets, reduction='mean', with_logits=True)

    return bfs_state_loss

@mx.compile
def step_loss_fn(bfs_output, bf_output, termination_probs, target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets):

    bf_loss_value = bf_loss_fn(bf_output, target_distance_bf, target_predecessor_bf)
    bfs_loss_value = bfs_loss_fn(bfs_output, target_bfs_state)

    bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean', with_logits=False)
    bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean', with_logits=False)

    total_step_loss = bf_loss_value + bfs_loss_value + bf_termination_loss + bfs_termination_loss

    return total_step_loss


def graph_execution_loss_fn(model, graph_data):
    accumulated_loss = mx.array(0.0)

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, MODEL_CONFIG['embed_dim']])

    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])

    num_steps = max(num_bf_steps, num_bfs_steps)
    
    
    for i in range(num_steps):
        # Check if samples exist
        bf_sample_exists = i < num_bf_steps and (i + 1) < num_bf_steps
        bfs_sample_exists = i < num_bfs_steps and (i + 1) < num_bfs_steps

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

        # Forward pass
        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)

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

        # Update for next step
        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss

    average_loss = accumulated_loss / (num_steps - 1)

    return average_loss


loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)

mx.eval(model.parameters())

# Initialize wandb
wandb.init(project="nge-train", config={**MODEL_CONFIG, **HYPERPARAMETERS})

def evaluate_model(model, dataset):

    accumulated_epoch_loss = mx.array(0.0)

    for i, graph_data in enumerate(dataset):
        
        loss, _ = loss_and_grad_fn(model, graph_data)

        accumulated_epoch_loss += loss

    avg_epoch_loss = accumulated_epoch_loss / len(dataset)

    return avg_epoch_loss

def train_model(model, dataset, optimizer, epochs):

    for epoch in range(epochs):

        accumulated_epoch_loss = mx.array(0.0)

        for i, graph_data in enumerate(dataset):
            
            loss, grads = loss_and_grad_fn(model, graph_data)

            optimizer.update(model, grads)

            mx.eval(model.parameters(), optimizer.state)

            accumulated_epoch_loss += loss

        avg_epoch_loss = accumulated_epoch_loss / len(dataset)

        print(f"Epoch {epoch}: loss = {avg_epoch_loss}")

        # Log to wandb
        wandb.log({"loss": float(avg_epoch_loss), "lr": float(optimizer.learning_rate)})

        if epoch % 10 == 0:
            val_loss = evaluate_model(model, val_dataset)
            wandb.log({"val_loss": float(val_loss)})

        if (epoch) % 50 == 0:
            random_idx = mx.random.randint(0, len(train_dataset)).item()
            _, norm  = print_execution_details(model, train_dataset[random_idx], MODEL_CONFIG['embed_dim'])
            wandb.log({"norm": float(norm)})


total_steps = HYPERPARAMETERS['epochs'] * len(train_dataset)
# decay_steps = int(total_steps * HYPERPARAMETERS['decay_steps'])
# warmup_steps = int(total_steps * HYPERPARAMETERS['warmup_steps'])

# lr_warmup = optim.linear_schedule(
#     init=0.0,
#     end=HYPERPARAMETERS['start_lr'],
#     steps=warmup_steps
# )

lr_decay = optim.cosine_decay(
    init=HYPERPARAMETERS['lr'],
    decay_steps=total_steps,
    end=0.0
)

# lr_scheduler = optim.join_schedules([lr_warmup, lr_decay], [warmup_steps])

optimizer = optim.Adam(learning_rate=lr_decay, weight_decay=1e-5)

train_model(model, train_dataset, optimizer, epochs=HYPERPARAMETERS['epochs']) 
test_loss = evaluate_model(model, test_dataset)
wandb.log({"test_loss": float(test_loss)})
