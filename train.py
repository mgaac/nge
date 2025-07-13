import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils

import mlx.optimizers as optim

import numpy as np

import wandb

from model import nge, aggregation_fn
from data.data import load_dataset
from utils import print_execution_details, calculate_losses_and_accuracies

MODEL_CONFIG = {
    'embed_dim': 128,
    'residual_connections': True,
    'agg_fn': aggregation_fn.MAX,
    'num_mp_layers': 2
}

HYPERPARAMETERS = {
    'epochs': 300,
    'start_lr': 5e-5,
    'end_lr': 1e-5,
    'decay_ratio': .8,
    'max_grad_norm': 1.0
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

    return (bf_distance_loss, bf_predecessor_loss, bfs_state_loss, bf_termination_loss, bfs_termination_loss), average_loss


def graph_execution_loss_scalar(model, graph_data):
    """Version of loss function that only returns scalar loss for gradient computation"""
    _, loss = graph_execution_loss_fn(model, graph_data)
    return loss


loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_scalar)

mx.eval(model.parameters())

# Initialize wandb
wandb.init(project="nge-train", config={**MODEL_CONFIG, **HYPERPARAMETERS})

def evaluate_model(model, dataset):

    accumulated_epoch_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])
    accumulated_accuracies = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i, graph_data in enumerate(dataset):
        
        aux_losses, loss, accuracies = calculate_losses_and_accuracies(model, graph_data, MODEL_CONFIG['embed_dim'])

        accumulated_epoch_loss += loss
        accumulated_aux_losses += aux_losses
        for j in range(5):
            accumulated_accuracies[j] += accuracies[j]

    avg_epoch_loss = accumulated_epoch_loss / len(dataset)
    avg_aux_losses = accumulated_aux_losses / len(dataset)
    avg_accuracies = [float(acc / len(dataset)) for acc in accumulated_accuracies]

    return avg_aux_losses, avg_epoch_loss, avg_accuracies

def train_model(model, dataset, optimizer, epochs):

    for epoch in range(epochs):

        accumulated_epoch_loss = mx.array(0.0)
        accumulated_aux_losses = mx.zeros([5])
        accumulated_accuracies = [0.0, 0.0, 0.0, 0.0, 0.0]

        for i, graph_data in enumerate(dataset):
            
            loss, grads = loss_and_grad_fn(model, graph_data)

            clipped_grads, norm = optim.clip_grad_norm(grads, max_norm=HYPERPARAMETERS['max_grad_norm'])

            optimizer.update(model, clipped_grads)

            mx.eval(model.parameters(), optimizer.state)

            accumulated_epoch_loss += loss
            
            # Get auxiliary losses and accuracies for this sample 
            aux_losses, _, train_accuracies = calculate_losses_and_accuracies(model, graph_data, MODEL_CONFIG['embed_dim'])
            accumulated_aux_losses += aux_losses
            accumulated_accuracies = list(mx.array(accumulated_accuracies) + mx.array(train_accuracies))

        avg_epoch_loss = accumulated_epoch_loss / len(dataset)
        avg_aux_losses = accumulated_aux_losses / len(dataset)
        avg_train_accuracies = [float(acc / len(dataset)) for acc in accumulated_accuracies]

        print(f"Epoch {epoch}: loss = {avg_epoch_loss}")

        # Log to wandb with organized structure
        wandb.log({
            # Main metrics (at root level for easy access)
            "loss": float(avg_epoch_loss),
            "lr": float(optimizer.learning_rate),
            "grad_norm": float(norm),
            
            # Loss breakdown
            "losses/bf_distance": float(avg_aux_losses[0]),
            "losses/bf_predecessor": float(avg_aux_losses[1]),
            "losses/bfs_state": float(avg_aux_losses[2]),
            "losses/bf_termination": float(avg_aux_losses[3]),
            "losses/bfs_termination": float(avg_aux_losses[4]),
            
            # Training accuracies
            "train_acc/bf_distance": avg_train_accuracies[0],
            "train_acc/bf_predecessor": avg_train_accuracies[1],
            "train_acc/bfs_state": avg_train_accuracies[2],
            "train_acc/bf_termination": avg_train_accuracies[3],
            "train_acc/bfs_termination": avg_train_accuracies[4]
        })

        if epoch % 5 == 0:
            val_aux_losses, val_loss, val_accuracies = evaluate_model(model, val_dataset)
            wandb.log({
                "val_loss": float(val_loss),
                "val_acc/bf_distance": float(val_accuracies[0]),
                "val_acc/bf_predecessor": float(val_accuracies[1]),
                "val_acc/bfs_state": float(val_accuracies[2]),
                "val_acc/bf_termination": float(val_accuracies[3]),
                "val_acc/bfs_termination": float(val_accuracies[4])
            })

        if (epoch) % 10 == 0:
            random_idx = mx.random.randint(0, len(train_dataset)).item()
            _, norm  = print_execution_details(model, train_dataset[random_idx], MODEL_CONFIG['embed_dim'])
            wandb.log({"debug/hidden_state_norm": float(norm)})


total_steps = HYPERPARAMETERS['epochs'] * len(train_dataset)
decay_steps = int(total_steps * HYPERPARAMETERS['decay_ratio'])
# warmup_steps = int(total_steps * HYPERPARAMETERS['warmup_steps'])

# lr_warmup = optim.linear_schedule(
#     init=0.0,
#     end=HYPERPARAMETERS['start_lr'],
#     steps=warmup_steps
# )

lr_decay = optim.cosine_decay(
    init=HYPERPARAMETERS['start_lr'],
    decay_steps=decay_steps,
    end=HYPERPARAMETERS['end_lr']
)

# lr_scheduler = optim.join_schedules([lr_warmup, lr_decay], [warmup_steps])

optimizer = optim.Adam(learning_rate=lr_decay)

train_model(model, train_dataset, optimizer, epochs=HYPERPARAMETERS['epochs']) 
test_aux_losses, test_loss, test_accuracies = evaluate_model(model, test_dataset)
wandb.log({
    "test_loss": float(test_loss),
    "test_acc/bf_distance": float(test_accuracies[0]),
    "test_acc/bf_predecessor": float(test_accuracies[1]), 
    "test_acc/bfs_state": float(test_accuracies[2]),
    "test_acc/bf_termination": float(test_accuracies[3]),
    "test_acc/bfs_termination": float(test_accuracies[4])
})
