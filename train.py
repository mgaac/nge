import mlx.core as mx
import mlx.nn as nn

import mlx.utils as utils
import mlx.optimizers as optim

import wandb    
import argparse

from model import nge, aggregation_fn
from data.data import load_dataset
from utils import print_execution_details, calculate_losses_and_accuracies, extract_per_head_magnitude_grads

mx.random.seed(42)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train NGE model')
parser.add_argument('--no-wandb', action='store_true', help='Disable wandb tracking')
args = parser.parse_args()

USE_WANDB = not args.no_wandb

MODEL_CONFIG = {
    'embed_dim': 32,
    'residual_connections': True,
    'agg_fn': aggregation_fn.MAX,
    'num_mp_layers': 2,
    'dropout': 0.1,
    'num_predecessor_layers': 5,
    'num_update_layers': 1,
}

HYPERPARAMETERS = {
    'epochs': 500,
    'start_lr':1e-5,
    'end_lr': 1e-5,
    'decay_ratio': .005,
    'max_grad_norm': 2.0,
    'batch_size': 5,
}

model = nge(**MODEL_CONFIG)

# Set model to training mode initially
model.train()

train_dataset = load_dataset('data/train_dataset.npz')
val_dataset = load_dataset('data/val_dataset.npz')
test_dataset = load_dataset('data/test_dataset.npz')


def graph_execution_loss_fn(model, graph_data):
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, MODEL_CONFIG['embed_dim'] * 2])

    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])

    num_steps = max(num_bf_steps, num_bfs_steps)
    
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

        # Forward pass
        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)

        # Compute losses
        if bf_sample_exists:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')

            # Convert invalid, denoted by -1, to a valid class, 0.
            valid_mask = (target_predecessor_bf != -1)                          # [num_nodes] bool
            safe_targets = mx.where(valid_mask, target_predecessor_bf,
                                    mx.zeros_like(target_predecessor_bf))  
            
            per_node_ce = nn.losses.cross_entropy(bf_predecessor_predictions, safe_targets, reduction='none')
            
            # Only consider loss over valid nodes
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom

            bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean', with_logits=True)
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

        raw_losses = mx.array([bf_distance_loss, bf_predecessor_loss, bfs_state_loss, bf_termination_loss, bfs_termination_loss])
        total_step_loss = mx.sum(raw_losses)

        # Update for next step
        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss
        accumulated_aux_losses += raw_losses

        bf_steps  = max(num_bf_steps  - 1, 0)
        bfs_steps = max(num_bfs_steps - 1, 0)
        effective_steps = max(bf_steps, bfs_steps, 1)

        average_loss = accumulated_loss / effective_steps
        per_task_counter = mx.array([
            max(bf_steps, 1),   # bf_distance
            max(bf_steps, 1),   # bf_predecessor
            max(bfs_steps, 1),  # bfs_state
            max(bf_steps, 1),   # bf_termination
            max(bfs_steps, 1),  # bfs_termination
        ], dtype=mx.float32)
        avg_aux_losses = accumulated_aux_losses / per_task_counter

    return average_loss, avg_aux_losses

# Initialize wandb
if USE_WANDB:
    wandb.init(project="nge", config={**MODEL_CONFIG, **HYPERPARAMETERS})
    print("Wandb tracking enabled")
else:
    print("Wandb tracking disabled")

def evaluate_model(model, dataset):
    # Set model to evaluation mode (disables dropout)
    model.eval()
    
    accumulated_epoch_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])
    accumulated_accuracies = mx.zeros([5])

    for graph_data in dataset:
        
        aux_losses, loss, accuracies = calculate_losses_and_accuracies(model, graph_data, MODEL_CONFIG['embed_dim'])

        accumulated_epoch_loss += loss
        accumulated_aux_losses += aux_losses
        accumulated_accuracies += accuracies

    avg_epoch_loss = accumulated_epoch_loss / len(dataset)
    avg_aux_losses = accumulated_aux_losses / len(dataset)
    avg_accuracies = accumulated_accuracies / len(dataset)
    
    # Set model back to training mode
    model.train()

    return avg_aux_losses, avg_epoch_loss, avg_accuracies

def train_model(model, dataset, optimizer, epochs, batch_size=1):
    # Set model to training mode (enables dropout)
    model.train()
    
    for epoch in range(epochs):
        
        loss_and_grad_fn = nn.value_and_grad(model, graph_execution_loss_fn)

        accumulated_epoch_loss = mx.array(0.0)
        accumulated_aux_losses = mx.zeros([5])
        accumulated_per_head_grads = {}

        permutation = mx.random.permutation(len(dataset))
        # We will index into the original Python list using this permutation
        # (do not convert the dataset itself to an mx.array)

        acc_batch_grads = None
        bucket_count = 0

        for idx_in_epoch, idx in enumerate(permutation):
            graph_data = dataset[int(idx.item())]

            (loss, aux_losses), grads = loss_and_grad_fn(model, graph_data)

            per_head_magnitude_grads = extract_per_head_magnitude_grads(grads)
            
            # Accumulate per-head gradients
            for head_name, grad_value in per_head_magnitude_grads.items():
                if head_name not in accumulated_per_head_grads:
                    accumulated_per_head_grads[head_name] = grad_value
                else:
                    accumulated_per_head_grads[head_name] += grad_value

            # --- Gradient accumulation ---
            if acc_batch_grads is None:
                # First gradient defines the tree structure
                acc_batch_grads = grads
            else:
                # Element-wise add across the gradient pytree
                acc_batch_grads = utils.tree_map(lambda a, b: a + b, acc_batch_grads, grads)
            bucket_count += 1

            end_of_bucket = (bucket_count == batch_size)
            end_of_epoch  = (idx_in_epoch + 1 == len(permutation))
            if end_of_bucket or end_of_epoch:
                # Average by actual bucket size (last bucket may be smaller)
                avg_grads = utils.tree_map(lambda x: x / bucket_count, acc_batch_grads)

                # Now actually clip the gradients
                avg_grads, norm = optim.clip_grad_norm(
                    avg_grads, max_norm=HYPERPARAMETERS['max_grad_norm']
                )

                optimizer.update(model, avg_grads)
                mx.eval(model.parameters(), optimizer.state)

                # Reset for next bucket
                acc_batch_grads = None
                bucket_count = 0

            # --- Book-keeping for epoch metrics ---
            accumulated_epoch_loss += loss
            accumulated_aux_losses += aux_losses

        avg_epoch_loss = accumulated_epoch_loss / len(dataset)
        avg_aux_losses = accumulated_aux_losses / len(dataset)
        
        # Compute average per-head gradients
        avg_per_head_grads = {
            head_name: grad_value / len(dataset)
            for head_name, grad_value in accumulated_per_head_grads.items()
        }
        
        print(f"Epoch {epoch}: loss = {avg_epoch_loss}")
        
        # Log to wandb with organized structure
        log_dict = {
            # Main metrics (at root level for easy access)
            "loss": float(avg_epoch_loss),
            "lr": float(optimizer.learning_rate),
            
            # Loss breakdown
            "losses/bf_distance": float(avg_aux_losses[0]),
            "losses/bf_predecessor": float(avg_aux_losses[1]),
            "losses/bfs_state": float(avg_aux_losses[2]),
            "losses/bf_termination": float(avg_aux_losses[3]) ,
            "losses/bfs_termination": float(avg_aux_losses[4]),
        }
        
        # Add per-head average gradients to the log
        for head_name, grad_value in avg_per_head_grads.items():
            log_dict[f"grad_avg/{head_name}"] = float(grad_value)
        
        if USE_WANDB:
            wandb.log(log_dict)

        if (epoch + 1) % 10 == 0:
            val_aux_losses, val_loss, val_accuracies = evaluate_model(model, val_dataset)

            # Evaluate on a random subsample of training dataset (same size as validation)
            train_subsample_size = len(val_dataset)
            train_subsample_indices = mx.random.permutation(len(train_dataset))[:train_subsample_size]
            train_subsample = [train_dataset[int(idx.item())] for idx in train_subsample_indices]
            _, _, train_accuracies = evaluate_model(model, train_subsample)

            if USE_WANDB:
                wandb.log({
                    "val_loss": float(val_loss),
                    "val_acc/bf_distance": float(val_accuracies[0]),
                    "val_acc/bf_predecessor": float(val_accuracies[1]),
                    "val_acc/bfs_state": float(val_accuracies[2]),
                    "val_acc/bf_termination": float(val_accuracies[3]),
                    "val_acc/bfs_termination": float(val_accuracies[4]),

                    "val_losses/bf_distance": float(val_aux_losses[0]),
                    "val_losses/bf_predecessor": float(val_aux_losses[1]),
                    "val_losses/bfs_state": float(val_aux_losses[2]),
                    "val_losses/bf_termination": float(val_aux_losses[3]),
                    "val_losses/bfs_termination": float(val_aux_losses[4]),

                    "train_acc/bf_distance": float(train_accuracies[0]),
                    "train_acc/bf_predecessor": float(train_accuracies[1]),
                    "train_acc/bfs_state": float(train_accuracies[2]),
                    "train_acc/bf_termination": float(train_accuracies[3]),
                    "train_acc/bfs_termination": float(train_accuracies[4]),

                })

            random_idx = mx.random.randint(0, len(train_dataset)).item()
            _, per_head_norms = print_execution_details(model, train_dataset[random_idx], MODEL_CONFIG['embed_dim'])
            
            # Log all the per-head norms to wandb
            if USE_WANDB:
                norm_log_dict = {
                    "debug/hidden_state_norm": float(per_head_norms['hidden_state']),
                    "debug/bf_distance_norm": float(per_head_norms['bf_distance']),
                    "debug/bf_predecessor_norm": float(per_head_norms['bf_predecessor']),
                    "debug/bfs_state_norm": float(per_head_norms['bfs_state']),
                    "debug/bf_termination_norm": float(per_head_norms['bf_termination']),
                    "debug/bfs_termination_norm": float(per_head_norms['bfs_termination'])
                }
                wandb.log(norm_log_dict)

total_steps = HYPERPARAMETERS['epochs'] * (len(train_dataset) / HYPERPARAMETERS['batch_size'])
decay_steps = int(total_steps * HYPERPARAMETERS['decay_ratio'])

lr_decay = optim.cosine_decay(
    init=HYPERPARAMETERS['start_lr'],
    decay_steps=decay_steps,
    end=HYPERPARAMETERS['end_lr']
)

optimizer = optim.Adam(learning_rate=lr_decay)

train_model(model, train_dataset, optimizer, epochs=HYPERPARAMETERS['epochs'], batch_size=HYPERPARAMETERS['batch_size']) 
test_aux_losses, test_loss, test_accuracies = evaluate_model(model, test_dataset)
if USE_WANDB:
    wandb.log({
        "test_loss": float(test_loss),
        "test_acc/bf_distance": float(test_accuracies[0]),
        "test_acc/bf_predecessor": float(test_accuracies[1]), 
        "test_acc/bfs_state": float(test_accuracies[2]),
        "test_acc/bf_termination": float(test_accuracies[3]),
        "test_acc/bfs_termination": float(test_accuracies[4])
    })
