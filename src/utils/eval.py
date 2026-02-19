"""Utility functions for model evaluation and debugging.

This module provides functions for calculating losses and accuracies,
as well as debugging tools for inspecting model execution.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils

from src.utils.termination import (
    compute_distance_termination_logits,
    get_distance_latent,
    needs_aux_latents,
    resolve_termination_settings,
)


def _normalize_selected_tasks(selected_tasks=None):
    if selected_tasks is None:
        return {"bf": True, "bfs": True}
    return {"bf": bool(selected_tasks.get("bf", False)), "bfs": bool(selected_tasks.get("bfs", False))}


def _effective_step_count(bf_steps: int, bfs_steps: int, selected_tasks: dict[str, bool]) -> int:
    if selected_tasks["bf"] and selected_tasks["bfs"]:
        return max(bf_steps, bfs_steps, 1)
    if selected_tasks["bf"]:
        return max(bf_steps, 1)
    if selected_tasks["bfs"]:
        return max(bfs_steps, 1)
    return 1


def extract_per_head_magnitude_grads(grads):
    """Extract the L2 norm of gradients for each model component.
    
    Args:
        grads: Gradient tree from value_and_grad
        
    Returns:
        Dictionary mapping component names to gradient magnitudes
    """
    head_names = set()
    utils.tree_map_with_path(lambda path, _: head_names.add(path.split('.')[0]), grads)

    per_head_magnitude_grads = {}
    for head_name in head_names:
        per_head_magnitude_grads[head_name] = utils.tree_reduce(
            lambda acc, x: acc + mx.sum(mx.square(x)), grads[head_name], 0.0
        ) ** 0.5
    return per_head_magnitude_grads

def print_execution_details(model, graph_data, embedding_dim, termination_cfg=None):
    """
    New comprehensive execution details printer with proper step tracking and averaging.
    Follows the exact logic from the loss computation in train.py.
    """
    # Initialize tracking variables
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])  # [bf_dist, bf_pred, bfs_state, bf_term, bfs_term]
    
    # Accuracy tracking
    bf_distance_correct_sum = 0
    bf_distance_total_sum = 0
    bf_predecessor_correct_sum = 0  
    bf_predecessor_total_sum = 0
    bfs_state_correct_sum = 0
    bfs_state_total_sum = 0
    bf_termination_correct_sum = 0
    bf_termination_total_sum = 0
    bfs_termination_correct_sum = 0
    bfs_termination_total_sum = 0
    
    # Per-head norm tracking
    accumulated_bf_distance_norms = mx.array(0.0)
    accumulated_bf_predecessor_norms = mx.array(0.0)
    accumulated_bfs_state_norms = mx.array(0.0)
    accumulated_bf_termination_norms = mx.array(0.0)
    accumulated_bfs_termination_norms = mx.array(0.0)
    accumulated_hidden_state_norms = mx.array(0.0)
    
    # Counters for averaging
    bf_distance_step_count = 0
    bf_predecessor_step_count = 0
    bfs_state_step_count = 0
    bf_termination_step_count = 0
    bfs_termination_step_count = 0
    total_step_count = 0
    
    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, 2 * embedding_dim])
    
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps)
    
    print(f"\n{'='*80}")
    print(f"EXECUTION DETAILS - Graph with {num_nodes} nodes")
    print(f"BF steps: {num_bf_steps}, BFS steps: {num_bfs_steps}")
    print(f"{'='*80}")
    
    termination_settings = resolve_termination_settings(termination_cfg)
    previous_distance_latent = None

    for i in range(num_steps):
        # Check if samples exist (matching train.py logic exactly)
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        
        if not (bf_sample_exists or bfs_sample_exists):
            continue
            
        print(f"\n{'='*60}")
        print(f"STEP {i} → {i+1}")
        print(f"BF sample exists: {bf_sample_exists}, BFS sample exists: {bfs_sample_exists}")
        print(f"{'='*60}")
        
        # Prepare data for current step (exactly as in train.py)
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
        
        # Compute losses and accuracies
        if bf_sample_exists:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            
            # BF Distance Loss
            bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')
            
            # BF Distance Accuracy (tolerance-based)
            distance_errors = mx.abs(bf_distance_predictions - target_distance_bf)
            distance_tolerance = 0.1  # On normalized [0,1] scale
            distance_correct_mask = distance_errors <= distance_tolerance
            bf_distance_correct = mx.sum(distance_correct_mask).item()
            bf_distance_total = num_nodes
            bf_distance_correct_sum += bf_distance_correct
            bf_distance_total_sum += bf_distance_total
            
            # Track BF distance norm
            bf_distance_norm = mx.linalg.norm(bf_distance_predictions)
            accumulated_bf_distance_norms += bf_distance_norm
            bf_distance_step_count += 1
            
            # BF Predecessor Loss (with masking)
            valid_mask = (target_predecessor_bf != -1)
            safe_targets = mx.where(valid_mask, target_predecessor_bf, mx.zeros_like(target_predecessor_bf))
            per_node_ce = nn.losses.cross_entropy(bf_predecessor_predictions, safe_targets, reduction='none')
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            
            # BF Predecessor Accuracy (only on valid nodes)
            pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            pred_correct_mask = (pred_argmax == target_predecessor_bf) & valid_mask
            bf_predecessor_correct = mx.sum(pred_correct_mask).item()
            bf_predecessor_total = mx.sum(valid_mask).item()
            bf_predecessor_correct_sum += bf_predecessor_correct
            bf_predecessor_total_sum += bf_predecessor_total
            
            # Track BF predecessor norm
            bf_predecessor_norm = mx.linalg.norm(bf_predecessor_predictions)
            accumulated_bf_predecessor_norms += bf_predecessor_norm
            bf_predecessor_step_count += 1
            
            # BF Termination Loss
            bf_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bf'],
                termination_targets['bf'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bf_termination_loss = mx.array(0.0)
            
            # BF Termination Accuracy
            bf_term_pred = mx.sigmoid(termination_logits['bf']) > 0.5  # Use sigmoid + 0.5 threshold
            bf_term_correct = (bf_term_pred.astype(mx.float32) == termination_targets['bf']).item()
            bf_termination_correct_sum += bf_term_correct
            bf_termination_total_sum += 1
            
            # Track BF termination norm
            bf_termination_norm = mx.linalg.norm(termination_logits['bf'])
            accumulated_bf_termination_norms += bf_termination_norm
            bf_termination_step_count += 1
            
            print("\nBF DISTANCE:")
            print(f"  Loss: {bf_distance_loss.item():.6f}")
            print(f"  Accuracy: {bf_distance_correct}/{bf_distance_total} = {bf_distance_correct/bf_distance_total:.3f}")
            print(f"  Predictions (first 10): {bf_distance_predictions[:10].tolist()}")
            print(f"  Targets (first 10): {target_distance_bf[:10].tolist()}")
            
            print("\nBF PREDECESSOR:")
            print(f"  Loss: {bf_predecessor_loss.item():.6f}")
            print(f"  Accuracy: {bf_predecessor_correct}/{bf_predecessor_total} = {bf_predecessor_correct/max(bf_predecessor_total,1):.3f}")
            print(f"  Valid nodes: {bf_predecessor_total}")
            print(f"  Predictions (argmax, first 10): {pred_argmax[:10].tolist()}")
            print(f"  Targets (first 10): {target_predecessor_bf[:10].tolist()}")
            
            print("\nBF TERMINATION:")
            print(f"  Loss: {bf_termination_loss.item():.6f}")
            print(
                f"  Logit: {termination_logits['bf'].item():.4f}, "
                f"Prob: {mx.sigmoid(termination_logits['bf']).item():.4f}"
            )
            print(f"  Target: {termination_targets['bf'].item()}, Correct: {bf_term_correct}")
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)
            
        if bfs_sample_exists:
            # BFS State Loss
            bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean', with_logits=True)
            
            # BFS State Accuracy
            bfs_state_pred = mx.sigmoid(bfs_output) > 0.5  # Use sigmoid + 0.5 threshold
            bfs_state_correct_mask = (bfs_state_pred.astype(mx.float32) == target_bfs_state)
            bfs_state_correct = mx.sum(bfs_state_correct_mask).item()
            bfs_state_total = num_nodes
            bfs_state_correct_sum += bfs_state_correct
            bfs_state_total_sum += bfs_state_total
            
            # Track BFS state norm
            bfs_state_norm = mx.linalg.norm(bfs_output)
            accumulated_bfs_state_norms += bfs_state_norm
            bfs_state_step_count += 1
            
            # BFS Termination Loss
            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bfs'],
                termination_targets['bfs'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bfs_termination_loss = mx.array(0.0)
            
            # BFS Termination Accuracy
            bfs_term_pred = mx.sigmoid(termination_logits['bfs']) > 0.5
            bfs_term_correct = (bfs_term_pred.astype(mx.float32) == termination_targets['bfs']).item()
            bfs_termination_correct_sum += bfs_term_correct
            bfs_termination_total_sum += 1
            
            # Track BFS termination norm
            bfs_termination_norm = mx.linalg.norm(termination_logits['bfs'])
            accumulated_bfs_termination_norms += bfs_termination_norm
            bfs_termination_step_count += 1
            
            print("\nBFS STATE:")
            print(f"  Loss: {bfs_state_loss.item():.6f}")
            print(f"  Accuracy: {bfs_state_correct}/{bfs_state_total} = {bfs_state_correct/bfs_state_total:.3f}")
            print(f"  Predictions (first 10): {bfs_state_pred[:10].astype(mx.int32).tolist()}")
            print(f"  Targets (first 10): {target_bfs_state[:10].astype(mx.int32).tolist()}")
            
            print("\nBFS TERMINATION:")
            print(f"  Loss: {bfs_termination_loss.item():.6f}")
            print(
                f"  Logit: {termination_logits['bfs'].item():.4f}, "
                f"Prob: {mx.sigmoid(termination_logits['bfs']).item():.4f}"
            )
            print(f"  Target: {termination_targets['bfs'].item()}, Correct: {bfs_term_correct}")
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)
            
        # Accumulate losses
        raw_losses = mx.array([bf_distance_loss, bf_predecessor_loss, bfs_state_loss, bf_termination_loss, bfs_termination_loss])
        total_step_loss = mx.sum(raw_losses)
        accumulated_loss += total_step_loss
        accumulated_aux_losses += raw_losses
        
        # Update hidden states
        previous_step_hidden_states = processed_embeddings
        
        # Track hidden state norm
        hidden_state_norm = mx.linalg.norm(processed_embeddings)
        accumulated_hidden_state_norms += hidden_state_norm
        total_step_count += 1
        
        print("\nSTEP SUMMARY:")
        print(f"  Total step loss: {total_step_loss.item():.6f}")
        print(f"  Hidden state norm: {hidden_state_norm.item():.4f}")
        
    # Calculate averages (matching train.py logic)
    bf_steps = max(num_bf_steps - 1, 0)
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
    
    # Calculate overall accuracies
    overall_bf_distance_acc = bf_distance_correct_sum / max(bf_distance_total_sum, 1)
    overall_bf_predecessor_acc = bf_predecessor_correct_sum / max(bf_predecessor_total_sum, 1)
    overall_bfs_state_acc = bfs_state_correct_sum / max(bfs_state_total_sum, 1)
    overall_bf_termination_acc = bf_termination_correct_sum / max(bf_termination_total_sum, 1)
    overall_bfs_termination_acc = bfs_termination_correct_sum / max(bfs_termination_total_sum, 1)
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Effective steps for averaging: {effective_steps}")
    print(f"BF steps executed: {bf_steps}, BFS steps executed: {bfs_steps}")
    print("\nAVERAGE LOSSES:")
    print(f"  Total: {average_loss.item():.6f}")
    print(f"  BF Distance: {avg_aux_losses[0].item():.6f}")
    print(f"  BF Predecessor: {avg_aux_losses[1].item():.6f}")
    print(f"  BFS State: {avg_aux_losses[2].item():.6f}")
    print(f"  BF Termination: {avg_aux_losses[3].item():.6f}")
    print(f"  BFS Termination: {avg_aux_losses[4].item():.6f}")
    print("\nOVERALL ACCURACIES:")
    print(f"  BF Distance: {overall_bf_distance_acc:.3f} ({bf_distance_correct_sum}/{bf_distance_total_sum})")
    print(f"  BF Predecessor: {overall_bf_predecessor_acc:.3f} ({bf_predecessor_correct_sum}/{bf_predecessor_total_sum})")
    print(f"  BFS State: {overall_bfs_state_acc:.3f} ({bfs_state_correct_sum}/{bfs_state_total_sum})")
    print(f"  BF Termination: {overall_bf_termination_acc:.3f} ({bf_termination_correct_sum}/{bf_termination_total_sum})")
    print(f"  BFS Termination: {overall_bfs_termination_acc:.3f} ({bfs_termination_correct_sum}/{bfs_termination_total_sum})")
    print(f"{'='*80}\n")
    
    # Calculate average per-head norms
    avg_bf_distance_norm = accumulated_bf_distance_norms / max(bf_distance_step_count, 1)
    avg_bf_predecessor_norm = accumulated_bf_predecessor_norms / max(bf_predecessor_step_count, 1)
    avg_bfs_state_norm = accumulated_bfs_state_norms / max(bfs_state_step_count, 1)
    avg_bf_termination_norm = accumulated_bf_termination_norms / max(bf_termination_step_count, 1)
    avg_bfs_termination_norm = accumulated_bfs_termination_norms / max(bfs_termination_step_count, 1)
    avg_hidden_state_norm = accumulated_hidden_state_norms / max(total_step_count, 1)
    
    # Return both the average loss and all the norms
    per_head_norms = {
        'bf_distance': avg_bf_distance_norm,
        'bf_predecessor': avg_bf_predecessor_norm,
        'bfs_state': avg_bfs_state_norm,
        'bf_termination': avg_bf_termination_norm,
        'bfs_termination': avg_bfs_termination_norm,
        'hidden_state': avg_hidden_state_norm
    }
    
    return average_loss, per_head_norms


def calculate_losses_and_accuracies(
    model,
    graph_data,
    embedding_dim=128,
    termination_cfg=None,
    selected_tasks=None,
):
    """
    Rewritten from scratch to match the exact loss computation logic from train.py.
    Returns: (aux_losses[5], total_loss, accuracies[5])
    """
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([5])  # [bf_dist, bf_pred, bfs_state, bf_term, bfs_term]
    
    # Accuracy counters - accumulate across all steps
    bf_distance_correct_sum = 0
    bf_distance_total_sum = 0
    bf_predecessor_correct_sum = 0
    bf_predecessor_total_sum = 0  
    bfs_state_correct_sum = 0
    bfs_state_total_sum = 0
    bf_termination_correct_sum = 0
    bf_termination_total_sum = 0
    bfs_termination_correct_sum = 0
    bfs_termination_total_sum = 0
    
    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, 2 * embedding_dim])
    
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps)
    
    termination_settings = resolve_termination_settings(termination_cfg)
    previous_distance_latent = None

    selected_tasks = _normalize_selected_tasks(selected_tasks)
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
        # Check if samples exist - exactly as in train.py
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        
        if not (bf_sample_exists or bfs_sample_exists):
            continue
            
        # Prepare data - exactly as in train.py
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
        
        # Compute losses and accuracies
        if bf_sample_exists and selected_tasks["bf"]:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            
            # BF Distance Loss - exactly as in train.py
            bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')
            
            # BF Distance Accuracy
            distance_errors = mx.abs(bf_distance_predictions - target_distance_bf)
            distance_tolerance = 0.1  # 10% tolerance on normalized scale
            distance_correct = mx.sum(distance_errors <= distance_tolerance).item()
            bf_distance_correct_sum += distance_correct
            bf_distance_total_sum += num_nodes
            
            # BF Predecessor Loss - exactly as in train.py
            valid_mask = (target_predecessor_bf != -1)
            safe_targets = mx.where(valid_mask, target_predecessor_bf, mx.zeros_like(target_predecessor_bf))
            
            # Only compute loss if we have valid targets
            if mx.sum(valid_mask) > 0:
                per_node_ce = nn.losses.cross_entropy(bf_predecessor_predictions, safe_targets, reduction='none', label_smoothing=1e-6)
                valid_mask_f = valid_mask.astype(mx.float32)
                denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
                bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            else:
                bf_predecessor_loss = mx.array(0.0)
            
            # BF Predecessor Accuracy - only count valid nodes
            pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            pred_correct = mx.sum((pred_argmax == target_predecessor_bf) & valid_mask).item()
            pred_total = mx.sum(valid_mask).item()
            bf_predecessor_correct_sum += pred_correct
            bf_predecessor_total_sum += pred_total
            
            # BF Termination Loss - exactly as in train.py
            bf_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bf'],
                termination_targets['bf'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bf_termination_loss = mx.array(0.0)
            
            # BF Termination Accuracy - since loss uses raw logits, we apply sigmoid for accuracy
            bf_term_prob = mx.sigmoid(termination_logits['bf'])
            bf_term_pred = (bf_term_prob > 0.5).astype(mx.float32)
            bf_term_correct = (bf_term_pred == termination_targets['bf']).item()
            bf_termination_correct_sum += bf_term_correct
            bf_termination_total_sum += 1
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)
            
        if bfs_sample_exists and selected_tasks["bfs"]:
            # BFS State Loss - exactly as in train.py
            bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean', with_logits=True)
            
            # BFS State Accuracy - apply sigmoid since loss uses logits
            bfs_state_probs = mx.sigmoid(bfs_output)
            bfs_state_pred = (bfs_state_probs > 0.5).astype(mx.float32)
            bfs_correct = mx.sum(bfs_state_pred == target_bfs_state).item()
            bfs_state_correct_sum += bfs_correct
            bfs_state_total_sum += num_nodes
            
            # BFS Termination Loss - exactly as in train.py  
            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bfs'],
                termination_targets['bfs'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bfs_termination_loss = mx.array(0.0)
            
            # BFS Termination Accuracy
            bfs_term_prob = mx.sigmoid(termination_logits['bfs'])
            bfs_term_pred = (bfs_term_prob > 0.5).astype(mx.float32)
            bfs_term_correct = (bfs_term_pred == termination_targets['bfs']).item()
            bfs_termination_correct_sum += bfs_term_correct
            bfs_termination_total_sum += 1
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)
            
        # Accumulate losses - exactly as in train.py
        raw_losses = mx.array(
            [bf_distance_loss, bf_predecessor_loss, bfs_state_loss, bf_termination_loss, bfs_termination_loss]
        )
        raw_losses = raw_losses * loss_mask
        total_step_loss = mx.sum(raw_losses)
        accumulated_loss += total_step_loss
        accumulated_aux_losses += raw_losses
        
        # Update hidden states for next step
        previous_step_hidden_states = processed_embeddings
        
    # Calculate averages - exactly as in train.py
    bf_steps = max(num_bf_steps - 1, 0)
    bfs_steps = max(num_bfs_steps - 1, 0)
    effective_steps = _effective_step_count(bf_steps, bfs_steps, selected_tasks)
    
    average_loss = accumulated_loss / effective_steps
    
    # Per-task averaging - exactly as in train.py
    per_task_counter = mx.array([
        max(bf_steps, 1),   # bf_distance
        max(bf_steps, 1),   # bf_predecessor  
        max(bfs_steps, 1),  # bfs_state
        max(bf_steps, 1),   # bf_termination
        max(bfs_steps, 1),  # bfs_termination
    ], dtype=mx.float32)
    avg_aux_losses = accumulated_aux_losses / per_task_counter
    
    # Calculate overall accuracies
    bf_distance_acc = bf_distance_correct_sum / max(bf_distance_total_sum, 1)
    bf_predecessor_acc = bf_predecessor_correct_sum / max(bf_predecessor_total_sum, 1)
    bfs_state_acc = bfs_state_correct_sum / max(bfs_state_total_sum, 1)
    bf_termination_acc = bf_termination_correct_sum / max(bf_termination_total_sum, 1)
    bfs_termination_acc = bfs_termination_correct_sum / max(bfs_termination_total_sum, 1)
    
    accuracies = mx.array([bf_distance_acc, bf_predecessor_acc, bfs_state_acc, bf_termination_acc, bfs_termination_acc])
    
    return avg_aux_losses, average_loss, accuracies


def calculate_accuracies(
    model,
    graph_data,
    embedding_dim=128,
    termination_cfg=None,
    selected_tasks=None,
):
    """
    Calculate only accuracies, following the exact same logic as calculate_losses_and_accuracies_v2.
    Returns: accuracies[5] array
    """
    # Accuracy counters - accumulate across all steps
    bf_distance_correct_sum = 0
    bf_distance_total_sum = 0
    bf_predecessor_correct_sum = 0
    bf_predecessor_total_sum = 0
    bfs_state_correct_sum = 0
    bfs_state_total_sum = 0
    bf_termination_correct_sum = 0
    bf_termination_total_sum = 0
    bfs_termination_correct_sum = 0
    bfs_termination_total_sum = 0
    
    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, 2 * embedding_dim])
    
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps)
    
    termination_settings = resolve_termination_settings(termination_cfg)
    previous_distance_latent = None

    selected_tasks = _normalize_selected_tasks(selected_tasks)

    for i in range(num_steps):
        # Check if samples exist
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        
        if not (bf_sample_exists or bfs_sample_exists):
            continue
            
        # Prepare data
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
        
        # Calculate accuracies
        if bf_sample_exists and selected_tasks["bf"]:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            
            # BF Distance Accuracy
            distance_errors = mx.abs(bf_distance_predictions - target_distance_bf)
            distance_tolerance = 0.1
            distance_correct = mx.sum(distance_errors <= distance_tolerance).item()
            bf_distance_correct_sum += distance_correct
            bf_distance_total_sum += num_nodes
            
            # BF Predecessor Accuracy
            valid_mask = (target_predecessor_bf != -1)
            pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            pred_correct = mx.sum((pred_argmax == target_predecessor_bf) & valid_mask).item()
            pred_total = mx.sum(valid_mask).item()
            bf_predecessor_correct_sum += pred_correct
            bf_predecessor_total_sum += pred_total
            
            # BF Termination Accuracy
            bf_term_prob = mx.sigmoid(termination_logits['bf'])
            bf_term_pred = (bf_term_prob > 0.5).astype(mx.float32)
            bf_term_correct = (bf_term_pred == termination_targets['bf']).item()
            bf_termination_correct_sum += bf_term_correct
            bf_termination_total_sum += 1
            
        if bfs_sample_exists and selected_tasks["bfs"]:
            # BFS State Accuracy
            bfs_state_probs = mx.sigmoid(bfs_output)
            bfs_state_pred = (bfs_state_probs > 0.5).astype(mx.float32)
            bfs_correct = mx.sum(bfs_state_pred == target_bfs_state).item()
            bfs_state_correct_sum += bfs_correct
            bfs_state_total_sum += num_nodes
            
            # BFS Termination Accuracy
            bfs_term_prob = mx.sigmoid(termination_logits['bfs'])
            bfs_term_pred = (bfs_term_prob > 0.5).astype(mx.float32)
            bfs_term_correct = (bfs_term_pred == termination_targets['bfs']).item()
            bfs_termination_correct_sum += bfs_term_correct
            bfs_termination_total_sum += 1
            
        # Update hidden states for next step
        previous_step_hidden_states = processed_embeddings
        
    # Calculate overall accuracies
    bf_distance_acc = bf_distance_correct_sum / max(bf_distance_total_sum, 1)
    bf_predecessor_acc = bf_predecessor_correct_sum / max(bf_predecessor_total_sum, 1) 
    bfs_state_acc = bfs_state_correct_sum / max(bfs_state_total_sum, 1)
    bf_termination_acc = bf_termination_correct_sum / max(bf_termination_total_sum, 1)
    bfs_termination_acc = bfs_termination_correct_sum / max(bfs_termination_total_sum, 1)
    
    return mx.array([bf_distance_acc, bf_predecessor_acc, bfs_state_acc, bf_termination_acc, bfs_termination_acc])


def _graph_failure_details(
    model,
    graph_data,
    embedding_dim=128,
    termination_cfg=None,
    selected_tasks=None,
    include_step_details=False,
):
    """Collect per-graph failure details for error analysis."""
    selected_tasks = _normalize_selected_tasks(selected_tasks)
    termination_settings = resolve_termination_settings(termination_cfg)

    num_nodes = int(graph_data["num_nodes"])
    num_bf_steps = len(graph_data["bf_distance_targets"])
    num_bfs_steps = len(graph_data["bfs_state_targets"])
    num_steps = max(num_bf_steps, num_bfs_steps)

    counts = {
        "bf_distance_correct": 0,
        "bf_distance_total": 0,
        "bf_predecessor_correct": 0,
        "bf_predecessor_total": 0,
        "bfs_state_correct": 0,
        "bfs_state_total": 0,
        "bf_termination_correct": 0,
        "bf_termination_total": 0,
        "bfs_termination_correct": 0,
        "bfs_termination_total": 0,
    }
    termination_confusion = {
        "bf_fp": 0,
        "bf_fn": 0,
        "bfs_fp": 0,
        "bfs_fn": 0,
    }

    previous_step_hidden_states = mx.zeros([num_nodes, 2 * embedding_dim])
    previous_distance_latent = None

    step_failures = []
    first_failure_step = None

    for i in range(num_steps):
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        if not (bf_sample_exists or bfs_sample_exists):
            continue

        if bfs_sample_exists:
            true_bfs_state = graph_data["bfs_state_targets"][i]
            target_bfs_state = graph_data["bfs_state_targets"][i + 1]
        else:
            true_bfs_state = graph_data["bfs_state_targets"][-1]
            target_bfs_state = graph_data["bfs_state_targets"][-1]

        if bf_sample_exists:
            true_distance_bf = graph_data["bf_distance_targets"][i]
            target_distance_bf = graph_data["bf_distance_targets"][i + 1]
            target_predecessor_bf = graph_data["bf_predecessor_targets"][i + 1]
        else:
            true_distance_bf = graph_data["bf_distance_targets"][-1]
            target_distance_bf = graph_data["bf_distance_targets"][-1]
            target_predecessor_bf = graph_data["bf_predecessor_targets"][-1]

        is_last_bf_step = (i + 1) == (num_bf_steps - 1)
        is_last_bfs_step = (i + 1) == (num_bfs_steps - 1)
        termination_targets = {
            "bf": mx.array(1.0 if is_last_bf_step else 0.0),
            "bfs": mx.array(1.0 if is_last_bfs_step else 0.0),
        }

        node_algo_features = mx.stack([true_bfs_state, true_distance_bf], axis=1)
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data["edge_matrix"])

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

        step_entry = {
            "step": int(i + 1),
            "bf_distance_incorrect": 0,
            "bf_predecessor_incorrect": 0,
            "bfs_state_incorrect": 0,
            "bf_termination_incorrect": 0,
            "bfs_termination_incorrect": 0,
        }

        if bf_sample_exists and selected_tasks["bf"]:
            bf_distance_predictions, bf_predecessor_predictions = bf_output

            distance_errors = mx.abs(bf_distance_predictions - target_distance_bf)
            distance_correct = int(mx.sum(distance_errors <= 0.1).item())
            counts["bf_distance_correct"] += distance_correct
            counts["bf_distance_total"] += num_nodes
            step_entry["bf_distance_incorrect"] = int(num_nodes - distance_correct)

            valid_mask = target_predecessor_bf != -1
            pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            pred_correct = int(mx.sum((pred_argmax == target_predecessor_bf) & valid_mask).item())
            pred_total = int(mx.sum(valid_mask).item())
            counts["bf_predecessor_correct"] += pred_correct
            counts["bf_predecessor_total"] += pred_total
            step_entry["bf_predecessor_incorrect"] = int(max(pred_total - pred_correct, 0))

            bf_term_prob = float(mx.sigmoid(termination_logits["bf"]).item())
            bf_term_pred = 1 if bf_term_prob > 0.5 else 0
            bf_term_target = int(termination_targets["bf"].item())
            bf_term_correct = int(bf_term_pred == bf_term_target)
            counts["bf_termination_correct"] += bf_term_correct
            counts["bf_termination_total"] += 1
            step_entry["bf_termination_incorrect"] = 1 - bf_term_correct
            if bf_term_pred == 1 and bf_term_target == 0:
                termination_confusion["bf_fp"] += 1
            elif bf_term_pred == 0 and bf_term_target == 1:
                termination_confusion["bf_fn"] += 1

        if bfs_sample_exists and selected_tasks["bfs"]:
            bfs_state_probs = mx.sigmoid(bfs_output)
            bfs_state_pred = (bfs_state_probs > 0.5).astype(mx.float32)
            bfs_correct = int(mx.sum(bfs_state_pred == target_bfs_state).item())
            counts["bfs_state_correct"] += bfs_correct
            counts["bfs_state_total"] += num_nodes
            step_entry["bfs_state_incorrect"] = int(num_nodes - bfs_correct)

            bfs_term_prob = float(mx.sigmoid(termination_logits["bfs"]).item())
            bfs_term_pred = 1 if bfs_term_prob > 0.5 else 0
            bfs_term_target = int(termination_targets["bfs"].item())
            bfs_term_correct = int(bfs_term_pred == bfs_term_target)
            counts["bfs_termination_correct"] += bfs_term_correct
            counts["bfs_termination_total"] += 1
            step_entry["bfs_termination_incorrect"] = 1 - bfs_term_correct
            if bfs_term_pred == 1 and bfs_term_target == 0:
                termination_confusion["bfs_fp"] += 1
            elif bfs_term_pred == 0 and bfs_term_target == 1:
                termination_confusion["bfs_fn"] += 1

        step_error_units = (
            step_entry["bf_distance_incorrect"]
            + step_entry["bf_predecessor_incorrect"]
            + step_entry["bfs_state_incorrect"]
            + step_entry["bf_termination_incorrect"]
            + step_entry["bfs_termination_incorrect"]
        )
        if step_error_units > 0:
            if first_failure_step is None:
                first_failure_step = int(i + 1)
            if include_step_details:
                step_failures.append(step_entry)

        previous_step_hidden_states = processed_embeddings

    accuracies = {
        "bf_distance": counts["bf_distance_correct"] / max(counts["bf_distance_total"], 1),
        "bf_predecessor": counts["bf_predecessor_correct"] / max(counts["bf_predecessor_total"], 1),
        "bfs_state": counts["bfs_state_correct"] / max(counts["bfs_state_total"], 1),
        "bf_termination": counts["bf_termination_correct"] / max(counts["bf_termination_total"], 1),
        "bfs_termination": counts["bfs_termination_correct"] / max(counts["bfs_termination_total"], 1),
    }
    incorrect = {
        "bf_distance": counts["bf_distance_total"] - counts["bf_distance_correct"],
        "bf_predecessor": counts["bf_predecessor_total"] - counts["bf_predecessor_correct"],
        "bfs_state": counts["bfs_state_total"] - counts["bfs_state_correct"],
        "bf_termination": counts["bf_termination_total"] - counts["bf_termination_correct"],
        "bfs_termination": counts["bfs_termination_total"] - counts["bfs_termination_correct"],
    }

    failed_tasks = []
    if selected_tasks["bf"] and incorrect["bf_distance"] > 0:
        failed_tasks.append("bf_distance")
    if selected_tasks["bf"] and incorrect["bf_predecessor"] > 0:
        failed_tasks.append("bf_predecessor")
    if selected_tasks["bfs"] and incorrect["bfs_state"] > 0:
        failed_tasks.append("bfs_state")
    if selected_tasks["bf"] and incorrect["bf_termination"] > 0:
        failed_tasks.append("bf_termination")
    if selected_tasks["bfs"] and incorrect["bfs_termination"] > 0:
        failed_tasks.append("bfs_termination")

    return {
        "num_nodes": num_nodes,
        "num_bf_steps": int(num_bf_steps),
        "num_bfs_steps": int(num_bfs_steps),
        "first_failure_step": first_failure_step,
        "accuracy": {k: float(v) for k, v in accuracies.items()},
        "incorrect": {k: int(v) for k, v in incorrect.items()},
        "termination_confusion": {k: int(v) for k, v in termination_confusion.items()},
        "failed_tasks": failed_tasks,
        "total_error_units": int(sum(incorrect.values())),
        "step_failures": step_failures if include_step_details else None,
    }


def analyze_failure_modes(
    model,
    dataset,
    embedding_dim=128,
    termination_cfg=None,
    selected_tasks=None,
    max_graphs=None,
    max_failure_records=200,
    include_step_details=False,
):
    """Analyze misclassification patterns and return per-graph failure summaries."""
    selected_tasks = _normalize_selected_tasks(selected_tasks)
    termination_settings = resolve_termination_settings(termination_cfg)

    model.eval()
    graphs = dataset if max_graphs is None else dataset[: max(max_graphs, 0)]

    failures_by_task = {
        "bf_distance": [],
        "bf_predecessor": [],
        "bfs_state": [],
        "bf_termination": [],
        "bfs_termination": [],
    }
    failed_graphs = []
    ranked_failed = []

    for graph_index, graph_data in enumerate(graphs):
        details = _graph_failure_details(
            model=model,
            graph_data=graph_data,
            embedding_dim=embedding_dim,
            termination_cfg=termination_cfg,
            selected_tasks=selected_tasks,
            include_step_details=include_step_details,
        )
        if not details["failed_tasks"]:
            continue

        details["graph_index"] = int(graph_index)
        for task_name in details["failed_tasks"]:
            failures_by_task[task_name].append(int(graph_index))
        ranked_failed.append((int(graph_index), int(details["total_error_units"])))

        if max_failure_records is None or len(failed_graphs) < max_failure_records:
            failed_graphs.append(details)

    ranked_failed.sort(key=lambda x: (-x[1], x[0]))
    ranked_failed_graph_indices = [idx for idx, _ in ranked_failed]

    return {
        "termination": {
            "mode": termination_settings["mode"],
            "distance": termination_settings["distance_type"],
            "latent": termination_settings["distance_latent"],
            "threshold": float(termination_settings["distance_threshold"]),
            "distance_signal": bool(termination_settings["distance_signal"]),
        },
        "selected_tasks": selected_tasks,
        "num_graphs_analyzed": int(len(graphs)),
        "num_failed_graphs": int(len(ranked_failed_graph_indices)),
        "failure_rate": float(len(ranked_failed_graph_indices) / max(len(graphs), 1)),
        "failures_by_task": failures_by_task,
        "ranked_failed_graph_indices": ranked_failed_graph_indices,
        "failed_graphs": failed_graphs,
    }


def safe_trained_model(model, sample_input_embeddings, sample_edge_matrix, output_path="trained_model.mlxfn"):
    """
    Save trained model for later use.
    
    Args:
        model: The trained NGE model
        sample_input_embeddings: Sample input embeddings array to trace the model
        sample_edge_matrix: Sample edge matrix to trace the model
        output_path: Path where the model will be saved (default: trained_model.mlxfn)
    
    Note:
        The exported function flattens the model outputs into a tuple of arrays:
        (bfs_output, bf_distance_predictions, bf_predecessor_predictions, 
         bf_termination_prob, bfs_termination_prob, processed_embeddings)
    """
    
    mx.eval(model.parameters())

    def call(input_embeddings, edge_matrix):
        bfs_output, bf_output, termination_probs, processed_embeddings = model((input_embeddings, edge_matrix))
        bf_distance_predictions, bf_predecessor_predictions = bf_output
        
        # Flatten outputs to a simple tuple of arrays for export
        return (bfs_output, 
                bf_distance_predictions, 
                bf_predecessor_predictions,
                termination_probs['bf'],
                termination_probs['bfs'],
                processed_embeddings)

    mx.export_function(output_path, call, (sample_input_embeddings, sample_edge_matrix))
