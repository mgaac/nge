import mlx.core as mx
import mlx.nn as nn

import numpy as np

def print_execution_details(model, graph_data, embedding_dim):
    accumulated_loss = mx.array(0.0)
    num_nodes = graph_data['num_nodes']
   
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])

    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])

    num_steps = max(num_bf_steps, num_bfs_steps)
    
    steps_executed = 0
    
    for i in range(num_steps):
        # Check if samples exist
        bf_sample_exists = i < num_bf_steps and (i + 1) < num_bf_steps
        bfs_sample_exists = i < num_bfs_steps and (i + 1) < num_bfs_steps

        # If neither sample exists, skip this step
        if not (bf_sample_exists or bfs_sample_exists):
            continue

        steps_executed += 1

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

        total_step_loss = bf_distance_loss + bf_predecessor_loss + bfs_state_loss + bf_termination_loss + bfs_termination_loss

        # Update for next step
        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss

        print(f"\n=== Step {steps_executed} ===")
        print(f"Total Loss: {total_step_loss.item():.6f}")
        
        # Loss breakdown
        print("Loss Breakdown:")
        print(f"  BF Distance:     {bf_distance_loss.item():.6f}")
        print(f"  BF Predecessor:  {bf_predecessor_loss.item():.6f}")
        print(f"  BFS State:       {bfs_state_loss.item():.6f}")
        print(f"  BF Termination:  {bf_termination_loss.item():.6f}")
        print(f"  BFS Termination: {bfs_termination_loss.item():.6f}")

        # Model outputs and statistics
        if bf_sample_exists:
            bf_distance_logits = bf_output[0]
            bf_predecessor_logits = bf_output[1]
            
            print("\nBF Distance:")
            print(f"  Logits: norm={mx.linalg.norm(bf_distance_logits).item():.6f}, std={mx.std(bf_distance_logits).item():.6f}")
            print(f"  Pred: {np.array(bf_distance_logits).round(4)}")
            print(f"  Targ: {np.array(target_distance_bf).round(4)}")
            
            print("\nBF Predecessor:")
            print(f"  Logits: norm={mx.linalg.norm(bf_predecessor_logits).item():.6f}, std={mx.std(bf_predecessor_logits).item():.6f}")
            print(f"  Pred: {np.argmax(np.array(bf_predecessor_logits), axis=-1)}")
            print(f"  Targ: {np.array(target_predecessor_bf)}")
            
            print("\nBF Termination:")
            bf_term_logit = termination_probs['bf']
            bf_term_prob  = mx.sigmoid(bf_term_logit)
            print(f"  Logit: norm={mx.linalg.norm(bf_term_logit).item():.6f}, value={bf_term_logit.item():.4f}")
            print(f"  Prob: {bf_term_prob.item():.4f}")
            print(f"  Targ: {termination_targets['bf'].item():.4f}")
            
        if bfs_sample_exists:
            print("\nBFS State:")
            print(f"  Logits: norm={mx.linalg.norm(bfs_output).item():.6f}, std={mx.std(bfs_output).item():.6f}")
            print(f"  Pred: {(np.array(bfs_output) > 0).astype(int)}")
            print(f"  Targ: {np.array(target_bfs_state).astype(int)}")
            
            print("\nBFS Termination:")
            bfs_term_logit = termination_probs['bfs']
            bfs_term_prob  = mx.sigmoid(bfs_term_logit)
            print(f"  Logit: norm={mx.linalg.norm(bfs_term_logit).item():.6f}, value={bfs_term_logit.item():.4f}")
            print(f"  Prob: {bfs_term_prob.item():.4f}")
            print(f"  Targ: {termination_targets['bfs'].item():.4f}")
            
        print(f"\nHidden State: norm={mx.linalg.norm(processed_embeddings).item():.6f}, std={mx.std(processed_embeddings).item():.6f}")

    # Calculate average loss correctly
    if steps_executed > 0:
        average_loss = accumulated_loss / steps_executed
    else:
        average_loss = mx.array(0.0)

    print(f"\n=== Summary ===")
    print(f"Steps executed: {steps_executed}")
    print(f"Average loss: {average_loss.item():.6f}")

    return average_loss, mx.linalg.norm(processed_embeddings)

def calculate_losses_and_accuracies(model, graph_data, embedding_dim=128):
    """
    Combined function that calculates both losses and accuracies in a single forward pass.
    Returns: (aux_losses, total_loss, accuracies)
    """
    accumulated_loss = mx.array(0.0)
    
    # Accuracy counters
    bf_distance_correct = 0
    bf_predecessor_correct = 0
    bfs_state_correct = 0
    bf_termination_correct = 0
    bfs_termination_correct = 0
    
    # Total sample counters for accurate percentage calculation
    bf_distance_total = 0
    bf_predecessor_total = 0
    bfs_state_total = 0
    bf_termination_total = 0
    bfs_termination_total = 0

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])

    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])

    num_steps = max(num_bf_steps, num_bfs_steps)
    
    # Loss accumulators for auxiliary losses
    bf_distance_loss_acc = mx.array(0.0)
    bf_predecessor_loss_acc = mx.array(0.0)
    bfs_state_loss_acc = mx.array(0.0)
    bf_termination_loss_acc = mx.array(0.0)
    bfs_termination_loss_acc = mx.array(0.0)
    
    steps_executed = 0
    bf_steps_executed = 0
    bfs_steps_executed = 0
    
    for i in range(num_steps):
        # Check if samples exist
        bf_sample_exists = i < num_bf_steps and (i + 1) < num_bf_steps
        bfs_sample_exists = i < num_bfs_steps and (i + 1) < num_bfs_steps

        # If neither sample exists, skip this step
        if not (bf_sample_exists or bfs_sample_exists):
            continue

        if bf_sample_exists:
            bf_steps_executed += 1
        if bfs_sample_exists:
            bfs_steps_executed += 1
        steps_executed += 1

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

        # Compute losses and accuracies
        if bf_sample_exists:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            
            # Losses
            bf_distance_loss = nn.losses.mse_loss(bf_distance_predictions, target_distance_bf, reduction='mean')
            # Masked cross-entropy for predecessor targets >= 0
            pred_mask = (target_predecessor_bf >= 0).astype(mx.float32)
            safe_targets = mx.where(pred_mask > 0, target_predecessor_bf, mx.zeros_like(target_predecessor_bf))
            ce_per_node = nn.losses.cross_entropy(bf_predecessor_predictions, safe_targets, reduction='none')
            mask_sum = mx.maximum(mx.sum(pred_mask), 1.0)
            bf_predecessor_loss = (mx.sum(ce_per_node * pred_mask) / mask_sum)
            bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean', with_logits=True)
            
            bf_distance_loss_acc += bf_distance_loss
            bf_predecessor_loss_acc += bf_predecessor_loss
            bf_termination_loss_acc += bf_termination_loss
            
            # Accuracies
            # BF Distance: correct if within 10% relative error or within 0.1 absolute error (for small values)
            distance_errors = mx.abs(bf_distance_predictions - target_distance_bf)
            relative_tolerance = 0.1 * mx.maximum(mx.abs(target_distance_bf), mx.array(1.0))  # minimum tolerance of 0.1
            absolute_tolerance = mx.array(0.1)
            distance_correct_mask = (distance_errors <= mx.maximum(relative_tolerance, absolute_tolerance))
            bf_distance_correct += mx.sum(distance_correct_mask).item()
            bf_distance_total += len(target_distance_bf)
            
            # BF Predecessor: correct if argmax matches target, only for valid targets
            pred_predecessors = mx.argmax(bf_predecessor_predictions, axis=-1)
            correct_mask = ((pred_predecessors == target_predecessor_bf).astype(mx.float32) * pred_mask)
            bf_predecessor_correct += mx.sum(correct_mask).item()
            bf_predecessor_total += int(mx.sum(pred_mask).item())
            
            # BF Termination: correct if (logit > 0.0) matches target
            bf_term_pred = (termination_probs['bf'] > 0.0).astype(mx.float32)
            bf_termination_correct += int(bf_term_pred.item() == termination_targets['bf'].item())
            bf_termination_total += 1

        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)

        if bfs_sample_exists:
            # Losses
            bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean', with_logits=True)
            bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean', with_logits=True)
            
            bfs_state_loss_acc += bfs_state_loss
            bfs_termination_loss_acc += bfs_termination_loss
            
            # Accuracies
            # BFS State: correct if sigmoid(logit) > 0.5 matches binary target
            bfs_probs = mx.sigmoid(bfs_output)
            bfs_pred = (bfs_probs > 0.5).astype(mx.float32)
            bfs_state_correct += mx.sum(bfs_pred == target_bfs_state).item()
            bfs_state_total += len(target_bfs_state)
            
            # BFS Termination: correct if sigmoid(logit) > 0.5 matches target  
            bfs_term_prob = mx.sigmoid(termination_probs['bfs'])
            bfs_term_pred = (bfs_term_prob > 0.5).astype(mx.float32)
            bfs_termination_correct += int(bfs_term_pred.item() == termination_targets['bfs'].item())
            bfs_termination_total += 1
            
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)

        total_step_loss = bf_distance_loss + bf_predecessor_loss + bfs_state_loss + bf_termination_loss + bfs_termination_loss

        # Update for next step
        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss

    # Calculate average losses
    if steps_executed > 0:
        average_loss = accumulated_loss / steps_executed
        avg_bf_distance_loss    = bf_distance_loss_acc    / max(bf_steps_executed, 1)
        avg_bf_predecessor_loss = bf_predecessor_loss_acc / max(bf_steps_executed, 1)
        avg_bf_termination_loss = bf_termination_loss_acc / max(bf_steps_executed, 1)
        avg_bfs_state_loss      = bfs_state_loss_acc      / max(bfs_steps_executed, 1)
        avg_bfs_termination_loss= bfs_termination_loss_acc/ max(bfs_steps_executed, 1)
    else:
        average_loss = mx.array(0.0)
        avg_bf_distance_loss = mx.array(0.0)
        avg_bf_predecessor_loss = mx.array(0.0)
        avg_bfs_state_loss = mx.array(0.0)
        avg_bf_termination_loss = mx.array(0.0)
        avg_bfs_termination_loss = mx.array(0.0)

    # Calculate accuracies (avoiding division by zero)
    bf_distance_acc = bf_distance_correct / bf_distance_total if bf_distance_total > 0 else 0.0
    bf_predecessor_acc = bf_predecessor_correct / bf_predecessor_total if bf_predecessor_total > 0 else 0.0
    bfs_state_acc = bfs_state_correct / bfs_state_total if bfs_state_total > 0 else 0.0
    bf_termination_acc = bf_termination_correct / bf_termination_total if bf_termination_total > 0 else 0.0
    bfs_termination_acc = bfs_termination_correct / bfs_termination_total if bfs_termination_total > 0 else 0.0

    aux_losses = mx.array([avg_bf_distance_loss, avg_bf_predecessor_loss, avg_bfs_state_loss, avg_bf_termination_loss, avg_bfs_termination_loss])
    accuracies = mx.array([bf_distance_acc, bf_predecessor_acc, bfs_state_acc, bf_termination_acc, bfs_termination_acc])

    return aux_losses, average_loss, accuracies

def calculate_accuracies(model, graph_data, embedding_dim=128):
    """
    Function that calculates only accuracies in a single forward pass.
    Returns: accuracies array [bf_distance_acc, bf_predecessor_acc, bfs_state_acc, bf_termination_acc, bfs_termination_acc]
    """
    # Accuracy counters
    bf_distance_correct = 0
    bf_predecessor_correct = 0
    bfs_state_correct = 0
    bf_termination_correct = 0
    bfs_termination_correct = 0
    
    # Total sample counters for accurate percentage calculation
    bf_distance_total = 0
    bf_predecessor_total = 0
    bfs_state_total = 0
    bf_termination_total = 0
    bfs_termination_total = 0

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])

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

        # Compute accuracies only
        if bf_sample_exists:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            
            # BF Distance: correct if within 10% relative error or within 0.1 absolute error (for small values)
            distance_errors = mx.abs(bf_distance_predictions - target_distance_bf)
            relative_tolerance = 0.1 * mx.maximum(mx.abs(target_distance_bf), mx.array(1.0))  # minimum tolerance of 0.1
            absolute_tolerance = mx.array(0.1)
            distance_correct_mask = (distance_errors <= mx.maximum(relative_tolerance, absolute_tolerance))
            bf_distance_correct += mx.sum(distance_correct_mask).item()
            bf_distance_total += len(target_distance_bf)
            
            # BF Predecessor: correct if argmax matches target
            pred_predecessors = mx.argmax(bf_predecessor_predictions, axis=-1)
            bf_predecessor_correct += mx.sum(pred_predecessors == target_predecessor_bf).item()
            bf_predecessor_total += len(target_predecessor_bf)
            
            # BF Termination: correct if (prediction > 0.5) matches target
            bf_term_pred = (termination_probs['bf'] > 0.5).astype(mx.float32)
            bf_termination_correct += int(bf_term_pred.item() == termination_targets['bf'].item())
            bf_termination_total += 1

        if bfs_sample_exists:
            # BFS State: correct if sigmoid(logit) > 0.5 matches binary target
            bfs_probs = mx.sigmoid(bfs_output)
            bfs_pred = (bfs_probs > 0.5).astype(mx.float32)
            bfs_state_correct += mx.sum(bfs_pred == target_bfs_state).item()
            bfs_state_total += len(target_bfs_state)
            
            # BFS Termination: correct if sigmoid(logit) > 0.5 matches target  
            bfs_term_prob = mx.sigmoid(termination_probs['bfs'])
            bfs_term_pred = (bfs_term_prob > 0.5).astype(mx.float32)
            bfs_termination_correct += int(bfs_term_pred.item() == termination_targets['bfs'].item())
            bfs_termination_total += 1

        # Update for next step
        previous_step_hidden_states = processed_embeddings

    # Calculate accuracies (avoiding division by zero)
    bf_distance_acc = bf_distance_correct / bf_distance_total if bf_distance_total > 0 else 0.0
    bf_predecessor_acc = bf_predecessor_correct / bf_predecessor_total if bf_predecessor_total > 0 else 0.0
    bfs_state_acc = bfs_state_correct / bfs_state_total if bfs_state_total > 0 else 0.0
    bf_termination_acc = bf_termination_correct / bf_termination_total if bf_termination_total > 0 else 0.0
    bfs_termination_acc = bfs_termination_correct / bfs_termination_total if bfs_termination_total > 0 else 0.0

    accuracies = mx.array([bf_distance_acc, bf_predecessor_acc, bfs_state_acc, bf_termination_acc, bfs_termination_acc])

    return accuracies