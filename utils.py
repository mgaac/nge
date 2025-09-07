import mlx.core as mx
import mlx.nn as nn
import numpy as np

def print_execution_details_v2(model, graph_data, embedding_dim):
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
    
    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])
    
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps)
    
    print(f"\n{'='*80}")
    print(f"EXECUTION DETAILS - Graph with {num_nodes} nodes")
    print(f"BF steps: {num_bf_steps}, BFS steps: {num_bfs_steps}")
    print(f"{'='*80}")
    
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
        node_algo_features = mx.concatenate([true_bfs_state, true_distance_bf]).reshape([-1, 2])
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])
        
        # Forward pass
        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)
        
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
            
            # BF Termination Loss
            bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
            
            # BF Termination Accuracy
            bf_term_pred = mx.sigmoid(termination_probs['bf']) > 0.5  # Use sigmoid + 0.5 threshold
            bf_term_correct = (bf_term_pred.astype(mx.float32) == termination_targets['bf']).item()
            bf_termination_correct_sum += bf_term_correct
            bf_termination_total_sum += 1
            
            print(f"\nBF DISTANCE:")
            print(f"  Loss: {bf_distance_loss.item():.6f}")
            print(f"  Accuracy: {bf_distance_correct}/{bf_distance_total} = {bf_distance_correct/bf_distance_total:.3f}")
            print(f"  Predictions (first 10): {bf_distance_predictions[:10].tolist()}")
            print(f"  Targets (first 10): {target_distance_bf[:10].tolist()}")
            
            print(f"\nBF PREDECESSOR:")
            print(f"  Loss: {bf_predecessor_loss.item():.6f}")
            print(f"  Accuracy: {bf_predecessor_correct}/{bf_predecessor_total} = {bf_predecessor_correct/max(bf_predecessor_total,1):.3f}")
            print(f"  Valid nodes: {bf_predecessor_total}")
            print(f"  Predictions (argmax, first 10): {pred_argmax[:10].tolist()}")
            print(f"  Targets (first 10): {target_predecessor_bf[:10].tolist()}")
            
            print(f"\nBF TERMINATION:")
            print(f"  Loss: {bf_termination_loss.item():.6f}")
            print(f"  Logit: {termination_probs['bf'].item():.4f}, Prob: {mx.sigmoid(termination_probs['bf']).item():.4f}")
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
            
            # BFS Termination Loss
            bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean', with_logits=True)
            
            # BFS Termination Accuracy
            bfs_term_pred = mx.sigmoid(termination_probs['bfs']) > 0.5
            bfs_term_correct = (bfs_term_pred.astype(mx.float32) == termination_targets['bfs']).item()
            bfs_termination_correct_sum += bfs_term_correct
            bfs_termination_total_sum += 1
            
            print(f"\nBFS STATE:")
            print(f"  Loss: {bfs_state_loss.item():.6f}")
            print(f"  Accuracy: {bfs_state_correct}/{bfs_state_total} = {bfs_state_correct/bfs_state_total:.3f}")
            print(f"  Predictions (first 10): {bfs_state_pred[:10].astype(mx.int32).tolist()}")
            print(f"  Targets (first 10): {target_bfs_state[:10].astype(mx.int32).tolist()}")
            
            print(f"\nBFS TERMINATION:")
            print(f"  Loss: {bfs_termination_loss.item():.6f}")
            print(f"  Logit: {termination_probs['bfs'].item():.4f}, Prob: {mx.sigmoid(termination_probs['bfs']).item():.4f}")
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
        
        print(f"\nSTEP SUMMARY:")
        print(f"  Total step loss: {total_step_loss.item():.6f}")
        print(f"  Hidden state norm: {mx.linalg.norm(processed_embeddings).item():.4f}")
        
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
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Effective steps for averaging: {effective_steps}")
    print(f"BF steps executed: {bf_steps}, BFS steps executed: {bfs_steps}")
    print(f"\nAVERAGE LOSSES:")
    print(f"  Total: {average_loss.item():.6f}")
    print(f"  BF Distance: {avg_aux_losses[0].item():.6f}")
    print(f"  BF Predecessor: {avg_aux_losses[1].item():.6f}")
    print(f"  BFS State: {avg_aux_losses[2].item():.6f}")
    print(f"  BF Termination: {avg_aux_losses[3].item():.6f}")
    print(f"  BFS Termination: {avg_aux_losses[4].item():.6f}")
    print(f"\nOVERALL ACCURACIES:")
    print(f"  BF Distance: {overall_bf_distance_acc:.3f} ({bf_distance_correct_sum}/{bf_distance_total_sum})")
    print(f"  BF Predecessor: {overall_bf_predecessor_acc:.3f} ({bf_predecessor_correct_sum}/{bf_predecessor_total_sum})")
    print(f"  BFS State: {overall_bfs_state_acc:.3f} ({bfs_state_correct_sum}/{bfs_state_total_sum})")
    print(f"  BF Termination: {overall_bf_termination_acc:.3f} ({bf_termination_correct_sum}/{bf_termination_total_sum})")
    print(f"  BFS Termination: {overall_bfs_termination_acc:.3f} ({bfs_termination_correct_sum}/{bfs_termination_total_sum})")
    print(f"{'='*80}\n")
    
    return average_loss, mx.linalg.norm(processed_embeddings)

def print_execution_details(model, graph_data, embedding_dim):
    accumulated_loss = mx.array(0.0)
    num_nodes = graph_data['num_nodes']

    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])

    num_bf_steps  = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps     = max(num_bf_steps, num_bfs_steps)

    steps_executed = 0

    for i in range(num_steps):
        # step-availability (need i and i+1)
        bf_sample_exists  = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        if not (bf_sample_exists or bfs_sample_exists):
            continue

        steps_executed += 1

        # targets for t -> t+1
        if bfs_sample_exists:
            true_bfs_state   = graph_data['bfs_state_targets'][i]
            target_bfs_state = graph_data['bfs_state_targets'][i + 1]
        else:
            true_bfs_state   = graph_data['bfs_state_targets'][-1]
            target_bfs_state = graph_data['bfs_state_targets'][-1]

        if bf_sample_exists:
            true_distance_bf    = graph_data['bf_distance_targets'][i]     # already normalized in [0,1]
            target_distance_bf  = graph_data['bf_distance_targets'][i + 1] # normalized
            target_predecessor  = graph_data['bf_predecessor_targets'][i + 1]  # int with -1 sentinel
        else:
            true_distance_bf    = graph_data['bf_distance_targets'][-1]
            target_distance_bf  = graph_data['bf_distance_targets'][-1]
            target_predecessor  = graph_data['bf_predecessor_targets'][-1]

        # termination targets (as floats 0/1)
        is_last_bf_step  = (i + 1) == (num_bf_steps  - 1)
        is_last_bfs_step = (i + 1) == (num_bfs_steps - 1)
        termination_targets = {
            'bf' : mx.array(1.0 if is_last_bf_step  else 0.0),
            'bfs': mx.array(1.0 if is_last_bfs_step else 0.0),
        }

        # model input (distances already normalized in dataset)
        node_algo_features = mx.concatenate([true_bfs_state, true_distance_bf]).reshape([-1, 2])
        input_embeddings   = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input        = (input_embeddings, graph_data['edge_matrix'])

        # forward
        bfs_output, bf_output, termination_logits, processed_embeddings = model(model_input)

        # losses
        if bf_sample_exists:
            bf_distance_pred, bf_pred_logits = bf_output  # shapes: [N], [N,N]

            # distance MSE on normalized values
            bf_distance_loss = nn.losses.mse_loss(bf_distance_pred, target_distance_bf, reduction='mean')

            # masked CE: ignore -1 (undefined)
            valid_mask      = (target_predecessor != -1)
            safe_targets    = mx.where(valid_mask, target_predecessor, mx.zeros_like(target_predecessor))
            ce_per_node     = nn.losses.cross_entropy(bf_pred_logits, safe_targets, reduction='none')
            denom           = mx.maximum(valid_mask.astype(mx.float32).sum(), mx.array(1.0))
            bf_predecessor_loss = (ce_per_node * valid_mask.astype(mx.float32)).sum() / denom

            # termination: logits in, with_logits=True
            bf_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bf'], termination_targets['bf'], reduction='mean', with_logits=True
            )
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)

        if bfs_sample_exists:
            bfs_state_loss = nn.losses.binary_cross_entropy(
                bfs_output, target_bfs_state, reduction='mean', with_logits=True
            )
            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bfs'], termination_targets['bfs'], reduction='mean', with_logits=True
            )
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)

        total_step_loss = bf_distance_loss + bf_predecessor_loss + bfs_state_loss + bf_termination_loss + bfs_termination_loss

        # update state
        previous_step_hidden_states = processed_embeddings
        accumulated_loss += total_step_loss

        # ---- diagnostics ----
        print(f"\n=== Step {steps_executed} ===")
        print(f"Total Loss: {total_step_loss.item():.6f}")
        print("Loss Breakdown:")
        print(f"  BF Distance:     {bf_distance_loss.item():.6f}")
        print(f"  BF Predecessor:  {bf_predecessor_loss.item():.6f}")
        print(f"  BFS State:       {bfs_state_loss.item():.6f}")
        print(f"  BF Termination:  {bf_termination_loss.item():.6f}")
        print(f"  BFS Termination: {bfs_termination_loss.item():.6f}")

        if bf_sample_exists:
            print("\nBF Distance:")
            print(f"  Pred norm={mx.linalg.norm(bf_distance_pred).item():.6f}, std={mx.std(bf_distance_pred).item():.6f}")
            print(f"  Pred: {np.array(bf_distance_pred).round(4)}")
            print(f"  Targ: {np.array(target_distance_bf).round(4)}")

            print("\nBF Predecessor:")
            print(f"  Logits: norm={mx.linalg.norm(bf_pred_logits).item():.6f}, std={mx.std(bf_pred_logits).item():.6f}")
            argmax_pred = np.argmax(np.array(bf_pred_logits), axis=-1)
            print(f"  Pred (argmax): {argmax_pred}")
            print(f"  Targ: {np.array(target_predecessor)}")

            print("\nBF Termination:")
            z = termination_logits['bf']
            p = mx.sigmoid(z)
            print(f"  Logit: {z.item():.4f}  Prob: {p.item():.4f}  Targ: {termination_targets['bf'].item():.4f}")

        if bfs_sample_exists:
            print("\nBFS State:")
            print(f"  Logits: norm={mx.linalg.norm(bfs_output).item():.6f}, std={mx.std(bfs_output).item():.6f}")
            print(f"  Pred>0: {(np.array(bfs_output) > 0).astype(int)}")
            print(f"  Targ:   {np.array(target_bfs_state).astype(int)}")

            print("\nBFS Termination:")
            z = termination_logits['bfs']
            p = mx.sigmoid(z)
            print(f"  Logit: {z.item():.4f}  Prob: {p.item():.4f}  Targ: {termination_targets['bfs'].item():.4f}")

        print(f"\nHidden State: norm={mx.linalg.norm(processed_embeddings).item():.6f}, std={mx.std(processed_embeddings).item():.6f}")

    avg_loss = accumulated_loss / steps_executed if steps_executed > 0 else mx.array(0.0)

    print(f"\n=== Summary ===")
    print(f"Steps executed: {steps_executed}")
    print(f"Average loss:   {avg_loss.item():.6f}")

    return avg_loss, mx.linalg.norm(processed_embeddings)

# Keep original function name for compatibility
def print_execution_details(model, graph_data, embedding_dim):
    """Wrapper that calls the new v2 implementation."""
    return print_execution_details_v2(model, graph_data, embedding_dim)

def calculate_losses_and_accuracies_v2(model, graph_data, embedding_dim=128):
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
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])
    
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps)
    
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
        node_algo_features = mx.concatenate([true_bfs_state, true_distance_bf]).reshape([-1, 2])
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])
        
        # Forward pass
        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)
        
        # Compute losses and accuracies
        if bf_sample_exists:
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
            per_node_ce = nn.losses.cross_entropy(bf_predecessor_predictions, safe_targets, reduction='none')
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            
            # BF Predecessor Accuracy - only count valid nodes
            pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            pred_correct = mx.sum((pred_argmax == target_predecessor_bf) & valid_mask).item()
            pred_total = mx.sum(valid_mask).item()
            bf_predecessor_correct_sum += pred_correct
            bf_predecessor_total_sum += pred_total
            
            # BF Termination Loss - exactly as in train.py
            bf_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bf'], termination_targets['bf'], reduction='mean')
            
            # BF Termination Accuracy - since loss uses raw logits, we apply sigmoid for accuracy
            bf_term_prob = mx.sigmoid(termination_probs['bf'])
            bf_term_pred = (bf_term_prob > 0.5).astype(mx.float32)
            bf_term_correct = (bf_term_pred == termination_targets['bf']).item()
            bf_termination_correct_sum += bf_term_correct
            bf_termination_total_sum += 1
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)
            
        if bfs_sample_exists:
            # BFS State Loss - exactly as in train.py
            bfs_state_loss = nn.losses.binary_cross_entropy(bfs_output, target_bfs_state, reduction='mean', with_logits=True)
            
            # BFS State Accuracy - apply sigmoid since loss uses logits
            bfs_state_probs = mx.sigmoid(bfs_output)
            bfs_state_pred = (bfs_state_probs > 0.5).astype(mx.float32)
            bfs_correct = mx.sum(bfs_state_pred == target_bfs_state).item()
            bfs_state_correct_sum += bfs_correct
            bfs_state_total_sum += num_nodes
            
            # BFS Termination Loss - exactly as in train.py  
            bfs_termination_loss = nn.losses.binary_cross_entropy(termination_probs['bfs'], termination_targets['bfs'], reduction='mean', with_logits=True)
            
            # BFS Termination Accuracy
            bfs_term_prob = mx.sigmoid(termination_probs['bfs'])
            bfs_term_pred = (bfs_term_prob > 0.5).astype(mx.float32)
            bfs_term_correct = (bfs_term_pred == termination_targets['bfs']).item()
            bfs_termination_correct_sum += bfs_term_correct
            bfs_termination_total_sum += 1
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)
            
        # Accumulate losses - exactly as in train.py
        raw_losses = mx.array([bf_distance_loss, bf_predecessor_loss, bfs_state_loss, bf_termination_loss, bfs_termination_loss])
        total_step_loss = mx.sum(raw_losses)
        accumulated_loss += total_step_loss
        accumulated_aux_losses += raw_losses
        
        # Update hidden states for next step
        previous_step_hidden_states = processed_embeddings
        
    # Calculate averages - exactly as in train.py
    bf_steps = max(num_bf_steps - 1, 0)
    bfs_steps = max(num_bfs_steps - 1, 0)
    effective_steps = max(bf_steps, bfs_steps, 1)
    
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

def calculate_losses_and_accuracies(model, graph_data, embedding_dim=128):
    """
    Returns: (aux_losses[5], total_loss, accuracies[5])
    aux_losses = [bf_dist, bf_pred, bfs_state, bf_term, bfs_term]  (averaged over *their* executed steps)
    accuracies = per-task accuracies with proper masking and thresholds.
    """
    accumulated_loss = mx.array(0.0)

    # accuracy counters
    bf_distance_correct = 0
    bf_predecessor_correct = 0
    bfs_state_correct = 0
    bf_termination_correct = 0
    bfs_termination_correct = 0

    # totals for denominators
    bf_distance_total = 0
    bf_predecessor_total = 0
    bfs_state_total = 0
    bf_termination_total = 0
    bfs_termination_total = 0

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])

    num_bf_steps  = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps     = max(num_bf_steps, num_bfs_steps)

    # loss accumulators
    bf_distance_loss_acc     = mx.array(0.0)
    bf_predecessor_loss_acc  = mx.array(0.0)
    bfs_state_loss_acc       = mx.array(0.0)
    bf_termination_loss_acc  = mx.array(0.0)
    bfs_termination_loss_acc = mx.array(0.0)

    steps_executed   = 0
    bf_steps_exec    = 0
    bfs_steps_exec   = 0

    for i in range(num_steps):
        bf_sample_exists  = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        if not (bf_sample_exists or bfs_sample_exists):
            continue

        steps_executed += 1
        if bf_sample_exists:  bf_steps_exec  += 1
        if bfs_sample_exists: bfs_steps_exec += 1

        if bfs_sample_exists:
            true_bfs_state   = graph_data['bfs_state_targets'][i]
            target_bfs_state = graph_data['bfs_state_targets'][i + 1]
        else:
            true_bfs_state   = graph_data['bfs_state_targets'][-1]
            target_bfs_state = graph_data['bfs_state_targets'][-1]

        if bf_sample_exists:
            true_distance_bf   = graph_data['bf_distance_targets'][i]       # normalized
            target_distance_bf = graph_data['bf_distance_targets'][i + 1]   # normalized
            target_predecessor = graph_data['bf_predecessor_targets'][i + 1]  # -1 = undefined
        else:
            true_distance_bf   = graph_data['bf_distance_targets'][-1]
            target_distance_bf = graph_data['bf_distance_targets'][-1]
            target_predecessor = graph_data['bf_predecessor_targets'][-1]

        is_last_bf_step  = (i + 1) == (num_bf_steps  - 1)
        is_last_bfs_step = (i + 1) == (num_bfs_steps - 1)
        termination_targets = {
            'bf' : mx.array(1.0 if is_last_bf_step  else 0.0),
            'bfs': mx.array(1.0 if is_last_bfs_step else 0.0),
        }

        node_algo_features = mx.concatenate([true_bfs_state, true_distance_bf]).reshape([-1, 2])
        input_embeddings   = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input        = (input_embeddings, graph_data['edge_matrix'])

        bfs_output, bf_output, termination_logits, processed_embeddings = model(model_input)

        # ----- losses + accuracies -----
        if bf_sample_exists:
            bf_distance_pred, bf_pred_logits = bf_output

            # distance loss (normalized)
            bf_dist_loss = nn.losses.mse_loss(bf_distance_pred, target_distance_bf, reduction='mean')
            bf_distance_loss_acc += bf_dist_loss

            # distance accuracy (tolerance on normalized scale)
            err = mx.abs(bf_distance_pred - target_distance_bf)
            tol = mx.array(0.1, dtype=err.dtype)  # 0.1 on [0,1] scale
            correct_mask = (err <= tol)
            bf_distance_correct += mx.sum(correct_mask).item()
            bf_distance_total   += len(target_distance_bf)

            # predecessor masked CE
            valid_mask   = (target_predecessor != -1)
            safe_targets = mx.where(valid_mask, target_predecessor, mx.zeros_like(target_predecessor))
            ce_per_node  = nn.losses.cross_entropy(bf_pred_logits, safe_targets, reduction='none')
            denom        = mx.maximum(valid_mask.astype(mx.float32).sum(), mx.array(1.0))
            bf_pred_loss = (ce_per_node * valid_mask.astype(mx.float32)).sum() / denom
            bf_predecessor_loss_acc += bf_pred_loss

            # predecessor accuracy (only valid)
            pred_argmax = mx.argmax(bf_pred_logits, axis=-1)
            bf_predecessor_correct += mx.sum((pred_argmax == target_predecessor) * valid_mask).item()
            bf_predecessor_total   += int(mx.sum(valid_mask).item())

            # termination (logits in, with_logits=True; threshold at 0 for accuracy)
            bf_term_loss = nn.losses.binary_cross_entropy(
                termination_logits['bf'], termination_targets['bf'], reduction='mean', with_logits=True
            )
            bf_termination_loss_acc += bf_term_loss
            bf_term_pred = (termination_logits['bf'] > 0.0).astype(mx.float32)
            bf_termination_correct += int(bf_term_pred.item() == termination_targets['bf'].item())
            bf_termination_total   += 1
        else:
            bf_dist_loss = mx.array(0.0)
            bf_pred_loss = mx.array(0.0)
            bf_term_loss = mx.array(0.0)

        if bfs_sample_exists:
            bfs_state_loss = nn.losses.binary_cross_entropy(
                bfs_output, target_bfs_state, reduction='mean', with_logits=True
            )
            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bfs'], termination_targets['bfs'], reduction='mean', with_logits=True
            )
            bfs_state_loss_acc       += bfs_state_loss
            bfs_termination_loss_acc += bfs_termination_loss

            # accuracies: logits threshold at 0
            bfs_state_pred = (bfs_output > 0.0).astype(mx.float32)
            bfs_state_correct += mx.sum(bfs_state_pred == target_bfs_state).item()
            bfs_state_total   += len(target_bfs_state)

            bfs_term_pred = (termination_logits['bfs'] > 0.0).astype(mx.float32)
            bfs_termination_correct += int(bfs_term_pred.item() == termination_targets['bfs'].item())
            bfs_termination_total   += 1
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)

        total_step_loss = bf_dist_loss + bf_pred_loss + bfs_state_loss + bf_term_loss + bfs_termination_loss
        accumulated_loss += total_step_loss

        previous_step_hidden_states = processed_embeddings

    # averages per task over their executed steps (avoid div by 0)
    if steps_executed > 0:
        avg_total_loss = accumulated_loss / steps_executed
        avg_bf_dist    = bf_distance_loss_acc     / max(bf_steps_exec, 1)
        avg_bf_pred    = bf_predecessor_loss_acc  / max(bf_steps_exec, 1)
        avg_bfs_state  = bfs_state_loss_acc       / max(bfs_steps_exec, 1)
        avg_bf_term    = bf_termination_loss_acc  / max(bf_steps_exec, 1)
        avg_bfs_term   = bfs_termination_loss_acc / max(bfs_steps_exec, 1)
    else:
        avg_total_loss = mx.array(0.0)
        avg_bf_dist = avg_bf_pred = avg_bfs_state = avg_bf_term = avg_bfs_term = mx.array(0.0)

    # accuracies with masking
    bf_distance_acc   = bf_distance_correct   / bf_distance_total   if bf_distance_total   > 0 else 0.0
    bf_predecessor_acc= bf_predecessor_correct/ bf_predecessor_total if bf_predecessor_total> 0 else 0.0
    bfs_state_acc     = bfs_state_correct     / bfs_state_total     if bfs_state_total     > 0 else 0.0
    bf_termination_acc= bf_termination_correct/ bf_termination_total if bf_termination_total> 0 else 0.0
    bfs_termination_acc= bfs_termination_correct/ bfs_termination_total if bfs_termination_total> 0 else 0.0

    aux_losses  = mx.array([avg_bf_dist, avg_bf_pred, avg_bfs_state, avg_bf_term, avg_bfs_term])
    accuracies  = mx.array([bf_distance_acc, bf_predecessor_acc, bfs_state_acc, bf_termination_acc, bfs_termination_acc])

    return aux_losses, avg_total_loss, accuracies

# Keep original function name for compatibility
def calculate_losses_and_accuracies(model, graph_data, embedding_dim=128):
    """Wrapper that calls the new v2 implementation."""
    return calculate_losses_and_accuracies_v2(model, graph_data, embedding_dim)

def calculate_accuracies_v2(model, graph_data, embedding_dim=128):
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
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])
    
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps)
    
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
        node_algo_features = mx.concatenate([true_bfs_state, true_distance_bf]).reshape([-1, 2])
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])
        
        # Forward pass
        bfs_output, bf_output, termination_probs, processed_embeddings = model(model_input)
        
        # Calculate accuracies
        if bf_sample_exists:
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
            bf_term_prob = mx.sigmoid(termination_probs['bf'])
            bf_term_pred = (bf_term_prob > 0.5).astype(mx.float32)
            bf_term_correct = (bf_term_pred == termination_targets['bf']).item()
            bf_termination_correct_sum += bf_term_correct
            bf_termination_total_sum += 1
            
        if bfs_sample_exists:
            # BFS State Accuracy
            bfs_state_probs = mx.sigmoid(bfs_output)
            bfs_state_pred = (bfs_state_probs > 0.5).astype(mx.float32)
            bfs_correct = mx.sum(bfs_state_pred == target_bfs_state).item()
            bfs_state_correct_sum += bfs_correct
            bfs_state_total_sum += num_nodes
            
            # BFS Termination Accuracy
            bfs_term_prob = mx.sigmoid(termination_probs['bfs'])
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

# Keep original function name for compatibility
def calculate_accuracies(model, graph_data, embedding_dim=128):
    """Wrapper that calls the new v2 implementation."""
    return calculate_accuracies_v2(model, graph_data, embedding_dim)