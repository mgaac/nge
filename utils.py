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
            print(f"  Logits: norm={mx.linalg.norm(termination_probs['bf']).item():.6f}")
            print(f"  Pred: {np.array(termination_probs['bf']).item():.4f}")
            print(f"  Targ: {np.array(termination_targets['bf']).item():.4f}")
            
        if bfs_sample_exists:
            print("\nBFS State:")
            print(f"  Logits: norm={mx.linalg.norm(bfs_output).item():.6f}, std={mx.std(bfs_output).item():.6f}")
            print(f"  Pred: {(np.array(bfs_output) > 0).astype(int)}")
            print(f"  Targ: {np.array(target_bfs_state).astype(int)}")
            
            print("\nBFS Termination:")
            print(f"  Logits: norm={mx.linalg.norm(termination_probs['bfs']).item():.6f}")
            print(f"  Pred: {np.array(termination_probs['bfs']).item():.4f}")
            print(f"  Targ: {np.array(termination_targets['bfs']).item():.4f}")
            
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