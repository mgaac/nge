import mlx.core as mx
import mlx.nn as nn
import numpy as np

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

def calculate_accuracies(model, graph_data, embedding_dim=128):
    bf_distance_correct = bf_predecessor_correct = 0
    bfs_state_correct = bf_termination_correct = bfs_termination_correct = 0
    bf_distance_total = bf_predecessor_total = bfs_state_total = bf_termination_total = bfs_termination_total = 0

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, embedding_dim])

    num_bf_steps  = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_steps     = max(num_bf_steps, num_bfs_steps)

    for i in range(num_steps):
        bf_sample_exists  = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        if not (bf_sample_exists or bfs_sample_exists):
            continue

        if bfs_sample_exists:
            true_bfs_state   = graph_data['bfs_state_targets'][i]
            target_bfs_state = graph_data['bfs_state_targets'][i + 1]
        else:
            true_bfs_state   = graph_data['bfs_state_targets'][-1]
            target_bfs_state = graph_data['bfs_state_targets'][-1]

        if bf_sample_exists:
            true_distance_bf   = graph_data['bf_distance_targets'][i]       # normalized
            target_distance_bf = graph_data['bf_distance_targets'][i + 1]   # normalized
            target_predecessor = graph_data['bf_predecessor_targets'][i + 1]
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

        if bf_sample_exists:
            bf_distance_pred, bf_pred_logits = bf_output

            # distance accuracy on normalized scale (tolerance 0.1)
            err = mx.abs(bf_distance_pred - target_distance_bf)
            tol = mx.array(0.1, dtype=err.dtype)
            bf_distance_correct += mx.sum(err <= tol).item()
            bf_distance_total   += len(target_distance_bf)

            # predecessor accuracy with masking
            valid_mask = (target_predecessor != -1)
            argmax_pred = mx.argmax(bf_pred_logits, axis=-1)
            bf_predecessor_correct += mx.sum((argmax_pred == target_predecessor) * valid_mask).item()
            bf_predecessor_total   += int(mx.sum(valid_mask).item())

            # termination accuracy: threshold at 0 on logits
            bf_term_pred = (termination_logits['bf'] > 0.0).astype(mx.float32)
            bf_termination_correct += int(bf_term_pred.item() == termination_targets['bf'].item())
            bf_termination_total   += 1

        if bfs_sample_exists:
            # BFS state: logits threshold at 0
            bfs_pred = (bfs_output > 0.0).astype(mx.float32)
            bfs_state_correct += mx.sum(bfs_pred == target_bfs_state).item()
            bfs_state_total   += len(target_bfs_state)

            bfs_term_pred = (termination_logits['bfs'] > 0.0).astype(mx.float32)
            bfs_termination_correct += int(bfs_term_pred.item() == termination_targets['bfs'].item())
            bfs_termination_total   += 1

        previous_step_hidden_states = processed_embeddings

    bf_distance_acc     = bf_distance_correct     / bf_distance_total     if bf_distance_total     > 0 else 0.0
    bf_predecessor_acc  = bf_predecessor_correct  / bf_predecessor_total  if bf_predecessor_total  > 0 else 0.0
    bfs_state_acc       = bfs_state_correct       / bfs_state_total       if bfs_state_total       > 0 else 0.0
    bf_termination_acc  = bf_termination_correct  / bf_termination_total  if bf_termination_total  > 0 else 0.0
    bfs_termination_acc = bfs_termination_correct / bfs_termination_total if bfs_termination_total > 0 else 0.0

    return mx.array([bf_distance_acc, bf_predecessor_acc, bfs_state_acc, bf_termination_acc, bfs_termination_acc])