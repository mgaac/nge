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
from src.utils.task_specs import (
    ALGORITHMS,
    METRIC_NAMES,
    build_node_algo_features,
    effective_step_count,
    execution_step_counts,
    metric_counters,
    metric_mask,
)


def _normalize_selected_tasks(selected_tasks=None):
    if selected_tasks is None:
        return {algorithm: True for algorithm in ALGORITHMS}
    return {
        algorithm: bool(selected_tasks.get(algorithm, False)) for algorithm in ALGORITHMS
    }


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
    """Print per-step execution details for all algorithm heads."""
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([len(METRIC_NAMES)])

    metric_counts = {
        metric_name: {"correct": 0, "total": 0} for metric_name in METRIC_NAMES
    }
    accumulated_norms = {
        "bf_distance": mx.array(0.0),
        "bf_predecessor": mx.array(0.0),
        "bfs_state": mx.array(0.0),
        "prim_state": mx.array(0.0),
        "prim_key": mx.array(0.0),
        "prim_predecessor": mx.array(0.0),
        "bf_termination": mx.array(0.0),
        "bfs_termination": mx.array(0.0),
        "prim_termination": mx.array(0.0),
        "hidden_state": mx.array(0.0),
    }
    norm_steps = {name: 0 for name in accumulated_norms}

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])

    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_prim_steps = len(graph_data['prim_key_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps, num_prim_steps)
    step_counts = execution_step_counts(graph_data)

    print(f"\n{'=' * 80}")
    print(f"EXECUTION DETAILS - Graph with {num_nodes} nodes")
    print(
        f"BF steps: {num_bf_steps}, BFS steps: {num_bfs_steps}, Prim steps: {num_prim_steps}"
    )
    print(f"{'=' * 80}")

    termination_settings = resolve_termination_settings(termination_cfg)
    previous_distance_latent = None

    for i in range(num_steps):
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        prim_sample_exists = (i + 1) < num_prim_steps

        if not (bf_sample_exists or bfs_sample_exists or prim_sample_exists):
            continue

        print(f"\n{'=' * 60}")
        print(f"STEP {i} -> {i + 1}")
        print(
            "Samples: "
            f"BF={bf_sample_exists}, BFS={bfs_sample_exists}, Prim={prim_sample_exists}"
        )
        print(f"{'=' * 60}")

        if bfs_sample_exists:
            true_bfs_state = graph_data['bfs_state_targets'][i]
            target_bfs_state = graph_data['bfs_state_targets'][i + 1]
        else:
            true_bfs_state = graph_data['bfs_state_targets'][-1]
            target_bfs_state = graph_data['bfs_state_targets'][-1]

        if bf_sample_exists:
            true_distance_bf = graph_data['bf_distance_targets'][i]
            target_distance_bf = graph_data['bf_distance_targets'][i + 1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][i + 1]
        else:
            true_distance_bf = graph_data['bf_distance_targets'][-1]
            target_distance_bf = graph_data['bf_distance_targets'][-1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][-1]

        if prim_sample_exists:
            true_prim_state = graph_data['prim_state_targets'][i]
            target_prim_state = graph_data['prim_state_targets'][i + 1]
            true_prim_key = graph_data['prim_key_targets'][i]
            target_prim_key = graph_data['prim_key_targets'][i + 1]
            target_prim_predecessor = graph_data['prim_predecessor_targets'][i + 1]
        else:
            true_prim_state = graph_data['prim_state_targets'][-1]
            target_prim_state = graph_data['prim_state_targets'][-1]
            true_prim_key = graph_data['prim_key_targets'][-1]
            target_prim_key = graph_data['prim_key_targets'][-1]
            target_prim_predecessor = graph_data['prim_predecessor_targets'][-1]

        termination_targets = {
            'bf': mx.array(1.0 if (i + 1) == (num_bf_steps - 1) else 0.0),
            'bfs': mx.array(1.0 if (i + 1) == (num_bfs_steps - 1) else 0.0),
            'prim': mx.array(1.0 if (i + 1) == (num_prim_steps - 1) else 0.0),
        }

        node_algo_features = build_node_algo_features(
            true_bfs_state,
            true_distance_bf,
            true_prim_state,
            true_prim_key,
        )
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])

        need_aux = needs_aux_latents(termination_settings)
        if need_aux:
            bfs_output, bf_output, prim_output, termination_probs, processed_embeddings, aux = model(
                model_input, return_latents=True
            )
        else:
            bfs_output, bf_output, prim_output, termination_probs, processed_embeddings = model(model_input)
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

        step_losses = []

        if bf_sample_exists:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            bf_distance_loss = nn.losses.mse_loss(
                bf_distance_predictions, target_distance_bf, reduction='mean'
            )
            distance_errors = mx.abs(bf_distance_predictions - target_distance_bf)
            bf_distance_correct = int(mx.sum(distance_errors <= 0.1).item())
            metric_counts['bf_distance']["correct"] += bf_distance_correct
            metric_counts['bf_distance']["total"] += num_nodes

            valid_mask = (target_predecessor_bf != -1)
            safe_targets = mx.where(valid_mask, target_predecessor_bf, mx.zeros_like(target_predecessor_bf))
            per_node_ce = nn.losses.cross_entropy(
                bf_predecessor_predictions,
                safe_targets,
                reduction='none',
            )
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            bf_pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            bf_pred_total = int(mx.sum(valid_mask).item())
            bf_pred_correct = int(mx.sum((bf_pred_argmax == target_predecessor_bf) & valid_mask).item())
            metric_counts['bf_predecessor']["correct"] += bf_pred_correct
            metric_counts['bf_predecessor']["total"] += bf_pred_total

            bf_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bf'],
                termination_targets['bf'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bf_termination_loss = mx.array(0.0)
            bf_term_correct = int(
                ((mx.sigmoid(termination_logits['bf']) > 0.5).astype(mx.float32) == termination_targets['bf']).item()
            )
            metric_counts['bf_termination']["correct"] += bf_term_correct
            metric_counts['bf_termination']["total"] += 1

            accumulated_norms['bf_distance'] += mx.linalg.norm(bf_distance_predictions)
            accumulated_norms['bf_predecessor'] += mx.linalg.norm(bf_predecessor_predictions)
            accumulated_norms['bf_termination'] += mx.linalg.norm(termination_logits['bf'])
            norm_steps['bf_distance'] += 1
            norm_steps['bf_predecessor'] += 1
            norm_steps['bf_termination'] += 1

            print("\nBF DISTANCE:")
            print(f"  Loss: {bf_distance_loss.item():.6f}")
            print(f"  Accuracy: {bf_distance_correct}/{num_nodes} = {bf_distance_correct / max(num_nodes, 1):.3f}")
            print(f"  Predictions (first 10): {bf_distance_predictions[:10].tolist()}")
            print(f"  Targets (first 10): {target_distance_bf[:10].tolist()}")

            print("\nBF PREDECESSOR:")
            print(f"  Loss: {bf_predecessor_loss.item():.6f}")
            print(f"  Accuracy: {bf_pred_correct}/{max(bf_pred_total, 1)} = {bf_pred_correct / max(bf_pred_total, 1):.3f}")
            print(f"  Predictions (argmax, first 10): {bf_pred_argmax[:10].tolist()}")
            print(f"  Targets (first 10): {target_predecessor_bf[:10].tolist()}")

            print("\nBF TERMINATION:")
            print(f"  Loss: {bf_termination_loss.item():.6f}")
            print(
                f"  Logit: {termination_logits['bf'].item():.4f}, "
                f"Prob: {mx.sigmoid(termination_logits['bf']).item():.4f}"
            )
            print(f"  Target: {termination_targets['bf'].item()}, Correct: {bf_term_correct}")

            step_losses.extend([bf_distance_loss, bf_predecessor_loss])
        else:
            bf_termination_loss = mx.array(0.0)

        if bfs_sample_exists:
            bfs_state_loss = nn.losses.binary_cross_entropy(
                bfs_output,
                target_bfs_state,
                reduction='mean',
                with_logits=True,
            )
            bfs_state_pred = (mx.sigmoid(bfs_output) > 0.5).astype(mx.float32)
            bfs_correct = int(mx.sum(bfs_state_pred == target_bfs_state).item())
            metric_counts['bfs_state']["correct"] += bfs_correct
            metric_counts['bfs_state']["total"] += num_nodes

            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bfs'],
                termination_targets['bfs'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bfs_termination_loss = mx.array(0.0)
            bfs_term_correct = int(
                ((mx.sigmoid(termination_logits['bfs']) > 0.5).astype(mx.float32) == termination_targets['bfs']).item()
            )
            metric_counts['bfs_termination']["correct"] += bfs_term_correct
            metric_counts['bfs_termination']["total"] += 1

            accumulated_norms['bfs_state'] += mx.linalg.norm(bfs_output)
            accumulated_norms['bfs_termination'] += mx.linalg.norm(termination_logits['bfs'])
            norm_steps['bfs_state'] += 1
            norm_steps['bfs_termination'] += 1

            print("\nBFS STATE:")
            print(f"  Loss: {bfs_state_loss.item():.6f}")
            print(f"  Accuracy: {bfs_correct}/{num_nodes} = {bfs_correct / max(num_nodes, 1):.3f}")
            print(f"  Predictions (first 10): {bfs_state_pred[:10].astype(mx.int32).tolist()}")
            print(f"  Targets (first 10): {target_bfs_state[:10].astype(mx.int32).tolist()}")

            print("\nBFS TERMINATION:")
            print(f"  Loss: {bfs_termination_loss.item():.6f}")
            print(
                f"  Logit: {termination_logits['bfs'].item():.4f}, "
                f"Prob: {mx.sigmoid(termination_logits['bfs']).item():.4f}"
            )
            print(f"  Target: {termination_targets['bfs'].item()}, Correct: {bfs_term_correct}")

            step_losses.extend([bfs_state_loss])
        else:
            bfs_termination_loss = mx.array(0.0)

        if prim_sample_exists:
            prim_state_predictions, prim_key_predictions, prim_predecessor_predictions = prim_output
            prim_state_loss = nn.losses.binary_cross_entropy(
                prim_state_predictions,
                target_prim_state,
                reduction='mean',
                with_logits=True,
            )
            prim_state_pred = (mx.sigmoid(prim_state_predictions) > 0.5).astype(mx.float32)
            prim_state_correct = int(mx.sum(prim_state_pred == target_prim_state).item())
            metric_counts['prim_state']["correct"] += prim_state_correct
            metric_counts['prim_state']["total"] += num_nodes

            prim_key_loss = nn.losses.mse_loss(
                prim_key_predictions,
                target_prim_key,
                reduction='mean',
            )
            prim_key_correct = int(mx.sum(mx.abs(prim_key_predictions - target_prim_key) <= 0.1).item())
            metric_counts['prim_key']["correct"] += prim_key_correct
            metric_counts['prim_key']["total"] += num_nodes

            valid_mask = (target_prim_predecessor != -1)
            safe_targets = mx.where(valid_mask, target_prim_predecessor, mx.zeros_like(target_prim_predecessor))
            per_node_ce = nn.losses.cross_entropy(
                prim_predecessor_predictions,
                safe_targets,
                reduction='none',
            )
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            prim_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            prim_pred_argmax = mx.argmax(prim_predecessor_predictions, axis=-1)
            prim_pred_total = int(mx.sum(valid_mask).item())
            prim_pred_correct = int(mx.sum((prim_pred_argmax == target_prim_predecessor) & valid_mask).item())
            metric_counts['prim_predecessor']["correct"] += prim_pred_correct
            metric_counts['prim_predecessor']["total"] += prim_pred_total

            prim_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['prim'],
                termination_targets['prim'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                prim_termination_loss = mx.array(0.0)
            prim_term_correct = int(
                ((mx.sigmoid(termination_logits['prim']) > 0.5).astype(mx.float32) == termination_targets['prim']).item()
            )
            metric_counts['prim_termination']["correct"] += prim_term_correct
            metric_counts['prim_termination']["total"] += 1

            accumulated_norms['prim_state'] += mx.linalg.norm(prim_state_predictions)
            accumulated_norms['prim_key'] += mx.linalg.norm(prim_key_predictions)
            accumulated_norms['prim_predecessor'] += mx.linalg.norm(prim_predecessor_predictions)
            accumulated_norms['prim_termination'] += mx.linalg.norm(termination_logits['prim'])
            norm_steps['prim_state'] += 1
            norm_steps['prim_key'] += 1
            norm_steps['prim_predecessor'] += 1
            norm_steps['prim_termination'] += 1

            print("\nPRIM STATE:")
            print(f"  Loss: {prim_state_loss.item():.6f}")
            print(f"  Accuracy: {prim_state_correct}/{num_nodes} = {prim_state_correct / max(num_nodes, 1):.3f}")
            print(f"  Predictions (first 10): {prim_state_pred[:10].astype(mx.int32).tolist()}")
            print(f"  Targets (first 10): {target_prim_state[:10].astype(mx.int32).tolist()}")

            print("\nPRIM KEY:")
            print(f"  Loss: {prim_key_loss.item():.6f}")
            print(f"  Accuracy: {prim_key_correct}/{num_nodes} = {prim_key_correct / max(num_nodes, 1):.3f}")
            print(f"  Predictions (first 10): {prim_key_predictions[:10].tolist()}")
            print(f"  Targets (first 10): {target_prim_key[:10].tolist()}")

            print("\nPRIM PREDECESSOR:")
            print(f"  Loss: {prim_predecessor_loss.item():.6f}")
            print(f"  Accuracy: {prim_pred_correct}/{max(prim_pred_total, 1)} = {prim_pred_correct / max(prim_pred_total, 1):.3f}")
            print(f"  Predictions (argmax, first 10): {prim_pred_argmax[:10].tolist()}")
            print(f"  Targets (first 10): {target_prim_predecessor[:10].tolist()}")

            print("\nPRIM TERMINATION:")
            print(f"  Loss: {prim_termination_loss.item():.6f}")
            print(
                f"  Logit: {termination_logits['prim'].item():.4f}, "
                f"Prob: {mx.sigmoid(termination_logits['prim']).item():.4f}"
            )
            print(f"  Target: {termination_targets['prim'].item()}, Correct: {prim_term_correct}")

            step_losses.extend([prim_state_loss, prim_key_loss, prim_predecessor_loss])
        else:
            prim_termination_loss = mx.array(0.0)

        raw_losses = mx.array(
            [
                bf_distance_loss if bf_sample_exists else mx.array(0.0),
                bf_predecessor_loss if bf_sample_exists else mx.array(0.0),
                bfs_state_loss if bfs_sample_exists else mx.array(0.0),
                prim_state_loss if prim_sample_exists else mx.array(0.0),
                prim_key_loss if prim_sample_exists else mx.array(0.0),
                prim_predecessor_loss if prim_sample_exists else mx.array(0.0),
                bf_termination_loss if bf_sample_exists else mx.array(0.0),
                bfs_termination_loss if bfs_sample_exists else mx.array(0.0),
                prim_termination_loss if prim_sample_exists else mx.array(0.0),
            ]
        )
        accumulated_aux_losses += raw_losses
        total_step_loss = mx.sum(raw_losses)
        accumulated_loss += total_step_loss

        previous_step_hidden_states = processed_embeddings
        accumulated_norms['hidden_state'] += mx.linalg.norm(processed_embeddings)
        norm_steps['hidden_state'] += 1

        print("\nSTEP SUMMARY:")
        print(f"  Total step loss: {total_step_loss.item():.6f}")
        print(f"  Hidden state norm: {mx.linalg.norm(processed_embeddings).item():.4f}")

    average_loss = accumulated_loss / effective_step_count(
        step_counts, {algorithm: True for algorithm in ALGORITHMS}
    )
    avg_aux_losses = accumulated_aux_losses / metric_counters(step_counts)

    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    print("\nAVERAGE LOSSES:")
    print(f"  Total: {average_loss.item():.6f}")
    for index, metric_name in enumerate(METRIC_NAMES):
        print(f"  {metric_name}: {avg_aux_losses[index].item():.6f}")

    print("\nOVERALL ACCURACIES:")
    for metric_name in METRIC_NAMES:
        correct = metric_counts[metric_name]["correct"]
        total = metric_counts[metric_name]["total"]
        print(f"  {metric_name}: {correct / max(total, 1):.3f} ({correct}/{total})")
    print(f"{'=' * 80}\n")

    per_head_norms = {
        name: accumulated_norms[name] / max(norm_steps[name], 1) for name in accumulated_norms
    }
    return average_loss, per_head_norms


def calculate_losses_and_accuracies(
    model,
    graph_data,
    embedding_dim=128,
    termination_cfg=None,
    selected_tasks=None,
):
    """Return average losses and accuracies, matching train.py logic."""
    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([len(METRIC_NAMES)])

    correct = {metric_name: 0 for metric_name in METRIC_NAMES}
    total = {metric_name: 0 for metric_name in METRIC_NAMES}

    num_nodes = graph_data['num_nodes']
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])

    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    num_prim_steps = len(graph_data['prim_key_targets'])
    num_steps = max(num_bf_steps, num_bfs_steps, num_prim_steps)

    termination_settings = resolve_termination_settings(termination_cfg)
    previous_distance_latent = None

    selected_tasks = _normalize_selected_tasks(selected_tasks)
    step_counts = execution_step_counts(graph_data)
    loss_scale = metric_mask(selected_tasks)

    for i in range(num_steps):
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        prim_sample_exists = (i + 1) < num_prim_steps

        if not (bf_sample_exists or bfs_sample_exists or prim_sample_exists):
            continue

        if bfs_sample_exists:
            true_bfs_state = graph_data['bfs_state_targets'][i]
            target_bfs_state = graph_data['bfs_state_targets'][i + 1]
        else:
            true_bfs_state = graph_data['bfs_state_targets'][-1]
            target_bfs_state = graph_data['bfs_state_targets'][-1]

        if bf_sample_exists:
            true_distance_bf = graph_data['bf_distance_targets'][i]
            target_distance_bf = graph_data['bf_distance_targets'][i + 1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][i + 1]
        else:
            true_distance_bf = graph_data['bf_distance_targets'][-1]
            target_distance_bf = graph_data['bf_distance_targets'][-1]
            target_predecessor_bf = graph_data['bf_predecessor_targets'][-1]

        if prim_sample_exists:
            true_prim_state = graph_data['prim_state_targets'][i]
            target_prim_state = graph_data['prim_state_targets'][i + 1]
            true_prim_key = graph_data['prim_key_targets'][i]
            target_prim_key = graph_data['prim_key_targets'][i + 1]
            target_prim_predecessor = graph_data['prim_predecessor_targets'][i + 1]
        else:
            true_prim_state = graph_data['prim_state_targets'][-1]
            target_prim_state = graph_data['prim_state_targets'][-1]
            true_prim_key = graph_data['prim_key_targets'][-1]
            target_prim_key = graph_data['prim_key_targets'][-1]
            target_prim_predecessor = graph_data['prim_predecessor_targets'][-1]

        termination_targets = {
            'bf': mx.array(1.0 if (i + 1) == (num_bf_steps - 1) else 0.0),
            'bfs': mx.array(1.0 if (i + 1) == (num_bfs_steps - 1) else 0.0),
            'prim': mx.array(1.0 if (i + 1) == (num_prim_steps - 1) else 0.0),
        }

        node_algo_features = build_node_algo_features(
            true_bfs_state,
            true_distance_bf,
            true_prim_state,
            true_prim_key,
        )
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data['edge_matrix'])

        need_aux = needs_aux_latents(termination_settings)
        if need_aux:
            bfs_output, bf_output, prim_output, termination_probs, processed_embeddings, aux = model(
                model_input, return_latents=True
            )
        else:
            bfs_output, bf_output, prim_output, termination_probs, processed_embeddings = model(model_input)
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

        if bf_sample_exists and selected_tasks["bf"]:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            bf_distance_loss = nn.losses.mse_loss(
                bf_distance_predictions, target_distance_bf, reduction='mean'
            )
            distance_correct = int(mx.sum(mx.abs(bf_distance_predictions - target_distance_bf) <= 0.1).item())
            correct["bf_distance"] += distance_correct
            total["bf_distance"] += num_nodes

            valid_mask = (target_predecessor_bf != -1)
            safe_targets = mx.where(valid_mask, target_predecessor_bf, mx.zeros_like(target_predecessor_bf))
            per_node_ce = nn.losses.cross_entropy(
                bf_predecessor_predictions,
                safe_targets,
                reduction='none',
                label_smoothing=1e-6,
            )
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            bf_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            pred_correct = int(mx.sum((pred_argmax == target_predecessor_bf) & valid_mask).item())
            pred_total = int(mx.sum(valid_mask).item())
            correct["bf_predecessor"] += pred_correct
            total["bf_predecessor"] += pred_total

            bf_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bf'],
                termination_targets['bf'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bf_termination_loss = mx.array(0.0)
            bf_term_correct = int(
                ((mx.sigmoid(termination_logits['bf']) > 0.5).astype(mx.float32) == termination_targets['bf']).item()
            )
            correct["bf_termination"] += bf_term_correct
            total["bf_termination"] += 1
        else:
            bf_distance_loss = mx.array(0.0)
            bf_predecessor_loss = mx.array(0.0)
            bf_termination_loss = mx.array(0.0)

        if bfs_sample_exists and selected_tasks["bfs"]:
            bfs_state_loss = nn.losses.binary_cross_entropy(
                bfs_output,
                target_bfs_state,
                reduction='mean',
                with_logits=True,
            )
            bfs_pred = (mx.sigmoid(bfs_output) > 0.5).astype(mx.float32)
            bfs_correct = int(mx.sum(bfs_pred == target_bfs_state).item())
            correct["bfs_state"] += bfs_correct
            total["bfs_state"] += num_nodes

            bfs_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['bfs'],
                termination_targets['bfs'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                bfs_termination_loss = mx.array(0.0)
            bfs_term_correct = int(
                ((mx.sigmoid(termination_logits['bfs']) > 0.5).astype(mx.float32) == termination_targets['bfs']).item()
            )
            correct["bfs_termination"] += bfs_term_correct
            total["bfs_termination"] += 1
        else:
            bfs_state_loss = mx.array(0.0)
            bfs_termination_loss = mx.array(0.0)

        if prim_sample_exists and selected_tasks["prim"]:
            prim_state_predictions, prim_key_predictions, prim_predecessor_predictions = prim_output
            prim_state_loss = nn.losses.binary_cross_entropy(
                prim_state_predictions,
                target_prim_state,
                reduction='mean',
                with_logits=True,
            )
            prim_state_pred = (mx.sigmoid(prim_state_predictions) > 0.5).astype(mx.float32)
            prim_state_correct = int(mx.sum(prim_state_pred == target_prim_state).item())
            correct["prim_state"] += prim_state_correct
            total["prim_state"] += num_nodes

            prim_key_loss = nn.losses.mse_loss(
                prim_key_predictions,
                target_prim_key,
                reduction='mean',
            )
            prim_key_correct = int(mx.sum(mx.abs(prim_key_predictions - target_prim_key) <= 0.1).item())
            correct["prim_key"] += prim_key_correct
            total["prim_key"] += num_nodes

            valid_mask = (target_prim_predecessor != -1)
            safe_targets = mx.where(valid_mask, target_prim_predecessor, mx.zeros_like(target_prim_predecessor))
            per_node_ce = nn.losses.cross_entropy(
                prim_predecessor_predictions,
                safe_targets,
                reduction='none',
                label_smoothing=1e-6,
            )
            valid_mask_f = valid_mask.astype(mx.float32)
            denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
            prim_predecessor_loss = (per_node_ce * valid_mask_f).sum() / denom
            prim_pred_argmax = mx.argmax(prim_predecessor_predictions, axis=-1)
            prim_pred_correct = int(mx.sum((prim_pred_argmax == target_prim_predecessor) & valid_mask).item())
            prim_pred_total = int(mx.sum(valid_mask).item())
            correct["prim_predecessor"] += prim_pred_correct
            total["prim_predecessor"] += prim_pred_total

            prim_termination_loss = nn.losses.binary_cross_entropy(
                termination_logits['prim'],
                termination_targets['prim'],
                reduction='mean',
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                prim_termination_loss = mx.array(0.0)
            prim_term_correct = int(
                ((mx.sigmoid(termination_logits['prim']) > 0.5).astype(mx.float32) == termination_targets['prim']).item()
            )
            correct["prim_termination"] += prim_term_correct
            total["prim_termination"] += 1
        else:
            prim_state_loss = mx.array(0.0)
            prim_key_loss = mx.array(0.0)
            prim_predecessor_loss = mx.array(0.0)
            prim_termination_loss = mx.array(0.0)

        raw_losses = mx.array(
            [
                bf_distance_loss,
                bf_predecessor_loss,
                bfs_state_loss,
                prim_state_loss,
                prim_key_loss,
                prim_predecessor_loss,
                bf_termination_loss,
                bfs_termination_loss,
                prim_termination_loss,
            ]
        )
        raw_losses = raw_losses * loss_scale
        accumulated_loss += mx.sum(raw_losses)
        accumulated_aux_losses += raw_losses
        previous_step_hidden_states = processed_embeddings

    average_loss = accumulated_loss / effective_step_count(step_counts, selected_tasks)
    avg_aux_losses = accumulated_aux_losses / metric_counters(step_counts)
    accuracies = mx.array(
        [correct[name] / max(total[name], 1) for name in METRIC_NAMES], dtype=mx.float32
    )
    return avg_aux_losses, average_loss, accuracies


def calculate_accuracies(
    model,
    graph_data,
    embedding_dim=128,
    termination_cfg=None,
    selected_tasks=None,
):
    """Return only mean accuracies over the selected tasks."""
    _, _, accuracies = calculate_losses_and_accuracies(
        model=model,
        graph_data=graph_data,
        embedding_dim=embedding_dim,
        termination_cfg=termination_cfg,
        selected_tasks=selected_tasks,
    )
    return accuracies


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
    num_prim_steps = len(graph_data["prim_key_targets"])
    num_steps = max(num_bf_steps, num_bfs_steps, num_prim_steps)

    correct = {metric_name: 0 for metric_name in METRIC_NAMES}
    total = {metric_name: 0 for metric_name in METRIC_NAMES}
    termination_confusion = {f"{algo}_{kind}": 0 for algo in ALGORITHMS for kind in ("fp", "fn")}
    termination_step_totals = {algo: {} for algo in ALGORITHMS}
    termination_step_mispredicts = {algo: {} for algo in ALGORITHMS}

    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])
    previous_distance_latent = None
    step_failures = []
    first_failure_step = None

    for i in range(num_steps):
        bf_sample_exists = (i + 1) < num_bf_steps
        bfs_sample_exists = (i + 1) < num_bfs_steps
        prim_sample_exists = (i + 1) < num_prim_steps
        if not (bf_sample_exists or bfs_sample_exists or prim_sample_exists):
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

        if prim_sample_exists:
            true_prim_state = graph_data["prim_state_targets"][i]
            target_prim_state = graph_data["prim_state_targets"][i + 1]
            true_prim_key = graph_data["prim_key_targets"][i]
            target_prim_key = graph_data["prim_key_targets"][i + 1]
            target_prim_predecessor = graph_data["prim_predecessor_targets"][i + 1]
        else:
            true_prim_state = graph_data["prim_state_targets"][-1]
            target_prim_state = graph_data["prim_state_targets"][-1]
            true_prim_key = graph_data["prim_key_targets"][-1]
            target_prim_key = graph_data["prim_key_targets"][-1]
            target_prim_predecessor = graph_data["prim_predecessor_targets"][-1]

        termination_targets = {
            "bf": mx.array(1.0 if (i + 1) == (num_bf_steps - 1) else 0.0),
            "bfs": mx.array(1.0 if (i + 1) == (num_bfs_steps - 1) else 0.0),
            "prim": mx.array(1.0 if (i + 1) == (num_prim_steps - 1) else 0.0),
        }

        node_algo_features = build_node_algo_features(
            true_bfs_state,
            true_distance_bf,
            true_prim_state,
            true_prim_key,
        )
        input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data["edge_matrix"])

        need_aux = needs_aux_latents(termination_settings)
        if need_aux:
            bfs_output, bf_output, prim_output, termination_probs, processed_embeddings, aux = model(
                model_input, return_latents=True
            )
        else:
            bfs_output, bf_output, prim_output, termination_probs, processed_embeddings = model(model_input)
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
            "prim_state_incorrect": 0,
            "prim_key_incorrect": 0,
            "prim_predecessor_incorrect": 0,
            "bf_termination_incorrect": 0,
            "bfs_termination_incorrect": 0,
            "prim_termination_incorrect": 0,
        }

        if bf_sample_exists and selected_tasks["bf"]:
            bf_distance_predictions, bf_predecessor_predictions = bf_output
            bf_distance_correct = int(mx.sum(mx.abs(bf_distance_predictions - target_distance_bf) <= 0.1).item())
            correct["bf_distance"] += bf_distance_correct
            total["bf_distance"] += num_nodes
            step_entry["bf_distance_incorrect"] = int(num_nodes - bf_distance_correct)

            valid_mask = target_predecessor_bf != -1
            bf_pred_argmax = mx.argmax(bf_predecessor_predictions, axis=-1)
            bf_pred_correct = int(mx.sum((bf_pred_argmax == target_predecessor_bf) & valid_mask).item())
            bf_pred_total = int(mx.sum(valid_mask).item())
            correct["bf_predecessor"] += bf_pred_correct
            total["bf_predecessor"] += bf_pred_total
            step_entry["bf_predecessor_incorrect"] = int(max(bf_pred_total - bf_pred_correct, 0))

            bf_term_prob = float(mx.sigmoid(termination_logits["bf"]).item())
            bf_term_pred = 1 if bf_term_prob > 0.5 else 0
            bf_term_target = int(termination_targets["bf"].item())
            bf_term_correct = int(bf_term_pred == bf_term_target)
            correct["bf_termination"] += bf_term_correct
            total["bf_termination"] += 1
            step_entry["bf_termination_incorrect"] = 1 - bf_term_correct
            step_key = int(i + 1)
            termination_step_totals["bf"][step_key] = termination_step_totals["bf"].get(step_key, 0) + 1
            if step_entry["bf_termination_incorrect"] > 0:
                termination_step_mispredicts["bf"][step_key] = termination_step_mispredicts["bf"].get(step_key, 0) + 1
            if bf_term_pred == 1 and bf_term_target == 0:
                termination_confusion["bf_fp"] += 1
            elif bf_term_pred == 0 and bf_term_target == 1:
                termination_confusion["bf_fn"] += 1

        if bfs_sample_exists and selected_tasks["bfs"]:
            bfs_state_pred = (mx.sigmoid(bfs_output) > 0.5).astype(mx.float32)
            bfs_correct = int(mx.sum(bfs_state_pred == target_bfs_state).item())
            correct["bfs_state"] += bfs_correct
            total["bfs_state"] += num_nodes
            step_entry["bfs_state_incorrect"] = int(num_nodes - bfs_correct)

            bfs_term_prob = float(mx.sigmoid(termination_logits["bfs"]).item())
            bfs_term_pred = 1 if bfs_term_prob > 0.5 else 0
            bfs_term_target = int(termination_targets["bfs"].item())
            bfs_term_correct = int(bfs_term_pred == bfs_term_target)
            correct["bfs_termination"] += bfs_term_correct
            total["bfs_termination"] += 1
            step_entry["bfs_termination_incorrect"] = 1 - bfs_term_correct
            step_key = int(i + 1)
            termination_step_totals["bfs"][step_key] = termination_step_totals["bfs"].get(step_key, 0) + 1
            if step_entry["bfs_termination_incorrect"] > 0:
                termination_step_mispredicts["bfs"][step_key] = termination_step_mispredicts["bfs"].get(step_key, 0) + 1
            if bfs_term_pred == 1 and bfs_term_target == 0:
                termination_confusion["bfs_fp"] += 1
            elif bfs_term_pred == 0 and bfs_term_target == 1:
                termination_confusion["bfs_fn"] += 1

        if prim_sample_exists and selected_tasks["prim"]:
            prim_state_predictions, prim_key_predictions, prim_predecessor_predictions = prim_output
            prim_state_pred = (mx.sigmoid(prim_state_predictions) > 0.5).astype(mx.float32)
            prim_state_correct = int(mx.sum(prim_state_pred == target_prim_state).item())
            correct["prim_state"] += prim_state_correct
            total["prim_state"] += num_nodes
            step_entry["prim_state_incorrect"] = int(num_nodes - prim_state_correct)

            prim_key_correct = int(mx.sum(mx.abs(prim_key_predictions - target_prim_key) <= 0.1).item())
            correct["prim_key"] += prim_key_correct
            total["prim_key"] += num_nodes
            step_entry["prim_key_incorrect"] = int(num_nodes - prim_key_correct)

            valid_mask = target_prim_predecessor != -1
            prim_pred_argmax = mx.argmax(prim_predecessor_predictions, axis=-1)
            prim_pred_correct = int(mx.sum((prim_pred_argmax == target_prim_predecessor) & valid_mask).item())
            prim_pred_total = int(mx.sum(valid_mask).item())
            correct["prim_predecessor"] += prim_pred_correct
            total["prim_predecessor"] += prim_pred_total
            step_entry["prim_predecessor_incorrect"] = int(max(prim_pred_total - prim_pred_correct, 0))

            prim_term_prob = float(mx.sigmoid(termination_logits["prim"]).item())
            prim_term_pred = 1 if prim_term_prob > 0.5 else 0
            prim_term_target = int(termination_targets["prim"].item())
            prim_term_correct = int(prim_term_pred == prim_term_target)
            correct["prim_termination"] += prim_term_correct
            total["prim_termination"] += 1
            step_entry["prim_termination_incorrect"] = 1 - prim_term_correct
            step_key = int(i + 1)
            termination_step_totals["prim"][step_key] = termination_step_totals["prim"].get(step_key, 0) + 1
            if step_entry["prim_termination_incorrect"] > 0:
                termination_step_mispredicts["prim"][step_key] = termination_step_mispredicts["prim"].get(step_key, 0) + 1
            if prim_term_pred == 1 and prim_term_target == 0:
                termination_confusion["prim_fp"] += 1
            elif prim_term_pred == 0 and prim_term_target == 1:
                termination_confusion["prim_fn"] += 1

        step_error_units = sum(value for key, value in step_entry.items() if key.endswith("_incorrect"))
        if step_error_units > 0:
            if first_failure_step is None:
                first_failure_step = int(i + 1)
            if include_step_details:
                step_failures.append(step_entry)

        previous_step_hidden_states = processed_embeddings

    accuracies = {
        metric_name: correct[metric_name] / max(total[metric_name], 1)
        for metric_name in METRIC_NAMES
    }
    incorrect = {
        metric_name: total[metric_name] - correct[metric_name]
        for metric_name in METRIC_NAMES
    }

    failed_tasks = []
    for metric_name in METRIC_NAMES:
        algorithm = metric_name.split("_")[0]
        algorithm = "bfs" if metric_name.startswith("bfs") else algorithm
        algorithm = "bf" if metric_name.startswith("bf") else algorithm
        algorithm = "prim" if metric_name.startswith("prim") else algorithm
        if selected_tasks.get(algorithm, False) and incorrect[metric_name] > 0:
            failed_tasks.append(metric_name)

    return {
        "num_nodes": num_nodes,
        "num_bf_steps": int(num_bf_steps),
        "num_bfs_steps": int(num_bfs_steps),
        "num_prim_steps": int(num_prim_steps),
        "first_failure_step": first_failure_step,
        "accuracy": {k: float(v) for k, v in accuracies.items()},
        "incorrect": {k: int(v) for k, v in incorrect.items()},
        "termination_confusion": {k: int(v) for k, v in termination_confusion.items()},
        "termination_step_totals": {
            algo: {int(step): int(count) for step, count in values.items()}
            for algo, values in termination_step_totals.items()
        },
        "termination_step_mispredicts": {
            algo: {int(step): int(count) for step, count in values.items()}
            for algo, values in termination_step_mispredicts.items()
        },
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

    failures_by_task = {metric_name: [] for metric_name in METRIC_NAMES}
    failed_graphs = []
    ranked_failed = []
    termination_step_totals = {algo: {} for algo in ALGORITHMS}
    termination_step_mispredicts = {algo: {} for algo in ALGORITHMS}

    for graph_index, graph_data in enumerate(graphs):
        details = _graph_failure_details(
            model=model,
            graph_data=graph_data,
            embedding_dim=embedding_dim,
            termination_cfg=termination_cfg,
            selected_tasks=selected_tasks,
            include_step_details=include_step_details,
        )

        for algo in ALGORITHMS:
            for step, count in details["termination_step_totals"][algo].items():
                step_key = int(step)
                termination_step_totals[algo][step_key] = (
                    termination_step_totals[algo].get(step_key, 0) + int(count)
                )
            for step, count in details["termination_step_mispredicts"][algo].items():
                step_key = int(step)
                termination_step_mispredicts[algo][step_key] = (
                    termination_step_mispredicts[algo].get(step_key, 0) + int(count)
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

    termination_step_rates = {algo: {} for algo in ALGORITHMS}
    for algo in ALGORITHMS:
        step_keys = sorted(set(termination_step_totals[algo].keys()))
        for step_key in step_keys:
            total = int(termination_step_totals[algo].get(step_key, 0))
            errors = int(termination_step_mispredicts[algo].get(step_key, 0))
            termination_step_rates[algo][step_key] = float(errors / max(total, 1))

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
        "termination_mispredict_distribution": {
            algo: {
                "counts_by_step": {
                    int(step): int(count)
                    for step, count in sorted(termination_step_mispredicts[algo].items())
                },
                "totals_by_step": {
                    int(step): int(count)
                    for step, count in sorted(termination_step_totals[algo].items())
                },
                "rate_by_step": {
                    int(step): float(rate)
                    for step, rate in sorted(termination_step_rates[algo].items())
                },
            }
            for algo in ALGORITHMS
        },
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
         prim_state_predictions, prim_key_predictions, prim_predecessor_predictions,
         bf_termination_prob, bfs_termination_prob, prim_termination_prob,
         processed_embeddings)
    """
    
    mx.eval(model.parameters())

    def call(input_embeddings, edge_matrix):
        bfs_output, bf_output, prim_output, termination_probs, processed_embeddings = model((input_embeddings, edge_matrix))
        bf_distance_predictions, bf_predecessor_predictions = bf_output
        (
            prim_state_predictions,
            prim_key_predictions,
            prim_predecessor_predictions,
        ) = prim_output
        
        # Flatten outputs to a simple tuple of arrays for export
        return (bfs_output,
                bf_distance_predictions,
                bf_predecessor_predictions,
                prim_state_predictions,
                prim_key_predictions,
                prim_predecessor_predictions,
                termination_probs['bf'],
                termination_probs['bfs'],
                termination_probs['prim'],
                processed_embeddings)

    mx.export_function(output_path, call, (sample_input_embeddings, sample_edge_matrix))
