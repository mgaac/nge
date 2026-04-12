"""Utility functions for model evaluation and debugging."""

from __future__ import annotations

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
    algorithm_display_name,
    algorithm_family,
    algorithm_target_lengths,
    build_node_algo_features,
    effective_step_count,
    execution_step_counts,
    feature_values_for_step,
    metric_counters,
    metric_names,
    metric_mask,
    selected_tasks_for_graph,
    targets_for_step,
    termination_targets_for_step,
)


def _normalize_selected_tasks(selected_tasks=None, algorithm_order=None):
    """Return a complete enabled/disabled mask for the configured algorithms."""
    algorithms = tuple(algorithm_order) if algorithm_order is not None else ALGORITHMS
    if selected_tasks is None:
        return {algorithm: True for algorithm in algorithms}
    return {
        algorithm: bool(selected_tasks.get(algorithm, False)) for algorithm in algorithms
    }


def extract_per_head_magnitude_grads(grads):
    """Extract the L2 norm of gradients for each top-level model component."""
    head_names = set()
    utils.tree_map_with_path(lambda path, _: head_names.add(path.split(".")[0]), grads)

    per_head_magnitude_grads = {}
    for head_name in head_names:
        per_head_magnitude_grads[head_name] = utils.tree_reduce(
            lambda acc, x: acc + mx.sum(mx.square(x)),
            grads[head_name],
            0.0,
        ) ** 0.5
    return per_head_magnitude_grads


def _normalize_model_forward(model_output, algorithm_order, return_latents=False):
    """Normalize legacy BF/BFS/Prim outputs and dynamic output dictionaries."""
    if return_latents:
        if isinstance(model_output[0], dict):
            return model_output
        bfs_output, bf_output, prim_output, termination_probs, processed_embeddings, aux = model_output
        outputs = {"bfs": bfs_output, "bf": bf_output, "prim": prim_output}
        return outputs, termination_probs, processed_embeddings, aux

    if isinstance(model_output[0], dict):
        return model_output
    bfs_output, bf_output, prim_output, termination_probs, processed_embeddings = model_output
    outputs = {"bfs": bfs_output, "bf": bf_output, "prim": prim_output}
    return outputs, termination_probs, processed_embeddings


def _eval_trees(*trees):
    """Force evaluation of MLX arrays nested inside pytrees."""
    leaves = []
    for tree in trees:
        if tree is None:
            continue
        leaves.extend(value for _, value in utils.tree_flatten(tree))
    if leaves:
        mx.eval(*leaves)


def _int_item(value) -> int:
    """Synchronize an MLX scalar and convert it to int."""
    mx.eval(value)
    return int(value.item())


def _float_item(value) -> float:
    """Synchronize an MLX scalar and convert it to float."""
    mx.eval(value)
    return float(value.item())


def _predicted_feature_values_for_next_step(algorithm_outputs, algorithm_order):
    """Convert decoded outputs into the recurrent state features for the next step."""
    feature_values = {}
    for algorithm in algorithm_order:
        family = algorithm_family(algorithm)
        output = algorithm_outputs[algorithm]

        if family == "state_mask":
            feature_values[f"{algorithm}_state"] = (
                mx.sigmoid(output) > 0.5
            ).astype(mx.float32)
        elif family == "shortest_path":
            distance_predictions, _ = output
            feature_values[f"{algorithm}_distance"] = distance_predictions
        elif family == "mst":
            state_predictions, key_predictions, _ = output
            feature_values[f"{algorithm}_state"] = (
                mx.sigmoid(state_predictions) > 0.5
            ).astype(mx.float32)
            feature_values[f"{algorithm}_key"] = key_predictions
        else:
            raise ValueError(f"Unsupported algorithm family: {family}")

    return feature_values


def _forward_step(
    model,
    graph_data,
    step_index,
    previous_step_hidden_states,
    previous_distance_latent,
    termination_settings,
    algorithm_order,
    step_counts,
    feature_values_override=None,
):
    """Run one execution step and return normalized outputs."""
    sample_exists = {algorithm: step_index < step_counts[algorithm] for algorithm in algorithm_order}
    if not any(sample_exists.values()):
        return None

    feature_values = (
        feature_values_for_step(graph_data, step_index, algorithm_order)
        if feature_values_override is None
        else feature_values_override
    )
    targets = targets_for_step(graph_data, step_index, algorithm_order)
    termination_targets = termination_targets_for_step(graph_data, step_index, algorithm_order)

    node_algo_features = build_node_algo_features(feature_values, algorithm_order)
    input_embeddings = mx.concatenate([previous_step_hidden_states, node_algo_features], axis=1)
    model_input = (input_embeddings, graph_data["edge_matrix"])

    need_aux = needs_aux_latents(termination_settings)
    if need_aux:
        algorithm_outputs, termination_probs, processed_embeddings, aux = _normalize_model_forward(
            model(model_input, return_latents=True),
            algorithm_order,
            return_latents=True,
        )
    else:
        algorithm_outputs, termination_probs, processed_embeddings = _normalize_model_forward(
            model(model_input),
            algorithm_order,
            return_latents=False,
        )
        aux = None

    if termination_settings["mode"] == "distance":
        current_latent = get_distance_latent(termination_settings, processed_embeddings, aux)
        termination_logits = compute_distance_termination_logits(
            settings=termination_settings,
            prev_latent=previous_distance_latent,
            current_latent=current_latent,
            algorithms=algorithm_order,
        )
        next_distance_latent = current_latent
    else:
        termination_logits = termination_probs
        next_distance_latent = previous_distance_latent

    _eval_trees(algorithm_outputs, termination_logits, processed_embeddings, aux)

    return {
        "sample_exists": sample_exists,
        "feature_values": feature_values,
        "targets": targets,
        "termination_targets": termination_targets,
        "algorithm_outputs": algorithm_outputs,
        "termination_logits": termination_logits,
        "processed_embeddings": processed_embeddings,
        "next_distance_latent": next_distance_latent,
        "next_feature_values": _predicted_feature_values_for_next_step(
            algorithm_outputs,
            algorithm_order,
        ),
    }


def _pointer_loss_and_accuracy(predictions, targets):
    """Return masked cross-entropy loss and pointer accuracy stats."""
    valid_mask = targets != -1
    safe_targets = mx.where(valid_mask, targets, mx.zeros_like(targets))
    per_node_ce = nn.losses.cross_entropy(
        predictions,
        safe_targets,
        reduction="none",
    )
    valid_mask_f = valid_mask.astype(mx.float32)
    denom = mx.maximum(valid_mask_f.sum(), mx.array(1.0))
    loss = (per_node_ce * valid_mask_f).sum() / denom

    pred_argmax = mx.argmax(predictions, axis=-1)
    total = _int_item(mx.sum(valid_mask))
    correct = _int_item(mx.sum((pred_argmax == targets) & valid_mask))
    return loss, pred_argmax, correct, total


def print_execution_details(model, graph_data, embedding_dim=128, termination_cfg=None):
    """Print per-step execution details for all configured algorithm heads."""
    del embedding_dim  # The value is part of the public API but not used here.

    algorithm_order = tuple(getattr(model, "algorithms", ALGORITHMS))
    current_metric_names = metric_names(algorithm_order)
    current_metric_index = {name: index for index, name in enumerate(current_metric_names)}
    selected_tasks = selected_tasks_for_graph(
        graph_data,
        {algorithm: True for algorithm in algorithm_order},
        algorithm_order,
    )
    termination_settings = resolve_termination_settings(termination_cfg)

    num_nodes = int(graph_data["num_nodes"])
    step_counts = execution_step_counts(graph_data, algorithm_order)
    sequence_lengths = algorithm_target_lengths(graph_data, algorithm_order)
    num_steps = max(step_counts.values(), default=0) + 1

    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([len(current_metric_names)])
    metric_counts = {
        metric_name: {"correct": 0, "total": 0} for metric_name in current_metric_names
    }
    accumulated_norms = {"hidden_state": mx.array(0.0)}
    norm_steps = {"hidden_state": 0}
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])
    previous_distance_latent = None
    current_feature_values = feature_values_for_step(graph_data, 0, algorithm_order)

    print(f"\n{'=' * 80}")
    print(f"EXECUTION DETAILS - Graph with {num_nodes} nodes")
    print(
        "Steps: "
        + ", ".join(
            f"{algorithm_display_name(algorithm)}={sequence_lengths[algorithm]}"
            for algorithm in algorithm_order
        )
    )
    print(f"{'=' * 80}")

    for step_index in range(num_steps):
        step_payload = _forward_step(
            model=model,
            graph_data=graph_data,
            step_index=step_index,
            previous_step_hidden_states=previous_step_hidden_states,
            previous_distance_latent=previous_distance_latent,
            termination_settings=termination_settings,
            algorithm_order=algorithm_order,
            step_counts=step_counts,
            feature_values_override=current_feature_values,
        )
        if step_payload is None:
            continue

        sample_exists = step_payload["sample_exists"]
        targets = step_payload["targets"]
        termination_targets = step_payload["termination_targets"]
        algorithm_outputs = step_payload["algorithm_outputs"]
        termination_logits = step_payload["termination_logits"]
        processed_embeddings = step_payload["processed_embeddings"]
        previous_distance_latent = step_payload["next_distance_latent"]
        current_feature_values = step_payload["next_feature_values"]

        print(f"\n{'=' * 60}")
        print(f"STEP {step_index} -> {step_index + 1}")
        print(
            "Samples: "
            + ", ".join(
                f"{algorithm_display_name(algorithm)}={sample_exists[algorithm]}"
                for algorithm in algorithm_order
            )
        )
        print(f"{'=' * 60}")

        raw_losses = mx.zeros([len(current_metric_names)])
        for algorithm in algorithm_order:
            family = algorithm_family(algorithm)
            display_name = algorithm_display_name(algorithm)
            if not sample_exists[algorithm] or not selected_tasks.get(algorithm, False):
                continue

            output = algorithm_outputs[algorithm]
            target = targets[algorithm]

            if family == "state_mask":
                state_loss = nn.losses.binary_cross_entropy(
                    output,
                    target["state"],
                    reduction="mean",
                    with_logits=True,
                )
                state_pred = (mx.sigmoid(output) > 0.5).astype(mx.float32)
                state_correct = _int_item(mx.sum(state_pred == target["state"]))
                state_metric = f"{algorithm}_state"
                metric_counts[state_metric]["correct"] += state_correct
                metric_counts[state_metric]["total"] += num_nodes
                raw_losses = raw_losses.at[current_metric_index[state_metric]].add(state_loss)

                print(f"\n{display_name.upper()} STATE:")
                print(f"  Loss: {float(state_loss):.6f}")
                print(
                    f"  Accuracy: {state_correct}/{num_nodes} = "
                    f"{state_correct / max(num_nodes, 1):.3f}"
                )
                print(
                    f"  Predictions (first 10): "
                    f"{state_pred[:10].astype(mx.int32).tolist()}"
                )
                print(
                    f"  Targets (first 10): "
                    f"{target['state'][:10].astype(mx.int32).tolist()}"
                )

            elif family == "shortest_path":
                distance_predictions, predecessor_predictions = output
                distance_loss = nn.losses.mse_loss(
                    distance_predictions,
                    target["distance"],
                    reduction="mean",
                )
                distance_correct = _int_item(
                    mx.sum(mx.abs(distance_predictions - target["distance"]) <= 0.1)
                )
                distance_metric = f"{algorithm}_distance"
                metric_counts[distance_metric]["correct"] += distance_correct
                metric_counts[distance_metric]["total"] += num_nodes
                raw_losses = raw_losses.at[current_metric_index[distance_metric]].add(distance_loss)

                predecessor_loss, predecessor_argmax, predecessor_correct, predecessor_total = (
                    _pointer_loss_and_accuracy(predecessor_predictions, target["predecessor"])
                )
                predecessor_metric = f"{algorithm}_predecessor"
                metric_counts[predecessor_metric]["correct"] += predecessor_correct
                metric_counts[predecessor_metric]["total"] += predecessor_total
                raw_losses = raw_losses.at[current_metric_index[predecessor_metric]].add(
                    predecessor_loss
                )

                print(f"\n{display_name.upper()} DISTANCE:")
                print(f"  Loss: {float(distance_loss):.6f}")
                print(
                    f"  Accuracy: {distance_correct}/{num_nodes} = "
                    f"{distance_correct / max(num_nodes, 1):.3f}"
                )
                print(
                    f"  Predictions (first 10): {distance_predictions[:10].tolist()}"
                )
                print(f"  Targets (first 10): {target['distance'][:10].tolist()}")

                print(f"\n{display_name.upper()} PREDECESSOR:")
                print(f"  Loss: {float(predecessor_loss):.6f}")
                print(
                    f"  Accuracy: {predecessor_correct}/{max(predecessor_total, 1)} = "
                    f"{predecessor_correct / max(predecessor_total, 1):.3f}"
                )
                print(
                    f"  Predictions (argmax, first 10): {predecessor_argmax[:10].tolist()}"
                )
                print(
                    f"  Targets (first 10): {target['predecessor'][:10].tolist()}"
                )

            elif family == "mst":
                state_predictions, key_predictions, predecessor_predictions = output
                state_loss = nn.losses.binary_cross_entropy(
                    state_predictions,
                    target["state"],
                    reduction="mean",
                    with_logits=True,
                )
                state_pred = (mx.sigmoid(state_predictions) > 0.5).astype(mx.float32)
                state_correct = _int_item(mx.sum(state_pred == target["state"]))
                state_metric = f"{algorithm}_state"
                metric_counts[state_metric]["correct"] += state_correct
                metric_counts[state_metric]["total"] += num_nodes
                raw_losses = raw_losses.at[current_metric_index[state_metric]].add(state_loss)

                key_loss = nn.losses.mse_loss(
                    key_predictions,
                    target["key"],
                    reduction="mean",
                )
                key_correct = _int_item(
                    mx.sum(mx.abs(key_predictions - target["key"]) <= 0.1)
                )
                key_metric = f"{algorithm}_key"
                metric_counts[key_metric]["correct"] += key_correct
                metric_counts[key_metric]["total"] += num_nodes
                raw_losses = raw_losses.at[current_metric_index[key_metric]].add(key_loss)

                predecessor_loss, predecessor_argmax, predecessor_correct, predecessor_total = (
                    _pointer_loss_and_accuracy(predecessor_predictions, target["predecessor"])
                )
                predecessor_metric = f"{algorithm}_predecessor"
                metric_counts[predecessor_metric]["correct"] += predecessor_correct
                metric_counts[predecessor_metric]["total"] += predecessor_total
                raw_losses = raw_losses.at[current_metric_index[predecessor_metric]].add(
                    predecessor_loss
                )

                print(f"\n{display_name.upper()} STATE:")
                print(f"  Loss: {float(state_loss):.6f}")
                print(
                    f"  Accuracy: {state_correct}/{num_nodes} = "
                    f"{state_correct / max(num_nodes, 1):.3f}"
                )
                print(
                    f"  Predictions (first 10): {state_pred[:10].astype(mx.int32).tolist()}"
                )
                print(
                    f"  Targets (first 10): {target['state'][:10].astype(mx.int32).tolist()}"
                )

                print(f"\n{display_name.upper()} KEY:")
                print(f"  Loss: {float(key_loss):.6f}")
                print(
                    f"  Accuracy: {key_correct}/{num_nodes} = "
                    f"{key_correct / max(num_nodes, 1):.3f}"
                )
                print(f"  Predictions (first 10): {key_predictions[:10].tolist()}")
                print(f"  Targets (first 10): {target['key'][:10].tolist()}")

                print(f"\n{display_name.upper()} PREDECESSOR:")
                print(f"  Loss: {float(predecessor_loss):.6f}")
                print(
                    f"  Accuracy: {predecessor_correct}/{max(predecessor_total, 1)} = "
                    f"{predecessor_correct / max(predecessor_total, 1):.3f}"
                )
                print(
                    f"  Predictions (argmax, first 10): {predecessor_argmax[:10].tolist()}"
                )
                print(
                    f"  Targets (first 10): {target['predecessor'][:10].tolist()}"
                )

            else:
                raise ValueError(f"Unsupported algorithm family: {family}")

            termination_metric = f"{algorithm}_termination"
            termination_loss = nn.losses.binary_cross_entropy(
                termination_logits[algorithm],
                termination_targets[algorithm],
                reduction="mean",
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                termination_loss = mx.array(0.0)
            termination_correct = _int_item(
                (mx.sigmoid(termination_logits[algorithm]) > 0.5).astype(mx.float32)
                == termination_targets[algorithm]
            )
            metric_counts[termination_metric]["correct"] += termination_correct
            metric_counts[termination_metric]["total"] += 1
            raw_losses = raw_losses.at[current_metric_index[termination_metric]].add(
                termination_loss
            )

            print(f"\n{display_name.upper()} TERMINATION:")
            print(f"  Loss: {float(termination_loss):.6f}")
            print(
                f"  Logit: {float(termination_logits[algorithm]):.4f}, "
                f"Prob: {float(mx.sigmoid(termination_logits[algorithm])):.4f}"
            )
            print(
                f"  Target: {float(termination_targets[algorithm]):.1f}, "
                f"Correct: {bool(termination_correct)}"
            )

        accumulated_aux_losses += raw_losses
        total_step_loss = mx.sum(raw_losses)
        accumulated_loss += total_step_loss
        previous_step_hidden_states = processed_embeddings
        accumulated_norms["hidden_state"] += mx.linalg.norm(processed_embeddings)
        norm_steps["hidden_state"] += 1

        print("\nSTEP SUMMARY:")
        print(f"  Total step loss: {float(total_step_loss):.6f}")
        print(f"  Hidden state norm: {float(mx.linalg.norm(processed_embeddings)):.4f}")

    average_loss = accumulated_loss / effective_step_count(step_counts, selected_tasks)
    avg_aux_losses = accumulated_aux_losses / metric_counters(step_counts, algorithm_order)

    print(f"\n{'=' * 80}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 80}")
    print("\nAVERAGE LOSSES:")
    print(f"  Total: {float(average_loss):.6f}")
    for index, metric_name in enumerate(current_metric_names):
        print(f"  {metric_name}: {float(avg_aux_losses[index]):.6f}")

    print("\nOVERALL ACCURACIES:")
    for metric_name in current_metric_names:
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
    """Return average losses and accuracies under autoregressive evaluation."""
    del embedding_dim  # Public API compatibility.

    algorithm_order = tuple(getattr(model, "algorithms", ALGORITHMS))
    current_metric_names = metric_names(algorithm_order)
    current_metric_index = {name: index for index, name in enumerate(current_metric_names)}

    accumulated_loss = mx.array(0.0)
    accumulated_aux_losses = mx.zeros([len(current_metric_names)])
    correct = {metric_name: 0 for metric_name in current_metric_names}
    total = {metric_name: 0 for metric_name in current_metric_names}

    num_nodes = int(graph_data["num_nodes"])
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])
    previous_distance_latent = None
    current_feature_values = feature_values_for_step(graph_data, 0, algorithm_order)

    selected_tasks = _normalize_selected_tasks(selected_tasks, algorithm_order)
    sample_selected_tasks = selected_tasks_for_graph(
        graph_data, selected_tasks, algorithm_order
    )
    step_counts = execution_step_counts(graph_data, algorithm_order)
    num_steps = max(step_counts.values(), default=0) + 1
    termination_settings = resolve_termination_settings(termination_cfg)
    loss_scale = metric_mask(sample_selected_tasks, algorithm_order)

    for step_index in range(num_steps):
        step_payload = _forward_step(
            model=model,
            graph_data=graph_data,
            step_index=step_index,
            previous_step_hidden_states=previous_step_hidden_states,
            previous_distance_latent=previous_distance_latent,
            termination_settings=termination_settings,
            algorithm_order=algorithm_order,
            step_counts=step_counts,
            feature_values_override=current_feature_values,
        )
        if step_payload is None:
            continue

        sample_exists = step_payload["sample_exists"]
        targets = step_payload["targets"]
        termination_targets = step_payload["termination_targets"]
        algorithm_outputs = step_payload["algorithm_outputs"]
        termination_logits = step_payload["termination_logits"]
        processed_embeddings = step_payload["processed_embeddings"]
        previous_distance_latent = step_payload["next_distance_latent"]
        current_feature_values = step_payload["next_feature_values"]

        raw_losses = mx.zeros([len(current_metric_names)])
        for algorithm in algorithm_order:
            if not sample_exists[algorithm] or not sample_selected_tasks.get(algorithm, False):
                continue

            family = algorithm_family(algorithm)
            target = targets[algorithm]
            output = algorithm_outputs[algorithm]

            if family == "state_mask":
                state_loss = nn.losses.binary_cross_entropy(
                    output,
                    target["state"],
                    reduction="mean",
                    with_logits=True,
                )
                state_pred = (mx.sigmoid(output) > 0.5).astype(mx.float32)
                state_correct = _int_item(mx.sum(state_pred == target["state"]))
                raw_losses = raw_losses.at[current_metric_index[f"{algorithm}_state"]].add(
                    state_loss
                )
                correct[f"{algorithm}_state"] += state_correct
                total[f"{algorithm}_state"] += num_nodes

            elif family == "shortest_path":
                distance_predictions, predecessor_predictions = output
                distance_loss = nn.losses.mse_loss(
                    distance_predictions,
                    target["distance"],
                    reduction="mean",
                )
                distance_correct = _int_item(
                    mx.sum(mx.abs(distance_predictions - target["distance"]) <= 0.1)
                )
                raw_losses = raw_losses.at[
                    current_metric_index[f"{algorithm}_distance"]
                ].add(distance_loss)
                correct[f"{algorithm}_distance"] += distance_correct
                total[f"{algorithm}_distance"] += num_nodes

                predecessor_loss, _, predecessor_correct, predecessor_total = (
                    _pointer_loss_and_accuracy(predecessor_predictions, target["predecessor"])
                )
                raw_losses = raw_losses.at[
                    current_metric_index[f"{algorithm}_predecessor"]
                ].add(predecessor_loss)
                correct[f"{algorithm}_predecessor"] += predecessor_correct
                total[f"{algorithm}_predecessor"] += predecessor_total

            elif family == "mst":
                state_predictions, key_predictions, predecessor_predictions = output
                state_loss = nn.losses.binary_cross_entropy(
                    state_predictions,
                    target["state"],
                    reduction="mean",
                    with_logits=True,
                )
                state_pred = (mx.sigmoid(state_predictions) > 0.5).astype(mx.float32)
                state_correct = _int_item(mx.sum(state_pred == target["state"]))
                raw_losses = raw_losses.at[current_metric_index[f"{algorithm}_state"]].add(
                    state_loss
                )
                correct[f"{algorithm}_state"] += state_correct
                total[f"{algorithm}_state"] += num_nodes

                key_loss = nn.losses.mse_loss(
                    key_predictions,
                    target["key"],
                    reduction="mean",
                )
                key_correct = _int_item(
                    mx.sum(mx.abs(key_predictions - target["key"]) <= 0.1)
                )
                raw_losses = raw_losses.at[current_metric_index[f"{algorithm}_key"]].add(
                    key_loss
                )
                correct[f"{algorithm}_key"] += key_correct
                total[f"{algorithm}_key"] += num_nodes

                predecessor_loss, _, predecessor_correct, predecessor_total = (
                    _pointer_loss_and_accuracy(predecessor_predictions, target["predecessor"])
                )
                raw_losses = raw_losses.at[
                    current_metric_index[f"{algorithm}_predecessor"]
                ].add(predecessor_loss)
                correct[f"{algorithm}_predecessor"] += predecessor_correct
                total[f"{algorithm}_predecessor"] += predecessor_total

            else:
                raise ValueError(f"Unsupported algorithm family: {family}")

            termination_loss = nn.losses.binary_cross_entropy(
                termination_logits[algorithm],
                termination_targets[algorithm],
                reduction="mean",
                with_logits=True,
            )
            if termination_settings["mode"] == "distance" and not termination_settings["distance_signal"]:
                termination_loss = mx.array(0.0)
            raw_losses = raw_losses.at[
                current_metric_index[f"{algorithm}_termination"]
            ].add(termination_loss)
            termination_correct = _int_item(
                (mx.sigmoid(termination_logits[algorithm]) > 0.5).astype(mx.float32)
                == termination_targets[algorithm]
            )
            correct[f"{algorithm}_termination"] += termination_correct
            total[f"{algorithm}_termination"] += 1

        raw_losses = raw_losses * loss_scale
        accumulated_loss += mx.sum(raw_losses)
        accumulated_aux_losses += raw_losses
        previous_step_hidden_states = processed_embeddings
        _eval_trees(
            raw_losses,
            accumulated_loss,
            accumulated_aux_losses,
            previous_step_hidden_states,
            current_feature_values,
        )

    average_loss = accumulated_loss / effective_step_count(step_counts, sample_selected_tasks)
    avg_aux_losses = accumulated_aux_losses / metric_counters(step_counts, algorithm_order)
    accuracies = mx.array(
        [correct[name] / max(total[name], 1) for name in current_metric_names],
        dtype=mx.float32,
    )
    _eval_trees(average_loss, avg_aux_losses, accuracies)
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
    """Collect per-graph failure details under autoregressive evaluation."""
    del embedding_dim  # Public API compatibility.

    algorithm_order = tuple(getattr(model, "algorithms", ALGORITHMS))
    current_metric_names = metric_names(algorithm_order)
    selected_tasks = _normalize_selected_tasks(selected_tasks, algorithm_order)
    sample_selected_tasks = selected_tasks_for_graph(
        graph_data, selected_tasks, algorithm_order
    )
    termination_settings = resolve_termination_settings(termination_cfg)

    num_nodes = int(graph_data["num_nodes"])
    sequence_lengths = algorithm_target_lengths(graph_data, algorithm_order)
    step_counts = execution_step_counts(graph_data, algorithm_order)
    num_steps = max(step_counts.values(), default=0) + 1

    correct = {metric_name: 0 for metric_name in current_metric_names}
    total = {metric_name: 0 for metric_name in current_metric_names}
    termination_confusion = {
        f"{algorithm}_{kind}": 0
        for algorithm in algorithm_order
        for kind in ("fp", "fn")
    }
    termination_step_totals = {algorithm: {} for algorithm in algorithm_order}
    termination_step_mispredicts = {algorithm: {} for algorithm in algorithm_order}

    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])
    previous_distance_latent = None
    current_feature_values = feature_values_for_step(graph_data, 0, algorithm_order)
    step_failures = []
    first_failure_step = None

    for step_index in range(num_steps):
        step_payload = _forward_step(
            model=model,
            graph_data=graph_data,
            step_index=step_index,
            previous_step_hidden_states=previous_step_hidden_states,
            previous_distance_latent=previous_distance_latent,
            termination_settings=termination_settings,
            algorithm_order=algorithm_order,
            step_counts=step_counts,
            feature_values_override=current_feature_values,
        )
        if step_payload is None:
            continue

        sample_exists = step_payload["sample_exists"]
        targets = step_payload["targets"]
        termination_targets = step_payload["termination_targets"]
        algorithm_outputs = step_payload["algorithm_outputs"]
        termination_logits = step_payload["termination_logits"]
        processed_embeddings = step_payload["processed_embeddings"]
        previous_distance_latent = step_payload["next_distance_latent"]
        current_feature_values = step_payload["next_feature_values"]

        step_entry = {"step": int(step_index + 1)}
        for metric_name in current_metric_names:
            step_entry[f"{metric_name}_incorrect"] = 0

        for algorithm in algorithm_order:
            if not sample_exists[algorithm] or not sample_selected_tasks.get(algorithm, False):
                continue

            family = algorithm_family(algorithm)
            target = targets[algorithm]
            output = algorithm_outputs[algorithm]

            if family == "state_mask":
                state_pred = (mx.sigmoid(output) > 0.5).astype(mx.float32)
                state_correct = _int_item(mx.sum(state_pred == target["state"]))
                correct[f"{algorithm}_state"] += state_correct
                total[f"{algorithm}_state"] += num_nodes
                step_entry[f"{algorithm}_state_incorrect"] = int(num_nodes - state_correct)

            elif family == "shortest_path":
                distance_predictions, predecessor_predictions = output
                distance_correct = _int_item(
                    mx.sum(mx.abs(distance_predictions - target["distance"]) <= 0.1)
                )
                correct[f"{algorithm}_distance"] += distance_correct
                total[f"{algorithm}_distance"] += num_nodes
                step_entry[f"{algorithm}_distance_incorrect"] = int(num_nodes - distance_correct)

                _, predecessor_argmax, predecessor_correct, predecessor_total = _pointer_loss_and_accuracy(
                    predecessor_predictions,
                    target["predecessor"],
                )
                del predecessor_argmax
                correct[f"{algorithm}_predecessor"] += predecessor_correct
                total[f"{algorithm}_predecessor"] += predecessor_total
                step_entry[f"{algorithm}_predecessor_incorrect"] = int(
                    max(predecessor_total - predecessor_correct, 0)
                )

            elif family == "mst":
                state_predictions, key_predictions, predecessor_predictions = output
                state_pred = (mx.sigmoid(state_predictions) > 0.5).astype(mx.float32)
                state_correct = _int_item(mx.sum(state_pred == target["state"]))
                correct[f"{algorithm}_state"] += state_correct
                total[f"{algorithm}_state"] += num_nodes
                step_entry[f"{algorithm}_state_incorrect"] = int(num_nodes - state_correct)

                key_correct = _int_item(
                    mx.sum(mx.abs(key_predictions - target["key"]) <= 0.1)
                )
                correct[f"{algorithm}_key"] += key_correct
                total[f"{algorithm}_key"] += num_nodes
                step_entry[f"{algorithm}_key_incorrect"] = int(num_nodes - key_correct)

                _, predecessor_argmax, predecessor_correct, predecessor_total = _pointer_loss_and_accuracy(
                    predecessor_predictions,
                    target["predecessor"],
                )
                del predecessor_argmax
                correct[f"{algorithm}_predecessor"] += predecessor_correct
                total[f"{algorithm}_predecessor"] += predecessor_total
                step_entry[f"{algorithm}_predecessor_incorrect"] = int(
                    max(predecessor_total - predecessor_correct, 0)
                )

            else:
                raise ValueError(f"Unsupported algorithm family: {family}")

            termination_prob = _float_item(mx.sigmoid(termination_logits[algorithm]))
            termination_pred = 1 if termination_prob > 0.5 else 0
            termination_target = _int_item(termination_targets[algorithm])
            termination_correct = int(termination_pred == termination_target)
            correct[f"{algorithm}_termination"] += termination_correct
            total[f"{algorithm}_termination"] += 1
            step_entry[f"{algorithm}_termination_incorrect"] = 1 - termination_correct

            step_key = int(step_index + 1)
            termination_step_totals[algorithm][step_key] = (
                termination_step_totals[algorithm].get(step_key, 0) + 1
            )
            if step_entry[f"{algorithm}_termination_incorrect"] > 0:
                termination_step_mispredicts[algorithm][step_key] = (
                    termination_step_mispredicts[algorithm].get(step_key, 0) + 1
                )
            if termination_pred == 1 and termination_target == 0:
                termination_confusion[f"{algorithm}_fp"] += 1
            elif termination_pred == 0 and termination_target == 1:
                termination_confusion[f"{algorithm}_fn"] += 1

        step_error_units = sum(
            value for key, value in step_entry.items() if key.endswith("_incorrect")
        )
        if step_error_units > 0:
            if first_failure_step is None:
                first_failure_step = int(step_index + 1)
            if include_step_details:
                step_failures.append(step_entry)

        previous_step_hidden_states = processed_embeddings
        _eval_trees(previous_step_hidden_states, current_feature_values)

    accuracies = {
        metric_name: correct[metric_name] / max(total[metric_name], 1)
        for metric_name in current_metric_names
    }
    incorrect = {
        metric_name: total[metric_name] - correct[metric_name]
        for metric_name in current_metric_names
    }

    failed_tasks = []
    for metric_name in current_metric_names:
        algorithm = metric_name.rsplit("_", 1)[0]
        if metric_name.endswith("_predecessor") or metric_name.endswith("_distance"):
            algorithm = metric_name.rsplit("_", 1)[0]
        if metric_name.endswith("_termination"):
            algorithm = metric_name[: -len("_termination")]
        if metric_name.endswith("_state"):
            algorithm = metric_name[: -len("_state")]
        if metric_name.endswith("_key"):
            algorithm = metric_name[: -len("_key")]
        if sample_selected_tasks.get(algorithm, False) and incorrect[metric_name] > 0:
            failed_tasks.append(metric_name)

    return {
        "num_nodes": num_nodes,
        "sequence_lengths": {algorithm: int(length) for algorithm, length in sequence_lengths.items()},
        "execution_step_counts": {algorithm: int(count) for algorithm, count in step_counts.items()},
        "first_failure_step": first_failure_step,
        "accuracy": {k: float(v) for k, v in accuracies.items()},
        "incorrect": {k: int(v) for k, v in incorrect.items()},
        "termination_confusion": {k: int(v) for k, v in termination_confusion.items()},
        "termination_step_totals": {
            algorithm: {int(step): int(count) for step, count in values.items()}
            for algorithm, values in termination_step_totals.items()
        },
        "termination_step_mispredicts": {
            algorithm: {int(step): int(count) for step, count in values.items()}
            for algorithm, values in termination_step_mispredicts.items()
        },
        "active_task": graph_data.get("active_task"),
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
    algorithm_order = tuple(getattr(model, "algorithms", ALGORITHMS))
    current_metric_names = metric_names(algorithm_order)
    selected_tasks = _normalize_selected_tasks(selected_tasks, algorithm_order)
    termination_settings = resolve_termination_settings(termination_cfg)

    model.eval()
    graphs = dataset if max_graphs is None else dataset[: max(max_graphs, 0)]

    failures_by_task = {metric_name: [] for metric_name in current_metric_names}
    failed_graphs = []
    ranked_failed = []
    termination_step_totals = {algorithm: {} for algorithm in algorithm_order}
    termination_step_mispredicts = {algorithm: {} for algorithm in algorithm_order}

    for graph_index, graph_data in enumerate(graphs):
        details = _graph_failure_details(
            model=model,
            graph_data=graph_data,
            embedding_dim=embedding_dim,
            termination_cfg=termination_cfg,
            selected_tasks=selected_tasks,
            include_step_details=include_step_details,
        )

        for algorithm in algorithm_order:
            for step, count in details["termination_step_totals"][algorithm].items():
                step_key = int(step)
                termination_step_totals[algorithm][step_key] = (
                    termination_step_totals[algorithm].get(step_key, 0) + int(count)
                )
            for step, count in details["termination_step_mispredicts"][algorithm].items():
                step_key = int(step)
                termination_step_mispredicts[algorithm][step_key] = (
                    termination_step_mispredicts[algorithm].get(step_key, 0) + int(count)
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

    termination_step_rates = {algorithm: {} for algorithm in algorithm_order}
    for algorithm in algorithm_order:
        step_keys = sorted(set(termination_step_totals[algorithm].keys()))
        for step_key in step_keys:
            total = int(termination_step_totals[algorithm].get(step_key, 0))
            errors = int(termination_step_mispredicts[algorithm].get(step_key, 0))
            termination_step_rates[algorithm][step_key] = float(errors / max(total, 1))

    return {
        "termination": {
            "mode": termination_settings["mode"],
            "distance": termination_settings["distance_type"],
            "latent": termination_settings["distance_latent"],
            "threshold": float(termination_settings["distance_threshold"]),
            "distance_signal": bool(termination_settings["distance_signal"]),
        },
        "selected_tasks": selected_tasks,
        "algorithms": list(algorithm_order),
        "algorithm_display_names": {
            algorithm: algorithm_display_name(algorithm) for algorithm in algorithm_order
        },
        "num_graphs_analyzed": int(len(graphs)),
        "num_failed_graphs": int(len(ranked_failed_graph_indices)),
        "failure_rate": float(len(ranked_failed_graph_indices) / max(len(graphs), 1)),
        "failures_by_task": failures_by_task,
        "termination_mispredict_distribution": {
            algorithm: {
                "counts_by_step": {
                    int(step): int(count)
                    for step, count in sorted(termination_step_mispredicts[algorithm].items())
                },
                "totals_by_step": {
                    int(step): int(count)
                    for step, count in sorted(termination_step_totals[algorithm].items())
                },
                "rate_by_step": {
                    int(step): float(rate)
                    for step, rate in sorted(termination_step_rates[algorithm].items())
                },
            }
            for algorithm in algorithm_order
        },
        "ranked_failed_graph_indices": ranked_failed_graph_indices,
        "failed_graphs": failed_graphs,
    }


def safe_trained_model(
    model,
    sample_input_embeddings,
    sample_edge_matrix,
    output_path="trained_model.mlxfn",
):
    """Export the legacy BF/BFS/Prim model format for downstream consumers."""
    if tuple(getattr(model, "algorithms", ALGORITHMS)) != ALGORITHMS:
        raise ValueError(
            "safe_trained_model currently supports only the legacy "
            "bf/bfs/prim model layout."
        )

    mx.eval(model.parameters())

    def call(input_embeddings, edge_matrix):
        bfs_output, bf_output, prim_output, termination_probs, processed_embeddings = model(
            (input_embeddings, edge_matrix)
        )
        bf_distance_predictions, bf_predecessor_predictions = bf_output
        (
            prim_state_predictions,
            prim_key_predictions,
            prim_predecessor_predictions,
        ) = prim_output

        return (
            bfs_output,
            bf_distance_predictions,
            bf_predecessor_predictions,
            prim_state_predictions,
            prim_key_predictions,
            prim_predecessor_predictions,
            termination_probs["bf"],
            termination_probs["bfs"],
            termination_probs["prim"],
            processed_embeddings,
        )

    mx.export_function(output_path, call, (sample_input_embeddings, sample_edge_matrix))
