"""Visualize algorithmic state dynamics during extra execution steps.

This script inspects model-generated algorithmic states over execution and
extra fake-continuation steps, then summarizes whether extra-step dynamics
look fixed-point-like or oscillatory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from src.analysis.common import (
    iter_execution_feature_values,
    load_analysis_dataset,
    load_model_from_checkpoint,
    resolve_checkpoint_path,
    resolve_config,
    resolve_dataset_path,
)
from src.utils.task_specs import (
    build_node_algo_features,
    execution_step_counts,
    normalize_algorithm_order,
)
from src.utils.termination import (
    compute_distance_termination_logits,
    get_distance_latent,
    needs_aux_latents,
    resolve_termination_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze predicted algorithmic states across execution + extra steps, "
            "and visualize whether extra-step behavior is stable or oscillatory."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory with config_resolved.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file/dir (defaults to latest in run-dir).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to use when --dataset is omitted.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path.")
    parser.add_argument(
        "--graph-index",
        type=int,
        required=True,
        help="Graph index to inspect.",
    )
    parser.add_argument(
        "--extra-steps",
        type=int,
        default=8,
        help="Number of fake-continuation steps after termination.",
    )
    parser.add_argument(
        "--fixed-tol",
        type=float,
        default=1e-3,
        help="Threshold on extra-step mean state delta for fixed-point classification.",
    )
    parser.add_argument(
        "--period2-ratio",
        type=float,
        default=0.65,
        help=(
            "Period-2 criterion: mean(lag-2 delta) <= period2-ratio * "
            "mean(consecutive delta) in extra region."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir for plot + JSON (default: run-dir/analysis/extra_step_state_dynamics).",
    )
    return parser.parse_args()


def count_base_steps(graph_data: dict, algorithm_order: tuple[str, ...]) -> int:
    return max(execution_step_counts(graph_data, algorithm_order).values(), default=0)


def _mean_l2_per_node(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if diff.ndim == 1:
        return float(np.linalg.norm(diff))
    return float(np.linalg.norm(diff, axis=1).mean())


def classify_extra_dynamics(
    extra_consecutive_state_deltas: List[float],
    extra_lag2_state_deltas: List[float],
    fixed_tol: float,
    period2_ratio: float,
) -> Dict[str, Any]:
    if not extra_consecutive_state_deltas:
        return {
            "label": "insufficient",
            "assessment": "unlikely",
            "reason": "Need at least two extra states to measure consecutive changes.",
        }

    mean_cons = float(np.mean(extra_consecutive_state_deltas))
    mean_lag2 = (
        float(np.mean(extra_lag2_state_deltas))
        if extra_lag2_state_deltas
        else None
    )

    if mean_cons <= fixed_tol:
        return {
            "label": "fixed_point_like",
            "assessment": "almost certain",
            "reason": (
                f"Mean extra-step consecutive state delta ({mean_cons:.6f}) "
                f"is <= fixed_tol ({fixed_tol:.6f})."
            ),
        }

    if mean_lag2 is not None and mean_lag2 <= period2_ratio * mean_cons:
        return {
            "label": "period2_like",
            "assessment": "very probable",
            "reason": (
                f"Mean lag-2 delta ({mean_lag2:.6f}) is <= "
                f"{period2_ratio:.3f} * mean consecutive delta ({mean_cons:.6f})."
            ),
        }

    return {
        "label": "drift_or_higher_period",
        "assessment": "plausible",
        "reason": (
            f"Extra-step consecutive state deltas remain above fixed_tol "
            f"(mean={mean_cons:.6f}) without strong period-2 signature."
        ),
    }


def plot_dynamics(
    *,
    output_path: Path,
    per_step: List[Dict[str, Any]],
    consecutive_state_deltas: List[float],
    bfs_flip_rates: List[float],
    bf_pred_change_rates: List[float],
    prim_flip_rates: List[float],
    prim_pred_change_rates: List[float],
    hidden_deltas: List[float],
    base_steps: int,
    title: str,
) -> None:
    steps = np.array([entry["step"] for entry in per_step], dtype=np.int32)

    bfs_active_fraction = np.array(
        [entry["bfs_active_fraction"] for entry in per_step], dtype=np.float64
    )
    bf_distance_mean = np.array(
        [entry["bf_distance_mean"] for entry in per_step], dtype=np.float64
    )
    bf_distance_std = np.array(
        [entry["bf_distance_std"] for entry in per_step], dtype=np.float64
    )
    prim_active_fraction = np.array(
        [entry["prim_active_fraction"] for entry in per_step], dtype=np.float64
    )
    prim_key_mean = np.array(
        [entry["prim_key_mean"] for entry in per_step], dtype=np.float64
    )
    prim_key_std = np.array(
        [entry["prim_key_std"] for entry in per_step], dtype=np.float64
    )
    bf_term_prob = np.array(
        [entry["bf_termination_prob"] for entry in per_step], dtype=np.float64
    )
    bfs_term_prob = np.array(
        [entry["bfs_termination_prob"] for entry in per_step], dtype=np.float64
    )
    prim_term_prob = np.array(
        [entry["prim_termination_prob"] for entry in per_step], dtype=np.float64
    )

    trans_steps = np.arange(2, len(per_step) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax0 = axes[0]
    ax0.plot(steps, bfs_active_fraction, marker="o", linewidth=1.5, label="BFS active frac")
    ax0.plot(steps, bf_distance_mean, marker="o", linewidth=1.5, label="BF distance mean")
    ax0.plot(steps, bf_distance_std, marker="o", linewidth=1.2, label="BF distance std")
    ax0.plot(steps, prim_active_fraction, marker="o", linewidth=1.2, label="Prim active frac")
    ax0.plot(steps, prim_key_mean, marker="o", linewidth=1.2, label="Prim key mean")
    ax0.plot(steps, prim_key_std, marker="o", linewidth=1.0, label="Prim key std")
    ax0.set_ylabel("State magnitude")
    ax0.grid(alpha=0.3)
    ax0.legend(loc="best", fontsize=8)

    ax1 = axes[1]
    ax1.plot(steps, bf_term_prob, marker="o", linewidth=1.4, label="BF term prob")
    ax1.plot(steps, bfs_term_prob, marker="o", linewidth=1.4, label="BFS term prob")
    ax1.plot(steps, prim_term_prob, marker="o", linewidth=1.4, label="Prim term prob")
    ax1.set_ylabel("Termination prob")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    ax2 = axes[2]
    ax2.plot(
        trans_steps,
        consecutive_state_deltas,
        marker="o",
        linewidth=1.5,
        label="state delta (consecutive)",
    )
    ax2.plot(trans_steps, hidden_deltas, marker="o", linewidth=1.2, label="hidden delta")
    ax2.plot(trans_steps, bfs_flip_rates, marker="o", linewidth=1.2, label="BFS flip rate")
    ax2.plot(
        trans_steps,
        bf_pred_change_rates,
        marker="o",
        linewidth=1.2,
        label="BF predecessor change rate",
    )
    ax2.plot(trans_steps, prim_flip_rates, marker="o", linewidth=1.2, label="Prim flip rate")
    ax2.plot(
        trans_steps,
        prim_pred_change_rates,
        marker="o",
        linewidth=1.2,
        label="Prim predecessor change rate",
    )
    ax2.set_ylabel("Change metric")
    ax2.set_xlabel("Execution step")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    boundary = base_steps + 0.5
    for ax in axes:
        ax.axvline(boundary, linestyle="--", linewidth=1.0, alpha=0.8, color="#d62728")
        ax.text(
            boundary + 0.03,
            0.97,
            "extra starts",
            transform=ax.get_xaxis_transform(),
            color="#d62728",
            fontsize=8,
            va="top",
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.extra_steps <= 0:
        raise ValueError("--extra-steps must be > 0 for extra-step state analysis.")
    if args.fixed_tol <= 0:
        raise ValueError("--fixed-tol must be > 0.")
    if args.period2_ratio <= 0:
        raise ValueError("--period2-ratio must be > 0.")

    config, run_dir = resolve_config(args.config, args.run_dir)
    algorithm_order = normalize_algorithm_order(config.model.algorithms)
    required_algorithms = ("bf", "bfs", "prim")
    missing = [algorithm for algorithm in required_algorithms if algorithm not in algorithm_order]
    if missing:
        raise ValueError(
            "extra_step_state_dynamics requires BF/BFS/Prim heads. Missing: "
            + ", ".join(missing)
        )
    dataset_path = resolve_dataset_path(
        args.dataset, args.split, config, algorithm_order=algorithm_order
    )
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, run_dir)

    model, checkpoint_step = load_model_from_checkpoint(config, checkpoint_path, run_dir)
    model.eval()

    dataset = load_analysis_dataset(dataset_path, algorithm_order)
    if args.graph_index < 0 or args.graph_index >= len(dataset):
        raise IndexError(
            f"--graph-index out of range: {args.graph_index} (dataset size={len(dataset)})"
        )
    graph_data = dataset[args.graph_index]
    num_nodes = int(graph_data["num_nodes"])
    base_steps = count_base_steps(graph_data, algorithm_order)

    termination_settings = resolve_termination_settings(config.model)
    need_aux = needs_aux_latents(termination_settings)

    previous_hidden = mx.zeros([num_nodes, model.processor_embed_dim])
    previous_distance_latent = None

    per_step: List[Dict[str, Any]] = []
    bfs_preds: List[np.ndarray] = []
    bf_dist_preds: List[np.ndarray] = []
    bf_pred_args: List[np.ndarray] = []
    prim_state_preds: List[np.ndarray] = []
    prim_key_preds: List[np.ndarray] = []
    prim_pred_args: List[np.ndarray] = []
    hidden_states: List[np.ndarray] = []
    flat_states: List[np.ndarray] = []

    for step_index, feature_values in enumerate(
        iter_execution_feature_values(graph_data, args.extra_steps, algorithm_order), start=1
    ):
        node_algo_features = build_node_algo_features(feature_values, algorithm_order)
        input_embeddings = mx.concatenate([previous_hidden, node_algo_features], axis=1)
        model_input = (input_embeddings, graph_data["edge_matrix"])

        if need_aux:
            forward_output = model(model_input, return_latents=True)
            if isinstance(forward_output[0], dict):
                algorithm_outputs, termination_probs, processed_embeddings, aux = forward_output
            else:
                bfs_output, bf_output, prim_output, termination_probs, processed_embeddings, aux = (
                    forward_output
                )
                algorithm_outputs = {
                    "bfs": bfs_output,
                    "bf": bf_output,
                    "prim": prim_output,
                }
        else:
            forward_output = model(model_input)
            if isinstance(forward_output[0], dict):
                algorithm_outputs, termination_probs, processed_embeddings = forward_output
            else:
                bfs_output, bf_output, prim_output, termination_probs, processed_embeddings = (
                    forward_output
                )
                algorithm_outputs = {
                    "bfs": bfs_output,
                    "bf": bf_output,
                    "prim": prim_output,
                }
            aux = None

        if termination_settings["mode"] == "distance":
            current_latent = get_distance_latent(termination_settings, processed_embeddings, aux)
            termination_logits = compute_distance_termination_logits(
                settings=termination_settings,
                prev_latent=previous_distance_latent,
                current_latent=current_latent,
                algorithms=algorithm_order,
            )
            previous_distance_latent = current_latent
        else:
            termination_logits = termination_probs

        bfs_output = algorithm_outputs["bfs"]
        bf_output = algorithm_outputs["bf"]
        prim_output = algorithm_outputs["prim"]
        bf_distance_pred, bf_pred_logits = bf_output
        prim_state_pred_logits, prim_key_pred, prim_pred_logits = prim_output
        bfs_pred = (mx.sigmoid(bfs_output) > 0.5).astype(mx.float32)
        prim_state_pred = (mx.sigmoid(prim_state_pred_logits) > 0.5).astype(mx.float32)
        bf_pred_arg = mx.argmax(bf_pred_logits, axis=-1)
        prim_pred_arg = mx.argmax(prim_pred_logits, axis=-1)

        bfs_np = np.array(bfs_pred, copy=False).astype(np.int32, copy=False)
        bf_dist_np = np.array(bf_distance_pred, copy=False).astype(np.float64, copy=False)
        bf_pred_arg_np = np.array(bf_pred_arg, copy=False).astype(np.int32, copy=False)
        prim_state_np = np.array(prim_state_pred, copy=False).astype(np.int32, copy=False)
        prim_key_np = np.array(prim_key_pred, copy=False).astype(np.float64, copy=False)
        prim_pred_arg_np = np.array(prim_pred_arg, copy=False).astype(np.int32, copy=False)
        hidden_np = np.array(processed_embeddings, copy=False).astype(np.float64, copy=False)

        bfs_preds.append(bfs_np)
        bf_dist_preds.append(bf_dist_np)
        bf_pred_args.append(bf_pred_arg_np)
        prim_state_preds.append(prim_state_np)
        prim_key_preds.append(prim_key_np)
        prim_pred_args.append(prim_pred_arg_np)
        hidden_states.append(hidden_np)

        bf_arg_norm = bf_pred_arg_np.astype(np.float64) / max(num_nodes - 1, 1)
        prim_arg_norm = prim_pred_arg_np.astype(np.float64) / max(num_nodes - 1, 1)
        flat_states.append(
            np.concatenate(
                [
                    bf_dist_np,
                    bfs_np.astype(np.float64),
                    bf_arg_norm,
                    prim_state_np.astype(np.float64),
                    prim_key_np,
                    prim_arg_norm,
                ],
                axis=0,
            )
        )

        per_step.append(
            {
                "step": int(step_index),
                "is_extra": bool(step_index > base_steps),
                "bfs_active_fraction": float(np.mean(bfs_np)),
                "bf_distance_mean": float(np.mean(bf_dist_np)),
                "bf_distance_std": float(np.std(bf_dist_np)),
                "prim_active_fraction": float(np.mean(prim_state_np)),
                "prim_key_mean": float(np.mean(prim_key_np)),
                "prim_key_std": float(np.std(prim_key_np)),
                "bf_termination_prob": float(mx.sigmoid(termination_logits["bf"]).item()),
                "bfs_termination_prob": float(mx.sigmoid(termination_logits["bfs"]).item()),
                "prim_termination_prob": float(mx.sigmoid(termination_logits["prim"]).item()),
                "hidden_norm": float(np.linalg.norm(hidden_np)),
            }
        )
        previous_hidden = processed_embeddings

    consecutive_state_deltas: List[float] = []
    bfs_flip_rates: List[float] = []
    bf_pred_change_rates: List[float] = []
    prim_flip_rates: List[float] = []
    prim_pred_change_rates: List[float] = []
    hidden_deltas: List[float] = []
    lag2_state_deltas: List[float | None] = [None]

    extra_consecutive_state_deltas: List[float] = []
    extra_lag2_state_deltas: List[float] = []

    for i in range(1, len(per_step)):
        delta_state = float(np.linalg.norm(flat_states[i] - flat_states[i - 1]) / np.sqrt(flat_states[i].shape[0]))
        delta_hidden = _mean_l2_per_node(hidden_states[i], hidden_states[i - 1])
        bfs_flip = float(np.mean(bfs_preds[i] != bfs_preds[i - 1]))
        bf_pred_change = float(np.mean(bf_pred_args[i] != bf_pred_args[i - 1]))
        prim_flip = float(np.mean(prim_state_preds[i] != prim_state_preds[i - 1]))
        prim_pred_change = float(np.mean(prim_pred_args[i] != prim_pred_args[i - 1]))

        consecutive_state_deltas.append(delta_state)
        hidden_deltas.append(delta_hidden)
        bfs_flip_rates.append(bfs_flip)
        bf_pred_change_rates.append(bf_pred_change)
        prim_flip_rates.append(prim_flip)
        prim_pred_change_rates.append(prim_pred_change)

        curr_is_extra = bool(per_step[i]["is_extra"])
        prev_is_extra = bool(per_step[i - 1]["is_extra"])
        if curr_is_extra and prev_is_extra:
            extra_consecutive_state_deltas.append(delta_state)

        if i >= 2:
            lag2 = float(np.linalg.norm(flat_states[i] - flat_states[i - 2]) / np.sqrt(flat_states[i].shape[0]))
            lag2_state_deltas.append(lag2)
            if curr_is_extra and bool(per_step[i - 2]["is_extra"]):
                extra_lag2_state_deltas.append(lag2)
        else:
            lag2_state_deltas.append(None)

    classification = classify_extra_dynamics(
        extra_consecutive_state_deltas=extra_consecutive_state_deltas,
        extra_lag2_state_deltas=extra_lag2_state_deltas,
        fixed_tol=args.fixed_tol,
        period2_ratio=args.period2_ratio,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            run_dir / "analysis" / "extra_step_state_dynamics"
            if run_dir
            else Path("analysis/extra_step_state_dynamics")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    title = (
        f"Extra-step algorithmic state dynamics (graph {args.graph_index}, "
        f"base={base_steps}, extra={args.extra_steps})"
    )
    plot_path = output_dir / f"graph_{args.graph_index}_extra_state_dynamics.png"
    plot_dynamics(
        output_path=plot_path,
        per_step=per_step,
        consecutive_state_deltas=consecutive_state_deltas,
        bfs_flip_rates=bfs_flip_rates,
        bf_pred_change_rates=bf_pred_change_rates,
        prim_flip_rates=prim_flip_rates,
        prim_pred_change_rates=prim_pred_change_rates,
        hidden_deltas=hidden_deltas,
        base_steps=base_steps,
        title=title,
    )

    transitions = []
    for i in range(1, len(per_step)):
        transitions.append(
            {
                "from_step": int(per_step[i - 1]["step"]),
                "to_step": int(per_step[i]["step"]),
                "from_is_extra": bool(per_step[i - 1]["is_extra"]),
                "to_is_extra": bool(per_step[i]["is_extra"]),
                "state_delta_l2": float(consecutive_state_deltas[i - 1]),
                "hidden_delta_mean_l2": float(hidden_deltas[i - 1]),
                "bfs_flip_rate": float(bfs_flip_rates[i - 1]),
                "bf_predecessor_change_rate": float(bf_pred_change_rates[i - 1]),
                "prim_flip_rate": float(prim_flip_rates[i - 1]),
                "prim_predecessor_change_rate": float(prim_pred_change_rates[i - 1]),
                "lag2_state_delta_l2": (
                    None if lag2_state_deltas[i] is None else float(lag2_state_deltas[i])
                ),
            }
        )

    report = {
        "metadata": {
            "config_name": config.name,
            "checkpoint_step": checkpoint_step,
            "run_dir": str(run_dir) if run_dir else None,
            "dataset": str(dataset_path),
            "split": args.split if args.dataset is None else None,
            "graph_index": int(args.graph_index),
            "num_nodes": int(num_nodes),
            "termination_mode": termination_settings["mode"],
            "termination_distance": termination_settings["distance_type"],
            "termination_latent": termination_settings["distance_latent"],
            "base_steps": int(base_steps),
            "extra_steps": int(args.extra_steps),
            "fixed_tol": float(args.fixed_tol),
            "period2_ratio": float(args.period2_ratio),
        },
        "classification": classification,
        "per_step": per_step,
        "transitions": transitions,
        "extra_region_summary": {
            "num_extra_states": int(sum(1 for s in per_step if s["is_extra"])),
            "num_extra_consecutive_transitions": int(len(extra_consecutive_state_deltas)),
            "mean_extra_consecutive_state_delta": (
                None
                if not extra_consecutive_state_deltas
                else float(np.mean(extra_consecutive_state_deltas))
            ),
            "mean_extra_lag2_state_delta": (
                None
                if not extra_lag2_state_deltas
                else float(np.mean(extra_lag2_state_deltas))
            ),
        },
        "plot_path": str(plot_path),
    }

    json_path = output_dir / f"graph_{args.graph_index}_extra_state_dynamics.json"
    json_path.write_text(json.dumps(report, indent=2))

    print(f"Saved plot to: {plot_path}")
    print(f"Saved report to: {json_path}")
    print(
        "Classification: "
        f"{classification['label']} ({classification['assessment']}) - {classification['reason']}"
    )


if __name__ == "__main__":
    main()
