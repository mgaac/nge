"""PCA over simultaneous autoregressive rollout segments grouped by active branch count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.algorithm_subspace import pca_fit_transform, plot_algorithm_delta_pca
from src.utils.task_specs import (
    ANALYSIS_LATENT_CHOICES,
    normalize_algorithm_order,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all configured algorithm branches simultaneously in autoregressive mode, "
            "segment the latent execution trace by the number of branches still active, "
            "and fit a separate PCA in each segment."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
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
        help="Checkpoint directory or file to load (defaults to latest in run-dir).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to use when --dataset is not provided.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset path (.npz). If omitted, analyze all unique split datasets in the config.",
    )
    parser.add_argument(
        "--graph-index",
        type=int,
        default=None,
        help="Analyze a single graph by index. Only valid when exactly one dataset source is used.",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap on the number of graphs analyzed after dataset loading.",
    )
    parser.add_argument(
        "--latent",
        type=str,
        choices=ANALYSIS_LATENT_CHOICES,
        default="processed",
        help="Which latent representation to analyze.",
    )
    parser.add_argument(
        "--node-agg",
        type=str,
        choices=["max", "min", "mean"],
        default="max",
        help="Aggregation over node dimension to form per-step embeddings.",
    )
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=64,
        help="Safety cap on autoregressive rollout length per graph.",
    )
    parser.add_argument(
        "--activity-source",
        type=str,
        choices=["state_change", "termination"],
        default="state_change",
        help=(
            "How to decide whether a branch remains active at the next step: "
            "branch-specific recurrent state change or predicted termination."
        ),
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=1.0e-4,
        help="Absolute tolerance used when --activity-source=state_change.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=2,
        help="Number of PCA components to fit per execution segment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write outputs. Default: <run-dir>/analysis/execution_segment_pca"
        ),
    )
    return parser.parse_args()


def aggregate_nodes(latent: np.ndarray, node_agg: str) -> np.ndarray:
    if node_agg == "max":
        return latent.max(axis=0)
    if node_agg == "min":
        return latent.min(axis=0)
    if node_agg == "mean":
        return latent.mean(axis=0)
    raise ValueError(f"Unknown node aggregation: {node_agg}")


def resolve_dataset_paths(
    config: Any,
    dataset_override: str | None,
    split: str,
    algorithm_order: Sequence[str],
) -> list[Path]:
    if dataset_override is not None:
        return [Path(dataset_override)]

    if config.data.task_paths:
        split_key = f"{split}_path"
        paths: list[Path] = []
        seen: set[str] = set()
        for algorithm in algorithm_order:
            if algorithm not in config.data.task_paths:
                continue
            raw_path = str(config.data.task_paths[algorithm][split_key])
            if raw_path in seen:
                continue
            seen.add(raw_path)
            paths.append(Path(raw_path))
        if paths:
            return paths

    if split == "train":
        return [Path(config.data.train_path)]
    if split == "val":
        return [Path(config.data.val_path)]
    if split == "test":
        return [Path(config.data.test_path)]
    raise ValueError(f"Unknown split: {split}")


def load_dataset_sources(
    dataset_paths: Sequence[Path],
    algorithm_order: Sequence[str],
) -> list[dict[str, Any]]:
    from src.analysis.common import load_analysis_dataset

    sources: list[dict[str, Any]] = []
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        sources.append(
            {
                "path": dataset_path,
                "graphs": load_analysis_dataset(dataset_path, algorithm_order),
            }
        )
    return sources


def select_graphs(
    dataset_sources: Sequence[dict[str, Any]],
    graph_index: int | None,
    max_graphs: int | None,
) -> list[dict[str, Any]]:
    if graph_index is not None:
        if len(dataset_sources) != 1:
            raise ValueError("--graph-index requires exactly one dataset source.")
        graphs = dataset_sources[0]["graphs"]
        if graph_index < 0 or graph_index >= len(graphs):
            raise IndexError(
                f"--graph-index out of range: {graph_index} (dataset size={len(graphs)})"
            )
        return [
            {
                "dataset_path": dataset_sources[0]["path"],
                "dataset_graph_index": int(graph_index),
                "graph": graphs[graph_index],
            }
        ]

    selected: list[dict[str, Any]] = []
    for source in dataset_sources:
        dataset_path = source["path"]
        for dataset_graph_index, graph in enumerate(source["graphs"]):
            selected.append(
                {
                    "dataset_path": dataset_path,
                    "dataset_graph_index": int(dataset_graph_index),
                    "graph": graph,
                }
            )
            if max_graphs is not None and len(selected) >= max_graphs:
                return selected
    return selected


def latent_from_forward_pass(
    latent_kind: str,
    processed_embeddings: Any,
    encoded_embeddings: Any,
    encoded_by_algorithm: Dict[str, Any],
    algorithm_order: Sequence[str],
) -> Any:
    if latent_kind == "processed":
        return processed_embeddings
    if latent_kind == "encoded":
        return encoded_embeddings
    if latent_kind.startswith("encoded_"):
        algorithm = latent_kind[len("encoded_") :]
        if algorithm not in encoded_by_algorithm:
            raise ValueError(
                f"Latent '{latent_kind}' is unavailable for algorithms {algorithm_order}."
            )
        return encoded_by_algorithm[algorithm]
    if latent_kind.startswith("processed_zero_") and latent_kind.endswith("_input"):
        return processed_embeddings
    raise ValueError(f"Unknown latent kind: {latent_kind}")


def decode_algorithm_outputs(
    model: Any,
    processed_embeddings: Any,
    encoded_by_algorithm: Dict[str, Any],
    edge_matrix: Any,
) -> Dict[str, Any]:
    from src.utils.task_specs import algorithm_family

    outputs: Dict[str, Any] = {}
    for algorithm in model.algorithms:
        family = algorithm_family(algorithm)
        decoder = getattr(model, f"{algorithm}_decoder")
        encoded = encoded_by_algorithm[algorithm]
        if family == "state_mask":
            outputs[algorithm] = decoder((processed_embeddings, encoded))
        else:
            outputs[algorithm] = decoder((processed_embeddings, encoded, edge_matrix))
    return outputs


def termination_logits_for_step(
    model: Any,
    processed_embeddings: Any,
    encoded_embeddings: Any,
    encoded_by_algorithm: Dict[str, Any],
    termination_settings: Dict[str, Any],
    previous_distance_latent: Any | None,
) -> tuple[Dict[str, Any], Any | None]:
    from src.utils.termination import (
        compute_distance_termination_logits,
        get_distance_latent,
        needs_aux_latents,
    )

    import mlx.core as mx

    avg_processed = mx.mean(processed_embeddings, axis=0)
    aux = {
        "encoded": encoded_embeddings,
        "avg_processed": avg_processed,
    }
    aux.update(
        {
            f"{algorithm}_encoded": encoded_by_algorithm[algorithm]
            for algorithm in model.algorithms
        }
    )

    if termination_settings["mode"] == "distance":
        current_distance_latent = get_distance_latent(
            termination_settings,
            processed_embeddings,
            aux if needs_aux_latents(termination_settings) else None,
        )
        logits = compute_distance_termination_logits(
            settings=termination_settings,
            prev_latent=previous_distance_latent,
            current_latent=current_distance_latent,
            algorithms=model.algorithms,
        )
        return logits, current_distance_latent

    logits = {}
    for algorithm in model.algorithms:
        termination_head = getattr(model, f"{algorithm}_termination")
        logits[algorithm] = termination_head(avg_processed).squeeze()
    return logits, previous_distance_latent


def collect_simultaneous_rollout(
    model: Any,
    graph_data: dict[str, Any],
    latent_kind: str,
    node_agg: str,
    max_rollout_steps: int,
    termination_cfg: Any,
    activity_source: str,
    activity_threshold: float,
) -> dict[str, Any]:
    from src.analysis.common import compute_forward_latents
    from src.utils.eval import _predicted_feature_values_for_next_step
    from src.utils.task_specs import (
        algorithm_family,
        build_node_algo_features,
        feature_values_for_step,
    )
    from src.utils.termination import resolve_termination_settings

    import mlx.core as mx

    if max_rollout_steps <= 0:
        raise ValueError("--max-rollout-steps must be positive.")

    algorithm_order = tuple(model.algorithms)
    num_nodes = int(graph_data["num_nodes"])
    edge_matrix = graph_data["edge_matrix"]
    previous_step_hidden_states = mx.zeros([num_nodes, model.processor_embed_dim])
    current_feature_values = feature_values_for_step(graph_data, 0, algorithm_order)
    previous_distance_latent = None
    termination_settings = resolve_termination_settings(termination_cfg)
    active_mask = {algorithm: True for algorithm in algorithm_order}

    step_embeddings: List[np.ndarray] = []
    step_indices: List[int] = []
    active_counts: List[int] = []
    active_signatures: List[str] = []
    inactive_steps: Dict[str, int | None] = {algorithm: None for algorithm in algorithm_order}

    for step_index in range(max_rollout_steps):
        node_algo_features = build_node_algo_features(current_feature_values, algorithm_order)
        input_embeddings = mx.concatenate(
            [previous_step_hidden_states, node_algo_features],
            axis=1,
        )

        zero_input_algorithms: tuple[str, ...] = ()
        if latent_kind.startswith("processed_zero_") and latent_kind.endswith("_input"):
            algorithm = latent_kind[len("processed_zero_") : -len("_input")]
            if algorithm not in algorithm_order:
                raise ValueError(
                    f"Latent '{latent_kind}' is unavailable for algorithms {algorithm_order}."
                )
            zero_input_algorithms = (algorithm,)

        processed_embeddings, encoded_embeddings, encoded_by_algorithm = compute_forward_latents(
            model,
            input_embeddings,
            edge_matrix,
            zero_input_algorithms=zero_input_algorithms,
        )
        latent = latent_from_forward_pass(
            latent_kind=latent_kind,
            processed_embeddings=processed_embeddings,
            encoded_embeddings=encoded_embeddings,
            encoded_by_algorithm=encoded_by_algorithm,
            algorithm_order=algorithm_order,
        )
        algorithm_outputs = decode_algorithm_outputs(
            model=model,
            processed_embeddings=processed_embeddings,
            encoded_by_algorithm=encoded_by_algorithm,
            edge_matrix=edge_matrix,
        )
        termination_logits, previous_distance_latent = termination_logits_for_step(
            model=model,
            processed_embeddings=processed_embeddings,
            encoded_embeddings=encoded_embeddings,
            encoded_by_algorithm=encoded_by_algorithm,
            termination_settings=termination_settings,
            previous_distance_latent=previous_distance_latent,
        )

        reduced = aggregate_nodes(
            np.asarray(latent, dtype=np.float32),
            node_agg=node_agg,
        )
        active_signature = tuple(
            algorithm for algorithm in algorithm_order if active_mask[algorithm]
        )
        step_embeddings.append(reduced.astype(np.float32, copy=False))
        step_indices.append(int(step_index))
        active_counts.append(len(active_signature))
        active_signatures.append("+".join(active_signature))

        previous_feature_values = current_feature_values
        current_feature_values = _predicted_feature_values_for_next_step(
            algorithm_outputs,
            algorithm_order,
        )
        previous_step_hidden_states = processed_embeddings

        if activity_source == "termination":
            deactivate_now = {
                algorithm: bool(
                    active_mask[algorithm]
                    and float(mx.sigmoid(termination_logits[algorithm]).item()) > 0.5
                )
                for algorithm in algorithm_order
            }
        else:
            deactivate_now = {}
            for algorithm in algorithm_order:
                if not active_mask[algorithm]:
                    deactivate_now[algorithm] = False
                    continue

                family = algorithm_family(algorithm)
                if family == "state_mask":
                    current_value = current_feature_values[f"{algorithm}_state"]
                    previous_value = previous_feature_values[f"{algorithm}_state"]
                elif family == "shortest_path":
                    current_value = current_feature_values[f"{algorithm}_distance"]
                    previous_value = previous_feature_values[f"{algorithm}_distance"]
                elif family == "mst":
                    current_state = current_feature_values[f"{algorithm}_state"]
                    current_key = current_feature_values[f"{algorithm}_key"]
                    previous_state = previous_feature_values[f"{algorithm}_state"]
                    previous_key = previous_feature_values[f"{algorithm}_key"]
                    state_delta = float(mx.max(mx.abs(current_state - previous_state)).item())
                    key_delta = float(mx.max(mx.abs(current_key - previous_key)).item())
                    deactivate_now[algorithm] = max(state_delta, key_delta) <= activity_threshold
                    continue
                else:
                    raise ValueError(f"Unsupported algorithm family: {family}")

                delta = float(mx.max(mx.abs(current_value - previous_value)).item())
                deactivate_now[algorithm] = delta <= activity_threshold

        for algorithm in algorithm_order:
            if deactivate_now[algorithm] and inactive_steps[algorithm] is None:
                inactive_steps[algorithm] = int(step_index)
        active_mask = {
            algorithm: active_mask[algorithm] and not deactivate_now[algorithm]
            for algorithm in algorithm_order
        }
        if not any(active_mask.values()):
            break

    if not step_embeddings:
        return {
            "embeddings": np.empty((0, model.processor_embed_dim), dtype=np.float32),
            "step_indices": np.empty((0,), dtype=np.int32),
            "active_counts": np.empty((0,), dtype=np.int32),
            "active_signatures": np.empty((0,), dtype=object),
            "inactive_steps": inactive_steps,
        }

    return {
        "embeddings": np.stack(step_embeddings, axis=0).astype(np.float32, copy=False),
        "step_indices": np.asarray(step_indices, dtype=np.int32),
        "active_counts": np.asarray(active_counts, dtype=np.int32),
        "active_signatures": np.asarray(active_signatures, dtype=object),
        "inactive_steps": inactive_steps,
    }


def build_segment_sets(rollouts: Sequence[dict[str, Any]]) -> Dict[int, dict[str, np.ndarray]]:
    segment_sets: Dict[int, dict[str, List[Any]]] = {}
    for rollout_index, rollout in enumerate(rollouts):
        embeddings = rollout["embeddings"]
        step_indices = rollout["step_indices"]
        active_counts = rollout["active_counts"]
        active_signatures = rollout["active_signatures"]
        source_labels = rollout["source_labels"]

        for step_row in range(int(embeddings.shape[0])):
            active_count = int(active_counts[step_row])
            if active_count <= 0:
                continue
            entry = segment_sets.setdefault(
                active_count,
                {
                    "vectors": [],
                    "rollout_labels": [],
                    "step_labels": [],
                    "active_signatures": [],
                    "source_labels": [],
                },
            )
            entry["vectors"].append(embeddings[step_row])
            entry["rollout_labels"].append(int(rollout_index))
            entry["step_labels"].append(int(step_indices[step_row]))
            entry["active_signatures"].append(str(active_signatures[step_row]))
            entry["source_labels"].append(str(source_labels[step_row]))

    finalized: Dict[int, dict[str, np.ndarray]] = {}
    for active_count, entry in segment_sets.items():
        finalized[active_count] = {
            "vectors": np.stack(entry["vectors"], axis=0).astype(np.float32, copy=False),
            "rollout_labels": np.asarray(entry["rollout_labels"], dtype=np.int32),
            "step_labels": np.asarray(entry["step_labels"], dtype=np.int32),
            "active_signatures": np.asarray(entry["active_signatures"], dtype=object),
            "source_labels": np.asarray(entry["source_labels"], dtype=object),
        }
    return finalized


def plot_segment_explained_variance(
    segment_labels: list[str],
    explained_variance_ratio: np.ndarray,
    output_path: Path,
) -> None:
    if explained_variance_ratio.ndim != 2 or explained_variance_ratio.shape[0] == 0:
        raise ValueError("Expected a non-empty [segments, components] explained-variance matrix.")

    num_components = int(explained_variance_ratio.shape[1])
    x = np.arange(len(segment_labels), dtype=np.float64)
    width = 0.7 / max(num_components, 1)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    offsets = (np.arange(num_components, dtype=np.float64) - (num_components - 1) / 2.0) * width

    for component_index in range(num_components):
        ax.bar(
            x + offsets[component_index],
            explained_variance_ratio[:, component_index],
            width=width * 0.92,
            color=colors[component_index % len(colors)],
            label=f"PC{component_index + 1}",
        )

    cumulative = explained_variance_ratio.sum(axis=1)
    ax.plot(
        x,
        cumulative,
        color="black",
        linewidth=1.4,
        marker="o",
        markersize=5,
        label=f"PC1-PC{num_components} cumulative",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(segment_labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Simultaneous-rollout segment PCA explained variance")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_segment_explained_variance_csv(
    segment_labels: list[str],
    explained_variance_ratio: np.ndarray,
    output_path: Path,
) -> None:
    component_headers = [
        f"pc{component_index + 1}_explained_variance_ratio"
        for component_index in range(explained_variance_ratio.shape[1])
    ]
    lines = [",".join(["segment", *component_headers, "cumulative_explained_variance_ratio"])]
    for segment_label, row in zip(segment_labels, explained_variance_ratio.tolist()):
        cumulative = float(sum(row))
        values = [segment_label, *[f"{float(value):.6f}" for value in row], f"{cumulative:.6f}"]
        lines.append(",".join(values))
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.pca_components <= 0:
        raise ValueError("--pca-components must be positive.")
    if args.max_rollout_steps <= 0:
        raise ValueError("--max-rollout-steps must be positive.")

    from src.analysis.common import (
        load_model_from_checkpoint,
        resolve_checkpoint_path,
        resolve_config,
    )

    config, run_dir = resolve_config(args.config, args.run_dir)
    algorithm_order = normalize_algorithm_order(config.model.algorithms)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, run_dir)
    model, checkpoint_step = load_model_from_checkpoint(config, checkpoint_path, run_dir)
    model.eval()

    dataset_paths = resolve_dataset_paths(
        config=config,
        dataset_override=args.dataset,
        split=args.split,
        algorithm_order=algorithm_order,
    )
    dataset_sources = load_dataset_sources(dataset_paths, algorithm_order)
    selected_graphs = select_graphs(
        dataset_sources=dataset_sources,
        graph_index=args.graph_index,
        max_graphs=args.max_graphs,
    )
    if not selected_graphs:
        raise ValueError("No graphs selected for analysis.")

    rollouts: list[dict[str, Any]] = []
    source_dataset_histogram: Dict[str, int] = {}
    inactive_step_summaries: Dict[str, List[int]] = {
        algorithm: [] for algorithm in algorithm_order
    }
    for graph_entry in selected_graphs:
        rollout = collect_simultaneous_rollout(
            model=model,
            graph_data=graph_entry["graph"],
            latent_kind=args.latent,
            node_agg=args.node_agg,
            max_rollout_steps=args.max_rollout_steps,
            termination_cfg=config.model,
            activity_source=args.activity_source,
            activity_threshold=args.activity_threshold,
        )
        dataset_label = str(graph_entry["dataset_path"])
        source_dataset_histogram[dataset_label] = source_dataset_histogram.get(dataset_label, 0) + 1
        rollout["source_labels"] = np.asarray(
            [dataset_label] * int(rollout["embeddings"].shape[0]),
            dtype=object,
        )
        rollout["dataset_path"] = dataset_label
        rollout["dataset_graph_index"] = int(graph_entry["dataset_graph_index"])
        for algorithm, step_index in rollout["inactive_steps"].items():
            if step_index is not None:
                inactive_step_summaries[algorithm].append(int(step_index))
        rollouts.append(rollout)

    segment_sets = build_segment_sets(rollouts)
    if not segment_sets:
        raise ValueError("No simultaneous-rollout segments were found.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else (
            Path(args.run_dir) / "analysis" / "execution_segment_pca"
            if args.run_dir is not None
            else Path("analysis/execution_segment_pca")
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_counts = sorted(segment_sets.keys(), reverse=True)
    segment_labels = [f"{count} active" for count in segment_counts]
    fitted_payloads: Dict[int, dict[str, np.ndarray]] = {}
    used_components_by_segment: Dict[int, int] = {}
    segment_summaries: Dict[str, dict[str, Any]] = {}

    for active_count in segment_counts:
        entry = segment_sets[active_count]
        payload, used_components = pca_fit_transform(
            entry["vectors"].astype(np.float64),
            args.pca_components,
        )
        fitted_payloads[active_count] = payload
        used_components_by_segment[active_count] = int(used_components)

        np.savez(output_dir / f"segment_{active_count}_active_pca.npz", **payload)
        plot_algorithm_delta_pca(
            projected=payload["projected"],
            step_indices=entry["step_labels"],
            output_path=output_dir / f"segment_{active_count}_active_pca.png",
            title=f"{active_count} active branches PCA",
            explained_ratio=payload["explained_variance_ratio"],
        )

        unique_steps, step_counts = np.unique(entry["step_labels"], return_counts=True)
        unique_signatures, signature_counts = np.unique(
            entry["active_signatures"], return_counts=True
        )
        unique_sources, source_counts = np.unique(entry["source_labels"], return_counts=True)
        segment_summaries[str(active_count)] = {
            "count": int(entry["vectors"].shape[0]),
            "pca_components_used": int(used_components),
            "explained_variance_ratio": payload["explained_variance_ratio"].astype(np.float64).tolist(),
            "step_histogram": {
                str(int(step)): int(count)
                for step, count in zip(unique_steps.tolist(), step_counts.tolist())
            },
            "active_signature_histogram": {
                str(signature): int(count)
                for signature, count in zip(unique_signatures.tolist(), signature_counts.tolist())
            },
            "source_dataset_histogram": {
                str(source): int(count)
                for source, count in zip(unique_sources.tolist(), source_counts.tolist())
            },
        }

    common_components = min(used_components_by_segment.values())
    explained_variance_matrix = np.stack(
        [
            fitted_payloads[active_count]["explained_variance_ratio"][:common_components]
            for active_count in segment_counts
        ],
        axis=0,
    ).astype(np.float64, copy=False)

    write_segment_explained_variance_csv(
        segment_labels=segment_labels,
        explained_variance_ratio=explained_variance_matrix,
        output_path=output_dir / "segment_explained_variance.csv",
    )
    plot_segment_explained_variance(
        segment_labels=segment_labels,
        explained_variance_ratio=explained_variance_matrix,
        output_path=output_dir / "segment_explained_variance.png",
    )

    summary = {
        "config_name": config.name,
        "checkpoint_step": checkpoint_step,
        "algorithms": list(algorithm_order),
        "dataset_paths": [str(path) for path in dataset_paths],
        "num_graphs": len(selected_graphs),
        "source_dataset_histogram": source_dataset_histogram,
        "graph_index": args.graph_index,
        "max_graphs": args.max_graphs,
        "latent": args.latent,
        "node_agg": args.node_agg,
        "max_rollout_steps": int(args.max_rollout_steps),
        "activity_source": args.activity_source,
        "activity_threshold": float(args.activity_threshold),
        "pca_components_requested": int(args.pca_components),
        "common_pca_components": int(common_components),
        "segment_order": segment_counts,
        "segment_labels": segment_labels,
        "segment_summaries": segment_summaries,
        "inactive_step_summary": {
            algorithm: {
                "count": len(steps),
                "mean_step": float(np.mean(np.asarray(steps, dtype=np.float64))) if steps else None,
                "min_step": int(min(steps)) if steps else None,
                "max_step": int(max(steps)) if steps else None,
            }
            for algorithm, steps in inactive_step_summaries.items()
        },
        "selected_graphs": [
            {
                "dataset_path": rollout["dataset_path"],
                "dataset_graph_index": int(rollout["dataset_graph_index"]),
                "rollout_steps": int(rollout["embeddings"].shape[0]),
                "inactive_steps": rollout["inactive_steps"],
            }
            for rollout in rollouts
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
