"""Relate transfer performance to directional subspace containment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.transfer_matrix import task_mean_nontermination_accuracy
from src.utils.task_specs import algorithm_display_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute ordered-pair correlations between transfer performance and "
            "directional subspace containment from algorithm_subspace outputs."
        )
    )
    parser.add_argument(
        "--subspace-summary",
        type=str,
        required=True,
        help="Path to summary.json produced by src.analysis.algorithm_subspace.",
    )
    parser.add_argument(
        "--transfer-summary",
        type=str,
        required=True,
        help="Path to <split>_transfer_matrix_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for CSV/JSON/plots. Default: "
            "<subspace-summary-dir>/transfer_subspace_correlation_<transfer-stem>"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def primary_score_for_target(
    metrics: dict[str, Any],
    target: str,
    primary_metric_specs: dict[str, Any],
) -> float | None:
    primary_spec = primary_metric_specs.get(target)
    if isinstance(primary_spec, str):
        key = f"acc/{primary_spec}"
        return float(metrics[key]) if key in metrics else None
    if isinstance(primary_spec, list):
        values = [
            float(metrics[f"acc/{metric}"])
            for metric in primary_spec
            if f"acc/{metric}" in metrics
        ]
        if values:
            return float(np.mean(values))
    return task_mean_nontermination_accuracy(metrics, target)


def ordered_pair_metrics(transfer_summary: dict[str, Any]) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    pair_results = transfer_summary.get("pair_results", [])
    pair_means: dict[tuple[str, str], float] = {}
    baselines: dict[str, float] = {}
    primary_metric_specs = transfer_summary.get("primary_metrics", {})

    for result in pair_results:
        source = str(result["source"])
        target = str(result["target"])
        metrics = result.get("metrics") or {}
        task_mean = primary_score_for_target(metrics, target, primary_metric_specs)
        if task_mean is None:
            continue
        pair_means[(source, target)] = float(task_mean)
        if source == target:
            baselines[target] = float(task_mean)

    fallback_baselines = transfer_summary.get("primary_baseline_by_target", {})
    for target in transfer_summary.get("targets", []):
        if target in baselines:
            continue
        fallback = fallback_baselines.get(target)
        if fallback is not None:
            baselines[str(target)] = float(fallback)

    return pair_means, baselines


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.shape[0]:
        stop = start + 1
        while stop < sorted_values.shape[0] and sorted_values[stop] == sorted_values[start]:
            stop += 1
        rank = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson_corr(average_ranks(x), average_ranks(y))


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if x.size < 2 or np.allclose(x, x[0]):
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def residual_outliers(
    rows: list[dict[str, Any]],
    predictor_key: str,
    response_key: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    predictor = np.array([float(row[predictor_key]) for row in rows], dtype=np.float64)
    response = np.array([float(row[response_key]) for row in rows], dtype=np.float64)
    line = fit_line(predictor, response)
    if line is None:
        return []
    slope, intercept = line
    residuals = response - (slope * predictor + intercept)
    ranked = sorted(
        (
            {
                "pair": rows[index]["pair"],
                "source": rows[index]["source"],
                "target": rows[index]["target"],
                "predictor": float(predictor[index]),
                "response": float(response[index]),
                "residual": float(residuals[index]),
                "abs_residual": float(abs(residuals[index])),
            }
            for index in range(len(rows))
        ),
        key=lambda item: item["abs_residual"],
        reverse=True,
    )
    return ranked[:top_k]


def correlation_summary(rows: list[dict[str, Any]], predictor_key: str, response_key: str) -> dict[str, Any]:
    predictor = np.array([float(row[predictor_key]) for row in rows], dtype=np.float64)
    response = np.array([float(row[response_key]) for row in rows], dtype=np.float64)
    return {
        "num_pairs": int(predictor.size),
        "pearson": pearson_corr(predictor, response),
        "spearman": spearman_corr(predictor, response),
    }


def write_pair_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair",
        "source",
        "target",
        "source_label",
        "target_label",
        "transfer_raw",
        "transfer_raw_delta",
        "transfer_relative_delta_percent",
        "containment_target_in_source",
        "containment_source_in_target",
        "symmetric_overlap",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_response_scatter(
    rows: list[dict[str, Any]],
    response_key: str,
    response_label: str,
    output_path: Path,
) -> None:
    predictors = [
        (
            "containment_target_in_source",
            "Target In Source Basis",
            "#1f77b4",
        ),
        (
            "containment_source_in_target",
            "Source In Target Basis",
            "#ff7f0e",
        ),
        (
            "symmetric_overlap",
            "Symmetric Overlap",
            "#2ca02c",
        ),
    ]
    fig, axes = plt.subplots(1, len(predictors), figsize=(17.5, 4.8), sharey=True)
    if len(predictors) == 1:
        axes = [axes]

    response = np.array([float(row[response_key]) for row in rows], dtype=np.float64)
    for ax, (predictor_key, title, color) in zip(axes, predictors):
        predictor = np.array([float(row[predictor_key]) for row in rows], dtype=np.float64)
        corr = correlation_summary(rows, predictor_key, response_key)
        ax.scatter(predictor, response, s=34, alpha=0.9, color=color, edgecolors="black", linewidths=0.25)
        line = fit_line(predictor, response)
        if line is not None:
            slope, intercept = line
            xs = np.linspace(float(np.min(predictor)), float(np.max(predictor)), 200)
            ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.0, alpha=0.8)
        for row in rows:
            ax.annotate(
                row["pair"],
                (float(row[predictor_key]), float(row[response_key])),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
                alpha=0.78,
            )
        pearson = corr["pearson"]
        spearman = corr["spearman"]
        ax.set_title(
            f"{title}\n"
            f"Pearson={pearson:.3f}  Spearman={spearman:.3f}"
            if pearson is not None and spearman is not None
            else title
        )
        ax.set_xlabel("Predictor value")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(response_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    subspace_summary_path = Path(args.subspace_summary)
    transfer_summary_path = Path(args.transfer_summary)
    subspace_summary = load_json(subspace_summary_path)
    transfer_summary = load_json(transfer_summary_path)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else subspace_summary_path.parent
        / f"transfer_subspace_correlation_{transfer_summary_path.stem.replace('_transfer_matrix_summary', '')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    subspace_order = list(subspace_summary["algorithms"])
    transfer_sources = list(transfer_summary["sources"])
    transfer_targets = list(transfer_summary["targets"])
    expected = set(subspace_order)
    if set(transfer_sources) != expected or set(transfer_targets) != expected:
        raise ValueError(
            "Transfer summary sources/targets do not match the subspace summary algorithms. "
            f"subspace={subspace_order}, sources={transfer_sources}, targets={transfer_targets}"
        )

    explained = np.asarray(subspace_summary["cross_explained_variance"], dtype=np.float64)
    overlap = np.asarray(subspace_summary["subspace_overlap_mean_canonical_corr"], dtype=np.float64)
    index = {algorithm: idx for idx, algorithm in enumerate(subspace_order)}

    pair_means, baselines = ordered_pair_metrics(transfer_summary)
    rows: list[dict[str, Any]] = []
    for source in transfer_sources:
        for target in transfer_targets:
            if source == target:
                continue
            pair_mean = pair_means.get((source, target))
            baseline = baselines.get(target)
            if pair_mean is None or baseline is None:
                continue
            relative_delta = np.nan if baseline == 0.0 else (pair_mean - baseline) * 100.0 / baseline
            source_idx = index[source]
            target_idx = index[target]
            rows.append(
                {
                    "pair": f"{source}->{target}",
                    "source": source,
                    "target": target,
                    "source_label": algorithm_display_name(source),
                    "target_label": algorithm_display_name(target),
                    "transfer_raw": float(pair_mean),
                    "transfer_raw_delta": float(pair_mean - baseline),
                    "transfer_relative_delta_percent": float(relative_delta),
                    "containment_target_in_source": float(explained[source_idx, target_idx]),
                    "containment_source_in_target": float(explained[target_idx, source_idx]),
                    "symmetric_overlap": float(overlap[source_idx, target_idx]),
                }
            )

    if not rows:
        raise ValueError("No off-diagonal ordered transfer pairs were available for analysis.")

    rows.sort(key=lambda row: (row["source"], row["target"]))
    correlations = {
        "transfer_raw_delta": {
            "containment_target_in_source": correlation_summary(
                rows, "containment_target_in_source", "transfer_raw_delta"
            ),
            "containment_source_in_target": correlation_summary(
                rows, "containment_source_in_target", "transfer_raw_delta"
            ),
            "symmetric_overlap": correlation_summary(
                rows, "symmetric_overlap", "transfer_raw_delta"
            ),
        },
        "transfer_relative_delta_percent": {
            "containment_target_in_source": correlation_summary(
                rows, "containment_target_in_source", "transfer_relative_delta_percent"
            ),
            "containment_source_in_target": correlation_summary(
                rows, "containment_source_in_target", "transfer_relative_delta_percent"
            ),
            "symmetric_overlap": correlation_summary(
                rows, "symmetric_overlap", "transfer_relative_delta_percent"
            ),
        },
    }

    summary = {
        "subspace_summary": str(subspace_summary_path),
        "transfer_summary": str(transfer_summary_path),
        "algorithm_order": subspace_order,
        "pair_count": len(rows),
        "predictor_definitions": {
            "containment_target_in_source": (
                "cross_explained_variance[source, target]: fraction of target variance "
                "explained by the source basis. This matches transfer source->target "
                "under the implementation where rows are basis algorithms and columns "
                "are evaluated target deltas."
            ),
            "containment_source_in_target": (
                "cross_explained_variance[target, source]: the flipped directional read."
            ),
            "symmetric_overlap": (
                "subspace_overlap_mean_canonical_corr[source, target]: mean canonical correlation."
            ),
        },
        "response_definitions": {
            "transfer_raw_delta": (
                "primary score(source->target) - own-processor baseline(target), "
                "where the primary score follows transfer_summary['primary_metrics'] "
                "when available and otherwise falls back to the mean non-termination accuracy."
            ),
            "transfer_relative_delta_percent": (
                "100 * (primary score(source->target) - baseline(target)) / baseline(target), "
                "with the same primary-score definition as above."
            ),
        },
        "target_baselines": baselines,
        "correlations": correlations,
        "largest_residual_outliers": {
            "transfer_raw_delta": residual_outliers(
                rows,
                predictor_key="containment_target_in_source",
                response_key="transfer_raw_delta",
            ),
            "transfer_relative_delta_percent": residual_outliers(
                rows,
                predictor_key="containment_target_in_source",
                response_key="transfer_relative_delta_percent",
            ),
        },
        "pairs": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_pair_table(output_dir / "ordered_pairs.csv", rows)
    plot_response_scatter(
        rows,
        response_key="transfer_raw_delta",
        response_label="Primary Transfer Raw Delta",
        output_path=output_dir / "transfer_raw_delta_scatter.png",
    )
    plot_response_scatter(
        rows,
        response_key="transfer_relative_delta_percent",
        response_label="Primary Transfer Relative Delta (%)",
        output_path=output_dir / "transfer_relative_delta_percent_scatter.png",
    )
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
