"""Fit independent PCA models for each fixed execution step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PCA independently for each execution step using "
            "embedding_trajectories outputs."
        )
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        required=True,
        help="Directory containing trajectories.npz and metadata.json.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Number of principal components to retain per step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "<trajectories-dir>/per_step_group_pca_k<pca-components>."
        ),
    )
    return parser.parse_args()


def pca_fit_transform(data: np.ndarray, n_components: int) -> Tuple[Dict[str, np.ndarray], int]:
    if data.ndim != 2:
        raise ValueError("PCA expects a 2D array.")
    num_samples, num_features = data.shape
    if num_samples == 0 or num_features == 0:
        raise ValueError("Empty data passed to PCA.")

    mean = data.mean(axis=0, keepdims=True)
    centered = data - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    max_components = min(vt.shape[0], n_components)

    components = vt[:max_components]
    projected = centered @ components.T

    if num_samples > 1:
        explained_variance = (singular_values * singular_values) / (num_samples - 1)
    else:
        explained_variance = singular_values * singular_values

    total_variance = explained_variance.sum()
    if total_variance > 0:
        explained_ratio = explained_variance[:max_components] / total_variance
    else:
        explained_ratio = np.zeros(max_components, dtype=np.float64)

    payload = {
        "projected": projected.astype(np.float32, copy=False),
        "components": components.astype(np.float32, copy=False),
        "mean": mean.squeeze(axis=0).astype(np.float32, copy=False),
        "explained_variance": explained_variance[:max_components].astype(np.float32, copy=False),
        "explained_variance_ratio": explained_ratio.astype(np.float32, copy=False),
    }
    return payload, max_components


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    if args.pca_components <= 0:
        raise ValueError("--pca-components must be positive.")

    trajectories_dir = args.trajectories_dir
    trajectories_path = trajectories_dir / "trajectories.npz"
    metadata_path = trajectories_dir / "metadata.json"

    if not trajectories_path.exists():
        raise FileNotFoundError(f"Missing file: {trajectories_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing file: {metadata_path}")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else trajectories_dir / f"per_step_group_pca_k{args.pca_components}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as handle:
        metadata = json.load(handle)

    payload = np.load(trajectories_path)
    trajectories = payload["trajectories"]
    if trajectories.ndim != 3:
        raise ValueError(
            f"Expected trajectories with shape [num_graphs, num_steps, latent_dim], got {trajectories.shape}"
        )

    num_graphs, num_steps, latent_dim = trajectories.shape
    requested_components = int(args.pca_components)
    used_components = min(requested_components, num_graphs, latent_dim)

    projected_by_step = np.zeros((num_steps, num_graphs, used_components), dtype=np.float32)
    components_by_step = np.zeros((num_steps, used_components, latent_dim), dtype=np.float32)
    means_by_step = np.zeros((num_steps, latent_dim), dtype=np.float32)
    explained_variance_by_step = np.zeros((num_steps, used_components), dtype=np.float32)
    explained_ratio_by_step = np.zeros((num_steps, used_components), dtype=np.float32)

    summary_steps = []
    for step in range(num_steps):
        step_data = trajectories[:, step, :]
        fit, fit_components = pca_fit_transform(step_data, requested_components)
        if fit_components != used_components:
            raise ValueError(
                f"Inconsistent PCA dimensionality at step {step}: "
                f"expected {used_components}, got {fit_components}"
            )
        projected_by_step[step] = fit["projected"]
        components_by_step[step] = fit["components"]
        means_by_step[step] = fit["mean"]
        explained_variance_by_step[step] = fit["explained_variance"]
        explained_ratio_by_step[step] = fit["explained_variance_ratio"]
        summary_steps.append(
            {
                "step": int(step),
                "explained_variance_ratio": fit["explained_variance_ratio"].astype(
                    np.float64, copy=False
                ).tolist(),
                "explained_variance": fit["explained_variance"].astype(
                    np.float64, copy=False
                ).tolist(),
            }
        )

    np.savez(
        output_dir / "per_step_group_pca.npz",
        projected_by_step=projected_by_step,
        components_by_step=components_by_step,
        means_by_step=means_by_step,
        explained_variance_by_step=explained_variance_by_step,
        explained_variance_ratio_by_step=explained_ratio_by_step,
    )

    summary = {
        "source_trajectories_dir": str(trajectories_dir),
        "source_dataset": metadata.get("dataset"),
        "num_graphs": int(num_graphs),
        "num_steps": int(num_steps),
        "latent_dim": int(latent_dim),
        "pca_components_requested": int(requested_components),
        "pca_components_used": int(used_components),
        "steps": summary_steps,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Saved per-step PCA to: {output_dir}")


if __name__ == "__main__":
    main()
