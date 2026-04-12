"""Run test-only 100-node eval for normal/frozen-random/identity CLRS runs.

This module writes per-run accuracy artifacts to:
  runs/<run>/analysis/eval_test_100n.json

It then builds a comparison summary/CSV/plot under:
  analysis/model_comparison_100n_test_only/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.train import (
    create_model,
    evaluate_model_accuracies_only,
    load_split_dataset,
)
from src.utils import load_config
from src.utils.checkpoint import CheckpointManager
from src.utils.task_specs import metric_dict, resolve_selected_tasks


DEFAULT_LABELS = ("normal", "frozen_random", "identity")
DEFAULT_RUN_DIRS = (
    "runs/20260317-220537_647125b-dirty7a8bf0e9_clrs-graphs",
    "runs/20260405-180449_2a33074-dirtye3b0c442_clrs-graphs-frozen-random-processor",
    "runs/20260406-104020_2a33074-dirtyf9f87a51_clrs-graphs-identity-processor",
)
DEFAULT_CONFIGS = (
    "configs/clrs_graphs_test_100n.yaml",
    "configs/clrs_graphs_frozen_random_processor_test_100n.yaml",
    "configs/clrs_graphs_identity_processor_test_100n.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run test-only 100n eval for normal/frozen-random/identity and "
            "build a comparison report."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/model_comparison_100n_test_only",
        help="Directory for summary CSV/JSON/plot comparison outputs.",
    )
    parser.add_argument(
        "--eval-file-name",
        type=str,
        default="eval_test_100n.json",
        help="Per-run analysis artifact name to write and compare.",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Only write per-run eval artifacts; skip cross-run comparison.",
    )
    return parser.parse_args()


def resolve_latest_checkpoint_path(run_dir: Path) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    manager = CheckpointManager(checkpoints_dir)
    latest_step = manager.get_latest_step()
    if latest_step is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    return checkpoints_dir / f"step_{latest_step:08d}"


def evaluate_test_only(
    *,
    label: str,
    run_dir: Path,
    config_path: Path,
    eval_file_name: str,
) -> Path:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found for '{label}': {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(
            f"100n config not found for '{label}': {config_path}"
        )

    config = load_config(config_path)
    selected_tasks = resolve_selected_tasks(config.training.tasks, config.model.algorithms)

    model = create_model(config)
    model.eval()

    checkpoint_path = resolve_latest_checkpoint_path(run_dir)
    manager = CheckpointManager(checkpoint_path.parent)
    model, _, step = manager.load(model, optimizer=None, checkpoint_path=checkpoint_path)

    test_dataset = load_split_dataset(config, "test", selected_tasks)
    test_accuracies = evaluate_model_accuracies_only(
        model=model,
        dataset=test_dataset,
        embed_dim=config.model.embed_dim,
        termination_cfg=config.model,
        selected_tasks=selected_tasks,
    )

    payload = {
        "checkpoint_step": int(step),
        "eval_mode": "accuracies_only",
        "selected_tasks": config.training.tasks,
        "termination": {
            "mode": config.model.termination_mode,
            "distance": config.model.termination_distance,
            "latent": config.model.termination_distance_latent,
            "threshold": float(config.model.termination_distance_threshold),
            "distance_signal": bool(config.model.termination_distance_signal),
        },
        "test": metric_dict("acc", test_accuracies, model.algorithms),
    }

    out_path = run_dir / "analysis" / eval_file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"[{label}] wrote {out_path}")
    return out_path


def run_comparison(
    *,
    labels: tuple[str, ...],
    run_dirs: tuple[Path, ...],
    eval_file_name: str,
    output_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "src.analysis.compare_eval_runs",
        "--run-dirs",
        *(str(run_dir) for run_dir in run_dirs),
        "--labels",
        *labels,
        "--eval-file-name",
        eval_file_name,
        "--split",
        "test",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    labels = tuple(DEFAULT_LABELS)
    run_dirs = tuple(Path(path) for path in DEFAULT_RUN_DIRS)
    configs = tuple(Path(path) for path in DEFAULT_CONFIGS)

    if not (len(labels) == len(run_dirs) == len(configs)):
        raise ValueError("Internal defaults must have matching lengths.")

    for label, run_dir, config_path in zip(labels, run_dirs, configs):
        evaluate_test_only(
            label=label,
            run_dir=run_dir,
            config_path=config_path,
            eval_file_name=args.eval_file_name,
        )

    if args.skip_compare:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_comparison(
        labels=labels,
        run_dirs=run_dirs,
        eval_file_name=args.eval_file_name,
        output_dir=output_dir,
    )
    print(f"Comparison written to {output_dir}")


if __name__ == "__main__":
    main()
