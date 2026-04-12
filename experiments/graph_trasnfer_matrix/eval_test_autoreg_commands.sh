#!/usr/bin/env bash
set -euo pipefail

matrix_dir="${1:-experiments/graph_trasnfer_matrix}"
manifest_path="${2:-experiments/manifest_resolved.json}"
conda_env="${CONDA_ENV:-mlx}"
max_retries="${MAX_RETRIES:-5}"
wandb_mode="${WANDB_MODE:-offline}"

tmp_runs="$(mktemp)"
trap 'rm -f "$tmp_runs"' EXIT

python - <<'PY' "$matrix_dir" "$manifest_path" > "$tmp_runs"
import json
import sys
from pathlib import Path

matrix_dir = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
run_root = Path("runs").resolve()


def source_run_dir_from_checkpoint(reference: str) -> Path:
    checkpoint_dir = Path(reference).resolve()
    if checkpoint_dir.name.startswith("step_") and checkpoint_dir.parent.name == "checkpoints":
        return checkpoint_dir.parents[1]
    raise ValueError(f"Unsupported checkpoint reference: {reference}")


def find_latest_run_for_name(experiment_name: str) -> Path | None:
    matches = []
    for run_dir in sorted(run_root.glob("*")):
        config_path = run_dir / "config_resolved.yaml"
        if not config_path.exists():
            continue
        config_text = config_path.read_text()
        if (
            f"name: {experiment_name}\n" in config_text
            or f'name: "{experiment_name}"\n' in config_text
            or f"name: '{experiment_name}'\n" in config_text
        ):
            matches.append(run_dir)
    return matches[-1] if matches else None


manifest = json.loads(manifest_path.read_text())

for reference in manifest["resolved_sources"].values():
    print(source_run_dir_from_checkpoint(reference))

for entry in manifest["experiments"]:
    run_dir = find_latest_run_for_name(entry["experiment_name"])
    if run_dir is not None:
        print(run_dir)
PY

sort -u "$tmp_runs" | while IFS= read -r run_dir; do
  [ -z "$run_dir" ] && continue
  echo "[eval] $run_dir"
  attempt=1
  while true; do
    if WANDB_MODE="$wandb_mode" conda run -n "$conda_env" --no-capture-output \
      python -m src.train --eval-only --run-dir "$run_dir"; then
      break
    fi
    if [ "$attempt" -ge "$max_retries" ]; then
      echo "[error] eval failed after ${max_retries} attempts: $run_dir" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "[retry] $run_dir (attempt ${attempt}/${max_retries})"
    sleep 2
  done
  cp "$run_dir/analysis/eval_only.json" "$run_dir/analysis/eval_only_autoreg.json"
done

WANDB_MODE="$wandb_mode" conda run -n "$conda_env" --no-capture-output \
python -m src.analysis.transfer_matrix collect-eval-only \
  --matrix-dir "$matrix_dir" \
  --split test \
  --eval-file-name eval_only_autoreg.json \
  --output-prefix test_autoreg
