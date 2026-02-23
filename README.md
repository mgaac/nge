# NGE (Neural Graph Execution)

NGE is an MLX research codebase for learning algorithm execution on graphs.  
It trains a message-passing model to predict Bellman-Ford (distance + predecessor) and BFS (reachability) state transitions, plus task termination signals.

## Capability overview

| Area | What is implemented |
| --- | --- |
| Data | Synthetic Erdos-Renyi + Barabasi-Albert graphs, bidirectional edges, self-loops, weighted edges, `.npz` serialization |
| Model | Shared encoder/processor architecture with BF/BFS heads and configurable aggregation (`SUM/AVG/MIN/MAX`) |
| Training | Config-driven runs, checkpointing, resume, JSONL metrics, optional W&B logging |
| Evaluation | Loss/accuracy reporting, eval-only mode, failure-mode analysis and debug traces |
| Analysis | Latent convergence, embedding trajectories + PCA, execution-length overlays, threshold sweeps, dataset step distributions |
| Reproducibility | Seed control, run metadata (`meta.json`), resolved config snapshots |

## Repository layout

- `src/model/` model definitions (`NGE`, aggregation enum, heads).
- `src/data/` dataset generation/loading and filtering utilities.
- `src/train.py` main training and evaluation entrypoint.
- `src/utils/` config, logging, checkpointing, evaluation, reproducibility helpers.
- `src/analysis/` post-training analysis scripts.
- `configs/` experiment configurations.
- `tests/` workflow and checkpointing tests.

## Setup

Run from repository root.

```bash
conda run -n mlx pip install -r requirements.txt
```

MLX targets Apple Silicon. This project is intended for macOS + Metal.

## Quick start

### 1. Generate default datasets

Generates `train/val/test` splits (`1500/100/100`, 20 nodes) into `data/`:

```bash
conda run -n mlx python -m src.data.dataset --preset --output-dir data
```

### 2. Optional: filter by BF/BFS execution-length relation

```bash
conda run -n mlx python -m src.data.filter_execution_length \
  --input data/train_dataset.npz \
  --output data/train_dataset_unequal_exec.npz \
  --relation unequal
```

### 3. Train

```bash
conda run -n mlx python -m src.train --config configs/baseline.yaml
```

### 4. Resume latest run

```bash
conda run -n mlx python -m src.train --config configs/baseline.yaml --resume
```

### 5. Eval-only

```bash
conda run -n mlx python -m src.train \
  --eval-only \
  --run-dir runs/<run_name> \
  --accuracies-only
```

## Configuration

Available configs:

- `configs/baseline.yaml`
- `configs/debug.yaml`
- `configs/large_model.yaml`
- `configs/termination.yaml`

Core schema:

```yaml
name: <string>
model:
  embed_dim: <int>
  residual_connections: <bool>
  agg_fn: <SUM|AVG|MIN|MAX>
  num_mp_layers: <int>
  dropout: <float>
  termination_mode: <head|distance>
  termination_distance_latent: <processed|encoded|encoded_bfs|encoded_bf>
  termination_distance: <l2|mean_l2|l1|mse>
  termination_distance_threshold: <float>
  termination_distance_signal: <bool>
training:
  epochs: <int>
  learning_rate: <float>
  max_grad_norm: <float>
  batch_size: <int>
  eval_interval: <int>
  seed: <int>
data:
  train_path: <path>
  val_path: <path>
  test_path: <path>
logging:
  use_wandb: <bool>
  wandb_project: <string>
  wandb_entity: <string>
  log_interval: <int>
  save_checkpoints: <bool>
  checkpoint_interval: <int>
```

## Training CLI (`src.train`)

| Flag | Purpose |
| --- | --- |
| `--config` | Path to YAML config |
| `--resume` | Resume latest run |
| `--run-dir` | Explicit run directory (resume/eval) |
| `--tasks {all,bf,bfs}` | Select optimized/evaluated tasks |
| `--eval-only` | Skip training and only evaluate |
| `--checkpoint` | Explicit checkpoint file/dir for eval |
| `--accuracies-only` | Eval-only: skip losses |
| `--analyze-failures` | Write failure analysis JSON and plots |
| `--failure-split` | Split for failure analysis |
| `--termination-mode` | Override `head` or `distance` at eval/train time |
| `--termination-threshold` | Override distance termination threshold |

## Analysis scripts

| Script | Output |
| --- | --- |
| `src.analysis.latent_convergence` | Distance-to-final / successive latent change plots + JSON |
| `src.analysis.embedding_trajectories` | Trajectories + step/trajectory PCA artifacts |
| `src.analysis.execution_length_step_overlay` | Overlayed average trajectories across execution lengths |
| `src.analysis.extra_step_state_dynamics` | Fake-continuation dynamics and stabilization diagnostics |
| `src.analysis.dataset_step_distribution` | BF/BFS step distribution plots + summary JSON |
| `src.analysis.termination_threshold_sweep` | Threshold vs termination-accuracy curves |

Example:

```bash
conda run -n mlx python -m src.analysis.dataset_step_distribution \
  --dataset data/test_dataset.npz
```

## Outputs per run

Each training run creates `runs/<run_name>/` with:

- `config_resolved.yaml`
- `meta.json` (git/environment/seed/config metadata)
- `metrics.jsonl`
- `checkpoints/`
- `analysis/` (if analysis/eval modes are enabled)

## Quality checks

```bash
conda run -n mlx python -m compileall -q src tests
conda run -n mlx python -m pytest -q
```

If `pytest` is unavailable in your environment, install dependencies from `requirements.txt` first.
