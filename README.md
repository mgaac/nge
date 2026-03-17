# NGE (Neural Graph Execution)

NGE is an MLX research codebase for learning algorithm execution on graphs.  
It trains a message-passing model to predict Bellman-Ford (distance + predecessor), BFS (reachability), and Prim (in-tree state + key + predecessor) state transitions, plus task termination signals.

## Capability overview

| Area | What is implemented |
| --- | --- |
| Data | Synthetic Erdos-Renyi + Barabasi-Albert graphs, bidirectional edges, self-loops, weighted edges, BF/BFS/Prim supervision, `.npz` serialization |
| Model | Shared encoder/processor architecture with BF/BFS/Prim heads and configurable aggregation (`SUM/AVG/MIN/MAX`) |
| Training | Config-driven runs, checkpointing, resume, JSONL metrics, optional W&B logging, explicit `mx.eval(...)` barriers for stable MLX gradient accumulation |
| Evaluation | Loss/accuracy reporting, eval-only mode, failure-mode analysis and debug traces |
| Analysis | Latent convergence, embedding trajectories + PCA, execution-length overlays, threshold sweeps, dataset step distributions, extra-step dynamics |
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

### 2. Optional: filter by execution-length relation

```bash
conda run -n mlx python -m src.data.filter_execution_length \
  --input data/train_dataset.npz \
  --output data/train_dataset_unequal_exec.npz \
  --relation unequal
```

The filter utility remains BF/BFS-compatible and also exposes Prim-aware relations such as `bf_gt_prim`, `prim_gt_bf`, `bfs_gt_prim`, `prim_gt_bfs`, and `all_equal`.

### 3. Optional: create isolated-execution datasets

This creates counterfactual variants where one algorithm evolves normally while the others are frozen at their initial target state. `balanced` emits BF-only, BFS-only, and Prim-only variants for each source graph.

```bash
conda run -n mlx python -m src.data.isolation_dataset \
  --input data/val_dataset.npz \
  --output data/val_dataset_isolated_exec.npz \
  --mode balanced
```

### 4. Train

```bash
conda run -n mlx python -m src.train --config configs/baseline.yaml
```

Task selection can be stored in the config:

```yaml
training:
  tasks: prim
```

The CLI still overrides it when needed:

```bash
conda run -n mlx python -m src.train --config configs/prims.yaml --tasks bfs
```

Transfer-style training is also config-driven. For example, to reuse a BF-trained processor for BFS:

```yaml
training:
  tasks: bfs
  init_checkpoint: runs/<bf_run>/checkpoints/step_<...>
  freeze_modules:
    - processor
    - bf_encoder
    - bf_decoder
    - bf_termination
    - prim_encoder
    - prim_decoder
    - prim_termination
  reset_modules:
    - bfs_encoder
    - bfs_decoder
    - bfs_termination
```

### 5. Resume latest run

```bash
conda run -n mlx python -m src.train --config configs/baseline.yaml --resume
```

### 6. Eval-only

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
  termination_distance_latent: <processed|encoded|encoded_bfs|encoded_bf|encoded_prim>
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
  tasks: <all|bf|bfs|prim>
  init_checkpoint: <path|null>
  freeze_modules: [<module_path>, ...]
  reset_modules: [<module_path>, ...]
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
  checkpoint_keep_last: <int>
```

## Training CLI (`src.train`)

| Flag | Purpose |
| --- | --- |
| `--config` | Path to YAML config |
| `--resume` | Resume latest run |
| `--run-dir` | Explicit run directory (resume/eval) |
| `--tasks {all,bf,bfs,prim}` | Override `training.tasks` for train/eval |
| `--eval-only` | Skip training and only evaluate |
| `--checkpoint` | Explicit checkpoint file/dir for eval |
| `--accuracies-only` | Eval-only: skip losses |
| `--analyze-failures` | Write failure analysis JSON and plots |
| `--failure-split` | Split for failure analysis |
| `--termination-mode` | Override `head` or `distance` at eval/train time |
| `--termination-threshold` | Override distance termination threshold |

`training.init_checkpoint` loads weights before training starts. `training.freeze_modules` zeroes gradients for the named module prefixes during optimization. `training.reset_modules` reinitializes named modules from a fresh model after the checkpoint is loaded. Valid top-level module names include `processor`, `bfs_encoder`, `bf_encoder`, `prim_encoder`, `bfs_decoder`, `bf_decoder`, `prim_decoder`, `bfs_termination`, `bf_termination`, and `prim_termination`.

## Analysis scripts

| Script | Output |
| --- | --- |
| `src.analysis.latent_convergence` | Distance-to-final / successive latent change plots + JSON |
| `src.analysis.embedding_trajectories` | Trajectories + step/trajectory PCA artifacts |
| `src.analysis.algorithm_subspace` | Algorithm-specific latent-delta subspace overlap / explained-variance analysis |
| `src.analysis.execution_length_step_overlay` | Overlayed average trajectories across execution lengths |
| `src.analysis.extra_step_state_dynamics` | Fake-continuation dynamics and stabilization diagnostics across BF/BFS/Prim states |
| `src.analysis.dataset_step_distribution` | BF/BFS/Prim step distribution plots + summary JSON |
| `src.analysis.pc_monotonicity` | Spearman/monotonicity checks between sequential latent distances and PCA coordinates near task termination |
| `src.analysis.termination_threshold_sweep` | Threshold vs BF/BFS/Prim termination-accuracy curves |

Example:

```bash
conda run -n mlx python -m src.analysis.dataset_step_distribution \
  --dataset data/test_dataset.npz
```

`src.analysis.embedding_trajectories` supports `--pca-components 2` and `--pca-components 3`; when plotting is enabled, three components produce a 3D PCA figure plus pairwise perspective views for `PC1-PC2`, `PC1-PC3`, and `PC2-PC3`. Step-wise PCA plots also overlay BF→BFS and BF→Prim reference segments built from the mean algorithm termination steps of the selected graphs.
`src.analysis.embedding_trajectories` also supports `--execution-window {all,bf,bfs,prim}` to truncate the analyzed trajectory prefix to the steps where a specific algorithm is still executing. For example, `--execution-window bfs` drops the post-BFS tail where only BF or Prim continue to evolve.
`src.analysis.algorithm_subspace` consumes an existing `embedding_trajectories` artifact, converts trajectories into per-step latent deltas, partitions those deltas by algorithm-active or algorithm-exclusive execution phases, and compares the resulting PCA subspaces via mean canonical correlations and cross explained-variance heatmaps.
`src.analysis.pc_monotonicity` tests hypotheses of the form "sequential latent-change magnitudes are monotonic with respect to PCk near task termination" and writes per-task scatter plots plus a Spearman heatmap.

## Outputs per run

Each training run creates `runs/<run_name>/` with:

- `config_resolved.yaml`
- `meta.json` (git/environment/seed/config metadata)
- `metrics.jsonl`
- `checkpoints/`
- `analysis/` (if analysis/eval modes are enabled)

When W&B is enabled, metrics are logged under split namespaces such as `train/*`, `val/*`, `train_eval/*`, and `test/*`, all indexed by the explicit `epoch` metric.
Checkpoint retention is controlled by `logging.checkpoint_keep_last`; after training, older checkpoints are pruned and only the most recent `N` are kept.

## Dataset Schema

Each graph record stores:

- `edge_matrix`, `num_nodes`, `source_node`
- `bf_distance_targets`, `bf_predecessor_targets`
- `bfs_state_targets`
- `prim_state_targets`, `prim_key_targets`, `prim_predecessor_targets`

The recurrent model input is the concatenation of the previous hidden state with the per-node algorithm state vector `[bfs_state, bf_distance, prim_state, prim_key]`.

## Quality checks

```bash
conda run -n mlx python -m compileall -q src tests
conda run -n mlx python -m pytest -q
```

If `pytest` is unavailable in your environment, install dependencies from `requirements.txt` first. MLX runtime checks require a working MLX environment on supported hardware.
