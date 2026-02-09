# NGE (Neural Graph Execution)

NGE is a research codebase that trains a message‑passing neural network to execute classic graph algorithms. It generates synthetic graphs, logs step‑wise Bellman‑Ford (BF) and BFS targets, trains a dual‑task model in MLX, and provides analysis utilities for latent convergence.

## Functionality index 
- Data generation: synthetic graph creation (Erdos‑Renyi + Barabasi‑Albert), self‑loops, bidirectional edges, uniform random weights, BF/BFS logging, dataset save/load to `.npz`.
- Model: MLX MPNN processor + BFS/BF encoders/decoders, termination heads, configurable aggregation (SUM/AVG/MIN/MAX), residual connections, dropout, configurable message‑passing depth.
- Training: config‑driven runs, per‑step BF/BFS losses, termination losses, optional gradient norm extraction, evaluation at intervals, checkpoints, run metadata, and JSONL metrics logging.
- Analysis: latent convergence script with per‑step distance curves for single graphs or datasets, optional custom distance functions, plotting + JSON outputs.
- Reproducibility: deterministic seeding, git + environment metadata, resolved config snapshots.
- Utilities: metrics history loader, checkpoint manager with latest marker, debug printing of execution details.
- Tests: workflow and checkpointing tests.

## Quick start
Install dependencies:
```bash
pip install -r requirements.txt
```

Generate datasets (writes `train_dataset.npz`, `val_dataset.npz`, `test_dataset.npz` into the current working directory):
```bash
python -m src.data.dataset
```

Train:
```bash
python -m src.train --config configs/baseline.yaml
```

Resume latest run:
```bash
python -m src.train --config configs/baseline.yaml --resume
```

## Project layout
- `src/model/` – NGE architecture (MPNN + BF/BFS encoders/decoders + termination heads).
- `src/data/` – synthetic graph generation and dataset save/load.
- `src/train.py` – training workflow and CLI.
- `src/utils/` – config, logging, checkpointing, reproducibility, evaluation utilities.
- `src/analysis/` – latent convergence analysis.
- `configs/` – experiment configs.
- `data/` – default dataset location used by configs.
- `runs/` – outputs (created at runtime).
- `tests/` – workflow/utility tests.

## Data generation details
Synthetic graphs are sampled per graph as either:
- Erdos‑Renyi (`p` edge probability), or
- Barabasi‑Albert (`m` edges per new node).

Edges are made bidirectional, and self‑loops are added. Uniform random edge weights are appended (default range `[0.2, 1.0]`). For each graph, the generator logs:
- Bellman‑Ford distance targets and predecessor targets (with a `-1` sentinel for invalid predecessors).
- BFS reachability targets.

Datasets are saved as compressed `.npz` files with per‑graph keys (e.g., `edge_matrix_0`, `bf_distance_targets_0`, ...).

## Model
NGE consists of:
- **Encoders:** BFS/BF linear encoders for per‑node inputs.
- **Processor:** an MPNN stack with configurable aggregation (`SUM`, `AVG`, `MIN`, `MAX`), residual connections, dropout, and depth `num_mp_layers`.
- **Decoders:**
  - BF distance head (regression),
  - BF predecessor head (edge‑wise logits),
  - BFS state head (reachability logits).
- **Termination heads:** separate BF and BFS logits computed from mean processed embeddings; optionally replaced by distance‑based termination with a fixed threshold on successive embedding change.

## Training workflow
Training runs are entirely config‑driven and produce reproducible artifacts.

Per‑step losses (computed when a target exists for that step):
- BF distance: MSE on normalized distances.
- BF predecessor: cross‑entropy on valid nodes only (masking `-1`).
- BFS state: binary cross‑entropy on reachability.
- BF/BFS termination: binary cross‑entropy on termination logits (head‑based) or distance‑based logits (fixed threshold minus successive‑embedding distance).

Training outputs include:
- run directory naming with timestamp + git SHA (+ dirty hash),
- `config_resolved.yaml`,
- `meta.json` (git/environment/seed/config),
- `metrics.jsonl` (append‑only),
- checkpoints in `runs/<run-name>/checkpoints/`.

## CLI
### Training (`src.train`)
```text
--config   Path to a YAML experiment config (required)
--resume   Resume from the latest run in runs/ (uses resolved config in that run)
--run-dir  Explicit run directory to resume (only with --resume)
--tasks    Tasks to optimize/evaluate: all|bf|bfs (default: all)
--eval-only  Skip training and run evaluation only (requires --run-dir or --checkpoint)
--checkpoint Checkpoint directory or file to load (for --eval-only)
--termination-threshold  Override termination_distance_threshold
--disable-distance-termination-signal  Disable BCE termination supervision in distance mode
```
Task mapping:
- `bf` trains/evaluates BF distance, BF predecessor, BF termination
- `bfs` trains/evaluates BFS state, BFS termination
- `all` enables both groups
Notes:
- In `--eval-only`, `--config` takes precedence over `--run-dir/config_resolved.yaml`.
- Threshold affects termination only when `termination_mode: distance`.

## Termination Modes
Termination can be configured in `model`:
- `termination_mode`: `head` (default) or `distance`
- `termination_distance_latent`: `processed` or `encoded`
- `termination_distance`: `l2`, `mean_l2`, `l1`, `mse`
- `termination_distance_threshold`: fixed threshold for distance mode
- `termination_distance_signal`: whether termination BCE supervision is applied in distance mode

## Analysis
### Latent convergence (`src.analysis.latent_convergence`)
Measures how embeddings change over execution.
Key options:
- `--mode successive` (step‑to‑step change) or `--mode to_final` (distance to final step)
- `--converge-threshold` and `--converge-patience` to estimate convergence step

### Embedding trajectories + PCA (`src.analysis.embedding_trajectories`)
Extracts trajectories and produces trajectory‑wise and step‑wise PCA.
Key options:
- `--latent` (`processed` or `encoded`) to choose which embeddings are tracked
- `--node-agg` (`max`, `min`, `mean`) to collapse node dimension per step
- `--step-policy` (`common`, `fixed`, `min`, `max`) to align graphs by execution length
- `--steps` when using `--step-policy fixed`
- `--pca` (`none`, `step`, `trajectory`, `both`) and `--pca-components`
- `--plot` to save 2D scatter plots with explained variance in the title
Outputs:
- `trajectories.npz` (N x T x D + flattened matrices + indices)
- `pca_step.npz` / `pca_trajectory.npz` with projections, components, mean, variance
- Optional plots in the output directory

### Latent convergence analysis (`src.analysis.latent_convergence`)
```text
--config         Path to a YAML config (ignored if --run-dir is used)
--run-dir        Run directory containing config_resolved.yaml + checkpoints
--checkpoint     Specific checkpoint file or directory (defaults to latest)
--split          Dataset split to use when --dataset is not provided (train|val|test)
--dataset        Override dataset path (.npz)
--graph-index    Analyze a single graph by index
--max-graphs     Limit number of graphs for dataset stats
--extra-steps    Run extra steps after termination
--latent         Which latent to probe (processed|encoded)
--distance       Built‑in distance metric (l2|l1|mse|cosine|mean_l2)
--distance-fn    Custom distance function module:function (overrides --distance)
--distance-input Input type for custom distance (numpy|mx)
--mode           Distance mode (successive|to_final)
--converge-threshold  Optional threshold for convergence detection
--converge-patience   Consecutive below-threshold steps to mark convergence
--output-dir     Output directory for plots + JSON
--title          Optional plot title override
```

Outputs (single‑graph):
- `graph_<index>_distance.png`
- `graph_<index>_distances.json`

Outputs (dataset):
- `dataset_distance.png`
- `dataset_stats.json` (mean/std/counts + metadata)

## Configuration
Configs in `configs/` define model, training, data paths, and logging:
- `configs/baseline.yaml`
- `configs/large_model.yaml`
- `configs/debug.yaml`

Schema (from `src/utils/config.py`):
```yaml
name: <string>
model:
  embed_dim: <int>
  residual_connections: <bool>
  agg_fn: <SUM|AVG|MIN|MAX>
  num_mp_layers: <int>
  dropout: <float>
  termination_mode: <head|distance>
  termination_distance_latent: <processed|encoded>
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

## Logging and checkpoints
- Metrics are appended to `metrics.jsonl` with `step`, `split`, `timestamp`, and `metrics`.
- Optional Weights & Biases logging is enabled via config (requires `wandb`).
- Checkpoints are saved as `model.safetensors` and `optimizer.safetensors` with `checkpoint.json` metadata and a `latest.json` pointer.

## Evaluation utilities
`src/utils/eval.py` provides:
- `calculate_losses_and_accuracies` (matches training loss logic).
- `calculate_accuracies` (accuracy‑only variant).
- `print_execution_details` (verbose step‑by‑step diagnostics).
- `extract_per_head_magnitude_grads` (per‑component gradient norms).

## Reproducibility
- `set_seed` for deterministic MLX RNG.
- `meta.json` captures git SHA/branch/dirty hash and environment info (Python + MLX versions, platform, hostname).
- Resolved configs are stored per‑run (`config_resolved.yaml`).
