# NGE (Neural Graph Execution)

NGE is a research codebase that trains a message‑passing neural network to execute classic graph algorithms. It generates synthetic graphs, logs step‑wise Bellman‑Ford (BF) and BFS targets, trains a dual‑task model in MLX, and provides analysis utilities for latent convergence.

## Functionality index 
- Data generation: synthetic graph creation (Erdos‑Renyi + Barabasi‑Albert), self‑loops, bidirectional edges, uniform random weights, BF/BFS logging, dataset save/load to `.npz`.
- Model: MLX MPNN processor + BFS/BF encoders/decoders, termination heads, configurable aggregation (SUM/AVG/MIN/MAX), residual connections, dropout, configurable message‑passing depth.
- Training: config‑driven runs, per‑step BF/BFS losses, termination losses, optional gradient norm extraction, evaluation at intervals, checkpoints, run metadata, and JSONL metrics logging.
- Analysis: latent convergence, embedding trajectories, execution-step distribution plots, threshold sweeps, and eval-only failure mode analysis with per-graph misclassification reports.
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
python -m src.data.dataset --preset
```

Build a dataset containing only graphs with a specific BF/BFS execution-length relation (for example, unequal lengths):
```bash
python -m src.data.filter_execution_length \
  --input data/train_dataset.npz \
  --output data/train_dataset_unequal_exec.npz \
  --relation unequal
```

Generate a custom single dataset (example: 100 graphs with 50 nodes each):
```bash
python -m src.data.dataset \
  --num-graphs 100 \
  --num-nodes 50 \
  --output data/dataset_100g_50n.npz
```

Train:
```bash
python -m src.train --config configs/baseline.yaml
```

Resume latest run:
```bash
python -m src.train --config configs/baseline.yaml --resume
```

Run code-quality checks:
```bash
conda run -n mlx ruff check src tests
conda run -n mlx python -m compileall -q src tests
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
- Optional filtering utility for execution-length relation (`unequal`, `equal`, `bf_gt_bfs`, `bfs_gt_bf`): `src.data.filter_execution_length`.

Datasets are saved as compressed `.npz` files with per‑graph keys (e.g., `edge_matrix_0`, `bf_distance_targets_0`, ...).

Dataset generation modes (`src.data.dataset`):
- `--preset` generates default `train/val/test` splits (`1500/100/100`, `20` nodes).
- Custom single dataset uses `--num-graphs`, `--num-nodes`, optional `--p`, `--m`, and `--output`.

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
--accuracies-only  In --eval-only mode, compute accuracies only (skip losses)
--analyze-failures  In --eval-only mode, write per-graph failure analysis JSON
--failure-split  Split for failure analysis: train|val|test (default: test)
--failure-max-graphs  Optional cap on number of graphs analyzed for failures
--failure-max-records  Max failed-graph records written to JSON
--failure-include-step-details  Include per-step mismatch counts per failed graph
--failure-debug-graphs  Comma-separated graph indices for debug execution traces
--failure-debug-top-k  Also dump debug traces for top-K failed graphs
--termination-mode  Override termination mode: head|distance
--termination-threshold  Override termination_distance_threshold
--termination-latent  Override termination_distance_latent
--disable-distance-termination-signal  Disable BCE termination supervision in distance mode
```
Task mapping:
- `bf` trains/evaluates BF distance, BF predecessor, BF termination
- `bfs` trains/evaluates BFS state, BFS termination
- `all` enables both groups
Notes:
- In `--eval-only`, `--config` takes precedence over `--run-dir/config_resolved.yaml`.
- `--accuracies-only` works for both head-mode and distance-mode checkpoints.
- You can force evaluation criterion with `--termination-mode distance` (or `head`) regardless of training mode.
- Threshold/latent overrides affect termination only when the effective mode is `distance`.

Eval-only examples:
```bash
# Accuracy-only evaluation for a standard (head-mode) run
python -m src.train --eval-only --run-dir runs/<run_name> --accuracies-only

# Evaluate the same checkpoint with distance-based termination at custom threshold
python -m src.train --eval-only --run-dir runs/<run_name> \
  --accuracies-only --termination-mode distance --termination-threshold 0.01 \
  --termination-latent processed

# Generate failure report + detailed debug dumps
python -m src.train --eval-only --run-dir runs/<run_name> --analyze-failures \
  --failure-split test --failure-debug-top-k 5
```
Failure-analysis outputs in `runs/<run_name>/analysis/`:
- `failure_modes_<split>.json` (per-graph failures + aggregated termination mispredict stats by step)
- `failure_termination_mispredict_distribution_<split>.png` (BF and BFS termination mispredict counts per execution step)
- optional debug traces under `failure_debug/<split>/`

## Termination Modes
Termination can be configured in `model`:
- `termination_mode`: `head` (default) or `distance`
- `termination_distance_latent`: `processed`, `encoded`, `encoded_bfs`, or `encoded_bf`
- `termination_distance`: `l2`, `mean_l2`, `l1`, `mse`
- `termination_distance_threshold`: fixed threshold for distance mode
- `termination_distance_signal`: whether termination BCE supervision is applied in distance mode

## Analysis
### Latent convergence (`src.analysis.latent_convergence`)
Measures how embeddings change over execution.
Key options:
- `--mode successive` (step‑to‑step change) or `--mode to_final` (distance to final step)
- `--latent` to choose representation:
  - `processed`: processor output
  - `encoded`: concatenated encoder outputs after layer norm
  - `encoded_bfs`: BFS encoder output only
  - `encoded_bf`: BF encoder output only
  - `processed_zero_bfs_input`: processor output when BFS encoder input is zeroed
  - `processed_zero_bf_input`: processor output when BF encoder input is zeroed
- `--converge-threshold` and `--converge-patience` to estimate convergence step
- Includes a distinct **terminal probe** in plots/JSON: one extra forward pass from final target state, measured as distance from the final analyzed latent

### Embedding trajectories + PCA (`src.analysis.embedding_trajectories`)
Extracts trajectories and produces trajectory‑wise and step‑wise PCA.
Key options:
- `--latent` to choose representation:
  - `processed`: processor output
  - `encoded`: concatenated encoder outputs after layer norm
  - `encoded_bfs`: BFS encoder output only
  - `encoded_bf`: BF encoder output only
  - `processed_zero_bfs_input`: processor output when BFS encoder input is zeroed
  - `processed_zero_bf_input`: processor output when BF encoder input is zeroed
- `--node-agg` (`max`, `min`, `mean`) to collapse node dimension per step
- `--graph-index` to analyze one specific graph (bypasses step-policy filtering)
- `--step-policy` (`common`, `fixed`, `min`, `max`) to align graphs by execution length
- `--steps` when using `--step-policy fixed`
- `--extra-steps` to fake-continue execution after algorithm termination by reusing final BF/BFS states as inputs
- When `--extra-steps > 0`, PCA bases are fit using only non-extra execution steps; extra-step tensors are projected into that fixed basis (not used to fit it)
- `--pca` (`none`, `step`, `trajectory`, `both`) and `--pca-components`
- `--plot` to save 2D PCA plots with explained variance in the title; trajectory-wise uses a gradient color mapping, step-wise uses one discrete color per execution step with subtle per-graph connecting lines, and both include legends. Step-wise plots also mark terminal probes (`X`) and their mean (`*`).
Outputs:
- `trajectories.npz` (N x T x D + flattened matrices + indices + terminal probes)
- `pca_step.npz` / `pca_trajectory.npz` with projections, components, mean, variance
- `pca_step_completion_probes.npz` when step PCA is enabled (projected terminal probes)
- Optional plots in the output directory

### Execution-length step overlay (`src.analysis.execution_length_step_overlay`)
Runs `embedding_trajectories` repeatedly for fixed execution lengths and overlays only the **average step coordinate trajectory** from each run.
Key options:
- `--run-dir` run containing checkpoints/config
- `--dataset` optional dataset override (`.npz`)
- `--latent` representation (`processed|encoded|encoded_bfs|encoded_bf|processed_zero_bfs_input|processed_zero_bf_input`)
- `--steps-min` / `--steps-max` sweep the **base** execution lengths (before fake continuation)
- `--node-agg` (`max|min|mean`)
- `--extra-steps` fake continuation steps appended after termination; each sub-run uses `total_steps = base_steps + extra_steps`
- `--keep-step-plots` to also save each per-step run PCA plot
Additionally, each series now includes a distinct **terminal-probe marker**: for graphs in that execution-length bucket, the script performs one extra forward pass from the final target state, projects those embeddings into the same per-run PCA basis, and plots their average location.
When `--extra-steps > 0`, those per-run PCA bases are fit without extra-step tensors (extra steps are projected only).
Outputs:
- `analysis/execution_length_step_overlay/avg_step_coordinate_overlay.png`
- `analysis/execution_length_step_overlay/avg_step_coordinate_overlay.json` (includes `completion_avg_coordinate` per series)
- per-step metadata under `analysis/execution_length_step_overlay/per_step_runs/steps_XX/`
Example:
```bash
python -m src.analysis.execution_length_step_overlay \
  --run-dir runs/<run_name> \
  --dataset data/dataset_20g_200n.npz \
  --latent processed_zero_bfs_input \
  --steps-min 0 --steps-max 8 \
  --extra-steps 2
```

### Dataset step distribution (`src.analysis.dataset_step_distribution`)
Visualizes BF and BFS execution-step distributions for a dataset.
Key options:
- `--dataset` input dataset (`.npz`, required)
- `--max-graphs` optional cap for quick inspection
- `--output-dir` optional output location (default: `analysis/dataset_step_distribution`)
Outputs:
- `execution_step_distribution.png` (BF and BFS bar plots)
- `execution_step_distribution.json` (distributions, summary stats, relation counts)
Example:
```bash
python -m src.analysis.dataset_step_distribution \
  --dataset data/test_dataset.npz
```

### Termination threshold sweep (`src.analysis.termination_threshold_sweep`)
Evaluates one checkpoint across a threshold grid and plots test (or chosen split) termination accuracy curves.
Key options:
- `--split` (`train`, `val`, `test`; default `test`)
- `--tasks` (`all`, `bf`, `bfs`)
- `--termination-mode` (`distance` default, or `head`) used during sweep evaluation
- `--termination-latent` to override distance latent (`processed|encoded|encoded_bfs|encoded_bf`)
- `--thresholds` for an explicit comma-separated list, or `--threshold-min/--threshold-max/--threshold-step` (default `0.005`) for step sweep
Notes:
- Legacy head-trained runs are supported because sweep mode defaults to `distance`.
Outputs:
- `<split>_threshold_vs_termination_accuracy.png` (BF and BFS termination accuracy vs threshold)
- `<split>_threshold_sweep.json` (thresholds, losses, and accuracies)

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
--latent         Which latent to probe (processed|encoded|encoded_bfs|encoded_bf|processed_zero_bfs_input|processed_zero_bf_input)
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
- `graph_<index>_distances.json` (`terminal_probe_distance` included)

Outputs (dataset):
- `dataset_distance.png`
- `dataset_stats.json` (mean/std/counts + `terminal_probe_distances`, probe mean/std, metadata)

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

## Logging and checkpoints
- Metrics are appended to `metrics.jsonl` with `step`, `split`, `timestamp`, and `metrics`.
- Optional Weights & Biases logging is enabled via config (requires `wandb`).
- Checkpoints are saved as `model.safetensors` and `optimizer.safetensors` with `checkpoint.json` metadata and a `latest.json` pointer.

## Evaluation utilities
`src/utils/eval.py` provides:
- `calculate_losses_and_accuracies` (matches training loss logic).
- `calculate_accuracies` (accuracy‑only variant).
- `analyze_failure_modes` (per-graph misclassification analysis and failure ranking).
- `print_execution_details` (verbose step‑by‑step diagnostics).
- `extract_per_head_magnitude_grads` (per‑component gradient norms).

## Reproducibility
- `set_seed` for deterministic MLX RNG.
- `meta.json` captures git SHA/branch/dirty hash and environment info (Python + MLX versions, platform, hostname).
- Resolved configs are stored per‑run (`config_resolved.yaml`).
