# NGE (Neural Graph Execution)

NGE is an MLX research codebase for learning algorithm execution on graphs.  
It trains a message-passing model to predict algorithm state transitions with a shared processor and per-algorithm heads. The legacy multitask setup covers Bellman-Ford, BFS, and Prim. Phase-1 CLRS-style extensions now add single-task support for Dijkstra and DAG shortest paths.

## Capability overview

| Area | What is implemented |
| --- | --- |
| Data | Synthetic graph datasets for the legacy BF/BFS/Prim multitask setup, a unified `clrs_algo` all-task dataset (all five algorithms per graph), plus single-task Dijkstra and DAG shortest paths datasets, schema-driven `.npz` serialization |
| Model | Shared encoder/processor architecture with configurable algorithm sets and per-algorithm heads (`state_mask`, `shortest_path`, `mst`) |
| Training | Config-driven runs, checkpointing, resume, JSONL metrics, optional W&B logging, explicit `mx.eval(...)` barriers for stable MLX gradient accumulation |
| Evaluation | Loss/accuracy reporting, eval-only mode, configurable task masking, failure-mode analysis and debug traces |
| Analysis | Latent convergence, embedding trajectories + PCA, execution-length overlays, threshold sweeps, dataset step distributions, extra-step dynamics for the legacy BF/BFS/Prim setup |
| Reproducibility | Seed control, run metadata (`meta.json`), resolved config snapshots |

## Repository layout

- `src/model/` model definitions (`NGE`, aggregation enum, heads).
- `src/data/` dataset generation/loading and filtering utilities.
- `src/train.py` main training and evaluation entrypoint.
- `src/utils/` config, logging, checkpointing, evaluation, reproducibility helpers.
- `src/analysis/` post-training analysis scripts.
- `configs/` experiment configurations.
- `skills/` project-local Codex skills for research workflow support.
- `tests/` workflow and checkpointing tests.

## Setup

Run from repository root.

```bash
conda run -n mlx pip install -r requirements.txt
```

MLX targets Apple Silicon. This project is intended for macOS + Metal.
Use `conda run -n mlx ...` for all commands and automation scripts. Do not rely on shell activation alone; `python`/`python3` may still resolve outside the intended environment.

## Quick start

### 1. Generate default datasets

Generates the legacy multitask `train/val/test` splits (`1500/100/100`, 20 nodes) into `data/`:

```bash
conda run -n mlx python -m src.data.dataset --preset --output-dir data
```

Generate Dijkstra-only splits:

```bash
conda run -n mlx python -m src.data.dataset \
  --preset \
  --task dijkstra \
  --output-dir data
```

Generate DAG shortest paths-only splits:

```bash
conda run -n mlx python -m src.data.dataset \
  --preset \
  --task dag_shortest_paths \
  --output-dir data
```

Generate unified all-task splits where each graph executes all five algorithms (`bf`, `bfs`, `prim`, `dijkstra`, `dag_shortest_paths`) on the same input graph:

```bash
conda run -n mlx python -m src.data.dataset \
  --preset \
  --task clrs_algo \
  --output-dir data
```

Train all currently supported graph tasks with a shared processor, CLRS-style, after generating all three dataset families:

```bash
conda run -n mlx python -m src.train --config configs/clrs_graphs.yaml
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
conda run -n mlx python -m src.train --config configs/prims_bf_bfs.yaml
```

Task selection can be stored in the config:

```yaml
training:
  tasks: prim
```

The CLI still overrides it when needed:

```bash
conda run -n mlx python -m src.train --config configs/prims_bf_bfs.yaml --tasks bfs
```

Algorithm branches are also config-driven. For example, Dijkstra uses a single shortest-path branch:

```yaml
model:
  algorithms:
    - dijkstra
training:
  tasks: dijkstra
```

The single-task configs (`configs/bf.yaml`, `configs/bfs.yaml`, `configs/prim.yaml`, `configs/dijkstra.yaml`, `configs/dag_shortest_paths.yaml`) now pin `model.algorithms` to exactly one task. That is the correct basis for canonical frozen-processor transfer.

Transfer-style training is also config-driven. For example, to reuse a BF-trained processor for BFS while keeping only the shared processor from the source checkpoint:

```yaml
model:
  algorithms:
    - bfs
training:
  tasks: bfs
  init_checkpoint: runs/<bf_run>/checkpoints/step_<...>
  init_checkpoint_modules:
    - processor
  freeze_modules:
    - processor
  reset_modules:
    - bfs_encoder
    - bfs_decoder
    - bfs_termination
```

### 5. Resume latest run

```bash
conda run -n mlx python -m src.train --config configs/prims_bf_bfs.yaml --resume
```

Resume now restores checkpoint state before loading validation/test splits. On
large CLRS mixed-task datasets this removes the previous startup stall caused by
eagerly materializing all three splits at process start.

### 6. Eval-only

```bash
conda run -n mlx python -m src.train \
  --eval-only \
  --run-dir runs/<run_name> \
  --accuracies-only
```

Evaluation is autoregressive over the explicit algorithm-state inputs. Step `t = 0`
uses the dataset's initial state, and every later step feeds back the model's
own step `t - 1` predictions as the next recurrent state inputs. Training
remains teacher-forced.

## Configuration

Available configs:

- `configs/bf.yaml`
- `configs/bfs.yaml`
- `configs/prim.yaml`
- `configs/prims_bf_bfs.yaml`
- `configs/termination.yaml`
- `configs/bfs-transfer.yaml`
- `configs/prim-transfer.yaml`
- `configs/dijkstra.yaml`
- `configs/dag_shortest_paths.yaml`
- `configs/clrs_graphs.yaml`
- `configs/transfer_matrix.template.yaml`

Core schema:

```yaml
name: <string>
model:
  embed_dim: <int>
  residual_connections: <bool>
  agg_fn: <SUM|AVG|MIN|MAX>
  num_mp_layers: <int>
  dropout: <float>
  algorithms: [<bf|bfs|prim|dijkstra|dag_shortest_paths>, ...]
  termination_mode: <head|distance>
  termination_distance_latent: <processed|encoded|encoded_<algorithm>>
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
  tasks: <all|bf|bfs|prim|dijkstra|dag_shortest_paths>
  init_checkpoint: <path|null>
  init_checkpoint_modules: [<module_path>, ...]
  freeze_modules: [<module_path>, ...]
  reset_modules: [<module_path>, ...]
data:
  train_path: <path>
  val_path: <path>
  test_path: <path>
  task_paths:
    <algorithm>:
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
| `--tasks {all,bf,bfs,prim,dijkstra,dag_shortest_paths}` | Override `training.tasks` for train/eval |
| `--eval-only` | Skip training and only evaluate |
| `--checkpoint` | Explicit checkpoint file/dir for eval |
| `--accuracies-only` | Eval-only: skip losses |
| `--analyze-failures` | Write failure analysis JSON and plots |
| `--failure-split` | Split for failure analysis |
| `--termination-mode` | Override `head` or `distance` at eval/train time |
| `--termination-threshold` | Override distance termination threshold |

`training.init_checkpoint` loads weights before training starts. `training.init_checkpoint_modules` restricts that initialization to specific module paths; if it is empty, the whole checkpoint is loaded. `training.freeze_modules` zeroes gradients for the named module prefixes during optimization. `training.reset_modules` reinitializes named modules from a fresh model after the checkpoint is loaded. Valid module names are derived from `model.algorithms`; e.g. `processor`, `bf_encoder`, `dijkstra_decoder`, or `dag_shortest_paths_termination`.

All evaluation paths that go through `src.train --eval-only`, validation during
training, failure analysis, or `print_execution_details` use the same
autoregressive rollout semantics for algorithm-state inputs.

When `data.task_paths` is present, training switches to CLRS-style task mixing: each algorithm is loaded from its own dataset family, each sample activates exactly one task head, and the shared processor is optimized across the union of all selected tasks.

## Transfer matrices

`src.analysis.transfer_matrix` scaffolds the canonical frozen-processor experiment

`x -> E_tau -> P_i (frozen) -> D_tau`

where each source processor `P_i` comes from a single-task source run and each target config keeps only the target algorithm branch.

Example workflow:

```bash
conda run -n mlx python -m src.analysis.transfer_matrix scaffold \
  --manifest configs/transfer_matrix.template.yaml \
  --output-dir experiments/graph_transfer_matrix
```

This writes one config per `(source, target)` pair plus `train_commands.sh`. Run the generated commands after filling in the source checkpoint paths.
The generated `train_commands.sh` intentionally uses plain `python -m src.train ...`; run it only after activating the intended environment yourself.

After training finishes, aggregate the completed runs into accuracy matrices:

```bash
conda run -n mlx python -m src.analysis.transfer_matrix collect \
  --matrix-dir experiments/graph_transfer_matrix \
  --split test
```

If you change evaluation semantics and need to recompute the matrix from fresh
`eval-only` artifacts, rerun eval on the source bank plus the discovered
transfer runs and then collect from those JSON files:

```bash
./experiments/graph_trasnfer_matrix/eval_test_autoreg_commands.sh
```

That script writes refreshed per-run `analysis/eval_only_autoreg.json` files and
rebuilds the matrix as `test_autoreg_*` under `experiments/graph_trasnfer_matrix/`.
It executes each eval as `conda run -n <env> --no-capture-output` and retries
failed runs up to `MAX_RETRIES` times (default `5`). Override defaults with
`CONDA_ENV=<env>`, `MAX_RETRIES=<n>`, and `WANDB_MODE=<mode>`.

The collector scans `runs/` for the latest run matching each generated experiment name, then writes:

- a primary transfer relative-change matrix (`<split>_primary_transfer_delta.csv/.png`)
- the corresponding raw primary matrix (`<split>_primary_transfer_matrix_raw.csv/.png`)
- one raw and one delta matrix per task metric (for example `test_bfs_state_delta.csv`)
- `pair_results` metadata in `<split>_transfer_matrix_summary.json`

Primary matrix values are built per target task as the mean of all non-termination
accuracies for that task (for example BF averages `bf_distance` and
`bf_predecessor`; Prim averages `prim_state`, `prim_key`, and `prim_predecessor`).
Scaffolding validates processor compatibility before emitting configs, and training re-checks the same constraint before copying the frozen processor. Legacy multi-branch source runs are rejected for the canonical single-task transfer matrix.
If the off-diagonal transfer runs are present but the diagonal self-transfer runs were never trained, the collector fills the diagonal from the source bank runs referenced by the manifest.
The `collect` path only reads configs and `metrics.jsonl`; it does not need to load MLX models.
Primary relative-change matrices are column-wise: for each target task, the baseline is the own-processor run for that task (diagonal self-transfer when available, otherwise source-bank baseline), and every entry is reported as `100 * (transfer_score - baseline) / baseline`.
To verify that the source bank and the completed transfer runs used the same shared hyperparameters and all reached the configured epoch count, run:

```bash
conda run -n mlx python -m src.analysis.transfer_matrix audit \
  --matrix-dir experiments/graph_transfer_matrix \
  --run-root runs
```

This writes `run_consistency_audit.json` inside the matrix directory. The audit checks the standard shared model/training/logging hyperparameters, the canonical frozen-processor invariants (`init_checkpoint_modules=["processor"]`, `freeze_modules=["processor"]`, target-only reset modules), and that the latest `test` record appears at `training.epochs` for both source-bank and transfer runs.

## Analysis scripts

| Script | Output |
| --- | --- |
| `src.analysis.latent_convergence` | Distance-to-final / successive latent change plots + JSON for any configured algorithm set |
| `src.analysis.embedding_trajectories` | Trajectories + step/trajectory PCA artifacts for any configured algorithm set |
| `src.analysis.per_step_group_pca` | Independent PCA at each fixed execution step `t` (across graphs), with per-step components/projections/variance summaries |
| `src.analysis.step_dynamics_diagnostics` | Global fixed-basis PCA projection, per-step/per-algorithm covariance spectra, cumulative variance curves, and principal-angle subspace drift |
| `src.analysis.algorithm_subspace` | Algorithm-specific latent-delta subspace overlap / explained-variance analysis from `embedding_trajectories` artifacts |
| `src.analysis.execution_segment_pca` | PCA over simultaneous autoregressive rollout segments grouped by the number of branches still active |
| `src.analysis.execution_length_step_overlay` | Overlayed average trajectories across execution lengths |
| `src.analysis.extra_step_state_dynamics` | Fake-continuation dynamics and stabilization diagnostics across BF/BFS/Prim states |
| `src.analysis.dataset_step_distribution` | Per-algorithm step distribution plots + summary JSON for the algorithms present in a dataset |
| `src.analysis.pc_monotonicity` | Spearman/monotonicity checks between sequential latent distances and PCA coordinates near task termination |
| `src.analysis.termination_threshold_sweep` | Threshold vs per-algorithm termination-accuracy curves |
| `src.analysis.transfer_matrix` | Frozen-processor transfer-matrix config generation and result aggregation |
| `src.analysis.transfer_subspace_correlation` | Correlate ordered transfer deltas against directional cross-explained-variance and symmetric subspace overlap |
| `src.analysis.eval_test_100n_triplet` | Test-only 100n eval for normal/frozen-random/identity runs + comparison CSV/JSON/plot |

Example:

```bash
conda run -n mlx python -m src.analysis.dataset_step_distribution \
  --dataset data/test_dataset.npz
```

Run the canonical test-only 100n comparison for the three CLRS processor variants (normal, frozen-random, identity):

```bash
conda run -n mlx python -m src.analysis.eval_test_100n_triplet
```

The comparison plot for this triplet now reports, per task, the mean of all
non-termination accuracy heads (for example BF averages distance+predecessor,
Prim averages state+key+predecessor).
It also overlays a fixed baseline line using the provided expected accuracies:
`bfs=0.679`, `bf=0.291`, `prim=0.314`, `dijkstra=0.161`, `dag_shortest_paths=0.260`.

`src.analysis.embedding_trajectories` supports `--pca-components 2` and `--pca-components 3`; when plotting is enabled, three components produce a 3D PCA figure plus pairwise perspective views for `PC1-PC2`, `PC1-PC3`, and `PC2-PC3`. Step-wise PCA plots also overlay BF→BFS and BF→Prim reference segments built from the mean algorithm termination steps of the selected graphs.
Step-wise plots now use a compact step-index colorbar instead of per-step legend entries, so large step ranges do not consume plot area with oversized keys.
`src.analysis.per_step_group_pca` answers a complementary question: for each fixed execution step `t`, it collects all graph embeddings at that step and fits a separate PCA inside that step-group. The script writes `per_step_group_pca.npz` (per-step projections/components/means/variance) and `summary.json` (per-step explained-variance entries).
`src.analysis.step_dynamics_diagnostics` runs four diagnostics on top of an `embedding_trajectories` artifact in one pass: (1) fit one global PCA over all steps/graphs and project every step cloud into that fixed basis, (2) compute per-algorithm active-set covariance spectra at each step, (3) plot cumulative per-step variance (`lambda_1`, `lambda_1+lambda_2`, `lambda_1+lambda_2+lambda_3`) over time, and (4) measure consecutive-step subspace drift via principal angles. It writes `diagnostics.npz`, `summary.json`, and four plots (`cumulative_variance_over_time.png`, `global_basis_cumulative_variance_over_time.png`, `principal_angle_drift.png`, `per_algorithm_cumulative_variance_top3.png`).
`src.analysis.embedding_trajectories` also supports `--execution-window {all,<algorithm>}` for every supported algorithm (`bf`, `bfs`, `prim`, `dijkstra`, `dag_shortest_paths`) and truncates the analyzed prefix to the steps where that algorithm is still executing.
Reference endpoints are clamped to the plotted step window, so algorithms whose mean end step falls one step beyond the visible range (for example DAG shortest paths at the shared horizon) still get a plotted endpoint marker.
When `src.analysis.embedding_trajectories` is called with an explicit `--dataset` and no `--output-dir`, it writes to `embedding_trajectories_<dataset_stem>` to avoid overwriting previous artifacts from other datasets. When `--isolate-input-algorithm <algo>` is set, it can either zero other recurrent inputs (`--isolate-other-inputs-mode zero_others`, default) or pin them to their final state (`--isolate-other-inputs-mode final_others`) while only the selected algorithm keeps its step-varying inputs. The `final_others` mode writes to `embedding_trajectories_<dataset_stem>_iso_<algo>_final_others`.
`src.analysis.algorithm_subspace` consumes one or more `embedding_trajectories` artifacts, uses the algorithm lists stored in their metadata, converts trajectories into per-step latent deltas, partitions those deltas by algorithm-active or algorithm-exclusive execution phases, and compares the resulting PCA subspaces via mean canonical correlations and cross explained-variance heatmaps. When invoked with only `--run-dir`, it auto-discovers full-window `embedding_trajectories*` artifacts under `runs/<name>/analysis/`. This is the intended path for CLRS-style mixed-task runs: keep BF/BFS/Prim in the shared legacy artifact and generate separate `embedding_trajectories_<algorithm>` artifacts for task-specific datasets such as Dijkstra and DAG shortest paths. Algorithms without active steps in the available sources are skipped and reported explicitly in the summary JSON. Use `--algorithm-order bf,bfs,prim,dijkstra,dag_shortest_paths` when you need a specific row/column and plot order, for example while comparing isolated single-task models.
For strict per-algorithm input isolation (basis for each algorithm built from runs where only that algorithm's recurrent inputs are non-zero), generate one isolated trajectories artifact per algorithm and run subspace with `--membership-mode isolated_inputs`:

```bash
for algo in bf bfs prim dijkstra dag_shortest_paths; do
  conda run -n mlx python -m src.analysis.embedding_trajectories \
    --run-dir runs/<full_model_run> \
    --dataset data/val_clrs_algo_dataset.npz \
    --isolate-input-algorithm "$algo"
done

conda run -n mlx python -m src.analysis.algorithm_subspace \
  --membership-mode isolated_inputs \
  --tasks all \
  --trajectories-dir runs/<full_model_run>/analysis/embedding_trajectories_val_clrs_algo_dataset_iso_bf \
  --trajectories-dir runs/<full_model_run>/analysis/embedding_trajectories_val_clrs_algo_dataset_iso_bfs \
  --trajectories-dir runs/<full_model_run>/analysis/embedding_trajectories_val_clrs_algo_dataset_iso_prim \
  --trajectories-dir runs/<full_model_run>/analysis/embedding_trajectories_val_clrs_algo_dataset_iso_dijkstra \
  --trajectories-dir runs/<full_model_run>/analysis/embedding_trajectories_val_clrs_algo_dataset_iso_dag_shortest_paths
```

To isolate each basis with non-selected algorithms fixed at final state instead of zero:

```bash
for algo in bf bfs prim dijkstra dag_shortest_paths; do
  conda run -n mlx python -m src.analysis.embedding_trajectories \
    --run-dir runs/<full_model_run> \
    --dataset data/val_clrs_algo_dataset.npz \
    --isolate-input-algorithm "$algo" \
    --isolate-other-inputs-mode final_others
done
```
`src.analysis.execution_segment_pca` answers a different question: it runs a simultaneous autoregressive rollout for every configured branch, tracks how many branches are still active at each step, partitions the latent execution states by that active-count, and fits a separate PCA inside each segment. By default, branch activity is defined by branch-specific recurrent state change (`--activity-source state_change`), which is the correct default for shared-distance-termination runs because `termination_mode=distance` emits the same termination logit for every branch. Set `--activity-source termination` only when you explicitly want to segment by the configured termination head/distance rule instead. The script writes per-segment PCA scatter plots plus a grouped explained-variance plot for the first `k` PCs (default `k=2`). When `data.task_paths` is present and `--dataset` is omitted, it analyzes all unique split datasets referenced by the config, so a full CLRS run can contribute graphs from the legacy BF/BFS/Prim family plus Dijkstra and DAG-shortest-paths families in one pass. This is the correct path when you want the model's own simultaneous `5 active -> 4 active -> ...` rollout dynamics instead of ground-truth dataset execution lengths:

```bash
conda run -n mlx python -m src.analysis.execution_segment_pca \
  --run-dir runs/<full_model_run> \
  --max-rollout-steps 64 \
  --activity-source state_change
```

`src.analysis.transfer_subspace_correlation` joins a transfer-matrix summary with an `algorithm_subspace` summary, constructs ordered source→target pairs, and reports Pearson/Spearman correlations between transfer deltas and three geometric predictors: target-in-source directional containment, the flipped directional read, and symmetric mean canonical correlation. It also writes scatter plots and an ordered-pair CSV so directional mistakes are easy to spot.
`src.analysis.pc_monotonicity` and `src.analysis.termination_threshold_sweep` now respect the run's configured algorithm set; use `--tasks <algorithm>` to analyze only one branch.
For configs with `data.task_paths`, analysis defaults to the shared legacy BF/BFS/Prim dataset when one exists. To analyze task-specific datasets such as Dijkstra or DAG shortest paths, pass `--dataset` explicitly.

## Codex research audit skill

This repository now includes a project-local Codex skill at `skills/deep-learning-experiment-sanitizer/`.
It is designed to critically audit deep learning experiments for:

- split and duplicate leakage
- target leakage and privileged inputs
- shortcut learning and confounds
- invalid evaluation claims or test-set contamination
- architecture-amplified failure modes, including graph-learning shortcuts

Use it when reviewing configs, dataset code, training loops, evaluation scripts, ablations, or result summaries that might otherwise overstate what a model has learned.

If you want Codex to auto-discover it globally, copy or symlink that folder into `${CODEX_HOME:-$HOME/.codex}/skills/`.

## Outputs per run

Each training run creates `runs/<run_name>/` with:

- `config_resolved.yaml`
- `meta.json` (git/environment/seed/config metadata)
- `metrics.jsonl`
- `checkpoints/`
- `analysis/` (if analysis/eval modes are enabled)

When W&B is enabled, metrics are logged under split namespaces such as `train/*`, `val/*`, `train_eval/*`, and `test/*`, all indexed by the explicit `epoch` metric.
Checkpoint retention is controlled by `logging.checkpoint_keep_last`; after training, older checkpoints are pruned and only the most recent `N` are kept.

## Dataset schema

Every graph record stores:

- `edge_matrix`, `num_nodes`, `source_node`
- algorithm-specific target tensors, depending on the dataset task family

Current schemas:

| Dataset task | Target tensors |
| --- | --- |
| `multitask` | `bf_distance_targets`, `bf_predecessor_targets`, `bfs_state_targets`, `prim_state_targets`, `prim_key_targets`, `prim_predecessor_targets` |
| `dijkstra` | `dijkstra_distance_targets`, `dijkstra_predecessor_targets` |
| `dag_shortest_paths` | `dag_shortest_paths_distance_targets`, `dag_shortest_paths_predecessor_targets` |

The recurrent model input is always the concatenation of the previous hidden state with the per-node algorithm-state features implied by `model.algorithms`. For the legacy multitask model this remains `[bfs_state, bf_distance, prim_state, prim_key]` to stay checkpoint-compatible with existing runs.

## Current scope

- Train/eval/data plumbing is dynamic for `bf`, `bfs`, `prim`, `dijkstra`, and `dag_shortest_paths`.
- `data.task_paths` enables CLRS-style mixed-task training over separate dataset families with one shared processor.
- The post-training trajectory/PCA analysis scripts are still primarily wired to the legacy BF/BFS/Prim workflow.

## Quality checks

```bash
conda run -n mlx python -m compileall -q src tests
conda run -n mlx python -m pytest -q
```

If `pytest` is unavailable in your environment, install dependencies from `requirements.txt` first. MLX runtime checks require a working MLX environment on supported hardware.
