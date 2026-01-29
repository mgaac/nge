# NGE (Neural Graph Execution)

Train a message-passing neural network to emulate graph algorithms (Bellman-Ford + BFS) on synthetic graphs using MLX.

## What it does
- Generates synthetic graphs, runs BF/BFS to create step-wise targets, and saves datasets as `.npz`
- Trains the NGE model with per-task losses and termination heads
- Logs metrics to `runs/<run-name>/metrics.jsonl` and saves checkpoints

## Features
- MLX-based training with lazy evaluation and unified memory model
- Synthetic graph generation (Erdos-Renyi + Barabasi-Albert) with weighted, bidirectional edges
- Dual-task supervision: BF distance + predecessor, BFS reachability, plus termination signals
- Deterministic, config-driven experiments with reproducibility metadata

## Project layout
- `src/model/` – NGE architecture (MPNN + BF/BFS decoders)
- `src/data/` – dataset generation + load/save utilities
- `src/train.py` – training/eval workflow
- `configs/` – experiment configs
- `runs/` – outputs (created at runtime)

## Quick start
Install deps:
```bash
pip install -r requirements.txt
```

Generate datasets:
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

## CLI flags (src.train)
```text
--config   Path to a YAML experiment config (required).
--resume   Resume from the latest run in runs/ (uses the resolved config in that run).
--run-dir  Explicit run directory to resume (only used with --resume).
```

## Outputs
- `runs/<run-name>/config_resolved.yaml`
- `runs/<run-name>/meta.json`
- `runs/<run-name>/metrics.jsonl`
- `runs/<run-name>/checkpoints/`

## Configs
YAML in `configs/` controls model, training, data paths, and logging.

## Tests
```bash
pytest
```
