#!/usr/bin/env zsh
set -e

caffeinate python -m src.train --config configs/dijkstra.yaml
caffeinate python -m src.train --config configs/prim.yaml
caffeinate python -m src.train --config configs/bfs.yaml
caffeinate python -m src.train --config configs/bf.yaml
caffeinate python -m src.train --config configs/dag_shortest_paths.yaml
