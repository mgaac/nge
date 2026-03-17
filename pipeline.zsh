#!/usr/bin/env zsh
set -e

caffeinate python3 -m src.train --config configs/bf.yaml
caffeinate python3 -m src.train --config configs/bfs.yaml
caffeinate python3 -m src.train --config configs/prim.yaml
