#!/usr/bin/env bash
set -euo pipefail

#python -m src.train --config experiments/graph_trasnfer_matrix/configs/bf__to__bf.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/bf__to__bfs.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/bf__to__prim.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/bf__to__dijkstra.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/bf__to__dag_shortest_paths.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/bfs__to__bf.yaml
#python -m src.train --config experiments/graph_trasnfer_matrix/configs/bfs__to__bfs.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/bfs__to__prim.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/bfs__to__dijkstra.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/bfs__to__dag_shortest_paths.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/prim__to__bf.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/prim__to__bfs.yaml
#python -m src.train --config experiments/graph_trasnfer_matrix/configs/prim__to__prim.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/prim__to__dijkstra.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/prim__to__dag_shortest_paths.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/dijkstra__to__bf.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/dijkstra__to__bfs.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/dijkstra__to__prim.yaml
#python -m src.train --config experiments/graph_trasnfer_matrix/configs/dijkstra__to__dijkstra.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/dijkstra__to__dag_shortest_paths.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/dag_shortest_paths__to__bf.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/dag_shortest_paths__to__bfs.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/dag_shortest_paths__to__prim.yaml
python -m src.train --config experiments/graph_trasnfer_matrix/configs/dag_shortest_paths__to__dijkstra.yaml
#python -m src.train --config experiments/graph_trasnfer_matrix/configs/dag_shortest_paths__to__dag_shortest_paths.yaml
