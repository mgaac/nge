#!/usr/bin/env bash
set -euo pipefail

python -m src.train --eval-only --run-dir runs/iso/20260319-155310_647125b-dirty1e7a69fa_bf --config configs/bf_test_100n.yaml
cp runs/iso/20260319-155310_647125b-dirty1e7a69fa_bf/analysis/eval_only.json runs/iso/20260319-155310_647125b-dirty1e7a69fa_bf/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/iso/20260319-133649_647125b-dirty1e7a69fa_bfs --config configs/bfs_test_100n.yaml
cp runs/iso/20260319-133649_647125b-dirty1e7a69fa_bfs/analysis/eval_only.json runs/iso/20260319-133649_647125b-dirty1e7a69fa_bfs/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/iso/20260319-100544_647125b-dirty1e7a69fa_prim --config configs/prim_test_100n.yaml
cp runs/iso/20260319-100544_647125b-dirty1e7a69fa_prim/analysis/eval_only.json runs/iso/20260319-100544_647125b-dirty1e7a69fa_prim/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/iso/20260319-082943_647125b-dirty1e7a69fa_dijkstra --config configs/dijkstra_test_100n.yaml
cp runs/iso/20260319-082943_647125b-dirty1e7a69fa_dijkstra/analysis/eval_only.json runs/iso/20260319-082943_647125b-dirty1e7a69fa_dijkstra/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/iso/20260319-162819_647125b-dirty1e7a69fa_dag-shortest-paths --config configs/dag_shortest_paths_test_100n.yaml
cp runs/iso/20260319-162819_647125b-dirty1e7a69fa_dag-shortest-paths/analysis/eval_only.json runs/iso/20260319-162819_647125b-dirty1e7a69fa_dag-shortest-paths/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-022504_647125b-dirty340505db_graph-transfer-matrix__bf__to__bfs --config configs/bfs_test_100n.yaml
cp runs/20260320-022504_647125b-dirty340505db_graph-transfer-matrix__bf__to__bfs/analysis/eval_only.json runs/20260320-022504_647125b-dirty340505db_graph-transfer-matrix__bf__to__bfs/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-030125_647125b-dirty340505db_graph-transfer-matrix__bf__to__prim --config configs/prim_test_100n.yaml
cp runs/20260320-030125_647125b-dirty340505db_graph-transfer-matrix__bf__to__prim/analysis/eval_only.json runs/20260320-030125_647125b-dirty340505db_graph-transfer-matrix__bf__to__prim/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-055745_647125b-dirty340505db_graph-transfer-matrix__bf__to__dijkstra --config configs/dijkstra_test_100n.yaml
cp runs/20260320-055745_647125b-dirty340505db_graph-transfer-matrix__bf__to__dijkstra/analysis/eval_only.json runs/20260320-055745_647125b-dirty340505db_graph-transfer-matrix__bf__to__dijkstra/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-072425_647125b-dirty340505db_graph-transfer-matrix__bf__to__dag_shortest_paths --config configs/dag_shortest_paths_test_100n.yaml
cp runs/20260320-072425_647125b-dirty340505db_graph-transfer-matrix__bf__to__dag_shortest_paths/analysis/eval_only.json runs/20260320-072425_647125b-dirty340505db_graph-transfer-matrix__bf__to__dag_shortest_paths/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-085257_647125b-dirty340505db_graph-transfer-matrix__bfs__to__bf --config configs/bf_test_100n.yaml
cp runs/20260320-085257_647125b-dirty340505db_graph-transfer-matrix__bfs__to__bf/analysis/eval_only.json runs/20260320-085257_647125b-dirty340505db_graph-transfer-matrix__bfs__to__bf/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-092650_647125b-dirty340505db_graph-transfer-matrix__bfs__to__prim --config configs/prim_test_100n.yaml
cp runs/20260320-092650_647125b-dirty340505db_graph-transfer-matrix__bfs__to__prim/analysis/eval_only.json runs/20260320-092650_647125b-dirty340505db_graph-transfer-matrix__bfs__to__prim/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-122510_647125b-dirty340505db_graph-transfer-matrix__bfs__to__dijkstra --config configs/dijkstra_test_100n.yaml
cp runs/20260320-122510_647125b-dirty340505db_graph-transfer-matrix__bfs__to__dijkstra/analysis/eval_only.json runs/20260320-122510_647125b-dirty340505db_graph-transfer-matrix__bfs__to__dijkstra/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-154923_647125b-dirty340505db_graph-transfer-matrix__bfs__to__dag_shortest_paths --config configs/dag_shortest_paths_test_100n.yaml
cp runs/20260320-154923_647125b-dirty340505db_graph-transfer-matrix__bfs__to__dag_shortest_paths/analysis/eval_only.json runs/20260320-154923_647125b-dirty340505db_graph-transfer-matrix__bfs__to__dag_shortest_paths/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-184435_647125b-dirty340505db_graph-transfer-matrix__prim__to__bf --config configs/bf_test_100n.yaml
cp runs/20260320-184435_647125b-dirty340505db_graph-transfer-matrix__prim__to__bf/analysis/eval_only.json runs/20260320-184435_647125b-dirty340505db_graph-transfer-matrix__prim__to__bf/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-211735_647125b-dirty340505db_graph-transfer-matrix__prim__to__bfs --config configs/bfs_test_100n.yaml
cp runs/20260320-211735_647125b-dirty340505db_graph-transfer-matrix__prim__to__bfs/analysis/eval_only.json runs/20260320-211735_647125b-dirty340505db_graph-transfer-matrix__prim__to__bfs/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260320-213710_647125b-dirty340505db_graph-transfer-matrix__prim__to__dijkstra --config configs/dijkstra_test_100n.yaml
cp runs/20260320-213710_647125b-dirty340505db_graph-transfer-matrix__prim__to__dijkstra/analysis/eval_only.json runs/20260320-213710_647125b-dirty340505db_graph-transfer-matrix__prim__to__dijkstra/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260321-015125_647125b-dirty340505db_graph-transfer-matrix__prim__to__dag_shortest_paths --config configs/dag_shortest_paths_test_100n.yaml
cp runs/20260321-015125_647125b-dirty340505db_graph-transfer-matrix__prim__to__dag_shortest_paths/analysis/eval_only.json runs/20260321-015125_647125b-dirty340505db_graph-transfer-matrix__prim__to__dag_shortest_paths/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260321-082025_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__bf --config configs/bf_test_100n.yaml
cp runs/20260321-082025_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__bf/analysis/eval_only.json runs/20260321-082025_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__bf/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260321-090335_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__bfs --config configs/bfs_test_100n.yaml
cp runs/20260321-090335_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__bfs/analysis/eval_only.json runs/20260321-090335_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__bfs/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260321-092542_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__prim --config configs/prim_test_100n.yaml
cp runs/20260321-092542_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__prim/analysis/eval_only.json runs/20260321-092542_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__prim/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260321-133403_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__dag_shortest_paths --config configs/dag_shortest_paths_test_100n.yaml
cp runs/20260321-133403_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__dag_shortest_paths/analysis/eval_only.json runs/20260321-133403_647125b-dirty340505db_graph-transfer-matrix__dijkstra__to__dag_shortest_paths/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260321-185056_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__bf --config configs/bf_test_100n.yaml
cp runs/20260321-185056_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__bf/analysis/eval_only.json runs/20260321-185056_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__bf/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260321-213241_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__bfs --config configs/bfs_test_100n.yaml
cp runs/20260321-213241_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__bfs/analysis/eval_only.json runs/20260321-213241_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__bfs/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260321-221735_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__prim --config configs/prim_test_100n.yaml
cp runs/20260321-221735_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__prim/analysis/eval_only.json runs/20260321-221735_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__prim/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260322-001124_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__dijkstra --config configs/dijkstra_test_100n.yaml
cp runs/20260322-001124_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__dijkstra/analysis/eval_only.json runs/20260322-001124_647125b-dirty340505db_graph-transfer-matrix__dag_shortest_paths__to__dijkstra/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260317-220537_647125b-dirty7a8bf0e9_clrs-graphs --config configs/clrs_graphs_test_100n.yaml
cp runs/20260317-220537_647125b-dirty7a8bf0e9_clrs-graphs/analysis/eval_only.json runs/20260317-220537_647125b-dirty7a8bf0e9_clrs-graphs/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260405-180449_2a33074-dirtye3b0c442_clrs-graphs-frozen-random-processor --config configs/clrs_graphs_frozen_random_processor_test_100n.yaml
cp runs/20260405-180449_2a33074-dirtye3b0c442_clrs-graphs-frozen-random-processor/analysis/eval_only.json runs/20260405-180449_2a33074-dirtye3b0c442_clrs-graphs-frozen-random-processor/analysis/eval_only_100n.json

python -m src.train --eval-only --run-dir runs/20260406-104020_2a33074-dirtyf9f87a51_clrs-graphs-identity-processor --config configs/clrs_graphs_identity_processor_test_100n.yaml
cp runs/20260406-104020_2a33074-dirtyf9f87a51_clrs-graphs-identity-processor/analysis/eval_only.json runs/20260406-104020_2a33074-dirtyf9f87a51_clrs-graphs-identity-processor/analysis/eval_only_100n.json

python -m src.analysis.transfer_matrix collect-eval-only --matrix-dir experiments/graph_trasnfer_matrix --split test --eval-file-name eval_only_100n.json --output-prefix test100n
python -m src.analysis.compare_eval_runs --split test --eval-file-name eval_only_100n.json --output-dir analysis/model_comparison_100n --labels normal frozen_random identity --run-dirs runs/20260317-220537_647125b-dirty7a8bf0e9_clrs-graphs runs/20260405-180449_2a33074-dirtye3b0c442_clrs-graphs-frozen-random-processor runs/20260406-104020_2a33074-dirtyf9f87a51_clrs-graphs-identity-processor
