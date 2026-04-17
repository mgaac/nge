from types import SimpleNamespace
from pathlib import Path

import numpy as np

from src.analysis.execution_segment_pca import (
    build_segment_sets,
    resolve_dataset_paths,
    select_graphs,
)


def test_resolve_dataset_paths_prefers_unique_task_paths():
    config = SimpleNamespace(
        data=SimpleNamespace(
            train_path="data/train_dataset.npz",
            val_path="data/val_dataset.npz",
            test_path="data/test_dataset.npz",
            task_paths={
                "bf": {
                    "train_path": "data/train_dataset.npz",
                    "val_path": "data/val_dataset.npz",
                    "test_path": "data/test_dataset.npz",
                },
                "bfs": {
                    "train_path": "data/train_dataset.npz",
                    "val_path": "data/val_dataset.npz",
                    "test_path": "data/test_dataset.npz",
                },
                "prim": {
                    "train_path": "data/train_dataset.npz",
                    "val_path": "data/val_dataset.npz",
                    "test_path": "data/test_dataset.npz",
                },
                "dijkstra": {
                    "train_path": "data/train_dijkstra_dataset.npz",
                    "val_path": "data/val_dijkstra_dataset.npz",
                    "test_path": "data/test_dijkstra_dataset.npz",
                },
                "dag_shortest_paths": {
                    "train_path": "data/train_dag_shortest_paths_dataset.npz",
                    "val_path": "data/val_dag_shortest_paths_dataset.npz",
                    "test_path": "data/test_dag_shortest_paths_dataset.npz",
                },
            },
        )
    )

    paths = resolve_dataset_paths(
        config=config,
        dataset_override=None,
        split="val",
        algorithm_order=("bf", "bfs", "prim", "dijkstra", "dag_shortest_paths"),
    )

    assert paths == [
        Path("data/val_dataset.npz"),
        Path("data/val_dijkstra_dataset.npz"),
        Path("data/val_dag_shortest_paths_dataset.npz"),
    ]


def test_select_graphs_flattens_sources_and_respects_limit():
    dataset_sources = [
        {"path": Path("a.npz"), "graphs": ["a0", "a1"]},
        {"path": Path("b.npz"), "graphs": ["b0", "b1"]},
    ]

    selected = select_graphs(dataset_sources=dataset_sources, graph_index=None, max_graphs=3)

    assert selected == [
        {"dataset_path": Path("a.npz"), "dataset_graph_index": 0, "graph": "a0"},
        {"dataset_path": Path("a.npz"), "dataset_graph_index": 1, "graph": "a1"},
        {"dataset_path": Path("b.npz"), "dataset_graph_index": 0, "graph": "b0"},
    ]


def test_build_segment_sets_groups_embeddings_by_active_count():
    rollouts = [
        {
            "embeddings": np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
            "step_indices": np.array([0, 1, 2], dtype=np.int32),
            "active_counts": np.array([5, 4, 3], dtype=np.int32),
            "active_signatures": np.array(
                ["bf+bfs+prim+dijkstra+dag_shortest_paths", "bf+bfs+prim+dijkstra", "bf+bfs+prim"],
                dtype=object,
            ),
            "source_labels": np.array(["a.npz", "a.npz", "a.npz"], dtype=object),
        },
        {
            "embeddings": np.array([[10.0, 1.0], [20.0, 2.0]], dtype=np.float32),
            "step_indices": np.array([0, 1], dtype=np.int32),
            "active_counts": np.array([5, 3], dtype=np.int32),
            "active_signatures": np.array(
                ["bf+bfs+prim+dijkstra+dag_shortest_paths", "prim+dijkstra+dag_shortest_paths"],
                dtype=object,
            ),
            "source_labels": np.array(["b.npz", "b.npz"], dtype=object),
        },
    ]

    segment_sets = build_segment_sets(rollouts)

    assert sorted(segment_sets) == [3, 4, 5]
    assert segment_sets[5]["vectors"].tolist() == [[1.0, 0.0], [10.0, 1.0]]
    assert segment_sets[4]["step_labels"].tolist() == [1]
    assert segment_sets[3]["source_labels"].tolist() == ["a.npz", "b.npz"]
