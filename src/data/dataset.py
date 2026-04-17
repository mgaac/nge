"""Dataset generation and I/O utilities for NGE experiments."""

from __future__ import annotations

import argparse
from math import inf
from pathlib import Path
from typing import Any, List, Tuple

import mlx.core as mx
import networkx as nx
import numpy as np

from src.utils.task_specs import (
    algorithm_family,
    algorithm_target_keys,
    normalize_algorithm_order,
)

MULTITASK_TASK = "multitask"
CLRS_ALGO_TASK = "clrs_algo"
TASK_CHOICES = (MULTITASK_TASK, CLRS_ALGO_TASK, "dijkstra", "dag_shortest_paths")
LEGACY_MULTITASK_KEYS = (
    "num_nodes",
    "edge_matrix",
    "source_node",
    "bf_distance_targets",
    "bf_predecessor_targets",
    "bfs_state_targets",
    "prim_state_targets",
    "prim_key_targets",
    "prim_predecessor_targets",
)


def _empty_edge_matrix(rows: int = 2) -> mx.array:
    """Return an empty edge matrix with the requested row count."""
    return mx.zeros([rows, 0], dtype=mx.float32)


def erdos_renyi_edge_matrix(num_nodes: int = 20, p: float = 0.2, directed: bool = False):
    """Generate an Erdos-Renyi graph edge matrix of shape (2, num_edges)."""
    graph = nx.erdos_renyi_graph(num_nodes, p, directed=directed)
    edges = list(graph.edges())
    if not edges:
        return _empty_edge_matrix(rows=2).astype(mx.int32)
    return mx.array(edges, dtype=mx.int32).T


def barabasi_albert_edge_matrix(num_nodes: int = 20, m: int = 2):
    """Generate a Barabasi-Albert graph edge matrix of shape (2, num_edges)."""
    graph = nx.barabasi_albert_graph(num_nodes, m)
    edges = list(graph.edges())
    if not edges:
        return _empty_edge_matrix(rows=2).astype(mx.int32)
    return mx.array(edges, dtype=mx.int32).T


def dag_edge_matrix(num_nodes: int = 20, p: float = 0.2):
    """Generate a random DAG edge matrix using a sampled topological order."""
    if num_nodes <= 0:
        return _empty_edge_matrix(rows=2).astype(mx.int32)

    order = np.random.permutation(num_nodes)
    edges = []
    for src_rank in range(num_nodes):
        for dst_rank in range(src_rank + 1, num_nodes):
            if np.random.rand() < p:
                edges.append((int(order[src_rank]), int(order[dst_rank])))

    if not edges:
        return _empty_edge_matrix(rows=2).astype(mx.int32)
    return mx.array(edges, dtype=mx.int32).T


def add_self_loops(edge_matrix: mx.array, num_nodes: int):
    """Add self-loops to every node in the graph."""
    if num_nodes == 0:
        return edge_matrix

    self_loop_edges = mx.array([[i, i] for i in range(num_nodes)], dtype=mx.int32).T
    if edge_matrix.size == 0:
        return self_loop_edges
    return mx.concatenate([edge_matrix.astype(mx.int32), self_loop_edges], axis=1)


def make_bidirectional_edges(edge_matrix: mx.array):
    """Add reverse edges for all non-self-loop edges."""
    if edge_matrix.size == 0:
        return edge_matrix

    num_edges = edge_matrix.shape[1]
    bidirectional_edges = []
    for edge_index in range(num_edges):
        u = int(edge_matrix[0, edge_index])
        v = int(edge_matrix[1, edge_index])
        if edge_matrix.shape[0] > 2:
            w = float(edge_matrix[2, edge_index])
            bidirectional_edges.append([u, v, w])
            if u != v:
                bidirectional_edges.append([v, u, w])
        else:
            bidirectional_edges.append([u, v])
            if u != v:
                bidirectional_edges.append([v, u])

    return mx.array(bidirectional_edges).T


def append_uniform_edge_weights(
    edge_matrix: mx.array,
    num_nodes: int,
    low: float = 0.2,
    high: float = 1.0,
):
    """Add self-loops, enforce bidirectionality, and append random edge weights."""
    if edge_matrix.size == 0 and num_nodes == 0:
        return _empty_edge_matrix(rows=3)

    edge_matrix = add_self_loops(edge_matrix, num_nodes)
    edge_matrix = make_bidirectional_edges(edge_matrix)

    num_edges = edge_matrix.shape[1]
    weights = mx.array(np.random.uniform(low, high, size=num_edges), dtype=mx.float32)
    return mx.concatenate([edge_matrix.astype(mx.float32), weights.reshape(1, -1)], axis=0)


def append_directed_edge_weights(
    edge_matrix: mx.array,
    num_nodes: int,
    low: float = 0.2,
    high: float = 1.0,
    self_loop_weight: float = 0.0,
):
    """Add self-loops and append random weights while preserving edge direction."""
    if edge_matrix.size == 0 and num_nodes == 0:
        return _empty_edge_matrix(rows=3)

    edge_matrix = add_self_loops(edge_matrix, num_nodes)
    num_edges = edge_matrix.shape[1]
    weights = np.random.uniform(low, high, size=num_edges).astype(np.float32)
    edge_array = np.array(edge_matrix, dtype=np.int32)
    loop_mask = edge_array[0] == edge_array[1]
    weights[loop_mask] = float(self_loop_weight)
    return mx.array(
        np.concatenate([edge_array.astype(np.float32), weights.reshape(1, -1)], axis=0),
        dtype=mx.float32,
    )


def bellman_ford_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> tuple[List[List[float]], List[List[int | None]]]:
    """Log Bellman-Ford distances and predecessors after each relaxation round."""
    if edges.size == 0:
        return [], []

    num_edges = edges.shape[1]
    in_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for edge_index in range(num_edges):
        u = int(edges[0, edge_index])
        v = int(edges[1, edge_index])
        w = float(edges[2, edge_index])
        in_neighbors[v].append((u, w))

    distance: List[float] = [inf] * num_nodes
    predecessor: List[int | None] = [None] * num_nodes
    distance[source] = 0.0
    predecessor[source] = source

    distance_log = [distance.copy()]
    predecessor_log = [predecessor.copy()]

    for _ in range(num_nodes - 1):
        new_distance = distance.copy()
        new_predecessor = predecessor.copy()
        updated = False

        for node in range(num_nodes):
            if node == source:
                continue

            best_cost = inf
            best_pred = None
            for neighbor, weight in in_neighbors[node]:
                candidate = distance[neighbor] + weight
                if candidate < best_cost or (
                    candidate == best_cost and best_pred is not None and neighbor < best_pred
                ):
                    best_cost = candidate
                    best_pred = neighbor

            if best_cost < new_distance[node]:
                new_distance[node] = best_cost
                new_predecessor[node] = best_pred
                updated = True

        distance = new_distance
        predecessor = new_predecessor
        distance_log.append(distance.copy())
        predecessor_log.append(predecessor.copy())

        if not updated:
            break

    return distance_log, predecessor_log


def dijkstra_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> tuple[List[List[float]], List[List[int | None]]]:
    """Log Dijkstra distances and predecessors after each extract-min step."""
    if edges.size == 0:
        return [], []

    neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for edge_index in range(edges.shape[1]):
        u = int(edges[0, edge_index])
        v = int(edges[1, edge_index])
        w = float(edges[2, edge_index])
        neighbors[u].append((v, w))

    distance: List[float] = [inf] * num_nodes
    predecessor: List[int | None] = [None] * num_nodes
    visited = [False] * num_nodes
    distance[source] = 0.0
    predecessor[source] = source

    distance_log = [distance.copy()]
    predecessor_log = [predecessor.copy()]

    for _ in range(num_nodes):
        next_node = None
        next_distance = inf
        for node in range(num_nodes):
            if visited[node]:
                continue
            if distance[node] < next_distance or (
                distance[node] == next_distance and next_node is not None and node < next_node
            ):
                next_node = node
                next_distance = distance[node]
            elif next_node is None:
                next_node = node
                next_distance = distance[node]

        if next_node is None or next_distance == inf:
            break

        visited[next_node] = True
        updated = False
        for neighbor, weight in neighbors[next_node]:
            if visited[neighbor]:
                continue
            candidate = distance[next_node] + weight
            current_pred = predecessor[neighbor]
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                predecessor[neighbor] = next_node
                updated = True
            elif candidate == distance[neighbor] and (
                current_pred is None or next_node < current_pred
            ):
                predecessor[neighbor] = next_node
                updated = True

        distance_log.append(distance.copy())
        predecessor_log.append(predecessor.copy())

        if not updated and all(visited[node] or distance[node] == inf for node in range(num_nodes)):
            break

    return distance_log, predecessor_log


def dag_shortest_paths_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> tuple[List[List[float]], List[List[int | None]]]:
    """Log DAG shortest paths after each topological relaxation step."""
    if edges.size == 0:
        return [], []

    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for edge_index in range(edges.shape[1]):
        u = int(edges[0, edge_index])
        v = int(edges[1, edge_index])
        if u == v:
            continue
        w = float(edges[2, edge_index])
        graph.add_edge(u, v, weight=w)
        neighbors[u].append((v, w))

    topological_order = list(nx.topological_sort(graph))
    distance: List[float] = [inf] * num_nodes
    predecessor: List[int | None] = [None] * num_nodes
    distance[source] = 0.0
    predecessor[source] = source

    distance_log = [distance.copy()]
    predecessor_log = [predecessor.copy()]

    for node in topological_order:
        if distance[node] != inf:
            for neighbor, weight in neighbors[node]:
                candidate = distance[node] + weight
                current_pred = predecessor[neighbor]
                if candidate < distance[neighbor]:
                    distance[neighbor] = candidate
                    predecessor[neighbor] = node
                elif candidate == distance[neighbor] and (
                    current_pred is None or node < current_pred
                ):
                    predecessor[neighbor] = node

        distance_log.append(distance.copy())
        predecessor_log.append(predecessor.copy())

    return distance_log, predecessor_log


def clean_distance_log(log: List[List[float]]) -> mx.array:
    """Normalize shortest-path distances to [0, 1] and map infinity to 1."""
    if not log or not log[-1]:
        return mx.array(log, dtype=mx.float32)

    final = log[-1]
    finite = [value for value in final if value != float("inf")]
    scale = (max(finite) + 1.0) if finite else 1.0

    normalized = []
    for state in log:
        normalized.append(
            [((scale if value == float("inf") else value) / scale) for value in state]
        )
    return mx.array(normalized, dtype=mx.float32)


def clean_key_log(log: List[List[float]]) -> mx.array:
    """Keep Prim keys in their natural scale and map infinity to 1."""
    if not log or not log[-1]:
        return mx.array(log, dtype=mx.float32)

    cleaned = []
    for state in log:
        cleaned.append([1.0 if value == float("inf") else value for value in state])
    return mx.array(cleaned, dtype=mx.float32)


def clean_predecessor_log(log: List[List[int | None]]) -> mx.array:
    """Replace missing predecessors with -1."""
    cleaned = []
    for state in log:
        cleaned.append([(-1 if value is None else value) for value in state])
    return mx.array(cleaned, dtype=mx.int32)


def bfs_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> List[List[int]]:
    """Log BFS reachability after each discovered layer."""
    if edges.size == 0:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for edge_index in range(edges.shape[1]):
        u = int(edges[0, edge_index])
        v = int(edges[1, edge_index])
        graph.add_edge(u, v)

    reachable = [0] * num_nodes
    reachable[source] = 1
    reachability_log = [reachable.copy()]

    for layer in nx.bfs_layers(graph, [source]):
        updated = False
        for node in layer:
            if not reachable[node]:
                reachable[node] = 1
                updated = True
        if updated:
            reachability_log.append(reachable.copy())

    return reachability_log


def prim_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> tuple[List[List[int]], List[List[float]], List[List[int | None]]]:
    """Log Prim execution on the source-connected component of an undirected graph."""
    if edges.size == 0:
        return [], [], []

    neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for edge_index in range(edges.shape[1]):
        u = int(edges[0, edge_index])
        v = int(edges[1, edge_index])
        w = float(edges[2, edge_index])
        neighbors[u].append((v, w))

    in_mst: List[int] = [0] * num_nodes
    key: List[float] = [inf] * num_nodes
    predecessor: List[int | None] = [None] * num_nodes

    key[source] = 0.0
    predecessor[source] = source

    in_mst_log = [in_mst.copy()]
    key_log = [key.copy()]
    predecessor_log = [predecessor.copy()]

    for _ in range(num_nodes):
        next_node = None
        next_key = inf
        for node in range(num_nodes):
            if in_mst[node]:
                continue
            node_key = key[node]
            if node_key < next_key or (
                node_key == next_key and next_node is not None and node < next_node
            ):
                next_key = node_key
                next_node = node
            elif next_node is None:
                next_key = node_key
                next_node = node

        if next_node is None or next_key == inf:
            break

        in_mst[next_node] = 1
        for neighbor, weight in neighbors[next_node]:
            if in_mst[neighbor]:
                continue
            current_pred = predecessor[neighbor]
            if weight < key[neighbor]:
                key[neighbor] = weight
                predecessor[neighbor] = next_node
            elif weight == key[neighbor] and (
                current_pred is None or next_node < current_pred
            ):
                predecessor[neighbor] = next_node

        in_mst_log.append(in_mst.copy())
        key_log.append(key.copy())
        predecessor_log.append(predecessor.copy())

    return in_mst_log, key_log, predecessor_log


def _sample_multitask_edge_matrix(num_nodes: int, p: float, m: int) -> mx.array:
    """Sample the legacy multitask graph family used for bf/bfs/prim."""
    if np.random.rand() < 0.5:
        return append_uniform_edge_weights(barabasi_albert_edge_matrix(num_nodes, m), num_nodes)
    return append_uniform_edge_weights(erdos_renyi_edge_matrix(num_nodes, p), num_nodes)


def _sample_directed_weighted_edge_matrix(num_nodes: int, p: float) -> mx.array:
    """Sample a directed weighted graph for Dijkstra."""
    return append_directed_edge_weights(
        erdos_renyi_edge_matrix(num_nodes, p, directed=True),
        num_nodes,
    )


def _sample_weighted_dag_edge_matrix(num_nodes: int, p: float) -> mx.array:
    """Sample a weighted DAG for DAG shortest paths."""
    return append_directed_edge_weights(dag_edge_matrix(num_nodes, p), num_nodes)


def _sample_clrs_algo_edge_matrix(num_nodes: int, p: float) -> mx.array:
    """Sample a shared DAG family used to execute all algorithms on the same graph."""
    return _sample_weighted_dag_edge_matrix(num_nodes, p)


def generated_dataset(
    num_graphs=100,
    num_nodes=20,
    p=0.2,
    m=2,
    task: str = MULTITASK_TASK,
):
    """Generate a dataset for the requested graph-algorithm task family."""
    if task not in TASK_CHOICES:
        raise ValueError(f"Unsupported task: {task}")

    dataset = []
    for _ in range(num_graphs):
        source_node = int(np.random.randint(0, num_nodes))

        if task == MULTITASK_TASK:
            edge_matrix = _sample_multitask_edge_matrix(num_nodes, p, m)

            bf_distance, bf_predecessor = bellman_ford_log(edge_matrix, source_node, num_nodes)
            bfs_reachability = bfs_log(edge_matrix, source_node, num_nodes)
            prim_state, prim_key, prim_predecessor = prim_log(edge_matrix, source_node, num_nodes)

            graph_dict = {
                "num_nodes": num_nodes,
                "edge_matrix": edge_matrix,
                "source_node": source_node,
                "bf_distance_targets": clean_distance_log(bf_distance),
                "bf_predecessor_targets": clean_predecessor_log(bf_predecessor),
                "bfs_state_targets": mx.array(bfs_reachability, dtype=mx.float32),
                "prim_state_targets": mx.array(prim_state, dtype=mx.float32),
                "prim_key_targets": clean_key_log(prim_key),
                "prim_predecessor_targets": clean_predecessor_log(prim_predecessor),
            }

        elif task == CLRS_ALGO_TASK:
            edge_matrix = _sample_clrs_algo_edge_matrix(num_nodes, p)

            bf_distance, bf_predecessor = bellman_ford_log(edge_matrix, source_node, num_nodes)
            bfs_reachability = bfs_log(edge_matrix, source_node, num_nodes)
            prim_state, prim_key, prim_predecessor = prim_log(edge_matrix, source_node, num_nodes)
            dijkstra_distance, dijkstra_predecessor = dijkstra_log(
                edge_matrix, source_node, num_nodes
            )
            dag_distance, dag_predecessor = dag_shortest_paths_log(
                edge_matrix, source_node, num_nodes
            )

            graph_dict = {
                "num_nodes": num_nodes,
                "edge_matrix": edge_matrix,
                "source_node": source_node,
                "bf_distance_targets": clean_distance_log(bf_distance),
                "bf_predecessor_targets": clean_predecessor_log(bf_predecessor),
                "bfs_state_targets": mx.array(bfs_reachability, dtype=mx.float32),
                "prim_state_targets": mx.array(prim_state, dtype=mx.float32),
                "prim_key_targets": clean_key_log(prim_key),
                "prim_predecessor_targets": clean_predecessor_log(prim_predecessor),
                "dijkstra_distance_targets": clean_distance_log(dijkstra_distance),
                "dijkstra_predecessor_targets": clean_predecessor_log(dijkstra_predecessor),
                "dag_shortest_paths_distance_targets": clean_distance_log(dag_distance),
                "dag_shortest_paths_predecessor_targets": clean_predecessor_log(
                    dag_predecessor
                ),
            }

        elif task == "dijkstra":
            edge_matrix = _sample_directed_weighted_edge_matrix(num_nodes, p)
            distance_log, predecessor_log = dijkstra_log(edge_matrix, source_node, num_nodes)
            graph_dict = {
                "num_nodes": num_nodes,
                "edge_matrix": edge_matrix,
                "source_node": source_node,
                "dijkstra_distance_targets": clean_distance_log(distance_log),
                "dijkstra_predecessor_targets": clean_predecessor_log(predecessor_log),
            }

        elif task == "dag_shortest_paths":
            edge_matrix = _sample_weighted_dag_edge_matrix(num_nodes, p)
            distance_log, predecessor_log = dag_shortest_paths_log(edge_matrix, source_node, num_nodes)
            graph_dict = {
                "num_nodes": num_nodes,
                "edge_matrix": edge_matrix,
                "source_node": source_node,
                "dag_shortest_paths_distance_targets": clean_distance_log(distance_log),
                "dag_shortest_paths_predecessor_targets": clean_predecessor_log(predecessor_log),
            }

        else:
            raise ValueError(f"Unsupported task: {task}")

        dataset.append(graph_dict)

    return dataset


def _inactive_targets(num_nodes: int, algorithm: str) -> dict[str, mx.array]:
    """Create a one-step inactive target payload for an algorithm."""
    family = algorithm_family(algorithm)
    keys = algorithm_target_keys(algorithm)
    if family == "state_mask":
        return {keys[0]: mx.zeros([1, num_nodes], dtype=mx.float32)}
    if family == "shortest_path":
        return {
            keys[0]: mx.zeros([1, num_nodes], dtype=mx.float32),
            keys[1]: mx.full([1, num_nodes], -1, dtype=mx.int32),
        }
    if family == "mst":
        return {
            keys[0]: mx.zeros([1, num_nodes], dtype=mx.float32),
            keys[1]: mx.zeros([1, num_nodes], dtype=mx.float32),
            keys[2]: mx.full([1, num_nodes], -1, dtype=mx.int32),
        }
    raise ValueError(f"Unsupported algorithm family: {family}")


def materialize_graph_sample(
    graph_dict: dict[str, Any],
    algorithm_order,
    active_task: str | None = None,
) -> dict[str, Any]:
    """Project a raw graph record into the full target schema expected by the model."""
    algorithms = normalize_algorithm_order(algorithm_order)
    num_nodes = int(graph_dict["num_nodes"])
    materialized = {
        "num_nodes": num_nodes,
        "edge_matrix": graph_dict["edge_matrix"],
        "source_node": int(graph_dict["source_node"]),
    }
    if active_task is not None:
        materialized["active_task"] = active_task

    for algorithm in algorithms:
        keys = algorithm_target_keys(algorithm)
        if active_task is None:
            has_targets = all(key in graph_dict for key in keys)
        else:
            has_targets = active_task == algorithm and all(key in graph_dict for key in keys)

        if has_targets:
            for key in keys:
                materialized[key] = graph_dict[key]
        else:
            materialized.update(_inactive_targets(num_nodes, algorithm))

    return materialized


def save_dataset(dataset, filename, task: str | None = None):
    """Save a dataset to disk using a schema-driven compressed NPZ format."""
    save_dict: dict[str, Any] = {"num_graphs": len(dataset)}
    if task is not None:
        save_dict["task"] = np.array(task)

    if not dataset:
        save_dict["schema_keys"] = np.array([], dtype=object)
        np.savez_compressed(filename, **save_dict)
        return

    schema_keys = list(dataset[0].keys())
    reference_set = set(schema_keys)
    for graph in dataset:
        if set(graph.keys()) != reference_set:
            raise ValueError("All dataset graphs must share the same schema.")

    save_dict["schema_keys"] = np.array(schema_keys, dtype=object)
    for index, graph_dict in enumerate(dataset):
        for key in schema_keys:
            value = graph_dict[key]
            if not isinstance(value, (int, float, np.integer, np.floating)):
                value = np.array(value)
            save_dict[f"{key}_{index}"] = value

    np.savez_compressed(filename, **save_dict)


def _load_generic_dataset(loaded) -> list[dict[str, Any]]:
    """Load a schema-driven dataset saved by save_dataset."""
    num_graphs = int(loaded["num_graphs"])
    schema_keys = [str(key) for key in loaded["schema_keys"].tolist()]
    dataset = []

    for index in range(num_graphs):
        graph_dict = {}
        for key in schema_keys:
            value = loaded[f"{key}_{index}"]
            if key in {"num_nodes", "source_node"}:
                graph_dict[key] = int(np.asarray(value).item())
            else:
                graph_dict[key] = mx.array(value)
        dataset.append(graph_dict)

    return dataset


def _load_legacy_multitask_dataset(loaded) -> list[dict[str, Any]]:
    """Load the original bf/bfs/prim dataset format."""
    num_graphs = int(loaded["num_graphs"])
    dataset = []

    for index in range(num_graphs):
        graph_dict = {
            "num_nodes": int(loaded[f"num_nodes_{index}"]),
            "edge_matrix": mx.array(loaded[f"edge_matrix_{index}"]),
            "source_node": int(loaded[f"source_node_{index}"]),
            "bf_distance_targets": mx.array(loaded[f"bf_distance_targets_{index}"]),
            "bf_predecessor_targets": mx.array(loaded[f"bf_predecessor_targets_{index}"]),
            "bfs_state_targets": mx.array(loaded[f"bfs_state_targets_{index}"]),
            "prim_state_targets": mx.array(loaded[f"prim_state_targets_{index}"]),
            "prim_key_targets": mx.array(loaded[f"prim_key_targets_{index}"]),
            "prim_predecessor_targets": mx.array(loaded[f"prim_predecessor_targets_{index}"]),
        }
        dataset.append(graph_dict)

    return dataset


def load_dataset(filename):
    """Load a dataset saved with save_dataset."""
    loaded = np.load(filename, allow_pickle=True)
    if "schema_keys" in loaded:
        return _load_generic_dataset(loaded)
    return _load_legacy_multitask_dataset(loaded)


def _default_output_name(task: str, num_graphs: int, num_nodes: int) -> str:
    """Return the default filename for single-dataset generation."""
    if task == MULTITASK_TASK:
        return f"dataset_{num_graphs}g_{num_nodes}n.npz"
    return f"{task}_dataset_{num_graphs}g_{num_nodes}n.npz"


def _preset_split_path(output_dir: Path, task: str, split: str) -> Path:
    """Return the default preset path for a split/task combination."""
    if task == MULTITASK_TASK:
        return output_dir / f"{split}_dataset.npz"
    return output_dir / f"{split}_{task}_dataset.npz"


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic NGE datasets.")
    parser.add_argument(
        "--preset",
        action="store_true",
        help="Generate default train/val/test datasets (1500/100/100 with 20 nodes).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=MULTITASK_TASK,
        choices=TASK_CHOICES,
        help=(
            "Dataset task family. "
            "'multitask' reproduces the legacy bf/bfs/prim setup. "
            "'clrs_algo' runs bf/bfs/prim/dijkstra/dag_shortest_paths on the same graph."
        ),
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=None,
        help="Number of graphs for single-dataset generation.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=20,
        help="Number of nodes per graph for single-dataset generation.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.2,
        help="Erdos-Renyi edge probability.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=2,
        help="Barabasi-Albert attachment parameter for the legacy multitask setup.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .npz path for single-dataset generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help=(
            "Default output directory for generated datasets. "
            "In preset mode, train/val/test are always written here."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.p <= 0 or args.p >= 1:
        raise ValueError("--p must be in (0, 1).")
    if args.num_nodes <= 0:
        raise ValueError("--num-nodes must be positive.")
    if args.m <= 0:
        raise ValueError("--m must be positive.")

    if args.preset:
        print(f"Generating preset datasets for task '{args.task}'...")
        train_dataset = generated_dataset(1500, 20, args.p, args.m, task=args.task)
        val_dataset = generated_dataset(100, 20, args.p, args.m, task=args.task)
        test_dataset = generated_dataset(100, 20, args.p, args.m, task=args.task)

        train_path = _preset_split_path(output_dir, args.task, "train")
        val_path = _preset_split_path(output_dir, args.task, "val")
        test_path = _preset_split_path(output_dir, args.task, "test")

        save_dataset(train_dataset, train_path, task=args.task)
        save_dataset(val_dataset, val_path, task=args.task)
        save_dataset(test_dataset, test_path, task=args.task)
        print(f"Preset datasets saved: {train_path}, {val_path}, {test_path}")
    else:
        if args.num_graphs is None:
            raise ValueError(
                "Use --preset for default splits, or provide --num-graphs for a single dataset."
            )
        if args.num_graphs <= 0:
            raise ValueError("--num-graphs must be positive.")

        dataset = generated_dataset(
            args.num_graphs,
            args.num_nodes,
            args.p,
            args.m,
            task=args.task,
        )
        output_path = (
            Path(args.output)
            if args.output
            else output_dir / _default_output_name(args.task, args.num_graphs, args.num_nodes)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(dataset, output_path, task=args.task)
        print(f"Saved {args.task} dataset to {output_path}")
