"""Dataset generation and I/O utilities for NGE experiments."""

import argparse
from math import inf
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import networkx as nx
import numpy as np


def erdos_renyi_edge_matrix(num_nodes: int = 20, p: float = 0.2):
    """Generate an Erdos-Renyi graph edge matrix of shape (2, num_edges)."""
    G = nx.erdos_renyi_graph(num_nodes, p)
    edges = list(G.edges())
    if not edges:
        return mx.array([])
    edge_matrix = mx.array(edges).T
    return edge_matrix


def barabasi_albert_edge_matrix(num_nodes: int = 20, m: int = 2):
    """Generate a Barabasi-Albert graph edge matrix of shape (2, num_edges)."""
    G = nx.barabasi_albert_graph(num_nodes, m)
    edges = list(G.edges())
    if not edges:
        return mx.array([])
    edge_matrix = mx.array(edges).T
    return edge_matrix


def add_self_loops(edge_matrix: mx.array, num_nodes: int):
    """Add self-loops to every node in the graph."""
    if num_nodes == 0:
        return edge_matrix

    self_loop_edges = [[i, i] for i in range(num_nodes)]
    if not self_loop_edges:
        return edge_matrix

    self_loops_array = mx.array(self_loop_edges).T

    if edge_matrix.size == 0:
        return self_loops_array
    return mx.concatenate([edge_matrix, self_loops_array], axis=1)


def make_bidirectional_edges(edge_matrix: mx.array):
    """Add reverse edges for all non-self-loop edges."""
    if edge_matrix.size == 0:
        return edge_matrix

    num_edges = edge_matrix.shape[1]
    bidirectional_edges = []

    for i in range(num_edges):
        u, v = int(edge_matrix[0, i]), int(edge_matrix[1, i])
        if edge_matrix.shape[0] > 2:
            w = edge_matrix[2, i]
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
        return edge_matrix

    edge_matrix = add_self_loops(edge_matrix, num_nodes)
    edge_matrix = make_bidirectional_edges(edge_matrix)

    num_edges = edge_matrix.shape[1]
    weights = mx.array(np.random.uniform(low, high, size=num_edges))
    weighted_matrix = mx.concatenate([edge_matrix, weights.reshape(1, -1)], axis=0)
    return weighted_matrix

def bellman_ford_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> tuple[List[List[float]], List[List[int | None]]]:
    """
    Paper-consistent BF logging:
      - source predecessor fixed to itself
      - update predecessor only on genuine relaxation
      - ties allowed (<=) but written only when distance improves
    """
    if edges.size == 0:
        return [], []

    num_edges = edges.shape[1]

    # Build in-neighborhoods with weights
    in_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for k in range(num_edges):
        u = int(edges[0, k])
        v = int(edges[1, k])
        w = float(edges[2, k])
        in_neighbors[v].append((u, w))

    # Init
    distance:    List[float]      = [inf] * num_nodes
    predecessor: List[int | None] = [None] * num_nodes
    distance[source]    = 0.0
    predecessor[source] = source  # p_s = s for all iterations

    distance_log    = [distance.copy()]
    predecessor_log = [predecessor.copy()]

    # Up to |V|-1 rounds
    for _ in range(num_nodes - 1):
        new_distance    = distance.copy()
        new_predecessor = predecessor.copy()
        updated = False

        for i in range(num_nodes):
            if i == source:
                continue

            best_cost = inf
            best_pred = None

            # Best incoming relaxation candidate
            for j, w in in_neighbors[i]:
                cand = distance[j] + w
                if cand <= best_cost:
                    best_cost = cand
                    best_pred = j

            # Write only on genuine relaxation
            if best_cost < new_distance[i]:
                new_distance[i]    = best_cost
                new_predecessor[i] = best_pred
                updated = True

        distance      = new_distance
        predecessor   = new_predecessor
        distance_log.append(distance.copy())
        predecessor_log.append(predecessor.copy())

        if not updated:
            break

    return distance_log, predecessor_log


def clean_bf_logs(log: List[List[float]] | List[List[int | None]]) -> mx.array:
    if not log or not log[-1]:
        return mx.array(log)

    is_distance = isinstance(log[0][0], float)

    if is_distance:
        final = log[-1]
        finite = [x for x in final if x != float('inf')]
        S = (max(finite) + 1.0) if finite else 1.0

        norm_log = []
        for state in log:
            state_norm = [((S if x == float('inf') else x) / S) for x in state]
            norm_log.append(state_norm)
        return mx.array(norm_log, dtype=mx.float32)

    else:
        # Predecessors: replace None with -1 (sentinel for "no predecessor")
        cleaned = []
        for state in log:
            cleaned_state = [(-1 if x is None else x) for i, x in enumerate(state)]
            cleaned.append(cleaned_state)
        return mx.array(cleaned, dtype=mx.int32)


def bfs_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> List[List[int]]:
    if edges.size == 0:
        return []
    # Build NetworkX graph for BFS (ignore weights for unweighted BFS)
    G = nx.Graph()

    # Add all nodes first to ensure they exist
    G.add_nodes_from(range(num_nodes))
    
    # Add edges (convert to undirected for standard BFS)
    for k in range(edges.shape[1]):
        u = int(edges[0, k])
        v = int(edges[1, k])
        G.add_edge(u, v)

    # Initialize reachability array: 1 if reachable, 0 if not
    reachable = [0] * num_nodes
    reachable[source] = 1

    reachability_log = [reachable.copy()]

    # Use NetworkX bfs_layers to get nodes layer by layer
    for layer in nx.bfs_layers(G, [source]):
        # Mark all nodes in this layer as reachable (if not already)
        updated = False
        for node in layer:
            if not reachable[node]:
                reachable[node] = 1
                updated = True
        if updated:
            reachability_log.append(reachable.copy())

    return reachability_log

def generated_dataset(num_graphs=100, num_nodes=20, p=0.2, m=2):

    dataset = []

    for _ in range(num_graphs):
        # Randomly choose graph type
        if mx.random.uniform() < 0.5:
            edge_matrix = append_uniform_edge_weights(barabasi_albert_edge_matrix(num_nodes, m), num_nodes)
        else:
            edge_matrix = append_uniform_edge_weights(erdos_renyi_edge_matrix(num_nodes, p), num_nodes)

        source_node = mx.random.randint(0, num_nodes).item()

        # Bellman-Ford logs
        bf_distance, bf_predecessor = bellman_ford_log(edge_matrix, source_node, num_nodes)
        bf_distance = clean_bf_logs(bf_distance)
        bf_predecessor = clean_bf_logs(bf_predecessor)

        # BFS logs
        bfs_reachability = bfs_log(edge_matrix, source_node, num_nodes)
        bfs_reachability = mx.array(bfs_reachability)

        graph_dict = {
            "num_nodes": num_nodes,
            "edge_matrix": edge_matrix,
            "source_node": source_node,
            "bf_distance_targets": bf_distance,
            "bf_predecessor_targets": bf_predecessor,
            "bfs_state_targets": bfs_reachability,
        }
        dataset.append(graph_dict)

    return dataset


def save_dataset(dataset, filename):
    """
    Save a dataset (as returned by generated_dataset) to disk using numpy's savez.
    This will store all arrays in a compressed .npz file.
    """
    # Flatten the dataset list for saving
    save_dict = {}
    save_dict["num_graphs"] = len(dataset)
    
    for i, graph_dict in enumerate(dataset):
        # Save each graph's data
        save_dict[f"num_nodes_{i}"] = graph_dict["num_nodes"]
        save_dict[f"edge_matrix_{i}"] = np.array(graph_dict["edge_matrix"])
        save_dict[f"source_node_{i}"] = graph_dict["source_node"]
        save_dict[f"bf_distance_targets_{i}"] = np.array(graph_dict["bf_distance_targets"])
        save_dict[f"bf_predecessor_targets_{i}"] = np.array(graph_dict["bf_predecessor_targets"])
        save_dict[f"bfs_state_targets_{i}"] = np.array(graph_dict["bfs_state_targets"])
    
    np.savez_compressed(filename, **save_dict)

def load_dataset(filename):
    """
    Load a dataset saved with save_dataset.
    Returns a list in the same format as generated_dataset.
    """
    loaded = np.load(filename, allow_pickle=True)
    num_graphs = int(loaded["num_graphs"])
    
    dataset = []
    for i in range(num_graphs):
        graph_dict = {
            "num_nodes": int(loaded[f"num_nodes_{i}"]),
            "edge_matrix": mx.array(loaded[f"edge_matrix_{i}"]),
            "source_node": int(loaded[f"source_node_{i}"]),
            "bf_distance_targets": mx.array(loaded[f"bf_distance_targets_{i}"]),
            "bf_predecessor_targets": mx.array(loaded[f"bf_predecessor_targets_{i}"]),
            "bfs_state_targets": mx.array(loaded[f"bfs_state_targets_{i}"]),
        }
        dataset.append(graph_dict)
    
    return dataset


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic NGE datasets.")
    parser.add_argument(
        "--preset",
        action="store_true",
        help="Generate default train/val/test datasets (1500/100/100 with 20 nodes).",
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
        help="Barabasi-Albert attachment parameter.",
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

    if args.preset:
        print("Generating preset datasets...")
        train_dataset = generated_dataset(1500, 20, 0.2, 2)
        val_dataset = generated_dataset(100, 20, 0.2, 2)
        test_dataset = generated_dataset(100, 20, 0.2, 2)

        train_path = output_dir / "train_dataset.npz"
        val_path = output_dir / "val_dataset.npz"
        test_path = output_dir / "test_dataset.npz"

        save_dataset(train_dataset, train_path)
        save_dataset(val_dataset, val_path)
        save_dataset(test_dataset, test_path)
        print(
            "Preset datasets saved: "
            f"{train_path}, {val_path}, {test_path}"
        )
    else:
        if args.num_graphs is None:
            raise ValueError(
                "Use --preset for default splits, or provide --num-graphs for a single dataset."
            )
        if args.num_graphs <= 0:
            raise ValueError("--num-graphs must be positive.")
        if args.num_nodes <= 0:
            raise ValueError("--num-nodes must be positive.")
        if args.p <= 0 or args.p >= 1:
            raise ValueError("--p must be in (0, 1).")
        if args.m <= 0:
            raise ValueError("--m must be positive.")

        dataset = generated_dataset(args.num_graphs, args.num_nodes, args.p, args.m)
        output_path = (
            Path(args.output)
            if args.output
            else output_dir / f"dataset_{args.num_graphs}g_{args.num_nodes}n.npz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(dataset, output_path)
        print(f"Saved dataset to {output_path}")
