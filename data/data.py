import mlx.core as mx
import networkx as nx
import numpy as np

from math import inf
from typing import List, Tuple, Union


def erdos_renyi_edge_matrix(num_nodes=20, p=0.2):
    # Generate Erdos-Renyi graph
    G = nx.erdos_renyi_graph(num_nodes, p)
    # Get edge list as (source, target) tuples
    edges = list(G.edges())
    if not edges:
        # No edges, return empty MLX array
        return mx.array([])
    # Convert to 2 x num_edges matrix
    edge_matrix = mx.array(edges).T  # shape: (2, num_edges)
    return edge_matrix

def barabasi_albert_edge_matrix(num_nodes=20, m=2):
    G = nx.barabasi_albert_graph(num_nodes, m)
    edges = list(G.edges())
    if not edges:
        return mx.array([])
    edge_matrix = mx.array(edges).T  # shape: (2, num_edges)
    return edge_matrix

def add_self_loops(edge_matrix, num_nodes):
    """
    Add self loops to each node in the graph.
    For each node i, adds edge (i,i).
    """
    if num_nodes == 0:
        return edge_matrix
    
    # Create self loop edges for all nodes (without weights)
    self_loop_edges = []
    for i in range(num_nodes):
        self_loop_edges.append([i, i])
    
    # Convert self loops to array
    if not self_loop_edges:
        return edge_matrix
    
    self_loops_array = mx.array(self_loop_edges).T
    
    # Concatenate with existing edges if any exist
    if edge_matrix.size == 0:
        return self_loops_array
    else:
        return mx.concatenate([edge_matrix, self_loops_array], axis=1)

def make_bidirectional_edges(edge_matrix):
    """
    Convert directed edges to bidirectional by adding reverse edges.
    For each edge (u,v,w), also adds (v,u,w).
    """
    if edge_matrix.size == 0:
        return edge_matrix
    
    num_edges = edge_matrix.shape[1]
    bidirectional_edges = []
    
    for i in range(num_edges):
        u, v = edge_matrix[0, i], edge_matrix[1, i]
        if edge_matrix.shape[0] > 2:  # Has weights
            w = edge_matrix[2, i]
            bidirectional_edges.append([u, v, w])
            bidirectional_edges.append([v, u, w])  # Reverse edge with same weight
        else:
            bidirectional_edges.append([u, v])
            bidirectional_edges.append([v, u])  # Reverse edge
    
    return mx.array(bidirectional_edges).T

def append_uniform_edge_weights(edge_matrix, num_nodes, low=0.2, high=1.0):
    if edge_matrix.size == 0 and num_nodes == 0:
        return edge_matrix
    
    # First add self loops (without weights initially)
    edge_matrix = add_self_loops(edge_matrix, num_nodes)
    
    # Then make edges bidirectional
    edge_matrix = make_bidirectional_edges(edge_matrix)
    
    num_edges = edge_matrix.shape[1]
    weights = mx.array(np.random.uniform(low, high, size=num_edges))
    # Stack as a new row: shape (3, num_edges)
    weighted_matrix = mx.concatenate([edge_matrix, weights.reshape(1, -1)], axis=0)
    return weighted_matrix

def bellman_ford_log(
    edges: mx.array,
    source: int,
    num_nodes: int,
) -> tuple[List[List[float]], List[List[int | None]]]:

    if edges.size == 0:
        return [], []

    num_edges = edges.shape[1]

    in_neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    for k in range(num_edges):
        u = int(edges[0, k])
        v = int(edges[1, k])
        w = float(edges[2, k])
        in_neighbors[v].append((u, w))

    distance      : List[float]      = [inf] * num_nodes
    predecessor   : List[int | None] = [None] * num_nodes
    distance[source] = 0.0
    predecessor[source] = source      # p_s = s  for *all* iterations

    distance_log    = [distance.copy()]
    predecessor_log = [predecessor.copy()]

    for _ in range(num_nodes - 1):

        new_distance    = distance.copy()
        new_predecessor = predecessor.copy()
        updated = False

        for i in range(num_nodes):
            if i == source:
                continue

            best_cost = inf
            best_pred = None

            for j, w in in_neighbors[i]:
                cand = distance[j] + w
                if cand <= best_cost:          # tie OK; write only on relaxation below
                    best_cost = cand
                    best_pred = j

            # write predecessor only on genuine relaxation
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

    is_distance_log = isinstance(log[0][0], float)

    if is_distance_log:
        final = log[-1]
        finite_vals = [x for x in final if x != float('inf')]
        # Per-graph scale
        S = (max(finite_vals) + 1.0) if finite_vals else 1.0

        normalized = []
        for state in log:
            # map inf -> S, then divide by S
            state_norm = [((S if x == float('inf') else x) / S) for x in state]
            normalized.append(state_norm)
        return mx.array(normalized, dtype=mx.float32)

    else:
        # predecessor log: None -> -1 sentinel
        cleaned_log = []
        for state in log:
            cleaned_state = [(-1 if x is None else x) for x in state]
            cleaned_log.append(cleaned_state)
        return mx.array(cleaned_log)



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


if __name__ == "__main__":
    # Only generate and save datasets when this script is run directly
    print("Generating datasets...")
    
    train_dataset = generated_dataset(1500, 20, 0.2, 2)
    val_dataset = generated_dataset(100, 20, 0.2, 2)
    test_dataset = generated_dataset(100, 20, 0.2, 2)

    save_dataset(train_dataset, "train_dataset.npz")
    save_dataset(val_dataset, "val_dataset.npz")
    save_dataset(test_dataset, "test_dataset.npz")
    
    print("Datasets saved successfully!")
