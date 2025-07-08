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

def append_uniform_edge_weights(edge_matrix, low=0.2, high=1.0):
    if edge_matrix.size == 0:
        return edge_matrix
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
                if cand <= best_cost:          # non-strict comparison
                    best_cost = cand
                    best_pred = j

            new_predecessor[i] = best_pred     # record arg-min j
            if best_cost < new_distance[i]:    # genuine relaxation?
                new_distance[i] = best_cost
                updated = True

        distance      = new_distance
        predecessor   = new_predecessor
        distance_log.append(distance.copy())
        predecessor_log.append(predecessor.copy())

        if not updated:
            break

    return distance_log, predecessor_log



def clean_bf_logs(log: List[List[float]] | List[List[int | None]]) -> mx.array:
    # Determine if this is a distance log (float) or predecessor log (int/None)
    is_distance_log = isinstance(log[0][0], float)
    # Defensive: handle empty log
    if not log or not log[-1]:
        return mx.array(log)

    # Compute replacements
    if is_distance_log:
        # Replace inf with max finite value + 1
        finite_vals = [x for x in log[-1] if x != float('inf')]
        inf_replacement = (max(finite_vals) + 1) if finite_vals else 1e6
        def clean_val(x):
            cleaned = inf_replacement if x == float('inf') else x
            return cleaned / inf_replacement
    else:
        # Replace None with 0
        def clean_val(x):
            return 0 if x is None else x

    cleaned_log = []
    for state in log:
        cleaned_state = [clean_val(x) for x in state]
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
            edge_matrix = append_uniform_edge_weights(barabasi_albert_edge_matrix(num_nodes, m))
        else:
            edge_matrix = append_uniform_edge_weights(erdos_renyi_edge_matrix(num_nodes, p))

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


save_dataset(generated_dataset(700, 20, 0.2, 2), "dataset.npz")

