import mlx.core as mx
import numpy as np
from collections import deque
import random
import math
from typing import List, Dict, Tuple, Optional

def generate_random_graph(num_nodes, edge_prob=0.3, directed=False):
    """Generate a random graph with self-loops for all nodes."""
    edges = []
    
    # Add self-loops for all nodes
    for i in range(num_nodes):
        edges.append((i, i))
    
    # Add random edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and random.random() < edge_prob:
                edges.append((i, j))
                if not directed:
                    edges.append((j, i))
    
    # Convert to connection matrix format [source_idx, target_idx, edge_weights]
    if edges:
        source_idx = mx.array([e[0] for e in edges])
        target_idx = mx.array([e[1] for e in edges])
        # All edge weights are 1 (as mentioned, no edge weights needed)
        edge_weights = mx.ones(len(edges))
        connection_matrix = mx.stack([source_idx, target_idx, edge_weights])
    else:
        connection_matrix = mx.zeros((3, 0))
    
    return connection_matrix

# === NEW GRAPH GENERATION FUNCTIONS ===

def generate_ladder_graph(num_nodes: int) -> mx.array:
    """Generate a ladder graph with self-loops and random edge weights."""
    if num_nodes < 2:
        raise ValueError("Ladder graph needs at least 2 nodes")
    
    edges = []
    
    # Add self-loops for all nodes
    for i in range(num_nodes):
        edges.append((i, i))
    
    # Create ladder structure (assuming even number of nodes for perfect ladder)
    # If odd, last node connects to previous one
    half = num_nodes // 2
    
    # Connect nodes in two rails
    for i in range(half - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
        if i + half < num_nodes:
            edges.append((i + half, i + half + 1))
            edges.append((i + half + 1, i + half))
    
    # Connect rungs between rails
    for i in range(half):
        if i + half < num_nodes:
            edges.append((i, i + half))
            edges.append((i + half, i))
    
    return _edges_to_connection_matrix(edges)

def generate_2d_grid_graph(num_nodes: int) -> mx.array:
    """Generate a 2D grid graph with self-loops and random edge weights."""
    # Find best grid dimensions
    rows = int(math.sqrt(num_nodes))
    cols = (num_nodes + rows - 1) // rows
    
    edges = []
    
    # Add self-loops
    for i in range(num_nodes):
        edges.append((i, i))
    
    # Add grid connections
    for i in range(num_nodes):
        row, col = divmod(i, cols)
        
        # Right neighbor
        if col + 1 < cols and i + 1 < num_nodes:
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        
        # Down neighbor
        if row + 1 < rows and i + cols < num_nodes:
            edges.append((i, i + cols))
            edges.append((i + cols, i))
    
    return _edges_to_connection_matrix(edges)

def generate_tree_prufer(num_nodes: int) -> mx.array:
    """Generate a tree using Prüfer sequences."""
    if num_nodes < 2:
        return _edges_to_connection_matrix([(0, 0)] if num_nodes == 1 else [])
    
    # Generate random Prüfer sequence
    prufer_sequence = [random.randint(0, num_nodes - 1) for _ in range(num_nodes - 2)]
    
    # Convert to tree edges
    degree = [1] * num_nodes
    for v in prufer_sequence:
        degree[v] += 1
    
    edges = []
    
    # Add self-loops
    for i in range(num_nodes):
        edges.append((i, i))
    
    # Build tree from Prüfer sequence
    for v in prufer_sequence:
        # Find leaf (degree 1 node with smallest index)
        for u in range(num_nodes):
            if degree[u] == 1:
                edges.append((u, v))
                edges.append((v, u))
                degree[u] -= 1
                degree[v] -= 1
                break
    
    # Add final edge between remaining degree-1 nodes
    remaining = [i for i in range(num_nodes) if degree[i] == 1]
    if len(remaining) == 2:
        edges.append((remaining[0], remaining[1]))
        edges.append((remaining[1], remaining[0]))
    
    return _edges_to_connection_matrix(edges)

def generate_erdos_renyi_graph(num_nodes: int) -> mx.array:
    """Generate Erdős-Rényi graph with p = min(log₂|V|/|V|, 0.5)."""
    if num_nodes <= 1:
        return _edges_to_connection_matrix([(0, 0)] if num_nodes == 1 else [])
    
    p = min(math.log2(num_nodes) / num_nodes, 0.5)
    
    edges = []
    
    # Add self-loops
    for i in range(num_nodes):
        edges.append((i, i))
    
    # Add random edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < p:
                edges.append((i, j))
                edges.append((j, i))
    
    return _edges_to_connection_matrix(edges)

def generate_barabasi_albert_graph(num_nodes: int) -> mx.array:
    """Generate Barabási-Albert preferential attachment graph."""
    if num_nodes < 2:
        return _edges_to_connection_matrix([(0, 0)] if num_nodes == 1 else [])
    
    m = random.choice([4, 5])  # Each new node attaches with 4 or 5 edges
    m = min(m, num_nodes - 1)  # Can't exceed available nodes
    
    edges = []
    
    # Add self-loops
    for i in range(num_nodes):
        edges.append((i, i))
    
    # Start with complete graph of m+1 nodes
    for i in range(min(m + 1, num_nodes)):
        for j in range(i + 1, min(m + 1, num_nodes)):
            edges.append((i, j))
            edges.append((j, i))
    
    # Add remaining nodes with preferential attachment
    degree = [0] * num_nodes
    for i, j, _ in [(e[0], e[1], 1) for e in edges if e[0] != e[1]]:
        degree[i] += 1
        degree[j] += 1
    
    for new_node in range(m + 1, num_nodes):
        # Select m existing nodes based on degree (preferential attachment)
        total_degree = sum(degree[:new_node])
        if total_degree == 0:
            # If no edges yet, connect to random nodes
            targets = random.sample(range(new_node), min(m, new_node))
        else:
            targets = []
            for _ in range(min(m, new_node)):
                # Roulette wheel selection
                r = random.random() * total_degree
                cumsum = 0
                for i in range(new_node):
                    cumsum += degree[i]
                    if cumsum >= r and i not in targets:
                        targets.append(i)
                        break
                if len(targets) == 0:  # Fallback
                    targets.append(random.randint(0, new_node - 1))
        
        # Add edges
        for target in targets:
            edges.append((new_node, target))
            edges.append((target, new_node))
            degree[new_node] += 1
            degree[target] += 1
    
    return _edges_to_connection_matrix(edges)

def generate_4_community_graph(num_nodes: int) -> mx.array:
    """Generate 4-community graph."""
    if num_nodes < 4:
        raise ValueError("4-community graph needs at least 4 nodes")
    
    # Divide nodes into 4 communities
    community_sizes = [num_nodes // 4] * 4
    for i in range(num_nodes % 4):
        community_sizes[i] += 1
    
    communities = []
    start = 0
    for size in community_sizes:
        communities.append(list(range(start, start + size)))
        start += size
    
    edges = []
    
    # Add self-loops
    for i in range(num_nodes):
        edges.append((i, i))
    
    # Intra-community edges (p=0.7)
    for community in communities:
        for i in community:
            for j in community:
                if i < j and random.random() < 0.7:
                    edges.append((i, j))
                    edges.append((j, i))
    
    # Inter-community edges (p=0.01)
    for i in range(4):
        for j in range(i + 1, 4):
            for node_i in communities[i]:
                for node_j in communities[j]:
                    if random.random() < 0.01:
                        edges.append((node_i, node_j))
                        edges.append((node_j, node_i))
    
    return _edges_to_connection_matrix(edges)

def generate_4_caveman_graph(num_nodes: int) -> mx.array:
    """Generate 4-caveman graph."""
    if num_nodes < 4:
        raise ValueError("4-caveman graph needs at least 4 nodes")
    
    # Divide nodes into 4 cliques
    clique_sizes = [num_nodes // 4] * 4
    for i in range(num_nodes % 4):
        clique_sizes[i] += 1
    
    cliques = []
    start = 0
    for size in clique_sizes:
        cliques.append(list(range(start, start + size)))
        start += size
    
    edges = []
    
    # Add self-loops
    for i in range(num_nodes):
        edges.append((i, i))
    
    # Start with complete cliques, then delete edges with probability 0.7
    for clique in cliques:
        for i in clique:
            for j in clique:
                if i < j:
                    if random.random() > 0.7:  # Keep edge with prob 0.3
                        edges.append((i, j))
                        edges.append((j, i))
    
    # Add shortcut edges between cliques
    num_shortcuts = int(0.025 * num_nodes)
    for _ in range(num_shortcuts):
        # Select two different cliques
        clique1_idx, clique2_idx = random.sample(range(4), 2)
        node1 = random.choice(cliques[clique1_idx])
        node2 = random.choice(cliques[clique2_idx])
        edges.append((node1, node2))
        edges.append((node2, node1))
    
    return _edges_to_connection_matrix(edges)

def _edges_to_connection_matrix(edges: List[Tuple[int, int]]) -> mx.array:
    """Convert edge list to connection matrix with random weights."""
    if not edges:
        return mx.zeros((3, 0))
    
    source_idx = mx.array([e[0] for e in edges])
    target_idx = mx.array([e[1] for e in edges])
    
    # Random edge weights from [0.2, 1.0]
    edge_weights = mx.random.uniform(low=0.2, high=1.0, shape=(len(edges),))
    
    return mx.stack([source_idx, target_idx, edge_weights])

# === DATASET GENERATION WITH CATEGORIES ===

def generate_graph_by_category(category: str, num_nodes: int) -> mx.array:
    """Generate a graph of the specified category."""
    if category == "random":
        return generate_random_graph(num_nodes, edge_prob=0.3)
    elif category == "ladder":
        return generate_ladder_graph(num_nodes)
    elif category == "grid":
        return generate_2d_grid_graph(num_nodes)
    elif category == "tree":
        return generate_tree_prufer(num_nodes)
    elif category == "erdos_renyi":
        return generate_erdos_renyi_graph(num_nodes)
    elif category == "barabasi_albert":
        return generate_barabasi_albert_graph(num_nodes)
    elif category == "4_community":
        return generate_4_community_graph(num_nodes)
    elif category == "4_caveman":
        return generate_4_caveman_graph(num_nodes)
    else:
        raise ValueError(f"Unknown graph category: {category}")

def run_bfs(num_nodes, connection_matrix, start_node=0):
    """Run BFS following the mathematical formulation: x_i^(t+1) = {1 if x_i^(t)=1 OR ∃j:(j,i)∈E ∧ x_j^(t)=1, 0 otherwise}."""
    # Build edge list for incoming edges to each node
    incoming_edges = [[] for _ in range(num_nodes)]
    source_idx = connection_matrix[0].tolist()
    target_idx = connection_matrix[1].tolist()
    edge_weights = connection_matrix[2].tolist()
    
    for src, tgt, weight in zip(source_idx, target_idx, edge_weights):
        if src != tgt:  # Ignore self-loops for BFS propagation
            incoming_edges[int(tgt)].append(int(src))
    
    # Initialize: x_i^(1) = {1 if i = s, 0 if i ≠ s}
    states = []
    x_current = [0.0] * num_nodes
    x_current[start_node] = 1.0
    states.append(mx.array(x_current))
    
    # Run BFS iterations until convergence
    for iteration in range(num_nodes):  # At most num_nodes iterations needed
        x_next = [0.0] * num_nodes
        
        for i in range(num_nodes):
            # x_i^(t+1) = 1 if x_i^(t) = 1 OR ∃j:(j,i)∈E ∧ x_j^(t) = 1
            if x_current[i] == 1.0:
                x_next[i] = 1.0
            else:
                # Check if any incoming neighbor is visited
                for j in incoming_edges[i]:
                    if x_current[j] == 1.0:
                        x_next[i] = 1.0
                        break
        
        # Check for convergence (no change from previous state)
        if x_next == x_current:
            # Add the final converged state before breaking
            states.append(mx.array(x_next))
            break
            
        x_current = x_next
        states.append(mx.array(x_current))
    
    return states

def run_bellman_ford(num_nodes, connection_matrix, start_node=0):
    """Run Bellman-Ford following mathematical formulation: x_i^(t+1) = min(x_i^(t), min_{(j,i)∈E} x_j^(t) + e_ji^(t))."""
    # Build incoming edge list for each node with weights
    incoming_edges = [[] for _ in range(num_nodes)]
    source_idx = connection_matrix[0].tolist()
    target_idx = connection_matrix[1].tolist()
    edge_weights = connection_matrix[2].tolist()
    
    for src, tgt, weight in zip(source_idx, target_idx, edge_weights):
        if src != tgt:  # Ignore self-loops for Bellman-Ford
            incoming_edges[int(tgt)].append((int(src), float(weight)))
    
    # Initialize: x_i^(1) = {0 if i = s, +∞ if i ≠ s}
    # Replace +∞ with length of longest shortest path + 1 (num_nodes for unweighted graphs)
    max_distance = num_nodes + 1
    
    # Calculate normalization factor: use num_nodes as the maximum attainable distance
    # This ensures distance values are normalized to [0, 1] range for better model training
    normalization_factor = float(num_nodes)
    
    distance_history = []
    predecessor_history = []
    
    x_current = [max_distance] * num_nodes
    x_current[start_node] = 0.0
    
    # Initialize predecessors according to formulation: p_i^(t) = {i if i = s, argmin(...) if i ≠ s}
    p_current = [0] * num_nodes
    p_current[start_node] = start_node
    
    # For all non-source nodes, compute initial argmin of incoming neighbors
    for i in range(num_nodes):
        if i == start_node:
            continue
        
        if not incoming_edges[i]:  # No incoming edges - node isolated
            p_current[i] = i  # Self-predecessor for isolated nodes
        else:
            # Compute argmin_{j:(j,i)∈E} x_j^(t) + e_ji^(t)
            best_cost = float('inf')
            best_predecessor = i  # default to self
            for j, weight in incoming_edges[i]:
                cost = x_current[j] + weight
                if cost < best_cost:
                    best_cost = cost
                    best_predecessor = j
            p_current[i] = best_predecessor
    
    # Normalize distances before storing
    x_normalized = [x / normalization_factor for x in x_current]
    distance_history.append(mx.array(x_normalized))
    predecessor_history.append(mx.array(p_current))
    
    # Bellman-Ford iterations - run for V-1 iterations or until convergence
    for iteration in range(num_nodes - 1):
        x_next = x_current.copy()
        p_next = [0] * num_nodes
        
        for i in range(num_nodes):
            if i == start_node:
                # p_s = s always (from formulation)
                p_next[i] = start_node
                continue
                
            # x_i^(t+1) = min(x_i^(t), min_{(j,i)∈E} x_j^(t) + e_ji^(t))
            min_distance = x_current[i]
            
            for j, weight in incoming_edges[i]:
                new_distance = x_current[j] + weight
                if new_distance < min_distance:
                    min_distance = new_distance
            
            x_next[i] = min_distance
            
            # p_i^(t) = argmin_{j:(j,i)∈E} x_j^(t) + e_ji^(t) (always compute for i ≠ s)
            if not incoming_edges[i]:  # No incoming edges
                p_next[i] = i  # Self-predecessor for isolated nodes
            else:
                best_cost = float('inf')
                best_predecessor = i  # default to self
                for j, weight in incoming_edges[i]:
                    cost = x_current[j] + weight
                    if cost < best_cost:
                        best_cost = cost
                        best_predecessor = j
                p_next[i] = best_predecessor
        
        # Check for convergence (no change from previous state)
        if x_next == x_current and p_next == p_current:
            # Add the final converged state before breaking
            x_normalized = [x / normalization_factor for x in x_next]
            distance_history.append(mx.array(x_normalized))
            predecessor_history.append(mx.array(p_next))
            break
            
        x_current = x_next
        p_current = p_next
        
        # Normalize distances before storing
        x_normalized = [x / normalization_factor for x in x_current]
        distance_history.append(mx.array(x_normalized))
        predecessor_history.append(mx.array(p_current))
    
    return distance_history, predecessor_history

def pad_algorithm_outputs(bfs_states, bf_distances, bf_predecessors):
    """Pad outputs so both algorithms have the same number of steps."""
    max_steps = max(len(bfs_states), len(bf_distances))
    
    # Pad BFS states
    while len(bfs_states) < max_steps:
        bfs_states.append(bfs_states[-1])
    
    # Pad BF outputs
    while len(bf_distances) < max_steps:
        bf_distances.append(bf_distances[-1])
        bf_predecessors.append(bf_predecessors[-1])
    
    return bfs_states, bf_distances, bf_predecessors, max_steps

def generate_graph_dataset(num_graphs=100, min_nodes=5, max_nodes=20, embedding_dim=128):
    """Generate a dataset of graphs with BFS and BF execution traces."""
    dataset = []
    
    for _ in range(num_graphs):
        num_nodes = random.randint(min_nodes, max_nodes)
        
        # Generate random graph
        connection_matrix = generate_random_graph(num_nodes, edge_prob=0.3)
        
        # Generate random node embeddings
        node_embeddings = mx.random.normal([num_nodes, embedding_dim], scale=math.sqrt(1.0 / embedding_dim))
        
        # Run algorithms
        start_node = random.randint(0, num_nodes - 1)
        bfs_states = run_bfs(num_nodes, connection_matrix, start_node)
        bf_distances, bf_predecessors = run_bellman_ford(num_nodes, connection_matrix, start_node)
        
        # Pad outputs
        bfs_states, bf_distances, bf_predecessors, total_steps = pad_algorithm_outputs(
            bfs_states, bf_distances, bf_predecessors
        )
        
        # Create termination targets - find when algorithms actually converged
        bfs_termination_step = len(bfs_states)
        for i in range(1, len(bfs_states)):
            if mx.array_equal(bfs_states[i], bfs_states[i-1]):
                bfs_termination_step = i
                break
        
        bf_termination_step = len(bf_distances)
        for i in range(1, len(bf_distances)):
            if mx.array_equal(bf_distances[i], bf_distances[i-1]):
                bf_termination_step = i
                break
        
        # Create step-wise data
        graph_data = {
            'node_embeddings': node_embeddings,
            'connection_matrix': connection_matrix,
            'steps': []
        }
        
        for step in range(total_steps):
            step_data = {
                'bfs_state_targets': bfs_states[step],
                'bf_distance_targets': bf_distances[step],
                'bf_predecessor_targets': bf_predecessors[step],
                'termination_targets': {
                    'bfs': mx.array(1.0 if step >= bfs_termination_step else 0.0),
                    'bf': mx.array(1.0 if step >= bf_termination_step else 0.0)
                }
            }
            graph_data['steps'].append(step_data)
        
        dataset.append(graph_data)
    
    return dataset

def save_dataset(dataset, filename='graph_dataset.npz'):
    """Save dataset to file."""
    # Convert to a format that can be saved
    save_dict = {}
    
    for i, graph in enumerate(dataset):
        # Save graph structure
        save_dict[f'graph_{i}_node_embeddings'] = np.array(graph['node_embeddings'])
        save_dict[f'graph_{i}_connection_matrix'] = np.array(graph['connection_matrix'])
        save_dict[f'graph_{i}_num_steps'] = len(graph['steps'])
        
        # Save step data
        for j, step in enumerate(graph['steps']):
            save_dict[f'graph_{i}_step_{j}_bfs_states'] = np.array(step['bfs_state_targets'])
            save_dict[f'graph_{i}_step_{j}_bf_distances'] = np.array(step['bf_distance_targets'])
            save_dict[f'graph_{i}_step_{j}_bf_predecessors'] = np.array(step['bf_predecessor_targets'])
            save_dict[f'graph_{i}_step_{j}_bfs_term'] = np.array(step['termination_targets']['bfs'])
            save_dict[f'graph_{i}_step_{j}_bf_term'] = np.array(step['termination_targets']['bf'])
    
    save_dict['num_graphs'] = len(dataset)
    np.savez(filename, **save_dict)
    print(f"Dataset saved to {filename}")

def save_extended_dataset(dataset: Dict[str, List], base_filename: str = 'extended_dataset'):
    """Save extended dataset with splits to separate files."""
    for split_name, split_data in dataset.items():
        filename = f"{base_filename}_{split_name}.npz"
        save_dict = {}
        
        for i, graph in enumerate(split_data):
            # Save graph metadata
            save_dict[f'graph_{i}_node_embeddings'] = np.array(graph['node_embeddings'])
            save_dict[f'graph_{i}_connection_matrix'] = np.array(graph['connection_matrix'])
            save_dict[f'graph_{i}_category'] = graph['category']
            save_dict[f'graph_{i}_split'] = graph['split']
            save_dict[f'graph_{i}_num_nodes'] = graph['num_nodes']
            save_dict[f'graph_{i}_start_node'] = graph['start_node']
            save_dict[f'graph_{i}_num_steps'] = len(graph['steps'])
            
            # Save step data
            for j, step in enumerate(graph['steps']):
                save_dict[f'graph_{i}_step_{j}_bfs_states'] = np.array(step['bfs_state_targets'])
                save_dict[f'graph_{i}_step_{j}_bf_distances'] = np.array(step['bf_distance_targets'])
                save_dict[f'graph_{i}_step_{j}_bf_predecessors'] = np.array(step['bf_predecessor_targets'])
                save_dict[f'graph_{i}_step_{j}_bfs_term'] = np.array(step['termination_targets']['bfs'])
                save_dict[f'graph_{i}_step_{j}_bf_term'] = np.array(step['termination_targets']['bf'])
        
        save_dict['num_graphs'] = len(split_data)
        np.savez(filename, **save_dict)
        print(f"{split_name.capitalize()} split saved to {filename}")

def load_dataset(filename='graph_dataset.npz'):
    """Load dataset from file."""
    data = np.load(filename)
    num_graphs = int(data['num_graphs'])
    
    dataset = []
    for i in range(num_graphs):
        graph_data = {
            'node_embeddings': mx.array(data[f'graph_{i}_node_embeddings']),
            'connection_matrix': mx.array(data[f'graph_{i}_connection_matrix']),
            'steps': []
        }
        
        num_steps = int(data[f'graph_{i}_num_steps'])
        for j in range(num_steps):
            step_data = {
                'bfs_state_targets': mx.array(data[f'graph_{i}_step_{j}_bfs_states']),
                'bf_distance_targets': mx.array(data[f'graph_{i}_step_{j}_bf_distances']),
                'bf_predecessor_targets': mx.array(data[f'graph_{i}_step_{j}_bf_predecessors']),
                'termination_targets': {
                    'bfs': mx.array(data[f'graph_{i}_step_{j}_bfs_term']),
                    'bf': mx.array(data[f'graph_{i}_step_{j}_bf_term'])
                }
            }
            graph_data['steps'].append(step_data)
        
        dataset.append(graph_data)
    
    return dataset

def load_extended_dataset(base_filename: str = 'extended_dataset') -> Dict[str, List]:
    """Load extended dataset with splits from separate files."""
    dataset = {}
    
    for split_name in ['train', 'val', 'test']:
        filename = f"{base_filename}_{split_name}.npz"
        try:
            data = np.load(filename)
            num_graphs = int(data['num_graphs'])
            
            split_data = []
            for i in range(num_graphs):
                graph_data = {
                    'node_embeddings': mx.array(data[f'graph_{i}_node_embeddings']),
                    'connection_matrix': mx.array(data[f'graph_{i}_connection_matrix']),
                    'category': str(data[f'graph_{i}_category']),
                    'split': str(data[f'graph_{i}_split']),
                    'num_nodes': int(data[f'graph_{i}_num_nodes']),
                    'start_node': int(data[f'graph_{i}_start_node']),
                    'steps': []
                }
                
                num_steps = int(data[f'graph_{i}_num_steps'])
                for j in range(num_steps):
                    step_data = {
                        'bfs_state_targets': mx.array(data[f'graph_{i}_step_{j}_bfs_states']),
                        'bf_distance_targets': mx.array(data[f'graph_{i}_step_{j}_bf_distances']),
                        'bf_predecessor_targets': mx.array(data[f'graph_{i}_step_{j}_bf_predecessors']),
                        'termination_targets': {
                            'bfs': mx.array(data[f'graph_{i}_step_{j}_bfs_term']),
                            'bf': mx.array(data[f'graph_{i}_step_{j}_bf_term'])
                        }
                    }
                    graph_data['steps'].append(step_data)
                
                split_data.append(graph_data)
            
            dataset[split_name] = split_data
            print(f"Loaded {len(split_data)} graphs from {filename}")
            
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping {split_name} split")
            dataset[split_name] = []
    
    return dataset

def generate_extended_dataset(
    dataset_type: str = "both",  # "basic", "extended", or "both"
    embedding_dim: int = 128
) -> Dict[str, List]:
    """
    Generate the extended dataset with proper splits and categories.
    
    Args:
        dataset_type: "basic" (original), "extended" (7 categories), or "both"
        embedding_dim: Dimension of node embeddings
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    
    # Define graph categories for extended dataset
    extended_categories = [
        "ladder", "grid", "tree", "erdos_renyi", 
        "barabasi_albert", "4_community", "4_caveman"
    ]
    
    # Define dataset splits configuration
    splits_config = {
        'train': {'graphs_per_category': 100, 'node_counts': [20]},
        'val': {'graphs_per_category': 5, 'node_counts': [20]},
        'test': {'graphs_per_category': 5, 'node_counts': [20, 50, 100]}
    }
    
    dataset = {'train': [], 'val': [], 'test': []}
    
    for split_name, config in splits_config.items():
        print(f"Generating {split_name} split...")
        
        for node_count in config['node_counts']:
            if dataset_type in ["basic", "both"]:
                # Generate basic random graphs
                for _ in range(config['graphs_per_category']):
                    graph_data = _generate_single_graph(
                        "random", node_count, embedding_dim, split_name
                    )
                    if graph_data:
                        dataset[split_name].append(graph_data)
            
            if dataset_type in ["extended", "both"]:
                # Generate extended category graphs
                for category in extended_categories:
                    for _ in range(config['graphs_per_category']):
                        graph_data = _generate_single_graph(
                            category, node_count, embedding_dim, split_name
                        )
                        if graph_data:
                            dataset[split_name].append(graph_data)
    
    return dataset

def _generate_single_graph(
    category: str, 
    num_nodes: int, 
    embedding_dim: int,
    split: str
) -> Optional[Dict]:
    """Generate a single graph with algorithm execution traces."""
    try:
        # Generate graph
        connection_matrix = generate_graph_by_category(category, num_nodes)
        
        # Generate random node embeddings
        node_embeddings = mx.random.normal(
            [num_nodes, embedding_dim], 
            scale=math.sqrt(1.0 / embedding_dim)
        )
        
        # Run algorithms
        start_node = random.randint(0, num_nodes - 1)
        bfs_states = run_bfs(num_nodes, connection_matrix, start_node)
        bf_distances, bf_predecessors = run_bellman_ford(num_nodes, connection_matrix, start_node)
        
        # Pad outputs
        bfs_states, bf_distances, bf_predecessors, total_steps = pad_algorithm_outputs(
            bfs_states, bf_distances, bf_predecessors
        )
        
        # Create termination targets - find when algorithms actually converged
        bfs_termination_step = len(bfs_states)
        for i in range(1, len(bfs_states)):
            if mx.array_equal(bfs_states[i], bfs_states[i-1]):
                bfs_termination_step = i
                break
        
        bf_termination_step = len(bf_distances)
        for i in range(1, len(bf_distances)):
            if mx.array_equal(bf_distances[i], bf_distances[i-1]):
                bf_termination_step = i
                break
        
        # Create step-wise data
        graph_data = {
            'node_embeddings': node_embeddings,
            'connection_matrix': connection_matrix,
            'category': category,
            'split': split,
            'num_nodes': num_nodes,
            'start_node': start_node,
            'steps': []
        }
        
        for step in range(total_steps):
            step_data = {
                'bfs_state_targets': bfs_states[step],
                'bf_distance_targets': bf_distances[step],
                'bf_predecessor_targets': bf_predecessors[step],
                'termination_targets': {
                    'bfs': mx.array(1.0 if step >= bfs_termination_step else 0.0),
                    'bf': mx.array(1.0 if step >= bf_termination_step else 0.0)
                }
            }
            graph_data['steps'].append(step_data)
        
        return graph_data
        
    except Exception as e:
        print(f"Error generating {category} graph with {num_nodes} nodes: {e}")
        return None

# === DATASET UTILITIES ===

def shuffle_dataset(dataset: List[Dict], seed: Optional[int] = None) -> List[Dict]:
    """Shuffle a dataset split."""
    if seed is not None:
        random.seed(seed)
    
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    return shuffled

def get_dataset_statistics(dataset: Dict[str, List]) -> Dict:
    """Get statistics about the dataset."""
    stats = {}
    
    for split_name, split_data in dataset.items():
        split_stats = {
            'total_graphs': len(split_data),
            'categories': {},
            'node_counts': {},
            'avg_steps': 0,
            'total_steps': 0
        }
        
        total_steps = 0
        for graph in split_data:
            category = graph.get('category', 'unknown')
            num_nodes = graph.get('num_nodes', 0)
            num_steps = len(graph['steps'])
            
            # Category counts
            if category not in split_stats['categories']:
                split_stats['categories'][category] = 0
            split_stats['categories'][category] += 1
            
            # Node count distribution
            if num_nodes not in split_stats['node_counts']:
                split_stats['node_counts'][num_nodes] = 0
            split_stats['node_counts'][num_nodes] += 1
            
            total_steps += num_steps
        
        split_stats['total_steps'] = total_steps
        split_stats['avg_steps'] = total_steps / len(split_data) if split_data else 0
        stats[split_name] = split_stats
    
    return stats

def create_data_loaders(dataset: Dict[str, List], batch_size: int = 32, shuffle_train: bool = True):
    """Create simple batch iterators for the dataset."""
    
    def batch_iterator(data: List[Dict], batch_size: int, shuffle: bool = False):
        """Simple batch iterator."""
        if shuffle:
            data = shuffle_dataset(data)
        
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    loaders = {}
    for split_name, split_data in dataset.items():
        shuffle = shuffle_train if split_name == 'train' else False
        loaders[split_name] = lambda data=split_data, bs=batch_size, shuf=shuffle: batch_iterator(data, bs, shuf)
    
    return loaders

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate graph datasets for algorithm learning')
    parser.add_argument('--dataset_type', choices=['basic', 'extended', 'both'], 
                       default='basic', help='Type of dataset to generate')
    parser.add_argument('--num_graphs', type=int, default=100,
                       help='Number of graphs for basic dataset (ignored for extended)')
    parser.add_argument('--min_nodes', type=int, default=5,
                       help='Minimum nodes for basic dataset')
    parser.add_argument('--max_nodes', type=int, default=20,
                       help='Maximum nodes for basic dataset')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Node embedding dimension')
    parser.add_argument('--output_prefix', type=str, default='graph_dataset',
                       help='Output filename prefix')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.dataset_type == 'basic':
        # Generate basic dataset (original functionality)
        print("Generating basic graph dataset...")
        dataset = generate_graph_dataset(
            num_graphs=args.num_graphs,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            embedding_dim=args.embedding_dim
        )
        
        # Save basic dataset
        save_dataset(dataset, f"{args.output_prefix}.npz")
        
        print(f"\nGenerated {len(dataset)} graphs")
        print(f"First graph has {dataset[0]['node_embeddings'].shape[0]} nodes")
        print(f"First graph has {len(dataset[0]['steps'])} execution steps")
        print(f"Connection matrix shape: {dataset[0]['connection_matrix'].shape}")
        
    else:
        # Generate extended dataset with proper splits
        print(f"Generating {args.dataset_type} graph dataset...")
        dataset = generate_extended_dataset(
            dataset_type=args.dataset_type,
            embedding_dim=args.embedding_dim
        )
        
        # Save extended dataset
        save_extended_dataset(dataset, args.output_prefix)
        
        # Print statistics
        stats = get_dataset_statistics(dataset)
        print("\n=== Dataset Statistics ===")
        for split_name, split_stats in stats.items():
            print(f"\n{split_name.upper()} Split:")
            print(f"  Total graphs: {split_stats['total_graphs']}")
            print(f"  Categories: {split_stats['categories']}")
            print(f"  Node counts: {split_stats['node_counts']}")
            print(f"  Average steps per graph: {split_stats['avg_steps']:.2f}")
        
        # Demonstrate data loaders
        print("\n=== Testing Data Loaders ===")
        loaders = create_data_loaders(dataset, batch_size=4)
        
        for split_name, loader_fn in loaders.items():
            batch_count = 0
            for batch in loader_fn():
                batch_count += 1
                if batch_count == 1:  # Show first batch info
                    print(f"{split_name} - First batch: {len(batch)} graphs")
                    if batch:
                        sample_graph = batch[0]
                        print(f"  Sample: {sample_graph['category']} graph with {sample_graph['num_nodes']} nodes")
            print(f"{split_name} - Total batches: {batch_count}") 