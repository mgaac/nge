import mlx.core as mx
import numpy as np
from collections import deque
import random

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

def run_bfs(num_nodes, connection_matrix, start_node=0):
    """Run BFS following the mathematical formulation: x_i^(t+1) = {1 if x_i^(t)=1 OR ∃j:(j,i)∈E ∧ x_j^(t)=1, 0 otherwise}."""
    # Build edge list for incoming edges to each node
    incoming_edges = [[] for _ in range(num_nodes)]
    source_idx = connection_matrix[0].tolist()
    target_idx = connection_matrix[1].tolist()
    
    for src, tgt in zip(source_idx, target_idx):
        if src != tgt:  # Ignore self-loops for BFS propagation
            incoming_edges[int(tgt)].append(int(src))
    
    # Initialize: x_i^(1) = {1 if i = s, 0 if i ≠ s}
    states = []
    x_current = [0.0] * num_nodes
    x_current[start_node] = 1.0
    states.append(mx.array(x_current))
    
    # Run BFS iterations
    for iteration in range(num_nodes):  # At most num_nodes iterations needed
        x_next = [0.0] * num_nodes
        changed = False
        
        for i in range(num_nodes):
            # x_i^(t+1) = 1 if x_i^(t) = 1 OR ∃j:(j,i)∈E ∧ x_j^(t) = 1
            if x_current[i] == 1.0:
                x_next[i] = 1.0
            else:
                # Check if any incoming neighbor is visited
                for j in incoming_edges[i]:
                    if x_current[j] == 1.0:
                        x_next[i] = 1.0
                        changed = True
                        break
        
        if not changed:
            break
            
        x_current = x_next
        states.append(mx.array(x_current))
    
    return states

def run_bellman_ford(num_nodes, connection_matrix, start_node=0):
    """Run Bellman-Ford following mathematical formulation: x_i^(t+1) = min(x_i^(t), min_{(j,i)∈E} x_j^(t) + e_ji^(t))."""
    # Build incoming edge list for each node
    incoming_edges = [[] for _ in range(num_nodes)]
    source_idx = connection_matrix[0].tolist()
    target_idx = connection_matrix[1].tolist()
    
    for src, tgt in zip(source_idx, target_idx):
        if src != tgt:  # Ignore self-loops for Bellman-Ford
            incoming_edges[int(tgt)].append(int(src))
    
    # Initialize: x_i^(1) = {0 if i = s, +∞ if i ≠ s}
    # Replace +∞ with length of longest shortest path + 1 (num_nodes for unweighted graphs)
    max_distance = num_nodes + 1
    
    distance_history = []
    predecessor_history = []
    
    x_current = [max_distance] * num_nodes
    x_current[start_node] = 0.0
    
    # Initialize predecessors: p_i^(t) = {i if i = s, -1 otherwise initially}
    p_current = [-1] * num_nodes
    p_current[start_node] = start_node
    
    distance_history.append(mx.array(x_current))
    predecessor_history.append(mx.array(p_current))
    
    # Bellman-Ford iterations
    for iteration in range(num_nodes - 1):
        x_next = x_current.copy()
        p_next = p_current.copy()
        changed = False
        
        for i in range(num_nodes):
            if i == start_node:
                continue
                
            # x_i^(t+1) = min(x_i^(t), min_{(j,i)∈E} x_j^(t) + e_ji^(t))
            # Since all edge weights are 1: x_i^(t+1) = min(x_i^(t), min_{(j,i)∈E} x_j^(t) + 1)
            min_distance = x_current[i]
            best_predecessor = p_current[i]
            
            for j in incoming_edges[i]:
                new_distance = x_current[j] + 1
                if new_distance < min_distance:
                    min_distance = new_distance
                    best_predecessor = j
                    changed = True
            
            x_next[i] = min_distance
            # p_i^(t) = argmin_{j:(j,i)∈E} x_j^(t) + e_ji^(t)
            if min_distance < max_distance:
                p_next[i] = best_predecessor
        
        if not changed:
            break
            
        x_current = x_next
        p_current = p_next
        distance_history.append(mx.array(x_current))
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
        node_embeddings = mx.random.normal([num_nodes, embedding_dim])
        
        # Run algorithms
        start_node = random.randint(0, num_nodes - 1)
        bfs_states = run_bfs(num_nodes, connection_matrix, start_node)
        bf_distances, bf_predecessors = run_bellman_ford(num_nodes, connection_matrix, start_node)
        
        # Pad outputs
        bfs_states, bf_distances, bf_predecessors, total_steps = pad_algorithm_outputs(
            bfs_states, bf_distances, bf_predecessors
        )
        
        # Create termination targets
        bfs_termination_step = len([s for s in bfs_states if not mx.array_equal(s, bfs_states[-1])])
        bf_termination_step = len([d for d in bf_distances if not mx.array_equal(d, bf_distances[-1])])
        
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

if __name__ == "__main__":
    # Generate dataset
    print("Generating graph dataset...")
    dataset = generate_graph_dataset(num_graphs=100, min_nodes=5, max_nodes=20)
    
    # Save dataset
    save_dataset(dataset)
    
    # Example of using the dataset
    print(f"\nGenerated {len(dataset)} graphs")
    print(f"First graph has {dataset[0]['node_embeddings'].shape[0]} nodes")
    print(f"First graph has {len(dataset[0]['steps'])} execution steps")
    print(f"Connection matrix shape: {dataset[0]['connection_matrix'].shape}") 