import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from generate_dataset import load_dataset, generate_graph_dataset
import argparse
import sys

class DatasetInspector:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def print_dataset_summary(self):
        """Print high-level statistics about the dataset."""
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        num_graphs = len(self.dataset)
        print(f"Number of graphs: {num_graphs}")
        
        # Collect statistics
        node_counts = []
        edge_counts = []
        step_counts = []
        
        for graph in self.dataset:
            node_counts.append(graph['node_embeddings'].shape[0])
            edge_counts.append(graph['connection_matrix'].shape[1])
            step_counts.append(len(graph['steps']))
        
        print(f"\nNode counts:")
        print(f"  Min: {min(node_counts)}, Max: {max(node_counts)}, Avg: {np.mean(node_counts):.1f}")
        
        print(f"\nEdge counts:")
        print(f"  Min: {min(edge_counts)}, Max: {max(edge_counts)}, Avg: {np.mean(edge_counts):.1f}")
        
        print(f"\nExecution steps:")
        print(f"  Min: {min(step_counts)}, Max: {max(step_counts)}, Avg: {np.mean(step_counts):.1f}")
        
        print(f"\nNode embedding dimension: {self.dataset[0]['node_embeddings'].shape[1]}")
    
    def inspect_graph(self, graph_idx):
        """Inspect a specific graph in detail."""
        if graph_idx >= len(self.dataset):
            print(f"Graph index {graph_idx} out of range. Dataset has {len(self.dataset)} graphs.")
            return
        
        graph = self.dataset[graph_idx]
        
        print("=" * 60)
        print(f"GRAPH {graph_idx} INSPECTION")
        print("=" * 60)
        
        num_nodes = graph['node_embeddings'].shape[0]
        num_edges = graph['connection_matrix'].shape[1]
        num_steps = len(graph['steps'])
        
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        print(f"Execution steps: {num_steps}")
        
        # Show connection matrix structure
        conn_matrix = graph['connection_matrix']
        source_idx = conn_matrix[0].tolist()
        target_idx = conn_matrix[1].tolist()
        edge_weights = conn_matrix[2].tolist()
        
        print(f"\nEdge structure (first 10 edges):")
        for i in range(min(10, len(source_idx))):
            print(f"  {int(source_idx[i])} -> {int(target_idx[i])} (weight: {edge_weights[i]:.1f})")
        if len(source_idx) > 10:
            print(f"  ... and {len(source_idx) - 10} more edges")
        
        # Show algorithm execution summary
        print(f"\nAlgorithm execution trace:")
        for step_idx, step in enumerate(graph['steps']):
            bfs_term = step['termination_targets']['bfs']
            bf_term = step['termination_targets']['bf']
            
            bfs_visited_count = int(mx.sum(step['bfs_state_targets']))
            bf_reachable_count = int(mx.sum(step['bf_distance_targets'] < num_nodes * 10))
            
            print(f"  Step {step_idx}: BFS visited={bfs_visited_count}/{num_nodes}, " +
                  f"BF reachable={bf_reachable_count}/{num_nodes}, " +
                  f"BFS_term={float(bfs_term):.0f}, BF_term={float(bf_term):.0f}")
    
    def visualize_graph(self, graph_idx, save_path=None):
        """Visualize a graph using networkx and matplotlib."""
        if graph_idx >= len(self.dataset):
            print(f"Graph index {graph_idx} out of range.")
            return
        
        graph = self.dataset[graph_idx]
        conn_matrix = graph['connection_matrix']
        
        # Create networkx graph
        G = nx.DiGraph()
        
        num_nodes = graph['node_embeddings'].shape[0]
        G.add_nodes_from(range(num_nodes))
        
        # Add edges (exclude self-loops for cleaner visualization)
        source_idx = conn_matrix[0].tolist()
        target_idx = conn_matrix[1].tolist()
        
        for src, tgt in zip(source_idx, target_idx):
            if src != tgt:  # Skip self-loops for visualization
                G.add_edge(int(src), int(tgt))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw the graph
        nx.draw(G, pos, 
                with_labels=True, 
                node_color='lightblue', 
                node_size=500,
                font_size=10,
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.7)
        
        plt.title(f"Graph {graph_idx} Structure\n{num_nodes} nodes, {len(source_idx)} edges")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def trace_algorithm_execution(self, graph_idx, algorithm='both'):
        """Show step-by-step algorithm execution."""
        if graph_idx >= len(self.dataset):
            print(f"Graph index {graph_idx} out of range.")
            return
        
        graph = self.dataset[graph_idx]
        
        print("=" * 60)
        print(f"ALGORITHM EXECUTION TRACE - Graph {graph_idx}")
        print("=" * 60)
        
        num_nodes = graph['node_embeddings'].shape[0]
        
        for step_idx, step in enumerate(graph['steps']):
            print(f"\n--- Step {step_idx} ---")
            
            if algorithm in ['bfs', 'both']:
                bfs_states = step['bfs_state_targets']
                visited_nodes = [i for i, visited in enumerate(bfs_states) if visited > 0.5]
                print(f"BFS visited nodes: {visited_nodes}")
                print(f"BFS termination: {float(step['termination_targets']['bfs'])}")
            
            if algorithm in ['bf', 'both']:
                bf_distances = step['bf_distance_targets']
                bf_predecessors = step['bf_predecessor_targets']
                
                print("Bellman-Ford distances:")
                for node_idx, distance in enumerate(bf_distances):
                    if distance < num_nodes * 10:  # Not infinite
                        pred = int(bf_predecessors[node_idx])
                        pred_str = str(pred) if pred >= 0 else "None"
                        print(f"  Node {node_idx}: distance={float(distance):.1f}, predecessor={pred_str}")
                
                print(f"BF termination: {float(step['termination_targets']['bf'])}")
    
    def plot_algorithm_convergence(self, graph_idx, save_path=None):
        """Plot how algorithms converge over time."""
        if graph_idx >= len(self.dataset):
            print(f"Graph index {graph_idx} out of range.")
            return
        
        graph = self.dataset[graph_idx]
        num_nodes = graph['node_embeddings'].shape[0]
        
        steps = []
        bfs_visited_counts = []
        bf_reachable_counts = []
        bfs_termination = []
        bf_termination = []
        
        for step_idx, step in enumerate(graph['steps']):
            steps.append(step_idx)
            
            # Count BFS visited nodes
            bfs_count = int(mx.sum(step['bfs_state_targets']))
            bfs_visited_counts.append(bfs_count)
            
            # Count BF reachable nodes
            bf_count = int(mx.sum(step['bf_distance_targets'] < num_nodes * 10))
            bf_reachable_counts.append(bf_count)
            
            # Termination flags
            bfs_termination.append(float(step['termination_targets']['bfs']))
            bf_termination.append(float(step['termination_targets']['bf']))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot node counts
        ax1.plot(steps, bfs_visited_counts, 'b-o', label='BFS Visited Nodes', markersize=4)
        ax1.plot(steps, bf_reachable_counts, 'r-s', label='BF Reachable Nodes', markersize=4)
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title(f'Algorithm Convergence - Graph {graph_idx}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, num_nodes + 1)
        
        # Plot termination flags
        ax2.plot(steps, bfs_termination, 'b-o', label='BFS Termination', markersize=4)
        ax2.plot(steps, bf_termination, 'r-s', label='BF Termination', markersize=4)
        ax2.set_xlabel('Execution Step')
        ax2.set_ylabel('Termination Flag')
        ax2.set_title('Algorithm Termination')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_graphs(self, graph_indices):
        """Compare multiple graphs side by side."""
        print("=" * 60)
        print("GRAPH COMPARISON")
        print("=" * 60)
        
        print(f"{'Graph':<8} {'Nodes':<8} {'Edges':<8} {'Steps':<8} {'BFS_Steps':<10} {'BF_Steps':<10}")
        print("-" * 60)
        
        for idx in graph_indices:
            if idx >= len(self.dataset):
                continue
            
            graph = self.dataset[idx]
            num_nodes = graph['node_embeddings'].shape[0]
            num_edges = graph['connection_matrix'].shape[1]
            num_steps = len(graph['steps'])
            
            # Find when each algorithm terminates
            bfs_term_step = num_steps
            bf_term_step = num_steps
            
            for step_idx, step in enumerate(graph['steps']):
                if float(step['termination_targets']['bfs']) > 0.5 and bfs_term_step == num_steps:
                    bfs_term_step = step_idx
                if float(step['termination_targets']['bf']) > 0.5 and bf_term_step == num_steps:
                    bf_term_step = step_idx
            
            print(f"{idx:<8} {num_nodes:<8} {num_edges:<8} {num_steps:<8} {bfs_term_step:<10} {bf_term_step:<10}")
    
    def inspect_raw_vectors(self, graph_idx, step_idx):
        """Inspect raw input vectors and target vectors for a specific step."""
        if graph_idx >= len(self.dataset):
            print(f"Graph index {graph_idx} out of range.")
            return
        
        graph = self.dataset[graph_idx]
        if step_idx >= len(graph['steps']):
            print(f"Step index {step_idx} out of range. Graph has {len(graph['steps'])} steps.")
            return
        
        print("=" * 80)
        print(f"RAW VECTORS INSPECTION - Graph {graph_idx}, Step {step_idx}")
        print("=" * 80)
        
        # Model inputs
        node_embeddings = graph['node_embeddings']
        connection_matrix = graph['connection_matrix']
        
        print(f"INPUT TO MODEL:")
        print(f"  Node embeddings shape: {node_embeddings.shape}")
        print(f"  Node embedding sample (first 3 dims of first 3 nodes):")
        for i in range(min(3, node_embeddings.shape[0])):
            sample_vals = node_embeddings[i][:3]
            print(f"    Node {i}: [{sample_vals[0]:.4f}, {sample_vals[1]:.4f}, {sample_vals[2]:.4f}, ...]")
        
        print(f"\n  Connection matrix shape: {connection_matrix.shape}")
        print(f"  Connection matrix sample (first 5 edges):")
        for i in range(min(5, connection_matrix.shape[1])):
            src, tgt, weight = connection_matrix[:, i]
            print(f"    Edge {i}: {int(src)} -> {int(tgt)} (weight: {float(weight):.1f})")
        
        # Target outputs
        step_data = graph['steps'][step_idx]
        
        print(f"\nTARGET OUTPUTS:")
        print(f"  BFS state targets shape: {step_data['bfs_state_targets'].shape}")
        bfs_targets = step_data['bfs_state_targets']
        visited_nodes = [i for i, val in enumerate(bfs_targets) if float(val) > 0.5]
        print(f"  BFS visited nodes: {visited_nodes}")
        print(f"  BFS raw values: {[float(bfs_targets[i]) for i in range(min(5, len(bfs_targets)))]}")
        
        print(f"\n  BF distance targets shape: {step_data['bf_distance_targets'].shape}")
        bf_distances = step_data['bf_distance_targets']
        print(f"  BF distance values (first 5 nodes): {[float(bf_distances[i]) for i in range(min(5, len(bf_distances)))]}")
        
        print(f"\n  BF predecessor targets shape: {step_data['bf_predecessor_targets'].shape}")
        bf_predecessors = step_data['bf_predecessor_targets']
        print(f"  BF predecessor values (first 5 nodes): {[int(bf_predecessors[i]) for i in range(min(5, len(bf_predecessors)))]}")
        
        print(f"\n  Termination targets:")
        print(f"    BFS termination: {float(step_data['termination_targets']['bfs'])}")
        print(f"    BF termination: {float(step_data['termination_targets']['bf'])}")
    
    def inspect_model_format(self, graph_idx, step_idx):
        """Show exactly what format the model expects and what targets should be."""
        if graph_idx >= len(self.dataset):
            print(f"Graph index {graph_idx} out of range.")
            return
        
        graph = self.dataset[graph_idx]
        if step_idx >= len(graph['steps']):
            print(f"Step index {step_idx} out of range.")
            return
        
        print("=" * 80)
        print(f"MODEL INPUT/OUTPUT FORMAT - Graph {graph_idx}, Step {step_idx}")
        print("=" * 80)
        
        # Prepare model input data
        node_embeddings = graph['node_embeddings']
        connection_matrix = graph['connection_matrix']
        model_input = (node_embeddings, connection_matrix)
        
        step_data = graph['steps'][step_idx]
        
        print("MODEL INPUT FORMAT:")
        print("  model_input = (node_embeddings, connection_matrix)")
        print(f"  node_embeddings: shape {node_embeddings.shape}, dtype {node_embeddings.dtype}")
        print(f"  connection_matrix: shape {connection_matrix.shape}, dtype {connection_matrix.dtype}")
        print("    connection_matrix[0] = source indices")
        print("    connection_matrix[1] = target indices") 
        print("    connection_matrix[2] = edge weights")
        
        print("\nEXPECTED MODEL OUTPUT FORMAT:")
        print("  bfs_output, bf_output, termination_probs = model(model_input)")
        print("  bfs_output: shape should match bfs_state_targets")
        print("  bf_output: (bf_distance_predictions, bf_predecessor_predictions)")
        print("  termination_probs: {'bfs': prob, 'bf': prob}")
        
        print("\nTARGET SHAPES FOR LOSS COMPUTATION:")
        print(f"  bfs_state_targets: {step_data['bfs_state_targets'].shape}")
        print(f"  bf_distance_targets: {step_data['bf_distance_targets'].shape}")
        print(f"  bf_predecessor_targets: {step_data['bf_predecessor_targets'].shape}")
        print(f"  termination_targets['bfs']: scalar")
        print(f"  termination_targets['bf']: scalar")
        
        # Show actual values for small examples
        if node_embeddings.shape[0] <= 8:
            print(f"\nFULL TARGETS FOR THIS STEP (small graph):")
            print(f"  BFS states: {[float(x) for x in step_data['bfs_state_targets']]}")
            print(f"  BF distances: {[float(x) for x in step_data['bf_distance_targets']]}")
            print(f"  BF predecessors: {[int(x) for x in step_data['bf_predecessor_targets']]}")
    
    def visualize_algorithm_states(self, graph_idx, step_idx, save_path=None):
        """Visualize the algorithm states on the graph structure."""
        if graph_idx >= len(self.dataset):
            print(f"Graph index {graph_idx} out of range.")
            return
        
        graph = self.dataset[graph_idx]
        if step_idx >= len(graph['steps']):
            print(f"Step index {step_idx} out of range.")
            return
        
        conn_matrix = graph['connection_matrix']
        step_data = graph['steps'][step_idx]
        
        # Create networkx graph
        G = nx.DiGraph()
        num_nodes = graph['node_embeddings'].shape[0]
        G.add_nodes_from(range(num_nodes))
        
        # Add edges (exclude self-loops for visualization)
        source_idx = conn_matrix[0].tolist()
        target_idx = conn_matrix[1].tolist()
        
        for src, tgt in zip(source_idx, target_idx):
            if src != tgt:
                G.add_edge(int(src), int(tgt))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # BFS visualization
        bfs_states = step_data['bfs_state_targets']
        bfs_colors = ['red' if float(bfs_states[i]) > 0.5 else 'lightblue' for i in range(num_nodes)]
        
        nx.draw(G, pos, ax=ax1,
                with_labels=True,
                node_color=bfs_colors,
                node_size=600,
                font_size=12,
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.8)
        ax1.set_title(f'BFS State - Step {step_idx}\n(Red = Visited)')
        
        # Bellman-Ford visualization
        bf_distances = step_data['bf_distance_targets']
        max_dist = max(float(d) for d in bf_distances)
        # Color by distance (blue = close, red = far)
        bf_colors = []
        for i in range(num_nodes):
            dist = float(bf_distances[i])
            if dist >= num_nodes + 1:  # Unreachable
                bf_colors.append('black')
            else:
                intensity = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
                bf_colors.append((intensity, intensity, 1.0))  # Blue gradient
        
        nx.draw(G, pos, ax=ax2,
                with_labels=True,
                node_color=bf_colors,
                node_size=600,
                font_size=12,
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.8)
        ax2.set_title(f'Bellman-Ford Distances - Step {step_idx}\n(Blue = Close, Black = Unreachable)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Algorithm state visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def interactive_mode(inspector):
    """Interactive mode for dataset inspection."""
    print("\n" + "=" * 60)
    print("INTERACTIVE DATASET INSPECTOR")
    print("=" * 60)
    print("Commands:")
    print("  summary                        - Show dataset summary")
    print("  inspect <graph_idx>            - Inspect specific graph")
    print("  visualize <graph_idx>          - Visualize graph structure")
    print("  trace <graph_idx> [alg]        - Show algorithm execution (alg: bfs/bf/both)")
    print("  plot <graph_idx>               - Plot algorithm convergence")
    print("  compare <idx1,idx2,idx3>       - Compare multiple graphs")
    print("  raw <graph_idx> <step_idx>     - Show raw input/target vectors")
    print("  format <graph_idx> <step_idx>  - Show model input/output format")
    print("  states <graph_idx> <step_idx>  - Visualize algorithm states on graph")
    print("  quit                           - Exit")
    print("-" * 60)
    
    while True:
        try:
            command = input("\n> ").strip().split()
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == 'quit':
                break
            elif cmd == 'summary':
                inspector.print_dataset_summary()
            elif cmd == 'inspect' and len(command) > 1:
                inspector.inspect_graph(int(command[1]))
            elif cmd == 'visualize' and len(command) > 1:
                inspector.visualize_graph(int(command[1]))
            elif cmd == 'trace' and len(command) > 1:
                alg = command[2] if len(command) > 2 else 'both'
                inspector.trace_algorithm_execution(int(command[1]), alg)
            elif cmd == 'plot' and len(command) > 1:
                inspector.plot_algorithm_convergence(int(command[1]))
            elif cmd == 'compare' and len(command) > 1:
                indices = [int(x.strip()) for x in command[1].split(',')]
                inspector.compare_graphs(indices)
            elif cmd == 'raw' and len(command) > 2:
                inspector.inspect_raw_vectors(int(command[1]), int(command[2]))
            elif cmd == 'format' and len(command) > 2:
                inspector.inspect_model_format(int(command[1]), int(command[2]))
            elif cmd == 'states' and len(command) > 2:
                inspector.visualize_algorithm_states(int(command[1]), int(command[2]))
            else:
                print("Invalid command or missing arguments.")
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            break

def main():
    parser = argparse.ArgumentParser(description='Inspect graph dataset')
    parser.add_argument('--file', type=str, default='graph_dataset.npz', 
                       help='Dataset file to inspect')
    parser.add_argument('--generate', action='store_true', 
                       help='Generate a new small dataset for testing')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--graph', type=int, default=0,
                       help='Graph index to inspect')
    parser.add_argument('--summary', action='store_true',
                       help='Show dataset summary')
    
    args = parser.parse_args()
    
    # Load or generate dataset
    if args.generate:
        print("Generating test dataset...")
        dataset = generate_graph_dataset(num_graphs=10, min_nodes=5, max_nodes=12)
    else:
        try:
            print(f"Loading dataset from {args.file}...")
            dataset = load_dataset(args.file)
        except FileNotFoundError:
            print(f"Dataset file {args.file} not found. Use --generate to create a test dataset.")
            return
    
    inspector = DatasetInspector(dataset)
    
    if args.interactive:
        interactive_mode(inspector)
    else:
        if args.summary:
            inspector.print_dataset_summary()
        else:
            inspector.print_dataset_summary()
            print()
            inspector.inspect_graph(args.graph)
            inspector.visualize_graph(args.graph)
            inspector.plot_algorithm_convergence(args.graph)

if __name__ == "__main__":
    main() 