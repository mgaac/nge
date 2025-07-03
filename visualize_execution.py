import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import mlx.core as mx
from typing import List, Dict, Tuple, Optional
import math

from generate_dataset import (
    generate_graph_by_category, run_bfs, run_bellman_ford,
    generate_random_graph
)

class GraphExecutionVisualizer:
    """Visualize graphs and their algorithm execution traces."""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        self.colors = {
            'unvisited': '#E8E8E8',
            'start': '#FF6B6B',
            'visited': '#4ECDC4',
            'current': '#FFE66D',
            'edge': '#95A5A6',
            'edge_active': '#E74C3C'
        }
    
    def connection_matrix_to_networkx(self, connection_matrix: mx.array, num_nodes: int) -> nx.Graph:
        """Convert connection matrix to NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        source_idx = connection_matrix[0].tolist()
        target_idx = connection_matrix[1].tolist()
        edge_weights = connection_matrix[2].tolist()
        
        for src, tgt, weight in zip(source_idx, target_idx, edge_weights):
            src, tgt = int(src), int(tgt)
            if src != tgt:  # Skip self-loops for visualization
                if G.has_edge(src, tgt):
                    # If edge already exists, keep the minimum weight
                    G[src][tgt]['weight'] = min(G[src][tgt]['weight'], weight)
                else:
                    G.add_edge(src, tgt, weight=weight)
        
        return G
    
    def get_layout(self, G: nx.Graph, layout_type: str = 'spring') -> Dict:
        """Get node positions using various layout algorithms."""
        if layout_type == 'spring':
            return nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout_type == 'circular':
            return nx.circular_layout(G)
        elif layout_type == 'kamada_kawai':
            return nx.kamada_kawai_layout(G)
        elif layout_type == 'grid':
            # Try to arrange in a grid pattern
            num_nodes = len(G.nodes())
            cols = int(math.sqrt(num_nodes))
            rows = (num_nodes + cols - 1) // cols
            pos = {}
            for i, node in enumerate(G.nodes()):
                row, col = divmod(i, cols)
                pos[node] = (col, rows - row - 1)
            return pos
        else:
            return nx.spring_layout(G, seed=42)
    
    def visualize_graph_structure(self, connection_matrix: mx.array, num_nodes: int, 
                                title: str = "Graph Structure", layout_type: str = 'spring',
                                ax: Optional[plt.Axes] = None) -> Tuple[nx.Graph, Dict]:
        """Visualize the basic graph structure."""
        G = self.connection_matrix_to_networkx(connection_matrix, num_nodes)
        pos = self.get_layout(G, layout_type)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Draw edges
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=self.colors['edge'], 
                              width=[w*2 for w in edge_weights], alpha=0.6)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=self.colors['unvisited'], 
                              node_size=800, edgecolors='black', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')
        
        # Add edge weight labels
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {edge: f'{weight:.1f}' for edge, weight in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        return G, pos
    
    def visualize_bfs_step(self, G: nx.Graph, pos: Dict, bfs_state: mx.array, 
                          start_node: int, step: int, ax: plt.Axes):
        """Visualize a single BFS step."""
        ax.clear()
        
        # Convert state to list
        state = bfs_state.tolist()
        
        # Color nodes based on BFS state
        node_colors = []
        for node in G.nodes():
            if node == start_node:
                node_colors.append(self.colors['start'])
            elif state[node] == 1.0:
                node_colors.append(self.colors['visited'])
            else:
                node_colors.append(self.colors['unvisited'])
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=self.colors['edge'], 
                              width=2, alpha=0.6)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=800, edgecolors='black', linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')
        
        ax.set_title(f'BFS Step {step}\nVisited: {[i for i, v in enumerate(state) if v == 1.0]}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def visualize_bellman_ford_step(self, G: nx.Graph, pos: Dict, bf_distances: mx.array,
                                  bf_predecessors: mx.array, start_node: int, step: int, ax: plt.Axes):
        """Visualize a single Bellman-Ford step."""
        ax.clear()
        
        # Convert to lists and denormalize distances
        distances = bf_distances.tolist()
        predecessors = bf_predecessors.tolist()
        
        # Denormalize distances (multiply by num_nodes)
        num_nodes = len(distances)
        distances_denorm = [d * num_nodes for d in distances]
        
        # Color nodes based on distances
        max_dist = max([d for d in distances_denorm if d < num_nodes])  # Exclude infinity
        node_colors = []
        node_labels = {}
        
        for node in G.nodes():
            if node == start_node:
                node_colors.append(self.colors['start'])
                node_labels[node] = f'{node}\n(0)'
            elif distances_denorm[node] >= num_nodes:  # Infinity
                node_colors.append(self.colors['unvisited'])
                node_labels[node] = f'{node}\n(∞)'
            else:
                # Color based on distance (closer = more blue, farther = more red)
                if max_dist > 0:
                    intensity = distances_denorm[node] / max_dist
                    color = plt.cm.viridis(1 - intensity)
                else:
                    color = self.colors['visited']
                node_colors.append(color)
                node_labels[node] = f'{node}\n({distances_denorm[node]:.1f})'
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=self.colors['edge'], 
                              width=2, alpha=0.6)
        
        # Highlight shortest path edges
        for node in G.nodes():
            if node != start_node and predecessors[node] >= 0:
                pred = predecessors[node]
                if G.has_edge(pred, node):
                    nx.draw_networkx_edges(G, pos, [(pred, node)], ax=ax, 
                                          edge_color=self.colors['edge_active'], 
                                          width=3, alpha=0.8)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=1000, edgecolors='black', linewidths=2)
        
        # Draw labels with distances
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=10, font_weight='bold')
        
        ax.set_title(f'Bellman-Ford Step {step}\nDistances from node {start_node}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def visualize_execution_comparison(self, connection_matrix: mx.array, num_nodes: int,
                                     start_node: int = 0, layout_type: str = 'spring',
                                     save_path: Optional[str] = None):
        """Visualize both BFS and Bellman-Ford execution side by side."""
        # Run algorithms
        bfs_states = run_bfs(num_nodes, connection_matrix, start_node)
        bf_distances, bf_predecessors = run_bellman_ford(num_nodes, connection_matrix, start_node)
        
        # Create graph
        G = self.connection_matrix_to_networkx(connection_matrix, num_nodes)
        pos = self.get_layout(G, layout_type)
        
        # Determine number of steps to show
        max_steps = max(len(bfs_states), len(bf_distances))
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 6 * max_steps))
        
        for step in range(max_steps):
            # BFS visualization
            ax_bfs = plt.subplot(max_steps, 3, step * 3 + 1)
            if step < len(bfs_states):
                self.visualize_bfs_step(G, pos, bfs_states[step], start_node, step, ax_bfs)
            else:
                ax_bfs.text(0.5, 0.5, 'BFS Complete', ha='center', va='center', 
                           transform=ax_bfs.transAxes, fontsize=16)
                ax_bfs.axis('off')
            
            # Bellman-Ford visualization  
            ax_bf = plt.subplot(max_steps, 3, step * 3 + 2)
            if step < len(bf_distances):
                self.visualize_bellman_ford_step(G, pos, bf_distances[step], 
                                               bf_predecessors[step], start_node, step, ax_bf)
            else:
                ax_bf.text(0.5, 0.5, 'Bellman-Ford Complete', ha='center', va='center',
                          transform=ax_bf.transAxes, fontsize=16)
                ax_bf.axis('off')
            
            # Algorithm state comparison
            ax_info = plt.subplot(max_steps, 3, step * 3 + 3)
            ax_info.axis('off')
            
            info_text = f"Step {step}\n\n"
            
            if step < len(bfs_states):
                visited_nodes = [i for i, v in enumerate(bfs_states[step].tolist()) if v == 1.0]
                info_text += f"BFS Visited: {visited_nodes}\n"
            else:
                info_text += "BFS: Complete\n"
            
            if step < len(bf_distances):
                distances = bf_distances[step].tolist()
                distances_denorm = [d * num_nodes for d in distances]
                reachable = [i for i, d in enumerate(distances_denorm) if d < num_nodes]
                info_text += f"BF Reachable: {reachable}\n"
                
                # Show shortest distances
                info_text += "Distances:\n"
                for i, d in enumerate(distances_denorm):
                    if d < num_nodes:
                        info_text += f"  Node {i}: {d:.1f}\n"
                    else:
                        info_text += f"  Node {i}: ∞\n"
            else:
                info_text += "BF: Complete\n"
            
            ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Algorithm Execution Comparison (Start: Node {start_node})', 
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return bfs_states, bf_distances, bf_predecessors
    
    def create_summary_visualization(self, connection_matrix: mx.array, num_nodes: int,
                                   graph_type: str = "Graph", start_node: int = 0,
                                   layout_type: str = 'spring', save_path: Optional[str] = None):
        """Create a summary visualization showing initial graph and final states."""
        # Run algorithms
        bfs_states = run_bfs(num_nodes, connection_matrix, start_node)
        bf_distances, bf_predecessors = run_bellman_ford(num_nodes, connection_matrix, start_node)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Graph structure
        G, pos = self.visualize_graph_structure(connection_matrix, num_nodes, 
                                               f"{graph_type} Structure", layout_type, axes[0,0])
        
        # Final BFS state
        self.visualize_bfs_step(G, pos, bfs_states[-1], start_node, len(bfs_states)-1, axes[0,1])
        
        # Final Bellman-Ford state
        self.visualize_bellman_ford_step(G, pos, bf_distances[-1], bf_predecessors[-1], 
                                       start_node, len(bf_distances)-1, axes[1,0])
        
        # Algorithm statistics
        axes[1,1].axis('off')
        
        # Count edges
        source_idx = connection_matrix[0].tolist()
        target_idx = connection_matrix[1].tolist()
        non_self_edges = [(s, t) for s, t in zip(source_idx, target_idx) if s != t]
        
        stats_text = f"{graph_type} Statistics\n\n"
        stats_text += f"Nodes: {num_nodes}\n"
        stats_text += f"Edges: {len(non_self_edges)}\n"
        stats_text += f"Start Node: {start_node}\n\n"
        stats_text += f"BFS Steps: {len(bfs_states)}\n"
        stats_text += f"BF Steps: {len(bf_distances)}\n\n"
        
        # BFS final state
        final_bfs = bfs_states[-1].tolist()
        visited_count = sum(final_bfs)
        stats_text += f"BFS Reached: {int(visited_count)}/{num_nodes} nodes\n"
        
        # BF final state
        final_bf = bf_distances[-1].tolist()
        final_bf_denorm = [d * num_nodes for d in final_bf]
        reachable_count = sum(1 for d in final_bf_denorm if d < num_nodes)
        stats_text += f"BF Reached: {reachable_count}/{num_nodes} nodes\n"
        
        axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                      fontsize=14, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'{graph_type} Execution Summary', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return bfs_states, bf_distances, bf_predecessors

def demo_visualization():
    """Demonstrate the visualization capabilities."""
    visualizer = GraphExecutionVisualizer()
    
    print("=== Graph Execution Visualization Demo ===\n")
    
    # Test different graph types
    graph_types = [
        ("random", 8),
        ("ladder", 8), 
        ("grid", 9),
        ("tree", 8)
    ]
    
    for graph_type, num_nodes in graph_types:
        print(f"Visualizing {graph_type} graph with {num_nodes} nodes...")
        
        # Generate graph
        connection_matrix = generate_graph_by_category(graph_type, num_nodes)
        
        # Create summary visualization
        visualizer.create_summary_visualization(
            connection_matrix, num_nodes, 
            graph_type=f"{graph_type.title()} Graph",
            start_node=0,
            layout_type='spring' if graph_type != 'grid' else 'grid'
        )
        
        print(f"Completed {graph_type} visualization\n")

def visualize_custom_graph(graph_type: str = "random", num_nodes: int = 10, 
                          start_node: int = 0, show_execution: bool = False):
    """Visualize a custom graph configuration."""
    visualizer = GraphExecutionVisualizer()
    
    print(f"Generating {graph_type} graph with {num_nodes} nodes...")
    connection_matrix = generate_graph_by_category(graph_type, num_nodes)
    
    if show_execution:
        # Show step-by-step execution
        print("Showing step-by-step algorithm execution...")
        return visualizer.visualize_execution_comparison(
            connection_matrix, num_nodes, start_node,
            layout_type='grid' if graph_type == 'grid' else 'spring'
        )
    else:
        # Show summary
        print("Showing execution summary...")
        return visualizer.create_summary_visualization(
            connection_matrix, num_nodes,
            graph_type=f"{graph_type.title()} Graph",
            start_node=start_node,
            layout_type='grid' if graph_type == 'grid' else 'spring'
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize graph algorithm execution')
    parser.add_argument('--graph_type', choices=['random', 'ladder', 'grid', 'tree', 'erdos_renyi', 'barabasi_albert', '4_community', '4_caveman'], 
                       default='random', help='Type of graph to generate')
    parser.add_argument('--num_nodes', type=int, default=8, help='Number of nodes')
    parser.add_argument('--start_node', type=int, default=0, help='Starting node for algorithms')
    parser.add_argument('--execution', action='store_true', help='Show step-by-step execution')
    parser.add_argument('--demo', action='store_true', help='Run demonstration of all graph types')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_visualization()
    else:
        visualize_custom_graph(
            graph_type=args.graph_type,
            num_nodes=args.num_nodes,
            start_node=args.start_node,
            show_execution=args.execution
        ) 