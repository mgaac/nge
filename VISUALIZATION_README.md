# Graph Algorithm Execution Visualization Tool

This visualization tool provides comprehensive visual analysis of BFS and Bellman-Ford algorithm execution on various graph structures. It helps understand how these algorithms propagate through different graph topologies and visualizes their step-by-step execution.

## Features

### 🎨 Visualization Capabilities
- **Graph Structure Display**: Visualize the basic graph structure with nodes, edges, and weights
- **Algorithm Execution Tracking**: Step-by-step visualization of BFS and Bellman-Ford execution
- **Side-by-Side Comparison**: Compare both algorithms running simultaneously
- **Multiple Layout Options**: Spring, circular, grid, and Kamada-Kawai layouts
- **Color-Coded States**: Intuitive color coding for different node states
- **Distance Visualization**: Visual representation of shortest path distances
- **Path Highlighting**: Highlight shortest path edges in Bellman-Ford

### 📊 Supported Graph Types
- Random graphs
- Ladder graphs
- 2D grid graphs  
- Trees (via Prüfer sequences)
- Erdős-Rényi graphs
- Barabási-Albert graphs
- 4-community graphs
- 4-caveman graphs

### 🎯 Visual Elements

#### Node Colors
- **Red**: Start node
- **Gray**: Unvisited nodes
- **Teal**: Visited nodes (BFS)
- **Gradient (Viridis)**: Distance-based coloring (Bellman-Ford)

#### Edge Visualization
- **Gray**: Regular edges
- **Red**: Active shortest path edges (Bellman-Ford)
- **Thickness**: Proportional to edge weights

## Usage

### Basic Python Usage

```python
from visualize_execution import GraphExecutionVisualizer, visualize_custom_graph

# Quick visualization
visualize_custom_graph('ladder', num_nodes=8, start_node=0)

# Step-by-step execution
visualize_custom_graph('grid', num_nodes=9, show_execution=True)
```

### Advanced Usage

```python
from visualize_execution import GraphExecutionVisualizer
from generate_dataset import generate_graph_by_category

# Create visualizer
visualizer = GraphExecutionVisualizer()

# Generate a graph
connection_matrix = generate_graph_by_category('tree', 10)

# Create summary visualization
visualizer.create_summary_visualization(
    connection_matrix, 10, 
    graph_type="Tree Graph",
    start_node=0,
    save_path='tree_summary.png'
)

# Step-by-step execution comparison
visualizer.visualize_execution_comparison(
    connection_matrix, 10, 
    start_node=0,
    save_path='tree_execution.png'
)
```

### Command Line Usage

```bash
# Basic visualization
python visualize_execution.py --graph_type grid --num_nodes 9

# Show step-by-step execution
python visualize_execution.py --graph_type tree --execution

# Run demonstration of all graph types
python visualize_execution.py --demo

# Custom configuration
python visualize_execution.py --graph_type barabasi_albert --num_nodes 15 --start_node 2
```

### Command Line Options

- `--graph_type`: Type of graph to generate (random, ladder, grid, tree, etc.)
- `--num_nodes`: Number of nodes in the graph
- `--start_node`: Starting node for algorithm execution
- `--execution`: Show step-by-step execution (vs. summary)
- `--demo`: Run demonstration of all graph types

## Visualization Types

### 1. Graph Structure Visualization
Shows the basic graph structure with:
- Node positions using various layout algorithms
- Edge connections with weights
- Clean, readable node labels

### 2. Summary Visualization (2x2 grid)
- **Top-left**: Original graph structure
- **Top-right**: Final BFS state
- **Bottom-left**: Final Bellman-Ford state
- **Bottom-right**: Algorithm statistics and summary

### 3. Step-by-Step Execution Comparison
- **Column 1**: BFS execution at each step
- **Column 2**: Bellman-Ford execution at each step  
- **Column 3**: Detailed state information and statistics

## Algorithm Visualization Details

### BFS (Breadth-First Search)
- Shows which nodes are visited at each step
- Red node indicates the start
- Teal nodes indicate visited nodes
- Gray nodes are unvisited

### Bellman-Ford (Shortest Paths)
- Displays shortest distances from start node
- Color gradient indicates distance (closer = darker)
- Shows predecessor relationships
- Highlights shortest path edges
- Infinity (∞) for unreachable nodes

## Layout Options

### Spring Layout (`'spring'`)
- Physics-based layout with attractive/repulsive forces
- Good for general graphs
- Default option for most graph types

### Grid Layout (`'grid'`)
- Arranges nodes in a rectangular grid
- Perfect for grid graphs
- Automatically used for grid graph type

### Circular Layout (`'circular'`)
- Nodes arranged in a circle
- Good for small graphs
- Emphasizes symmetry

### Kamada-Kawai Layout (`'kamada_kawai'`)
- Energy-based layout algorithm
- Good for complex networks
- Minimizes edge crossing

## Examples and Use Cases

### Research and Education
- **Algorithm Understanding**: Visualize how BFS and Bellman-Ford propagate
- **Graph Theory**: Understand different graph topologies
- **Debugging**: Identify issues in algorithm implementations
- **Presentations**: Create clear visualizations for papers/talks

### Development and Testing
- **Algorithm Validation**: Verify correct algorithm behavior
- **Dataset Analysis**: Understand your training data
- **Performance Analysis**: Compare algorithm execution across graph types
- **Debug Neural Network Training**: Visualize target states for learning

### Interactive Exploration
```python
# Explore different graph structures
for graph_type in ['ladder', 'grid', 'tree', 'erdos_renyi']:
    visualize_custom_graph(graph_type, num_nodes=10, start_node=0)

# Compare algorithm behavior on complex graphs
visualize_custom_graph('4_community', num_nodes=20, show_execution=True)
```

## Customization

### Color Schemes
Modify the color scheme by updating the `colors` dictionary:

```python
visualizer = GraphExecutionVisualizer()
visualizer.colors.update({
    'visited': '#FF9999',  # Light red for visited
    'start': '#0066CC',    # Blue for start
    'unvisited': '#DDDDDD' # Light gray for unvisited
})
```

### Figure Size
```python
visualizer = GraphExecutionVisualizer(figsize=(20, 12))
```

### Custom Layouts
Add custom node positioning:

```python
# Custom positions
pos = {0: (0, 0), 1: (1, 0), 2: (0.5, 1)}
# Use in visualization functions
```

## Output Files

All visualization functions support saving to files:

```python
# Save high-resolution images
visualizer.create_summary_visualization(
    connection_matrix, num_nodes,
    save_path='my_graph.png'  # Supports PNG, PDF, SVG
)
```

## Performance Considerations

- **Large Graphs**: For graphs with >25 nodes, consider using summary view instead of step-by-step
- **Many Steps**: Step-by-step visualization can be large for graphs with many algorithm steps
- **Layout Calculation**: Spring layout can be slow for large graphs; use grid or circular for speed
- **File Size**: High DPI saves create large files; adjust DPI as needed

## Dependencies

Required packages:
- `matplotlib`: For plotting and visualization
- `networkx`: For graph layout algorithms
- `numpy`: For array operations
- `mlx.core`: For MLX array operations

Install with:
```bash
pip install matplotlib networkx numpy
```

## Tips and Best Practices

1. **Start Small**: Begin with small graphs (8-10 nodes) to understand the visualization
2. **Choose Appropriate Layout**: Use grid layout for grid graphs, spring for others
3. **Save Important Visualizations**: Use `save_path` parameter to keep useful visualizations
4. **Compare Algorithms**: Use step-by-step view to see how BFS and Bellman-Ford differ
5. **Experiment with Start Nodes**: Try different start nodes to see how it affects execution
6. **Use Summary for Overview**: Summary view is better for understanding final states
7. **Step-by-step for Learning**: Use detailed execution view to understand algorithm mechanics

## Troubleshooting

### Common Issues

**Issue**: "FigureCanvasAgg is non-interactive"
**Solution**: This warning appears in non-GUI environments. Use `save_path` parameter to save files instead of displaying.

**Issue**: Layout looks cluttered
**Solution**: Try different layout options or reduce the number of nodes.

**Issue**: Visualization takes too long
**Solution**: Use simpler layouts (circular, grid) for large graphs.

**Issue**: Can't see edge weights
**Solution**: Ensure your graph has reasonable edge weights and try zooming in.

## Integration with Neural Algorithm Learning

This visualization tool integrates seamlessly with the algorithm learning pipeline:

```python
# Visualize training data
from generate_dataset import load_extended_dataset

dataset = load_extended_dataset('extended_dataset')
sample_graph = dataset['train'][0]

# Visualize a training example
visualizer.create_summary_visualization(
    sample_graph['connection_matrix'],
    sample_graph['num_nodes'],
    graph_type=f"{sample_graph['category']} (Training Sample)",
    start_node=sample_graph['start_node']
)
```

This helps you understand what your neural network is learning to predict and can identify potential issues in the training data. 