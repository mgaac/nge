# Dataset Inspector Usage Guide

The dataset inspector (`inspect_dataset.py`) is a comprehensive tool for exploring and analyzing your graph dataset. It provides multiple ways to visualize and understand the algorithm execution traces.

## Quick Start

```bash
# Show dataset summary
python inspect_dataset.py --summary

# Inspect a specific graph
python inspect_dataset.py --graph 0

# Run in interactive mode
python inspect_dataset.py --interactive

# Generate a test dataset and inspect it
python inspect_dataset.py --generate --interactive
```

## Command Line Options

- `--file <filename>`: Specify dataset file (default: `graph_dataset.npz`)
- `--generate`: Generate a new test dataset instead of loading from file
- `--interactive`: Run in interactive mode with commands
- `--graph <index>`: Inspect specific graph by index
- `--summary`: Show only dataset summary statistics

## Interactive Commands

When running with `--interactive`, you can use these commands:

### Basic Analysis
- `summary` - Show dataset overview with statistics
- `inspect <graph_idx>` - Detailed inspection of a specific graph
- `compare <idx1,idx2,idx3>` - Compare multiple graphs side by side

### Algorithm Traces
- `trace <graph_idx>` - Show both BFS and Bellman-Ford execution steps
- `trace <graph_idx> bfs` - Show only BFS execution trace
- `trace <graph_idx> bf` - Show only Bellman-Ford execution trace

### Raw Data Inspection
- `raw <graph_idx> <step_idx>` - Show raw input/target vectors for specific step
- `format <graph_idx> <step_idx>` - Show exact model input/output format
- `states <graph_idx> <step_idx>` - Visualize algorithm states on graph structure

### Visualizations
- `visualize <graph_idx>` - Show graph structure (requires matplotlib)
- `plot <graph_idx>` - Plot algorithm convergence over time

### Control
- `quit` - Exit interactive mode

## Output Explanation

### Dataset Summary
- **Node counts**: Range and average number of nodes per graph
- **Edge counts**: Range and average number of edges per graph  
- **Execution steps**: Range and average algorithm execution steps
- **Embedding dimension**: Size of node feature vectors

### Graph Inspection
- **Edge structure**: Shows source→target connections with weights
- **Algorithm trace**: Step-by-step progress of both algorithms
  - `BFS visited`: Number of nodes visited by BFS at each step
  - `BF reachable`: Number of nodes reachable by Bellman-Ford
  - `Termination flags`: When each algorithm finished (0=running, 1=done)

### Algorithm Traces
- **BFS**: Shows which nodes are visited at each step
- **Bellman-Ford**: Shows distances from source and predecessor nodes

### Graph Comparison
Tabular view comparing:
- Graph index, node count, edge count
- Total execution steps
- When BFS terminated (BFS_Steps)
- When Bellman-Ford terminated (BF_Steps)

## Understanding the Data

### Algorithm Execution
1. Both algorithms start from a random source node
2. They execute step-by-step until completion
3. **Padding**: Once an algorithm finishes, its output is repeated until both are done
4. **Termination targets**: Binary flags indicating when each algorithm actually finished

### Key Insights
- **BFS**: Explores nodes level by level, tracks visited states
- **Bellman-Ford**: Computes shortest distances, tracks predecessors for path reconstruction
- **Self-loops**: All nodes have self-connections but are ignored during algorithm execution
- **Edge weights**: All set to 1.0 for unweighted shortest paths

## Example Session

```bash
python inspect_dataset.py --interactive

> summary                    # Overview of entire dataset
> inspect 0                  # Detailed look at first graph
> trace 0 bfs               # See BFS execution step-by-step
> trace 0 bf                # See Bellman-Ford execution  
> raw 0 1                   # Raw vectors at step 1
> format 0 1                # Model input/output format
> states 0 1                # Visualize algorithm states
> compare 0,1,2,3           # Compare first 4 graphs
> visualize 0               # Show graph structure
> plot 0                    # Plot convergence curves
> quit
```

## Mathematical Formulation Compliance

The updated dataset generator now follows the exact mathematical formulation:

### Algorithm Initialization
- **BFS**: x_i^(1) = {1 if i = s, 0 if i ≠ s} where s is the source node
- **Bellman-Ford**: x_i^(1) = {0 if i = s, +∞ if i ≠ s} where s is the source node

### Algorithm Execution  
- **BFS**: x_i^(t+1) = {1 if x_i^(t) = 1 OR ∃j:(j,i) ∈ E ∧ x_j^(t) = 1, 0 otherwise}
- **Bellman-Ford**: x_i^(t+1) = min(x_i^(t), min_{(j,i)∈E} x_j^(t) + e_ji^(t))

### Output Format
- **BFS**: y_i^(t) = x_i^(t+1) (visited states)
- **Bellman-Ford**: p_i^(t) = argmin_{j:(j,i)∈E} x_j^(t) + e_ji^(t) (predecessors)

This tool is essential for:
- **Debugging**: Verify algorithm implementations follow mathematical formulation
- **Analysis**: Understand dataset characteristics and complexity  
- **Validation**: Ensure your model receives properly formatted targets
- **Exploration**: Find interesting examples for testing
- **Raw Data Inspection**: View exact vectors and formats expected by your model 