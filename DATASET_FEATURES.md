# Expanded Graph Dataset Generation Features

## Overview

The `generate_dataset.py` script has been significantly expanded to support both the original basic dataset generation and seven new structural graph categories with proper dataset splits and training utilities.

## New Features

### 1. Seven Structural Graph Categories

All graphs include self-loops and real-valued edge weights drawn uniformly from [0.2, 1.0]:

- **Ladder graphs**: Linear ladder structure with two rails connected by rungs
- **2D grid graphs**: Regular grid topology 
- **Trees**: Generated using uniformly random Prüfer sequences
- **Erdős–Rényi graphs**: Random graphs with p = min(log₂|V|/|V|, 0.5)
- **Barabási–Albert graphs**: Preferential attachment graphs (4-5 edges per new node)
- **4-Community graphs**: Four disjoint Erdős–Rényi subgraphs (p=0.7) with inter-community edges (p=0.01)
- **4-Caveman graphs**: Start from cliques, delete intra-clique edges (p=0.7), add 0.025|V| random shortcuts

### 2. Proper Dataset Splits

Following the specification:

| Split      | Graphs per category | Node counts        |
|------------|--------------------|--------------------|
| Training   | 100                | 20                 |
| Validation | 5                  | 20                 |
| Testing    | 5                  | 20, 50, 100       |

### 3. Dataset Generation Options

Three generation modes:
- `basic`: Original random graph functionality only
- `extended`: Seven structural categories only  
- `both`: Random graphs + seven structural categories (8 total)

### 4. Training Utilities

- **Shuffling**: `shuffle_dataset()` with optional seeding
- **Statistics**: `get_dataset_statistics()` for comprehensive dataset analysis
- **Data loaders**: `create_data_loaders()` for batch iteration
- **Split management**: Separate save/load functions for train/val/test splits

## Usage Examples

### Command Line Interface

```bash
# Generate basic dataset (original functionality)
python generate_dataset.py --dataset_type basic --num_graphs 100

# Generate extended dataset with all 7 categories
python generate_dataset.py --dataset_type extended --seed 42

# Generate combined dataset (basic + extended)
python generate_dataset.py --dataset_type both --output_prefix my_dataset

# Full parameter control
python generate_dataset.py \
    --dataset_type both \
    --embedding_dim 256 \
    --output_prefix large_dataset \
    --seed 42
```

### Programmatic Usage

```python
from generate_dataset import *

# Generate extended dataset
dataset = generate_extended_dataset(dataset_type="both", embedding_dim=128)

# Get statistics
stats = get_dataset_statistics(dataset)
print(f"Training graphs: {stats['train']['total_graphs']}")

# Create data loaders
loaders = create_data_loaders(dataset, batch_size=32)

# Iterate through training batches
for batch in loaders['train']():
    for graph in batch:
        print(f"Graph: {graph['category']}, Nodes: {graph['num_nodes']}")
```

### Loading Datasets

```python
# Load basic dataset
basic_data = load_dataset('graph_dataset.npz')

# Load extended dataset with splits
extended_data = load_extended_dataset('test_extended')

# Access specific splits
train_graphs = extended_data['train']
val_graphs = extended_data['val'] 
test_graphs = extended_data['test']
```

## Algorithm Execution Traces

Each graph includes complete execution traces for:

- **BFS**: Binary node states (visited/unvisited) for each timestep
- **Bellman-Ford**: Distance estimates and predecessor tracking for each timestep
- **Termination detection**: Binary indicators for when each algorithm converges

### Data Structure

```python
graph_data = {
    'node_embeddings': mx.array,      # [num_nodes, embedding_dim]
    'connection_matrix': mx.array,    # [3, num_edges] - [src, tgt, weight]
    'category': str,                  # Graph category
    'split': str,                     # 'train', 'val', or 'test'
    'num_nodes': int,                 # Number of nodes
    'start_node': int,                # Algorithm start node
    'steps': [                        # Execution trace
        {
            'bfs_state_targets': mx.array,        # Binary node states
            'bf_distance_targets': mx.array,      # Distance estimates
            'bf_predecessor_targets': mx.array,   # Predecessor tracking
            'termination_targets': {
                'bfs': mx.array,      # BFS termination indicator
                'bf': mx.array        # Bellman-Ford termination indicator
            }
        },
        # ... more steps
    ]
}
```

## File Organization

- **Basic dataset**: Single file `graph_dataset.npz`
- **Extended dataset**: Split files `{prefix}_{split}.npz`
  - `{prefix}_train.npz`
  - `{prefix}_val.npz` 
  - `{prefix}_test.npz`

## Key Improvements

1. **Edge weights**: All graphs now use real-valued weights from [0.2, 1.0]
2. **Algorithm correctness**: BFS and Bellman-Ford properly handle weighted edges
3. **Scalability**: Proper train/val/test splits for machine learning
4. **Flexibility**: Choose between basic, extended, or combined datasets
5. **Reproducibility**: Seed control for consistent generation
6. **Training utilities**: Built-in shuffling, batching, and statistics

## Backward Compatibility

The original `generate_graph_dataset()` function remains unchanged, ensuring existing code continues to work while providing access to all new features through the expanded API. 