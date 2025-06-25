
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from train import model, step_loss
from generate_dataset import generate_graph_dataset, load_dataset

def train_model(dataset, num_epochs=10, learning_rate=0.001):
    """Train the model on the generated dataset."""
    
    # Create optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_steps = 0
        
        for graph_idx, graph_data in enumerate(dataset):
            # Get graph structure
            node_embeddings = graph_data['node_embeddings']
            connection_matrix = graph_data['connection_matrix']
            
            # Process each step in the algorithm execution
            for step_idx, step_data in enumerate(graph_data['steps']):
                # Get targets for this step
                bfs_state_targets = step_data['bfs_state_targets']
                bf_distance_targets = step_data['bf_distance_targets']
                bf_predecessor_targets = step_data['bf_predecessor_targets']
                termination_targets = step_data['termination_targets']
                
                # Forward pass
                bfs_output, bf_output, termination_probs = model((node_embeddings, connection_matrix))
                
                # Calculate loss
                loss = step_loss(
                    bf_output, 
                    bfs_output, 
                    bfs_state_targets, 
                    bf_distance_targets, 
                    bf_predecessor_targets, 
                    termination_probs,
                    termination_targets
                )
                
                # Compute gradients
                loss_value, grads = mx.value_and_grad(model, loss)(model)
                
                # Update parameters
                optimizer.update(model, grads)
                
                total_loss += loss_value
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def demonstrate_data_format():
    """Show the format of the generated data."""
    print("Generating a small dataset to demonstrate format...")
    dataset = generate_graph_dataset(num_graphs=2, min_nodes=5, max_nodes=8)
    
    print("\n=== Dataset Structure ===")
    print(f"Number of graphs: {len(dataset)}")
    
    for i, graph in enumerate(dataset):
        print(f"\nGraph {i}:")
        print(f"  - Node embeddings shape: {graph['node_embeddings'].shape}")
        print(f"  - Connection matrix shape: {graph['connection_matrix'].shape}")
        print(f"  - Number of algorithm steps: {len(graph['steps'])}")
        
        if graph['steps']:
            step = graph['steps'][0]
            print(f"\n  First step targets:")
            print(f"    - BFS states shape: {step['bfs_state_targets'].shape}")
            print(f"    - BF distances shape: {step['bf_distance_targets'].shape}")
            print(f"    - BF predecessors shape: {step['bf_predecessor_targets'].shape}")
            print(f"    - BFS termination: {step['termination_targets']['bfs']}")
            print(f"    - BF termination: {step['termination_targets']['bf']}")

if __name__ == "__main__":
    # Demonstrate data format
    demonstrate_data_format()
    
    print("\n" + "="*50 + "\n")
    
    # Generate training dataset
    print("Generating training dataset...")
    train_dataset = generate_graph_dataset(num_graphs=50, min_nodes=5, max_nodes=15)
    
    # Train the model
    print("\nStarting training...")
    train_model(train_dataset, num_epochs=5, learning_rate=0.001) 