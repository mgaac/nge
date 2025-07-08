import mlx.core as mx

# Global flag to control printing
VERBOSE = False

def set_verbose(verbose):
    """Set global verbose flag"""
    global VERBOSE
    VERBOSE = verbose

def log_graph_execution_details(graph_data, num_steps, model_config):
    """Log graph execution details"""
    if not VERBOSE:
        return
    
    num_nodes = graph_data['num_nodes']
    num_bf_steps = len(graph_data['bf_distance_targets'])
    num_bfs_steps = len(graph_data['bfs_state_targets'])
    
    print(f"\n=== Graph Execution Details ===")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of BF steps: {num_bf_steps}")
    print(f"Number of BFS steps: {num_bfs_steps}")
    print(f"Total steps: {num_steps}")
    print(f"Hidden state dimension: {model_config['embedding_dim']}")


def log_step_details(step_idx, bf_sample_exists, bfs_sample_exists):
    """Log step details"""
    if not VERBOSE:
        return
    
    print(f"\n--- Step {step_idx} ---")
    
    if not (bf_sample_exists or bfs_sample_exists):
        print("Skipping step - no valid samples")


def log_model_inputs(input_embeddings, edge_matrix, previous_step_hidden_states, node_algo_features, 
                    true_bfs_state, true_distance_bf, true_predecessor_bf, num_nodes):
    """Log model inputs"""
    if not VERBOSE:
        return
    
    # Calculate both total norm and average per-node norm for better insight
    total_norm = float(mx.linalg.norm(previous_step_hidden_states))
    # Calculate norm for each node individually, then average
    node_norms = mx.linalg.norm(previous_step_hidden_states, axis=1)
    avg_node_norm = float(mx.mean(node_norms))
    
    print(f"Model Inputs:")
    print(f"  - Input embeddings shape: {input_embeddings.shape}")
    print(f"  - Edge matrix shape: {edge_matrix.shape}")
    print(f"  - Hidden states total norm: {total_norm:.4f}")
    print(f"  - Hidden states avg per-node norm: {avg_node_norm:.4f}")
    print(f"  - Node algo features shape: {node_algo_features.shape}")
    print(f"  - Node algo features (first 3 nodes):")
    for j in range(min(3, num_nodes)):
        print(f"    Node {j}: BFS={float(true_bfs_state[j]):.2f}, Dist={float(true_distance_bf[j]):.2f}, Pred={float(true_predecessor_bf[j]):.2f}")


def log_model_outputs(bfs_output, bf_output, termination_probs, processed_embeddings, num_nodes):
    """Log model outputs"""
    if not VERBOSE:
        return
    
    bf_distance_predictions, bf_predecessor_predictions = bf_output
    
    # Calculate both total norm and average per-node norm for better insight
    total_norm = float(mx.linalg.norm(processed_embeddings))
    # Calculate norm for each node individually, then average
    node_norms = mx.linalg.norm(processed_embeddings, axis=1)
    avg_node_norm = float(mx.mean(node_norms))
    
    print(f"Model Outputs:")
    print(f"  - BFS output shape: {bfs_output.shape}")
    print(f"  - BF distance predictions shape: {bf_distance_predictions.shape}")
    print(f"  - BF predecessor predictions shape: {bf_predecessor_predictions.shape}")
    print(f"  - Processed embeddings total norm: {total_norm:.4f}")
    print(f"  - Processed embeddings avg per-node norm: {avg_node_norm:.4f}")
    print(f"  - Termination probs: BF={float(termination_probs['bf']):.4f}, BFS={float(termination_probs['bfs']):.4f}")
    
    # Print first few values of outputs
    print(f"  - BFS output (first 3 nodes): {[float(bfs_output[j]) for j in range(min(3, num_nodes))]}")
    print(f"  - BF distance predictions (first 3 nodes): {[float(bf_distance_predictions[j]) for j in range(min(3, num_nodes))]}")
    
    # Handle BF predecessor predictions which might be logits over all possible predecessors
    try:
        if bf_predecessor_predictions.ndim == 2 and bf_predecessor_predictions.shape[1] > 1:
            # It's logits, take argmax to get predicted predecessor
            pred_predecessors = mx.argmax(bf_predecessor_predictions, axis=1)
            print(f"  - BF predecessor predictions (first 3 nodes): {[int(pred_predecessors[j]) for j in range(min(3, num_nodes))]}")
        else:
            # It's already scalar predictions
            print(f"  - BF predecessor predictions (first 3 nodes): {[float(bf_predecessor_predictions[j]) for j in range(min(3, num_nodes))]}")
    except Exception as e:
        print(f"  - BF predecessor predictions shape: {bf_predecessor_predictions.shape}, error printing values: {e}")


def log_targets(target_bfs_state, target_distance_bf, target_predecessor_bf, termination_targets, num_nodes):
    """Log targets"""
    if not VERBOSE:
        return
    
    print(f"Targets:")
    print(f"  - Target BFS state shape: {target_bfs_state.shape}")
    print(f"  - Target BF distance shape: {target_distance_bf.shape}")
    print(f"  - Target BF predecessor shape: {target_predecessor_bf.shape}")
    print(f"  - Termination targets: BF={float(termination_targets['bf']):.0f}, BFS={float(termination_targets['bfs']):.0f}")
    print(f"  - Target BFS state (first 3 nodes): {[float(target_bfs_state[j]) for j in range(min(3, num_nodes))]}")
    print(f"  - Target BF distance (first 3 nodes): {[float(target_distance_bf[j]) for j in range(min(3, num_nodes))]}")
    print(f"  - Target BF predecessor (first 3 nodes): {[float(target_predecessor_bf[j]) for j in range(min(3, num_nodes))]}")


def log_losses(bf_distance_loss, bf_predecessor_loss, bfs_state_loss, bf_termination_loss, bfs_termination_loss, total_step_loss):
    """Log losses"""
    if not VERBOSE:
        return
    
    print(f"Losses:")
    print(f"  - BF distance loss: {float(bf_distance_loss):.6f}")
    print(f"  - BF predecessor loss: {float(bf_predecessor_loss):.6f}")
    print(f"  - BFS state loss: {float(bfs_state_loss):.6f}")
    print(f"  - BF termination loss: {float(bf_termination_loss):.6f}")
    print(f"  - BFS termination loss: {float(bfs_termination_loss):.6f}")
    print(f"  - Total step loss: {float(total_step_loss):.6f}")


def log_execution_summary(average_loss):
    """Log execution summary"""
    if not VERBOSE:
        return
    
    print(f"\n=== Execution Summary ===")
    print(f"Average loss: {float(average_loss):.6f}")


def log_epoch_loss(epoch, avg_epoch_loss, component_losses=None):
    """Log epoch loss with optional component breakdown"""
    if component_losses is None:
        print(f"Epoch {epoch} loss: {avg_epoch_loss}")
    else:
        print(f"\nEpoch {epoch} total loss: {avg_epoch_loss:.6f}")
        print(f"  - BF distance loss: {component_losses['bf_distance']:.6f}")
        print(f"  - BF predecessor loss: {component_losses['bf_predecessor']:.6f}")
        print(f"  - BFS state loss: {component_losses['bfs_state']:.6f}")
        print(f"  - BF termination loss: {component_losses['bf_termination']:.6f}")
        print(f"  - BFS termination loss: {component_losses['bfs_termination']:.6f}") 

import wandb
from typing import Dict, Optional, Any

class WandbLogger:
    """Clean WandB logger that isolates tracking from ML logic"""
    
    def __init__(self, project_name: str = "nge-training", enabled: bool = True):
        self.enabled = enabled
        self.run = None
        if self.enabled:
            try:
                self.run = wandb.init(project=project_name, resume="allow")
            except Exception as e:
                print(f"Warning: Failed to initialize WandB: {e}")
                self.enabled = False
    
    def log_config(self, config: Dict[str, Any]):
        """Log training configuration"""
        if self.enabled and self.run:
            wandb.config.update(config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics"""
        if self.enabled and self.run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch-level metrics"""
        if self.enabled and self.run:
            epoch_metrics = {f"epoch_{k}": v for k, v in metrics.items()}
            epoch_metrics["epoch"] = epoch
            wandb.log(epoch_metrics, step=epoch)
    
    def log_graph_metrics(self, graph_idx: int, metrics: Dict[str, float], step: int):
        """Log per-graph metrics for first graph monitoring"""
        if self.enabled and self.run and graph_idx == 0:  # Only log first graph
            graph_metrics = {f"first_graph/{k}": v for k, v in metrics.items()}
            wandb.log(graph_metrics, step=step)
    
    def finish(self):
        """Clean shutdown"""
        if self.enabled and self.run:
            wandb.finish()

# Global logger instance
_wandb_logger = None

def init_wandb(project_name: str = "nge-training", enabled: bool = True) -> WandbLogger:
    """Initialize global WandB logger"""
    global _wandb_logger
    _wandb_logger = WandbLogger(project_name, enabled)
    return _wandb_logger

def log_wandb_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """Log metrics if WandB is enabled"""
    if _wandb_logger:
        _wandb_logger.log_metrics(metrics, step)

def log_wandb_epoch(epoch: int, metrics: Dict[str, float]):
    """Log epoch metrics if WandB is enabled"""
    if _wandb_logger:
        _wandb_logger.log_epoch(epoch, metrics)

def log_wandb_graph(graph_idx: int, metrics: Dict[str, float], step: int):
    """Log graph metrics if WandB is enabled"""
    if _wandb_logger:
        _wandb_logger.log_graph_metrics(graph_idx, metrics, step)

def finish_wandb():
    """Finish WandB logging"""
    if _wandb_logger:
        _wandb_logger.finish() 