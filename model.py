import mlx.core as mx
import mlx.nn as nn

from enum import Enum

class aggregation_fn(Enum):
    SUM = 1
    AVG = 2
    MIN = 4
    MAX = 5

class mp_layer(nn.Module):
    def __init__(self, embed_dim: int, residual_connections: bool, dropout: float, agg_fn: Enum):
        super().__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embed_dim = embed_dim

        self.residual_connections = residual_connections
        self.agg_fn = agg_fn

        self.source_message_fn = nn.Linear(embed_dim, embed_dim, bias=False)
        self.target_message_fn = nn.Linear(embed_dim, embed_dim, bias=False)

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.update_fn = nn.Linear(embed_dim, embed_dim)
        
        # Use MLX's built-in Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, connection_matrix, node_embeddings):

        num_nodes = node_embeddings.shape[0]

        # Get edge weights and reshape to [num_edges, 1] for broadcasting
        edge_weights = mx.expand_dims(connection_matrix[2], axis=-1)

        source_idx = connection_matrix[self.source_idx].astype(mx.int32)
        target_idx = connection_matrix[self.target_idx].astype(mx.int32)

        source_embeddings = self.source_message_fn(node_embeddings)
        target_embeddings = self.target_message_fn(node_embeddings)

        filtered_source_embeddings = mx.take(source_embeddings, source_idx, axis=0)
        filtered_target_embeddings = mx.take(target_embeddings, target_idx, axis=0)

        message = filtered_source_embeddings + filtered_target_embeddings

        message = message * edge_weights

        message = nn.relu(message)
        
        # Apply dropout to the messages (features), not the graph structure
        message = self.dropout(message)
    
        if (self.agg_fn == aggregation_fn.SUM):
            agg_message = mx.zeros([num_nodes, self.embed_dim])
            agg_message = agg_message.at[target_idx].add(message)

        elif (self.agg_fn == aggregation_fn.AVG):
            agg_message = mx.zeros([num_nodes, self.embed_dim])
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([num_nodes, 1]).at[target_idx].add(1)
            # Fix: Use small positive epsilon instead of large negative number
            agg_message = agg_message / mx.maximum(denominator, 1e-6)

        elif (self.agg_fn == aggregation_fn.MAX):
            # Use more reasonable initialization value and handle nodes with no incoming edges
            agg_message = mx.full([num_nodes, self.embed_dim], -1e3)
            agg_message = agg_message.at[target_idx].maximum(message)
            # For nodes with no incoming messages, reset to zero
            has_incoming = mx.zeros([num_nodes, 1]).at[target_idx].add(1) > 0
            agg_message = mx.where(has_incoming, agg_message, mx.zeros_like(agg_message))

        elif (self.agg_fn == aggregation_fn.MIN):
            # Use more reasonable initialization value and handle nodes with no incoming edges  
            agg_message = mx.full([num_nodes, self.embed_dim], 1e3)
            agg_message = agg_message.at[target_idx].minimum(message)
            # For nodes with no incoming messages, reset to zero
            has_incoming = mx.zeros([num_nodes, 1]).at[target_idx].add(1) > 0
            agg_message = mx.where(has_incoming, agg_message, mx.zeros_like(agg_message))

        agg_message = self.layer_norm(agg_message)

        new_node_embeddings = self.update_fn(agg_message)
        new_node_embeddings = nn.relu(new_node_embeddings)

        if (self.residual_connections):
            new_node_embeddings = new_node_embeddings + node_embeddings

        return new_node_embeddings

class mpnn(nn.Module):
    def __init__(self, embed_dim: int, residual_connections: bool, agg_fn: Enum, num_mp_layers: int, dropout: float = 0.0):
        super(mpnn, self).__init__()

        self.embed_dim = embed_dim
        self.residual_connections = residual_connections
        self.agg_fn = agg_fn

        # Fix: Use proper module list for parameter tracking
        self.mp_layers = [
            mp_layer(embed_dim, residual_connections, dropout, agg_fn)
            for _ in range(num_mp_layers)
        ]

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        assert node_embeddings.shape[1] == self.embed_dim, f'Incorrect node embedding size. Expected {self.embed_dim}, got {node_embeddings.shape[1]}'

        for mp_layer in self.mp_layers:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)

        return node_embeddings
    
class bfs_decoder(nn.Module):
    def __init__(self, embed_dim: int, expansion: float = 4.0):
        super(bfs_decoder, self).__init__()

        self.embed_dim = embed_dim
        hidden_dim = int(embed_dim * expansion)
        self.bfs_state_outputs = [nn.Linear(embed_dim, hidden_dim), nn.Linear(hidden_dim, 1)]

    def __call__(self, data):
        node_embeddings, _ = data

        # BFS state prediction
        bfs_state_predictions = nn.relu(self.bfs_state_outputs[0](node_embeddings))
        bfs_state_predictions = self.bfs_state_outputs[1](bfs_state_predictions)

        bfs_state_predictions = mx.sigmoid(bfs_state_predictions.squeeze())

        return bfs_state_predictions

class bf_decoder(nn.Module):
    def __init__(self, embed_dim: int, expansion: float = 4.0):
        super(bf_decoder, self).__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embed_dim = embed_dim
        
        # Simplified Bellman-Ford distance head with proper initialization
        # Use a single linear layer with proper initialization and output constraint
        hidden_dim = int(embed_dim * expansion)
        self.bf_distance_outputs = [nn.Linear(embed_dim, hidden_dim), nn.Linear(hidden_dim, 1)]
        
        # Improved predecessor prediction - use edge-based approach
        edge_hidden_dim = int(embed_dim * expansion)
        self.bf_predecessor_head = nn.Linear(2 * embed_dim, edge_hidden_dim)
        self.bf_predecessor_output = nn.Linear(edge_hidden_dim, 1)

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        num_nodes = node_embeddings.shape[0]

        source_idx = connection_matrix[self.source_idx].astype(mx.int32)
        target_idx = connection_matrix[self.target_idx].astype(mx.int32)

        # Simplified distance prediction with stability improvements
        bf_distance_predictions = nn.relu(self.bf_distance_outputs[0](node_embeddings))
        bf_distance_predictions = self.bf_distance_outputs[1](bf_distance_predictions)
        
        # Apply ReLU to ensure non-negative distances (shortest paths can't be negative)
        # Add small epsilon for numerical stability
        bf_distance_predictions = nn.relu(bf_distance_predictions) + 1e-6
        bf_distance_predictions = bf_distance_predictions.squeeze()

        # Bellman-Ford predecessor predictions with efficient neighborhood-aware selection
        source_embeddings = mx.take(node_embeddings, source_idx, axis=0)
        target_embeddings = mx.take(node_embeddings, target_idx, axis=0)
        
        # Process edge features
        concatenated_embeddings = mx.concat([source_embeddings, target_embeddings], axis=1)
        edge_features = nn.relu(self.bf_predecessor_head(concatenated_embeddings))
        edge_scores = self.bf_predecessor_output(edge_features).squeeze()

        # Create adjacency-aware predecessor predictions
        # Initialize with very negative values to ensure invalid connections are never selected
        bf_predecessor_predictions = mx.full([num_nodes, num_nodes], -1e6)
        
        # Only populate scores for actual edges in the graph using in-place assignment
        bf_predecessor_predictions[target_idx, source_idx] = edge_scores
        
        # Apply softmax per target node (each row represents a target node's choices)
        bf_predecessor_predictions = nn.softmax(bf_predecessor_predictions, axis=1)
        
        return bf_distance_predictions, bf_predecessor_predictions
    
class nge(nn.Module):
    def __init__(self, embed_dim: int, residual_connections: bool, agg_fn: Enum, num_mp_layers: int, dropout: float = 0.0, expansion: float = 4.0):
        super(nge, self).__init__()


        # Separate encoders for each algorithm
        # 3 is the number of additional features (bfs_state, bf_distance, bf_predecessor)
        self.encoder = nn.Linear(embed_dim + 3, embed_dim)

        # Separate decoders for each algorithm
        self.bfs_decoder = bfs_decoder(embed_dim, expansion)
        self.bf_decoder = bf_decoder(embed_dim, expansion)

        # Separate termination heads for each algorithm 
        self.bfs_termination = nn.Linear(embed_dim, 1, bias=True)
        self.bf_termination = nn.Linear(embed_dim, 1, bias=True)
    
        # Shared processor
        self.processor = mpnn(embed_dim, residual_connections, agg_fn, num_mp_layers, dropout)

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        # Process through both algorithm-specific encoders
        encoded_embeddings = self.encoder(node_embeddings)
        
        # Process through shared MPNN
        processed_embeddings = self.processor((encoded_embeddings, connection_matrix))
        
        # Process through algorithm-specific decoders
        bfs_output = self.bfs_decoder((processed_embeddings, connection_matrix))
        bf_output = self.bf_decoder((processed_embeddings, connection_matrix))

        avg_embeddings = mx.mean(processed_embeddings, axis=0)

        bfs_termination_prob = mx.sigmoid(self.bfs_termination(avg_embeddings).squeeze())

        bf_termination_prob = mx.sigmoid(self.bf_termination(avg_embeddings).squeeze())
        
        termination_probs = {
            'bfs': bfs_termination_prob,
            'bf': bf_termination_prob
        }

        return bfs_output, bf_output, termination_probs, processed_embeddings
    