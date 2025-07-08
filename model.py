import mlx.core as mx
import mlx.nn as nn

from enum import Enum

class aggregation_fn(Enum):
    SUM = 1
    AVG = 2
    MIN = 4
    MAX = 5

class mp_layer(nn.Module):
    def __init__(self, embedding_dim: int, skip_connections: bool, aggregation_fn: Enum):
        super().__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embedding_dim = embedding_dim

        self.skip_connections = skip_connections
        self.aggregation_fn = aggregation_fn

        self.source_message_fn = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.target_message_fn = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.update_fn = nn.Linear(embedding_dim, embedding_dim)

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


        if (self.aggregation_fn == aggregation_fn.SUM):
            agg_message = mx.zeros([num_nodes, self.embedding_dim])
            agg_message = agg_message.at[target_idx].add(message)

        elif (self.aggregation_fn == aggregation_fn.AVG):
            agg_message = mx.zeros([num_nodes, self.embedding_dim])
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([num_nodes, 1]).at[target_idx].add(1)
            agg_message = agg_message /  mx.maximum(denominator, -1e6)

        elif (self.aggregation_fn == aggregation_fn.MAX):
            agg_message = mx.full([num_nodes, self.embedding_dim], -1e6)
            agg_message = agg_message.at[target_idx].maximum(message)

        elif (self.aggregation_fn == aggregation_fn.MIN):
            agg_message = mx.full([num_nodes, self.embedding_dim], 1e6)
            agg_message = agg_message.at[target_idx].minimum(message)

        new_node_embeddings = self.update_fn(agg_message)
        new_node_embeddings = nn.relu(new_node_embeddings)

        if (self.skip_connections):
            new_node_embeddings = new_node_embeddings + node_embeddings

        return new_node_embeddings

class mpnn(nn.Module):
    def __init__(self, embedding_dim: int, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
        super(mpnn, self).__init__()

        self.embedding_dim = embedding_dim
        self.skip_connections = skip_connections
        self.aggregation_function_fn = aggregation_fn

        self.mp_layer = [
            mp_layer(embedding_dim, skip_connections, aggregation_fn)
            for _ in range(num_mp_layers)
        ]

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        assert node_embeddings.shape[1] == self.embedding_dim, f'Incorrect node embedding size. Expected {self.embedding_dim}, got {node_embeddings.shape[1]}'

        for mp_layer in self.mp_layer:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)

        return node_embeddings
    
class bfs_decoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super(bfs_decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.bfs_state_outputs = nn.Linear(embedding_dim, 1)

    def __call__(self, data):
        node_embeddings, _ = data

        # BFS state prediction
        bfs_state_predictions = self.bfs_state_outputs(node_embeddings)
        bfs_state_predictions = mx.sigmoid(bfs_state_predictions.squeeze())

        return bfs_state_predictions

class bf_decoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super(bf_decoder, self).__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embedding_dim = embedding_dim
        
        # Simplified Bellman-Ford distance head with proper initialization
        # Use a single linear layer with proper initialization and output constraint
        self.bf_distance_head = nn.Linear(embedding_dim, 64)
        self.bf_distance_output = nn.Linear(64, 1)
        
        self.bf_predecessor_prob = nn.Linear(2 * embedding_dim, 1)

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        num_nodes = node_embeddings.shape[0]

        source_idx = connection_matrix[self.source_idx].astype(mx.int32)
        target_idx = connection_matrix[self.target_idx].astype(mx.int32)

        # Simplified distance prediction with stability improvements
        distance_features = nn.relu(self.bf_distance_head(node_embeddings))
        bf_distance_predictions = self.bf_distance_output(distance_features)
        
        # Apply ReLU to ensure non-negative distances (shortest paths can't be negative)
        # Add small epsilon for numerical stability
        bf_distance_predictions = nn.relu(bf_distance_predictions) + 1e-6
        bf_distance_predictions = bf_distance_predictions.squeeze()

        # Bellman-Ford predecessor predictions with efficient neighborhood-aware selection
        source_embeddings = mx.take(node_embeddings, source_idx, axis=0)
        target_embeddings = mx.take(node_embeddings, target_idx, axis=0)
        concatenated_embeddings = mx.concat([source_embeddings, target_embeddings], axis=1)
        edge_scores = self.bf_predecessor_prob(concatenated_embeddings)
        edge_scores = edge_scores.squeeze()

        # Efficient predecessor logits matrix with proper neighborhood-aware selection
        # Initialize with small negative values for numerical stability
        bf_predecessor_predictions = mx.full([num_nodes, num_nodes], -1)
        
        # Use scatter operation to efficiently populate valid edge scores
        # This allows each target node to select among its incoming neighbors
        bf_predecessor_predictions = bf_predecessor_predictions.at[target_idx, source_idx].add(edge_scores + 1)
        bf_predecessor_predictions = mx.softmax(bf_predecessor_predictions, axis=1)
        
        return bf_distance_predictions, bf_predecessor_predictions
    
class nge(nn.Module):
    def __init__(self, embedding_dim: int, skip_connections: bool, aggregation_fn: Enum, num_mp_layers: int):
        super(nge, self).__init__()


        # Separate encoders for each algorithm
        # 3 is the number of additional features (bfs_state, bf_distance, bf_predecessor)
        self.encoder = nn.Linear(embedding_dim + 3, embedding_dim)

        # Separate decoders for each algorithm
        self.bfs_decoder = bfs_decoder(embedding_dim)
        self.bf_decoder = bf_decoder(embedding_dim)

        # Separate termination heads for each algorithm 
        self.bfs_termination = nn.Linear(embedding_dim, 1, bias=True)
        self.bf_termination = nn.Linear(embedding_dim, 1, bias=True)
    
        # Shared processor
        self.processor = mpnn(embedding_dim, skip_connections, aggregation_fn, num_mp_layers)

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
    