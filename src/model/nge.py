"""Neural Graph Execution (NGE) model implementation.

This module implements a graph neural network for executing graph algorithms,
specifically Bellman-Ford and BFS, using message passing neural networks.
"""

import mlx.core as mx
import mlx.nn as nn

from enum import Enum


class AggregationFn(Enum):
    """Aggregation functions for message passing."""

    SUM = 1
    AVG = 2
    MIN = 4
    MAX = 5


class MPLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        residual_connections: bool,
        dropout: float,
        agg_fn: Enum,
    ):
        super().__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embed_dim = embed_dim

        self.residual_connections = residual_connections
        self.agg_fn = agg_fn

        self.message_fn = nn.Linear(2 * embed_dim + 1, embed_dim, bias=True)

        self.embed_ln = nn.LayerNorm(2 * embed_dim)
        self.update_ln = nn.LayerNorm(embed_dim)

        self.update_fn = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, connection_matrix, node_embeddings):
        num_nodes = node_embeddings.shape[0]

        edge_weights = mx.expand_dims(connection_matrix[2], axis=-1)

        source_idx = connection_matrix[self.source_idx].astype(mx.int32)
        target_idx = connection_matrix[self.target_idx].astype(mx.int32)

        source_embeddings = mx.take(node_embeddings, source_idx, axis=0)
        target_embeddings = mx.take(node_embeddings, target_idx, axis=0)

        message_in = mx.concatenate([source_embeddings, target_embeddings], axis=1)
        message_in = self.embed_ln(message_in)
        message = self.message_fn(mx.concatenate([message_in, edge_weights], axis=1))

        if self.agg_fn == AggregationFn.SUM:
            agg_message = mx.zeros([num_nodes, self.embed_dim])
            agg_message = agg_message.at[target_idx].add(message)

        elif self.agg_fn == AggregationFn.AVG:
            agg_message = mx.zeros([num_nodes, self.embed_dim])
            agg_message = agg_message.at[target_idx].add(message)
            denominator = mx.zeros([num_nodes, 1]).at[target_idx].add(1)
            agg_message = agg_message / mx.maximum(denominator, 1e-9)

        elif self.agg_fn == AggregationFn.MAX:
            agg_message = mx.full([num_nodes, self.embed_dim], -1e6)
            agg_message = agg_message.at[target_idx].maximum(message)
            has_incoming = mx.zeros([num_nodes, 1]).at[target_idx].add(1) > 0
            agg_message = mx.where(
                has_incoming, agg_message, mx.zeros_like(agg_message)
            )

        elif self.agg_fn == AggregationFn.MIN:
            agg_message = mx.full([num_nodes, self.embed_dim], 1e6)
            agg_message = agg_message.at[target_idx].minimum(message)
            has_incoming = mx.zeros([num_nodes, 1]).at[target_idx].add(1) > 0
            agg_message = mx.where(
                has_incoming, agg_message, mx.zeros_like(agg_message)
            )

        agg_message = nn.relu(self.update_fn(agg_message))
        new_node_embeddings = self.update_ln(agg_message) + node_embeddings
        new_node_embeddings = self.dropout(new_node_embeddings)

        return new_node_embeddings


class MPNN(nn.Module):
    """Message Passing Neural Network."""

    def __init__(
        self,
        embed_dim: int,
        residual_connections: bool,
        agg_fn: Enum,
        num_mp_layers: int,
        dropout: float = 0.0,
    ):
        super(MPNN, self).__init__()

        self.embed_dim = embed_dim
        self.residual_connections = residual_connections
        self.agg_fn = agg_fn

        self.mp_layers = [
            MPLayer(embed_dim, residual_connections, dropout, agg_fn)
            for _ in range(num_mp_layers)
        ]

    def __call__(self, data):
        node_embeddings, connection_matrix = data

        assert node_embeddings.shape[1] == self.embed_dim, (
            f"Incorrect node embedding size. Expected {self.embed_dim}, got {node_embeddings.shape[1]}"
        )

        for mp_layer in self.mp_layers:
            node_embeddings = mp_layer(connection_matrix, node_embeddings)

        return node_embeddings


class BFSDecoder(nn.Module):
    """Decoder for BFS state predictions."""

    def __init__(self, embed_dim: int):
        super(BFSDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.bfs_state_outputs = nn.Linear(int(embed_dim * 3), 1, bias=False)
        self.layer_norm = nn.LayerNorm(embed_dim * 3)

    def __call__(self, data):
        processed_embeddings, encoded_embeddings = data

        input = mx.concatenate([processed_embeddings, encoded_embeddings], axis=1)
        input = self.layer_norm(input)

        bfs_state_predictions = self.bfs_state_outputs(input).squeeze()

        return bfs_state_predictions


class BFDecoder(nn.Module):
    """Decoder for Bellman-Ford distance and predecessor predictions."""

    def __init__(self, embed_dim: int):
        super(BFDecoder, self).__init__()

        self.source_idx = 0
        self.target_idx = 1

        self.embed_dim = embed_dim

        # Simple linear heads as per paper
        # processed_embeddings is 2*embed_dim, encoded is embed_dim

        self.distance_head = nn.Linear(3 * embed_dim, 1, bias=True)
        self.predecessor_head = nn.Linear(6 * embed_dim + 1, 1, bias=True)

        self.distance_ln = nn.LayerNorm(3 * embed_dim)
        self.predecessor_ln = nn.LayerNorm(6 * embed_dim + 1)

    def __call__(self, data):
        processed_embeddings, encoded_embeddings, connection_matrix = data

        num_nodes = processed_embeddings.shape[0]

        edge_weights = mx.expand_dims(connection_matrix[2], axis=-1)

        source_idx = connection_matrix[self.source_idx].astype(mx.int32)
        target_idx = connection_matrix[self.target_idx].astype(mx.int32)

        # Distance prediction: concatenate processed + encoded for each node
        joint_embeddings = mx.concatenate(
            [processed_embeddings, encoded_embeddings], axis=1
        )
        joint_embeddings = self.distance_ln(joint_embeddings)
        bf_distance_predictions = self.distance_head(joint_embeddings).squeeze()

        # Predecessor prediction: concatenate source and target (don't add)
        encoded_source_embeddings = mx.take(encoded_embeddings, source_idx, axis=0)
        encoded_target_embeddings = mx.take(encoded_embeddings, target_idx, axis=0)

        processed_source_embeddings = mx.take(processed_embeddings, source_idx, axis=0)
        processed_target_embeddings = mx.take(processed_embeddings, target_idx, axis=0)

        # Concatenate all features (preserves directional information)
        concatenated_embeddings = mx.concatenate(
            [
                encoded_source_embeddings,
                encoded_target_embeddings,
                processed_source_embeddings,
                processed_target_embeddings,
                edge_weights,
            ],
            axis=1,
        )
        concatenated_embeddings = self.predecessor_ln(concatenated_embeddings)

        edge_logits = self.predecessor_head(concatenated_embeddings).squeeze()

        # Use -inf instead of -1e6 for better numerical stability
        bf_predecessor_predictions = mx.full([num_nodes, num_nodes], -1e6)
        bf_predecessor_predictions[target_idx, source_idx] = edge_logits

        return bf_distance_predictions, bf_predecessor_predictions


class NGE(nn.Module):
    """Neural Graph Execution model for executing graph algorithms."""

    def __init__(
        self,
        embed_dim: int,
        residual_connections: bool,
        agg_fn: Enum,
        num_mp_layers: int,
        dropout: float = 0.0,
    ):
        super(NGE, self).__init__()

        self.embed_dim = embed_dim
        self.ln = nn.LayerNorm(2 * embed_dim)

        self.bfs_encoder = nn.Linear(2 * embed_dim + 2, embed_dim)
        self.bf_encoder = nn.Linear(2 * embed_dim + 2, embed_dim)

        self.bfs_decoder = BFSDecoder(embed_dim)
        self.bf_decoder = BFDecoder(embed_dim)

        self.bfs_termination = nn.Linear(2 * embed_dim, 1, bias=True)
        self.bf_termination = nn.Linear(2 * embed_dim, 1, bias=True)

        self.processor = MPNN(
            2 * embed_dim,
            residual_connections,
            agg_fn,
            num_mp_layers,
            dropout,
        )

    def __call__(self, data, return_latents: bool = False):
        node_embeddings, connection_matrix = data

        bfs_encoded_embeddings = self.bfs_encoder(node_embeddings)
        bf_encoded_embeddings = self.bf_encoder(node_embeddings)

        encoded_embeddings = mx.concatenate(
            [bfs_encoded_embeddings, bf_encoded_embeddings], axis=1
        )
        encoded_embeddings = self.ln(encoded_embeddings)

        processed_embeddings = self.processor((encoded_embeddings, connection_matrix))

        bfs_output = self.bfs_decoder((processed_embeddings, bfs_encoded_embeddings))
        bf_output = self.bf_decoder(
            (processed_embeddings, bf_encoded_embeddings, connection_matrix)
        )

        avg_embeddings = mx.mean(processed_embeddings, axis=0)

        bfs_termination_prob = self.bfs_termination(avg_embeddings).squeeze()

        bf_termination_prob = self.bf_termination(avg_embeddings).squeeze()

        termination_probs = {"bfs": bfs_termination_prob, "bf": bf_termination_prob}

        if return_latents:
            aux = {
                "bfs_encoded": bfs_encoded_embeddings,
                "bf_encoded": bf_encoded_embeddings,
                "encoded": encoded_embeddings,
                "avg_processed": avg_embeddings,
            }
            return bfs_output, bf_output, termination_probs, processed_embeddings, aux

        return bfs_output, bf_output, termination_probs, processed_embeddings
