"""Neural Graph Execution (NGE) model implementation.

This module implements a graph neural network for executing graph algorithms,
specifically Bellman-Ford, BFS, and Prim, using message passing neural networks.
"""

import mlx.core as mx
import mlx.nn as nn

from enum import Enum

from src.utils.task_specs import (
    DEFAULT_ALGORITHMS,
    algorithm_family,
    input_feature_dim,
    normalize_algorithm_order,
    processor_algorithm_order,
)


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


class StateMaskDecoder(nn.Module):
    """Decoder for node-wise binary state predictions."""

    def __init__(self, embed_dim: int, processor_embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.processor_embed_dim = processor_embed_dim
        self.input_dim = processor_embed_dim + embed_dim
        self.state_outputs = nn.Linear(self.input_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(self.input_dim)

    def __call__(self, data):
        processed_embeddings, encoded_embeddings = data

        input = mx.concatenate([processed_embeddings, encoded_embeddings], axis=1)
        input = self.layer_norm(input)

        state_predictions = self.state_outputs(input).squeeze()

        return state_predictions


class EdgePointerHead(nn.Module):
    """Shared edge-wise pointer head for predecessor prediction."""

    def __init__(self, embed_dim: int, processor_embed_dim: int):
        super().__init__()

        self.source_idx = 0
        self.target_idx = 1
        self.embed_dim = embed_dim
        self.processor_embed_dim = processor_embed_dim
        self.input_dim = 2 * embed_dim + 2 * processor_embed_dim + 1

        self.pointer_head = nn.Linear(self.input_dim, 1, bias=True)
        self.pointer_ln = nn.LayerNorm(self.input_dim)

    def __call__(self, encoded_embeddings, processed_embeddings, connection_matrix):
        num_nodes = processed_embeddings.shape[0]

        edge_weights = mx.expand_dims(connection_matrix[2], axis=-1)

        source_idx = connection_matrix[self.source_idx].astype(mx.int32)
        target_idx = connection_matrix[self.target_idx].astype(mx.int32)

        encoded_source_embeddings = mx.take(encoded_embeddings, source_idx, axis=0)
        encoded_target_embeddings = mx.take(encoded_embeddings, target_idx, axis=0)

        processed_source_embeddings = mx.take(processed_embeddings, source_idx, axis=0)
        processed_target_embeddings = mx.take(processed_embeddings, target_idx, axis=0)

        pointer_input = mx.concatenate(
            [
                encoded_source_embeddings,
                encoded_target_embeddings,
                processed_source_embeddings,
                processed_target_embeddings,
                edge_weights,
            ],
            axis=1,
        )
        pointer_input = self.pointer_ln(pointer_input)

        edge_logits = self.pointer_head(pointer_input).reshape(-1)

        # MLX documents dedicated put/scatter ops for indexed writes. Flattening the
        # destination avoids the expensive advanced multi-index assignment path.
        predecessor_predictions = mx.put_along_axis(
            mx.full([num_nodes * num_nodes], -1e6),
            target_idx * num_nodes + source_idx,
            edge_logits,
            axis=None,
        ).reshape(num_nodes, num_nodes)
        return predecessor_predictions


class ShortestPathDecoder(nn.Module):
    """Decoder for distance and predecessor predictions."""

    def __init__(self, embed_dim: int, processor_embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.processor_embed_dim = processor_embed_dim
        self.distance_input_dim = processor_embed_dim + embed_dim

        self.distance_head = nn.Linear(self.distance_input_dim, 1, bias=True)
        self.distance_ln = nn.LayerNorm(self.distance_input_dim)
        self.pointer_head = EdgePointerHead(embed_dim, processor_embed_dim)

    def __call__(self, data):
        processed_embeddings, encoded_embeddings, connection_matrix = data

        # Distance prediction: concatenate processed + encoded for each node
        joint_embeddings = mx.concatenate(
            [processed_embeddings, encoded_embeddings], axis=1
        )
        joint_embeddings = self.distance_ln(joint_embeddings)
        distance_predictions = self.distance_head(joint_embeddings).squeeze()

        predecessor_predictions = self.pointer_head(
            encoded_embeddings, processed_embeddings, connection_matrix
        )

        return distance_predictions, predecessor_predictions


class PrimDecoder(nn.Module):
    """Decoder for Prim state, key, and predecessor predictions."""

    def __init__(self, embed_dim: int, processor_embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.processor_embed_dim = processor_embed_dim
        self.state_input_dim = processor_embed_dim + embed_dim

        self.state_head = nn.Linear(self.state_input_dim, 1, bias=True)
        self.key_head = nn.Linear(self.state_input_dim, 1, bias=True)
        self.state_ln = nn.LayerNorm(self.state_input_dim)
        self.pointer_head = EdgePointerHead(embed_dim, processor_embed_dim)

    def __call__(self, data):
        processed_embeddings, encoded_embeddings, connection_matrix = data

        joint_embeddings = mx.concatenate(
            [processed_embeddings, encoded_embeddings], axis=1
        )
        joint_embeddings = self.state_ln(joint_embeddings)

        prim_state_predictions = self.state_head(joint_embeddings).squeeze()
        prim_key_predictions = self.key_head(joint_embeddings).squeeze()
        prim_predecessor_predictions = self.pointer_head(
            encoded_embeddings, processed_embeddings, connection_matrix
        )

        return (
            prim_state_predictions,
            prim_key_predictions,
            prim_predecessor_predictions,
        )


class NGE(nn.Module):
    """Neural Graph Execution model for executing graph algorithms."""

    def __init__(
        self,
        embed_dim: int,
        residual_connections: bool,
        agg_fn: Enum,
        num_mp_layers: int,
        dropout: float = 0.0,
        algorithms: tuple[str, ...] | None = None,
    ):
        super(NGE, self).__init__()

        self.algorithms = normalize_algorithm_order(algorithms)
        self.processor_algorithms = processor_algorithm_order(self.algorithms)
        self.embed_dim = embed_dim
        self.processor_embed_dim = len(self.algorithms) * embed_dim
        self.input_dim = self.processor_embed_dim + input_feature_dim(self.algorithms)
        self.ln = nn.LayerNorm(self.processor_embed_dim)

        for algorithm in self.algorithms:
            encoder = nn.Linear(self.input_dim, embed_dim)
            setattr(self, f"{algorithm}_encoder", encoder)

            family = algorithm_family(algorithm)
            if family == "state_mask":
                decoder = StateMaskDecoder(embed_dim, self.processor_embed_dim)
            elif family == "shortest_path":
                decoder = ShortestPathDecoder(embed_dim, self.processor_embed_dim)
            elif family == "mst":
                decoder = PrimDecoder(embed_dim, self.processor_embed_dim)
            else:
                raise ValueError(f"Unsupported algorithm family: {family}")
            setattr(self, f"{algorithm}_decoder", decoder)
            setattr(
                self,
                f"{algorithm}_termination",
                nn.Linear(self.processor_embed_dim, 1, bias=True),
            )

        self.processor = MPNN(
            self.processor_embed_dim,
            residual_connections,
            agg_fn,
            num_mp_layers,
            dropout,
        )

    def __call__(self, data, return_latents: bool = False):
        node_embeddings, connection_matrix = data

        encoded_by_algorithm = {}
        for algorithm in self.algorithms:
            encoder = getattr(self, f"{algorithm}_encoder")
            encoded_by_algorithm[algorithm] = encoder(node_embeddings)

        encoded_embeddings = mx.concatenate(
            [encoded_by_algorithm[algorithm] for algorithm in self.processor_algorithms],
            axis=1,
        )
        encoded_embeddings = self.ln(encoded_embeddings)

        processed_embeddings = self.processor((encoded_embeddings, connection_matrix))

        algorithm_outputs = {}
        for algorithm in self.algorithms:
            family = algorithm_family(algorithm)
            decoder = getattr(self, f"{algorithm}_decoder")
            encoded = encoded_by_algorithm[algorithm]
            if family == "state_mask":
                algorithm_outputs[algorithm] = decoder((processed_embeddings, encoded))
            else:
                algorithm_outputs[algorithm] = decoder(
                    (processed_embeddings, encoded, connection_matrix)
                )

        avg_embeddings = mx.mean(processed_embeddings, axis=0)

        termination_probs = {}
        for algorithm in self.algorithms:
            termination_head = getattr(self, f"{algorithm}_termination")
            termination_probs[algorithm] = termination_head(avg_embeddings).squeeze()

        if return_latents:
            aux = {
                "encoded": encoded_embeddings,
                "avg_processed": avg_embeddings,
            }
            aux.update(
                {
                    f"{algorithm}_encoded": encoded_by_algorithm[algorithm]
                    for algorithm in self.algorithms
                }
            )
            if self.algorithms == DEFAULT_ALGORITHMS:
                return (
                    algorithm_outputs["bfs"],
                    algorithm_outputs["bf"],
                    algorithm_outputs["prim"],
                    termination_probs,
                    processed_embeddings,
                    aux,
                )
            return algorithm_outputs, termination_probs, processed_embeddings, aux

        if self.algorithms == DEFAULT_ALGORITHMS:
            return (
                algorithm_outputs["bfs"],
                algorithm_outputs["bf"],
                algorithm_outputs["prim"],
                termination_probs,
                processed_embeddings,
            )
        return algorithm_outputs, termination_probs, processed_embeddings
