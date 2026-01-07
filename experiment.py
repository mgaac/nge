from enum import Enum

import mlx.core as mx
import mlx.nn as nn

import wandb

from data.data import load_dataset 

class ConvergenceMetric(Enum):
    COSINE_SIMILARITY = "cosine"
    L2_NORM = "l2"
    
    def compute(self, prev_embeddings, curr_embeddings):
        if self == ConvergenceMetric.COSINE_SIMILARITY:
            return 1 - nn.losses.cosine_similarity(prev_embeddings, curr_embeddings, axis=-1).mean()
        elif self == ConvergenceMetric.L2_NORM:
            return mx.linalg.norm(curr_embeddings - prev_embeddings)
        else:
            raise ValueError(f"Unknown convergence metric: {self}")

EXP_CONFIG = {
    "num_bf_steps": 5,
    "convergence_metric": ConvergenceMetric.COSINE_SIMILARITY,
    "commit": ""
}

train_dataset = load_dataset("data/train_dataset.npz")

model = mx.import_function("trained_model.mlxfn")

wandb.init(project="stc-experiment", config=EXP_CONFIG)


for sample in train_dataset:
    num_bf_steps = len(sample['bf_distance_targets'])

    if num_bf_steps != EXP_CONFIG['num_bf_steps']:
        continue
    
    input_embeddings = mx.zeros([sample['num_nodes'], 66])
    edge_matrix = sample['edge_matrix']

    for i in range(num_bf_steps):
        input = (input_embeddings, edge_matrix)

        output = model(input)

        _, _, _, processed_embeddings = output






