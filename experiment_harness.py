import copy
import os
import wandb

from train import (
    MODEL_CONFIG,
    HYPERPARAMETERS,
    nge,
    train_model,
    evaluate_model,
    model as base_model,
    train_dataset,
    val_dataset,
    test_dataset,
    optim,
)

def run_experiments(config_list, save_dir="trained_models"):
    """
    Runs a queue of training configurations sequentially.
    Each configuration is a dict with keys for model and hyperparameters.
    Trains, logs, and saves the model for each configuration.
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx, config in enumerate(config_list):
        # Prepare config
        model_config = copy.deepcopy(MODEL_CONFIG)
        hyperparams = copy.deepcopy(HYPERPARAMETERS)
        model_config.update(config.get("model", {}))
        hyperparams.update(config.get("hyperparameters", {}))

        # Set up wandb for this run
        wandb.init(
            project="nge-train",
            config={**model_config, **hyperparams},
            reinit=True,
        )

        # Build model and optimizer
        model = nge(**model_config)
        total_steps = hyperparams['epochs'] * len(train_dataset)
        lr_decay = optim.cosine_decay(
            init=hyperparams['lr'],
            decay_steps=total_steps,
            end=0.0
        )
        optimizer = optim.Adam(learning_rate=lr_decay, weight_decay=1e-5)

        # Train
        train_model(model, train_dataset, optimizer, epochs=hyperparams['epochs'])

        # Evaluate
        val_loss = evaluate_model(model, val_dataset)
        test_loss = evaluate_model(model, test_dataset)
        wandb.log({"val_loss": float(val_loss), "test_loss": float(test_loss)})

        # Save model
        model_save_path = os.path.join(save_dir, f"model_{idx}.npz")
        # Save model parameters using mlx.utils
        import mlx.utils as utils
        utils.save(model_save_path, model.parameters())

        wandb.finish()

# Example usage:
# config_queue = [
#     {"model": {"embed_dim": 64}, "hyperparameters": {"epochs": 100, "lr": 1e-4}},
#     {"model": {"embed_dim": 128, "num_mp_layers": 3}, "hyperparameters": {"epochs": 200, "lr": 5e-5}},
# ]
# run_experiments(config_queue)
