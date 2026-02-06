"""Configuration loading and validation.

This module provides utilities for loading experiment configurations from YAML files
and validating them against expected schemas.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embed_dim: int = 32
    residual_connections: bool = True
    agg_fn: str = "MAX"  # SUM, AVG, MIN, MAX
    num_mp_layers: int = 2
    dropout: float = 0.1
    termination_mode: str = "head"  # head, distance
    termination_distance_latent: str = "processed"  # processed, encoded
    termination_distance: str = "mean_l2"  # l2, mean_l2, l1, mse
    termination_distance_threshold: float = 0.01


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 500
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    batch_size: int = 10
    eval_interval: int = 10
    seed: int = 42


@dataclass
class DataConfig:
    """Dataset configuration."""
    train_path: str = "data/train_dataset.npz"
    val_path: str = "data/val_dataset.npz"
    test_path: str = "data/test_dataset.npz"


@dataclass
class LoggingConfig:
    """Logging and tracking configuration."""
    use_wandb: bool = False
    wandb_project: str = "nge"
    wandb_entity: str = ""
    log_interval: int = 1
    save_checkpoints: bool = True
    checkpoint_interval: int = 50


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'model': asdict(self.model),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'logging': asdict(self.logging),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(
            name=config_dict.get('name', 'default'),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
        )


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig object with validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    return ExperimentConfig.from_dict(config_dict)


def save_config(config: ExperimentConfig, output_path: str | Path) -> None:
    """Save experiment configuration to YAML file.
    
    Args:
        config: ExperimentConfig object to save
        output_path: Path where to save the YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def validate_config(config: ExperimentConfig) -> None:
    """Validate experiment configuration.
    
    Args:
        config: ExperimentConfig to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate model config
    if config.model.embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {config.model.embed_dim}")
    
    if config.model.agg_fn not in ["SUM", "AVG", "MIN", "MAX"]:
        raise ValueError(f"Invalid agg_fn: {config.model.agg_fn}")
    
    if config.model.num_mp_layers <= 0:
        raise ValueError(f"num_mp_layers must be positive, got {config.model.num_mp_layers}")
    
    if not 0 <= config.model.dropout < 1:
        raise ValueError(f"dropout must be in [0, 1), got {config.model.dropout}")

    if config.model.termination_mode not in ["head", "distance"]:
        raise ValueError(f"Invalid termination_mode: {config.model.termination_mode}")

    if config.model.termination_distance_latent not in ["processed", "encoded"]:
        raise ValueError(
            "termination_distance_latent must be 'processed' or 'encoded', "
            f"got {config.model.termination_distance_latent}"
        )

    if config.model.termination_distance not in ["l2", "mean_l2", "l1", "mse"]:
        raise ValueError(
            "termination_distance must be one of: l2, mean_l2, l1, mse. "
            f"Got {config.model.termination_distance}"
        )

    if config.model.termination_distance_threshold < 0:
        raise ValueError(
            "termination_distance_threshold must be non-negative, "
            f"got {config.model.termination_distance_threshold}"
        )
    
    # Validate training config
    if config.training.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.training.epochs}")
    
    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.training.learning_rate}")
    
    if config.training.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.training.batch_size}")
    
    # Validate data paths exist
    for path_name, path in [
        ('train_path', config.data.train_path),
        ('val_path', config.data.val_path),
        ('test_path', config.data.test_path)
    ]:
        if not Path(path).exists():
            raise ValueError(f"Data file not found: {path_name}={path}")
