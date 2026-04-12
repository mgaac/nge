"""Configuration loading and validation.

This module provides utilities for loading experiment configurations from YAML files
and validating them against expected schemas.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field, asdict

from src.utils.task_specs import (
    DEFAULT_ALGORITHMS,
    SELECT_TASK_CHOICES,
    TERMINATION_LATENT_CHOICES,
    normalize_algorithm_order,
)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embed_dim: int = 32
    residual_connections: bool = True
    agg_fn: str = "MAX"  # SUM, AVG, MIN, MAX
    num_mp_layers: int = 2
    dropout: float = 0.1
    algorithms: list[str] = field(default_factory=lambda: list(DEFAULT_ALGORITHMS))
    termination_mode: str = "head"  # head, distance
    termination_distance_latent: str = "processed"  # processed, encoded, encoded_bfs, encoded_bf, encoded_prim
    termination_distance: str = "mean_l2"  # l2, mean_l2, l1, mse
    termination_distance_threshold: float = 0.01
    termination_distance_signal: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 500
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    batch_size: int = 10
    eval_interval: int = 10
    seed: int = 42
    tasks: str = "all"
    init_checkpoint: str | None = None
    init_checkpoint_modules: list[str] = field(default_factory=list)
    freeze_modules: list[str] = field(default_factory=list)
    reset_modules: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    """Dataset configuration."""
    train_path: str = "data/train_dataset.npz"
    val_path: str = "data/val_dataset.npz"
    test_path: str = "data/test_dataset.npz"
    task_paths: Dict[str, Dict[str, str]] | None = None


@dataclass
class LoggingConfig:
    """Logging and tracking configuration."""
    use_wandb: bool = False
    wandb_project: str = "nge"
    wandb_entity: str = ""
    log_interval: int = 1
    save_checkpoints: bool = True
    checkpoint_interval: int = 50
    checkpoint_keep_last: int = 5


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
    
    if config.model.num_mp_layers < 0:
        raise ValueError(
            "num_mp_layers must be non-negative, "
            f"got {config.model.num_mp_layers}"
        )

    if not 0 <= config.model.dropout < 1:
        raise ValueError(f"dropout must be in [0, 1), got {config.model.dropout}")

    normalize_algorithm_order(config.model.algorithms)

    if config.model.termination_mode not in ["head", "distance"]:
        raise ValueError(f"Invalid termination_mode: {config.model.termination_mode}")

    if config.model.termination_distance_latent not in TERMINATION_LATENT_CHOICES:
        raise ValueError(
            "termination_distance_latent must be one of: "
            f"{', '.join(TERMINATION_LATENT_CHOICES)}, "
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

    if not isinstance(config.model.termination_distance_signal, bool):
        raise ValueError(
            "termination_distance_signal must be boolean, "
            f"got {type(config.model.termination_distance_signal).__name__}"
        )
    
    # Validate training config
    if config.training.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.training.epochs}")
    
    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.training.learning_rate}")
    
    if config.training.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.training.batch_size}")

    if config.training.tasks not in SELECT_TASK_CHOICES:
        raise ValueError(
            "training.tasks must be one of: "
            f"{', '.join(SELECT_TASK_CHOICES)}. Got {config.training.tasks}"
        )
    if config.training.tasks != "all" and config.training.tasks not in config.model.algorithms:
        raise ValueError(
            "training.tasks must be 'all' or one of model.algorithms. "
            f"Got {config.training.tasks} for algorithms {config.model.algorithms}"
        )

    if config.training.init_checkpoint is not None and not isinstance(
        config.training.init_checkpoint, str
    ):
        raise ValueError("training.init_checkpoint must be a string path or null.")

    for field_name in ("init_checkpoint_modules", "freeze_modules", "reset_modules"):
        field_value = getattr(config.training, field_name)
        if not isinstance(field_value, list) or not all(
            isinstance(entry, str) and entry.strip() for entry in field_value
        ):
            raise ValueError(
                f"training.{field_name} must be a list of non-empty strings."
            )

    if config.training.init_checkpoint_modules and config.training.init_checkpoint is None:
        raise ValueError(
            "training.init_checkpoint_modules requires training.init_checkpoint."
        )

    overlap = sorted(set(config.training.freeze_modules) & set(config.training.reset_modules))
    if overlap:
        raise ValueError(
            "training.freeze_modules and training.reset_modules overlap: "
            + ", ".join(overlap)
        )

    if config.logging.log_interval <= 0:
        raise ValueError(f"log_interval must be positive, got {config.logging.log_interval}")

    if config.logging.checkpoint_interval <= 0:
        raise ValueError(
            f"checkpoint_interval must be positive, got {config.logging.checkpoint_interval}"
        )

    if config.logging.checkpoint_keep_last <= 0:
        raise ValueError(
            "checkpoint_keep_last must be positive, "
            f"got {config.logging.checkpoint_keep_last}"
        )
    
    # Validate data paths exist
    for path_name, path in [
        ('train_path', config.data.train_path),
        ('val_path', config.data.val_path),
        ('test_path', config.data.test_path)
    ]:
        if config.data.task_paths is None and not Path(path).exists():
            raise ValueError(f"Data file not found: {path_name}={path}")

    if config.data.task_paths is not None:
        if not isinstance(config.data.task_paths, dict) or not config.data.task_paths:
            raise ValueError("data.task_paths must be a non-empty mapping when provided.")
        for algorithm in config.model.algorithms:
            if algorithm not in config.data.task_paths:
                raise ValueError(
                    f"Missing data.task_paths entry for configured algorithm: {algorithm}"
                )
            split_paths = config.data.task_paths[algorithm]
            if not isinstance(split_paths, dict):
                raise ValueError(
                    f"data.task_paths.{algorithm} must be a mapping of split names to paths."
                )
            for split_key in ("train_path", "val_path", "test_path"):
                if split_key not in split_paths:
                    raise ValueError(
                        f"Missing {split_key} in data.task_paths.{algorithm}"
                    )
                if not Path(split_paths[split_key]).exists():
                    raise ValueError(
                        f"Data file not found: data.task_paths.{algorithm}.{split_key}="
                        f"{split_paths[split_key]}"
                    )
