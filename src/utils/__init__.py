"""Utility functions for training and evaluation."""
from .eval import (
    analyze_failure_modes,
    calculate_accuracies,
    calculate_losses_and_accuracies,
    extract_per_head_magnitude_grads,
    print_execution_details,
)
from .config import ExperimentConfig, load_config, save_config, validate_config
from .repro import set_seed, get_git_info, get_env_info, create_run_metadata, save_run_metadata, generate_run_name
from .checkpoint import CheckpointManager
from .logging import MetricsLogger, load_metrics_history

__all__ = [
    'calculate_losses_and_accuracies',
    'calculate_accuracies',
    'analyze_failure_modes',
    'extract_per_head_magnitude_grads',
    'print_execution_details',
    'ExperimentConfig',
    'load_config',
    'save_config',
    'validate_config',
    'set_seed',
    'get_git_info',
    'get_env_info',
    'create_run_metadata',
    'save_run_metadata',
    'generate_run_name',
    'CheckpointManager',
    'MetricsLogger',
    'load_metrics_history',
]
