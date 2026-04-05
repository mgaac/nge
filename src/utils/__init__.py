"""Utility exports with lazy loading.

Importing submodules such as ``src.utils.config`` should not eagerly import MLX.
The package-level conveniences remain available through ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "analyze_failure_modes": (".eval", "analyze_failure_modes"),
    "calculate_accuracies": (".eval", "calculate_accuracies"),
    "calculate_losses_and_accuracies": (".eval", "calculate_losses_and_accuracies"),
    "extract_per_head_magnitude_grads": (".eval", "extract_per_head_magnitude_grads"),
    "print_execution_details": (".eval", "print_execution_details"),
    "ExperimentConfig": (".config", "ExperimentConfig"),
    "load_config": (".config", "load_config"),
    "save_config": (".config", "save_config"),
    "validate_config": (".config", "validate_config"),
    "set_seed": (".repro", "set_seed"),
    "get_git_info": (".repro", "get_git_info"),
    "get_env_info": (".repro", "get_env_info"),
    "create_run_metadata": (".repro", "create_run_metadata"),
    "save_run_metadata": (".repro", "save_run_metadata"),
    "generate_run_name": (".repro", "generate_run_name"),
    "CheckpointManager": (".checkpoint", "CheckpointManager"),
    "MetricsLogger": (".logging", "MetricsLogger"),
    "load_metrics_history": (".logging", "load_metrics_history"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
