"""Metrics logging utilities with JSONL format.

This module provides append-only JSONL logging for training metrics
and optional Weights & Biases integration.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class MetricsLogger:
    """Append-only JSONL metrics logger with optional W&B integration.
    
    Each metric entry is written as a single JSON line containing:
    - step: training step number
    - split: data split (train/val/test)
    - metrics: dictionary of metric values
    - timestamp: ISO format timestamp
    """
    
    def __init__(
        self,
        log_file: Path,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize metrics logger.
        
        Args:
            log_file: Path to JSONL log file
            use_wandb: Whether to enable Weights & Biases logging
            wandb_project: W&B project name (required if use_wandb=True)
            wandb_entity: W&B entity name (optional)
            wandb_config: Configuration to log to W&B (optional)
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or append to log file
        if not self.log_file.exists():
            self.log_file.touch()
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None
        self.wandb_step_metric = "epoch"
        
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not installed, logging locally only")
                self.use_wandb = False
            elif wandb_project is None:
                raise ValueError("wandb_project required when use_wandb=True")
            else:
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=wandb_config,
                )
                self.wandb_run.define_metric(self.wandb_step_metric)
                for split_name in ("train", "val", "train_eval", "test"):
                    self.wandb_run.define_metric(
                        f"{split_name}/*",
                        step_metric=self.wandb_step_metric,
                    )
    
    def log(self, step: int, metrics: Dict[str, Any], split: str = "train") -> None:
        """Log metrics for a given step.
        
        Args:
            step: Training step number
            metrics: Dictionary of metric name -> value
            split: Data split name (train/val/test)
        """
        from datetime import datetime
        
        # Create log entry
        entry = {
            'step': step,
            'split': split,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }
        
        # Append to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Log to W&B if enabled
        if self.use_wandb and self.wandb_run is not None:
            wandb_metrics = {self.wandb_step_metric: step}
            for key, value in metrics.items():
                # Convert to Python scalar if needed
                if hasattr(value, 'item'):
                    value = value.item()

                wandb_metrics[f"{split}/{key}"] = value

            self.wandb_run.log(wandb_metrics)
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary metrics (e.g., final test results).
        
        Args:
            summary: Dictionary of summary metrics
        """
        if self.use_wandb and self.wandb_run is not None:
            for key, value in summary.items():
                if hasattr(value, 'item'):
                    value = value.item()
                wandb.run.summary[key] = value
    
    def finish(self) -> None:
        """Finish logging and close W&B run if active."""
        if self.use_wandb and self.wandb_run is not None:
            wandb.finish()
    
    def read_metrics(self, split: Optional[str] = None) -> list[Dict[str, Any]]:
        """Read metrics from log file.
        
        Args:
            split: Optional split to filter by
            
        Returns:
            List of metric entries
        """
        metrics = []
        
        if not self.log_file.exists():
            return metrics
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if split is None or entry.get('split') == split:
                        metrics.append(entry)
        
        return metrics
    
    def validate_log_file(self) -> Tuple[bool, Optional[str]]:
        """Validate that log file is valid JSONL with monotonic steps.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.log_file.exists():
            return True, None
        
        try:
            last_step = -1
            line_num = 0
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    line_num += 1
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        return False, f"Invalid JSON on line {line_num}: {e}"
                    
                    if 'step' not in entry:
                        return False, f"Missing 'step' field on line {line_num}"
                    
                    current_step = entry['step']
                    if current_step < last_step:
                        return False, f"Non-monotonic steps: {last_step} -> {current_step} on line {line_num}"
                    
                    last_step = current_step
            
            return True, None
            
        except Exception as e:
            return False, f"Error reading log file: {e}"


def load_metrics_history(log_file: Path) -> Dict[str, list]:
    """Load complete metrics history from JSONL file.
    
    Args:
        log_file: Path to JSONL metrics file
        
    Returns:
        Dictionary mapping metric names to lists of (step, value) tuples
    """
    metrics_history = {}
    
    if not log_file.exists():
        return metrics_history
    
    with open(log_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            entry = json.loads(line)
            step = entry['step']
            split = entry.get('split', 'train')
            
            for metric_name, value in entry['metrics'].items():
                key = f"{split}_{metric_name}" if split != "train" else metric_name
                
                if key not in metrics_history:
                    metrics_history[key] = []
                
                metrics_history[key].append((step, value))
    
    return metrics_history
