"""Checkpoint management for model and optimizer state.

This module provides utilities for:
- Saving model and optimizer state
- Loading checkpoints with resume support
- Managing checkpoint files and symlinks
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils


def _tree_flatten(tree):
    """Flatten a pytree into a flat name->array dict (MLX version compatible)."""
    try:
        flat = utils.tree_flatten(tree, destination={})
    except TypeError:
        flat = utils.tree_flatten(tree)
    return dict(flat)


def _tree_unflatten(tree):
    """Unflatten a flat tree produced by _tree_flatten."""
    return utils.tree_unflatten(list(tree.items()) if isinstance(tree, dict) else tree)

class CheckpointManager:
    """Manages saving and loading of training checkpoints.
    
    Each checkpoint includes:
    - Model weights
    - Optimizer state
    - Training step number
    - Additional metadata
    
    The manager maintains a 'latest' marker file pointing to the most recent checkpoint.
    """
    
    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory where checkpoints will be saved
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_file = self.checkpoint_dir / "latest.json"
    
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        step: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save checkpoint with model and optimizer state.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            step: Current training step
            metadata: Optional additional metadata to save
            
        Returns:
            Path to saved checkpoint directory
        """
        # Create checkpoint directory for this step
        ckpt_name = f"step_{step:08d}"
        ckpt_path = self.checkpoint_dir / ckpt_name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights (flatten tree to a flat name->array dict)
        model_weights = _tree_flatten(model.parameters())
        mx.save_safetensors(str(ckpt_path / "model.safetensors"), model_weights)
        
        # Save optimizer state
        # Save optimizer state (flatten tree to a flat name->array dict)
        optimizer_state = _tree_flatten(optimizer.state)
        mx.save_safetensors(str(ckpt_path / "optimizer.safetensors"), optimizer_state)
        
        # Save checkpoint metadata
        ckpt_metadata = {
            'step': step,
            'model_file': 'model.safetensors',
            'optimizer_file': 'optimizer.safetensors',
        }
        if metadata:
            ckpt_metadata['metadata'] = metadata
        
        with open(ckpt_path / "checkpoint.json", 'w') as f:
            json.dump(ckpt_metadata, f, indent=2)
        
        # Update latest marker
        latest_info = {
            'checkpoint_name': ckpt_name,
            'step': step,
            'path': str(ckpt_path),
        }
        with open(self.latest_file, 'w') as f:
            json.dump(latest_info, f, indent=2)
        
        return ckpt_path
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        checkpoint_path: Optional[Path] = None
    ) -> Tuple[nn.Module, Optional[optim.Optimizer], int]:
        """Load checkpoint and restore model and optimizer state.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            checkpoint_path: Optional path to specific checkpoint. If None, loads latest.
            
        Returns:
            Tuple of (model, optimizer, step)
            
        Raises:
            FileNotFoundError: If checkpoint not found
            ValueError: If checkpoint is malformed
        """
        # Determine which checkpoint to load
        if checkpoint_path is None:
            if not self.latest_file.exists():
                raise FileNotFoundError("No checkpoints found (latest.json missing)")
            
            with open(self.latest_file, 'r') as f:
                latest_info = json.load(f)
            checkpoint_name = latest_info.get("checkpoint_name")
            if checkpoint_name:
                checkpoint_path = self.checkpoint_dir / checkpoint_name
            else:
                checkpoint_path = Path(latest_info["path"])
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint metadata
        metadata_file = checkpoint_path / "checkpoint.json"
        if not metadata_file.exists():
            raise ValueError(f"Checkpoint metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            ckpt_metadata = json.load(f)
        
        step = ckpt_metadata['step']
        
        # Load model weights
        model_file = checkpoint_path / ckpt_metadata['model_file']
        if not model_file.exists():
            raise ValueError(f"Model weights not found: {model_file}")
        
        model_weights = mx.load(str(model_file))
        model.update(_tree_unflatten(model_weights))
        
        # Load optimizer state if optimizer provided
        if optimizer is not None:
            optimizer_file = checkpoint_path / ckpt_metadata['optimizer_file']
            if not optimizer_file.exists():
                raise ValueError(f"Optimizer state not found: {optimizer_file}")
            
            optimizer_state = mx.load(str(optimizer_file))
            optimizer.state = _tree_unflatten(optimizer_state)
        
        return model, optimizer, step
    
    def list_checkpoints(self) -> list[Dict[str, Any]]:
        """List all available checkpoints.
        
        Returns:
            List of checkpoint info dictionaries sorted by step
        """
        checkpoints = []
        
        for ckpt_dir in sorted(self.checkpoint_dir.glob("step_*")):
            metadata_file = ckpt_dir / "checkpoint.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                checkpoints.append({
                    'name': ckpt_dir.name,
                    'path': ckpt_dir,
                    'step': metadata['step'],
                })
        
        return sorted(checkpoints, key=lambda x: x['step'])
    
    def get_latest_step(self) -> Optional[int]:
        """Get the step number of the latest checkpoint.
        
        Returns:
            Latest step number, or None if no checkpoints exist
        """
        if not self.latest_file.exists():
            return None
        
        with open(self.latest_file, 'r') as f:
            latest_info = json.load(f)
        
        return latest_info.get('step')
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Remove old checkpoints, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        # Remove oldest checkpoints
        for ckpt in checkpoints[:-keep_last_n]:
            ckpt_path = Path(ckpt['path'])
            if ckpt_path.exists():
                shutil.rmtree(ckpt_path)
