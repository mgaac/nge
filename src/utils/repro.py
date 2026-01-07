"""Reproducibility utilities for capturing experiment context.

This module provides functions for:
- Seeding RNGs deterministically
- Capturing git information (commit SHA, dirty diff)
- Recording environment metadata (Python version, MLX version, etc.)
"""

import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import mlx.core as mx


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    mx.random.seed(seed)


def get_git_info(repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get git repository information.
    
    Args:
        repo_path: Path to git repository (defaults to current directory)
        
    Returns:
        Dictionary with git information:
        - commit_sha: Current commit SHA
        - branch: Current branch name
        - is_dirty: Whether repository has uncommitted changes
        - dirty_diff_hash: Hash of uncommitted changes (if any)
        - remote_url: Remote URL (if configured)
    """
    if repo_path is None:
        repo_path = Path.cwd()
    
    git_info = {
        'commit_sha': None,
        'branch': None,
        'is_dirty': False,
        'dirty_diff_hash': None,
        'remote_url': None,
    }
    
    try:
        # Get current commit SHA
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        git_info['commit_sha'] = result.stdout.strip()
        
        # Get current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        git_info['branch'] = result.stdout.strip()
        
        # Check if repository is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        git_info['is_dirty'] = len(result.stdout.strip()) > 0
        
        # If dirty, compute hash of uncommitted changes
        if git_info['is_dirty']:
            result = subprocess.run(
                ['git', 'diff', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            diff_content = result.stdout
            git_info['dirty_diff_hash'] = hashlib.sha256(diff_content.encode()).hexdigest()[:8]
        
        # Get remote URL
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False  # Don't fail if remote not configured
        )
        if result.returncode == 0:
            git_info['remote_url'] = result.stdout.strip()
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not a git repository
        pass
    
    return git_info


def get_env_info() -> Dict[str, Any]:
    """Get environment information.
    
    Returns:
        Dictionary with environment information:
        - python_version: Python version string
        - mlx_version: MLX version string
        - platform: OS platform
        - hostname: Machine hostname
    """
    import platform
    
    return {
        'python_version': sys.version,
        'mlx_version': mx.__version__,
        'platform': platform.platform(),
        'hostname': platform.node(),
    }


def create_run_metadata(config_dict: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Create comprehensive run metadata.
    
    Args:
        config_dict: Experiment configuration dictionary
        seed: Random seed used for the run
        
    Returns:
        Dictionary with complete run metadata including:
        - timestamp
        - git info
        - environment info
        - seed
        - config
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'git': get_git_info(),
        'environment': get_env_info(),
        'seed': seed,
        'config': config_dict,
    }
    
    return metadata


def save_run_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """Save run metadata to JSON file.
    
    Args:
        metadata: Metadata dictionary to save
        output_path: Path where to save metadata JSON
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def generate_run_name(exp_name: str, git_info: Optional[Dict[str, Any]] = None) -> str:
    """Generate deterministic run directory name.
    
    Format: YYYYMMDD-HHMMSS_<gitsha>_<expname>
    If git info not available: YYYYMMDD-HHMMSS_<expname>
    
    Args:
        exp_name: Experiment name from config
        git_info: Optional git information dictionary
        
    Returns:
        Run directory name string
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if git_info and git_info.get('commit_sha'):
        git_sha = git_info['commit_sha'][:7]
        if git_info.get('is_dirty'):
            git_sha += f"-dirty{git_info.get('dirty_diff_hash', '')}"
        return f"{timestamp}_{git_sha}_{exp_name}"
    else:
        return f"{timestamp}_{exp_name}"
