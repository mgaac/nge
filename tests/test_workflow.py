"""Scaffold tests for NGE research workflow.

These tests validate the core research infrastructure to ensure:
1. Run directories are created properly with all required artifacts
2. Metrics logging is valid and monotonic
3. Checkpointing and resume work correctly
4. Configuration is deterministic
5. Reproducibility metadata is captured
"""

import json
import tempfile
import shutil
from pathlib import Path

import pytest
import mlx.core as mx
import mlx.optimizers as optim

from src.data.dataset import generated_dataset, load_dataset, save_dataset
from src.model import NGE, AggregationFn
from src.utils import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    LoggingConfig,
    load_config,
    save_config,
    set_seed,
    create_run_metadata,
    save_run_metadata,
    generate_run_name,
    CheckpointManager,
    MetricsLogger,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_config():
    """Create sample experiment configuration."""
    return ExperimentConfig(
        name="test_exp",
        model=ModelConfig(embed_dim=16, num_mp_layers=1),
        training=TrainingConfig(epochs=10, seed=42),
        data=DataConfig(
            train_path="data/train_dataset.npz",
            val_path="data/val_dataset.npz",
            test_path="data/test_dataset.npz",
        ),
        logging=LoggingConfig(use_wandb=False),
    )


@pytest.fixture
def sample_model():
    """Create sample model for testing."""
    return NGE(
        embed_dim=16,
        residual_connections=True,
        agg_fn=AggregationFn.MAX,
        num_mp_layers=1,
        dropout=0.0,
    )


def test_run_directory_creation_stable(temp_dir, sample_config):
    """Test 1: Run directory creation is stable and contains required artifacts.
    
    Validates:
    - Run directory is created
    - meta.json exists and contains required fields
    - config_resolved.yaml exists and is valid
    - checkpoints subdirectory exists
    """
    # Create run directory structure
    run_dir = temp_dir / "runs" / "test_run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    
    # Save config
    save_config(sample_config, run_dir / "config_resolved.yaml")
    
    # Create and save metadata
    metadata = create_run_metadata(sample_config.to_dict(), sample_config.training.seed)
    save_run_metadata(metadata, run_dir / "meta.json")
    
    # Validate directory structure
    assert run_dir.exists(), "Run directory not created"
    assert (run_dir / "checkpoints").exists(), "Checkpoints directory not created"
    assert (run_dir / "meta.json").exists(), "meta.json not created"
    assert (run_dir / "config_resolved.yaml").exists(), "config_resolved.yaml not created"
    
    # Validate meta.json content
    with open(run_dir / "meta.json", 'r') as f:
        meta = json.load(f)
    
    required_fields = ['timestamp', 'git', 'environment', 'seed', 'config']
    for field in required_fields:
        assert field in meta, f"Missing required field in meta.json: {field}"
    
    assert meta['seed'] == sample_config.training.seed, "Seed not recorded correctly"
    assert 'python_version' in meta['environment'], "Python version not captured"
    assert 'mlx_version' in meta['environment'], "MLX version not captured"
    
    # Validate config can be reloaded
    loaded_config = load_config(run_dir / "config_resolved.yaml")
    assert loaded_config.name == sample_config.name, "Config not saved/loaded correctly"
    assert loaded_config.model.embed_dim == sample_config.model.embed_dim


def test_metrics_jsonl_valid_and_monotonic(temp_dir):
    """Test 2: metrics.jsonl is valid JSON per line and has monotonic steps.
    
    Validates:
    - Each line is valid JSON
    - Step numbers are monotonically non-decreasing
    - Required fields are present
    """
    log_file = temp_dir / "metrics.jsonl"
    
    # Create logger and log some metrics
    logger = MetricsLogger(log_file, use_wandb=False)
    
    # Log training metrics
    for step in [0, 1, 2, 5, 10, 10, 15]:  # Note: 10 appears twice (valid)
        logger.log(step, {"loss": float(step * 0.1), "acc": 0.5}, split="train")
    
    # Log validation metrics
    logger.log(10, {"loss": 0.8, "acc": 0.6}, split="val")
    
    # Validate log file
    assert log_file.exists(), "Metrics log file not created"
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 8, f"Expected 8 log entries, got {len(lines)}"
    
    last_step = -1
    for i, line in enumerate(lines):
        # Validate JSON
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON on line {i+1}: {e}")
        
        # Validate required fields
        assert 'step' in entry, f"Missing 'step' field on line {i+1}"
        assert 'split' in entry, f"Missing 'split' field on line {i+1}"
        assert 'metrics' in entry, f"Missing 'metrics' field on line {i+1}"
        assert 'timestamp' in entry, f"Missing 'timestamp' field on line {i+1}"
        
        # Validate monotonic steps
        current_step = entry['step']
        assert current_step >= last_step, \
            f"Non-monotonic steps: {last_step} -> {current_step} on line {i+1}"
        last_step = current_step
    
    # Test logger validation method
    is_valid, error = logger.validate_log_file()
    assert is_valid, f"Log file validation failed: {error}"


def test_checkpoint_resume_correctness(temp_dir, sample_model):
    """Test 3: Resume restores step and continues correctly.
    
    Validates:
    - Checkpoint saves model and optimizer state
    - Loading checkpoint restores correct step
    - Model parameters are restored correctly
    - Latest checkpoint marker works
    """
    checkpoint_dir = temp_dir / "checkpoints"
    manager = CheckpointManager(checkpoint_dir)
    
    # Create optimizer
    optimizer = optim.Adam(learning_rate=1e-3)
    
    # Do a fake training step to create optimizer state
    # (needed because optimizer state is only initialized after first update)
    dummy_grads = {k: mx.zeros_like(v) for k, v in sample_model.parameters().items()}
    optimizer.update(sample_model, dummy_grads)
    mx.eval(sample_model.parameters(), optimizer.state)
    
    # Save initial parameters
    initial_params = {k: v.copy() for k, v in sample_model.parameters().items()}
    
    # Save checkpoint at step 100
    ckpt_path = manager.save(sample_model, optimizer, step=100, metadata={'test': 'value'})
    
    assert ckpt_path.exists(), "Checkpoint directory not created"
    assert (ckpt_path / "model.safetensors").exists(), "Model weights not saved"
    assert (ckpt_path / "optimizer.safetensors").exists(), "Optimizer state not saved"
    assert (ckpt_path / "checkpoint.json").exists(), "Checkpoint metadata not saved"
    
    # Modify model parameters
    for param in sample_model.parameters().values():
        param[:] = param + 1.0
    mx.eval(sample_model.parameters())
    
    # Verify parameters changed
    for key, initial_val in initial_params.items():
        current_val = sample_model.parameters()[key]
        assert not mx.allclose(initial_val, current_val), \
            f"Parameter {key} should have changed"
    
    # Load checkpoint
    new_model = NGE(
        embed_dim=16,
        residual_connections=True,
        agg_fn=AggregationFn.MAX,
        num_mp_layers=1,
        dropout=0.0,
    )
    new_optimizer = optim.Adam(learning_rate=1e-3)
    
    loaded_model, loaded_optimizer, loaded_step = manager.load(
        new_model, new_optimizer
    )
    
    # Validate step restored correctly
    assert loaded_step == 100, f"Expected step 100, got {loaded_step}"
    
    # Validate parameters restored correctly
    for key, initial_val in initial_params.items():
        loaded_val = loaded_model.parameters()[key]
        assert mx.allclose(initial_val, loaded_val, atol=1e-6), \
            f"Parameter {key} not restored correctly"
    
    # Test latest checkpoint marker
    latest_step = manager.get_latest_step()
    assert latest_step == 100, f"Latest step should be 100, got {latest_step}"


def test_dataset_roundtrip_preserves_prim_targets(temp_dir):
    """Generated datasets must serialize and restore Prim supervision."""
    dataset = generated_dataset(num_graphs=2, num_nodes=6, p=0.5, m=2)
    path = temp_dir / "dataset.npz"
    save_dataset(dataset, path)
    loaded = load_dataset(path)

    assert len(loaded) == len(dataset)
    sample = loaded[0]
    assert "prim_state_targets" in sample
    assert "prim_key_targets" in sample
    assert "prim_predecessor_targets" in sample
    assert sample["prim_state_targets"].shape[0] == sample["prim_key_targets"].shape[0]
    assert sample["prim_key_targets"].shape[0] == sample["prim_predecessor_targets"].shape[0]


def test_model_forward_returns_prim_outputs():
    """Model forward pass must expose Prim heads and termination logits."""
    dataset = generated_dataset(num_graphs=1, num_nodes=6, p=0.5, m=2)
    graph = dataset[0]
    model = NGE(
        embed_dim=8,
        residual_connections=True,
        agg_fn=AggregationFn.MAX,
        num_mp_layers=1,
        dropout=0.0,
    )

    num_nodes = graph["num_nodes"]
    previous_hidden = mx.zeros([num_nodes, model.processor_embed_dim])
    node_algo_features = mx.stack(
        [
            graph["bfs_state_targets"][0],
            graph["bf_distance_targets"][0],
            graph["prim_state_targets"][0],
            graph["prim_key_targets"][0],
        ],
        axis=1,
    )
    model_input = (
        mx.concatenate([previous_hidden, node_algo_features], axis=1),
        graph["edge_matrix"],
    )

    bfs_output, bf_output, prim_output, termination_probs, processed = model(model_input)
    bf_distance, bf_predecessor = bf_output
    prim_state, prim_key, prim_predecessor = prim_output

    assert bfs_output.shape == (num_nodes,)
    assert bf_distance.shape == (num_nodes,)
    assert bf_predecessor.shape == (num_nodes, num_nodes)
    assert prim_state.shape == (num_nodes,)
    assert prim_key.shape == (num_nodes,)
    assert prim_predecessor.shape == (num_nodes, num_nodes)
    assert set(termination_probs.keys()) == {"bf", "bfs", "prim"}
    assert processed.shape == (num_nodes, model.processor_embed_dim)
    
    # Save another checkpoint
    manager.save(sample_model, optimizer, step=200)
    
    latest_step = manager.get_latest_step()
    assert latest_step == 200, f"Latest step should be updated to 200, got {latest_step}"
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    assert len(checkpoints) == 2, f"Expected 2 checkpoints, got {len(checkpoints)}"
    assert checkpoints[0]['step'] == 100
    assert checkpoints[1]['step'] == 200


def test_config_resolution_deterministic(temp_dir, sample_config):
    """Test 4: Config resolution is deterministic.
    
    Validates:
    - Same config file produces same resolved config
    - Config can be saved and reloaded identically
    - No non-deterministic fields in resolved config
    """
    config_path = temp_dir / "config.yaml"
    
    # Save config
    save_config(sample_config, config_path)
    
    # Load config multiple times
    loaded_1 = load_config(config_path)
    loaded_2 = load_config(config_path)
    loaded_3 = load_config(config_path)
    
    # Convert to dicts for comparison
    dict_1 = loaded_1.to_dict()
    dict_2 = loaded_2.to_dict()
    dict_3 = loaded_3.to_dict()
    
    # Validate determinism
    assert dict_1 == dict_2, "Config loading not deterministic (1 vs 2)"
    assert dict_2 == dict_3, "Config loading not deterministic (2 vs 3)"
    
    # Validate all fields match original
    original_dict = sample_config.to_dict()
    assert dict_1 == original_dict, "Loaded config doesn't match original"
    
    # Save again and compare
    config_path_2 = temp_dir / "config2.yaml"
    save_config(loaded_1, config_path_2)
    
    # Read both files and compare content
    with open(config_path, 'r') as f1, open(config_path_2, 'r') as f2:
        content_1 = f1.read()
        content_2 = f2.read()
    
    assert content_1 == content_2, "Saved configs are not identical"


def test_seeding_called_and_recorded(temp_dir, sample_config):
    """Test 5: Seeding function is called and recorded in meta.json.
    
    Validates:
    - set_seed is called with correct seed
    - Seed is recorded in metadata
    - RNG state is deterministic after seeding
    """
    seed = sample_config.training.seed
    
    # Set seed
    set_seed(seed)
    
    # Generate random numbers
    random_1 = mx.random.uniform(shape=(10,))
    
    # Reset seed and generate again
    set_seed(seed)
    random_2 = mx.random.uniform(shape=(10,))
    
    # Validate determinism
    assert mx.allclose(random_1, random_2, atol=1e-6), \
        "RNG not deterministic after seeding"
    
    # Create metadata
    metadata = create_run_metadata(sample_config.to_dict(), seed)
    
    # Validate seed recorded
    assert 'seed' in metadata, "Seed not in metadata"
    assert metadata['seed'] == seed, f"Seed {metadata['seed']} doesn't match {seed}"
    
    # Save and reload metadata
    meta_path = temp_dir / "meta.json"
    save_run_metadata(metadata, meta_path)
    
    with open(meta_path, 'r') as f:
        loaded_meta = json.load(f)
    
    assert loaded_meta['seed'] == seed, "Seed not preserved in saved metadata"
    
    # Validate metadata has config
    assert 'config' in loaded_meta, "Config not in metadata"
    assert loaded_meta['config']['training']['seed'] == seed, \
        "Seed in config doesn't match"


def test_run_name_generation():
    """Test run name generation is deterministic and includes git info."""
    exp_name = "test_experiment"
    
    # Without git info
    run_name_1 = generate_run_name(exp_name, git_info=None)
    assert exp_name in run_name_1, "Experiment name not in run name"
    
    # With git info (clean)
    git_info_clean = {
        'commit_sha': 'abc123def456',
        'is_dirty': False,
    }
    run_name_2 = generate_run_name(exp_name, git_info=git_info_clean)
    assert 'abc123d' in run_name_2, "Git SHA not in run name"
    assert exp_name in run_name_2, "Experiment name not in run name"
    assert 'dirty' not in run_name_2, "Clean repo should not have 'dirty' in name"
    
    # With git info (dirty)
    git_info_dirty = {
        'commit_sha': 'abc123def456',
        'is_dirty': True,
        'dirty_diff_hash': 'xyz789',
    }
    run_name_3 = generate_run_name(exp_name, git_info=git_info_dirty)
    assert 'abc123d' in run_name_3, "Git SHA not in run name"
    assert 'dirty' in run_name_3, "Dirty repo should have 'dirty' in name"
    assert exp_name in run_name_3, "Experiment name not in run name"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
