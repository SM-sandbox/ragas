"""
Config Loader for RAG Evaluation Suite

Loads and validates configuration files for checkpoint, run, and experiment evaluations.
This is the single source of truth for config loading across all runners.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

# Config file paths
CHECKPOINT_CONFIG_PATH = CONFIG_DIR / "checkpoint_config.yaml"
RUN_CONFIG_PATH = CONFIG_DIR / "run_config.yaml"
EXPERIMENT_CONFIG_PATH = CONFIG_DIR / "experiment_config.yaml"


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def validate_config(config: Dict[str, Any], config_type: str) -> None:
    """
    Validate that config has all required fields.
    
    Raises ConfigValidationError if validation fails.
    """
    errors = []
    
    # Required top-level sections
    required_sections = ["generator", "judge", "retrieval", "execution"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    if errors:
        raise ConfigValidationError(f"Config validation failed: {errors}")
    
    # Generator required fields
    gen_required = ["model", "temperature", "seed", "reasoning_effort", "max_output_tokens"]
    for field in gen_required:
        if field not in config.get("generator", {}):
            errors.append(f"Generator missing required field: {field}")
    
    # Judge required fields
    judge_required = ["model", "temperature", "seed", "reasoning_effort"]
    for field in judge_required:
        if field not in config.get("judge", {}):
            errors.append(f"Judge missing required field: {field}")
    
    # Validate reasoning_effort values
    gen_effort = config.get("generator", {}).get("reasoning_effort", "")
    if gen_effort not in ["low", "high"]:
        errors.append(f"Generator reasoning_effort must be 'low' or 'high', got: {gen_effort}")
    
    judge_effort = config.get("judge", {}).get("reasoning_effort", "")
    if judge_effort not in ["low", "high"]:
        errors.append(f"Judge reasoning_effort must be 'low' or 'high', got: {judge_effort}")
    
    # Retrieval required fields
    ret_required = ["recall_k", "precision_k"]
    for field in ret_required:
        if field not in config.get("retrieval", {}):
            errors.append(f"Retrieval missing required field: {field}")
    
    # Execution required fields
    exec_required = ["workers"]
    for field in exec_required:
        if field not in config.get("execution", {}):
            errors.append(f"Execution missing required field: {field}")
    
    if errors:
        raise ConfigValidationError(f"Config validation failed for {config_type}: {errors}")


def load_checkpoint_config() -> Dict[str, Any]:
    """
    Load the checkpoint configuration.
    
    This is the locked gold standard config - should not be modified.
    """
    config = load_yaml(CHECKPOINT_CONFIG_PATH)
    validate_config(config, "checkpoint")
    return config


def load_run_config() -> Dict[str, Any]:
    """
    Load the run configuration.
    
    This is the default config for full evaluation runs.
    """
    config = load_yaml(RUN_CONFIG_PATH)
    validate_config(config, "run")
    return config


def load_experiment_config() -> Dict[str, Any]:
    """
    Load the experiment configuration.
    
    This is the flexible config for ad-hoc experiments.
    """
    config = load_yaml(EXPERIMENT_CONFIG_PATH)
    validate_config(config, "experiment")
    return config


def load_config(config_type: str = "run", config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration by type or from a custom path.
    
    Args:
        config_type: "checkpoint", "run", or "experiment"
        config_path: Optional custom config file path
        
    Returns:
        Validated configuration dictionary
    """
    if config_path:
        config = load_yaml(config_path)
        validate_config(config, "custom")
        return config
    
    if config_type == "checkpoint":
        return load_checkpoint_config()
    elif config_type == "run":
        return load_run_config()
    elif config_type == "experiment":
        return load_experiment_config()
    else:
        raise ValueError(f"Unknown config_type: {config_type}. Must be checkpoint|run|experiment")


def get_generator_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract generator config from full config."""
    return config.get("generator", {})


def get_judge_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract judge config from full config."""
    return config.get("judge", {})


def get_retrieval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract retrieval config from full config."""
    return config.get("retrieval", {})


def get_execution_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract execution config from full config."""
    return config.get("execution", {})


def merge_config_with_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge config with CLI overrides.
    
    Args:
        config: Base configuration
        overrides: Dictionary of overrides (e.g., {"generator": {"model": "new-model"}})
        
    Returns:
        Merged configuration
    """
    import copy
    merged = copy.deepcopy(config)
    
    for key, value in overrides.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value
    
    return merged
