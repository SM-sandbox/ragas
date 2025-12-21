"""
Unit tests for config file validation.

Ensures all config files (checkpoint, experiment) have required fields
with correct values for reproducibility and consistency.
"""

import pytest
import yaml
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "config"
CHECKPOINT_CONFIG = CONFIG_DIR / "checkpoint_config.yaml"
EXPERIMENT_CONFIG = CONFIG_DIR / "experiment_config.yaml"


def load_config(path: Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# CHECKPOINT CONFIG TESTS
# =============================================================================

class TestCheckpointConfig:
    """Tests for checkpoint_config.yaml - the locked gold standard."""
    
    @pytest.fixture
    def config(self):
        """Load checkpoint config."""
        return load_config(CHECKPOINT_CONFIG)
    
    def test_config_exists(self):
        """Checkpoint config file must exist."""
        assert CHECKPOINT_CONFIG.exists(), f"Missing: {CHECKPOINT_CONFIG}"
    
    def test_has_schema_version(self, config):
        """Must have schema_version."""
        assert "schema_version" in config
        assert config["schema_version"] == "1.0"
    
    def test_has_config_type(self, config):
        """Must have config_type = checkpoint."""
        assert config.get("config_type") == "checkpoint"
    
    def test_has_client(self, config):
        """Must have client."""
        assert "client" in config
        assert config["client"] == "BFAI"
    
    # -------------------------------------------------------------------------
    # GENERATOR CONFIG
    # -------------------------------------------------------------------------
    
    def test_has_generator_section(self, config):
        """Must have generator section."""
        assert "generator" in config
        assert isinstance(config["generator"], dict)
    
    def test_generator_has_model(self, config):
        """Generator must have model."""
        assert "model" in config["generator"]
        assert config["generator"]["model"] == "gemini-3-flash-preview"
    
    def test_generator_has_temperature(self, config):
        """Generator must have temperature = 0.0."""
        assert "temperature" in config["generator"]
        assert config["generator"]["temperature"] == 0.0
    
    def test_generator_has_seed(self, config):
        """Generator must have seed = 42."""
        assert "seed" in config["generator"]
        assert config["generator"]["seed"] == 42
    
    def test_generator_has_reasoning_effort(self, config):
        """Generator must have reasoning_effort = low."""
        assert "reasoning_effort" in config["generator"]
        assert config["generator"]["reasoning_effort"] == "low"
    
    def test_generator_reasoning_effort_valid(self, config):
        """Generator reasoning_effort must be low or high (no medium in Gemini 3)."""
        effort = config["generator"]["reasoning_effort"]
        assert effort in ["low", "high"], f"Invalid reasoning_effort: {effort}"
    
    def test_generator_has_max_output_tokens(self, config):
        """Generator must have max_output_tokens."""
        assert "max_output_tokens" in config["generator"]
        assert config["generator"]["max_output_tokens"] == 8192
    
    # -------------------------------------------------------------------------
    # JUDGE CONFIG
    # -------------------------------------------------------------------------
    
    def test_has_judge_section(self, config):
        """Must have judge section."""
        assert "judge" in config
        assert isinstance(config["judge"], dict)
    
    def test_judge_has_model(self, config):
        """Judge must have model."""
        assert "model" in config["judge"]
        assert config["judge"]["model"] == "gemini-3-flash-preview"
    
    def test_judge_has_temperature(self, config):
        """Judge must have temperature = 0.0."""
        assert "temperature" in config["judge"]
        assert config["judge"]["temperature"] == 0.0
    
    def test_judge_has_seed(self, config):
        """Judge must have seed = 42."""
        assert "seed" in config["judge"]
        assert config["judge"]["seed"] == 42
    
    def test_judge_has_reasoning_effort(self, config):
        """Judge must have reasoning_effort = low."""
        assert "reasoning_effort" in config["judge"]
        assert config["judge"]["reasoning_effort"] == "low"
    
    def test_judge_reasoning_effort_valid(self, config):
        """Judge reasoning_effort must be low or high (no medium in Gemini 3)."""
        effort = config["judge"]["reasoning_effort"]
        assert effort in ["low", "high"], f"Invalid reasoning_effort: {effort}"
    
    # -------------------------------------------------------------------------
    # RETRIEVAL CONFIG
    # -------------------------------------------------------------------------
    
    def test_has_retrieval_section(self, config):
        """Must have retrieval section."""
        assert "retrieval" in config
    
    def test_retrieval_has_recall_k(self, config):
        """Retrieval must have recall_k."""
        assert "recall_k" in config["retrieval"]
        assert config["retrieval"]["recall_k"] == 100
    
    def test_retrieval_has_precision_k(self, config):
        """Retrieval must have precision_k."""
        assert "precision_k" in config["retrieval"]
        assert config["retrieval"]["precision_k"] == 25
    
    # -------------------------------------------------------------------------
    # EXECUTION CONFIG
    # -------------------------------------------------------------------------
    
    def test_has_execution_section(self, config):
        """Must have execution section."""
        assert "execution" in config
    
    def test_execution_has_workers(self, config):
        """Execution must have workers = 100 (smart throttler handles concurrency)."""
        assert "workers" in config["execution"]
        assert config["execution"]["workers"] == 100


# =============================================================================
# EXPERIMENT CONFIG TESTS
# =============================================================================

class TestExperimentConfig:
    """Tests for experiment_config.yaml - the flexible experiment configuration."""
    
    @pytest.fixture
    def config(self):
        """Load experiment config."""
        return load_config(EXPERIMENT_CONFIG)
    
    def test_config_exists(self):
        """Experiment config file must exist."""
        assert EXPERIMENT_CONFIG.exists(), f"Missing: {EXPERIMENT_CONFIG}"
    
    def test_has_config_type(self, config):
        """Must have config_type = experiment."""
        assert config.get("config_type") == "experiment"
    
    # -------------------------------------------------------------------------
    # GENERATOR CONFIG
    # -------------------------------------------------------------------------
    
    def test_has_generator_section(self, config):
        """Must have generator section."""
        assert "generator" in config
    
    def test_generator_has_model(self, config):
        """Generator must have model."""
        assert "model" in config["generator"]
    
    def test_generator_has_temperature(self, config):
        """Generator must have temperature."""
        assert "temperature" in config["generator"]
    
    def test_generator_has_seed(self, config):
        """Generator must have seed."""
        assert "seed" in config["generator"]
    
    def test_generator_has_reasoning_effort(self, config):
        """Generator must have reasoning_effort."""
        assert "reasoning_effort" in config["generator"]
        assert config["generator"]["reasoning_effort"] in ["low", "high"]
    
    def test_generator_has_max_output_tokens(self, config):
        """Generator must have max_output_tokens."""
        assert "max_output_tokens" in config["generator"]
    
    # -------------------------------------------------------------------------
    # JUDGE CONFIG
    # -------------------------------------------------------------------------
    
    def test_has_judge_section(self, config):
        """Must have judge section."""
        assert "judge" in config
    
    def test_judge_has_model(self, config):
        """Judge must have model."""
        assert "model" in config["judge"]
    
    def test_judge_has_temperature(self, config):
        """Judge must have temperature."""
        assert "temperature" in config["judge"]
    
    def test_judge_has_seed(self, config):
        """Judge must have seed."""
        assert "seed" in config["judge"]
    
    def test_judge_has_reasoning_effort(self, config):
        """Judge must have reasoning_effort."""
        assert "reasoning_effort" in config["judge"]
        assert config["judge"]["reasoning_effort"] in ["low", "high"]
    
    # -------------------------------------------------------------------------
    # EXECUTION CONFIG
    # -------------------------------------------------------------------------
    
    def test_execution_has_workers(self, config):
        """Execution must have workers = 100 (smart throttler handles concurrency)."""
        assert "workers" in config["execution"]
        assert config["execution"]["workers"] == 100


# =============================================================================
# CROSS-CONFIG CONSISTENCY TESTS
# =============================================================================

class TestConfigConsistency:
    """Tests for consistency across all config files."""
    
    @pytest.fixture
    def all_configs(self):
        """Load all configs."""
        return {
            "checkpoint": load_config(CHECKPOINT_CONFIG),
            "experiment": load_config(EXPERIMENT_CONFIG),
        }
    
    def test_all_have_generator_section(self, all_configs):
        """All configs must have generator section."""
        for name, config in all_configs.items():
            assert "generator" in config, f"{name} missing generator section"
    
    def test_all_have_judge_section(self, all_configs):
        """All configs must have judge section."""
        for name, config in all_configs.items():
            assert "judge" in config, f"{name} missing judge section"
    
    def test_all_generators_have_required_fields(self, all_configs):
        """All generator configs must have required fields."""
        required = ["model", "temperature", "seed", "reasoning_effort", "max_output_tokens"]
        for name, config in all_configs.items():
            for field in required:
                assert field in config["generator"], f"{name} generator missing {field}"
    
    def test_all_judges_have_required_fields(self, all_configs):
        """All judge configs must have required fields."""
        required = ["model", "temperature", "seed", "reasoning_effort"]
        for name, config in all_configs.items():
            for field in required:
                assert field in config["judge"], f"{name} judge missing {field}"
    
    def test_all_have_temperature_zero(self, all_configs):
        """All configs must have temperature = 0.0 for reproducibility."""
        for name, config in all_configs.items():
            assert config["generator"]["temperature"] == 0.0, f"{name} generator temp != 0.0"
            assert config["judge"]["temperature"] == 0.0, f"{name} judge temp != 0.0"
    
    def test_all_have_seed_42(self, all_configs):
        """All configs must have seed = 42 for reproducibility."""
        for name, config in all_configs.items():
            assert config["generator"]["seed"] == 42, f"{name} generator seed != 42"
            assert config["judge"]["seed"] == 42, f"{name} judge seed != 42"
    
    def test_all_have_reasoning_effort_low(self, all_configs):
        """All configs must have reasoning_effort = low."""
        for name, config in all_configs.items():
            assert config["generator"]["reasoning_effort"] == "low", f"{name} generator reasoning != low"
            assert config["judge"]["reasoning_effort"] == "low", f"{name} judge reasoning != low"
    
    def test_all_have_workers_100(self, all_configs):
        """All configs must have workers = 100 (smart throttler handles concurrency)."""
        for name, config in all_configs.items():
            assert config["execution"]["workers"] == 100, f"{name} workers != 100"
    
    def test_checkpoint_matches_gold_standard(self, all_configs):
        """Checkpoint config must use gold standard models."""
        checkpoint = all_configs["checkpoint"]
        assert checkpoint["generator"]["model"] == "gemini-3-flash-preview"
        assert checkpoint["judge"]["model"] == "gemini-3-flash-preview"


# =============================================================================
# REQUIRED FIELDS SCHEMA TEST
# =============================================================================

class TestRequiredFieldsSchema:
    """Test that all required fields are documented and present."""
    
    GENERATOR_REQUIRED = [
        "model",
        "temperature",
        "seed",
        "reasoning_effort",
        "max_output_tokens",
    ]
    
    JUDGE_REQUIRED = [
        "model",
        "temperature",
        "seed",
        "reasoning_effort",
    ]
    
    RETRIEVAL_REQUIRED = [
        "recall_k",
        "precision_k",
    ]
    
    EXECUTION_REQUIRED = [
        "workers",
    ]
    
    @pytest.fixture
    def checkpoint(self):
        return load_config(CHECKPOINT_CONFIG)
    
    def test_generator_has_all_required(self, checkpoint):
        """Generator must have all required fields."""
        for field in self.GENERATOR_REQUIRED:
            assert field in checkpoint["generator"], f"Generator missing required field: {field}"
    
    def test_judge_has_all_required(self, checkpoint):
        """Judge must have all required fields."""
        for field in self.JUDGE_REQUIRED:
            assert field in checkpoint["judge"], f"Judge missing required field: {field}"
    
    def test_retrieval_has_all_required(self, checkpoint):
        """Retrieval must have all required fields."""
        for field in self.RETRIEVAL_REQUIRED:
            assert field in checkpoint["retrieval"], f"Retrieval missing required field: {field}"
    
    def test_execution_has_all_required(self, checkpoint):
        """Execution must have all required fields."""
        for field in self.EXECUTION_REQUIRED:
            assert field in checkpoint["execution"], f"Execution missing required field: {field}"


# =============================================================================
# SEED PASSTHROUGH TESTS
# =============================================================================

class TestSeedPassthrough:
    """Tests to verify seed=42 is passed through the code to Gemini API."""
    
    def test_gemini_client_generate_accepts_seed(self):
        """generate() function must accept seed parameter."""
        from lib.clients.gemini_client import generate
        import inspect
        sig = inspect.signature(generate)
        assert "seed" in sig.parameters, "generate() must accept seed parameter"
    
    def test_gemini_client_generate_for_judge_accepts_seed(self):
        """generate_for_judge() must accept seed parameter."""
        from lib.clients.gemini_client import generate_for_judge
        import inspect
        sig = inspect.signature(generate_for_judge)
        assert "seed" in sig.parameters, "generate_for_judge() must accept seed parameter"
    
    def test_gemini_client_generate_for_rag_accepts_seed(self):
        """generate_for_rag() must accept seed parameter."""
        from lib.clients.gemini_client import generate_for_rag
        import inspect
        sig = inspect.signature(generate_for_rag)
        assert "seed" in sig.parameters, "generate_for_rag() must accept seed parameter"
    
    def test_config_loader_returns_judge_seed(self):
        """Config loader must return judge seed = 42."""
        from lib.core.config_loader import load_config, get_judge_config
        config = load_config(config_type="checkpoint")
        judge_config = get_judge_config(config)
        assert "seed" in judge_config, "Judge config must have seed"
        assert judge_config["seed"] == 42, "Judge seed must be 42"
    
    def test_config_loader_returns_generator_seed(self):
        """Config loader must return generator seed = 42."""
        from lib.core.config_loader import load_config, get_generator_config
        config = load_config(config_type="checkpoint")
        gen_config = get_generator_config(config)
        assert "seed" in gen_config, "Generator config must have seed"
        assert gen_config["seed"] == 42, "Generator seed must be 42"
    
    def test_all_configs_have_seed_42(self):
        """All config files must have seed=42 for both generator and judge."""
        configs = [
            ("checkpoint", CHECKPOINT_CONFIG),
            ("experiment", EXPERIMENT_CONFIG),
        ]
        for name, path in configs:
            config = load_config(path)
            assert config["generator"]["seed"] == 42, f"{name} generator seed must be 42"
            assert config["judge"]["seed"] == 42, f"{name} judge seed must be 42"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
