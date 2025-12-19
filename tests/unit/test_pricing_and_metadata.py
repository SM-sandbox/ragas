"""
Unit tests for pricing, cost estimation, and metadata features.

Tests:
- Pricing config loading
- Cost calculation
- Orchestrator metadata
- 6-bucket quality breakdown
- Directory naming convention
"""

import pytest
import yaml
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "config"
PRICING_CONFIG = CONFIG_DIR / "model_pricing.yaml"


# =============================================================================
# PRICING CONFIG TESTS
# =============================================================================

class TestPricingConfig:
    """Tests for model_pricing.yaml configuration."""
    
    @pytest.fixture
    def pricing(self):
        """Load pricing config."""
        with open(PRICING_CONFIG) as f:
            return yaml.safe_load(f)
    
    def test_pricing_config_exists(self):
        """Pricing config file must exist."""
        assert PRICING_CONFIG.exists(), f"Missing: {PRICING_CONFIG}"
    
    def test_has_schema_version(self, pricing):
        """Must have schema_version."""
        assert "schema_version" in pricing
        assert pricing["schema_version"] == "1.0"
    
    def test_has_models_section(self, pricing):
        """Must have models section."""
        assert "models" in pricing
        assert isinstance(pricing["models"], dict)
    
    def test_has_gemini_3_flash(self, pricing):
        """Must have gemini-3-flash-preview pricing."""
        assert "gemini-3-flash-preview" in pricing["models"]
    
    def test_has_gemini_3_pro(self, pricing):
        """Must have gemini-3-pro-preview pricing."""
        assert "gemini-3-pro-preview" in pricing["models"]
    
    def test_model_has_required_fields(self, pricing):
        """Each model must have input, output, thinking, cached pricing."""
        required_fields = ["input_per_1m", "output_per_1m", "thinking_per_1m", "cached_per_1m"]
        for model_name, model_pricing in pricing["models"].items():
            for field in required_fields:
                assert field in model_pricing, f"{model_name} missing {field}"
    
    def test_flash_cheaper_than_pro(self, pricing):
        """Flash should be cheaper than Pro."""
        flash = pricing["models"]["gemini-3-flash-preview"]
        pro = pricing["models"]["gemini-3-pro-preview"]
        assert flash["input_per_1m"] < pro["input_per_1m"]
        assert flash["output_per_1m"] < pro["output_per_1m"]
    
    def test_has_default_fallback(self, pricing):
        """Must have default fallback pricing."""
        assert "default" in pricing


# =============================================================================
# PRICING MODULE TESTS
# =============================================================================

class TestPricingModule:
    """Tests for lib/core/pricing.py module."""
    
    def test_get_model_pricing_flash(self):
        """get_model_pricing returns correct rates for flash."""
        from lib.core.pricing import get_model_pricing
        pricing = get_model_pricing("gemini-3-flash-preview")
        assert pricing["input_per_1m"] == 0.10
        assert pricing["output_per_1m"] == 0.40
    
    def test_get_model_pricing_pro(self):
        """get_model_pricing returns correct rates for pro."""
        from lib.core.pricing import get_model_pricing
        pricing = get_model_pricing("gemini-3-pro-preview")
        assert pricing["input_per_1m"] == 1.25
        assert pricing["output_per_1m"] == 10.00
    
    def test_get_model_pricing_unknown_returns_default(self):
        """Unknown model returns default pricing."""
        from lib.core.pricing import get_model_pricing
        pricing = get_model_pricing("unknown-model-xyz")
        assert "input_per_1m" in pricing
        assert "output_per_1m" in pricing
    
    def test_calculate_token_cost(self):
        """calculate_token_cost computes correct cost."""
        from lib.core.pricing import calculate_token_cost, get_model_pricing
        
        tokens = {"prompt": 1000, "completion": 100, "thinking": 0, "cached": 0}
        pricing = get_model_pricing("gemini-3-flash-preview")
        
        cost = calculate_token_cost(tokens, pricing)
        
        # 1000 input tokens at $0.10/1M = $0.0001
        # 100 output tokens at $0.40/1M = $0.00004
        expected = 0.0001 + 0.00004
        assert abs(cost - expected) < 0.0000001
    
    def test_calculate_token_cost_with_thinking(self):
        """calculate_token_cost includes thinking tokens."""
        from lib.core.pricing import calculate_token_cost, get_model_pricing
        
        tokens = {"prompt": 1000, "completion": 100, "thinking": 500, "cached": 0}
        pricing = get_model_pricing("gemini-3-flash-preview")
        
        cost = calculate_token_cost(tokens, pricing)
        
        # Should be more than without thinking
        tokens_no_thinking = {"prompt": 1000, "completion": 100, "thinking": 0, "cached": 0}
        cost_no_thinking = calculate_token_cost(tokens_no_thinking, pricing)
        
        assert cost > cost_no_thinking


# =============================================================================
# ORCHESTRATOR METADATA TESTS
# =============================================================================

class TestOrchestratorMetadata:
    """Tests for orchestrator metadata in cloud mode."""
    
    def test_cloud_constants_defined(self):
        """Cloud orchestrator constants must be defined."""
        from lib.core.evaluator import (
            CLOUD_RUN_URL, CLOUD_PROJECT_ID, CLOUD_SERVICE, 
            CLOUD_REGION, CLOUD_ENVIRONMENT
        )
        assert CLOUD_RUN_URL is not None
        assert CLOUD_PROJECT_ID == "bfai-prod"
        assert CLOUD_SERVICE == "bfai-api"
        assert CLOUD_REGION == "us-east1"
        assert CLOUD_ENVIRONMENT == "production"
    
    def test_cloud_url_is_valid(self):
        """Cloud Run URL must be valid HTTPS URL."""
        from lib.core.evaluator import CLOUD_RUN_URL
        assert CLOUD_RUN_URL.startswith("https://")
        assert ".run.app" in CLOUD_RUN_URL


# =============================================================================
# DIRECTORY NAMING TESTS
# =============================================================================

class TestDirectoryNaming:
    """Tests for directory naming convention."""
    
    def test_naming_format_documented(self):
        """Directory naming format should be in evaluator docstring or comments."""
        from lib.core.evaluator import GoldEvaluator
        import inspect
        source = inspect.getsource(GoldEvaluator._create_run_directory)
        # Check that the format is documented
        assert "local|cloud" in source or "local" in source
        assert "p{K}" in source or "precision" in source.lower()


# =============================================================================
# TOKEN TRACKING TESTS
# =============================================================================

class TestTokenTracking:
    """Tests for separate generator and judge token tracking."""
    
    def test_generate_for_judge_returns_metadata(self):
        """generate_for_judge with return_metadata=True returns tokens."""
        from lib.clients.gemini_client import generate_for_judge
        import inspect
        sig = inspect.signature(generate_for_judge)
        assert "return_metadata" in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
