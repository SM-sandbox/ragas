"""
Unit tests for core/models.py - Model Registry Integration.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")

import pytest
from lib.utils.models import (
    get_model,
    get_approved_models,
    get_models_with_thinking,
    get_thinking_config,
    validate_model,
    ModelInfo,
)


class TestGetModel:
    """Tests for get_model function."""
    
    def test_get_valid_model(self):
        """Should return ModelInfo for valid model ID."""
        model = get_model("gemini-2.5-flash")
        assert model is not None
        assert model.id == "gemini-2.5-flash"
        assert model.family == "gemini-2.5"
    
    def test_get_invalid_model_returns_none(self):
        """Should return None for invalid model ID."""
        model = get_model("invalid-model-id")
        assert model is None
    
    def test_get_model_has_all_fields(self):
        """ModelInfo should have all expected fields."""
        model = get_model("gemini-2.5-flash")
        assert model is not None
        
        # Check all fields exist
        assert hasattr(model, "id")
        assert hasattr(model, "name")
        assert hasattr(model, "family")
        assert hasattr(model, "version")
        assert hasattr(model, "status")
        assert hasattr(model, "cost_tier")
        assert hasattr(model, "supports_thinking")
        assert hasattr(model, "thinking_config_type")
        assert hasattr(model, "max_input_tokens")
        assert hasattr(model, "max_output_tokens")


class TestGetApprovedModels:
    """Tests for get_approved_models function."""
    
    def test_returns_list(self):
        """Should return a list of models."""
        models = get_approved_models()
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_all_items_are_model_info(self):
        """All items should be ModelInfo instances."""
        models = get_approved_models()
        for model in models:
            assert isinstance(model, ModelInfo)
    
    def test_includes_key_models(self):
        """Should include key Gemini models."""
        models = get_approved_models()
        model_ids = [m.id for m in models]
        
        assert "gemini-2.5-flash" in model_ids
        assert "gemini-2.5-pro" in model_ids


class TestGetModelsWithThinking:
    """Tests for get_models_with_thinking function."""
    
    def test_returns_only_thinking_models(self):
        """Should only return models that support thinking."""
        models = get_models_with_thinking()
        for model in models:
            assert model.supports_thinking is True
    
    def test_includes_gemini_25_models(self):
        """Should include Gemini 2.5 models which support thinking."""
        models = get_models_with_thinking()
        model_ids = [m.id for m in models]
        
        assert "gemini-2.5-flash" in model_ids
        assert "gemini-2.5-pro" in model_ids


class TestGetThinkingConfig:
    """Tests for get_thinking_config function."""
    
    def test_gemini_25_returns_budget(self):
        """Gemini 2.5 models should return thinking_budget."""
        config = get_thinking_config("gemini-2.5-flash", "high")
        assert "thinking_budget" in config
        assert isinstance(config["thinking_budget"], int)
        assert config["thinking_budget"] > 0
    
    def test_gemini_3_returns_level(self):
        """Gemini 3 models should return thinking_level."""
        config = get_thinking_config("gemini-3-pro-preview", "high")
        assert "thinking_level" in config
        assert config["thinking_level"] == "HIGH"
    
    def test_non_thinking_model_returns_empty(self):
        """Non-thinking models should return empty dict."""
        config = get_thinking_config("gemini-2.0-flash", "high")
        assert config == {}
    
    def test_invalid_model_returns_empty(self):
        """Invalid model should return empty dict."""
        config = get_thinking_config("invalid-model", "high")
        assert config == {}
    
    def test_effort_levels(self):
        """Different effort levels should return different budgets."""
        low = get_thinking_config("gemini-2.5-flash", "low")
        high = get_thinking_config("gemini-2.5-flash", "high")
        
        assert low["thinking_budget"] < high["thinking_budget"]


class TestValidateModel:
    """Tests for validate_model function."""
    
    def test_valid_model_passes(self):
        """Valid model should pass validation."""
        result = validate_model("gemini-2.5-flash")
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_invalid_model_fails(self):
        """Invalid model should fail validation."""
        result = validate_model("invalid-model")
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_require_thinking_on_thinking_model(self):
        """Thinking model should pass when thinking required."""
        result = validate_model("gemini-2.5-flash", require_thinking=True)
        assert result.valid is True
    
    def test_require_thinking_on_non_thinking_model(self):
        """Non-thinking model should fail when thinking required."""
        result = validate_model("gemini-2.0-flash", require_thinking=True)
        assert result.valid is False
        assert any("thinking" in e.lower() for e in result.errors)
    
    def test_experimental_model_has_warning(self):
        """Experimental model should have warning."""
        result = validate_model("gemini-2.0-flash-thinking-exp")
        assert len(result.warnings) > 0
        assert any("experimental" in w.lower() for w in result.warnings)


class TestModelInfoToDict:
    """Tests for ModelInfo.to_dict method."""
    
    def test_to_dict_returns_dict(self):
        """to_dict should return a dictionary."""
        model = get_model("gemini-2.5-flash")
        assert model is not None
        
        d = model.to_dict()
        assert isinstance(d, dict)
    
    def test_to_dict_has_all_keys(self):
        """to_dict should include all model properties."""
        model = get_model("gemini-2.5-flash")
        assert model is not None
        
        d = model.to_dict()
        expected_keys = [
            "id", "name", "family", "version", "status",
            "cost_tier", "supports_thinking", "thinking_config_type",
            "max_input_tokens", "max_output_tokens",
        ]
        for key in expected_keys:
            assert key in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
