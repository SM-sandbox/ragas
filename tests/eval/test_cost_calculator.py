"""
Unit tests for cost_calculator.py
"""

import pytest
import sys
from pathlib import Path

# Add scripts/eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "eval"))

from cost_calculator import get_pricing, calculate_cost, calculate_run_cost, PRICING


class TestGetPricing:
    """Tests for get_pricing function."""
    
    def test_exact_match_gemini_25_flash(self):
        """Should return exact pricing for gemini-2.5-flash."""
        pricing = get_pricing("gemini-2.5-flash")
        assert pricing["input"] == 0.075
        assert pricing["output"] == 0.30
        assert pricing["thinking"] == 0.30
    
    def test_exact_match_gemini_3_flash(self):
        """Should return exact pricing for gemini-3-flash-preview."""
        pricing = get_pricing("gemini-3-flash-preview")
        assert pricing["input"] == 0.10
        assert pricing["output"] == 0.40
        assert pricing["thinking"] == 0.40
    
    def test_prefix_match(self):
        """Should match by prefix for versioned model names."""
        pricing = get_pricing("gemini-2.5-flash-preview-05-20")
        assert pricing["input"] == 0.075
    
    def test_unknown_model_returns_default(self):
        """Should return default pricing for unknown models."""
        pricing = get_pricing("unknown-model-xyz")
        assert pricing == PRICING["default"]
    
    def test_case_insensitive(self):
        """Should be case-insensitive."""
        pricing = get_pricing("GEMINI-2.5-FLASH")
        assert pricing["input"] == 0.075


class TestCalculateCost:
    """Tests for calculate_cost function."""
    
    def test_basic_cost_calculation(self):
        """Should calculate cost correctly for basic inputs."""
        costs = calculate_cost(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            model="gemini-2.5-flash",
        )
        assert costs["input_cost"] == 0.075
        assert costs["output_cost"] == 0.30
        assert costs["total_cost"] == 0.375
    
    def test_thinking_tokens_cost(self):
        """Should include thinking tokens in cost."""
        costs = calculate_cost(
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
            thinking_tokens=500_000,
            model="gemini-2.5-flash",
        )
        assert costs["thinking_cost"] == 0.15  # 0.5M * $0.30
        assert costs["total_cost"] == 0.075 + 0.15 + 0.15  # input + output + thinking
    
    def test_cached_tokens_cost(self):
        """Should calculate cached tokens at discounted rate."""
        costs = calculate_cost(
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=1_000_000,
            model="gemini-2.5-flash",
        )
        assert costs["cached_cost"] == 0.01875  # 75% discount
    
    def test_zero_tokens_returns_zero_cost(self):
        """Should return zero cost for zero tokens."""
        costs = calculate_cost(
            prompt_tokens=0,
            completion_tokens=0,
            model="gemini-2.5-flash",
        )
        assert costs["total_cost"] == 0.0
    
    def test_small_token_count(self):
        """Should handle small token counts correctly."""
        costs = calculate_cost(
            prompt_tokens=1000,  # 0.001M tokens
            completion_tokens=100,
            model="gemini-2.5-flash",
        )
        # 1000 tokens = 0.001M * $0.075 = $0.000075
        assert costs["input_cost"] == pytest.approx(0.000075, rel=1e-4)


class TestCalculateRunCost:
    """Tests for calculate_run_cost function."""
    
    def test_run_cost_with_question_count(self):
        """Should calculate per-question cost."""
        costs = calculate_run_cost(
            total_prompt_tokens=458_000,
            total_completion_tokens=137_400,
            model="gemini-2.5-flash",
            question_count=458,
        )
        assert costs["question_count"] == 458
        assert costs["cost_per_question"] > 0
        assert costs["cost_per_question"] == pytest.approx(
            costs["total_cost"] / 458, rel=1e-4
        )
    
    def test_run_cost_gemini_3_more_expensive(self):
        """Gemini 3 should be more expensive than Gemini 2.5."""
        tokens = {
            "total_prompt_tokens": 1_000_000,
            "total_completion_tokens": 500_000,
            "question_count": 100,
        }
        
        cost_25 = calculate_run_cost(**tokens, model="gemini-2.5-flash")
        cost_3 = calculate_run_cost(**tokens, model="gemini-3-flash-preview")
        
        assert cost_3["total_cost"] > cost_25["total_cost"]
    
    def test_realistic_eval_run_cost(self):
        """Test realistic cost for 458 question eval run."""
        # Realistic: ~4000 prompt tokens, ~300 completion tokens per question
        costs = calculate_run_cost(
            total_prompt_tokens=458 * 4000,
            total_completion_tokens=458 * 300,
            model="gemini-2.5-flash",
            question_count=458,
        )
        
        # Should be under $1 for the full run
        assert costs["total_cost"] < 1.0
        # Should be around $0.15-0.20
        assert 0.10 < costs["total_cost"] < 0.50
    
    def test_zero_question_count_no_division_error(self):
        """Should handle zero question count without division error."""
        costs = calculate_run_cost(
            total_prompt_tokens=1000,
            total_completion_tokens=100,
            model="gemini-2.5-flash",
            question_count=0,
        )
        # max(0, 1) = 1, so cost_per_question should equal total_cost
        assert costs["cost_per_question"] == costs["total_cost"]
    
    def test_negative_tokens_handled(self):
        """Should handle negative tokens (edge case, shouldn't happen but be safe)."""
        costs = calculate_run_cost(
            total_prompt_tokens=-1000,
            total_completion_tokens=100,
            model="gemini-2.5-flash",
            question_count=1,
        )
        # Negative input cost
        assert costs["input_cost"] < 0


class TestEdgeCases:
    """Edge case and robustness tests."""
    
    def test_very_large_token_count(self):
        """Should handle very large token counts."""
        costs = calculate_cost(
            prompt_tokens=1_000_000_000,  # 1 billion tokens
            completion_tokens=500_000_000,
            model="gemini-2.5-flash",
        )
        assert costs["total_cost"] > 0
        assert costs["input_cost"] == 75.0  # 1B / 1M * $0.075
    
    def test_rounding_precision(self):
        """Should round to 6 decimal places."""
        costs = calculate_cost(
            prompt_tokens=1,
            completion_tokens=1,
            model="gemini-2.5-flash",
        )
        # Very small numbers should still be rounded properly
        assert len(str(costs["input_cost"]).split(".")[-1]) <= 6
    
    def test_all_pricing_keys_present(self):
        """All pricing dicts should have required keys."""
        required_keys = {"input", "output", "thinking", "cached"}
        for model, pricing in PRICING.items():
            assert required_keys.issubset(pricing.keys()), f"{model} missing keys"
    
    def test_pricing_values_non_negative(self):
        """All pricing values should be non-negative."""
        for model, pricing in PRICING.items():
            for key, value in pricing.items():
                assert value >= 0, f"{model}.{key} is negative"
    
    def test_cost_dict_has_all_keys(self):
        """calculate_cost should return all expected keys."""
        costs = calculate_cost(100, 100)
        expected_keys = {"input_cost", "output_cost", "thinking_cost", "cached_cost", "total_cost"}
        assert expected_keys == set(costs.keys())
    
    def test_run_cost_dict_has_all_keys(self):
        """calculate_run_cost should return all expected keys."""
        costs = calculate_run_cost(100, 100, question_count=10)
        expected_keys = {"input_cost", "output_cost", "thinking_cost", "cached_cost", 
                         "total_cost", "cost_per_question", "question_count"}
        assert expected_keys == set(costs.keys())


class TestIdempotency:
    """Idempotency tests - same input should always produce same output."""
    
    def test_calculate_cost_idempotent(self):
        """Same inputs should produce identical outputs."""
        args = {"prompt_tokens": 5000, "completion_tokens": 300, "model": "gemini-2.5-flash"}
        result1 = calculate_cost(**args)
        result2 = calculate_cost(**args)
        assert result1 == result2
    
    def test_calculate_run_cost_idempotent(self):
        """Same inputs should produce identical outputs."""
        args = {
            "total_prompt_tokens": 458000,
            "total_completion_tokens": 137400,
            "model": "gemini-2.5-flash",
            "question_count": 458,
        }
        result1 = calculate_run_cost(**args)
        result2 = calculate_run_cost(**args)
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
