"""
End-to-End tests for Gemini Client multi-region cascade.

These tests make REAL API calls and cost money. Run sparingly.

Usage:
    # Run e2e tests only
    pytest tests/eval/test_gemini_cascade_e2e.py -v -m e2e
    
    # Skip e2e tests (default in CI)
    pytest tests/eval/ -v -m "not e2e"
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.clients.gemini_client import (
    generate,
    generate_json,
    generate_for_judge,
    health_check,
    health_check_all_endpoints,
    reset_client,
    get_model_info,
    VERTEX_AI_REGIONS,
    VERTEX_PROJECT_ID,
    DEFAULT_MODEL,
)


# Mark all tests in this module as e2e (skipped by default)
pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def reset_before_test():
    """Reset clients before each test for clean state."""
    reset_client()
    yield
    reset_client()


class TestHealthCheckE2E:
    """E2E tests for health check functions."""
    
    def test_health_check_returns_healthy(self):
        """Health check should return healthy status with real API."""
        result = health_check()
        
        assert result["status"] == "healthy"
        assert "endpoint_used" in result
        assert "cascade_stats" in result
        assert result["response"] is not None
        print(f"\n  Health check result: {result}")
    
    def test_health_check_shows_endpoint(self):
        """Health check should show which endpoint was used."""
        result = health_check()
        
        # Should use primary Vertex AI region
        assert result["endpoint_used"].startswith("vertex_ai:")
        assert result["used_fallback"] is False
        print(f"\n  Endpoint used: {result['endpoint_used']}")
    
    def test_health_check_all_endpoints(self):
        """Test all endpoints individually."""
        result = health_check_all_endpoints()
        
        print(f"\n  Overall status: {result['overall_status']}")
        print(f"  Summary: {result['summary']}")
        
        # Check Vertex AI regions
        for region, status in result["vertex_ai_regions"].items():
            print(f"  Vertex AI {region}: {status['status']}")
            if status["status"] == "healthy":
                print(f"    Latency: {status['latency_ms']}ms")
        
        # Check AI Studio
        if result["ai_studio"]:
            print(f"  AI Studio: {result['ai_studio']['status']}")
            if result["ai_studio"]["status"] == "healthy":
                print(f"    Latency: {result['ai_studio']['latency_ms']}ms")
        
        # At least primary region should be healthy
        assert result["summary"]["healthy_count"] >= 1


class TestGenerateE2E:
    """E2E tests for generate function."""
    
    def test_generate_simple(self):
        """Simple generation should work."""
        result = generate("What is 2+2? Answer with just the number.")
        
        assert result["text"] is not None
        assert "4" in result["text"]
        assert result["endpoint_used"].startswith("vertex_ai:")
        assert result["cascade_stats"]["total_attempts"] == 1
        print(f"\n  Response: {result['text'][:100]}")
        print(f"  Endpoint: {result['endpoint_used']}")
    
    def test_generate_with_thinking(self):
        """Generation with thinking level should work."""
        result = generate(
            "What is 15 * 23?",
            thinking_level="LOW",
            max_output_tokens=500,
        )
        
        assert result["text"] is not None
        assert "345" in result["text"]
        print(f"\n  Response: {result['text'][:100]}")
    
    def test_generate_returns_token_counts(self):
        """Generation should return token usage."""
        result = generate("Say hello.", max_output_tokens=50)
        
        assert "llm_metadata" in result
        assert result["llm_metadata"]["prompt_tokens"] > 0
        assert result["llm_metadata"]["completion_tokens"] > 0
        print(f"\n  Tokens: prompt={result['llm_metadata']['prompt_tokens']}, "
              f"completion={result['llm_metadata']['completion_tokens']}")
    
    def test_generate_cascade_stats(self):
        """Generation should include cascade statistics."""
        result = generate("Say OK.")
        
        stats = result["cascade_stats"]
        assert stats["total_attempts"] >= 1
        assert stats["vertex_ai_attempts"] >= 1
        assert stats["total_latency_ms"] > 0
        print(f"\n  Cascade stats: {stats}")


class TestGenerateJsonE2E:
    """E2E tests for JSON generation."""
    
    def test_generate_json_simple(self):
        """JSON generation should return parsed dict."""
        result = generate_json(
            "Return a JSON object with name='test' and value=42"
        )
        
        assert isinstance(result, dict)
        assert "name" in result or "value" in result
        print(f"\n  JSON result: {result}")
    
    def test_generate_json_complex(self):
        """JSON generation with complex structure."""
        result = generate_json(
            "Extract the following into JSON: John is 30 years old and lives in NYC. "
            "Return {name: string, age: number, city: string}"
        )
        
        assert isinstance(result, dict)
        print(f"\n  JSON result: {result}")


class TestGenerateForJudgeE2E:
    """E2E tests for judge generation."""
    
    def test_judge_returns_scores(self):
        """Judge should return structured scores."""
        prompt = """You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth.

Question: What is the capital of France?
Ground Truth: Paris is the capital of France.
RAG Answer: The capital of France is Paris, a major European city.
Context: France is a country in Western Europe. Paris is its capital and largest city.

Score 1-5 for each (5=best):
1. correctness - factually correct vs ground truth?
2. completeness - covers key points?
3. faithfulness - faithful to context, no hallucinations?
4. relevance - relevant to question?
5. clarity - clear and well-structured?

Respond with JSON containing: correctness, completeness, faithfulness, relevance, clarity, overall_score (all 1-5), and verdict (pass|partial|fail)."""
        
        result = generate_for_judge(prompt, return_metadata=True)
        
        assert "judgment" in result
        judgment = result["judgment"]
        assert "correctness" in judgment
        assert "verdict" in judgment
        assert judgment["verdict"] in ["pass", "partial", "fail"]
        
        print(f"\n  Judgment: {judgment}")
        print(f"  Tokens: {result['tokens']}")


class TestModelInfoE2E:
    """E2E tests for model info."""
    
    def test_model_info_correct(self):
        """Model info should reflect current configuration."""
        info = get_model_info()
        
        assert info["model_id"] == DEFAULT_MODEL
        assert info["vertex_project"] == VERTEX_PROJECT_ID
        assert info["vertex_regions"] == VERTEX_AI_REGIONS
        assert info["ai_studio_fallback_enabled"] is True
        
        print(f"\n  Model info: {info}")


class TestCascadeResilienceE2E:
    """E2E tests for cascade resilience (harder to test without inducing failures)."""
    
    def test_multiple_sequential_calls(self):
        """Multiple sequential calls should all succeed."""
        results = []
        for i in range(3):
            result = generate(f"Say the number {i}.", max_output_tokens=20)
            results.append(result)
        
        for i, result in enumerate(results):
            assert result["text"] is not None
            assert result["endpoint_used"].startswith("vertex_ai:")
            print(f"\n  Call {i}: endpoint={result['endpoint_used']}")
    
    def test_client_reuse(self):
        """Clients should be reused across calls."""
        # First call initializes client
        result1 = generate("Say A.", max_output_tokens=10)
        
        # Second call should reuse client (faster)
        result2 = generate("Say B.", max_output_tokens=10)
        
        # Both should use same endpoint (primary)
        assert result1["endpoint_used"] == result2["endpoint_used"]
        print(f"\n  Both calls used: {result1['endpoint_used']}")


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
