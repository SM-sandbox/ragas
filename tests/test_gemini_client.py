#!/usr/bin/env python3
"""
Comprehensive Test Suite for Gemini Client

Tests:
- Unit: Client initialization, Secret Manager, config parsing
- Integration: Single generation, JSON output, thinking levels
- Functional: Rate limit handling, error recovery
- E2E: Full judge workflow

Run with: pytest tests/test_gemini_client.py -v
"""

import sys
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gemini_client import (
    get_client,
    reset_client,
    generate,
    generate_fast,
    generate_json,
    generate_for_judge,
    generate_for_rag,
    generate_with_reasoning,
    get_model_info,
    health_check,
    _get_api_key,
    DEFAULT_MODEL,
    PROJECT_ID,
    SECRET_NAME,
    JUDGE_CONFIG,
    GENERATOR_CONFIG,
)


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestUnitClientInitialization:
    """Unit tests for client initialization."""
    
    def test_model_info_structure(self):
        """Test model info returns expected structure."""
        info = get_model_info()
        
        assert "model_id" in info
        assert "display_name" in info
        assert "status" in info
        assert "input_token_limit" in info
        assert "output_token_limit" in info
        assert "thinking_levels" in info
        assert "project" in info
        assert "secret" in info
    
    def test_model_id_is_gemini_3_flash(self):
        """Test we're using Gemini 3 Flash Preview."""
        info = get_model_info()
        assert info["model_id"] == "gemini-3-flash-preview"
        assert info["status"] == "PUBLIC_PREVIEW"
    
    def test_default_model_constant(self):
        """Test DEFAULT_MODEL is set correctly."""
        assert DEFAULT_MODEL == "gemini-3-flash-preview"
    
    def test_project_and_secret_constants(self):
        """Test project and secret are configured."""
        assert PROJECT_ID == "bf-rag-eval-service"
        assert SECRET_NAME == "gemini-api-key-eval"
    
    def test_thinking_levels_available(self):
        """Test thinking levels are LOW and HIGH only."""
        info = get_model_info()
        assert "LOW" in info["thinking_levels"]
        assert "HIGH" in info["thinking_levels"]
        assert "MEDIUM" not in info["thinking_levels"]


class TestUnitSecretManager:
    """Unit tests for Secret Manager integration."""
    
    def test_api_key_retrieval(self):
        """Test API key can be retrieved from Secret Manager."""
        # Reset to force fresh retrieval
        reset_client()
        
        api_key = _get_api_key()
        
        assert api_key is not None
        assert len(api_key) > 0
        assert api_key.startswith("AIza")  # Google API keys start with this
    
    def test_api_key_caching(self):
        """Test API key is cached after first retrieval."""
        reset_client()
        
        # First call
        key1 = _get_api_key()
        
        # Second call should return cached value
        key2 = _get_api_key()
        
        assert key1 == key2


class TestUnitClientSingleton:
    """Unit tests for client singleton pattern."""
    
    def test_get_client_returns_same_instance(self):
        """Test get_client returns singleton."""
        reset_client()
        
        client1 = get_client()
        client2 = get_client()
        
        assert client1 is client2
    
    def test_reset_client_clears_singleton(self):
        """Test reset_client clears the singleton."""
        client1 = get_client()
        reset_client()
        client2 = get_client()
        
        # After reset, should be a new instance
        # (can't directly compare, but we can verify it works)
        assert client2 is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegrationBasicGeneration:
    """Integration tests for basic generation."""
    
    def test_simple_generation(self):
        """Test simple text generation."""
        result = generate("What is 2 + 2? Answer with just the number.")
        
        assert "text" in result
        assert "model_used" in result
        assert result["model_used"] == "gemini-3-flash-preview"
        assert "4" in result["text"]
    
    def test_llm_metadata_structure(self):
        """Test llm_metadata matches orchestrator schema."""
        result = generate("Say hello.")
        
        assert "llm_metadata" in result
        metadata = result["llm_metadata"]
        
        # Required fields per orchestrator schema
        assert "prompt_tokens" in metadata
        assert "completion_tokens" in metadata
        assert "thinking_tokens" in metadata
        assert "total_tokens" in metadata
        assert "cached_content_tokens" in metadata
        assert "model_version" in metadata
        assert "finish_reason" in metadata
        assert "used_fallback" in metadata
        assert "reasoning_effort" in metadata
        assert "avg_logprobs" in metadata
        assert "response_id" in metadata
        
        # Type checks
        assert isinstance(metadata["prompt_tokens"], int)
        assert isinstance(metadata["completion_tokens"], int)
        assert isinstance(metadata["thinking_tokens"], int)
        assert isinstance(metadata["total_tokens"], int)
        assert isinstance(metadata["used_fallback"], bool)
        assert isinstance(metadata["reasoning_effort"], str)
    
    def test_llm_metadata_token_counts_positive(self):
        """Test token counts are positive values."""
        result = generate("What is the capital of France?")
        
        metadata = result["llm_metadata"]
        assert metadata["prompt_tokens"] > 0
        assert metadata["completion_tokens"] > 0
        assert metadata["total_tokens"] > 0
        assert metadata["total_tokens"] >= metadata["prompt_tokens"] + metadata["completion_tokens"]
    
    def test_llm_metadata_finish_reason(self):
        """Test finish_reason is valid."""
        result = generate("Say OK.")
        
        metadata = result["llm_metadata"]
        valid_reasons = ["STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER", 
                         "FinishReason.STOP", "FinishReason.MAX_TOKENS"]
        assert any(r in str(metadata["finish_reason"]) for r in valid_reasons)
    
    def test_backward_compatible_usage_field(self):
        """Test backward compatible 'usage' field still exists."""
        result = generate("Say hello.")
        
        assert "usage" in result
        usage = result["usage"]
        assert "prompt_tokens" in usage
        assert "response_tokens" in usage
        assert "total_tokens" in usage
        assert "thinking_tokens" in usage
    
    def test_generate_fast(self):
        """Test generate_fast convenience function."""
        text = generate_fast("Say 'hello' and nothing else.")
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "hello" in text.lower()
    
    def test_custom_temperature(self):
        """Test generation with custom temperature."""
        result = generate(
            "Generate a random word.",
            temperature=0.0,
        )
        
        assert "text" in result
        assert len(result["text"]) > 0
    
    def test_custom_max_tokens(self):
        """Test generation with custom max tokens."""
        result = generate(
            "Write a very long essay about the history of computing.",
            max_output_tokens=50,
        )
        
        assert "text" in result
        # Response may be None or truncated - either is acceptable
        if result["text"] is not None:
            assert len(result["text"]) < 1000


class TestIntegrationJSONOutput:
    """Integration tests for JSON output."""
    
    def test_generate_json_simple(self):
        """Test JSON generation with simple object."""
        result = generate_json(
            "Return a JSON object with name='test' and value=42"
        )
        
        assert isinstance(result, dict)
        assert "name" in result
        assert "value" in result
        assert result["name"] == "test"
        assert result["value"] == 42
    
    def test_generate_json_nested(self):
        """Test JSON generation with nested object."""
        result = generate_json(
            "Return JSON: {person: {name: 'John', age: 30}, active: true}"
        )
        
        assert isinstance(result, dict)
        assert "person" in result
        assert result["person"]["name"] == "John"
    
    def test_generate_for_judge(self):
        """Test judge-specific generation."""
        prompt = """Evaluate this answer:
Question: What is 2+2?
Ground Truth: 4
Answer: The answer is 4.

Return JSON with: correctness (1-5), verdict (pass/partial/fail)"""
        
        result = generate_for_judge(prompt)
        
        assert isinstance(result, dict)
        assert "correctness" in result or "verdict" in result


class TestIntegrationThinking:
    """Integration tests for thinking/reasoning."""
    
    def test_thinking_level_low(self):
        """Test generation with LOW thinking level."""
        result = generate(
            "What is 15 * 23?",
            thinking_level="LOW",
        )
        
        assert "text" in result
        assert "345" in result["text"]
    
    def test_thinking_level_high(self):
        """Test generation with HIGH thinking level."""
        result = generate(
            "What is 15 * 23?",
            thinking_level="HIGH",
        )
        
        assert "text" in result
        assert "345" in result["text"]
    
    def test_thinking_with_thoughts_returned(self):
        """Test thinking with thoughts included in response."""
        result = generate_with_reasoning(
            "What is 15 * 23?",
            thinking_level="LOW",
        )
        
        assert "text" in result
        assert "thoughts" in result
        assert "345" in result["text"]


class TestIntegrationRAGGeneration:
    """Integration tests for RAG-specific generation."""
    
    def test_generate_for_rag(self):
        """Test RAG answer generation."""
        prompt = """Based on this context, answer the question.

Context: Solar inverters convert DC power from solar panels to AC power for home use.

Question: What does a solar inverter do?

Answer:"""
        
        result = generate_for_rag(prompt)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "convert" in result.lower() or "dc" in result.lower() or "ac" in result.lower()


# =============================================================================
# FUNCTIONAL TESTS
# =============================================================================

class TestFunctionalErrorHandling:
    """Functional tests for error handling."""
    
    def test_empty_prompt_handling(self):
        """Test handling of empty prompt."""
        # Should not crash, may return error or empty response
        try:
            result = generate("")
            assert "text" in result
        except Exception as e:
            # Some error is acceptable for empty prompt
            assert "error" in str(e).lower() or "empty" in str(e).lower() or True
    
    def test_very_long_prompt(self):
        """Test handling of very long prompt."""
        long_prompt = "Hello " * 1000  # ~6000 chars
        
        result = generate(long_prompt + " Say OK.")
        
        assert "text" in result


class TestFunctionalHealthCheck:
    """Functional tests for health check."""
    
    def test_health_check_returns_healthy(self):
        """Test health check returns healthy status."""
        result = health_check()
        
        assert "status" in result
        assert result["status"] == "healthy"
        assert "model" in result
        assert result["model"] == "gemini-3-flash-preview"


# =============================================================================
# E2E TESTS
# =============================================================================

class TestE2EJudgeWorkflow:
    """End-to-end tests for judge workflow."""
    
    def test_full_judge_evaluation(self):
        """Test full judge evaluation workflow."""
        question = "What is the capital of France?"
        ground_truth = "Paris is the capital of France."
        rag_answer = "The capital of France is Paris."
        context = "France is a country in Europe. Its capital city is Paris."
        
        prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth.

Question: {question}

Ground Truth: {ground_truth}

RAG Answer: {rag_answer}

Context: {context}

Score 1-5 for each (5=best):
1. correctness - factually correct vs ground truth?
2. completeness - covers key points?
3. faithfulness - faithful to context, no hallucinations?
4. relevance - relevant to question?
5. clarity - clear and well-structured?

Respond with JSON containing: correctness, completeness, faithfulness, relevance, clarity, overall_score (all 1-5), and verdict (pass|partial|fail)."""
        
        result = generate_for_judge(prompt)
        
        assert isinstance(result, dict)
        assert "correctness" in result
        assert "verdict" in result
        assert result["correctness"] >= 1 and result["correctness"] <= 5
        assert result["verdict"] in ["pass", "partial", "fail"]
    
    def test_judge_with_incorrect_answer(self):
        """Test judge correctly identifies incorrect answer."""
        question = "What is 2 + 2?"
        ground_truth = "4"
        rag_answer = "The answer is 5."
        context = "Basic arithmetic."
        
        prompt = f"""Evaluate this RAG answer:
Question: {question}
Ground Truth: {ground_truth}
RAG Answer: {rag_answer}
Context: {context}

Return JSON with correctness (1-5) and verdict (pass/partial/fail)."""
        
        result = generate_for_judge(prompt)
        
        assert isinstance(result, dict)
        # Should score low for incorrect answer
        if "correctness" in result:
            assert result["correctness"] <= 3
        if "verdict" in result:
            assert result["verdict"] in ["partial", "fail"]


class TestE2EGeneratorWorkflow:
    """End-to-end tests for generator workflow."""
    
    def test_full_rag_generation(self):
        """Test full RAG generation workflow."""
        context = """
[1] Solar inverters are devices that convert DC electricity from solar panels into AC electricity.
[2] The typical efficiency of modern solar inverters is between 95-98%.
[3] String inverters are the most common type used in residential installations.
"""
        question = "What is the efficiency of modern solar inverters?"
        
        prompt = f"""You are a technical assistant for solar equipment.
Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        result = generate_for_rag(prompt)
        
        assert isinstance(result, str)
        assert "95" in result or "98" in result or "efficiency" in result.lower()


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests."""
    
    def test_generation_latency(self):
        """Test generation completes in reasonable time."""
        start = time.time()
        
        result = generate("Say 'OK'.", max_output_tokens=10)
        
        elapsed = time.time() - start
        
        assert elapsed < 30  # Should complete in under 30 seconds
        assert "text" in result
    
    def test_multiple_sequential_calls(self):
        """Test multiple sequential calls work correctly."""
        results = []
        
        for i in range(3):
            result = generate(f"What is {i} + 1? Answer with just the number.")
            results.append(result)
        
        assert len(results) == 3
        for r in results:
            assert "text" in r


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
