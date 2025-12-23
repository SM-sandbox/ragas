"""
Comprehensive tests for Gemini Client with global endpoint and AI Studio fallback.

Test Matrix:
    - test_global_endpoint_success: global endpoint works → no fallback
    - test_fallback_to_ai_studio: global fails → AI Studio works
    - test_all_endpoints_fail: Everything fails → proper error
    - test_non_retryable_error_no_fallback: 400 error → fail fast
    - test_auth_error_no_fallback: 401/403 → fail fast
    - test_cascade_logging: Verify proper logging at each step
    - test_response_includes_endpoint_used: Response metadata shows which endpoint
    - test_cascade_stats_tracking: Stats correctly track attempts and failures
    - test_client_caching: Clients are cached and reused
    - test_rate_limit_retry_within_endpoint: 429 retries before fallback
"""

import pytest
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.clients.gemini_client import (
    # Core functions
    generate,
    _generate_content_with_cascade,
    is_retryable_error,
    # Client functions
    get_vertex_client,
    get_ai_studio_client,
    reset_client,
    # Classes
    APIError,
    CascadeExhaustedError,
    CascadeStats,
    EndpointType,
    EndpointResult,
    # Config
    VERTEX_AI_REGIONS,
    VERTEX_PROJECT_ID,
    DEFAULT_MODEL,
    ENABLE_AI_STUDIO_FALLBACK,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_clients():
    """Reset all cached clients before each test."""
    reset_client()
    yield
    reset_client()


@pytest.fixture
def mock_response():
    """Create a mock successful response."""
    response = Mock()
    response.text = "Test response"
    response.model_version = "gemini-3-flash-preview"
    response.response_id = "test-response-id"
    
    # Mock usage metadata
    usage = Mock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    usage.total_token_count = 150
    usage.cached_content_token_count = 0
    response.usage_metadata = usage
    
    # Mock candidates
    candidate = Mock()
    candidate.finish_reason = "STOP"
    candidate.avg_logprobs = -0.5
    content = Mock()
    content.parts = []
    candidate.content = content
    response.candidates = [candidate]
    
    return response


@pytest.fixture
def mock_config():
    """Create a mock generation config."""
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=100,
    )


# =============================================================================
# ERROR CLASSIFICATION TESTS
# =============================================================================

class TestIsRetryableError:
    """Tests for error classification logic."""
    
    def test_429_is_retryable(self):
        """429 rate limit should trigger cascade."""
        error = Exception("Error 429: Resource exhausted")
        assert is_retryable_error(error) is True
    
    def test_resource_exhausted_is_retryable(self):
        """Resource exhausted should trigger cascade."""
        error = Exception("RESOURCE_EXHAUSTED: Quota exceeded")
        assert is_retryable_error(error) is True
    
    def test_503_is_retryable(self):
        """503 service unavailable should trigger cascade."""
        error = Exception("503 Service Unavailable")
        assert is_retryable_error(error) is True
    
    def test_timeout_is_retryable(self):
        """Timeout should trigger cascade."""
        error = Exception("Request timed out after 60s")
        assert is_retryable_error(error) is True
    
    def test_deadline_exceeded_is_retryable(self):
        """Deadline exceeded should trigger cascade."""
        error = Exception("DEADLINE_EXCEEDED: Operation took too long")
        assert is_retryable_error(error) is True
    
    def test_504_is_retryable(self):
        """504 gateway timeout should trigger cascade."""
        error = Exception("504 Gateway Timeout")
        assert is_retryable_error(error) is True
    
    def test_404_is_retryable(self):
        """404 not found should trigger cascade (model might be in another region)."""
        error = Exception("404: Model not found in this region")
        assert is_retryable_error(error) is True
    
    def test_400_is_not_retryable(self):
        """400 bad request should NOT trigger cascade."""
        error = Exception("400 Bad Request: Invalid prompt")
        assert is_retryable_error(error) is False
    
    def test_401_is_not_retryable(self):
        """401 unauthorized should NOT trigger cascade."""
        error = Exception("401 Unauthorized: Invalid credentials")
        assert is_retryable_error(error) is False
    
    def test_403_is_not_retryable(self):
        """403 forbidden should NOT trigger cascade."""
        error = Exception("403 Forbidden: Permission denied")
        assert is_retryable_error(error) is False
    
    def test_invalid_request_is_not_retryable(self):
        """Invalid request should NOT trigger cascade."""
        error = Exception("Invalid request: missing required field")
        assert is_retryable_error(error) is False
    
    def test_unknown_error_is_retryable(self):
        """Unknown errors should trigger cascade (be resilient)."""
        error = Exception("Something unexpected happened")
        assert is_retryable_error(error) is True


# =============================================================================
# CASCADE TESTS
# =============================================================================

class TestGlobalEndpointSuccess:
    """Tests for when global endpoint succeeds."""
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_global_endpoint_success(self, mock_get_client, mock_response, mock_config):
        """Global endpoint works → no fallback needed."""
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        response, endpoint, stats = _generate_content_with_cascade(
            DEFAULT_MODEL, "test prompt", mock_config
        )
        
        assert response == mock_response
        assert endpoint == "vertex_ai:global"
        assert stats.total_attempts == 1
        assert stats.vertex_ai_attempts == 1
        assert stats.ai_studio_attempts == 0
        assert stats.successful_endpoint == "vertex_ai:global"
        assert len(stats.failed_endpoints) == 0
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_response_includes_endpoint_used(self, mock_get_client, mock_response):
        """Response metadata shows which endpoint was used."""
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        result = generate("test prompt")
        
        assert "endpoint_used" in result
        assert result["endpoint_used"] == "vertex_ai:global"
        assert "cascade_stats" in result
        assert result["cascade_stats"]["total_attempts"] == 1


class TestFallbackToAIStudio:
    """Tests for AI Studio fallback."""
    
    @patch('lib.clients.gemini_client.get_ai_studio_client')
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_fallback_to_ai_studio(self, mock_get_vertex, mock_get_ai_studio, mock_response, mock_config):
        """Global endpoint fails → AI Studio works."""
        # Global endpoint fails
        mock_vertex_client = Mock()
        mock_vertex_client.models.generate_content.side_effect = Exception("429 Quota exceeded")
        mock_get_vertex.return_value = mock_vertex_client
        
        # AI Studio succeeds
        mock_ai_studio_client = Mock()
        mock_ai_studio_client.models.generate_content.return_value = mock_response
        mock_get_ai_studio.return_value = mock_ai_studio_client
        
        response, endpoint, stats = _generate_content_with_cascade(
            DEFAULT_MODEL, "test prompt", mock_config
        )
        
        assert response == mock_response
        assert endpoint == "ai_studio"
        assert stats.total_attempts == 2  # 1 global + 1 AI Studio
        assert stats.vertex_ai_attempts == 1
        assert stats.ai_studio_attempts == 1
        assert stats.successful_endpoint == "ai_studio"
        assert len(stats.failed_endpoints) == 1  # Just global
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_used_fallback_flag_in_metadata(self, mock_get_vertex, mock_response):
        """Response metadata correctly indicates fallback was used."""
        # Make all Vertex AI fail
        mock_vertex_client = Mock()
        mock_vertex_client.models.generate_content.side_effect = Exception("429")
        mock_get_vertex.return_value = mock_vertex_client
        
        with patch('lib.clients.gemini_client.get_ai_studio_client') as mock_ai_studio:
            mock_ai_studio_client = Mock()
            mock_ai_studio_client.models.generate_content.return_value = mock_response
            mock_ai_studio.return_value = mock_ai_studio_client
            
            result = generate("test prompt")
            
            assert result["endpoint_used"] == "ai_studio"
            assert result["llm_metadata"]["used_fallback"] is True


class TestAllEndpointsFail:
    """Tests for when all endpoints fail."""
    
    @patch('lib.clients.gemini_client.get_ai_studio_client')
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_all_endpoints_fail(self, mock_get_vertex, mock_get_ai_studio, mock_config):
        """All endpoints fail → CascadeExhaustedError with stats."""
        # All clients fail
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("429 Quota exceeded everywhere")
        mock_get_vertex.return_value = mock_client
        mock_get_ai_studio.return_value = mock_client
        
        with pytest.raises(CascadeExhaustedError) as exc_info:
            _generate_content_with_cascade(DEFAULT_MODEL, "test prompt", mock_config)
        
        error = exc_info.value
        assert "All endpoints exhausted" in str(error)
        assert error.stats.total_attempts == 2  # 1 global + 1 AI Studio
        assert len(error.stats.failed_endpoints) == 2
    
    @patch('lib.clients.gemini_client.ENABLE_AI_STUDIO_FALLBACK', False)
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_global_fail_no_ai_studio_fallback(self, mock_get_vertex, mock_config):
        """Global fails, AI Studio disabled → error after 1 attempt."""
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("429")
        mock_get_vertex.return_value = mock_client
        
        # Need to reimport to pick up the patched constant
        from lib.clients import gemini_client
        original_value = gemini_client.ENABLE_AI_STUDIO_FALLBACK
        gemini_client.ENABLE_AI_STUDIO_FALLBACK = False
        
        try:
            with pytest.raises(CascadeExhaustedError) as exc_info:
                _generate_content_with_cascade(DEFAULT_MODEL, "test prompt", mock_config)
            
            error = exc_info.value
            assert error.stats.total_attempts == 1  # Only global attempt
            assert error.stats.ai_studio_attempts == 0
        finally:
            gemini_client.ENABLE_AI_STUDIO_FALLBACK = original_value


class TestNonRetryableErrors:
    """Tests for non-retryable errors that should fail fast."""
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_400_error_no_cascade(self, mock_get_vertex, mock_config):
        """400 Bad Request → fail fast, no cascade."""
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("400 Bad Request: Invalid prompt")
        mock_get_vertex.return_value = mock_client
        
        with pytest.raises(APIError) as exc_info:
            _generate_content_with_cascade(DEFAULT_MODEL, "test prompt", mock_config)
        
        assert "Non-retryable error" in str(exc_info.value)
        # Should have only tried global endpoint
        mock_get_vertex.assert_called_once_with("global")
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_401_error_no_cascade(self, mock_get_vertex, mock_config):
        """401 Unauthorized → fail fast, no cascade."""
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("401 Unauthorized")
        mock_get_vertex.return_value = mock_client
        
        with pytest.raises(APIError) as exc_info:
            _generate_content_with_cascade(DEFAULT_MODEL, "test prompt", mock_config)
        
        assert "Non-retryable error" in str(exc_info.value)
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_403_error_no_cascade(self, mock_get_vertex, mock_config):
        """403 Forbidden → fail fast, no cascade."""
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("403 Permission denied")
        mock_get_vertex.return_value = mock_client
        
        with pytest.raises(APIError) as exc_info:
            _generate_content_with_cascade(DEFAULT_MODEL, "test prompt", mock_config)
        
        assert "Non-retryable error" in str(exc_info.value)


class TestCascadeStats:
    """Tests for cascade statistics tracking."""
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_cascade_stats_tracking(self, mock_get_vertex, mock_response, mock_config):
        """Stats correctly track attempts, failures, and latency."""
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_vertex.return_value = mock_client
        
        response, endpoint, stats = _generate_content_with_cascade(
            DEFAULT_MODEL, "test prompt", mock_config
        )
        
        assert stats.total_attempts == 1
        assert stats.vertex_ai_attempts == 1
        assert stats.ai_studio_attempts == 0
        assert len(stats.failed_endpoints) == 0
        assert stats.successful_endpoint == "vertex_ai:global"
        assert stats.total_latency_ms > 0  # Should have some latency recorded


class TestClientCaching:
    """Tests for client caching behavior."""
    
    @patch('lib.clients.gemini_client.genai.Client')
    def test_vertex_client_cached(self, mock_client_class):
        """Vertex AI clients are cached and reused."""
        mock_client_class.return_value = Mock()
        
        # Get client twice for same region
        c1 = get_vertex_client("global")
        c2 = get_vertex_client("global")
        
        # Should be same instance
        assert c1 is c2
        
        # Should only create once
        assert mock_client_class.call_count == 1
    
    @patch('lib.clients.gemini_client._get_ai_studio_api_key')
    @patch('lib.clients.gemini_client.genai.Client')
    def test_ai_studio_client_cached(self, mock_client_class, mock_get_key):
        """AI Studio client is cached and reused."""
        mock_client_class.return_value = Mock()
        mock_get_key.return_value = "test-api-key"
        
        client1 = get_ai_studio_client()
        client2 = get_ai_studio_client()
        
        assert client1 is client2
        # API key should only be fetched once
        assert mock_get_key.call_count == 1
    
    def test_reset_client_clears_cache(self):
        """reset_client() clears all cached clients."""
        from lib.clients import gemini_client
        
        # Add something to cache
        gemini_client._client_cache["test"] = Mock()
        gemini_client._ai_studio_api_key = "test-key"
        
        reset_client()
        
        assert len(gemini_client._client_cache) == 0
        assert gemini_client._ai_studio_api_key is None


class TestRateLimitRetryWithinEndpoint:
    """Tests for retry behavior within a single endpoint."""
    
    @patch('lib.clients.gemini_client.time.sleep')
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_rate_limit_retry_before_cascade(self, mock_get_vertex, mock_sleep, mock_response, mock_config):
        """429 retries within endpoint before cascading."""
        call_count = [0]
        
        def generate_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 Resource exhausted")
            return mock_response
        
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = generate_side_effect
        mock_get_vertex.return_value = mock_client
        
        response, endpoint, stats = _generate_content_with_cascade(
            DEFAULT_MODEL, "test prompt", mock_config
        )
        
        # Should succeed on global endpoint after retry
        assert endpoint == "vertex_ai:global"
        assert stats.total_attempts == 1  # Only counted as 1 attempt (retries are internal)
        # Should have slept for backoff
        assert mock_sleep.called


class TestLogging:
    """Tests for proper logging during cascade."""
    
    @patch('lib.clients.gemini_client.logger')
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_cascade_logging(self, mock_get_vertex, mock_logger, mock_response, mock_config):
        """Verify proper logging at each cascade step."""
        call_count = [0]
        
        def side_effect(region):
            mock_client = Mock()
            call_count[0] += 1
            if call_count[0] == 1:
                mock_client.models.generate_content.side_effect = Exception("429")
            else:
                mock_client.models.generate_content.return_value = mock_response
            return mock_client
        
        mock_get_vertex.side_effect = side_effect
        
        _generate_content_with_cascade(DEFAULT_MODEL, "test prompt", mock_config)
        
        # Should have warning for failed region
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        assert len(warning_calls) >= 1
        
        # Should have info for success
        info_calls = [call for call in mock_logger.info.call_args_list]
        assert len(info_calls) >= 1


# =============================================================================
# INTEGRATION TESTS (require real config but mock API)
# =============================================================================

class TestGenerateFunction:
    """Integration tests for the main generate() function."""
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_generate_returns_endpoint_info(self, mock_get_vertex, mock_response):
        """generate() returns endpoint and cascade info."""
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_vertex.return_value = mock_client
        
        result = generate("What is 2+2?")
        
        assert "text" in result
        assert "endpoint_used" in result
        assert "cascade_stats" in result
        assert "llm_metadata" in result
        assert result["endpoint_used"].startswith("vertex_ai:")
    
    @patch('lib.clients.gemini_client.get_vertex_client')
    def test_generate_with_all_options(self, mock_get_vertex, mock_response):
        """generate() works with all optional parameters."""
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_vertex.return_value = mock_client
        
        result = generate(
            "Test prompt",
            model="gemini-3-flash-preview",
            thinking_level="LOW",
            temperature=0.0,
            max_output_tokens=1000,
            seed=42,
        )
        
        assert result["text"] == "Test response"
        assert result["model_used"] == "gemini-3-flash-preview"


# =============================================================================
# DATACLASS TESTS
# =============================================================================

class TestDataClasses:
    """Tests for dataclass behavior."""
    
    def test_cascade_stats_defaults(self):
        """CascadeStats has correct defaults."""
        stats = CascadeStats()
        assert stats.total_attempts == 0
        assert stats.vertex_ai_attempts == 0
        assert stats.ai_studio_attempts == 0
        assert stats.successful_endpoint is None
        assert stats.failed_endpoints == []
        assert stats.total_latency_ms == 0.0
    
    def test_endpoint_result_name(self):
        """EndpointResult.endpoint_name works correctly."""
        vertex_result = EndpointResult(
            success=True,
            endpoint_type=EndpointType.VERTEX_AI,
            region="us-east1"
        )
        assert vertex_result.endpoint_name == "vertex_ai:us-east1"
        
        ai_studio_result = EndpointResult(
            success=True,
            endpoint_type=EndpointType.AI_STUDIO
        )
        assert ai_studio_result.endpoint_name == "ai_studio"


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Tests for configuration values."""
    
    def test_vertex_regions_configured(self):
        """Vertex AI uses global endpoint only (required for gemini-3-flash-preview)."""
        assert len(VERTEX_AI_REGIONS) == 1
        assert VERTEX_AI_REGIONS[0] == "global"
    
    def test_vertex_project_configured(self):
        """Vertex AI project is properly configured."""
        assert VERTEX_PROJECT_ID == "bfai-prod"
    
    def test_ai_studio_fallback_enabled(self):
        """AI Studio fallback is enabled by default."""
        assert ENABLE_AI_STUDIO_FALLBACK is True
    
    def test_default_model_configured(self):
        """Default model is properly configured for Vertex AI global endpoint."""
        assert DEFAULT_MODEL == "gemini-3-flash-preview"  # Vertex AI global + AI Studio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
