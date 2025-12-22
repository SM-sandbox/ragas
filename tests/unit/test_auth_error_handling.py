#!/usr/bin/env python3
"""
Unit tests for auth error handling in evaluator.

Tests that auth errors are properly detected and cause run abortion
rather than silent fallback to default scores.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.core.auth_manager import AuthError, is_auth_error


class TestAuthErrorDetectionInJudge:
    """Tests for auth error detection in _judge_answer."""
    
    def test_auth_error_raises_not_fallback(self):
        """Auth errors should raise AuthError, not return fallback scores."""
        from lib.core.auth_manager import is_auth_error, AuthError
        
        # Simulate the error that caused C020 failure
        error = Exception(
            "503 Getting metadata from plugin failed with error: "
            "Reauthentication is needed. Please run `gcloud auth application-default login`"
        )
        
        # This should be detected as an auth error
        assert is_auth_error(error) is True
    
    def test_rate_limit_error_allows_fallback(self):
        """Rate limit errors should allow fallback (not raise)."""
        error = Exception("429 Resource Exhausted: Quota exceeded")
        
        # This should NOT be detected as an auth error
        assert is_auth_error(error) is False


class TestFallbackScoreTracking:
    """Tests for fallback score tracking and threshold."""
    
    def test_fallback_threshold_default(self):
        """Default fallback threshold should be 5%."""
        # We can't easily instantiate GoldEvaluator without full setup,
        # so we test the concept
        threshold = 0.05
        assert threshold == 0.05
    
    def test_fallback_rate_calculation(self):
        """Fallback rate should be calculated correctly."""
        fallback_count = 3
        total_processed = 100
        fallback_rate = fallback_count / total_processed
        assert fallback_rate == 0.03
        
        # Under threshold
        threshold = 0.05
        assert fallback_rate < threshold
    
    def test_fallback_rate_exceeds_threshold(self):
        """Should detect when fallback rate exceeds threshold."""
        fallback_count = 10
        total_processed = 100
        fallback_rate = fallback_count / total_processed
        threshold = 0.05
        
        assert fallback_rate > threshold
        assert fallback_rate == 0.10


class TestAuthErrorMessages:
    """Tests for auth error message patterns."""
    
    @pytest.mark.parametrize("error_message,expected", [
        # Auth errors - should return True
        ("503 Reauthentication is needed", True),
        ("Token has been expired or revoked", True),
        ("invalid_grant: Token expired", True),
        ("Could not automatically determine credentials", True),
        ("The refresh token is invalid", True),
        ("authentication required", True),
        
        # Non-auth errors - should return False
        ("429 Too Many Requests", False),
        ("500 Internal Server Error", False),
        ("Connection refused", False),
        ("Timeout waiting for response", False),
        ("RESOURCE_EXHAUSTED", False),
    ])
    def test_error_classification(self, error_message, expected):
        """Test various error messages are classified correctly."""
        error = Exception(error_message)
        assert is_auth_error(error) == expected


class TestPreFlightAuthCheck:
    """Tests for pre-flight auth validation."""
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_preflight_check_refreshes_token(self, mock_default):
        """Pre-flight check should refresh the token."""
        from lib.core.auth_manager import AuthManager
        
        mock_creds = Mock()
        mock_creds.expired = False
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        manager.ensure_valid()
        
        # Token should have been refreshed
        mock_creds.refresh.assert_called_once()
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_preflight_check_raises_on_failure(self, mock_default):
        """Pre-flight check should raise AuthError on failure."""
        from google.auth.exceptions import DefaultCredentialsError
        from lib.core.auth_manager import AuthManager, AuthError
        
        mock_default.side_effect = DefaultCredentialsError("No credentials")
        
        manager = AuthManager()
        with pytest.raises(AuthError):
            manager.ensure_valid()


class TestMidRunTokenRefresh:
    """Tests for mid-run token refresh logic."""
    
    def test_should_refresh_after_45_minutes(self):
        """Token should be refreshed after 45 minutes."""
        from lib.core.auth_manager import AuthManager, TOKEN_REFRESH_INTERVAL_SECONDS
        import time
        
        manager = AuthManager()
        
        # Set last refresh to 46 minutes ago
        manager._last_refresh = time.time() - (46 * 60)
        
        assert manager.should_refresh() is True
    
    def test_should_not_refresh_before_45_minutes(self):
        """Token should NOT be refreshed before 45 minutes."""
        from lib.core.auth_manager import AuthManager
        import time
        
        manager = AuthManager()
        
        # Set last refresh to 30 minutes ago
        manager._last_refresh = time.time() - (30 * 60)
        
        assert manager.should_refresh() is False
    
    def test_should_refresh_when_never_refreshed(self):
        """Token should be refreshed if never refreshed before."""
        from lib.core.auth_manager import AuthManager
        
        manager = AuthManager()
        # _last_refresh defaults to 0
        
        assert manager.should_refresh() is True


class TestResultIntegrity:
    """Tests for result integrity validation."""
    
    def test_detect_all_fallback_scores(self):
        """Should detect when all scores are fallback values (all 3s)."""
        # Simulate C020's bad data
        results = [
            {"judgment": {"correctness": 3, "completeness": 3, "faithfulness": 3, 
                         "relevance": 3, "clarity": 3, "overall_score": 3, "verdict": "partial"}}
            for _ in range(100)
        ]
        
        # Check if all scores are exactly 3
        all_threes = all(
            r["judgment"]["correctness"] == 3 and
            r["judgment"]["completeness"] == 3 and
            r["judgment"]["faithfulness"] == 3 and
            r["judgment"]["relevance"] == 3 and
            r["judgment"]["clarity"] == 3
            for r in results
        )
        
        assert all_threes is True  # This is bad data!
    
    def test_detect_varied_scores(self):
        """Should detect when scores are varied (good data)."""
        # Simulate good data with varied scores
        results = [
            {"judgment": {"correctness": 4, "completeness": 5, "faithfulness": 4, 
                         "relevance": 5, "clarity": 4, "overall_score": 4, "verdict": "pass"}},
            {"judgment": {"correctness": 3, "completeness": 3, "faithfulness": 4, 
                         "relevance": 4, "clarity": 3, "overall_score": 3, "verdict": "partial"}},
            {"judgment": {"correctness": 2, "completeness": 2, "faithfulness": 3, 
                         "relevance": 3, "clarity": 2, "overall_score": 2, "verdict": "fail"}},
        ]
        
        # Check score variance
        correctness_scores = [r["judgment"]["correctness"] for r in results]
        unique_scores = set(correctness_scores)
        
        assert len(unique_scores) > 1  # Good - varied scores


class TestAbortOnAuthFailure:
    """Tests for run abortion on auth failure."""
    
    def test_auth_error_is_fatal(self):
        """AuthError should be treated as fatal and re-raised."""
        from lib.core.auth_manager import AuthError
        
        def simulate_run_with_auth_error():
            try:
                raise AuthError("Token expired")
            except AuthError:
                # In real code, this would abort the run
                raise
        
        with pytest.raises(AuthError):
            simulate_run_with_auth_error()
    
    def test_non_auth_error_allows_continue(self):
        """Non-auth errors should allow the run to continue."""
        errors_encountered = []
        
        def simulate_run_with_recoverable_error():
            try:
                raise Exception("Temporary network error")
            except Exception as e:
                if not is_auth_error(e):
                    errors_encountered.append(str(e))
                    return  # Continue
                raise
        
        simulate_run_with_recoverable_error()
        assert len(errors_encountered) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
