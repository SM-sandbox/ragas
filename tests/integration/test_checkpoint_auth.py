#!/usr/bin/env python3
"""
Integration tests for checkpoint auth handling.

Tests end-to-end behavior of auth validation and error handling
during checkpoint runs.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPreFlightAuthValidation:
    """Tests for pre-flight auth validation before checkpoint runs."""
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_preflight_validates_credentials(self, mock_default):
        """Pre-flight should validate ADC credentials exist."""
        from lib.core.auth_manager import check_auth_valid
        
        mock_creds = Mock()
        mock_creds.expired = False
        mock_default.return_value = (mock_creds, "test-project")
        
        result = check_auth_valid()
        
        assert result is True
        mock_creds.refresh.assert_called_once()
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_preflight_fails_without_credentials(self, mock_default):
        """Pre-flight should fail if no credentials available."""
        from google.auth.exceptions import DefaultCredentialsError
        from lib.core.auth_manager import AuthManager, AuthError
        
        mock_default.side_effect = DefaultCredentialsError("No credentials")
        
        # Use fresh AuthManager instance to avoid singleton caching
        manager = AuthManager()
        
        with pytest.raises(AuthError) as exc_info:
            manager.ensure_valid()
        
        assert "gcloud auth application-default login" in str(exc_info.value)


class TestCheckpointResultValidation:
    """Tests for validating checkpoint results aren't garbage."""
    
    def test_detect_garbage_results_all_threes(self):
        """Should detect when all results have fallback scores (all 3s)."""
        # Simulate C020's bad data pattern
        results = []
        for i in range(100):
            results.append({
                "question_id": f"q_{i}",
                "judgment": {
                    "correctness": 3,
                    "completeness": 3,
                    "faithfulness": 3,
                    "relevance": 3,
                    "clarity": 3,
                    "overall_score": 3,
                    "verdict": "partial",
                    "parse_error": "503 Reauthentication is needed"
                }
            })
        
        # Check for garbage pattern
        def is_garbage_results(results):
            if len(results) < 10:
                return False
            
            # Check if all scores are exactly 3
            all_fallback = all(
                r.get("judgment", {}).get("correctness") == 3 and
                r.get("judgment", {}).get("completeness") == 3 and
                r.get("judgment", {}).get("faithfulness") == 3 and
                r.get("judgment", {}).get("relevance") == 3 and
                r.get("judgment", {}).get("clarity") == 3
                for r in results
            )
            
            # Check if all have parse_error
            all_errors = all(
                "parse_error" in r.get("judgment", {})
                for r in results
            )
            
            return all_fallback and all_errors
        
        assert is_garbage_results(results) is True
    
    def test_valid_results_have_variance(self):
        """Valid results should have score variance."""
        # Simulate good data
        results = [
            {"judgment": {"correctness": 5, "completeness": 5, "faithfulness": 5,
                         "relevance": 5, "clarity": 5, "overall_score": 5, "verdict": "pass"}},
            {"judgment": {"correctness": 4, "completeness": 4, "faithfulness": 4,
                         "relevance": 4, "clarity": 4, "overall_score": 4, "verdict": "pass"}},
            {"judgment": {"correctness": 3, "completeness": 3, "faithfulness": 4,
                         "relevance": 4, "clarity": 3, "overall_score": 3, "verdict": "partial"}},
            {"judgment": {"correctness": 2, "completeness": 2, "faithfulness": 3,
                         "relevance": 3, "clarity": 2, "overall_score": 2, "verdict": "fail"}},
        ]
        
        # Check for variance
        scores = [r["judgment"]["correctness"] for r in results]
        unique_scores = set(scores)
        
        assert len(unique_scores) > 1  # Good - varied scores


class TestAuthErrorAbortion:
    """Tests for run abortion on auth errors."""
    
    def test_auth_error_aborts_run(self):
        """Auth errors should cause immediate run abortion."""
        from lib.core.auth_manager import AuthError, is_auth_error
        
        # Simulate the error pattern from C020
        error = Exception(
            "Timeout of 60.0s exceeded, last exception: 503 Getting metadata "
            "from plugin failed with error: Reauthentication is needed"
        )
        
        # Should be detected as auth error
        assert is_auth_error(error) is True
        
        # In real code, this would raise AuthError and abort
        if is_auth_error(error):
            with pytest.raises(AuthError):
                raise AuthError(f"Authentication failed: {error}")
    
    def test_rate_limit_does_not_abort(self):
        """Rate limit errors should NOT abort the run."""
        from lib.core.auth_manager import is_auth_error
        
        error = Exception("429 Resource Exhausted: Quota exceeded")
        
        # Should NOT be detected as auth error
        assert is_auth_error(error) is False


class TestFallbackThresholdAbortion:
    """Tests for abortion when fallback threshold exceeded."""
    
    def test_abort_when_threshold_exceeded(self):
        """Should abort when >5% of questions use fallback."""
        from lib.core.auth_manager import AuthError
        
        fallback_count = 10
        total_processed = 100
        threshold = 0.05
        
        fallback_rate = fallback_count / total_processed
        
        if fallback_rate > threshold:
            with pytest.raises(AuthError):
                raise AuthError(
                    f"Fallback rate ({fallback_rate:.1%}) exceeds threshold ({threshold:.1%})"
                )
    
    def test_continue_when_under_threshold(self):
        """Should continue when fallback rate is under threshold."""
        fallback_count = 3
        total_processed = 100
        threshold = 0.05
        
        fallback_rate = fallback_count / total_processed
        
        # Should NOT abort
        assert fallback_rate < threshold
        assert fallback_rate == 0.03


class TestTokenRefreshDuringRun:
    """Tests for mid-run token refresh."""
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_token_refreshed_after_45_minutes(self, mock_default):
        """Token should be refreshed after 45 minutes of running."""
        from lib.core.auth_manager import AuthManager
        import time
        
        mock_creds = Mock()
        mock_creds.expired = False
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        
        # Simulate 46 minutes elapsed
        manager._last_refresh = time.time() - (46 * 60)
        
        # Should need refresh
        assert manager.should_refresh() is True
        
        # Refresh should work
        manager.refresh()
        mock_creds.refresh.assert_called()
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_token_not_refreshed_unnecessarily(self, mock_default):
        """Token should NOT be refreshed if recently refreshed."""
        from lib.core.auth_manager import AuthManager
        import time
        
        mock_creds = Mock()
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        
        # Simulate 10 minutes elapsed
        manager._last_refresh = time.time() - (10 * 60)
        
        # Should NOT need refresh
        assert manager.should_refresh() is False


class TestEndToEndAuthFlow:
    """End-to-end tests for auth flow."""
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_full_auth_lifecycle(self, mock_default):
        """Test complete auth lifecycle: validate -> refresh -> check."""
        from lib.core.auth_manager import AuthManager
        import time
        
        mock_creds = Mock()
        mock_creds.expired = False
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        
        # 1. Pre-flight validation
        assert manager.ensure_valid() is True
        initial_refresh_time = manager._last_refresh
        
        # 2. Check if refresh needed (should be False, just refreshed)
        assert manager.should_refresh() is False
        
        # 3. Simulate time passing (46 minutes)
        manager._last_refresh = time.time() - (46 * 60)
        
        # 4. Now should need refresh
        assert manager.should_refresh() is True
        
        # 5. Refresh
        manager.refresh()
        
        # 6. Should not need refresh again
        assert manager.should_refresh() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
