#!/usr/bin/env python3
"""
Unit tests for auth_manager module.

Tests ADC credential refresh, validation, and error detection.
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.core.auth_manager import (
    AuthManager, AuthError,
    is_auth_error, is_rate_limit_error,
    get_auth_manager, refresh_adc_credentials,
    check_auth_valid, should_refresh_token,
    TOKEN_REFRESH_INTERVAL_SECONDS
)


class TestIsAuthError:
    """Tests for is_auth_error() function."""
    
    def test_detects_503_with_reauthentication(self):
        """503 + reauthentication message = auth error."""
        error = Exception("503 Getting metadata from plugin failed: Reauthentication is needed")
        assert is_auth_error(error) is True
    
    def test_detects_token_expired(self):
        """Token expired message = auth error."""
        error = Exception("Token has been expired or revoked")
        assert is_auth_error(error) is True
    
    def test_detects_invalid_grant(self):
        """Invalid grant = auth error."""
        error = Exception("invalid_grant: Token has been expired")
        assert is_auth_error(error) is True
    
    def test_detects_credentials_error(self):
        """Credentials error = auth error."""
        error = Exception("Could not automatically determine credentials")
        assert is_auth_error(error) is True
    
    def test_detects_refresh_token_error(self):
        """Refresh token error = auth error."""
        error = Exception("The refresh token is invalid or expired")
        assert is_auth_error(error) is True
    
    def test_not_auth_error_for_rate_limit(self):
        """Rate limit errors should NOT be classified as auth errors."""
        error = Exception("429 Resource Exhausted: Quota exceeded")
        assert is_auth_error(error) is False
    
    def test_not_auth_error_for_generic_500(self):
        """Generic 500 without auth message = not auth error."""
        error = Exception("500 Internal Server Error")
        assert is_auth_error(error) is False
    
    def test_not_auth_error_for_network_error(self):
        """Network errors = not auth error."""
        error = Exception("Connection refused")
        assert is_auth_error(error) is False
    
    def test_503_without_auth_message_not_auth_error(self):
        """503 without auth-related message = not auth error."""
        error = Exception("503 Service Unavailable")
        assert is_auth_error(error) is False


class TestIsRateLimitError:
    """Tests for is_rate_limit_error() function."""
    
    def test_detects_429(self):
        """429 status = rate limit."""
        error = Exception("429 Too Many Requests")
        assert is_rate_limit_error(error) is True
    
    def test_detects_resource_exhausted(self):
        """Resource exhausted = rate limit."""
        error = Exception("RESOURCE_EXHAUSTED: Quota exceeded")
        assert is_rate_limit_error(error) is True
    
    def test_detects_quota_exceeded(self):
        """Quota exceeded = rate limit."""
        error = Exception("Quota exceeded for this API")
        assert is_rate_limit_error(error) is True
    
    def test_not_rate_limit_for_auth_error(self):
        """Auth errors should NOT be classified as rate limit."""
        error = Exception("503 Reauthentication is needed")
        assert is_rate_limit_error(error) is False
    
    def test_not_rate_limit_for_generic_error(self):
        """Generic errors = not rate limit."""
        error = Exception("Something went wrong")
        assert is_rate_limit_error(error) is False


class TestAuthManager:
    """Tests for AuthManager class."""
    
    def test_singleton_pattern(self):
        """get_auth_manager returns same instance."""
        # Reset singleton for test
        import lib.core.auth_manager as am
        am._auth_manager = None
        
        manager1 = get_auth_manager()
        manager2 = get_auth_manager()
        assert manager1 is manager2
    
    def test_should_refresh_initially_true(self):
        """should_refresh returns True when never refreshed."""
        manager = AuthManager()
        assert manager.should_refresh() is True
    
    def test_should_refresh_false_after_refresh(self):
        """should_refresh returns False immediately after refresh."""
        manager = AuthManager()
        manager._last_refresh = time.time()
        assert manager.should_refresh() is False
    
    def test_should_refresh_true_after_interval(self):
        """should_refresh returns True after TOKEN_REFRESH_INTERVAL_SECONDS."""
        manager = AuthManager()
        # Set last refresh to 46 minutes ago
        manager._last_refresh = time.time() - (46 * 60)
        assert manager.should_refresh() is True
    
    def test_get_time_since_refresh_infinite_when_never_refreshed(self):
        """get_time_since_refresh returns infinity when never refreshed."""
        manager = AuthManager()
        assert manager.get_time_since_refresh() == float('inf')
    
    def test_get_time_since_refresh_accurate(self):
        """get_time_since_refresh returns accurate time."""
        manager = AuthManager()
        manager._last_refresh = time.time() - 100
        elapsed = manager.get_time_since_refresh()
        assert 99 <= elapsed <= 101  # Allow 1 second tolerance
    
    def test_reset_clears_state(self):
        """reset() clears all state."""
        manager = AuthManager()
        manager._credentials = Mock()
        manager._project = "test-project"
        manager._last_refresh = time.time()
        
        manager.reset()
        
        assert manager._credentials is None
        assert manager._project is None
        assert manager._last_refresh == 0
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_get_credentials_caches_result(self, mock_default):
        """_get_credentials caches credentials after first call."""
        mock_creds = Mock()
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        creds1, proj1 = manager._get_credentials()
        creds2, proj2 = manager._get_credentials()
        
        # Should only call google.auth.default once
        assert mock_default.call_count == 1
        assert creds1 is creds2
        assert proj1 == proj2 == "test-project"
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_get_credentials_raises_auth_error_on_failure(self, mock_default):
        """_get_credentials raises AuthError when no credentials found."""
        from google.auth.exceptions import DefaultCredentialsError
        mock_default.side_effect = DefaultCredentialsError("No credentials")
        
        manager = AuthManager()
        with pytest.raises(AuthError) as exc_info:
            manager._get_credentials()
        
        assert "gcloud auth application-default login" in str(exc_info.value)
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_refresh_updates_last_refresh_time(self, mock_default):
        """refresh() updates _last_refresh timestamp."""
        mock_creds = Mock()
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        before = time.time()
        manager.refresh()
        after = time.time()
        
        assert before <= manager._last_refresh <= after
        mock_creds.refresh.assert_called_once()
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_refresh_raises_auth_error_on_refresh_failure(self, mock_default):
        """refresh() raises AuthError when token refresh fails."""
        from google.auth.exceptions import RefreshError
        mock_creds = Mock()
        mock_creds.refresh.side_effect = RefreshError("Refresh failed")
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        with pytest.raises(AuthError) as exc_info:
            manager.refresh()
        
        assert "Token refresh failed" in str(exc_info.value)
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_ensure_valid_refreshes_and_validates(self, mock_default):
        """ensure_valid() refreshes token and validates credentials."""
        mock_creds = Mock()
        mock_creds.expired = False
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        result = manager.ensure_valid()
        
        assert result is True
        mock_creds.refresh.assert_called_once()
    
    @patch('lib.core.auth_manager.google.auth.default')
    def test_ensure_valid_raises_if_expired_after_refresh(self, mock_default):
        """ensure_valid() raises AuthError if credentials still expired after refresh."""
        mock_creds = Mock()
        mock_creds.expired = True
        mock_default.return_value = (mock_creds, "test-project")
        
        manager = AuthManager()
        with pytest.raises(AuthError) as exc_info:
            manager.ensure_valid()
        
        assert "expired immediately after refresh" in str(exc_info.value)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    @patch('lib.core.auth_manager.get_auth_manager')
    def test_refresh_adc_credentials_calls_manager(self, mock_get_manager):
        """refresh_adc_credentials() delegates to AuthManager."""
        mock_manager = Mock()
        mock_manager.refresh.return_value = True
        mock_get_manager.return_value = mock_manager
        
        result = refresh_adc_credentials()
        
        assert result is True
        mock_manager.refresh.assert_called_once()
    
    @patch('lib.core.auth_manager.get_auth_manager')
    def test_check_auth_valid_calls_manager(self, mock_get_manager):
        """check_auth_valid() delegates to AuthManager."""
        mock_manager = Mock()
        mock_manager.ensure_valid.return_value = True
        mock_get_manager.return_value = mock_manager
        
        result = check_auth_valid()
        
        assert result is True
        mock_manager.ensure_valid.assert_called_once()
    
    @patch('lib.core.auth_manager.get_auth_manager')
    def test_should_refresh_token_calls_manager(self, mock_get_manager):
        """should_refresh_token() delegates to AuthManager."""
        mock_manager = Mock()
        mock_manager.should_refresh.return_value = True
        mock_get_manager.return_value = mock_manager
        
        result = should_refresh_token()
        
        assert result is True
        mock_manager.should_refresh.assert_called_once()


class TestTokenRefreshInterval:
    """Tests for token refresh interval configuration."""
    
    def test_refresh_interval_is_45_minutes(self):
        """TOKEN_REFRESH_INTERVAL_SECONDS should be 45 minutes."""
        assert TOKEN_REFRESH_INTERVAL_SECONDS == 45 * 60
        assert TOKEN_REFRESH_INTERVAL_SECONDS == 2700


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
