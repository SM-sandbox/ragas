"""
Auth Manager for RAG Eval Suite

Handles ADC (Application Default Credentials) refresh and validation
to prevent silent failures during long-running checkpoint evaluations.

Features:
- Pre-flight auth validation before runs
- Mid-run token refresh (every 45 minutes)
- Auth error detection and classification
"""

import logging
import time
from typing import Optional, Tuple

import google.auth
import google.auth.transport.requests
from google.auth.exceptions import RefreshError, DefaultCredentialsError

logger = logging.getLogger(__name__)

# Token refresh interval (45 minutes = 2700 seconds)
# ADC tokens expire after 1 hour, so refresh at 45 min for safety margin
TOKEN_REFRESH_INTERVAL_SECONDS = 45 * 60

# Auth error patterns to detect
AUTH_ERROR_PATTERNS = [
    "reauthentication is needed",
    "reauthenticate",
    "token expired",
    "credentials",
    "invalid_grant",
    "token has been expired",
    "refresh token",
    "authentication required",
]


class AuthError(Exception):
    """Raised when authentication fails and cannot be recovered."""
    pass


class AuthManager:
    """
    Manages ADC credentials for long-running evaluation jobs.
    
    Usage:
        auth = AuthManager()
        auth.ensure_valid()  # Pre-flight check
        
        # During long run:
        if auth.should_refresh():
            auth.refresh()
    """
    
    def __init__(self):
        self._credentials: Optional[google.auth.credentials.Credentials] = None
        self._project: Optional[str] = None
        self._last_refresh: float = 0
        self._request = google.auth.transport.requests.Request()
    
    def _get_credentials(self) -> Tuple[google.auth.credentials.Credentials, str]:
        """Get or load ADC credentials."""
        if self._credentials is None:
            try:
                self._credentials, self._project = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                logger.info(f"Loaded ADC credentials for project: {self._project}")
            except DefaultCredentialsError as e:
                raise AuthError(
                    f"No valid credentials found. Run 'gcloud auth application-default login'. Error: {e}"
                )
        return self._credentials, self._project
    
    def refresh(self) -> bool:
        """
        Force refresh the ADC token.
        
        Returns:
            True if refresh succeeded, False otherwise
        """
        try:
            credentials, _ = self._get_credentials()
            credentials.refresh(self._request)
            self._last_refresh = time.time()
            logger.info("ADC token refreshed successfully")
            return True
        except RefreshError as e:
            logger.error(f"Failed to refresh ADC token: {e}")
            raise AuthError(f"Token refresh failed. Run 'gcloud auth application-default login'. Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error refreshing token: {e}")
            raise AuthError(f"Token refresh failed unexpectedly: {e}")
    
    def should_refresh(self) -> bool:
        """Check if token should be refreshed (>45 min since last refresh)."""
        if self._last_refresh == 0:
            return True
        elapsed = time.time() - self._last_refresh
        return elapsed >= TOKEN_REFRESH_INTERVAL_SECONDS
    
    def ensure_valid(self) -> bool:
        """
        Pre-flight check: ensure credentials are valid and refreshed.
        
        Raises:
            AuthError: If credentials cannot be validated
            
        Returns:
            True if credentials are valid
        """
        logger.info("Running pre-flight auth check...")
        
        # Get credentials
        credentials, project = self._get_credentials()
        
        # Force refresh to ensure we have a fresh token
        self.refresh()
        
        # Verify token is valid by checking expiry
        if hasattr(credentials, 'expired') and credentials.expired:
            raise AuthError("Credentials expired immediately after refresh")
        
        logger.info(f"Pre-flight auth check passed for project: {project}")
        return True
    
    def get_time_since_refresh(self) -> float:
        """Get seconds since last token refresh."""
        if self._last_refresh == 0:
            return float('inf')
        return time.time() - self._last_refresh
    
    def reset(self):
        """Reset the auth manager state (useful for testing)."""
        self._credentials = None
        self._project = None
        self._last_refresh = 0


# Singleton instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get the singleton AuthManager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def refresh_adc_credentials() -> bool:
    """Convenience function to refresh ADC credentials."""
    return get_auth_manager().refresh()


def check_auth_valid() -> bool:
    """Convenience function for pre-flight auth check."""
    return get_auth_manager().ensure_valid()


def should_refresh_token() -> bool:
    """Convenience function to check if refresh is needed."""
    return get_auth_manager().should_refresh()


def is_auth_error(error: Exception) -> bool:
    """
    Determine if an exception is an authentication error.
    
    Args:
        error: The exception to check
        
    Returns:
        True if this is an auth error that requires re-authentication
    """
    error_str = str(error).lower()
    
    # Check for 503 status code (often indicates auth issues with Google APIs)
    if "503" in error_str:
        # 503 + auth-related message = auth error
        for pattern in AUTH_ERROR_PATTERNS:
            if pattern in error_str:
                return True
    
    # Check for explicit auth error patterns
    for pattern in AUTH_ERROR_PATTERNS:
        if pattern in error_str:
            return True
    
    # Check for specific exception types
    if isinstance(error, (RefreshError, DefaultCredentialsError)):
        return True
    
    return False


def is_rate_limit_error(error: Exception) -> bool:
    """
    Determine if an exception is a rate limit error (not auth).
    
    Args:
        error: The exception to check
        
    Returns:
        True if this is a rate limit error
    """
    error_str = str(error).lower()
    rate_limit_patterns = [
        "429",
        "resource_exhausted",
        "quota exceeded",
        "rate limit",
        "too many requests",
    ]
    
    for pattern in rate_limit_patterns:
        if pattern in error_str:
            return True
    
    return False
