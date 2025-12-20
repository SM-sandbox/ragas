"""
Integration tests for SmartRateLimiter.

Tests cover:
- Pre-throttling behavior
- Model rotation flag behavior
- Statistics and monitoring
- Sliding window behavior
- Worker staggering
- Edge cases
"""

import pytest
import time

from lib.core.smart_throttler.rate_limiter import (
    SmartRateLimiter,
    RateLimitConfig,
    UsageRecord,
    ThrottleReason,
    reset_rate_limiter,
)


class TestRateLimiterPreThrottling:
    """Tests for pre-throttling behavior."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_rate_limiter()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_rate_limiter()
    
    def test_pre_throttle_at_threshold(self):
        """Test that throttling kicks in at threshold."""
        config = RateLimitConfig(rpm_limit=10, threshold=0.9)
        limiter = SmartRateLimiter(config)
        
        # Add 9 records (90% of limit)
        for _ in range(9):
            limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        
        stats = limiter.get_stats()
        
        assert stats.is_throttled is True
        assert stats.throttle_reason == ThrottleReason.RPM_LIMIT
    
    def test_no_throttle_below_threshold(self):
        """Test no throttling below threshold."""
        config = RateLimitConfig(rpm_limit=10, threshold=0.9)
        limiter = SmartRateLimiter(config)
        
        # Add 5 records (50% of limit)
        for _ in range(5):
            limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        
        stats = limiter.get_stats()
        
        assert stats.is_throttled is False
        assert stats.throttle_reason == ThrottleReason.NONE


class TestModelRotationFlag:
    """Tests for model rotation flag behavior."""
    
    def setup_method(self):
        reset_rate_limiter()
    
    def teardown_method(self):
        reset_rate_limiter()
    
    def test_rotation_disabled_returns_none(self):
        """Test that rotation returns None when disabled."""
        config = RateLimitConfig(enable_model_rotation=False)
        limiter = SmartRateLimiter(config)
        
        result = limiter.mark_model_rate_limited("gemini-3-flash-preview")
        
        assert result is None
    
    def test_rotation_enabled_returns_new_model(self):
        """Test that rotation returns new model when enabled."""
        config = RateLimitConfig(
            enable_model_rotation=True,
            rotation_models=["model-a", "model-b", "model-c"],
        )
        limiter = SmartRateLimiter(config)
        
        result = limiter.mark_model_rate_limited("model-a")
        
        assert result == "model-b"


class TestSlidingWindowBehavior:
    """Tests for sliding window behavior."""
    
    def setup_method(self):
        reset_rate_limiter()
    
    def teardown_method(self):
        reset_rate_limiter()
    
    def test_old_records_expire(self):
        """Test that old records are removed from window."""
        config = RateLimitConfig(window_size=1)
        limiter = SmartRateLimiter(config)
        
        limiter._usage_records.append(UsageRecord(time.time() - 2, 100, "test"))
        
        assert limiter._get_current_rpm() == 0
    
    def test_recent_records_kept(self):
        """Test that recent records are kept in window."""
        config = RateLimitConfig(window_size=60)
        limiter = SmartRateLimiter(config)
        
        limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        
        assert limiter._get_current_rpm() == 1


class TestEdgeCasesIntegration:
    """Integration tests for edge cases."""
    
    def setup_method(self):
        reset_rate_limiter()
    
    def teardown_method(self):
        reset_rate_limiter()
    
    def test_zero_limits_handled(self):
        """Test that zero limits are handled gracefully."""
        config = RateLimitConfig(rpm_limit=0, tpm_limit=0)
        limiter = SmartRateLimiter(config)
        
        stats = limiter.get_stats()
        
        assert stats.rpm_utilization == 0
        assert stats.tpm_utilization == 0
    
    def test_very_high_limits(self):
        """Test handling of very high limits."""
        config = RateLimitConfig(rpm_limit=1000000, tpm_limit=1000000000)
        limiter = SmartRateLimiter(config)
        
        for _ in range(100):
            limiter._usage_records.append(UsageRecord(time.time(), 10000, "test"))
        
        stats = limiter.get_stats()
        
        assert stats.is_throttled is False
