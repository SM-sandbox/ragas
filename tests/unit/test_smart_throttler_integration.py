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
    configure_rate_limiter,
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
        """Reset singletons before each test."""
        reset_rate_limiter()
    
    def teardown_method(self):
        """Clean up after each test."""
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
    
    def test_rotation_tracks_cooldowns(self):
        """Test that rotation tracks model cooldowns."""
        config = RateLimitConfig(
            enable_model_rotation=True,
            model_cooldown_seconds=60.0,
        )
        limiter = SmartRateLimiter(config)
        
        limiter.mark_model_rate_limited("gemini-3-flash-preview")
        
        assert "gemini-3-flash-preview" in limiter._model_cooldowns


class TestRateLimiterStatistics:
    """Tests for rate limiter statistics."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_rate_limiter()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_rate_limiter()
    
    def test_stats_utilization_calculation(self):
        """Test utilization percentage calculation."""
        config = RateLimitConfig(rpm_limit=100, tpm_limit=10000)
        limiter = SmartRateLimiter(config)
        
        # Add 50 records with 50 tokens each = 2500 tokens
        for _ in range(50):
            limiter._usage_records.append(UsageRecord(time.time(), 50, "test"))
        
        stats = limiter.get_stats()
        
        assert stats.current_rpm == 50
        assert stats.current_tpm == 2500
        assert stats.rpm_utilization == 0.5  # 50/100
        assert stats.tpm_utilization == 0.25  # 2500/10000
    
    def test_stats_active_model(self):
        """Test that stats include active model."""
        config = RateLimitConfig(rotation_models=["model-a", "model-b"])
        limiter = SmartRateLimiter(config)
        
        stats = limiter.get_stats()
        
        assert stats.active_model == "model-a"
    
    def test_stats_models_on_cooldown(self):
        """Test that stats include models on cooldown."""
        config = RateLimitConfig(
            enable_model_rotation=True,
            model_cooldown_seconds=60.0,
        )
        limiter = SmartRateLimiter(config)
        
        limiter.mark_model_rate_limited("gemini-3-flash-preview")
        
        stats = limiter.get_stats()
        
        assert "gemini-3-flash-preview" in stats.models_on_cooldown


class TestSlidingWindowBehavior:
    """Tests for sliding window behavior."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_rate_limiter()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_rate_limiter()
    
    def test_old_records_expire(self):
        """Test that old records are removed from window."""
        config = RateLimitConfig(window_size=1)  # 1 second window
        limiter = SmartRateLimiter(config)
        
        # Add record from 2 seconds ago
        limiter._usage_records.append(UsageRecord(time.time() - 2, 100, "test"))
        
        # Should be cleaned up
        assert limiter._get_current_rpm() == 0
    
    def test_recent_records_kept(self):
        """Test that recent records are kept in window."""
        config = RateLimitConfig(window_size=60)
        limiter = SmartRateLimiter(config)
        
        # Add recent record
        limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        
        # Should be kept
        assert limiter._get_current_rpm() == 1
    
    def test_window_respects_size(self):
        """Test that window respects configured size."""
        config = RateLimitConfig(window_size=5)  # 5 second window
        limiter = SmartRateLimiter(config)
        
        # Add records at different times
        limiter._usage_records.append(UsageRecord(time.time() - 10, 100, "old"))
        limiter._usage_records.append(UsageRecord(time.time() - 3, 100, "recent"))
        limiter._usage_records.append(UsageRecord(time.time(), 100, "now"))
        
        # Only 2 should be in window
        assert limiter._get_current_rpm() == 2


class TestWorkerStaggering:
    """Tests for worker staggering behavior."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_rate_limiter()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_rate_limiter()
    
    def test_stagger_delay_within_bounds(self):
        """Test that stagger delays are within configured bounds."""
        config = RateLimitConfig(max_stagger_delay=2.0)
        limiter = SmartRateLimiter(config)
        
        for i in range(100):
            delay = limiter.get_stagger_delay(i)
            assert 0 <= delay <= 2.0
    
    def test_stagger_delay_consistent_for_worker(self):
        """Test that same worker gets same delay."""
        limiter = SmartRateLimiter()
        
        delay1 = limiter.get_stagger_delay(42)
        delay2 = limiter.get_stagger_delay(42)
        
        assert delay1 == delay2
    
    def test_stagger_delays_vary_across_workers(self):
        """Test that different workers get different delays."""
        limiter = SmartRateLimiter()
        
        delays = set()
        for i in range(50):
            delays.add(limiter.get_stagger_delay(i))
        
        # Should have multiple unique delays
        assert len(delays) > 1


class TestEdgeCasesIntegration:
    """Integration tests for edge cases."""
    
    def setup_method(self):
        """Reset singletons before each test."""
        reset_rate_limiter()
    
    def teardown_method(self):
        """Clean up after each test."""
        reset_rate_limiter()
    
    def test_zero_limits_handled(self):
        """Test that zero limits are handled gracefully."""
        config = RateLimitConfig(rpm_limit=0, tpm_limit=0)
        limiter = SmartRateLimiter(config)
        
        stats = limiter.get_stats()
        
        # Should not crash
        assert stats.rpm_utilization == 0
        assert stats.tpm_utilization == 0
    
    def test_very_high_limits(self):
        """Test handling of very high limits."""
        config = RateLimitConfig(rpm_limit=1000000, tpm_limit=1000000000)
        limiter = SmartRateLimiter(config)
        
        # Add some records
        for _ in range(100):
            limiter._usage_records.append(UsageRecord(time.time(), 10000, "test"))
        
        stats = limiter.get_stats()
        
        # Should not be throttled
        assert stats.is_throttled is False
    
    def test_empty_rotation_models(self):
        """Test handling of empty rotation models list."""
        config = RateLimitConfig(rotation_models=[])
        limiter = SmartRateLimiter(config)
        
        # Should use default model
        assert limiter._active_model == "gemini-3-flash-preview"
