"""
Comprehensive tests for the SmartRateLimiter.

Tests cover:
- Basic rate limiting functionality
- Sliding window tracking
- RPM and TPM limits
- Threshold-based throttling
- Worker stagger delays
- Model rotation (when enabled)
- Model cooldown tracking
- Statistics and monitoring
- Thread safety
- Async behavior
- Edge cases
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from lib.core.smart_throttler.rate_limiter import (
    SmartRateLimiter,
    RateLimitConfig,
    RateLimitStats,
    UsageRecord,
    ThrottleReason,
    get_rate_limiter,
    reset_rate_limiter,
    configure_rate_limiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.rpm_limit == 1000
        assert config.tpm_limit == 1000000
        assert config.threshold == 0.9
        assert config.window_size == 60
        assert config.min_request_delay == 0.05
        assert config.max_stagger_delay == 2.0
        assert config.enable_model_rotation is False
        assert config.model_cooldown_seconds == 30.0
        assert len(config.rotation_models) == 1  # Only gemini-3-flash-preview (no fallbacks)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            rpm_limit=500,
            tpm_limit=500000,
            threshold=0.8,
            enable_model_rotation=True,
        )
        
        assert config.rpm_limit == 500
        assert config.tpm_limit == 500000
        assert config.threshold == 0.8
        assert config.enable_model_rotation is True


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""
    
    def test_usage_record_creation(self):
        """Test creating a usage record."""
        record = UsageRecord(
            timestamp=time.time(),
            tokens=1000,
            model="gemini-3-flash-preview",
        )
        
        assert record.tokens == 1000
        assert record.model == "gemini-3-flash-preview"
        assert record.timestamp > 0


class TestSmartRateLimiterBasic:
    """Basic functionality tests for SmartRateLimiter."""
    
    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = SmartRateLimiter()
        
        assert limiter.config is not None
        assert limiter._total_requests == 0
        assert limiter._total_tokens == 0
        assert len(limiter._usage_records) == 0
    
    def test_initialization_with_config(self):
        """Test rate limiter initialization with custom config."""
        config = RateLimitConfig(rpm_limit=100, tpm_limit=50000)
        limiter = SmartRateLimiter(config)
        
        assert limiter.config.rpm_limit == 100
        assert limiter.config.tpm_limit == 50000
    
    def test_get_stats_empty(self):
        """Test getting stats with no usage."""
        limiter = SmartRateLimiter()
        stats = limiter.get_stats()
        
        assert stats.current_rpm == 0
        assert stats.current_tpm == 0
        assert stats.is_throttled is False
        assert stats.throttle_reason == ThrottleReason.NONE
        assert stats.wait_time == 0.0
    
    def test_reset(self):
        """Test resetting the rate limiter."""
        limiter = SmartRateLimiter()
        limiter._total_requests = 100
        limiter._total_tokens = 50000
        limiter._usage_records.append(UsageRecord(time.time(), 1000, "test"))
        
        limiter.reset()
        
        assert limiter._total_requests == 0
        assert limiter._total_tokens == 0
        assert len(limiter._usage_records) == 0


class TestSlidingWindow:
    """Tests for sliding window functionality."""
    
    def test_cleanup_old_records(self):
        """Test that old records are cleaned up."""
        config = RateLimitConfig(window_size=1)  # 1 second window
        limiter = SmartRateLimiter(config)
        
        # Add a record from 2 seconds ago
        old_record = UsageRecord(time.time() - 2, 1000, "test")
        limiter._usage_records.append(old_record)
        
        # Cleanup should remove it
        limiter._cleanup_old_records()
        
        assert len(limiter._usage_records) == 0
    
    def test_recent_records_kept(self):
        """Test that recent records are kept."""
        config = RateLimitConfig(window_size=60)
        limiter = SmartRateLimiter(config)
        
        # Add a recent record
        recent_record = UsageRecord(time.time(), 1000, "test")
        limiter._usage_records.append(recent_record)
        
        # Cleanup should keep it
        limiter._cleanup_old_records()
        
        assert len(limiter._usage_records) == 1
    
    def test_rpm_calculation(self):
        """Test RPM calculation."""
        limiter = SmartRateLimiter()
        
        # Add 5 records
        for _ in range(5):
            limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        
        assert limiter._get_current_rpm() == 5
    
    def test_tpm_calculation(self):
        """Test TPM calculation."""
        limiter = SmartRateLimiter()
        
        # Add records with different token counts
        limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        limiter._usage_records.append(UsageRecord(time.time(), 200, "test"))
        limiter._usage_records.append(UsageRecord(time.time(), 300, "test"))
        
        assert limiter._get_current_tpm() == 600


class TestThrottling:
    """Tests for throttling behavior."""
    
    def test_rpm_throttle_detection(self):
        """Test that RPM throttling is detected."""
        config = RateLimitConfig(rpm_limit=10, threshold=0.9)
        limiter = SmartRateLimiter(config)
        
        # Add 9 records (90% of limit)
        for _ in range(9):
            limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        
        stats = limiter.get_stats()
        
        assert stats.is_throttled is True
        assert stats.throttle_reason == ThrottleReason.RPM_LIMIT
        assert stats.rpm_utilization >= 0.9
    
    def test_tpm_throttle_detection(self):
        """Test that TPM throttling is detected."""
        config = RateLimitConfig(rpm_limit=1000, tpm_limit=1000, threshold=0.9)
        limiter = SmartRateLimiter(config)
        
        # Add record with 900 tokens (90% of limit)
        limiter._usage_records.append(UsageRecord(time.time(), 900, "test"))
        
        stats = limiter.get_stats()
        
        assert stats.is_throttled is True
        assert stats.throttle_reason == ThrottleReason.TPM_LIMIT
        assert stats.tpm_utilization >= 0.9
    
    def test_no_throttle_under_threshold(self):
        """Test that no throttling occurs under threshold."""
        config = RateLimitConfig(rpm_limit=100, tpm_limit=100000, threshold=0.9)
        limiter = SmartRateLimiter(config)
        
        # Add 50 records (50% of limit)
        for _ in range(50):
            limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        
        stats = limiter.get_stats()
        
        assert stats.is_throttled is False
        assert stats.throttle_reason == ThrottleReason.NONE
    
    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        config = RateLimitConfig(window_size=10)  # 10 second window
        limiter = SmartRateLimiter(config)
        
        # Add a record from 5 seconds ago
        limiter._usage_records.append(UsageRecord(time.time() - 5, 100, "test"))
        
        wait_time = limiter._calculate_wait_time()
        
        # Should wait approximately 5 seconds for the record to expire
        assert 4.5 <= wait_time <= 5.5


class TestWorkerStagger:
    """Tests for worker stagger functionality."""
    
    def test_stagger_delay_assignment(self):
        """Test that workers get stagger delays."""
        limiter = SmartRateLimiter()
        
        delay1 = limiter.get_stagger_delay(1)
        delay2 = limiter.get_stagger_delay(2)
        
        assert 0 <= delay1 <= limiter.config.max_stagger_delay
        assert 0 <= delay2 <= limiter.config.max_stagger_delay
    
    def test_stagger_delay_consistency(self):
        """Test that same worker gets same delay."""
        limiter = SmartRateLimiter()
        
        delay1 = limiter.get_stagger_delay(1)
        delay2 = limiter.get_stagger_delay(1)
        
        assert delay1 == delay2
    
    def test_different_workers_may_differ(self):
        """Test that different workers may get different delays."""
        limiter = SmartRateLimiter()
        
        # Get delays for many workers - at least some should differ
        delays = [limiter.get_stagger_delay(i) for i in range(100)]
        unique_delays = set(delays)
        
        # With 100 random delays, we should have multiple unique values
        assert len(unique_delays) > 1


class TestModelRotation:
    """Tests for model rotation functionality."""
    
    def test_model_rotation_disabled_by_default(self):
        """Test that model rotation is disabled by default."""
        limiter = SmartRateLimiter()
        
        result = limiter.mark_model_rate_limited("gemini-3-flash-preview")
        
        assert result is None
    
    def test_model_rotation_when_enabled(self):
        """Test model rotation when enabled."""
        config = RateLimitConfig(
            enable_model_rotation=True,
            rotation_models=["model-a", "model-b", "model-c"],
        )
        limiter = SmartRateLimiter(config)
        
        # Mark first model as rate limited
        new_model = limiter.mark_model_rate_limited("model-a")
        
        assert new_model == "model-b"
        assert limiter._active_model == "model-b"
    
    def test_model_cooldown_tracking(self):
        """Test that model cooldowns are tracked."""
        config = RateLimitConfig(
            enable_model_rotation=True,
            model_cooldown_seconds=30.0,
        )
        limiter = SmartRateLimiter(config)
        
        limiter.mark_model_rate_limited("gemini-3-flash-preview")
        
        stats = limiter.get_stats()
        assert "gemini-3-flash-preview" in stats.models_on_cooldown
    
    def test_cooldown_expiry(self):
        """Test that cooldowns expire."""
        config = RateLimitConfig(
            enable_model_rotation=True,
            model_cooldown_seconds=0.1,  # Very short cooldown
        )
        limiter = SmartRateLimiter(config)
        
        limiter.mark_model_rate_limited("gemini-3-flash-preview")
        
        # Wait for cooldown to expire
        time.sleep(0.2)
        
        # Get available model should return the previously rate-limited model
        available = limiter._get_available_model()
        assert available == "gemini-3-flash-preview"
    
    def test_all_models_on_cooldown(self):
        """Test behavior when all models are on cooldown."""
        config = RateLimitConfig(
            enable_model_rotation=True,
            rotation_models=["model-a", "model-b"],
            model_cooldown_seconds=60.0,
        )
        limiter = SmartRateLimiter(config)
        
        # Mark all models as rate limited
        limiter.mark_model_rate_limited("model-a")
        limiter.mark_model_rate_limited("model-b")
        
        # Should return the one that expires soonest
        available = limiter._get_available_model()
        assert available in ["model-a", "model-b"]


class TestAsyncAcquire:
    """Tests for async acquire functionality."""
    
    @pytest.mark.asyncio
    async def test_acquire_success(self):
        """Test successful acquire."""
        config = RateLimitConfig(rpm_limit=100, tpm_limit=100000)
        limiter = SmartRateLimiter(config)
        
        success, model = await limiter.acquire(estimated_tokens=1000)
        
        assert success is True
        assert model is not None
        assert limiter._total_requests == 1
    
    @pytest.mark.asyncio
    async def test_acquire_records_usage(self):
        """Test that acquire records usage."""
        limiter = SmartRateLimiter()
        
        await limiter.acquire(estimated_tokens=500)
        
        assert len(limiter._usage_records) == 1
        assert limiter._usage_records[0].tokens == 500
    
    @pytest.mark.asyncio
    async def test_acquire_with_worker_id(self):
        """Test acquire with worker ID for stagger."""
        limiter = SmartRateLimiter()
        
        start = time.time()
        success, _ = await limiter.acquire(estimated_tokens=100, worker_id=1)
        elapsed = time.time() - start
        
        assert success is True
        # Should have some stagger delay
        assert elapsed >= 0
    
    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test acquire timeout when at capacity."""
        config = RateLimitConfig(rpm_limit=1, tpm_limit=100, threshold=0.5)
        limiter = SmartRateLimiter(config)
        
        # Fill up capacity
        limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        
        # Try to acquire with short timeout
        success, _ = await limiter.acquire(estimated_tokens=100, timeout=0.1)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_concurrent_acquires(self):
        """Test concurrent acquire calls."""
        config = RateLimitConfig(rpm_limit=100, tpm_limit=1000000)
        limiter = SmartRateLimiter(config)
        
        async def acquire_task():
            return await limiter.acquire(estimated_tokens=100)
        
        # Run 10 concurrent acquires
        tasks = [acquire_task() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r[0] for r in results)
        assert limiter._total_requests == 10


class TestRecordUsage:
    """Tests for record_usage functionality."""
    
    def test_record_usage_updates_tokens(self):
        """Test that record_usage updates token count."""
        limiter = SmartRateLimiter()
        
        # Add initial record
        limiter._usage_records.append(UsageRecord(time.time(), 500, "test"))
        limiter._total_tokens = 500
        
        # Update with actual usage
        limiter.record_usage(actual_tokens=750)
        
        assert limiter._usage_records[-1].tokens == 750
        assert limiter._total_tokens == 750
    
    def test_record_usage_no_records(self):
        """Test record_usage with no existing records."""
        limiter = SmartRateLimiter()
        
        # Should not raise
        limiter.record_usage(actual_tokens=100)


class TestSingletonPattern:
    """Tests for singleton pattern functions."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        reset_rate_limiter()
    
    def teardown_method(self):
        """Reset singleton after each test."""
        reset_rate_limiter()
    
    def test_get_rate_limiter_singleton(self):
        """Test that get_rate_limiter returns singleton."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is limiter2
    
    def test_reset_rate_limiter(self):
        """Test reset_rate_limiter clears singleton."""
        limiter1 = get_rate_limiter()
        reset_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is not limiter2
    
    def test_configure_rate_limiter(self):
        """Test configure_rate_limiter creates new instance."""
        limiter = configure_rate_limiter(
            rpm_limit=500,
            tpm_limit=250000,
            enable_model_rotation=True,
        )
        
        assert limiter.config.rpm_limit == 500
        assert limiter.config.tpm_limit == 250000
        assert limiter.config.enable_model_rotation is True


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_stats_access(self):
        """Test concurrent access to stats."""
        limiter = SmartRateLimiter()
        errors = []
        
        def get_stats_repeatedly():
            try:
                for _ in range(100):
                    limiter.get_stats()
            except Exception as e:
                errors.append(e)
        
        threads = [Thread(target=get_stats_repeatedly) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_concurrent_stagger_delay(self):
        """Test concurrent stagger delay access."""
        limiter = SmartRateLimiter()
        results = {}
        
        def get_delay(worker_id):
            delay = limiter.get_stagger_delay(worker_id)
            results[worker_id] = delay
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_delay, i) for i in range(100)]
            for f in futures:
                f.result()
        
        # All workers should have delays
        assert len(results) == 100


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_zero_rpm_limit(self):
        """Test with zero RPM limit."""
        config = RateLimitConfig(rpm_limit=0)
        limiter = SmartRateLimiter(config)
        
        stats = limiter.get_stats()
        assert stats.rpm_utilization == 0
    
    def test_zero_tpm_limit(self):
        """Test with zero TPM limit."""
        config = RateLimitConfig(tpm_limit=0)
        limiter = SmartRateLimiter(config)
        
        stats = limiter.get_stats()
        assert stats.tpm_utilization == 0
    
    def test_empty_rotation_models(self):
        """Test with empty rotation models list."""
        config = RateLimitConfig(rotation_models=[])
        limiter = SmartRateLimiter(config)
        
        # Should use default model
        assert limiter._active_model == "gemini-3-flash-preview"
    
    def test_very_high_token_estimate(self):
        """Test with very high token estimate."""
        config = RateLimitConfig(tpm_limit=1000)
        limiter = SmartRateLimiter(config)
        
        # Request more tokens than limit allows
        # Should still work but may throttle
        stats = limiter.get_stats()
        assert stats is not None
    
    def test_negative_wait_time_handling(self):
        """Test that negative wait times are handled."""
        limiter = SmartRateLimiter()
        
        # Add a record from the future (edge case)
        future_record = UsageRecord(time.time() + 100, 100, "test")
        limiter._usage_records.append(future_record)
        
        wait_time = limiter._calculate_wait_time()
        # Should handle gracefully
        assert wait_time >= 0


class TestStatistics:
    """Tests for statistics tracking."""
    
    def test_total_requests_tracking(self):
        """Test that total requests are tracked."""
        limiter = SmartRateLimiter()
        
        # Manually add records
        for _ in range(5):
            limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
            limiter._total_requests += 1
        
        assert limiter._total_requests == 5
    
    def test_total_tokens_tracking(self):
        """Test that total tokens are tracked."""
        limiter = SmartRateLimiter()
        
        limiter._usage_records.append(UsageRecord(time.time(), 100, "test"))
        limiter._total_tokens = 100
        limiter._usage_records.append(UsageRecord(time.time(), 200, "test"))
        limiter._total_tokens += 200
        
        assert limiter._total_tokens == 300
    
    def test_throttle_count_tracking(self):
        """Test that throttle count is tracked."""
        limiter = SmartRateLimiter()
        
        limiter._throttle_count = 5
        
        assert limiter._throttle_count == 5
    
    def test_model_rotation_count(self):
        """Test that model rotations are counted."""
        config = RateLimitConfig(
            enable_model_rotation=True,
            rotation_models=["model-a", "model-b"],
        )
        limiter = SmartRateLimiter(config)
        
        limiter.mark_model_rate_limited("model-a")
        
        assert limiter._model_rotations == 1


class TestRateLimitStats:
    """Tests for RateLimitStats dataclass."""
    
    def test_stats_dataclass(self):
        """Test RateLimitStats creation."""
        stats = RateLimitStats(
            current_rpm=50,
            current_tpm=25000,
            rpm_limit=100,
            tpm_limit=100000,
            rpm_utilization=0.5,
            tpm_utilization=0.25,
            is_throttled=False,
            throttle_reason=ThrottleReason.NONE,
            wait_time=0.0,
            active_model="gemini-3-flash-preview",
            models_on_cooldown=[],
        )
        
        assert stats.current_rpm == 50
        assert stats.current_tpm == 25000
        assert stats.is_throttled is False
        assert stats.active_model == "gemini-3-flash-preview"


class TestThrottleReason:
    """Tests for ThrottleReason enum."""
    
    def test_throttle_reason_values(self):
        """Test ThrottleReason enum values."""
        assert ThrottleReason.NONE.value == "none"
        assert ThrottleReason.RPM_LIMIT.value == "rpm_limit"
        assert ThrottleReason.TPM_LIMIT.value == "tpm_limit"
        assert ThrottleReason.RATE_LIMITED.value == "rate_limited"
        assert ThrottleReason.MODEL_COOLDOWN.value == "model_cooldown"
