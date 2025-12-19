"""
Comprehensive tests for SmartRateLimiter.

Test categories:
1. Unit tests - Individual methods
2. Integration tests - Full acquire/release cycles
3. Concurrency tests - Thread safety
4. Edge case tests - Boundary conditions

Run with: pytest tests/unit/test_rate_limiter.py -v
"""

import pytest
import asyncio
import time
import threading
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.core.rate_limiter import (
    SmartRateLimiter,
    RateLimiterConfig,
    get_limiter_for_model,
    DEFAULT_LIMITER,
)


# =============================================================================
# UNIT TESTS - Basic Functionality
# =============================================================================

class TestRateLimiterInit:
    """Test rate limiter initialization."""
    
    def test_default_init(self):
        """Default init uses sensible defaults."""
        limiter = SmartRateLimiter()
        assert limiter.rpm_limit == 20000
        assert limiter.tpm_limit == 1000000
        assert limiter.config.headroom == 0.90
    
    def test_custom_limits(self):
        """Custom limits are respected."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=500000)
        assert limiter.rpm_limit == 1000
        assert limiter.tpm_limit == 500000
    
    def test_config_object(self):
        """Config object overrides individual params."""
        config = RateLimiterConfig(rpm_limit=5000, tpm_limit=250000, headroom=0.80)
        limiter = SmartRateLimiter(config=config)
        assert limiter.rpm_limit == 5000
        assert limiter.tpm_limit == 250000
        assert limiter.config.headroom == 0.80
    
    def test_effective_limits(self):
        """Effective limits apply headroom."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=100000, headroom=0.90)
        assert limiter.effective_rpm == 900
        assert limiter.effective_tpm == 90000


class TestRateLimiterTracking:
    """Test RPM and TPM tracking."""
    
    def test_initial_usage_zero(self):
        """Initial usage is zero."""
        limiter = SmartRateLimiter()
        assert limiter.get_current_rpm() == 0
        assert limiter.get_current_tpm() == 0
    
    def test_record_increases_counts(self):
        """Recording requests increases counts."""
        limiter = SmartRateLimiter()
        limiter._record_request(1000)
        assert limiter.get_current_rpm() == 1
        assert limiter.get_current_tpm() == 1000
    
    def test_multiple_records(self):
        """Multiple records accumulate."""
        limiter = SmartRateLimiter()
        limiter._record_request(500)
        limiter._record_request(750)
        limiter._record_request(250)
        assert limiter.get_current_rpm() == 3
        assert limiter.get_current_tpm() == 1500
    
    def test_get_usage_returns_dict(self):
        """get_usage returns comprehensive stats."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=100000)
        limiter._record_request(5000)
        
        usage = limiter.get_usage()
        assert "current_rpm" in usage
        assert "current_tpm" in usage
        assert "rpm_limit" in usage
        assert "tpm_limit" in usage
        assert "rpm_utilization" in usage
        assert "tpm_utilization" in usage
        assert usage["current_rpm"] == 1
        assert usage["current_tpm"] == 5000
        assert usage["rpm_utilization"] == 0.001  # 1/1000


class TestSlidingWindow:
    """Test sliding window expiration."""
    
    def test_old_entries_expire(self):
        """Entries older than window are removed."""
        config = RateLimiterConfig(window_seconds=1)  # 1 second window for testing
        limiter = SmartRateLimiter(config=config)
        
        limiter._record_request(1000)
        assert limiter.get_current_rpm() == 1
        
        # Wait for window to expire
        time.sleep(1.1)
        assert limiter.get_current_rpm() == 0
        assert limiter.get_current_tpm() == 0
    
    def test_partial_expiration(self):
        """Only old entries expire, recent ones remain."""
        config = RateLimiterConfig(window_seconds=1)
        limiter = SmartRateLimiter(config=config)
        
        limiter._record_request(1000)
        time.sleep(0.6)
        limiter._record_request(2000)
        
        # First should expire, second should remain
        time.sleep(0.5)
        assert limiter.get_current_rpm() == 1
        assert limiter.get_current_tpm() == 2000


class TestCanAcquire:
    """Test can_acquire check."""
    
    def test_can_acquire_when_empty(self):
        """Can acquire when no requests recorded."""
        limiter = SmartRateLimiter(rpm_limit=100, tpm_limit=10000)
        assert limiter.can_acquire(1000) is True
    
    def test_cannot_acquire_at_rpm_limit(self):
        """Cannot acquire when at RPM limit."""
        limiter = SmartRateLimiter(rpm_limit=10, tpm_limit=1000000, headroom=1.0)
        
        # Fill up to limit
        for _ in range(10):
            limiter._record_request(100)
        
        assert limiter.can_acquire(100) is False
    
    def test_cannot_acquire_at_tpm_limit(self):
        """Cannot acquire when at TPM limit."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=10000, headroom=1.0)
        
        limiter._record_request(10000)
        assert limiter.can_acquire(1000) is False
    
    def test_headroom_applied(self):
        """Headroom is applied to limits."""
        limiter = SmartRateLimiter(rpm_limit=100, tpm_limit=100000, headroom=0.50)
        
        # Fill to 50% (effective limit)
        for _ in range(50):
            limiter._record_request(1000)
        
        assert limiter.can_acquire(1000) is False


class TestWaitTimeCalculation:
    """Test wait time calculation."""
    
    def test_no_wait_when_under_limit(self):
        """No wait needed when under limits."""
        limiter = SmartRateLimiter(rpm_limit=100, tpm_limit=100000)
        wait = limiter._calculate_wait_time(10, 5000, 1000)
        assert wait == 0.0
    
    def test_wait_when_at_rpm_limit(self):
        """Wait calculated when at RPM limit."""
        config = RateLimiterConfig(rpm_limit=10, tpm_limit=1000000, headroom=1.0, window_seconds=60)
        limiter = SmartRateLimiter(config=config)
        
        # Fill to limit
        for _ in range(10):
            limiter._record_request(100)
        
        wait = limiter._calculate_wait_time(10, 1000, 100)
        assert wait > 0


# =============================================================================
# INTEGRATION TESTS - Acquire/Release Cycles
# =============================================================================

class TestAcquireSync:
    """Test synchronous acquire."""
    
    def test_acquire_sync_returns_wait_time(self):
        """acquire_sync returns time spent waiting."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=100000)
        wait_time = limiter.acquire_sync(1000)
        assert isinstance(wait_time, float)
        assert wait_time >= 0
    
    def test_acquire_sync_records_request(self):
        """acquire_sync records the request."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=100000)
        limiter.acquire_sync(5000)
        assert limiter.get_current_rpm() == 1
        assert limiter.get_current_tpm() == 5000
    
    def test_acquire_sync_multiple(self):
        """Multiple acquire_sync calls accumulate."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=100000)
        limiter.acquire_sync(1000)
        limiter.acquire_sync(2000)
        limiter.acquire_sync(3000)
        assert limiter.get_current_rpm() == 3
        assert limiter.get_current_tpm() == 6000


class TestAcquireAsync:
    """Test async acquire."""
    
    @pytest.mark.asyncio
    async def test_acquire_returns_wait_time(self):
        """acquire returns time spent waiting."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=100000)
        wait_time = await limiter.acquire(1000)
        assert isinstance(wait_time, float)
        assert wait_time >= 0
    
    @pytest.mark.asyncio
    async def test_acquire_records_request(self):
        """acquire records the request."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=100000)
        await limiter.acquire(5000)
        assert limiter.get_current_rpm() == 1
        assert limiter.get_current_tpm() == 5000
    
    @pytest.mark.asyncio
    async def test_acquire_multiple_concurrent(self):
        """Multiple concurrent acquires work correctly."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=100000)
        
        async def do_acquire():
            await limiter.acquire(1000)
        
        await asyncio.gather(*[do_acquire() for _ in range(10)])
        assert limiter.get_current_rpm() == 10
        assert limiter.get_current_tpm() == 10000


class TestThrottling:
    """Test throttling behavior."""
    
    def test_throttles_at_limit(self):
        """Requests are throttled when at limit."""
        config = RateLimiterConfig(
            rpm_limit=5,
            tpm_limit=1000000,
            headroom=1.0,
            window_seconds=2,
            stagger_ms=10,
        )
        limiter = SmartRateLimiter(config=config)
        
        # Fill to limit
        for _ in range(5):
            limiter.acquire_sync(100)
        
        # Next acquire should wait
        start = time.time()
        limiter.acquire_sync(100)
        elapsed = time.time() - start
        
        # Should have waited some time (at least stagger)
        assert elapsed >= 0.01


# =============================================================================
# CONCURRENCY TESTS - Thread Safety
# =============================================================================

class TestThreadSafety:
    """Test thread safety of rate limiter."""
    
    def test_concurrent_acquires_thread_safe(self):
        """Concurrent acquires from multiple threads are safe."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=1000000)
        num_threads = 50
        
        def do_acquire():
            limiter.acquire_sync(1000)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(do_acquire) for _ in range(num_threads)]
            for f in futures:
                f.result()
        
        assert limiter.get_current_rpm() == num_threads
        assert limiter.get_current_tpm() == num_threads * 1000
    
    def test_concurrent_reads_safe(self):
        """Concurrent reads don't cause issues."""
        limiter = SmartRateLimiter(rpm_limit=1000, tpm_limit=1000000)
        
        # Add some data
        for _ in range(10):
            limiter._record_request(1000)
        
        def do_read():
            for _ in range(100):
                limiter.get_current_rpm()
                limiter.get_current_tpm()
                limiter.get_usage()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(do_read) for _ in range(10)]
            for f in futures:
                f.result()
        
        # Should complete without errors
        assert limiter.get_current_rpm() == 10


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_tokens(self):
        """Zero tokens is valid."""
        limiter = SmartRateLimiter()
        limiter.acquire_sync(0)
        assert limiter.get_current_rpm() == 1
        assert limiter.get_current_tpm() == 0
    
    def test_large_token_count(self):
        """Large token counts work."""
        limiter = SmartRateLimiter(tpm_limit=10000000)
        limiter.acquire_sync(5000000)
        assert limiter.get_current_tpm() == 5000000
    
    def test_reset_clears_all(self):
        """reset() clears all tracking data."""
        limiter = SmartRateLimiter()
        limiter.acquire_sync(1000)
        limiter.acquire_sync(2000)
        
        limiter.reset()
        
        assert limiter.get_current_rpm() == 0
        assert limiter.get_current_tpm() == 0
        usage = limiter.get_usage()
        assert usage["total_requests"] == 0
    
    def test_very_small_window(self):
        """Very small window works correctly."""
        config = RateLimiterConfig(window_seconds=1, rpm_limit=100, tpm_limit=100000)
        limiter = SmartRateLimiter(config=config)
        
        limiter.acquire_sync(1000)
        assert limiter.get_current_rpm() == 1
        
        time.sleep(1.1)
        assert limiter.get_current_rpm() == 0


# =============================================================================
# MODEL-SPECIFIC LIMITER TESTS
# =============================================================================

class TestModelSpecificLimiter:
    """Test get_limiter_for_model function."""
    
    def test_flash_model_limits(self):
        """Flash models get 20K RPM."""
        limiter = get_limiter_for_model("gemini-3-flash-preview")
        assert limiter.rpm_limit == 20000
        assert limiter.tpm_limit == 1000000
    
    def test_pro_model_limits(self):
        """Pro models get 2K RPM."""
        limiter = get_limiter_for_model("gemini-3-pro-preview")
        assert limiter.rpm_limit == 2000
        assert limiter.tpm_limit == 1000000
    
    def test_25_flash_limits(self):
        """2.5 flash gets 20K RPM."""
        limiter = get_limiter_for_model("gemini-2.5-flash")
        assert limiter.rpm_limit == 20000
    
    def test_25_pro_limits(self):
        """2.5 pro gets 2K RPM, 2M TPM."""
        limiter = get_limiter_for_model("gemini-2.5-pro")
        assert limiter.rpm_limit == 2000
        assert limiter.tpm_limit == 2000000
    
    def test_unknown_model_defaults(self):
        """Unknown model gets flash defaults."""
        limiter = get_limiter_for_model("unknown-model-xyz")
        assert limiter.rpm_limit == 20000
        assert limiter.tpm_limit == 1000000
    
    def test_default_limiter_exists(self):
        """DEFAULT_LIMITER is configured correctly."""
        assert DEFAULT_LIMITER.rpm_limit == 20000
        assert DEFAULT_LIMITER.tpm_limit == 1000000


# =============================================================================
# STATS TRACKING TESTS
# =============================================================================

class TestStatsTracking:
    """Test statistics tracking."""
    
    def test_total_requests_tracked(self):
        """Total requests are tracked."""
        limiter = SmartRateLimiter()
        limiter.acquire_sync(1000)
        limiter.acquire_sync(2000)
        limiter.acquire_sync(3000)
        
        usage = limiter.get_usage()
        assert usage["total_requests"] == 3
    
    def test_total_tokens_tracked(self):
        """Total tokens are tracked."""
        limiter = SmartRateLimiter()
        limiter.acquire_sync(1000)
        limiter.acquire_sync(2000)
        
        usage = limiter.get_usage()
        assert usage["total_tokens"] == 3000


# =============================================================================
# STAGGER TESTS
# =============================================================================

class TestStaggering:
    """Test request staggering."""
    
    def test_stagger_prevents_burst(self):
        """Stagger prevents all requests firing at once."""
        config = RateLimiterConfig(
            rpm_limit=1000,
            tpm_limit=1000000,
            stagger_ms=50,
        )
        limiter = SmartRateLimiter(config=config)
        
        start = time.time()
        for _ in range(5):
            limiter.acquire_sync(100)
        elapsed = time.time() - start
        
        # Should take at least 4 * 50ms = 200ms due to staggering
        assert elapsed >= 0.15  # Allow some tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
