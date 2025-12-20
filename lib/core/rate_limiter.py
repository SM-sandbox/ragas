"""
SmartRateLimiter - Real-time RPM and TPM tracking with pre-emptive throttling.

Features:
- Tracks requests per minute (RPM) and tokens per minute (TPM) in real-time
- Uses sliding window for accurate rate tracking
- Pre-emptively throttles at 90% capacity (configurable headroom)
- Async acquire() method that waits for capacity
- Thread-safe for concurrent access
- Staggered request release to avoid bursts

Usage:
    limiter = SmartRateLimiter(rpm_limit=20000, tpm_limit=1000000)
    
    async def make_request():
        await limiter.acquire(estimated_tokens=5000)
        # ... make API call ...
        limiter.record_completion(actual_tokens=4500)
"""

import asyncio
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, Tuple


@dataclass
class RateLimiterConfig:
    """Configuration for SmartRateLimiter."""
    rpm_limit: int = 20000  # Requests per minute limit
    tpm_limit: int = 1000000  # Tokens per minute limit
    headroom: float = 0.90  # Throttle at 90% of limit
    window_seconds: int = 60  # Sliding window size
    min_wait_ms: int = 10  # Minimum wait between checks
    max_wait_ms: int = 5000  # Maximum wait before retry
    stagger_ms: int = 5  # Stagger between concurrent releases (was 50, reduced for high RPM limits)
    max_concurrent: int = 30  # Maximum concurrent API calls (semaphore)


class SmartRateLimiter:
    """
    Real-time rate limiter that tracks RPM and TPM with pre-emptive throttling.
    
    - Tracks RPM and TPM in real-time using sliding windows
    - Staggers worker requests (not all at once)
    - Pre-emptively throttles before hitting limits
    - Thread-safe for concurrent access
    """
    
    def __init__(
        self,
        rpm_limit: int = 20000,
        tpm_limit: int = 1000000,
        headroom: float = 0.90,
        max_concurrent: int = 30,
        config: Optional[RateLimiterConfig] = None,
    ):
        """
        Initialize the rate limiter.
        
        Args:
            rpm_limit: Maximum requests per minute
            tpm_limit: Maximum tokens per minute
            headroom: Fraction of limit to use (0.90 = throttle at 90%)
            max_concurrent: Maximum concurrent API calls (semaphore)
            config: Optional full config object
        """
        if config:
            self.config = config
        else:
            self.config = RateLimiterConfig(
                rpm_limit=rpm_limit,
                tpm_limit=tpm_limit,
                headroom=headroom,
                max_concurrent=max_concurrent,
            )
        
        # Sliding windows: (timestamp, value)
        self._request_times: Deque[float] = deque()
        self._token_records: Deque[Tuple[float, int]] = deque()
        
        # Thread safety
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
        # Semaphore to cap concurrent API calls
        self._semaphore = threading.Semaphore(self.config.max_concurrent)
        self._async_semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._in_flight = 0  # Track current in-flight requests
        
        # Stats
        self._total_requests = 0
        self._total_tokens = 0
        self._total_waits = 0
        self._total_wait_time = 0.0
        
        # Stagger control
        self._last_acquire_time = 0.0
    
    @property
    def rpm_limit(self) -> int:
        return self.config.rpm_limit
    
    @property
    def tpm_limit(self) -> int:
        return self.config.tpm_limit
    
    @property
    def effective_rpm(self) -> int:
        """Effective RPM limit after headroom."""
        return int(self.config.rpm_limit * self.config.headroom)
    
    @property
    def effective_tpm(self) -> int:
        """Effective TPM limit after headroom."""
        return int(self.config.tpm_limit * self.config.headroom)
    
    def _cleanup_old_entries(self, now: float) -> None:
        """Remove entries older than the sliding window."""
        cutoff = now - self.config.window_seconds
        
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        
        while self._token_records and self._token_records[0][0] < cutoff:
            self._token_records.popleft()
    
    def get_current_rpm(self) -> int:
        """Get current requests per minute in the sliding window."""
        with self._lock:
            now = time.time()
            self._cleanup_old_entries(now)
            return len(self._request_times)
    
    def get_current_tpm(self) -> int:
        """Get current tokens per minute in the sliding window."""
        with self._lock:
            now = time.time()
            self._cleanup_old_entries(now)
            return sum(tokens for _, tokens in self._token_records)
    
    def get_usage(self) -> dict:
        """Get current usage stats."""
        with self._lock:
            now = time.time()
            self._cleanup_old_entries(now)
            current_rpm = len(self._request_times)
            current_tpm = sum(tokens for _, tokens in self._token_records)
            
            return {
                "current_rpm": current_rpm,
                "current_tpm": current_tpm,
                "rpm_limit": self.config.rpm_limit,
                "tpm_limit": self.config.tpm_limit,
                "max_concurrent": self.config.max_concurrent,
                "in_flight": self._in_flight,
                "rpm_utilization": current_rpm / self.config.rpm_limit if self.config.rpm_limit > 0 else 0,
                "tpm_utilization": current_tpm / self.config.tpm_limit if self.config.tpm_limit > 0 else 0,
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "total_waits": self._total_waits,
                "avg_wait_time": self._total_wait_time / self._total_waits if self._total_waits > 0 else 0,
            }
    
    def _calculate_wait_time(self, current_rpm: int, current_tpm: int, estimated_tokens: int) -> float:
        """Calculate how long to wait before acquiring capacity."""
        # Check if we're under limits
        if current_rpm < self.effective_rpm and current_tpm + estimated_tokens < self.effective_tpm:
            return 0.0
        
        # Calculate wait based on oldest entry expiring
        now = time.time()
        wait_time = 0.0
        
        if current_rpm >= self.effective_rpm and self._request_times:
            # Wait for oldest request to expire
            oldest_request = self._request_times[0]
            wait_time = max(wait_time, oldest_request + self.config.window_seconds - now)
        
        if current_tpm + estimated_tokens >= self.effective_tpm and self._token_records:
            # Wait for enough tokens to expire
            tokens_needed = (current_tpm + estimated_tokens) - self.effective_tpm
            tokens_freed = 0
            for ts, tokens in self._token_records:
                tokens_freed += tokens
                if tokens_freed >= tokens_needed:
                    wait_time = max(wait_time, ts + self.config.window_seconds - now)
                    break
        
        # Clamp wait time
        min_wait = self.config.min_wait_ms / 1000.0
        max_wait = self.config.max_wait_ms / 1000.0
        return max(min_wait, min(wait_time, max_wait))
    
    def _record_request(self, estimated_tokens: int) -> None:
        """Record a request in the sliding window."""
        now = time.time()
        self._request_times.append(now)
        self._token_records.append((now, estimated_tokens))
        self._total_requests += 1
        self._total_tokens += estimated_tokens
    
    def can_acquire(self, estimated_tokens: int = 1000) -> bool:
        """Check if we can acquire capacity without waiting."""
        with self._lock:
            now = time.time()
            self._cleanup_old_entries(now)
            current_rpm = len(self._request_times)
            current_tpm = sum(tokens for _, tokens in self._token_records)
            return current_rpm < self.effective_rpm and current_tpm + estimated_tokens < self.effective_tpm
    
    async def acquire(self, estimated_tokens: int = 1000) -> float:
        """
        Wait until we have capacity, then record the request.
        
        Args:
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            Time spent waiting (seconds)
        """
        total_wait = 0.0
        
        async with self._async_lock:
            while True:
                with self._lock:
                    now = time.time()
                    self._cleanup_old_entries(now)
                    current_rpm = len(self._request_times)
                    current_tpm = sum(tokens for _, tokens in self._token_records)
                    
                    # Check if we have capacity
                    if current_rpm < self.effective_rpm and current_tpm + estimated_tokens < self.effective_tpm:
                        # Stagger releases to avoid bursts
                        stagger_wait = max(0, self._last_acquire_time + (self.config.stagger_ms / 1000.0) - now)
                        if stagger_wait > 0:
                            # Release lock during stagger wait
                            pass
                        else:
                            self._record_request(estimated_tokens)
                            self._last_acquire_time = time.time()
                            if total_wait > 0:
                                self._total_waits += 1
                                self._total_wait_time += total_wait
                            return total_wait
                    
                    wait_time = self._calculate_wait_time(current_rpm, current_tpm, estimated_tokens)
                
                # Wait outside the lock
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    total_wait += wait_time
                else:
                    # Small stagger wait
                    stagger = self.config.stagger_ms / 1000.0
                    await asyncio.sleep(stagger)
                    total_wait += stagger
    
    def acquire_sync(self, estimated_tokens: int = 1000) -> float:
        """
        Synchronous version of acquire for non-async code.
        
        Acquires the semaphore first (blocking if max_concurrent reached),
        then checks RPM/TPM limits.
        
        Args:
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            Time spent waiting (seconds)
        """
        total_wait = 0.0
        wait_start = time.time()
        
        # First, acquire semaphore (blocks if too many concurrent requests)
        self._semaphore.acquire()
        with self._lock:
            self._in_flight += 1
        
        try:
            while True:
                with self._lock:
                    now = time.time()
                    self._cleanup_old_entries(now)
                    current_rpm = len(self._request_times)
                    current_tpm = sum(tokens for _, tokens in self._token_records)
                    
                    # Check if we have capacity
                    if current_rpm < self.effective_rpm and current_tpm + estimated_tokens < self.effective_tpm:
                        # Stagger releases - wait only the remaining stagger time
                        stagger_wait = max(0, self._last_acquire_time + (self.config.stagger_ms / 1000.0) - now)
                        if stagger_wait > 0:
                            # Release lock, wait for stagger, then re-acquire
                            pass  # Will wait below
                        else:
                            self._record_request(estimated_tokens)
                            self._last_acquire_time = time.time()
                            total_wait = time.time() - wait_start
                            if total_wait > 0.01:  # Only count significant waits
                                self._total_waits += 1
                                self._total_wait_time += total_wait
                            return total_wait
                        wait_time = stagger_wait
                    else:
                        wait_time = self._calculate_wait_time(current_rpm, current_tpm, estimated_tokens)
                
                # Wait outside the lock - only wait the calculated time, not a fixed stagger
                if wait_time > 0:
                    time.sleep(wait_time)
                else:
                    # Minimal sleep to prevent busy-wait, but very short
                    time.sleep(0.001)  # 1ms
        except Exception:
            # On error, release semaphore and re-raise
            self.release()
            raise
    
    def release(self) -> None:
        """
        Release the semaphore after an API call completes.
        
        Must be called after acquire_sync() when the API call is done.
        """
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
        self._semaphore.release()
    
    def get_in_flight(self) -> int:
        """Get the current number of in-flight requests."""
        with self._lock:
            return self._in_flight
    
    def record_completion(self, actual_tokens: int) -> None:
        """
        Update the token count with actual tokens used (optional refinement).
        
        This can be called after a request completes to adjust the token
        count if the actual tokens differ from the estimate.
        """
        # For simplicity, we don't retroactively adjust - the estimate is used
        # This method is here for future enhancement if needed
        pass
    
    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._request_times.clear()
            self._token_records.clear()
            self._total_requests = 0
            self._total_tokens = 0
            self._total_waits = 0
            self._total_wait_time = 0.0
            self._last_acquire_time = 0.0


# Default limiter instance for gemini-3-flash (20K RPM, 1M TPM)
DEFAULT_LIMITER = SmartRateLimiter(
    rpm_limit=20000,
    tpm_limit=1000000,
    headroom=0.90,
)


def get_limiter_for_model(model: str) -> SmartRateLimiter:
    """
    Get a rate limiter configured for a specific model.
    
    Args:
        model: Model name (e.g., 'gemini-3-flash-preview')
        
    Returns:
        Configured SmartRateLimiter instance
    """
    # Model-specific limits based on bfai-prod quotas
    MODEL_LIMITS = {
        "gemini-3-flash": {"rpm": 20000, "tpm": 1000000},
        "gemini-3-flash-preview": {"rpm": 20000, "tpm": 1000000},
        "gemini-2.5-flash": {"rpm": 20000, "tpm": 1000000},
        "gemini-3-pro": {"rpm": 2000, "tpm": 1000000},
        "gemini-3-pro-preview": {"rpm": 2000, "tpm": 1000000},
        "gemini-2.5-pro": {"rpm": 2000, "tpm": 2000000},
    }
    
    # Find matching model config
    for model_key, limits in MODEL_LIMITS.items():
        if model_key in model.lower():
            return SmartRateLimiter(
                rpm_limit=limits["rpm"],
                tpm_limit=limits["tpm"],
                headroom=0.90,
            )
    
    # Default to flash limits
    return SmartRateLimiter(rpm_limit=20000, tpm_limit=1000000, headroom=0.90)
