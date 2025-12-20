"""
Smart Rate Limiter for Gemini API.

This module provides intelligent rate limiting with:
- Sliding window tracking for RPM and TPM
- Pre-emptive throttling (90% threshold)
- Staggered worker starts
- Optional model rotation
- Real-time quota monitoring

Usage:
    from lib.core.smart_throttler import SmartRateLimiter, get_rate_limiter
    
    # Get singleton instance
    limiter = get_rate_limiter()
    
    # Acquire permission before making API call
    await limiter.acquire(estimated_tokens=1000)
    
    # Record actual usage after call
    limiter.record_usage(actual_tokens=1500)
"""

import asyncio
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque, Tuple
from threading import Lock, Semaphore
from enum import Enum

logger = logging.getLogger(__name__)


class ThrottleReason(str, Enum):
    """Reasons for throttling."""
    NONE = "none"
    RPM_LIMIT = "rpm_limit"
    TPM_LIMIT = "tpm_limit"
    RATE_LIMITED = "rate_limited"
    MODEL_COOLDOWN = "model_cooldown"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # Requests per minute limit
    rpm_limit: int = 1000
    # Tokens per minute limit
    tpm_limit: int = 1000000
    # Threshold percentage (0.0-1.0) - throttle before hitting limit
    threshold: float = 0.9
    # Sliding window size in seconds
    window_size: int = 60
    # Minimum delay between requests (seconds)
    min_request_delay: float = 0.05
    # Maximum stagger delay for worker starts (seconds)
    max_stagger_delay: float = 2.0
    # Enable model rotation when rate limited
    # NOTE: Disabled by default - smart throttler is good enough to wait for gemini-3-flash-preview
    enable_model_rotation: bool = False
    # Cooldown period for rate-limited models (seconds)
    model_cooldown_seconds: float = 30.0
    # Models available for rotation (in priority order)
    # NOTE: Model rotation is disabled by default. The smart throttler will wait
    # for gemini-3-flash-preview to become available rather than falling back.
    # Set enable_model_rotation=True to enable fallback behavior.
    rotation_models: List[str] = field(default_factory=lambda: [
        "gemini-3-flash-preview",  # Primary baseline model - always use this
        # "gemini-2.5-flash",      # Fallback 1 (disabled)
        # "gemini-2.0-flash",      # Fallback 2 (disabled)
    ])
    # Maximum concurrent requests (prevents burst overwhelming API)
    max_concurrent_requests: int = 20


@dataclass
class UsageRecord:
    """Record of a single API request."""
    timestamp: float
    tokens: int
    model: str


@dataclass
class RateLimitStats:
    """Current rate limit statistics."""
    current_rpm: int
    current_tpm: int
    rpm_limit: int
    tpm_limit: int
    rpm_utilization: float
    tpm_utilization: float
    is_throttled: bool
    throttle_reason: ThrottleReason
    wait_time: float
    active_model: str
    models_on_cooldown: List[str]


class SmartRateLimiter:
    """
    Intelligent rate limiter with sliding window tracking.
    
    Features:
    - Tracks RPM and TPM in real-time using sliding windows
    - Pre-emptively throttles at configurable threshold (default 90%)
    - Staggers worker starts to prevent burst patterns
    - Optional model rotation when rate limited
    - Thread-safe for concurrent access
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize the rate limiter."""
        self.config = config or RateLimitConfig()
        self._lock = Lock()
        self._async_lock = asyncio.Lock()
        
        # Semaphore to limit concurrent requests (prevents burst overwhelming API)
        self._semaphore = Semaphore(self.config.max_concurrent_requests)
        
        # Sliding window of usage records
        self._usage_records: Deque[UsageRecord] = deque()
        
        # Model cooldown tracking
        self._model_cooldowns: Dict[str, float] = {}
        
        # Current active model
        self._active_model: str = self.config.rotation_models[0] if self.config.rotation_models else "gemini-3-flash-preview"
        
        # Statistics
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._throttle_count: int = 0
        self._model_rotations: int = 0
        
        # Worker stagger tracking
        self._worker_start_times: Dict[int, float] = {}
        
    def _cleanup_old_records(self) -> None:
        """Remove records outside the sliding window."""
        cutoff = time.time() - self.config.window_size
        while self._usage_records and self._usage_records[0].timestamp < cutoff:
            self._usage_records.popleft()
    
    def _get_current_rpm(self) -> int:
        """Get current requests per minute."""
        self._cleanup_old_records()
        return len(self._usage_records)
    
    def _get_current_tpm(self) -> int:
        """Get current tokens per minute."""
        self._cleanup_old_records()
        return sum(r.tokens for r in self._usage_records)
    
    def _calculate_wait_time(self) -> float:
        """Calculate how long to wait before next request is safe."""
        if not self._usage_records:
            return 0.0
        
        # Find oldest record and calculate when it will expire
        oldest = self._usage_records[0]
        time_until_expiry = (oldest.timestamp + self.config.window_size) - time.time()
        
        # Add small buffer
        return max(0.0, time_until_expiry + 0.1)
    
    def _get_available_model(self) -> Optional[str]:
        """Get the next available model (not on cooldown)."""
        current_time = time.time()
        
        # Clean up expired cooldowns
        self._model_cooldowns = {
            model: expiry 
            for model, expiry in self._model_cooldowns.items() 
            if expiry > current_time
        }
        
        # Find first available model
        for model in self.config.rotation_models:
            if model not in self._model_cooldowns:
                return model
        
        # All models on cooldown - return the one that expires soonest
        if self._model_cooldowns:
            soonest = min(self._model_cooldowns.items(), key=lambda x: x[1])
            return soonest[0]
        
        return self._active_model
    
    def mark_model_rate_limited(self, model: str) -> Optional[str]:
        """
        Mark a model as rate limited and optionally rotate to another.
        
        Returns the new model to use, or None if rotation is disabled.
        """
        with self._lock:
            cooldown_until = time.time() + self.config.model_cooldown_seconds
            self._model_cooldowns[model] = cooldown_until
            
            logger.warning(f"Model {model} rate limited, cooldown until {cooldown_until}")
            
            if not self.config.enable_model_rotation:
                return None
            
            new_model = self._get_available_model()
            if new_model and new_model != model:
                self._active_model = new_model
                self._model_rotations += 1
                logger.info(f"Rotating to model: {new_model}")
                return new_model
            
            return None
    
    def get_stats(self) -> RateLimitStats:
        """Get current rate limit statistics."""
        with self._lock:
            current_rpm = self._get_current_rpm()
            current_tpm = self._get_current_tpm()
            
            rpm_util = current_rpm / self.config.rpm_limit if self.config.rpm_limit > 0 else 0
            tpm_util = current_tpm / self.config.tpm_limit if self.config.tpm_limit > 0 else 0
            
            is_throttled = rpm_util >= self.config.threshold or tpm_util >= self.config.threshold
            
            if rpm_util >= self.config.threshold:
                reason = ThrottleReason.RPM_LIMIT
            elif tpm_util >= self.config.threshold:
                reason = ThrottleReason.TPM_LIMIT
            else:
                reason = ThrottleReason.NONE
            
            wait_time = self._calculate_wait_time() if is_throttled else 0.0
            
            models_on_cooldown = list(self._model_cooldowns.keys())
            
            return RateLimitStats(
                current_rpm=current_rpm,
                current_tpm=current_tpm,
                rpm_limit=self.config.rpm_limit,
                tpm_limit=self.config.tpm_limit,
                rpm_utilization=rpm_util,
                tpm_utilization=tpm_util,
                is_throttled=is_throttled,
                throttle_reason=reason,
                wait_time=wait_time,
                active_model=self._active_model,
                models_on_cooldown=models_on_cooldown,
            )
    
    def get_stagger_delay(self, worker_id: int) -> float:
        """
        Get a stagger delay for a worker to prevent burst patterns.
        
        Each worker gets a consistent delay based on its ID.
        """
        with self._lock:
            if worker_id not in self._worker_start_times:
                # Assign a random stagger delay for this worker
                delay = random.uniform(0, self.config.max_stagger_delay)
                self._worker_start_times[worker_id] = delay
            return self._worker_start_times[worker_id]
    
    async def acquire(
        self, 
        estimated_tokens: int = 1000,
        worker_id: Optional[int] = None,
        timeout: float = 60.0
    ) -> Tuple[bool, str]:
        """
        Acquire permission to make an API request.
        
        This method will block until capacity is available or timeout is reached.
        
        Args:
            estimated_tokens: Estimated tokens for this request
            worker_id: Optional worker ID for stagger delay
            timeout: Maximum time to wait for capacity
            
        Returns:
            Tuple of (success, model_to_use)
        """
        start_time = time.time()
        
        # Apply stagger delay for worker if provided
        if worker_id is not None:
            stagger = self.get_stagger_delay(worker_id)
            if stagger > 0:
                await asyncio.sleep(stagger)
                # Clear stagger after first use
                with self._lock:
                    self._worker_start_times.pop(worker_id, None)
        
        async with self._async_lock:
            while True:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning(f"Rate limiter timeout after {elapsed:.1f}s")
                    return False, self._active_model
                
                with self._lock:
                    current_rpm = self._get_current_rpm()
                    current_tpm = self._get_current_tpm()
                    
                    rpm_headroom = self.config.rpm_limit * self.config.threshold - current_rpm
                    tpm_headroom = self.config.tpm_limit * self.config.threshold - current_tpm
                    
                    # Check if we have capacity
                    if rpm_headroom >= 1 and tpm_headroom >= estimated_tokens:
                        # Record the request
                        record = UsageRecord(
                            timestamp=time.time(),
                            tokens=estimated_tokens,
                            model=self._active_model,
                        )
                        self._usage_records.append(record)
                        self._total_requests += 1
                        self._total_tokens += estimated_tokens
                        
                        return True, self._active_model
                    
                    # Calculate wait time
                    wait_time = self._calculate_wait_time()
                    self._throttle_count += 1
                
                # Wait and retry
                actual_wait = min(wait_time, timeout - elapsed, 5.0)  # Cap at 5s per iteration
                if actual_wait > 0:
                    logger.debug(f"Rate limiter waiting {actual_wait:.2f}s (RPM: {current_rpm}, TPM: {current_tpm})")
                    await asyncio.sleep(actual_wait)
                else:
                    # Small delay to prevent tight loop
                    await asyncio.sleep(self.config.min_request_delay)
    
    def acquire_sync(self, estimated_tokens: int = 1000) -> Tuple[bool, str]:
        """
        Synchronous version of acquire for use in ThreadPoolExecutor.
        
        Uses a semaphore to limit concurrent requests and prevent burst patterns.
        This is the key method for preventing 100 workers from overwhelming the API.
        
        Args:
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            Tuple of (success, model_to_use)
        """
        # Acquire semaphore - blocks if too many concurrent requests
        self._semaphore.acquire()
        
        try:
            with self._lock:
                # Check current usage
                current_rpm = self._get_current_rpm()
                current_tpm = self._get_current_tpm()
                
                rpm_headroom = self.config.rpm_limit * self.config.threshold - current_rpm
                tpm_headroom = self.config.tpm_limit * self.config.threshold - current_tpm
                
                # If we're near limits, wait a bit
                if rpm_headroom < 10 or tpm_headroom < estimated_tokens:
                    wait_time = self._calculate_wait_time()
                    if wait_time > 0:
                        self._throttle_count += 1
                        # Release lock while waiting
                        self._lock.release()
                        try:
                            time.sleep(min(wait_time, 5.0))
                        finally:
                            self._lock.acquire()
                
                # Record the request
                record = UsageRecord(
                    timestamp=time.time(),
                    tokens=estimated_tokens,
                    model=self._active_model,
                )
                self._usage_records.append(record)
                self._total_requests += 1
                self._total_tokens += estimated_tokens
                
                return True, self._active_model
        except Exception as e:
            logger.error(f"Error in acquire_sync: {e}")
            return False, self._active_model
    
    def release_sync(self) -> None:
        """Release the semaphore after a request completes."""
        self._semaphore.release()
    
    def record_usage(self, actual_tokens: int, model: Optional[str] = None) -> None:
        """
        Record actual token usage after a request completes.
        
        Call this to update the token count if it differs from the estimate.
        """
        with self._lock:
            if self._usage_records:
                # Update the most recent record with actual tokens
                # This is a simplification - in practice you'd want to track by request ID
                latest = self._usage_records[-1]
                token_diff = actual_tokens - latest.tokens
                if token_diff != 0:
                    # Create a new record with the difference
                    self._usage_records[-1] = UsageRecord(
                        timestamp=latest.timestamp,
                        tokens=actual_tokens,
                        model=model or latest.model,
                    )
                    self._total_tokens += token_diff
    
    def reset(self) -> None:
        """Reset all rate limiting state."""
        with self._lock:
            self._usage_records.clear()
            self._model_cooldowns.clear()
            self._worker_start_times.clear()
            self._total_requests = 0
            self._total_tokens = 0
            self._throttle_count = 0
            self._model_rotations = 0
            if self.config.rotation_models:
                self._active_model = self.config.rotation_models[0]


# Singleton instance
_rate_limiter: Optional[SmartRateLimiter] = None
_rate_limiter_lock = Lock()


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> SmartRateLimiter:
    """
    Get the singleton rate limiter instance.
    
    Args:
        config: Optional configuration. Only used on first call.
        
    Returns:
        The singleton SmartRateLimiter instance.
    """
    global _rate_limiter
    
    with _rate_limiter_lock:
        if _rate_limiter is None:
            _rate_limiter = SmartRateLimiter(config)
        return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the singleton rate limiter (mainly for testing)."""
    global _rate_limiter
    
    with _rate_limiter_lock:
        if _rate_limiter is not None:
            _rate_limiter.reset()
        _rate_limiter = None


def configure_rate_limiter(
    rpm_limit: int = 1000,
    tpm_limit: int = 1000000,
    threshold: float = 0.9,
    enable_model_rotation: bool = False,
    rotation_models: Optional[List[str]] = None,
    max_concurrent_requests: int = 20,
) -> SmartRateLimiter:
    """
    Configure and return the rate limiter with custom settings.
    
    This resets any existing rate limiter.
    """
    reset_rate_limiter()
    
    config = RateLimitConfig(
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        threshold=threshold,
        enable_model_rotation=enable_model_rotation,
        rotation_models=rotation_models or [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-3-flash",
            "gemini-2.5-pro",
            "gemini-3-pro",
        ],
        max_concurrent_requests=max_concurrent_requests,
    )
    
    return get_rate_limiter(config)
