"""
Retry Policy for Smart Throttler.

Central retry logic with jittered backoff.
Retries are the "emergency brake" - they inform the adaptive controller
to reduce admitted rate and prevent retry storms.
"""

import asyncio
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, TypeVar, Any


class RetryOutcome(str, Enum):
    """Outcome of a retry attempt."""
    SUCCESS = "success"
    RETRY = "retry"
    EXHAUSTED = "exhausted"
    NON_RETRYABLE = "non_retryable"


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_attempts: int = 6
    base_delay_ms: int = 250
    max_delay_ms: int = 30000
    jitter: bool = True
    jitter_factor: float = 0.5  # +/- 50% jitter
    
    # Retry storm prevention
    max_concurrent_retries: int = 5
    retry_spacing_ms: int = 100  # Minimum time between retries


@dataclass
class RetryStats:
    """Statistics for retry policy."""
    total_attempts: int
    total_retries: int
    total_successes: int
    total_exhausted: int
    total_non_retryable: int
    current_concurrent_retries: int
    avg_retry_delay_ms: float


T = TypeVar('T')


class RetryPolicy:
    """
    Central retry policy with jittered exponential backoff.
    
    Key features:
    - Jittered exponential backoff with caps
    - Retry storm prevention (limits concurrent retries)
    - Informs adaptive controller on retries
    - Distinguishes 429 from 5xx handling
    """
    
    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        on_retry_callback: Optional[Callable[[int, float, str], None]] = None,
    ):
        """
        Initialize the retry policy.
        
        Args:
            config: Retry configuration
            on_retry_callback: Called on each retry with (attempt, delay_ms, error_type)
        """
        self._config = config or RetryConfig()
        self._on_retry = on_retry_callback
        self._lock = threading.Lock()
        
        # Retry storm prevention
        self._concurrent_retries = 0
        self._retry_semaphore = threading.Semaphore(self._config.max_concurrent_retries)
        self._last_retry_time = 0.0
        
        # Statistics
        self._total_attempts = 0
        self._total_retries = 0
        self._total_successes = 0
        self._total_exhausted = 0
        self._total_non_retryable = 0
        self._total_delay_ms = 0.0
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for an attempt with jitter.
        
        Args:
            attempt: Current attempt number (1-indexed)
            
        Returns:
            Delay in milliseconds
        """
        # Exponential backoff: base * 2^(attempt-1)
        delay = self._config.base_delay_ms * (2 ** (attempt - 1))
        
        # Cap at max delay
        delay = min(delay, self._config.max_delay_ms)
        
        # Apply jitter
        if self._config.jitter:
            jitter_range = delay * self._config.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(self._config.base_delay_ms, delay)  # Don't go below base
        
        return delay
    
    def _should_retry(self, error: Exception) -> tuple[bool, str]:
        """
        Determine if an error is retryable.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Tuple of (should_retry, error_type)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # 429 Resource Exhausted - always retry
        if "429" in error_str or "resource_exhausted" in error_str or "resourceexhausted" in error_str:
            return True, "429"
        
        # 5xx Server Errors - retry
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True, "5xx"
        
        # Connection errors - retry
        if any(term in error_str for term in ["connection", "timeout", "network", "unavailable"]):
            return True, "connection"
        
        # Rate limit errors from various APIs
        if "rate" in error_str and "limit" in error_str:
            return True, "rate_limit"
        
        # Non-retryable errors
        if any(term in error_str for term in ["invalid", "bad request", "400", "401", "403", "404"]):
            return False, "client_error"
        
        # Default: don't retry unknown errors
        return False, error_type
    
    def _wait_for_retry_slot(self) -> bool:
        """
        Wait for a retry slot (prevents retry storms).
        
        Returns:
            True if slot acquired, False if should give up
        """
        # Acquire semaphore
        acquired = self._retry_semaphore.acquire(timeout=30.0)
        if not acquired:
            return False
        
        with self._lock:
            self._concurrent_retries += 1
            
            # Ensure minimum spacing between retries
            now = time.time() * 1000
            time_since_last = now - self._last_retry_time
            if time_since_last < self._config.retry_spacing_ms:
                wait_ms = self._config.retry_spacing_ms - time_since_last
                time.sleep(wait_ms / 1000)
            
            self._last_retry_time = time.time() * 1000
        
        return True
    
    def _release_retry_slot(self) -> None:
        """Release a retry slot."""
        with self._lock:
            self._concurrent_retries = max(0, self._concurrent_retries - 1)
        self._retry_semaphore.release()
    
    def execute_with_retry(
        self,
        func: Callable[[], T],
        on_error: Optional[Callable[[Exception, int], None]] = None,
    ) -> tuple[Optional[T], RetryOutcome, int]:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            on_error: Optional callback on each error
            
        Returns:
            Tuple of (result, outcome, attempts)
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self._config.max_attempts + 1):
            with self._lock:
                self._total_attempts += 1
            
            try:
                result = func()
                with self._lock:
                    self._total_successes += 1
                return result, RetryOutcome.SUCCESS, attempt
                
            except Exception as e:
                last_error = e
                should_retry, error_type = self._should_retry(e)
                
                if on_error:
                    on_error(e, attempt)
                
                if not should_retry:
                    with self._lock:
                        self._total_non_retryable += 1
                    return None, RetryOutcome.NON_RETRYABLE, attempt
                
                if attempt >= self._config.max_attempts:
                    with self._lock:
                        self._total_exhausted += 1
                    return None, RetryOutcome.EXHAUSTED, attempt
                
                # Wait for retry slot
                if not self._wait_for_retry_slot():
                    with self._lock:
                        self._total_exhausted += 1
                    return None, RetryOutcome.EXHAUSTED, attempt
                
                try:
                    # Calculate and apply delay
                    delay_ms = self._calculate_delay(attempt)
                    
                    with self._lock:
                        self._total_retries += 1
                        self._total_delay_ms += delay_ms
                    
                    # Notify callback
                    if self._on_retry:
                        self._on_retry(attempt, delay_ms, error_type)
                    
                    time.sleep(delay_ms / 1000)
                finally:
                    self._release_retry_slot()
        
        with self._lock:
            self._total_exhausted += 1
        return None, RetryOutcome.EXHAUSTED, self._config.max_attempts
    
    async def execute_with_retry_async(
        self,
        func: Callable[[], Any],
        on_error: Optional[Callable[[Exception, int], None]] = None,
    ) -> tuple[Any, RetryOutcome, int]:
        """
        Execute an async function with retry logic.
        
        Args:
            func: Async function to execute
            on_error: Optional callback on each error
            
        Returns:
            Tuple of (result, outcome, attempts)
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self._config.max_attempts + 1):
            with self._lock:
                self._total_attempts += 1
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()
                with self._lock:
                    self._total_successes += 1
                return result, RetryOutcome.SUCCESS, attempt
                
            except Exception as e:
                last_error = e
                should_retry, error_type = self._should_retry(e)
                
                if on_error:
                    on_error(e, attempt)
                
                if not should_retry:
                    with self._lock:
                        self._total_non_retryable += 1
                    return None, RetryOutcome.NON_RETRYABLE, attempt
                
                if attempt >= self._config.max_attempts:
                    with self._lock:
                        self._total_exhausted += 1
                    return None, RetryOutcome.EXHAUSTED, attempt
                
                # Calculate and apply delay
                delay_ms = self._calculate_delay(attempt)
                
                with self._lock:
                    self._total_retries += 1
                    self._total_delay_ms += delay_ms
                
                # Notify callback
                if self._on_retry:
                    self._on_retry(attempt, delay_ms, error_type)
                
                await asyncio.sleep(delay_ms / 1000)
        
        with self._lock:
            self._total_exhausted += 1
        return None, RetryOutcome.EXHAUSTED, self._config.max_attempts
    
    def get_stats(self) -> RetryStats:
        """Get current retry statistics."""
        with self._lock:
            avg_delay = 0.0
            if self._total_retries > 0:
                avg_delay = self._total_delay_ms / self._total_retries
            
            return RetryStats(
                total_attempts=self._total_attempts,
                total_retries=self._total_retries,
                total_successes=self._total_successes,
                total_exhausted=self._total_exhausted,
                total_non_retryable=self._total_non_retryable,
                current_concurrent_retries=self._concurrent_retries,
                avg_retry_delay_ms=avg_delay,
            )
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._total_attempts = 0
            self._total_retries = 0
            self._total_successes = 0
            self._total_exhausted = 0
            self._total_non_retryable = 0
            self._total_delay_ms = 0.0
