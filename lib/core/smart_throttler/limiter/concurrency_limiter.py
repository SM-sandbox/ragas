"""
Concurrency Limiter for Smart Throttler.

Provides semaphore-based max in-flight request limiting.
Thread-safe and supports both sync and async usage.
"""

import asyncio
import threading
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager, asynccontextmanager


@dataclass
class ConcurrencyStats:
    """Statistics for concurrency limiter."""
    max_concurrency: int
    current_in_flight: int
    total_acquired: int
    total_rejected: int
    total_timeouts: int


class ConcurrencyLimiter:
    """
    Semaphore-based concurrency limiter.
    
    Limits the maximum number of in-flight requests to prevent
    overwhelming the API with too many concurrent requests.
    
    Thread-safe for use with ThreadPoolExecutor.
    """
    
    def __init__(self, max_concurrency: int = 16):
        """
        Initialize the concurrency limiter.
        
        Args:
            max_concurrency: Maximum concurrent requests allowed
        """
        self._max_concurrency = max_concurrency
        self._semaphore = threading.Semaphore(max_concurrency)
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._current_in_flight = 0
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_timeouts = 0
    
    def _get_async_semaphore(self) -> asyncio.Semaphore:
        """Get or create the async semaphore."""
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self._max_concurrency)
        return self._async_semaphore
    
    @property
    def max_concurrency(self) -> int:
        """Get the maximum concurrency limit."""
        return self._max_concurrency
    
    @property
    def current_in_flight(self) -> int:
        """Get the current number of in-flight requests."""
        with self._lock:
            return self._current_in_flight
    
    @property
    def available_slots(self) -> int:
        """Get the number of available slots."""
        with self._lock:
            return self._max_concurrency - self._current_in_flight
    
    def try_acquire(self) -> bool:
        """
        Try to acquire a slot without blocking.
        
        Returns:
            True if acquired, False if no slots available
        """
        acquired = self._semaphore.acquire(blocking=False)
        if acquired:
            with self._lock:
                self._current_in_flight += 1
                self._total_acquired += 1
        else:
            with self._lock:
                self._total_rejected += 1
        return acquired
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            True if acquired, False if timeout
        """
        acquired = self._semaphore.acquire(blocking=True, timeout=timeout)
        if acquired:
            with self._lock:
                self._current_in_flight += 1
                self._total_acquired += 1
        else:
            with self._lock:
                self._total_timeouts += 1
        return acquired
    
    def release(self) -> None:
        """Release a slot."""
        with self._lock:
            if self._current_in_flight > 0:
                self._current_in_flight -= 1
        self._semaphore.release()
    
    async def acquire_async(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot asynchronously.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)
            
        Returns:
            True if acquired, False if timeout
        """
        sem = self._get_async_semaphore()
        
        if timeout is None:
            await sem.acquire()
            with self._lock:
                self._current_in_flight += 1
                self._total_acquired += 1
            return True
        
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout)
            with self._lock:
                self._current_in_flight += 1
                self._total_acquired += 1
            return True
        except asyncio.TimeoutError:
            with self._lock:
                self._total_timeouts += 1
            return False
    
    def release_async(self) -> None:
        """Release a slot (async version)."""
        with self._lock:
            if self._current_in_flight > 0:
                self._current_in_flight -= 1
        sem = self._get_async_semaphore()
        sem.release()
    
    @contextmanager
    def slot(self, timeout: Optional[float] = None):
        """
        Context manager for acquiring a slot.
        
        Args:
            timeout: Maximum time to wait
            
        Yields:
            True if acquired
            
        Raises:
            TimeoutError: If timeout exceeded
        """
        acquired = self.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Failed to acquire concurrency slot within {timeout}s")
        try:
            yield True
        finally:
            self.release()
    
    @asynccontextmanager
    async def slot_async(self, timeout: Optional[float] = None):
        """
        Async context manager for acquiring a slot.
        
        Args:
            timeout: Maximum time to wait
            
        Yields:
            True if acquired
            
        Raises:
            TimeoutError: If timeout exceeded
        """
        acquired = await self.acquire_async(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Failed to acquire concurrency slot within {timeout}s")
        try:
            yield True
        finally:
            self.release_async()
    
    def get_stats(self) -> ConcurrencyStats:
        """Get current statistics."""
        with self._lock:
            return ConcurrencyStats(
                max_concurrency=self._max_concurrency,
                current_in_flight=self._current_in_flight,
                total_acquired=self._total_acquired,
                total_rejected=self._total_rejected,
                total_timeouts=self._total_timeouts,
            )
    
    def update_max_concurrency(self, new_max: int) -> None:
        """
        Update the maximum concurrency limit.
        
        Note: This creates a new semaphore. Existing waiters may be affected.
        
        Args:
            new_max: New maximum concurrency
        """
        with self._lock:
            self._max_concurrency = new_max
            self._semaphore = threading.Semaphore(new_max)
            self._async_semaphore = asyncio.Semaphore(new_max)
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._total_acquired = 0
            self._total_rejected = 0
            self._total_timeouts = 0
