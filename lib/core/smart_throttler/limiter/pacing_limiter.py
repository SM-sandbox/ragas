"""
Pacing Limiter for Smart Throttler.

Provides smooth request pacing using token bucket algorithm.
Prevents second-level spikes and "21st request within a minute" patterns.
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PacingStats:
    """Statistics for pacing limiter."""
    current_tokens: float
    max_tokens: float
    refill_rate: float
    total_requests: int
    total_waited_ms: float
    total_delays: int


class TokenBucket:
    """
    Token bucket for rate limiting.
    
    Tokens are added at a constant rate up to a maximum.
    Each request consumes one token. If no tokens are available,
    the request must wait.
    """
    
    def __init__(
        self,
        rate: float,
        burst: float = 2.0,
    ):
        """
        Initialize the token bucket.
        
        Args:
            rate: Tokens per second to add
            burst: Maximum tokens (burst allowance multiplier)
        """
        self._rate = rate
        self._max_tokens = rate * burst
        self._tokens = self._max_tokens  # Start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        
        # Statistics
        self._total_requests = 0
        self._total_waited_ms = 0.0
        self._total_delays = 0
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now
    
    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without blocking.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False if insufficient tokens
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_requests += 1
                return True
            return False
    
    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens, blocking if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        start = time.monotonic()
        deadline = start + timeout if timeout else float('inf')
        
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._total_requests += 1
                    return True
                
                # Calculate wait time
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._rate
            
            # Check timeout
            now = time.monotonic()
            if now + wait_time > deadline:
                return False
            
            # Wait for tokens
            actual_wait = min(wait_time, deadline - now)
            if actual_wait > 0:
                with self._lock:
                    self._total_delays += 1
                    self._total_waited_ms += actual_wait * 1000
                time.sleep(actual_wait)
    
    async def acquire_async(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens asynchronously.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if acquired, False if timeout
        """
        start = time.monotonic()
        deadline = start + timeout if timeout else float('inf')
        
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._total_requests += 1
                    return True
                
                # Calculate wait time
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._rate
            
            # Check timeout
            now = time.monotonic()
            if now + wait_time > deadline:
                return False
            
            # Wait for tokens
            actual_wait = min(wait_time, deadline - now)
            if actual_wait > 0:
                with self._lock:
                    self._total_delays += 1
                    self._total_waited_ms += actual_wait * 1000
                await asyncio.sleep(actual_wait)
    
    def get_wait_time(self, tokens: float = 1.0) -> float:
        """
        Get the time to wait for tokens without acquiring.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Seconds to wait (0 if tokens available)
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            tokens_needed = tokens - self._tokens
            return tokens_needed / self._rate
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens
    
    def update_rate(self, new_rate: float) -> None:
        """
        Update the token refill rate.
        
        Args:
            new_rate: New tokens per second
        """
        with self._lock:
            self._refill()  # Refill with old rate first
            self._rate = new_rate
            self._max_tokens = new_rate * (self._max_tokens / self._rate) if self._rate > 0 else new_rate * 2
    
    def get_stats(self) -> PacingStats:
        """Get current statistics."""
        with self._lock:
            self._refill()
            return PacingStats(
                current_tokens=self._tokens,
                max_tokens=self._max_tokens,
                refill_rate=self._rate,
                total_requests=self._total_requests,
                total_waited_ms=self._total_waited_ms,
                total_delays=self._total_delays,
            )


class PacingLimiter:
    """
    Request pacing limiter using token bucket.
    
    Smooths request traffic to prevent bursts that could trigger
    rate limiting. Uses minute-level smoothing to avoid the
    "21st request within a minute" problem.
    """
    
    def __init__(
        self,
        admitted_rps: float = 1.0,
        burst_allowance: float = 2.0,
    ):
        """
        Initialize the pacing limiter.
        
        Args:
            admitted_rps: Admitted requests per second
            burst_allowance: Burst multiplier (e.g., 2.0 = allow 2x burst)
        """
        self._bucket = TokenBucket(rate=admitted_rps, burst=burst_allowance)
        self._admitted_rps = admitted_rps
        self._burst_allowance = burst_allowance
    
    def try_acquire(self) -> bool:
        """Try to acquire permission without blocking."""
        return self._bucket.try_acquire()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission, blocking if necessary."""
        return self._bucket.acquire(timeout=timeout)
    
    async def acquire_async(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission asynchronously."""
        return await self._bucket.acquire_async(timeout=timeout)
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request."""
        return self._bucket.get_wait_time()
    
    def update_rate(self, new_rps: float) -> None:
        """
        Update the admitted rate.
        
        Args:
            new_rps: New requests per second
        """
        self._admitted_rps = new_rps
        self._bucket.update_rate(new_rps)
    
    @property
    def admitted_rps(self) -> float:
        """Get current admitted RPS."""
        return self._admitted_rps
    
    def get_stats(self) -> PacingStats:
        """Get current statistics."""
        return self._bucket.get_stats()
