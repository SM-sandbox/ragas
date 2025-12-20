"""
Token Budget Limiter for Smart Throttler.

Limits tokens per minute using estimated tokens at admission time.
Reconciles against observed tokens when available.
Prevents large-token steps from silently consuming the budget.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional, Tuple


class TokenBudgetStatus(str, Enum):
    """Status of token budget."""
    OK = "ok"
    SOFT_LIMIT = "soft_limit"
    HARD_LIMIT = "hard_limit"


@dataclass
class TokenRecord:
    """Record of token usage."""
    timestamp: float
    estimated_tokens: int
    actual_tokens: Optional[int] = None
    step_name: str = ""
    request_id: str = ""


@dataclass
class TokenBudgetStats:
    """Statistics for token budget limiter."""
    current_tokens_in_window: int
    soft_limit: int
    hard_limit: int
    utilization: float
    status: TokenBudgetStatus
    requests_in_window: int
    tokens_by_step: dict


class TokenBudgetLimiter:
    """
    Token budget limiter with soft and hard limits.
    
    Uses estimated tokens at admission time and reconciles
    against actual tokens when available. Tracks usage per step
    to identify which steps are consuming the budget.
    """
    
    def __init__(
        self,
        soft_tokens_per_minute: int = 200000,
        hard_tokens_per_minute: int = 240000,
        window_size_seconds: int = 60,
    ):
        """
        Initialize the token budget limiter.
        
        Args:
            soft_tokens_per_minute: Soft limit (start throttling)
            hard_tokens_per_minute: Hard limit (reject requests)
            window_size_seconds: Sliding window size
        """
        self._soft_limit = soft_tokens_per_minute
        self._hard_limit = hard_tokens_per_minute
        self._window_size = window_size_seconds
        self._lock = threading.Lock()
        
        # Sliding window of token records
        self._records: Deque[TokenRecord] = deque()
        
        # Statistics
        self._total_estimated = 0
        self._total_actual = 0
        self._total_requests = 0
        self._soft_limit_hits = 0
        self._hard_limit_hits = 0
    
    def _cleanup_old_records(self) -> None:
        """Remove records outside the sliding window."""
        cutoff = time.time() - self._window_size
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()
    
    def _get_current_tokens(self) -> int:
        """Get current tokens in the window."""
        self._cleanup_old_records()
        total = 0
        for record in self._records:
            # Use actual tokens if available, otherwise estimated
            total += record.actual_tokens if record.actual_tokens is not None else record.estimated_tokens
        return total
    
    def _get_tokens_by_step(self) -> dict:
        """Get token usage by step."""
        self._cleanup_old_records()
        by_step = {}
        for record in self._records:
            tokens = record.actual_tokens if record.actual_tokens is not None else record.estimated_tokens
            by_step[record.step_name] = by_step.get(record.step_name, 0) + tokens
        return by_step
    
    def check_budget(self, estimated_tokens: int) -> Tuple[TokenBudgetStatus, float]:
        """
        Check if a request can be admitted.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            Tuple of (status, wait_time_seconds)
        """
        with self._lock:
            current = self._get_current_tokens()
            projected = current + estimated_tokens
            
            if projected >= self._hard_limit:
                # Calculate wait time for oldest record to expire
                if self._records:
                    oldest = self._records[0]
                    wait_time = (oldest.timestamp + self._window_size) - time.time()
                    return TokenBudgetStatus.HARD_LIMIT, max(0.0, wait_time + 0.1)
                return TokenBudgetStatus.HARD_LIMIT, 1.0
            
            if projected >= self._soft_limit:
                return TokenBudgetStatus.SOFT_LIMIT, 0.0
            
            return TokenBudgetStatus.OK, 0.0
    
    def try_acquire(
        self,
        estimated_tokens: int,
        step_name: str = "",
        request_id: str = "",
    ) -> Tuple[bool, TokenBudgetStatus]:
        """
        Try to acquire token budget.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            step_name: Name of the step
            request_id: Optional request ID for tracking
            
        Returns:
            Tuple of (success, status)
        """
        with self._lock:
            current = self._get_current_tokens()
            projected = current + estimated_tokens
            
            if projected >= self._hard_limit:
                self._hard_limit_hits += 1
                return False, TokenBudgetStatus.HARD_LIMIT
            
            # Record the request
            record = TokenRecord(
                timestamp=time.time(),
                estimated_tokens=estimated_tokens,
                step_name=step_name,
                request_id=request_id,
            )
            self._records.append(record)
            self._total_estimated += estimated_tokens
            self._total_requests += 1
            
            if projected >= self._soft_limit:
                self._soft_limit_hits += 1
                return True, TokenBudgetStatus.SOFT_LIMIT
            
            return True, TokenBudgetStatus.OK
    
    def acquire(
        self,
        estimated_tokens: int,
        step_name: str = "",
        request_id: str = "",
        timeout: Optional[float] = None,
    ) -> Tuple[bool, TokenBudgetStatus]:
        """
        Acquire token budget, waiting if necessary.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            step_name: Name of the step
            request_id: Optional request ID
            timeout: Maximum time to wait
            
        Returns:
            Tuple of (success, status)
        """
        start = time.time()
        deadline = start + timeout if timeout else float('inf')
        
        while True:
            success, status = self.try_acquire(estimated_tokens, step_name, request_id)
            if success:
                return success, status
            
            # Check timeout
            now = time.time()
            if now >= deadline:
                return False, status
            
            # Calculate wait time
            _, wait_time = self.check_budget(estimated_tokens)
            actual_wait = min(wait_time, deadline - now, 5.0)  # Cap at 5s per iteration
            
            if actual_wait > 0:
                time.sleep(actual_wait)
            else:
                time.sleep(0.1)  # Small delay to prevent tight loop
    
    def record_actual_tokens(
        self,
        request_id: str,
        actual_tokens: int,
    ) -> None:
        """
        Record actual token usage for a request.
        
        Args:
            request_id: Request ID to update
            actual_tokens: Actual tokens used
        """
        with self._lock:
            # Find and update the record
            for record in reversed(self._records):
                if record.request_id == request_id:
                    record.actual_tokens = actual_tokens
                    self._total_actual += actual_tokens
                    break
    
    def record_actual_tokens_for_latest(self, actual_tokens: int) -> None:
        """
        Record actual tokens for the most recent request.
        
        Use this when request_id is not available.
        
        Args:
            actual_tokens: Actual tokens used
        """
        with self._lock:
            if self._records:
                record = self._records[-1]
                record.actual_tokens = actual_tokens
                self._total_actual += actual_tokens
    
    def get_stats(self) -> TokenBudgetStats:
        """Get current statistics."""
        with self._lock:
            current = self._get_current_tokens()
            by_step = self._get_tokens_by_step()
            
            utilization = current / self._hard_limit if self._hard_limit > 0 else 0.0
            
            if current >= self._hard_limit:
                status = TokenBudgetStatus.HARD_LIMIT
            elif current >= self._soft_limit:
                status = TokenBudgetStatus.SOFT_LIMIT
            else:
                status = TokenBudgetStatus.OK
            
            return TokenBudgetStats(
                current_tokens_in_window=current,
                soft_limit=self._soft_limit,
                hard_limit=self._hard_limit,
                utilization=utilization,
                status=status,
                requests_in_window=len(self._records),
                tokens_by_step=by_step,
            )
    
    def get_headroom(self) -> int:
        """Get remaining token headroom before hard limit."""
        with self._lock:
            current = self._get_current_tokens()
            return max(0, self._hard_limit - current)
    
    def update_limits(
        self,
        soft_limit: Optional[int] = None,
        hard_limit: Optional[int] = None,
    ) -> None:
        """
        Update the token limits.
        
        Args:
            soft_limit: New soft limit (or None to keep current)
            hard_limit: New hard limit (or None to keep current)
        """
        with self._lock:
            if soft_limit is not None:
                self._soft_limit = soft_limit
            if hard_limit is not None:
                self._hard_limit = hard_limit
    
    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._records.clear()
            self._total_estimated = 0
            self._total_actual = 0
            self._total_requests = 0
            self._soft_limit_hits = 0
            self._hard_limit_hits = 0
