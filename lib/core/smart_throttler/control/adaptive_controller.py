"""
Adaptive Controller for Smart Throttler.

Implements AIMD-style congestion control:
- Additive Increase: Slowly increase rate when healthy
- Multiplicative Decrease: Quickly decrease rate on 429 or latency inflation

This is the "steering wheel" of the throttler - retries are the "emergency brake".
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple


@dataclass
class ControllerConfig:
    """Configuration for adaptive controller."""
    # Rate limits
    initial_admitted_rps: float = 1.0
    min_admitted_rps: float = 0.1
    max_admitted_rps: float = 10.0
    
    # AIMD parameters
    additive_increase_per_sec: float = 0.02
    multiplicative_decrease_factor: float = 0.7
    
    # Evaluation window
    evaluate_window_s: int = 15
    
    # Target 429 rate (near zero)
    target_429_rate: float = 0.002
    
    # Latency guard
    latency_p95_threshold_ms: int = 8000
    latency_decrease_factor: float = 0.8


@dataclass
class ControllerState:
    """Current state of the adaptive controller."""
    current_admitted_rps: float
    token_budget_scaler: float
    last_adjustment_time: float
    consecutive_healthy_windows: int
    consecutive_unhealthy_windows: int
    is_backing_off: bool


@dataclass
class ControllerOutput:
    """Output from controller evaluation."""
    admitted_rps: float
    token_budget_scaler: float
    action: str  # "increase", "decrease", "hold"
    reason: str


@dataclass
class RequestOutcome:
    """Outcome of a single request."""
    timestamp: float
    success: bool
    is_429: bool
    is_5xx: bool
    latency_ms: float
    tokens_used: int = 0


class AdaptiveController:
    """
    AIMD-style adaptive rate controller.
    
    Monitors request outcomes and adjusts admitted rate:
    - Increases rate slowly when healthy (additive increase)
    - Decreases rate quickly on errors (multiplicative decrease)
    
    Inputs:
    - Rolling 429 rate
    - Rolling retry rate
    - Latency p50 and p95
    - Queue depth
    
    Outputs:
    - current_admitted_rps
    - token_budget_scaler (optional)
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        """
        Initialize the adaptive controller.
        
        Args:
            config: Controller configuration
        """
        self._config = config or ControllerConfig()
        self._lock = threading.Lock()
        
        # Current state
        self._current_rps = self._config.initial_admitted_rps
        self._token_budget_scaler = 1.0
        self._last_adjustment = time.time()
        self._consecutive_healthy = 0
        self._consecutive_unhealthy = 0
        self._is_backing_off = False
        
        # Rolling window of outcomes
        self._outcomes: Deque[RequestOutcome] = deque()
        
        # Statistics
        self._total_increases = 0
        self._total_decreases = 0
        self._total_holds = 0
    
    def record_outcome(
        self,
        success: bool,
        is_429: bool = False,
        is_5xx: bool = False,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
    ) -> None:
        """
        Record the outcome of a request.
        
        Args:
            success: Whether request succeeded
            is_429: Whether request got 429 error
            is_5xx: Whether request got 5xx error
            latency_ms: Request latency in milliseconds
            tokens_used: Tokens used by request
        """
        outcome = RequestOutcome(
            timestamp=time.time(),
            success=success,
            is_429=is_429,
            is_5xx=is_5xx,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )
        
        with self._lock:
            self._outcomes.append(outcome)
            self._cleanup_old_outcomes()
            
            # Immediate decrease on 429
            if is_429:
                self._apply_decrease("429_error")
    
    def _cleanup_old_outcomes(self) -> None:
        """Remove outcomes outside the evaluation window."""
        cutoff = time.time() - self._config.evaluate_window_s
        while self._outcomes and self._outcomes[0].timestamp < cutoff:
            self._outcomes.popleft()
    
    def _get_window_stats(self) -> Tuple[int, int, int, float, float]:
        """
        Get statistics for the current window.
        
        Returns:
            Tuple of (total, 429_count, 5xx_count, latency_p50, latency_p95)
        """
        self._cleanup_old_outcomes()
        
        if not self._outcomes:
            return 0, 0, 0, 0.0, 0.0
        
        total = len(self._outcomes)
        count_429 = sum(1 for o in self._outcomes if o.is_429)
        count_5xx = sum(1 for o in self._outcomes if o.is_5xx)
        
        latencies = sorted(o.latency_ms for o in self._outcomes if o.latency_ms > 0)
        if latencies:
            p50_idx = int(len(latencies) * 0.5)
            p95_idx = int(len(latencies) * 0.95)
            latency_p50 = latencies[min(p50_idx, len(latencies) - 1)]
            latency_p95 = latencies[min(p95_idx, len(latencies) - 1)]
        else:
            latency_p50 = 0.0
            latency_p95 = 0.0
        
        return total, count_429, count_5xx, latency_p50, latency_p95
    
    def _apply_decrease(self, reason: str) -> None:
        """Apply multiplicative decrease."""
        old_rps = self._current_rps
        self._current_rps = max(
            self._config.min_admitted_rps,
            self._current_rps * self._config.multiplicative_decrease_factor
        )
        self._is_backing_off = True
        self._consecutive_healthy = 0
        self._consecutive_unhealthy += 1
        self._total_decreases += 1
        self._last_adjustment = time.time()
    
    def _apply_increase(self) -> None:
        """Apply additive increase."""
        elapsed = time.time() - self._last_adjustment
        increase = self._config.additive_increase_per_sec * elapsed
        
        old_rps = self._current_rps
        self._current_rps = min(
            self._config.max_admitted_rps,
            self._current_rps + increase
        )
        self._is_backing_off = False
        self._consecutive_healthy += 1
        self._consecutive_unhealthy = 0
        self._total_increases += 1
        self._last_adjustment = time.time()
    
    def evaluate(self, queue_depth: int = 0) -> ControllerOutput:
        """
        Evaluate current state and adjust rate if needed.
        
        Should be called periodically (e.g., every evaluate_window_s).
        
        Args:
            queue_depth: Current queue depth
            
        Returns:
            ControllerOutput with new rate and action taken
        """
        with self._lock:
            total, count_429, count_5xx, latency_p50, latency_p95 = self._get_window_stats()
            
            # Calculate rates
            rate_429 = count_429 / total if total > 0 else 0.0
            rate_5xx = count_5xx / total if total > 0 else 0.0
            
            # Check for problems
            has_429_problem = rate_429 > self._config.target_429_rate
            has_latency_problem = latency_p95 > self._config.latency_p95_threshold_ms
            has_5xx_problem = rate_5xx > 0.05  # 5% 5xx is a problem
            
            if has_429_problem:
                self._apply_decrease("429_rate_exceeded")
                return ControllerOutput(
                    admitted_rps=self._current_rps,
                    token_budget_scaler=self._token_budget_scaler,
                    action="decrease",
                    reason=f"429 rate {rate_429:.3f} > target {self._config.target_429_rate}",
                )
            
            if has_latency_problem:
                # Use latency-specific decrease factor
                old_rps = self._current_rps
                self._current_rps = max(
                    self._config.min_admitted_rps,
                    self._current_rps * self._config.latency_decrease_factor
                )
                self._last_adjustment = time.time()
                self._total_decreases += 1
                
                return ControllerOutput(
                    admitted_rps=self._current_rps,
                    token_budget_scaler=self._token_budget_scaler,
                    action="decrease",
                    reason=f"Latency p95 {latency_p95:.0f}ms > threshold {self._config.latency_p95_threshold_ms}ms",
                )
            
            if has_5xx_problem:
                self._apply_decrease("5xx_rate_exceeded")
                return ControllerOutput(
                    admitted_rps=self._current_rps,
                    token_budget_scaler=self._token_budget_scaler,
                    action="decrease",
                    reason=f"5xx rate {rate_5xx:.3f} > 5%",
                )
            
            # All healthy - consider increasing
            if total >= 5:  # Need minimum samples
                self._apply_increase()
                return ControllerOutput(
                    admitted_rps=self._current_rps,
                    token_budget_scaler=self._token_budget_scaler,
                    action="increase",
                    reason=f"Healthy window: 429={rate_429:.4f}, p95={latency_p95:.0f}ms",
                )
            
            # Not enough data - hold
            self._total_holds += 1
            return ControllerOutput(
                admitted_rps=self._current_rps,
                token_budget_scaler=self._token_budget_scaler,
                action="hold",
                reason=f"Insufficient samples ({total})",
            )
    
    def get_current_rps(self) -> float:
        """Get current admitted RPS."""
        with self._lock:
            return self._current_rps
    
    def get_state(self) -> ControllerState:
        """Get current controller state."""
        with self._lock:
            return ControllerState(
                current_admitted_rps=self._current_rps,
                token_budget_scaler=self._token_budget_scaler,
                last_adjustment_time=self._last_adjustment,
                consecutive_healthy_windows=self._consecutive_healthy,
                consecutive_unhealthy_windows=self._consecutive_unhealthy,
                is_backing_off=self._is_backing_off,
            )
    
    def force_decrease(self, factor: Optional[float] = None) -> None:
        """
        Force a rate decrease (e.g., from retry policy).
        
        Args:
            factor: Decrease factor (default: config multiplicative_decrease_factor)
        """
        with self._lock:
            factor = factor or self._config.multiplicative_decrease_factor
            self._current_rps = max(
                self._config.min_admitted_rps,
                self._current_rps * factor
            )
            self._is_backing_off = True
            self._last_adjustment = time.time()
            self._total_decreases += 1
    
    def set_rate(self, rps: float) -> None:
        """
        Set the admitted rate directly.
        
        Args:
            rps: New rate (will be clamped to min/max)
        """
        with self._lock:
            self._current_rps = max(
                self._config.min_admitted_rps,
                min(self._config.max_admitted_rps, rps)
            )
            self._last_adjustment = time.time()
    
    def reset(self) -> None:
        """Reset controller to initial state."""
        with self._lock:
            self._current_rps = self._config.initial_admitted_rps
            self._token_budget_scaler = 1.0
            self._last_adjustment = time.time()
            self._consecutive_healthy = 0
            self._consecutive_unhealthy = 0
            self._is_backing_off = False
            self._outcomes.clear()
            self._total_increases = 0
            self._total_decreases = 0
            self._total_holds = 0
