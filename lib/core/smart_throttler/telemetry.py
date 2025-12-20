"""
Telemetry for Smart Throttler.

Provides structured logging, metrics collection, and periodic health summaries.
Designed for multi-day pipeline runs with minimal dependencies.
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional, Any


logger = logging.getLogger("smart_throttler")


@dataclass
class ThrottlerMetrics:
    """Current metrics snapshot."""
    timestamp: float = field(default_factory=time.time)
    
    # Request metrics
    requests_submitted: int = 0
    requests_admitted: int = 0
    requests_queued: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    
    # In-flight
    in_flight: int = 0
    
    # Error metrics
    count_429: int = 0
    count_5xx: int = 0
    retry_count: int = 0
    
    # Latency (milliseconds)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_avg_ms: float = 0.0
    
    # Rate control
    current_admitted_rps: float = 0.0
    target_rps: float = 0.0
    
    # Token budget
    token_budget_used: int = 0
    token_budget_limit: int = 0
    token_budget_utilization: float = 0.0
    
    # Queue depth
    queue_depth_total: int = 0
    queue_depth_interactive: int = 0
    queue_depth_standard: int = 0
    queue_depth_background: int = 0
    
    # Cache
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class LatencyRecord:
    """Record of a request latency."""
    timestamp: float
    latency_ms: float
    step_name: str = ""


class TelemetryCollector:
    """
    Collects and reports telemetry for the Smart Throttler.
    
    Features:
    - Structured logging with JSON output option
    - Rolling latency percentile calculation
    - Periodic health summary logging
    - Thread-safe metric updates
    """
    
    def __init__(
        self,
        summary_interval_s: int = 60,
        latency_window_s: int = 300,
        enable_json_logs: bool = False,
    ):
        """
        Initialize the telemetry collector.
        
        Args:
            summary_interval_s: Interval for periodic summaries
            latency_window_s: Window for latency percentile calculation
            enable_json_logs: Use JSON format for logs
        """
        self._summary_interval = summary_interval_s
        self._latency_window = latency_window_s
        self._json_logs = enable_json_logs
        self._lock = threading.Lock()
        
        # Current metrics
        self._metrics = ThrottlerMetrics()
        
        # Latency tracking
        self._latencies: Deque[LatencyRecord] = deque()
        
        # Summary timing
        self._last_summary_time = time.time()
        self._summary_count = 0
        
        # Historical metrics for trend analysis
        self._history: Deque[ThrottlerMetrics] = deque(maxlen=60)
        
        # Start time for uptime calculation
        self._start_time = time.time()
    
    def record_request_submitted(self) -> None:
        """Record a request submission."""
        with self._lock:
            self._metrics.requests_submitted += 1
    
    def record_request_admitted(self) -> None:
        """Record a request admission."""
        with self._lock:
            self._metrics.requests_admitted += 1
            self._metrics.in_flight += 1
    
    def record_request_queued(self) -> None:
        """Record a request being queued."""
        with self._lock:
            self._metrics.requests_queued += 1
    
    def record_request_completed(
        self,
        latency_ms: float,
        step_name: str = "",
        tokens_used: int = 0,
    ) -> None:
        """
        Record a completed request.
        
        Args:
            latency_ms: Request latency in milliseconds
            step_name: Name of the step
            tokens_used: Tokens used by the request
        """
        with self._lock:
            self._metrics.requests_completed += 1
            self._metrics.in_flight = max(0, self._metrics.in_flight - 1)
            
            # Record latency
            self._latencies.append(LatencyRecord(
                timestamp=time.time(),
                latency_ms=latency_ms,
                step_name=step_name,
            ))
            
            # Update token usage
            self._metrics.token_budget_used += tokens_used
            
            # Cleanup old latencies
            self._cleanup_old_latencies()
            
            # Update percentiles
            self._update_latency_percentiles()
    
    def record_request_failed(self, is_429: bool = False, is_5xx: bool = False) -> None:
        """
        Record a failed request.
        
        Args:
            is_429: Whether this was a 429 error
            is_5xx: Whether this was a 5xx error
        """
        with self._lock:
            self._metrics.requests_failed += 1
            self._metrics.in_flight = max(0, self._metrics.in_flight - 1)
            
            if is_429:
                self._metrics.count_429 += 1
            if is_5xx:
                self._metrics.count_5xx += 1
    
    def record_retry(self) -> None:
        """Record a retry attempt."""
        with self._lock:
            self._metrics.retry_count += 1
    
    def record_cache_access(self, hit: bool) -> None:
        """
        Record a cache access.
        
        Args:
            hit: Whether it was a cache hit
        """
        with self._lock:
            if hit:
                self._metrics.cache_hits += 1
            else:
                self._metrics.cache_misses += 1
            
            total = self._metrics.cache_hits + self._metrics.cache_misses
            if total > 0:
                self._metrics.cache_hit_rate = self._metrics.cache_hits / total
    
    def update_rate_control(
        self,
        current_rps: float,
        target_rps: float,
    ) -> None:
        """
        Update rate control metrics.
        
        Args:
            current_rps: Current admitted RPS
            target_rps: Target RPS
        """
        with self._lock:
            self._metrics.current_admitted_rps = current_rps
            self._metrics.target_rps = target_rps
    
    def update_token_budget(
        self,
        used: int,
        limit: int,
    ) -> None:
        """
        Update token budget metrics.
        
        Args:
            used: Tokens used in current window
            limit: Token limit
        """
        with self._lock:
            self._metrics.token_budget_used = used
            self._metrics.token_budget_limit = limit
            if limit > 0:
                self._metrics.token_budget_utilization = used / limit
    
    def update_queue_depth(
        self,
        total: int,
        by_priority: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Update queue depth metrics.
        
        Args:
            total: Total queue depth
            by_priority: Queue depth by priority
        """
        with self._lock:
            self._metrics.queue_depth_total = total
            if by_priority:
                self._metrics.queue_depth_interactive = by_priority.get("interactive", 0)
                self._metrics.queue_depth_standard = by_priority.get("standard", 0)
                self._metrics.queue_depth_background = by_priority.get("background", 0)
    
    def _cleanup_old_latencies(self) -> None:
        """Remove latencies outside the window."""
        cutoff = time.time() - self._latency_window
        while self._latencies and self._latencies[0].timestamp < cutoff:
            self._latencies.popleft()
    
    def _update_latency_percentiles(self) -> None:
        """Update latency percentile metrics."""
        if not self._latencies:
            return
        
        latencies = sorted(r.latency_ms for r in self._latencies)
        n = len(latencies)
        
        self._metrics.latency_avg_ms = sum(latencies) / n
        self._metrics.latency_p50_ms = latencies[int(n * 0.5)]
        self._metrics.latency_p95_ms = latencies[min(int(n * 0.95), n - 1)]
    
    def get_metrics(self) -> ThrottlerMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            # Create a copy with current timestamp
            metrics = ThrottlerMetrics(
                timestamp=time.time(),
                requests_submitted=self._metrics.requests_submitted,
                requests_admitted=self._metrics.requests_admitted,
                requests_queued=self._metrics.requests_queued,
                requests_completed=self._metrics.requests_completed,
                requests_failed=self._metrics.requests_failed,
                in_flight=self._metrics.in_flight,
                count_429=self._metrics.count_429,
                count_5xx=self._metrics.count_5xx,
                retry_count=self._metrics.retry_count,
                latency_p50_ms=self._metrics.latency_p50_ms,
                latency_p95_ms=self._metrics.latency_p95_ms,
                latency_avg_ms=self._metrics.latency_avg_ms,
                current_admitted_rps=self._metrics.current_admitted_rps,
                target_rps=self._metrics.target_rps,
                token_budget_used=self._metrics.token_budget_used,
                token_budget_limit=self._metrics.token_budget_limit,
                token_budget_utilization=self._metrics.token_budget_utilization,
                queue_depth_total=self._metrics.queue_depth_total,
                queue_depth_interactive=self._metrics.queue_depth_interactive,
                queue_depth_standard=self._metrics.queue_depth_standard,
                queue_depth_background=self._metrics.queue_depth_background,
                cache_hits=self._metrics.cache_hits,
                cache_misses=self._metrics.cache_misses,
                cache_hit_rate=self._metrics.cache_hit_rate,
            )
            return metrics
    
    def maybe_log_summary(self) -> bool:
        """
        Log a summary if interval has elapsed.
        
        Returns:
            True if summary was logged
        """
        now = time.time()
        with self._lock:
            if now - self._last_summary_time < self._summary_interval:
                return False
            
            self._last_summary_time = now
            self._summary_count += 1
            
            # Store in history
            self._history.append(self.get_metrics())
        
        # Log summary
        self._log_summary()
        return True
    
    def _log_summary(self) -> None:
        """Log a health summary."""
        metrics = self.get_metrics()
        uptime_s = time.time() - self._start_time
        uptime_h = uptime_s / 3600
        
        if self._json_logs:
            summary = {
                "type": "throttler_summary",
                "uptime_hours": round(uptime_h, 2),
                **metrics.to_dict(),
            }
            logger.info(json.dumps(summary))
        else:
            # Single-line human-readable summary
            rate_429 = 0.0
            if metrics.requests_completed > 0:
                rate_429 = metrics.count_429 / metrics.requests_completed
            
            summary = (
                f"[Throttler] uptime={uptime_h:.1f}h | "
                f"rps={metrics.current_admitted_rps:.2f} | "
                f"in_flight={metrics.in_flight} | "
                f"queue={metrics.queue_depth_total} | "
                f"429s={metrics.count_429} ({rate_429:.2%}) | "
                f"retries={metrics.retry_count} | "
                f"p95={metrics.latency_p95_ms:.0f}ms | "
                f"tokens={metrics.token_budget_used}/{metrics.token_budget_limit} "
                f"({metrics.token_budget_utilization:.0%})"
            )
            logger.info(summary)
    
    def force_log_summary(self) -> None:
        """Force log a summary regardless of interval."""
        with self._lock:
            self._last_summary_time = 0
        self.maybe_log_summary()
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics = ThrottlerMetrics()
            self._latencies.clear()
            self._history.clear()
            self._start_time = time.time()
            self._last_summary_time = time.time()
            self._summary_count = 0


# Singleton instance
_telemetry: Optional[TelemetryCollector] = None
_telemetry_lock = threading.Lock()


def get_telemetry() -> TelemetryCollector:
    """Get the singleton TelemetryCollector instance."""
    global _telemetry
    with _telemetry_lock:
        if _telemetry is None:
            _telemetry = TelemetryCollector()
        return _telemetry


def reset_telemetry() -> None:
    """Reset the singleton TelemetryCollector."""
    global _telemetry
    with _telemetry_lock:
        if _telemetry:
            _telemetry.reset()
        _telemetry = None
