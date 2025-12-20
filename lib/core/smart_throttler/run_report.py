"""
Run Report for Smart Throttler.

Comprehensive tracking and JSON export of all throttler metrics for post-run analysis.
Tracks every error code, timing, rate adjustments, and provides detailed diagnostics.
"""

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ErrorRecord:
    """Record of a single error occurrence."""
    timestamp: float
    error_type: str  # "429", "5xx", "timeout", "connection", etc.
    error_code: Optional[int]
    error_message: str
    step_name: str
    request_id: str
    retry_attempt: int
    latency_ms: float


@dataclass
class RateAdjustment:
    """Record of a rate adjustment."""
    timestamp: float
    old_rps: float
    new_rps: float
    reason: str  # "429_detected", "latency_guard", "healthy_increase", etc.
    trigger_metric: str  # What triggered it


@dataclass
class RequestRecord:
    """Record of a single request."""
    request_id: str
    timestamp: float
    step_name: str
    priority: str
    flow_id: Optional[str]
    estimated_tokens: int
    actual_tokens: int
    queue_time_ms: float
    api_time_ms: float
    total_time_ms: float
    success: bool
    was_cached: bool
    retry_count: int
    error_type: Optional[str]


@dataclass
class RunSummary:
    """Summary statistics for a run."""
    # Timing
    start_time: float
    end_time: float
    duration_s: float
    
    # Request counts
    total_requests: int
    successful_requests: int
    failed_requests: int
    cached_requests: int
    
    # Error breakdown
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    # Retry stats
    total_retries: int = 0
    requests_with_retries: int = 0
    
    # Rate control
    initial_rps: float = 0.0
    final_rps: float = 0.0
    min_rps: float = 0.0
    max_rps: float = 0.0
    rate_adjustments: int = 0
    
    # Token budget
    total_tokens_used: int = 0
    peak_token_utilization: float = 0.0
    
    # Latency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_avg_ms: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # Queue
    peak_queue_depth: int = 0
    avg_queue_time_ms: float = 0.0


@dataclass 
class RunReport:
    """Complete run report with all tracking data."""
    # Metadata
    run_id: str
    run_name: str
    start_time: str
    end_time: str
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Summary
    summary: Optional[RunSummary] = None
    
    # Detailed records (can be large)
    errors: List[ErrorRecord] = field(default_factory=list)
    rate_adjustments: List[RateAdjustment] = field(default_factory=list)
    
    # Per-step breakdown
    step_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Time series (sampled every N seconds)
    time_series: List[Dict[str, Any]] = field(default_factory=list)


class RunReporter:
    """
    Tracks all throttler activity and generates comprehensive run reports.
    
    Usage:
        reporter = RunReporter(run_name="filtration_batch_1")
        reporter.start()
        
        # ... run your pipeline ...
        
        reporter.stop()
        reporter.save_report("reports/run_001.json")
    """
    
    def __init__(
        self,
        run_name: str = "unnamed_run",
        sample_interval_s: float = 10.0,
        output_dir: str = "throttler_reports",
    ):
        """
        Initialize the run reporter.
        
        Args:
            run_name: Human-readable name for this run
            sample_interval_s: How often to sample time series data
            output_dir: Directory for report output
        """
        self._run_name = run_name
        self._run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._sample_interval = sample_interval_s
        self._output_dir = Path(output_dir)
        
        self._lock = threading.Lock()
        self._running = False
        self._sample_thread: Optional[threading.Thread] = None
        
        # Timing
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        # Counters
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._cached_requests = 0
        self._total_retries = 0
        self._requests_with_retries = 0
        
        # Error tracking
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._errors: List[ErrorRecord] = []
        
        # Rate tracking
        self._rate_adjustments: List[RateAdjustment] = []
        self._current_rps = 0.0
        self._initial_rps = 0.0
        self._min_rps = float('inf')
        self._max_rps = 0.0
        
        # Token tracking
        self._total_tokens = 0
        self._peak_token_utilization = 0.0
        
        # Latency tracking
        self._latencies: List[float] = []
        
        # Queue tracking
        self._peak_queue_depth = 0
        self._queue_times: List[float] = []
        
        # Per-step stats
        self._step_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "successes": 0,
                "failures": 0,
                "tokens": 0,
                "latencies": [],
                "errors": defaultdict(int),
            }
        )
        
        # Time series
        self._time_series: List[Dict[str, Any]] = []
        
        # Config snapshot
        self._config_snapshot: Dict[str, Any] = {}
    
    def start(self, config_snapshot: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking."""
        with self._lock:
            self._start_time = time.time()
            self._running = True
            if config_snapshot:
                self._config_snapshot = config_snapshot
        
        # Start sampling thread
        self._sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sample_thread.start()
    
    def stop(self) -> None:
        """Stop tracking."""
        with self._lock:
            self._end_time = time.time()
            self._running = False
    
    def record_request(
        self,
        request_id: str,
        step_name: str,
        success: bool,
        estimated_tokens: int,
        actual_tokens: int,
        queue_time_ms: float,
        api_time_ms: float,
        total_time_ms: float,
        was_cached: bool = False,
        retry_count: int = 0,
        error_type: Optional[str] = None,
        priority: str = "standard",
        flow_id: Optional[str] = None,
    ) -> None:
        """Record a completed request."""
        with self._lock:
            self._total_requests += 1
            
            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1
            
            if was_cached:
                self._cached_requests += 1
            
            if retry_count > 0:
                self._total_retries += retry_count
                self._requests_with_retries += 1
            
            self._total_tokens += actual_tokens
            self._latencies.append(api_time_ms)
            self._queue_times.append(queue_time_ms)
            
            # Per-step tracking
            stats = self._step_stats[step_name]
            stats["requests"] += 1
            if success:
                stats["successes"] += 1
            else:
                stats["failures"] += 1
            stats["tokens"] += actual_tokens
            stats["latencies"].append(api_time_ms)
            
            if error_type:
                stats["errors"][error_type] += 1
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        step_name: str,
        request_id: str,
        retry_attempt: int = 0,
        latency_ms: float = 0.0,
        error_code: Optional[int] = None,
    ) -> None:
        """Record an error occurrence."""
        with self._lock:
            self._error_counts[error_type] += 1
            
            self._errors.append(ErrorRecord(
                timestamp=time.time(),
                error_type=error_type,
                error_code=error_code,
                error_message=error_message[:500],  # Truncate long messages
                step_name=step_name,
                request_id=request_id,
                retry_attempt=retry_attempt,
                latency_ms=latency_ms,
            ))
    
    def record_rate_adjustment(
        self,
        old_rps: float,
        new_rps: float,
        reason: str,
        trigger_metric: str = "",
    ) -> None:
        """Record a rate adjustment."""
        with self._lock:
            self._current_rps = new_rps
            self._min_rps = min(self._min_rps, new_rps)
            self._max_rps = max(self._max_rps, new_rps)
            
            if self._initial_rps == 0.0:
                self._initial_rps = old_rps
            
            self._rate_adjustments.append(RateAdjustment(
                timestamp=time.time(),
                old_rps=old_rps,
                new_rps=new_rps,
                reason=reason,
                trigger_metric=trigger_metric,
            ))
    
    def record_queue_depth(self, depth: int) -> None:
        """Record current queue depth."""
        with self._lock:
            self._peak_queue_depth = max(self._peak_queue_depth, depth)
    
    def record_token_utilization(self, utilization: float) -> None:
        """Record token budget utilization."""
        with self._lock:
            self._peak_token_utilization = max(self._peak_token_utilization, utilization)
    
    def _sample_loop(self) -> None:
        """Background thread to sample time series data."""
        while self._running:
            self._take_sample()
            time.sleep(self._sample_interval)
    
    def _take_sample(self) -> None:
        """Take a time series sample."""
        with self._lock:
            elapsed = time.time() - (self._start_time or time.time())
            
            # Calculate current metrics
            recent_latencies = self._latencies[-100:] if self._latencies else [0]
            avg_latency = sum(recent_latencies) / len(recent_latencies)
            
            rps = 0.0
            if elapsed > 0:
                rps = self._total_requests / elapsed
            
            sample = {
                "elapsed_s": elapsed,
                "timestamp": time.time(),
                "total_requests": self._total_requests,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "current_rps": self._current_rps,
                "actual_rps": rps,
                "error_rate": self._failed_requests / max(1, self._total_requests),
                "avg_latency_ms": avg_latency,
                "queue_depth": self._peak_queue_depth,
                "total_tokens": self._total_tokens,
            }
            
            self._time_series.append(sample)
    
    def get_summary(self) -> RunSummary:
        """Generate run summary."""
        with self._lock:
            start = self._start_time or time.time()
            end = self._end_time or time.time()
            duration = end - start
            
            # Calculate latency percentiles
            sorted_latencies = sorted(self._latencies) if self._latencies else [0]
            n = len(sorted_latencies)
            
            p50 = sorted_latencies[int(n * 0.5)] if n > 0 else 0
            p95 = sorted_latencies[int(n * 0.95)] if n > 0 else 0
            p99 = sorted_latencies[int(n * 0.99)] if n > 0 else 0
            avg = sum(sorted_latencies) / n if n > 0 else 0
            
            # Calculate queue time
            avg_queue = sum(self._queue_times) / len(self._queue_times) if self._queue_times else 0
            
            return RunSummary(
                start_time=start,
                end_time=end,
                duration_s=duration,
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                cached_requests=self._cached_requests,
                error_counts=dict(self._error_counts),
                total_retries=self._total_retries,
                requests_with_retries=self._requests_with_retries,
                initial_rps=self._initial_rps,
                final_rps=self._current_rps,
                min_rps=self._min_rps if self._min_rps != float('inf') else 0,
                max_rps=self._max_rps,
                rate_adjustments=len(self._rate_adjustments),
                total_tokens_used=self._total_tokens,
                peak_token_utilization=self._peak_token_utilization,
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                latency_p99_ms=p99,
                latency_avg_ms=avg,
                requests_per_second=self._total_requests / duration if duration > 0 else 0,
                tokens_per_second=self._total_tokens / duration if duration > 0 else 0,
                peak_queue_depth=self._peak_queue_depth,
                avg_queue_time_ms=avg_queue,
            )
    
    def get_report(self) -> RunReport:
        """Generate complete run report."""
        summary = self.get_summary()
        
        # Convert step stats
        step_stats = {}
        for step_name, stats in self._step_stats.items():
            latencies = stats["latencies"]
            step_stats[step_name] = {
                "requests": stats["requests"],
                "successes": stats["successes"],
                "failures": stats["failures"],
                "success_rate": stats["successes"] / max(1, stats["requests"]),
                "tokens": stats["tokens"],
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "errors": dict(stats["errors"]),
            }
        
        return RunReport(
            run_id=self._run_id,
            run_name=self._run_name,
            start_time=datetime.fromtimestamp(summary.start_time).isoformat(),
            end_time=datetime.fromtimestamp(summary.end_time).isoformat(),
            config_snapshot=self._config_snapshot,
            summary=summary,
            errors=self._errors[-1000:],  # Last 1000 errors
            rate_adjustments=self._rate_adjustments,
            step_stats=step_stats,
            time_series=self._time_series,
        )
    
    def save_report(self, filepath: Optional[str] = None) -> str:
        """
        Save report to JSON file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(self._output_dir / f"{self._run_id}.json")
        
        report = self.get_report()
        
        # Convert to dict for JSON serialization
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        report_dict = to_dict(report)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        return filepath
    
    def print_summary(self) -> None:
        """Print a human-readable summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print(f"RUN REPORT: {self._run_name}")
        print("=" * 70)
        
        print(f"\n📊 OVERVIEW")
        print(f"  Duration: {summary.duration_s:.1f}s")
        print(f"  Total Requests: {summary.total_requests:,}")
        print(f"  Success Rate: {summary.successful_requests / max(1, summary.total_requests):.1%}")
        print(f"  Cached: {summary.cached_requests:,}")
        
        print(f"\n❌ ERRORS")
        print(f"  Failed Requests: {summary.failed_requests:,}")
        print(f"  Total Retries: {summary.total_retries:,}")
        if summary.error_counts:
            for error_type, count in sorted(summary.error_counts.items(), key=lambda x: -x[1]):
                print(f"    {error_type}: {count:,}")
        
        print(f"\n⚡ RATE CONTROL")
        print(f"  Initial RPS: {summary.initial_rps:.2f}")
        print(f"  Final RPS: {summary.final_rps:.2f}")
        print(f"  Min/Max RPS: {summary.min_rps:.2f} / {summary.max_rps:.2f}")
        print(f"  Rate Adjustments: {summary.rate_adjustments:,}")
        
        print(f"\n⏱️  LATENCY")
        print(f"  p50: {summary.latency_p50_ms:.0f}ms")
        print(f"  p95: {summary.latency_p95_ms:.0f}ms")
        print(f"  p99: {summary.latency_p99_ms:.0f}ms")
        print(f"  avg: {summary.latency_avg_ms:.0f}ms")
        
        print(f"\n📈 THROUGHPUT")
        print(f"  Requests/sec: {summary.requests_per_second:.2f}")
        print(f"  Tokens/sec: {summary.tokens_per_second:.0f}")
        print(f"  Total Tokens: {summary.total_tokens_used:,}")
        
        print(f"\n📋 QUEUE")
        print(f"  Peak Depth: {summary.peak_queue_depth:,}")
        print(f"  Avg Queue Time: {summary.avg_queue_time_ms:.0f}ms")
        
        print("\n" + "=" * 70)


# Singleton instance
_reporter: Optional[RunReporter] = None
_reporter_lock = threading.Lock()


def get_reporter(run_name: str = "default") -> RunReporter:
    """Get or create the singleton RunReporter."""
    global _reporter
    with _reporter_lock:
        if _reporter is None:
            _reporter = RunReporter(run_name=run_name)
        return _reporter


def reset_reporter() -> None:
    """Reset the singleton RunReporter."""
    global _reporter
    with _reporter_lock:
        if _reporter:
            _reporter.stop()
        _reporter = None
