"""
Heartbeat Display for Smart Throttler.

Real-time terminal display showing progress and throttler metrics.
Updates every N seconds with current status.
"""

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ProgressState:
    """Current progress state."""
    # Job progress
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    step_name: str = ""
    
    # Throttler metrics
    current_rps: float = 0.0
    target_rps: float = 0.0
    in_flight: int = 0
    queue_depth: int = 0
    
    # Error tracking
    error_429_count: int = 0
    error_5xx_count: int = 0
    retry_count: int = 0
    
    # Token budget
    tokens_used: int = 0
    tokens_limit: int = 0
    
    # Latency
    latency_p95_ms: float = 0.0
    latency_avg_ms: float = 0.0
    
    # Timing
    start_time: float = 0.0
    last_update: float = 0.0


class HeartbeatDisplay:
    """
    Real-time terminal display for throttler progress.
    
    Shows:
    - Job progress (X/Y items, Z% complete)
    - Current RPS and rate control status
    - Error counts (429s, 5xx, retries)
    - Token budget utilization
    - Latency metrics
    - ETA
    
    Usage:
        heartbeat = HeartbeatDisplay(interval_s=10)
        heartbeat.start(total_items=5000, step_name="filtration")
        
        # Update as you process
        heartbeat.update(completed=100, current_rps=3.5, ...)
        
        heartbeat.stop()
    """
    
    def __init__(
        self,
        interval_s: float = 10.0,
        output_func: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize heartbeat display.
        
        Args:
            interval_s: Seconds between heartbeat outputs
            output_func: Custom output function (default: print to stderr)
        """
        self._interval = interval_s
        self._output = output_func or (lambda s: print(s, file=sys.stderr))
        
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._state = ProgressState()
        self._beat_count = 0
    
    def start(
        self,
        total_items: int,
        step_name: str = "",
    ) -> None:
        """
        Start the heartbeat display.
        
        Args:
            total_items: Total items to process
            step_name: Name of the current step
        """
        with self._lock:
            self._state = ProgressState(
                total_items=total_items,
                step_name=step_name,
                start_time=time.time(),
                last_update=time.time(),
            )
            self._running = True
            self._beat_count = 0
        
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        
        # Initial message
        self._output(f"\n🚀 Starting {step_name or 'job'}: {total_items:,} items")
    
    def stop(self) -> None:
        """Stop the heartbeat display."""
        with self._lock:
            self._running = False
        
        # Final summary
        self._print_final_summary()
    
    def update(
        self,
        completed: Optional[int] = None,
        failed: Optional[int] = None,
        current_rps: Optional[float] = None,
        target_rps: Optional[float] = None,
        in_flight: Optional[int] = None,
        queue_depth: Optional[int] = None,
        error_429_count: Optional[int] = None,
        error_5xx_count: Optional[int] = None,
        retry_count: Optional[int] = None,
        tokens_used: Optional[int] = None,
        tokens_limit: Optional[int] = None,
        latency_p95_ms: Optional[float] = None,
        latency_avg_ms: Optional[float] = None,
    ) -> None:
        """
        Update the progress state.
        
        Only updates fields that are provided (not None).
        """
        with self._lock:
            if completed is not None:
                self._state.completed_items = completed
            if failed is not None:
                self._state.failed_items = failed
            if current_rps is not None:
                self._state.current_rps = current_rps
            if target_rps is not None:
                self._state.target_rps = target_rps
            if in_flight is not None:
                self._state.in_flight = in_flight
            if queue_depth is not None:
                self._state.queue_depth = queue_depth
            if error_429_count is not None:
                self._state.error_429_count = error_429_count
            if error_5xx_count is not None:
                self._state.error_5xx_count = error_5xx_count
            if retry_count is not None:
                self._state.retry_count = retry_count
            if tokens_used is not None:
                self._state.tokens_used = tokens_used
            if tokens_limit is not None:
                self._state.tokens_limit = tokens_limit
            if latency_p95_ms is not None:
                self._state.latency_p95_ms = latency_p95_ms
            if latency_avg_ms is not None:
                self._state.latency_avg_ms = latency_avg_ms
            
            self._state.last_update = time.time()
    
    def increment(self, success: bool = True) -> None:
        """Increment completed (or failed) count by 1."""
        with self._lock:
            if success:
                self._state.completed_items += 1
            else:
                self._state.failed_items += 1
            self._state.last_update = time.time()
    
    def _heartbeat_loop(self) -> None:
        """Background thread for periodic heartbeat."""
        while self._running:
            time.sleep(self._interval)
            if self._running:
                self._print_heartbeat()
    
    def _print_heartbeat(self) -> None:
        """Print a single heartbeat line."""
        with self._lock:
            state = self._state
            self._beat_count += 1
        
        # Calculate progress
        total = state.total_items or 1
        completed = state.completed_items
        failed = state.failed_items
        progress_pct = (completed + failed) / total * 100
        
        # Calculate ETA
        elapsed = time.time() - state.start_time
        if completed > 0:
            rate = completed / elapsed
            remaining = total - completed - failed
            eta_s = remaining / rate if rate > 0 else 0
            eta_str = self._format_duration(eta_s)
        else:
            eta_str = "calculating..."
        
        # Calculate error rate
        total_processed = completed + failed
        error_rate = failed / total_processed * 100 if total_processed > 0 else 0
        
        # Token utilization
        token_pct = state.tokens_used / state.tokens_limit * 100 if state.tokens_limit > 0 else 0
        
        # Build heartbeat as multi-line block
        step_prefix = f"[{state.step_name}]" if state.step_name else ""
        
        # Build warning indicators
        warnings = []
        if state.error_429_count > 0 and state.error_429_count % 10 == 0:
            warnings.append("⚠️ 429s")
        if error_rate > 5:
            warnings.append(f"⚠️ {error_rate:.1f}% errors")
        if token_pct > 90:
            warnings.append("⚠️ token budget")
        
        warning_str = f"  {' | '.join(warnings)}" if warnings else ""
        
        lines = [
            f"💓 HEARTBEAT {step_prefix} ─────────────────────────────────",
            f"   Progress: {completed:,}/{total:,} ({progress_pct:.1f}%)  |  ETA: {eta_str}",
            f"   RPS: {state.current_rps:.1f}/{state.target_rps:.1f}  |  in-flight: {state.in_flight}  |  p95: {state.latency_p95_ms:.0f}ms",
            f"   429s: {state.error_429_count}  |  5xx: {state.error_5xx_count}  |  retries: {state.retry_count}  |  tokens: {token_pct:.0f}%",
        ]
        
        if warning_str:
            lines.append(warning_str)
        
        line = "\n".join(lines)
        
        self._output(line)
    
    def _print_final_summary(self) -> None:
        """Print final summary when stopping."""
        with self._lock:
            state = self._state
        
        elapsed = time.time() - state.start_time
        total_processed = state.completed_items + state.failed_items
        success_rate = state.completed_items / total_processed * 100 if total_processed > 0 else 0
        
        self._output("\n" + "─" * 60)
        self._output(f"✅ {state.step_name or 'Job'} Complete")
        self._output(f"   Processed: {total_processed:,}/{state.total_items:,}")
        self._output(f"   Success Rate: {success_rate:.1f}%")
        self._output(f"   Duration: {self._format_duration(elapsed)}")
        self._output(f"   Avg RPS: {total_processed / elapsed:.2f}" if elapsed > 0 else "")
        self._output(f"   429 Errors: {state.error_429_count}")
        self._output(f"   Total Retries: {state.retry_count}")
        self._output("─" * 60 + "\n")
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def force_heartbeat(self) -> None:
        """Force an immediate heartbeat output."""
        self._print_heartbeat()


# Singleton instance
_heartbeat: Optional[HeartbeatDisplay] = None
_heartbeat_lock = threading.Lock()


def get_heartbeat(interval_s: float = 10.0) -> HeartbeatDisplay:
    """Get or create the singleton HeartbeatDisplay."""
    global _heartbeat
    with _heartbeat_lock:
        if _heartbeat is None:
            _heartbeat = HeartbeatDisplay(interval_s=interval_s)
        return _heartbeat


def reset_heartbeat() -> None:
    """Reset the singleton HeartbeatDisplay."""
    global _heartbeat
    with _heartbeat_lock:
        if _heartbeat:
            _heartbeat.stop()
        _heartbeat = None
