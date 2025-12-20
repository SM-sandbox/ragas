"""
Health Model for Smart Throttler.

Classifies overload as transient vs persistent to determine
when model failover should be recommended.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional


class OverloadType(str, Enum):
    """Type of overload condition."""
    NONE = "none"
    TRANSIENT = "transient"
    PERSISTENT = "persistent"


class HealthStatus(str, Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthSnapshot:
    """Snapshot of health metrics at a point in time."""
    timestamp: float
    rate_429: float
    rate_5xx: float
    latency_p95_ms: float
    queue_depth: int
    admitted_rps: float


@dataclass
class HealthState:
    """Current health state."""
    status: HealthStatus
    overload_type: OverloadType
    should_failover: bool
    reason: str
    consecutive_unhealthy_minutes: int
    last_healthy_time: float


class HealthModel:
    """
    Health model for classifying overload conditions.
    
    Distinguishes between:
    - Transient overload: Brief spikes that will resolve
    - Persistent overload: Sustained issues requiring intervention
    
    Persistent overload triggers:
    - Sustained 429 above threshold for N minutes even after rate reduction
    - Repeated hard failures unrelated to request sizing
    """
    
    def __init__(
        self,
        persistent_threshold_minutes: int = 5,
        transient_threshold_minutes: int = 1,
        rate_429_threshold: float = 0.1,
        rate_5xx_threshold: float = 0.05,
    ):
        """
        Initialize the health model.
        
        Args:
            persistent_threshold_minutes: Minutes of unhealthy before persistent
            transient_threshold_minutes: Minutes of unhealthy before transient
            rate_429_threshold: 429 rate threshold for unhealthy
            rate_5xx_threshold: 5xx rate threshold for unhealthy
        """
        self._persistent_threshold = persistent_threshold_minutes * 60
        self._transient_threshold = transient_threshold_minutes * 60
        self._rate_429_threshold = rate_429_threshold
        self._rate_5xx_threshold = rate_5xx_threshold
        
        self._lock = threading.Lock()
        
        # Rolling window of health snapshots (1 per minute)
        self._snapshots: Deque[HealthSnapshot] = deque(maxlen=60)
        
        # State tracking
        self._last_healthy_time = time.time()
        self._unhealthy_start_time: Optional[float] = None
        self._current_status = HealthStatus.HEALTHY
        self._current_overload = OverloadType.NONE
        
        # Failover tracking
        self._failover_recommended = False
        self._failover_reason = ""
    
    def record_snapshot(
        self,
        rate_429: float,
        rate_5xx: float,
        latency_p95_ms: float,
        queue_depth: int,
        admitted_rps: float,
    ) -> None:
        """
        Record a health snapshot.
        
        Should be called periodically (e.g., every minute).
        
        Args:
            rate_429: Current 429 error rate
            rate_5xx: Current 5xx error rate
            latency_p95_ms: Current p95 latency
            queue_depth: Current queue depth
            admitted_rps: Current admitted RPS
        """
        snapshot = HealthSnapshot(
            timestamp=time.time(),
            rate_429=rate_429,
            rate_5xx=rate_5xx,
            latency_p95_ms=latency_p95_ms,
            queue_depth=queue_depth,
            admitted_rps=admitted_rps,
        )
        
        with self._lock:
            self._snapshots.append(snapshot)
            self._evaluate_health(snapshot)
    
    def _evaluate_health(self, snapshot: HealthSnapshot) -> None:
        """Evaluate health based on latest snapshot."""
        now = time.time()
        
        # Determine if current snapshot is healthy
        is_healthy = (
            snapshot.rate_429 < self._rate_429_threshold and
            snapshot.rate_5xx < self._rate_5xx_threshold
        )
        
        if is_healthy:
            self._last_healthy_time = now
            self._unhealthy_start_time = None
            self._current_status = HealthStatus.HEALTHY
            self._current_overload = OverloadType.NONE
            self._failover_recommended = False
            self._failover_reason = ""
            return
        
        # Not healthy - track duration
        if self._unhealthy_start_time is None:
            self._unhealthy_start_time = now
        
        unhealthy_duration = now - self._unhealthy_start_time
        
        # Classify overload type
        if unhealthy_duration >= self._persistent_threshold:
            self._current_overload = OverloadType.PERSISTENT
            self._current_status = HealthStatus.CRITICAL
            self._failover_recommended = True
            self._failover_reason = (
                f"Persistent overload for {unhealthy_duration/60:.1f} minutes. "
                f"429 rate: {snapshot.rate_429:.2%}, 5xx rate: {snapshot.rate_5xx:.2%}"
            )
        elif unhealthy_duration >= self._transient_threshold:
            self._current_overload = OverloadType.TRANSIENT
            self._current_status = HealthStatus.UNHEALTHY
        else:
            self._current_overload = OverloadType.NONE
            self._current_status = HealthStatus.DEGRADED
    
    def get_state(self) -> HealthState:
        """Get current health state."""
        with self._lock:
            now = time.time()
            unhealthy_minutes = 0
            
            if self._unhealthy_start_time is not None:
                unhealthy_minutes = int((now - self._unhealthy_start_time) / 60)
            
            return HealthState(
                status=self._current_status,
                overload_type=self._current_overload,
                should_failover=self._failover_recommended,
                reason=self._failover_reason,
                consecutive_unhealthy_minutes=unhealthy_minutes,
                last_healthy_time=self._last_healthy_time,
            )
    
    def should_failover(self) -> bool:
        """Check if model failover is recommended."""
        with self._lock:
            return self._failover_recommended
    
    def get_recent_snapshots(self, minutes: int = 5) -> list:
        """
        Get recent health snapshots.
        
        Args:
            minutes: Number of minutes of history
            
        Returns:
            List of HealthSnapshot
        """
        cutoff = time.time() - (minutes * 60)
        with self._lock:
            return [s for s in self._snapshots if s.timestamp >= cutoff]
    
    def get_trend(self) -> str:
        """
        Get health trend (improving, stable, degrading).
        
        Returns:
            Trend string
        """
        with self._lock:
            if len(self._snapshots) < 3:
                return "unknown"
            
            recent = list(self._snapshots)[-3:]
            rates = [s.rate_429 + s.rate_5xx for s in recent]
            
            if rates[-1] < rates[0] * 0.8:
                return "improving"
            elif rates[-1] > rates[0] * 1.2:
                return "degrading"
            else:
                return "stable"
    
    def acknowledge_failover(self) -> None:
        """Acknowledge that failover has been initiated."""
        with self._lock:
            self._failover_recommended = False
            self._failover_reason = ""
    
    def reset(self) -> None:
        """Reset health model state."""
        with self._lock:
            self._snapshots.clear()
            self._last_healthy_time = time.time()
            self._unhealthy_start_time = None
            self._current_status = HealthStatus.HEALTHY
            self._current_overload = OverloadType.NONE
            self._failover_recommended = False
            self._failover_reason = ""
