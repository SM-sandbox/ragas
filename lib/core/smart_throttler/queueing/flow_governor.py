"""
Flow Governor for Smart Throttler.

Enforces max_in_flight_per_flow to preserve ordering for chained workflows
without serializing all work.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from contextlib import contextmanager, asynccontextmanager


@dataclass
class FlowState:
    """State of a flow."""
    flow_id: str
    in_flight: int = 0
    total_requests: int = 0
    last_activity: float = field(default_factory=time.time)
    waiting: int = 0


@dataclass
class FlowGovernorStats:
    """Statistics for flow governor."""
    active_flows: int
    total_in_flight: int
    total_waiting: int
    flows: Dict[str, FlowState]


class FlowGovernor:
    """
    Flow governor for chained workflow ordering.
    
    Ensures that requests within a flow are processed in order by
    limiting the number of in-flight requests per flow (default 1).
    
    This preserves ordering for chained workflows without serializing
    all work across different flows.
    """
    
    def __init__(self, max_in_flight_per_flow: int = 1):
        """
        Initialize the flow governor.
        
        Args:
            max_in_flight_per_flow: Maximum concurrent requests per flow
        """
        self._max_in_flight = max_in_flight_per_flow
        self._lock = threading.Lock()
        
        # Flow state tracking
        self._flows: Dict[str, FlowState] = {}
        
        # Condition variables for waiting
        self._conditions: Dict[str, threading.Condition] = {}
        
        # Async events for waiting
        self._async_events: Dict[str, asyncio.Event] = {}
        
        # Cleanup threshold (seconds of inactivity)
        self._cleanup_threshold = 3600  # 1 hour
    
    def _get_or_create_flow(self, flow_id: str) -> FlowState:
        """Get or create flow state."""
        if flow_id not in self._flows:
            self._flows[flow_id] = FlowState(flow_id=flow_id)
            self._conditions[flow_id] = threading.Condition(self._lock)
        return self._flows[flow_id]
    
    def _cleanup_stale_flows(self) -> None:
        """Remove flows that have been inactive."""
        now = time.time()
        stale = [
            fid for fid, state in self._flows.items()
            if state.in_flight == 0 and 
               state.waiting == 0 and
               now - state.last_activity > self._cleanup_threshold
        ]
        for fid in stale:
            del self._flows[fid]
            self._conditions.pop(fid, None)
            self._async_events.pop(fid, None)
    
    def try_acquire(self, flow_id: Optional[str]) -> bool:
        """
        Try to acquire a slot for a flow without blocking.
        
        Args:
            flow_id: Flow ID (None = no flow constraint)
            
        Returns:
            True if acquired, False if flow is at capacity
        """
        if flow_id is None:
            return True
        
        with self._lock:
            flow = self._get_or_create_flow(flow_id)
            
            if flow.in_flight < self._max_in_flight:
                flow.in_flight += 1
                flow.total_requests += 1
                flow.last_activity = time.time()
                return True
            
            return False
    
    def acquire(self, flow_id: Optional[str], timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot for a flow, blocking if necessary.
        
        Args:
            flow_id: Flow ID (None = no flow constraint)
            timeout: Maximum time to wait
            
        Returns:
            True if acquired, False if timeout
        """
        if flow_id is None:
            return True
        
        with self._lock:
            flow = self._get_or_create_flow(flow_id)
            condition = self._conditions[flow_id]
            
            deadline = time.time() + timeout if timeout else None
            flow.waiting += 1
            
            try:
                while flow.in_flight >= self._max_in_flight:
                    if deadline:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            return False
                        condition.wait(timeout=remaining)
                    else:
                        condition.wait()
                
                flow.in_flight += 1
                flow.total_requests += 1
                flow.last_activity = time.time()
                return True
            finally:
                flow.waiting -= 1
    
    async def acquire_async(self, flow_id: Optional[str], timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot for a flow asynchronously.
        
        Args:
            flow_id: Flow ID (None = no flow constraint)
            timeout: Maximum time to wait
            
        Returns:
            True if acquired, False if timeout
        """
        if flow_id is None:
            return True
        
        start = time.time()
        deadline = start + timeout if timeout else None
        
        while True:
            with self._lock:
                flow = self._get_or_create_flow(flow_id)
                
                if flow.in_flight < self._max_in_flight:
                    flow.in_flight += 1
                    flow.total_requests += 1
                    flow.last_activity = time.time()
                    return True
            
            # Check timeout
            if deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                await asyncio.sleep(min(0.1, remaining))
            else:
                await asyncio.sleep(0.1)
    
    def release(self, flow_id: Optional[str]) -> None:
        """
        Release a slot for a flow.
        
        Args:
            flow_id: Flow ID (None = no-op)
        """
        if flow_id is None:
            return
        
        with self._lock:
            if flow_id in self._flows:
                flow = self._flows[flow_id]
                if flow.in_flight > 0:
                    flow.in_flight -= 1
                    flow.last_activity = time.time()
                
                # Notify waiters
                if flow_id in self._conditions:
                    self._conditions[flow_id].notify()
            
            # Periodic cleanup
            self._cleanup_stale_flows()
    
    @contextmanager
    def slot(self, flow_id: Optional[str], timeout: Optional[float] = None):
        """
        Context manager for acquiring a flow slot.
        
        Args:
            flow_id: Flow ID
            timeout: Maximum time to wait
            
        Yields:
            True if acquired
            
        Raises:
            TimeoutError: If timeout exceeded
        """
        acquired = self.acquire(flow_id, timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Failed to acquire flow slot for {flow_id} within {timeout}s")
        try:
            yield True
        finally:
            self.release(flow_id)
    
    @asynccontextmanager
    async def slot_async(self, flow_id: Optional[str], timeout: Optional[float] = None):
        """
        Async context manager for acquiring a flow slot.
        
        Args:
            flow_id: Flow ID
            timeout: Maximum time to wait
            
        Yields:
            True if acquired
            
        Raises:
            TimeoutError: If timeout exceeded
        """
        acquired = await self.acquire_async(flow_id, timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Failed to acquire flow slot for {flow_id} within {timeout}s")
        try:
            yield True
        finally:
            self.release(flow_id)
    
    def get_flow_state(self, flow_id: str) -> Optional[FlowState]:
        """Get state of a specific flow."""
        with self._lock:
            return self._flows.get(flow_id)
    
    def get_stats(self) -> FlowGovernorStats:
        """Get current statistics."""
        with self._lock:
            total_in_flight = sum(f.in_flight for f in self._flows.values())
            total_waiting = sum(f.waiting for f in self._flows.values())
            
            return FlowGovernorStats(
                active_flows=len(self._flows),
                total_in_flight=total_in_flight,
                total_waiting=total_waiting,
                flows=dict(self._flows),
            )
    
    def update_max_in_flight(self, new_max: int) -> None:
        """Update the maximum in-flight per flow."""
        with self._lock:
            self._max_in_flight = new_max
            # Notify all waiters to re-check
            for condition in self._conditions.values():
                condition.notify_all()
    
    def clear(self) -> None:
        """Clear all flow state."""
        with self._lock:
            self._flows.clear()
            self._conditions.clear()
            self._async_events.clear()
