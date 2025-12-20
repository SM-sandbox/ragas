"""
Priority Queue for Smart Throttler.

Provides weighted priority queueing with deadlines and max queue limits.
Supports interactive, standard, and background priority classes.
"""

import asyncio
import heapq
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Any, Callable


class Priority(IntEnum):
    """Priority levels (lower number = higher priority)."""
    INTERACTIVE = 0
    STANDARD = 1
    BACKGROUND = 2


@dataclass
class PriorityConfig:
    """Configuration for a priority class."""
    weight: int = 1
    max_queue: int = 10000
    deadline_s: Optional[float] = None


@dataclass(order=True)
class QueuedRequest:
    """
    A request in the priority queue.
    
    Ordered by (priority, timestamp) for fair scheduling within priority.
    """
    priority: Priority = field(compare=True)
    timestamp: float = field(compare=True)
    request_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    step_name: str = field(compare=False, default="")
    flow_id: Optional[str] = field(compare=False, default=None)
    estimated_tokens: int = field(compare=False, default=0)
    deadline: Optional[float] = field(compare=False, default=None)
    payload: Any = field(compare=False, default=None)
    
    # Event for signaling when request is admitted
    _admitted_event: Optional[asyncio.Event] = field(compare=False, default=None, repr=False)
    _admitted: bool = field(compare=False, default=False)
    _cancelled: bool = field(compare=False, default=False)
    
    def is_expired(self) -> bool:
        """Check if request has exceeded its deadline."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline
    
    def time_until_deadline(self) -> Optional[float]:
        """Get seconds until deadline (None if no deadline)."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())


@dataclass
class QueueStats:
    """Statistics for the priority queue."""
    total_queued: int
    by_priority: Dict[str, int]
    total_admitted: int
    total_expired: int
    total_rejected: int
    oldest_request_age_s: float


class PriorityQueue:
    """
    Weighted priority queue with deadline support.
    
    Implements weighted fair queueing where higher-weight priorities
    get more slots. Supports deadlines and max queue limits per priority.
    """
    
    def __init__(
        self,
        priority_configs: Optional[Dict[str, PriorityConfig]] = None,
    ):
        """
        Initialize the priority queue.
        
        Args:
            priority_configs: Configuration per priority level
        """
        self._lock = threading.Lock()
        
        # Default configurations
        self._configs: Dict[Priority, PriorityConfig] = {
            Priority.INTERACTIVE: PriorityConfig(weight=5, max_queue=200, deadline_s=10),
            Priority.STANDARD: PriorityConfig(weight=2, max_queue=20000, deadline_s=600),
            Priority.BACKGROUND: PriorityConfig(weight=1, max_queue=500000, deadline_s=7200),
        }
        
        # Override with provided configs
        if priority_configs:
            for name, config in priority_configs.items():
                priority = self._parse_priority(name)
                self._configs[priority] = config
        
        # Separate heaps per priority for weighted scheduling
        self._queues: Dict[Priority, List[QueuedRequest]] = {
            Priority.INTERACTIVE: [],
            Priority.STANDARD: [],
            Priority.BACKGROUND: [],
        }
        
        # Request lookup by ID
        self._requests: Dict[str, QueuedRequest] = {}
        
        # Statistics
        self._total_admitted = 0
        self._total_expired = 0
        self._total_rejected = 0
        
        # Weighted round-robin state
        self._weight_counters: Dict[Priority, int] = {p: 0 for p in Priority}
    
    def _parse_priority(self, name: str) -> Priority:
        """Parse priority from string name."""
        name_lower = name.lower()
        if name_lower == "interactive":
            return Priority.INTERACTIVE
        elif name_lower == "standard":
            return Priority.STANDARD
        elif name_lower == "background":
            return Priority.BACKGROUND
        else:
            return Priority.STANDARD
    
    def _cleanup_expired(self, priority: Priority) -> int:
        """Remove expired requests from a queue. Returns count removed."""
        queue = self._queues[priority]
        expired_count = 0
        
        # Filter out expired requests
        valid = []
        for req in queue:
            if req.is_expired() or req._cancelled:
                self._requests.pop(req.request_id, None)
                expired_count += 1
            else:
                valid.append(req)
        
        if expired_count > 0:
            self._queues[priority] = valid
            heapq.heapify(self._queues[priority])
            self._total_expired += expired_count
        
        return expired_count
    
    def enqueue(
        self,
        priority: str = "standard",
        step_name: str = "",
        flow_id: Optional[str] = None,
        estimated_tokens: int = 0,
        deadline_s: Optional[float] = None,
        payload: Any = None,
    ) -> Optional[QueuedRequest]:
        """
        Add a request to the queue.
        
        Args:
            priority: Priority level name
            step_name: Name of the step
            flow_id: Optional flow ID for ordering
            estimated_tokens: Estimated tokens for the request
            deadline_s: Deadline in seconds from now (overrides config)
            payload: Optional payload to store with request
            
        Returns:
            QueuedRequest if enqueued, None if rejected (queue full)
        """
        prio = self._parse_priority(priority)
        config = self._configs[prio]
        
        with self._lock:
            # Cleanup expired first
            self._cleanup_expired(prio)
            
            # Check queue limit
            if len(self._queues[prio]) >= config.max_queue:
                self._total_rejected += 1
                return None
            
            # Calculate deadline
            deadline = None
            if deadline_s is not None:
                deadline = time.time() + deadline_s
            elif config.deadline_s is not None:
                deadline = time.time() + config.deadline_s
            
            # Create request
            request = QueuedRequest(
                priority=prio,
                timestamp=time.time(),
                step_name=step_name,
                flow_id=flow_id,
                estimated_tokens=estimated_tokens,
                deadline=deadline,
                payload=payload,
            )
            
            # Add to queue
            heapq.heappush(self._queues[prio], request)
            self._requests[request.request_id] = request
            
            return request
    
    def dequeue(self) -> Optional[QueuedRequest]:
        """
        Get the next request using weighted fair queueing.
        
        Returns:
            Next QueuedRequest or None if all queues empty
        """
        with self._lock:
            # Try priorities in order, respecting weights
            for priority in Priority:
                self._cleanup_expired(priority)
                
                config = self._configs[priority]
                queue = self._queues[priority]
                
                if not queue:
                    continue
                
                # Check weight counter
                self._weight_counters[priority] += 1
                if self._weight_counters[priority] >= config.weight:
                    self._weight_counters[priority] = 0
                    
                    # Pop from this queue
                    request = heapq.heappop(queue)
                    self._requests.pop(request.request_id, None)
                    self._total_admitted += 1
                    request._admitted = True
                    return request
            
            # If weighted scheduling didn't yield, try any non-empty queue
            for priority in Priority:
                queue = self._queues[priority]
                if queue:
                    request = heapq.heappop(queue)
                    self._requests.pop(request.request_id, None)
                    self._total_admitted += 1
                    request._admitted = True
                    return request
            
            return None
    
    def peek(self) -> Optional[QueuedRequest]:
        """Peek at the next request without removing it."""
        with self._lock:
            for priority in Priority:
                self._cleanup_expired(priority)
                queue = self._queues[priority]
                if queue:
                    return queue[0]
            return None
    
    def cancel(self, request_id: str) -> bool:
        """
        Cancel a queued request.
        
        Args:
            request_id: ID of request to cancel
            
        Returns:
            True if cancelled, False if not found
        """
        with self._lock:
            request = self._requests.get(request_id)
            if request:
                request._cancelled = True
                self._requests.pop(request_id, None)
                return True
            return False
    
    def get_queue_depth(self, priority: Optional[str] = None) -> int:
        """
        Get current queue depth.
        
        Args:
            priority: Specific priority or None for total
            
        Returns:
            Number of queued requests
        """
        with self._lock:
            if priority:
                prio = self._parse_priority(priority)
                self._cleanup_expired(prio)
                return len(self._queues[prio])
            else:
                total = 0
                for prio in Priority:
                    self._cleanup_expired(prio)
                    total += len(self._queues[prio])
                return total
    
    def get_stats(self) -> QueueStats:
        """Get current queue statistics."""
        with self._lock:
            by_priority: Dict[str, int] = {}
            oldest_age = 0.0
            now = time.time()
            
            for priority in Priority:
                self._cleanup_expired(priority)
                queue = self._queues[priority]
                by_priority[priority.name.lower()] = len(queue)
                
                if queue:
                    oldest = min(r.timestamp for r in queue)
                    age = now - oldest
                    oldest_age = max(oldest_age, age)
            
            return QueueStats(
                total_queued=sum(by_priority.values()),
                by_priority=by_priority,
                total_admitted=self._total_admitted,
                total_expired=self._total_expired,
                total_rejected=self._total_rejected,
                oldest_request_age_s=oldest_age,
            )
    
    def clear(self, priority: Optional[str] = None) -> int:
        """
        Clear queued requests.
        
        Args:
            priority: Specific priority or None for all
            
        Returns:
            Number of requests cleared
        """
        with self._lock:
            cleared = 0
            if priority:
                prio = self._parse_priority(priority)
                cleared = len(self._queues[prio])
                for req in self._queues[prio]:
                    self._requests.pop(req.request_id, None)
                self._queues[prio] = []
            else:
                for prio in Priority:
                    cleared += len(self._queues[prio])
                    for req in self._queues[prio]:
                        self._requests.pop(req.request_id, None)
                    self._queues[prio] = []
            return cleared
