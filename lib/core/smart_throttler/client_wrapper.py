"""
Client Wrapper for Smart Throttler.

The primary interface for callers. Wraps Gemini API calls with:
- Admission control and queueing
- Token estimation using step profiles
- Retry handling
- Outcome recording for calibration and adaptive control

This is the single entry point for all throttled API calls.
"""

import asyncio
import threading
import time
import uuid
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from lib.core.smart_throttler.throttler_config.loader import ThrottlerConfig, get_default_config
from lib.core.smart_throttler.profiles import StepProfile, get_profile_manager
from lib.core.smart_throttler.limiter.concurrency_limiter import ConcurrencyLimiter
from lib.core.smart_throttler.limiter.pacing_limiter import PacingLimiter
from lib.core.smart_throttler.limiter.token_budget_limiter import TokenBudgetLimiter
from lib.core.smart_throttler.queueing.priority_queue import PriorityQueue
from lib.core.smart_throttler.queueing.flow_governor import FlowGovernor
from lib.core.smart_throttler.control.adaptive_controller import AdaptiveController, ControllerConfig
from lib.core.smart_throttler.control.health_model import HealthModel
from lib.core.smart_throttler.retry_policy import RetryPolicy, RetryConfig, RetryOutcome
from lib.core.smart_throttler.cache import ResponseCache, CacheConfig
from lib.core.smart_throttler.telemetry import get_telemetry

if TYPE_CHECKING:
    pass


@dataclass
class RequestSpec:
    """
    Specification for a throttled request.
    
    Contains all information needed to route, prioritize, and execute a request.
    """
    model: str = "gemini-3-flash-preview"
    step_name: str = ""
    priority: str = "standard"
    flow_id: Optional[str] = None
    deadline_s: Optional[float] = None
    prompt: str = ""
    user_content: str = ""
    temperature: float = 0.0
    max_tokens: int = 65536
    idempotency_hint: Optional[str] = None
    
    # Generated fields
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    estimated_tokens: int = 0
    submit_time: float = field(default_factory=time.time)


@dataclass
class ThrottledResponse:
    """
    Response from a throttled request.
    
    Contains the API response plus metadata about throttling.
    """
    text: str
    success: bool
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Timing
    queue_time_ms: float = 0.0
    api_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Throttling info
    was_queued: bool = False
    was_cached: bool = False
    retry_count: int = 0
    
    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class AdmissionPermit:
    """
    Permit for an admitted request.
    
    Used with context manager to ensure proper cleanup.
    """
    request_id: str
    step_name: str
    flow_id: Optional[str]
    estimated_tokens: int
    admit_time: float
    
    # References for cleanup
    _client: Optional["ThrottledClient"] = field(repr=False, default=None)
    _released: bool = False
    
    def record_outcome(
        self,
        success: bool,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        is_429: bool = False,
        is_5xx: bool = False,
    ) -> None:
        """
        Record the outcome of the request.
        
        Args:
            success: Whether request succeeded
            tokens_used: Actual tokens used
            latency_ms: Request latency
            is_429: Whether got 429 error
            is_5xx: Whether got 5xx error
        """
        if self._client:
            self._client._record_outcome(
                request_id=self.request_id,
                step_name=self.step_name,
                success=success,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                is_429=is_429,
                is_5xx=is_5xx,
            )
    
    def release(self) -> None:
        """Release resources held by this permit."""
        if not self._released and self._client:
            self._client._release_permit(self)
            self._released = True


class ThrottledClient:
    """
    Production-grade throttled client for Gemini API.
    
    Provides:
    - Smoothing and burst prevention
    - Concurrency limits
    - Token budget limits with step profiles
    - Adaptive throughput control
    - Priority queueing
    - Flow ordering for chained workflows
    - Caching for eval reruns
    - Comprehensive telemetry
    
    Usage:
        client = ThrottledClient()
        
        # Simple usage
        response = await client.generate(
            model="gemini-3-flash-preview",
            step_name="bronze",
            prompt="...",
            user_content="...",
        )
        
        # With flow ordering
        response = await client.generate(
            step_name="bronze",
            flow_id="case_123",
            priority="standard",
            prompt="...",
            user_content="...",
        )
    """
    
    def __init__(
        self,
        config: Optional[ThrottlerConfig] = None,
        api_client: Optional[Any] = None,
    ):
        """
        Initialize the throttled client.
        
        Args:
            config: Throttler configuration
            api_client: Optional Gemini API client (uses default if not provided)
        """
        self._config = config or get_default_config()
        self._api_client = api_client
        self._lock = threading.Lock()
        
        # Initialize components
        self._init_components()
        
        # Background evaluation thread
        self._running = True
        self._eval_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self._eval_thread.start()
    
    def _init_components(self) -> None:
        """Initialize all throttler components."""
        cfg = self._config
        
        # Profile manager
        self._profiles = get_profile_manager()
        for step_name, profile_cfg in cfg.step_profiles.items():
            self._profiles.register_profile(StepProfile(
                step_name=step_name,
                initial_estimated_prompt_tokens=profile_cfg.initial_estimated_prompt_tokens,
                variance_factor=profile_cfg.variance_factor,
            ))
        
        # Concurrency limiter
        self._concurrency = ConcurrencyLimiter(max_concurrency=cfg.max_concurrency)
        
        # Pacing limiter
        self._pacing = PacingLimiter(
            admitted_rps=cfg.initial_admitted_rps,
            burst_allowance=cfg.pacing.burst_allowance,
        )
        
        # Token budget limiter
        self._token_budget = TokenBudgetLimiter(
            soft_tokens_per_minute=cfg.token_budget.soft_tokens_per_minute,
            hard_tokens_per_minute=cfg.token_budget.hard_tokens_per_minute,
        )
        
        # Priority queue
        self._queue = PriorityQueue()
        
        # Flow governor
        self._flow_governor = FlowGovernor(
            max_in_flight_per_flow=cfg.flows.max_in_flight_per_flow,
        )
        
        # Adaptive controller
        controller_cfg = ControllerConfig(
            initial_admitted_rps=cfg.initial_admitted_rps,
            min_admitted_rps=cfg.min_admitted_rps,
            max_admitted_rps=cfg.max_admitted_rps,
            additive_increase_per_sec=cfg.controller.additive_increase_per_sec,
            multiplicative_decrease_factor=cfg.controller.multiplicative_decrease_factor,
            evaluate_window_s=cfg.controller.evaluate_window_s,
            target_429_rate=cfg.target_429_rate,
            latency_p95_threshold_ms=cfg.controller.latency_guard.p95_ms_threshold,
            latency_decrease_factor=cfg.controller.latency_guard.latency_decrease_factor,
        )
        self._controller = AdaptiveController(config=controller_cfg)
        
        # Health model
        self._health = HealthModel()
        
        # Retry policy
        retry_cfg = RetryConfig(
            max_attempts=cfg.retries.max_attempts,
            base_delay_ms=cfg.retries.base_delay_ms,
            max_delay_ms=cfg.retries.max_delay_ms,
            jitter=cfg.retries.jitter,
        )
        self._retry = RetryPolicy(
            config=retry_cfg,
            on_retry_callback=self._on_retry,
        )
        
        # Cache
        cache_cfg = CacheConfig(
            enabled=cfg.cache.enabled,
            ttl_s=cfg.cache.ttl_s,
            max_entries=cfg.cache.max_entries,
        )
        self._cache = ResponseCache(config=cache_cfg)
        
        # Telemetry
        self._telemetry = get_telemetry()
    
    def _on_retry(self, attempt: int, delay_ms: float, error_type: str) -> None:
        """Callback when a retry occurs."""
        self._telemetry.record_retry()
        
        # Inform controller to reduce rate
        if error_type == "429":
            self._controller.force_decrease()
    
    def _evaluation_loop(self) -> None:
        """Background loop for periodic evaluation."""
        while self._running:
            try:
                # Evaluate controller
                queue_depth = self._queue.get_queue_depth()
                output = self._controller.evaluate(queue_depth=queue_depth)
                
                # Update pacing rate
                self._pacing.update_rate(output.admitted_rps)
                
                # Update telemetry
                self._telemetry.update_rate_control(
                    current_rps=output.admitted_rps,
                    target_rps=self._config.max_admitted_rps,
                )
                
                token_stats = self._token_budget.get_stats()
                self._telemetry.update_token_budget(
                    used=token_stats.current_tokens_in_window,
                    limit=token_stats.hard_limit,
                )
                
                queue_stats = self._queue.get_stats()
                self._telemetry.update_queue_depth(
                    total=queue_stats.total_queued,
                    by_priority=queue_stats.by_priority,
                )
                
                # Maybe log summary
                self._telemetry.maybe_log_summary()
                
                # Record health snapshot
                metrics = self._telemetry.get_metrics()
                rate_429 = 0.0
                rate_5xx = 0.0
                if metrics.requests_completed > 0:
                    rate_429 = metrics.count_429 / metrics.requests_completed
                    rate_5xx = metrics.count_5xx / metrics.requests_completed
                
                self._health.record_snapshot(
                    rate_429=rate_429,
                    rate_5xx=rate_5xx,
                    latency_p95_ms=metrics.latency_p95_ms,
                    queue_depth=queue_stats.total_queued,
                    admitted_rps=output.admitted_rps,
                )
                
            except Exception:
                pass  # Don't crash the evaluation loop
            
            time.sleep(self._config.controller.evaluate_window_s)
    
    def _estimate_tokens(self, spec: RequestSpec) -> int:
        """Estimate tokens for a request."""
        prompt_length = len(spec.prompt) + len(spec.user_content)
        return self._profiles.estimate_tokens(spec.step_name, prompt_length)
    
    def _record_outcome(
        self,
        request_id: str,
        step_name: str,
        success: bool,
        tokens_used: int,
        latency_ms: float,
        is_429: bool,
        is_5xx: bool,
    ) -> None:
        """Record the outcome of a request."""
        # Update controller
        self._controller.record_outcome(
            success=success,
            is_429=is_429,
            is_5xx=is_5xx,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
        )
        
        # Update profile calibration
        if tokens_used > 0:
            self._profiles.record_observation(step_name, tokens_used)
        
        # Update token budget with actual
        if tokens_used > 0:
            self._token_budget.record_actual_tokens(request_id, tokens_used)
        
        # Update telemetry
        if success:
            self._telemetry.record_request_completed(
                latency_ms=latency_ms,
                step_name=step_name,
                tokens_used=tokens_used,
            )
        else:
            self._telemetry.record_request_failed(is_429=is_429, is_5xx=is_5xx)
    
    def _release_permit(self, permit: AdmissionPermit) -> None:
        """Release resources held by a permit."""
        self._concurrency.release()
        self._flow_governor.release(permit.flow_id)
    
    def _try_admit(
        self,
        spec: RequestSpec,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, Optional[AdmissionPermit]]:
        """
        Try to admit a request through all limiters.
        
        Args:
            spec: Request specification
            timeout: Maximum time to wait
            
        Returns:
            Tuple of (admitted, permit)
        """
        start = time.time()
        deadline = start + timeout if timeout else None
        
        # 1. Acquire concurrency slot
        remaining = (deadline - time.time()) if deadline else None
        if not self._concurrency.acquire(timeout=remaining):
            return False, None
        
        try:
            # 2. Acquire pacing slot
            remaining = (deadline - time.time()) if deadline else None
            if not self._pacing.acquire(timeout=remaining):
                self._concurrency.release()
                return False, None
            
            # 3. Acquire token budget
            remaining = (deadline - time.time()) if deadline else None
            success, status = self._token_budget.acquire(
                estimated_tokens=spec.estimated_tokens,
                step_name=spec.step_name,
                request_id=spec.request_id,
                timeout=remaining,
            )
            if not success:
                self._concurrency.release()
                return False, None
            
            # 4. Acquire flow slot
            remaining = (deadline - time.time()) if deadline else None
            if not self._flow_governor.acquire(spec.flow_id, timeout=remaining):
                self._concurrency.release()
                return False, None
            
            # All acquired - create permit
            permit = AdmissionPermit(
                request_id=spec.request_id,
                step_name=spec.step_name,
                flow_id=spec.flow_id,
                estimated_tokens=spec.estimated_tokens,
                admit_time=time.time(),
                _client=self,
            )
            
            self._telemetry.record_request_admitted()
            return True, permit
            
        except Exception:
            self._concurrency.release()
            raise
    
    @contextmanager
    def acquire(
        self,
        step_name: str = "",
        priority: str = "standard",
        flow_id: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Context manager for manual admission control.
        
        Use this when you want to make your own API calls but still
        go through the throttler.
        
        Args:
            step_name: Name of the step
            priority: Priority level
            flow_id: Optional flow ID
            estimated_tokens: Token estimate (uses profile if not provided)
            timeout: Maximum time to wait
            
        Yields:
            AdmissionPermit
            
        Example:
            with client.acquire(step_name="bronze") as permit:
                result = my_api_call()
                permit.record_outcome(success=True, tokens_used=result.tokens)
        """
        spec = RequestSpec(
            step_name=step_name,
            priority=priority,
            flow_id=flow_id,
        )
        
        if estimated_tokens:
            spec.estimated_tokens = estimated_tokens
        else:
            spec.estimated_tokens = self._profiles.estimate_tokens(step_name)
        
        self._telemetry.record_request_submitted()
        
        admitted, permit = self._try_admit(spec, timeout=timeout)
        if not admitted or permit is None:
            raise TimeoutError(f"Failed to acquire admission within {timeout}s")
        
        try:
            yield permit
        finally:
            permit.release()
    
    @asynccontextmanager
    async def acquire_async(
        self,
        step_name: str = "",
        priority: str = "standard",
        flow_id: Optional[str] = None,
        estimated_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Async context manager for manual admission control.
        
        Args:
            step_name: Name of the step
            priority: Priority level
            flow_id: Optional flow ID
            estimated_tokens: Token estimate
            timeout: Maximum time to wait
            
        Yields:
            AdmissionPermit
        """
        # Run sync admission in thread pool
        loop = asyncio.get_event_loop()
        
        spec = RequestSpec(
            step_name=step_name,
            priority=priority,
            flow_id=flow_id,
        )
        
        if estimated_tokens:
            spec.estimated_tokens = estimated_tokens
        else:
            spec.estimated_tokens = self._profiles.estimate_tokens(step_name)
        
        self._telemetry.record_request_submitted()
        
        admitted, permit = await loop.run_in_executor(
            None,
            lambda: self._try_admit(spec, timeout=timeout)
        )
        
        if not admitted or permit is None:
            raise TimeoutError(f"Failed to acquire admission within {timeout}s")
        
        try:
            yield permit
        finally:
            permit.release()
    
    def generate_sync(
        self,
        prompt: str,
        user_content: str,
        model: str = "gemini-3-flash-preview",
        step_name: str = "",
        priority: str = "standard",
        flow_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 65536,
        deadline_s: Optional[float] = None,
    ) -> ThrottledResponse:
        """
        Generate content synchronously with throttling.
        
        Args:
            prompt: System prompt
            user_content: User content
            model: Model name
            step_name: Step name for profiling
            priority: Priority level
            flow_id: Optional flow ID for ordering
            temperature: Temperature
            max_tokens: Max output tokens
            deadline_s: Deadline in seconds
            
        Returns:
            ThrottledResponse
        """
        start_time = time.time()
        
        spec = RequestSpec(
            model=model,
            step_name=step_name,
            priority=priority,
            flow_id=flow_id,
            deadline_s=deadline_s,
            prompt=prompt,
            user_content=user_content,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        spec.estimated_tokens = self._estimate_tokens(spec)
        
        self._telemetry.record_request_submitted()
        
        # Check cache first
        if self._cache.enabled:
            hit, cached = self._cache.get(
                model=model,
                step_name=step_name,
                prompt=prompt + user_content,
                temperature=temperature,
            )
            if hit:
                self._telemetry.record_cache_access(hit=True)
                return ThrottledResponse(
                    text=cached,
                    success=True,
                    was_cached=True,
                    total_time_ms=(time.time() - start_time) * 1000,
                )
            self._telemetry.record_cache_access(hit=False)
        
        # Try to admit
        timeout = deadline_s or 600  # Default 10 min timeout
        admitted, permit = self._try_admit(spec, timeout=timeout)
        
        if not admitted or permit is None:
            return ThrottledResponse(
                text="",
                success=False,
                error="Admission timeout",
                error_type="timeout",
                total_time_ms=(time.time() - start_time) * 1000,
            )
        
        queue_time = (time.time() - start_time) * 1000
        
        try:
            # Execute with retry
            api_start = time.time()
            
            def make_call():
                # Note: This client wrapper is designed for stem-pipeline's genai_client.
                # For ragas integration, use the SmartRateLimiter directly with
                # lib.clients.gemini_client.generate_for_judge()
                raise NotImplementedError(
                    "ThrottledClient.generate_sync() is not yet adapted for ragas. "
                    "Use SmartRateLimiter directly with lib.clients.gemini_client."
                )
            
            result, outcome, attempts = self._retry.execute_with_retry(make_call)
            
            api_time = (time.time() - api_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            if outcome == RetryOutcome.SUCCESS and result:
                # Record success
                permit.record_outcome(
                    success=True,
                    tokens_used=result.metadata.total_tokens,
                    latency_ms=api_time,
                )
                
                # Cache result
                if self._cache.enabled:
                    self._cache.put(
                        model=model,
                        step_name=step_name,
                        prompt=prompt + user_content,
                        response=result.text,
                        temperature=temperature,
                    )
                
                return ThrottledResponse(
                    text=result.text,
                    success=True,
                    prompt_tokens=result.metadata.prompt_tokens,
                    completion_tokens=result.metadata.completion_tokens,
                    total_tokens=result.metadata.total_tokens,
                    queue_time_ms=queue_time,
                    api_time_ms=api_time,
                    total_time_ms=total_time,
                    was_queued=queue_time > 100,
                    retry_count=attempts - 1,
                )
            else:
                permit.record_outcome(success=False, latency_ms=api_time)
                return ThrottledResponse(
                    text="",
                    success=False,
                    error=f"Request failed after {attempts} attempts",
                    error_type=str(outcome),
                    queue_time_ms=queue_time,
                    api_time_ms=api_time,
                    total_time_ms=total_time,
                    retry_count=attempts - 1,
                )
                
        finally:
            permit.release()
    
    async def generate(
        self,
        prompt: str,
        user_content: str,
        model: str = "gemini-3-flash-preview",
        step_name: str = "",
        priority: str = "standard",
        flow_id: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 65536,
        deadline_s: Optional[float] = None,
    ) -> ThrottledResponse:
        """
        Generate content asynchronously with throttling.
        
        This is the primary async interface for throttled generation.
        
        Args:
            prompt: System prompt
            user_content: User content
            model: Model name
            step_name: Step name for profiling
            priority: Priority level
            flow_id: Optional flow ID for ordering
            temperature: Temperature
            max_tokens: Max output tokens
            deadline_s: Deadline in seconds
            
        Returns:
            ThrottledResponse
        """
        # Run sync version in thread pool for now
        # TODO: Implement fully async version
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_sync(
                prompt=prompt,
                user_content=user_content,
                model=model,
                step_name=step_name,
                priority=priority,
                flow_id=flow_id,
                temperature=temperature,
                max_tokens=max_tokens,
                deadline_s=deadline_s,
            )
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "concurrency": self._concurrency.get_stats().__dict__,
            "pacing": self._pacing.get_stats().__dict__,
            "token_budget": self._token_budget.get_stats().__dict__,
            "queue": self._queue.get_stats().__dict__,
            "flow_governor": self._flow_governor.get_stats().__dict__,
            "controller": self._controller.get_state().__dict__,
            "health": self._health.get_state().__dict__,
            "retry": self._retry.get_stats().__dict__,
            "cache": self._cache.get_stats().__dict__,
            "telemetry": self._telemetry.get_metrics().to_dict(),
            "profiles": self._profiles.get_calibration_stats(),
        }
    
    def shutdown(self) -> None:
        """Shutdown the throttled client."""
        self._running = False
        if self._eval_thread.is_alive():
            self._eval_thread.join(timeout=5.0)


# Singleton instance
_throttler: Optional[ThrottledClient] = None
_throttler_lock = threading.Lock()


def get_throttler(config: Optional[ThrottlerConfig] = None) -> ThrottledClient:
    """
    Get the singleton ThrottledClient instance.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        ThrottledClient instance
    """
    global _throttler
    with _throttler_lock:
        if _throttler is None:
            _throttler = ThrottledClient(config=config)
        return _throttler


def reset_throttler() -> None:
    """Reset the singleton ThrottledClient."""
    global _throttler
    with _throttler_lock:
        if _throttler:
            _throttler.shutdown()
        _throttler = None
