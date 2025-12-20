"""
Smart Throttler Package for Gemini API.

This package provides production-grade rate limiting with:
- Smart Rate Limiter: Sliding window RPM/TPM tracking with pre-emptive throttling
- Concurrency, pacing, and token budget limiters
- Adaptive controller with AIMD (TCP-style congestion control)
- Priority queueing and flow ordering
- Retry policy with jittered backoff
- Response caching for eval reruns
- Comprehensive telemetry

Quick Start (Legacy API - compatible with existing ragas code):
    from lib.core.smart_throttler import SmartRateLimiter, get_rate_limiter
    
    limiter = get_rate_limiter()
    await limiter.acquire(estimated_tokens=1000)
    # ... make API call ...
    limiter.release_sync()

Advanced API (Full throttler with adaptive control):
    from lib.core.smart_throttler import get_throttler
    
    client = get_throttler()
    with client.acquire(step_name="judge") as permit:
        # ... make API call ...
        permit.record_outcome(success=True, tokens_used=1500)
"""

# =============================================================================
# Legacy Rate Limiter API (Primary for ragas integration)
# =============================================================================

from lib.core.smart_throttler.rate_limiter import (
    SmartRateLimiter,
    RateLimitConfig,
    RateLimitStats,
    UsageRecord,
    ThrottleReason,
    get_rate_limiter,
    reset_rate_limiter,
    configure_rate_limiter,
)

# =============================================================================
# Component Imports (for advanced usage)
# =============================================================================

from lib.core.smart_throttler.profiles import (
    StepProfile,
    ProfileManager,
    get_profile_manager,
    reset_profile_manager,
)

from lib.core.smart_throttler.limiter import (
    ConcurrencyLimiter,
    PacingLimiter,
    TokenBudgetLimiter,
)

from lib.core.smart_throttler.queueing import (
    PriorityQueue,
    QueuedRequest,
    FlowGovernor,
)

from lib.core.smart_throttler.control import (
    AdaptiveController,
    ControllerState,
    ControllerOutput,
    HealthModel,
    HealthStatus,
    OverloadType,
)

from lib.core.smart_throttler.retry_policy import (
    RetryPolicy,
    RetryConfig,
    RetryOutcome,
)

from lib.core.smart_throttler.cache import (
    ResponseCache,
    CacheConfig,
    CacheStats,
)

from lib.core.smart_throttler.telemetry import (
    TelemetryCollector,
    ThrottlerMetrics,
    get_telemetry,
    reset_telemetry,
)

__all__ = [
    # Legacy Rate Limiter (primary API for ragas)
    "SmartRateLimiter",
    "RateLimitConfig",
    "RateLimitStats",
    "UsageRecord",
    "ThrottleReason",
    "get_rate_limiter",
    "reset_rate_limiter",
    "configure_rate_limiter",
    # Profiles
    "StepProfile",
    "ProfileManager",
    "get_profile_manager",
    "reset_profile_manager",
    # Limiters
    "ConcurrencyLimiter",
    "PacingLimiter",
    "TokenBudgetLimiter",
    # Queueing
    "PriorityQueue",
    "QueuedRequest",
    "FlowGovernor",
    # Control
    "AdaptiveController",
    "ControllerState",
    "ControllerOutput",
    "HealthModel",
    "HealthStatus",
    "OverloadType",
    # Retry
    "RetryPolicy",
    "RetryConfig",
    "RetryOutcome",
    # Cache
    "ResponseCache",
    "CacheConfig",
    "CacheStats",
    # Telemetry
    "TelemetryCollector",
    "ThrottlerMetrics",
    "get_telemetry",
    "reset_telemetry",
]
