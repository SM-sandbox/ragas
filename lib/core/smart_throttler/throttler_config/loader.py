"""
YAML Configuration Loader for Smart Throttler.

Provides configuration loading, validation, and strong defaults aimed at
avoiding 429 errors while maximizing throughput.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class LatencyGuardConfig:
    """Configuration for latency-based throttling."""
    p95_ms_threshold: int = 8000
    latency_decrease_factor: float = 0.8


@dataclass
class ControllerConfig:
    """Configuration for the adaptive controller."""
    additive_increase_per_sec: float = 0.02
    multiplicative_decrease_factor: float = 0.7
    evaluate_window_s: int = 15
    latency_guard: LatencyGuardConfig = field(default_factory=LatencyGuardConfig)


@dataclass
class PacingConfig:
    """Configuration for request pacing."""
    burst_allowance: float = 2.0


@dataclass
class TokenBudgetConfig:
    """Configuration for token budget limiting."""
    enabled: bool = True
    soft_tokens_per_minute: int = 200000
    hard_tokens_per_minute: int = 240000


@dataclass
class PriorityConfig:
    """Configuration for a priority class."""
    weight: int = 1
    max_queue: int = 10000
    deadline_s: Optional[int] = None


@dataclass
class FlowConfig:
    """Configuration for flow ordering."""
    max_in_flight_per_flow: int = 1


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_attempts: int = 6
    base_delay_ms: int = 250
    max_delay_ms: int = 30000
    jitter: bool = True


@dataclass
class CacheConfig:
    """Configuration for response caching."""
    enabled: bool = False
    ttl_s: int = 86400
    max_entries: int = 50000


@dataclass
class StepProfileConfig:
    """Configuration for a step profile."""
    initial_estimated_prompt_tokens: int = 3000
    variance_factor: float = 1.2


@dataclass
class ModelOverrideConfig:
    """Configuration overrides for a specific model."""
    max_concurrency: Optional[int] = None
    initial_admitted_rps: Optional[float] = None
    max_admitted_rps: Optional[float] = None
    min_admitted_rps: Optional[float] = None


@dataclass
class ThrottlerConfig:
    """Complete Smart Throttler configuration."""
    enabled: bool = True
    
    # Concurrency and rate limits
    max_concurrency: int = 16
    initial_admitted_rps: float = 1.0
    min_admitted_rps: float = 0.1
    max_admitted_rps: float = 10.0
    
    # Target 429 rate (near zero)
    target_429_rate: float = 0.002
    
    # Sub-configurations
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    pacing: PacingConfig = field(default_factory=PacingConfig)
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    flows: FlowConfig = field(default_factory=FlowConfig)
    retries: RetryConfig = field(default_factory=RetryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Priority configurations
    priorities: Dict[str, PriorityConfig] = field(default_factory=lambda: {
        "interactive": PriorityConfig(weight=5, max_queue=200, deadline_s=10),
        "standard": PriorityConfig(weight=2, max_queue=20000, deadline_s=600),
        "background": PriorityConfig(weight=1, max_queue=500000, deadline_s=7200),
    })
    
    # Step profiles
    step_profiles: Dict[str, StepProfileConfig] = field(default_factory=lambda: {
        "filtration": StepProfileConfig(initial_estimated_prompt_tokens=2800, variance_factor=1.2),
        "bronze": StepProfileConfig(initial_estimated_prompt_tokens=3000, variance_factor=1.2),
        "verb_noun_noun": StepProfileConfig(initial_estimated_prompt_tokens=12000, variance_factor=1.3),
    })
    
    # Model-specific overrides
    model_overrides: Dict[str, ModelOverrideConfig] = field(default_factory=dict)


def get_default_config() -> ThrottlerConfig:
    """Get the default configuration with strong safe defaults."""
    return ThrottlerConfig()


def _parse_latency_guard(data: Dict[str, Any]) -> LatencyGuardConfig:
    """Parse latency guard configuration."""
    return LatencyGuardConfig(
        p95_ms_threshold=data.get("p95_ms_threshold", 8000),
        latency_decrease_factor=data.get("latency_decrease_factor", 0.8),
    )


def _parse_controller(data: Dict[str, Any]) -> ControllerConfig:
    """Parse controller configuration."""
    latency_guard = _parse_latency_guard(data.get("latency_guard", {}))
    return ControllerConfig(
        additive_increase_per_sec=data.get("additive_increase_per_sec", 0.02),
        multiplicative_decrease_factor=data.get("multiplicative_decrease_factor", 0.7),
        evaluate_window_s=data.get("evaluate_window_s", 15),
        latency_guard=latency_guard,
    )


def _parse_priority(data: Dict[str, Any]) -> PriorityConfig:
    """Parse priority configuration."""
    return PriorityConfig(
        weight=data.get("weight", 1),
        max_queue=data.get("max_queue", 10000),
        deadline_s=data.get("deadline_s"),
    )


def _parse_step_profile(data: Dict[str, Any]) -> StepProfileConfig:
    """Parse step profile configuration."""
    return StepProfileConfig(
        initial_estimated_prompt_tokens=data.get("initial_estimated_prompt_tokens", 3000),
        variance_factor=data.get("variance_factor", 1.2),
    )


def _parse_model_override(data: Dict[str, Any]) -> ModelOverrideConfig:
    """Parse model override configuration."""
    return ModelOverrideConfig(
        max_concurrency=data.get("max_concurrency"),
        initial_admitted_rps=data.get("initial_admitted_rps"),
        max_admitted_rps=data.get("max_admitted_rps"),
        min_admitted_rps=data.get("min_admitted_rps"),
    )


def load_config(data: Dict[str, Any]) -> ThrottlerConfig:
    """
    Load configuration from a dictionary.
    
    Args:
        data: Configuration dictionary (typically from YAML)
        
    Returns:
        Validated ThrottlerConfig
    """
    # Get the smart_throttler section if present
    if "smart_throttler" in data:
        data = data["smart_throttler"]
    
    # Parse defaults section if present
    defaults = data.get("defaults", data)
    
    # Parse sub-configurations
    controller = _parse_controller(defaults.get("controller", {}))
    
    pacing = PacingConfig(
        burst_allowance=defaults.get("pacing", {}).get("burst_allowance", 2.0),
    )
    
    token_budget_data = defaults.get("token_budget", {})
    token_budget = TokenBudgetConfig(
        enabled=token_budget_data.get("enabled", True),
        soft_tokens_per_minute=token_budget_data.get("soft_tokens_per_minute", 200000),
        hard_tokens_per_minute=token_budget_data.get("hard_tokens_per_minute", 240000),
    )
    
    flows = FlowConfig(
        max_in_flight_per_flow=data.get("flows", {}).get("max_in_flight_per_flow", 1),
    )
    
    retries_data = data.get("retries", {})
    retries = RetryConfig(
        max_attempts=retries_data.get("max_attempts", 6),
        base_delay_ms=retries_data.get("base_delay_ms", 250),
        max_delay_ms=retries_data.get("max_delay_ms", 30000),
        jitter=retries_data.get("jitter", True),
    )
    
    cache_data = data.get("cache", {})
    cache = CacheConfig(
        enabled=cache_data.get("enabled", False),
        ttl_s=cache_data.get("ttl_s", 86400),
        max_entries=cache_data.get("max_entries", 50000),
    )
    
    # Parse priorities
    priorities = {}
    for name, pdata in data.get("priorities", {}).items():
        priorities[name] = _parse_priority(pdata)
    if not priorities:
        priorities = get_default_config().priorities
    
    # Parse step profiles
    step_profiles = {}
    for name, pdata in data.get("step_profiles", {}).items():
        step_profiles[name] = _parse_step_profile(pdata)
    if not step_profiles:
        step_profiles = get_default_config().step_profiles
    
    # Parse model overrides
    model_overrides = {}
    for name, odata in data.get("model_overrides", {}).items():
        model_overrides[name] = _parse_model_override(odata)
    
    return ThrottlerConfig(
        enabled=data.get("enabled", True),
        max_concurrency=defaults.get("max_concurrency", 16),
        initial_admitted_rps=defaults.get("initial_admitted_rps", 1.0),
        min_admitted_rps=defaults.get("min_admitted_rps", 0.1),
        max_admitted_rps=defaults.get("max_admitted_rps", 10.0),
        target_429_rate=defaults.get("target_429_rate", 0.002),
        controller=controller,
        pacing=pacing,
        token_budget=token_budget,
        flows=flows,
        retries=retries,
        cache=cache,
        priorities=priorities,
        step_profiles=step_profiles,
        model_overrides=model_overrides,
    )


def load_config_from_file(path: str) -> ThrottlerConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Validated ThrottlerConfig
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return load_config(data or {})


def validate_config(config: ThrottlerConfig) -> List[str]:
    """
    Validate configuration and return list of warnings.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of warning messages (empty if all valid)
    """
    warnings = []
    
    # Check rate limits
    if config.max_admitted_rps < config.min_admitted_rps:
        warnings.append(
            f"max_admitted_rps ({config.max_admitted_rps}) < min_admitted_rps ({config.min_admitted_rps})"
        )
    
    if config.initial_admitted_rps > config.max_admitted_rps:
        warnings.append(
            f"initial_admitted_rps ({config.initial_admitted_rps}) > max_admitted_rps ({config.max_admitted_rps})"
        )
    
    if config.initial_admitted_rps < config.min_admitted_rps:
        warnings.append(
            f"initial_admitted_rps ({config.initial_admitted_rps}) < min_admitted_rps ({config.min_admitted_rps})"
        )
    
    # Check token budget
    if config.token_budget.enabled:
        if config.token_budget.hard_tokens_per_minute < config.token_budget.soft_tokens_per_minute:
            warnings.append(
                f"hard_tokens_per_minute ({config.token_budget.hard_tokens_per_minute}) < "
                f"soft_tokens_per_minute ({config.token_budget.soft_tokens_per_minute})"
            )
    
    # Check controller
    if config.controller.multiplicative_decrease_factor >= 1.0:
        warnings.append(
            f"multiplicative_decrease_factor ({config.controller.multiplicative_decrease_factor}) >= 1.0, "
            "this will not decrease rate on errors"
        )
    
    if config.controller.additive_increase_per_sec <= 0:
        warnings.append(
            f"additive_increase_per_sec ({config.controller.additive_increase_per_sec}) <= 0, "
            "rate will never increase"
        )
    
    # Check priorities have valid weights
    for name, priority in config.priorities.items():
        if priority.weight <= 0:
            warnings.append(f"Priority '{name}' has non-positive weight: {priority.weight}")
    
    return warnings


def find_config_file() -> Optional[str]:
    """
    Find the configuration file in standard locations.
    
    Searches in order:
    1. SMART_THROTTLER_CONFIG environment variable
    2. ./smart_throttler.yaml
    3. ./config/smart_throttler.yaml
    4. ~/.config/smart_throttler.yaml
    
    Returns:
        Path to config file or None if not found
    """
    # Check environment variable
    env_path = os.environ.get("SMART_THROTTLER_CONFIG")
    if env_path and Path(env_path).exists():
        return env_path
    
    # Check standard locations
    locations = [
        Path("smart_throttler.yaml"),
        Path("config/smart_throttler.yaml"),
        Path.home() / ".config" / "smart_throttler.yaml",
    ]
    
    for loc in locations:
        if loc.exists():
            return str(loc)
    
    return None
