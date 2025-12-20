"""
Rate Limiting Configuration.

Provides auto-configuration based on cached GCP quotas.

Usage:
    from lib.core.smart_throttler.config import get_recommended_config, get_model_quota
    
    # Get recommended config for a model
    config = get_recommended_config("gemini-2.5-flash")
    
    # Get quota for a specific model
    quota = get_model_quota("gemini-2.5-flash")
"""

from typing import Dict, Any, Optional
from lib.core.smart_throttler.quota_checker import load_cached_quotas, QUOTA_CACHE_FILE

# Re-export for convenience
__all__ = ["get_model_quota", "get_recommended_config", "QUOTA_CACHE_FILE"]

# Default quotas if cache is not available (conservative estimates)
DEFAULT_QUOTAS = {
    "gemini-2.5-flash": {"rpm": 1000, "tpm": 1000000},
    "gemini-2.0-flash": {"rpm": 1000, "tpm": 1000000},
    "gemini-3-flash": {"rpm": 1000, "tpm": 1000000},
    "gemini-2.5-pro": {"rpm": 200, "tpm": 500000},
    "gemini-3-pro": {"rpm": 200, "tpm": 500000},
}

# Safety margin - use this percentage of actual quota
SAFETY_MARGIN = 0.5  # 50% of actual quota for safety


def get_model_quota(model: str) -> Dict[str, int]:
    """
    Get quota for a specific model.
    
    Checks cache first, falls back to defaults.
    
    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        
    Returns:
        Dict with rpm and tpm
    """
    # Try cache first
    cached = load_cached_quotas()
    if cached:
        quotas = cached.get("quotas", {})
        
        # Try exact match
        if model in quotas:
            return {"rpm": quotas[model]["rpm"], "tpm": quotas[model]["tpm"]}
        
        # Try base model match (e.g., "gemini-2.5-flash-preview" -> "gemini-2.5-flash")
        for cached_model, vals in quotas.items():
            if model.startswith(cached_model) or cached_model.startswith(model.split("-preview")[0]):
                return {"rpm": vals["rpm"], "tpm": vals["tpm"]}
    
    # Fall back to defaults
    for default_model, vals in DEFAULT_QUOTAS.items():
        if model.startswith(default_model) or default_model in model:
            return vals
    
    # Ultimate fallback
    return {"rpm": 1000, "tpm": 1000000}


def get_recommended_config(
    model: str,
    max_concurrent: Optional[int] = None,
    safety_margin: float = SAFETY_MARGIN,
) -> Dict[str, Any]:
    """
    Get recommended rate limiter configuration for a model.
    
    Uses cached quotas with a safety margin.
    
    Args:
        model: Model name
        max_concurrent: Override for max concurrent requests
        safety_margin: Fraction of quota to use (default 0.5 = 50%)
        
    Returns:
        Dict with recommended configuration values
    """
    quota = get_model_quota(model)
    
    # Apply safety margin
    rpm_limit = int(quota["rpm"] * safety_margin)
    tpm_limit = int(quota["tpm"] * safety_margin)
    
    # Calculate recommended concurrent requests
    # Rule of thumb: RPM / 60 = requests per second, aim for ~5s of buffer
    if max_concurrent is None:
        max_concurrent = min(100, max(10, rpm_limit // 60))
    
    return {
        "rpm_limit": rpm_limit,
        "tpm_limit": tpm_limit,
        "max_concurrent_requests": max_concurrent,
        "threshold": 0.9,
        "source_quota": quota,
        "safety_margin": safety_margin,
    }


def get_all_model_quotas() -> Dict[str, Dict[str, int]]:
    """
    Get quotas for all known models.
    
    Returns:
        Dict mapping model names to their quotas
    """
    cached = load_cached_quotas()
    if cached:
        return cached.get("quotas", DEFAULT_QUOTAS)
    return DEFAULT_QUOTAS


def print_quota_summary() -> None:
    """Print a summary of all model quotas."""
    cached = load_cached_quotas()
    
    if cached:
        print(f"Project: {cached.get('project', 'unknown')}")
        print(f"Tier: {cached.get('tier', 'unknown')}")
        print(f"Fetched: {cached.get('fetched_at', 'unknown')}")
        print()
        
        quotas = cached.get("quotas", {})
        for model in sorted(quotas.keys()):
            vals = quotas[model]
            print(f"{model}:")
            print(f"  RPM: {vals['rpm']:,}")
            print(f"  TPM: {vals['tpm']:,}")
    else:
        print("No cached quotas found. Run refresh_quotas() to fetch from GCP.")
        print()
        print("Using defaults:")
        for model, vals in DEFAULT_QUOTAS.items():
            print(f"{model}:")
            print(f"  RPM: {vals['rpm']:,}")
            print(f"  TPM: {vals['tpm']:,}")
