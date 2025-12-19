"""
Model Pricing Utilities

Loads pricing from config/model_pricing.yaml once at import time.
Provides simple functions to get rates and calculate costs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

# Path to pricing config
PRICING_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "model_pricing.yaml"

# Load pricing once at module import
def _load_pricing() -> Dict[str, Any]:
    """Load pricing config from YAML file (called once at import)."""
    with open(PRICING_CONFIG_PATH) as f:
        return yaml.safe_load(f)

# Module-level cache - loaded once
_PRICING = _load_pricing()


def get_model_pricing(model: str) -> Dict[str, float]:
    """
    Get pricing rates for a specific model.
    
    Returns dict with:
        - input_per_1m: cost per 1M input tokens
        - output_per_1m: cost per 1M output tokens
        - thinking_per_1m: cost per 1M thinking tokens
        - cached_per_1m: cost per 1M cached tokens
    """
    models = _PRICING.get("models", {})
    
    if model in models:
        return models[model]
    
    # Return default if model not found
    return _PRICING.get("default", {
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
        "thinking_per_1m": 0.70,
        "cached_per_1m": 0.025,
    })


def calculate_token_cost(
    tokens: Dict[str, int],
    pricing: Dict[str, float],
) -> float:
    """
    Calculate cost for tokens using pre-loaded pricing rates.
    
    Args:
        tokens: dict with prompt, completion, thinking, cached counts
        pricing: dict with input_per_1m, output_per_1m, thinking_per_1m, cached_per_1m
    
    Returns:
        Total cost in USD
    """
    # Use 'or 0' to handle None values
    input_cost = ((tokens.get("prompt") or 0) / 1_000_000) * (pricing.get("input_per_1m") or 0.10)
    output_cost = ((tokens.get("completion") or 0) / 1_000_000) * (pricing.get("output_per_1m") or 0.40)
    thinking_cost = ((tokens.get("thinking") or 0) / 1_000_000) * (pricing.get("thinking_per_1m") or 0.70)
    cached_cost = ((tokens.get("cached") or 0) / 1_000_000) * (pricing.get("cached_per_1m") or 0.025)
    
    return round(input_cost + output_cost + thinking_cost + cached_cost, 8)
