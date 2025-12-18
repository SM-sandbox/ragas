"""
Cost Calculator for Gemini Models

Calculates cost based on token usage and model pricing.
Pricing is per 1M tokens.
"""

from typing import Dict, Optional

# Pricing per 1M tokens (as of Dec 2025)
PRICING = {
    "gemini-2.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "thinking": 0.30,
        "cached": 0.01875,  # 75% discount on cached
    },
    "gemini-2.5-flash-preview": {
        "input": 0.075,
        "output": 0.30,
        "thinking": 0.30,
        "cached": 0.01875,
    },
    "gemini-3-flash-preview": {
        "input": 0.10,
        "output": 0.40,
        "thinking": 0.40,
        "cached": 0.025,
    },
    "gemini-2.0-flash": {
        "input": 0.075,
        "output": 0.30,
        "thinking": 0.0,  # No thinking tokens
        "cached": 0.01875,
    },
    # Fallback for unknown models
    "default": {
        "input": 0.10,
        "output": 0.40,
        "thinking": 0.40,
        "cached": 0.025,
    },
}


def get_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a model, with fallback to default."""
    # Normalize model name
    model_lower = model.lower()
    
    # Try exact match first
    if model_lower in PRICING:
        return PRICING[model_lower]
    
    # Try prefix match
    for key in PRICING:
        if model_lower.startswith(key):
            return PRICING[key]
    
    return PRICING["default"]


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    thinking_tokens: int = 0,
    cached_tokens: int = 0,
    model: str = "gemini-2.5-flash",
) -> Dict[str, float]:
    """
    Calculate cost for a generation.
    
    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        thinking_tokens: Number of reasoning tokens
        cached_tokens: Number of cached tokens (reduces input cost)
        model: Model name for pricing lookup
        
    Returns:
        Dict with input_cost, output_cost, thinking_cost, cached_cost, total_cost
    """
    pricing = get_pricing(model)
    
    # Calculate costs (pricing is per 1M tokens)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    thinking_cost = (thinking_tokens / 1_000_000) * pricing["thinking"]
    cached_cost = (cached_tokens / 1_000_000) * pricing["cached"]
    
    total_cost = input_cost + output_cost + thinking_cost + cached_cost
    
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "thinking_cost": round(thinking_cost, 6),
        "cached_cost": round(cached_cost, 6),
        "total_cost": round(total_cost, 6),
    }


def calculate_run_cost(
    total_prompt_tokens: int,
    total_completion_tokens: int,
    total_thinking_tokens: int = 0,
    total_cached_tokens: int = 0,
    model: str = "gemini-2.5-flash",
    question_count: int = 1,
) -> Dict[str, float]:
    """
    Calculate cost for an entire evaluation run.
    
    Returns:
        Dict with total costs and per-question cost
    """
    costs = calculate_cost(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        thinking_tokens=total_thinking_tokens,
        cached_tokens=total_cached_tokens,
        model=model,
    )
    
    costs["cost_per_question"] = round(costs["total_cost"] / max(question_count, 1), 6)
    costs["question_count"] = question_count
    
    return costs


if __name__ == "__main__":
    # Test cost calculation
    print("Cost Calculator Test")
    print("=" * 50)
    
    # Example: 458 questions, ~4000 prompt tokens, ~300 completion tokens each
    total_prompt = 458 * 4000
    total_completion = 458 * 300
    
    for model in ["gemini-2.5-flash", "gemini-3-flash-preview"]:
        costs = calculate_run_cost(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            model=model,
            question_count=458,
        )
        print(f"\n{model}:")
        print(f"  Total cost: ${costs['total_cost']:.4f}")
        print(f"  Per question: ${costs['cost_per_question']:.6f}")
