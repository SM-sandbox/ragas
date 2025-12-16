"""
Model Registry Integration for BFAI Eval Suite.

Wraps the orchestrator's approved_models registry to provide:
- Model validation
- Thinking config lookup
- Model metadata for reports
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Add orchestrator to path
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")

from services.api.core.approved_models import (
    APPROVED_MODELS,
    get_model as _get_model,
    get_approved_models as _get_approved_models,
    get_models_with_thinking as _get_models_with_thinking,
    get_thinking_config_for_model,
    ModelCard,
    ModelStatus,
    ThinkingConfigType,
)


@dataclass
class ModelInfo:
    """
    Model information for eval reports.
    
    Simplified view of ModelCard for reporting purposes.
    """
    id: str
    name: str
    family: str
    version: str
    status: str
    cost_tier: str
    supports_thinking: bool
    thinking_config_type: str
    can_disable_thinking: bool
    min_thinking_budget: int
    max_thinking_budget: int
    default_thinking_budget: int
    max_input_tokens: int
    max_output_tokens: int
    notes: str
    
    @classmethod
    def from_model_card(cls, card: ModelCard) -> "ModelInfo":
        """Create ModelInfo from orchestrator ModelCard."""
        return cls(
            id=card.id,
            name=card.name,
            family=card.family,
            version=card.version,
            status=card.status.value,
            cost_tier=card.cost_tier,
            supports_thinking=card.supports_thinking,
            thinking_config_type=card.thinking.config_type.value,
            can_disable_thinking=card.thinking.can_disable,
            min_thinking_budget=card.thinking.min_budget,
            max_thinking_budget=card.thinking.max_budget,
            default_thinking_budget=card.thinking.default_budget,
            max_input_tokens=card.max_input_tokens,
            max_output_tokens=card.max_output_tokens,
            notes=card.notes,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "status": self.status,
            "cost_tier": self.cost_tier,
            "supports_thinking": self.supports_thinking,
            "thinking_config_type": self.thinking_config_type,
            "can_disable_thinking": self.can_disable_thinking,
            "min_thinking_budget": self.min_thinking_budget,
            "max_thinking_budget": self.max_thinking_budget,
            "default_thinking_budget": self.default_thinking_budget,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "notes": self.notes,
        }


def get_model(model_id: str) -> Optional[ModelInfo]:
    """
    Get model info by ID.
    
    Args:
        model_id: The model identifier (e.g., "gemini-2.5-flash")
        
    Returns:
        ModelInfo if found, None otherwise
    """
    card = _get_model(model_id)
    if card is None:
        return None
    return ModelInfo.from_model_card(card)


def get_approved_models() -> List[ModelInfo]:
    """Get all approved models."""
    return [ModelInfo.from_model_card(card) for card in _get_approved_models()]


def get_models_with_thinking() -> List[ModelInfo]:
    """Get all models that support thinking/reasoning."""
    return [ModelInfo.from_model_card(card) for card in _get_models_with_thinking()]


def get_thinking_config(model_id: str, effort: str = "medium") -> Dict[str, Any]:
    """
    Get thinking configuration for a model and effort level.
    
    Args:
        model_id: The model identifier
        effort: "none", "low", "medium", "high"
        
    Returns:
        Dict with thinking_budget or thinking_level, or empty dict
        
    Examples:
        >>> get_thinking_config("gemini-2.5-flash", "high")
        {"thinking_budget": 24576}
        
        >>> get_thinking_config("gemini-3-pro-preview", "high")
        {"thinking_level": "HIGH"}
        
        >>> get_thinking_config("gemini-2.0-flash", "high")
        {}
    """
    return get_thinking_config_for_model(model_id, effort)


@dataclass
class ModelValidationResult:
    """Result of model validation."""
    valid: bool
    model_id: str
    model_info: Optional[ModelInfo]
    errors: List[str]
    warnings: List[str]


def validate_model(
    model_id: str,
    require_thinking: bool = False,
    effort: Optional[str] = None,
) -> ModelValidationResult:
    """
    Validate a model for use in experiments.
    
    Args:
        model_id: The model identifier to validate
        require_thinking: If True, model must support thinking
        effort: If provided, validate thinking config for this effort level
        
    Returns:
        ModelValidationResult with validation status and any errors/warnings
    """
    errors = []
    warnings = []
    
    # Check if model exists
    model_info = get_model(model_id)
    if model_info is None:
        return ModelValidationResult(
            valid=False,
            model_id=model_id,
            model_info=None,
            errors=[f"Model '{model_id}' not found in approved models registry"],
            warnings=[],
        )
    
    # Check status
    if model_info.status == "deprecated":
        errors.append(f"Model '{model_id}' is deprecated and should not be used")
    elif model_info.status == "experimental":
        warnings.append(f"Model '{model_id}' is experimental - results may vary")
    elif model_info.status == "preview":
        warnings.append(f"Model '{model_id}' is in preview - API may change")
    
    # Check thinking support
    if require_thinking and not model_info.supports_thinking:
        errors.append(f"Model '{model_id}' does not support thinking/reasoning")
    
    # Validate effort level
    if effort is not None:
        if effort.lower() not in ("none", "low", "medium", "high"):
            errors.append(f"Invalid effort level '{effort}' - must be none/low/medium/high")
        elif effort.lower() == "none" and model_info.supports_thinking and not model_info.can_disable_thinking:
            errors.append(f"Model '{model_id}' cannot disable thinking (min budget: {model_info.min_thinking_budget})")
    
    return ModelValidationResult(
        valid=len(errors) == 0,
        model_id=model_id,
        model_info=model_info,
        errors=errors,
        warnings=warnings,
    )


def get_model_summary_table() -> str:
    """
    Generate a markdown table summarizing all approved models.
    
    Returns:
        Markdown table string
    """
    models = get_approved_models()
    
    lines = [
        "| Model ID | Family | Status | Thinking | Budget Range | Cost |",
        "|----------|--------|--------|----------|--------------|------|",
    ]
    
    for m in models:
        thinking = "✓" if m.supports_thinking else "✗"
        if m.supports_thinking:
            if m.thinking_config_type == "level":
                budget = "LOW/HIGH"
            else:
                budget = f"{m.min_thinking_budget}-{m.max_thinking_budget}"
        else:
            budget = "N/A"
        
        lines.append(f"| {m.id} | {m.family} | {m.status} | {thinking} | {budget} | {m.cost_tier} |")
    
    return "\n".join(lines)
