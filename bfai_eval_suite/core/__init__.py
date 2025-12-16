"""
BFAI Eval Suite Core Module.

Provides model registry integration, pre-flight checks, metrics, and reporting.
"""

from .models import (
    get_model,
    get_approved_models,
    get_models_with_thinking,
    get_thinking_config,
    validate_model,
    ModelInfo,
)

from .preflight import (
    run_preflight_checks,
    PreflightResult,
    PreflightCheck,
)

from .report import (
    generate_report,
    ReportConfig,
)

__all__ = [
    # Models
    "get_model",
    "get_approved_models", 
    "get_models_with_thinking",
    "get_thinking_config",
    "validate_model",
    "ModelInfo",
    # Preflight
    "run_preflight_checks",
    "PreflightResult",
    "PreflightCheck",
    # Report
    "generate_report",
    "ReportConfig",
]
