"""
DEPRECATED: Use lib/ instead.

This module re-exports from lib/ for backward compatibility.
"""
import warnings
warnings.warn(
    "Importing from src/ is deprecated. Use lib/ instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from lib for compatibility
from lib.utils.models import (
    get_model,
    get_approved_models,
    get_models_with_thinking,
    get_thinking_config,
    validate_model,
    ModelInfo,
)

from lib.utils.preflight import (
    run_preflight_checks,
    PreflightResult,
    PreflightCheck,
)

from lib.utils.report import (
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
