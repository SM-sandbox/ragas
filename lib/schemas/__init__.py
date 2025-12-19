"""
Evaluation Output Schemas

This module defines versioned schemas for evaluation outputs.
All evaluation results should conform to these schemas for consistency
and to enable meaningful comparisons across runs.

Schema Versioning:
- Major version (2.x -> 3.x): Breaking changes, old data may not validate
- Minor version (2.0 -> 2.1): Additive changes, backward compatible

Current Schema: v2.0
"""

from lib.schemas.eval_output_v2 import (
    SCHEMA_VERSION,
    EvalOutput,
    EnvironmentInfo,
    GeneratorConfig,
    JudgeConfig,
    RetrievalConfig,
    CorpusInfo,
    ExecutionInfo,
    QuestionResult,
    validate_eval_output,
)

__all__ = [
    "SCHEMA_VERSION",
    "EvalOutput",
    "EnvironmentInfo",
    "GeneratorConfig",
    "JudgeConfig",
    "RetrievalConfig",
    "CorpusInfo",
    "ExecutionInfo",
    "QuestionResult",
    "validate_eval_output",
]
