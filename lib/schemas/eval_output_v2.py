"""
Evaluation Output Schema v2.0

Comprehensive schema for RAG evaluation outputs with full metadata
for reproducibility and troubleshooting.

Schema Version History:
- v1.0: Original schema (implicit)
- v1.1: Added retry_stats, errors, skipped sections
- v2.0: Comprehensive metadata overhaul - git info, config hashes,
        full model configs, per-question details
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


SCHEMA_VERSION = "2.0"
SCHEMA_VERSION_MAJOR = 2
SCHEMA_VERSION_MINOR = 0


@dataclass
class EnvironmentInfo:
    """System and environment context for the run."""
    git_commit: str
    git_branch: str
    git_dirty: bool
    config_file: str
    config_hash: str
    hostname: str
    user: str
    os_name: str
    os_version: str
    python_version: str
    timezone: str
    start_time_utc: str
    end_time_utc: Optional[str] = None
    ragas_version: str = "1.0.0"


@dataclass
class GeneratorConfig:
    """Generator model configuration."""
    model: str
    model_version: Optional[str] = None
    api_endpoint: Optional[str] = None
    temperature: float = 0.0
    seed: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: int = 8192
    reasoning_effort: str = "low"  # low | high (no medium in Gemini 3)
    safety_settings: str = "default"
    system_instruction: Optional[str] = None


@dataclass
class JudgeConfig:
    """Judge model configuration."""
    model: str
    model_version: Optional[str] = None
    temperature: float = 0.0
    seed: Optional[int] = None
    prompt_template_version: str = "v1.0"
    prompt_template_hash: Optional[str] = None


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    index_job_id: str
    embedding_model: str
    embedding_dimension: int
    embedding_model_version: Optional[str] = None
    recall_k: int = 100
    precision_k: int = 25
    hybrid_enabled: bool = True
    hybrid_alpha: float = 0.5
    reranking_enabled: bool = True
    reranking_model: str = "semantic-ranker-default@latest"
    reranking_top_k: Optional[int] = None


@dataclass
class CorpusInfo:
    """Corpus metadata."""
    file_path: str
    file_hash: str
    question_count: int
    sample_size: Optional[int] = None
    distribution: Optional[Dict[str, int]] = None


@dataclass
class ExecutionInfo:
    """Execution context."""
    mode: str  # "local" | "cloud"
    run_type: str  # "checkpoint" | "run" | "experiment"
    workers: int
    timeout_per_question_s: int = 120
    max_retries: int = 5
    checkpoint_interval: int = 10
    cloud_endpoint: Optional[str] = None
    cloud_revision: Optional[str] = None
    cloud_region: Optional[str] = None


@dataclass
class TimingInfo:
    """Timing breakdown for a question."""
    retrieval_s: float = 0.0
    rerank_s: float = 0.0
    generation_s: float = 0.0
    judge_s: float = 0.0
    total_s: float = 0.0
    # Cloud-specific
    cloud_retrieve_s: Optional[float] = None
    cloud_query_s: Optional[float] = None


@dataclass
class TokenInfo:
    """Token usage for a question."""
    prompt: int = 0
    completion: int = 0
    thinking: int = 0
    cached: int = 0
    total: int = 0


@dataclass
class JudgmentInfo:
    """LLM judge evaluation scores."""
    correctness: int = 0
    completeness: int = 0
    faithfulness: int = 0
    relevance: int = 0
    clarity: int = 0
    overall_score: int = 0
    verdict: str = "unknown"  # "pass" | "partial" | "fail"
    parse_error: Optional[str] = None


@dataclass
class LLMMetadata:
    """LLM response metadata."""
    model: str
    model_version: Optional[str] = None
    finish_reason: Optional[str] = None
    reasoning_effort: Optional[str] = None
    used_fallback: bool = False
    avg_logprobs: Optional[float] = None
    response_id: Optional[str] = None
    temperature: float = 0.0
    has_citations: bool = False


@dataclass
class RetryInfo:
    """Retry tracking for a question."""
    attempts: int = 1
    recovered: bool = False
    error: Optional[str] = None


@dataclass
class QuestionResult:
    """Complete result for a single question."""
    # Identity
    question_id: str
    question_type: str  # "single_hop" | "multi_hop"
    difficulty: str  # "easy" | "medium" | "hard"
    
    # Input (for debugging)
    question_text: str
    expected_answer: str
    source_documents: List[str]
    
    # Output
    generated_answer: str
    retrieved_doc_names: List[str]
    context_char_count: int
    
    # Evaluation
    recall_hit: bool
    mrr: float
    judgment: JudgmentInfo
    
    # Performance
    timing: TimingInfo
    tokens: TokenInfo
    llm_metadata: LLMMetadata
    
    # Metadata
    request_timestamp_utc: str
    answer_length: int
    retrieval_candidates: int
    retry_info: RetryInfo
    
    # Error tracking
    error: Optional[str] = None
    error_phase: Optional[str] = None


@dataclass
class MetricsAggregate:
    """Aggregated metrics across all questions."""
    precision_k: int
    total: int
    completed: int
    recall_at_100: float
    mrr: float
    pass_rate: float
    partial_rate: float
    fail_rate: float
    acceptable_rate: float
    correctness_avg: float = 0.0
    completeness_avg: float = 0.0
    faithfulness_avg: float = 0.0
    relevance_avg: float = 0.0
    clarity_avg: float = 0.0
    overall_score_avg: float = 0.0


@dataclass
class LatencyAggregate:
    """Aggregated latency statistics."""
    total_avg_s: float
    total_min_s: float
    total_max_s: float
    by_phase: Dict[str, float]


@dataclass
class TokenAggregate:
    """Aggregated token statistics."""
    prompt_total: int
    completion_total: int
    thinking_total: int
    cached_total: int
    total: int


@dataclass
class AnswerStats:
    """Answer length statistics."""
    avg_length_chars: float
    min_length_chars: int
    max_length_chars: int


@dataclass
class QualityInfo:
    """Quality metrics."""
    finish_reason_distribution: Dict[str, int]
    fallback_rate: float


@dataclass
class RetryStats:
    """Aggregate retry statistics."""
    total_questions: int
    succeeded_first_try: int
    succeeded_after_retry: int
    failed_all_retries: int
    total_retry_attempts: int
    avg_attempts: float


@dataclass
class ErrorStats:
    """Aggregate error statistics."""
    total_errors: int
    by_phase: Dict[str, int]
    error_messages: List[str]


@dataclass
class SkippedStats:
    """Skipped question statistics."""
    count: int
    reasons: Dict[str, int]
    question_ids: List[str]


@dataclass
class BreakdownStats:
    """Breakdown by category."""
    total: int
    pass_count: int
    partial: int
    fail: int
    pass_rate: float


@dataclass
class EvalOutput:
    """
    Complete evaluation output schema v2.0
    
    This is the top-level structure for all evaluation outputs.
    All fields are required unless marked Optional.
    """
    # Schema version for forward compatibility
    schema_version: str
    
    # Run identification
    run_id: str
    run_type: str  # "checkpoint" | "run" | "experiment"
    client: str
    timestamp: str
    
    # Comprehensive configuration
    environment: EnvironmentInfo
    generator_config: GeneratorConfig
    judge_config: JudgeConfig
    retrieval_config: RetrievalConfig
    corpus: CorpusInfo
    execution: ExecutionInfo
    
    # Aggregated results
    metrics: MetricsAggregate
    latency: LatencyAggregate
    tokens: TokenAggregate
    answer_stats: AnswerStats
    quality: QualityInfo
    retry_stats: RetryStats
    errors: ErrorStats
    skipped: SkippedStats
    
    # Breakdowns
    breakdown_by_type: Dict[str, BreakdownStats]
    breakdown_by_difficulty: Dict[str, BreakdownStats]
    
    # Per-question results
    results: List[QuestionResult]
    
    # Optional notes
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def get_schema_version(cls) -> str:
        """Get the current schema version."""
        return SCHEMA_VERSION


def validate_eval_output(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate evaluation output data against schema v2.0.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check schema version
    schema_version = data.get("schema_version", "")
    if not schema_version:
        errors.append("Missing required field: schema_version")
    elif not schema_version.startswith("2."):
        errors.append(f"Schema version mismatch: expected 2.x, got {schema_version}")
    
    # Required top-level fields
    required_fields = [
        "run_id", "run_type", "client", "timestamp",
        "environment", "generator_config", "judge_config",
        "retrieval_config", "corpus", "execution",
        "metrics", "latency", "tokens", "answer_stats",
        "quality", "retry_stats", "errors", "skipped",
        "breakdown_by_type", "breakdown_by_difficulty", "results"
    ]
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate run_type
    if data.get("run_type") not in ["checkpoint", "run", "experiment"]:
        errors.append(f"Invalid run_type: {data.get('run_type')}. Must be checkpoint|run|experiment")
    
    # Validate environment
    env = data.get("environment", {})
    env_required = ["git_commit", "git_branch", "config_file", "config_hash", 
                    "hostname", "start_time_utc"]
    for field in env_required:
        if field not in env:
            errors.append(f"Missing required environment field: {field}")
    
    # Validate generator_config
    gen = data.get("generator_config", {})
    if "model" not in gen:
        errors.append("Missing required generator_config field: model")
    if gen.get("temperature") is None:
        errors.append("Missing required generator_config field: temperature")
    
    # Validate judge_config
    judge = data.get("judge_config", {})
    if "model" not in judge:
        errors.append("Missing required judge_config field: model")
    
    # Validate retrieval_config
    ret = data.get("retrieval_config", {})
    ret_required = ["index_job_id", "embedding_model", "embedding_dimension", 
                    "recall_k", "precision_k"]
    for field in ret_required:
        if field not in ret:
            errors.append(f"Missing required retrieval_config field: {field}")
    
    # Validate corpus
    corpus = data.get("corpus", {})
    corpus_required = ["file_path", "file_hash", "question_count"]
    for field in corpus_required:
        if field not in corpus:
            errors.append(f"Missing required corpus field: {field}")
    
    # Validate execution
    exec_info = data.get("execution", {})
    if "mode" not in exec_info:
        errors.append("Missing required execution field: mode")
    if exec_info.get("mode") not in ["local", "cloud"]:
        errors.append(f"Invalid execution mode: {exec_info.get('mode')}. Must be local|cloud")
    
    # Validate metrics
    metrics = data.get("metrics", {})
    metrics_required = ["precision_k", "total", "completed", "recall_at_100", 
                        "mrr", "pass_rate", "partial_rate", "fail_rate"]
    for field in metrics_required:
        if field not in metrics:
            errors.append(f"Missing required metrics field: {field}")
    
    # Validate results array
    results = data.get("results", [])
    if not isinstance(results, list):
        errors.append("results must be a list")
    elif len(results) > 0:
        # Validate first result as sample
        first = results[0]
        result_required = ["question_id", "question_type", "difficulty"]
        for field in result_required:
            if field not in first:
                errors.append(f"Missing required result field: {field}")
    
    return len(errors) == 0, errors


def get_empty_question_result(question_id: str) -> Dict[str, Any]:
    """Get a template for a question result with all fields."""
    return {
        "question_id": question_id,
        "question_type": "",
        "difficulty": "",
        "question_text": "",
        "expected_answer": "",
        "source_documents": [],
        "generated_answer": "",
        "retrieved_doc_names": [],
        "context_char_count": 0,
        "recall_hit": False,
        "mrr": 0.0,
        "judgment": {
            "correctness": 0,
            "completeness": 0,
            "faithfulness": 0,
            "relevance": 0,
            "clarity": 0,
            "overall_score": 0,
            "verdict": "unknown",
            "parse_error": None
        },
        "timing": {
            "retrieval_s": 0.0,
            "rerank_s": 0.0,
            "generation_s": 0.0,
            "judge_s": 0.0,
            "total_s": 0.0
        },
        "tokens": {
            "prompt": 0,
            "completion": 0,
            "thinking": 0,
            "cached": 0,
            "total": 0
        },
        "llm_metadata": {
            "model": "",
            "model_version": None,
            "finish_reason": None,
            "reasoning_effort": None,
            "used_fallback": False,
            "avg_logprobs": None,
            "response_id": None,
            "temperature": 0.0,
            "has_citations": False
        },
        "request_timestamp_utc": "",
        "answer_length": 0,
        "retrieval_candidates": 0,
        "retry_info": {
            "attempts": 1,
            "recovered": False,
            "error": None
        },
        "error": None,
        "error_phase": None
    }
