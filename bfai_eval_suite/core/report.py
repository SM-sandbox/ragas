"""
Standardized Report Generation for BFAI Eval Suite.

Generates comprehensive markdown reports with all metrics, timing, and configuration.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .models import ModelInfo, get_model
from .preflight import PreflightResult


@dataclass
class RetrievalMetrics:
    """Retrieval performance metrics."""
    recall_at_100: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_15: float = 0.0
    precision_at_20: float = 0.0
    precision_at_25: float = 0.0
    mrr_at_10: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "recall_at_100": self.recall_at_100,
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "precision_at_15": self.precision_at_15,
            "precision_at_20": self.precision_at_20,
            "precision_at_25": self.precision_at_25,
            "mrr_at_10": self.mrr_at_10,
        }


@dataclass
class TimingMetrics:
    """Timing metrics for a phase."""
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    total: float = 0.0
    count: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "avg": self.avg,
            "min": self.min,
            "max": self.max,
            "total": self.total,
            "count": self.count,
        }


@dataclass
class TokenMetrics:
    """Token usage metrics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_thinking_tokens: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "thinking_tokens": self.thinking_tokens,
            "total_tokens": self.total_tokens,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_thinking_tokens": self.avg_thinking_tokens,
        }


@dataclass
class AnswerLengthMetrics:
    """Answer length metrics in characters."""
    avg_chars: float = 0.0
    min_chars: int = 0
    max_chars: int = 0
    total_chars: int = 0
    avg_words: float = 0.0
    min_words: int = 0
    max_words: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_chars": self.avg_chars,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "total_chars": self.total_chars,
            "avg_words": self.avg_words,
            "min_words": self.min_words,
            "max_words": self.max_words,
        }


@dataclass
class JudgeMetrics:
    """LLM Judge evaluation metrics."""
    pass_count: int = 0
    partial_count: int = 0
    fail_count: int = 0
    error_count: int = 0
    total_count: int = 0
    pass_rate: float = 0.0
    correctness: float = 0.0
    completeness: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    clarity: float = 0.0
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdicts": {
                "pass": self.pass_count,
                "partial": self.partial_count,
                "fail": self.fail_count,
                "error": self.error_count,
                "total": self.total_count,
            },
            "pass_rate": self.pass_rate,
            "scores": {
                "correctness": self.correctness,
                "completeness": self.completeness,
                "faithfulness": self.faithfulness,
                "relevance": self.relevance,
                "clarity": self.clarity,
                "overall_score": self.overall_score,
            },
        }


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    name: str
    model_id: str
    temperature: float = 0.0
    context_size: int = 10
    thinking_effort: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking_level: Optional[str] = None
    embedding_model: str = ""
    job_id: str = ""
    corpus_name: str = ""
    corpus_count: int = 0
    recall_top_k: int = 100
    reranker_model: str = "semantic-ranker-default@latest"
    judge_model: str = "gemini-2.5-flash"
    judge_temperature: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "context_size": self.context_size,
            "thinking_effort": self.thinking_effort,
            "thinking_budget": self.thinking_budget,
            "thinking_level": self.thinking_level,
            "embedding_model": self.embedding_model,
            "job_id": self.job_id,
            "corpus_name": self.corpus_name,
            "corpus_count": self.corpus_count,
            "recall_top_k": self.recall_top_k,
            "reranker_model": self.reranker_model,
            "judge_model": self.judge_model,
            "judge_temperature": self.judge_temperature,
        }


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: str = "reports"
    include_preflight: bool = True
    include_detailed_results: bool = False
    format: str = "markdown"  # markdown, json, or both


@dataclass
class ExperimentReport:
    """Complete experiment report data."""
    config: ExperimentConfig
    model_info: Optional[ModelInfo] = None
    preflight: Optional[PreflightResult] = None
    retrieval_metrics: Optional[RetrievalMetrics] = None
    timing: Dict[str, TimingMetrics] = field(default_factory=dict)
    tokens: Optional[TokenMetrics] = None
    answer_length: Optional[AnswerLengthMetrics] = None
    judge_metrics: Optional[JudgeMetrics] = None
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "config": self.config.to_dict(),
            "model_info": self.model_info.to_dict() if self.model_info else None,
            "preflight": self.preflight.to_dict() if self.preflight else None,
            "retrieval_metrics": self.retrieval_metrics.to_dict() if self.retrieval_metrics else None,
            "timing": {k: v.to_dict() for k, v in self.timing.items()},
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "answer_length": self.answer_length.to_dict() if self.answer_length else None,
            "judge_metrics": self.judge_metrics.to_dict() if self.judge_metrics else None,
            "detailed_results_count": len(self.detailed_results),
        }


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    - Under 60s: "45.2s"
    - 1-60 min: "2m 30s"
    - Over 60 min: "2h 8m 30s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def generate_report(report: ExperimentReport, config: Optional[ReportConfig] = None) -> str:
    """
    Generate a comprehensive markdown report.
    
    Args:
        report: ExperimentReport with all data
        config: Optional ReportConfig for output settings
        
    Returns:
        Markdown string
    """
    if config is None:
        config = ReportConfig()
    
    lines = []
    exp = report.config
    
    # Title
    lines.append(f"# {exp.name} Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Duration:** {format_duration(report.duration_seconds)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 1. Configuration
    lines.append("## 1. Configuration")
    lines.append("")
    lines.append("### Experiment Settings")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Experiment Name | {exp.name} |")
    lines.append(f"| Model | {exp.model_id} |")
    lines.append(f"| Temperature | {exp.temperature} |")
    lines.append(f"| Context Size | {exp.context_size} chunks |")
    
    if exp.thinking_effort:
        lines.append(f"| Thinking Effort | {exp.thinking_effort} |")
    if exp.thinking_budget is not None:
        lines.append(f"| Thinking Budget | {exp.thinking_budget} tokens |")
    if exp.thinking_level:
        lines.append(f"| Thinking Level | {exp.thinking_level} |")
    
    lines.append(f"| Embedding Model | {exp.embedding_model or 'N/A'} |")
    lines.append(f"| Job ID | {exp.job_id or 'N/A'} |")
    lines.append(f"| Corpus | {exp.corpus_name} ({exp.corpus_count} questions) |")
    lines.append(f"| Recall Top-K | {exp.recall_top_k} |")
    lines.append(f"| Reranker | {exp.reranker_model} |")
    lines.append(f"| Judge Model | {exp.judge_model} (temp {exp.judge_temperature}) |")
    lines.append("")
    
    # Model Info
    if report.model_info:
        m = report.model_info
        lines.append("### Model Details")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Model ID | {m.id} |")
        lines.append(f"| Display Name | {m.name} |")
        lines.append(f"| Family | {m.family} |")
        lines.append(f"| Version | {m.version} |")
        lines.append(f"| Status | {m.status} |")
        lines.append(f"| Cost Tier | {m.cost_tier} |")
        lines.append(f"| Supports Thinking | {'Yes' if m.supports_thinking else 'No'} |")
        if m.supports_thinking:
            lines.append(f"| Thinking Config Type | {m.thinking_config_type} |")
            lines.append(f"| Can Disable Thinking | {'Yes' if m.can_disable_thinking else 'No'} |")
            lines.append(f"| Thinking Budget Range | {m.min_thinking_budget} - {m.max_thinking_budget} |")
        lines.append(f"| Max Input Tokens | {m.max_input_tokens:,} |")
        lines.append(f"| Max Output Tokens | {m.max_output_tokens:,} |")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # 2. Pre-flight Results
    if config.include_preflight and report.preflight:
        lines.append(report.preflight.to_markdown())
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 3. Retrieval Metrics
    if report.retrieval_metrics:
        rm = report.retrieval_metrics
        lines.append("## 3. Retrieval Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Recall@100 | {rm.recall_at_100*100:.1f}% |")
        lines.append(f"| Precision@5 | {rm.precision_at_5*100:.1f}% |")
        lines.append(f"| Precision@10 | {rm.precision_at_10*100:.1f}% |")
        lines.append(f"| Precision@15 | {rm.precision_at_15*100:.1f}% |")
        lines.append(f"| Precision@20 | {rm.precision_at_20*100:.1f}% |")
        lines.append(f"| Precision@25 | {rm.precision_at_25*100:.1f}% |")
        lines.append(f"| MRR@10 | {rm.mrr_at_10:.3f} |")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 4. LLM Judge Results
    if report.judge_metrics:
        jm = report.judge_metrics
        lines.append("## 4. LLM Judge Results")
        lines.append("")
        lines.append("### Verdict Distribution")
        lines.append("")
        lines.append("| Verdict | Count | Percentage |")
        lines.append("|---------|-------|------------|")
        total = jm.total_count or 1
        lines.append(f"| ✅ Pass | {jm.pass_count} | {jm.pass_count/total*100:.1f}% |")
        lines.append(f"| ⚠️ Partial | {jm.partial_count} | {jm.partial_count/total*100:.1f}% |")
        lines.append(f"| ❌ Fail | {jm.fail_count} | {jm.fail_count/total*100:.1f}% |")
        lines.append(f"| 🔴 Error | {jm.error_count} | {jm.error_count/total*100:.1f}% |")
        lines.append("")
        lines.append(f"**Pass Rate: {jm.pass_rate*100:.1f}%**")
        lines.append("")
        lines.append("### Quality Scores (1-5 scale)")
        lines.append("")
        lines.append("| Dimension | Score |")
        lines.append("|-----------|-------|")
        lines.append(f"| **Overall Score** | **{jm.overall_score:.2f}** |")
        lines.append(f"| Correctness | {jm.correctness:.2f} |")
        lines.append(f"| Completeness | {jm.completeness:.2f} |")
        lines.append(f"| Faithfulness | {jm.faithfulness:.2f} |")
        lines.append(f"| Relevance | {jm.relevance:.2f} |")
        lines.append(f"| Clarity | {jm.clarity:.2f} |")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 5. Answer Length
    if report.answer_length:
        al = report.answer_length
        lines.append("## 5. Answer Length")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Avg Characters | {al.avg_chars:,.0f} |")
        lines.append(f"| Min Characters | {al.min_chars:,} |")
        lines.append(f"| Max Characters | {al.max_chars:,} |")
        lines.append(f"| Avg Words | {al.avg_words:,.0f} |")
        lines.append(f"| Min Words | {al.min_words:,} |")
        lines.append(f"| Max Words | {al.max_words:,} |")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 6. Token Usage
    if report.tokens:
        tk = report.tokens
        lines.append("## 6. Token Usage")
        lines.append("")
        lines.append("| Metric | Total | Avg/Query |")
        lines.append("|--------|-------|-----------|")
        lines.append(f"| Prompt Tokens | {tk.prompt_tokens:,} | {tk.avg_prompt_tokens:,.0f} |")
        lines.append(f"| Completion Tokens | {tk.completion_tokens:,} | {tk.avg_completion_tokens:,.0f} |")
        if tk.thinking_tokens > 0:
            lines.append(f"| Thinking/Reasoning Tokens | {tk.thinking_tokens:,} | {tk.avg_thinking_tokens:,.0f} |")
        lines.append(f"| **Total** | **{tk.total_tokens:,}** | |")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 7. Timing Breakdown
    if report.timing:
        lines.append("## 7. Timing Breakdown")
        lines.append("")
        lines.append("| Phase | Avg | Min | Max | Total |")
        lines.append("|-------|-----|-----|-----|-------|")
        
        phase_order = ["retrieval", "reranking", "generation", "judge", "total"]
        for phase in phase_order:
            if phase in report.timing:
                t = report.timing[phase]
                lines.append(f"| {phase.title()} | {t.avg:.1f}s | {t.min:.1f}s | {t.max:.1f}s | {format_duration(t.total)} |")
        
        # Add any other phases not in the standard order
        for phase, t in report.timing.items():
            if phase not in phase_order:
                lines.append(f"| {phase.title()} | {t.avg:.1f}s | {t.min:.1f}s | {t.max:.1f}s | {format_duration(t.total)} |")
        
        lines.append("")
        
        # Questions per second
        if "total" in report.timing and report.timing["total"].count > 0:
            qps = report.timing["total"].count / report.timing["total"].total if report.timing["total"].total > 0 else 0
            lines.append(f"**Throughput:** {qps:.2f} questions/second")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # 8. Summary
    lines.append("## 8. Summary")
    lines.append("")
    
    if report.judge_metrics:
        jm = report.judge_metrics
        lines.append(f"- **Pass Rate:** {jm.pass_rate*100:.1f}%")
        lines.append(f"- **Overall Score:** {jm.overall_score:.2f}/5")
    
    if "generation" in report.timing:
        lines.append(f"- **Avg Generation Time:** {report.timing['generation'].avg:.2f}s")
    
    if report.tokens:
        lines.append(f"- **Total Tokens Used:** {report.tokens.total_tokens:,}")
    
    lines.append(f"- **Total Duration:** {format_duration(report.duration_seconds)}")
    lines.append("")
    
    return "\n".join(lines)


def save_report(
    report: ExperimentReport,
    output_dir: str,
    filename: str,
    config: Optional[ReportConfig] = None,
) -> Dict[str, str]:
    """
    Save report to files.
    
    Args:
        report: ExperimentReport with all data
        output_dir: Directory to save reports
        filename: Base filename (without extension)
        config: Optional ReportConfig
        
    Returns:
        Dict with paths to saved files
    """
    if config is None:
        config = ReportConfig()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved = {}
    
    # Save markdown
    if config.format in ("markdown", "both"):
        md_path = output_path / f"{filename}.md"
        md_content = generate_report(report, config)
        with open(md_path, "w") as f:
            f.write(md_content)
        saved["markdown"] = str(md_path)
    
    # Save JSON
    if config.format in ("json", "both"):
        json_path = output_path / f"{filename}.json"
        json_data = report.to_dict()
        if config.include_detailed_results:
            json_data["detailed_results"] = report.detailed_results
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        saved["json"] = str(json_path)
    
    return saved


def create_report_from_results(
    experiment_name: str,
    model_id: str,
    results: List[Dict[str, Any]],
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentReport:
    """
    Create an ExperimentReport from raw experiment results.
    
    Args:
        experiment_name: Name of the experiment
        model_id: Model ID used
        results: List of per-question result dictionaries
        config_overrides: Optional config overrides
        
    Returns:
        ExperimentReport ready for report generation
    """
    import statistics
    
    config_overrides = config_overrides or {}
    
    # Build config
    exp_config = ExperimentConfig(
        name=experiment_name,
        model_id=model_id,
        temperature=config_overrides.get("temperature", 0.0),
        context_size=config_overrides.get("context_size", 10),
        thinking_effort=config_overrides.get("thinking_effort"),
        thinking_budget=config_overrides.get("thinking_budget"),
        embedding_model=config_overrides.get("embedding_model", ""),
        job_id=config_overrides.get("job_id", ""),
        corpus_name=config_overrides.get("corpus_name", ""),
        corpus_count=len(results),
        recall_top_k=config_overrides.get("recall_top_k", 100),
        reranker_model=config_overrides.get("reranker_model", "semantic-ranker-default@latest"),
        judge_model=config_overrides.get("judge_model", "gemini-2.5-flash"),
        judge_temperature=config_overrides.get("judge_temperature", 0.0),
    )
    
    # Get model info
    model_info = get_model(model_id)
    
    # Calculate timing metrics
    timing = {}
    for phase in ["retrieval", "reranking", "generation", "judge"]:
        time_key = f"{phase}_time"
        times = [r.get(time_key, 0) for r in results if time_key in r]
        if times:
            timing[phase] = TimingMetrics(
                avg=statistics.mean(times),
                min=min(times),
                max=max(times),
                total=sum(times),
                count=len(times),
            )
    
    # Total timing
    total_times = []
    for r in results:
        t = sum(r.get(f"{p}_time", 0) for p in ["retrieval", "reranking", "generation", "judge"])
        total_times.append(t)
    if total_times:
        timing["total"] = TimingMetrics(
            avg=statistics.mean(total_times),
            min=min(total_times),
            max=max(total_times),
            total=sum(total_times),
            count=len(total_times),
        )
    
    # Calculate judge metrics
    verdicts = [r.get("judgment", {}).get("verdict", "error") for r in results]
    judge_metrics = JudgeMetrics(
        pass_count=verdicts.count("pass"),
        partial_count=verdicts.count("partial"),
        fail_count=verdicts.count("fail"),
        error_count=verdicts.count("error"),
        total_count=len(verdicts),
        pass_rate=verdicts.count("pass") / len(verdicts) if verdicts else 0,
    )
    
    # Calculate quality scores
    valid_results = [r for r in results if r.get("judgment", {}).get("verdict") != "error"]
    for score_key in ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]:
        scores = [r.get("judgment", {}).get(score_key, 0) for r in valid_results 
                  if isinstance(r.get("judgment", {}).get(score_key), (int, float)) and r.get("judgment", {}).get(score_key) > 0]
        if scores:
            setattr(judge_metrics, score_key, statistics.mean(scores))
    
    # Token metrics (if available)
    tokens = None
    prompt_tokens = [r.get("prompt_tokens", 0) for r in results if r.get("prompt_tokens")]
    completion_tokens = [r.get("completion_tokens", 0) for r in results if r.get("completion_tokens")]
    thinking_tokens = [r.get("thinking_tokens", 0) for r in results if r.get("thinking_tokens")]
    if prompt_tokens or completion_tokens:
        tokens = TokenMetrics(
            prompt_tokens=sum(prompt_tokens),
            completion_tokens=sum(completion_tokens),
            thinking_tokens=sum(thinking_tokens),
            total_tokens=sum(prompt_tokens) + sum(completion_tokens) + sum(thinking_tokens),
            avg_prompt_tokens=statistics.mean(prompt_tokens) if prompt_tokens else 0,
            avg_completion_tokens=statistics.mean(completion_tokens) if completion_tokens else 0,
            avg_thinking_tokens=statistics.mean(thinking_tokens) if thinking_tokens else 0,
        )
    
    # Answer length metrics
    answer_length = None
    answers = [r.get("answer", "") for r in results if r.get("answer")]
    if answers:
        char_lengths = [len(a) for a in answers]
        word_counts = [len(a.split()) for a in answers]
        answer_length = AnswerLengthMetrics(
            avg_chars=statistics.mean(char_lengths),
            min_chars=min(char_lengths),
            max_chars=max(char_lengths),
            total_chars=sum(char_lengths),
            avg_words=statistics.mean(word_counts),
            min_words=min(word_counts),
            max_words=max(word_counts),
        )
    
    return ExperimentReport(
        config=exp_config,
        model_info=model_info,
        timing=timing,
        tokens=tokens,
        answer_length=answer_length,
        judge_metrics=judge_metrics,
        detailed_results=results,
        duration_seconds=timing.get("total", TimingMetrics()).total,
    )
