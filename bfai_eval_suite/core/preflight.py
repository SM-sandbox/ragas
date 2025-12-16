"""
Pre-flight Check System for BFAI Eval Suite.

Runs comprehensive checks before experiments to ensure:
- GCP authentication is valid
- Orchestrator components are accessible
- Required resources exist
- Model configuration is valid
"""

import sys
import os
import json
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum

# Add orchestrator to path
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")


class CheckStatus(Enum):
    """Status of a pre-flight check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class PreflightCheck:
    """Result of a single pre-flight check."""
    name: str
    status: CheckStatus
    message: str
    details: Optional[str] = None
    duration_ms: float = 0.0
    
    @property
    def passed(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP)
    
    @property
    def icon(self) -> str:
        icons = {
            CheckStatus.PASS: "✅",
            CheckStatus.FAIL: "❌",
            CheckStatus.WARN: "⚠️",
            CheckStatus.SKIP: "⏭️",
        }
        return icons.get(self.status, "❓")


@dataclass
class PreflightResult:
    """Complete pre-flight check results."""
    checks: List[PreflightCheck] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_duration_ms: float = 0.0
    
    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)
    
    @property
    def critical_failures(self) -> List[PreflightCheck]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]
    
    @property
    def warnings(self) -> List[PreflightCheck]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]
    
    def to_markdown(self) -> str:
        """Generate markdown report of pre-flight checks."""
        lines = [
            "## Pre-flight Checks",
            "",
            f"**Timestamp:** {self.timestamp}",
            f"**Duration:** {self.total_duration_ms:.0f}ms",
            "",
            "| Check | Status | Message |",
            "|-------|--------|---------|",
        ]
        
        for check in self.checks:
            lines.append(f"| {check.name} | {check.icon} {check.status.value} | {check.message} |")
        
        if self.critical_failures:
            lines.extend([
                "",
                "### ❌ Critical Failures",
                "",
            ])
            for check in self.critical_failures:
                lines.append(f"- **{check.name}:** {check.message}")
                if check.details:
                    lines.append(f"  - Details: {check.details}")
        
        if self.warnings:
            lines.extend([
                "",
                "### ⚠️ Warnings",
                "",
            ])
            for check in self.warnings:
                lines.append(f"- **{check.name}:** {check.message}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_duration_ms": self.total_duration_ms,
            "all_passed": self.all_passed,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "duration_ms": c.duration_ms,
                }
                for c in self.checks
            ],
        }


def _timed_check(name: str, check_fn: Callable[[], PreflightCheck]) -> PreflightCheck:
    """Run a check function and time it."""
    import time
    start = time.time()
    try:
        result = check_fn()
        result.duration_ms = (time.time() - start) * 1000
        return result
    except Exception as e:
        return PreflightCheck(
            name=name,
            status=CheckStatus.FAIL,
            message=f"Check raised exception: {str(e)}",
            duration_ms=(time.time() - start) * 1000,
        )


def check_gcp_auth() -> PreflightCheck:
    """Check GCP authentication is valid."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return PreflightCheck(
                name="GCP Auth",
                status=CheckStatus.PASS,
                message="GCP authentication valid",
            )
        else:
            return PreflightCheck(
                name="GCP Auth",
                status=CheckStatus.FAIL,
                message="GCP authentication failed",
                details=result.stderr.strip() if result.stderr else "No access token returned",
            )
    except subprocess.TimeoutExpired:
        return PreflightCheck(
            name="GCP Auth",
            status=CheckStatus.FAIL,
            message="GCP auth check timed out",
        )
    except FileNotFoundError:
        return PreflightCheck(
            name="GCP Auth",
            status=CheckStatus.FAIL,
            message="gcloud CLI not found",
        )


def check_project_access(project_id: str = "civic-athlete-473921-c0") -> PreflightCheck:
    """Check access to GCP project."""
    try:
        result = subprocess.run(
            ["gcloud", "projects", "describe", project_id, "--format=json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return PreflightCheck(
                name="Project Access",
                status=CheckStatus.PASS,
                message=f"Access to project {project_id} confirmed",
            )
        else:
            return PreflightCheck(
                name="Project Access",
                status=CheckStatus.FAIL,
                message=f"Cannot access project {project_id}",
                details=result.stderr.strip(),
            )
    except Exception as e:
        return PreflightCheck(
            name="Project Access",
            status=CheckStatus.FAIL,
            message=f"Project access check failed: {str(e)}",
        )


def check_orchestrator_import() -> PreflightCheck:
    """Check orchestrator modules can be imported."""
    try:
        from services.api.core.approved_models import get_approved_models
        from services.api.retrieval.vector_search import VectorSearchRetriever
        from services.api.ranking.google_ranker import GoogleRanker
        from services.api.generation.gemini import GeminiAnswerGenerator
        from libs.core.gcp_config import get_jobs_config
        
        return PreflightCheck(
            name="Orchestrator Import",
            status=CheckStatus.PASS,
            message="All orchestrator modules imported successfully",
        )
    except ImportError as e:
        return PreflightCheck(
            name="Orchestrator Import",
            status=CheckStatus.FAIL,
            message="Failed to import orchestrator modules",
            details=str(e),
        )


def check_model_registry() -> PreflightCheck:
    """Check model registry is accessible and populated."""
    try:
        from services.api.core.approved_models import get_approved_models
        
        models = get_approved_models()
        if len(models) == 0:
            return PreflightCheck(
                name="Model Registry",
                status=CheckStatus.FAIL,
                message="Model registry is empty",
            )
        
        model_ids = [m.id for m in models]
        return PreflightCheck(
            name="Model Registry",
            status=CheckStatus.PASS,
            message=f"Found {len(models)} approved models",
            details=", ".join(model_ids[:5]) + ("..." if len(model_ids) > 5 else ""),
        )
    except Exception as e:
        return PreflightCheck(
            name="Model Registry",
            status=CheckStatus.FAIL,
            message=f"Model registry check failed: {str(e)}",
        )


def check_job_config(job_id: str) -> PreflightCheck:
    """Check job configuration exists."""
    try:
        from libs.core.gcp_config import get_jobs_config
        
        jobs = get_jobs_config()
        if job_id not in jobs:
            available = list(jobs.keys())[:5]
            return PreflightCheck(
                name="Job Config",
                status=CheckStatus.FAIL,
                message=f"Job '{job_id}' not found",
                details=f"Available jobs: {', '.join(available)}...",
            )
        
        job = jobs[job_id]
        return PreflightCheck(
            name="Job Config",
            status=CheckStatus.PASS,
            message=f"Job '{job_id}' found",
            details=f"Index: {job.get('index_name', 'N/A')}",
        )
    except Exception as e:
        return PreflightCheck(
            name="Job Config",
            status=CheckStatus.FAIL,
            message=f"Job config check failed: {str(e)}",
        )


def check_corpus_file(corpus_path: str) -> PreflightCheck:
    """Check corpus file exists and is valid JSON."""
    path = Path(corpus_path)
    
    if not path.exists():
        return PreflightCheck(
            name="Corpus File",
            status=CheckStatus.FAIL,
            message=f"Corpus file not found: {corpus_path}",
        )
    
    try:
        with open(path) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data.get("questions", data.get("items", [])))
        else:
            count = 0
        
        return PreflightCheck(
            name="Corpus File",
            status=CheckStatus.PASS,
            message=f"Corpus loaded: {count} items",
            details=str(path),
        )
    except json.JSONDecodeError as e:
        return PreflightCheck(
            name="Corpus File",
            status=CheckStatus.FAIL,
            message="Corpus file is not valid JSON",
            details=str(e),
        )


def check_model_valid(model_id: str, require_thinking: bool = False) -> PreflightCheck:
    """Check requested model is valid."""
    from .models import validate_model
    
    result = validate_model(model_id, require_thinking=require_thinking)
    
    if not result.valid:
        return PreflightCheck(
            name="Model Validation",
            status=CheckStatus.FAIL,
            message=f"Model '{model_id}' validation failed",
            details="; ".join(result.errors),
        )
    
    if result.warnings:
        return PreflightCheck(
            name="Model Validation",
            status=CheckStatus.WARN,
            message=f"Model '{model_id}' valid with warnings",
            details="; ".join(result.warnings),
        )
    
    return PreflightCheck(
        name="Model Validation",
        status=CheckStatus.PASS,
        message=f"Model '{model_id}' is valid",
        details=f"Family: {result.model_info.family}, Status: {result.model_info.status}",
    )


def check_api_connectivity() -> PreflightCheck:
    """Quick ping to Gemini API to verify connectivity."""
    try:
        from vertexai.generative_models import GenerativeModel
        
        model = GenerativeModel("gemini-2.0-flash")
        response = model.generate_content("Say 'OK' if you can hear me.", 
                                          generation_config={"max_output_tokens": 10})
        
        if response and response.text:
            return PreflightCheck(
                name="API Connectivity",
                status=CheckStatus.PASS,
                message="Gemini API responding",
            )
        else:
            return PreflightCheck(
                name="API Connectivity",
                status=CheckStatus.WARN,
                message="Gemini API returned empty response",
            )
    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "429" in error_str:
            return PreflightCheck(
                name="API Connectivity",
                status=CheckStatus.WARN,
                message="API rate limited (quota issue)",
                details=str(e)[:200],
            )
        return PreflightCheck(
            name="API Connectivity",
            status=CheckStatus.FAIL,
            message="Gemini API connectivity failed",
            details=str(e)[:200],
        )


def check_metrics_data(corpus_path: str, cache_path: Optional[str] = None) -> PreflightCheck:
    """
    Validate that corpus and retrieval cache have all required fields for metrics.
    
    This ensures we can compute precision, recall, MRR, etc.
    """
    from .metrics import validate_metrics_data
    
    # Load corpus
    try:
        with open(corpus_path) as f:
            corpus = json.load(f)
    except Exception as e:
        return PreflightCheck(
            name="Metrics Data",
            status=CheckStatus.FAIL,
            message=f"Cannot load corpus: {e}",
        )
    
    # Load cache if provided
    cache = None
    if cache_path:
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception as e:
            return PreflightCheck(
                name="Metrics Data",
                status=CheckStatus.WARN,
                message=f"Cannot load retrieval cache: {e}",
            )
    
    # Validate
    result = validate_metrics_data(corpus, cache)
    
    if not result.valid:
        return PreflightCheck(
            name="Metrics Data",
            status=CheckStatus.FAIL,
            message="Missing required fields for metrics",
            details="; ".join(result.errors),
        )
    
    if result.warnings:
        return PreflightCheck(
            name="Metrics Data",
            status=CheckStatus.WARN,
            message="Metrics data has warnings",
            details="; ".join(result.warnings),
        )
    
    # Build coverage summary
    coverage_items = []
    for field, pct in result.corpus_fields_present.items():
        if pct >= 0.99:
            coverage_items.append(f"{field}:✓")
    
    return PreflightCheck(
        name="Metrics Data",
        status=CheckStatus.PASS,
        message="All required fields present for metrics",
        details=f"Corpus fields: {', '.join(coverage_items[:5])}",
    )


@dataclass
class PreflightConfig:
    """Configuration for pre-flight checks."""
    job_id: str = "bfai__eval66a_g1_1536_tt"
    corpus_path: str = ""
    retrieval_cache_path: str = ""  # For metrics validation
    model_id: str = "gemini-2.5-flash"
    require_thinking: bool = False
    skip_api_check: bool = False
    skip_metrics_check: bool = False
    project_id: str = "civic-athlete-473921-c0"


def run_preflight_checks(config: Optional[PreflightConfig] = None) -> PreflightResult:
    """
    Run all pre-flight checks.
    
    Args:
        config: Optional configuration for checks
        
    Returns:
        PreflightResult with all check results
    """
    import time
    start = time.time()
    
    if config is None:
        config = PreflightConfig()
    
    result = PreflightResult()
    
    # Core checks (always run)
    result.checks.append(_timed_check("GCP Auth", check_gcp_auth))
    result.checks.append(_timed_check("Project Access", lambda: check_project_access(config.project_id)))
    result.checks.append(_timed_check("Orchestrator Import", check_orchestrator_import))
    result.checks.append(_timed_check("Model Registry", check_model_registry))
    
    # Job config check
    if config.job_id:
        result.checks.append(_timed_check("Job Config", lambda: check_job_config(config.job_id)))
    
    # Corpus file check
    if config.corpus_path:
        result.checks.append(_timed_check("Corpus File", lambda: check_corpus_file(config.corpus_path)))
    
    # Model validation
    if config.model_id:
        result.checks.append(_timed_check("Model Validation", 
                                          lambda: check_model_valid(config.model_id, config.require_thinking)))
    
    # Metrics data validation (ensures we can compute precision, recall, MRR)
    if config.corpus_path and not config.skip_metrics_check:
        result.checks.append(_timed_check("Metrics Data", 
                                          lambda: check_metrics_data(config.corpus_path, config.retrieval_cache_path)))
    elif config.skip_metrics_check:
        result.checks.append(PreflightCheck(
            name="Metrics Data",
            status=CheckStatus.SKIP,
            message="Skipped (skip_metrics_check=True)",
        ))
    
    # API connectivity (optional, can be slow)
    if not config.skip_api_check:
        result.checks.append(_timed_check("API Connectivity", check_api_connectivity))
    else:
        result.checks.append(PreflightCheck(
            name="API Connectivity",
            status=CheckStatus.SKIP,
            message="Skipped (skip_api_check=True)",
        ))
    
    result.total_duration_ms = (time.time() - start) * 1000
    
    return result


def print_preflight_summary(result: PreflightResult) -> None:
    """Print a summary of pre-flight checks to console."""
    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECKS")
    print("=" * 60)
    
    for check in result.checks:
        status_str = f"{check.icon} {check.status.value.upper():6}"
        print(f"{status_str} | {check.name:20} | {check.message}")
    
    print("-" * 60)
    
    if result.all_passed:
        print(f"✅ All checks passed ({result.total_duration_ms:.0f}ms)")
    else:
        failures = len(result.critical_failures)
        print(f"❌ {failures} critical failure(s) - cannot proceed")
        for check in result.critical_failures:
            print(f"   - {check.name}: {check.message}")
    
    print("=" * 60 + "\n")
