"""
Metadata Collection Helpers

Functions for capturing comprehensive metadata about the execution environment,
git state, configuration, and system information for evaluation runs.
"""

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any


def get_git_info(repo_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get git repository information.
    
    Returns:
        Dict with git_commit, git_branch, git_dirty, git_tag
    """
    if repo_path is None:
        repo_path = Path(__file__).parent.parent.parent
    
    result = {
        "git_commit": "unknown",
        "git_branch": "unknown",
        "git_dirty": False,
        "git_tag": None,
    }
    
    try:
        # Get commit hash
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if commit.returncode == 0:
            result["git_commit"] = commit.stdout.strip()[:12]  # Short hash
        
        # Get branch name
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if branch.returncode == 0:
            result["git_branch"] = branch.stdout.strip()
        
        # Check if dirty (uncommitted changes)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if status.returncode == 0:
            result["git_dirty"] = len(status.stdout.strip()) > 0
        
        # Get tag if on a tag
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if tag.returncode == 0:
            result["git_tag"] = tag.stdout.strip()
            
    except Exception:
        pass  # Return defaults if git commands fail
    
    return result


def get_system_info() -> Dict[str, str]:
    """
    Get system and environment information.
    
    Returns:
        Dict with hostname, user, os_name, os_version, python_version, timezone
    """
    return {
        "hostname": platform.node(),
        "user": os.getenv("USER", os.getenv("USERNAME", "unknown")),
        "os_name": platform.system(),
        "os_version": platform.release(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timezone": datetime.now().astimezone().tzname() or "UTC",
    }


def get_file_hash(file_path: Path) -> str:
    """
    Calculate SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        First 16 characters of SHA256 hash, or "not_found" if file doesn't exist
    """
    if not file_path.exists():
        return "not_found"
    
    try:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
    except Exception:
        return "error"


def get_config_hash(config_path: Path) -> str:
    """
    Get hash of a configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        SHA256 hash (first 16 chars) of the config file
    """
    return get_file_hash(config_path)


def get_utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.
    
    Returns:
        ISO format timestamp with Z suffix (e.g., "2025-12-19T14:30:00Z")
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_local_timestamp() -> str:
    """
    Get current local timestamp in ISO format.
    
    Returns:
        ISO format timestamp (e.g., "2025-12-19T09:30:00-05:00")
    """
    return datetime.now().astimezone().isoformat()


def build_environment_info(
    config_file: str,
    config_path: Path,
    start_time_utc: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build complete environment info dictionary.
    
    Args:
        config_file: Name of config file used (e.g., "checkpoint_config.yaml")
        config_path: Full path to config file
        start_time_utc: Optional start time (defaults to now)
        
    Returns:
        Complete environment info dict matching EnvironmentInfo schema
    """
    git_info = get_git_info()
    sys_info = get_system_info()
    
    return {
        "git_commit": git_info["git_commit"],
        "git_branch": git_info["git_branch"],
        "git_dirty": git_info["git_dirty"],
        "config_file": config_file,
        "config_hash": get_config_hash(config_path),
        "hostname": sys_info["hostname"],
        "user": sys_info["user"],
        "os_name": sys_info["os_name"],
        "os_version": sys_info["os_version"],
        "python_version": sys_info["python_version"],
        "timezone": sys_info["timezone"],
        "start_time_utc": start_time_utc or get_utc_timestamp(),
        "end_time_utc": None,
        "ragas_version": "1.0.0",
    }


def build_generator_config(
    model: str,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
    max_output_tokens: int = 8192,
    seed: Optional[int] = None,
    model_version: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build generator config dictionary.
    
    Returns:
        Complete generator config dict matching GeneratorConfig schema
    """
    return {
        "model": model,
        "model_version": model_version,
        "api_endpoint": api_endpoint or "generativelanguage.googleapis.com",
        "temperature": temperature,
        "seed": seed,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
        "reasoning_effort": reasoning_effort,
        "safety_settings": "default",
        "system_instruction": None,
    }


def build_judge_config(
    model: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    model_version: Optional[str] = None,
    prompt_template_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build judge config dictionary.
    
    Returns:
        Complete judge config dict matching JudgeConfig schema
    """
    return {
        "model": model,
        "model_version": model_version,
        "temperature": temperature,
        "seed": seed,
        "prompt_template_version": "v1.0",
        "prompt_template_hash": prompt_template_hash,
    }


def build_retrieval_config(
    index_job_id: str,
    embedding_model: str,
    embedding_dimension: int,
    recall_k: int = 100,
    precision_k: int = 25,
    hybrid_enabled: bool = True,
    hybrid_alpha: float = 0.5,
    reranking_enabled: bool = True,
    reranking_model: str = "semantic-ranker-default@latest",
    embedding_model_version: Optional[str] = None,
    reranking_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build retrieval config dictionary.
    
    Returns:
        Complete retrieval config dict matching RetrievalConfig schema
    """
    return {
        "index_job_id": index_job_id,
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "embedding_model_version": embedding_model_version,
        "recall_k": recall_k,
        "precision_k": precision_k,
        "hybrid_enabled": hybrid_enabled,
        "hybrid_alpha": hybrid_alpha,
        "reranking_enabled": reranking_enabled,
        "reranking_model": reranking_model,
        "reranking_top_k": reranking_top_k,
    }


def build_corpus_info(
    file_path: str,
    question_count: int,
    sample_size: Optional[int] = None,
    distribution: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Build corpus info dictionary.
    
    Args:
        file_path: Path to corpus file (relative to project root)
        question_count: Total questions in corpus
        sample_size: If sampled, how many questions
        distribution: Question distribution by type/difficulty
        
    Returns:
        Complete corpus info dict matching CorpusInfo schema
    """
    # Get absolute path for hashing
    project_root = Path(__file__).parent.parent.parent
    abs_path = project_root / file_path
    
    return {
        "file_path": file_path,
        "file_hash": get_file_hash(abs_path),
        "question_count": question_count,
        "sample_size": sample_size,
        "distribution": distribution,
    }


def build_execution_info(
    mode: str,
    run_type: str,
    workers: int,
    timeout_per_question_s: int = 120,
    max_retries: int = 5,
    checkpoint_interval: int = 10,
    cloud_endpoint: Optional[str] = None,
    cloud_revision: Optional[str] = None,
    cloud_region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build execution info dictionary.
    
    Args:
        mode: "local" or "cloud"
        run_type: "checkpoint", "run", or "experiment"
        workers: Number of parallel workers
        
    Returns:
        Complete execution info dict matching ExecutionInfo schema
    """
    return {
        "mode": mode,
        "run_type": run_type,
        "workers": workers,
        "timeout_per_question_s": timeout_per_question_s,
        "max_retries": max_retries,
        "checkpoint_interval": checkpoint_interval,
        "cloud_endpoint": cloud_endpoint,
        "cloud_revision": cloud_revision,
        "cloud_region": cloud_region,
    }


def get_judge_prompt_hash() -> str:
    """
    Get hash of the judge prompt template.
    
    This should be updated whenever the judge prompt changes.
    """
    judge_prompt = """You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth.

Question: {question}

Ground Truth: {ground_truth}

RAG Answer: {answer}

Context (first 2000 chars): {context}

Score 1-5 for each (5=best):
1. correctness - factually correct vs ground truth?
2. completeness - covers key points?
3. faithfulness - faithful to context, no hallucinations?
4. relevance - relevant to question?
5. clarity - clear and well-structured?

Respond with JSON containing: correctness, completeness, faithfulness, relevance, clarity, overall_score (all 1-5), and verdict (pass|partial|fail)."""
    
    return hashlib.sha256(judge_prompt.encode()).hexdigest()[:16]
