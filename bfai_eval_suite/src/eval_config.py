"""
Default evaluation configuration for BrightFox RAG evaluation suite.

Standard settings used across all evaluation scripts:
- Gemini 2.5 Flash for LLM operations
- 15 parallel workers (optimal balance of speed vs rate limits)
- 5 retries with exponential backoff
- Progress bar with live metrics
- JSON output with timestamps
- Checkpointing every 10 items
"""

import os

# GCP Configuration
GCP_PROJECT = os.environ.get("GCP_PROJECT", "civic-athlete-473921-c0")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-east1")
GCP_LLM_LOCATION = os.environ.get("GCP_LLM_LOCATION", "us-central1")

# LLM Configuration
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY"

# Execution Configuration
DEFAULT_WORKERS = 15  # Optimal: 25 hits rate limits, 10 is slow, 15 is sweet spot
DEFAULT_RETRIES = 5   # With exponential backoff
DEFAULT_TIMEOUT = 60  # Seconds per request
CHECKPOINT_INTERVAL = 10  # Save progress every N items

# Retrieval Configuration
DEFAULT_TOP_K = 12
USE_RERANKER = False

# Output Configuration
OUTPUT_FORMAT = "json"
INCLUDE_TIMING = True
INCLUDE_CHAR_COUNT = True

# Progress Bar Configuration
PROGRESS_BAR_CONFIG = {
    "unit": "q",  # questions
    "dynamic_ncols": True,
    "leave": True,
    "smoothing": 0.1,
}


def get_progress_bar_format(metrics: dict = None) -> str:
    """
    Generate progress bar postfix with live metrics.
    
    Args:
        metrics: Dict with current metrics like score, pass_count, fail_count
        
    Returns:
        Formatted string for tqdm postfix
    """
    if not metrics:
        return ""
    
    parts = []
    if "score" in metrics:
        parts.append(f"avg={metrics['score']:.2f}")
    if "pass" in metrics:
        parts.append(f"pass={metrics['pass']}")
    if "fail" in metrics:
        parts.append(f"fail={metrics['fail']}")
    if "buckets" in metrics:
        # Show distribution like "5:10|4:25|3:15|2:5|1:2"
        bucket_str = "|".join(f"{k}:{v}" for k, v in sorted(metrics['buckets'].items(), reverse=True))
        parts.append(bucket_str)
    
    return ", ".join(parts)


# Why 15 workers?
# ----------------
# Through empirical testing:
# - 25 workers: Hits Gemini rate limits frequently, causes retries and slowdowns
# - 20 workers: Occasional rate limit hits
# - 15 workers: Sweet spot - fast execution without rate limit issues
# - 10 workers: Works but noticeably slower
# 
# 15 workers achieves ~3-4 questions/second sustained throughput
# which is optimal for the Gemini API quotas on this project.
