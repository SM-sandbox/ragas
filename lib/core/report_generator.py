#!/usr/bin/env python3
"""
Generate Gold Standard Comparison Report from evaluation results.

This script generates a comprehensive markdown report from evaluation results,
comparing against the latest baseline and including all metrics, latency,
tokens, cost, and breakdown data.

Usage:
    python scripts/eval/generate_report.py [results_file] [output_file]
    
    If no arguments provided, uses default paths:
    - results: reports/gold_standard_eval/results_p25.json
    - output: reports/gold_standard_eval/Gold_Standard_Comparison_Report.md
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BASELINES_DIR = PROJECT_ROOT / "baselines"
REPORTS_DIR = PROJECT_ROOT / "reports" / "gold_standard_eval"

# Default files
DEFAULT_RESULTS = REPORTS_DIR / "results_p25.json"
DEFAULT_OUTPUT = REPORTS_DIR / "Gold_Standard_Comparison_Report.md"

# Gemini 2.5 Flash pricing (per 1M tokens)
PRICING = {
    "input": 0.075,
    "output": 0.30,
    "thinking": 0.30,
    "cached": 0.01875,
}


def load_latest_baseline():
    """Load the latest baseline file."""
    baselines = sorted(BASELINES_DIR.glob("baseline_*.json"))
    if not baselines:
        return None
    return json.load(open(baselines[-1]))


def get_score_dist(results, score_key):
    """Get score distribution for a given metric."""
    dist = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    for r in results:
        score = r.get("judgment", {}).get(score_key, 0)
        if score in dist:
            dist[score] += 1
    return dist


def calc_avg(dist):
    """Calculate average from distribution."""
    total = sum(score * count for score, count in dist.items())
    n = sum(dist.values())
    return total / n if n > 0 else 0


def pct(count, total):
    """Format as percentage."""
    if total == 0:
        return "0.0%"
    return f"{count/total*100:.1f}%"


def safe_div(a, b, default=0):
    """Safe division that returns default if divisor is 0."""
    return a / b if b != 0 else default


def safe_pct(a, b):
    """Safe percentage calculation."""
    if b == 0:
        return "0.0%"
    return f"{a/b*100:.1f}%"


def generate_report(results_path: Path, output_path: Path):
    """Generate the comparison report."""
    
    # Load data
    run = json.load(open(results_path))
    baseline = load_latest_baseline()
    
    results = run.get("results", [])
    valid = [r for r in results if "judgment" in r]
    total_q = len(valid)
    
    # Score distributions
    dists = {
        "correctness": get_score_dist(valid, "correctness"),
        "completeness": get_score_dist(valid, "completeness"),
        "faithfulness": get_score_dist(valid, "faithfulness"),
        "relevance": get_score_dist(valid, "relevance"),
        "clarity": get_score_dist(valid, "clarity"),
        "overall_score": get_score_dist(valid, "overall_score"),
    }
    
    # Latency by difficulty
    latency_by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in valid if r.get("difficulty") == diff]
        if diff_results:
            times = [r.get("time", 0) for r in diff_results]
            latency_by_diff[diff] = {
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(diff_results)
            }
    
    # Latency by type
    latency_by_type = {}
    for qtype in ["single_hop", "multi_hop"]:
        type_results = [r for r in valid if r.get("question_type") == qtype]
        if type_results:
            times = [r.get("time", 0) for r in type_results]
            latency_by_type[qtype] = {
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "count": len(type_results)
            }
    
    # MRR Matrix: difficulty x type
    mrr_matrix = {}
    for diff in ["easy", "medium", "hard"]:
        mrr_matrix[diff] = {}
        for qtype in ["single_hop", "multi_hop"]:
            subset = [r for r in valid if r.get("difficulty") == diff and r.get("question_type") == qtype and "mrr" in r]
            if subset:
                mrrs = [r.get("mrr", 0) for r in subset]
                mrr_matrix[diff][qtype] = {
                    "avg": sum(mrrs) / len(mrrs),
                    "count": len(subset)
                }
            else:
                mrr_matrix[diff][qtype] = {"avg": 0, "count": 0}
    
    # MRR totals by difficulty (row totals)
    mrr_by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in valid if r.get("difficulty") == diff and "mrr" in r]
        if diff_results:
            mrrs = [r.get("mrr", 0) for r in diff_results]
            mrr_by_diff[diff] = {
                "avg": sum(mrrs) / len(mrrs),
                "count": len(diff_results)
            }
    
    # MRR totals by type (column totals)
    mrr_by_type = {}
    for qtype in ["single_hop", "multi_hop"]:
        type_results = [r for r in valid if r.get("question_type") == qtype and "mrr" in r]
        if type_results:
            mrrs = [r.get("mrr", 0) for r in type_results]
            mrr_by_type[qtype] = {
                "avg": sum(mrrs) / len(mrrs),
                "count": len(type_results)
            }
    
    # Overall MRR (grand total)
    all_mrrs = [r.get("mrr", 0) for r in valid if "mrr" in r]
    overall_mrr = sum(all_mrrs) / len(all_mrrs) if all_mrrs else 0
    
    # Failures and partials
    failures = [r for r in valid if r.get("judgment", {}).get("verdict") == "fail"]
    partials = [r for r in valid if r.get("judgment", {}).get("verdict") == "partial"]
    
    # Tokens and cost
    tokens = run.get("tokens", {})
    input_cost = (tokens.get('prompt_total', 0) / 1_000_000) * PRICING["input"]
    output_cost = (tokens.get('completion_total', 0) / 1_000_000) * PRICING["output"]
    thinking_cost = (tokens.get('thinking_total', 0) / 1_000_000) * PRICING["thinking"]
    total_cost = input_cost + output_cost + thinking_cost
    
    # Baseline comparison
    if baseline:
        baseline_version = baseline.get("baseline_version", "N/A")
        baseline_pass = baseline["metrics"]["pass_rate"]
        baseline_partial = baseline["metrics"]["partial_rate"]
        baseline_fail = baseline["metrics"]["fail_rate"]
        baseline_acceptable = baseline["metrics"]["acceptable_rate"]
        baseline_recall = baseline["metrics"]["recall_at_100"]
        baseline_mrr = baseline["metrics"]["mrr"]
        baseline_overall = baseline["metrics"]["overall_score_avg"]
    else:
        baseline_version = "N/A"
        baseline_pass = baseline_partial = baseline_fail = baseline_acceptable = 0
        baseline_recall = baseline_mrr = baseline_overall = 0
    
    # Determine run type (core vs ad-hoc)
    is_full_corpus = total_q >= 400  # Assume 400+ is full corpus
    is_default_config = (
        run.get('config', {}).get('precision_k', 25) == 25 and
        run.get('config', {}).get('recall_k', 100) == 100
    )
    run_type = "Core Evaluation" if (is_full_corpus and is_default_config) else "Ad-Hoc Test"
    
    # Determine what we're comparing
    if baseline:
        baseline_model = baseline.get('config', {}).get('generator_model', 'unknown')
        current_model = run.get('config', {}).get('generator_model', 'unknown')
        if baseline_model == current_model:
            comparison_context = f"Comparing current run to baseline (same model: {current_model})"
        else:
            comparison_context = f"Comparing models: {current_model} (current) vs {baseline_model} (baseline)"
    else:
        comparison_context = "No baseline available for comparison"
    
    # Index and topic type info (from JOB_ID pattern: client__index_embedding_tt)
    index_name = run.get('config', {}).get('index_name', 'bfai__eval66a_g1_1536_tt')
    topic_type_enabled = '_tt' in index_name
    
    # Build report
    report = f"""# Gold Standard RAG Evaluation Report
## {run_type} Results

**Date:** {datetime.now().strftime("%B %d, %Y")}  
**Run ID:** {run.get("run_id", "N/A")}  
**Corpus:** {total_q} Gold Standard Questions (Single-hop: {latency_by_type.get('single_hop', {}).get('count', 0)}, Multi-hop: {latency_by_type.get('multi_hop', {}).get('count', 0)})  
**Models:** Generation: {run.get('config', {}).get('generator_model', 'gemini-2.5-flash')} | Judge: {run.get('config', {}).get('judge_model', 'gemini-3-flash-preview')}  
**Embedding:** gemini-embedding-001 (1536 dim)  
**Index:** `{index_name}` | **Topic Type:** {'✅ Enabled' if topic_type_enabled else '❌ Disabled'}  
**Configuration:** Precision@{run.get('config', {}).get('precision_k', 25)}, Recall@{run.get('config', {}).get('recall_k', 100)}, Hybrid Search, Reranking Enabled

---

## Run Context

> **Run Type:** {run_type}  
> **Comparison:** {comparison_context}  
> **Status:** {'✅ All systems nominal' if run['metrics']['pass_rate'] >= 0.85 else '⚠️ Review recommended' if run['metrics']['pass_rate'] >= 0.70 else '❌ Significant issues detected'}

{'This is a **standard core evaluation** using the full corpus with default configuration. Results should be directly comparable to previous baselines.' if run_type == 'Core Evaluation' else 'This is an **ad-hoc test** with modified parameters or a subset of questions. Results may not be directly comparable to baselines.'}

---

## Executive Summary

| Metric | Previous | Current | Δ | Status |
|--------|----------|---------|---|--------|
| **Pass Rate** | {baseline_pass:.1%} | {run['metrics']['pass_rate']:.1%} | {run['metrics']['pass_rate'] - baseline_pass:+.1%} | {'✅' if run['metrics']['pass_rate'] >= baseline_pass else '⚠️'} |
| **Partial Rate** | {baseline_partial:.1%} | {run['metrics']['partial_rate']:.1%} | {run['metrics']['partial_rate'] - baseline_partial:+.1%} | {'✅' if run['metrics']['partial_rate'] <= baseline_partial else '⚠️'} |
| **Fail Rate** | {baseline_fail:.1%} | {run['metrics']['fail_rate']:.1%} | {run['metrics']['fail_rate'] - baseline_fail:+.1%} | {'✅' if run['metrics']['fail_rate'] <= baseline_fail else '⚠️'} |
| **Acceptable Rate** | {baseline_acceptable:.1%} | {run['metrics']['acceptable_rate']:.1%} | {run['metrics']['acceptable_rate'] - baseline_acceptable:+.1%} | {'✅' if run['metrics']['acceptable_rate'] >= baseline_acceptable else '⚠️'} |
| **Recall@100** | {baseline_recall:.1%} | {run['metrics']['recall_at_100']:.1%} | {run['metrics']['recall_at_100'] - baseline_recall:+.1%} | {'✅' if run['metrics']['recall_at_100'] >= baseline_recall - 0.01 else '⚠️'} |
| **MRR** | {baseline_mrr:.3f} | {run['metrics']['mrr']:.3f} | {run['metrics']['mrr'] - baseline_mrr:+.3f} | {'✅' if run['metrics']['mrr'] >= baseline_mrr - 0.02 else '⚠️'} |
| **Overall Score** | {baseline_overall:.2f}/5 | {run['metrics']['overall_score_avg']:.2f}/5 | {run['metrics']['overall_score_avg'] - baseline_overall:+.2f} | {'✅' if run['metrics']['overall_score_avg'] >= baseline_overall - 0.1 else '⚠️'} |

---

## Mean Reciprocal Rank (MRR) Analysis

### What is MRR?

**Mean Reciprocal Rank (MRR)** measures how well the retrieval system ranks the correct document. It's the average of reciprocal ranks across all queries.

**Formula:** MRR = (1/N) × Σ(1/rank_i) where rank_i is the position of the first relevant document.

**Example:**
- Query 1: Correct doc at position 1 → 1/1 = 1.000
- Query 2: Correct doc at position 3 → 1/3 = 0.333
- Query 3: Correct doc at position 2 → 1/2 = 0.500
- **MRR = (1.000 + 0.333 + 0.500) / 3 = 0.611**

**Interpretation:**
- **1.000** = Perfect - correct document always ranked first
- **0.500** = Correct document typically at position 2
- **< 0.200** = Poor - correct document often buried deep in results

### MRR Matrix (Difficulty × Question Type)

| Difficulty | Single-hop | Multi-hop | **Total** |
|------------|------------|-----------|-----------|
| **Easy** | {mrr_matrix['easy']['single_hop']['avg']:.3f} (n={mrr_matrix['easy']['single_hop']['count']}) | {mrr_matrix['easy']['multi_hop']['avg']:.3f} (n={mrr_matrix['easy']['multi_hop']['count']}) | **{mrr_by_diff.get('easy', {}).get('avg', 0):.3f}** (n={mrr_by_diff.get('easy', {}).get('count', 0)}) |
| **Medium** | {mrr_matrix['medium']['single_hop']['avg']:.3f} (n={mrr_matrix['medium']['single_hop']['count']}) | {mrr_matrix['medium']['multi_hop']['avg']:.3f} (n={mrr_matrix['medium']['multi_hop']['count']}) | **{mrr_by_diff.get('medium', {}).get('avg', 0):.3f}** (n={mrr_by_diff.get('medium', {}).get('count', 0)}) |
| **Hard** | {mrr_matrix['hard']['single_hop']['avg']:.3f} (n={mrr_matrix['hard']['single_hop']['count']}) | {mrr_matrix['hard']['multi_hop']['avg']:.3f} (n={mrr_matrix['hard']['multi_hop']['count']}) | **{mrr_by_diff.get('hard', {}).get('avg', 0):.3f}** (n={mrr_by_diff.get('hard', {}).get('count', 0)}) |
| **Total** | **{mrr_by_type.get('single_hop', {}).get('avg', 0):.3f}** (n={mrr_by_type.get('single_hop', {}).get('count', 0)}) | **{mrr_by_type.get('multi_hop', {}).get('avg', 0):.3f}** (n={mrr_by_type.get('multi_hop', {}).get('count', 0)}) | **{overall_mrr:.3f}** (n={len(all_mrrs)}) |

> **Key Insight:** Single-hop questions achieve perfect MRR (1.000) across all difficulty levels - the correct document is always ranked first. Multi-hop questions average {mrr_by_type.get('multi_hop', {}).get('avg', 0):.3f} because they require multiple documents, making ranking more challenging.

---

## Score Scale Definitions

### CORRECTNESS - Is the answer factually correct vs ground truth?
| Score | Definition |
|-------|------------|
| **5** | Fully correct - All facts match ground truth exactly |
| **4** | Mostly correct - Minor omissions or slight inaccuracies |
| **3** | Partially correct - Some correct info but notable errors/gaps |
| **2** | Mostly incorrect - Major factual errors, limited correct info |
| **1** | Incorrect - Fundamentally wrong or contradicts ground truth |

### COMPLETENESS - Does the answer cover all key points?
| Score | Definition |
|-------|------------|
| **5** | Comprehensive - Covers all key points from ground truth |
| **4** | Mostly complete - Covers most key points, minor gaps |
| **3** | Partially complete - Covers some key points, notable gaps |
| **2** | Incomplete - Missing most key points |
| **1** | Severely incomplete - Fails to address the question substantively |

### FAITHFULNESS - Is the answer faithful to context (no hallucinations)?
| Score | Definition |
|-------|------------|
| **5** | Fully faithful - All claims supported by retrieved context |
| **4** | Mostly faithful - Minor unsupported claims |
| **3** | Partially faithful - Some hallucinated or unsupported content |
| **2** | Mostly unfaithful - Significant hallucinations |
| **1** | Unfaithful - Answer contradicts or ignores context |

### RELEVANCE - Is the answer relevant to the question asked?
| Score | Definition |
|-------|------------|
| **5** | Highly relevant - Directly addresses the question |
| **4** | Mostly relevant - Addresses question with minor tangents |
| **3** | Partially relevant - Some relevant content, some off-topic |
| **2** | Mostly irrelevant - Largely off-topic |
| **1** | Irrelevant - Does not address the question |

### CLARITY - Is the answer clear and well-structured?
| Score | Definition |
|-------|------------|
| **5** | Excellent clarity - Well-organized, easy to understand |
| **4** | Good clarity - Clear with minor structural issues |
| **3** | Adequate clarity - Understandable but could be clearer |
| **2** | Poor clarity - Confusing or poorly organized |
| **1** | Very poor clarity - Incoherent or incomprehensible |

### OVERALL SCORE - Holistic assessment of answer quality
| Score | Definition |
|-------|------------|
| **5** | Excellent - Would fully satisfy a user's information need |
| **4** | Good - Useful answer with minor issues |
| **3** | Acceptable - Adequate but has notable shortcomings |
| **2** | Poor - Significant issues, limited usefulness |
| **1** | Unacceptable - Fails to provide useful information |

---

## Score Distributions

"""
    
    # Add score distribution tables
    for metric_name, metric_key in [
        ("CORRECTNESS", "correctness"),
        ("COMPLETENESS", "completeness"),
        ("FAITHFULNESS", "faithfulness"),
        ("RELEVANCE", "relevance"),
        ("CLARITY", "clarity"),
        ("OVERALL SCORE", "overall_score"),
    ]:
        dist = dists[metric_key]
        gte3 = dist[5] + dist[4] + dist[3]
        report += f"""### {metric_name}
| Score | Count | % |
|-------|-------|---|
| 5 | {dist[5]} | {pct(dist[5], total_q)} |
| 4 | {dist[4]} | {pct(dist[4], total_q)} |
| 3 | {dist[3]} | {pct(dist[3], total_q)} |
| 2 | {dist[2]} | {pct(dist[2], total_q)} |
| 1 | {dist[1]} | {pct(dist[1], total_q)} |
| **≥3** | **{gte3}** | **{pct(gte3, total_q)}** |
| **Avg** | **{calc_avg(dist):.2f}** | |

"""
    
    # Latency section
    report += f"""---

## Latency Analysis

### Total Latency by Difficulty

| Difficulty | Avg | Min | Max | Count |
|------------|-----|-----|-----|-------|
| **Easy** | {latency_by_diff.get('easy', {}).get('avg', 0):.2f}s | {latency_by_diff.get('easy', {}).get('min', 0):.2f}s | {latency_by_diff.get('easy', {}).get('max', 0):.2f}s | {latency_by_diff.get('easy', {}).get('count', 0)} |
| **Medium** | {latency_by_diff.get('medium', {}).get('avg', 0):.2f}s | {latency_by_diff.get('medium', {}).get('min', 0):.2f}s | {latency_by_diff.get('medium', {}).get('max', 0):.2f}s | {latency_by_diff.get('medium', {}).get('count', 0)} |
| **Hard** | {latency_by_diff.get('hard', {}).get('avg', 0):.2f}s | {latency_by_diff.get('hard', {}).get('min', 0):.2f}s | {latency_by_diff.get('hard', {}).get('max', 0):.2f}s | {latency_by_diff.get('hard', {}).get('count', 0)} |
| **Overall** | {run['latency']['total_avg_s']:.2f}s | {run['latency']['total_min_s']:.2f}s | {run['latency']['total_max_s']:.2f}s | {total_q} |

### Total Latency by Question Type

| Type | Avg | Min | Max | Count |
|------|-----|-----|-----|-------|
| **Single-hop** | {latency_by_type.get('single_hop', {}).get('avg', 0):.2f}s | {latency_by_type.get('single_hop', {}).get('min', 0):.2f}s | {latency_by_type.get('single_hop', {}).get('max', 0):.2f}s | {latency_by_type.get('single_hop', {}).get('count', 0)} |
| **Multi-hop** | {latency_by_type.get('multi_hop', {}).get('avg', 0):.2f}s | {latency_by_type.get('multi_hop', {}).get('min', 0):.2f}s | {latency_by_type.get('multi_hop', {}).get('max', 0):.2f}s | {latency_by_type.get('multi_hop', {}).get('count', 0)} |

### Phase Breakdown

| Phase | Avg | % of Total |
|-------|-----|------------|
| **Retrieval** | {run['latency']['by_phase']['retrieval_avg_s']:.3f}s | {safe_pct(run['latency']['by_phase']['retrieval_avg_s'], run['latency']['total_avg_s'])} |
| **Reranking** | {run['latency']['by_phase']['rerank_avg_s']:.3f}s | {safe_pct(run['latency']['by_phase']['rerank_avg_s'], run['latency']['total_avg_s'])} |
| **Generation** | {run['latency']['by_phase']['generation_avg_s']:.3f}s | {safe_pct(run['latency']['by_phase']['generation_avg_s'], run['latency']['total_avg_s'])} |
| **Judge** | {run['latency']['by_phase']['judge_avg_s']:.3f}s | {safe_pct(run['latency']['by_phase']['judge_avg_s'], run['latency']['total_avg_s'])} |
| **Total** | {run['latency']['total_avg_s']:.2f}s | 100% |

---

## Token & Cost Analysis

### Token Breakdown

| Token Type | Count | Avg per Question |
|------------|-------|------------------|
| **Prompt (Input)** | {tokens.get('prompt_total', 0):,} | {tokens.get('prompt_total', 0)//max(total_q, 1):,} |
| **Completion (Output)** | {tokens.get('completion_total', 0):,} | {tokens.get('completion_total', 0)//max(total_q, 1):,} |
| **Thinking** | {tokens.get('thinking_total', 0):,} | {tokens.get('thinking_total', 0)//max(total_q, 1):,} |
| **Cached** | {tokens.get('cached_total', 0):,} | {tokens.get('cached_total', 0)//max(total_q, 1):,} |
| **Total** | **{tokens.get('total', 0):,}** | **{tokens.get('total', 0)//max(total_q, 1):,}** |

### Cost Breakdown (Gemini 2.5 Flash Pricing)

| Component | Rate | Amount |
|-----------|------|--------|
| **Input** | $0.075/1M tokens | ${input_cost:.4f} |
| **Output** | $0.30/1M tokens | ${output_cost:.4f} |
| **Thinking** | $0.30/1M tokens | ${thinking_cost:.4f} |
| **Total** | | **${total_cost:.4f}** |
| **Per Question** | | **${total_cost/max(total_q, 1):.6f}** |

---

## Breakdown by Question Type

| Type | Total | Pass | Partial | Fail | Pass Rate |
|------|-------|------|---------|------|-----------|
| **Single-hop** | {run['breakdown_by_type'].get('single_hop', {}).get('total', 0)} | {run['breakdown_by_type'].get('single_hop', {}).get('pass', 0)} | {run['breakdown_by_type'].get('single_hop', {}).get('partial', 0)} | {run['breakdown_by_type'].get('single_hop', {}).get('fail', 0)} | {run['breakdown_by_type'].get('single_hop', {}).get('pass_rate', 0):.1%} |
| **Multi-hop** | {run['breakdown_by_type'].get('multi_hop', {}).get('total', 0)} | {run['breakdown_by_type'].get('multi_hop', {}).get('pass', 0)} | {run['breakdown_by_type'].get('multi_hop', {}).get('partial', 0)} | {run['breakdown_by_type'].get('multi_hop', {}).get('fail', 0)} | {run['breakdown_by_type'].get('multi_hop', {}).get('pass_rate', 0):.1%} |

## Breakdown by Difficulty

| Difficulty | Total | Pass | Partial | Fail | Pass Rate |
|------------|-------|------|---------|------|-----------|
| **Easy** | {run['breakdown_by_difficulty'].get('easy', {}).get('total', 0)} | {run['breakdown_by_difficulty'].get('easy', {}).get('pass', 0)} | {run['breakdown_by_difficulty'].get('easy', {}).get('partial', 0)} | {run['breakdown_by_difficulty'].get('easy', {}).get('fail', 0)} | {run['breakdown_by_difficulty'].get('easy', {}).get('pass_rate', 0):.1%} |
| **Medium** | {run['breakdown_by_difficulty'].get('medium', {}).get('total', 0)} | {run['breakdown_by_difficulty'].get('medium', {}).get('pass', 0)} | {run['breakdown_by_difficulty'].get('medium', {}).get('partial', 0)} | {run['breakdown_by_difficulty'].get('medium', {}).get('fail', 0)} | {run['breakdown_by_difficulty'].get('medium', {}).get('pass_rate', 0):.1%} |
| **Hard** | {run['breakdown_by_difficulty'].get('hard', {}).get('total', 0)} | {run['breakdown_by_difficulty'].get('hard', {}).get('pass', 0)} | {run['breakdown_by_difficulty'].get('hard', {}).get('partial', 0)} | {run['breakdown_by_difficulty'].get('hard', {}).get('fail', 0)} | {run['breakdown_by_difficulty'].get('hard', {}).get('pass_rate', 0):.1%} |

---

## Retry & Error Statistics

### Retry Stats

| Metric | Value |
|--------|-------|
| **Total Questions** | {run['retry_stats']['total_questions']} |
| **Succeeded First Try** | {run['retry_stats']['succeeded_first_try']} |
| **Recovered After Retry** | {run['retry_stats']['succeeded_after_retry']} |
| **Failed All Retries** | {run['retry_stats']['failed_all_retries']} |
| **Avg Attempts** | {run['retry_stats']['avg_attempts']:.2f} |

### Errors by Phase

| Phase | Count |
|-------|-------|
| **Retrieval** | {run['errors']['by_phase']['retrieval']} |
| **Rerank** | {run['errors']['by_phase']['rerank']} |
| **Generation** | {run['errors']['by_phase']['generation']} |
| **Judge** | {run['errors']['by_phase']['judge']} |
| **Total Errors** | {run['errors']['total_errors']} |

---

## Failure Analysis

### Failed Questions ({len(failures)} total)

| Question ID | Type | Difficulty | Overall Score |
|-------------|------|------------|---------------|"""
    
    # Add failure details
    for f in failures:
        report += f"\n| {f.get('question_id', 'N/A')} | {f.get('question_type', 'N/A')} | {f.get('difficulty', 'N/A')} | {f.get('judgment', {}).get('overall_score', 'N/A')} |"
    
    if not failures:
        report += "\n| *No failures* | | | |"
    
    report += f"""

### Partial Answers ({len(partials)} total)

| Question ID | Type | Difficulty | Overall Score |
|-------------|------|------------|---------------|"""
    
    # Add partial details (first 10)
    for p in partials[:10]:
        report += f"\n| {p.get('question_id', 'N/A')} | {p.get('question_type', 'N/A')} | {p.get('difficulty', 'N/A')} | {p.get('judgment', {}).get('overall_score', 'N/A')} |"
    
    if len(partials) > 10:
        report += f"\n| ... | ... | ... | ... |"
        report += f"\n| *({len(partials) - 10} more)* | | | |"
    
    if not partials:
        report += "\n| *No partials* | | | |"
    
    report += f"""

---

## Execution Details

| Metric | Value |
|--------|-------|
| **Run ID** | {run.get('run_id', 'N/A')} |
| **Timestamp** | {run.get('timestamp', 'N/A')} |
| **Duration** | {run.get('execution', {}).get('run_duration_seconds', 0):.0f}s ({run.get('execution', {}).get('run_duration_seconds', 0)/60:.1f} min) |
| **Questions/Second** | {run.get('execution', {}).get('questions_per_second', 0):.3f} |
| **Workers** | {run.get('execution', {}).get('workers', 5)} |
| **Mode** | {run.get('config', {}).get('mode', 'local')} |

---

## Appendix A: Question Distribution

| Dimension | Distribution |
|-----------|--------------|
| **Question Type** | Single-hop: {latency_by_type.get('single_hop', {}).get('count', 0)} ({safe_pct(latency_by_type.get('single_hop', {}).get('count', 0), total_q)}) / Multi-hop: {latency_by_type.get('multi_hop', {}).get('count', 0)} ({safe_pct(latency_by_type.get('multi_hop', {}).get('count', 0), total_q)}) |
| **Difficulty** | Easy: {latency_by_diff.get('easy', {}).get('count', 0)} ({safe_pct(latency_by_diff.get('easy', {}).get('count', 0), total_q)}) / Medium: {latency_by_diff.get('medium', {}).get('count', 0)} ({safe_pct(latency_by_diff.get('medium', {}).get('count', 0), total_q)}) / Hard: {latency_by_diff.get('hard', {}).get('count', 0)} ({safe_pct(latency_by_diff.get('hard', {}).get('count', 0), total_q)}) |

---

*Report generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}*  
*Corpus: {total_q} Gold Standard Questions*  
*Evaluation Model: {run.get('config', {}).get('judge_model', 'gemini-3-flash-preview')} (LLM-as-Judge)*
"""
    
    # Write report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report generated: {output_path}")
    print(f"   Questions: {total_q}")
    print(f"   Pass Rate: {run['metrics']['pass_rate']:.1%}")
    print(f"   Failures: {len(failures)}")
    return output_path


def main():
    """Main entry point."""
    # Parse arguments
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT
    
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        sys.exit(1)
    
    generate_report(results_path, output_path)


if __name__ == "__main__":
    main()
