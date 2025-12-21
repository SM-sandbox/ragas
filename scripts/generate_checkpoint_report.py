#!/usr/bin/env python3
"""
Generate Checkpoint Report from evaluation results.

Generates a formatted markdown report from checkpoint/run results JSON,
placing the report alongside the JSON in the same directory.

Usage:
    python scripts/generate_checkpoint_report.py C013
    python scripts/generate_checkpoint_report.py --path /path/to/results.json
    python scripts/generate_checkpoint_report.py --list  # List available checkpoints
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "clients_eval_data" / "BFAI" / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "clients_eval_data" / "BFAI" / "runs"
REGISTRY_PATH = CHECKPOINTS_DIR / "registry.json"

# Gemini 3 Flash pricing (per 1M tokens)
PRICING = {
    "input": 0.075,
    "output": 0.30,
    "thinking": 0.30,
    "cached": 0.01875,
}


def load_registry():
    """Load the checkpoint registry."""
    if REGISTRY_PATH.exists():
        return json.load(open(REGISTRY_PATH))
    return {"entries": [], "gold_baseline": None}


def find_checkpoint_folder(checkpoint_id: str) -> Path:
    """Find the folder for a checkpoint ID."""
    # Check checkpoints directory
    for folder in CHECKPOINTS_DIR.iterdir():
        if folder.is_dir() and folder.name.startswith(f"{checkpoint_id}__"):
            return folder
    
    # Check runs directory
    for folder in RUNS_DIR.iterdir():
        if folder.is_dir() and folder.name.startswith(f"{checkpoint_id}__"):
            return folder
    
    return None


def list_checkpoints():
    """List all available checkpoints."""
    registry = load_registry()
    
    print("\n=== Available Checkpoints ===\n")
    
    gold = registry.get("gold_baseline", {})
    if gold:
        print(f"  GOLD BASELINE: {gold.get('id')} (Pass: {gold.get('pass_rate', 0):.1%}, Acceptable: {gold.get('acceptable_rate', 0):.1%})")
        print()
    
    for entry in registry.get("entries", []):
        marker = " **GOLD**" if entry.get("id") == gold.get("id") else ""
        print(f"  {entry.get('id')}: {entry.get('date')} | {entry.get('mode')} | {entry.get('config_summary')} | Pass: {entry.get('pass_rate', 0):.1%}{marker}")
    
    print("\n=== Available Runs ===\n")
    if RUNS_DIR.exists():
        for folder in sorted(RUNS_DIR.iterdir()):
            if folder.is_dir() and folder.name.startswith("R"):
                results_file = folder / "results.json"
                if results_file.exists():
                    data = json.load(open(results_file))
                    metrics = data.get("metrics", {})
                    print(f"  {folder.name[:4]}: Pass: {metrics.get('pass_rate', 0):.1%}")
    
    print()


def calculate_cost(tokens: dict) -> dict:
    """Calculate cost from token counts."""
    prompt = tokens.get("prompt_total", 0)
    completion = tokens.get("completion_total", 0)
    thinking = tokens.get("thinking_total", 0)
    cached = tokens.get("cached_total", 0)
    
    input_cost = (prompt / 1_000_000) * PRICING["input"]
    output_cost = (completion / 1_000_000) * PRICING["output"]
    thinking_cost = (thinking / 1_000_000) * PRICING["thinking"]
    cached_cost = (cached / 1_000_000) * PRICING["cached"]
    
    total_cost = input_cost + output_cost + thinking_cost - cached_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "thinking_cost": thinking_cost,
        "cached_savings": cached_cost,
        "total_cost": max(0, total_cost),
    }


def get_breakdown_by_type(results: list) -> dict:
    """Get pass/fail breakdown by question type."""
    by_type = defaultdict(lambda: {"total": 0, "pass": 0, "partial": 0, "fail": 0})
    
    for r in results:
        qtype = "single-hop" if r.get("question_type", "").startswith("sh") else "multi-hop"
        verdict = r.get("verdict", "").lower()
        by_type[qtype]["total"] += 1
        if verdict == "pass":
            by_type[qtype]["pass"] += 1
        elif verdict == "partial":
            by_type[qtype]["partial"] += 1
        elif verdict == "fail":
            by_type[qtype]["fail"] += 1
    
    return dict(by_type)


def get_breakdown_by_difficulty(results: list) -> dict:
    """Get pass/fail breakdown by difficulty."""
    by_diff = defaultdict(lambda: {"total": 0, "pass": 0, "partial": 0, "fail": 0})
    
    for r in results:
        diff = r.get("difficulty", "unknown")
        verdict = r.get("verdict", "").lower()
        by_diff[diff]["total"] += 1
        if verdict == "pass":
            by_diff[diff]["pass"] += 1
        elif verdict == "partial":
            by_diff[diff]["partial"] += 1
        elif verdict == "fail":
            by_diff[diff]["fail"] += 1
    
    return dict(by_diff)


def get_failures(results: list) -> list:
    """Get list of failed questions."""
    failures = []
    for r in results:
        if r.get("verdict", "").lower() == "fail":
            failures.append({
                "id": r.get("question_id", "unknown"),
                "type": r.get("question_type", "unknown"),
                "difficulty": r.get("difficulty", "unknown"),
                "overall_score": r.get("judgment", {}).get("overall_score", 0),
            })
    return failures


def generate_report(results_path: Path, output_path: Path = None, baseline_data: dict = None):
    """
    Generate a markdown report from results JSON.
    
    Args:
        results_path: Path to results.json
        output_path: Optional output path (default: REPORT.md alongside results)
        baseline_data: Optional baseline results dict for comparison
    """
    
    # Load results
    data = json.load(open(results_path))
    
    # Determine output path (same directory as results)
    if output_path is None:
        output_path = results_path.parent / "REPORT.md"
    
    # Extract data
    config = data.get("config", {})
    metrics = data.get("metrics", {})
    latency = data.get("latency", {})
    tokens = data.get("tokens", {})
    index_info = data.get("index", {})
    results = data.get("results", [])
    
    # Calculate derived values
    cost = calculate_cost(tokens)
    by_type = get_breakdown_by_type(results)
    by_diff = get_breakdown_by_difficulty(results)
    failures = get_failures(results)
    
    # Build report
    lines = []
    
    # Header
    checkpoint_id = results_path.parent.name.split("__")[0]
    lines.append(f"# Checkpoint Report: {checkpoint_id}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Results File:** `{results_path.name}`")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Pass Rate** | {metrics.get('pass_rate', 0):.1%} |")
    lines.append(f"| **Partial Rate** | {metrics.get('partial_rate', 0):.1%} |")
    lines.append(f"| **Fail Rate** | {metrics.get('fail_rate', 0):.1%} |")
    lines.append(f"| **Acceptable Rate** | {metrics.get('acceptable_rate', 0):.1%} |")
    lines.append(f"| **Overall Score** | {metrics.get('overall_score_avg', 0):.2f}/5 |")
    lines.append(f"| **Questions** | {metrics.get('total', 0)} |")
    lines.append("")
    
    # Comparison to Gold Baseline
    if baseline_data:
        baseline_metrics = baseline_data.get("metrics", {})
        baseline_latency = baseline_data.get("latency", {})
        baseline_cost = baseline_data.get("cost_summary_usd", {})
        baseline_config = baseline_data.get("config", {})
        
        lines.append("## Comparison to Gold Baseline")
        lines.append("")
        lines.append("### Quality Metrics")
        lines.append("")
        lines.append("| Metric | Baseline | This Run | Delta | Status |")
        lines.append("|--------|----------|----------|-------|--------|")
        
        # Pass rate
        b_pass = baseline_metrics.get("pass_rate", 0)
        t_pass = metrics.get("pass_rate", 0)
        d_pass = t_pass - b_pass
        s_pass = "✅" if d_pass >= -0.02 else "⚠️" if d_pass >= -0.05 else "❌"
        lines.append(f"| **Pass Rate** | {b_pass:.1%} | {t_pass:.1%} | {d_pass:+.1%} | {s_pass} |")
        
        # Acceptable rate
        b_accept = baseline_metrics.get("acceptable_rate", 0)
        t_accept = metrics.get("acceptable_rate", 0)
        d_accept = t_accept - b_accept
        s_accept = "✅" if d_accept >= -0.01 else "⚠️" if d_accept >= -0.02 else "❌"
        lines.append(f"| **Acceptable Rate** | {b_accept:.1%} | {t_accept:.1%} | {d_accept:+.1%} | {s_accept} |")
        
        # Fail rate (lower is better)
        b_fail = baseline_metrics.get("fail_rate", 0)
        t_fail = metrics.get("fail_rate", 0)
        d_fail = t_fail - b_fail
        s_fail = "✅" if d_fail <= 0.01 else "⚠️" if d_fail <= 0.02 else "❌"
        lines.append(f"| **Fail Rate** | {b_fail:.1%} | {t_fail:.1%} | {d_fail:+.1%} | {s_fail} |")
        
        # MRR
        b_mrr = baseline_metrics.get("mrr", 0)
        t_mrr = metrics.get("mrr", 0)
        d_mrr = t_mrr - b_mrr
        s_mrr = "✅" if d_mrr >= -0.02 else "⚠️"
        lines.append(f"| **MRR** | {b_mrr:.3f} | {t_mrr:.3f} | {d_mrr:+.3f} | {s_mrr} |")
        
        # Recall
        b_recall = baseline_metrics.get("recall_at_100", 0)
        t_recall = metrics.get("recall_at_100", 0)
        d_recall = t_recall - b_recall
        s_recall = "✅" if d_recall >= -0.01 else "⚠️"
        lines.append(f"| **Recall@100** | {b_recall:.1%} | {t_recall:.1%} | {d_recall:+.1%} | {s_recall} |")
        
        # Overall score
        b_score = baseline_metrics.get("overall_score_avg", 0)
        t_score = metrics.get("overall_score_avg", 0)
        d_score = t_score - b_score
        s_score = "✅" if d_score >= -0.1 else "⚠️"
        lines.append(f"| **Overall Score** | {b_score:.2f}/5 | {t_score:.2f}/5 | {d_score:+.2f} | {s_score} |")
        lines.append("")
        
        # Latency comparison
        lines.append("### Latency")
        lines.append("")
        lines.append("| Metric | Baseline | This Run | Delta | Speedup |")
        lines.append("|--------|----------|----------|-------|---------|")
        
        b_lat = baseline_latency.get("total_avg_s", 0)
        t_lat = latency.get("total_avg_s", 0)
        d_lat = t_lat - b_lat
        speedup = b_lat / t_lat if t_lat > 0 else 0
        s_lat = "✅" if speedup >= 0.9 else "⚠️" if speedup >= 0.5 else "❌"
        lines.append(f"| **Avg Latency** | {b_lat:.1f}s | {t_lat:.1f}s | {d_lat:+.1f}s | {speedup:.2f}x {s_lat} |")
        lines.append("")
        
        # Cost comparison
        lines.append("### Cost")
        lines.append("")
        lines.append("| Metric | Baseline | This Run | Delta |")
        lines.append("|--------|----------|----------|-------|")
        
        b_cost = baseline_cost.get("total", 0)
        t_cost = data.get("cost_summary_usd", {}).get("total", 0)
        d_cost = t_cost - b_cost
        lines.append(f"| **Total Cost** | ${b_cost:.4f} | ${t_cost:.4f} | ${d_cost:+.4f} |")
        
        b_per_q = baseline_cost.get("avg_per_question", 0)
        t_per_q = data.get("cost_summary_usd", {}).get("avg_per_question", 0)
        d_per_q = t_per_q - b_per_q
        lines.append(f"| **Per Question** | ${b_per_q:.6f} | ${t_per_q:.6f} | ${d_per_q:+.6f} |")
        lines.append("")
        
        # Breakdown comparison
        baseline_by_type = baseline_data.get("breakdown_by_type", {})
        baseline_by_diff = baseline_data.get("breakdown_by_difficulty", {})
        this_by_type = data.get("breakdown_by_type", {})
        this_by_diff = data.get("breakdown_by_difficulty", {})
        
        lines.append("### By Question Type")
        lines.append("")
        lines.append("| Type | Baseline | This Run | Delta |")
        lines.append("|------|----------|----------|-------|")
        for qtype in ["single_hop", "multi_hop"]:
            b_rate = baseline_by_type.get(qtype, {}).get("pass_rate", 0)
            t_rate = this_by_type.get(qtype, {}).get("pass_rate", 0)
            d_rate = t_rate - b_rate
            lines.append(f"| **{qtype.replace('_', '-').title()}** | {b_rate:.1%} | {t_rate:.1%} | {d_rate:+.1%} |")
        lines.append("")
        
        lines.append("### By Difficulty")
        lines.append("")
        lines.append("| Difficulty | Baseline | This Run | Delta |")
        lines.append("|------------|----------|----------|-------|")
        for diff in ["easy", "medium", "hard"]:
            b_rate = baseline_by_diff.get(diff, {}).get("pass_rate", 0)
            t_rate = this_by_diff.get(diff, {}).get("pass_rate", 0)
            d_rate = t_rate - b_rate
            lines.append(f"| **{diff.title()}** | {b_rate:.1%} | {t_rate:.1%} | {d_rate:+.1%} |")
        lines.append("")
        
        # Key finding (auto-generated)
        lines.append("### Key Finding")
        lines.append("")
        issues = []
        if d_pass < -0.02:
            issues.append(f"pass rate dropped {abs(d_pass):.1%}")
        if d_accept < -0.01:
            issues.append(f"acceptable rate dropped {abs(d_accept):.1%}")
        if speedup < 0.9:
            issues.append(f"latency increased {abs(d_lat):.1f}s")
        
        if not issues:
            lines.append("✅ **All metrics within acceptable range of baseline.**")
        else:
            lines.append(f"⚠️ **Regressions detected:** {', '.join(issues)}")
        lines.append("")
    
    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| **Generator Model** | {config.get('generator_model', 'unknown')} |")
    lines.append(f"| **Reasoning Effort** | {config.get('generator_reasoning_effort', 'unknown')} |")
    lines.append(f"| **Temperature** | {config.get('temperature', 0)} |")
    lines.append(f"| **Precision@K** | {config.get('precision_k', 25)} |")
    lines.append(f"| **Recall@K** | {config.get('recall_k', 100)} |")
    lines.append(f"| **Workers** | {config.get('workers', 1)} |")
    lines.append(f"| **Judge Model** | {config.get('judge_model', 'unknown')} |")
    lines.append("")
    
    # Retrieval Metrics
    lines.append("## Retrieval Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Recall@100** | {metrics.get('recall_at_100', 0):.1%} |")
    lines.append(f"| **MRR** | {metrics.get('mrr', 0):.3f} |")
    lines.append("")
    
    # Quality Scores
    lines.append("## Quality Scores")
    lines.append("")
    lines.append("| Dimension | Average |")
    lines.append("|-----------|---------|")
    lines.append(f"| **Correctness** | {metrics.get('correctness_avg', 0):.2f}/5 |")
    lines.append(f"| **Completeness** | {metrics.get('completeness_avg', 0):.2f}/5 |")
    lines.append(f"| **Faithfulness** | {metrics.get('faithfulness_avg', 0):.2f}/5 |")
    lines.append(f"| **Relevance** | {metrics.get('relevance_avg', 0):.2f}/5 |")
    lines.append(f"| **Clarity** | {metrics.get('clarity_avg', 0):.2f}/5 |")
    lines.append(f"| **Overall** | {metrics.get('overall_score_avg', 0):.2f}/5 |")
    lines.append("")
    
    # Latency Analysis
    lines.append("## Latency Analysis")
    lines.append("")
    by_phase = latency.get("by_phase", {})
    total_avg = latency.get("total_avg_s", 0)
    
    lines.append("| Phase | Avg Time | % of Total |")
    lines.append("|-------|----------|------------|")
    
    retrieval = by_phase.get("retrieval_avg_s", 0)
    rerank = by_phase.get("rerank_avg_s", 0)
    generation = by_phase.get("generation_avg_s", 0)
    judge = by_phase.get("judge_avg_s", 0)
    
    if total_avg > 0:
        lines.append(f"| **Retrieval** | {retrieval:.2f}s | {retrieval/total_avg*100:.1f}% |")
        lines.append(f"| **Reranking** | {rerank:.2f}s | {rerank/total_avg*100:.1f}% |")
        lines.append(f"| **Generation** | {generation:.2f}s | {generation/total_avg*100:.1f}% |")
        lines.append(f"| **Judge** | {judge:.2f}s | {judge/total_avg*100:.1f}% |")
        lines.append(f"| **Total** | {total_avg:.2f}s | 100% |")
    lines.append("")
    
    lines.append(f"**Min Latency:** {latency.get('total_min_s', 0):.2f}s")
    lines.append(f"**Max Latency:** {latency.get('total_max_s', 0):.2f}s")
    lines.append("")
    
    # Token & Cost Analysis
    lines.append("## Token & Cost Analysis")
    lines.append("")
    total_q = metrics.get("total", 1)
    
    lines.append("### Token Breakdown")
    lines.append("")
    lines.append("| Token Type | Total | Per Question |")
    lines.append("|------------|-------|--------------|")
    lines.append(f"| **Prompt (Input)** | {tokens.get('prompt_total', 0):,} | {tokens.get('prompt_total', 0)/total_q:,.0f} |")
    lines.append(f"| **Completion (Output)** | {tokens.get('completion_total', 0):,} | {tokens.get('completion_total', 0)/total_q:,.0f} |")
    lines.append(f"| **Thinking** | {tokens.get('thinking_total', 0):,} | {tokens.get('thinking_total', 0)/total_q:,.0f} |")
    lines.append(f"| **Cached** | {tokens.get('cached_total', 0):,} | {tokens.get('cached_total', 0)/total_q:,.0f} |")
    lines.append(f"| **Total** | {tokens.get('total', 0):,} | {tokens.get('total', 0)/total_q:,.0f} |")
    lines.append("")
    
    lines.append("### Cost Estimate (Gemini 3 Flash)")
    lines.append("")
    lines.append("| Component | Cost |")
    lines.append("|-----------|------|")
    lines.append(f"| **Input** | ${cost['input_cost']:.4f} |")
    lines.append(f"| **Output** | ${cost['output_cost']:.4f} |")
    lines.append(f"| **Thinking** | ${cost['thinking_cost']:.4f} |")
    lines.append(f"| **Cached Savings** | -${cost['cached_savings']:.4f} |")
    lines.append(f"| **Total** | **${cost['total_cost']:.4f}** |")
    lines.append(f"| **Per Question** | ${cost['total_cost']/total_q:.6f} |")
    lines.append(f"| **Per 1,000 Questions** | ${cost['total_cost']/total_q*1000:.2f} |")
    lines.append("")
    
    # Breakdown by Type
    lines.append("## Breakdown by Question Type")
    lines.append("")
    lines.append("| Type | Total | Pass | Partial | Fail | Pass Rate |")
    lines.append("|------|-------|------|---------|------|-----------|")
    for qtype, counts in sorted(by_type.items()):
        rate = counts["pass"] / counts["total"] * 100 if counts["total"] > 0 else 0
        lines.append(f"| **{qtype.title()}** | {counts['total']} | {counts['pass']} | {counts['partial']} | {counts['fail']} | {rate:.1f}% |")
    lines.append("")
    
    # Breakdown by Difficulty
    lines.append("## Breakdown by Difficulty")
    lines.append("")
    lines.append("| Difficulty | Total | Pass | Partial | Fail | Pass Rate |")
    lines.append("|------------|-------|------|---------|------|-----------|")
    for diff in ["easy", "medium", "hard"]:
        if diff in by_diff:
            counts = by_diff[diff]
            rate = counts["pass"] / counts["total"] * 100 if counts["total"] > 0 else 0
            lines.append(f"| **{diff.title()}** | {counts['total']} | {counts['pass']} | {counts['partial']} | {counts['fail']} | {rate:.1f}% |")
    lines.append("")
    
    # Failures
    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.append(f"**Total Failures:** {len(failures)}")
        lines.append("")
        lines.append("| Question ID | Type | Difficulty | Overall Score |")
        lines.append("|-------------|------|------------|---------------|")
        for f in failures[:20]:  # Limit to 20
            lines.append(f"| {f['id']} | {f['type']} | {f['difficulty']} | {f['overall_score']}/5 |")
        if len(failures) > 20:
            lines.append(f"| ... | ... | ... | ({len(failures) - 20} more) |")
        lines.append("")
    
    # Index Info
    lines.append("## Index Information")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| **Job ID** | {index_info.get('job_id', 'unknown')} |")
    lines.append(f"| **Mode** | {index_info.get('mode', 'unknown')} |")
    lines.append(f"| **Documents** | {index_info.get('document_count', 0)} |")
    lines.append(f"| **Embedding Model** | {index_info.get('embedding_model', 'unknown')} |")
    lines.append(f"| **Embedding Dimension** | {index_info.get('embedding_dimension', 0)} |")
    lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append(f"*Checkpoint: {checkpoint_id}*")
    lines.append(f"*Judge Model: {config.get('judge_model', 'unknown')}*")
    
    # Write report
    report_content = "\n".join(lines)
    output_path.write_text(report_content)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate checkpoint report from results JSON")
    parser.add_argument("checkpoint_id", nargs="?", help="Checkpoint ID (e.g., C013) or path to results.json")
    parser.add_argument("--path", help="Direct path to results.json")
    parser.add_argument("--list", action="store_true", help="List available checkpoints")
    parser.add_argument("--output", "-o", help="Output path for report (default: alongside results.json)")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
        return
    
    # Determine results path
    if args.path:
        results_path = Path(args.path)
    elif args.checkpoint_id:
        if args.checkpoint_id.endswith(".json"):
            results_path = Path(args.checkpoint_id)
        else:
            folder = find_checkpoint_folder(args.checkpoint_id)
            if folder is None:
                print(f"Error: Checkpoint '{args.checkpoint_id}' not found")
                print("Use --list to see available checkpoints")
                sys.exit(1)
            results_path = folder / "results.json"
    else:
        parser.print_help()
        sys.exit(1)
    
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    # Load gold baseline data for comparison
    registry = load_registry()
    baseline_data = None
    gold_baseline = registry.get("gold_baseline")
    if gold_baseline:
        baseline_folder = CHECKPOINTS_DIR / gold_baseline.get("folder", "")
        baseline_results = baseline_folder / "results.json"
        if baseline_results.exists():
            baseline_data = json.load(open(baseline_results))
            print(f"Comparing against baseline: {gold_baseline.get('id')} ({gold_baseline.get('folder')})")
    
    # Determine output path
    output_path = Path(args.output) if args.output else None
    
    # Generate report
    print(f"Generating report from: {results_path}")
    report_path = generate_report(results_path, output_path, baseline_data)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
