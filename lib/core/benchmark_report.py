#!/usr/bin/env python3
"""
Benchmark Report Generator

Generates comparison reports between a test run and the gold standard benchmark.
Outputs both JSON (for programmatic use) and Markdown (for human reading).

Usage:
    python scripts/eval/benchmark_report.py --test results.json --benchmark v1.3
    python scripts/eval/benchmark_report.py --test results.json  # Uses latest benchmark
    
Output:
    - reports/gold_standard_eval/report_BFAI__R002__2025-12-19.json
    - reports/gold_standard_eval/final_reports/R002_Benchmark_Report_2025-12-19.md
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
REPORTS_DIR = PROJECT_ROOT / "reports" / "gold_standard_eval"
FINAL_REPORTS_DIR = REPORTS_DIR / "final_reports"

# Thresholds for regression detection
THRESHOLDS = {
    "pass_rate": 0.02,        # 2% regression is significant
    "fail_rate": 0.02,
    "recall_at_100": 0.01,
    "mrr": 0.02,
    "overall_score_avg": 0.1,
    "latency": 1.0,           # 1 second slower is significant
}


def get_next_run_id() -> str:
    """Get the next sequential run ID (R001, R002, etc.)."""
    existing = list(FINAL_REPORTS_DIR.glob("R*_*.md"))
    if not existing:
        return "R001"
    
    # Extract run numbers
    run_nums = []
    for f in existing:
        match = re.match(r"R(\d+)_", f.name)
        if match:
            run_nums.append(int(match.group(1)))
    
    next_num = max(run_nums) + 1 if run_nums else 1
    return f"R{next_num:03d}"


def load_benchmark(client: str, version: Optional[str] = None) -> Tuple[Dict, Path]:
    """Load a benchmark file.
    
    Args:
        client: Client name (e.g., "BFAI")
        version: Benchmark version (e.g., "1.3") or None for latest
        
    Returns:
        Tuple of (benchmark data, benchmark path)
    """
    client_dir = BENCHMARKS_DIR / client
    if not client_dir.exists():
        raise FileNotFoundError(f"No benchmarks found for client: {client}")
    
    if version:
        # Find specific version
        pattern = f"benchmark_{client}_v{version}__*.json"
        matches = list(client_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Benchmark v{version} not found for {client}")
        benchmark_path = matches[0]
    else:
        # Get latest (by filename, which includes date)
        benchmarks = sorted(client_dir.glob(f"benchmark_{client}_v*.json"))
        if not benchmarks:
            raise FileNotFoundError(f"No benchmarks found for {client}")
        benchmark_path = benchmarks[-1]
    
    with open(benchmark_path) as f:
        return json.load(f), benchmark_path


def load_test_results(path: Path) -> Dict:
    """Load test results file."""
    with open(path) as f:
        return json.load(f)


def compare_metrics(benchmark: Dict, test: Dict) -> Dict:
    """Compare metrics between benchmark and test."""
    comparison = {}
    
    b_metrics = benchmark.get("metrics", {})
    t_metrics = test.get("metrics", {})
    
    metric_keys = [
        "pass_rate", "partial_rate", "fail_rate", "acceptable_rate",
        "recall_at_100", "mrr", "overall_score_avg"
    ]
    
    for key in metric_keys:
        b_val = b_metrics.get(key, 0)
        t_val = t_metrics.get(key, 0)
        delta = t_val - b_val
        
        # Determine status
        threshold = THRESHOLDS.get(key, 0.02)
        if key == "fail_rate":
            # For fail rate, higher is worse
            if delta > threshold:
                status = "regression"
            elif delta < -threshold:
                status = "improvement"
            else:
                status = "match"
        else:
            # For other metrics, higher is better
            if delta < -threshold:
                status = "regression"
            elif delta > threshold:
                status = "improvement"
            else:
                status = "match"
        
        comparison[key] = {
            "benchmark": round(b_val, 4),
            "test": round(t_val, 4),
            "delta": round(delta, 4),
            "delta_pct": round(delta * 100, 2),
            "status": status
        }
    
    return comparison


def compare_latency(benchmark: Dict, test: Dict) -> Dict:
    """Compare latency between benchmark and test."""
    b_lat = benchmark.get("latency", {})
    t_lat = test.get("latency", {})
    
    # Total latency
    b_total = b_lat.get("total_avg_s", 0)
    t_total = t_lat.get("total_avg_s", 0)
    
    # Client experience (excluding judge)
    b_client = b_lat.get("client_experience_s", b_total)
    t_client = t_lat.get("client_experience_s", t_total)
    
    # If test doesn't have client_experience_s, calculate it
    if "client_experience_s" not in t_lat and "by_phase" in t_lat:
        judge = t_lat["by_phase"].get("judge_avg_s", 0)
        t_client = t_total - judge
    
    return {
        "total": {
            "benchmark": round(b_total, 2),
            "test": round(t_total, 2),
            "delta": round(t_total - b_total, 2),
            "speedup": round(b_total / t_total, 2) if t_total > 0 else 0
        },
        "client_experience": {
            "benchmark": round(b_client, 2),
            "test": round(t_client, 2),
            "delta": round(t_client - b_client, 2),
            "speedup": round(b_client / t_client, 2) if t_client > 0 else 0
        }
    }


def compare_cost(benchmark: Dict, test: Dict) -> Dict:
    """Compare cost between benchmark and test."""
    b_cost = benchmark.get("cost", {})
    t_cost = test.get("cost", {})
    
    # If test doesn't have cost, estimate from tokens
    if not t_cost and "tokens" in test:
        tokens = test["tokens"]
        t_cost = {
            "total_usd": (
                tokens.get("prompt_total", 0) * 0.075 / 1e6 +
                tokens.get("completion_total", 0) * 0.30 / 1e6 +
                tokens.get("thinking_total", 0) * 0.30 / 1e6
            ),
            "per_question_usd": 0
        }
        q_count = test.get("metrics", {}).get("total", 458)
        t_cost["per_question_usd"] = t_cost["total_usd"] / q_count if q_count > 0 else 0
    
    return {
        "total": {
            "benchmark": round(b_cost.get("total_usd", 0), 4),
            "test": round(t_cost.get("total_usd", 0), 4),
            "delta": round(t_cost.get("total_usd", 0) - b_cost.get("total_usd", 0), 4)
        },
        "per_question": {
            "benchmark": round(b_cost.get("per_question_usd", 0), 6),
            "test": round(t_cost.get("per_question_usd", 0), 6),
            "delta": round(t_cost.get("per_question_usd", 0) - b_cost.get("per_question_usd", 0), 6)
        }
    }


def compare_breakdowns(benchmark: Dict, test: Dict) -> Dict:
    """Compare breakdown by type and difficulty."""
    result = {
        "by_type": {},
        "by_difficulty": {}
    }
    
    # By type
    for qtype in ["single_hop", "multi_hop"]:
        b_data = benchmark.get("breakdown_by_type", {}).get(qtype, {})
        t_data = test.get("breakdown_by_type", {}).get(qtype, {})
        
        b_rate = b_data.get("pass_rate", 0)
        t_rate = t_data.get("pass_rate", 0)
        
        result["by_type"][qtype] = {
            "benchmark_pass_rate": round(b_rate, 4),
            "test_pass_rate": round(t_rate, 4),
            "delta": round(t_rate - b_rate, 4),
            "benchmark_counts": {
                "pass": b_data.get("pass", 0),
                "partial": b_data.get("partial", 0),
                "fail": b_data.get("fail", 0)
            },
            "test_counts": {
                "pass": t_data.get("pass", 0),
                "partial": t_data.get("partial", 0),
                "fail": t_data.get("fail", 0)
            }
        }
    
    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        b_data = benchmark.get("breakdown_by_difficulty", {}).get(diff, {})
        t_data = test.get("breakdown_by_difficulty", {}).get(diff, {})
        
        b_rate = b_data.get("pass_rate", 0)
        t_rate = t_data.get("pass_rate", 0)
        
        result["by_difficulty"][diff] = {
            "benchmark_pass_rate": round(b_rate, 4),
            "test_pass_rate": round(t_rate, 4),
            "delta": round(t_rate - b_rate, 4),
            "benchmark_counts": {
                "pass": b_data.get("pass", 0),
                "partial": b_data.get("partial", 0),
                "fail": b_data.get("fail", 0)
            },
            "test_counts": {
                "pass": t_data.get("pass", 0),
                "partial": t_data.get("partial", 0),
                "fail": t_data.get("fail", 0)
            }
        }
    
    return result


def compare_failures(benchmark: Dict, test: Dict) -> Dict:
    """Compare failures between benchmark and test."""
    b_failures = set(f["question_id"] for f in benchmark.get("failures", []))
    
    # Get test failures from results
    t_failures = set()
    for r in test.get("results", []):
        if r.get("judgment", {}).get("verdict") == "fail":
            t_failures.add(r["question_id"])
    
    return {
        "benchmark_only": sorted(b_failures - t_failures),  # Fixed in test
        "test_only": sorted(t_failures - b_failures),       # New failures
        "both": sorted(b_failures & t_failures),            # Still failing
        "benchmark_count": len(b_failures),
        "test_count": len(t_failures)
    }


def generate_summary(comparison: Dict) -> Dict:
    """Generate summary with verdict and recommendations."""
    regressions = []
    improvements = []
    
    # Check metrics
    for key, data in comparison["metrics"].items():
        if data["status"] == "regression":
            regressions.append(key)
        elif data["status"] == "improvement":
            improvements.append(key)
    
    # Check latency
    if comparison["latency"]["client_experience"]["delta"] > THRESHOLDS["latency"]:
        regressions.append("latency")
    elif comparison["latency"]["client_experience"]["delta"] < -THRESHOLDS["latency"]:
        improvements.append("latency")
    
    # Determine verdict
    if regressions:
        if "pass_rate" in regressions or "fail_rate" in regressions:
            verdict = "FAIL"
        else:
            verdict = "WARN"
    else:
        verdict = "PASS"
    
    # Generate key finding
    if verdict == "PASS":
        if improvements:
            key_finding = f"Test matches or exceeds benchmark. Improvements in: {', '.join(improvements)}"
        else:
            key_finding = "Test matches benchmark on all metrics."
    elif verdict == "WARN":
        key_finding = f"Minor regressions detected in: {', '.join(regressions)}"
    else:
        key_finding = f"Significant regressions in: {', '.join(regressions)}"
    
    # Recommendation
    if verdict == "PASS" and improvements:
        recommendation = "Consider promoting this run to the new benchmark."
    elif verdict == "PASS":
        recommendation = "No action needed. Test meets benchmark standards."
    elif verdict == "WARN":
        recommendation = "Review regressions before deploying. May be acceptable."
    else:
        recommendation = "Do not deploy. Investigate and fix regressions."
    
    return {
        "verdict": verdict,
        "regressions": regressions,
        "improvements": improvements,
        "key_finding": key_finding,
        "recommendation": recommendation
    }


def generate_report_json(
    benchmark: Dict,
    benchmark_path: Path,
    test: Dict,
    test_path: Path,
    run_id: str
) -> Dict:
    """Generate the full report JSON."""
    
    # Compare everything
    metrics_comparison = compare_metrics(benchmark, test)
    latency_comparison = compare_latency(benchmark, test)
    cost_comparison = compare_cost(benchmark, test)
    breakdown_comparison = compare_breakdowns(benchmark, test)
    failures_comparison = compare_failures(benchmark, test)
    
    comparison = {
        "metrics": metrics_comparison,
        "latency": latency_comparison,
        "cost": cost_comparison,
        "breakdown_by_type": breakdown_comparison["by_type"],
        "breakdown_by_difficulty": breakdown_comparison["by_difficulty"]
    }
    
    summary = generate_summary(comparison)
    
    # Build report
    report = {
        "schema_version": "2.0",
        "report_id": run_id,
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "generated_timestamp": datetime.now().isoformat(),
        "client": benchmark.get("client", "BFAI"),
        
        "benchmark": {
            "version": benchmark.get("benchmark_version", "unknown"),
            "type": benchmark.get("benchmark_type", "unknown"),
            "file": benchmark_path.name,
            "date": benchmark.get("created_date", "unknown")
        },
        
        "test": {
            "run_id": run_id,
            "type": test.get("index", {}).get("mode", "local"),
            "file": test_path.name,
            "timestamp": test.get("timestamp", "unknown")
        },
        
        "config_match": check_config_match(benchmark, test),
        "comparison": comparison,
        "summary": summary,
        "failures": failures_comparison,
        
        "files": {
            "benchmark_json": str(benchmark_path),
            "test_json": str(test_path),
            "report_json": "",  # Will be filled in
            "report_markdown": ""  # Will be filled in
        }
    }
    
    return report


def check_config_match(benchmark: Dict, test: Dict) -> Dict:
    """Check if configurations match."""
    b_config = benchmark.get("config", {})
    t_config = test.get("config", {})
    
    params = {}
    all_matched = True
    
    check_keys = [
        ("model", "generator_model"),
        ("reasoning_effort", "reasoning_effort"),
        ("temperature", "temperature"),
        ("recall_top_k", "recall_k"),
        ("precision_top_n", "precision_k"),
    ]
    
    for b_key, t_key in check_keys:
        b_val = b_config.get(b_key)
        t_val = t_config.get(t_key, t_config.get(b_key))
        match = b_val == t_val
        if not match:
            all_matched = False
        params[b_key] = {
            "benchmark": b_val,
            "test": t_val,
            "match": match
        }
    
    return {
        "all_matched": all_matched,
        "parameters": params
    }


def generate_markdown_report(report: Dict) -> str:
    """Generate markdown report from report JSON."""
    
    b = report["benchmark"]
    t = report["test"]
    c = report["comparison"]
    s = report["summary"]
    
    # Status emoji
    verdict_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    status_emoji = {"match": "✅", "improvement": "🏆", "regression": "⚠️"}
    
    md = f"""# Gold Standard Benchmark Report

## {report['report_id']}: Test vs Benchmark Comparison

**Report ID:** {report['report_id']}  
**Date:** {report['generated_date']}  
**Client:** {report['client']}  
**Verdict:** {verdict_emoji.get(s['verdict'], '')} **{s['verdict']}**

---

## Benchmark Reference

| Property | Value |
|----------|-------|
| **Version** | v{b['version']} |
| **Type** | {b['type']} |
| **Date** | {b['date']} |
| **File** | `{b['file']}` |

---

## Test Run

| Property | Value |
|----------|-------|
| **Run ID** | {t['run_id']} |
| **Type** | {t['type']} |
| **Timestamp** | {t['timestamp']} |
| **File** | `{t['file']}` |

---

## Executive Summary

| Metric | Benchmark | Test | Δ | Status |
|--------|-----------|------|---|--------|
"""
    
    # Add metrics
    for key, data in c["metrics"].items():
        status = status_emoji.get(data["status"], "")
        if "rate" in key or key in ["mrr", "recall_at_100"]:
            md += f"| **{key}** | {data['benchmark']:.1%} | {data['test']:.1%} | {data['delta']:+.1%} | {status} |\n"
        else:
            md += f"| **{key}** | {data['benchmark']:.2f} | {data['test']:.2f} | {data['delta']:+.2f} | {status} |\n"
    
    md += f"""
### Key Finding

**{s['key_finding']}**

### Recommendation

{s['recommendation']}

---

## Latency Analysis

| Metric | Benchmark | Test | Δ | Speedup |
|--------|-----------|------|---|---------|
| **Total (with Judge)** | {c['latency']['total']['benchmark']}s | {c['latency']['total']['test']}s | {c['latency']['total']['delta']:+.2f}s | {c['latency']['total']['speedup']}x |
| **Client Experience** | {c['latency']['client_experience']['benchmark']}s | {c['latency']['client_experience']['test']}s | {c['latency']['client_experience']['delta']:+.2f}s | {c['latency']['client_experience']['speedup']}x |

> **Note:** Client Experience excludes judge latency (eval-only). This is what end users experience.

---

## Cost Analysis

| Metric | Benchmark | Test | Δ |
|--------|-----------|------|---|
| **Total Cost** | ${c['cost']['total']['benchmark']:.4f} | ${c['cost']['total']['test']:.4f} | ${c['cost']['total']['delta']:+.4f} |
| **Per Question** | ${c['cost']['per_question']['benchmark']:.6f} | ${c['cost']['per_question']['test']:.6f} | ${c['cost']['per_question']['delta']:+.6f} |

---

## Breakdown by Question Type

| Type | Benchmark | Test | Δ |
|------|-----------|------|---|
"""
    
    for qtype, data in c["breakdown_by_type"].items():
        md += f"| **{qtype}** | {data['benchmark_pass_rate']:.1%} | {data['test_pass_rate']:.1%} | {data['delta']:+.1%} |\n"
    
    md += """
---

## Breakdown by Difficulty

| Difficulty | Benchmark | Test | Δ |
|------------|-----------|------|---|
"""
    
    for diff, data in c["breakdown_by_difficulty"].items():
        md += f"| **{diff}** | {data['benchmark_pass_rate']:.1%} | {data['test_pass_rate']:.1%} | {data['delta']:+.1%} |\n"
    
    # Failures
    f = report["failures"]
    md += f"""
---

## Failures Analysis

| Category | Count | Question IDs |
|----------|-------|--------------|
| **Benchmark Only** (fixed) | {len(f['benchmark_only'])} | {', '.join(f['benchmark_only'][:5])}{'...' if len(f['benchmark_only']) > 5 else ''} |
| **Test Only** (new) | {len(f['test_only'])} | {', '.join(f['test_only'][:5])}{'...' if len(f['test_only']) > 5 else ''} |
| **Both** (persistent) | {len(f['both'])} | {', '.join(f['both'][:5])}{'...' if len(f['both']) > 5 else ''} |

---

## Configuration Match

| Parameter | Benchmark | Test | Match |
|-----------|-----------|------|-------|
"""
    
    for param, data in report["config_match"]["parameters"].items():
        match = "✅" if data["match"] else "❌"
        md += f"| **{param}** | {data['benchmark']} | {data['test']} | {match} |\n"
    
    md += f"""
---

*Report generated: {report['generated_timestamp']}*  
*Benchmark: v{b['version']} ({b['type']})*  
*Report ID: {report['report_id']}*
"""
    
    return md


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument("--test", required=True, help="Path to test results JSON")
    parser.add_argument("--benchmark", help="Benchmark version (e.g., 1.3). Uses latest if not specified.")
    parser.add_argument("--client", default="BFAI", help="Client name (default: BFAI)")
    parser.add_argument("--run-id", help="Run ID (auto-generated if not specified)")
    parser.add_argument("--output-dir", help="Output directory (default: reports/gold_standard_eval)")
    
    args = parser.parse_args()
    
    # Load files
    test_path = Path(args.test)
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        sys.exit(1)
    
    try:
        benchmark, benchmark_path = load_benchmark(args.client, args.benchmark)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    test = load_test_results(test_path)
    
    # Get run ID
    run_id = args.run_id or get_next_run_id()
    
    # Generate report
    print(f"📊 Generating report {run_id}...")
    print(f"   Benchmark: v{benchmark.get('benchmark_version')} ({benchmark_path.name})")
    print(f"   Test: {test_path.name}")
    
    report = generate_report_json(benchmark, benchmark_path, test, test_path, run_id)
    
    # Output paths
    output_dir = Path(args.output_dir) if args.output_dir else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    FINAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    json_path = output_dir / f"report_{args.client}__{run_id}__{date_str}.json"
    md_path = FINAL_REPORTS_DIR / f"{run_id}_Benchmark_Report_{date_str}.md"
    
    # Update file paths in report
    report["files"]["report_json"] = str(json_path)
    report["files"]["report_markdown"] = str(md_path)
    
    # Write JSON
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Write Markdown
    md_content = generate_markdown_report(report)
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    # Summary
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"📋 Report: {run_id}")
    print(f"{'='*60}")
    print(f"Verdict: {s['verdict']}")
    print(f"Key Finding: {s['key_finding']}")
    if s['regressions']:
        print(f"Regressions: {', '.join(s['regressions'])}")
    if s['improvements']:
        print(f"Improvements: {', '.join(s['improvements'])}")
    print(f"\n✅ JSON: {json_path}")
    print(f"✅ Markdown: {md_path}")


if __name__ == "__main__":
    main()
