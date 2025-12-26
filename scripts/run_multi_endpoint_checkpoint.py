#!/usr/bin/env python3
"""
Multi-Endpoint Checkpoint Runner

Runs checkpoint evaluations against multiple Cloud Run endpoints sequentially,
then generates a comparison report against the gold baseline.

Usage:
  python scripts/run_multi_endpoint_checkpoint.py              # Run all endpoints
  python scripts/run_multi_endpoint_checkpoint.py --quick 36   # Quick mode (36 questions)
  python scripts/run_multi_endpoint_checkpoint.py --dry-run    # Show config only
"""

import os
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.core.evaluator import GoldEvaluator, load_corpus
from lib.core.config_loader import load_config

# =============================================================================
# ENDPOINT CONFIGURATION
# =============================================================================

ENDPOINTS = [
    {
        "name": "prod",
        "url": "https://bfai-app-ppfq5ahfsq-ue.a.run.app",
        "description": "Production bfai-app",
        "project": "bfai-prod",
    },
    {
        "name": "stage",
        "url": "https://bfai-app-xcshqh7sqq-ue.a.run.app",
        "description": "Staging bfai-app",
        "project": "bfai-stage",
    },
    {
        "name": "dev",
        "url": "https://bfai-app-eb2qyzdzvq-ue.a.run.app",
        "description": "Development bfai-app",
        "project": "bfai-dev",
    },
    {
        "name": "grag-v3-dev",
        "url": "https://grag-v3-app-eb2qyzdzvq-uc.a.run.app",
        "description": "gRAG v3 refactor (us-central)",
        "project": "bfai-dev",
    },
]

# Gold baseline for comparison
GOLD_BASELINE = {
    "id": "C034",
    "pass_rate": 0.932,
    "acceptable_rate": 0.976,
    "fail_rate": 0.024,
    "mrr": 0.74,
    "recall_at_100": 0.991,
    "overall_score_avg": 4.81,
}


def print_header():
    """Print header with current date/time."""
    print("\n" + "=" * 70)
    print("  MULTI-ENDPOINT CHECKPOINT RUNNER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_endpoints():
    """Print the endpoints to be tested."""
    print(f"\n{'─' * 70}")
    print("  ENDPOINTS TO TEST")
    print(f"{'─' * 70}")
    for i, ep in enumerate(ENDPOINTS, 1):
        print(f"\n  [{i}] {ep['name']}")
        print(f"      URL: {ep['url']}")
        print(f"      Description: {ep['description']}")
    print(f"\n{'─' * 70}")


def run_single_endpoint(endpoint: dict, questions: list, config: dict) -> dict:
    """Run checkpoint evaluation against a single endpoint."""
    print(f"\n{'='*70}")
    print(f"  RUNNING: {endpoint['name']}")
    print(f"  URL: {endpoint['url']}")
    print(f"{'='*70}\n")
    
    try:
        evaluator = GoldEvaluator(
            precision_k=config.get("retrieval", {}).get("precision_k", 25),
            recall_k=config.get("retrieval", {}).get("recall_k", 100),
            workers=config.get("execution", {}).get("workers", 100),
            generator_reasoning=config.get("generator", {}).get("reasoning_effort", "low"),
            cloud_mode=True,
            model=config.get("generator", {}).get("model", "gemini-3-flash-preview"),
            config_type="checkpoint",
            cloud_url=endpoint["url"],
            endpoint_name=endpoint["name"],
        )
        
        output = evaluator.run(questions)
        
        # Extract key metrics
        metrics = output.get("metrics", {})
        result = {
            "endpoint": endpoint["name"],
            "url": endpoint["url"],
            "status": "success",
            "pass_rate": metrics.get("pass_rate", 0),
            "acceptable_rate": metrics.get("acceptable_rate", 0),
            "fail_rate": metrics.get("fail_rate", 0),
            "partial_rate": metrics.get("partial_rate", 0),
            "mrr": metrics.get("mrr", 0),
            "recall_at_100": metrics.get("recall_at_100", 0),
            "overall_score_avg": metrics.get("overall_score_avg", 0),
            "total_questions": metrics.get("total", 0),
            "run_duration_s": output.get("run_duration_seconds", 0),
            "total_cost_usd": output.get("total_cost_usd", 0),
            "latency_avg_s": output.get("latency", {}).get("total_avg_s", 0),
            "results_file": str(evaluator.results_file) if hasattr(evaluator, 'results_file') else None,
        }
        
        print(f"\n✅ {endpoint['name']} completed: pass_rate={result['pass_rate']:.1%}")
        return result
        
    except Exception as e:
        print(f"\n❌ {endpoint['name']} FAILED: {e}")
        return {
            "endpoint": endpoint["name"],
            "url": endpoint["url"],
            "status": "failed",
            "error": str(e),
        }


def generate_comparison_report(results: list, output_dir: Path) -> Path:
    """Generate a markdown comparison report."""
    report_path = output_dir / f"multi_endpoint_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, "w") as f:
        f.write("# Multi-Endpoint Checkpoint Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Gold Baseline:** C034 (pass_rate={GOLD_BASELINE['pass_rate']:.1%}, acceptable={GOLD_BASELINE['acceptable_rate']:.1%})\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Endpoint | Status | Pass Rate | Δ Baseline | Acceptable | Fail | MRR | Latency |\n")
        f.write("|----------|--------|-----------|------------|------------|------|-----|--------|\n")
        
        for r in results:
            if r["status"] == "success":
                delta = r["pass_rate"] - GOLD_BASELINE["pass_rate"]
                delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
                delta_emoji = "🟢" if delta >= 0 else "🔴"
                f.write(f"| {r['endpoint']} | ✅ | {r['pass_rate']:.1%} | {delta_emoji} {delta_str} | {r['acceptable_rate']:.1%} | {r['fail_rate']:.1%} | {r['mrr']:.3f} | {r['latency_avg_s']:.1f}s |\n")
            else:
                f.write(f"| {r['endpoint']} | ❌ FAILED | - | - | - | - | - | - |\n")
        
        # Detailed results
        f.write("\n## Detailed Results\n\n")
        for r in results:
            f.write(f"### {r['endpoint']}\n\n")
            if r["status"] == "success":
                f.write(f"- **URL:** `{r['url']}`\n")
                f.write(f"- **Pass Rate:** {r['pass_rate']:.1%} (baseline: {GOLD_BASELINE['pass_rate']:.1%})\n")
                f.write(f"- **Acceptable Rate:** {r['acceptable_rate']:.1%}\n")
                f.write(f"- **Fail Rate:** {r['fail_rate']:.1%}\n")
                f.write(f"- **MRR:** {r['mrr']:.3f}\n")
                f.write(f"- **Recall@100:** {r['recall_at_100']:.1%}\n")
                f.write(f"- **Overall Score Avg:** {r['overall_score_avg']:.2f}\n")
                f.write(f"- **Questions:** {r['total_questions']}\n")
                f.write(f"- **Duration:** {r['run_duration_s']:.1f}s\n")
                f.write(f"- **Cost:** ${r['total_cost_usd']:.4f}\n")
                f.write(f"- **Avg Latency:** {r['latency_avg_s']:.1f}s\n")
                if r.get("results_file"):
                    f.write(f"- **Results File:** `{r['results_file']}`\n")
            else:
                f.write(f"- **Status:** FAILED\n")
                f.write(f"- **Error:** {r.get('error', 'Unknown')}\n")
            f.write("\n")
        
        # Recommendations
        f.write("## Analysis\n\n")
        
        # Find grag-v3 result for special attention
        grag_result = next((r for r in results if r["endpoint"] == "grag-v3-dev"), None)
        if grag_result and grag_result["status"] == "success":
            delta = grag_result["pass_rate"] - GOLD_BASELINE["pass_rate"]
            if delta >= 0:
                f.write(f"### gRAG v3 Refactor: 🟢 PASS\n\n")
                f.write(f"The gRAG v3 refactor shows **{delta:+.1%}** improvement over baseline.\n\n")
            elif delta > -0.02:
                f.write(f"### gRAG v3 Refactor: 🟡 ACCEPTABLE\n\n")
                f.write(f"The gRAG v3 refactor shows **{delta:.1%}** vs baseline (within tolerance).\n\n")
            else:
                f.write(f"### gRAG v3 Refactor: 🔴 REGRESSION\n\n")
                f.write(f"The gRAG v3 refactor shows **{delta:.1%}** regression vs baseline.\n\n")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Endpoint Checkpoint Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", type=int, default=0,
                        help="Quick mode: run N questions only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show configuration without running")
    parser.add_argument("--endpoints", type=str, default="all",
                        help="Comma-separated endpoint names to run (default: all)")
    
    args = parser.parse_args()
    
    print_header()
    print_endpoints()
    
    # Load checkpoint config
    config = load_config(config_type="checkpoint")
    
    # Filter endpoints if specified
    if args.endpoints != "all":
        endpoint_names = [e.strip() for e in args.endpoints.split(",")]
        endpoints_to_run = [e for e in ENDPOINTS if e["name"] in endpoint_names]
    else:
        endpoints_to_run = ENDPOINTS
    
    # Load corpus
    questions = load_corpus(test_mode=False)
    if args.quick > 0:
        questions = questions[:args.quick]
        print(f"\nQUICK MODE: Running {len(questions)} questions")
    else:
        print(f"\nFULL MODE: Running {len(questions)} questions")
    
    if args.dry_run:
        print("\n[DRY RUN - No evaluation performed]")
        return
    
    # Confirm
    print(f"\n⚠️  This will run {len(endpoints_to_run)} checkpoints × {len(questions)} questions each.")
    print(f"    Estimated time: {len(endpoints_to_run) * 15} minutes (sequential)")
    confirm = input("\n    Proceed? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("\n    Aborted.")
        return
    
    # Run each endpoint sequentially
    results = []
    for endpoint in endpoints_to_run:
        result = run_single_endpoint(endpoint, questions, config)
        results.append(result)
    
    # Generate comparison report
    output_dir = Path(__file__).parent.parent / "reports" / "multi_endpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = generate_comparison_report(results, output_dir)
    
    # Save raw results as JSON
    results_path = output_dir / f"multi_endpoint_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "baseline": GOLD_BASELINE,
            "results": results,
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  MULTI-ENDPOINT CHECKPOINT COMPLETE")
    print("=" * 70)
    print(f"\n  Report: {report_path}")
    print(f"  Results: {results_path}")
    print("\n  Summary:")
    for r in results:
        if r["status"] == "success":
            delta = r["pass_rate"] - GOLD_BASELINE["pass_rate"]
            emoji = "🟢" if delta >= 0 else "🔴"
            print(f"    {emoji} {r['endpoint']}: {r['pass_rate']:.1%} (Δ {delta:+.1%})")
        else:
            print(f"    ❌ {r['endpoint']}: FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
