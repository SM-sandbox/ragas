#!/usr/bin/env python3
"""
Core Eval - RAG Pipeline Evaluation with Baseline Comparison

Main entry point for running RAG evaluations and comparing to baselines.
Outputs results as JSONL for easy processing and generates comparison reports.

Usage:
  python scripts/eval/core_eval.py --client BFAI --workers 5
  python scripts/eval/core_eval.py --client BFAI --quick 10  # Test with 10 questions
  python scripts/eval/core_eval.py --client BFAI --update-baseline  # Save as new baseline
"""

import sys
import json
import time
import argparse
import uuid
import socket
import platform
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/gRAG_v3")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.core.baseline_manager import get_latest_baseline, save_baseline, compare_to_baseline, format_comparison_report
from lib.core.cost_calculator import calculate_run_cost

# Import the evaluator
from lib.core.evaluator import GoldEvaluator, load_corpus, OUTPUT_DIR

# Directories
RUNS_DIR = Path(__file__).parent.parent.parent / "runs"
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "core_eval"


def generate_run_id() -> str:
    """Generate unique run ID."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def get_run_folder(run_id: str, config: dict) -> Path:
    """Generate run folder path."""
    model = config.get("generator_model", "unknown").replace("/", "_")
    precision = config.get("precision_k", 25)
    date = datetime.now().strftime("%Y-%m-%d")
    folder_name = f"{date}__{model}__p{precision}__{run_id[-8:]}"
    return RUNS_DIR / folder_name


def save_jsonl(results: list, path: Path):
    """Save results as JSONL (one JSON object per line)."""
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def run_evaluation(
    client: str = "BFAI",
    workers: int = 5,
    precision_k: int = 25,
    quick: int = 0,
    test_mode: bool = False,
    update_baseline: bool = False,
) -> dict:
    """
    Run full evaluation and compare to baseline.
    
    Args:
        client: Client name for baseline lookup
        workers: Number of parallel workers
        precision_k: Precision@K setting
        quick: If > 0, run only this many questions
        test_mode: If True, run 30 questions (5 per bucket)
        update_baseline: If True, save results as new baseline
        
    Returns:
        Evaluation output dict
    """
    run_id = generate_run_id()
    run_start = time.time()
    
    print(f"\n{'='*60}")
    print(f"CORE EVAL - {client}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")
    
    # Load baseline
    baseline = get_latest_baseline(client)
    if baseline:
        print(f"Baseline: v{baseline.get('baseline_version')} ({baseline.get('created_date')})")
        print(f"  Pass rate: {baseline.get('metrics', {}).get('pass_rate', 0):.1%}")
    else:
        print("No baseline found - this will be the first run")
    
    # Load corpus
    questions = load_corpus(test_mode=test_mode)
    if quick > 0:
        questions = questions[:quick]
        print(f"QUICK MODE: Running {len(questions)} questions")
    
    # Run evaluation
    evaluator = GoldEvaluator(precision_k=precision_k, workers=workers)
    output = evaluator.run(questions)
    
    # Add run metadata
    output["run_id"] = run_id
    output["execution"]["hostname"] = socket.gethostname()
    output["execution"]["python_version"] = platform.python_version()
    output["execution"]["run_duration_seconds"] = time.time() - run_start
    
    # Calculate cost
    tokens = output.get("tokens", {})
    generator_model = output.get("config", {}).get("generator_model", "gemini-2.5-flash")
    cost = calculate_run_cost(
        total_prompt_tokens=tokens.get("prompt_total", 0),
        total_completion_tokens=tokens.get("completion_total", 0),
        total_thinking_tokens=tokens.get("thinking_total", 0),
        total_cached_tokens=tokens.get("cached_total", 0),
        model=generator_model,
        question_count=len(questions),
    )
    output["cost"] = cost
    
    # Create run folder and save outputs
    run_folder = get_run_folder(run_id, output.get("config", {}))
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Save summary JSON
    summary_path = run_folder / "run_summary.json"
    summary = {k: v for k, v in output.items() if k != "results"}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save results as JSONL
    jsonl_path = run_folder / "results.jsonl"
    save_jsonl(output.get("results", []), jsonl_path)
    
    print(f"\nRun saved: {run_folder}")
    print(f"  Summary: {summary_path.name}")
    print(f"  Results: {jsonl_path.name}")
    
    # Compare to baseline
    if baseline:
        comparison = compare_to_baseline(output, baseline)
        
        print(f"\n{'='*60}")
        print("COMPARISON TO BASELINE")
        print(f"{'='*60}")
        
        # Show key deltas
        for key in ["pass_rate", "fail_rate", "acceptable_rate"]:
            if key in comparison["deltas"]:
                delta = comparison["deltas"][key]
                sign = "+" if delta["delta"] > 0 else ""
                print(f"  {key}: {delta['baseline']:.1%} → {delta['current']:.1%} ({sign}{delta['delta']:.1%})")
        
        # Show regressions/improvements
        if comparison["regressions"]:
            print(f"\n⚠️  REGRESSIONS: {', '.join(comparison['regressions'])}")
        if comparison["improvements"]:
            print(f"✅ IMPROVEMENTS: {', '.join(comparison['improvements'])}")
        
        # Save comparison report
        report_md = format_comparison_report(comparison)
        report_path = run_folder / "comparison.md"
        with open(report_path, "w") as f:
            f.write(f"# Core Eval Report\n\n")
            f.write(f"**Run ID:** {run_id}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Client:** {client}\n\n")
            f.write(report_md)
        
        output["comparison"] = comparison
    
    # Update baseline if requested
    if update_baseline:
        # Convert output to baseline format
        baseline_data = {
            "schema_version": "1.0",
            "baseline_version": None,  # Auto-incremented
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "client": client,
            "environment": {"type": "local", "orchestrator_url": None},
            "index": output.get("index", {}),
            "corpus": {
                "file": "QA_BFAI_gold_v1-0__q458.json",
                "question_count": len(questions),
            },
            "config": output.get("config", {}),
            "metrics": output.get("metrics", {}),
            "latency": output.get("latency", {}),
            "tokens": output.get("tokens", {}),
            "cost": output.get("cost", {}),
            "answer_stats": output.get("answer_stats", {}),
            "execution": output.get("execution", {}),
            "quality": output.get("quality", {}),
        }
        
        baseline_path = save_baseline(baseline_data, client)
        print(f"\n✅ Baseline saved: {baseline_path}")
    
    # Print cost summary
    print(f"\n{'='*60}")
    print("COST SUMMARY")
    print(f"{'='*60}")
    print(f"Total cost: ${cost['total_cost']:.4f}")
    print(f"Per question: ${cost['cost_per_question']:.6f}")
    print(f"{'='*60}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Core RAG Evaluation")
    parser.add_argument("--client", type=str, default="BFAI", help="Client name")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")
    parser.add_argument("--precision", type=int, default=25, help="Precision@K")
    parser.add_argument("--quick", type=int, default=0, help="Quick test: N questions only")
    parser.add_argument("--test", action="store_true", help="Test mode: 30 questions")
    parser.add_argument("--update-baseline", action="store_true", help="Save as new baseline")
    args = parser.parse_args()
    
    run_evaluation(
        client=args.client,
        workers=args.workers,
        precision_k=args.precision,
        quick=args.quick,
        test_mode=args.test,
        update_baseline=args.update_baseline,
    )


if __name__ == "__main__":
    main()
