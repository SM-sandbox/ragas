#!/usr/bin/env python3
"""
Generate a sample report from context_100 experiment using the new report template.
"""

import sys
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")

from core.report import (
    RetrievalMetrics,
    ReportConfig,
    generate_report,
    save_report,
    create_report_from_results,
)
from core.preflight import run_preflight_checks, PreflightConfig
from core.metrics import compute_retrieval_metrics


def load_experiment_results(experiment_name: str) -> list:
    """Load results from experiment checkpoint file."""
    base_dir = Path(__file__).parent.parent
    checkpoint_file = base_dir / "experiments" / "2025-12-15_temp_context_sweep" / f"{experiment_name}_checkpoint.jsonl"
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    
    results = []
    with open(checkpoint_file) as f:
        for line in f:
            results.append(json.loads(line))
    
    return results


def main():
    print("=" * 70)
    print("GENERATING SAMPLE REPORT: context_100")
    print("=" * 70)
    
    # Load results
    print("\n1. Loading experiment results...")
    results = load_experiment_results("context_100")
    print(f"   Loaded {len(results)} results")
    
    # Run pre-flight checks (quick version)
    print("\n2. Running pre-flight checks...")
    preflight_config = PreflightConfig(
        job_id="bfai__eval66a_g1_1536_tt",
        corpus_path=str(Path(__file__).parent.parent / "corpus" / "qa_corpus_200.json"),
        model_id="gemini-2.5-flash",
        skip_api_check=True,  # Skip for speed
    )
    preflight_result = run_preflight_checks(preflight_config)
    
    for check in preflight_result.checks:
        print(f"   {check.icon} {check.name}: {check.message}")
    
    # Create report from results
    print("\n3. Creating report...")
    report = create_report_from_results(
        experiment_name="Context Size 100 Experiment",
        model_id="gemini-2.5-flash",
        results=results,
        config_overrides={
            "temperature": 0.0,
            "context_size": 100,
            "embedding_model": "gemini-1536-RETRIEVAL_QUERY",
            "job_id": "bfai__eval66a_g1_1536_tt",
            "corpus_name": "qa_corpus_200",
            "recall_top_k": 100,
            "reranker_model": "semantic-ranker-default@latest",
            "judge_model": "gemini-2.5-flash",
            "judge_temperature": 0.0,
        },
    )
    
    # Add preflight results
    report.preflight = preflight_result
    
    # Compute REAL retrieval metrics from cache
    print("\n3b. Computing retrieval metrics from cache...")
    cache_file = Path(__file__).parent.parent / "experiments" / "2025-12-15_temp_context_sweep" / "retrieval_cache.json"
    with open(cache_file) as f:
        retrieval_cache = json.load(f)
    
    retrieval_results = list(retrieval_cache.values())
    metrics = compute_retrieval_metrics(retrieval_results, k_values=[5, 10, 15, 20, 25, 50, 100])
    
    print(f"   Recall@100: {metrics.recall_at_k[100]*100:.1f}%")
    print(f"   Precision@10: {metrics.precision_at_k[10]*100:.1f}%")
    print(f"   MRR@10: {metrics.mrr_at_k[10]:.3f}")
    
    report.retrieval_metrics = RetrievalMetrics(
        recall_at_100=metrics.recall_at_k[100],
        precision_at_5=metrics.precision_at_k[5],
        precision_at_10=metrics.precision_at_k[10],
        precision_at_15=metrics.precision_at_k[15],
        precision_at_20=metrics.precision_at_k[20],
        precision_at_25=metrics.precision_at_k[25],
        mrr_at_10=metrics.mrr_at_k[10],
    )
    
    # Generate markdown report
    print("\n4. Generating markdown report...")
    report_config = ReportConfig(
        output_dir=str(Path(__file__).parent.parent / "reports"),
        include_preflight=True,
        format="both",
    )
    
    saved = save_report(
        report=report,
        output_dir=report_config.output_dir,
        filename="context_100_NEW_FORMAT",
        config=report_config,
    )
    
    print(f"\n5. Reports saved:")
    for fmt, path in saved.items():
        print(f"   {fmt}: {path}")
    
    # Print the markdown to console
    print("\n" + "=" * 70)
    print("GENERATED REPORT PREVIEW")
    print("=" * 70)
    
    md_content = generate_report(report, report_config)
    print(md_content)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
