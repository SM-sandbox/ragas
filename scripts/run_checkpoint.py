#!/usr/bin/env python3
"""
Checkpoint Runner - Standard CI/CD Evaluation

Runs a checkpoint evaluation with LOCKED configuration.
The ONLY variable that changes is: local vs cloud mode.

Usage:
  python scripts/run_checkpoint.py              # Interactive (prompts for mode)
  python scripts/run_checkpoint.py --cloud      # Cloud mode (default)
  python scripts/run_checkpoint.py --local      # Local mode
  python scripts/run_checkpoint.py --dry-run    # Show config without running
"""

# Suppress verbose gRPC/absl logs BEFORE importing google libraries
import os
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")  # 0=INFO, 1=WARNING, 2=ERROR

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.core.evaluator import GoldEvaluator, load_corpus
from lib.core.config_loader import load_config

# =============================================================================
# CHECKPOINT CONFIGURATION (LOCKED)
# =============================================================================

def get_checkpoint_config() -> dict:
    """Load and return the locked checkpoint configuration."""
    config = load_config(config_type="checkpoint")
    
    # Build display config
    return {
        # Client & Corpus
        "client": config.get("client", "BFAI"),
        "corpus": config.get("corpus", {}).get("file", "QA_BFAI_gold_v1-0__q458.json"),
        "questions": 458,  # Full corpus
        
        # Index
        "index": config.get("index", {}).get("job_id", "bfai__eval66a_g1_1536_tt"),
        
        # Generator
        "generator_model": config.get("generator", {}).get("model", "gemini-3-flash-preview"),
        "generator_reasoning": config.get("generator", {}).get("reasoning_effort", "low"),
        "generator_temperature": config.get("generator", {}).get("temperature", 0.0),
        "generator_seed": config.get("generator", {}).get("seed", 42),
        
        # Judge
        "judge_model": config.get("judge", {}).get("model", "gemini-3-flash-preview"),
        "judge_reasoning": config.get("judge", {}).get("reasoning_effort", "low"),
        "judge_temperature": config.get("judge", {}).get("temperature", 0.0),
        "judge_seed": config.get("judge", {}).get("seed", 42),
        
        # Retrieval
        "recall_k": config.get("retrieval", {}).get("recall_k", 100),
        "precision_k": config.get("retrieval", {}).get("precision_k", 25),
        "enable_hybrid": config.get("retrieval", {}).get("enable_hybrid", True),
        "rrf_alpha": config.get("retrieval", {}).get("rrf_alpha", 0.5),
        "enable_reranking": config.get("retrieval", {}).get("enable_reranking", True),
        "ranking_model": config.get("retrieval", {}).get("ranking_model", "semantic-ranker-512@latest"),
        
        # Execution
        "workers": config.get("execution", {}).get("workers", 100),
        "timeout_per_question_s": config.get("execution", {}).get("timeout_per_question_s", 120),
    }


def print_header():
    """Print header with current date/time."""
    print("\n" + "=" * 70)
    print("  CHECKPOINT RUNNER - Standard CI/CD Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_config(config: dict, mode: str):
    """Print the locked checkpoint configuration."""
    print(f"\n{'─' * 70}")
    print("  LOCKED CHECKPOINT CONFIGURATION")
    print(f"{'─' * 70}")
    
    print(f"\n  Mode: {'☁️  CLOUD' if mode == 'cloud' else '💻 LOCAL'}")
    
    print(f"\n  Client & Corpus:")
    print(f"    Client:     {config['client']}")
    print(f"    Corpus:     {config['corpus']}")
    print(f"    Questions:  {config['questions']}")
    print(f"    Index:      {config['index']}")
    
    print(f"\n  Generator Model:")
    print(f"    Model:      {config['generator_model']}")
    print(f"    Reasoning:  {config['generator_reasoning']}")
    print(f"    Temp:       {config['generator_temperature']}")
    print(f"    Seed:       {config['generator_seed']}")
    
    print(f"\n  Judge Model:")
    print(f"    Model:      {config['judge_model']}")
    print(f"    Reasoning:  {config['judge_reasoning']}")
    print(f"    Temp:       {config['judge_temperature']}")
    print(f"    Seed:       {config['judge_seed']}")
    
    print(f"\n  Retrieval:")
    print(f"    Recall@K:   {config['recall_k']}")
    print(f"    Precision@K: {config['precision_k']}")
    print(f"    Hybrid:     {config['enable_hybrid']} (alpha={config['rrf_alpha']})")
    print(f"    Reranking:  {config['enable_reranking']}")
    print(f"    Ranker:     {config['ranking_model']}")
    
    print(f"\n  Execution:")
    print(f"    Workers:    {config['workers']} (smart throttler limits concurrency)")
    print(f"    Timeout:    {config['timeout_per_question_s']}s per question")
    
    print(f"\n{'─' * 70}")


def run_checkpoint(mode: str, dry_run: bool = False):
    """Run a checkpoint evaluation."""
    config = get_checkpoint_config()
    
    # Get sample_size from config (for quick checkpoint runs)
    raw_config = load_config(config_type="checkpoint")
    sample_size = raw_config.get("corpus", {}).get("sample_size")
    
    print_header()
    print_config(config, mode)
    
    if dry_run:
        print("\n[DRY RUN - No evaluation performed]")
        return
    
    # Confirm before running
    num_questions = sample_size if sample_size else 458
    print(f"\n⚠️  This will run a checkpoint evaluation ({num_questions} questions).")
    print("    Estimated time: 10-15 minutes (cloud), 20-30 minutes (local)")
    confirm = input("\n    Proceed? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("\n    Aborted.")
        return
    
    print("\n🚀 Starting checkpoint evaluation...")
    print("=" * 70)
    
    # Load corpus and optionally sample
    questions = load_corpus(test_mode=False)
    if sample_size and sample_size < len(questions):
        questions = questions[:sample_size]
        print(f"Using {sample_size} questions (from config sample_size)")
    
    # Create evaluator with checkpoint config
    evaluator = GoldEvaluator(
        precision_k=config["precision_k"],
        recall_k=config["recall_k"],
        workers=config["workers"],
        generator_reasoning=config["generator_reasoning"],
        cloud_mode=(mode == "cloud"),
        model=config["generator_model"],
        config_type="checkpoint",
    )
    
    # Run evaluation
    output = evaluator.run(questions)
    
    print("\n" + "=" * 70)
    print("✅ CHECKPOINT COMPLETE")
    print("=" * 70)
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint Runner - Standard CI/CD Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_checkpoint.py              # Interactive mode
  python scripts/run_checkpoint.py --cloud      # Cloud mode (default)
  python scripts/run_checkpoint.py --local      # Local mode
  python scripts/run_checkpoint.py --dry-run    # Show config only
        """
    )
    
    parser.add_argument("--cloud", action="store_true", 
                        help="Run against Cloud Run endpoint (default)")
    parser.add_argument("--local", action="store_true",
                        help="Run against local gRAG_v3 pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show configuration without running")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.local:
        mode = "local"
    elif args.cloud:
        mode = "cloud"
    else:
        # Interactive mode
        print_header()
        print("\n  Select mode:")
        print("    [1] Cloud (default) - Hit Cloud Run endpoint")
        print("    [2] Local - Use local gRAG_v3 pipeline")
        choice = input("\n  Enter choice [1]: ").strip()
        mode = "local" if choice == "2" else "cloud"
    
    run_checkpoint(mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
