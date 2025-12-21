#!/usr/bin/env python3
"""
Experiment Runner - Exploratory Evaluation with Variable Changes

Runs an experiment evaluation where you can change any parameter.
Use this to test new models, different K values, or other configurations.

Usage:
  python scripts/run_experiment.py              # Interactive (prompts for changes)
  python scripts/run_experiment.py --cloud      # Cloud mode with defaults
  python scripts/run_experiment.py --local      # Local mode with defaults
  python scripts/run_experiment.py --recall 200 # Change recall@K to 200
  python scripts/run_experiment.py --dry-run    # Show config without running
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.core.evaluator import GoldEvaluator, load_corpus
from lib.core.config_loader import load_config

# =============================================================================
# EXPERIMENT CONFIGURATION (FLEXIBLE)
# =============================================================================

def get_experiment_config() -> dict:
    """Load and return the experiment configuration (can be modified)."""
    config = load_config(config_type="experiment")
    
    return {
        # Client & Corpus
        "client": config.get("client", "BFAI"),
        "corpus": config.get("corpus", {}).get("file", "QA_BFAI_gold_v1-0__q458.json"),
        "questions": 458,  # Default to full corpus
        "quick": 0,  # 0 = full, N = run N questions only
        
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
    print("  EXPERIMENT RUNNER - Exploratory Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_config(config: dict, mode: str, changes: list = None):
    """Print the experiment configuration with highlighted changes."""
    print("\n" + "─" * 70)
    print("  EXPERIMENT CONFIGURATION")
    print("─" * 70)
    
    changes = changes or []
    
    def highlight(key, value):
        if key in changes:
            return f"  ⚡ {value} [CHANGED]"
        return f"     {value}"
    
    print(f"\n  Mode: {'☁️  CLOUD' if mode == 'cloud' else '💻 LOCAL'}")
    
    print("\n  Client & Corpus:")
    print(highlight("client", f"Client:     {config['client']}"))
    print(highlight("corpus", f"Corpus:     {config['corpus']}"))
    print(highlight("questions", f"Questions:  {config['questions'] if config['quick'] == 0 else config['quick']}"))
    print(highlight("index", f"Index:      {config['index']}"))
    
    print("\n  Generator Model:")
    print(highlight("generator_model", f"Model:      {config['generator_model']}"))
    print(highlight("generator_reasoning", f"Reasoning:  {config['generator_reasoning']}"))
    print(highlight("generator_temperature", f"Temp:       {config['generator_temperature']}"))
    print(highlight("generator_seed", f"Seed:       {config['generator_seed']}"))
    
    print("\n  Judge Model:")
    print(highlight("judge_model", f"Model:      {config['judge_model']}"))
    print(highlight("judge_reasoning", f"Reasoning:  {config['judge_reasoning']}"))
    print(highlight("judge_temperature", f"Temp:       {config['judge_temperature']}"))
    print(highlight("judge_seed", f"Seed:       {config['judge_seed']}"))
    
    print("\n  Retrieval:")
    print(highlight("recall_k", f"Recall@K:   {config['recall_k']}"))
    print(highlight("precision_k", f"Precision@K: {config['precision_k']}"))
    print(highlight("enable_hybrid", f"Hybrid:     {config['enable_hybrid']} (alpha={config['rrf_alpha']})"))
    print(highlight("enable_reranking", f"Reranking:  {config['enable_reranking']}"))
    print(highlight("ranking_model", f"Ranker:     {config['ranking_model']}"))
    
    print("\n  Execution:")
    print(highlight("workers", f"Workers:    {config['workers']} (smart throttler limits concurrency)"))
    print(highlight("timeout", f"Timeout:    {config['timeout_per_question_s']}s per question"))
    
    print("\n" + "─" * 70)


def interactive_config(config: dict) -> tuple:
    """Interactively configure the experiment. Returns (config, changes, mode)."""
    changes = []
    
    print("\n  Select mode:")
    print("    [1] Cloud (default) - Hit Cloud Run endpoint")
    print("    [2] Local - Use local gRAG_v3 pipeline")
    choice = input("\n  Enter choice [1]: ").strip()
    mode = "local" if choice == "2" else "cloud"
    
    print("\n  Which parameters would you like to change?")
    print("  (Press ENTER to keep default, or type new value)")
    print("─" * 50)
    
    # Recall@K
    recall = input(f"\n  Recall@K [{config['recall_k']}]: ").strip()
    if recall:
        config["recall_k"] = int(recall)
        changes.append("recall_k")
    
    # Precision@K
    precision = input(f"  Precision@K [{config['precision_k']}]: ").strip()
    if precision:
        config["precision_k"] = int(precision)
        changes.append("precision_k")
    
    # Generator model
    print("\n  Generator model options:")
    print("    - gemini-3-flash-preview (default)")
    print("    - gemini-2.5-flash")
    print("    - gemini-2.0-flash")
    model = input(f"  Generator model [{config['generator_model']}]: ").strip()
    if model:
        config["generator_model"] = model
        changes.append("generator_model")
    
    # Reasoning effort
    reasoning = input(f"  Generator reasoning [low/high] [{config['generator_reasoning']}]: ").strip()
    if reasoning and reasoning in ("low", "high"):
        config["generator_reasoning"] = reasoning
        changes.append("generator_reasoning")
    
    # Quick mode
    quick = input(f"  Quick mode (0=full, N=N questions) [{config['quick']}]: ").strip()
    if quick:
        config["quick"] = int(quick)
        if config["quick"] > 0:
            changes.append("questions")
    
    return config, changes, mode


def run_experiment(config: dict, mode: str, changes: list = None, dry_run: bool = False):
    """Run an experiment evaluation."""
    print_header()
    print_config(config, mode, changes)
    
    if dry_run:
        print("\n[DRY RUN - No evaluation performed]")
        return
    
    # Confirm before running
    num_questions = config["quick"] if config["quick"] > 0 else config["questions"]
    print(f"\n⚠️  This will run an EXPERIMENT evaluation ({num_questions} questions).")
    if num_questions == 458:
        print("    Estimated time: 10-15 minutes (cloud), 20-30 minutes (local)")
    confirm = input("\n    Proceed? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("\n    Aborted.")
        return
    
    print("\n🧪 Starting experiment evaluation...")
    print("=" * 70)
    
    # Load corpus
    questions = load_corpus(test_mode=False)
    if config["quick"] > 0:
        questions = questions[:config["quick"]]
        print(f"QUICK MODE: Running {len(questions)} questions only")
    
    # Create evaluator with experiment config
    evaluator = GoldEvaluator(
        precision_k=config["precision_k"],
        recall_k=config["recall_k"],
        workers=config["workers"],
        generator_reasoning=config["generator_reasoning"],
        cloud_mode=(mode == "cloud"),
        model=config["generator_model"],
        config_type="experiment",
    )
    
    # Run evaluation
    output = evaluator.run(questions)
    
    print("\n" + "=" * 70)
    print("🧪 EXPERIMENT COMPLETE")
    print("=" * 70)
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Experiment Runner - Exploratory Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_experiment.py                    # Interactive mode
  python scripts/run_experiment.py --cloud            # Cloud mode with defaults
  python scripts/run_experiment.py --local            # Local mode with defaults
  python scripts/run_experiment.py --recall 200       # Change recall@K to 200
  python scripts/run_experiment.py --precision 50     # Change precision@K to 50
  python scripts/run_experiment.py --model gemini-2.5-flash  # Different model
  python scripts/run_experiment.py --quick 30         # Quick test with 30 questions
  python scripts/run_experiment.py --dry-run          # Show config only
        """
    )
    
    parser.add_argument("--cloud", action="store_true", 
                        help="Run against Cloud Run endpoint")
    parser.add_argument("--local", action="store_true",
                        help="Run against local gRAG_v3 pipeline")
    parser.add_argument("--recall", type=int, default=None,
                        help="Recall@K value (default: 100)")
    parser.add_argument("--precision", type=int, default=None,
                        help="Precision@K value (default: 25)")
    parser.add_argument("--model", type=str, default=None,
                        help="Generator model (default: gemini-3-flash-preview)")
    parser.add_argument("--reasoning", type=str, choices=["low", "high"], default=None,
                        help="Generator reasoning effort (default: low)")
    parser.add_argument("--quick", type=int, default=None,
                        help="Quick mode: run N questions only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show configuration without running")
    
    args = parser.parse_args()
    
    # Load base config
    config = get_experiment_config()
    changes = []
    
    # Apply CLI overrides
    if args.recall is not None:
        config["recall_k"] = args.recall
        changes.append("recall_k")
    if args.precision is not None:
        config["precision_k"] = args.precision
        changes.append("precision_k")
    if args.model is not None:
        config["generator_model"] = args.model
        changes.append("generator_model")
    if args.reasoning is not None:
        config["generator_reasoning"] = args.reasoning
        changes.append("generator_reasoning")
    if args.quick is not None:
        config["quick"] = args.quick
        changes.append("questions")
    
    # Determine mode
    if args.local:
        mode = "local"
    elif args.cloud:
        mode = "cloud"
    elif changes or args.dry_run:
        # If CLI args provided, default to cloud
        mode = "cloud"
    else:
        # Interactive mode
        print_header()
        config, changes, mode = interactive_config(config)
    
    run_experiment(config, mode, changes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
