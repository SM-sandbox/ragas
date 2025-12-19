#!/usr/bin/env python3
"""
Eval Runner - Interactive CLI for RAG Evaluation

Shows current defaults and lets you override any setting before running.
Idempotent: temperature=0.0 ensures same inputs produce same outputs.

Usage:
  python scripts/eval/eval_runner.py              # Interactive mode
  python scripts/eval/eval_runner.py --run        # Run with defaults (no prompts)
  python scripts/eval/eval_runner.py --dry-run    # Show config without running
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/gRAG_v3")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.core.baseline_manager import get_latest_baseline, list_baselines

# =============================================================================
# DEFAULTS - Edit these to change the default configuration
# =============================================================================

DEFAULTS = {
    # Client & Corpus
    "client": "BFAI",
    "corpus": "QA_BFAI_gold_v1-0__q458.json",
    
    # Index
    "job_id": "bfai__eval66a_g1_1536_tt",
    
    # Models
    "generator_model": "gemini-2.5-flash",
    "judge_model": "gemini-2.0-flash",
    "reasoning_effort": "low",  # low | medium | high
    
    # Retrieval
    "precision_k": 25,
    "recall_k": 100,
    "enable_hybrid": True,
    "enable_reranking": True,
    
    # Execution
    "workers": 5,
    "temperature": 0.0,  # MUST be 0.0 for idempotency
    "max_retries": 5,  # Retry attempts per question (no fallback)
    
    # Environment
    "mode": "local",  # "local" or "cloud"
    "endpoint": None,  # Cloud Run endpoint URL (only for cloud mode)
    
    # Run options
    "quick": 0,  # 0 = full run, N = run N questions only
    "test_mode": False,  # True = 30 questions (5 per bucket)
    "update_baseline": False,  # True = save results as new baseline
}

# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_header():
    """Print header with current date/time."""
    print("\n" + "=" * 60)
    print("  EVAL RUNNER - RAG Pipeline Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def print_config(config: dict, title: str = "Current Configuration"):
    """Print configuration in a readable format."""
    print(f"\n{title}")
    print("-" * 40)
    
    sections = {
        "Client & Corpus": ["client", "corpus", "job_id"],
        "Models": ["generator_model", "judge_model", "reasoning_effort"],
        "Retrieval": ["precision_k", "recall_k", "enable_hybrid", "enable_reranking"],
        "Execution": ["workers", "temperature", "max_retries"],
        "Environment": ["mode", "endpoint"],
        "Run Options": ["quick", "test_mode", "update_baseline"],
    }
    
    for section, keys in sections.items():
        print(f"\n  {section}:")
        for key in keys:
            value = config.get(key, "N/A")
            print(f"    {key}: {value}")


def print_baseline_info(client: str):
    """Print info about current baseline."""
    baseline = get_latest_baseline(client)
    if baseline:
        print(f"\n📊 Current Baseline: v{baseline.get('baseline_version')} ({baseline.get('created_date')})")
        metrics = baseline.get("metrics", {})
        print(f"   Pass: {metrics.get('pass_rate', 0):.1%} | "
              f"Partial: {metrics.get('partial_rate', 0):.1%} | "
              f"Fail: {metrics.get('fail_rate', 0):.1%}")
    else:
        print(f"\n⚠️  No baseline found for {client}")


def get_user_input(prompt: str, default, value_type=str):
    """Get user input with default value."""
    if value_type == bool:
        default_str = "y" if default else "n"
        response = input(f"{prompt} [{default_str}]: ").strip().lower()
        if response == "":
            return default
        return response in ("y", "yes", "true", "1")
    else:
        response = input(f"{prompt} [{default}]: ").strip()
        if response == "":
            return default
        return value_type(response)


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

def interactive_config() -> dict:
    """Interactively configure the run."""
    config = DEFAULTS.copy()
    
    print_header()
    print_config(config, "Default Configuration")
    print_baseline_info(config["client"])
    
    print("\n" + "-" * 40)
    print("Press ENTER to keep defaults, or type new value")
    print("-" * 40)
    
    # Quick settings
    print("\n📋 RUN TYPE:")
    
    run_type = input("  Run type [full/quick/test]: ").strip().lower()
    if run_type == "quick":
        config["quick"] = get_user_input("    How many questions?", 10, int)
    elif run_type == "test":
        config["test_mode"] = True
        config["quick"] = 0
    
    # Key overrides
    print("\n⚙️  KEY SETTINGS (press ENTER to keep default):")
    
    config["workers"] = get_user_input("  Workers", config["workers"], int)
    config["precision_k"] = get_user_input("  Precision@K", config["precision_k"], int)
    
    # Model override
    model_change = input("  Change generator model? [n]: ").strip().lower()
    if model_change in ("y", "yes"):
        print("    Options: gemini-2.5-flash, gemini-3-flash-preview")
        config["generator_model"] = get_user_input("    Generator model", config["generator_model"])
    
    # Baseline update
    config["update_baseline"] = get_user_input("  Save as new baseline?", config["update_baseline"], bool)
    
    # Confirm
    print_config(config, "\n✅ Final Configuration")
    
    confirm = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("Aborted.")
        sys.exit(0)
    
    return config


# =============================================================================
# RUN EVALUATION
# =============================================================================

def run_evaluation(config: dict):
    """Run the evaluation with the given config."""
    from evaluations.baseline.run_baseline import run_evaluation as core_run
    
    print("\n🚀 Starting evaluation...")
    print("-" * 40)
    
    output = core_run(
        client=config["client"],
        workers=config["workers"],
        precision_k=config["precision_k"],
        quick=config["quick"],
        test_mode=config["test_mode"],
        update_baseline=config["update_baseline"],
    )
    
    return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive RAG Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_runner.py              # Interactive mode
  python eval_runner.py --run        # Run with defaults
  python eval_runner.py --dry-run    # Show config only
  python eval_runner.py --quick 10   # Quick run with 10 questions
  python eval_runner.py --workers 3  # Override workers
  python eval_runner.py --cloud --endpoint https://orchestrator-xyz.run.app  # Cloud mode
        """
    )
    
    parser.add_argument("--run", action="store_true", 
                        help="Run immediately with defaults (no prompts)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show configuration without running")
    parser.add_argument("--quick", type=int, default=0,
                        help="Quick mode: run N questions only")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 30 questions (5 per bucket)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--precision", type=int, default=None,
                        help="Precision@K setting")
    parser.add_argument("--model", type=str, default=None,
                        help="Generator model override")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Save results as new baseline")
    parser.add_argument("--list-baselines", action="store_true",
                        help="List all baselines and exit")
    parser.add_argument("--cloud", action="store_true",
                        help="Use cloud mode (hit Cloud Run endpoint instead of local)")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Cloud Run endpoint URL (required for --cloud)")
    
    args = parser.parse_args()
    
    # List baselines mode
    if args.list_baselines:
        print("\n📊 Available Baselines:")
        print("-" * 40)
        for b in list_baselines():
            print(f"  {b['client']} v{b['version']} ({b['date']}) - {b['question_count']} questions")
        return
    
    # Build config from defaults + CLI overrides
    config = DEFAULTS.copy()
    
    if args.quick > 0:
        config["quick"] = args.quick
    if args.test:
        config["test_mode"] = True
    if args.workers is not None:
        config["workers"] = args.workers
    if args.precision is not None:
        config["precision_k"] = args.precision
    if args.model is not None:
        config["generator_model"] = args.model
    if args.cloud:
        config["mode"] = "cloud"
        if args.endpoint:
            config["endpoint"] = args.endpoint
        elif not config.get("endpoint"):
            print("❌ Error: --cloud requires --endpoint URL")
            sys.exit(1)
    if args.update_baseline:
        config["update_baseline"] = True
    
    # Dry run mode
    if args.dry_run:
        print_header()
        print_config(config, "Configuration (DRY RUN)")
        print_baseline_info(config["client"])
        print("\n[Dry run - no evaluation performed]")
        return
    
    # Direct run mode (no prompts)
    if args.run or args.quick > 0 or args.test:
        print_header()
        print_config(config, "Running with Configuration")
        run_evaluation(config)
        return
    
    # Interactive mode
    config = interactive_config()
    run_evaluation(config)


if __name__ == "__main__":
    main()
