#!/usr/bin/env python3
"""
CI/CD Evaluation Runner

Runs reduced evaluation for CI gating with strict thresholds.
Exit code 0 = pass, 1 = regression detected.

Usage:
  python evaluations/cicd/run_cicd_eval.py --quick 30          # Quick regression check
  python evaluations/cicd/run_cicd_eval.py --validate-config   # Config validation only
  python evaluations/cicd/run_cicd_eval.py --check-imports     # Import check only
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Thresholds for CI pass/fail
THRESHOLDS = {
    "pass_rate": 0.85,      # Minimum pass rate (85%)
    "fail_rate": 0.08,      # Maximum fail rate (8%)
    "error_count": 0,       # Maximum errors allowed
}


def check_imports() -> bool:
    """Validate all imports resolve correctly."""
    print("Checking imports...")
    errors = []
    
    # Core imports
    try:
        from lib.core.baseline_manager import list_baselines, get_latest_baseline
        print("  ✓ lib.core.baseline_manager")
    except ImportError as e:
        errors.append(f"lib.core.baseline_manager: {e}")
    
    try:
        from lib.core.cost_calculator import calculate_run_cost
        print("  ✓ lib.core.cost_calculator")
    except ImportError as e:
        errors.append(f"lib.core.cost_calculator: {e}")
    
    try:
        from lib.core.evaluator import GoldEvaluator, load_corpus
        print("  ✓ lib.core.evaluator")
    except ImportError as e:
        errors.append(f"lib.core.evaluator: {e}")
    
    # Client imports
    try:
        from lib.clients.gemini_client import get_model_info
        print("  ✓ lib.clients.gemini_client")
    except ImportError as e:
        errors.append(f"lib.clients.gemini_client: {e}")
    
    # Utils imports
    try:
        from lib.utils.metrics import compute_retrieval_metrics
        print("  ✓ lib.utils.metrics")
    except ImportError as e:
        errors.append(f"lib.utils.metrics: {e}")
    
    try:
        from lib.utils.models import get_model, get_approved_models
        print("  ✓ lib.utils.models")
    except ImportError as e:
        errors.append(f"lib.utils.models: {e}")
    
    # Entry point imports
    try:
        from eval_runners.baseline.run_baseline import run_evaluation
        print("  ✓ eval_runners.baseline.run_baseline")
    except ImportError as e:
        errors.append(f"evaluations.baseline.run_baseline: {e}")
    
    if errors:
        print(f"\n❌ Import errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("\n✓ All imports OK")
    return True


def validate_config() -> bool:
    """Validate configuration files load correctly."""
    print("Validating configuration...")
    errors = []
    
    # Check eval_config.yaml
    config_path = Path(__file__).parent.parent.parent / "config" / "eval_config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            if not config:
                errors.append("eval_config.yaml is empty")
            elif "generator" not in config:
                errors.append("eval_config.yaml missing 'generator' section")
            else:
                print(f"  ✓ eval_config.yaml (schema_version: {config.get('schema_version', 'unknown')})")
        except Exception as e:
            errors.append(f"eval_config.yaml: {e}")
    else:
        errors.append(f"eval_config.yaml not found at {config_path}")
    
    # Check baselines exist
    baselines_dir = Path(__file__).parent.parent.parent / "baselines"
    if baselines_dir.exists():
        baselines = list(baselines_dir.glob("baseline_*.json"))
        if baselines:
            print(f"  ✓ Found {len(baselines)} baseline(s)")
        else:
            errors.append("No baseline files found")
    else:
        errors.append(f"Baselines directory not found at {baselines_dir}")
    
    # Check corpus exists
    corpus_path = Path(__file__).parent.parent.parent / "clients" / "BFAI" / "qa" / "QA_BFAI_gold_v1-0__q458.json"
    if corpus_path.exists():
        try:
            with open(corpus_path) as f:
                corpus = json.load(f)
            questions = corpus.get("questions", corpus)
            print(f"  ✓ Corpus: {len(questions)} questions")
        except Exception as e:
            errors.append(f"Corpus file: {e}")
    else:
        errors.append(f"Corpus not found at {corpus_path}")
    
    if errors:
        print(f"\n❌ Config errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("\n✓ All config OK")
    return True


def run_quick_eval(num_questions: int = 30) -> dict:
    """Run quick evaluation and return results."""
    from eval_runners.baseline.run_baseline import run_evaluation
    
    print(f"\nRunning quick evaluation ({num_questions} questions)...")
    print("-" * 40)
    
    output = run_evaluation(
        client="BFAI",
        workers=5,
        precision_k=25,
        quick=num_questions,
        test_mode=False,
        update_baseline=False,
    )
    
    return output


def check_thresholds(output: dict) -> tuple[bool, list[str]]:
    """Check if results meet CI thresholds."""
    metrics = output.get("metrics", {})
    results = output.get("results", [])
    
    failures = []
    
    # Check pass rate
    pass_rate = metrics.get("pass_rate", 0)
    if pass_rate < THRESHOLDS["pass_rate"]:
        failures.append(f"pass_rate {pass_rate:.1%} < {THRESHOLDS['pass_rate']:.0%}")
    
    # Check fail rate
    fail_rate = metrics.get("fail_rate", 1)
    if fail_rate > THRESHOLDS["fail_rate"]:
        failures.append(f"fail_rate {fail_rate:.1%} > {THRESHOLDS['fail_rate']:.0%}")
    
    # Check for errors
    error_count = sum(1 for r in results if r.get("error"))
    if error_count > THRESHOLDS["error_count"]:
        failures.append(f"error_count {error_count} > {THRESHOLDS['error_count']}")
    
    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(
        description="CI/CD Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--quick", type=int, default=30,
                        help="Number of questions for quick eval (default: 30)")
    parser.add_argument("--validate-config", action="store_true",
                        help="Only validate configuration")
    parser.add_argument("--check-imports", action="store_true",
                        help="Only check imports")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only run checks")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  CI/CD EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Import check only
    if args.check_imports:
        success = check_imports()
        sys.exit(0 if success else 1)
    
    # Config validation only
    if args.validate_config:
        success = validate_config()
        sys.exit(0 if success else 1)
    
    # Full CI check
    all_passed = True
    
    # Step 1: Check imports
    if not check_imports():
        all_passed = False
    
    # Step 2: Validate config
    if not validate_config():
        all_passed = False
    
    # Step 3: Run evaluation (unless skipped)
    if not args.skip_eval and all_passed:
        output = run_quick_eval(args.quick)
        
        # Check thresholds
        passed, failures = check_thresholds(output)
        
        print("\n" + "=" * 60)
        print("CI/CD RESULTS")
        print("=" * 60)
        
        metrics = output.get("metrics", {})
        print(f"  Pass Rate: {metrics.get('pass_rate', 0):.1%}")
        print(f"  Fail Rate: {metrics.get('fail_rate', 0):.1%}")
        print(f"  MRR: {metrics.get('mrr', 0):.3f}")
        
        if passed:
            print("\n✅ CI/CD PASSED - All thresholds met")
        else:
            print(f"\n❌ CI/CD FAILED - Threshold violations:")
            for f in failures:
                print(f"  - {f}")
            all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("❌ CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
