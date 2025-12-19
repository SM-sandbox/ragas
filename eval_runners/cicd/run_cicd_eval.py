#!/usr/bin/env python3
"""
CI/CD Evaluation Runner (Checkpoint Mode)

Runs reduced evaluation for CI gating with strict thresholds.
Exit code 0 = pass, 1 = regression detected.

⚠️  IMPORTANT: This runner uses LOCKED configuration from checkpoint_config.yaml
⚠️  The config file is hash-validated to prevent accidental changes.
⚠️  To modify checkpoint settings, update the config AND the hash below.

Usage:
  python eval_runners/cicd/run_cicd_eval.py                    # Run checkpoint eval
  python eval_runners/cicd/run_cicd_eval.py --validate-config  # Config validation only
  python eval_runners/cicd/run_cicd_eval.py --check-imports    # Import check only
  python eval_runners/cicd/run_cicd_eval.py --force            # Skip hash validation (NOT RECOMMENDED)
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# Add repo root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CHECKPOINT CONFIG PROTECTION
# =============================================================================
# This hash ensures checkpoint_config.yaml hasn't been accidentally modified.
# To update: 1) Modify checkpoint_config.yaml  2) Run: shasum -a 256 config/checkpoint_config.yaml
#            3) Update the hash below  4) Commit both changes together

CHECKPOINT_CONFIG_HASH = "8097ea0832ce991dd1a3d3c4b2d7102c46af6680bb715849e443da98d4878f8d"
CHECKPOINT_CONFIG_PATH = PROJECT_ROOT / "config" / "checkpoint_config.yaml"


def validate_checkpoint_config_hash(force: bool = False) -> bool:
    """
    Validate that checkpoint_config.yaml hasn't been modified.
    
    Returns True if valid, False if modified (unless force=True).
    """
    if not CHECKPOINT_CONFIG_PATH.exists():
        print(f"❌ ERROR: checkpoint_config.yaml not found at {CHECKPOINT_CONFIG_PATH}")
        return False
    
    actual_hash = hashlib.sha256(CHECKPOINT_CONFIG_PATH.read_bytes()).hexdigest()
    
    if actual_hash != CHECKPOINT_CONFIG_HASH:
        print("\n" + "=" * 60)
        print("⚠️  CHECKPOINT CONFIG MODIFIED")
        print("=" * 60)
        print(f"Expected hash: {CHECKPOINT_CONFIG_HASH}")
        print(f"Actual hash:   {actual_hash}")
        print("\nThe checkpoint configuration has been modified.")
        print("This could cause inconsistent checkpoint results.")
        print("\nTo fix:")
        print("  1. Revert changes to config/checkpoint_config.yaml, OR")
        print("  2. Update CHECKPOINT_CONFIG_HASH in this file if change is intentional")
        print("\nUse --force to bypass this check (NOT RECOMMENDED)")
        print("=" * 60 + "\n")
        
        if force:
            print("⚠️  --force flag set, continuing despite hash mismatch...\n")
            return True
        return False
    
    print("  ✓ checkpoint_config.yaml hash validated")
    return True


def load_checkpoint_config() -> dict:
    """Load the checkpoint configuration."""
    import yaml
    with open(CHECKPOINT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_thresholds() -> dict:
    """Get thresholds from checkpoint config."""
    try:
        config = load_checkpoint_config()
        thresholds = config.get("thresholds", {})
        return {
            "pass_rate": thresholds.get("min_pass_rate", 0.85),
            "fail_rate": thresholds.get("max_fail_rate", 0.08),
            "error_count": thresholds.get("max_error_count", 0),
        }
    except Exception:
        # Fallback defaults
        return {
            "pass_rate": 0.85,
            "fail_rate": 0.08,
            "error_count": 0,
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


def validate_config(force: bool = False) -> bool:
    """Validate configuration files load correctly."""
    print("Validating configuration...")
    errors = []
    
    # Check checkpoint_config.yaml hash (CRITICAL)
    if not validate_checkpoint_config_hash(force=force):
        return False
    
    # Load and validate checkpoint config
    try:
        config = load_checkpoint_config()
        if not config:
            errors.append("checkpoint_config.yaml is empty")
        elif config.get("config_type") != "checkpoint":
            errors.append("checkpoint_config.yaml has wrong config_type")
        else:
            print(f"  ✓ checkpoint_config.yaml (schema_version: {config.get('schema_version', 'unknown')})")
    except Exception as e:
        errors.append(f"checkpoint_config.yaml: {e}")
    
    # Check baselines exist (using new path)
    baselines_dir = PROJECT_ROOT / "clients_qa_gold" / "BFAI" / "baselines"
    if baselines_dir.exists():
        baselines = list(baselines_dir.glob("baseline_*.json"))
        if baselines:
            print(f"  ✓ Found {len(baselines)} baseline(s)")
        else:
            errors.append("No baseline files found")
    else:
        errors.append(f"Baselines directory not found at {baselines_dir}")
    
    # Check corpus exists (using new path)
    corpus_path = PROJECT_ROOT / "clients_qa_gold" / "BFAI" / "qa" / "QA_BFAI_gold_v1-0__q458.json"
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


def run_checkpoint_eval() -> dict:
    """
    Run checkpoint evaluation using LOCKED configuration.
    
    All settings come from checkpoint_config.yaml - no overrides allowed.
    """
    from eval_runners.baseline.run_baseline import run_evaluation
    
    # Load config (already validated)
    config = load_checkpoint_config()
    
    # Extract settings from config
    client = config.get("client", "BFAI")
    corpus_config = config.get("corpus", {})
    num_questions = corpus_config.get("sample_size", 30)
    retrieval = config.get("retrieval", {})
    precision_k = retrieval.get("precision_k", 25)
    execution = config.get("execution", {})
    workers = execution.get("workers", 5)
    
    print(f"\nRunning CHECKPOINT evaluation...")
    print(f"  Config: checkpoint_config.yaml (LOCKED)")
    print(f"  Client: {client}")
    print(f"  Questions: {num_questions}")
    print(f"  Precision@K: {precision_k}")
    print(f"  Workers: {workers}")
    print("-" * 40)
    
    output = run_evaluation(
        client=client,
        workers=workers,
        precision_k=precision_k,
        quick=num_questions,
        test_mode=False,
        update_baseline=False,
    )
    
    return output


def check_thresholds(output: dict) -> tuple[bool, list[str]]:
    """Check if results meet CI thresholds from checkpoint config."""
    metrics = output.get("metrics", {})
    results = output.get("results", [])
    thresholds = get_thresholds()
    
    failures = []
    
    # Check pass rate
    pass_rate = metrics.get("pass_rate", 0)
    if pass_rate < thresholds["pass_rate"]:
        failures.append(f"pass_rate {pass_rate:.1%} < {thresholds['pass_rate']:.0%}")
    
    # Check fail rate
    fail_rate = metrics.get("fail_rate", 1)
    if fail_rate > thresholds["fail_rate"]:
        failures.append(f"fail_rate {fail_rate:.1%} > {thresholds['fail_rate']:.0%}")
    
    # Check for errors
    error_count = sum(1 for r in results if r.get("error"))
    if error_count > thresholds["error_count"]:
        failures.append(f"error_count {error_count} > {thresholds['error_count']}")
    
    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser(
        description="CI/CD Evaluation Runner (Checkpoint Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: This runner uses LOCKED configuration from checkpoint_config.yaml.
      The config file is hash-validated to prevent accidental changes.
      Use --force to bypass hash validation (NOT RECOMMENDED).
        """
    )
    
    parser.add_argument("--validate-config", action="store_true",
                        help="Only validate configuration")
    parser.add_argument("--check-imports", action="store_true",
                        help="Only check imports")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only run checks")
    parser.add_argument("--force", action="store_true",
                        help="Bypass config hash validation (NOT RECOMMENDED)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  CI/CD CHECKPOINT EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Config: checkpoint_config.yaml (LOCKED)")
    print("=" * 60)
    
    # Import check only
    if args.check_imports:
        success = check_imports()
        sys.exit(0 if success else 1)
    
    # Config validation only
    if args.validate_config:
        success = validate_config(force=args.force)
        sys.exit(0 if success else 1)
    
    # Full CI check
    all_passed = True
    
    # Step 1: Check imports
    if not check_imports():
        all_passed = False
    
    # Step 2: Validate config (including hash check)
    if not validate_config(force=args.force):
        all_passed = False
        if not args.force:
            print("\n❌ Aborting due to config validation failure.")
            sys.exit(1)
    
    # Step 3: Run checkpoint evaluation (unless skipped)
    if not args.skip_eval and all_passed:
        output = run_checkpoint_eval()
        
        # Check thresholds
        passed, failures = check_thresholds(output)
        
        print("\n" + "=" * 60)
        print("CHECKPOINT RESULTS")
        print("=" * 60)
        
        metrics = output.get("metrics", {})
        thresholds = get_thresholds()
        print(f"  Pass Rate: {metrics.get('pass_rate', 0):.1%} (threshold: ≥{thresholds['pass_rate']:.0%})")
        print(f"  Fail Rate: {metrics.get('fail_rate', 0):.1%} (threshold: ≤{thresholds['fail_rate']:.0%})")
        print(f"  MRR: {metrics.get('mrr', 0):.3f}")
        
        if passed:
            print("\n✅ CHECKPOINT PASSED - All thresholds met")
        else:
            print(f"\n❌ CHECKPOINT FAILED - Threshold violations:")
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
