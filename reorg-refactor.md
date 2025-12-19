# Repository Reorganization Plan

**Date:** 2025-12-19  
**Status:** AWAITING APPROVAL  
**Scope:** File/folder moves only - no logic changes

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Principles](#2-design-principles)
3. [Current vs Proposed Structure](#3-current-vs-proposed-structure)
4. [Move Manifest](#4-move-manifest)
5. [Required Updates Checklist](#5-required-updates-checklist)
6. [Execution Mode Boundaries](#6-execution-mode-boundaries)
7. [Rollback Plan](#7-rollback-plan)

---

## 1. Executive Summary

### Current State Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| Two config directories (`config/`, `configs/`) | Medium | Confusion about canonical config location |
| 15 scripts in `scripts/eval/` with mixed concerns | Medium | Hard to find entry points vs utilities |
| Reports fragmented across 11 subdirectories | Low | Difficult to navigate |
| `_upstream_ragas/` at root level | Low | Clutters root, should be archived |
| No clear execution mode separation | High | Unclear which code serves baseline vs CI vs ad hoc |
| Test file in wrong location (`scripts/eval/test_parallel.py`) | Low | Should be in `tests/` |

### Goals

1. **Execution mode clarity** - Obvious where to find baseline, CI/CD, and ad hoc entry points
2. **Lifecycle phase grouping** - Group by evaluation lifecycle where appropriate
3. **Shared primitives isolation** - Core utilities clearly separated from mode-specific code
4. **Archive consolidation** - All deprecated/archived content in `_archive/`
5. **Config consolidation** - Single config location

---

## 2. Design Principles

- **Execution mode clarity** - Baseline, CI/CD, and ad hoc evaluations have distinct entry points
- **Shared evaluation primitives** - Core components (metrics, models, clients) are reusable across all modes
- **Minimal duplication** - Consolidate overlapping scripts
- **Lifecycle phases where explicit** - Use numbered prefixes only when order is enforced
- **Archive at root** - `_archive/` sorts to top, keeps active folders contiguous
- **No behavioral changes** - Only moves and import fixes

---

## 3. Current vs Proposed Structure

### Current Structure

```
ragas/
├── _upstream_ragas/           # Archived upstream (at root)
├── baselines/                 # ✓ Keep
├── benchmarks/                # Underutilized
├── clients/BFAI/              # ✓ Keep
├── config/                    # JSON configs
├── configs/                   # YAML configs (duplicate!)
├── docs/                      # ✓ Keep
├── reports/                   # 11 subdirs, fragmented
│   ├── archive/
│   ├── executive/
│   ├── experiments/
│   ├── foundational/
│   ├── gemini3_comparison/
│   ├── gemini3_eval/
│   ├── gemini3_test/
│   ├── gold_standard_500/
│   ├── gold_standard_eval/
│   └── orchestrator_eval/
├── runs/                      # ✓ Keep
├── scripts/
│   ├── archive/               # 18 archived scripts
│   ├── corpus/                # 8 QA generation scripts
│   ├── eval/                  # 15 scripts (mixed concerns)
│   ├── output/
│   ├── reports/
│   └── setup/                 # 3 setup scripts
├── src/                       # Core modules
└── tests/
    ├── eval/                  # Eval-specific tests
    ├── integration/
    ├── smoke/
    └── unit/
```

### Proposed Structure

```
ragas/
├── _archive/                          # All archived content (sorts to top)
│   ├── scripts/                       # From scripts/archive/
│   ├── reports/                       # From reports/archive/
│   └── upstream_ragas/                # From _upstream_ragas/
│
├── baselines/                         # ✓ Unchanged
├── clients/BFAI/                      # ✓ Unchanged
├── runs/                              # ✓ Unchanged
│
├── config/                            # CONSOLIDATED config
│   ├── eval_config.yaml               # From configs/
│   ├── gold_standard_500.json         # Already here
│   └── ragas-cloud-run-invoker.json   # Already here
│
├── docs/                              # ✓ Unchanged
│
├── evaluations/                       # NEW: Execution mode entry points
│   ├── baseline/                      # Scheduled evaluations
│   │   └── run_baseline.py            # Entry point for daily runs
│   ├── cicd/                          # CI/CD gating evaluations
│   │   └── run_cicd_eval.py           # Entry point for CI checks
│   └── adhoc/                         # Ad hoc experiments
│       ├── run_adhoc.py               # Interactive entry point
│       └── experiments/               # One-off experiment scripts
│
├── lib/                               # NEW: Shared evaluation primitives
│   ├── __init__.py
│   ├── core/                          # Core evaluation logic
│   │   ├── __init__.py
│   │   ├── evaluator.py               # From run_gold_eval.py (GoldEvaluator)
│   │   ├── baseline_manager.py        # From scripts/eval/
│   │   ├── cost_calculator.py         # From scripts/eval/
│   │   └── report_generator.py        # From scripts/eval/generate_report.py
│   ├── clients/                       # External service clients
│   │   ├── __init__.py
│   │   └── gemini_client.py           # From src/
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── metrics.py                 # From src/
│       ├── models.py                  # From src/
│       └── preflight.py               # From src/
│
├── reports/                           # SIMPLIFIED report structure
│   ├── core_eval/                     # From gold_standard_eval/
│   ├── experiments/                   # Consolidated experiments
│   └── executive/                     # ✓ Keep
│
├── scripts/                           # SLIMMED: Setup and corpus only
│   ├── corpus/                        # ✓ Keep (QA generation)
│   └── setup/                         # ✓ Keep (one-time setup)
│
├── src/                               # DEPRECATED: Redirect to lib/
│   └── __init__.py                    # Re-exports from lib/ for compatibility
│
└── tests/                             # ✓ Mostly unchanged
    ├── eval/                          # + test_parallel.py moved here
    ├── integration/
    ├── smoke/
    └── unit/
```

---

## 4. Move Manifest

### Batch 1: Archive Consolidation

```bash
# Create _archive structure
mkdir -p _archive/scripts
mkdir -p _archive/reports
mkdir -p _archive/upstream_ragas

# Move archived scripts
git mv scripts/archive/* _archive/scripts/

# Move archived reports
git mv reports/archive/* _archive/reports/

# Move upstream ragas
git mv _upstream_ragas/* _archive/upstream_ragas/
rmdir _upstream_ragas
```

### Batch 2: Config Consolidation

```bash
# Move YAML config to config/
git mv configs/eval_config.yaml config/

# Remove empty configs directory
rmdir configs
```

### Batch 3: Create Evaluations Structure

```bash
# Create evaluations directories
mkdir -p evaluations/baseline
mkdir -p evaluations/cicd
mkdir -p evaluations/adhoc/experiments
```

### Batch 4: Create lib/ Structure and Move Core Components

```bash
# Create lib structure
mkdir -p lib/core
mkdir -p lib/clients
mkdir -p lib/utils

# Move core evaluation components
git mv scripts/eval/baseline_manager.py lib/core/
git mv scripts/eval/cost_calculator.py lib/core/
git mv scripts/eval/generate_report.py lib/core/report_generator.py

# Move client code
git mv src/gemini_client.py lib/clients/

# Move utilities
git mv src/metrics.py lib/utils/
git mv src/models.py lib/utils/
git mv src/preflight.py lib/utils/
git mv src/report.py lib/utils/
git mv src/eval_config.py lib/utils/
git mv src/schema_loader.py lib/utils/
```

### Batch 5: Move Evaluation Entry Points

```bash
# Baseline entry point
git mv scripts/eval/core_eval.py evaluations/baseline/run_baseline.py

# Ad hoc entry point
git mv scripts/eval/eval_runner.py evaluations/adhoc/run_adhoc.py

# Move experiment scripts to adhoc/experiments
git mv scripts/eval/gemini3_comparison_eval.py evaluations/adhoc/experiments/
git mv scripts/eval/run_gemini3_eval.py evaluations/adhoc/experiments/
git mv scripts/eval/run_gemini3_test.py evaluations/adhoc/experiments/
git mv scripts/eval/e2e_orchestrator_test.py evaluations/adhoc/experiments/
git mv scripts/eval/run_orchestrator_eval.py evaluations/adhoc/experiments/
git mv scripts/eval/gold_standard_eval.py evaluations/adhoc/experiments/
```

### Batch 6: Move Primary Evaluator

```bash
# Move main evaluator to lib/core
git mv scripts/eval/run_gold_eval.py lib/core/evaluator.py
```

### Batch 7: Move Remaining Scripts

```bash
# Move preflight check to lib/utils
git mv scripts/eval/preflight_check.py lib/utils/

# Move benchmark report to lib/core
git mv scripts/eval/benchmark_report.py lib/core/

# Move test file to tests
git mv scripts/eval/test_parallel.py tests/eval/
```

### Batch 8: Consolidate Reports

```bash
# Rename gold_standard_eval to core_eval
git mv reports/gold_standard_eval reports/core_eval

# Move experiment reports to experiments/
git mv reports/gemini3_comparison reports/experiments/gemini3_comparison
git mv reports/gemini3_eval reports/experiments/gemini3_eval
git mv reports/gemini3_test reports/experiments/gemini3_test
git mv reports/orchestrator_eval reports/experiments/orchestrator_eval
git mv reports/foundational reports/experiments/foundational

# Remove empty directories
rmdir reports/gold_standard_500
```

### Batch 9: Clean Up src/

```bash
# Remove files that were moved (keep __init__.py for compatibility)
git rm src/archive.py
git rm src/config.py
git rm src/generator.py
git rm src/judge.py
git rm src/orchestrator_client.py
git rm src/question_generator.py
git rm src/question_rater.py
git rm src/ragas_evaluator.py
git rm src/reranker.py
git rm src/retriever.py
git rm src/vector_search.py
```

### Batch 10: Clean Up Empty Directories

```bash
# Remove empty scripts/eval (should be empty after moves)
rmdir scripts/eval
rmdir scripts/output
rmdir scripts/reports

# Remove benchmarks if empty/unused
# (Keep if actively used)
```

---

## 5. Required Updates Checklist

### Import Updates Required

After moves, these files need import path updates:

| File | Current Import | New Import |
|------|----------------|------------|
| `evaluations/baseline/run_baseline.py` | `from baseline_manager import ...` | `from lib.core.baseline_manager import ...` |
| `evaluations/baseline/run_baseline.py` | `from cost_calculator import ...` | `from lib.core.cost_calculator import ...` |
| `evaluations/baseline/run_baseline.py` | `from run_gold_eval import ...` | `from lib.core.evaluator import ...` |
| `evaluations/adhoc/run_adhoc.py` | `from baseline_manager import ...` | `from lib.core.baseline_manager import ...` |
| `evaluations/adhoc/run_adhoc.py` | `from core_eval import ...` | `from evaluations.baseline.run_baseline import ...` |
| `lib/core/evaluator.py` | `from gemini_client import ...` | `from lib.clients.gemini_client import ...` |
| `tests/eval/*.py` | `sys.path.insert(...scripts/eval)` | `from lib.core import ...` |

### sys.path.insert Removals

All `sys.path.insert(0, "/Users/scottmacon/...")` lines should be removed after proper package setup.

**Files requiring sys.path cleanup:**
- `evaluations/baseline/run_baseline.py`
- `evaluations/adhoc/run_adhoc.py`
- `evaluations/adhoc/experiments/*.py`
- `lib/core/evaluator.py`
- `lib/utils/preflight.py`
- `tests/eval/*.py`
- `tests/unit/*.py`

### __init__.py Files to Create

```bash
touch lib/__init__.py
touch lib/core/__init__.py
touch lib/clients/__init__.py
touch lib/utils/__init__.py
touch evaluations/__init__.py
touch evaluations/baseline/__init__.py
touch evaluations/cicd/__init__.py
touch evaluations/adhoc/__init__.py
```

### Documentation Updates

| File | Update Needed |
|------|---------------|
| `README.md` | Update project structure, quick start commands |
| `docs/EVAL_SYSTEM.md` | Update script paths |
| `docs/EVAL_RUNBOOK.md` | Update command examples |
| `docs/CODE_ORGANIZATION.md` | Update to reflect new structure |

### CI/CD Configuration (if exists)

- Update any GitHub Actions workflows
- Update any Cloud Run job definitions
- Update any cron job scripts

---

## 6. Execution Mode Boundaries

### After Reorganization

#### Baseline Scheduled Evaluations

**Entry Point:** `evaluations/baseline/run_baseline.py`

```bash
# Daily scheduled run
python evaluations/baseline/run_baseline.py --client BFAI

# With baseline update
python evaluations/baseline/run_baseline.py --client BFAI --update-baseline
```

**Components Used:**
- `lib/core/evaluator.py` - GoldEvaluator class
- `lib/core/baseline_manager.py` - Baseline comparison
- `lib/core/cost_calculator.py` - Cost tracking

#### CI/CD Gating Evaluations

**Entry Point:** `evaluations/cicd/run_cicd_eval.py` (to be created)

```bash
# CI check with reduced dataset
python evaluations/cicd/run_cicd_eval.py --quick 30 --fail-on-regression
```

**Components Used:**
- Same as baseline, with:
  - Reduced dataset (30 questions)
  - Strict threshold checking
  - Exit code for CI pass/fail

#### Ad Hoc Evaluations

**Entry Point:** `evaluations/adhoc/run_adhoc.py`

```bash
# Interactive mode
python evaluations/adhoc/run_adhoc.py

# Quick test
python evaluations/adhoc/run_adhoc.py --quick 10 --model gemini-3-flash-preview
```

**Experiments:** `evaluations/adhoc/experiments/`

---

## 7. Rollback Plan

### Before Starting

```bash
# Create backup branch
git checkout -b backup/pre-reorg-$(date +%Y%m%d)
git push origin backup/pre-reorg-$(date +%Y%m%d)

# Create reorg branch
git checkout main
git checkout -b reorg
```

### If Issues Arise

```bash
# Option 1: Reset to main
git checkout main
git branch -D reorg

# Option 2: Reset specific batch
git log --oneline  # Find commit before problematic batch
git reset --hard <commit-hash>

# Option 3: Full restore from backup
git checkout backup/pre-reorg-YYYYMMDD
```

### Verification After Each Batch

```bash
# Run tests
python -m pytest tests/eval/ -v --ignore=tests/eval/test_e2e.py

# Verify imports work
python -c "from lib.core.baseline_manager import list_baselines; print(list_baselines())"

# Verify entry points
python evaluations/baseline/run_baseline.py --dry-run
```

---

## Appendix A: Files Not Moved

These files remain in place:

| File/Directory | Reason |
|----------------|--------|
| `baselines/` | Already well-organized |
| `runs/` | Output directory, no change needed |
| `clients/BFAI/` | Client data structure is correct |
| `docs/` | Documentation location is standard |
| `scripts/corpus/` | QA generation is distinct from evaluation |
| `scripts/setup/` | One-time setup scripts are distinct |
| `benchmarks/` | Keep for now, evaluate later |

---

## Appendix B: New Files to Create

### evaluations/cicd/run_cicd_eval.py

Stub for CI/CD entry point (implementation in Stage B):

```python
#!/usr/bin/env python3
"""
CI/CD Evaluation Runner

Runs reduced evaluation for CI gating with strict thresholds.
Exit code 0 = pass, 1 = regression detected.

Usage:
  python evaluations/cicd/run_cicd_eval.py --quick 30
"""
# TODO: Implement in Stage B
pass
```

### lib/__init__.py

```python
"""
BFAI Eval Suite Library

Shared evaluation primitives for baseline, CI/CD, and ad hoc evaluations.
"""
from .core import baseline_manager, cost_calculator, evaluator
from .clients import gemini_client
from .utils import metrics, models, preflight
```

### src/__init__.py (Updated for Compatibility)

```python
"""
DEPRECATED: Use lib/ instead.

This module re-exports from lib/ for backward compatibility.
"""
import warnings
warnings.warn(
    "Importing from src/ is deprecated. Use lib/ instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for compatibility
from lib.utils.models import *
from lib.utils.preflight import *
from lib.utils.report import *
```

---

**AWAITING APPROVAL TO PROCEED WITH PHASE 4 EXECUTION**
