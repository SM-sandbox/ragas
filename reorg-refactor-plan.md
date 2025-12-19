# Runs & Experiments Reorganization Plan

**Created:** 2025-12-19
**Status:** Approved, Ready for Implementation

---

## Overview

This document details the reorganization of evaluation outputs into a structured 3-tier system with consistent naming conventions and registry tracking.

## 3-Tier Evaluation System

| Type | Code | Purpose | Config | Corpus |
|------|------|---------|--------|--------|
| **Checkpoint** | `C###` | CICD/daily health checks - locked baseline config | Fixed | Full 458 questions |
| **Run** | `R###` | Full evaluation with variations (testing hypotheses) | Variable | Full 458 questions |
| **Experiment** | `E###` | Quick exploratory tests | Variable | Partial (30, 100, etc.) |

### Key Distinctions

- **Checkpoints (C###):** Same config every time. Used for CICD gating, daily regression. "Is the system still working?"
- **Runs (R###):** Full corpus, but varying something (model, precision, recall). "What if I change X?"
- **Experiments (E###):** Quick, exploratory, smaller corpus. "Let me quickly check something."

---

## Naming Conventions

### Baselines (Gold Standard)
```
baseline_gold__{client}__{version}__{date}__q{count}.json
```
**Example:** `baseline_gold__BFAI__v2__2025-12-18__q458.json`

### Checkpoints (C###)
```
C{NNN}__{YYYY-MM-DD}__{mode}__{config-summary}/
```
**Example:** `C001__2025-12-19__C__p25-flash/`

### Runs (R###)
```
R{NNN}__{YYYY-MM-DD}__{mode}__{config-summary}/
```
**Example:** `R001__2025-12-18__C__p12-test/`

### Experiments (E###)
```
E{NNN}__{YYYY-MM-DD}__{mode}__{description}/
```
**Example:** `E001__2025-12-17__L__quick30-temp-sweep/`

### Mode Codes
- `L` = Local (gRAG_v3 pipeline running locally)
- `C` = Cloud (Cloud Run endpoint)

---

## Implementation Steps

### Step 1: Rename Baselines

**Action:** Rename existing baseline files to new convention

| Current | New |
|---------|-----|
| `baselines/baseline_BFAI_v1__2025-12-17__q458.json` | `baselines/baseline_gold__BFAI__v1__2025-12-17__q458.json` |
| `baselines/baseline_BFAI_v2__2025-12-18__q458.json` | `baselines/baseline_gold__BFAI__v2__2025-12-18__q458.json` |

**Create:** `baselines/REGISTRY.md` and `baselines/registry.json`

### Step 2: Create `checkpoints/` Directory

**Action:** Create new directory and move checkpoint data from `reports/core_eval/`

| Source | Destination |
|--------|-------------|
| `reports/core_eval/checkpoint_p25_cloud.json` + `results_p25_cloud.json` | `checkpoints/C001__2025-12-19__C__p25-flash/` |
| `reports/core_eval/checkpoint_p12_cloud.json` + `results_p12_cloud.json` | `checkpoints/C002__2025-12-18__C__p12-flash/` |
| `reports/core_eval/checkpoint_p25.json` + `results_p25.json` | `checkpoints/C003__2025-12-19__L__p25-flash/` |
| `reports/core_eval/checkpoint_p25_cloud_gemini_3_pro_preview.json` + results | `checkpoints/C004__2025-12-19__C__p25-pro/` |

**Create:** `checkpoints/REGISTRY.md` and `checkpoints/registry.json`

### Step 3: Reorganize `runs/` with R### Naming

**Action:** Rename existing run folders

| Current | New |
|---------|-----|
| `runs/2025-12-18__gemini-2.5-flash__p25__30a5b7b1/` | `runs/R001__2025-12-18__L__p25-flash-2.5/` |
| `runs/2025-12-18__gemini-2.5-flash__p25__716d048c/` | `runs/R002__2025-12-18__L__p25-flash-2.5/` |
| `runs/2025-12-18__gemini-2.5-flash__p25__9d6623bd/` | `runs/R003__2025-12-18__L__p25-flash-2.5/` |
| `runs/2025-12-18__gemini-3-flash-preview__p25__f978a2b8/` | `runs/R004__2025-12-18__C__p25-flash-3/` |
| `runs/2025-12-19__gemini-3-flash-preview__p25__6907389a/` | `runs/R005__2025-12-19__L__p25-flash-3/` |

**Create:** `runs/REGISTRY.md` and `runs/registry.json`

### Step 4: Create `experiments/` Directory

**Action:** Move experimental data from `reports/experiments/`

| Source | Destination |
|--------|-------------|
| `reports/experiments/gemini3_eval/` | `experiments/E001__2025-12-17__L__gemini3-eval/` |
| `reports/experiments/gemini3_test/` | `experiments/E002__2025-12-17__L__gemini3-test/` |
| `reports/experiments/gemini3_comparison/` | `experiments/E003__2025-12-18__L__gemini3-comparison/` |
| `reports/experiments/orchestrator_eval/` | `experiments/E004__2025-12-18__L__orchestrator-eval/` |
| `reports/experiments/foundational/` | `experiments/foundational/` (keep as reference docs) |

**Create:** `experiments/REGISTRY.md` and `experiments/registry.json`

### Step 5: Create Documentation

**Action:** Create `docs/RUNS_AND_EXPERIMENTS.md` with:
- 3-tier system explanation
- Naming conventions
- When to use each type
- How to register new entries
- Directory structure

### Step 6: Update Code References

**Action:** Update `lib/core/baseline_manager.py`:
- Update `BASELINE_PATTERN` regex for new naming
- Update `save_baseline()` to use new naming convention
- Update `list_baselines()` to parse new format

### Step 7: Clean Up `reports/`

**Action:** After moving data, consolidate `reports/`:
```
reports/
├── executive/           # Keep - executive summaries
├── analysis/            # Create - move failure analysis files here
└── archive/             # Move old/superseded reports
```

### Step 8: Run Fresh Evaluation

**Action:** Run a fresh C### checkpoint evaluation to validate the system works end-to-end

---

## Final Directory Structure

```
ragas/
├── baselines/                         # Gold standard baselines
│   ├── baseline_gold__BFAI__v1__2025-12-17__q458.json
│   ├── baseline_gold__BFAI__v2__2025-12-18__q458.json
│   ├── REGISTRY.md
│   └── registry.json
│
├── checkpoints/                       # CICD/daily health checks (C###)
│   ├── C001__2025-12-19__C__p25-flash/
│   │   ├── checkpoint.json
│   │   ├── results.json
│   │   └── summary.md
│   ├── C002__2025-12-18__C__p12-flash/
│   ├── REGISTRY.md
│   └── registry.json
│
├── runs/                              # Full evaluation runs (R###)
│   ├── R001__2025-12-18__L__p25-flash-2.5/
│   │   ├── results.jsonl
│   │   ├── run_summary.json
│   │   └── comparison.md
│   ├── R002__2025-12-18__L__p25-flash-2.5/
│   ├── REGISTRY.md
│   └── registry.json
│
├── experiments/                       # Quick experiments (E###)
│   ├── E001__2025-12-17__L__gemini3-eval/
│   ├── foundational/                  # Reference documentation
│   ├── REGISTRY.md
│   └── registry.json
│
├── reports/
│   ├── executive/                     # Executive summaries
│   ├── analysis/                      # Deep-dive analysis
│   └── archive/                       # Old reports
│
└── docs/
    └── RUNS_AND_EXPERIMENTS.md        # System documentation
```

---

## Registry File Formats

### REGISTRY.md (Human-readable)

```markdown
# Checkpoints Registry

| ID | Date | Mode | Config | Questions | Pass Rate | Notes |
|----|------|------|--------|-----------|-----------|-------|
| C001 | 2025-12-19 | Cloud | p25-flash | 458 | 92.6% | Daily health check |
| C002 | 2025-12-18 | Cloud | p12-flash | 458 | 86.2% | Precision comparison |
```

### registry.json (Programmatic)

```json
{
  "schema_version": "1.0",
  "type": "checkpoints",
  "entries": [
    {
      "id": "C001",
      "date": "2025-12-19",
      "mode": "cloud",
      "config_summary": "p25-flash",
      "questions": 458,
      "pass_rate": 0.926,
      "folder": "C001__2025-12-19__C__p25-flash",
      "notes": "Daily health check"
    }
  ]
}
```

---

## Verification Checklist

- [ ] Baselines renamed and registry created
- [ ] Checkpoints directory created with C### folders
- [ ] Runs reorganized with R### naming
- [ ] Experiments directory created with E### folders
- [ ] All registry files created (.md and .json)
- [ ] Documentation created in docs/
- [ ] baseline_manager.py updated
- [ ] reports/ cleaned up
- [ ] Fresh evaluation runs successfully
