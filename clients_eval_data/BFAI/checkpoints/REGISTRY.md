# Checkpoints Registry

CICD and daily health check runs against the gold standard baseline.

## Gold Baseline

**All CI/CD and test runs compare against this locked baseline:**

| Field | Value |
|-------|-------|
| **ID** | C012 |
| **Folder** | `C012__2025-12-19__cloud__p25-3-flash-low__q458` |
| **Locked Date** | 2025-12-20 |
| **Acceptable Rate** | 98.5% |
| **Pass Rate** | 92.8% |

> ⚠️ **Do not change the gold baseline without explicit approval.** Future runs (C013, C014, etc.) always compare against C012.

## Checkpoint vs Run Nomenclature

| Type | Purpose | Location | When to Use |
|------|---------|----------|-------------|
| **Checkpoint** | Daily health check + CI/CD gating. **Same config every time.** Compares to Gold Baseline. | `checkpoints/C{NNN}__...` | Scheduled runs, CI/CD gates, production validation |
| **Run** | Experimental. **Test one variable** (model, temp, precision, etc.). | `runs/R{NNN}__...` | A/B testing, experiments, debugging |

### Key Differences

- **Checkpoints** use locked configuration (`checkpoint_config.yaml`) - no overrides allowed
- **Runs** can vary any parameter for experimentation
- **Checkpoints** always compare against Gold Baseline (C012)
- **Runs** may compare against any baseline or each other

## Naming Convention

```text
C{NNN}__{YYYY-MM-DD}__{mode}__{config-summary}__q{count}/   # Checkpoint
R{NNN}__{YYYY-MM-DD}__{mode}__{config-summary}__q{count}/   # Run
```

- **Mode:** `local` or `cloud`
- **Config:** `p{precision}-{model}-{reasoning}` (e.g., `p25-3-flash-low`)

## Registry

| ID | Date | Mode | Config | Questions | Pass Rate | Fail Rate | Notes |
|----|------|------|--------|-----------|-----------|-----------|-------|
| C001 | 2025-12-19 | Cloud | p25-flash | 458 | 92.6% | 0.9% | Cloud baseline p25 |
| C002 | 2025-12-18 | Cloud | p12-flash | 458 | 86.2% | 3.5% | Cloud p12 comparison |
| C003 | 2025-12-19 | Local | p25-flash | 458 | 92.4% | 1.1% | Local baseline p25 |
| C004 | 2025-12-19 | Cloud | p25-pro | 458 | 92.1% | 1.1% | Gemini 3 Pro test |
| **C012** | 2025-12-19 | Cloud | p25-3-flash-low | 458 | 92.8% | 1.5% | **GOLD BASELINE** |
| C013 | 2025-12-20 | Local | p25-3-flash-low | 458 | 91.9% | 1.5% | Smart Throttler integration test |

## Purpose

Checkpoints are used for:

- **CICD gating:** Verify the system hasn't regressed
- **Daily health checks:** Ensure production stability
- **Baseline comparison:** Compare against gold standard metrics

## Folder Contents

Each checkpoint folder contains:

- `checkpoint.json` - Full checkpoint data with all question results
- `results.json` - Aggregated metrics and summary
- `summary.md` - Human-readable summary (optional)
