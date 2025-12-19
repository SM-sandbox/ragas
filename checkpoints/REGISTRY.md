# Checkpoints Registry

CICD and daily health check runs against the gold standard baseline.

## Naming Convention

```text
C{NNN}__{YYYY-MM-DD}__{mode}__{config-summary}/
```

- **Mode:** `L` = Local, `C` = Cloud

## Registry

| ID | Date | Mode | Config | Questions | Pass Rate | Fail Rate | Notes |
|----|------|------|--------|-----------|-----------|-----------|-------|
| C001 | 2025-12-19 | Cloud | p25-flash | 458 | 92.6% | 0.9% | Cloud baseline p25 |
| C002 | 2025-12-18 | Cloud | p12-flash | 458 | 86.2% | 3.5% | Cloud p12 comparison |
| C003 | 2025-12-19 | Local | p25-flash | 458 | 92.4% | 1.1% | Local baseline p25 |
| C004 | 2025-12-19 | Cloud | p25-pro | 458 | 92.1% | 1.1% | Gemini 3 Pro test |

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
