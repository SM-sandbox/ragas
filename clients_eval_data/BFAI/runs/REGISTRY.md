# Runs Registry

Full evaluation runs with variations against the gold standard corpus.

## Naming Convention

```text
R{NNN}__{YYYY-MM-DD}__{mode}__{config-summary}/
```

- **Mode:** `L` = Local, `C` = Cloud

## Registry

| ID | Date | Mode | Config | Questions | Pass Rate | Fail Rate | Notes |
|----|------|------|--------|-----------|-----------|-----------|-------|
| R001 | 2025-12-18 | Local | p25-flash-2.5 | 458 | 85.6% | 3.9% | Gemini 2.5 Flash baseline |
| R002 | 2025-12-18 | Local | p25-flash-2.5 | 458 | 85.6% | 3.9% | Gemini 2.5 Flash run 2 |
| R003 | 2025-12-18 | Local | p25-flash-2.5 | 458 | 85.6% | 3.9% | Gemini 2.5 Flash run 3 |
| R004 | 2025-12-18 | Cloud | p25-flash-3 | 458 | 92.6% | 0.9% | Gemini 3 Flash Preview cloud |
| R005 | 2025-12-19 | Local | p25-flash-3 | 458 | 92.4% | 1.1% | Gemini 3 Flash Preview local |

## Purpose

Runs are used for:

- **Hypothesis testing:** What happens if I change precision, model, or config?
- **Model comparison:** Compare different generator models
- **Configuration tuning:** Find optimal settings

## Folder Contents

Each run folder contains:

- `results.jsonl` - Per-question results (one JSON object per line)
- `run_summary.json` - Aggregated metrics and configuration
- `comparison.md` - Comparison to baseline (if applicable)
