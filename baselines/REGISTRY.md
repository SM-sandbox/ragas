# Baselines Registry

Gold standard baseline snapshots for RAG evaluation comparison.

## Naming Convention

```text
baseline_gold__{client}__{version}__{date}__q{count}.json
```

## Registry

| Version | Client | Date | Questions | Pass Rate | Fail Rate | Model | Notes |
|---------|--------|------|-----------|-----------|-----------|-------|-------|
| v1 | BFAI | 2025-12-17 | 458 | 85.6% | 3.9% | gemini-2.5-flash | Initial baseline |
| v2 | BFAI | 2025-12-18 | 458 | 92.4% | 1.1% | gemini-3-flash-preview | Major improvement over v1 |

## Usage

Baselines are used by `lib/core/baseline_manager.py` for:

- Loading the latest baseline for comparison
- Comparing new runs against established metrics
- Gating CICD pipelines based on regression thresholds

## Adding a New Baseline

1. Run a full evaluation with `--update-baseline` flag
2. Verify metrics meet quality thresholds
3. Update this registry with the new entry
4. Update `registry.json` programmatically or manually

