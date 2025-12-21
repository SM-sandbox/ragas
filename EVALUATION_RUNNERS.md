# Evaluation Runners

This document provides an overview of the evaluation entry points for the RAG pipeline.

## Quick Start

### Run a Checkpoint (CI/CD / Daily Validation)
```bash
python scripts/run_checkpoint.py --cloud    # Cloud mode (default)
python scripts/run_checkpoint.py --local    # Local mode
python scripts/run_checkpoint.py            # Interactive (prompts for mode)
```

### Run an Experiment (Exploratory Testing)
```bash
python scripts/run_experiment.py --cloud --recall 200   # Test recall@200
python scripts/run_experiment.py --cloud --model gemini-2.5-flash  # Different model
python scripts/run_experiment.py                        # Interactive mode
```

---

## Checkpoint vs Experiment

| Aspect | Checkpoint | Experiment |
|--------|------------|------------|
| **Purpose** | CI/CD, daily validation, regression testing | Exploratory testing, parameter tuning |
| **Configuration** | LOCKED (cannot change) | FLEXIBLE (change anything) |
| **Output Directory** | `checkpoints/C###` | `experiments/E###` |
| **Variable** | Only local vs cloud | Any parameter |
| **Use Case** | "Did we regress?" | "What if we try X?" |

---

## Checkpoint Configuration (LOCKED)

Checkpoints use a **fixed configuration** to ensure consistent comparisons over time.

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Client** | BFAI | Fixed |
| **Corpus** | QA_BFAI_gold_v1-0__q458.json | 458 questions |
| **Index** | bfai__eval66a_g1_1536_tt | Fixed |
| **Generator Model** | gemini-3-flash-preview | Fixed |
| **Generator Reasoning** | low | Fixed |
| **Generator Temperature** | 0.0 | Reproducibility |
| **Generator Seed** | 42 | Fixed |
| **Judge Model** | gemini-3-flash-preview | Fixed |
| **Judge Reasoning** | low | Fixed |
| **Judge Temperature** | 0.0 | Fixed |
| **Judge Seed** | 42 | Fixed |
| **Recall@K** | 100 | Fixed |
| **Precision@K** | 25 | Fixed |
| **Hybrid Search** | enabled | Fixed |
| **RRF Alpha** | 0.5 | 50/50 keyword + semantic |
| **Reranking** | enabled | Fixed |
| **Workers** | 100 | Smart throttler limits to 20 concurrent |
| **Mode** | local OR cloud | **Only variable** |

### Checkpoint Entry Point
```bash
python scripts/run_checkpoint.py [--cloud|--local] [--dry-run]
```

**Options:**
- `--cloud` - Run against Cloud Run endpoint (default)
- `--local` - Run against local gRAG_v3 pipeline
- `--dry-run` - Show configuration without running

---

## Experiment Configuration (FLEXIBLE)

Experiments start with the same defaults as checkpoints but allow you to change any parameter.

### Experiment Entry Point
```bash
python scripts/run_experiment.py [options]
```

**Options:**
- `--cloud` - Run against Cloud Run endpoint
- `--local` - Run against local gRAG_v3 pipeline
- `--recall N` - Change recall@K (default: 100)
- `--precision N` - Change precision@K (default: 25)
- `--model NAME` - Change generator model
- `--reasoning [low|high]` - Change reasoning effort
- `--quick N` - Run only N questions (for quick tests)
- `--dry-run` - Show configuration without running

### Example Experiments
```bash
# Test higher recall
python scripts/run_experiment.py --cloud --recall 200

# Test different model
python scripts/run_experiment.py --cloud --model gemini-2.5-flash

# Quick test with 30 questions
python scripts/run_experiment.py --cloud --quick 30

# Multiple changes
python scripts/run_experiment.py --cloud --recall 200 --precision 50 --reasoning high
```

---

## Output Structure

### Checkpoints
```
clients_eval_data/BFAI/checkpoints/
├── C016__2025-12-21__cloud__p25-3-flash-low__q458/
│   ├── results.json              # Full results
│   ├── checkpoint.json           # Progress checkpoint
│   └── checkpoint_report_C016.md # Generated report
├── registry.json                 # Checkpoint registry
└── ...
```

### Experiments
```
clients_eval_data/BFAI/experiments/
├── E001__2025-12-21__cloud__p25-3-flash-low__q458/
│   ├── results.json
│   ├── checkpoint.json
│   └── checkpoint_report_E001.md
└── ...
```

---

## Smart Throttler

Both checkpoints and experiments use the **Smart Throttler** to manage API rate limits:

- **Max Concurrent Requests**: 20 (via semaphore)
- **RPM Limit**: 1000 requests/minute
- **TPM Limit**: 1,000,000 tokens/minute
- **Threshold**: 90% (pre-emptive throttling)

You can specify `--workers 100` but the throttler will only allow 20 concurrent requests at a time. This prevents overwhelming the API while maintaining high throughput.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/checkpoint_config.yaml` | Locked checkpoint configuration |
| `config/experiment_config.yaml` | Default experiment configuration |
| `config/model_pricing.yaml` | Model pricing for cost estimates |

---

## Troubleshooting

### "Registry not found" Warning
The registry is auto-created on first checkpoint run. This warning is harmless.

### Slow Cloud Mode
Cloud mode includes a warmup phase (~30s) to wake up Cloud Run instances. This is normal.

### Rate Limiting
If you see throttling messages, the smart throttler is working correctly. It will automatically wait and retry.

---

## See Also

- `scripts/run_checkpoint.py` - Checkpoint runner source
- `scripts/run_experiment.py` - Experiment runner source
- `lib/core/evaluator.py` - Core evaluation logic
- `lib/core/smart_throttler/` - Rate limiting implementation
