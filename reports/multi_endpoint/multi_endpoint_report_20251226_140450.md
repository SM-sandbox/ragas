# Multi-Endpoint Checkpoint Comparison Report

**Generated:** 2025-12-26 14:04:50

**Gold Baseline:** C034 (pass_rate=93.2%, acceptable=97.6%)

## Summary

| Endpoint | Status | Pass Rate | Δ Baseline | Acceptable | Fail | MRR | Latency |
|----------|--------|-----------|------------|------------|------|-----|--------|
| stage | ✅ | 91.7% | 🔴 -1.5% | 98.0% | 2.0% | 0.737 | 55.0s |
| dev | ✅ | 91.3% | 🔴 -1.9% | 98.3% | 1.7% | 0.738 | 54.2s |

## Detailed Results

### stage

- **URL:** `https://bfai-app-xcshqh7sqq-ue.a.run.app`
- **Pass Rate:** 91.7% (baseline: 93.2%)
- **Acceptable Rate:** 98.0%
- **Fail Rate:** 2.0%
- **MRR:** 0.737
- **Recall@100:** 99.1%
- **Overall Score Avg:** 4.81
- **Questions:** 458
- **Duration:** 0.0s
- **Cost:** $0.0000
- **Avg Latency:** 55.0s
- **Results File:** `/Users/scottmacon/Documents/GitHub/ragas/clients_eval_data/BFAI/checkpoints/C064__2025-12-26__cloud__p25-3-flash-low__q458/results.json`

### dev

- **URL:** `https://bfai-app-eb2qyzdzvq-ue.a.run.app`
- **Pass Rate:** 91.3% (baseline: 93.2%)
- **Acceptable Rate:** 98.3%
- **Fail Rate:** 1.7%
- **MRR:** 0.738
- **Recall@100:** 99.1%
- **Overall Score Avg:** 4.81
- **Questions:** 458
- **Duration:** 0.0s
- **Cost:** $0.0000
- **Avg Latency:** 54.2s
- **Results File:** `/Users/scottmacon/Documents/GitHub/ragas/clients_eval_data/BFAI/checkpoints/C065__2025-12-26__cloud__p25-3-flash-low__q458/results.json`

## Analysis

