# Multi-Endpoint Checkpoint Comparison Report

**Generated:** 2025-12-26 14:35:32

**Gold Baseline:** C034 (pass_rate=93.2%, acceptable=97.6%)

## Summary

| Endpoint | Status | Pass Rate | Δ Baseline | Acceptable | Fail | MRR | Latency |
|----------|--------|-----------|------------|------------|------|-----|--------|
| stage | ✅ | 92.1% | 🔴 -1.1% | 98.5% | 1.5% | 0.739 | 58.9s |
| dev | ✅ | 90.6% | 🔴 -2.6% | 98.0% | 2.0% | 0.739 | 61.2s |

## Detailed Results

### stage

- **URL:** `https://bfai-app-xcshqh7sqq-ue.a.run.app`
- **Pass Rate:** 92.1% (baseline: 93.2%)
- **Acceptable Rate:** 98.5%
- **Fail Rate:** 1.5%
- **MRR:** 0.739
- **Recall@100:** 99.1%
- **Overall Score Avg:** 4.81
- **Questions:** 458
- **Duration:** 0.0s
- **Cost:** $0.0000
- **Avg Latency:** 58.9s
- **Results File:** `/Users/scottmacon/Documents/GitHub/ragas/clients_eval_data/BFAI/checkpoints/C066__2025-12-26__cloud__p25-3-flash-low__q458/results.json`

### dev

- **URL:** `https://bfai-app-eb2qyzdzvq-ue.a.run.app`
- **Pass Rate:** 90.6% (baseline: 93.2%)
- **Acceptable Rate:** 98.0%
- **Fail Rate:** 2.0%
- **MRR:** 0.739
- **Recall@100:** 99.1%
- **Overall Score Avg:** 4.79
- **Questions:** 458
- **Duration:** 0.0s
- **Cost:** $0.0000
- **Avg Latency:** 61.2s
- **Results File:** `/Users/scottmacon/Documents/GitHub/ragas/clients_eval_data/BFAI/checkpoints/C067__2025-12-26__cloud__p25-3-flash-low__q458/results.json`

## Analysis

