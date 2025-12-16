# Temperature & Context Size Sweep - Complete Results

**Date:** 2025-12-16 09:24

---

## Experiment Overview

- **Questions:** 224
- **Embedding:** gemini-1536-RETRIEVAL_QUERY  
- **Generation Model:** gemini-2.5-flash
- **Judge Model:** gemini-2.5-flash (temp 0.0)
- **Retrieval:** Top 100 (hybrid 50/50)

---

## Temperature Sweep Results (Context Size = 10)

| Temp | Pass Rate | Overall | Correctness | Completeness | Faithfulness | Gen Time |
|------|-----------|---------|-------------|--------------|--------------|----------|
| 0.0 | 62.9% | 4.04 | 4.06 | 3.77 | 4.58 | 6.34s |
| 0.1 | 62.5% | 4.03 | 4.06 | 3.77 | 4.58 | 6.50s |
| 0.2 | 62.5% | 4.03 | 4.06 | 3.77 | 4.58 | 6.61s |
| 0.3 | 62.9% | 4.03 | 4.05 | 3.77 | 4.58 | 6.69s |

**Finding:** Temperature has minimal impact (0.0-0.3 all produce ~same results). Use 0.0 for determinism.

---

## Context Size Sweep Results (Temperature = 0.0)

| Context | Pass Rate | Overall | Correctness | Completeness | Faithfulness | Gen Time |
|---------|-----------|---------|-------------|--------------|--------------|----------|
| 5 | 57.6% | 3.89 | 4.00 | 3.53 | 4.60 | 6.08s |
| 10 | 62.9% | 4.04 | 4.06 | 3.77 | 4.58 | 6.88s |
| 15 | 66.1% | 4.15 | 4.17 | 3.94 | 4.63 | 7.25s |
| 20 | 71.4% | 4.29 | 4.23 | 4.12 | 4.62 | 7.13s |
| 25 | 73.7% | 4.38 | 4.37 | 4.23 | 4.72 | 7.39s |
| 50 | 73.7% | 4.40 | 4.45 | 4.23 | 4.76 | 7.49s |
| 100 | 76.3% | 4.44 | 4.45 | 4.29 | 4.72 | 8.97s |

**Best Context Size:** 100 chunks (Overall Score: 4.44)

---

## Key Findings

### 1. Temperature Has No Meaningful Impact
All temperatures (0.0-0.3) produced virtually identical results. Stick with **temp 0.0** for deterministic outputs.

### 2. More Context = Better Answers
Quality improves consistently as context size increases:
- 5 chunks → 25 chunks: Pass rate improves significantly
- Completeness benefits most from more context

### 3. Optimal Configuration
- **Temperature:** 0.0
- **Context Size:** 100 chunks (pending 50/100 results)

---

## Timing Summary

| Experiment | Gen Time (avg) |
|------------|----------------|
| context_10 | 6.88s |
| context_100 | 8.97s |
| context_15 | 7.25s |
| context_20 | 7.13s |
| context_25 | 7.39s |
| context_5 | 6.08s |
| context_50 | 7.49s |
| temp_0.0 | 6.34s |
| temp_0.1 | 6.50s |
| temp_0.2 | 6.61s |
| temp_0.3 | 6.69s |
