# BFAI Benchmarks

Gold standard benchmarks for the BFAI RAG evaluation corpus.

## Current Benchmark

| Version | Type | Date | Pass Rate | Fail Rate | Client Latency |
|---------|------|------|-----------|-----------|----------------|
| **v1.3** | cloud | 2025-12-18 | 92.6% | 0.9% | 8.14s |

**File:** `benchmark_BFAI_v1.3__cloud__2025-12-18.json`

## Benchmark History

| Version | Type | Date | Pass Rate | Notes |
|---------|------|------|-----------|-------|
| v1.3 | cloud | 2025-12-18 | 92.6% | First cloud benchmark, Gemini 3 Flash Preview |
| v1.2 | local | 2025-12-18 | 92.4% | Local baseline v2, identical config to cloud |
| v1.1 | local | 2025-12-17 | 88.9% | Precision@25, improved from v1.0 |
| v1.0 | local | 2025-12-15 | 85.6% | Initial benchmark |

## Configuration

All v1.x benchmarks use:

- **Model:** gemini-3-flash-preview
- **Reasoning:** low
- **Temperature:** 0.0
- **Recall Top K:** 100
- **Precision Top N:** 25
- **Hybrid Search:** enabled
- **Reranking:** enabled
- **Embedding:** gemini-embedding-001 (1536d)
- **Index:** bfai__eval66a_g1_1536_tt

## Corpus

- **File:** QA_BFAI_gold_v1-0__q458.json
- **Questions:** 458
- **Single-hop:** 222
- **Multi-hop:** 236
- **Easy:** 161, **Medium:** 161, **Hard:** 136

## Why v1.3 is the Benchmark

1. **Cloud Run** - Represents actual user experience
2. **Highest pass rate** - 92.6% (best achieved)
3. **Lowest fail rate** - 0.9% (best achieved)
4. **Fastest client latency** - 8.14s (excludes judge)
5. **Matched config** - Identical to local baseline for apples-to-apples

## Promoting a New Benchmark

To promote a test run to the new benchmark:

```bash
python scripts/eval/benchmark_manager.py \
  --set-benchmark \
  --source <results_file.json> \
  --version 1.4
```

This will:
1. Move current benchmark to `archive/`
2. Create new benchmark file
3. Update this README

## Archive

Previous benchmarks are stored in `archive/` for historical reference.
