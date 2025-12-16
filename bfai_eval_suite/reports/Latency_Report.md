# Latency Report

**Generated:** 2025-12-14 23:09:02

This report shows timing data for retrieval, generation, and LLM judge calls across all embedding configurations.

## Summary Table

| Config | Retrieval (s) | Generation (s) | Judge (s) | Total (s) | Samples |
|--------|---------------|----------------|-----------|-----------|---------|
| azure-text-embedding-3-large | 0.70 | 1.53 | 4.23 | 4.72 | 1320 |
| gemini-1536-RETRIEVAL_QUERY | 1.24 | 4.64 | 10.20 | 16.07 | 448 |
| gemini-RETRIEVAL_QUERY | 3.21 | 4.52 | 9.90 | 17.63 | 1862 |
| gemini-SEMANTIC_SIMILARITY | 1.65 | 5.65 | 12.24 | 19.55 | 956 |
| gemini-with-tasktype | 1.62 | 2.48 | 5.89 | 10.00 | 30 |
| text-embedding-005 | 0.30 | 4.47 | 8.54 | 13.26 | 1006 |

## Detailed Statistics

### azure-text-embedding-3-large

| Metric | Avg (s) | Min (s) | Max (s) | Samples |
|--------|---------|---------|---------|---------|
| Retrieval | 0.698 | 0.193 | 13.270 | 1302 |
| Generation | 1.526 | 0.365 | 10.967 | 1104 |
| Judge/LLM | 4.225 | 1.025 | 27.903 | 658 |
| Total | 4.720 | 0.027 | 35.492 | 1320 |

### gemini-1536-RETRIEVAL_QUERY

| Metric | Avg (s) | Min (s) | Max (s) | Samples |
|--------|---------|---------|---------|---------|
| Retrieval | 1.238 | 0.862 | 3.844 | 448 |
| Generation | 4.642 | 1.143 | 45.305 | 448 |
| Judge/LLM | 10.195 | 1.990 | 124.222 | 448 |
| Total | 16.075 | 5.067 | 132.545 | 448 |

### gemini-RETRIEVAL_QUERY

| Metric | Avg (s) | Min (s) | Max (s) | Samples |
|--------|---------|---------|---------|---------|
| Retrieval | 3.206 | 0.736 | 128.282 | 1862 |
| Generation | 4.517 | 1.055 | 134.655 | 1862 |
| Judge/LLM | 9.904 | 2.201 | 185.301 | 1862 |
| Total | 17.627 | 4.584 | 192.895 | 1862 |

### gemini-SEMANTIC_SIMILARITY

| Metric | Avg (s) | Min (s) | Max (s) | Samples |
|--------|---------|---------|---------|---------|
| Retrieval | 1.655 | 0.731 | 66.243 | 956 |
| Generation | 5.653 | 1.123 | 283.137 | 956 |
| Judge/LLM | 12.238 | 1.966 | 149.088 | 956 |
| Total | 19.546 | 4.665 | 295.248 | 956 |

### gemini-with-tasktype

| Metric | Avg (s) | Min (s) | Max (s) | Samples |
|--------|---------|---------|---------|---------|
| Retrieval | 1.620 | 0.937 | 2.237 | 30 |
| Generation | 2.485 | 1.461 | 4.219 | 30 |
| Judge/LLM | 5.891 | 3.061 | 12.816 | 30 |
| Total | 9.996 | 6.055 | 16.457 | 30 |

### text-embedding-005

| Metric | Avg (s) | Min (s) | Max (s) | Samples |
|--------|---------|---------|---------|---------|
| Retrieval | 0.304 | 0.113 | 4.139 | 1006 |
| Generation | 4.469 | 1.021 | 25.438 | 996 |
| Judge/LLM | 8.536 | 2.210 | 46.920 | 1006 |
| Total | 13.264 | 3.918 | 49.573 | 1006 |

## Observations

- **Fastest Retrieval:** text-embedding-005 (0.30s avg)
- **Fastest Generation:** azure-text-embedding-3-large (1.53s avg)
- **Fastest Total:** azure-text-embedding-3-large (4.72s avg)