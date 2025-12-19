# Benchmarks

This folder contains **gold standard benchmarks** for RAG evaluation. A benchmark represents the best-known performance baseline that all future tests are compared against.

## Folder Structure

```
benchmarks/
├── README.md                    # This file
├── BFAI/                        # Client-specific benchmarks
│   ├── benchmark_BFAI_v1.3__cloud__2025-12-18.json  # Current benchmark
│   ├── archive/                 # Previous benchmarks
│   └── README.md                # Client benchmark history
├── BENCHMARK_SCHEMA.json        # JSON schema for benchmarks
└── REPORT_SCHEMA.json           # JSON schema for comparison reports
```

## Naming Convention

### Benchmark Files

```
benchmark_{CLIENT}_v{MAJOR}.{MINOR}__{TYPE}__{DATE}.json
```

| Component | Description | Example |
|-----------|-------------|---------|
| `CLIENT` | Client identifier | `BFAI` |
| `MAJOR.MINOR` | Version number | `1.3` |
| `TYPE` | Environment type | `local` or `cloud` |
| `DATE` | Creation date | `2025-12-18` |

**Example:** `benchmark_BFAI_v1.3__cloud__2025-12-18.json`

### Version Numbering

- **Major version** (1.x → 2.x): Significant changes
  - New model (e.g., Gemini 2.5 → Gemini 3)
  - New index or embedding
  - Major configuration change
  
- **Minor version** (1.2 → 1.3): Incremental improvements
  - Parameter tuning
  - Bug fixes
  - Same model/index

## Benchmark vs Baseline

| Term | Purpose | Location |
|------|---------|----------|
| **Benchmark** | Gold standard for comparison | `benchmarks/` |
| **Baseline** | Historical snapshots | `baselines/` |

- **Benchmark** = "This is what we compare against"
- **Baseline** = "This is what we achieved at a point in time"

When a test run exceeds the benchmark, you may promote it to the new benchmark.

## Workflow

### 1. Running a Test

```bash
# Run evaluation against Cloud Run
python scripts/eval/run_gold_eval.py --cloud --workers 5

# Output: reports/gold_standard_eval/results_p25_cloud.json
```

### 2. Comparing to Benchmark

```bash
# Generate comparison report
python scripts/eval/generate_report.py \
  --test reports/gold_standard_eval/results_p25_cloud.json \
  --benchmark v1.3

# Output: 
#   - reports/gold_standard_eval/report_BFAI__R002__2025-12-19.json
#   - reports/gold_standard_eval/final_reports/R002_*.md
```

### 3. Promoting a New Benchmark

When a test run is better than the current benchmark:

```bash
# Set new benchmark
python scripts/eval/benchmark_manager.py \
  --set-benchmark \
  --source reports/gold_standard_eval/results_p25_cloud.json \
  --version 1.4

# This will:
#   1. Archive the current benchmark
#   2. Create new benchmark file
#   3. Update the README
```

## Benchmark Criteria

A test run should be promoted to benchmark when:

1. **Quality**: Pass rate ≥ current benchmark
2. **Reliability**: Fail rate ≤ current benchmark
3. **Latency**: Client experience ≤ current benchmark (or within 10%)
4. **Cost**: Per-question cost ≤ current benchmark (or within 10%)
5. **Stability**: Run completed without errors

## Current Benchmarks

| Client | Version | Type | Date | Pass Rate | Fail Rate |
|--------|---------|------|------|-----------|-----------|
| BFAI | v1.3 | cloud | 2025-12-18 | 92.6% | 0.9% |

## JSON Schema

See `BENCHMARK_SCHEMA.json` for the complete schema definition.

Key sections:
- `environment`: Local or Cloud configuration
- `config`: Model, precision, recall settings
- `metrics`: Pass/partial/fail rates, recall, MRR
- `latency`: Total and per-phase timing
- `tokens`: Input/output/thinking token counts
- `cost`: USD cost estimates
- `breakdown_by_type`: Single-hop vs multi-hop
- `breakdown_by_difficulty`: Easy/medium/hard
- `failures`: List of failed question IDs
