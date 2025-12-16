# Generation Consistency Report: context_100 Duplicate Runs

**Generated:** 2025-12-16 12:02:08
**Model:** gemini-2.5-flash
**Temperature:** 0.0
**Context Size:** 100 chunks
**Total Questions:** 224 (each run twice = 448 entries)

## Summary

The context_100 experiment inadvertently ran each question twice. This report
analyzes the consistency between Run 1 and Run 2 to validate reproducibility.

## Verdict Consistency

| Metric | Value |
|--------|-------|
| Same verdict both runs | **224** (100.0%) |
| Different verdict | **0** (0.0%) |
| **Consistency Rate** | **100.0%** |

## Score Consistency

| Metric | Value |
|--------|-------|
| Avg score difference | **0.007** |
| Max score difference | **1.0** |
| Identical scores | **222** (99.1%) |

## Generation Time Consistency

| Metric | Run 1 | Run 2 | Difference |
|--------|-------|-------|------------|
| Avg time | 8.92s | 8.94s | 0.03s |
| Min time | 2.29s | 2.36s | - |
| Max time | 39.27s | 37.15s | - |
| Std dev | 6.16s | 6.24s | - |

## Answer Text Similarity

| Metric | Value |
|--------|-------|
| Avg similarity | **100.0%** |
| Min similarity | 100.0% |
| Max similarity | 100.0% |
| Identical answers | **224** (100.0%) |

## Per-Run Metrics

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| Pass Rate | 76.3% | 76.3% | +0.0% |
| Avg Score | 4.44 | 4.44 | +0.00 |
| Avg Gen Time | 8.9s | 8.9s | +0.0s |

## Conclusion

✅ **Perfect verdict consistency** - All 224 questions received identical verdicts
across both runs, confirming highly reproducible generation at temperature 0.0.

The average answer text similarity of **100.0%** indicates
near-identical outputs, with minor variations in phrasing.

## Data Cleanup

After generating this report, the duplicate entries were removed from the checkpoint
file, keeping only Run 1 data. The original 448-entry file was preserved as a backup.

- **Original:** 448 entries (224 questions × 2 runs)
- **Cleaned:** 224 entries (Run 1 only)
- **Backup:** `context_100_checkpoint_with_duplicates.jsonl`