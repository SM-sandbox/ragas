# Orchestrator vs Direct Comparison Report

**Generated:** 2025-12-15 08:57:25
**Job ID:** bfai__eval66_g1_768_tt
**Config:** gemini-RETRIEVAL_QUERY

## Summary

| Metric | Orchestrator | Direct | Difference |
|--------|--------------|--------|------------|
| avg_score | 4.33 | 1.01 | +3.33 |
| pass_rate | 83.3% | 0.0% | +83.3% |

## Timing Comparison

| Phase | Orchestrator | Direct | Overhead |
|-------|--------------|--------|----------|
| Total (client) | 4.87s | 8.19s | -3.31s |
| Retrieval | 0.29s | 1.43s | -1.14s |
| Generation | 4.57s | 1.90s | +2.67s |

## Success/Failure

- **Orchestrator:** 12/224 successful
- **Direct:** 224/224 successful

## Conclusion

⚠️ **Score difference of 3.33** - May need investigation.