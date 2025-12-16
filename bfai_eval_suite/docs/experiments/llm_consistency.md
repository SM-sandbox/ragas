# LLM Consistency Test

**Status:** Planned  
**Priority:** Medium

## Objective

Measure the variance in LLM judge scores when evaluating the same RAG responses multiple times. This helps establish confidence intervals for our evaluation metrics.

## Methodology

1. Select a subset of questions (e.g., 50 questions)
2. Run the full RAG pipeline + judge evaluation 5 times for each question
3. Calculate variance in scores across runs
4. Report standard deviation and confidence intervals

## Metrics to Track

| Metric | Description |
|--------|-------------|
| **Score Variance** | Standard deviation of overall scores across runs |
| **Verdict Consistency** | % of questions with same pass/partial/fail across all runs |
| **Per-Dimension Variance** | Variance in correctness, completeness, faithfulness, relevance, clarity |

## Expected Outcomes

- Establish margin of error for evaluation scores
- Identify questions with high variance (may indicate ambiguous ground truth)
- Validate that temperature=0 produces consistent results

## Configuration

```python
# Recommended settings for consistency testing
LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0  # Deterministic output
NUM_RUNS = 5
SAMPLE_SIZE = 50  # Questions to test
```

## Status

**Not yet implemented.** This experiment is planned to validate the reliability of our LLM-as-judge evaluation approach.

## Related

- [Embedding Model Comparison](embedding_model_comparison.md) - Uses LLM judge
- [Question Relevance](question_relevance.md) - Also uses LLM evaluation
