# Evaluation Metrics

The LLM-as-Judge evaluates each RAG answer on **6 dimensions** plus an overall score and pass/fail verdict.

## Overall Score (1-5)

The holistic quality rating of the RAG answer, considering all factors. This is the **primary metric** for comparison.

| Score | Meaning |
|-------|---------|
| **5** | Excellent - Answer is accurate, complete, and well-articulated |
| **4** | Good - Answer is mostly correct with minor gaps |
| **3** | Partial - Answer has some correct information but significant gaps |
| **2** | Poor - Answer is mostly incorrect or incomplete |
| **1** | Fail - Answer is wrong or irrelevant |

## Pass Rate (%)

Percentage of questions where the RAG system achieved an overall score of **4 or higher**. This represents the "production-ready" quality threshold.

| Verdict | Criteria |
|---------|----------|
| **Pass** | Overall score ≥ 4 |
| **Partial** | Overall score = 3 |
| **Fail** | Overall score ≤ 2 |

## Correctness (1-5)

Does the RAG answer match the ground truth **factually**? Measures whether the key facts and information are accurate.

| Score | Meaning |
|-------|---------|
| **5** | All facts match ground truth exactly |
| **3** | Some facts correct, some missing or wrong |
| **1** | Facts are incorrect or contradictory |

## Completeness (1-5)

Does the RAG answer cover **all key points** from the ground truth? Measures information coverage.

| Score | Meaning |
|-------|---------|
| **5** | All key points from ground truth are addressed |
| **3** | Some key points covered, others missing |
| **1** | Most key points are missing |

## Faithfulness (1-5)

Is the RAG answer **grounded in the retrieved context**? Measures whether the answer only uses information from the retrieved documents (no hallucination).

| Score | Meaning |
|-------|---------|
| **5** | Every claim is supported by retrieved context |
| **3** | Some claims supported, some unsupported |
| **1** | Answer contains significant hallucinated content |

## Relevance (1-5)

Does the RAG answer **directly address the question** asked? Measures how well the answer targets the specific question.

| Score | Meaning |
|-------|---------|
| **5** | Directly and precisely answers the question |
| **3** | Partially addresses the question |
| **1** | Answer is off-topic or tangential |

## Clarity (1-5)

Is the RAG answer **well-written and easy to understand**? Measures communication quality.

| Score | Meaning |
|-------|---------|
| **5** | Clear, well-structured, professional |
| **3** | Understandable but could be clearer |
| **1** | Confusing, poorly written, or incoherent |

## How Scores Are Derived

1. **Question + Ground Truth** are provided to the Judge LLM
2. **Retrieved Context** (top-12 chunks) is provided
3. **Generated Answer** from the RAG system is provided
4. **Judge LLM** (Gemini 2.5 Flash) evaluates on all 6 dimensions
5. **JSON response** with scores 1-5 for each dimension plus verdict

```json
{
  "correctness": 4,
  "completeness": 4,
  "faithfulness": 5,
  "relevance": 4,
  "clarity": 5,
  "overall_score": 4,
  "verdict": "pass",
  "explanation": "Answer correctly addresses the question..."
}
```

## Metric Correlations

From our experiments, we observed:

- **Correctness and Relevance** tend to correlate strongly
- **Faithfulness** is independent - a wrong answer can still be faithful to context
- **Clarity** is usually high unless the LLM produces malformed output
- **Completeness** is the hardest to achieve - requires comprehensive retrieval
