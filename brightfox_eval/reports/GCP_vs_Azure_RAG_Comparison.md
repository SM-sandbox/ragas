# GCP vs Azure RAG Comparison

**Date:** December 14, 2024  
**Evaluation Corpus:** 193 questions (apples-to-apples subset)

---

## Executive Summary

**Winner: GCP Vertex AI** outperforms Azure AI Search by **+0.23 points** (5.9% improvement) and **+13% pass rate** on an apples-to-apples comparison.

---

## Apples-to-Apples Comparison

### Why This Comparison Is Fair

Both configurations use:
- **Same retrieval approach:** Pure vector search, top-12 results, no reranker
- **Same judge:** Gemini 2.5 Flash (eliminates judge bias)
- **Same question set:** 193 questions (excluding 30 questions from documents missing in Azure)
- **Same corpus intent:** Technical documents for energy/electrical domain

The only differences are the platform-specific components:
- **Retrieval:** Azure AI Search vs GCP Vertex AI Vector Search
- **Embeddings:** Azure text-embedding-3-large (3072 dim) vs GCP gemini-embedding-001 (768 dim)
- **Generation LLM:** Azure GPT-4.1-mini vs GCP Gemini 2.5 Flash

### Results

| Metric | Azure | GCP | Delta | Winner |
|--------|-------|-----|-------|--------|
| **Overall Score** | 3.90 | 4.13 | +0.23 | GCP |
| **Pass Rate** | 56% | 69% | +13% | GCP |
| **Correctness** | 3.53 | 4.06 | +0.53 | GCP |
| **Completeness** | 3.41 | 3.92 | +0.51 | GCP |
| **Faithfulness** | 3.65 | 3.84 | +0.19 | GCP |
| **Relevance** | 3.53 | 4.06 | +0.53 | GCP |
| **Clarity** | 3.65 | 3.84 | +0.19 | GCP |

### Configuration Details

| Component | Azure | GCP |
|-----------|-------|-----|
| **Search Service** | Azure AI Search (asosearch-stg) | Vertex AI Vector Search |
| **Index** | bf-demo (55 docs, 8,691 chunks) | eval_66 (65 docs, 6,059 chunks) |
| **Embedding Model** | text-embedding-3-large | gemini-embedding-001 |
| **Embedding Dimensions** | 3072 | 768 |
| **Generation LLM** | GPT-4.1-mini | Gemini 2.5 Flash |
| **Retrieval Top-K** | 12 | 12 |
| **Reranker** | None | None |
| **Judge LLM** | Gemini 2.5 Flash | Gemini 2.5 Flash |

---

## Document Corpus Analysis

The Q&A evaluation corpus contains **224 questions** derived from **40 unique source documents**.

### Corpus Coverage

| Index | Total Docs | Q&A Source Docs Present | Coverage |
|-------|------------|------------------------|----------|
| GCP | 65 | 40/40 | **100%** |
| Azure | 55 | 32/40 | **80%** |

### Questions Excluded from Comparison

30 questions were excluded because their source documents are missing from Azure:

| Document | Questions |
|----------|-----------|
| 1-SO50016-BOYNE-MOUNTAIN-RESORT_SLD.0.12C.ADD-ON.SO87932.IFC | 4 |
| AcquiSuite-Basics---External | 2 |
| Acquisuite Backup and Restore | 4 |
| DiGiWR21WR31ModemBasicsv1 | 6 |
| EPEC_1200-6000A UL 891 SWBD (Crit Power) Flyer | 6 |
| Manual-PVI-Central-250-300 | 4 |
| PFMG - OM Agreement (WSD)_Executed | 2 |
| PVI-6000-OUTD-US Service Manual | 2 |
| **Total** | **30** |

---

## Key Observations

1. **GCP wins despite smaller embeddings:** GCP's 768-dimension embeddings outperform Azure's 3072-dimension embeddings, suggesting embedding quality matters more than size.

2. **Correctness is the biggest gap:** GCP scores +0.53 higher on correctness, indicating better retrieval of factually relevant content.

3. **Azure has corpus gaps:** 8 documents (20% of Q&A sources) are missing from Azure, which would further hurt Azure's score if included.

4. **Same judge eliminates bias:** Using Gemini 2.5 Flash as the judge for both platforms ensures the comparison reflects actual RAG quality, not judge preferences.

---

## Metric Definitions

### Overall Score (1-5)
The holistic quality rating of the RAG answer, considering all factors. This is the primary metric for comparison.

- **5:** Excellent - Answer is accurate, complete, and well-articulated
- **4:** Good - Answer is mostly correct with minor gaps
- **3:** Partial - Answer has some correct information but significant gaps
- **2:** Poor - Answer is mostly incorrect or incomplete
- **1:** Fail - Answer is wrong or irrelevant

### Pass Rate (%)
Percentage of questions where the RAG system achieved an overall score of **4 or higher**. This represents the "production-ready" quality threshold.

- **Pass:** Overall score ≥ 4
- **Partial:** Overall score = 3
- **Fail:** Overall score ≤ 2

### Correctness (1-5)
Does the RAG answer match the ground truth factually? Measures whether the key facts and information are accurate.

- **5:** All facts match ground truth exactly
- **3:** Some facts correct, some missing or wrong
- **1:** Facts are incorrect or contradictory

### Completeness (1-5)
Does the RAG answer cover all key points from the ground truth? Measures information coverage.

- **5:** All key points from ground truth are addressed
- **3:** Some key points covered, others missing
- **1:** Most key points are missing

### Faithfulness (1-5)
Is the RAG answer grounded in the retrieved context? Measures whether the answer only uses information from the retrieved documents (no hallucination).

- **5:** Every claim is supported by retrieved context
- **3:** Some claims supported, some unsupported
- **1:** Answer contains significant hallucinated content

### Relevance (1-5)
Does the RAG answer directly address the question asked? Measures how well the answer targets the specific question.

- **5:** Directly and precisely answers the question
- **3:** Partially addresses the question
- **1:** Answer is off-topic or tangential

### Clarity (1-5)
Is the RAG answer well-written and easy to understand? Measures communication quality.

- **5:** Clear, well-structured, professional
- **3:** Understandable but could be clearer
- **1:** Confusing, poorly written, or incoherent

---

## Methodology

### Evaluation Pipeline

1. **Question Selection:** 193 questions from Q&A corpus (excluding questions from documents missing in Azure)
2. **Retrieval:** Top-12 chunks retrieved via vector search (no reranker)
3. **Generation:** Platform-specific LLM generates answer from retrieved context
4. **Judging:** Gemini 2.5 Flash evaluates answer against ground truth on all 6 dimensions
5. **Scoring:** Metrics aggregated across all questions

### Judge Prompt
The LLM judge evaluates each RAG answer on the 6 dimensions above, returning scores 1-5 for each plus an overall score and pass/partial/fail verdict.

### Ground Truth
Each question has a human-verified ground truth answer derived from the source document. The judge compares RAG answers against this ground truth.

---

*Generated by BrightFox AI Evaluation Framework*
