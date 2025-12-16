# BrightFox RAG Evaluation

End-to-end RAG evaluation pipeline for the BrightFox SCADA/Solar document corpus.

## Overview

This project evaluates RAG systems using:

- **Vertex AI Vector Search** or **Azure AI Search** for retrieval
- **Gemini 2.5 Flash** or **Azure OpenAI** for generation
- **LLM-as-Judge** methodology for evaluation
- Multiple embedding models (gemini-embedding-001, text-embedding-005, text-embedding-3-large)

## Project Structure

```
brightfox_eval/
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── .gitignore
│
├── src/                      # Core reusable code
│   ├── retriever.py          # Vector search
│   ├── reranker.py           # Reranker logic
│   ├── generator.py          # LLM generation
│   ├── judge.py              # LLM-as-judge evaluation
│   └── vector_search.py      # Vertex AI client
│
├── corpus/                   # Master Q&A corpus
│   ├── qa_corpus_200.json    # 224 questions
│   ├── document_inventory.md # Master doc list
│   └── knowledge_graph.json  # Document relationships
│
├── experiments/              # Test runs (data gitignored, synced to GCS)
│   ├── 2024-12-14_embedding_model_comparison/
│   ├── 2024-12-14_embedding_dimension_test/
│   └── 2024-12-14_azure_vs_gcp/
│
├── reports/                  # Final polished reports
│   └── GCP_vs_Azure_RAG_Comparison.md
│
├── scripts/                  # Runnable test scripts
│   ├── azure_comparison.py
│   ├── embedding_comparison_direct.py
│   └── precision_test.py
│
└── data/                     # Source document chunks
    └── all_chunks.json
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with GCP
gcloud auth application-default login

# 3. Run embedding comparison
cd scripts/
python embedding_comparison_direct.py --mode recall-only

# 4. Run Azure comparison
python azure_comparison.py --filter-missing
```

## Completed Experiments

| Experiment | Finding |
|------------|---------|
| **Embedding Model Comparison** | gemini-embedding-001 with RETRIEVAL_QUERY task type wins |
| **Embedding Dimension Test** | 768 dimensions sufficient, 1536 shows no improvement |
| **Azure vs GCP** | GCP wins by +0.23 score, +13% pass rate |

See `experiments/*/README.md` for details on each test.

## Evaluation Metrics

- **Overall Score (1-5)** - Holistic quality rating
- **Pass Rate** - % of questions scoring ≥4
- **Correctness** - Factual accuracy vs ground truth
- **Completeness** - Coverage of key points
- **Faithfulness** - Grounded in retrieved context
- **Relevance** - Directly answers the question
- **Clarity** - Well-written and clear

## GCS Backup

Experiment data is synced to GCS:

```bash
gsutil -m rsync -r experiments/ gs://brightfox-eval-experiments/
```

## GCP Resources

| Resource | Value |
|----------|-------|
| Project | civic-athlete-473921-c0 |
| Location | us-east1 |
| Best Embedding | gemini-embedding-001 (768 dim, RETRIEVAL_QUERY) |
| LLM | gemini-2.5-flash |
