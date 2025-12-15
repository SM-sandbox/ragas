# BrightFox RAG Evaluation

Welcome to the BrightFox RAG Evaluation documentation. This project provides a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems using the BrightFox SCADA/Solar document corpus.

## What This Project Does

1. **Evaluates RAG quality** using LLM-as-Judge methodology
2. **Compares platforms** (GCP Vertex AI vs Azure AI Search)
3. **Tests embedding models** to find optimal configurations
4. **Generates Q&A test corpora** from source documents

## Quick Links

| Section | Description |
|---------|-------------|
| [Architecture](architecture.md) | System design and data flow |
| [Metrics](metrics.md) | What each evaluation metric means |
| [Experiments](experiments/index.md) | Completed tests and findings |
| [Runbooks](runbooks/index.md) | How to run evaluations |

## Key Findings

| Experiment | Winner | Delta |
|------------|--------|-------|
| **Embedding Model** | gemini-embedding-001 (RETRIEVAL_QUERY) | +0.15 vs text-embedding-005 |
| **Embedding Dimensions** | 768 sufficient | 1536 shows no improvement |
| **Platform Comparison** | GCP Vertex AI | +0.23 score, +13% pass rate vs Azure |

## Project Structure

```
brightfox_eval/
├── src/           # Core reusable code
├── corpus/        # Q&A test corpus (224 questions)
├── experiments/   # Test runs with data
├── reports/       # Polished comparison reports
├── scripts/       # Runnable evaluation scripts
├── docs/          # This documentation
└── data/          # Source document chunks
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with GCP
gcloud auth application-default login

# Run an evaluation
cd scripts/
python embedding_comparison_direct.py --mode recall-only
```
