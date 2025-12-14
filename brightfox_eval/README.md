# BrightFox RAG Evaluation

End-to-end RAG evaluation pipeline using Ragas for the BrightFox SCADA/Solar document corpus.

## Overview

This project evaluates a RAG system using:
- **Vertex AI Vector Search** for document retrieval
- **Gemini 1.5 Pro** for LLM operations
- **text-embedding-005** for embeddings
- **Ragas** for evaluation metrics

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with GCP
gcloud auth application-default login

# 3. Run the full pipeline
python run_full_pipeline.py
```

## Pipeline Steps

1. **Download Chunks** - Fetches 65 document chunk files from GCS
2. **Generate Questions** - Creates 25 single-hop + 25 multi-hop questions
3. **Rate Questions** - LLM rates each question 1-5 for quality
4. **Filter Questions** - Keeps only 4s and 5s (discards 3 and below)
5. **Run Evaluation** - Executes Ragas evaluation on filtered questions

## Configuration

Edit `config.py` to modify:
- GCP project/location
- Vector Search endpoint
- LLM model
- Number of questions to generate
- Minimum quality score threshold

## Output Files

All outputs are saved to `output/`:
- `generated_questions.json` - All generated questions
- `rated_questions.json` - Questions with quality ratings
- `filtered_questions.json` - High-quality questions only
- `discarded_questions.json` - Low-quality questions (for reference)
- `ragas_evaluation_results.json` - Final evaluation metrics

## Metrics

The evaluation produces these Ragas metrics:
- **Faithfulness** - Is the answer grounded in the context?
- **Answer Relevancy** - Is the answer relevant to the question?
- **Context Precision** - Are the retrieved contexts relevant?
- **Context Recall** - Does the context cover the ground truth?

## Project Structure

```
brightfox_eval/
├── config.py              # Configuration
├── run_full_pipeline.py   # Main entry point
├── scripts/
│   └── download_chunks.py # GCS chunk downloader
├── vector_search.py       # Vertex AI Vector Search client
├── llm_client.py          # Gemini LLM client
├── question_generator.py  # Question generation
├── question_rater.py      # Question quality rating
├── rag_retriever.py       # RAG retrieval
├── ragas_evaluator.py     # Ragas evaluation
├── data/                  # Downloaded chunks
└── output/                # Evaluation results
```

## GCP Resources

| Resource | Value |
|----------|-------|
| Project | civic-athlete-473921-c0 |
| Location | us-east1 |
| Vector Search Endpoint | 1807654290668388352 |
| Deployed Index | idx_brightfoxai_evalv3_autoscale |
| Embedding Model | text-embedding-005 |
| LLM Model | gemini-1.5-pro-002 |
