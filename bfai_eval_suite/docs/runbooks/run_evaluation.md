# Run Evaluation

How to run a RAG evaluation against the Q&A corpus.

## Prerequisites

1. **GCP Authentication**
   ```bash
   gcloud auth application-default login
   ```

2. **Azure Environment Variables** (for Azure comparison only)
   ```bash
   export AZURE_SEARCH_KEY="your-search-key"
   export AZURE_OPENAI_KEY="your-openai-key"
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## GCP Embedding Comparison

Test different embedding models on GCP:

```bash
cd scripts/

# Recall-only (no reranker)
python embedding_comparison_direct.py --mode recall-only

# With reranker
python embedding_comparison_direct.py --mode rerank

# Single config only
python embedding_comparison_direct.py --config gemini-RETRIEVAL_QUERY --mode recall-only
```

## Azure vs GCP Comparison

```bash
cd scripts/

# Full 224 questions
python azure_comparison.py --workers 8

# Filtered (exclude docs missing in Azure)
python azure_comparison.py --filter-missing --workers 8
```

## Output

Results are saved to `experiments/` with timestamps:

- `embedding_comparison_*.json` - GCP results
- `azure_evaluation_*.json` - Azure results

## Viewing Results

```python
import json

with open('experiments/2024-12-14_azure_vs_gcp/data/azure_evaluation_*.json') as f:
    data = json.load(f)
    
print(f"Score: {data['metrics']['overall_score']:.2f}")
print(f"Pass Rate: {data['metrics']['pass_rate']*100:.0f}%")
```
