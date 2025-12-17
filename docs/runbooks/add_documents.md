# Add New Documents

How to add new documents to the evaluation corpus.

## Steps

### 1. Add Document Chunks to GCS

Upload chunked documents to the GCS bucket:

```bash
gsutil cp your_chunks.json gs://your-bucket/chunks/
```

### 2. Update Vector Search Index

Re-index the Vector Search endpoint with new embeddings.

### 3. Generate Q&A Pairs

Use the Q&A generator to create test questions:

```bash
cd scripts/
python generate_qa_200.py --documents "new_doc_1,new_doc_2"
```

### 4. Update Document Inventory

Add the new documents to `corpus/document_inventory.md`.

### 5. Re-run Evaluation

```bash
python embedding_comparison_direct.py --mode recall-only
```
