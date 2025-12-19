# BFAI GCS Manifest

**Client:** BFAI (BrightFox AI Demo Suite)  
**Bucket:** `gs://bfai-eval-suite/BFAI/`  
**Last Updated:** 2025-12-17

---

## Corpus Data

| Local Path | GCS Path | Description |
|------------|----------|-------------|
| `corpus/documents/` | `gs://bfai-eval-suite/BFAI/corpus/documents/` | 65 source PDFs |
| `corpus/metadata/` | `gs://bfai-eval-suite/BFAI/corpus/metadata/` | Per-document analysis |
| `corpus/chunks/` | `gs://bfai-eval-suite/BFAI/corpus/chunks/` | Chunked text for retrieval |
| `corpus/knowledge_graph.json` | `gs://bfai-eval-suite/BFAI/corpus/knowledge_graph.json` | Entity relationships |

---

## QA Test Sets

| Local Path | GCS Path | Description |
|------------|----------|-------------|
| `qa/QA_BFAI_gold_v1-0__q458.json` | `gs://bfai-eval-suite/BFAI/qa/` | Current gold corpus (458 questions) |
| `qa/archive/` | `gs://bfai-eval-suite/BFAI/qa/archive/` | Old QA versions |

---

## Test Runs

| TID | Date | Type | GCS Path |
|-----|------|------|----------|
| TID_01 | 2024-12-14 | Ad-Hoc | `gs://bfai-eval-suite/BFAI/tests/TID_01/` |
| TID_02 | 2024-12-14 | Ad-Hoc | `gs://bfai-eval-suite/BFAI/tests/TID_02/` |
| TID_03 | 2024-12-14 | Ad-Hoc | `gs://bfai-eval-suite/BFAI/tests/TID_03/` |
| TID_04 | 2025-12-15 | Ad-Hoc | `gs://bfai-eval-suite/BFAI/tests/TID_04/` |
| TID_05 | 2025-12-15 | Ad-Hoc | `gs://bfai-eval-suite/BFAI/tests/TID_05/` |
| TID_06 | 2025-12-15 | Ad-Hoc | `gs://bfai-eval-suite/BFAI/tests/TID_06/` |
| TID_07 | 2025-12-15 | Ad-Hoc | `gs://bfai-eval-suite/BFAI/tests/TID_07/` |
| TID_08 | 2025-12-16 | Core | `gs://bfai-eval-suite/BFAI/tests/TID_08/` |
| TID_09 | 2025-12-16 | Core | `gs://bfai-eval-suite/BFAI/tests/TID_09/` |
| TID_10 | 2025-12-17 | Ad-Hoc | `gs://bfai-eval-suite/BFAI/tests/TID_10/` |

---

## Sync Commands

### Upload local to GCS

```bash
# Upload corpus
gcloud storage cp -r clients/BFAI/corpus/ gs://bfai-eval-suite/BFAI/corpus/

# Upload QA
gcloud storage cp -r clients/BFAI/qa/ gs://bfai-eval-suite/BFAI/qa/

# Upload specific test
gcloud storage cp -r clients/BFAI/tests/TID_XX/ gs://bfai-eval-suite/BFAI/tests/TID_XX/
```

### Download from GCS to local

```bash
# Download corpus (new machine setup)
gcloud storage cp -r gs://bfai-eval-suite/BFAI/corpus/ clients/BFAI/corpus/

# Download specific test
gcloud storage cp -r gs://bfai-eval-suite/BFAI/tests/TID_XX/ clients/BFAI/tests/TID_XX/
```

### Verify sync

```bash
gcloud storage ls gs://bfai-eval-suite/BFAI/
gcloud storage ls gs://bfai-eval-suite/BFAI/corpus/
gcloud storage ls gs://bfai-eval-suite/BFAI/tests/
```
