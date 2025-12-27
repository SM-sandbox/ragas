# CI/CD Integration Plan for RAG Evaluation

## Overview

This document outlines the plan to integrate the ragas evaluation suite into the gRAG_v3 CI/CD pipeline. The goal is to automatically run evaluations after each deployment to dev/stage and gate production deployments on passing thresholds.

## Current State

### What Exists

| Component | Status | Location |
| --------- | ------ | -------- |
| Checkpoint runner | ✅ Working | `scripts/run_checkpoint.py` |
| CI/CD eval runner | ✅ Working | `eval_runners/cicd/run_cicd_eval.py` |
| Locked config | ✅ Working | `config/checkpoint_config.yaml` (hash-protected) |
| Evaluator | ✅ Working | `lib/core/evaluator.py` |
| Smart throttler | ✅ Working | `lib/core/smart_throttler/` |
| Local checkpoints | ✅ Working | `clients_eval_data/BFAI/checkpoints/` |
| GCS upload | ❌ Manual | `gsutil cp` commands in README |

### Current Flow

```text
Manual: python scripts/run_checkpoint.py --cloud
         │
         ▼
    Evaluator runs 458 questions against Cloud Run endpoint
         │
         ▼
    Results saved to clients_eval_data/BFAI/checkpoints/C###/
         │
         ▼
    Registry updated (registry.json)
         │
         ▼
    Report generated (checkpoint_report_C###.md)
         │
         ▼
    Manual: gsutil cp to GCS (optional)
```

---

## Target State

### Automated CI/CD Flow

```text
GitHub Actions: Deploy to bfai-dev
         │
         ▼
    Cloud Deploy: Push new revision to Cloud Run
         │
         ▼
    Post-deploy step: Trigger ragas evaluation
         │
         ▼
    Docker container runs checkpoint evaluation
         │
         ▼
    Results uploaded to GCS automatically
         │
         ▼
    Pass/Fail gate based on thresholds
         │
         ▼
    ✅ Pass → Continue to stage/prod
    ❌ Fail → Block deployment, notify team
```

---

## Implementation Plan

### Phase 1: Dockerize the Evaluation Suite

**Goal**: Create a Docker image that can run checkpoint evaluations.

#### 1.1 Create Dockerfile

```dockerfile
# ragas/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy evaluation code
COPY lib/ lib/
COPY eval_runners/ eval_runners/
COPY scripts/ scripts/
COPY config/ config/
COPY clients_qa_gold/ clients_qa_gold/

# Create output directory
RUN mkdir -p /app/output

# Entry point
ENTRYPOINT ["python", "eval_runners/cicd/run_cicd_eval.py"]
```

#### 1.2 Update requirements.txt

Ensure all dependencies are pinned:

```txt
google-cloud-aiplatform>=1.38.0
google-cloud-storage>=2.10.0
google-genai>=1.54.0
pyyaml>=6.0
httpx>=0.25.0
tenacity>=8.2.0
```

#### 1.3 Build and Push to Artifact Registry

```bash
# Build
docker build -t us-east1-docker.pkg.dev/bfai-prod/bfai-images/ragas-eval:latest .

# Push
docker push us-east1-docker.pkg.dev/bfai-prod/bfai-images/ragas-eval:latest
```

---

### Phase 2: Add GCS Upload to Evaluator

**Goal**: Automatically upload checkpoint results to GCS after each run.

#### 2.1 Add GCS Upload Function

```python
# lib/core/gcs_uploader.py

from google.cloud import storage
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

GCS_BUCKET = "bfai-eval-suite"
GCS_PREFIX = "checkpoints"


def upload_checkpoint_to_gcs(run_dir: Path, client: str = "BFAI") -> str:
    """
    Upload checkpoint results to GCS.
    
    Args:
        run_dir: Local directory containing checkpoint results
        client: Client name (e.g., "BFAI")
        
    Returns:
        GCS URI of uploaded checkpoint
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    
    run_id = run_dir.name  # e.g., "C020__2025-12-27__cloud__p25-3-flash-low__q458"
    gcs_prefix = f"{GCS_PREFIX}/{client}/{run_id}"
    
    uploaded_files = []
    for local_file in run_dir.glob("*"):
        if local_file.is_file():
            blob_name = f"{gcs_prefix}/{local_file.name}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_file))
            uploaded_files.append(blob_name)
            logger.info(f"Uploaded: gs://{GCS_BUCKET}/{blob_name}")
    
    gcs_uri = f"gs://{GCS_BUCKET}/{gcs_prefix}"
    logger.info(f"Checkpoint uploaded to: {gcs_uri}")
    
    return gcs_uri
```

#### 2.2 Integrate into Evaluator

Add to `lib/core/evaluator.py` after `_generate_report()`:

```python
def _upload_to_gcs(self) -> None:
    """Upload checkpoint results to GCS."""
    try:
        from lib.core.gcs_uploader import upload_checkpoint_to_gcs
        gcs_uri = upload_checkpoint_to_gcs(self.run_dir, client="BFAI")
        print(f"  Uploaded to GCS: {gcs_uri}")
    except Exception as e:
        print(f"  Warning: GCS upload failed: {e}")
```

Call it at the end of `run()`:

```python
# After _generate_report()
self._upload_to_gcs()
```

---

### Phase 3: CI/CD Integration

**Goal**: Trigger evaluation automatically after deployment.

#### 3.1 Option A: Cloud Build Step

Add to `gRAG_v3/cloudbuild.yaml`:

```yaml
# After deploy step
- name: 'gcr.io/cloud-builders/docker'
  id: 'run-eval'
  entrypoint: 'bash'
  args:
    - '-c'
    - |
      docker run --rm \
        -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa-key.json \
        -v /secrets:/secrets \
        us-east1-docker.pkg.dev/bfai-prod/bfai-images/ragas-eval:latest \
        --cloud
  waitFor: ['deploy-dev']
```

#### 3.2 Option B: GitHub Actions Workflow

Create `.github/workflows/post-deploy-eval.yml`:

```yaml
name: Post-Deploy Evaluation

on:
  workflow_run:
    workflows: ["Deploy to Dev"]
    types: [completed]
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    
    steps:
      - name: Checkout ragas repo
        uses: actions/checkout@v4
        with:
          repository: your-org/ragas
          
      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
          
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
        
      - name: Run Evaluation
        run: |
          docker run --rm \
            -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa-key.json \
            -v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/sa-key.json \
            us-east1-docker.pkg.dev/bfai-prod/bfai-images/ragas-eval:latest \
            --cloud
            
      - name: Check Results
        run: |
          # Exit code from Docker container determines pass/fail
          echo "Evaluation completed"
```

#### 3.3 Option C: Cloud Run Job (Recommended)

Create a Cloud Run Job that can be triggered after deployment:

```bash
# Create the job
gcloud run jobs create ragas-eval \
  --image=us-east1-docker.pkg.dev/bfai-prod/bfai-images/ragas-eval:latest \
  --region=us-east1 \
  --project=bfai-prod \
  --service-account=bfai-orchestrator-sa@bfai-prod.iam.gserviceaccount.com \
  --memory=4Gi \
  --cpu=2 \
  --task-timeout=30m \
  --max-retries=0

# Trigger after deployment
gcloud run jobs execute ragas-eval --region=us-east1 --project=bfai-prod
```

Add to Cloud Deploy pipeline:

```yaml
# deploy/clouddeploy.yaml
stages:
  - targetId: dev
    profiles: [dev]
    postdeploy:
      actions:
        - name: run-eval
          command: gcloud run jobs execute ragas-eval --region=us-east1 --project=bfai-prod --wait
```

---

### Phase 4: Pass/Fail Gating

**Goal**: Block deployments that fail quality thresholds.

#### 4.1 Thresholds (from checkpoint_config.yaml)

```yaml
thresholds:
  min_pass_rate: 0.85      # Minimum 85% pass rate
  max_fail_rate: 0.08      # Maximum 8% fail rate
  max_error_count: 0       # No errors allowed
```

#### 4.2 Exit Codes

The `run_cicd_eval.py` already returns proper exit codes:

- `exit(0)` = All thresholds met, deployment can proceed
- `exit(1)` = Threshold violation, block deployment

#### 4.3 Notification on Failure

Add Slack/email notification:

```python
# In run_cicd_eval.py, after threshold check
if not passed:
    send_failure_notification(
        channel="#bfai-alerts",
        message=f"❌ Eval failed: {failures}",
        run_id=output.get("run_id"),
        gcs_uri=output.get("gcs_uri"),
    )
```

---

## GCS Storage Structure

```text
gs://bfai-eval-suite/
├── checkpoints/
│   └── BFAI/
│       ├── C020__2025-12-27__cloud__p25-3-flash-low__q458/
│       │   ├── results.json
│       │   ├── checkpoint.json
│       │   └── checkpoint_report_C020.md
│       ├── C021__2025-12-28__cloud__p25-3-flash-low__q458/
│       │   └── ...
│       └── registry.json  (master registry)
├── baselines/
│   └── BFAI/
│       └── baseline_v1.json
└── corpus/
    └── BFAI/
        └── QA_BFAI_gold_v1-0__q458.json
```

---

## Timeline Estimate

| Phase | Effort | Dependencies |
| ----- | ------ | ------------ |
| Phase 1: Dockerize | 2-4 hours | None |
| Phase 2: GCS Upload | 2-3 hours | Phase 1 |
| Phase 3: CI/CD Integration | 4-6 hours | Phase 1, 2 |
| Phase 4: Pass/Fail Gating | 2-3 hours | Phase 3 |

**Total: 10-16 hours** (1-2 days of focused work)

---

## Decisions (Confirmed Dec 27, 2025)

| Decision | Choice | Rationale |
| -------- | ------ | --------- |
| **CI/CD Trigger** | Cloud Run Job | Can be triggered from Cloud Deploy AND manually for debugging |
| **Corpus Size** | Full 458 questions | Known baseline, can trim later once working |
| **Environments** | Gate stage, require for prod | Dev is informational only |
| **Notification** | TBD | Slack recommended |

---

## Quick Start (MVP)

For a minimal viable integration:

1. **Create Dockerfile** (Phase 1.1)
2. **Build and push image** (Phase 1.3)
3. **Create Cloud Run Job** (Phase 3.3)
4. **Add to Cloud Deploy postdeploy** (Phase 3.3)

This gets you automated evals after each deploy in ~4-6 hours.

---

## Appendix A: Full Dockerfile

```dockerfile
# ragas/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY lib/ lib/
COPY eval_runners/ eval_runners/
COPY scripts/ scripts/
COPY config/ config/
COPY clients_qa_gold/ clients_qa_gold/

# Create output directory for checkpoints
RUN mkdir -p /app/clients_eval_data/BFAI/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command: run CI/CD checkpoint evaluation
ENTRYPOINT ["python", "eval_runners/cicd/run_cicd_eval.py"]

# Override with --cloud or --local
CMD ["--cloud"]
```

---

## Appendix B: Cloud Run Job Manifest

```yaml
# ragas/deploy/cloudrun-job.yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: ragas-eval
  annotations:
    run.googleapis.com/launch-stage: BETA
spec:
  template:
    spec:
      template:
        spec:
          containers:
            - image: us-east1-docker.pkg.dev/bfai-prod/bfai-images/ragas-eval:latest
              resources:
                limits:
                  cpu: "2"
                  memory: 4Gi
              env:
                - name: EVAL_MODE
                  value: "cloud"
          serviceAccountName: bfai-orchestrator-sa@bfai-prod.iam.gserviceaccount.com
          timeoutSeconds: 1800  # 30 minutes
```

---

*Document Version: 1.0*  
*Created: December 27, 2025*  
*Author: gRAG_v3 Architecture Team*
