# Experiments

This section documents all completed evaluation experiments with findings and methodology.

## Completed Experiments

| Experiment | Date | Finding |
|------------|------|---------|
| [Embedding Model Comparison](embedding_model_comparison.md) | Dec 14, 2024 | gemini-embedding-001 with RETRIEVAL_QUERY wins |
| [Embedding Dimension Test](embedding_dimension_test.md) | Dec 14, 2024 | 768 dimensions sufficient |
| [Azure vs GCP](azure_vs_gcp.md) | Dec 14, 2024 | GCP wins by +0.23 score |

## Experiment Structure

Each experiment follows this structure:

```
experiments/YYYY-MM-DD_experiment_name/
├── README.md           # What we tested, what we found
├── config.json         # Test parameters (optional)
├── results_summary.json # Aggregated metrics (optional)
└── data/               # Raw results (gitignored, synced to GCS)
```

## Data Storage

Experiment data files are **gitignored** and synced to GCS:

```bash
# Sync to GCS
gsutil -m rsync -r experiments/ gs://brightfox-eval-experiments/

# Restore from GCS
gsutil -m rsync -r gs://brightfox-eval-experiments/ experiments/
```

## Running New Experiments

See [Create New Experiment](../runbooks/create_experiment.md) runbook.
