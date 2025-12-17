# Create New Experiment

How to set up and run a new evaluation experiment.

## Steps

### 1. Create Experiment Directory

```bash
mkdir -p experiments/$(date +%Y-%m-%d)_your_experiment_name/data
```

### 2. Create README.md

Create `experiments/YYYY-MM-DD_experiment_name/README.md` with:

```markdown
# Experiment Name

**Date:** YYYY-MM-DD
**Status:** In Progress

## Objective

What are you trying to test or prove?

## Configuration

- Corpus: X questions
- Retrieval: Top-K chunks
- Models tested: ...

## Results

(Fill in after running)

## How to Reproduce

(Commands to re-run)
```

### 3. Run Your Test

```bash
cd scripts/
python your_test_script.py
```

### 4. Move Results to Experiment Directory

```bash
mv output/*.json experiments/YYYY-MM-DD_experiment_name/data/
```

### 5. Update README with Results

Fill in the Results section with your findings.

### 6. Sync to GCS

```bash
gsutil -m rsync -r experiments/ gs://brightfox-eval-experiments/
```

### 7. Update Documentation

Add your experiment to `docs/experiments/index.md` and create a summary page.
