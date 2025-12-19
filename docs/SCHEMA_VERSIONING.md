# Evaluation Output Schema Versioning

This document describes the versioning strategy for evaluation output schemas and how to handle schema changes.

## Current Version

**Schema Version: 2.0**

Location: `lib/schemas/eval_output_v2.py`

## Versioning Strategy

We use **semantic versioning** for schemas:

```
MAJOR.MINOR
  │     │
  │     └── Additive changes (backward compatible)
  │
  └── Breaking changes (old data may not validate)
```

### Major Version Changes (e.g., 2.x → 3.x)

A **major version bump** is required when:

- Removing a required field
- Renaming a field
- Changing a field's type (e.g., string → integer)
- Changing the structure of nested objects
- Removing enum values from constrained fields

**Impact:** Old evaluation outputs will NOT validate against the new schema.

### Minor Version Changes (e.g., 2.0 → 2.1)

A **minor version bump** is appropriate when:

- Adding new optional fields
- Adding new enum values to existing fields
- Adding new optional nested objects
- Relaxing constraints (e.g., making a required field optional)

**Impact:** Old evaluation outputs WILL still validate against the new schema.

## How to Update the Schema

### Step 1: Modify the Schema

Edit `lib/schemas/eval_output_v2.py`:

```python
# Update version constants
SCHEMA_VERSION = "2.1"  # or "3.0" for breaking changes
SCHEMA_VERSION_MAJOR = 2  # or 3
SCHEMA_VERSION_MINOR = 1  # or 0
```

### Step 2: Update the JSON Schema

Edit `lib/schemas/eval_output_v2.json` to match the Python changes.

### Step 3: Update the Validator

If adding new required fields, update `validate_eval_output()` in `eval_output_v2.py`.

### Step 4: Update Tests

Add tests for new fields in `tests/unit/test_schemas.py`.

### Step 5: Run Tests

```bash
python3 -m pytest tests/unit/test_schemas.py -v
```

All 45+ tests must pass.

### Step 6: Update Documentation

Add an entry to the Version History below.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **2.0** | 2025-12-19 | Comprehensive metadata overhaul: git info, config hashes, full model configs (seed, top_p, top_k), per-question details (question_text, expected_answer, generated_answer) |
| **1.1** | 2025-12-18 | Added retry_stats, errors, skipped sections |
| **1.0** | 2025-12-17 | Original schema (implicit) |

## Schema Validation

### Python Validation

```python
from lib.schemas import validate_eval_output

data = load_json("results.json")
is_valid, errors = validate_eval_output(data)

if not is_valid:
    print(f"Validation errors: {errors}")
```

### JSON Schema Validation

Use any JSON Schema validator with `lib/schemas/eval_output_v2.json`:

```bash
# Using ajv-cli
ajv validate -s lib/schemas/eval_output_v2.json -d results.json
```

## Backward Compatibility

The validator accepts any `2.x` version:

```python
# These all pass version check:
{"schema_version": "2.0", ...}  # Current
{"schema_version": "2.1", ...}  # Future minor
{"schema_version": "2.9", ...}  # Future minor

# These fail version check:
{"schema_version": "1.0", ...}  # Old major
{"schema_version": "3.0", ...}  # Future major
```

## Required Fields by Section

### Top-Level (Required)
- `schema_version` - Must be "2.x"
- `run_id` - Unique run identifier
- `run_type` - "checkpoint" | "run" | "experiment"
- `client` - Client identifier (e.g., "BFAI")
- `timestamp` - ISO 8601 timestamp

### Environment (Required)
- `git_commit` - Short commit hash
- `git_branch` - Branch name
- `config_file` - Config file used
- `config_hash` - SHA256 of config file
- `hostname` - Machine hostname
- `start_time_utc` - Run start time

### Generator Config (Required)
- `model` - Model name
- `temperature` - Sampling temperature

### Judge Config (Required)
- `model` - Judge model name

### Retrieval Config (Required)
- `index_job_id` - Vector index job ID
- `embedding_model` - Embedding model name
- `embedding_dimension` - Vector dimension
- `recall_k` - Retrieval candidates
- `precision_k` - After reranking

### Corpus (Required)
- `file_path` - Path to corpus file
- `file_hash` - SHA256 of corpus file
- `question_count` - Total questions

### Execution (Required)
- `mode` - "local" | "cloud"
- `run_type` - "checkpoint" | "run" | "experiment"
- `workers` - Parallel workers

### Metrics (Required)
- `precision_k`, `total`, `completed`
- `recall_at_100`, `mrr`
- `pass_rate`, `partial_rate`, `fail_rate`

## Testing Schema Changes

When modifying the schema, ensure these test classes pass:

1. **TestSchemaVersion** - Version format and constants
2. **TestValidateEvalOutput** - All validation rules
3. **TestGetEmptyQuestionResult** - Question result template
4. **TestMetadataHelpers** - Metadata collection functions
5. **TestDataclasses** - Dataclass creation
6. **TestJSONSchemaFile** - JSON schema structure
7. **TestSchemaVersionCompatibility** - Version acceptance/rejection
8. **TestIntegration** - End-to-end creation and serialization

Run the full test suite:

```bash
python3 -m pytest tests/unit/test_schemas.py -v
```

Expected: **45 tests passing**
