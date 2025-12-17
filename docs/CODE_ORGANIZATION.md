# Code Organization & Architecture Review

**Version:** 1.0  
**Date:** December 17, 2025  
**Purpose:** Architectural review, organization recommendations, and refactoring suggestions

---

## Table of Contents

1. [Current Structure](#1-current-structure)
2. [Assessment](#2-assessment)
3. [Recommended Reorganization](#3-recommended-reorganization)
4. [Naming Conventions](#4-naming-conventions)
5. [Test Organization](#5-test-organization)
6. [Data Management](#6-data-management)
7. [Refactoring Recommendations](#7-refactoring-recommendations)

---

## 1. Current Structure

```
bfai_eval_suite/
├── core/                    # Core utilities (metrics, models, preflight, report)
├── src/                     # Source modules (retriever, generator, judge, reranker)
├── scripts/                 # 30 scripts (mix of experiments, utilities, runners)
├── tests/                   # Unit, integration, smoke tests
├── corpus/                  # QA corpus files (multiple versions, checkpoints)
├── corpus_pdfs/             # Source PDF documents
├── doc_metadata/            # Extracted document metadata
├── experiments/             # Date-stamped experiment outputs
├── reports/                 # Generated reports (archive, executive, foundational)
├── docs/                    # Documentation (runbooks, architecture)
├── config/                  # Configuration files
├── data/                    # (empty - unused)
└── [root scripts]           # 8 standalone scripts at root level
```

### Current Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| **Script sprawl** | Medium | 30 scripts in `/scripts`, 8 at root - unclear which are current |
| **Duplicate functionality** | Medium | Multiple question generators, multiple eval runners |
| **Hardcoded paths** | High | `sys.path.insert(0, "/Users/scottmacon/...")` in every script |
| **Mixed concerns** | Medium | `/scripts` has experiments, utilities, and runners mixed |
| **Corpus versioning** | Medium | Unclear naming: `v2`, `gold_500`, `gold_100` |
| **Unused directories** | Low | `/data` is empty |

---

## 2. Assessment

### What's Working Well ✅

1. **`/core` module** - Clean separation of metrics, models, preflight, report
2. **`/src` module** - Good abstraction of retriever, generator, judge, reranker
3. **`/tests` structure** - Proper unit/integration/smoke separation
4. **`/reports` organization** - Archive, executive, foundational, experiments
5. **`/docs` with runbooks** - Good documentation structure
6. **`.gitignore`** - Properly ignores large data files and caches

### What Needs Improvement ⚠️

1. **Script organization** - No clear distinction between:
   - One-off experiments
   - Reusable utilities
   - Production runners
   
2. **Root-level scripts** - Should be consolidated:
   - `generate_qa_200.py`, `generate_qa_simple.py`, `generate_qa_from_kg.py`
   - `generate_questions_v2.py`
   - `build_doc_metadata.py`, `build_knowledge_graph.py`
   - `question_generator.py`, `question_rater.py`
   - `ragas_evaluator.py`, `orchestrator_client.py`

3. **Hardcoded dependencies** - Every script has:
   ```python
   sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")
   ```

---

## 3. Recommended Reorganization

### Proposed Structure

```
bfai_eval_suite/
├── core/                        # Core utilities (KEEP AS-IS)
│   ├── __init__.py
│   ├── metrics.py
│   ├── models.py
│   ├── preflight.py
│   └── report.py
│
├── src/                         # Source modules (KEEP AS-IS)
│   ├── __init__.py
│   ├── retriever.py
│   ├── generator.py
│   ├── judge.py
│   ├── reranker.py
│   └── eval_config.py
│
├── runners/                     # ⭐ NEW: Production evaluation runners
│   ├── __init__.py
│   ├── gold_eval.py            # Main gold standard eval (from run_gold_eval.py)
│   ├── e2e_eval.py             # E2E orchestrator eval
│   ├── comparison_eval.py      # A/B comparison runner
│   └── preflight.py            # Pre-flight checks
│
├── generators/                  # ⭐ NEW: Question generation tools
│   ├── __init__.py
│   ├── question_generator.py   # Core generator class
│   ├── batch_generator.py      # Batch generation
│   ├── relevance_scorer.py     # Domain relevance scoring
│   └── quality_rater.py        # Question quality rating
│
├── experiments/                 # ⭐ REORGANIZE: One-off experiments
│   ├── archive/                # Old experiments (date-stamped)
│   ├── embedding_comparison.py
│   ├── temperature_sweep.py
│   ├── context_size_sweep.py
│   └── retrieval_metrics.py
│
├── scripts/                     # ⭐ SLIM DOWN: Utilities only
│   ├── download_chunks.py
│   ├── generate_latency_report.py
│   └── generate_sample_report.py
│
├── tests/                       # Tests (KEEP AS-IS)
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   └── fixtures/               # ⭐ NEW: Test data fixtures
│       └── sample_corpus.json
│
├── corpus/                      # Corpus files (RENAME - see Section 4)
├── reports/                     # Reports (KEEP AS-IS)
├── docs/                        # Documentation (KEEP AS-IS)
├── config/                      # Configuration (KEEP AS-IS)
│
└── [root level]
    ├── README.md
    ├── requirements.txt
    ├── .env.example
    ├── .gitignore
    └── pyproject.toml          # ⭐ NEW: Package configuration
```

### Migration Steps

1. **Create `/runners`** - Move production eval scripts
2. **Create `/generators`** - Consolidate question generation
3. **Archive old scripts** - Move to `/experiments/archive`
4. **Add `pyproject.toml`** - Enable proper imports without `sys.path` hacks
5. **Update imports** - Replace hardcoded paths with package imports

---

## 4. Naming Conventions

### Corpus Files

**Current (Confusing):**
```
qa_corpus_v2.json           # What's v2? How many questions?
qa_corpus_gold_500.json     # Is it 500 questions? Or target 500?
qa_corpus_gold_100.json     # Subset of gold_500?
```

**Proposed Convention:**
```
corpus_<type>_<version>_q<count>.json

Examples:
corpus_gold_v1_q458.json    # Gold standard, version 1, 458 questions
corpus_gold_v1_q100.json    # Gold standard sample, 100 questions
corpus_full_v2_q458.json    # Full corpus (before gold filtering)
corpus_test_v1_q30.json     # Test subset
```

**Metadata in file:**
```json
{
  "metadata": {
    "name": "corpus_gold_v1_q458",
    "version": "1.0",
    "created": "2025-12-17",
    "question_count": 458,
    "description": "Gold standard corpus - domain relevance 4-5 only",
    "filters_applied": ["domain_relevance >= 4"],
    "parent_corpus": "corpus_full_v2_q500"
  },
  "questions": [...]
}
```

### Script Naming

| Type | Convention | Example |
|------|------------|---------|
| Runner | `run_<what>.py` | `run_gold_eval.py` |
| Generator | `generate_<what>.py` | `generate_questions.py` |
| Analyzer | `analyze_<what>.py` | `analyze_failures.py` |
| Utility | `<verb>_<noun>.py` | `download_chunks.py` |
| Test | `test_<module>.py` | `test_metrics.py` |

### Report Naming

```
<type>_<experiment>_<date>.md

Examples:
comparison_p25_vs_p12_20251217.md
analysis_failure_modes_20251217.md
benchmark_latency_20251217.md
```

---

## 5. Test Organization

### Current Structure ✅
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with real dependencies
└── smoke/          # Quick sanity checks
```

### Recommended Additions

```
tests/
├── unit/
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_preflight.py
│   ├── test_judge.py          # ⭐ ADD: Judge parsing tests
│   └── test_generator.py      # ⭐ ADD: Generator tests
│
├── integration/
│   ├── test_retrieval.py      # ⭐ ADD: Retrieval integration
│   ├── test_generation.py     # ⭐ ADD: Generation integration
│   └── test_full_pipeline.py  # ⭐ ADD: E2E pipeline test
│
├── smoke/
│   └── test_smoke.py
│
├── fixtures/                   # ⭐ NEW
│   ├── sample_corpus.json     # 10 questions for testing
│   ├── sample_chunks.json     # Mock retrieval results
│   └── sample_judgments.json  # Mock judge responses
│
└── conftest.py                # ⭐ ADD: Shared fixtures
```

### Test Data Management

**DO NOT commit to git:**
- Full corpus files (>1MB)
- Checkpoint files
- Raw experiment outputs

**DO commit to git:**
- Small fixture files (<100KB)
- Sample data for tests
- Expected output snapshots

**Add to `.gitignore`:**
```
# Test outputs
tests/output/
tests/**/*.log

# Large fixtures (if any)
tests/fixtures/large/
```

---

## 6. Data Management

### What Goes Where

| Data Type | Location | Git? | Cloud? |
|-----------|----------|------|--------|
| Source PDFs | `corpus_pdfs/` | ❌ | ✅ GCS |
| Corpus JSON (<1MB) | `corpus/` | ✅ | ✅ GCS |
| Corpus JSON (>1MB) | `corpus/` | ❌ | ✅ GCS |
| Checkpoints | `corpus/` | ❌ | Optional |
| Doc metadata | `doc_metadata/` | ✅ | ✅ GCS |
| Experiment results | `experiments/` | ❌ | ✅ GCS |
| Reports (MD) | `reports/` | ✅ | ✅ GCS |
| Reports (JSON) | `reports/` | ❌ | ✅ GCS |
| Test fixtures | `tests/fixtures/` | ✅ | ❌ |

### Recommended `.gitignore` Additions

```gitignore
# Large corpus files
corpus/*_q500*.json
corpus/*_q458*.json
corpus/*checkpoint*.json

# Experiment raw data
experiments/**/results*.json
experiments/**/checkpoint*.json

# Report raw data
reports/**/*.json
!reports/**/config.json

# Keep small test fixtures
!tests/fixtures/*.json
```

---

## 7. Refactoring Recommendations

### Priority 1: Fix Hardcoded Paths (HIGH)

**Problem:** Every script has:
```python
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")
```

**Solution:** Create `pyproject.toml` for proper package management:

```toml
[project]
name = "bfai_eval_suite"
version = "1.0.0"

[tool.setuptools.packages.find]
where = ["."]

[project.optional-dependencies]
dev = ["pytest", "mypy", "black"]
```

Then install in dev mode:
```bash
pip install -e .
pip install -e /path/to/sm-dev-01
```

Scripts become:
```python
# No more sys.path hacks!
from libs.core.gcp_config import get_jobs_config
from bfai_eval_suite.core.metrics import calculate_metrics
```

### Priority 2: Consolidate Question Generators (MEDIUM)

**Current:** 8 different question generation scripts
- `generate_qa_200.py`
- `generate_qa_simple.py`
- `generate_qa_from_kg.py`
- `generate_questions_v2.py`
- `question_generator.py`
- `scripts/fill_gaps_simple.py`
- `scripts/fill_gaps_pro.py`
- `scripts/generate_batch_questions.py`

**Proposed:** Single unified generator with config:
```python
# generators/question_generator.py
class QuestionGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
    
    def generate_single_hop(self, chunk: str) -> Question: ...
    def generate_multi_hop(self, chunks: List[str]) -> Question: ...
    def generate_batch(self, chunks: List[str], n: int) -> List[Question]: ...
    def fill_gaps(self, corpus: Corpus, target: int) -> List[Question]: ...
```

### Priority 3: Create Shared Config (MEDIUM)

**Problem:** Config scattered across scripts:
```python
# In run_gold_eval.py
JOB_ID = "bfai__eval66a_g1_1536_tt"
CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_gold_500.json"

# In e2e_orchestrator_test.py
JOB_ID = "bfai__eval66a_g1_1536_tt"
CORPUS_FILE = "corpus/qa_corpus_200.json"
```

**Solution:** Centralized config:
```python
# config/eval_config.py
from dataclasses import dataclass

@dataclass
class EvalConfig:
    job_id: str = "bfai__eval66a_g1_1536_tt"
    corpus_path: str = "corpus/corpus_gold_v1_q458.json"
    precision_k: int = 25
    recall_k: int = 100
    checkpoint_interval: int = 10
    
    @classmethod
    def from_json(cls, path: str) -> "EvalConfig":
        ...
```

### Priority 4: Abstract LLM Judge (LOW)

**Current:** Judge logic duplicated in multiple scripts with slight variations.

**Proposed:** Single judge module with configurable prompts:
```python
# src/judge.py (enhance existing)
class LLMJudge:
    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.0):
        ...
    
    def judge(self, question: str, ground_truth: str, answer: str, context: str) -> Judgment:
        ...
    
    def judge_batch(self, items: List[JudgeInput], workers: int = 1) -> List[Judgment]:
        ...
```

---

## Summary

### Immediate Actions (Do Now)

1. ✅ Rename corpus files with new convention
2. ✅ Update `.gitignore` for large files
3. ⬜ Create `pyproject.toml`

### Short-Term (This Week)

1. ⬜ Create `/runners` directory, move production scripts
2. ⬜ Create `/generators` directory, consolidate generators
3. ⬜ Archive old experiments to `/experiments/archive`
4. ⬜ Add test fixtures

### Medium-Term (This Month)

1. ⬜ Remove `sys.path` hacks after package setup
2. ⬜ Consolidate question generators into single module
3. ⬜ Create shared config module
4. ⬜ Add integration tests for full pipeline

---

*This document should be reviewed and updated as the codebase evolves.*
