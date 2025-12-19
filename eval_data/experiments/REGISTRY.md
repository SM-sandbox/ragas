# Experiments Registry

Quick exploratory tests and hypothesis validation.

## Naming Convention

```text
E{NNN}__{YYYY-MM-DD}__{mode}__{description}/
```

- **Mode:** `L` = Local, `C` = Cloud

## Registry

| ID | Date | Mode | Description | Questions | Notes |
|----|------|------|-------------|-----------|-------|
| E001 | 2025-12-17 | Local | gemini3-eval | ~30 | Gemini 3 Flash initial evaluation |
| E002 | 2025-12-17 | Local | gemini3-test | ~30 | Gemini 3 Flash test runs |
| E003 | 2025-12-18 | Local | gemini3-comparison | ~100 | Model comparison tests |
| E004 | 2025-12-18 | Local | orchestrator-eval | ~50 | Orchestrator endpoint testing |

## Reference Documents

The following markdown documents provide analysis and findings:

- `Context_Size_Sweep.md` - Context window size experiments
- `Model_Comparison_Flash_vs_Pro.md` - Flash vs Pro model comparison
- `Temperature_Sweep.md` - Temperature parameter experiments
- `foundational/` - Foundational experiment documentation

## Purpose

Experiments are used for:

- **Quick hypothesis testing:** Does this change help?
- **Parameter exploration:** What's the optimal value?
- **Debugging:** Why is this failing?

## Folder Contents

Each experiment folder may contain:

- JSON result files with timestamps
- Markdown analysis documents
- Configuration snapshots
