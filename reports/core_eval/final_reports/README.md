# Gold Standard Evaluation - Final Reports

This folder contains the three final reports from the Gold Standard Evaluation.

## Reports

| Report | Description |
|--------|-------------|
| **Gold_Standard_Comparison_Report.md** | Main evaluation comparing P@12 vs P@25 configurations on 458 questions |
| **Failure_Archetype_Report.md** | Analysis of 18 failures classified into 6 archetypes |
| **Failure_Rerun_Comparison_Report.md** | Before/after comparison of failures with enhanced settings |

## Key Findings

- **96.1% pass rate** with P@25 configuration
- **18 failures** (3.9%) classified into 6 archetypes
- **67% of failures improved** with enhanced settings (Recall@200, P@100, Pro)
- **5.1x latency cost** for enhanced settings (42.6s vs 8.3s)

## Supporting Data

| File | Location | Description |
|------|----------|-------------|
| `failure_analysis_for_engineering.json` | Parent folder | Comprehensive JSON for engineering investigation |
| `question_id_mapping.json` | `corpus/` | Old to new question ID mapping |
| `qa_corpus_gold_500.json` | `corpus/` | Full corpus with new IDs (sh_easy_001, mh_hard_003, etc.) |

## Question ID Nomenclature

Questions now use descriptive IDs:
- `sh_easy_001` - Single-hop, easy, question 1
- `sh_med_015` - Single-hop, medium, question 15
- `mh_hard_042` - Multi-hop, hard, question 42

---

*Generated: December 17, 2025*
