# Failure Analysis Package - C020

Generated: 2025-12-22 17:28

## Contents

1. **FAILURE_ANALYSIS_REPORT.md** - Human-readable summary with recommendations
2. **failure_analysis.json** - Complete analysis data with all details
3. **failures_by_archetype.json** - Failures grouped by root cause type
4. **chunking_issues.json** - Specific chunking/metadata issues to investigate

## Quick Stats

- Total Failures Analyzed: 24
- Top Archetype: INCOMPLETE_CONTEXT

## Archetype Distribution

- **INCOMPLETE_CONTEXT**: 12 (50.0%)
- **WRONG_DOCUMENT**: 8 (33.3%)
- **HALLUCINATION**: 2 (8.3%)
- **COMPLEX_REASONING**: 2 (8.3%)

## How to Use

1. Start with `FAILURE_ANALYSIS_REPORT.md` for overview
2. Check `chunking_issues.json` for specific files to investigate
3. Use `failures_by_archetype.json` to focus on specific failure types
4. Reference `failure_analysis.json` for complete details

## Priority Actions

Focus on **INCOMPLETE_CONTEXT** and **WRONG_DOCUMENT** failures first - 
these are typically addressable through chunking and metadata improvements.
