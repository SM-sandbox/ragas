# Question Relevance Evaluation

**Date:** December 14, 2024  
**Status:** Completed

## Objective

Evaluate each Q&A question for domain relevance - is this a question a real SCADA/solar technician would actually ask, or is it trivial/irrelevant noise?

## Relevance Scale

| Score | Label | Description |
|-------|-------|-------------|
| 5 | Critical | Core technical knowledge a field tech MUST know |
| 4 | Relevant | Useful domain knowledge for the job |
| 3 | Marginal | Somewhat useful but not essential |
| 2 | Low Value | Trivial or overly specific to document |
| 1 | Irrelevant | Not useful for domain work |

## Examples of Low-Quality Questions

Questions that should be filtered out:

- "What is the company address?" (company info)
- "What is the document revision number?" (document metadata)
- "Who downloaded this PDF?" (watermark content)
- "What is the copyright notice?" (legal boilerplate)

## Configuration

- **LLM:** Gemini 2.5 Flash
- **Workers:** 15 (parallel)
- **Retries:** 5 with exponential backoff

## How to Run

```bash
cd scripts/
python question_relevance_evaluator.py --workers 15
```

## Results

**Date Run:** December 14, 2024  
**Total Questions:** 224  
**Average Relevance:** 4.82 / 5.0  
**Total Time:** 138.4 seconds (3.64s per question)

### Distribution

| Score | Label | Count | Percentage |
|-------|-------|-------|------------|
| 5 | Critical | 36 | 16.1% |
| 4 | Relevant | 186 | 83.0% |
| 3 | Marginal | 0 | 0.0% |
| 2 | Low Value | 1 | 0.4% |
| 1 | Irrelevant | 1 | 0.4% |

### Key Finding

**The Q&A corpus is high quality.** Only 2 out of 224 questions (0.9%) were flagged as low-quality candidates for removal.

### Low-Quality Questions Identified

1. **"What color is the text in the AlsoEnergy logo?"**
   - Score: 1 (Irrelevant)
   - Flags: company_info, non_technical
   - Reason: Logo color is branding, not technical knowledge

2. **"What is the revision number and date of this CPS SCA Series manual?"**
   - Score: 2 (Low Value)
   - Flags: document_metadata
   - Reason: Document metadata, not equipment knowledge

## Output

The evaluator produces:

- Distribution of scores (how many 5s, 4s, etc.)
- List of low-scoring questions (1-2) for potential removal
- Recommended "clean corpus" excluding irrelevant questions
