# Failure Analysis Report - C020

**Generated:** December 22, 2025  
**Total Failures Analyzed:** 24

---

## Executive Summary

| Archetype | Count | % | Description |
|-----------|-------|---|-------------|
| **INCOMPLETE_CONTEXT** | 12 | 50.0% | Retrieved chunks missing full answer |
| **WRONG_DOCUMENT** | 8 | 33.3% | Relevant doc ranked poorly (low MRR) |
| **HALLUCINATION** | 2 | 8.3% | Generated plausible but incorrect info |
| **COMPLEX_REASONING** | 2 | 8.3% | Multi-step reasoning failed |

### Key Insight

**83% of failures are retrieval-related** (INCOMPLETE_CONTEXT + WRONG_DOCUMENT). 
These are addressable through better chunking and metadata strategies.

---

## Breakdown by Difficulty

| Difficulty | HALLUCINATION | INCOMPLETE_CONTEXT | WRONG_DOCUMENT | COMPLEX_REASONING |
|------------|---|---|---|---|
| **Easy** | 2 | 5 | 5 | 0 |
| **Medium** | 0 | 5 | 0 | 1 |
| **Hard** | 0 | 2 | 3 | 1 |

---

## Chunking Issues to Investigate

- **sh_hard_020**: The chunk containing the most relevant information was too granular or did not encompass all the necessary details for a complete answer, leading to an incomplete context for the LLM.
- **sh_med_027**: The retrieved chunks, despite being relevant, were too narrow or incomplete to contain all the necessary facts for a correct answer. This could be due to overly granular chunking, or the context window not capturing enough surrounding information.
- **sh_med_044**: The specific chunk(s) retrieved, while deemed most relevant, did not contain all the necessary details to fully answer the question. This could be due to chunks being too small, or critical information being split across multiple chunks that were not all retrieved or properly synthesized. The 'Source Files: []' also indicates a potential logging or data integrity issue.
- **sh_easy_048**: Potentially, the necessary information was present in the source document but was fragmented or incomplete within the specific chunk(s) provided to the LLM, or the source document itself was incomplete for the given question.
- **sh_hard_046**: The chunk containing the relevant information was missing the specific detail required for a correct answer. This suggests the chunking strategy might have separated critical information, or the crucial detail was not present in the indexed document within a single retrievable unit.
- **sh_med_073**: The relevant information might have been fragmented across multiple chunks, or the single retrieved chunk, despite being the most relevant, did not contain all necessary details for a complete and correct answer.
- **mh_easy_025**: The necessary information for the multi-hop question might have been split across multiple chunks, and not all required chunks were retrieved, or individual chunks were too small to provide a complete piece of information for the multi-hop reasoning.
- **mh_med_013**: The necessary information for the multi-hop question might have been fragmented across multiple chunks, or individual chunks were too small to capture the complete logical flow or required facts, leading to an incomplete context being presented to the LLM.
- **mh_med_021**: The context, while relevant, might have been fragmented or not optimally structured (e.g., too small chunks, or key related facts separated) to facilitate the complex multi-hop reasoning required by the LLM.
- **sh_easy_069**: Context assembly/passing failure. The issue is not with the chunking strategy itself, but with the mechanism that extracts and provides chunks from the retrieved documents to the LLM.

---

## Detailed Failures

### COMPLEX_REASONING (2 failures)

**mh_med_021** (medium, multi_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=2, comp=4, faith=5
- **Root Cause:** Despite retrieving relevant and highly ranked context (MRR 1.0, Recall Hit True, Relevance 5/5), the LLM failed to perform the necessary multi-hop reasoning or synthesis required to correctly answer the question. The LLM remained faithful to the context (Faithfulness 5/5), but its inability to connect the dots led to an incorrect (Correctness 2/5) and partially complete (Completeness 4/5) answer.
- **Recommendation:** 1. **LLM Improvement:** Implement advanced prompting techniques like Chain-of-Thought (CoT) or self-reflection to guide the LLM through multi-hop reasoning. Consider fine-tuning the LLM for complex synthesis tasks. 2. **Retrieval/Chunking Improvement:** Evaluate chunking strategy for multi-hop questions. Consider larger, more semantically coherent chunks or graph-based retrieval to ensure all interconnected facts are presented together to the LLM.

**mh_hard_068** (hard, multi_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=2, comp=2, faith=5
- **Root Cause:** The LLM failed to synthesize information from multiple retrieved chunks to answer a multi-hop question, likely because crucial information, though present (Recall Hit: True), was not highly ranked (MRR: 0.333), making the effective context for the LLM incomplete.
- **Recommendation:** Improve retrieval ranking (e.g., re-ranking, better embedding models, hybrid search) to ensure all necessary information for multi-hop questions is at the top. Additionally, explore LLM fine-tuning or advanced prompt engineering strategies to enhance its multi-hop reasoning and synthesis capabilities.

---

### HALLUCINATION (2 failures)

**sh_easy_026** (easy, single_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=5, comp=5, faith=1
- **Root Cause:** The LLM generated an answer that was factually correct and complete but was not supported by or derived from the provided context, indicating a failure to ground its response in the retrieved information despite relevant context being available.
- **Recommendation:** Implement stronger prompt engineering directives to enforce grounding (e.g., 'Answer *only* from the provided context. If the information is not present, state that you cannot answer based on the context.') and consider using a more context-aware LLM or fine-tuning for faithfulness.

**sh_easy_079** (easy, single_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=1, comp=1, faith=1
- **Root Cause:** Despite perfect retrieval of relevant context (MRR 1.0, Recall Hit True, Relevance 5/5), the LLM generated an answer that was incorrect, incomplete, and unfaithful to the source material, indicating it either ignored the context or misinterpreted it severely.
- **Recommendation:** Implement stronger prompt engineering to enforce grounding of the LLM's response to the provided context. Consider fine-tuning the LLM for improved faithfulness and explore adding post-generation fact-checking or confidence scoring mechanisms to detect unsupported claims.

---

### INCOMPLETE_CONTEXT (12 failures)

**sh_hard_020** (hard, single_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=1, comp=1, faith=5
- **Root Cause:** The retrieved context, despite being highly relevant (Relevance 5/5) and correctly ranked (MRR 1.0), did not contain all the necessary information to provide a complete and correct answer. The LLM faithfully generated an answer based on this insufficient context (Faithfulness 5/5), leading to low correctness and completeness scores.
- **Recommendation:** Review the chunking strategy for documents related to this question type. Consider increasing chunk size or implementing overlapping chunks to ensure that complete pieces of information are captured within single retrieval units. If the full answer inherently spans multiple distinct pieces of information, explore multi-hop retrieval strategies.

**sh_med_027** (medium, single_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=2, comp=3, faith=4
- **Root Cause:** The retrieved context, while highly relevant and correctly identified (MRR 1.0), did not contain all the necessary information to formulate a complete and correct answer. The LLM, being faithful to the provided context, could only produce a partial and ultimately incorrect response.
- **Recommendation:** Review and adjust the chunking strategy to ensure that essential information is not split across multiple chunks. Consider increasing chunk size, implementing overlapping chunks, or using context expansion techniques (e.g., retrieving surrounding chunks) to provide a more comprehensive context to the LLM.

**sh_med_044** (medium, single_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=3, comp=4, faith=5
- **Root Cause:** The retrieval system successfully identified the most relevant chunk (MRR 1.0), but this chunk was only partially relevant (Relevance 3/5) and lacked key information required for a fully correct and complete answer. The LLM faithfully used the provided, albeit insufficient, context.
- **Recommendation:** 1. **Address Data Logging:** Investigate and fix the 'Source Files: []' issue to ensure proper traceability of retrieved documents. 2. **Context Analysis:** Manually review the actual question, the RAG answer, and the retrieved context for 'sh_med_044' to pinpoint the exact missing information. 3. **Chunking Strategy Review:** Based on the context analysis, evaluate if the current chunking strategy (size, overlap, metadata inclusion) is effectively capturing complete units of information relevant to typical questions. Consider increasing chunk size or implementing more sophisticated chunking methods (e.g., semantic chunking). 4. **Retrieval Enhancement:** If the relevant information is spread across multiple documents or parts of a document, explore multi-hop retrieval, re-ranking, or query expansion techniques to gather a more comprehensive context.

**sh_easy_048** (easy, single_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=2, comp=5, faith=5
- **Root Cause:** The LLM faithfully generated an answer based on the retrieved context (Faithfulness 5/5), but the context itself was missing critical information, leading to an incorrect answer (Correctness 2/5) despite high relevance and successful retrieval (MRR 1.0).
- **Recommendation:** Review the content of the source documents and the chunking strategy for this topic. Ensure that complete and sufficient information units are captured within chunks to allow for correct answer generation. Consider techniques like larger chunk sizes, semantic chunking, or multi-chunk retrieval to provide more comprehensive context.

**sh_hard_046** (hard, single_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=2, comp=4, faith=5
- **Root Cause:** The retrieved context, despite being highly relevant and perfectly ranked (MRR 1.0), was missing a critical piece of information necessary to fully and correctly answer the question. The LLM faithfully generated an answer based on the available, but incomplete, context, leading to low correctness despite high faithfulness.
- **Recommendation:** Review the source document(s) related to the question's topic to identify the missing information. Adjust the chunking strategy to ensure all critical details for a given concept or answer are contained within a single chunk. If the information is not present in the source documents, update the knowledge base.

*...and 7 more*

---

### WRONG_DOCUMENT (8 failures)

**mh_easy_012** (easy, multi_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=2, comp=5, faith=3
- **Root Cause:** The correct information was retrieved (Recall Hit: True) but ranked very poorly (MRR: 0.125), meaning the most relevant chunks were not presented at the top of the context to the LLM. This led the LLM to work with effectively incomplete or poorly prioritized context, resulting in an incorrect answer with some unfaithful elements.
- **Recommendation:** Improve the ranking algorithm (e.g., implement a re-ranking step, fine-tune the embedding model, or explore hybrid search) to ensure the most relevant chunks are consistently ranked higher and presented at the top of the context window for the LLM.

**mh_easy_038** (easy, multi_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=1, comp=1, faith=3
- **Root Cause:** The retrieval system failed to rank the most relevant document at the top (MRR 0.5), leading the LLM to process a sub-optimal or less relevant context. This resulted in a largely incorrect and incomplete answer, despite the relevant document being present in the retrieved set. The missing source files also hinder debugging.
- **Recommendation:** 1. Improve retrieval ranking: Implement or fine-tune re-ranking models (e.g., cross-encoders) to ensure the most relevant documents are consistently prioritized, aiming for a higher MRR. 2. Enhance source attribution: Fix the 'Source Files: []' issue to ensure all retrieved chunks are properly linked to their original documents for better traceability and debugging. 3. Evaluate context window utilization: Investigate how the LLM utilizes the provided context, especially when the top-ranked documents are not the most relevant.

**mh_easy_034** (easy, multi_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=2, comp=5, faith=5
- **Root Cause:** The most critical information needed for the multi-hop question was retrieved but ranked poorly (MRR 0.5), causing the LLM to prioritize less accurate or incomplete context from higher-ranked chunks, leading to an incorrect answer despite being faithful to the context it processed.
- **Recommendation:** Improve retrieval ranking to ensure the most critical chunks for multi-hop questions are consistently at the top. Implement re-ranking models or fine-tune the retriever to prioritize highly relevant and crucial information more effectively.

**mh_easy_059** (easy, multi_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=2, comp=3, faith=5
- **Root Cause:** The correct and necessary information was retrieved (Recall Hit: True) but was ranked very poorly (MRR: 0.0625). This poor ranking meant the LLM likely did not effectively utilize the critical information, leading to an incorrect and incomplete answer despite being faithful to the context it did prioritize.
- **Recommendation:** Improve the ranking algorithm to ensure that the most relevant chunks are prioritized and presented at the top of the retrieved context. This could involve implementing a re-ranking step, fine-tuning the embedding model, or enhancing query understanding for better retrieval.

**mh_easy_043** (easy, multi_hop)

> N/A...

- **Ground Truth:** N/A...
- **Scores:** corr=3, comp=2, faith=4
- **Root Cause:** The retrieval system successfully identified the relevant information (Recall Hit: True) but failed to rank it highly (MRR: 0.0526). This resulted in the LLM receiving a context where the crucial details were buried or effectively missing, leading to an incomplete answer.
- **Recommendation:** Implement a robust reranking mechanism to ensure highly relevant chunks are prioritized and presented at the top of the context provided to the LLM. Additionally, investigate the initial retriever's performance to improve the quality of the initial candidate set.

*...and 3 more*

---


## Recommendations

### High Priority
1. Review chunking strategy for documents with INCOMPLETE_CONTEXT failures
2. Investigate reranking for WRONG_DOCUMENT cases
3. Check metadata extraction for NUMERICAL_PRECISION issues

### Medium Priority
4. Enhance prompts for COMPLEX_REASONING questions
5. Review judge calibration for NO_FAILURE cases

---

*Report generated by automated failure analysis pipeline.*
