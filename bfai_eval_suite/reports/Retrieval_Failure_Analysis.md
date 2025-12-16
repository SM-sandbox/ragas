# Retrieval Failure Analysis Report

**Generated:** 2025-12-15
**Test:** Pure Semantic Retrieval (160 questions, top 100 results)

---

## 1. Summary

Out of 160 questions tested, a small number failed to retrieve the correct document in the top 100 results:

| Embedding Config | Recall@100 | Failures | Failure Rate |
|------------------|------------|----------|--------------|
| text-embedding-005 | 99.4% | 1 | 0.6% |
| gemini-768-RETRIEVAL_QUERY | 97.5% | 4 | 2.5% |
| gemini-1536-RETRIEVAL_QUERY | 97.5% | 4 | 2.5% |
| gemini-3072-RETRIEVAL_QUERY | 97.5% | 4 | 2.5% |

**Total unique failing questions: 5**

---

## 2. Failing Questions

### Failure #1: Document-Reference + Metadata
**Question:** "What is the nominal AC voltage of the Balance of System (BoS) described in this document?"
**Expected Doc:** `800VAC BoS_Comprehensive_4.2MVA Skid Sales Flyer V2.24`
**Failed On:** text-embedding-005 only

**Analysis:**
- Question contains "this document" which requires conversational context
- The answer (800VAC) is actually in the document filename, not necessarily prominent in the text
- Gemini models with RETRIEVAL_QUERY task type successfully retrieved this

---

### Failure #2: Metadata Question
**Question:** "What is the revision date listed on the document?"
**Expected Doc:** `Intersect Power Easley Project POD (MAIN) Rev 2024-7-30 508`
**Failed On:** All Gemini configs

**Analysis:**
- Extremely generic question - "revision date" could apply to ANY document
- No semantic content to differentiate from other documents
- The date (2024-7-30) is in the filename but question has no unique identifiers
- **Archetype: METADATA_QUERY** - questions about document properties rather than content

---

### Failure #3: Visual Content
**Question:** "What is the name of the company associated with the logo?"
**Expected Doc:** `DiGiWR21WR31ModemBasicsv1`
**Failed On:** All Gemini configs

**Analysis:**
- Question asks about a **logo** - visual content that text embeddings cannot capture
- No company name mentioned in the question to match against
- The document name gives no hint about the company (Digi International)
- **Archetype: VISUAL_CONTENT** - questions requiring image understanding

---

### Failure #4: Visual Content + Document Reference
**Question:** "What is the name of the company associated with the logo featured in the document?"
**Expected Doc:** `EPEC_1200-6000A UL 891 SWBD for Solar_BESS Flyer V12.23`
**Failed On:** All Gemini configs

**Analysis:**
- Same issue as #3 - asking about visual logo content
- "the document" requires context that retrieval cannot provide
- **Archetype: VISUAL_CONTENT**

---

### Failure #5: Ambiguous Technical Question
**Question:** "What are the available communication protocols for advanced diagnostics and metering data from the circuit breakers?"
**Expected Doc:** `EPEC_1200-6000A UL 891 SWBD (Crit Power) Flyer`
**Failed On:** All Gemini configs

**Analysis:**
- Technical question that SHOULD be retrievable
- Multiple EPEC documents exist with similar content (Solar_BESS, Industrial, Crit Power versions)
- The question may be matching chunks from the wrong EPEC variant
- **Archetype: AMBIGUOUS_SOURCE** - multiple similar documents could answer the question

---

## 3. Failure Archetypes

| Archetype | Count | Description |
|-----------|-------|-------------|
| **VISUAL_CONTENT** | 2 | Questions about logos, images, diagrams that text embeddings cannot capture |
| **METADATA_QUERY** | 1 | Questions about document properties (dates, revision numbers) rather than content |
| **DOCUMENT_REFERENCE** | 2 | Questions using "this document" requiring conversational context |
| **AMBIGUOUS_SOURCE** | 1 | Multiple similar documents could answer; wrong variant retrieved |

Note: Some questions have multiple archetypes.

---

## 4. Root Causes

### 4.1 Visual Content Cannot Be Retrieved
Text embeddings only capture textual content. Questions about:
- Logos
- Diagrams
- Images
- Visual layouts

...will fail unless the visual content is described in text (e.g., alt text, captions).

### 4.2 Generic/Metadata Questions Lack Semantic Signal
Questions like "What is the revision date?" have no unique semantic content to match against a specific document. These questions assume the user already has a document in context.

### 4.3 Document Reference Without Context
Questions containing "this document" or "the document" assume conversational context that pure retrieval cannot provide. These are valid questions in a chat context but not for standalone retrieval.

### 4.4 Similar Documents Create Ambiguity
When multiple documents cover similar topics (e.g., EPEC SWBD variants), the retrieval may return chunks from the wrong variant.

---

## 5. Recommendations

### For Question Corpus Quality
1. **Remove or flag VISUAL_CONTENT questions** - These cannot be answered by text-based RAG
2. **Remove DOCUMENT_REFERENCE questions** - Add document name to question if testing retrieval
3. **Add specificity to METADATA questions** - Include unique identifiers

### For Production RAG
1. **Implement image captioning** during ingestion to capture visual content
2. **Include document metadata** in chunk text (title, date, version)
3. **Use conversation context** to resolve "this document" references

### For Evaluation
1. **Separate test sets** by question archetype
2. **Exclude unanswerable questions** from retrieval metrics
3. **Track failure archetypes** to identify systematic issues

---

## 6. Interesting Finding

**text-embedding-005 retrieved 1 document that Gemini models missed:**
- The "800VAC BoS" question succeeded with text-embedding-005 but failed with all Gemini configs
- This suggests the RETRIEVAL_QUERY task type may over-optimize for certain patterns

**Gemini models are more consistent with each other:**
- All 3 Gemini configs (768, 1536, 3072) failed on the exact same 4 questions
- This suggests the task type has more impact than dimensionality
