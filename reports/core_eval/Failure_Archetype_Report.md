# Failure Archetype Analysis Report

**Date:** December 17, 2025  
**Corpus:** Gold Standard v1 (458 questions)  
**Config:** Precision@25  
**Analyzer:** Gemini 2.5 Flash  

---

## Executive Summary

Of 458 gold standard questions, **18 failed** (3.9% fail rate, scores 1-2). LLM-based archetype classification identified **6 distinct failure patterns**.

| Archetype | Count | % of Failures | Description |
|-----------|-------|---------------|-------------|
| **INCOMPLETE_CONTEXT** | 6 | 33.3% | Retrieved chunks missing full answer |
| **WRONG_DOCUMENT** | 4 | 22.2% | Low MRR - relevant doc ranked poorly |
| **HALLUCINATION** | 3 | 16.7% | Generated plausible but incorrect info |
| **NO_FAILURE** | 2 | 11.1% | Judge disagreement (actually correct) |
| **COMPLEX_REASONING** | 2 | 11.1% | Multi-step reasoning failed |
| **NUMERICAL_PRECISION** | 1 | 5.6% | Exact numbers wrong or missing |

### Key Insight

**55% of failures are retrieval-related** (INCOMPLETE_CONTEXT + WRONG_DOCUMENT). These are addressable through better chunking and reranking strategies.

---

## Archetype Definitions

### 1. INCOMPLETE_CONTEXT (33.3%)
The retrieval system found relevant documents, but the specific chunks passed to the LLM didn't contain all information needed to answer the question completely.

**Root Causes:**
- Chunking splits related facts across boundaries
- Multi-part questions require info from multiple chunks
- Technical tables/specs fragmented during ingestion

**Example:**
> **Q:** For the Southwire Multi-Conductor CU 600 V cable with Stock Number 662507, what is its allowable ampacity at 90°C, and which NEC table are these ampacities based upon?
> 
> **Issue:** Chunk contained ampacity but not NEC table reference, or vice versa.

---

### 2. WRONG_DOCUMENT (22.2%)
The correct document was retrieved (recall=True) but ranked poorly (low MRR), meaning less relevant content was prioritized.

**Root Causes:**
- Embedding similarity favored wrong document variant
- Multiple similar documents in corpus (e.g., EPEC variants)
- Query terms matched irrelevant content more strongly

**Example:**
> **Q:** What are the primary equipment families for the PVI-CENTRAL-250-US and SCH125KTL-DO/US-600 inverters?
> 
> **Issue:** MRR=0.11 - correct doc ranked 9th despite being recalled.

---

### 3. HALLUCINATION (16.7%)
The LLM generated plausible-sounding but incorrect information, either from parametric knowledge or misinterpretation of context.

**Root Causes:**
- LLM relied on training data instead of retrieved context
- Ambiguous context led to incorrect inference
- Model "filled in" missing details incorrectly

**Example:**
> **Q:** What is the maximum efficiency of the BDM-1200-LV/BDM-1600-LV Micro Inverter?
> 
> **Ground Truth:** 99.9%
> **Issue:** Model generated different efficiency value despite context availability.

---

### 4. NO_FAILURE (11.1%)
The RAG system actually answered correctly, but the judge scored it as a failure. This represents judge calibration issues or edge cases.

**Root Causes:**
- Judge prompt interpretation differences
- Semantic equivalence not recognized
- Overly strict scoring criteria

**Example:**
> **Q:** What hazardous condition was observed at a solar array involving a panel junction box?
> 
> **Issue:** Answer was correct but judge scored low - likely formatting/phrasing difference.

---

### 5. COMPLEX_REASONING (11.1%)
Questions requiring multi-hop reasoning or synthesis across multiple documents failed despite good retrieval.

**Root Causes:**
- LLM couldn't synthesize info from multiple chunks
- Comparison questions require structured reasoning
- Missing explicit reasoning chain in prompt

**Example:**
> **Q:** Evaluate which microinverter (Yotta DPI-480 vs NEP BDM-2000) is better suited for 65°C ambient, considering cooling mechanisms, input current limits, and thermal warnings.
> 
> **Issue:** Model failed to identify and prioritize the critical thermal derating warning.

---

### 6. NUMERICAL_PRECISION (5.6%)
Questions requiring exact numerical values failed due to extraction or matching errors.

**Root Causes:**
- Multiple similar product numbers in context
- Numerical specifications require exact matching
- Table extraction may have errors

**Example:**
> **Q:** What is the Siemens product number for a 125A TMTU circuit breaker with 25KAIC @ 480V?
> 
> **Issue:** Slight mismatch in numerical specification led to wrong product number.

---

## Detailed Failure Examples

### Example 1: INCOMPLETE_CONTEXT

**Question ID:** q_0110  
**Type:** single_hop | **Difficulty:** hard

**Question:**
> For the Southwire Multi-Conductor CU 600 V FR-XLPE LCT Shielded LSZH Jacket Control Cable with Stock Number 662507, which is a 12 AWG cable with 12 conductors, what is its allowable ampacity for continuous operation at 90°C, and which NEC table and edition are these ampacities based upon?

**Ground Truth:**
> The allowable ampacity for continuous operation at 90°C for Stock Number 662507 (12 AWG, 12 conductors) is 15 Amp. These ampacities are based on Table 310.15 (B)(16) of the NEC, 2017 Edition.

**Scores:** correctness=1, completeness=1, faithfulness=5, relevance=5  
**Retrieval:** recall=True, MRR=1.0

**Analysis:**
Despite perfect retrieval (MRR=1.0), the chunks passed to the LLM didn't contain both the ampacity value AND the NEC table reference together. The chunking strategy split this related information.

**Recommendation:**
Improve chunking to keep technical specifications with their source references. Consider semantic chunking that respects document structure.

---

### Example 2: WRONG_DOCUMENT

**Question ID:** q_0345  
**Type:** multi_hop | **Difficulty:** hard

**Question:**
> Considering a Southern States EV-2 Aluminum Vertical Disconnect Switch is deployed in a 362 kV system, what is the required phase spacing and recommended insulator type if the application demands a 2000A continuous current and a BIL of 1050 kV?

**Ground Truth:**
> For a Southern States EV-2 at 362 kV, Document 1 confirms it can support 2000A continuous current with 1050 kV BIL. Document 2's application chart specifies phase spacing and insulator type for this configuration.

**Scores:** correctness=1, completeness=1, faithfulness=1, relevance=5  
**Retrieval:** recall=True, MRR=0.14

**Analysis:**
The relevant documents were recalled but ranked 7th position (MRR=0.14). The reranker failed to prioritize the correct technical specifications over similar but less relevant content.

**Recommendation:**
Implement cross-encoder reranking for technical queries. Consider query decomposition for multi-hop questions.

---

### Example 3: COMPLEX_REASONING

**Question ID:** q_0457  
**Type:** multi_hop | **Difficulty:** hard

**Question:**
> An industrial solar project requires maximum PV power harvesting at 65°C ambient. Evaluate which microinverter (Yotta DPI-480 vs NEP BDM-2000) is better suited, considering cooling mechanisms, input current limits, and thermal warnings.

**Ground Truth:**
> Both have max operating temp of +65°C. The Yotta DPI-480 uses natural convection cooling and has a critical warning: "Inverter may enter low power mode in environments with poor ventilation." This makes the NEP BDM-2000 potentially more suitable for extreme conditions.

**Scores:** correctness=1, completeness=4, faithfulness=5, relevance=5  
**Retrieval:** recall=True, MRR=0.5

**Analysis:**
The model failed to identify and prioritize the critical thermal derating warning from the Yotta datasheet. It provided a comparison but missed the key differentiating factor.

**Recommendation:**
Enhance prompts to explicitly look for warnings and caveats. Consider a two-stage approach: extract facts first, then synthesize comparison.

---

## Recommendations by Priority

### High Priority (Address Now)

1. **Improve Chunking Strategy**
   - Keep related technical specs together
   - Respect document structure (tables, sections)
   - Include metadata (section headers) in chunks

2. **Implement Cross-Encoder Reranking**
   - Current MRR issues show ranking failures
   - Cross-encoder can significantly improve precision

### Medium Priority (Next Sprint)

3. **Enhanced Prompt Engineering**
   - Explicitly instruct LLM to cite sources
   - Add instructions to identify warnings/caveats
   - Implement "I don't know" responses for missing info

4. **Multi-Hop Query Handling**
   - Decompose complex queries into sub-queries
   - Implement iterative retrieval for multi-part questions

### Low Priority (Future)

5. **Judge Calibration**
   - Review NO_FAILURE cases for judge prompt improvements
   - Consider multi-judge consensus for edge cases

6. **Table Extraction**
   - Improve structured data extraction from PDFs
   - Preserve table relationships in chunks

---

## Appendix: All Failed Questions

| ID | Archetype | MRR | Correctness | Question (truncated) |
|----|-----------|-----|-------------|---------------------|
| q_0009 | HALLUCINATION | 1.00 | 1 | Maximum efficiency of BDM-1200-LV... |
| q_0025 | NUMERICAL_PRECISION | 1.00 | 1 | Siemens product number for 125A TMTU... |
| q_0085 | INCOMPLETE_CONTEXT | 1.00 | 1 | Mechanical Interlock application... |
| q_0110 | INCOMPLETE_CONTEXT | 1.00 | 1 | Southwire cable ampacity and NEC table... |
| q_0136 | NO_FAILURE | 1.00 | 1 | Hazardous condition at solar array... |
| q_0141 | NO_FAILURE | 1.00 | 1 | DiGi WR21/31 wireless metrics... |
| q_0163 | INCOMPLETE_CONTEXT | 1.00 | 1 | Meta Power torque specifications... |
| q_0238 | INCOMPLETE_CONTEXT | 1.00 | 2 | PVI-6000-TL-OUTD power output... |
| q_0247 | HALLUCINATION | 1.00 | 1 | Audible Sound-Level Test purpose... |
| q_0263 | COMPLEX_REASONING | 1.00 | 2 | EV-2 current rating and BIL... |
| q_0318 | WRONG_DOCUMENT | 0.11 | 1 | Inverter equipment families... |
| q_0324 | WRONG_DOCUMENT | 0.50 | 1 | UL 489 Calibration Tests... |
| q_0345 | WRONG_DOCUMENT | 0.14 | 1 | EV-2 phase spacing and insulator... |
| q_0353 | WRONG_DOCUMENT | 0.33 | 1 | GFCI safety standard revision... |
| q_0457 | COMPLEX_REASONING | 0.50 | 1 | Microinverter thermal comparison... |
| q_0488 | INCOMPLETE_CONTEXT | 1.00 | 1 | Siemens 3VA and Southwire cable... |
| q_0501 | INCOMPLETE_CONTEXT | 1.00 | 1 | Garbled document content... |
| q_0511 | HALLUCINATION | 1.00 | 1 | High-Voltage Bonding Cable DC rating... |

---

*Report generated by automated failure archetype analysis pipeline.*
