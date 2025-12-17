"""
Retrieval Metrics Calculation for BFAI Eval Suite.

Computes precision, recall, MRR, and other retrieval quality metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import statistics


@dataclass
class RetrievalMetricsResult:
    """Complete retrieval metrics for an experiment."""
    # Recall
    recall_at_k: Dict[int, float]  # {k: recall_value}
    
    # Precision
    precision_at_k: Dict[int, float]  # {k: precision_value}
    
    # MRR (Mean Reciprocal Rank)
    mrr_at_k: Dict[int, float]  # {k: mrr_value}
    
    # Hit rate (at least one relevant doc in top k)
    hit_rate_at_k: Dict[int, float]
    
    # Per-question details for debugging
    per_question_metrics: List[Dict[str, Any]]
    
    # Summary
    total_questions: int
    questions_with_relevant_docs: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr_at_k": self.mrr_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "total_questions": self.total_questions,
            "questions_with_relevant_docs": self.questions_with_relevant_docs,
        }


def normalize_doc_name(name: str) -> str:
    """Normalize document name for comparison (remove .pdf, lowercase, strip)."""
    if not name:
        return ""
    normalized = name.lower().strip()
    # Remove common extensions
    for ext in [".pdf", ".docx", ".doc", ".txt", ".md"]:
        if normalized.endswith(ext):
            normalized = normalized[:-len(ext)]
    return normalized


def is_relevant_doc(retrieved_doc_name: str, expected_source: str) -> bool:
    """Check if a retrieved document matches the expected source."""
    return normalize_doc_name(retrieved_doc_name) == normalize_doc_name(expected_source)


def compute_retrieval_metrics(
    retrieval_results: List[Dict[str, Any]],
    k_values: List[int] = [5, 10, 15, 20, 25, 50, 100],
) -> RetrievalMetricsResult:
    """
    Compute retrieval metrics from cached retrieval results.
    
    Args:
        retrieval_results: List of dicts with keys:
            - expected_source: The ground truth document name
            - reranked_docs: List of retrieved docs with 'doc_name' field
        k_values: List of k values to compute metrics at
        
    Returns:
        RetrievalMetricsResult with all metrics
    """
    recall_at_k = {k: [] for k in k_values}
    precision_at_k = {k: [] for k in k_values}
    mrr_at_k = {k: [] for k in k_values}
    hit_rate_at_k = {k: [] for k in k_values}
    per_question_metrics = []
    
    questions_with_relevant = 0
    
    for item in retrieval_results:
        expected_source = item.get("expected_source", "")
        reranked_docs = item.get("reranked_docs", [])
        
        if not expected_source:
            # Skip questions without expected source
            continue
        
        # Find positions of relevant documents
        relevant_positions = []
        for i, doc in enumerate(reranked_docs):
            doc_name = doc.get("doc_name", "")
            if is_relevant_doc(doc_name, expected_source):
                relevant_positions.append(i + 1)  # 1-indexed
        
        has_relevant = len(relevant_positions) > 0
        if has_relevant:
            questions_with_relevant += 1
        
        first_relevant_pos = relevant_positions[0] if relevant_positions else None
        
        question_metrics = {
            "question": item.get("question", "")[:100],
            "expected_source": expected_source,
            "relevant_positions": relevant_positions,
            "first_relevant_position": first_relevant_pos,
            "total_retrieved": len(reranked_docs),
        }
        
        for k in k_values:
            # Recall@k: Did we find the relevant doc in top k?
            # For single-source questions, recall is binary (0 or 1)
            relevant_in_top_k = any(pos <= k for pos in relevant_positions)
            recall_at_k[k].append(1.0 if relevant_in_top_k else 0.0)
            
            # Precision@k: What fraction of top k docs are relevant?
            relevant_count_in_top_k = sum(1 for pos in relevant_positions if pos <= k)
            precision_at_k[k].append(relevant_count_in_top_k / k)
            
            # Hit rate@k: Same as recall for single-source (binary)
            hit_rate_at_k[k].append(1.0 if relevant_in_top_k else 0.0)
            
            # MRR@k: Reciprocal of first relevant position (if in top k)
            if first_relevant_pos and first_relevant_pos <= k:
                mrr_at_k[k].append(1.0 / first_relevant_pos)
            else:
                mrr_at_k[k].append(0.0)
        
        per_question_metrics.append(question_metrics)
    
    # Aggregate
    return RetrievalMetricsResult(
        recall_at_k={k: statistics.mean(v) if v else 0.0 for k, v in recall_at_k.items()},
        precision_at_k={k: statistics.mean(v) if v else 0.0 for k, v in precision_at_k.items()},
        mrr_at_k={k: statistics.mean(v) if v else 0.0 for k, v in mrr_at_k.items()},
        hit_rate_at_k={k: statistics.mean(v) if v else 0.0 for k, v in hit_rate_at_k.items()},
        per_question_metrics=per_question_metrics,
        total_questions=len(retrieval_results),
        questions_with_relevant_docs=questions_with_relevant,
    )


@dataclass
class MetricsValidationResult:
    """Result of validating that all required data is present for metrics."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    
    # Field coverage stats
    corpus_fields_present: Dict[str, float]  # field -> % of items with field
    cache_fields_present: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "corpus_fields_present": self.corpus_fields_present,
            "cache_fields_present": self.cache_fields_present,
        }


def validate_metrics_data(
    corpus: List[Dict[str, Any]],
    retrieval_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> MetricsValidationResult:
    """
    Validate that corpus and retrieval cache have all required fields for metrics.
    
    Required corpus fields:
        - question: The question text
        - answer: Ground truth answer
        - source_document OR source_documents: Expected source document name(s)
        
    Required retrieval cache fields (per item):
        - expected_source: Expected source document
        - reranked_docs: List with doc_name field
        
    Returns:
        MetricsValidationResult with validation status and details
    """
    errors = []
    warnings = []
    
    # Check corpus fields - source_document can be singular or plural
    required_corpus_fields = ["question", "answer"]
    optional_corpus_fields = ["node_id", "context", "difficulty", "question_type"]
    
    corpus_fields_present = {}
    for field in required_corpus_fields + optional_corpus_fields + ["source_document", "source_documents"]:
        count = sum(1 for item in corpus if item.get(field))
        corpus_fields_present[field] = count / len(corpus) if corpus else 0.0
    
    # Check source_document coverage (either singular or plural)
    source_coverage = sum(1 for item in corpus if item.get("source_document") or item.get("source_documents"))
    corpus_fields_present["source_document_combined"] = source_coverage / len(corpus) if corpus else 0.0
    
    # Validate required corpus fields
    for field in required_corpus_fields:
        coverage = corpus_fields_present.get(field, 0.0)
        if coverage < 1.0:
            if coverage == 0.0:
                errors.append(f"Corpus missing required field '{field}' (0% coverage)")
            else:
                errors.append(f"Corpus field '{field}' only {coverage*100:.1f}% complete")
    
    # Check source document coverage
    source_coverage = corpus_fields_present.get("source_document_combined", 0.0)
    if source_coverage < 1.0:
        if source_coverage == 0.0:
            errors.append("Corpus missing source_document/source_documents field (0% coverage)")
        else:
            warnings.append(f"Source document field only {source_coverage*100:.1f}% complete")
    
    # Check retrieval cache if provided
    cache_fields_present = {}
    if retrieval_cache:
        required_cache_fields = ["expected_source", "reranked_docs"]
        
        cache_items = list(retrieval_cache.values())
        for field in required_cache_fields:
            count = sum(1 for item in cache_items if item.get(field))
            cache_fields_present[field] = count / len(cache_items) if cache_items else 0.0
        
        # Check reranked_docs have doc_name
        docs_with_name = 0
        total_docs = 0
        for item in cache_items:
            for doc in item.get("reranked_docs", []):
                total_docs += 1
                if doc.get("doc_name"):
                    docs_with_name += 1
        
        if total_docs > 0:
            cache_fields_present["reranked_docs.doc_name"] = docs_with_name / total_docs
        
        # Validate required cache fields
        for field in required_cache_fields:
            coverage = cache_fields_present.get(field, 0.0)
            if coverage < 1.0:
                if coverage == 0.0:
                    errors.append(f"Retrieval cache missing required field '{field}'")
                else:
                    warnings.append(f"Retrieval cache field '{field}' only {coverage*100:.1f}% complete")
        
        # Check doc_name coverage
        doc_name_coverage = cache_fields_present.get("reranked_docs.doc_name", 0.0)
        if doc_name_coverage < 0.95:
            warnings.append(f"Only {doc_name_coverage*100:.1f}% of retrieved docs have 'doc_name'")
    
    return MetricsValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        corpus_fields_present=corpus_fields_present,
        cache_fields_present=cache_fields_present,
    )
