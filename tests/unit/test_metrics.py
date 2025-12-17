"""
Unit tests for core/metrics.py - Retrieval Metrics Calculation.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from core.metrics import (
    normalize_doc_name,
    is_relevant_doc,
    compute_retrieval_metrics,
    validate_metrics_data,
    RetrievalMetricsResult,
    MetricsValidationResult,
)


class TestNormalizeDocName:
    """Tests for normalize_doc_name function."""
    
    def test_removes_pdf_extension(self):
        assert normalize_doc_name("Document.pdf") == "document"
    
    def test_removes_docx_extension(self):
        assert normalize_doc_name("Document.docx") == "document"
    
    def test_lowercases(self):
        assert normalize_doc_name("MyDocument") == "mydocument"
    
    def test_strips_whitespace(self):
        assert normalize_doc_name("  Document  ") == "document"
    
    def test_handles_empty_string(self):
        assert normalize_doc_name("") == ""
    
    def test_handles_none_like(self):
        assert normalize_doc_name("") == ""


class TestIsRelevantDoc:
    """Tests for is_relevant_doc function."""
    
    def test_exact_match(self):
        assert is_relevant_doc("Document", "Document") is True
    
    def test_case_insensitive(self):
        assert is_relevant_doc("DOCUMENT", "document") is True
    
    def test_pdf_extension_match(self):
        assert is_relevant_doc("Document.pdf", "Document") is True
    
    def test_both_have_extension(self):
        assert is_relevant_doc("Document.pdf", "Document.pdf") is True
    
    def test_no_match(self):
        assert is_relevant_doc("Document1", "Document2") is False


class TestComputeRetrievalMetrics:
    """Tests for compute_retrieval_metrics function."""
    
    def test_perfect_retrieval(self):
        """When relevant doc is always first, metrics should be perfect."""
        results = [
            {
                "expected_source": "Doc A",
                "reranked_docs": [
                    {"doc_name": "Doc A.pdf"},
                    {"doc_name": "Doc B.pdf"},
                ]
            },
            {
                "expected_source": "Doc B",
                "reranked_docs": [
                    {"doc_name": "Doc B.pdf"},
                    {"doc_name": "Doc A.pdf"},
                ]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[5, 10])
        
        assert metrics.recall_at_k[5] == 1.0
        assert metrics.recall_at_k[10] == 1.0
        assert metrics.mrr_at_k[5] == 1.0  # First position = 1/1 = 1.0
        assert metrics.mrr_at_k[10] == 1.0
    
    def test_no_relevant_docs(self):
        """When no relevant docs found, metrics should be zero."""
        results = [
            {
                "expected_source": "Doc A",
                "reranked_docs": [
                    {"doc_name": "Doc B.pdf"},
                    {"doc_name": "Doc C.pdf"},
                ]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[5])
        
        assert metrics.recall_at_k[5] == 0.0
        assert metrics.mrr_at_k[5] == 0.0
    
    def test_relevant_doc_at_position_3(self):
        """MRR should be 1/3 when relevant doc is at position 3."""
        results = [
            {
                "expected_source": "Doc A",
                "reranked_docs": [
                    {"doc_name": "Doc B.pdf"},
                    {"doc_name": "Doc C.pdf"},
                    {"doc_name": "Doc A.pdf"},
                ]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[5])
        
        assert metrics.recall_at_k[5] == 1.0
        assert abs(metrics.mrr_at_k[5] - 1/3) < 0.001
    
    def test_precision_calculation(self):
        """Precision@k should be relevant_count / k."""
        results = [
            {
                "expected_source": "Doc A",
                "reranked_docs": [
                    {"doc_name": "Doc A.pdf"},  # Relevant
                    {"doc_name": "Doc A.pdf"},  # Relevant (duplicate)
                    {"doc_name": "Doc B.pdf"},
                    {"doc_name": "Doc C.pdf"},
                    {"doc_name": "Doc D.pdf"},
                ]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[5])
        
        # 2 relevant docs in top 5 = 2/5 = 0.4
        assert abs(metrics.precision_at_k[5] - 0.4) < 0.001
    
    def test_recall_at_different_k(self):
        """Recall should change based on k value."""
        results = [
            {
                "expected_source": "Doc A",
                "reranked_docs": [
                    {"doc_name": "Doc B.pdf"},
                    {"doc_name": "Doc C.pdf"},
                    {"doc_name": "Doc D.pdf"},
                    {"doc_name": "Doc E.pdf"},
                    {"doc_name": "Doc F.pdf"},
                    {"doc_name": "Doc A.pdf"},  # Position 6
                ]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[5, 10])
        
        assert metrics.recall_at_k[5] == 0.0  # Not in top 5
        assert metrics.recall_at_k[10] == 1.0  # In top 10
    
    def test_returns_per_question_metrics(self):
        """Should include per-question details."""
        results = [
            {
                "question": "What is X?",
                "expected_source": "Doc A",
                "reranked_docs": [{"doc_name": "Doc A.pdf"}]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[5])
        
        assert len(metrics.per_question_metrics) == 1
        assert metrics.per_question_metrics[0]["first_relevant_position"] == 1
    
    def test_counts_questions_with_relevant_docs(self):
        """Should track how many questions have relevant docs."""
        results = [
            {
                "expected_source": "Doc A",
                "reranked_docs": [{"doc_name": "Doc A.pdf"}]
            },
            {
                "expected_source": "Doc B",
                "reranked_docs": [{"doc_name": "Doc C.pdf"}]  # No match
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[5])
        
        assert metrics.total_questions == 2
        assert metrics.questions_with_relevant_docs == 1


class TestValidateMetricsData:
    """Tests for validate_metrics_data function."""
    
    def test_valid_corpus(self):
        """Valid corpus should pass validation."""
        corpus = [
            {
                "question": "What is X?",
                "answer": "X is Y",
                "source_document": "Doc A",
            }
        ]
        
        result = validate_metrics_data(corpus)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_missing_question_field(self):
        """Missing question field should fail."""
        corpus = [
            {
                "answer": "X is Y",
                "source_document": "Doc A",
            }
        ]
        
        result = validate_metrics_data(corpus)
        
        assert result.valid is False
        assert any("question" in e for e in result.errors)
    
    def test_missing_source_document(self):
        """Missing source_document should fail."""
        corpus = [
            {
                "question": "What is X?",
                "answer": "X is Y",
            }
        ]
        
        result = validate_metrics_data(corpus)
        
        assert result.valid is False
        assert any("source_document" in e for e in result.errors)
    
    def test_partial_coverage_fails(self):
        """Partial field coverage should fail."""
        corpus = [
            {"question": "Q1", "answer": "A1", "source_document": "D1"},
            {"question": "Q2", "answer": "A2"},  # Missing source_document
        ]
        
        result = validate_metrics_data(corpus)
        
        assert result.valid is False
    
    def test_cache_validation(self):
        """Should validate retrieval cache if provided."""
        corpus = [
            {"question": "Q1", "answer": "A1", "source_document": "D1"},
        ]
        cache = {
            "q1": {
                "expected_source": "D1",
                "reranked_docs": [{"doc_name": "D1.pdf"}]
            }
        }
        
        result = validate_metrics_data(corpus, cache)
        
        assert result.valid is True
        assert "expected_source" in result.cache_fields_present
    
    def test_cache_missing_expected_source(self):
        """Cache missing expected_source should fail."""
        corpus = [
            {"question": "Q1", "answer": "A1", "source_document": "D1"},
        ]
        cache = {
            "q1": {
                "reranked_docs": [{"doc_name": "D1.pdf"}]
            }
        }
        
        result = validate_metrics_data(corpus, cache)
        
        assert result.valid is False
        assert any("expected_source" in e for e in result.errors)
    
    def test_reports_field_coverage(self):
        """Should report field coverage percentages."""
        corpus = [
            {"question": "Q1", "answer": "A1", "source_document": "D1", "difficulty": "easy"},
            {"question": "Q2", "answer": "A2", "source_document": "D2"},
        ]
        
        result = validate_metrics_data(corpus)
        
        assert result.corpus_fields_present["question"] == 1.0
        assert result.corpus_fields_present["difficulty"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
