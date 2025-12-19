"""
Unit tests for lib/utils/metrics.py

Tests retrieval metrics calculation including recall, precision, MRR, and hit rate.
"""

import pytest
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.utils.metrics import (
    normalize_doc_name,
    is_relevant_doc,
    compute_retrieval_metrics,
    RetrievalMetricsResult,
)


class TestNormalizeDocName:
    """Tests for normalize_doc_name function."""
    
    def test_normalize_removes_pdf_extension(self):
        """Normalize removes .pdf extension."""
        result = normalize_doc_name("document.pdf")
        assert result == "document"
    
    def test_normalize_removes_docx_extension(self):
        """Normalize removes .docx extension."""
        result = normalize_doc_name("report.docx")
        assert result == "report"
    
    def test_normalize_lowercases(self):
        """Normalize lowercases the name."""
        result = normalize_doc_name("MyDocument.PDF")
        assert result == "mydocument"
    
    def test_normalize_strips_whitespace(self):
        """Normalize strips whitespace."""
        result = normalize_doc_name("  document.pdf  ")
        assert result == "document"
    
    def test_normalize_handles_empty_string(self):
        """Normalize handles empty string."""
        result = normalize_doc_name("")
        assert result == ""
    
    def test_normalize_handles_none(self):
        """Normalize handles None input."""
        result = normalize_doc_name(None)
        assert result == ""
    
    def test_normalize_multiple_extensions(self):
        """Normalize only removes known extensions."""
        result = normalize_doc_name("file.backup.pdf")
        assert result == "file.backup"


class TestIsRelevantDoc:
    """Tests for is_relevant_doc function."""
    
    def test_exact_match(self):
        """Exact match returns True."""
        assert is_relevant_doc("document.pdf", "document.pdf") is True
    
    def test_case_insensitive_match(self):
        """Case insensitive match returns True."""
        assert is_relevant_doc("Document.PDF", "document.pdf") is True
    
    def test_extension_agnostic_match(self):
        """Match ignoring extension returns True."""
        assert is_relevant_doc("document.pdf", "document") is True
        assert is_relevant_doc("document", "document.pdf") is True
    
    def test_no_match(self):
        """Non-matching documents return False."""
        assert is_relevant_doc("document1.pdf", "document2.pdf") is False
    
    def test_partial_match_returns_false(self):
        """Partial match returns False."""
        assert is_relevant_doc("document", "doc") is False


class TestComputeRetrievalMetrics:
    """Tests for compute_retrieval_metrics function."""
    
    def test_perfect_retrieval(self):
        """Perfect retrieval returns 1.0 for all metrics."""
        results = [
            {
                "expected_source": "doc1.pdf",
                "reranked_docs": [{"doc_name": "doc1.pdf"}, {"doc_name": "doc2.pdf"}]
            },
            {
                "expected_source": "doc2.pdf",
                "reranked_docs": [{"doc_name": "doc2.pdf"}, {"doc_name": "doc1.pdf"}]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[1, 2, 5])
        
        # Perfect recall at k=1 since relevant doc is always first
        assert metrics.recall_at_k[1] == 1.0
        assert metrics.mrr_at_k[1] == 1.0
        assert metrics.hit_rate_at_k[1] == 1.0
    
    def test_no_relevant_docs(self):
        """No relevant docs returns 0.0 for metrics."""
        results = [
            {
                "expected_source": "doc1.pdf",
                "reranked_docs": [{"doc_name": "wrong1.pdf"}, {"doc_name": "wrong2.pdf"}]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[1, 2])
        
        assert metrics.recall_at_k[1] == 0.0
        assert metrics.recall_at_k[2] == 0.0
        assert metrics.mrr_at_k[2] == 0.0
    
    def test_relevant_doc_at_position_2(self):
        """Relevant doc at position 2 affects MRR."""
        results = [
            {
                "expected_source": "doc1.pdf",
                "reranked_docs": [{"doc_name": "wrong.pdf"}, {"doc_name": "doc1.pdf"}]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[1, 2, 5])
        
        # Not found at k=1
        assert metrics.recall_at_k[1] == 0.0
        assert metrics.hit_rate_at_k[1] == 0.0
        
        # Found at k=2
        assert metrics.recall_at_k[2] == 1.0
        assert metrics.hit_rate_at_k[2] == 1.0
        
        # MRR should be 0.5 (1/2)
        assert metrics.mrr_at_k[2] == 0.5
    
    def test_empty_results(self):
        """Empty results returns zeros."""
        metrics = compute_retrieval_metrics([], k_values=[5, 10])
        
        assert metrics.total_questions == 0
        assert metrics.recall_at_k[5] == 0.0
    
    def test_to_dict(self):
        """to_dict returns proper dictionary."""
        results = [
            {
                "expected_source": "doc1.pdf",
                "reranked_docs": [{"doc_name": "doc1.pdf"}]
            },
        ]
        
        metrics = compute_retrieval_metrics(results, k_values=[5])
        result_dict = metrics.to_dict()
        
        assert "recall_at_k" in result_dict
        assert "precision_at_k" in result_dict
        assert "mrr_at_k" in result_dict
        assert "total_questions" in result_dict
        assert result_dict["total_questions"] == 1


class TestRetrievalMetricsResult:
    """Tests for RetrievalMetricsResult dataclass."""
    
    def test_dataclass_creation(self):
        """Can create RetrievalMetricsResult."""
        result = RetrievalMetricsResult(
            recall_at_k={5: 0.9, 10: 0.95},
            precision_at_k={5: 0.2, 10: 0.1},
            mrr_at_k={5: 0.8, 10: 0.85},
            hit_rate_at_k={5: 0.95, 10: 0.98},
            per_question_metrics=[],
            total_questions=100,
            questions_with_relevant_docs=95,
        )
        
        assert result.recall_at_k[5] == 0.9
        assert result.total_questions == 100
    
    def test_to_dict_includes_all_fields(self):
        """to_dict includes all expected fields."""
        result = RetrievalMetricsResult(
            recall_at_k={5: 0.9},
            precision_at_k={5: 0.2},
            mrr_at_k={5: 0.8},
            hit_rate_at_k={5: 0.95},
            per_question_metrics=[],
            total_questions=100,
            questions_with_relevant_docs=95,
        )
        
        d = result.to_dict()
        
        assert "recall_at_k" in d
        assert "precision_at_k" in d
        assert "mrr_at_k" in d
        assert "hit_rate_at_k" in d
        assert "total_questions" in d
        assert "questions_with_relevant_docs" in d
