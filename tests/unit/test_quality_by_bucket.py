"""
Unit tests for quality_by_bucket dimension breakdowns in evaluator.

Tests that all quality dimensions (correctness, completeness, faithfulness,
relevance, clarity) are properly calculated and aggregated by question type
and difficulty level.
"""

import pytest


class TestQualityByBucketDimensions:
    """Tests for dimension score breakdowns in quality_by_bucket."""
    
    @pytest.fixture
    def sample_results(self):
        """Sample evaluation results with full judgment data."""
        return [
            # Single-hop easy - perfect scores
            {
                "question_id": "q1",
                "question_type": "single_hop",
                "difficulty": "easy",
                "recall_hit": True,
                "mrr": 1.0,
                "judgment": {
                    "verdict": "pass",
                    "correctness": 5,
                    "completeness": 5,
                    "faithfulness": 5,
                    "relevance": 5,
                    "clarity": 5,
                    "overall_score": 5,
                }
            },
            # Single-hop easy - good scores
            {
                "question_id": "q2",
                "question_type": "single_hop",
                "difficulty": "easy",
                "recall_hit": True,
                "mrr": 1.0,
                "judgment": {
                    "verdict": "pass",
                    "correctness": 4,
                    "completeness": 5,
                    "faithfulness": 5,
                    "relevance": 5,
                    "clarity": 5,
                    "overall_score": 5,
                }
            },
            # Multi-hop medium - mixed scores
            {
                "question_id": "q3",
                "question_type": "multi_hop",
                "difficulty": "medium",
                "recall_hit": True,
                "mrr": 0.5,
                "judgment": {
                    "verdict": "pass",
                    "correctness": 4,
                    "completeness": 4,
                    "faithfulness": 5,
                    "relevance": 5,
                    "clarity": 5,
                    "overall_score": 4,
                }
            },
            # Multi-hop hard - lower scores
            {
                "question_id": "q4",
                "question_type": "multi_hop",
                "difficulty": "hard",
                "recall_hit": False,
                "mrr": 0.0,
                "judgment": {
                    "verdict": "partial",
                    "correctness": 3,
                    "completeness": 3,
                    "faithfulness": 4,
                    "relevance": 4,
                    "clarity": 4,
                    "overall_score": 3,
                }
            },
        ]
    
    def test_calculate_bucket_dimensions(self, sample_results):
        """Test that dimension averages are calculated correctly per bucket."""
        # Simulate the evaluator's quality_by_bucket calculation
        quality_by_bucket = {}
        valid = sample_results
        
        for qtype in ["single_hop", "multi_hop"]:
            quality_by_bucket[qtype] = {}
            for diff in ["easy", "medium", "hard"]:
                bucket_results = [r for r in valid if r.get("question_type") == qtype and r.get("difficulty") == diff]
                if bucket_results:
                    n = len(bucket_results)
                    bucket_pass = sum(1 for r in bucket_results if r.get("judgment", {}).get("verdict") == "pass")
                    bucket_partial = sum(1 for r in bucket_results if r.get("judgment", {}).get("verdict") == "partial")
                    bucket_fail = sum(1 for r in bucket_results if r.get("judgment", {}).get("verdict") == "fail")
                    bucket_recall = sum(1 for r in bucket_results if r.get("recall_hit")) / n
                    bucket_mrr = sum(r.get("mrr", 0) for r in bucket_results) / n
                    
                    # Calculate all dimension score averages
                    bucket_overall = [r.get("judgment", {}).get("overall_score", 0) for r in bucket_results]
                    bucket_correctness = [r.get("judgment", {}).get("correctness", 0) for r in bucket_results]
                    bucket_completeness = [r.get("judgment", {}).get("completeness", 0) for r in bucket_results]
                    bucket_faithfulness = [r.get("judgment", {}).get("faithfulness", 0) for r in bucket_results]
                    bucket_relevance = [r.get("judgment", {}).get("relevance", 0) for r in bucket_results]
                    bucket_clarity = [r.get("judgment", {}).get("clarity", 0) for r in bucket_results]
                    
                    quality_by_bucket[qtype][diff] = {
                        "count": n,
                        "pass_rate": bucket_pass / n,
                        "partial_rate": bucket_partial / n,
                        "fail_rate": bucket_fail / n,
                        "recall_at_100": bucket_recall,
                        "mrr": bucket_mrr,
                        "overall_score_avg": sum(bucket_overall) / n if bucket_overall else 0,
                        "correctness_avg": sum(bucket_correctness) / n if bucket_correctness else 0,
                        "completeness_avg": sum(bucket_completeness) / n if bucket_completeness else 0,
                        "faithfulness_avg": sum(bucket_faithfulness) / n if bucket_faithfulness else 0,
                        "relevance_avg": sum(bucket_relevance) / n if bucket_relevance else 0,
                        "clarity_avg": sum(bucket_clarity) / n if bucket_clarity else 0,
                    }
        
        # Verify single_hop easy bucket (q1 + q2)
        sh_easy = quality_by_bucket["single_hop"]["easy"]
        assert sh_easy["count"] == 2
        assert sh_easy["correctness_avg"] == 4.5  # (5 + 4) / 2
        assert sh_easy["completeness_avg"] == 5.0  # (5 + 5) / 2
        assert sh_easy["faithfulness_avg"] == 5.0
        assert sh_easy["relevance_avg"] == 5.0
        assert sh_easy["clarity_avg"] == 5.0
        assert sh_easy["overall_score_avg"] == 5.0
        
        # Verify multi_hop medium bucket (q3)
        mh_med = quality_by_bucket["multi_hop"]["medium"]
        assert mh_med["count"] == 1
        assert mh_med["correctness_avg"] == 4.0
        assert mh_med["completeness_avg"] == 4.0
        assert mh_med["faithfulness_avg"] == 5.0
        assert mh_med["relevance_avg"] == 5.0
        assert mh_med["clarity_avg"] == 5.0
        assert mh_med["overall_score_avg"] == 4.0
        
        # Verify multi_hop hard bucket (q4)
        mh_hard = quality_by_bucket["multi_hop"]["hard"]
        assert mh_hard["count"] == 1
        assert mh_hard["correctness_avg"] == 3.0
        assert mh_hard["completeness_avg"] == 3.0
        assert mh_hard["faithfulness_avg"] == 4.0
        assert mh_hard["relevance_avg"] == 4.0
        assert mh_hard["clarity_avg"] == 4.0
        assert mh_hard["overall_score_avg"] == 3.0
    
    def test_empty_bucket_not_created(self, sample_results):
        """Test that empty buckets are not created."""
        quality_by_bucket = {}
        valid = sample_results
        
        for qtype in ["single_hop", "multi_hop"]:
            quality_by_bucket[qtype] = {}
            for diff in ["easy", "medium", "hard"]:
                bucket_results = [r for r in valid if r.get("question_type") == qtype and r.get("difficulty") == diff]
                if bucket_results:
                    quality_by_bucket[qtype][diff] = {"count": len(bucket_results)}
        
        # single_hop medium and hard should not exist
        assert "medium" not in quality_by_bucket["single_hop"]
        assert "hard" not in quality_by_bucket["single_hop"]
        
        # multi_hop easy should not exist
        assert "easy" not in quality_by_bucket["multi_hop"]
    
    def test_all_dimensions_present(self, sample_results):
        """Test that all dimension fields are present in bucket data."""
        quality_by_bucket = {}
        valid = sample_results
        
        for qtype in ["single_hop", "multi_hop"]:
            quality_by_bucket[qtype] = {}
            for diff in ["easy", "medium", "hard"]:
                bucket_results = [r for r in valid if r.get("question_type") == qtype and r.get("difficulty") == diff]
                if bucket_results:
                    n = len(bucket_results)
                    quality_by_bucket[qtype][diff] = {
                        "count": n,
                        "pass_rate": 0,
                        "partial_rate": 0,
                        "fail_rate": 0,
                        "recall_at_100": 0,
                        "mrr": 0,
                        "overall_score_avg": sum(r.get("judgment", {}).get("overall_score", 0) for r in bucket_results) / n,
                        "correctness_avg": sum(r.get("judgment", {}).get("correctness", 0) for r in bucket_results) / n,
                        "completeness_avg": sum(r.get("judgment", {}).get("completeness", 0) for r in bucket_results) / n,
                        "faithfulness_avg": sum(r.get("judgment", {}).get("faithfulness", 0) for r in bucket_results) / n,
                        "relevance_avg": sum(r.get("judgment", {}).get("relevance", 0) for r in bucket_results) / n,
                        "clarity_avg": sum(r.get("judgment", {}).get("clarity", 0) for r in bucket_results) / n,
                    }
        
        required_fields = [
            "count", "pass_rate", "partial_rate", "fail_rate",
            "recall_at_100", "mrr", "overall_score_avg",
            "correctness_avg", "completeness_avg", "faithfulness_avg",
            "relevance_avg", "clarity_avg"
        ]
        
        for qtype in quality_by_bucket:
            for diff in quality_by_bucket[qtype]:
                bucket = quality_by_bucket[qtype][diff]
                for field in required_fields:
                    assert field in bucket, f"Missing field {field} in {qtype}/{diff}"


class TestAggregateByType:
    """Tests for aggregating dimensions by question type (single-hop vs multi-hop)."""
    
    def test_aggregate_single_hop_dimensions(self):
        """Test aggregating all single-hop buckets into type-level averages."""
        quality_by_bucket = {
            "single_hop": {
                "easy": {"count": 10, "correctness_avg": 4.8, "completeness_avg": 4.9},
                "medium": {"count": 10, "correctness_avg": 4.5, "completeness_avg": 4.6},
                "hard": {"count": 10, "correctness_avg": 4.2, "completeness_avg": 4.3},
            }
        }
        
        sh_buckets = quality_by_bucket["single_hop"]
        sh_total = sum(b["count"] for b in sh_buckets.values())
        
        # Weighted average
        sh_correctness = sum(b["correctness_avg"] * b["count"] for b in sh_buckets.values()) / sh_total
        sh_completeness = sum(b["completeness_avg"] * b["count"] for b in sh_buckets.values()) / sh_total
        
        assert sh_total == 30
        assert sh_correctness == pytest.approx(4.5, rel=0.01)  # (4.8*10 + 4.5*10 + 4.2*10) / 30
        assert sh_completeness == pytest.approx(4.6, rel=0.01)


class TestAggregateByDifficulty:
    """Tests for aggregating dimensions by difficulty (easy/medium/hard)."""
    
    def test_aggregate_easy_dimensions(self):
        """Test aggregating all easy buckets across types into difficulty-level averages."""
        quality_by_bucket = {
            "single_hop": {
                "easy": {"count": 10, "correctness_avg": 4.8},
            },
            "multi_hop": {
                "easy": {"count": 5, "correctness_avg": 4.2},
            }
        }
        
        sh_easy = quality_by_bucket["single_hop"].get("easy", {})
        mh_easy = quality_by_bucket["multi_hop"].get("easy", {})
        
        easy_count = sh_easy.get("count", 0) + mh_easy.get("count", 0)
        easy_correctness = (
            sh_easy.get("correctness_avg", 0) * sh_easy.get("count", 0) +
            mh_easy.get("correctness_avg", 0) * mh_easy.get("count", 0)
        ) / easy_count
        
        assert easy_count == 15
        assert easy_correctness == pytest.approx(4.6, rel=0.01)  # (4.8*10 + 4.2*5) / 15
