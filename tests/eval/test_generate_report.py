"""
Tests for generate_report.py - Report generation functionality.

Tests cover:
- Score distribution calculation
- Latency aggregation by difficulty and type
- Cost calculation
- Report structure and content
- Baseline comparison logic
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "eval"))

from generate_report import (
    get_score_dist,
    calc_avg,
    pct,
    generate_report,
    PRICING,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_results():
    """Sample evaluation results for testing."""
    return [
        {
            "question_id": "q1",
            "question_type": "single_hop",
            "difficulty": "easy",
            "time": 5.0,
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
        {
            "question_id": "q2",
            "question_type": "multi_hop",
            "difficulty": "medium",
            "time": 10.0,
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
        {
            "question_id": "q3",
            "question_type": "single_hop",
            "difficulty": "hard",
            "time": 15.0,
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
        {
            "question_id": "q4",
            "question_type": "multi_hop",
            "difficulty": "hard",
            "time": 20.0,
            "judgment": {
                "verdict": "fail",
                "correctness": 2,
                "completeness": 2,
                "faithfulness": 3,
                "relevance": 3,
                "clarity": 3,
                "overall_score": 2,
            }
        },
    ]


@pytest.fixture
def sample_run_data(sample_results):
    """Complete run data structure for testing."""
    return {
        "run_id": "test_run_001",
        "timestamp": "2025-12-18T12:00:00",
        "schema_version": "1.1",
        "metrics": {
            "pass_rate": 0.5,
            "partial_rate": 0.25,
            "fail_rate": 0.25,
            "acceptable_rate": 0.75,
            "recall_at_100": 0.99,
            "mrr": 0.72,
            "overall_score_avg": 3.5,
        },
        "latency": {
            "total_avg_s": 12.5,
            "total_min_s": 5.0,
            "total_max_s": 20.0,
            "by_phase": {
                "retrieval_avg_s": 0.25,
                "rerank_avg_s": 0.18,
                "generation_avg_s": 10.0,
                "judge_avg_s": 2.07,
            }
        },
        "tokens": {
            "prompt_total": 100000,
            "completion_total": 10000,
            "thinking_total": 5000,
            "cached_total": 0,
            "total": 115000,
        },
        "config": {
            "generator_model": "gemini-2.5-flash",
            "judge_model": "gemini-3-flash-preview",
            "precision_k": 25,
            "recall_k": 100,
        },
        "execution": {
            "run_duration_seconds": 120,
            "questions_per_second": 0.033,
            "workers": 5,
        },
        "retry_stats": {
            "total_questions": 4,
            "succeeded_first_try": 4,
            "succeeded_after_retry": 0,
            "failed_all_retries": 0,
            "avg_attempts": 1.0,
        },
        "errors": {
            "total_errors": 0,
            "by_phase": {
                "retrieval": 0,
                "rerank": 0,
                "generation": 0,
                "judge": 0,
            }
        },
        "breakdown_by_type": {
            "single_hop": {"total": 2, "pass": 1, "partial": 1, "fail": 0, "pass_rate": 0.5},
            "multi_hop": {"total": 2, "pass": 1, "partial": 0, "fail": 1, "pass_rate": 0.5},
        },
        "breakdown_by_difficulty": {
            "easy": {"total": 1, "pass": 1, "partial": 0, "fail": 0, "pass_rate": 1.0},
            "medium": {"total": 1, "pass": 1, "partial": 0, "fail": 0, "pass_rate": 1.0},
            "hard": {"total": 2, "pass": 0, "partial": 1, "fail": 1, "pass_rate": 0.0},
        },
        "results": sample_results,
    }


# =============================================================================
# Test Score Distribution
# =============================================================================

class TestScoreDistribution:
    """Tests for score distribution calculation."""
    
    def test_get_score_dist_correctness(self, sample_results):
        """Test score distribution for correctness."""
        dist = get_score_dist(sample_results, "correctness")
        assert dist[5] == 1  # q1
        assert dist[4] == 1  # q2
        assert dist[3] == 1  # q3
        assert dist[2] == 1  # q4
        assert dist[1] == 0
    
    def test_get_score_dist_overall(self, sample_results):
        """Test score distribution for overall score."""
        dist = get_score_dist(sample_results, "overall_score")
        assert dist[5] == 1
        assert dist[4] == 1
        assert dist[3] == 1
        assert dist[2] == 1
        assert dist[1] == 0
    
    def test_get_score_dist_empty_results(self):
        """Test score distribution with empty results."""
        dist = get_score_dist([], "correctness")
        assert dist == {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
    
    def test_get_score_dist_missing_judgment(self):
        """Test score distribution with missing judgment."""
        results = [{"question_id": "q1"}]  # No judgment
        dist = get_score_dist(results, "correctness")
        assert dist == {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}


class TestCalcAvg:
    """Tests for average calculation from distribution."""
    
    def test_calc_avg_uniform(self):
        """Test average with uniform distribution."""
        dist = {5: 1, 4: 1, 3: 1, 2: 1, 1: 1}
        avg = calc_avg(dist)
        assert avg == 3.0  # (5+4+3+2+1) / 5
    
    def test_calc_avg_all_fives(self):
        """Test average with all 5s."""
        dist = {5: 10, 4: 0, 3: 0, 2: 0, 1: 0}
        avg = calc_avg(dist)
        assert avg == 5.0
    
    def test_calc_avg_empty(self):
        """Test average with empty distribution."""
        dist = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        avg = calc_avg(dist)
        assert avg == 0
    
    def test_calc_avg_weighted(self):
        """Test weighted average."""
        dist = {5: 2, 4: 0, 3: 0, 2: 0, 1: 0}
        avg = calc_avg(dist)
        assert avg == 5.0


class TestPct:
    """Tests for percentage formatting."""
    
    def test_pct_half(self):
        """Test 50% formatting."""
        assert pct(50, 100) == "50.0%"
    
    def test_pct_full(self):
        """Test 100% formatting."""
        assert pct(100, 100) == "100.0%"
    
    def test_pct_zero(self):
        """Test 0% formatting."""
        assert pct(0, 100) == "0.0%"
    
    def test_pct_decimal(self):
        """Test decimal percentage."""
        assert pct(1, 3) == "33.3%"


# =============================================================================
# Test Cost Calculation
# =============================================================================

class TestCostCalculation:
    """Tests for cost calculation in report."""
    
    def test_pricing_constants(self):
        """Test that pricing constants are defined."""
        assert "input" in PRICING
        assert "output" in PRICING
        assert "thinking" in PRICING
        assert "cached" in PRICING
    
    def test_pricing_values(self):
        """Test pricing values are reasonable."""
        assert PRICING["input"] == 0.075
        assert PRICING["output"] == 0.30
        assert PRICING["thinking"] == 0.30
        assert PRICING["cached"] == 0.01875
    
    def test_cost_calculation_formula(self):
        """Test cost calculation formula."""
        tokens = {
            "prompt_total": 1_000_000,
            "completion_total": 100_000,
            "thinking_total": 50_000,
            "cached_total": 0,
        }
        
        input_cost = (tokens["prompt_total"] / 1_000_000) * PRICING["input"]
        output_cost = (tokens["completion_total"] / 1_000_000) * PRICING["output"]
        thinking_cost = (tokens["thinking_total"] / 1_000_000) * PRICING["thinking"]
        
        assert input_cost == 0.075
        assert output_cost == 0.03
        assert thinking_cost == 0.015
        
        total = input_cost + output_cost + thinking_cost
        assert total == 0.12


# =============================================================================
# Test Report Generation
# =============================================================================

class TestReportGeneration:
    """Tests for full report generation."""
    
    def test_generate_report_creates_file(self, sample_run_data, tmp_path):
        """Test that generate_report creates output file."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        assert output_path.exists()
    
    def test_generate_report_contains_metrics(self, sample_run_data, tmp_path):
        """Test that report contains key metrics."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        # Check for key sections
        assert "Executive Summary" in content
        assert "Score Distributions" in content
        assert "Latency Analysis" in content
        assert "Token & Cost Analysis" in content
        assert "Breakdown by Question Type" in content
        assert "Breakdown by Difficulty" in content
    
    def test_generate_report_contains_latency_by_type(self, sample_run_data, tmp_path):
        """Test that report contains latency by question type."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        assert "Single-hop" in content
        assert "Multi-hop" in content
        assert "Total Latency by Question Type" in content
    
    def test_generate_report_contains_latency_by_difficulty(self, sample_run_data, tmp_path):
        """Test that report contains latency by difficulty."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        assert "Easy" in content
        assert "Medium" in content
        assert "Hard" in content
        assert "Total Latency by Difficulty" in content
    
    def test_generate_report_contains_failures(self, sample_run_data, tmp_path):
        """Test that report contains failure analysis."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        assert "Failure Analysis" in content
        assert "Failed Questions" in content
        assert "q4" in content  # The failed question


class TestBaselineComparison:
    """Tests for baseline comparison in report."""
    
    def test_report_with_baseline(self, sample_run_data, tmp_path):
        """Test report generation with baseline comparison."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        baseline = {
            "metrics": {
                "pass_rate": 0.4,
                "partial_rate": 0.3,
                "fail_rate": 0.3,
                "acceptable_rate": 0.7,
                "recall_at_100": 0.98,
                "mrr": 0.70,
                "overall_score_avg": 3.2,
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=baseline):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        # Should show delta
        assert "Previous" in content
        assert "Current" in content
        assert "Δ" in content
    
    def test_report_without_baseline(self, sample_run_data, tmp_path):
        """Test report generation without baseline."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        # Should still generate report
        assert "Executive Summary" in content


# =============================================================================
# Test Report Structure
# =============================================================================

class TestReportStructure:
    """Tests for report markdown structure."""
    
    def test_report_has_title(self, sample_run_data, tmp_path):
        """Test that report has proper title."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        assert content.startswith("# Gold Standard RAG Evaluation Report")
    
    def test_report_has_metadata(self, sample_run_data, tmp_path):
        """Test that report has metadata section."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        assert "**Date:**" in content
        assert "**Run ID:**" in content
        assert "**Corpus:**" in content
        assert "**Models:**" in content
    
    def test_report_has_score_definitions(self, sample_run_data, tmp_path):
        """Test that report has score scale definitions."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        assert "CORRECTNESS" in content
        assert "COMPLETENESS" in content
        assert "FAITHFULNESS" in content
        assert "RELEVANCE" in content
        assert "CLARITY" in content
        assert "OVERALL SCORE" in content
    
    def test_report_has_execution_details(self, sample_run_data, tmp_path):
        """Test that report has execution details."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        
        assert "Execution Details" in content
        assert "Duration" in content
        assert "Workers" in content


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in report generation."""
    
    def test_empty_results(self, tmp_path):
        """Test report with empty results."""
        run_data = {
            "run_id": "empty_run",
            "timestamp": "2025-12-18T12:00:00",
            "metrics": {
                "pass_rate": 0,
                "partial_rate": 0,
                "fail_rate": 0,
                "acceptable_rate": 0,
                "recall_at_100": 0,
                "mrr": 0,
                "overall_score_avg": 0,
            },
            "latency": {
                "total_avg_s": 0,
                "total_min_s": 0,
                "total_max_s": 0,
                "by_phase": {
                    "retrieval_avg_s": 0,
                    "rerank_avg_s": 0,
                    "generation_avg_s": 0,
                    "judge_avg_s": 0,
                }
            },
            "tokens": {
                "prompt_total": 0,
                "completion_total": 0,
                "thinking_total": 0,
                "cached_total": 0,
                "total": 0,
            },
            "config": {},
            "execution": {},
            "retry_stats": {
                "total_questions": 0,
                "succeeded_first_try": 0,
                "succeeded_after_retry": 0,
                "failed_all_retries": 0,
                "avg_attempts": 0,
            },
            "errors": {
                "total_errors": 0,
                "by_phase": {"retrieval": 0, "rerank": 0, "generation": 0, "judge": 0}
            },
            "breakdown_by_type": {},
            "breakdown_by_difficulty": {},
            "results": [],
        }
        
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        assert output_path.exists()
    
    def test_all_failures(self, tmp_path):
        """Test report with all failures."""
        run_data = {
            "run_id": "fail_run",
            "timestamp": "2025-12-18T12:00:00",
            "metrics": {
                "pass_rate": 0,
                "partial_rate": 0,
                "fail_rate": 1.0,
                "acceptable_rate": 0,
                "recall_at_100": 0.5,
                "mrr": 0.3,
                "overall_score_avg": 1.5,
            },
            "latency": {
                "total_avg_s": 10,
                "total_min_s": 5,
                "total_max_s": 15,
                "by_phase": {
                    "retrieval_avg_s": 0.2,
                    "rerank_avg_s": 0.1,
                    "generation_avg_s": 8,
                    "judge_avg_s": 1.7,
                }
            },
            "tokens": {"prompt_total": 1000, "completion_total": 100, "thinking_total": 50, "cached_total": 0, "total": 1150},
            "config": {},
            "execution": {},
            "retry_stats": {"total_questions": 2, "succeeded_first_try": 0, "succeeded_after_retry": 0, "failed_all_retries": 2, "avg_attempts": 5.0},
            "errors": {"total_errors": 2, "by_phase": {"retrieval": 0, "rerank": 0, "generation": 2, "judge": 0}},
            "breakdown_by_type": {"single_hop": {"total": 2, "pass": 0, "partial": 0, "fail": 2, "pass_rate": 0}},
            "breakdown_by_difficulty": {"hard": {"total": 2, "pass": 0, "partial": 0, "fail": 2, "pass_rate": 0}},
            "results": [
                {"question_id": "q1", "question_type": "single_hop", "difficulty": "hard", "time": 10, "judgment": {"verdict": "fail", "correctness": 1, "completeness": 1, "faithfulness": 2, "relevance": 2, "clarity": 2, "overall_score": 1}},
                {"question_id": "q2", "question_type": "single_hop", "difficulty": "hard", "time": 10, "judgment": {"verdict": "fail", "correctness": 2, "completeness": 2, "faithfulness": 2, "relevance": 2, "clarity": 2, "overall_score": 2}},
            ],
        }
        
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "q1" in content
        assert "q2" in content


class TestMRRSection:
    """Tests for MRR section in report."""
    
    def test_report_contains_mrr_section(self, sample_run_data, tmp_path):
        """Test that report contains MRR Analysis section."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "Mean Reciprocal Rank (MRR) Analysis" in content
        assert "What is MRR?" in content
    
    def test_mrr_definition_and_example(self, sample_run_data, tmp_path):
        """Test that MRR section contains definition and example."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "Formula:" in content
        assert "Example:" in content
        assert "Interpretation:" in content
        assert "1/1 = 1.000" in content  # Example calculation
    
    def test_mrr_matrix_table(self, sample_run_data, tmp_path):
        """Test that MRR matrix table is present with difficulty x type."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "MRR Matrix" in content
        assert "Difficulty" in content
        assert "Single-hop" in content
        assert "Multi-hop" in content
        assert "Easy" in content
        assert "Medium" in content
        assert "Hard" in content
        assert "Total" in content
    
    def test_mrr_key_insight(self, sample_run_data, tmp_path):
        """Test that MRR section contains key insight."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "Key Insight:" in content


class TestIndexAndTopicType:
    """Tests for index and topic type info in report."""
    
    def test_report_contains_index_info(self, sample_run_data, tmp_path):
        """Test that report contains index information."""
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "Index:" in content
        assert "Topic Type:" in content
    
    def test_topic_type_enabled_detection(self, sample_run_data, tmp_path):
        """Test that topic type enabled is detected from index name."""
        sample_run_data['config']['index_name'] = 'bfai__eval66a_g1_1536_tt'
        
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "✅ Enabled" in content
    
    def test_topic_type_disabled_detection(self, sample_run_data, tmp_path):
        """Test that topic type disabled is detected from index name."""
        sample_run_data['config']['index_name'] = 'bfai__eval66a_g1_1536'  # No _tt
        
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "❌ Disabled" in content


class TestRunContext:
    """Tests for run context section in report."""
    
    def test_core_evaluation_detection(self, sample_run_data, tmp_path):
        """Test that core evaluation is detected for full corpus with default config."""
        # Make it look like a core eval (400+ questions, default config)
        sample_run_data['results'] = sample_run_data['results'] * 100  # 500 results
        sample_run_data['config']['precision_k'] = 25
        sample_run_data['config']['recall_k'] = 100
        
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "Core Evaluation" in content
        assert "standard core evaluation" in content
    
    def test_adhoc_test_detection(self, sample_run_data, tmp_path):
        """Test that ad-hoc test is detected for small corpus or non-default config."""
        # Small corpus = ad-hoc
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=None):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "Ad-Hoc Test" in content
        assert "ad-hoc test" in content
    
    def test_comparison_context_same_model(self, sample_run_data, tmp_path):
        """Test comparison context when models are the same."""
        baseline = {
            "config": {"generator_model": "gemini-3-flash-preview"},
            "metrics": {
                "pass_rate": 0.9, "partial_rate": 0.05, "fail_rate": 0.05,
                "acceptable_rate": 0.95, "recall_at_100": 0.99, "mrr": 0.75,
                "overall_score_avg": 4.5,
            }
        }
        sample_run_data['config']['generator_model'] = 'gemini-3-flash-preview'
        
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=baseline):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "same model" in content
    
    def test_comparison_context_different_model(self, sample_run_data, tmp_path):
        """Test comparison context when models are different."""
        baseline = {
            "config": {"generator_model": "gemini-2.5-flash"},
            "metrics": {
                "pass_rate": 0.85, "partial_rate": 0.1, "fail_rate": 0.05,
                "acceptable_rate": 0.95, "recall_at_100": 0.99, "mrr": 0.75,
                "overall_score_avg": 4.2,
            }
        }
        sample_run_data['config']['generator_model'] = 'gemini-3-flash-preview'
        
        results_path = tmp_path / "results.json"
        output_path = tmp_path / "report.md"
        
        with open(results_path, 'w') as f:
            json.dump(sample_run_data, f)
        
        with patch('generate_report.load_latest_baseline', return_value=baseline):
            generate_report(results_path, output_path)
        
        content = output_path.read_text()
        assert "Comparing models" in content
