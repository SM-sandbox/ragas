"""
Integration tests for baseline workflow.

Tests the full baseline save → load → compare cycle.
"""

import json
import pytest
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestBaselineWorkflow:
    """Integration tests for baseline workflow."""
    
    def test_save_load_compare_cycle(self, tmp_path):
        """Full baseline save → load → compare cycle."""
        from lib.core.baseline_manager import (
            save_baseline,
            load_baseline,
            compare_to_baseline,
            format_comparison_report,
            BASELINES_DIR,
        )
        import lib.core.baseline_manager as bm
        
        # Setup temp directory
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        original_dir = bm.BASELINES_DIR
        bm.BASELINES_DIR = baselines_dir
        
        try:
            # Create baseline data
            baseline_data = {
                "schema_version": "1.0",
                "client": "INTTEST",
                "corpus": {"question_count": 50},
                "metrics": {
                    "pass_rate": 0.90,
                    "partial_rate": 0.07,
                    "fail_rate": 0.03,
                    "acceptable_rate": 0.97,
                    "recall_at_100": 0.99,
                    "mrr": 0.75,
                    "overall_score_avg": 4.65,
                },
                "latency": {"total_avg_s": 8.0},
                "cost": {"total_cost_usd": 0.05},
            }
            
            # Save baseline
            path = save_baseline(baseline_data, "INTTEST")
            assert path.exists()
            
            # Load baseline
            loaded = load_baseline(str(path))
            assert loaded["client"] == "INTTEST"
            assert loaded["metrics"]["pass_rate"] == 0.90
            
            # Create current run with regression
            current_run = {
                "metrics": {
                    "pass_rate": 0.85,  # Regression
                    "partial_rate": 0.10,
                    "fail_rate": 0.05,  # Regression
                    "acceptable_rate": 0.95,
                    "recall_at_100": 0.98,
                    "mrr": 0.73,
                    "overall_score_avg": 4.50,
                },
                "latency": {"total_avg_s": 9.0},
                "cost": {"total_cost_usd": 0.06},
            }
            
            # Compare
            comparison = compare_to_baseline(current_run, loaded)
            
            # Verify regression detected
            assert "pass_rate" in comparison["regressions"]
            assert "fail_rate" in comparison["regressions"]
            
            # Verify deltas calculated
            assert comparison["deltas"]["pass_rate"]["delta"] == pytest.approx(-0.05, abs=0.001)
            
            # Format report
            report = format_comparison_report(comparison)
            assert "Regression" in report or "regression" in report
            assert "pass_rate" in report
            
        finally:
            bm.BASELINES_DIR = original_dir
    
    def test_multiple_baselines_versioning(self, tmp_path):
        """Multiple baselines are versioned correctly."""
        from lib.core.baseline_manager import (
            save_baseline,
            list_baselines,
        )
        import lib.core.baseline_manager as bm
        
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        original_dir = bm.BASELINES_DIR
        bm.BASELINES_DIR = baselines_dir
        
        try:
            # Save 3 baselines
            for i in range(3):
                data = {
                    "schema_version": "1.0",
                    "client": "VERTEST",
                    "corpus": {"question_count": 100},
                    "metrics": {"pass_rate": 0.90 + i * 0.01},
                }
                save_baseline(data, "VERTEST")
            
            # List baselines
            baselines = list_baselines("VERTEST")
            
            # Should have 3 baselines
            assert len(baselines) == 3
            
            # Versions should be 1, 2, 3
            versions = sorted([b["version"] for b in baselines])
            assert versions == ["1", "2", "3"]
            
        finally:
            bm.BASELINES_DIR = original_dir


class TestCICDWorkflow:
    """Integration tests for CI/CD workflow."""
    
    def test_cicd_import_check(self):
        """CI/CD import check passes."""
        from evaluations.cicd.run_cicd_eval import check_imports
        
        result = check_imports()
        assert result is True
    
    def test_cicd_config_validation(self):
        """CI/CD config validation passes."""
        from evaluations.cicd.run_cicd_eval import validate_config
        
        result = validate_config()
        assert result is True
    
    def test_cicd_threshold_check_pass(self):
        """CI/CD threshold check passes with good metrics."""
        from evaluations.cicd.run_cicd_eval import check_thresholds
        
        output = {
            "metrics": {
                "pass_rate": 0.90,
                "fail_rate": 0.03,
            },
            "results": [
                {"question_id": "q1"},
                {"question_id": "q2"},
            ]
        }
        
        passed, failures = check_thresholds(output)
        assert passed is True
        assert len(failures) == 0
    
    def test_cicd_threshold_check_fail(self):
        """CI/CD threshold check fails with bad metrics."""
        from evaluations.cicd.run_cicd_eval import check_thresholds
        
        output = {
            "metrics": {
                "pass_rate": 0.80,  # Below 85% threshold
                "fail_rate": 0.10,  # Above 8% threshold
            },
            "results": []
        }
        
        passed, failures = check_thresholds(output)
        assert passed is False
        assert len(failures) >= 1


class TestFixtureLoading:
    """Tests for loading test fixtures."""
    
    def test_load_tiny_corpus(self):
        """Can load tiny corpus fixture."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "tiny_corpus.json"
        
        with open(fixture_path) as f:
            corpus = json.load(f)
        
        assert "questions" in corpus
        assert len(corpus["questions"]) == 3
        
        # Check question structure
        q = corpus["questions"][0]
        assert "question_id" in q
        assert "question" in q
        assert "ground_truth_answer" in q
    
    def test_load_sample_baseline(self):
        """Can load sample baseline fixture."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_baseline.json"
        
        with open(fixture_path) as f:
            baseline = json.load(f)
        
        assert baseline["schema_version"] == "1.0"
        assert baseline["client"] == "TEST"
        assert "metrics" in baseline
        assert baseline["metrics"]["pass_rate"] == 0.90
    
    def test_load_sample_results(self):
        """Can load sample results fixture."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_results.jsonl"
        
        results = []
        with open(fixture_path) as f:
            for line in f:
                results.append(json.loads(line))
        
        assert len(results) == 3
        
        # Check result structure
        r = results[0]
        assert "question_id" in r
        assert "judgment" in r
        assert "verdict" in r["judgment"]
