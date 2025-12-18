"""
Unit tests for baseline_manager.py
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add scripts/eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "eval"))

from baseline_manager import (
    parse_baseline_filename,
    list_baselines,
    get_latest_baseline,
    load_baseline,
    save_baseline,
    get_baseline_path,
    compare_to_baseline,
    format_comparison_report,
    BASELINES_DIR,
)


class TestParseBaselineFilename:
    """Tests for parse_baseline_filename function."""
    
    def test_valid_filename(self):
        """Should parse valid baseline filename."""
        result = parse_baseline_filename("baseline_BFAI_v1__2025-12-17__q458.json")
        assert result == {
            "client": "BFAI",
            "version": "1",
            "date": "2025-12-17",
            "question_count": 458,
        }
    
    def test_multi_digit_version(self):
        """Should handle multi-digit versions."""
        result = parse_baseline_filename("baseline_BFAI_v12__2025-12-17__q458.json")
        assert result["version"] == "12"
    
    def test_different_client(self):
        """Should handle different client names."""
        result = parse_baseline_filename("baseline_ACME_v1__2025-01-01__q100.json")
        assert result["client"] == "ACME"
        assert result["question_count"] == 100
    
    def test_invalid_filename_returns_none(self):
        """Should return None for invalid filenames."""
        assert parse_baseline_filename("not_a_baseline.json") is None
        assert parse_baseline_filename("baseline_BFAI.json") is None
        assert parse_baseline_filename("random_file.txt") is None


class TestCompareToBaseline:
    """Tests for compare_to_baseline function."""
    
    @pytest.fixture
    def baseline(self):
        """Sample baseline data."""
        return {
            "baseline_version": "1",
            "created_date": "2025-12-17",
            "metrics": {
                "pass_rate": 0.856,
                "partial_rate": 0.105,
                "fail_rate": 0.039,
                "acceptable_rate": 0.961,
                "recall_at_100": 0.991,
                "mrr": 0.717,
                "overall_score_avg": 4.71,
            },
            "latency": {
                "total_avg_s": 8.3,
            },
            "cost": {
                "total_cost_usd": 0.15,
            },
        }
    
    def test_no_change(self, baseline):
        """Should show zero deltas when current matches baseline."""
        comparison = compare_to_baseline(baseline, baseline)
        
        assert comparison["deltas"]["pass_rate"]["delta"] == 0
        assert len(comparison["regressions"]) == 0
        assert len(comparison["improvements"]) == 0
    
    def test_improvement_detected(self, baseline):
        """Should detect improvements."""
        current = {
            "metrics": {
                "pass_rate": 0.90,  # +4.4% improvement
                "partial_rate": 0.08,
                "fail_rate": 0.02,
                "acceptable_rate": 0.98,
                "recall_at_100": 0.995,
                "mrr": 0.75,
                "overall_score_avg": 4.80,
            },
            "latency": {"total_avg_s": 8.0},
            "cost": {"total_cost_usd": 0.14},
        }
        
        comparison = compare_to_baseline(current, baseline)
        
        assert "pass_rate" in comparison["improvements"]
        assert comparison["deltas"]["pass_rate"]["delta"] > 0
    
    def test_regression_detected(self, baseline):
        """Should detect regressions."""
        current = {
            "metrics": {
                "pass_rate": 0.80,  # -5.6% regression
                "partial_rate": 0.12,
                "fail_rate": 0.08,  # +4.1% regression
                "acceptable_rate": 0.92,
                "recall_at_100": 0.98,
                "mrr": 0.70,
                "overall_score_avg": 4.50,
            },
            "latency": {"total_avg_s": 10.0},
            "cost": {"total_cost_usd": 0.20},
        }
        
        comparison = compare_to_baseline(current, baseline)
        
        assert "pass_rate" in comparison["regressions"]
        assert "fail_rate" in comparison["regressions"]
    
    def test_delta_calculation(self, baseline):
        """Should calculate deltas correctly."""
        current = {
            "metrics": {
                "pass_rate": 0.90,
                "partial_rate": 0.07,
                "fail_rate": 0.03,
                "acceptable_rate": 0.97,
                "recall_at_100": 0.995,
                "mrr": 0.75,
                "overall_score_avg": 4.80,
            },
            "latency": {"total_avg_s": 7.5},
            "cost": {"total_cost_usd": 0.12},
        }
        
        comparison = compare_to_baseline(current, baseline)
        
        # Check pass_rate delta
        assert comparison["deltas"]["pass_rate"]["baseline"] == 0.856
        assert comparison["deltas"]["pass_rate"]["current"] == 0.90
        assert comparison["deltas"]["pass_rate"]["delta"] == pytest.approx(0.044, rel=1e-2)


class TestFormatComparisonReport:
    """Tests for format_comparison_report function."""
    
    def test_generates_markdown(self):
        """Should generate valid markdown."""
        comparison = {
            "baseline_version": "1",
            "baseline_date": "2025-12-17",
            "deltas": {
                "pass_rate": {"baseline": 0.856, "current": 0.90, "delta": 0.044},
                "fail_rate": {"baseline": 0.039, "current": 0.03, "delta": -0.009},
            },
            "regressions": [],
            "improvements": ["pass_rate"],
        }
        
        report = format_comparison_report(comparison)
        
        assert "## Comparison to Baseline" in report
        assert "v1" in report
        assert "2025-12-17" in report
        assert "Improvements" in report
        assert "pass_rate" in report
    
    def test_shows_regressions(self):
        """Should show regressions section when present."""
        comparison = {
            "baseline_version": "1",
            "baseline_date": "2025-12-17",
            "deltas": {
                "pass_rate": {"baseline": 0.856, "current": 0.80, "delta": -0.056},
            },
            "regressions": ["pass_rate"],
            "improvements": [],
        }
        
        report = format_comparison_report(comparison)
        
        assert "Regressions" in report
        assert "⚠️" in report


class TestListBaselines:
    """Tests for list_baselines function."""
    
    def test_returns_list(self):
        """Should return a list."""
        result = list_baselines()
        assert isinstance(result, list)
    
    def test_filters_by_client(self):
        """Should filter by client when specified."""
        # This test depends on actual baselines existing
        all_baselines = list_baselines()
        bfai_baselines = list_baselines("BFAI")
        
        # All BFAI baselines should have client == BFAI
        for b in bfai_baselines:
            assert b["client"].upper() == "BFAI"


class TestGetLatestBaseline:
    """Tests for get_latest_baseline function."""
    
    def test_returns_dict_or_none(self):
        """Should return dict or None."""
        result = get_latest_baseline("BFAI")
        assert result is None or isinstance(result, dict)
    
    def test_bfai_baseline_exists(self):
        """BFAI baseline should exist (we created it)."""
        result = get_latest_baseline("BFAI")
        assert result is not None
        assert result.get("client") == "BFAI"
        assert result.get("baseline_version") == "1"
    
    def test_nonexistent_client_returns_none(self):
        """Should return None for client with no baselines."""
        result = get_latest_baseline("NONEXISTENT_CLIENT_XYZ")
        assert result is None


class TestGetBaselinePath:
    """Tests for get_baseline_path function."""
    
    def test_generates_correct_path(self):
        """Should generate correct filename."""
        path = get_baseline_path("BFAI", "1", "2025-12-17", 458)
        assert path.name == "baseline_BFAI_v1__2025-12-17__q458.json"
    
    def test_path_is_in_baselines_dir(self):
        """Path should be in baselines directory."""
        path = get_baseline_path("TEST", "2", "2025-01-01", 100)
        assert path.parent == BASELINES_DIR
    
    def test_different_versions(self):
        """Should handle different version numbers."""
        path1 = get_baseline_path("BFAI", "1", "2025-12-17", 458)
        path2 = get_baseline_path("BFAI", "10", "2025-12-17", 458)
        assert "v1__" in path1.name
        assert "v10__" in path2.name


class TestLoadBaseline:
    """Tests for load_baseline function."""
    
    def test_load_existing_baseline(self):
        """Should load existing baseline file."""
        baselines = list_baselines("BFAI")
        if baselines:
            data = load_baseline(baselines[0]["path"])
            assert isinstance(data, dict)
            assert "client" in data or "metrics" in data
    
    def test_load_nonexistent_raises_error(self):
        """Should raise error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_baseline("/nonexistent/path/baseline.json")


class TestSaveBaseline:
    """Tests for save_baseline function."""
    
    @pytest.fixture
    def temp_baselines_dir(self, tmp_path, monkeypatch):
        """Use temp directory for baselines."""
        import baseline_manager
        monkeypatch.setattr(baseline_manager, "BASELINES_DIR", tmp_path)
        return tmp_path
    
    def test_save_creates_file(self, temp_baselines_dir):
        """Should create baseline file."""
        data = {
            "client": "TEST",
            "corpus": {"question_count": 100},
            "metrics": {"pass_rate": 0.85},
        }
        path = save_baseline(data, "TEST", version="1", date="2025-01-01")
        assert path.exists()
    
    def test_save_auto_increments_version(self, temp_baselines_dir):
        """Should auto-increment version when not specified."""
        data = {"corpus": {"question_count": 50}}
        
        # Save first version
        path1 = save_baseline(data, "AUTOTEST", date="2025-01-01")
        assert "v1__" in path1.name
        
        # Save second version (should auto-increment)
        path2 = save_baseline(data, "AUTOTEST", date="2025-01-02")
        assert "v2__" in path2.name
    
    def test_save_uses_today_date_if_not_specified(self, temp_baselines_dir):
        """Should use today's date when not specified."""
        from datetime import datetime
        data = {"corpus": {"question_count": 10}}
        path = save_baseline(data, "DATETEST", version="1")
        today = datetime.now().strftime("%Y-%m-%d")
        assert today in path.name
    
    def test_save_load_roundtrip(self, temp_baselines_dir):
        """Saved data should match loaded data."""
        original = {
            "client": "ROUNDTRIP",
            "corpus": {"question_count": 200},
            "metrics": {"pass_rate": 0.90, "mrr": 0.75},
            "config": {"model": "gemini-2.5-flash"},
        }
        path = save_baseline(original, "ROUNDTRIP", version="1", date="2025-06-15")
        loaded = load_baseline(str(path))
        
        assert loaded["client"] == original["client"]
        assert loaded["metrics"] == original["metrics"]
        assert loaded["config"] == original["config"]


class TestCompareToBaselineEdgeCases:
    """Edge cases for compare_to_baseline."""
    
    def test_missing_metrics_in_baseline(self):
        """Should handle missing metrics gracefully."""
        baseline = {"baseline_version": "1", "created_date": "2025-01-01"}
        current = {"metrics": {"pass_rate": 0.85}}
        
        comparison = compare_to_baseline(current, baseline)
        assert "pass_rate" in comparison["deltas"]
        assert comparison["deltas"]["pass_rate"]["baseline"] == 0
    
    def test_missing_metrics_in_current(self):
        """Should handle missing metrics in current run."""
        baseline = {"metrics": {"pass_rate": 0.85}, "baseline_version": "1"}
        current = {"metrics": {}}
        
        comparison = compare_to_baseline(current, baseline)
        assert comparison["deltas"]["pass_rate"]["current"] == 0
    
    def test_empty_dicts(self):
        """Should handle empty dicts without crashing."""
        comparison = compare_to_baseline({}, {})
        assert isinstance(comparison, dict)
        assert "deltas" in comparison
    
    def test_threshold_boundary_no_regression(self):
        """Delta just at threshold should not be flagged."""
        baseline = {"metrics": {"pass_rate": 0.85}, "baseline_version": "1"}
        current = {"metrics": {"pass_rate": 0.831}}  # -0.019 > -0.02 threshold
        
        comparison = compare_to_baseline(current, baseline)
        # Just above threshold, so no regression
        assert "pass_rate" not in comparison["regressions"]
    
    def test_threshold_boundary_regression(self):
        """Delta just below threshold should be flagged."""
        baseline = {"metrics": {"pass_rate": 0.85}, "baseline_version": "1"}
        current = {"metrics": {"pass_rate": 0.82}}  # -0.03 < -0.02 threshold
        
        comparison = compare_to_baseline(current, baseline)
        assert "pass_rate" in comparison["regressions"]


class TestFormatComparisonReportEdgeCases:
    """Edge cases for format_comparison_report."""
    
    def test_empty_deltas(self):
        """Should handle empty deltas."""
        comparison = {
            "baseline_version": "1",
            "baseline_date": "2025-01-01",
            "deltas": {},
            "regressions": [],
            "improvements": [],
        }
        report = format_comparison_report(comparison)
        assert "Comparison to Baseline" in report
    
    def test_no_regressions_or_improvements(self):
        """Should not show sections if empty."""
        comparison = {
            "baseline_version": "1",
            "baseline_date": "2025-01-01",
            "deltas": {"pass_rate": {"baseline": 0.85, "current": 0.85, "delta": 0}},
            "regressions": [],
            "improvements": [],
        }
        report = format_comparison_report(comparison)
        assert "Regressions" not in report
        assert "Improvements" not in report


class TestIdempotency:
    """Idempotency tests for baseline operations."""
    
    def test_list_baselines_idempotent(self):
        """Listing baselines should be idempotent."""
        result1 = list_baselines("BFAI")
        result2 = list_baselines("BFAI")
        assert result1 == result2
    
    def test_compare_to_baseline_idempotent(self):
        """Comparison should be idempotent."""
        baseline = {"metrics": {"pass_rate": 0.85}, "baseline_version": "1"}
        current = {"metrics": {"pass_rate": 0.90}}
        
        result1 = compare_to_baseline(current, baseline)
        result2 = compare_to_baseline(current, baseline)
        assert result1 == result2
    
    def test_format_comparison_report_idempotent(self):
        """Report formatting should be idempotent."""
        comparison = {
            "baseline_version": "1",
            "baseline_date": "2025-01-01",
            "deltas": {"pass_rate": {"baseline": 0.85, "current": 0.90, "delta": 0.05}},
            "regressions": [],
            "improvements": ["pass_rate"],
        }
        result1 = format_comparison_report(comparison)
        result2 = format_comparison_report(comparison)
        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
