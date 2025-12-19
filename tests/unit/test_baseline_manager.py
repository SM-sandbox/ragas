"""
Unit tests for lib/core/baseline_manager.py

Tests baseline CRUD operations, filename parsing, comparison logic, and versioning.
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import sys

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.core.baseline_manager import (
    parse_baseline_filename,
    get_baseline_path,
    list_baselines,
    get_latest_baseline,
    load_baseline,
    save_baseline,
    compare_to_baseline,
    format_comparison_report,
    BASELINES_DIR,
)


class TestParseBaselineFilename:
    """Tests for parse_baseline_filename function."""
    
    def test_parse_valid_filename(self):
        """Parse valid baseline filename."""
        result = parse_baseline_filename("baseline_BFAI_v1__2025-12-17__q458.json")
        assert result is not None
        assert result["client"] == "BFAI"
        assert result["version"] == "1"
        assert result["date"] == "2025-12-17"
        assert result["question_count"] == 458
    
    def test_parse_valid_filename_v2(self):
        """Parse valid baseline filename with version 2."""
        result = parse_baseline_filename("baseline_BFAI_v2__2025-12-18__q458.json")
        assert result is not None
        assert result["version"] == "2"
        assert result["date"] == "2025-12-18"
    
    def test_parse_invalid_filename_no_prefix(self):
        """Reject filename without baseline_ prefix."""
        result = parse_baseline_filename("BFAI_v1__2025-12-17__q458.json")
        assert result is None
    
    def test_parse_invalid_filename_wrong_format(self):
        """Reject filename with wrong format."""
        result = parse_baseline_filename("baseline_BFAI_2025-12-17.json")
        assert result is None
    
    def test_parse_invalid_filename_no_version(self):
        """Reject filename without version."""
        result = parse_baseline_filename("baseline_BFAI__2025-12-17__q458.json")
        assert result is None
    
    def test_parse_invalid_filename_bad_date(self):
        """Reject filename with invalid date format."""
        result = parse_baseline_filename("baseline_BFAI_v1__12-17-2025__q458.json")
        assert result is None


class TestGetBaselinePath:
    """Tests for get_baseline_path function."""
    
    def test_generate_path(self):
        """Generate correct baseline path."""
        path = get_baseline_path("BFAI", "1", "2025-12-17", 458)
        assert path.name == "baseline_BFAI_v1__2025-12-17__q458.json"
        assert path.parent == BASELINES_DIR
    
    def test_generate_path_different_client(self):
        """Generate path for different client."""
        path = get_baseline_path("TEST", "3", "2025-01-01", 100)
        assert path.name == "baseline_TEST_v3__2025-01-01__q100.json"


class TestListBaselines:
    """Tests for list_baselines function."""
    
    def test_list_baselines_returns_list(self):
        """list_baselines returns a list."""
        result = list_baselines()
        assert isinstance(result, list)
    
    def test_list_baselines_filtered_by_client(self):
        """list_baselines filters by client."""
        result = list_baselines(client="BFAI")
        for b in result:
            assert b["client"].upper() == "BFAI"
    
    def test_list_baselines_sorted_by_date(self):
        """list_baselines returns sorted by date descending."""
        result = list_baselines()
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i]["date"] >= result[i + 1]["date"]


class TestGetLatestBaseline:
    """Tests for get_latest_baseline function."""
    
    def test_get_latest_baseline_bfai(self):
        """Get latest baseline for BFAI client."""
        result = get_latest_baseline("BFAI")
        # Should return a dict or None
        assert result is None or isinstance(result, dict)
        if result:
            assert "baseline_version" in result or "metrics" in result
    
    def test_get_latest_baseline_nonexistent_client(self):
        """Get latest baseline for non-existent client returns None."""
        result = get_latest_baseline("NONEXISTENT_CLIENT_XYZ")
        assert result is None


class TestLoadBaseline:
    """Tests for load_baseline function."""
    
    def test_load_baseline_from_fixture(self):
        """Load baseline from fixture file."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "sample_baseline.json"
        if fixture_path.exists():
            result = load_baseline(str(fixture_path))
            assert isinstance(result, dict)
            assert result["schema_version"] == "1.0"
            assert result["client"] == "TEST"
            assert result["metrics"]["pass_rate"] == 0.90


class TestSaveBaseline:
    """Tests for save_baseline function."""
    
    def test_save_baseline_creates_file(self, tmp_path):
        """save_baseline creates a file."""
        # Create a temporary baselines directory
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        
        # Patch BASELINES_DIR temporarily
        import lib.core.baseline_manager as bm
        original_dir = bm.BASELINES_DIR
        bm.BASELINES_DIR = baselines_dir
        
        try:
            data = {
                "schema_version": "1.0",
                "baseline_version": None,
                "created_date": "2025-12-19",
                "client": "TEST",
                "corpus": {"question_count": 100},
                "metrics": {"pass_rate": 0.90},
            }
            
            path = save_baseline(data, "TEST")
            assert path.exists()
            
            # Verify content
            with open(path) as f:
                saved = json.load(f)
            assert saved["metrics"]["pass_rate"] == 0.90
        finally:
            bm.BASELINES_DIR = original_dir
    
    def test_save_baseline_auto_increments_version(self, tmp_path):
        """save_baseline auto-increments version."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        
        import lib.core.baseline_manager as bm
        original_dir = bm.BASELINES_DIR
        bm.BASELINES_DIR = baselines_dir
        
        try:
            # Create first baseline
            data1 = {"schema_version": "1.0", "client": "TEST", "corpus": {"question_count": 100}, "metrics": {}}
            path1 = save_baseline(data1, "TEST")
            
            # Create second baseline
            data2 = {"schema_version": "1.0", "client": "TEST", "corpus": {"question_count": 100}, "metrics": {}}
            path2 = save_baseline(data2, "TEST")
            
            # Verify versions by filename (v1 and v2)
            assert "v1" in path1.name
            assert "v2" in path2.name
        finally:
            bm.BASELINES_DIR = original_dir


class TestCompareToBaseline:
    """Tests for compare_to_baseline function."""
    
    def test_compare_detects_regression(self):
        """compare_to_baseline detects regression."""
        current = {
            "metrics": {
                "pass_rate": 0.85,
                "fail_rate": 0.05,
                "acceptable_rate": 0.95,
                "overall_score_avg": 4.5,
            }
        }
        baseline = {
            "metrics": {
                "pass_rate": 0.92,
                "fail_rate": 0.02,
                "acceptable_rate": 0.98,
                "overall_score_avg": 4.7,
            }
        }
        
        result = compare_to_baseline(current, baseline)
        
        assert "deltas" in result
        assert "regressions" in result
        assert "improvements" in result
        
        # Pass rate dropped - should be a regression
        assert "pass_rate" in result["regressions"]
        # Fail rate increased - should be a regression
        assert "fail_rate" in result["regressions"]
    
    def test_compare_detects_improvement(self):
        """compare_to_baseline detects improvement."""
        current = {
            "metrics": {
                "pass_rate": 0.95,
                "fail_rate": 0.01,
                "acceptable_rate": 0.99,
                "overall_score_avg": 4.8,
            }
        }
        baseline = {
            "metrics": {
                "pass_rate": 0.90,
                "fail_rate": 0.05,  # Higher fail rate so decrease is > threshold
                "acceptable_rate": 0.97,
                "overall_score_avg": 4.6,
            }
        }
        
        result = compare_to_baseline(current, baseline)
        
        # Pass rate improved
        assert "pass_rate" in result["improvements"]
        # Fail rate decreased by 0.04 which is > 0.02 threshold
        assert "fail_rate" in result["improvements"]
    
    def test_compare_handles_missing_metrics(self):
        """compare_to_baseline handles missing metrics gracefully."""
        current = {"metrics": {"pass_rate": 0.90}}
        baseline = {"metrics": {"pass_rate": 0.90, "extra_metric": 0.5}}
        
        # Should not raise
        result = compare_to_baseline(current, baseline)
        assert result is not None


class TestFormatComparisonReport:
    """Tests for format_comparison_report function."""
    
    def test_format_report_returns_string(self):
        """format_comparison_report returns a string."""
        comparison = {
            "baseline_version": "1",
            "baseline_date": "2025-12-19",
            "deltas": {
                "pass_rate": {"baseline": 0.90, "current": 0.85, "delta": -0.05}
            },
            "regressions": ["pass_rate"],
            "improvements": [],
        }
        
        result = format_comparison_report(comparison)
        assert isinstance(result, str)
        assert "pass_rate" in result
    
    def test_format_report_includes_regressions(self):
        """format_comparison_report includes regressions."""
        comparison = {
            "baseline_version": "1",
            "baseline_date": "2025-12-19",
            "deltas": {
                "pass_rate": {"baseline": 0.90, "current": 0.85, "delta": -0.05},
                "fail_rate": {"baseline": 0.02, "current": 0.05, "delta": 0.03},
            },
            "regressions": ["pass_rate", "fail_rate"],
            "improvements": [],
        }
        
        result = format_comparison_report(comparison)
        assert "Regression" in result or "regression" in result.lower()


class TestVersionAutoIncrement:
    """Tests for version auto-increment in save_baseline."""
    
    def test_first_baseline_gets_version_1(self, tmp_path):
        """First baseline for a client gets version 1."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        
        import lib.core.baseline_manager as bm
        original_dir = bm.BASELINES_DIR
        bm.BASELINES_DIR = baselines_dir
        
        try:
            data = {"schema_version": "1.0", "client": "NEWCLIENT", "corpus": {"question_count": 10}}
            path = save_baseline(data, "NEWCLIENT")
            
            with open(path) as f:
                saved = json.load(f)
            assert saved.get("baseline_version") is None or saved.get("baseline_version") == "1"
        finally:
            bm.BASELINES_DIR = original_dir
    
    def test_subsequent_baseline_increments_version(self, tmp_path):
        """Subsequent baselines increment version number."""
        baselines_dir = tmp_path / "baselines"
        baselines_dir.mkdir()
        
        # Create existing baseline file
        existing = baselines_dir / "baseline_TEST_v3__2025-12-19__q100.json"
        existing.write_text('{"baseline_version": "3"}')
        
        import lib.core.baseline_manager as bm
        original_dir = bm.BASELINES_DIR
        bm.BASELINES_DIR = baselines_dir
        
        try:
            data = {"schema_version": "1.0", "client": "TEST", "corpus": {"question_count": 100}}
            path = save_baseline(data, "TEST")
            
            # Should be v4 since v3 exists
            assert "v4" in path.name
        finally:
            bm.BASELINES_DIR = original_dir
