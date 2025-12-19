"""
Unit tests for core/preflight.py - Pre-flight Check System.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")

import pytest
from lib.utils.preflight import (
    PreflightCheck,
    PreflightResult,
    run_preflight_checks,
)


class TestPreflightCheck:
    """Tests for PreflightCheck dataclass."""
    
    def test_pass_status_is_passed(self):
        """PASS status should be considered passed."""
        check = PreflightCheck(
            name="Test",
            status=CheckStatus.PASS,
            message="OK",
        )
        assert check.passed is True
    
    def test_fail_status_is_not_passed(self):
        """FAIL status should not be considered passed."""
        check = PreflightCheck(
            name="Test",
            status=CheckStatus.FAIL,
            message="Failed",
        )
        assert check.passed is False
    
    def test_warn_status_is_passed(self):
        """WARN status should be considered passed (non-blocking)."""
        check = PreflightCheck(
            name="Test",
            status=CheckStatus.WARN,
            message="Warning",
        )
        assert check.passed is True
    
    def test_skip_status_is_passed(self):
        """SKIP status should be considered passed."""
        check = PreflightCheck(
            name="Test",
            status=CheckStatus.SKIP,
            message="Skipped",
        )
        assert check.passed is True
    
    def test_icon_mapping(self):
        """Each status should have correct icon."""
        assert PreflightCheck(name="", status=CheckStatus.PASS, message="").icon == "✅"
        assert PreflightCheck(name="", status=CheckStatus.FAIL, message="").icon == "❌"
        assert PreflightCheck(name="", status=CheckStatus.WARN, message="").icon == "⚠️"
        assert PreflightCheck(name="", status=CheckStatus.SKIP, message="").icon == "⏭️"


class TestPreflightResult:
    """Tests for PreflightResult dataclass."""
    
    def test_all_passed_with_all_pass(self):
        """all_passed should be True when all checks pass."""
        result = PreflightResult(checks=[
            PreflightCheck(name="A", status=CheckStatus.PASS, message="OK"),
            PreflightCheck(name="B", status=CheckStatus.PASS, message="OK"),
        ])
        assert result.all_passed is True
    
    def test_all_passed_with_one_fail(self):
        """all_passed should be False when any check fails."""
        result = PreflightResult(checks=[
            PreflightCheck(name="A", status=CheckStatus.PASS, message="OK"),
            PreflightCheck(name="B", status=CheckStatus.FAIL, message="Failed"),
        ])
        assert result.all_passed is False
    
    def test_critical_failures_list(self):
        """critical_failures should list only FAIL checks."""
        result = PreflightResult(checks=[
            PreflightCheck(name="A", status=CheckStatus.PASS, message="OK"),
            PreflightCheck(name="B", status=CheckStatus.FAIL, message="Failed"),
            PreflightCheck(name="C", status=CheckStatus.WARN, message="Warning"),
        ])
        failures = result.critical_failures
        assert len(failures) == 1
        assert failures[0].name == "B"
    
    def test_warnings_list(self):
        """warnings should list only WARN checks."""
        result = PreflightResult(checks=[
            PreflightCheck(name="A", status=CheckStatus.PASS, message="OK"),
            PreflightCheck(name="B", status=CheckStatus.WARN, message="Warning"),
        ])
        warnings = result.warnings
        assert len(warnings) == 1
        assert warnings[0].name == "B"
    
    def test_to_markdown_returns_string(self):
        """to_markdown should return a string."""
        result = PreflightResult(checks=[
            PreflightCheck(name="Test", status=CheckStatus.PASS, message="OK"),
        ])
        md = result.to_markdown()
        assert isinstance(md, str)
        assert "Pre-flight Checks" in md
    
    def test_to_dict_returns_dict(self):
        """to_dict should return a dictionary."""
        result = PreflightResult(checks=[
            PreflightCheck(name="Test", status=CheckStatus.PASS, message="OK"),
        ])
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "checks" in d
        assert "all_passed" in d


class TestIndividualChecks:
    """Tests for individual check functions."""
    
    def test_check_gcp_auth_returns_check(self):
        """check_gcp_auth should return a PreflightCheck."""
        result = check_gcp_auth()
        assert isinstance(result, PreflightCheck)
        assert result.name == "GCP Auth"
    
    def test_check_orchestrator_import_returns_check(self):
        """check_orchestrator_import should return a PreflightCheck."""
        result = check_orchestrator_import()
        assert isinstance(result, PreflightCheck)
        assert result.name == "Orchestrator Import"
    
    def test_check_model_registry_returns_check(self):
        """check_model_registry should return a PreflightCheck."""
        result = check_model_registry()
        assert isinstance(result, PreflightCheck)
        assert result.name == "Model Registry"
    
    def test_check_corpus_file_with_valid_file(self):
        """check_corpus_file should pass for valid file."""
        corpus_path = str(Path(__file__).parent.parent.parent / "corpus" / "qa_corpus_200.json")
        result = check_corpus_file(corpus_path)
        assert result.status == CheckStatus.PASS
    
    def test_check_corpus_file_with_invalid_file(self):
        """check_corpus_file should fail for missing file."""
        result = check_corpus_file("/nonexistent/file.json")
        assert result.status == CheckStatus.FAIL
    
    def test_check_model_valid_with_valid_model(self):
        """check_model_valid should pass for valid model."""
        result = check_model_valid("gemini-2.5-flash")
        assert result.status == CheckStatus.PASS
    
    def test_check_model_valid_with_invalid_model(self):
        """check_model_valid should fail for invalid model."""
        result = check_model_valid("invalid-model")
        assert result.status == CheckStatus.FAIL


class TestRunPreflightChecks:
    """Tests for run_preflight_checks function."""
    
    def test_returns_preflight_result(self):
        """Should return a PreflightResult."""
        config = PreflightConfig(skip_api_check=True)
        result = run_preflight_checks(config)
        assert isinstance(result, PreflightResult)
    
    def test_includes_core_checks(self):
        """Should include core checks."""
        config = PreflightConfig(skip_api_check=True)
        result = run_preflight_checks(config)
        
        check_names = [c.name for c in result.checks]
        assert "GCP Auth" in check_names
        assert "Orchestrator Import" in check_names
        assert "Model Registry" in check_names
    
    def test_skips_api_check_when_configured(self):
        """Should skip API check when skip_api_check=True."""
        config = PreflightConfig(skip_api_check=True)
        result = run_preflight_checks(config)
        
        api_check = next(c for c in result.checks if c.name == "API Connectivity")
        assert api_check.status == CheckStatus.SKIP
    
    def test_includes_job_config_check_when_provided(self):
        """Should include job config check when job_id provided."""
        config = PreflightConfig(
            job_id="bfai__eval66a_g1_1536_tt",
            skip_api_check=True,
        )
        result = run_preflight_checks(config)
        
        check_names = [c.name for c in result.checks]
        assert "Job Config" in check_names
    
    def test_tracks_total_duration(self):
        """Should track total duration."""
        config = PreflightConfig(skip_api_check=True)
        result = run_preflight_checks(config)
        
        assert result.total_duration_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
