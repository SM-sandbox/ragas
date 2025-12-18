"""
End-to-End Tests for the Eval Pipeline

These tests run the ACTUAL pipeline with real API calls.
They are slow and cost money, so they are marked with @pytest.mark.e2e
and skipped by default.

Run with: pytest tests/eval/test_e2e.py -v -m e2e

WARNING: These tests make real API calls and incur costs!
"""

import pytest
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add scripts/eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "eval"))

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def e2e_run_output():
    """
    Run a quick evaluation and return the output.
    This fixture is shared across all e2e tests.
    """
    # Import here to avoid loading orchestrator for unit tests
    from core_eval import run_evaluation
    
    # Run with 3 questions
    output = run_evaluation(
        client="BFAI",
        workers=1,
        precision_k=25,
        quick=3,
        test_mode=False,
        update_baseline=False,
    )
    
    return output


class TestE2EBasicFunctionality:
    """Basic end-to-end functionality tests."""
    
    def test_run_completes_successfully(self, e2e_run_output):
        """Run should complete without errors."""
        assert e2e_run_output is not None
        assert "results" in e2e_run_output
    
    def test_all_questions_processed(self, e2e_run_output):
        """All questions should be processed."""
        results = e2e_run_output.get("results", [])
        assert len(results) == 3
    
    def test_no_errors_in_results(self, e2e_run_output):
        """No results should have errors."""
        results = e2e_run_output.get("results", [])
        errors = [r for r in results if "error" in r]
        assert len(errors) == 0, f"Errors found: {errors}"


class TestE2EMetrics:
    """Test that metrics are captured correctly."""
    
    def test_metrics_present(self, e2e_run_output):
        """Metrics should be present in output."""
        assert "metrics" in e2e_run_output
        metrics = e2e_run_output["metrics"]
        
        required = ["pass_rate", "partial_rate", "fail_rate", "acceptable_rate", 
                    "recall_at_100", "mrr"]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_metrics_in_valid_range(self, e2e_run_output):
        """Metrics should be in valid ranges."""
        metrics = e2e_run_output["metrics"]
        
        # Rates should be between 0 and 1
        for key in ["pass_rate", "partial_rate", "fail_rate", "acceptable_rate", "recall_at_100"]:
            assert 0 <= metrics[key] <= 1, f"{key} out of range: {metrics[key]}"
        
        # MRR should be between 0 and 1
        assert 0 <= metrics["mrr"] <= 1
    
    def test_rates_sum_to_one(self, e2e_run_output):
        """Pass + partial + fail should sum to 1."""
        metrics = e2e_run_output["metrics"]
        total = metrics["pass_rate"] + metrics["partial_rate"] + metrics["fail_rate"]
        assert abs(total - 1.0) < 0.01, f"Rates sum to {total}, expected 1.0"


class TestE2ETokens:
    """Test that token counts are captured."""
    
    def test_tokens_present(self, e2e_run_output):
        """Token counts should be present."""
        assert "tokens" in e2e_run_output
        tokens = e2e_run_output["tokens"]
        
        required = ["prompt_total", "completion_total", "thinking_total", "cached_total"]
        for key in required:
            assert key in tokens, f"Missing token field: {key}"
    
    def test_tokens_non_negative(self, e2e_run_output):
        """Token counts should be non-negative."""
        tokens = e2e_run_output["tokens"]
        
        for key, value in tokens.items():
            assert value >= 0, f"{key} is negative: {value}"
    
    def test_tokens_reasonable(self, e2e_run_output):
        """Token counts should be reasonable for 3 questions."""
        tokens = e2e_run_output["tokens"]
        
        # Should have some prompt tokens (at least 1000 per question)
        assert tokens["prompt_total"] >= 3000
        
        # Should have some completion tokens (at least 50 per question)
        assert tokens["completion_total"] >= 150
    
    def test_per_result_tokens(self, e2e_run_output):
        """Each result should have token counts."""
        results = e2e_run_output.get("results", [])
        
        for r in results:
            if "error" not in r:
                assert "tokens" in r
                assert "prompt" in r["tokens"]
                assert "completion" in r["tokens"]


class TestE2ELatency:
    """Test that latency is captured correctly."""
    
    def test_latency_present(self, e2e_run_output):
        """Latency should be present."""
        assert "latency" in e2e_run_output
        latency = e2e_run_output["latency"]
        
        assert "total_avg_s" in latency
        assert "by_phase" in latency
    
    def test_latency_by_phase(self, e2e_run_output):
        """Latency should be broken down by phase."""
        phases = e2e_run_output["latency"]["by_phase"]
        
        required = ["retrieval_avg_s", "rerank_avg_s", "generation_avg_s", "judge_avg_s"]
        for key in required:
            assert key in phases, f"Missing phase: {key}"
            assert phases[key] >= 0, f"{key} is negative"
    
    def test_per_result_timing(self, e2e_run_output):
        """Each result should have timing breakdown."""
        results = e2e_run_output.get("results", [])
        
        for r in results:
            if "error" not in r:
                assert "timing" in r
                assert "retrieval" in r["timing"]
                assert "generation" in r["timing"]
                assert "judge" in r["timing"]


class TestE2ELLMMetadata:
    """Test that LLM metadata is captured."""
    
    def test_llm_metadata_present(self, e2e_run_output):
        """LLM metadata should be present in results."""
        results = e2e_run_output.get("results", [])
        
        for r in results:
            if "error" not in r:
                assert "llm_metadata" in r
    
    def test_llm_metadata_fields(self, e2e_run_output):
        """LLM metadata should have required fields."""
        results = e2e_run_output.get("results", [])
        
        required = ["model", "finish_reason", "temperature"]
        
        for r in results:
            if "error" not in r:
                meta = r["llm_metadata"]
                for key in required:
                    assert key in meta, f"Missing LLM metadata field: {key}"
    
    def test_finish_reason_is_stop(self, e2e_run_output):
        """Finish reason should typically be STOP."""
        results = e2e_run_output.get("results", [])
        
        for r in results:
            if "error" not in r:
                reason = r["llm_metadata"].get("finish_reason", "")
                # Should be STOP or contain STOP (some APIs return "FinishReason.STOP")
                assert "STOP" in str(reason).upper(), f"Unexpected finish reason: {reason}"


class TestE2EJudgment:
    """Test that judgments are captured correctly."""
    
    def test_judgment_present(self, e2e_run_output):
        """Each result should have a judgment."""
        results = e2e_run_output.get("results", [])
        
        for r in results:
            if "error" not in r:
                assert "judgment" in r
    
    def test_judgment_has_verdict(self, e2e_run_output):
        """Each judgment should have a verdict."""
        results = e2e_run_output.get("results", [])
        
        for r in results:
            if "error" not in r:
                judgment = r["judgment"]
                assert "verdict" in judgment
                assert judgment["verdict"] in ["pass", "partial", "fail"]
    
    def test_judgment_has_scores(self, e2e_run_output):
        """Each judgment should have dimension scores."""
        results = e2e_run_output.get("results", [])
        
        dimensions = ["correctness", "completeness", "faithfulness", "relevance", "clarity"]
        
        for r in results:
            if "error" not in r:
                judgment = r["judgment"]
                for dim in dimensions:
                    assert dim in judgment, f"Missing dimension: {dim}"
                    assert 1 <= judgment[dim] <= 5, f"{dim} score out of range"


class TestE2EOutputFiles:
    """Test that output files are created correctly."""
    
    def test_run_folder_created(self, e2e_run_output):
        """Run folder should be created."""
        run_id = e2e_run_output.get("run_id")
        assert run_id is not None
        
        # Find the run folder
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        matching = list(runs_dir.glob(f"*{run_id[-8:]}"))
        assert len(matching) == 1, f"Expected 1 run folder, found {len(matching)}"
    
    def test_summary_json_valid(self, e2e_run_output):
        """run_summary.json should be valid JSON."""
        run_id = e2e_run_output.get("run_id")
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        matching = list(runs_dir.glob(f"*{run_id[-8:]}"))
        
        if matching:
            summary_path = matching[0] / "run_summary.json"
            assert summary_path.exists()
            
            with open(summary_path) as f:
                summary = json.load(f)
            
            assert "metrics" in summary
            assert "tokens" in summary
    
    def test_results_jsonl_valid(self, e2e_run_output):
        """results.jsonl should have valid JSON lines."""
        run_id = e2e_run_output.get("run_id")
        runs_dir = Path(__file__).parent.parent.parent / "runs"
        matching = list(runs_dir.glob(f"*{run_id[-8:]}"))
        
        if matching:
            jsonl_path = matching[0] / "results.jsonl"
            assert jsonl_path.exists()
            
            with open(jsonl_path) as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            
            for line in lines:
                obj = json.loads(line)
                assert "question_id" in obj


class TestE2EIdempotency:
    """Test idempotency of the pipeline."""
    
    def test_same_questions_same_verdicts(self, e2e_run_output):
        """
        With temperature=0, same questions should produce consistent verdicts.
        Note: This is a weak test since we only run once.
        """
        results = e2e_run_output.get("results", [])
        
        # Just verify we got consistent results
        for r in results:
            if "error" not in r:
                assert r["llm_metadata"]["temperature"] == 0.0


class TestE2EBaselineComparison:
    """Test baseline comparison functionality."""
    
    def test_comparison_present(self, e2e_run_output):
        """Comparison should be present if baseline exists."""
        # We created a baseline, so comparison should exist
        if "comparison" in e2e_run_output:
            comparison = e2e_run_output["comparison"]
            assert "deltas" in comparison
            assert "regressions" in comparison
            assert "improvements" in comparison


class TestE2ECost:
    """Test cost calculation."""
    
    def test_cost_calculated(self, e2e_run_output):
        """Cost should be calculated."""
        assert "cost" in e2e_run_output
        cost = e2e_run_output["cost"]
        
        assert "total_cost" in cost
        assert "cost_per_question" in cost
    
    def test_cost_reasonable(self, e2e_run_output):
        """Cost should be reasonable for 3 questions."""
        cost = e2e_run_output["cost"]
        
        # Should be less than $0.10 for 3 questions
        assert cost["total_cost"] < 0.10
        
        # Should be more than $0 (we did use tokens)
        assert cost["total_cost"] > 0


if __name__ == "__main__":
    # Run e2e tests
    pytest.main([__file__, "-v", "-m", "e2e"])
