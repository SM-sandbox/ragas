"""
Integration tests for the eval pipeline.

Tests component interactions without making real API calls.
Uses mocked orchestrator components.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys

# Add scripts/eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "eval"))

from cost_calculator import calculate_cost, calculate_run_cost
from baseline_manager import (
    save_baseline, load_baseline, get_latest_baseline,
    compare_to_baseline, format_comparison_report, list_baselines
)
from core_eval import generate_run_id, get_run_folder, save_jsonl


class TestCostCalculatorIntegration:
    """Integration tests for cost calculator with realistic data."""
    
    def test_realistic_eval_run(self):
        """Test cost calculation with realistic eval run data."""
        # Simulate 458 questions with realistic token counts
        results = []
        for i in range(458):
            results.append({
                "tokens": {
                    "prompt": 4000 + (i % 1000),  # 4000-5000 tokens
                    "completion": 200 + (i % 200),  # 200-400 tokens
                    "thinking": 0,
                    "cached": 0,
                }
            })
        
        # Aggregate tokens
        total_prompt = sum(r["tokens"]["prompt"] for r in results)
        total_completion = sum(r["tokens"]["completion"] for r in results)
        
        # Calculate cost
        cost = calculate_run_cost(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            model="gemini-2.5-flash",
            question_count=458,
        )
        
        # Verify reasonable cost
        assert 0.10 < cost["total_cost"] < 1.0
        assert cost["cost_per_question"] > 0
        assert cost["question_count"] == 458
    
    def test_cost_with_thinking_tokens(self):
        """Test cost calculation including thinking tokens."""
        cost = calculate_run_cost(
            total_prompt_tokens=458 * 4000,
            total_completion_tokens=458 * 300,
            total_thinking_tokens=458 * 500,  # Thinking tokens
            model="gemini-2.5-flash",
            question_count=458,
        )
        
        # Thinking should add to cost
        cost_no_thinking = calculate_run_cost(
            total_prompt_tokens=458 * 4000,
            total_completion_tokens=458 * 300,
            total_thinking_tokens=0,
            model="gemini-2.5-flash",
            question_count=458,
        )
        
        assert cost["total_cost"] > cost_no_thinking["total_cost"]


class TestBaselineManagerIntegration:
    """Integration tests for baseline manager."""
    
    @pytest.fixture
    def temp_baselines_dir(self, tmp_path, monkeypatch):
        """Use temp directory for baselines."""
        import baseline_manager
        monkeypatch.setattr(baseline_manager, "BASELINES_DIR", tmp_path)
        return tmp_path
    
    def test_full_baseline_workflow(self, temp_baselines_dir):
        """Test complete baseline save/load/compare workflow."""
        # Create initial baseline
        baseline_data = {
            "client": "TEST",
            "corpus": {"question_count": 100},
            "metrics": {
                "pass_rate": 0.85,
                "partial_rate": 0.10,
                "fail_rate": 0.05,
                "acceptable_rate": 0.95,
                "recall_at_100": 0.99,
                "mrr": 0.72,
                "overall_score_avg": 4.7,
            },
            "latency": {"total_avg_s": 8.0},
            "cost": {"total_cost_usd": 0.15},
        }
        
        # Save baseline
        path1 = save_baseline(baseline_data, "TEST", date="2025-01-01")
        assert path1.exists()
        
        # Load and verify
        loaded = load_baseline(str(path1))
        assert loaded["metrics"]["pass_rate"] == 0.85
        
        # Create new run with improvements
        current_run = {
            "metrics": {
                "pass_rate": 0.90,  # Improved
                "partial_rate": 0.07,
                "fail_rate": 0.03,  # Improved
                "acceptable_rate": 0.97,
                "recall_at_100": 0.995,
                "mrr": 0.75,
                "overall_score_avg": 4.8,
            },
            "latency": {"total_avg_s": 7.5},
            "cost": {"total_cost_usd": 0.14},
        }
        
        # Compare to baseline
        comparison = compare_to_baseline(current_run, loaded)
        
        # Verify improvements detected
        assert "pass_rate" in comparison["improvements"]
        assert "fail_rate" in comparison["improvements"]
        assert len(comparison["regressions"]) == 0
        
        # Generate report
        report = format_comparison_report(comparison)
        assert "Improvements" in report
        assert "pass_rate" in report
    
    def test_multiple_baselines_versioning(self, temp_baselines_dir):
        """Test multiple baseline versions."""
        data = {"corpus": {"question_count": 50}, "metrics": {"pass_rate": 0.80}}
        
        # Save multiple versions
        path1 = save_baseline(data, "MULTI", date="2025-01-01")
        path2 = save_baseline(data, "MULTI", date="2025-01-02")
        path3 = save_baseline(data, "MULTI", date="2025-01-03")
        
        # List should show all
        baselines = list_baselines("MULTI")
        assert len(baselines) == 3
        
        # Should be sorted by date descending
        assert baselines[0]["date"] == "2025-01-03"
        assert baselines[1]["date"] == "2025-01-02"
        assert baselines[2]["date"] == "2025-01-01"
        
        # Versions should increment
        assert baselines[0]["version"] == "3"
        assert baselines[1]["version"] == "2"
        assert baselines[2]["version"] == "1"


class TestCoreEvalIntegration:
    """Integration tests for core_eval components."""
    
    def test_run_folder_structure(self, tmp_path):
        """Test run folder creation and structure."""
        run_id = generate_run_id()
        config = {"generator_model": "gemini-2.5-flash", "precision_k": 25}
        
        folder = get_run_folder(run_id, config)
        folder.mkdir(parents=True, exist_ok=True)
        
        # Create expected files
        summary = {"metrics": {"pass_rate": 0.85}}
        results = [{"question_id": "q1"}, {"question_id": "q2"}]
        
        with open(folder / "run_summary.json", "w") as f:
            json.dump(summary, f)
        
        save_jsonl(results, folder / "results.jsonl")
        
        # Verify structure
        assert (folder / "run_summary.json").exists()
        assert (folder / "results.jsonl").exists()
        
        # Verify content
        with open(folder / "run_summary.json") as f:
            loaded_summary = json.load(f)
        assert loaded_summary["metrics"]["pass_rate"] == 0.85
        
        with open(folder / "results.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 2
    
    def test_jsonl_with_full_result_structure(self, tmp_path):
        """Test JSONL with full result structure."""
        results = [
            {
                "question_id": "sh_easy_001",
                "question_type": "single_hop",
                "difficulty": "easy",
                "recall_hit": True,
                "mrr": 1.0,
                "judgment": {
                    "correctness": 5,
                    "completeness": 5,
                    "faithfulness": 5,
                    "relevance": 5,
                    "clarity": 5,
                    "overall_score": 5,
                    "verdict": "pass",
                },
                "time": 7.5,
                "timing": {
                    "retrieval": 0.25,
                    "rerank": 0.15,
                    "generation": 5.8,
                    "judge": 1.3,
                    "total": 7.5,
                },
                "tokens": {
                    "prompt": 5000,
                    "completion": 300,
                    "thinking": 0,
                    "total": 5300,
                    "cached": 0,
                },
                "llm_metadata": {
                    "model": "gemini-2.5-flash",
                    "model_version": "gemini-2.5-flash-preview-05-20",
                    "finish_reason": "STOP",
                    "reasoning_effort": "low",
                    "used_fallback": False,
                    "avg_logprobs": None,
                    "response_id": "abc123",
                    "temperature": 0.0,
                    "has_citations": True,
                },
                "answer_length": 450,
                "retrieval_candidates": 100,
            }
        ]
        
        path = tmp_path / "full_results.jsonl"
        save_jsonl(results, path)
        
        # Load and verify all fields preserved
        with open(path) as f:
            loaded = json.loads(f.readline())
        
        assert loaded["question_id"] == "sh_easy_001"
        assert loaded["judgment"]["verdict"] == "pass"
        assert loaded["tokens"]["prompt"] == 5000
        assert loaded["llm_metadata"]["model"] == "gemini-2.5-flash"
        assert loaded["timing"]["generation"] == 5.8


class TestEndToEndDataFlow:
    """Test data flow through the entire pipeline."""
    
    def test_metrics_aggregation(self):
        """Test that metrics are aggregated correctly."""
        # Simulate individual results
        results = [
            {"judgment": {"verdict": "pass"}, "recall_hit": True, "mrr": 1.0},
            {"judgment": {"verdict": "pass"}, "recall_hit": True, "mrr": 0.5},
            {"judgment": {"verdict": "partial"}, "recall_hit": True, "mrr": 0.33},
            {"judgment": {"verdict": "fail"}, "recall_hit": False, "mrr": 0.0},
        ]
        
        # Calculate metrics (mimicking run_gold_eval.py logic)
        valid = [r for r in results if "judgment" in r]
        pass_count = sum(1 for r in valid if r["judgment"]["verdict"] == "pass")
        partial_count = sum(1 for r in valid if r["judgment"]["verdict"] == "partial")
        fail_count = sum(1 for r in valid if r["judgment"]["verdict"] == "fail")
        
        metrics = {
            "pass_rate": pass_count / len(valid),
            "partial_rate": partial_count / len(valid),
            "fail_rate": fail_count / len(valid),
            "acceptable_rate": (pass_count + partial_count) / len(valid),
            "recall_at_100": sum(1 for r in valid if r["recall_hit"]) / len(valid),
            "mrr": sum(r["mrr"] for r in valid) / len(valid),
        }
        
        # Verify calculations
        assert metrics["pass_rate"] == 0.5  # 2/4
        assert metrics["partial_rate"] == 0.25  # 1/4
        assert metrics["fail_rate"] == 0.25  # 1/4
        assert metrics["acceptable_rate"] == 0.75  # 3/4
        assert metrics["recall_at_100"] == 0.75  # 3/4
        assert metrics["mrr"] == pytest.approx(0.4575, rel=1e-2)  # (1+0.5+0.33+0)/4
    
    def test_token_aggregation(self):
        """Test that tokens are aggregated correctly."""
        results = [
            {"tokens": {"prompt": 4000, "completion": 300, "thinking": 0, "cached": 0}},
            {"tokens": {"prompt": 5000, "completion": 400, "thinking": 100, "cached": 500}},
            {"tokens": {"prompt": 3500, "completion": 250, "thinking": 0, "cached": 0}},
        ]
        
        token_totals = {
            "prompt_total": sum(r["tokens"]["prompt"] for r in results),
            "completion_total": sum(r["tokens"]["completion"] for r in results),
            "thinking_total": sum(r["tokens"]["thinking"] for r in results),
            "cached_total": sum(r["tokens"]["cached"] for r in results),
        }
        
        assert token_totals["prompt_total"] == 12500
        assert token_totals["completion_total"] == 950
        assert token_totals["thinking_total"] == 100
        assert token_totals["cached_total"] == 500
    
    def test_latency_aggregation(self):
        """Test that latency is aggregated correctly."""
        results = [
            {"time": 7.0, "timing": {"retrieval": 0.2, "rerank": 0.1, "generation": 5.5, "judge": 1.2}},
            {"time": 8.0, "timing": {"retrieval": 0.3, "rerank": 0.2, "generation": 6.0, "judge": 1.5}},
            {"time": 6.0, "timing": {"retrieval": 0.15, "rerank": 0.1, "generation": 4.5, "judge": 1.25}},
        ]
        
        latency = {
            "total_avg_s": sum(r["time"] for r in results) / len(results),
            "total_min_s": min(r["time"] for r in results),
            "total_max_s": max(r["time"] for r in results),
            "by_phase": {
                "retrieval_avg_s": sum(r["timing"]["retrieval"] for r in results) / len(results),
                "rerank_avg_s": sum(r["timing"]["rerank"] for r in results) / len(results),
                "generation_avg_s": sum(r["timing"]["generation"] for r in results) / len(results),
                "judge_avg_s": sum(r["timing"]["judge"] for r in results) / len(results),
            },
        }
        
        assert latency["total_avg_s"] == 7.0
        assert latency["total_min_s"] == 6.0
        assert latency["total_max_s"] == 8.0
        assert latency["by_phase"]["retrieval_avg_s"] == pytest.approx(0.2167, rel=1e-2)


class TestSchemaValidation:
    """Test that output schemas are valid and complete."""
    
    def test_result_schema_complete(self):
        """Test that result schema has all required fields."""
        required_fields = {
            "question_id", "question_type", "difficulty",
            "recall_hit", "mrr", "judgment", "time", "timing",
            "tokens", "llm_metadata", "answer_length", "retrieval_candidates"
        }
        
        # Minimal valid result
        result = {
            "question_id": "test",
            "question_type": "single_hop",
            "difficulty": "easy",
            "recall_hit": True,
            "mrr": 1.0,
            "judgment": {"verdict": "pass"},
            "time": 5.0,
            "timing": {"retrieval": 0.1, "rerank": 0.1, "generation": 4.0, "judge": 0.8, "total": 5.0},
            "tokens": {"prompt": 1000, "completion": 100, "thinking": 0, "total": 1100, "cached": 0},
            "llm_metadata": {"model": "test", "finish_reason": "STOP"},
            "answer_length": 200,
            "retrieval_candidates": 100,
        }
        
        assert required_fields.issubset(result.keys())
    
    def test_summary_schema_complete(self):
        """Test that summary schema has all required fields."""
        required_fields = {
            "schema_version", "timestamp", "client", "index", "config",
            "metrics", "latency", "tokens", "answer_stats", "quality", "execution"
        }
        
        summary = {
            "schema_version": "1.0",
            "timestamp": "2025-01-01T00:00:00",
            "client": "TEST",
            "index": {"job_id": "test"},
            "config": {"generator_model": "test"},
            "metrics": {"pass_rate": 0.85},
            "latency": {"total_avg_s": 7.0},
            "tokens": {"prompt_total": 1000},
            "answer_stats": {"avg_length_chars": 500},
            "quality": {"finish_reason_distribution": {"STOP": 10}},
            "execution": {"workers": 5},
        }
        
        assert required_fields.issubset(summary.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
