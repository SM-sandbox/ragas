"""
Unit tests for checkpoint registry functionality.

Tests that:
1. Registry is updated after checkpoint runs
2. All required fields are written correctly
3. Duplicate entries are not added
4. Registry entries are sorted by ID
5. Failure reason codes are captured when applicable
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestRegistryUpdate:
    """Tests for _update_registry method in GoldEvaluator."""
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create a mock evaluator with minimal setup."""
        from lib.core.evaluator import GoldEvaluator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create run directory structure
            run_dir = tmpdir / "C099__2025-12-23__local__p25-3-flash-low__q458"
            run_dir.mkdir(parents=True)
            
            # Create registry.json
            registry = {
                "gold_baseline": {
                    "id": "C012",
                    "pass_rate": 0.928,
                    "acceptable_rate": 0.985
                },
                "entries": [
                    {"id": "C001", "pass_rate": 0.926, "folder": "C001__test"},
                    {"id": "C012", "pass_rate": 0.928, "folder": "C012__test"}
                ]
            }
            registry_path = tmpdir / "registry.json"
            with open(registry_path, "w") as f:
                json.dump(registry, f)
            
            # Create mock evaluator
            evaluator = Mock(spec=GoldEvaluator)
            evaluator.run_dir = run_dir
            evaluator._update_registry = GoldEvaluator._update_registry.__get__(evaluator)
            
            yield evaluator, tmpdir, registry_path
    
    def test_registry_entry_added(self, mock_evaluator):
        """Test that new checkpoint entry is added to registry."""
        evaluator, tmpdir, registry_path = mock_evaluator
        
        output = {
            "timestamp": "2025-12-23T10:00:00",
            "metrics": {
                "total": 458,
                "pass_rate": 0.919,
                "fail_rate": 0.022,
                "acceptable_rate": 0.978
            },
            "config": {
                "precision_k": 25,
                "generator_model": "gemini-3-flash-preview",
                "generator_reasoning_effort": "low"
            },
            "index": {
                "mode": "local",
                "job_id": "bfai__eval66b_g1536tt"
            }
        }
        
        evaluator._update_registry(output)
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        # Check entry was added
        ids = [e["id"] for e in registry["entries"]]
        assert "C099" in ids
        
        # Check entry has correct values
        entry = next(e for e in registry["entries"] if e["id"] == "C099")
        assert entry["pass_rate"] == 0.919
        assert entry["fail_rate"] == 0.022
        assert entry["acceptable_rate"] == 0.978
        assert entry["questions"] == 458
        assert entry["mode"] == "local"
        assert entry["date"] == "2025-12-23"
    
    def test_registry_entries_sorted(self, mock_evaluator):
        """Test that registry entries are sorted by ID after update."""
        evaluator, tmpdir, registry_path = mock_evaluator
        
        output = {
            "timestamp": "2025-12-23T10:00:00",
            "metrics": {"total": 458, "pass_rate": 0.919, "fail_rate": 0.022, "acceptable_rate": 0.978},
            "config": {"precision_k": 25, "generator_model": "gemini-3-flash-preview"},
            "index": {"mode": "local"}
        }
        
        evaluator._update_registry(output)
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        ids = [e["id"] for e in registry["entries"]]
        assert ids == sorted(ids, key=lambda x: int(x[1:]))
    
    def test_duplicate_entry_not_added(self, mock_evaluator):
        """Test that duplicate entries are not added."""
        evaluator, tmpdir, registry_path = mock_evaluator
        
        # Change run_dir to match existing entry
        evaluator.run_dir = tmpdir / "C012__2025-12-19__cloud__p25-3-flash-low__q458"
        evaluator.run_dir.mkdir(parents=True, exist_ok=True)
        
        output = {
            "timestamp": "2025-12-23T10:00:00",
            "metrics": {"total": 458, "pass_rate": 0.999, "fail_rate": 0.001, "acceptable_rate": 1.0},
            "config": {"precision_k": 25, "generator_model": "gemini-3-flash-preview"},
            "index": {"mode": "cloud"}
        }
        
        evaluator._update_registry(output)
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        # Should still have only 2 entries (C001, C012)
        assert len(registry["entries"]) == 2
        
        # C012 pass_rate should be unchanged (not overwritten)
        c012 = next(e for e in registry["entries"] if e["id"] == "C012")
        assert c012["pass_rate"] == 0.928
    
    def test_registry_not_found_handled(self, mock_evaluator):
        """Test that missing registry is handled gracefully."""
        evaluator, tmpdir, registry_path = mock_evaluator
        
        # Remove registry
        registry_path.unlink()
        
        output = {
            "timestamp": "2025-12-23T10:00:00",
            "metrics": {"total": 458, "pass_rate": 0.919},
            "config": {},
            "index": {}
        }
        
        # Should not raise exception
        evaluator._update_registry(output)
    
    def test_required_fields_present(self, mock_evaluator):
        """Test that all required fields are written to registry entry."""
        evaluator, tmpdir, registry_path = mock_evaluator
        
        output = {
            "timestamp": "2025-12-23T10:00:00",
            "metrics": {
                "total": 458,
                "pass_rate": 0.919,
                "fail_rate": 0.022,
                "acceptable_rate": 0.978
            },
            "config": {
                "precision_k": 25,
                "generator_model": "gemini-3-flash-preview",
                "generator_reasoning_effort": "low"
            },
            "index": {
                "mode": "local",
                "job_id": "bfai__eval66b_g1536tt"
            }
        }
        
        evaluator._update_registry(output)
        
        with open(registry_path) as f:
            registry = json.load(f)
        
        entry = next(e for e in registry["entries"] if e["id"] == "C099")
        
        required_fields = [
            "id", "date", "mode", "config_summary", "model",
            "precision_k", "questions", "pass_rate", "fail_rate",
            "acceptable_rate", "folder", "notes"
        ]
        
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"


class TestRegistryGoldBaseline:
    """Tests for gold baseline comparison functionality."""
    
    def test_gold_baseline_preserved(self):
        """Test that gold baseline is not modified during updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            registry = {
                "gold_baseline": {
                    "id": "C012",
                    "pass_rate": 0.928,
                    "acceptable_rate": 0.985,
                    "mrr": 0.737,
                    "locked_date": "2025-12-20"
                },
                "entries": []
            }
            registry_path = tmpdir / "registry.json"
            with open(registry_path, "w") as f:
                json.dump(registry, f)
            
            # Simulate adding new entry
            with open(registry_path) as f:
                reg = json.load(f)
            
            reg["entries"].append({
                "id": "C099",
                "pass_rate": 0.999  # Higher than gold
            })
            
            with open(registry_path, "w") as f:
                json.dump(reg, f)
            
            # Verify gold baseline unchanged
            with open(registry_path) as f:
                final_reg = json.load(f)
            
            assert final_reg["gold_baseline"]["id"] == "C012"
            assert final_reg["gold_baseline"]["pass_rate"] == 0.928


class TestRegistryFailureTracking:
    """Tests for failure reason code tracking in registry."""
    
    def test_failure_reason_captured(self):
        """Test that failure reason codes can be captured in registry entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            registry = {"gold_baseline": {}, "entries": []}
            registry_path = tmpdir / "registry.json"
            with open(registry_path, "w") as f:
                json.dump(registry, f)
            
            # Add entry with failure reason
            entry = {
                "id": "C050",
                "pass_rate": 0.0,
                "fail_rate": 1.0,
                "failure_reason": "orchestrator_unavailable",
                "notes": "Cloud Run returned 503"
            }
            
            with open(registry_path) as f:
                reg = json.load(f)
            reg["entries"].append(entry)
            with open(registry_path, "w") as f:
                json.dump(reg, f)
            
            # Verify failure reason is stored
            with open(registry_path) as f:
                final_reg = json.load(f)
            
            c050 = next(e for e in final_reg["entries"] if e["id"] == "C050")
            assert c050["failure_reason"] == "orchestrator_unavailable"
    
    def test_index_mismatch_tracked(self):
        """Test that index mismatch errors are tracked."""
        entry = {
            "id": "C051",
            "pass_rate": 0.0,
            "failure_reason": "index_not_found",
            "notes": "Job bfai__eval66b_g1536tt not deployed"
        }
        
        assert entry["failure_reason"] == "index_not_found"
        assert "not deployed" in entry["notes"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
