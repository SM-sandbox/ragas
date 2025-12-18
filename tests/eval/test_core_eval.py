"""
Unit tests for core_eval.py
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add scripts/eval to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "eval"))

from core_eval import (
    generate_run_id,
    get_run_folder,
    save_jsonl,
    RUNS_DIR,
)


class TestGenerateRunId:
    """Tests for generate_run_id function."""
    
    def test_returns_string(self):
        """Should return a string."""
        run_id = generate_run_id()
        assert isinstance(run_id, str)
    
    def test_starts_with_run_prefix(self):
        """Should start with 'run_'."""
        run_id = generate_run_id()
        assert run_id.startswith("run_")
    
    def test_contains_timestamp(self):
        """Should contain date/time components."""
        run_id = generate_run_id()
        # Format: run_YYYYMMDD_HHMMSS_xxxxxxxx
        parts = run_id.split("_")
        assert len(parts) >= 3
        # Date part should be 8 digits
        assert len(parts[1]) == 8
        assert parts[1].isdigit()
    
    def test_unique_ids(self):
        """Should generate unique IDs."""
        ids = [generate_run_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique
    
    def test_contains_uuid_suffix(self):
        """Should contain UUID hex suffix."""
        run_id = generate_run_id()
        # Last part should be 8 hex chars
        suffix = run_id.split("_")[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)


class TestGetRunFolder:
    """Tests for get_run_folder function."""
    
    def test_returns_path(self):
        """Should return a Path object."""
        config = {"generator_model": "gemini-2.5-flash", "precision_k": 25}
        folder = get_run_folder("run_20251218_120000_abc12345", config)
        assert isinstance(folder, Path)
    
    def test_folder_in_runs_dir(self):
        """Should be in runs directory."""
        config = {"generator_model": "gemini-2.5-flash", "precision_k": 25}
        folder = get_run_folder("run_20251218_120000_abc12345", config)
        assert folder.parent == RUNS_DIR
    
    def test_folder_contains_model_name(self):
        """Should contain model name."""
        config = {"generator_model": "gemini-2.5-flash", "precision_k": 25}
        folder = get_run_folder("run_20251218_120000_abc12345", config)
        assert "gemini-2.5-flash" in folder.name
    
    def test_folder_contains_precision(self):
        """Should contain precision setting."""
        config = {"generator_model": "gemini-2.5-flash", "precision_k": 12}
        folder = get_run_folder("run_20251218_120000_abc12345", config)
        assert "p12" in folder.name
    
    def test_folder_contains_run_id_suffix(self):
        """Should contain run ID suffix."""
        config = {"generator_model": "gemini-2.5-flash", "precision_k": 25}
        folder = get_run_folder("run_20251218_120000_abc12345", config)
        assert "abc12345" in folder.name
    
    def test_handles_model_with_slash(self):
        """Should handle model names with slashes."""
        config = {"generator_model": "models/gemini-2.5-flash", "precision_k": 25}
        folder = get_run_folder("run_20251218_120000_abc12345", config)
        assert "/" not in folder.name  # Slashes replaced
    
    def test_handles_missing_config_keys(self):
        """Should handle missing config keys with defaults."""
        folder = get_run_folder("run_20251218_120000_abc12345", {})
        assert "unknown" in folder.name
        assert "p25" in folder.name


class TestSaveJsonl:
    """Tests for save_jsonl function."""
    
    def test_creates_file(self, tmp_path):
        """Should create JSONL file."""
        results = [{"id": 1}, {"id": 2}]
        path = tmp_path / "test.jsonl"
        save_jsonl(results, path)
        assert path.exists()
    
    def test_one_json_per_line(self, tmp_path):
        """Should write one JSON object per line."""
        results = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 3, "name": "c"}]
        path = tmp_path / "test.jsonl"
        save_jsonl(results, path)
        
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
    
    def test_each_line_valid_json(self, tmp_path):
        """Each line should be valid JSON."""
        results = [
            {"question_id": "q1", "verdict": "pass"},
            {"question_id": "q2", "verdict": "fail"},
        ]
        path = tmp_path / "test.jsonl"
        save_jsonl(results, path)
        
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                assert "question_id" in obj
    
    def test_empty_list(self, tmp_path):
        """Should handle empty list."""
        path = tmp_path / "empty.jsonl"
        save_jsonl([], path)
        assert path.exists()
        assert path.read_text() == ""
    
    def test_preserves_data_types(self, tmp_path):
        """Should preserve data types."""
        results = [{
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"a": 1},
        }]
        path = tmp_path / "types.jsonl"
        save_jsonl(results, path)
        
        loaded = json.loads(path.read_text().strip())
        assert loaded["int"] == 42
        assert loaded["float"] == 3.14
        assert loaded["bool"] is True
        assert loaded["null"] is None
        assert loaded["list"] == [1, 2, 3]
        assert loaded["nested"] == {"a": 1}
    
    def test_roundtrip(self, tmp_path):
        """Saved data should match when loaded."""
        original = [
            {"question_id": "sh_easy_001", "verdict": "pass", "score": 4.5},
            {"question_id": "sh_easy_002", "verdict": "partial", "score": 3.0},
        ]
        path = tmp_path / "roundtrip.jsonl"
        save_jsonl(original, path)
        
        loaded = []
        with open(path) as f:
            for line in f:
                loaded.append(json.loads(line))
        
        assert loaded == original


class TestIdempotency:
    """Idempotency tests for core_eval functions."""
    
    def test_get_run_folder_idempotent(self):
        """Same inputs should produce same folder path."""
        config = {"generator_model": "gemini-2.5-flash", "precision_k": 25}
        run_id = "run_20251218_120000_abc12345"
        
        result1 = get_run_folder(run_id, config)
        result2 = get_run_folder(run_id, config)
        assert result1 == result2
    
    def test_save_jsonl_idempotent(self, tmp_path):
        """Same data should produce identical files."""
        results = [{"id": 1}, {"id": 2}]
        
        path1 = tmp_path / "test1.jsonl"
        path2 = tmp_path / "test2.jsonl"
        
        save_jsonl(results, path1)
        save_jsonl(results, path2)
        
        assert path1.read_text() == path2.read_text()


class TestEdgeCases:
    """Edge case tests."""
    
    def test_save_jsonl_special_characters(self, tmp_path):
        """Should handle special characters in data."""
        results = [{
            "text": "Hello\nWorld\t\"quoted\"",
            "unicode": "日本語 🎉",
            "backslash": "path\\to\\file",
        }]
        path = tmp_path / "special.jsonl"
        save_jsonl(results, path)
        
        loaded = json.loads(path.read_text().strip())
        assert loaded["text"] == "Hello\nWorld\t\"quoted\""
        assert loaded["unicode"] == "日本語 🎉"
    
    def test_save_jsonl_large_data(self, tmp_path):
        """Should handle large datasets."""
        results = [{"id": i, "data": "x" * 1000} for i in range(1000)]
        path = tmp_path / "large.jsonl"
        save_jsonl(results, path)
        
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
