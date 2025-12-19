"""
Unit tests for evaluation output schemas.

Tests schema validation, versioning, and data integrity.
"""

import pytest
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.schemas.eval_output_v2 import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_MAJOR,
    SCHEMA_VERSION_MINOR,
    validate_eval_output,
    get_empty_question_result,
    EvalOutput,
    EnvironmentInfo,
    GeneratorConfig,
    JudgeConfig,
    RetrievalConfig,
    CorpusInfo,
    ExecutionInfo,
    QuestionResult,
    MetricsAggregate,
    LatencyAggregate,
    TokenAggregate,
    AnswerStats,
    QualityInfo,
    RetryStats,
    ErrorStats,
    SkippedStats,
    BreakdownStats,
)
from lib.core.metadata import (
    get_git_info,
    get_system_info,
    get_file_hash,
    get_utc_timestamp,
    build_environment_info,
    build_generator_config,
    build_judge_config,
    build_retrieval_config,
    build_corpus_info,
    build_execution_info,
    get_judge_prompt_hash,
)


class TestSchemaVersion:
    """Tests for schema versioning."""
    
    def test_schema_version_format(self):
        """Schema version should be in X.Y format."""
        assert "." in SCHEMA_VERSION
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)
    
    def test_schema_version_major(self):
        """Major version should match."""
        assert SCHEMA_VERSION.startswith(f"{SCHEMA_VERSION_MAJOR}.")
    
    def test_schema_version_minor(self):
        """Minor version should match."""
        assert SCHEMA_VERSION == f"{SCHEMA_VERSION_MAJOR}.{SCHEMA_VERSION_MINOR}"
    
    def test_current_version_is_2(self):
        """Current schema version should be 2.x."""
        assert SCHEMA_VERSION_MAJOR == 2


class TestValidateEvalOutput:
    """Tests for validate_eval_output function."""
    
    @pytest.fixture
    def minimal_valid_output(self):
        """Minimal valid evaluation output."""
        return {
            "schema_version": "2.0",
            "run_id": "test_run_001",
            "run_type": "experiment",
            "client": "BFAI",
            "timestamp": "2025-12-19T14:30:00Z",
            "environment": {
                "git_commit": "abc123",
                "git_branch": "main",
                "git_dirty": False,
                "config_file": "experiment_config.yaml",
                "config_hash": "def456",
                "hostname": "test-host",
                "user": "testuser",
                "os_name": "Darwin",
                "os_version": "23.1.0",
                "python_version": "3.11.5",
                "timezone": "UTC",
                "start_time_utc": "2025-12-19T14:30:00Z",
            },
            "generator_config": {
                "model": "gemini-2.5-flash",
                "temperature": 0.0,
            },
            "judge_config": {
                "model": "gemini-2.0-flash",
            },
            "retrieval_config": {
                "index_job_id": "test_job",
                "embedding_model": "gemini-embedding-001",
                "embedding_dimension": 1536,
                "recall_k": 100,
                "precision_k": 25,
            },
            "corpus": {
                "file_path": "test/corpus.json",
                "file_hash": "ghi789",
                "question_count": 100,
            },
            "execution": {
                "mode": "local",
                "run_type": "experiment",
                "workers": 5,
            },
            "metrics": {
                "precision_k": 25,
                "total": 100,
                "completed": 100,
                "recall_at_100": 0.95,
                "mrr": 0.75,
                "pass_rate": 0.85,
                "partial_rate": 0.10,
                "fail_rate": 0.05,
            },
            "latency": {
                "total_avg_s": 8.5,
                "total_min_s": 2.0,
                "total_max_s": 30.0,
                "by_phase": {},
            },
            "tokens": {
                "prompt_total": 100000,
                "completion_total": 20000,
                "thinking_total": 5000,
                "cached_total": 0,
                "total": 125000,
            },
            "answer_stats": {
                "avg_length_chars": 1500,
                "min_length_chars": 100,
                "max_length_chars": 5000,
            },
            "quality": {
                "finish_reason_distribution": {"STOP": 100},
                "fallback_rate": 0.0,
            },
            "retry_stats": {
                "total_questions": 100,
                "succeeded_first_try": 98,
                "succeeded_after_retry": 2,
                "failed_all_retries": 0,
                "total_retry_attempts": 102,
                "avg_attempts": 1.02,
            },
            "errors": {
                "total_errors": 0,
                "by_phase": {},
                "error_messages": [],
            },
            "skipped": {
                "count": 0,
                "reasons": {},
                "question_ids": [],
            },
            "breakdown_by_type": {},
            "breakdown_by_difficulty": {},
            "results": [],
        }
    
    def test_valid_output_passes(self, minimal_valid_output):
        """Valid output should pass validation."""
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert is_valid, f"Validation failed: {errors}"
        assert len(errors) == 0
    
    def test_missing_schema_version_fails(self, minimal_valid_output):
        """Missing schema_version should fail."""
        del minimal_valid_output["schema_version"]
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert not is_valid
        assert any("schema_version" in e for e in errors)
    
    def test_wrong_schema_version_fails(self, minimal_valid_output):
        """Wrong schema version should fail."""
        minimal_valid_output["schema_version"] = "1.0"
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert not is_valid
        assert any("Schema version mismatch" in e for e in errors)
    
    def test_invalid_run_type_fails(self, minimal_valid_output):
        """Invalid run_type should fail."""
        minimal_valid_output["run_type"] = "invalid"
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert not is_valid
        assert any("run_type" in e for e in errors)
    
    def test_valid_run_types(self, minimal_valid_output):
        """All valid run_types should pass."""
        for run_type in ["checkpoint", "run", "experiment"]:
            minimal_valid_output["run_type"] = run_type
            minimal_valid_output["execution"]["run_type"] = run_type
            is_valid, errors = validate_eval_output(minimal_valid_output)
            assert is_valid, f"run_type '{run_type}' failed: {errors}"
    
    def test_missing_environment_fields_fails(self, minimal_valid_output):
        """Missing required environment fields should fail."""
        del minimal_valid_output["environment"]["git_commit"]
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert not is_valid
        assert any("git_commit" in e for e in errors)
    
    def test_missing_generator_model_fails(self, minimal_valid_output):
        """Missing generator model should fail."""
        del minimal_valid_output["generator_config"]["model"]
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert not is_valid
        assert any("generator_config" in e and "model" in e for e in errors)
    
    def test_missing_metrics_fields_fails(self, minimal_valid_output):
        """Missing required metrics fields should fail."""
        del minimal_valid_output["metrics"]["pass_rate"]
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert not is_valid
        assert any("pass_rate" in e for e in errors)
    
    def test_invalid_execution_mode_fails(self, minimal_valid_output):
        """Invalid execution mode should fail."""
        minimal_valid_output["execution"]["mode"] = "hybrid"
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert not is_valid
        assert any("mode" in e for e in errors)
    
    def test_results_with_question(self, minimal_valid_output):
        """Results with valid question should pass."""
        minimal_valid_output["results"] = [{
            "question_id": "Q001",
            "question_type": "single_hop",
            "difficulty": "easy",
        }]
        is_valid, errors = validate_eval_output(minimal_valid_output)
        assert is_valid, f"Validation failed: {errors}"


class TestGetEmptyQuestionResult:
    """Tests for get_empty_question_result helper."""
    
    def test_returns_dict(self):
        """Should return a dictionary."""
        result = get_empty_question_result("Q001")
        assert isinstance(result, dict)
    
    def test_has_question_id(self):
        """Should have the provided question_id."""
        result = get_empty_question_result("Q123")
        assert result["question_id"] == "Q123"
    
    def test_has_all_required_fields(self):
        """Should have all required fields."""
        result = get_empty_question_result("Q001")
        required_fields = [
            "question_id", "question_type", "difficulty",
            "question_text", "expected_answer", "source_documents",
            "generated_answer", "retrieved_doc_names", "context_char_count",
            "recall_hit", "mrr", "judgment", "timing", "tokens",
            "llm_metadata", "request_timestamp_utc", "answer_length",
            "retrieval_candidates", "retry_info", "error", "error_phase"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
    
    def test_judgment_has_all_fields(self):
        """Judgment should have all score fields."""
        result = get_empty_question_result("Q001")
        judgment = result["judgment"]
        for field in ["correctness", "completeness", "faithfulness", 
                      "relevance", "clarity", "overall_score", "verdict"]:
            assert field in judgment, f"Missing judgment field: {field}"
    
    def test_timing_has_all_phases(self):
        """Timing should have all phase fields."""
        result = get_empty_question_result("Q001")
        timing = result["timing"]
        for field in ["retrieval_s", "rerank_s", "generation_s", "judge_s", "total_s"]:
            assert field in timing, f"Missing timing field: {field}"


class TestMetadataHelpers:
    """Tests for metadata helper functions."""
    
    def test_get_git_info_returns_dict(self):
        """get_git_info should return a dictionary."""
        info = get_git_info()
        assert isinstance(info, dict)
        assert "git_commit" in info
        assert "git_branch" in info
        assert "git_dirty" in info
    
    def test_get_system_info_returns_dict(self):
        """get_system_info should return a dictionary."""
        info = get_system_info()
        assert isinstance(info, dict)
        assert "hostname" in info
        assert "python_version" in info
        assert "os_name" in info
    
    def test_get_file_hash_nonexistent(self):
        """get_file_hash should return 'not_found' for missing files."""
        result = get_file_hash(Path("/nonexistent/file.txt"))
        assert result == "not_found"
    
    def test_get_file_hash_existing(self):
        """get_file_hash should return hash for existing files."""
        # Use this test file itself
        result = get_file_hash(Path(__file__))
        assert result != "not_found"
        assert result != "error"
        assert len(result) == 16  # First 16 chars of SHA256
    
    def test_get_utc_timestamp_format(self):
        """get_utc_timestamp should return ISO format with Z suffix."""
        ts = get_utc_timestamp()
        assert ts.endswith("Z")
        assert "T" in ts
        # Should be parseable
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    
    def test_build_environment_info(self):
        """build_environment_info should return complete dict."""
        info = build_environment_info(
            config_file="test_config.yaml",
            config_path=Path(__file__),  # Use this file as dummy
        )
        assert info["config_file"] == "test_config.yaml"
        assert "git_commit" in info
        assert "hostname" in info
        assert "start_time_utc" in info
    
    def test_build_generator_config(self):
        """build_generator_config should return complete dict."""
        config = build_generator_config(
            model="gemini-2.5-flash",
            temperature=0.0,
            reasoning_effort="low",
        )
        assert config["model"] == "gemini-2.5-flash"
        assert config["temperature"] == 0.0
        assert config["reasoning_effort"] == "low"
        assert "max_output_tokens" in config
    
    def test_build_judge_config(self):
        """build_judge_config should return complete dict."""
        config = build_judge_config(
            model="gemini-2.0-flash",
            temperature=0.0,
        )
        assert config["model"] == "gemini-2.0-flash"
        assert config["temperature"] == 0.0
        assert "prompt_template_version" in config
    
    def test_build_retrieval_config(self):
        """build_retrieval_config should return complete dict."""
        config = build_retrieval_config(
            index_job_id="test_job",
            embedding_model="gemini-embedding-001",
            embedding_dimension=1536,
        )
        assert config["index_job_id"] == "test_job"
        assert config["embedding_model"] == "gemini-embedding-001"
        assert config["embedding_dimension"] == 1536
        assert "recall_k" in config
        assert "precision_k" in config
    
    def test_build_corpus_info(self):
        """build_corpus_info should return complete dict."""
        info = build_corpus_info(
            file_path="test/corpus.json",
            question_count=100,
        )
        assert info["file_path"] == "test/corpus.json"
        assert info["question_count"] == 100
        assert "file_hash" in info
    
    def test_build_execution_info(self):
        """build_execution_info should return complete dict."""
        info = build_execution_info(
            mode="local",
            run_type="experiment",
            workers=5,
        )
        assert info["mode"] == "local"
        assert info["run_type"] == "experiment"
        assert info["workers"] == 5
        assert "timeout_per_question_s" in info
    
    def test_get_judge_prompt_hash_consistent(self):
        """get_judge_prompt_hash should return consistent hash."""
        hash1 = get_judge_prompt_hash()
        hash2 = get_judge_prompt_hash()
        assert hash1 == hash2
        assert len(hash1) == 16


class TestDataclasses:
    """Tests for dataclass definitions."""
    
    def test_environment_info_creation(self):
        """EnvironmentInfo should be creatable."""
        env = EnvironmentInfo(
            git_commit="abc123",
            git_branch="main",
            git_dirty=False,
            config_file="test.yaml",
            config_hash="def456",
            hostname="test-host",
            user="testuser",
            os_name="Darwin",
            os_version="23.1.0",
            python_version="3.11.5",
            timezone="UTC",
            start_time_utc="2025-12-19T14:30:00Z",
        )
        assert env.git_commit == "abc123"
        assert env.git_branch == "main"
    
    def test_generator_config_defaults(self):
        """GeneratorConfig should have sensible defaults."""
        config = GeneratorConfig(model="gemini-2.5-flash")
        assert config.model == "gemini-2.5-flash"
        assert config.temperature == 0.0
        assert config.max_output_tokens == 8192
        assert config.reasoning_effort == "low"
    
    def test_metrics_aggregate_creation(self):
        """MetricsAggregate should be creatable."""
        metrics = MetricsAggregate(
            precision_k=25,
            total=100,
            completed=100,
            recall_at_100=0.95,
            mrr=0.75,
            pass_rate=0.85,
            partial_rate=0.10,
            fail_rate=0.05,
            acceptable_rate=0.95,
        )
        assert metrics.pass_rate == 0.85
        assert metrics.total == 100


class TestJSONSchemaFile:
    """Tests for the JSON Schema file."""
    
    @pytest.fixture
    def json_schema(self):
        """Load the JSON schema file."""
        schema_path = Path(__file__).parent.parent.parent / "lib" / "schemas" / "eval_output_v2.json"
        with open(schema_path) as f:
            return json.load(f)
    
    def test_schema_is_valid_json(self, json_schema):
        """JSON schema should be valid JSON."""
        assert isinstance(json_schema, dict)
    
    def test_schema_has_required_fields(self, json_schema):
        """Schema should define required fields."""
        assert "required" in json_schema
        assert "schema_version" in json_schema["required"]
        assert "run_id" in json_schema["required"]
        assert "results" in json_schema["required"]
    
    def test_schema_has_definitions(self, json_schema):
        """Schema should have definitions for nested types."""
        assert "definitions" in json_schema
        assert "EnvironmentInfo" in json_schema["definitions"]
        assert "GeneratorConfig" in json_schema["definitions"]
        assert "QuestionResult" in json_schema["definitions"]
    
    def test_schema_version_pattern(self, json_schema):
        """Schema version should have pattern for 2.x."""
        version_prop = json_schema["properties"]["schema_version"]
        assert "pattern" in version_prop
        assert "2" in version_prop["pattern"]
    
    def test_run_type_enum(self, json_schema):
        """run_type should be enum with valid values."""
        run_type_prop = json_schema["properties"]["run_type"]
        assert "enum" in run_type_prop
        assert "checkpoint" in run_type_prop["enum"]
        assert "run" in run_type_prop["enum"]
        assert "experiment" in run_type_prop["enum"]


class TestSchemaVersionCompatibility:
    """Tests for schema version compatibility."""
    
    def test_v2_0_is_current(self):
        """v2.0 should be the current version."""
        assert SCHEMA_VERSION == "2.0"
    
    def test_validate_accepts_2_x_versions(self):
        """Validator should accept any 2.x version."""
        data = {"schema_version": "2.1"}  # Future minor version
        is_valid, errors = validate_eval_output(data)
        # Should fail for other reasons, but not schema version
        schema_errors = [e for e in errors if "Schema version" in e]
        assert len(schema_errors) == 0
    
    def test_validate_rejects_1_x_versions(self):
        """Validator should reject 1.x versions."""
        data = {"schema_version": "1.1"}
        is_valid, errors = validate_eval_output(data)
        assert not is_valid
        assert any("Schema version mismatch" in e for e in errors)
    
    def test_validate_rejects_3_x_versions(self):
        """Validator should reject 3.x versions (future major)."""
        data = {"schema_version": "3.0"}
        is_valid, errors = validate_eval_output(data)
        assert not is_valid
        assert any("Schema version mismatch" in e for e in errors)


class TestIntegration:
    """Integration tests for schema and metadata together."""
    
    def test_full_output_creation(self):
        """Test creating a full output using helpers."""
        from lib.core.metadata import (
            build_environment_info,
            build_generator_config,
            build_judge_config,
            build_retrieval_config,
            build_corpus_info,
            build_execution_info,
        )
        
        output = {
            "schema_version": SCHEMA_VERSION,
            "run_id": "test_integration_001",
            "run_type": "experiment",
            "client": "BFAI",
            "timestamp": get_utc_timestamp(),
            "environment": build_environment_info(
                config_file="experiment_config.yaml",
                config_path=Path(__file__),
            ),
            "generator_config": build_generator_config(
                model="gemini-2.5-flash",
                temperature=0.0,
            ),
            "judge_config": build_judge_config(
                model="gemini-2.0-flash",
            ),
            "retrieval_config": build_retrieval_config(
                index_job_id="test_job",
                embedding_model="gemini-embedding-001",
                embedding_dimension=1536,
            ),
            "corpus": build_corpus_info(
                file_path="test/corpus.json",
                question_count=100,
            ),
            "execution": build_execution_info(
                mode="local",
                run_type="experiment",
                workers=5,
            ),
            "metrics": {
                "precision_k": 25,
                "total": 100,
                "completed": 100,
                "recall_at_100": 0.95,
                "mrr": 0.75,
                "pass_rate": 0.85,
                "partial_rate": 0.10,
                "fail_rate": 0.05,
            },
            "latency": {"total_avg_s": 8.5, "total_min_s": 2.0, "total_max_s": 30.0, "by_phase": {}},
            "tokens": {"prompt_total": 100000, "completion_total": 20000, "thinking_total": 5000, "cached_total": 0, "total": 125000},
            "answer_stats": {"avg_length_chars": 1500, "min_length_chars": 100, "max_length_chars": 5000},
            "quality": {"finish_reason_distribution": {"STOP": 100}, "fallback_rate": 0.0},
            "retry_stats": {"total_questions": 100, "succeeded_first_try": 100, "succeeded_after_retry": 0, "failed_all_retries": 0, "total_retry_attempts": 100, "avg_attempts": 1.0},
            "errors": {"total_errors": 0, "by_phase": {}, "error_messages": []},
            "skipped": {"count": 0, "reasons": {}, "question_ids": []},
            "breakdown_by_type": {},
            "breakdown_by_difficulty": {},
            "results": [],
        }
        
        is_valid, errors = validate_eval_output(output)
        assert is_valid, f"Integration test failed: {errors}"
    
    def test_output_serializes_to_json(self):
        """Test that output can be serialized to JSON."""
        result = get_empty_question_result("Q001")
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        # Should be parseable back
        parsed = json.loads(json_str)
        assert parsed["question_id"] == "Q001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
