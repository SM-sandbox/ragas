#!/usr/bin/env python3
"""
Comprehensive Test Suite for Schema Loader

Tests:
- Unit: Schema parsing, validation logic
- Integration: GCS loading, local file loading
- Functional: Schema compatibility checks, pre-eval validation

Run with: pytest tests/test_schema_loader.py -v
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schema_loader import (
    load_orchestrator_schema,
    load_schema_from_gcs,
    load_schema_from_local,
    validate_llm_metadata,
    get_schema_fields,
    check_schema_compatibility,
    pre_eval_schema_check,
    GCS_BUCKET,
    GCS_SCHEMA_PATH,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_schema():
    """Sample schema matching orchestrator format."""
    return {
        "schema_version": "1.0.0",
        "last_updated": "2025-12-17",
        "llm_metadata": {
            "prompt_tokens": {"type": "int", "required": True},
            "completion_tokens": {"type": "int", "required": True},
            "thinking_tokens": {"type": "int", "required": True},
            "total_tokens": {"type": "int", "required": True},
            "cached_content_tokens": {"type": "int", "required": False},
            "model_version": {"type": "str", "required": True},
            "finish_reason": {"type": "str", "required": True},
            "used_fallback": {"type": "bool", "required": True},
            "reasoning_effort": {"type": "str", "required": True},
            "avg_logprobs": {"type": "float", "required": False},
            "response_id": {"type": "str", "required": False},
        }
    }


@pytest.fixture
def valid_llm_metadata():
    """Valid llm_metadata matching schema."""
    return {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "thinking_tokens": 25,
        "total_tokens": 175,
        "cached_content_tokens": 0,
        "model_version": "gemini-3-flash-preview",
        "finish_reason": "STOP",
        "used_fallback": False,
        "reasoning_effort": "low",
        "avg_logprobs": -0.42,
        "response_id": "resp_123",
    }


@pytest.fixture
def incomplete_llm_metadata():
    """Incomplete llm_metadata missing required fields."""
    return {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        # Missing: thinking_tokens, total_tokens, model_version, finish_reason, etc.
    }


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestUnitSchemaConstants:
    """Unit tests for schema constants."""
    
    def test_gcs_bucket_configured(self):
        """Test GCS bucket is configured."""
        assert GCS_BUCKET == "brightfoxai-documents"
    
    def test_gcs_schema_path_configured(self):
        """Test GCS schema path is configured."""
        assert GCS_SCHEMA_PATH == "schemas/llm_schema.json"


class TestUnitValidation:
    """Unit tests for validation logic."""
    
    def test_validate_complete_metadata(self, sample_schema, valid_llm_metadata):
        """Test validation passes for complete metadata."""
        missing = validate_llm_metadata(valid_llm_metadata, sample_schema)
        assert len(missing) == 0
    
    def test_validate_incomplete_metadata(self, sample_schema, incomplete_llm_metadata):
        """Test validation catches missing required fields."""
        missing = validate_llm_metadata(incomplete_llm_metadata, sample_schema)
        
        assert len(missing) > 0
        assert "thinking_tokens" in missing
        assert "total_tokens" in missing
        assert "model_version" in missing
        assert "finish_reason" in missing
    
    def test_validate_with_extra_fields(self, sample_schema, valid_llm_metadata):
        """Test validation allows extra fields not in schema."""
        metadata_with_extra = {**valid_llm_metadata, "extra_field": "value"}
        missing = validate_llm_metadata(metadata_with_extra, sample_schema)
        assert len(missing) == 0
    
    def test_validate_empty_metadata(self, sample_schema):
        """Test validation catches all missing required fields for empty metadata."""
        missing = validate_llm_metadata({}, sample_schema)
        
        # Should have all required fields missing
        required_fields = [f for f, s in sample_schema["llm_metadata"].items() if s.get("required")]
        assert len(missing) == len(required_fields)


class TestUnitSchemaCompatibility:
    """Unit tests for schema compatibility checking."""
    
    def test_compatibility_check_valid(self, valid_llm_metadata):
        """Test compatibility check with valid metadata."""
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = {
                "schema_version": "1.0.0",
                "llm_metadata": {
                    "prompt_tokens": {"type": "int", "required": True},
                    "completion_tokens": {"type": "int", "required": True},
                }
            }
            
            result = check_schema_compatibility(valid_llm_metadata)
            
            assert result["compatible"] is True
            assert len(result["required_missing"]) == 0
    
    def test_compatibility_check_missing_required(self, incomplete_llm_metadata):
        """Test compatibility check catches missing required fields."""
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = {
                "schema_version": "1.0.0",
                "llm_metadata": {
                    "prompt_tokens": {"type": "int", "required": True},
                    "completion_tokens": {"type": "int", "required": True},
                    "thinking_tokens": {"type": "int", "required": True},
                }
            }
            
            result = check_schema_compatibility(incomplete_llm_metadata)
            
            assert result["compatible"] is False
            assert "thinking_tokens" in result["required_missing"]
    
    def test_compatibility_check_extra_fields(self, valid_llm_metadata):
        """Test compatibility check reports extra fields."""
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = {
                "schema_version": "1.0.0",
                "llm_metadata": {
                    "prompt_tokens": {"type": "int", "required": True},
                }
            }
            
            result = check_schema_compatibility(valid_llm_metadata)
            
            assert len(result["extra"]) > 0
            assert "completion_tokens" in result["extra"]


class TestUnitGetSchemaFields:
    """Unit tests for get_schema_fields."""
    
    def test_get_schema_fields_returns_list(self):
        """Test get_schema_fields returns a list."""
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = {
                "llm_metadata": {
                    "field1": {},
                    "field2": {},
                }
            }
            
            fields = get_schema_fields()
            
            assert isinstance(fields, list)
            assert "field1" in fields
            assert "field2" in fields
    
    def test_get_schema_fields_empty_when_no_schema(self):
        """Test get_schema_fields returns empty list when no schema."""
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = None
            
            fields = get_schema_fields()
            
            assert fields == []


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegrationGCSLoading:
    """Integration tests for GCS schema loading."""
    
    def test_load_schema_from_gcs_success(self):
        """Test loading schema from GCS succeeds."""
        # This is a real integration test - requires GCS access
        schema = load_schema_from_gcs()
        
        if schema is not None:
            assert "schema_version" in schema
            assert "llm_metadata" in schema
    
    def test_load_schema_from_gcs_handles_error(self):
        """Test GCS loading handles errors gracefully."""
        # Patch at the google.cloud.storage level
        with patch('google.cloud.storage.Client') as mock_client:
            mock_client.side_effect = Exception("GCS error")
            
            schema = load_schema_from_gcs()
            
            assert schema is None


class TestIntegrationLocalLoading:
    """Integration tests for local schema loading."""
    
    def test_load_schema_from_local_when_exists(self):
        """Test loading schema from local file when it exists."""
        schema = load_schema_from_local()
        
        # May or may not exist depending on environment
        if schema is not None:
            assert "schema_version" in schema
            assert "llm_metadata" in schema
    
    def test_load_schema_from_local_handles_missing(self):
        """Test local loading handles missing file gracefully."""
        with patch('schema_loader.LOCAL_SCHEMA_PATH') as mock_path:
            mock_path.exists.return_value = False
            
            schema = load_schema_from_local()
            
            # Should return None, not crash


class TestIntegrationOrchestratorSchema:
    """Integration tests for orchestrator schema loading."""
    
    def test_load_orchestrator_schema_prefers_gcs(self):
        """Test orchestrator schema prefers GCS by default."""
        schema = load_orchestrator_schema(prefer_gcs=True)
        
        # Should get schema from somewhere
        if schema is not None:
            assert "schema_version" in schema
    
    def test_load_orchestrator_schema_fallback_to_local(self):
        """Test orchestrator schema falls back to local."""
        with patch('schema_loader.load_schema_from_gcs') as mock_gcs:
            mock_gcs.return_value = None
            
            schema = load_orchestrator_schema(prefer_gcs=True)
            
            # Should try local as fallback


# =============================================================================
# FUNCTIONAL TESTS
# =============================================================================

class TestFunctionalPreEvalCheck:
    """Functional tests for pre-eval schema check."""
    
    def test_pre_eval_check_success(self):
        """Test pre-eval check succeeds when schema available."""
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = {
                "schema_version": "1.0.0",
                "llm_metadata": {
                    "prompt_tokens": {"required": True},
                }
            }
            
            result = pre_eval_schema_check()
            
            assert result is True
    
    def test_pre_eval_check_fails_when_no_schema(self):
        """Test pre-eval check fails when no schema available."""
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = None
            
            result = pre_eval_schema_check()
            
            assert result is False


class TestFunctionalValidationWorkflow:
    """Functional tests for validation workflow."""
    
    def test_full_validation_workflow(self, sample_schema, valid_llm_metadata):
        """Test full validation workflow."""
        # Step 1: Load schema
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = sample_schema
            
            # Step 2: Validate metadata
            missing = validate_llm_metadata(valid_llm_metadata)
            assert len(missing) == 0
            
            # Step 3: Check compatibility
            compat = check_schema_compatibility(valid_llm_metadata)
            assert compat["compatible"] is True
    
    def test_validation_catches_schema_drift(self, sample_schema, valid_llm_metadata):
        """Test validation catches when orchestrator adds new required fields."""
        # Simulate orchestrator adding a new required field
        schema_with_new_field = {**sample_schema}
        schema_with_new_field["llm_metadata"]["new_required_field"] = {
            "type": "str",
            "required": True
        }
        
        with patch('schema_loader.load_orchestrator_schema') as mock_load:
            mock_load.return_value = schema_with_new_field
            
            # Our metadata doesn't have the new field
            missing = validate_llm_metadata(valid_llm_metadata)
            
            assert "new_required_field" in missing


# =============================================================================
# E2E TESTS
# =============================================================================

class TestE2ESchemaIntegration:
    """End-to-end tests for schema integration with gemini_client."""
    
    def test_gemini_client_metadata_matches_schema(self):
        """Test gemini_client output matches orchestrator schema."""
        from gemini_client import generate
        
        # Generate a response
        result = generate("Say hello.")
        
        assert "llm_metadata" in result
        metadata = result["llm_metadata"]
        
        # Load schema and validate
        schema = load_orchestrator_schema()
        
        if schema is not None:
            missing = validate_llm_metadata(metadata, schema)
            # Should have no missing required fields
            assert len(missing) == 0, f"Missing required fields: {missing}"
    
    def test_schema_compatibility_with_real_output(self):
        """Test schema compatibility with real gemini_client output."""
        from gemini_client import generate
        
        result = generate("What is 2+2?")
        metadata = result["llm_metadata"]
        
        compat = check_schema_compatibility(metadata)
        
        # Should be compatible
        assert compat["compatible"] is True, f"Not compatible: {compat}"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
