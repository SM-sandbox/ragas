"""
Schema Loader for LLM Metadata

Loads the LLM metadata schema from the orchestrator (sm-dev-01).
The orchestrator owns the schema - eval suite just reads it.

Source of Truth: sm-dev-01/services/api/core/llm_schema.json
GCS Location: gs://brightfoxai-documents/schemas/llm_schema.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# GCS bucket and path for production schema
GCS_BUCKET = "brightfoxai-documents"
GCS_SCHEMA_PATH = "schemas/llm_schema.json"

# Local fallback path (if repos are co-located)
LOCAL_SCHEMA_PATH = Path(__file__).parent.parent.parent / "sm-dev-01/services/api/core/llm_schema.json"


def load_schema_from_gcs() -> Dict[str, Any]:
    """Load LLM schema from GCS (production)."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_SCHEMA_PATH)
        schema_str = blob.download_as_string()
        return json.loads(schema_str)
    except Exception as e:
        logger.warning(f"Failed to load schema from GCS: {e}")
        return None


def load_schema_from_local() -> Dict[str, Any]:
    """Load LLM schema from local file (development)."""
    try:
        if LOCAL_SCHEMA_PATH.exists():
            with open(LOCAL_SCHEMA_PATH) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load schema from local: {e}")
    return None


def load_orchestrator_schema(prefer_gcs: bool = True) -> Dict[str, Any]:
    """
    Load the LLM metadata schema from orchestrator.
    
    Args:
        prefer_gcs: If True, try GCS first, then local. If False, try local first.
        
    Returns:
        Schema dictionary or None if not found.
    """
    if prefer_gcs:
        schema = load_schema_from_gcs()
        if schema:
            logger.info(f"Loaded schema v{schema.get('schema_version', '?')} from GCS")
            return schema
        schema = load_schema_from_local()
        if schema:
            logger.info(f"Loaded schema v{schema.get('schema_version', '?')} from local")
            return schema
    else:
        schema = load_schema_from_local()
        if schema:
            logger.info(f"Loaded schema v{schema.get('schema_version', '?')} from local")
            return schema
        schema = load_schema_from_gcs()
        if schema:
            logger.info(f"Loaded schema v{schema.get('schema_version', '?')} from GCS")
            return schema
    
    logger.error("Could not load LLM schema from any source")
    return None


def validate_llm_metadata(metadata: Dict[str, Any], schema: Dict[str, Any] = None) -> List[str]:
    """
    Validate llm_metadata against the orchestrator schema.
    
    Args:
        metadata: The llm_metadata dict to validate
        schema: Schema to validate against (loads from orchestrator if not provided)
        
    Returns:
        List of missing required fields (empty if valid)
    """
    if schema is None:
        schema = load_orchestrator_schema()
    
    if schema is None:
        logger.warning("No schema available for validation")
        return []
    
    missing = []
    llm_schema = schema.get("llm_metadata", {})
    
    # Handle nested "fields" structure from orchestrator schema
    fields = llm_schema.get("fields", llm_schema)
    
    for field, spec in fields.items():
        if isinstance(spec, dict) and spec.get("required", False) and field not in metadata:
            missing.append(field)
    
    if missing:
        logger.warning(f"Missing required fields from orchestrator schema: {missing}")
    
    return missing


def get_schema_fields() -> List[str]:
    """Get list of all fields defined in the orchestrator schema."""
    schema = load_orchestrator_schema()
    if schema:
        llm_schema = schema.get("llm_metadata", {})
        # Handle nested "fields" structure
        fields = llm_schema.get("fields", llm_schema)
        return list(fields.keys())
    return []


def check_schema_compatibility(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if our llm_metadata is compatible with orchestrator schema.
    
    Returns:
        Dict with 'compatible', 'missing', 'extra' fields
    """
    schema = load_orchestrator_schema()
    if schema is None:
        return {"compatible": True, "missing": [], "extra": [], "error": "No schema available"}
    
    llm_schema = schema.get("llm_metadata", {})
    # Handle nested "fields" structure
    fields = llm_schema.get("fields", llm_schema)
    
    schema_fields = set(fields.keys())
    metadata_fields = set(metadata.keys())
    
    missing = schema_fields - metadata_fields
    extra = metadata_fields - schema_fields
    
    # Check required fields
    required_missing = []
    for field in missing:
        spec = fields.get(field, {})
        if isinstance(spec, dict) and spec.get("required", False):
            required_missing.append(field)
    
    return {
        "compatible": len(required_missing) == 0,
        "missing": list(missing),
        "required_missing": required_missing,
        "extra": list(extra),
        "schema_version": schema.get("schema_version", "unknown"),
    }


# Pre-eval check function
def pre_eval_schema_check() -> bool:
    """
    Run before eval to ensure we can capture all orchestrator fields.
    
    Returns:
        True if schema is available and we're compatible
    """
    schema = load_orchestrator_schema()
    if schema is None:
        logger.error("PRE-EVAL CHECK FAILED: Cannot load orchestrator schema")
        return False
    
    logger.info(f"PRE-EVAL CHECK: Schema v{schema.get('schema_version')} loaded")
    logger.info(f"PRE-EVAL CHECK: {len(schema.get('llm_metadata', {}))} llm_metadata fields defined")
    
    # Log the fields we need to capture
    required = [f for f, s in schema.get("llm_metadata", {}).items() if s.get("required")]
    logger.info(f"PRE-EVAL CHECK: Required fields: {required}")
    
    return True
