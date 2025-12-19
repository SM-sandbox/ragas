"""
Unit tests to validate Cloud Run config matches Local baseline config.

These tests MUST pass before running cloud eval to ensure apples-to-apples comparison.
"""

import pytest
import json
import requests
from pathlib import Path
from google.oauth2 import service_account
from google.auth.transport.requests import Request


# Expected config values (from local baseline)
EXPECTED_CONFIG = {
    "model": "gemini-3-flash-preview",
    "temperature": 0.0,
    "recall_top_k": 100,
    "precision_top_n": 25,
    "enable_reranking": True,
    "retrieval_mode": "hybrid",
    "alpha": 0.5,
    "reasoning_effort": "low",
    "embedding_model": "gemini-embedding-001",
    "embedding_dimension": 1536,
}

# Cloud Run endpoint
CLOUD_RUN_URL = "https://bfai-api-ppfq5ahfsq-ue.a.run.app"
SA_KEY_PATH = Path(__file__).parent.parent / "config" / "ragas-cloud-run-invoker.json"
JOB_ID = "bfai__eval66a_g1_1536_tt"


def get_cloud_token():
    """Get ID token for Cloud Run authentication."""
    creds = service_account.IDTokenCredentials.from_service_account_file(
        str(SA_KEY_PATH), target_audience=CLOUD_RUN_URL
    )
    creds.refresh(Request())
    return creds.token


class TestCloudConfigEndpoint:
    """Tests for Cloud Run /config endpoint."""
    
    @pytest.fixture(scope="class")
    def cloud_config(self):
        """Fetch config from Cloud Run endpoint."""
        token = get_cloud_token()
        response = requests.get(
            f"{CLOUD_RUN_URL}/config",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    
    def test_default_model_matches(self, cloud_config):
        """Cloud Run default model must match local baseline."""
        assert cloud_config["defaultModel"] == EXPECTED_CONFIG["model"], \
            f"Model mismatch: cloud={cloud_config['defaultModel']}, expected={EXPECTED_CONFIG['model']}"
    
    def test_default_temperature_matches(self, cloud_config):
        """Cloud Run default temperature must match local baseline."""
        assert cloud_config["defaultTemperature"] == EXPECTED_CONFIG["temperature"], \
            f"Temperature mismatch: cloud={cloud_config['defaultTemperature']}, expected={EXPECTED_CONFIG['temperature']}"
    
    def test_default_recall_top_k_matches(self, cloud_config):
        """Cloud Run default recall_top_k must match local baseline."""
        assert cloud_config["defaultRecallTopK"] == EXPECTED_CONFIG["recall_top_k"], \
            f"Recall mismatch: cloud={cloud_config['defaultRecallTopK']}, expected={EXPECTED_CONFIG['recall_top_k']}"
    
    def test_default_precision_top_n_matches(self, cloud_config):
        """Cloud Run default precision_top_n must match local baseline."""
        assert cloud_config["defaultPrecisionTopN"] == EXPECTED_CONFIG["precision_top_n"], \
            f"Precision mismatch: cloud={cloud_config['defaultPrecisionTopN']}, expected={EXPECTED_CONFIG['precision_top_n']}"
    
    def test_default_reranking_enabled(self, cloud_config):
        """Cloud Run default reranking must be enabled."""
        assert cloud_config["defaultEnableReranking"] == EXPECTED_CONFIG["enable_reranking"], \
            f"Reranking mismatch: cloud={cloud_config['defaultEnableReranking']}, expected={EXPECTED_CONFIG['enable_reranking']}"
    
    def test_default_retrieval_mode_hybrid(self, cloud_config):
        """Cloud Run default retrieval mode must be hybrid."""
        assert cloud_config["defaultRetrievalMode"] == EXPECTED_CONFIG["retrieval_mode"], \
            f"Retrieval mode mismatch: cloud={cloud_config['defaultRetrievalMode']}, expected={EXPECTED_CONFIG['retrieval_mode']}"
    
    def test_default_alpha_matches(self, cloud_config):
        """Cloud Run default alpha must match local baseline."""
        assert cloud_config["defaultAlpha"] == EXPECTED_CONFIG["alpha"], \
            f"Alpha mismatch: cloud={cloud_config['defaultAlpha']}, expected={EXPECTED_CONFIG['alpha']}"
    
    def test_default_reasoning_effort_matches(self, cloud_config):
        """Cloud Run default reasoning effort must match local baseline."""
        assert cloud_config["defaultReasoningEffort"] == EXPECTED_CONFIG["reasoning_effort"], \
            f"Reasoning mismatch: cloud={cloud_config['defaultReasoningEffort']}, expected={EXPECTED_CONFIG['reasoning_effort']}"


class TestCloudRetrieveEndpoint:
    """Tests for Cloud Run /retrieve endpoint."""
    
    @pytest.fixture(scope="class")
    def retrieve_response(self):
        """Fetch retrieve response from Cloud Run endpoint."""
        token = get_cloud_token()
        response = requests.post(
            f"{CLOUD_RUN_URL}/retrieve",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"query": "What is the voltage rating?", "job_id": JOB_ID, "recall_top_k": 100},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    
    def test_retrieve_returns_100_candidates(self, retrieve_response):
        """Cloud Run /retrieve must return 100 candidates for apples-to-apples recall."""
        chunks = retrieve_response.get("chunks", [])
        assert len(chunks) == 100, \
            f"Expected 100 candidates, got {len(chunks)}"
    
    def test_retrieve_has_required_fields(self, retrieve_response):
        """Each chunk must have required fields."""
        chunks = retrieve_response.get("chunks", [])
        assert len(chunks) > 0, "No chunks returned"
        
        required_fields = ["doc_name", "text", "score"]
        for chunk in chunks[:5]:  # Check first 5
            for field in required_fields:
                assert field in chunk, f"Missing field '{field}' in chunk"
    
    def test_retrieve_recall_top_k_matches_request(self, retrieve_response):
        """Response recall_top_k must match request."""
        assert retrieve_response.get("recall_top_k") == 100, \
            f"Expected recall_top_k=100, got {retrieve_response.get('recall_top_k')}"


class TestCloudQueryEndpoint:
    """Tests for Cloud Run /query endpoint."""
    
    @pytest.fixture(scope="class")
    def query_response(self):
        """Fetch query response from Cloud Run endpoint."""
        token = get_cloud_token()
        response = requests.post(
            f"{CLOUD_RUN_URL}/query",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={
                "query": "What is the voltage rating?",
                "job_id": JOB_ID,
                "top_k": 25,
                "model": "gemini-3-flash-preview",
                "reasoning_effort": "low",
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    
    def test_query_uses_correct_model(self, query_response):
        """Query must use the specified model."""
        metadata = query_response.get("metadata", {})
        model = metadata.get("model", "")
        assert "gemini-3" in model or "gemini-3-flash" in model, \
            f"Expected gemini-3-flash model, got {model}"
    
    def test_query_returns_answer(self, query_response):
        """Query must return an answer."""
        answer = query_response.get("answer", "")
        assert len(answer) > 0, "No answer returned"
    
    def test_query_returns_sources(self, query_response):
        """Query must return sources."""
        sources = query_response.get("sources", [])
        assert len(sources) > 0, "No sources returned"
    
    def test_query_has_citations(self, query_response):
        """Query must have citations in metadata."""
        metadata = query_response.get("metadata", {})
        assert metadata.get("has_citations") == True, \
            "Expected has_citations=True in metadata"
    
    def test_query_reasoning_effort_matches(self, query_response):
        """Query reasoning effort must match request."""
        metadata = query_response.get("metadata", {})
        reasoning = metadata.get("reasoning_effort", "")
        assert reasoning == "low", \
            f"Expected reasoning_effort='low', got '{reasoning}'"


class TestLocalConfig:
    """Tests for local config file."""
    
    @pytest.fixture(scope="class")
    def local_config(self):
        """Load local dev config."""
        config_path = Path(__file__).parent.parent.parent / "gRAG_v3" / ".dev-config.json"
        with open(config_path) as f:
            return json.load(f)
    
    def test_local_model_is_gemini_3(self, local_config):
        """Local config must use gemini-3-flash-preview."""
        model = local_config["ai_models"]["ai_model"]
        assert model == "gemini-3-flash-preview", \
            f"Expected gemini-3-flash-preview, got {model}"
    
    def test_local_recall_top_k(self, local_config):
        """Local config recall_top_k must be 100."""
        recall = local_config["rag_defaults"]["recall_top_k"]
        assert recall == 100, f"Expected 100, got {recall}"
    
    def test_local_precision_top_n(self, local_config):
        """Local config precision_top_n must be 25."""
        precision = local_config["rag_defaults"]["precision_top_n"]
        assert precision == 25, f"Expected 25, got {precision}"
    
    def test_local_hybrid_enabled(self, local_config):
        """Local config must have hybrid search enabled."""
        hybrid = local_config["rag_defaults"]["enable_hybrid"]
        assert hybrid == True, f"Expected True, got {hybrid}"
    
    def test_local_reranking_enabled(self, local_config):
        """Local config must have reranking enabled."""
        reranking = local_config["rag_defaults"]["enable_reranking"]
        assert reranking == True, f"Expected True, got {reranking}"
    
    def test_local_temperature_zero(self, local_config):
        """Local config temperature must be 0.0."""
        temp = local_config["rag_defaults"]["temperature"]
        assert temp == 0.0, f"Expected 0.0, got {temp}"


def run_all_config_checks():
    """Run all config checks and return summary."""
    print("="*70)
    print("CLOUD RUN CONFIG VALIDATION")
    print("="*70)
    
    try:
        token = get_cloud_token()
        
        # Check /config endpoint
        print("\n1. Checking /config endpoint...")
        response = requests.get(
            f"{CLOUD_RUN_URL}/config",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        response.raise_for_status()
        config = response.json()
        
        checks = [
            ("defaultModel", config.get("defaultModel"), EXPECTED_CONFIG["model"]),
            ("defaultTemperature", config.get("defaultTemperature"), EXPECTED_CONFIG["temperature"]),
            ("defaultRecallTopK", config.get("defaultRecallTopK"), EXPECTED_CONFIG["recall_top_k"]),
            ("defaultPrecisionTopN", config.get("defaultPrecisionTopN"), EXPECTED_CONFIG["precision_top_n"]),
            ("defaultEnableReranking", config.get("defaultEnableReranking"), EXPECTED_CONFIG["enable_reranking"]),
            ("defaultRetrievalMode", config.get("defaultRetrievalMode"), EXPECTED_CONFIG["retrieval_mode"]),
            ("defaultAlpha", config.get("defaultAlpha"), EXPECTED_CONFIG["alpha"]),
            ("defaultReasoningEffort", config.get("defaultReasoningEffort"), EXPECTED_CONFIG["reasoning_effort"]),
        ]
        
        all_pass = True
        for name, actual, expected in checks:
            status = "✅" if actual == expected else "❌"
            if actual != expected:
                all_pass = False
            print(f"  {status} {name}: {actual} (expected: {expected})")
        
        # Check /retrieve endpoint
        print("\n2. Checking /retrieve endpoint...")
        response = requests.post(
            f"{CLOUD_RUN_URL}/retrieve",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"query": "test", "job_id": JOB_ID, "recall_top_k": 100},
            timeout=60,
        )
        response.raise_for_status()
        retrieve = response.json()
        chunks = retrieve.get("chunks", [])
        status = "✅" if len(chunks) == 100 else "❌"
        if len(chunks) != 100:
            all_pass = False
        print(f"  {status} Returns 100 candidates: {len(chunks)}")
        
        print("\n" + "="*70)
        if all_pass:
            print("✅ ALL CONFIG CHECKS PASSED - Ready for eval!")
        else:
            print("❌ SOME CHECKS FAILED - Fix config before running eval!")
        print("="*70)
        
        return all_pass
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    run_all_config_checks()
