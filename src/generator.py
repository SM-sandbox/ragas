"""
LLM client using Google GenAI SDK with Gemini 3 Flash.

Updated: Dec 2025 - Migrated to google-genai SDK with API key auth
"""
import json
from typing import Dict, Any, Optional

from gemini_client import (
    generate,
    generate_json as _generate_json,
    generate_for_rag,
    get_model_info,
    get_client,
)


class LLMClient:
    """Client for Gemini 3 Flash via google-genai SDK"""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the shared client
        self.client = get_client()
        model_info = get_model_info()
        self.model_name = model_name or model_info["model_id"]
        print(f"LLMClient initialized with {self.model_name}")
        
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        """Generate text from a prompt"""
        result = generate(
            prompt,
            model=self.model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return result["text"]
    
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Generate JSON response from a prompt"""
        return _generate_json(
            prompt,
            model=self.model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )


if __name__ == "__main__":
    # Test the client
    client = LLMClient()
    
    response = client.generate("What is a solar inverter? Answer in one sentence.")
    print(f"Response: {response}")
    
    json_response = client.generate_json("Return JSON with name='test' and value=123")
    print(f"JSON Response: {json_response}")
