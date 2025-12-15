"""LLM client using Google GenAI SDK with Vertex AI"""
import json
from typing import Dict, Any, Optional
from google import genai

from config import config


class LLMClient:
    """Client for Vertex AI Gemini via google-genai SDK"""
    
    def __init__(self, model_name: Optional[str] = None):
        # Use Vertex AI backend with ADC
        self.client = genai.Client(
            vertexai=True, 
            project=config.GCP_PROJECT_ID, 
            location=config.GCP_LLM_LOCATION
        )
        self.model_name = model_name or config.LLM_MODEL
        
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> str:
        """Generate text from a prompt"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text
    
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Generate JSON response from a prompt"""
        # Add JSON instruction to prompt
        json_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON. No markdown, no code blocks, just raw JSON."""
        
        response = self.generate(json_prompt, temperature, max_tokens)
        
        # Clean up response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        return json.loads(response)


if __name__ == "__main__":
    # Test the client
    client = LLMClient()
    
    response = client.generate("What is a solar inverter? Answer in one sentence.")
    print(f"Response: {response}")
