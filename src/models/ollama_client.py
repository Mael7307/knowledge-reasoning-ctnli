"""Ollama client implementation for local models"""

from typing import Optional
import ollama

from .base import BaseModel


class OllamaClient(BaseModel):
    """Client for Ollama (local models)"""

    def __init__(self, model_name: str, ollama_model_name: Optional[str] = None):
        """
        Initialize Ollama client.

        Args:
            model_name: Name identifier for the model (e.g., "llama3.2")
            ollama_model_name: Actual Ollama model name (e.g., "llama3.2:latest")
                              If None, uses model_name
        """
        super().__init__(model_name)
        self.ollama_model_name = ollama_model_name or model_name

    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 1.0, **kwargs) -> str:
        """Generate response using Ollama"""
        response = ollama.chat(
            model=self.ollama_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content'].strip()

