"""Base model interface for LLM clients"""

from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """Base class for all LLM model clients"""

    def __init__(self, model_name: str):
        """
        Initialize the model client.

        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 1.0, **kwargs) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text response
        """
        pass

    def generate_multiple(self, prompt: str, num_runs: int = 10, **kwargs) -> List[str]:
        """
        Generate multiple responses for the same prompt.

        Args:
            prompt: Input prompt
            num_runs: Number of runs to generate
            **kwargs: Additional parameters passed to generate()

        Returns:
            List of generated responses
        """
        responses = []
        for i in range(num_runs):
            print(f"   → Run {i + 1}/{num_runs}")
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses

