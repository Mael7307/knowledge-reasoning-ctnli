"""OpenAI and Azure OpenAI client implementations"""

import os
from typing import Optional
from openai import OpenAI, AzureOpenAI

from .base import BaseModel


class OpenAIClient(BaseModel):
    """Client for OpenAI API (non-Azure)"""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.

        Args:
            model_name: Name of the OpenAI model (e.g., "gpt-4o", "o3")
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var
        """
        super().__init__(model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided via api_key parameter or OPENAI_API_KEY env var")
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 1.0, **kwargs) -> str:
        """Generate response using OpenAI API"""
        max_completion_tokens = kwargs.get("max_completion_tokens", max_tokens)
        max_retries = kwargs.get("max_retries", 3)

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries:
                    print(f"⚠️ Max retries reached: {e}")
                    raise
                print(f"⚠️ Attempt {attempt} failed: {e}, retrying...")


class AzureOpenAIClient(BaseModel):
    """Client for Azure OpenAI API"""

    def __init__(self, model_name: str, api_key: str, api_version: str, endpoint: str):
        """
        Initialize Azure OpenAI client.

        Args:
            model_name: Name of the model (e.g., "deepseek-r1")
            api_key: Azure OpenAI API key
            api_version: API version (e.g., "2024-02-15-preview")
            endpoint: Azure endpoint URL
        """
        super().__init__(model_name)
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        # Azure models are typically prefixed with "lunar-"
        self.azure_model_name = f"lunar-{model_name}"

    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 1.0, **kwargs) -> str:
        """Generate response using Azure OpenAI API"""
        max_retries = kwargs.get("max_retries", 3)

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.azure_model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                status = getattr(e, 'status_code', None)
                print(f"⚠️ Attempt {attempt} failed with error: {e} (status: {status})")
                if attempt == max_retries:
                    print("⚠️ Max retries reached, returning empty response")
                    return ""
                print("   Retrying...")

