"""Google Gemini client implementation"""

import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .base import BaseModel


class GeminiClient(BaseModel):
    """Client for Google Gemini API"""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize Gemini client.

        Args:
            model_name: Name of the Gemini model (e.g., "gemini-2.5-pro")
            api_key: Gemini API key
        """
        super().__init__(model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 1.0, **kwargs) -> str:
        """Generate response using Gemini API"""
        retries = kwargs.get("retries", 10)
        wait_time = kwargs.get("wait_time", 60)

        for attempt in range(1, retries + 1):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": kwargs.get("max_output_tokens", max_tokens * 5)  # Gemini uses larger defaults
                    },
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    }
                )

                # Handle response extraction
                if hasattr(response, "text") and response.text:
                    return response.text.strip()

                # Fallback: collect text from parts
                texts = []
                for cand in getattr(response, "candidates", []):
                    if cand.content and cand.content.parts:
                        for p in cand.content.parts:
                            if hasattr(p, "text") and p.text:
                                texts.append(p.text)
                            elif isinstance(p, dict) and "text" in p:
                                texts.append(p["text"])
                content = "\n".join(texts).strip()
                return content if content else f"ERROR: No text content in response"

            except Exception as e:
                print(f"   ⚠️ Error during generation (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    print(f"   ⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("   ❌ All retries failed.")
                    return f"ERROR: {e}"

