"""Experiment configuration"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for running experiments"""

    model_type: str  # "openai", "azure_openai", "gemini", "ollama"
    model_name: str  # e.g., "gpt-4o", "deepseek-r1", "gemini-2.5-pro"
    data_dir: str  # Directory containing data files
    output_dir: str  # Directory to save results
    prompt_type: str  # "direct" or "cot"
    task_type: str  # "nli" (entailment/neutral/contradiction) or "factual" (True/False)
    input_files: List[str]  # List of JSON data files to process
    num_runs: int = 10  # Number of runs per example
    max_tokens: int = 2000
    temperature: float = 1.0

    # Optional: model-specific config
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    endpoint: Optional[str] = None
    ollama_model_name: Optional[str] = None

    def get_prompt_path(self, base_dir: Path) -> Path:
        """Get path to prompt template"""
        prompt_dir = base_dir / "prompts" / self.task_type
        prompt_filename = f"{self.prompt_type}.txt"
        return prompt_dir / prompt_filename

    def get_output_filename(self, data_filename: str) -> str:
        """Get output filename for a data file"""
        base_name = Path(data_filename).stem
        suffix = "cot_res" if self.prompt_type == "cot" else "res"
        return f"{base_name}_{suffix}.json"

