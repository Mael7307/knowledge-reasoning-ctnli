"""Main experiment runner"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from ..models import BaseModel, OpenAIClient, AzureOpenAIClient, GeminiClient, OllamaClient
from .config import ExperimentConfig


def create_model_client(config: ExperimentConfig) -> BaseModel:
    """Create appropriate model client based on config"""
    if config.model_type == "openai":
        return OpenAIClient(
            model_name=config.model_name,
            api_key=config.api_key
        )
    elif config.model_type == "azure_openai":
        if not all([config.api_key, config.api_version, config.endpoint]):
            raise ValueError("Azure OpenAI requires api_key, api_version, and endpoint")
        return AzureOpenAIClient(
            model_name=config.model_name,
            api_key=config.api_key,
            api_version=config.api_version,
            endpoint=config.endpoint
        )
    elif config.model_type == "gemini":
        if not config.api_key:
            raise ValueError("Gemini requires api_key")
        return GeminiClient(
            model_name=config.model_name,
            api_key=config.api_key
        )
    elif config.model_type == "ollama":
        return OllamaClient(
            model_name=config.model_name,
            ollama_model_name=config.ollama_model_name
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


class ExperimentRunner:
    """Runs experiments with LLMs on cognitive gap tasks"""

    def __init__(self, config: ExperimentConfig, project_root: Path = None):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration
            project_root: Path to project root directory. If None, uses current working directory
        """
        self.config = config
        self.project_root = project_root or Path.cwd()
        self.model = create_model_client(config)

    def load_prompt_template(self) -> str:
        """Load prompt template from file"""
        prompt_path = self.config.get_prompt_path(self.project_root)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def process_example(self, example: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
        """
        Process a single example.

        Args:
            example: Example dict with premise, statement, label, etc.
            prompt_template: Prompt template string with {premise} and {statement} placeholders

        Returns:
            Result dict with premise, statement, label, and responses
        """
        premise_raw = example.get("premise", "")
        premise = " ".join(premise_raw).strip() if isinstance(premise_raw, list) else premise_raw.strip()
        statement = example.get("statement", "").strip()
        label = example.get("label", "")

        message = prompt_template.format(
            premise=premise,
            statement=statement
        )

        print(f"   → Generating {self.config.num_runs} responses...")
        responses = self.model.generate_multiple(
            message,
            num_runs=self.config.num_runs,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )

        result = {
            "premise": premise,
            "statement": statement,
            "label": label,
            "responses": responses
        }

        # Include additional fields if present
        for key in ["reasoning_type", "Reasoning type"]:
            if key in example:
                result[key] = example[key]

        return result

    def process_file(self, filename: str) -> None:
        """
        Process a single data file and save results.

        Args:
            filename: Name of the data file (e.g., "causal.json")
        """
        data_path = self.project_root / self.config.data_dir / filename
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        output_dir = self.project_root / self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = self.config.get_output_filename(filename)
        output_path = output_dir / output_filename

        # Load examples
        with open(data_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)

        # Load prompt template
        prompt_template = self.load_prompt_template()

        # Process all examples
        all_results = {}
        print(f"\n📂 Processing {filename} ({len(examples)} examples)")
        for ex_id, ex in examples.items():
            print(f"🔍 Example {ex_id}")
            result = self.process_example(ex, prompt_template)
            all_results[ex_id] = result
            print(f"   ✅ Completed example {ex_id}\n")

        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved results to {output_path}\n")

    def run(self) -> None:
        """Run experiments for all configured input files"""
        print(f"🚀 Starting experiments")
        print(f"   Model: {self.config.model_type}/{self.config.model_name}")
        print(f"   Task: {self.config.task_type}")
        print(f"   Prompt: {self.config.prompt_type}")
        print(f"   Files: {self.config.input_files}\n")

        for filename in self.config.input_files:
            try:
                self.process_file(filename)
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                raise

        print("✅ All experiments completed!")

