#!/usr/bin/env python3
"""CLI script to run experiments"""

import argparse
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiments import ExperimentRunner, ExperimentConfig
from src.models import OpenAIClient, AzureOpenAIClient, GeminiClient, OllamaClient


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_experiment_config(
    model_type: str,
    model_name: str,
    task_type: str,
    prompt_type: str,
    data_dir: str,
    output_dir: str,
    input_files: list,
    config_dict: dict,
    num_runs: int = 10,
    **kwargs
) -> ExperimentConfig:
    """Create experiment configuration from arguments and config dict"""
    
    # Extract model-specific config
    api_key = None
    api_version = None
    endpoint = None
    
    if model_type == "openai":
        api_key = kwargs.get("api_key") or config_dict.get("openai", {}).get("api_key")
    elif model_type == "azure_openai":
        config_key = f"lunar-{model_name}"
        model_config = config_dict.get(config_key, {})
        api_key = kwargs.get("api_key") or model_config.get("api_key")
        api_version = kwargs.get("api_version") or model_config.get("version")
        endpoint = kwargs.get("endpoint") or model_config.get("endpoint")
    elif model_type == "gemini":
        api_key = kwargs.get("api_key") or config_dict.get("gemini", {}).get("api_key")

    return ExperimentConfig(
        model_type=model_type,
        model_name=model_name,
        data_dir=data_dir,
        output_dir=output_dir,
        prompt_type=prompt_type,
        task_type=task_type,
        input_files=input_files,
        num_runs=num_runs,
        api_key=api_key,
        api_version=api_version,
        endpoint=endpoint,
        ollama_model_name=kwargs.get("ollama_model_name"),
    )


def main():
    parser = argparse.ArgumentParser(description="Run cognitive gap experiments")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["openai", "azure_openai", "gemini", "ollama"],
        help="Type of model provider"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name (e.g., 'gpt-4o', 'deepseek-r1', 'gemini-2.5-pro')"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["nli", "factual"],
        help="Task type: 'nli' for entailment/neutral/contradiction, 'factual' for True/False"
    )
    parser.add_argument(
        "--prompt-type",
        required=True,
        choices=["direct", "cot"],
        help="Prompt type: 'direct' or 'cot' (chain-of-thought)"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing data files (relative to project root)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save results (relative to project root)"
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="List of input JSON files to process"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs per example (default: 10)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to config.yaml file (default: config/config.yaml)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: current working directory)"
    )
    parser.add_argument(
        "--api-key",
        help="Override API key from config file"
    )
    parser.add_argument(
        "--api-version",
        help="Override API version for Azure OpenAI"
    )
    parser.add_argument(
        "--endpoint",
        help="Override endpoint for Azure OpenAI"
    )
    parser.add_argument(
        "--ollama-model-name",
        help="Ollama model name (if different from model-name)"
    )

    args = parser.parse_args()

    # Load config
    config_dict = {}
    if args.config.exists():
        config_dict = load_config(args.config)
    elif args.model_type in ["openai", "azure_openai", "gemini"]:
        print("⚠️ Warning: Config file not found. Using provided API keys or environment variables.")

    # Create experiment config
    config = create_experiment_config(
        model_type=args.model_type,
        model_name=args.model_name,
        task_type=args.task_type,
        prompt_type=args.prompt_type,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        input_files=args.input_files,
        config_dict=config_dict,
        num_runs=args.num_runs,
        api_key=args.api_key,
        api_version=args.api_version,
        endpoint=args.endpoint,
        ollama_model_name=args.ollama_model_name
    )

    # Run experiments
    runner = ExperimentRunner(config, project_root=args.project_root)
    runner.run()


if __name__ == "__main__":
    main()

