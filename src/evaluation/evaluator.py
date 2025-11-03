"""Main evaluation logic"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .metrics import extract_output, calculate_accuracy, calculate_f1


class Evaluator:
    """Evaluates model results against gold labels"""

    def __init__(self, project_root: Path = None):
        """
        Initialize evaluator.

        Args:
            project_root: Path to project root directory. If None, uses current working directory
        """
        self.project_root = project_root or Path.cwd()

    def evaluate_file(
        self,
        result_path: Path,
        gold_path: Path,
        extract_label_fn: Optional[callable] = None
    ) -> Tuple[int, int, int]:
        """
        Evaluate a single result file against gold labels.

        Args:
            result_path: Path to results JSON file
            gold_path: Path to gold labels JSON file
            extract_label_fn: Optional function to extract label from gold data.
                            If None, uses str(gold_data[ex_id]["label"]).lower().strip()

        Returns:
            Tuple of (correct, total, missing_output) counts
        """
        if not gold_path.exists():
            print(f"⚠️ Skipping {result_path.name} (no gold data at {gold_path})")
            return 0, 0, 0

        with open(gold_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        if extract_label_fn is None:
            extract_label_fn = lambda ex_id, gold_data: str(gold_data.get(ex_id, {}).get("label", "")).lower().strip()

        correct = 0
        total = 0
        missing_output = 0

        for ex_id, result_entry in results.items():
            gold_label = extract_label_fn(ex_id, gold_data)
            responses = result_entry.get("responses", [])

            if not gold_label:
                continue

            for resp in responses:
                pred = extract_output(resp)
                if not pred:
                    missing_output += 1
                    continue

                # For factual tasks, check if gold label appears in prediction
                # For NLI tasks, do exact match or substring match
                if gold_label in pred:
                    correct += 1
                total += 1

        return correct, total, missing_output

    def evaluate_directory(
        self,
        results_dir: Path,
        data_dir: Path,
        model_name: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all result files in a directory.

        Args:
            results_dir: Directory containing result files (organized by model)
            data_dir: Directory containing gold label files
            model_name: Specific model to evaluate. If None, evaluates all models

        Returns:
            Dict mapping model -> task -> metrics
        """
        scores = defaultdict(lambda: defaultdict(dict))
        total_missing = 0

        # Get models to evaluate
        if model_name:
            model_dirs = [results_dir / model_name] if (results_dir / model_name).exists() else []
        else:
            model_dirs = [results_dir / d for d in results_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            model = model_dir.name
            if model.startswith("llama3.2_1b"):  # Skip small models
                continue

            for result_file in model_dir.glob("*_res.json"):
                # Extract task name from filename
                base_name = result_file.stem.replace("_cot_res", "").replace("_res", "")
                task = base_name
                prompt_type = "cot" if "_cot" in result_file.stem else "direct"

                gold_path = data_dir / f"{task}.json"
                correct, total, missing = self.evaluate_file(result_file, gold_path)

                if total > 0:
                    accuracy = correct / total
                    scores[model][f"{task}_{prompt_type}"] = {
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total
                    }

                total_missing += missing

        return dict(scores), total_missing

    def generate_latex_table(
        self,
        scores: Dict[str, Dict[str, Dict[str, float]]],
        metric: str = "accuracy"
    ) -> List[str]:
        """
        Generate LaTeX table rows from evaluation scores.

        Args:
            scores: Dict from evaluate_directory
            metric: Metric to display ("accuracy" or "f1")

        Returns:
            List of LaTeX table rows
        """
        rows = []
        for model in sorted(scores):
            for task_prompt in sorted(scores[model]):
                value = scores[model][task_prompt].get(metric, 0.0)
                rows.append(f"{model} & {task_prompt} & {value:.3f} \\\\")
        return rows

