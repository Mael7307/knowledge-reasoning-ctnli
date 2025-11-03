#!/usr/bin/env python3
"""CLI script to evaluate experiment results"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing results (organized by model)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing gold label files"
    )
    parser.add_argument(
        "--model",
        help="Specific model to evaluate (if not provided, evaluates all models)"
    )
    parser.add_argument(
        "--output-format",
        choices=["latex", "json", "table"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--metric",
        choices=["accuracy", "f1"],
        default="accuracy",
        help="Metric to display (default: accuracy)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (default: current working directory)"
    )

    args = parser.parse_args()

    evaluator = Evaluator(project_root=args.project_root)
    scores, total_missing = evaluator.evaluate_directory(
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        model_name=args.model
    )

    # Output results
    if args.output_format == "latex":
        rows = evaluator.generate_latex_table(scores, metric=args.metric)
        print(f"\\begin{{tabular}}{{l l c}}")
        print(f"\\textbf{{Model}} & \\textbf{{Task}} & \\textbf{{{args.metric.capitalize()}}} \\\\ \\hline")
        for row in rows:
            print(row)
        print("\\end{tabular}")
    elif args.output_format == "json":
        import json
        print(json.dumps(scores, indent=2))
    else:  # table
        print("\n" + "=" * 60)
        print(f"Evaluation Results ({args.metric.upper()})")
        print("=" * 60)
        for model in sorted(scores):
            print(f"\n{model}:")
            for task in sorted(scores[model]):
                value = scores[model][task].get(args.metric, 0.0)
                correct = scores[model][task].get("correct", 0)
                total = scores[model][task].get("total", 0)
                print(f"  {task:30s} {value:.3f} ({correct}/{total})")
        print("\n" + "=" * 60)

    if total_missing > 0:
        print(f"\n⚠️ Total responses missing 'output:' tag: {total_missing}")


if __name__ == "__main__":
    main()

