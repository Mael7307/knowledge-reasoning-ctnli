#!/usr/bin/env python3
"""
Helper script to migrate data and results from the old structure to the new structure.

This script helps reorganize:
- Data files from cl_results_and_prompts/*_data/ to data/
- Results from cl_results_and_prompts/*_results/ to results/
- Prompts are already copied (but can be verified)
"""

import shutil
import argparse
from pathlib import Path


def migrate_data(old_data_dir: Path, new_data_dir: Path):
    """Migrate data files"""
    if not old_data_dir.exists():
        print(f"⚠️ Source data directory not found: {old_data_dir}")
        return

    new_data_dir.mkdir(parents=True, exist_ok=True)

    for dataset_dir in old_data_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name.endswith("_data"):
            dataset_name = dataset_dir.name.replace("_data", "")
            target_dir = new_data_dir / dataset_name
            target_dir.mkdir(exist_ok=True)

            print(f"📁 Migrating {dataset_dir.name} -> data/{dataset_name}")
            for json_file in dataset_dir.glob("*.json"):
                target_file = target_dir / json_file.name
                if target_file.exists():
                    print(f"   ⚠️ Skipping {json_file.name} (already exists)")
                else:
                    shutil.copy2(json_file, target_file)
                    print(f"   ✅ Copied {json_file.name}")


def migrate_results(old_results_dir: Path, new_results_dir: Path):
    """Migrate result files"""
    if not old_results_dir.exists():
        print(f"⚠️ Source results directory not found: {old_results_dir}")
        return

    new_results_dir.mkdir(parents=True, exist_ok=True)

    for dataset_dir in old_results_dir.iterdir():
        if dataset_dir.is_dir() and dataset_dir.name.endswith("_results"):
            dataset_name = dataset_dir.name.replace("_results", "")
            target_dir = new_results_dir / dataset_name
            target_dir.mkdir(exist_ok=True)

            print(f"📁 Migrating {dataset_dir.name} -> results/{dataset_name}")

            # Copy model subdirectories
            for model_dir in dataset_dir.iterdir():
                if model_dir.is_dir():
                    target_model_dir = target_dir / model_dir.name
                    target_model_dir.mkdir(exist_ok=True)

                    for json_file in model_dir.glob("*_res.json"):
                        target_file = target_model_dir / json_file.name
                        if target_file.exists():
                            print(f"   ⚠️ Skipping {model_dir.name}/{json_file.name} (already exists)")
                        else:
                            shutil.copy2(json_file, target_file)
                            print(f"   ✅ Copied {model_dir.name}/{json_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Migrate data and results to new structure")
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("cl_results_and_prompts"),
        help="Old directory containing data and results"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="New data directory"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="New results directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually copying files"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be copied\n")
    else:
        print("🚀 Migrating data and results...\n")

    # Migrate data
    print("=" * 60)
    print("MIGRATING DATA FILES")
    print("=" * 60)
    if not args.dry_run:
        migrate_data(args.old_dir, args.data_dir)
    else:
        print(f"Would migrate from {args.old_dir}/*_data/ to {args.data_dir}/")

    print("\n" + "=" * 60)
    print("MIGRATING RESULTS FILES")
    print("=" * 60)
    if not args.dry_run:
        migrate_results(args.old_dir, args.results_dir)
    else:
        print(f"Would migrate from {args.old_dir}/*_results/ to {args.results_dir}/")

    print("\n✅ Migration complete!")


if __name__ == "__main__":
    main()

