#!/usr/bin/env python3
"""
Cleanup script for removing temporary files and results
Useful for clean demonstrations to professors
"""

import shutil
import argparse
from pathlib import Path


class ProjectCleaner:
    """Clean up project files for demonstrations"""

    def __init__(self, project_root="."):
        self.project_root = Path(project_root)

    def clean_demo_files(self):
        """Remove demo images and visualizations"""
        print("Cleaning demo files...")

        demo_dir = self.project_root / "demo_images"
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            print(f"Removed: {demo_dir}")
        else:
            print("No demo files found")

    def clean_results(self):
        """Remove result files and logs"""
        print("Cleaning result files...")

        result_files = [
            "evaluation_results.json",
            "comprehensive_evaluation.json",
            "final_results.json",
            "presentation_results.json",
            "training_results.txt",
        ]

        removed_count = 0
        for file_name in result_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                file_path.unlink()
                print(f"Removed: {file_path}")
                removed_count += 1

        if removed_count == 0:
            print("No result files found")

    def clean_temp_files(self):
        """Remove temporary files and caches"""
        print("Cleaning temporary files...")

        temp_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/runs",
            "**/logs",
        ]

        removed_count = 0
        for pattern in temp_patterns:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"Removed: {path}")
                removed_count += 1

        if removed_count == 0:
            print("No temporary files found")

    def clean_models(self, force_confirm=False):
        """Remove trained models (VERY CAREFUL!)"""
        print("DANGER: This will remove all trained models!")
        print("This includes:")
        print("- PDLPR trained model (pdlpr_final.pth)")
        print("- Baseline model (baseline_model.pth)")
        print("- YOLOv5 models (*.pt)")
        print("\nThese models took hours to train and cannot be recovered!")

        if not force_confirm:
            confirm1 = input("Type 'DELETE_MODELS' to confirm: ")
            if confirm1 != "DELETE_MODELS":
                print("Model removal cancelled - incorrect confirmation")
                return

            confirm2 = input("Are you absolutely sure? Type 'YES_DELETE': ")
            if confirm2 != "YES_DELETE":
                print("Model removal cancelled - double confirmation failed")
                return

        models_dir = self.project_root / "models"
        if models_dir.exists():
            removed_count = 0
            for model_file in models_dir.glob("*.pth"):
                model_file.unlink()
                print(f"Removed model: {model_file}")
                removed_count += 1
            for model_file in models_dir.glob("*.pt"):
                model_file.unlink()
                print(f"Removed model: {model_file}")
                removed_count += 1

            if removed_count > 0:
                print(f"{removed_count} models removed - you will need to retrain!")
            else:
                print("No models found to remove")
        else:
            print("Models directory not found")

    def clean_data_cache(self):
        """DISABLED: Remove processed data cache"""
        print("Data cache cleaning is DISABLED for safety")
        print("Processed data contains your training subset (1546.3 MB)")
        print("Removing this would require re-downloading and processing CCPD dataset")
        print("Use manual deletion if absolutely necessary")

    def quick_clean(self):
        """Quick clean for demonstrations (safe)"""
        print("=" * 50)
        print("QUICK CLEAN FOR DEMONSTRATION")
        print("=" * 50)
        print("Removing: demo files, results, temporary files")
        print("Preserving: models, training data")

        self.clean_demo_files()
        self.clean_results()
        self.clean_temp_files()

        print("\nQuick clean completed!")
        print("Models and training data preserved")
        print("Project ready for fresh demonstration")

    def deep_clean(self):
        """Deep clean with model removal option"""
        print("=" * 50)
        print("DEEP CLEAN OPTIONS")
        print("=" * 50)
        print("This will remove:")
        print("Demo files and results")
        print("Temporary files and caches")
        print("Training data (PROTECTED)")
        print("Trained models (OPTIONAL)")

        print("\nChoose deep clean option:")
        print("1. Safe deep clean (preserve models)")
        print("2. Full deep clean (remove models too)")
        print("3. Cancel")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            self.clean_demo_files()
            self.clean_results()
            self.clean_temp_files()
            print("\nSafe deep clean completed!")
            print("Models and training data preserved")

        elif choice == "2":
            self.clean_demo_files()
            self.clean_results()
            self.clean_temp_files()
            print("\nNow removing models...")
            self.clean_models()
            print("\nFull deep clean completed!")
            print("Training data still preserved")

        else:
            print("Deep clean cancelled")

    def list_cleanable_items(self):
        """List what can be cleaned"""
        print("=" * 50)
        print("CLEANABLE ITEMS")
        print("=" * 50)

        # Demo files
        demo_dir = self.project_root / "demo_images"
        if demo_dir.exists():
            demo_count = len(list(demo_dir.rglob("*")))
            print(f"Demo files: {demo_count} items in {demo_dir}")
        else:
            print("Demo files: None found")

        # Result files
        result_files = [
            "evaluation_results.json",
            "comprehensive_evaluation.json",
            "final_results.json",
            "presentation_results.json",
        ]

        result_count = 0
        for file_name in result_files:
            if (self.project_root / file_name).exists():
                result_count += 1
        print(f"Result files: {result_count} files")

        # Models
        models_dir = self.project_root / "models"
        if models_dir.exists():
            model_count = len(list(models_dir.glob("*.pth"))) + len(
                list(models_dir.glob("*.pt"))
            )
            print(f"Trained models: {model_count} files")
        else:
            print("Trained models: None found")

        # Processed data
        processed_dir = self.project_root / "data" / "processed"
        if processed_dir.exists():
            data_size = sum(
                f.stat().st_size for f in processed_dir.rglob("*") if f.is_file()
            )
            print(f"Processed data: {data_size / (1024 * 1024):.1f} MB")
        else:
            print("Processed data: None found")


def main():
    parser = argparse.ArgumentParser(description="Project Cleanup Utility")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick clean (demo files, results, temp files)",
    )
    parser.add_argument(
        "--deep", action="store_true", help="Deep clean (everything including models)"
    )
    parser.add_argument(
        "--demo-only", action="store_true", help="Clean only demo files"
    )
    parser.add_argument(
        "--results-only", action="store_true", help="Clean only result files"
    )
    parser.add_argument(
        "--temp-only", action="store_true", help="Clean only temporary files"
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Remove trained models (requires double confirmation)",
    )
    parser.add_argument("--list", action="store_true", help="List what can be cleaned")
    parser.add_argument(
        "--project-root", type=str, default=".", help="Project root directory"
    )

    args = parser.parse_args()

    cleaner = ProjectCleaner(args.project_root)

    if args.list:
        cleaner.list_cleanable_items()
    elif args.quick:
        cleaner.quick_clean()
    elif args.deep:
        cleaner.deep_clean()
    elif args.demo_only:
        cleaner.clean_demo_files()
    elif args.results_only:
        cleaner.clean_results()
    elif args.temp_only:
        cleaner.clean_temp_files()
    elif args.models_only:
        cleaner.clean_models()
    else:
        # Interactive mode
        print("=" * 50)
        print("PROJECT CLEANUP UTILITY")
        print("=" * 50)
        print("Choose cleanup option:")
        print("1. Quick clean (demo + results + temp) [SAFE]")
        print("2. Deep clean options")
        print("3. Demo files only")
        print("4. Result files only")
        print("5. Temporary files only")
        print("6. Remove models (DANGEROUS)")
        print("7. List cleanable items")
        print("8. Exit")

        choice = input("\nEnter choice (1-8): ").strip()

        if choice == "1":
            cleaner.quick_clean()
        elif choice == "2":
            cleaner.deep_clean()
        elif choice == "3":
            cleaner.clean_demo_files()
        elif choice == "4":
            cleaner.clean_results()
        elif choice == "5":
            cleaner.clean_temp_files()
        elif choice == "6":
            cleaner.clean_models()
        elif choice == "7":
            cleaner.list_cleanable_items()
        elif choice == "8":
            print("Cleanup cancelled")
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
