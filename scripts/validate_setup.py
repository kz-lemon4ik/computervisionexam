#!/usr/bin/env python3
"""
Validate project setup and dependencies
Check that everything is working correctly
"""

import sys
import importlib
from pathlib import Path


def check_dependencies():
    """Check required dependencies"""
    print("Checking dependencies...")

    required_packages = ["torch", "cv2", "numpy", "sklearn"]

    missing_packages = []

    for package in required_packages:
        try:
            if package == "cv2":
                import cv2

                print(f"OpenCV: {cv2.__version__}")
            elif package == "torch":
                import torch

                print(f"PyTorch: {torch.__version__}")
            elif package == "numpy":
                import numpy as np

                print(f"NumPy: {np.__version__}")
            elif package == "sklearn":
                import sklearn

                print(f"Scikit-learn: {sklearn.__version__}")
            else:
                importlib.import_module(package)
                print(f"{package}: Available")
        except ImportError:
            missing_packages.append(package)
            print(f"{package}: Missing")

    return len(missing_packages) == 0


def check_project_structure():
    """Check project directory structure"""
    print("\nChecking project structure...")

    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/utils",
        "src/detection",
        "src/recognition",
        "scripts",
        "tests",
        "models",
        "data",
    ]

    required_files = [
        "main.py",
        "demo.py",
        "run_tests.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "src/detection/__init__.py",
        "src/recognition/__init__.py",
        "tests/__init__.py",
    ]

    missing_items = []

    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"Directory: {dir_path}")
        else:
            missing_items.append(dir_path)
            print(f"Directory: {dir_path} (missing)")

    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"File: {file_path}")
        else:
            missing_items.append(file_path)
            print(f"File: {file_path} (missing)")

    return len(missing_items) == 0


def check_imports():
    """Check that custom modules can be imported"""
    print("\nChecking custom module imports...")

    sys.path.append("src")

    modules_to_test = [
        "data.data_loader",
        "models.baseline",
        "utils.config",
        "utils.metrics",
        "detection.yolo_model",
        "recognition.pdlpr_model",
        "pipeline",
    ]

    import_errors = []

    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"Import: {module}")
        except Exception as e:
            import_errors.append((module, str(e)))
            print(f"Import: {module} - {e}")

    return len(import_errors) == 0


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")

    try:
        sys.path.append("src")

        # Test data loader
        from data.data_loader import CCPDDataLoader

        loader = CCPDDataLoader("dummy")
        print("Data loader creation")

        # Test baseline model
        from models.baseline import BaselineCNN
        import torch

        model = BaselineCNN()
        test_input = torch.randn(1, 3, 64, 128)
        output = model(test_input)
        print(f"Baseline model forward pass: {output.shape}")

        # Test PDLPR model
        from recognition.pdlpr_model import create_pdlpr_model

        pdlpr_model = create_pdlpr_model()
        pdlpr_output = pdlpr_model(test_input)
        print(f"PDLPR model forward pass: {pdlpr_output.shape}")

        # Test pipeline
        from pipeline import create_pipeline

        pipeline = create_pipeline()
        print("Pipeline creation")

        return True

    except Exception as e:
        print(f"Functionality test failed: {e}")
        return False


def check_data_availability():
    """Check if data files are available"""
    print("\nChecking data availability...")

    data_files = ["data/raw/CCPD2019.tar.xz"]

    available_data = []

    for data_file in data_files:
        if Path(data_file).exists():
            size_mb = Path(data_file).stat().st_size / (1024 * 1024)
            print(f"Data: {data_file} ({size_mb:.1f} MB)")
            available_data.append(data_file)
        else:
            print(f"Data: {data_file} (not found)")

    return len(available_data) > 0


def main():
    """Run validation checks"""
    print("License Plate Recognition - Setup Validation")
    print("=" * 60)

    checks = [
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Module Imports", check_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Data Availability", check_data_availability),
    ]

    results = {}

    for check_name, check_func in checks:
        print(f"\n{'-' * 20} {check_name} {'-' * 20}")
        results[check_name] = check_func()

    # Summary
    print(f"\n{'=' * 60}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    passed = 0
    total = len(checks)

    for check_name, passed_check in results.items():
        status = "PASS" if passed_check else "FAIL"
        print(f"{check_name}: {status}")
        if passed_check:
            passed += 1

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("Project setup is complete and working!")
    else:
        print("Some issues need to be resolved")
        print("\nNext steps:")
        if not results.get("Dependencies", True):
            print("- Install missing dependencies: pip install -r requirements.txt")
        if not results.get("Data Availability", True):
            print("- Download CCPD dataset to data/raw/")
        print("- Run tests: python run_tests.py")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
