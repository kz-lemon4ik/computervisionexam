#!/usr/bin/env python3
"""
Project setup script
Initialize project structure and validate environment
"""

import sys
from pathlib import Path


def create_directories():
    """Create necessary project directories"""
    dirs = [
        "data/raw",
        "data/processed",
        "data/yolo/train/images",
        "data/yolo/train/labels",
        "data/yolo/val/images",
        "data/yolo/val/labels",
        "models/optimized",
        "demo_images/input",
        "demo_images/output",
        "results",
        "logs",
    ]

    print("Creating project directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"{dir_path}")

    print("Directories created successfully!")


def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8+ required")
        return False

    print(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install required packages"""
    print("Installing requirements...")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Requirements installed")
            return True
        else:
            print(f"Installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Installation error: {e}")
        return False


def download_sample_data():
    """Create sample data for testing"""
    import cv2
    import numpy as np

    print("Creating sample data...")

    # Create sample car image
    sample_dir = Path("demo_images/input")
    sample_path = sample_dir / "sample_car.jpg"

    if not sample_path.exists():
        # Generate synthetic car image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # Add car body
        cv2.rectangle(img, (100, 200), (540, 400), (80, 80, 120), -1)

        # Add license plate
        cv2.rectangle(img, (220, 320), (420, 380), (255, 255, 255), -1)
        cv2.rectangle(img, (220, 320), (420, 380), (0, 0, 0), 2)

        # Add some text-like features
        cv2.putText(
            img, "ABC123", (240, 355), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )

        cv2.imwrite(str(sample_path), img)
        print(f"Created sample image: {sample_path}")
    else:
        print(f"Sample image exists: {sample_path}")


def validate_setup():
    """Validate project setup"""
    print("Validating setup...")

    try:
        # Test imports
        import torch

        print("Core dependencies available")

        # Test project modules
        sys.path.append("src")
        from utils.config import ALL_CHARS

        print(f"Project modules accessible ({len(ALL_CHARS)} characters)")

        # Test basic functionality
        from models.baseline import BaselineCNN

        model = BaselineCNN()
        test_input = torch.randn(1, 3, 64, 128)
        output = model(test_input)
        print(f"Models functional (output: {output.shape})")

        return True

    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def main():
    """Main setup function"""
    print("License Plate Recognition - Project Setup")
    print("=" * 50)

    success_steps = 0
    total_steps = 5

    # Step 1: Check Python version
    if check_python_version():
        success_steps += 1

    # Step 2: Create directories
    try:
        create_directories()
        success_steps += 1
    except Exception as e:
        print(f"Directory creation failed: {e}")

    # Step 3: Install requirements
    if install_requirements():
        success_steps += 1

    # Step 4: Create sample data
    try:
        download_sample_data()
        success_steps += 1
    except Exception as e:
        print(f"Sample data creation failed: {e}")

    # Step 5: Validate setup
    if validate_setup():
        success_steps += 1

    # Summary
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    print(f"Completed: {success_steps}/{total_steps} steps")

    if success_steps == total_steps:
        print("Project setup completed successfully!")
        print("\nNext steps:")
        print("1. Download CCPD dataset to data/raw/")
        print("2. Run: python main.py --demo")
        print("3. Run: python run_tests.py")
        print("4. Start training: python src/models/baseline.py --train")
        return True
    else:
        print("Setup incomplete - some steps failed")
        print("\nTroubleshooting:")
        print("- Check Python version (3.8+ required)")
        print("- Install dependencies manually: pip install -r requirements.txt")
        print("- Check file permissions")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
