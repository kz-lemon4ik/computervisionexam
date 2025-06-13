#!/usr/bin/env python3
"""
License Plate Recognition Demo
Simple demo script for testing the pipeline
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.append('src')
from main import LicensePlateRecognizer, create_demo_image

def run_demo():
    """Run basic demo of license plate recognition"""
    print("License Plate Recognition Demo")
    print("=" * 30)
    
    # Create demo image
    demo_image = create_demo_image()
    
    # Initialize recognizer
    recognizer = LicensePlateRecognizer()
    
    # Process image
    plate_text, confidence, info = recognizer.recognize_plate(demo_image)
    
    print(f"Demo image: {demo_image}")
    print(f"Recognition result: {plate_text}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Characters: {info.get('characters', [])}")
    
    print("\nDemo completed!")
    print("To train the model: python src/models/baseline.py --train")
    print("To process your image: python main.py --input your_image.jpg")

if __name__ == "__main__":
    run_demo()