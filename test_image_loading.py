#!/usr/bin/env python3

import os
import cv2
from pathlib import Path

def test_image_loading():
    print(f"Current working directory: {os.getcwd()}")
    
    # Test different ways to load images
    base_paths = [
        "/home/lemon/untitled folder/computervisionexam/data/processed/ccpd_subset",
        "data/processed/ccpd_subset"
    ]
    
    # Get first few image files
    full_path = Path("/home/lemon/untitled folder/computervisionexam/data/processed/ccpd_subset")
    image_files = list(full_path.glob('*.jpg'))[:3]
    
    print(f"Found {len(image_files)} image files")
    
    for base_path in base_paths:
        print(f"\n=== Testing base path: {base_path} ===")
        
        for img_file in image_files:
            filename = img_file.name
            
            # Test different path constructions
            test_paths = [
                os.path.join(base_path, filename),
                f"{base_path}/{filename}",
                str(Path(base_path) / filename),
                str(img_file)  # Full absolute path
            ]
            
            for test_path in test_paths:
                print(f"  Testing: {test_path}")
                print(f"    Path exists: {Path(test_path).exists()}")
                
                try:
                    image = cv2.imread(test_path)
                    if image is not None:
                        print(f"    ✓ Loaded successfully: {image.shape}")
                    else:
                        print(f"    ✗ cv2.imread returned None")
                except Exception as e:
                    print(f"    ✗ Exception: {e}")
                print()
                
            break  # Only test first image for each base path

if __name__ == "__main__":
    test_image_loading()