#!/usr/bin/env python3

import os
import sys
from pathlib import Path

print("Debug: Checking paths and files")
print("=" * 50)

# Check current working directory
print(f"Current working directory: {os.getcwd()}")

# Check if the data path exists
data_path = Path("/home/lemon/untitled folder/computervisionexam/data/processed/ccpd_subset")
print(f"Data path exists: {data_path.exists()}")

if data_path.exists():
    # List first 10 files
    image_files = list(data_path.glob('*.jpg'))[:10]
    print(f"Found {len(image_files)} image files (showing first 10):")
    
    for i, img_file in enumerate(image_files):
        print(f"  {i+1}: {img_file.name}")
        
    # Check for the specific problematic file
    problematic_file = "0566-21_43-384&365_637&552-598&466_384&552_423&451_637&365-0_0_24_24_15_30_32-85-65.jpg"
    problematic_path = data_path / problematic_file
    print(f"\nProblematic file exists: {problematic_path.exists()}")
    print(f"Problematic file path: {problematic_path}")
    
    # Search for any files starting with 0566
    files_0566 = list(data_path.glob('0566*'))
    print(f"Files starting with 0566: {len(files_0566)}")
    for f in files_0566:
        print(f"  {f.name}")
else:
    print("Data directory does not exist!")

print("=" * 50)