#!/usr/bin/env python3
"""
CCPD dataset exploration script
Basic analysis of Chinese City Parking Dataset
"""

import sys
from pathlib import Path
sys.path.append('src')

from data.data_loader import CCPDDataLoader

def explore_ccpd_dataset():
    """Basic exploration of CCPD dataset"""
    
    # Dataset path
    archive_path = "data/raw/CCPD2019.tar.xz"
    
    if not Path(archive_path).exists():
        print(f"Dataset not found at {archive_path}")
        print("Please download CCPD dataset first")
        return
    
    # Initialize loader
    loader = CCPDDataLoader(archive_path)
    
    print("CCPD Dataset Exploration")
    print("=" * 40)
    
    # Test filename parsing
    sample_filenames = [
        "0849-20_39-202&370_540&580-540&580_235&465_202&370_507&485-0_0_28_26_28_28_31-95-382.jpg",
        "0047-5_1-424&451_523&491-523&482_428&491_424&460_519&451-0_0_26_22_25_24_27-121-15.jpg"
    ]
    
    print(f"Character mapping:")
    print(f"  Chinese provinces: {len(loader.chinese_chars)} characters")
    print(f"  Alphanumeric: {len(loader.alphanumeric)} characters") 
    print(f"  Total vocabulary: {len(loader.all_chars)} characters")
    print()
    
    print("Sample filename parsing:")
    for i, filename in enumerate(sample_filenames, 1):
        print(f"Sample {i}:")
        result = loader.parse_filename(filename)
        if result:
            print(f"  Plate text: {result['plate_text']}")
            print(f"  Characters: {result['characters']}")
            print(f"  Bbox shape: {result['bbox'].shape}")
        print()
    
    print("Dataset structure:")
    print("  Format: Area-AgentType-TimeOfDay&Province_coordinates-bbox_coordinates-landmarks-characters-brightness-blur.jpg")
    print("  Archive format: tar.xz compressed")
    print("  Image format: JPEG")

if __name__ == "__main__":
    explore_ccpd_dataset()