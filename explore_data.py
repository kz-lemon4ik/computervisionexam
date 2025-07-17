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
    data_path = "data/processed/ccpd_subset"
    
    if not Path(data_path).exists():
        print(f"Dataset not found at {data_path}")
        print("Please check dataset path")
        return
    
    # Initialize loader
    loader = CCPDDataLoader(data_path)
    
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
    
    # Load actual dataset
    annotations = loader.load_dataset(max_samples=1000)
    
    print("Sample filename parsing:")
    for i, filename in enumerate(sample_filenames, 1):
        print(f"Sample {i}:")
        result = loader.parse_filename(filename)
        if result:
            print(f"  Plate text: {result['plate_text']}")
            print(f"  Characters: {result['characters']}")
            print(f"  Bbox shape: {result['bbox'].shape}")
        print()
    
    if annotations:
        print(f"Loaded {len(annotations)} samples")
        
        # Character statistics
        char_counts = {}
        plate_lengths = []
        
        for ann in annotations[:100]:  # Sample for stats
            for char in ann['characters']:
                char_counts[char] = char_counts.get(char, 0) + 1
            plate_lengths.append(len(ann['characters']))
        
        print(f"Average plate length: {sum(plate_lengths)/len(plate_lengths):.1f}")
        print(f"Most common characters: {sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
        
        # Train/val split test
        train_data, val_data = loader.get_train_val_split(annotations)
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
    
    print("\nDataset structure:")
    print("  Format: Area-AgentType-TimeOfDay&Province_coordinates-bbox_coordinates-landmarks-characters-brightness-blur.jpg")
    print("  Processing: Reduced subset for development")
    print("  Image format: JPEG")

if __name__ == "__main__":
    explore_ccpd_dataset()