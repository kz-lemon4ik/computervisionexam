#!/usr/bin/env python3
"""
Simple test for CCPD data loader
"""

import sys
sys.path.append('src')

from data.data_loader import CCPDDataLoader

def test_filename_parsing():
    """Test CCPD filename parsing functionality"""
    
    loader = CCPDDataLoader("dummy_path")
    
    # Test case
    test_filename = "0849-20_39-202&370_540&580-540&580_235&465_202&370_507&485-0_0_28_26_28_28_31-95-382.jpg"
    
    print("Testing CCPD filename parsing...")
    print(f"Input: {test_filename}")
    
    result = loader.parse_filename(test_filename)
    
    if result:
        print("✓ Parsing successful")
        print(f"  Plate text: {result['plate_text']}")
        print(f"  Number of characters: {len(result['characters'])}")
        print(f"  Bbox coordinates: {result['bbox'].tolist()}")
        
        # Basic validation
        assert len(result['characters']) == 7, "Expected 7 characters"
        assert result['bbox'].shape == (2, 2), "Expected 2x2 bbox"
        assert len(result['plate_text']) == 7, "Plate text length mismatch"
        
        print("✓ All assertions passed")
        
    else:
        print("✗ Parsing failed")
        return False
    
    return True

def test_character_mapping():
    """Test character vocabulary"""
    
    loader = CCPDDataLoader("dummy_path")
    
    print("\nTesting character mapping...")
    print(f"Chinese characters: {len(loader.chinese_chars)}")
    print(f"Alphanumeric: {len(loader.alphanumeric)}")
    print(f"Total vocabulary: {len(loader.all_chars)}")
    
    # Check no duplicates
    all_chars_set = set(loader.all_chars)
    assert len(all_chars_set) == len(loader.all_chars), "Duplicate characters found"
    
    print("✓ Character mapping tests passed")
    
    return True

def test_dataset_loading():
    """Test loading actual dataset"""
    from pathlib import Path
    
    data_path = "data/processed/ccpd_subset"
    
    print(f"\nTesting dataset loading from {data_path}...")
    
    if not Path(data_path).exists():
        print(f"✗ Dataset path not found: {data_path}")
        return False
        
    loader = CCPDDataLoader(data_path)
    
    try:
        annotations = loader.load_dataset(max_samples=10)
        
        if annotations:
            print(f"✓ Loaded {len(annotations)} samples")
            
            # Test first annotation
            first = annotations[0]
            print(f"  Sample plate: {first['plate_text']}")
            print(f"  Characters: {first['characters']}")
            print(f"  Image path exists: {Path(first['image_path']).exists()}")
            
            # Test train/val split
            train_data, val_data = loader.get_train_val_split(annotations)
            print(f"  Train samples: {len(train_data)}")
            print(f"  Val samples: {len(val_data)}")
            
            print("✓ Dataset loading tests passed")
            return True
        else:
            print("✗ No annotations loaded")
            return False
            
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

if __name__ == "__main__":
    print("CCPD Data Loader Tests")
    print("=" * 30)
    
    success = True
    success &= test_filename_parsing()
    success &= test_character_mapping()
    success &= test_dataset_loading()
    
    print("\n" + "=" * 30)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)