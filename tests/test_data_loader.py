#!/usr/bin/env python3
"""
Test data loading functionality
"""

import unittest
import sys
from pathlib import Path

sys.path.append('src')
from data.data_loader import CCPDDataLoader

class TestCCPDDataLoader(unittest.TestCase):
    """Test CCPD data loader"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = CCPDDataLoader("dummy_path")
    
    def test_character_mapping(self):
        """Test character vocabulary"""
        # Test Chinese characters
        self.assertEqual(len(self.loader.chinese_chars), 31)
        self.assertIn("京", self.loader.chinese_chars)
        self.assertIn("沪", self.loader.chinese_chars)
        
        # Test alphanumeric
        self.assertEqual(len(self.loader.alphanumeric), 36)
        self.assertIn("0", self.loader.alphanumeric)
        self.assertIn("A", self.loader.alphanumeric)
        
        # Test total vocabulary
        self.assertEqual(len(self.loader.all_chars), 67)
        
        # Test no duplicates
        self.assertEqual(len(set(self.loader.all_chars)), len(self.loader.all_chars))
    
    def test_filename_parsing(self):
        """Test CCPD filename parsing"""
        test_filename = "0849-20_39-202&370_540&580-540&580_235&465_202&370_507&485-0_0_28_26_28_28_31-95-382.jpg"
        
        result = self.loader.parse_filename(test_filename)
        
        self.assertIsNotNone(result)
        self.assertIn('plate_text', result)
        self.assertIn('characters', result)
        self.assertIn('bbox', result)
        
        # Check character count
        self.assertEqual(len(result['characters']), 7)
        self.assertEqual(len(result['plate_text']), 7)
        
        # Check bbox shape
        self.assertEqual(result['bbox'].shape, (2, 2))
    
    def test_invalid_filename(self):
        """Test handling of invalid filenames"""
        invalid_filenames = [
            "invalid.jpg",
            "123.jpg",
            "not-a-ccpd-filename.jpg"
        ]
        
        for filename in invalid_filenames:
            result = self.loader.parse_filename(filename)
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()