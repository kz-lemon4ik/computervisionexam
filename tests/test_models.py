#!/usr/bin/env python3
"""
Test model functionality
"""

import unittest
import torch
import numpy as np
import sys

sys.path.append('src')
from models.baseline import BaselineCNN, SyntheticDataset
from recognition.pdlpr_model import create_pdlpr_model
from utils.metrics import MetricsTracker, calculate_metrics

class TestBaselineModel(unittest.TestCase):
    """Test baseline CNN model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = BaselineCNN()
        self.batch_size = 4
        self.test_input = torch.randn(self.batch_size, 3, 64, 128)
    
    def test_model_creation(self):
        """Test model creation"""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.num_chars, 67)
        self.assertEqual(self.model.sequence_length, 7)
    
    def test_forward_pass(self):
        """Test forward pass"""
        output = self.model(self.test_input)
        
        # Check output shape
        expected_shape = (self.batch_size, 7, 67)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is valid probabilities
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_model_parameters(self):
        """Test model has trainable parameters"""
        params = list(self.model.parameters())
        self.assertGreater(len(params), 0)
        
        total_params = sum(p.numel() for p in params)
        self.assertGreater(total_params, 1000)  # Reasonable number of parameters

class TestPDLPRModel(unittest.TestCase):
    """Test PDLPR model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = create_pdlpr_model()
        self.batch_size = 2
        self.test_input = torch.randn(self.batch_size, 3, 64, 128)
    
    def test_model_creation(self):
        """Test PDLPR model creation"""
        self.assertIsNotNone(self.model)
    
    def test_forward_pass(self):
        """Test forward pass"""
        output = self.model(self.test_input)
        
        # Check output shape
        expected_shape = (self.batch_size, 7, 67)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output is valid
        self.assertTrue(torch.all(torch.isfinite(output)))

class TestSyntheticDataset(unittest.TestCase):
    """Test synthetic dataset"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.dataset = SyntheticDataset(num_samples=10)
    
    def test_dataset_length(self):
        """Test dataset length"""
        self.assertEqual(len(self.dataset), 10)
    
    def test_dataset_item(self):
        """Test dataset item retrieval"""
        image, labels = self.dataset[0]
        
        # Check image tensor
        self.assertEqual(image.shape, (3, 64, 128))
        self.assertTrue(torch.all(image >= 0) and torch.all(image <= 1))
        
        # Check labels
        self.assertEqual(labels.shape, (7,))
        self.assertTrue(torch.all(labels >= 0) and torch.all(labels < 67))

class TestMetrics(unittest.TestCase):
    """Test metrics calculation"""
    
    def test_character_accuracy(self):
        """Test character accuracy calculation"""
        from utils.metrics import character_accuracy
        
        # Perfect match
        pred = np.array([[1, 2, 3], [4, 5, 6]])
        target = np.array([[1, 2, 3], [4, 5, 6]])
        acc = character_accuracy(pred, target)
        self.assertEqual(acc, 1.0)
        
        # Partial match
        pred = np.array([[1, 2, 3], [4, 5, 6]])
        target = np.array([[1, 2, 0], [4, 0, 6]])
        acc = character_accuracy(pred, target)
        self.assertEqual(acc, 4/6)  # 4 correct out of 6
    
    def test_sequence_accuracy(self):
        """Test sequence accuracy calculation"""
        from utils.metrics import sequence_accuracy
        
        # Perfect match
        pred = np.array([[1, 2, 3], [4, 5, 6]])
        target = np.array([[1, 2, 3], [4, 5, 6]])
        acc = sequence_accuracy(pred, target)
        self.assertEqual(acc, 1.0)
        
        # One sequence wrong
        pred = np.array([[1, 2, 3], [4, 5, 6]])
        target = np.array([[1, 2, 0], [4, 5, 6]])
        acc = sequence_accuracy(pred, target)
        self.assertEqual(acc, 0.5)  # 1 correct out of 2
    
    def test_metrics_tracker(self):
        """Test metrics tracker"""
        tracker = MetricsTracker()
        
        # Add some data
        pred = torch.randn(2, 7, 67)
        target = torch.randint(0, 67, (2, 7))
        
        tracker.update(pred, target, 0.5)
        
        # Check metrics computation
        metrics = tracker.compute()
        self.assertIn('character_accuracy', metrics)
        self.assertIn('sequence_accuracy', metrics)
        self.assertIn('avg_loss', metrics)

if __name__ == '__main__':
    unittest.main()