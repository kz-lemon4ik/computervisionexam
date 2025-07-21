#!/usr/bin/env python3
"""
Test pipeline functionality
"""

import unittest
import numpy as np
import cv2
import sys

sys.path.append("src")
from pipeline import create_pipeline, IntegratedPipeline
from detection.yolo_model import YOLOv5Detector


class TestYOLODetector(unittest.TestCase):
    """Test YOLO detection functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.detector = YOLOv5Detector()
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # Add mock license plate
        cv2.rectangle(self.test_image, (200, 300), (440, 380), (255, 255, 255), -1)

    def test_detector_creation(self):
        """Test detector creation"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.5)

    def test_plate_detection(self):
        """Test plate detection"""
        detections = self.detector.detect_plates(self.test_image)

        self.assertIsInstance(detections, list)
        self.assertGreater(len(detections), 0)

        # Check detection format
        detection = detections[0]
        self.assertIn("bbox", detection)
        self.assertIn("confidence", detection)
        self.assertIn("class", detection)

    def test_plate_cropping(self):
        """Test plate cropping"""
        detections = self.detector.detect_plates(self.test_image)
        crops = self.detector.crop_plates(self.test_image, detections)

        self.assertIsInstance(crops, list)
        self.assertGreater(len(crops), 0)

        # Check crop format
        crop = crops[0]
        self.assertIn("image", crop)
        self.assertIn("bbox", crop)
        self.assertIn("confidence", crop)

        # Check crop image is valid
        crop_image = crop["image"]
        self.assertGreater(crop_image.size, 0)


class TestIntegratedPipeline(unittest.TestCase):
    """Test integrated pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = create_pipeline()
        self.test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100

        # Add mock license plate
        cv2.rectangle(self.test_image, (200, 300), (440, 380), (255, 255, 255), -1)

    def test_pipeline_creation(self):
        """Test pipeline creation"""
        self.assertIsNotNone(self.pipeline)
        self.assertIsInstance(self.pipeline, IntegratedPipeline)

    def test_end_to_end_recognition(self):
        """Test end-to-end recognition"""
        results = self.pipeline.detect_and_recognize(self.test_image)

        self.assertIsInstance(results, list)

        if results:  # If detection succeeded
            result = results[0]
            self.assertIn("plate_text", result)
            self.assertIn("bbox", result)
            self.assertIn("detection_confidence", result)
            self.assertIn("recognition_confidence", result)
            self.assertIn("method", result)

            # Check text format
            plate_text = result["plate_text"]
            self.assertIsInstance(plate_text, str)
            self.assertGreater(len(plate_text), 0)

    def test_batch_processing(self):
        """Test batch processing"""
        # Create multiple test images
        test_images = [self.test_image for _ in range(3)]

        results = self.pipeline.process_batch(test_images)

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)

    def test_performance_tracking(self):
        """Test performance tracking"""
        # Process an image to generate stats
        self.pipeline.detect_and_recognize(self.test_image)

        # Get performance stats
        stats = self.pipeline.get_performance_stats()

        self.assertIsInstance(stats, dict)
        # Should have some timing information
        self.assertGreater(len(stats), 0)


class TestPipelineValidation(unittest.TestCase):
    """Test pipeline validation and error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = create_pipeline()

    def test_invalid_image_handling(self):
        """Test handling of invalid images"""
        # Test with None
        results = self.pipeline.detect_and_recognize(None)
        self.assertEqual(results, [])

        # Test with invalid path
        results = self.pipeline.detect_and_recognize("nonexistent.jpg")
        self.assertEqual(results, [])

    def test_empty_detection(self):
        """Test handling when no plates are detected"""
        # Create image with no plate-like features
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)

        results = self.pipeline.detect_and_recognize(empty_image)
        # Should return empty list or handle gracefully
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTest(unittest.makeSuite(TestYOLODetector))
    suite.addTest(unittest.makeSuite(TestIntegratedPipeline))
    suite.addTest(unittest.makeSuite(TestPipelineValidation))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
