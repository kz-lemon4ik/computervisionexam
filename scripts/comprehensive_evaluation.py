#!/usr/bin/env python3
"""
Comprehensive evaluation script for license plate recognition system
Provides detailed performance analysis and comparison metrics
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
import torch

sys.path.append("src")
from detection.yolo_inference import YOLOv5Inference
from recognition.pdlpr_inference import PDLPRInference
from models.baseline import BaselineCNN
from data.data_loader import CCPDDataLoader


class ComprehensiveEvaluator:
    """Complete system evaluation and analysis"""

    def __init__(self, data_dir: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_dir = Path(data_dir)

        # Load test dataset
        self.loader = CCPDDataLoader(str(self.data_dir))
        self.test_annotations = self.loader.load_dataset(max_samples=200)

        # Initialize models
        self.yolo_detector = YOLOv5Inference("models/yolo_plate_detection.pt")
        self.pdlpr_recognizer = PDLPRInference(
            "models/pdlpr_final.pth", device=self.device
        )

        # Load baseline if available
        self.baseline_model = None
        if Path("models/baseline_model.pth").exists():
            self.baseline_model = BaselineCNN()
            state_dict = torch.load(
                "models/baseline_model.pth", map_location=self.device
            )
            self.baseline_model.load_state_dict(state_dict)
            self.baseline_model.to(self.device)
            self.baseline_model.eval()

    def evaluate_detection_performance(self) -> Dict:
        """Evaluate YOLOv5 detection performance"""

        print("Evaluating detection performance...")

        total_images = 0
        images_with_detections = 0
        detection_times = []

        for annotation in self.test_annotations[:50]:
            image_path = annotation["image_path"]

            total_images += 1

            # Run detection
            start_time = time.time()
            detections = self.yolo_detector.detect_plates(image_path)
            detection_time = time.time() - start_time
            detection_times.append(detection_time)

            if len(detections) > 0:
                images_with_detections += 1

        results = {
            "total_images_tested": total_images,
            "images_with_detections": images_with_detections,
            "detection_success_rate": images_with_detections / total_images
            if total_images > 0
            else 0,
            "average_detection_time": np.mean(detection_times),
            "detection_fps": 1 / np.mean(detection_times) if detection_times else 0,
        }

        return results

    def evaluate_recognition_performance(self) -> Dict:
        """Evaluate character recognition performance"""

        print("Evaluating recognition performance...")

        total_characters = 0
        correct_characters = 0
        total_sequences = 0
        correct_sequences = 0
        recognition_times = []

        for annotation in self.test_annotations[:50]:
            image_path = annotation["image_path"]
            ground_truth_text = annotation.get("chars", "")

            if len(str(ground_truth_text)) == 0:
                continue

            # Get detected plates
            detections = self.yolo_detector.detect_plates(image_path)

            if not detections:
                continue

            # Process first detection
            detection = detections[0]
            bbox = detection["bbox"]

            # Extract plate region
            import cv2

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            x, y, w, h = bbox
            plate_roi = image[y : y + h, x : x + w]

            if plate_roi.size == 0:
                continue

            # Run recognition
            start_time = time.time()
            recognized_text = self.pdlpr_recognizer.predict_text(plate_roi)
            recognition_time = time.time() - start_time
            recognition_times.append(recognition_time)

            # Calculate metrics
            total_sequences += 1
            if recognized_text == ground_truth_text:
                correct_sequences += 1

            # Character-level accuracy
            min_len = min(len(recognized_text), len(ground_truth_text))
            for i in range(min_len):
                total_characters += 1
                if recognized_text[i] == ground_truth_text[i]:
                    correct_characters += 1

            # Add remaining characters as incorrect
            total_characters += abs(len(recognized_text) - len(ground_truth_text))

        results = {
            "character_accuracy": correct_characters / total_characters
            if total_characters > 0
            else 0,
            "sequence_accuracy": correct_sequences / total_sequences
            if total_sequences > 0
            else 0,
            "average_recognition_time": np.mean(recognition_times)
            if recognition_times
            else 0,
            "recognition_fps": 1 / np.mean(recognition_times)
            if recognition_times
            else 0,
            "total_sequences_tested": total_sequences,
        }

        return results

    def evaluate_end_to_end_performance(self) -> Dict:
        """Evaluate complete pipeline performance"""

        print("Evaluating end-to-end pipeline...")

        successful_processing = 0
        total_processing_time = []
        error_count = 0

        for annotation in self.test_annotations[:30]:
            image_path = annotation["image_path"]

            try:
                start_time = time.time()

                # Detection
                detections = self.yolo_detector.detect_plates(image_path)

                # Recognition
                recognized_plates = []
                if detections:
                    import cv2

                    image = cv2.imread(str(image_path))
                    if image is not None:
                        for detection in detections:
                            bbox = detection["bbox"]
                            x, y, w, h = bbox
                            plate_roi = image[y : y + h, x : x + w]

                            if plate_roi.size > 0:
                                recognized_text = self.pdlpr_recognizer.predict_text(
                                    plate_roi
                                )
                                recognized_plates.append(recognized_text)

                processing_time = time.time() - start_time
                total_processing_time.append(processing_time)

                if recognized_plates:
                    successful_processing += 1

            except Exception as e:
                error_count += 1
                print(f"Error processing {image_path}: {e}")

        results = {
            "successful_processing_rate": successful_processing
            / len(self.test_annotations[:30]),
            "average_total_time": np.mean(total_processing_time)
            if total_processing_time
            else 0,
            "system_throughput": 1 / np.mean(total_processing_time)
            if total_processing_time
            else 0,
            "error_rate": error_count / len(self.test_annotations[:30]),
            "total_images_tested": len(self.test_annotations[:30]),
        }

        return results

    def _calculate_iou(self, box1: List, box2) -> float:
        """Calculate Intersection over Union (IoU) between two boxes"""

        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1

        # Handle different formats for box2
        try:
            if hasattr(box2, "__len__") and len(box2) >= 4:
                # Format: [x, y, w, h] or [x1, y1, x2, y2]
                if len(box2) == 4:
                    x1_2, y1_2, w2, h2 = box2
                    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
                else:
                    x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[2], box2[3]
            elif hasattr(box2, "__len__") and len(box2) == 2:
                # Format: [[x1,y1], [x2,y2]]
                if hasattr(box2[0], "__len__"):
                    x1_2, y1_2 = box2[0]
                    x2_2, y2_2 = box2[1]
                else:
                    # Format: [x, y] - skip this case
                    return 0.0
            else:
                return 0.0
        except:
            return 0.0

        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def generate_comprehensive_report(self) -> Dict:
        """Generate complete evaluation report"""

        print("=" * 60)
        print("COMPREHENSIVE SYSTEM EVALUATION")
        print("=" * 60)

        # Run all evaluations
        detection_results = self.evaluate_detection_performance()
        recognition_results = self.evaluate_recognition_performance()
        pipeline_results = self.evaluate_end_to_end_performance()

        # Compile complete report
        report = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_configuration": {
                "device": self.device,
                "yolo_model": "YOLOv5",
                "recognition_model": "PDLPR",
                "dataset": "CCPD",
                "test_images": len(self.test_annotations),
            },
            "detection_performance": detection_results,
            "recognition_performance": recognition_results,
            "pipeline_performance": pipeline_results,
        }

        # Print summary
        self._print_evaluation_summary(report)

        return report

    def _print_evaluation_summary(self, report: Dict):
        """Print formatted evaluation summary"""

        print("\nDETECTION PERFORMANCE:")
        print("-" * 30)
        det = report["detection_performance"]
        print(f"Success Rate: {det['detection_success_rate']:.3f}")
        print(
            f"Images with detections: {det['images_with_detections']}/{det['total_images_tested']}"
        )
        print(f"Detection Speed: {det['detection_fps']:.1f} FPS")

        print("\nRECOGNITION PERFORMANCE:")
        print("-" * 30)
        rec = report["recognition_performance"]
        print(f"Character Accuracy: {rec['character_accuracy']:.3f}")
        print(f"Sequence Accuracy: {rec['sequence_accuracy']:.3f}")
        print(f"Recognition Speed: {rec['recognition_fps']:.1f} FPS")

        print("\nPIPELINE PERFORMANCE:")
        print("-" * 30)
        pipe = report["pipeline_performance"]
        print(f"Success Rate: {pipe['successful_processing_rate']:.3f}")
        print(f"System Throughput: {pipe['system_throughput']:.1f} images/sec")
        print(f"Error Rate: {pipe['error_rate']:.3f}")

        print("\n" + "=" * 60)

    def save_report(
        self, report: Dict, filename: str = "comprehensive_evaluation.json"
    ):
        """Save evaluation report to file"""

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Evaluation report saved to: {filename}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive System Evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/ccpd_subset",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comprehensive_evaluation.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(args.data_dir)

    # Generate report
    report = evaluator.generate_comprehensive_report()

    # Save results
    evaluator.save_report(report, args.output)


if __name__ == "__main__":
    main()
