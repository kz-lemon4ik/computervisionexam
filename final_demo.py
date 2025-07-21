#!/usr/bin/env python3
"""
License Plate Recognition System - Final Demonstration
YOLOv5 + PDLPR Pipeline Implementation

This script demonstrates the complete license plate recognition system
combining YOLOv5 for detection and PDLPR for character recognition.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import time
import json
import argparse
from typing import List, Dict

sys.path.append("src")
from detection.yolo_inference import YOLOv5Inference
from recognition.pdlpr_inference import PDLPRInference
from models.baseline import BaselineCNN
from data.data_loader import CCPDDataLoader


class LicensePlateRecognitionDemo:
    """Complete demonstration of license plate recognition system"""

    def __init__(self, yolo_model=None, pdlpr_model=None, baseline_model=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"System running on: {self.device}")

        # Initialize models
        self.yolo_detector = YOLOv5Inference(
            yolo_model or "models/yolo_plate_detection.pt"
        )
        self.pdlpr_recognizer = PDLPRInference(
            pdlpr_model or "models/pdlpr_final.pth", device=self.device
        )

        # Load baseline for comparison
        self.baseline_model = None
        if baseline_model and Path(baseline_model).exists():
            self.baseline_model = BaselineCNN()
            state_dict = torch.load(baseline_model, map_location=self.device)
            self.baseline_model.load_state_dict(state_dict)
            self.baseline_model.to(self.device)
            self.baseline_model.eval()

        self.results_history = []

    def process_single_image(self, image_path: str) -> Dict:
        """Process single image through complete pipeline"""

        start_time = time.time()

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return {"error": f"Failed to load image: {image_path}"}

        # Step 1: License plate detection
        detection_start = time.time()
        detections = self.yolo_detector.detect_plates(str(image_path))
        detection_time = time.time() - detection_start

        # Step 2: Character recognition
        recognition_start = time.time()
        plates_recognized = []

        for detection in detections:
            bbox = detection["bbox"]
            x, y, w, h = bbox
            plate_roi = image[y : y + h, x : x + w]

            if plate_roi.size == 0:
                continue

            # PDLPR recognition
            recognized_text = ""
            if self.pdlpr_recognizer.model_loaded:
                recognized_text = self.pdlpr_recognizer.predict_text(plate_roi)

            plates_recognized.append(
                {
                    "bbox": bbox,
                    "text": recognized_text,
                    "detection_confidence": detection["confidence"],
                    "recognition_method": "PDLPR",
                }
            )

        recognition_time = time.time() - recognition_start
        total_time = time.time() - start_time

        result = {
            "image_path": str(image_path),
            "plates": plates_recognized,
            "performance": {
                "detection_time": detection_time,
                "recognition_time": recognition_time,
                "total_time": total_time,
                "plates_detected": len(plates_recognized),
            },
        }

        self.results_history.append(result)
        return result

    def batch_evaluation(self, data_dir: str, max_images: int = 50) -> Dict:
        """Evaluate system on batch of images"""

        print(f"Running batch evaluation on {max_images} images...")
        data_path = Path(data_dir)

        if not data_path.exists():
            print(f"Data directory not found: {data_dir}")
            return {}

        # Load CCPD dataset
        loader = CCPDDataLoader(str(data_path))
        annotations = loader.load_dataset(max_samples=max_images)

        if not annotations:
            print("No annotations found")
            return {}

        results = []
        total_plates = 0
        correct_detections = 0
        processing_times = []

        for i, annotation in enumerate(annotations[:max_images]):
            if i % 10 == 0:
                print(f"Processing image {i + 1}/{min(max_images, len(annotations))}")

            result = self.process_single_image(annotation["image_path"])
            if "error" not in result:
                results.append(result)
                total_plates += len(annotation.get("bbox", []))
                correct_detections += result["performance"]["plates_detected"]
                processing_times.append(result["performance"]["total_time"])

        # Calculate metrics
        detection_accuracy = (
            correct_detections / total_plates if total_plates > 0 else 0
        )
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        throughput = len(results) / sum(processing_times) if processing_times else 0

        evaluation_results = {
            "total_images_processed": len(results),
            "total_plates_expected": total_plates,
            "total_plates_detected": correct_detections,
            "detection_accuracy": detection_accuracy,
            "average_processing_time": avg_processing_time,
            "throughput_fps": throughput,
            "system_performance": {
                "min_time": min(processing_times) if processing_times else 0,
                "max_time": max(processing_times) if processing_times else 0,
                "std_time": np.std(processing_times) if processing_times else 0,
            },
        }

        return evaluation_results

    def demonstrate_system(self, demo_images: List[str] = None):
        """Live demonstration of the system"""

        print("=" * 60)
        print("LICENSE PLATE RECOGNITION SYSTEM DEMONSTRATION")
        print("=" * 60)
        print("Architecture: YOLOv5 Detection + PDLPR Recognition")
        print("Framework: PyTorch")
        print(f"Device: {self.device}")
        print()

        if demo_images is None:
            # Create demo images if none provided
            demo_images = self._create_demo_images()

        for i, image_path in enumerate(demo_images):
            print(f"Processing Image {i + 1}: {Path(image_path).name}")
            print("-" * 40)

            result = self.process_single_image(image_path)

            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            # Display results
            perf = result["performance"]
            print(f"Detection time: {perf['detection_time'] * 1000:.1f}ms")
            print(f"Recognition time: {perf['recognition_time'] * 1000:.1f}ms")
            print(f"Total processing time: {perf['total_time'] * 1000:.1f}ms")
            print(f"Plates detected: {perf['plates_detected']}")

            for j, plate in enumerate(result["plates"]):
                bbox = plate["bbox"]
                text = plate["text"]
                conf = plate["detection_confidence"]
                print(f"  Plate {j + 1}: '{text}' (confidence: {conf:.3f})")
                print(f"  Location: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")

            print()

        # Summary statistics
        if self.results_history:
            self._print_summary_statistics()

    def _create_demo_images(self) -> List[str]:
        """Create demonstration images"""

        demo_dir = Path("demo_images")
        demo_dir.mkdir(exist_ok=True)

        demo_images = []
        for i in range(3):
            # Create synthetic car image
            img = np.ones((480, 640, 3), dtype=np.uint8) * 60

            # Car body
            cv2.rectangle(img, (100, 200), (540, 400), (80, 80, 120), -1)
            cv2.rectangle(img, (120, 220), (520, 380), (100, 100, 140), -1)

            # License plate
            plate_x = 270 + i * 5
            cv2.rectangle(
                img, (plate_x, 350), (plate_x + 100, 385), (255, 255, 255), -1
            )
            cv2.rectangle(img, (plate_x, 350), (plate_x + 100, 385), (0, 0, 0), 2)

            # Save image
            demo_path = demo_dir / f"demo_vehicle_{i + 1}.jpg"
            cv2.imwrite(str(demo_path), img)
            demo_images.append(str(demo_path))

        return demo_images

    def _print_summary_statistics(self):
        """Print summary statistics for all processed images"""

        if not self.results_history:
            return

        print("=" * 60)
        print("SYSTEM PERFORMANCE SUMMARY")
        print("=" * 60)

        total_images = len(self.results_history)
        total_plates = sum(len(r["plates"]) for r in self.results_history)

        processing_times = [
            r["performance"]["total_time"] for r in self.results_history
        ]
        detection_times = [
            r["performance"]["detection_time"] for r in self.results_history
        ]
        recognition_times = [
            r["performance"]["recognition_time"] for r in self.results_history
        ]

        print(f"Images processed: {total_images}")
        print(f"Total plates detected: {total_plates}")
        print(f"Average plates per image: {total_plates / total_images:.2f}")
        print()
        print("Timing Performance:")
        print(f"  Average total time: {np.mean(processing_times) * 1000:.1f}ms")
        print(f"  Average detection time: {np.mean(detection_times) * 1000:.1f}ms")
        print(f"  Average recognition time: {np.mean(recognition_times) * 1000:.1f}ms")
        print(
            f"  System throughput: {total_images / sum(processing_times):.1f} images/sec"
        )
        print()

        # Model status
        print("Model Configuration:")
        print("  YOLOv5 Detection: Loaded")
        print(
            f"  PDLPR Recognition: {'Loaded' if self.pdlpr_recognizer.model_loaded else 'Not Loaded'}"
        )
        print(
            f"  Baseline CNN: {'Loaded' if self.baseline_model is not None else 'Not Loaded'}"
        )

    def save_results(self, output_file: str = "evaluation_results.json"):
        """Save evaluation results to file"""

        output_data = {
            "system_info": {
                "device": self.device,
                "models": {
                    "yolo_detector": "YOLOv5",
                    "character_recognizer": "PDLPR",
                    "pdlpr_loaded": self.pdlpr_recognizer.model_loaded,
                },
            },
            "results": self.results_history,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="License Plate Recognition System Demo"
    )
    parser.add_argument("--demo", action="store_true", help="Run live demonstration")
    parser.add_argument("--evaluate", type=str, help="Run batch evaluation on dataset")
    parser.add_argument(
        "--images", type=str, nargs="+", help="Specific images to process"
    )
    parser.add_argument(
        "--max_images", type=int, default=50, help="Maximum images for evaluation"
    )
    parser.add_argument(
        "--save_results",
        type=str,
        default="evaluation_results.json",
        help="File to save results",
    )

    args = parser.parse_args()

    # Initialize system
    demo_system = LicensePlateRecognitionDemo()

    if args.demo:
        # Live demonstration
        demo_images = args.images if args.images else None
        demo_system.demonstrate_system(demo_images)

    elif args.evaluate:
        # Batch evaluation
        results = demo_system.batch_evaluation(args.evaluate, args.max_images)

        print("\nEVALUATION RESULTS:")
        print("=" * 40)
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value:.4f}")
            else:
                print(
                    f"{key}: {value:.4f}"
                    if isinstance(value, float)
                    else f"{key}: {value}"
                )

    elif args.images:
        # Process specific images
        for image_path in args.images:
            result = demo_system.process_single_image(image_path)
            print(f"Processed: {image_path}")
            if "error" not in result:
                perf = result["performance"]
                print(f"  Processing time: {perf['total_time'] * 1000:.1f}ms")
                print(f"  Plates detected: {perf['plates_detected']}")

    else:
        print("Please specify --demo, --evaluate, or --images")
        return

    # Save results
    if args.save_results and demo_system.results_history:
        demo_system.save_results(args.save_results)


if __name__ == "__main__":
    main()
