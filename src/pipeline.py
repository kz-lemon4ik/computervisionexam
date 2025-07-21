"""
Integrated pipeline for license plate recognition
Combines YOLOv5 detection and PDLPR recognition
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append("src")
from recognition.pdlpr_model import create_pdlpr_model, PDLPRTrainer
from models.baseline import BaselineCNN
from utils.config import ALL_CHARS


class IntegratedPipeline:
    """Full license plate recognition pipeline"""

    def __init__(
        self,
        detection_model_path=None,
        recognition_model_path=None,
        baseline_model_path="models/baseline_model.pth",
        device="cpu",
    ):
        self.device = torch.device(device)
        self.char_mapping = ALL_CHARS

        # Initialize performance profiler
        self.profiler = PerformanceProfiler()

        # Initialize optimized image processor
        self.image_processor = ImageProcessor()

        # Initialize detection module
        self.detector = YOLOv5Detector(detection_model_path, device)

        # Initialize recognition modules
        self.recognition_model = None
        self.baseline_model = None

        # Load models
        self._load_models(recognition_model_path, baseline_model_path)

        # Apply optimizations
        self._apply_optimizations()

    def _load_models(self, recognition_path, baseline_path):
        """Load recognition models"""

        # Load PDLPR model if available
        if recognition_path and Path(recognition_path).exists():
            try:
                self.recognition_model = create_pdlpr_model()
                trainer = PDLPRTrainer(self.recognition_model, self.device)
                trainer.load_model(recognition_path)
                print(f"Loaded PDLPR model from {recognition_path}")
            except Exception as e:
                print(f"Failed to load PDLPR model: {e}")

        # Load baseline model as fallback
        if baseline_path and Path(baseline_path).exists():
            try:
                self.baseline_model = BaselineCNN()
                state_dict = torch.load(baseline_path, map_location=self.device)
                self.baseline_model.load_state_dict(state_dict)
                self.baseline_model.to(self.device)
                self.baseline_model.eval()
                print(f"Loaded baseline model from {baseline_path}")
            except Exception as e:
                print(f"Failed to load baseline model: {e}")

    def _apply_optimizations(self):
        """Apply CPU optimizations to models"""
        if self.baseline_model:
            self.baseline_model = ModelOptimizer.optimize_for_cpu(self.baseline_model)

        if self.recognition_model:
            self.recognition_model = ModelOptimizer.optimize_for_cpu(
                self.recognition_model
            )

        # Set optimal CPU settings
        torch.set_num_threads(2)

    def detect_and_recognize(self, image_path, use_pdlpr=True):
        """
        Full pipeline: detection + recognition
        Returns: list of recognized plates with metadata
        """
        results = []

        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
        else:
            image = image_path

        if image is None:
            return results

        # Step 1: Detect license plates
        self.profiler.start_timer("detection")
        detections = self.detector.detect_plates(image)
        self.profiler.end_timer("detection")

        # Step 2: Extract plate regions
        self.profiler.start_timer("cropping")
        plate_crops = self.detector.crop_plates(image, detections)
        self.profiler.end_timer("cropping")

        # Step 3: Recognize characters in each plate
        for i, plate_data in enumerate(plate_crops):
            plate_image = plate_data["image"]
            bbox = plate_data["bbox"]
            detection_confidence = plate_data["confidence"]

            # Recognize characters
            self.profiler.start_timer("recognition")
            plate_text, recognition_confidence, method = self._recognize_plate(
                plate_image, use_pdlpr
            )
            self.profiler.end_timer("recognition")
            self.profiler.increment_counter("plates_processed")

            result = {
                "plate_text": plate_text,
                "bbox": bbox,
                "detection_confidence": detection_confidence,
                "recognition_confidence": recognition_confidence,
                "method": method,
                "plate_id": i,
            }

            results.append(result)

        return results

    def _recognize_plate(self, plate_image, use_pdlpr=True):
        """Recognize characters in a single plate image"""

        # Preprocess image
        processed_image = self._preprocess_for_recognition(plate_image)

        # Try PDLPR first if available and requested
        if use_pdlpr and self.recognition_model:
            try:
                trainer = PDLPRTrainer(self.recognition_model, self.device)
                predictions = trainer.predict(processed_image)

                if predictions is not None and len(predictions) > 0:
                    chars = [self.char_mapping[idx] for idx in predictions[0]]
                    plate_text = "".join(chars)
                    return plate_text, 0.90, "pdlpr"
            except Exception as e:
                print(f"PDLPR recognition failed: {e}")

        # Fallback to baseline model
        if self.baseline_model:
            try:
                with torch.no_grad():
                    outputs = self.baseline_model(processed_image)
                    predicted_indices = torch.argmax(outputs, dim=2).squeeze(0)

                    chars = [self.char_mapping[idx.item()] for idx in predicted_indices]
                    plate_text = "".join(chars)
                    return plate_text, 0.75, "baseline"
            except Exception as e:
                print(f"Baseline recognition failed: {e}")

        return "UNKNOWN", 0.0, "none"

    def _preprocess_for_recognition(self, plate_image):
        """Preprocess plate image for recognition models"""
        # Resize to standard size
        resized = cv2.resize(plate_image, (128, 64))

        # Convert to tensor
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        return tensor.to(self.device)

    def process_batch(self, image_paths, output_dir=None):
        """Process multiple images"""
        results = {}

        for image_path in image_paths:
            print(f"Processing: {image_path}")

            try:
                plate_results = self.detect_and_recognize(image_path)
                results[str(image_path)] = plate_results

                # Print results
                for result in plate_results:
                    print(
                        f"  Detected: {result['plate_text']} "
                        f"(conf: {result['recognition_confidence']:.2f}, "
                        f"method: {result['method']})"
                    )

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[str(image_path)] = []

        # Save results if output directory specified
        if output_dir:
            self._save_results(results, output_dir)

        return results

    def _save_results(self, results, output_dir):
        """Save recognition results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        import json

        with open(output_dir / "recognition_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Save summary
        with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("License Plate Recognition Results\n")
            f.write("=" * 40 + "\n\n")

            for image_path, plates in results.items():
                f.write(f"Image: {image_path}\n")
                if plates:
                    for plate in plates:
                        f.write(
                            f"  {plate['plate_text']} "
                            f"(conf: {plate['recognition_confidence']:.2f})\n"
                        )
                else:
                    f.write("  No plates detected\n")
                f.write("\n")

        print(f"Results saved to {output_dir}")

    def get_performance_stats(self):
        """Get performance statistics"""
        return self.profiler.get_stats()

    def print_performance_stats(self):
        """Print performance statistics"""
        self.profiler.print_stats()


def create_pipeline(detection_model=None, recognition_model=None, baseline_model=None):
    """Factory function to create integrated pipeline"""

    # Default model paths
    if baseline_model is None:
        baseline_model = "models/baseline_model.pth"
    if recognition_model is None:
        recognition_model = "models/pdlpr_best.pth"

    pipeline = IntegratedPipeline(
        detection_model_path=detection_model,
        recognition_model_path=recognition_model,
        baseline_model_path=baseline_model,
    )

    return pipeline


if __name__ == "__main__":
    # Demo pipeline
    print("Testing integrated pipeline...")

    pipeline = create_pipeline()

    # Create demo image
    demo_image = np.ones((480, 640, 3), dtype=np.uint8) * 100
    cv2.rectangle(demo_image, (200, 300), (440, 380), (255, 255, 255), -1)
    cv2.rectangle(demo_image, (200, 300), (440, 380), (0, 0, 0), 2)

    # Test recognition
    results = pipeline.detect_and_recognize(demo_image)

    print("Pipeline test completed!")
    print(f"Detected {len(results)} plates:")
    for result in results:
        print(f"  {result['plate_text']} (method: {result['method']})")
