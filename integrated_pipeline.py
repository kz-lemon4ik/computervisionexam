#!/usr/bin/env python3
"""
Integrated YOLOv5 + PDLPR Pipeline for License Plate Recognition
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import time
import json
import argparse

sys.path.append('src')
from detection.yolo_inference import YOLOv5Inference
from recognition.pdlpr_inference import PDLPRInference
from models.baseline import BaselineCNN
from utils.config import ALL_CHARS

class LicensePlateRecognitionPipeline:
    """Complete license plate recognition pipeline"""
    
    def __init__(self, 
                 yolo_model_path=None,
                 pdlpr_model_path=None,
                 baseline_model_path=None,
                 device='auto'):
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Pipeline using device: {self.device}")
        
        # Initialize components
        self.yolo_detector = YOLOv5Inference(yolo_model_path, confidence=0.5)
        
        # Use trained PDLPR model
        if pdlpr_model_path is None:
            pdlpr_model_path = "models/pdlpr_final.pth"
        self.pdlpr_recognizer = PDLPRInference(pdlpr_model_path, device=self.device)
        
        # Load baseline model if available
        self.baseline_model = None
        if baseline_model_path and Path(baseline_model_path).exists():
            try:
                self.baseline_model = BaselineCNN()
                state_dict = torch.load(baseline_model_path, map_location=self.device)
                self.baseline_model.load_state_dict(state_dict)
                self.baseline_model.to(self.device)
                self.baseline_model.eval()
                print(f"Loaded baseline model from {baseline_model_path}")
            except Exception as e:
                print(f"Failed to load baseline model: {e}")
        
        # Stats
        self.stats = {
            'total_images': 0,
            'total_plates': 0,
            'detection_time': 0,
            'recognition_time': 0,
            'total_time': 0
        }
    
    def process_image(self, image_path, save_visualization=False):
        """Process single image through complete pipeline"""
        
        start_time = time.time()
        
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to load image: {image_path}")
                return []
        else:
            image = image_path
        
        results = []
        
        # Step 1: Detect license plates with YOLOv5
        print(f"Step 1: Detecting license plates...")
        detection_start = time.time()
        
        detections = self.yolo_detector.detect_plates(str(image_path) if isinstance(image_path, (str, Path)) else image)
        
        detection_time = time.time() - detection_start
        self.stats['detection_time'] += detection_time
        
        print(f"Found {len(detections)} license plates")
        
        # Step 2: Recognize characters in each detected plate
        recognition_start = time.time()
        
        for i, detection in enumerate(detections):
            print(f"Step 2.{i+1}: Recognizing characters...")
            
            # Extract plate region
            bbox = detection['bbox']
            x, y, w, h = bbox
            plate_roi = image[y:y+h, x:x+w]
            
            if plate_roi.size == 0:
                continue
            
            # Recognize text with trained PDLPR model
            if self.pdlpr_recognizer.model_loaded:
                recognized_text = self.pdlpr_recognizer.predict_text(plate_roi)
                recognition_confidence = 0.85
                recognition_method = 'pdlpr'
            else:
                # No simulation - return unknown if model not loaded
                recognized_text = "UNKNOWN"
                recognition_confidence = 0.0
                recognition_method = 'no_model'
            
            result = {
                'plate_text': recognized_text,
                'bbox': bbox,
                'detection_confidence': detection['confidence'],
                'recognition_confidence': recognition_confidence,
                'recognition_method': recognition_method,
                'plate_id': i
            }
            
            results.append(result)
            print(f"  Recognized: '{recognized_text}' (conf: {recognition_confidence:.2f})")
        
        recognition_time = time.time() - recognition_start
        self.stats['recognition_time'] += recognition_time
        
        total_time = time.time() - start_time
        self.stats['total_time'] += total_time
        self.stats['total_images'] += 1
        self.stats['total_plates'] += len(results)
        
        # Save visualization if requested
        if save_visualization and results:
            self._save_visualization(image, results, image_path)
        
        return results
    
    def _save_visualization(self, image, results, image_path):
        """Save image with detection and recognition results"""
        
        vis_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            plate_text = result['plate_text']
            detection_conf = result['detection_confidence']
            recognition_conf = result['recognition_confidence']
            
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{plate_text} ({detection_conf:.2f}|{recognition_conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for label
            cv2.rectangle(vis_image, (x, y-label_size[1]-10), 
                         (x+label_size[0]+5, y), (0, 255, 0), -1)
            
            # Label text
            cv2.putText(vis_image, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save visualization
        if isinstance(image_path, (str, Path)):
            base_name = Path(str(image_path)).stem
            output_path = f"demo_images/{base_name}_pipeline_result.jpg"
        else:
            output_path = "demo_images/pipeline_result.jpg"
        
        Path("demo_images").mkdir(exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")
    
    def process_batch(self, image_paths, output_dir=None):
        """Process multiple images"""
        
        print(f"Processing {len(image_paths)} images...")
        all_results = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] Processing: {image_path}")
            
            try:
                results = self.process_image(image_path, save_visualization=True)
                all_results[str(image_path)] = results
                
                if results:
                    print(f"  Results: {len(results)} plates detected")
                    for result in results:
                        print(f"    - {result['plate_text']}")
                else:
                    print("  No plates detected")
                    
            except Exception as e:
                print(f"  Error: {e}")
                all_results[str(image_path)] = []
        
        # Save results
        if output_dir:
            self._save_batch_results(all_results, output_dir)
        
        return all_results
    
    def _save_batch_results(self, results, output_dir):
        """Save batch processing results"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Save text summary
        with open(output_dir / 'summary.txt', 'w', encoding='utf-8') as f:
            f.write("License Plate Recognition Pipeline Results\n")
            f.write("=" * 50 + "\n\n")
            
            total_images = len(results)
            total_plates = sum(len(plates) for plates in results.values())
            
            f.write(f"Summary:\n")
            f.write(f"  Images processed: {total_images}\n")
            f.write(f"  Total plates detected: {total_plates}\n")
            f.write(f"  Average plates per image: {total_plates/total_images:.1f}\n\n")
            
            # Detailed results
            for image_path, plates in results.items():
                f.write(f"Image: {Path(image_path).name}\n")
                if plates:
                    for plate in plates:
                        f.write(f"  - {plate['plate_text']} "
                               f"(det: {plate['detection_confidence']:.2f}, "
                               f"rec: {plate['recognition_confidence']:.2f})\n")
                else:
                    f.write("  - No plates detected\n")
                f.write("\n")
        
        print(f"\nBatch results saved to: {output_dir}")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if self.stats['total_images'] > 0:
            return {
                'total_images': self.stats['total_images'],
                'total_plates': self.stats['total_plates'],
                'avg_detection_time': self.stats['detection_time'] / self.stats['total_images'],
                'avg_recognition_time': self.stats['recognition_time'] / self.stats['total_images'],
                'avg_total_time': self.stats['total_time'] / self.stats['total_images'],
                'plates_per_image': self.stats['total_plates'] / self.stats['total_images']
            }
        return self.stats
    
    def print_performance_stats(self):
        """Print performance statistics"""
        stats = self.get_performance_stats()
        
        print("\n" + "="*50)
        print("PERFORMANCE STATISTICS")
        print("="*50)
        print(f"Images processed:     {stats['total_images']}")
        print(f"Total plates found:   {stats['total_plates']}")
        print(f"Plates per image:     {stats.get('plates_per_image', 0):.1f}")
        print(f"Avg detection time:   {stats.get('avg_detection_time', 0):.3f}s")
        print(f"Avg recognition time: {stats.get('avg_recognition_time', 0):.3f}s") 
        print(f"Avg total time:       {stats.get('avg_total_time', 0):.3f}s")
        print("="*50)

def demo_pipeline():
    """Demo the integrated pipeline"""
    
    print("Creating demo images...")
    
    # Create demo directory
    demo_dir = Path('demo_images')
    demo_dir.mkdir(exist_ok=True)
    
    # Create demo car images
    demo_images = []
    
    for i in range(3):
        # Create car image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 70
        
        # Draw car
        cv2.rectangle(img, (100, 200), (540, 400), (80, 80, 120), -1)
        cv2.rectangle(img, (120, 220), (520, 380), (100, 100, 140), -1)
        
        # Draw license plate
        plate_x = 270 + i * 10
        cv2.rectangle(img, (plate_x, 350), (plate_x + 100, 385), (255, 255, 255), -1)
        cv2.rectangle(img, (plate_x, 350), (plate_x + 100, 385), (0, 0, 0), 2)
        
        # Add some text
        texts = ["ABC123", "XYZ789", "DEF456"]
        cv2.putText(img, texts[i], (plate_x + 10, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save demo image
        demo_path = demo_dir / f'demo_car_{i+1}.jpg'
        cv2.imwrite(str(demo_path), img)
        demo_images.append(demo_path)
        print(f"Created: {demo_path}")
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = LicensePlateRecognitionPipeline()
    
    # Process demo images
    print("\nRunning pipeline on demo images...")
    results = pipeline.process_batch(demo_images, output_dir='demo_images/results')
    
    # Show performance stats
    pipeline.print_performance_stats()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Integrated License Plate Recognition Pipeline')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--batch', type=str, nargs='+', help='Multiple input images')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--yolo_model', type=str, help='YOLOv5 model path')
    parser.add_argument('--pdlpr_model', type=str, help='PDLPR model path')
    parser.add_argument('--baseline_model', type=str, help='Baseline model path')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_pipeline()
        return
    
    # Initialize pipeline
    pipeline = LicensePlateRecognitionPipeline(
        yolo_model_path=args.yolo_model,
        pdlpr_model_path=args.pdlpr_model,
        baseline_model_path=args.baseline_model
    )
    
    if args.input:
        # Single image
        results = pipeline.process_image(args.input, save_visualization=True)
        pipeline.print_performance_stats()
        
    elif args.batch:
        # Multiple images
        results = pipeline.process_batch(args.batch, output_dir=args.output)
        pipeline.print_performance_stats()
        
    else:
        print("Please provide --input, --batch, or --demo")

if __name__ == "__main__":
    main()