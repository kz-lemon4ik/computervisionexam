#!/usr/bin/env python3
"""
Simple integrated pipeline for license plate recognition
Uses existing YOLOv5 detection + PDLPR recognition
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

class SimplePipeline:
    """Simple license plate recognition pipeline"""
    
    def __init__(self, yolo_model=None, pdlpr_model=None, device='auto'):
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Pipeline using device: {self.device}")
        
        # Initialize components
        self.yolo_detector = YOLOv5Inference(yolo_model, confidence=0.5)
        self.pdlpr_recognizer = PDLPRInference(pdlpr_model, device=self.device)
        
        # Stats
        self.stats = {
            'images_processed': 0,
            'plates_detected': 0,
            'total_time': 0
        }
    
    def process_image(self, image_path):
        """Process single image"""
        
        start_time = time.time()
        
        # Convert Path to string if needed
        image_path_str = str(image_path)
        
        # Load image
        image = cv2.imread(image_path_str)
        if image is None:
            print(f"Failed to load image: {image_path_str}")
            return []
        
        print(f"Processing image: {Path(image_path_str).name}")
        
        # Step 1: Detect license plates
        print("Step 1: Detecting license plates...")
        detections = self.yolo_detector.detect_plates(image_path_str)
        print(f"Found {len(detections)} license plates")
        
        results = []
        
        # Step 2: Recognize characters in each plate
        for i, detection in enumerate(detections):
            print(f"Step 2.{i+1}: Recognizing characters...")
            
            # Extract plate region
            bbox = detection['bbox']
            x, y, w, h = bbox
            plate_roi = image[y:y+h, x:x+w]
            
            if plate_roi.size == 0:
                continue
            
            # Recognize text
            if self.pdlpr_recognizer.model_loaded:
                recognized_text = self.pdlpr_recognizer.predict_text(plate_roi)
                method = 'pdlpr'
                confidence = 0.85
            else:
                recognized_text = self.pdlpr_recognizer.simulate_recognition(plate_roi)
                method = 'simulated'
                confidence = 0.60
            
            result = {
                'plate_text': recognized_text,
                'bbox': bbox,
                'detection_confidence': detection['confidence'],
                'recognition_confidence': confidence,
                'method': method,
                'plate_id': i
            }
            
            results.append(result)
            print(f"  Recognized: '{recognized_text}' (method: {method}, conf: {confidence:.2f})")
        
        # Save visualization
        if results:
            self._save_visualization(image, results, image_path_str)
        
        # Update stats
        processing_time = time.time() - start_time
        self.stats['images_processed'] += 1
        self.stats['plates_detected'] += len(results)
        self.stats['total_time'] += processing_time
        
        print(f"Processing completed in {processing_time:.2f}s")
        
        return results
    
    def _save_visualization(self, image, results, image_path):
        """Save visualization of results"""
        
        vis_image = image.copy()
        
        for result in results:
            bbox = result['bbox']
            plate_text = result['plate_text']
            det_conf = result['detection_confidence']
            rec_conf = result['recognition_confidence']
            
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{plate_text} ({det_conf:.2f}|{rec_conf:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for label
            cv2.rectangle(vis_image, (x, y-label_size[1]-10), 
                         (x+label_size[0]+5, y), (0, 255, 0), -1)
            
            # Label text
            cv2.putText(vis_image, label, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save visualization
        base_name = Path(image_path).stem
        output_path = f"demo_images/{base_name}_result.jpg"
        Path("demo_images").mkdir(exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved: {output_path}")
    
    def process_batch(self, image_paths):
        """Process multiple images"""
        
        print(f"Processing {len(image_paths)} images...")
        all_results = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"\n[{i+1}/{len(image_paths)}] Processing: {Path(image_path).name}")
            
            try:
                results = self.process_image(image_path)
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
        self._save_batch_results(all_results)
        
        return all_results
    
    def _save_batch_results(self, results):
        """Save batch results"""
        
        output_dir = Path("demo_images/results")
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
    
    def print_stats(self):
        """Print processing statistics"""
        
        print("\n" + "=" * 50)
        print("PIPELINE STATISTICS")
        print("=" * 50)
        print(f"Images processed:     {self.stats['images_processed']}")
        print(f"Total plates found:   {self.stats['plates_detected']}")
        if self.stats['images_processed'] > 0:
            print(f"Plates per image:     {self.stats['plates_detected']/self.stats['images_processed']:.1f}")
            print(f"Avg processing time:  {self.stats['total_time']/self.stats['images_processed']:.2f}s")
        print("=" * 50)

def create_demo_images():
    """Create demo car images with license plates"""
    
    print("Creating demo images...")
    
    demo_dir = Path('demo_images')
    demo_dir.mkdir(exist_ok=True)
    
    demo_images = []
    
    for i in range(3):
        # Create car image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 70
        
        # Draw car body
        cv2.rectangle(img, (100, 200), (540, 400), (80, 80, 120), -1)
        cv2.rectangle(img, (120, 220), (520, 380), (100, 100, 140), -1)
        
        # Draw license plate
        plate_x = 270 + i * 10
        cv2.rectangle(img, (plate_x, 350), (plate_x + 100, 385), (255, 255, 255), -1)
        cv2.rectangle(img, (plate_x, 350), (plate_x + 100, 385), (0, 0, 0), 2)
        
        # Add plate text
        texts = ["ABC123", "XYZ789", "DEF456"]
        cv2.putText(img, texts[i], (plate_x + 10, 375), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save demo image
        demo_path = demo_dir / f'demo_car_{i+1}.jpg'
        cv2.imwrite(str(demo_path), img)
        demo_images.append(demo_path)
        print(f"Created: {demo_path}")
    
    return demo_images

def main():
    parser = argparse.ArgumentParser(description='Simple License Plate Recognition Pipeline')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--batch', type=str, nargs='+', help='Multiple input images')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--yolo_model', type=str, help='YOLOv5 model path')
    parser.add_argument('--pdlpr_model', type=str, help='PDLPR model path')
    
    args = parser.parse_args()
    
    if args.demo:
        # Create demo images
        demo_images = create_demo_images()
        
        # Initialize pipeline
        print("\nInitializing pipeline...")
        pipeline = SimplePipeline(args.yolo_model, args.pdlpr_model)
        
        # Process demo images
        print("\nRunning pipeline on demo images...")
        results = pipeline.process_batch(demo_images)
        
        # Show stats
        pipeline.print_stats()
        
        return
    
    # Initialize pipeline
    pipeline = SimplePipeline(args.yolo_model, args.pdlpr_model)
    
    if args.input:
        # Single image
        results = pipeline.process_image(args.input)
        pipeline.print_stats()
        
    elif args.batch:
        # Multiple images
        results = pipeline.process_batch(args.batch)
        pipeline.print_stats()
        
    else:
        print("Please provide --input, --batch, or --demo")

if __name__ == "__main__":
    main()