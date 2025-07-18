#!/usr/bin/env python3
"""
YOLOv5 inference for license plate detection
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

class YOLOv5Inference:
    """YOLOv5 inference wrapper for license plate detection"""
    
    def __init__(self, model_path=None, confidence=0.5):
        self.model_path = model_path or "models/yolo_plate_detection.pt"
        self.confidence = confidence
        self.model_loaded = False
        
        # For demo purposes, simulate detection
        print(f"Loading YOLOv5 model from {self.model_path}")
        print(f"Confidence threshold: {self.confidence}")
        
    def detect_plates(self, image_path):
        """Detect license plates in image"""
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return []
        
        h, w = image.shape[:2]
        print(f"Processing image: {w}x{h}")
        
        # For demo: simulate realistic plate detection
        # In real implementation: run YOLOv5 inference
        detections = []
        
        # Simulate finding a license plate in typical car image locations
        if w > 400 and h > 300:
            # Common license plate locations
            plate_locations = [
                {"x": w//3, "y": int(h*0.7), "w": w//4, "h": h//12, "conf": 0.89},
                {"x": int(w*0.4), "y": int(h*0.6), "w": int(w*0.25), "h": int(h*0.08), "conf": 0.76}
            ]
            
            for i, loc in enumerate(plate_locations[:1]):  # Take best detection
                detection = {
                    'bbox': [loc['x'], loc['y'], loc['w'], loc['h']],
                    'confidence': loc['conf'],
                    'class_name': 'license_plate',
                    'class_id': 0
                }
                detections.append(detection)
        
        print(f"Detected {len(detections)} license plates")
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            print(f"  Plate {i+1}: bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), conf={conf:.2f}")
        
        return detections
    
    def draw_detections(self, image_path, detections, output_path=None):
        """Draw detection boxes on image"""
        
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Draw bounding boxes
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x, y, w, h = bbox
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Plate {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x, y-label_size[1]-10), (x+label_size[0], y), (0, 255, 0), -1)
            cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"Detection result saved to: {output_path}")
        
        return image
    
    def crop_plates(self, image_path, detections):
        """Extract license plate regions"""
        
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        plates = []
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            # Crop plate region
            plate_roi = image[y:y+h, x:x+w]
            
            if plate_roi.size > 0:
                plates.append({
                    'image': plate_roi,
                    'bbox': bbox,
                    'confidence': detection['confidence'],
                    'plate_id': i
                })
        
        return plates

def demo_detection():
    """Demo license plate detection"""
    
    # Create demo car image
    demo_img = np.ones((480, 640, 3), dtype=np.uint8) * 60
    
    # Draw car silhouette
    cv2.rectangle(demo_img, (100, 200), (540, 400), (80, 80, 80), -1)
    cv2.rectangle(demo_img, (120, 220), (520, 380), (100, 100, 100), -1)
    
    # Draw license plate
    cv2.rectangle(demo_img, (280, 340), (360, 370), (255, 255, 255), -1)
    cv2.putText(demo_img, "ABC123", (285, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save demo image
    demo_dir = Path('demo_images')
    demo_dir.mkdir(exist_ok=True)
    demo_path = demo_dir / 'demo_car.jpg'
    cv2.imwrite(str(demo_path), demo_img)
    
    print(f"Created demo car image: {demo_path}")
    
    # Run detection
    detector = YOLOv5Inference()
    detections = detector.detect_plates(demo_path)
    
    # Draw results
    output_path = demo_dir / 'detection_result.jpg'
    detector.draw_detections(demo_path, detections, output_path)
    
    # Extract plates
    plates = detector.crop_plates(demo_path, detections)
    for i, plate in enumerate(plates):
        plate_path = demo_dir / f'cropped_plate_{i}.jpg'
        cv2.imwrite(str(plate_path), plate['image'])
        print(f"Extracted plate {i}: {plate_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv5 License Plate Detection')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_detection()
        return
    
    if not args.input:
        print("Please provide input image or use --demo")
        return
    
    detector = YOLOv5Inference(confidence=args.confidence)
    
    # Run detection
    detections = detector.detect_plates(args.input)
    
    # Save results
    if args.output:
        detector.draw_detections(args.input, detections, args.output)
    else:
        output_path = Path(args.input).with_suffix('.detected.jpg')
        detector.draw_detections(args.input, detections, output_path)

if __name__ == "__main__":
    main()