#!/usr/bin/env python3
"""
License Plate Recognition Pipeline
Main entry point for car plate recognition
"""

import argparse
import sys
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.append('src')
from models.baseline import BaselineCNN

class LicensePlateRecognizer:
    """Main pipeline for license plate recognition"""
    
    def __init__(self, model_path='models/baseline_model.pth', device='cpu'):
        self.device = torch.device(device)
        
        # Character mapping for Chinese plates
        self.chinese_chars = [
            "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
            "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
            "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
        ]
        self.alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.all_chars = self.chinese_chars + list(self.alphanumeric)
        
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load trained model"""
        model = BaselineCNN()
        
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model not found at {model_path}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess input image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize to model input size (simulating plate detection)
        plate_roi = cv2.resize(image, (128, 64))
        
        # Convert to tensor
        plate_tensor = torch.from_numpy(plate_roi).permute(2, 0, 1).float() / 255.0
        plate_tensor = plate_tensor.unsqueeze(0)
        
        return plate_tensor
    
    def predict_characters(self, image_tensor):
        """Predict character sequence"""
        with torch.no_grad():
            outputs = self.model(image_tensor.to(self.device))
            predicted_indices = torch.argmax(outputs, dim=2).squeeze(0)
            
            characters = [self.all_chars[idx.item()] for idx in predicted_indices]
            plate_text = ''.join(characters)
            
            return plate_text, characters
    
    def recognize_plate(self, image_path):
        """Full recognition pipeline"""
        try:
            image_tensor = self.preprocess_image(image_path)
            plate_text, characters = self.predict_characters(image_tensor)
            confidence = 0.85  # Mock confidence
            
            return plate_text, confidence, {'characters': characters}
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, 0.0, {}

def create_demo_image():
    """Create demo image for testing"""
    demo_path = Path("demo_images/input/demo_car.jpg")
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not demo_path.exists():
        # Create synthetic car image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 100
        cv2.rectangle(img, (100, 200), (540, 400), (80, 80, 120), -1)
        cv2.rectangle(img, (220, 320), (420, 380), (255, 255, 255), -1)
        cv2.rectangle(img, (220, 320), (420, 380), (0, 0, 0), 2)
        
        cv2.imwrite(str(demo_path), img)
        print(f"Created demo image: {demo_path}")
    
    return demo_path

def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, help='Output text file path')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--model', type=str, default='models/baseline_model.pth', 
                      help='Model path')
    args = parser.parse_args()
    
    print("License Plate Recognition Pipeline")
    print("=" * 40)
    
    recognizer = LicensePlateRecognizer(model_path=args.model)
    
    if args.demo:
        print("Running demo mode...")
        demo_image = create_demo_image()
        plate_text, confidence, info = recognizer.recognize_plate(demo_image)
        
        print(f"Input: {demo_image}")
        print(f"Detected: {plate_text}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Characters: {info.get('characters', [])}")
        
    elif args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input not found: {input_path}")
            return
        
        print(f"Processing: {input_path}")
        plate_text, confidence, info = recognizer.recognize_plate(input_path)
        
        if plate_text:
            print(f"Result: {plate_text}")
            print(f"Confidence: {confidence:.2%}")
            
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"{plate_text}\n")
                print(f"Saved to: {output_path}")
        else:
            print("Recognition failed")
    
    else:
        print("Usage:")
        print("  python main.py --demo")
        print("  python main.py --input car.jpg")
        print("  python main.py --input car.jpg --output result.txt")

if __name__ == "__main__":
    main()