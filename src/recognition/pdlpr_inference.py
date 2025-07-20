#!/usr/bin/env python3
"""
PDLPR inference for character recognition
"""

import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse

sys.path.append('src')
from recognition.pdlpr_model import create_pdlpr_model, PDLPRTrainer
from utils.config import ALL_CHARS

class PDLPRInference:
    """PDLPR inference wrapper"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.chars = ALL_CHARS
        
        # Create model
        self.model = create_pdlpr_model(
            num_classes=len(self.chars),
            sequence_length=7
        )
        self.trainer = PDLPRTrainer(self.model, device)
        
        # Load model if available
        if model_path and Path(model_path).exists():
            self.trainer.load_model(model_path)
            self.model_loaded = True
            print(f"Loaded PDLPR model from {model_path}")
        else:
            self.model_loaded = False
            print("No trained model available, using random initialization")
    
    def preprocess_image(self, image):
        """Preprocess image for PDLPR"""
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input size
        target_height, target_width = 64, 128
        image = cv2.resize(image, (target_width, target_height))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension
        
        return image
    
    def predict_text(self, image):
        """Predict license plate text from image"""
        
        # Preprocess image
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            return ""
        
        processed_image = self.preprocess_image(image)
        
        # Run inference
        predictions = self.trainer.predict(processed_image)
        
        # Convert predictions to text
        if predictions is not None and len(predictions) > 0:
            # Handle different prediction formats
            pred_data = predictions[0]
            if hasattr(pred_data, 'cpu'):
                pred_data = pred_data.cpu().numpy()
            text = self.decode_predictions(pred_data)
        else:
            text = ""
        
        return text
    
    def decode_predictions(self, predictions):
        """Convert predicted indices to text"""
        text = ""
        try:
            # Handle different prediction formats
            if hasattr(predictions, '__iter__'):
                for idx in predictions:
                    idx_val = int(idx) if hasattr(idx, '__int__') else idx
                    if 0 <= idx_val < len(self.chars):
                        text += self.chars[idx_val]
                    else:
                        text += "?"  # Unknown character
            else:
                # Single prediction
                idx_val = int(predictions) if hasattr(predictions, '__int__') else predictions
                if 0 <= idx_val < len(self.chars):
                    text += self.chars[idx_val]
                else:
                    text += "?"
        except Exception as e:
            print(f"Error decoding predictions: {e}")
            text = "ERROR"
        
        return text
    
    def simulate_recognition(self, image):
        """Simulate character recognition for demo purposes"""
        
        # For demo purposes, return simulated Chinese license plate
        simulated_plates = [
            "京A12345",
            "沪B67890", 
            "粤C11111",
            "川D22222",
            "苏E33333"
        ]
        
        # Select based on image properties for consistency
        h, w = image.shape[:2]
        plate_idx = (h + w) % len(simulated_plates)
        
        return simulated_plates[plate_idx]

def demo_pdlpr_recognition():
    """Demo PDLPR character recognition"""
    
    # Create demo license plate image
    demo_img = np.ones((64, 128, 3), dtype=np.uint8) * 255  # White background
    
    # Add some plate-like features
    cv2.rectangle(demo_img, (10, 15), (118, 49), (200, 200, 200), -1)  # Plate background
    cv2.putText(demo_img, "ABC123", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save demo image
    demo_dir = Path('demo_images')
    demo_dir.mkdir(exist_ok=True)
    demo_path = demo_dir / 'demo_plate.jpg'
    cv2.imwrite(str(demo_path), demo_img)
    
    print(f"Created demo plate image: {demo_path}")
    
    # Initialize PDLPR
    recognizer = PDLPRInference()
    
    # Run recognition
    if recognizer.model_loaded:
        recognized_text = recognizer.predict_text(demo_img)
        print(f"PDLPR Recognition: {recognized_text}")
    else:
        recognized_text = recognizer.simulate_recognition(demo_img)
        print(f"Simulated Recognition: {recognized_text}")
    
    return recognized_text

def main():
    parser = argparse.ArgumentParser(description='PDLPR Character Recognition')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--model', type=str, default='models/pdlpr_best.pth', help='Model path')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_pdlpr_recognition()
        return
    
    if not args.input:
        print("Please provide input image or use --demo")
        return
    
    # Initialize recognizer
    recognizer = PDLPRInference(args.model, args.device)
    
    # Run recognition
    result = recognizer.predict_text(args.input)
    print(f"Recognized text: {result}")

if __name__ == "__main__":
    main()