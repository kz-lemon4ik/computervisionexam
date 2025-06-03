import os
import cv2
import numpy as np
import tarfile
from pathlib import Path

class CCPDDataLoader:
    """CCPD dataset loader for license plate recognition"""
    
    def __init__(self, archive_path):
        self.archive_path = archive_path
        self.chinese_chars = [
            "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
            "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
            "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"
        ]
        self.alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.all_chars = self.chinese_chars + list(self.alphanumeric)
        
    def parse_filename(self, filename):
        """Parse CCPD filename to extract bounding box and characters"""
        try:
            basename = filename.split('.')[0]
            parts = basename.split('-')
            
            # Extract bounding box coordinates
            bbox_str = parts[2]
            bbox_parts = bbox_str.split('_')
            bbox = []
            for part in bbox_parts:
                x, y = map(int, part.split('&'))
                bbox.append([x, y])
            
            # Extract character labels
            chars_str = parts[4]
            char_indices = list(map(int, chars_str.split('_')))
            characters = [self.all_chars[i] for i in char_indices]
            plate_text = ''.join(characters)
            
            return {
                'bbox': np.array(bbox),
                'characters': characters,
                'plate_text': plate_text,
                'filename': filename
            }
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return None
    
    def load_image(self, image_path):
        """Load image from path"""
        return cv2.imread(str(image_path))
    
    def extract_samples(self, output_dir, num_samples=100):
        """Extract sample images from archive"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        annotations = []
        extracted_count = 0
        
        with tarfile.open(self.archive_path, 'r:xz') as tar:
            members = tar.getmembers()
            image_members = [m for m in members if m.name.endswith('.jpg')]
            
            for member in image_members[:num_samples]:
                tar.extract(member, output_dir)
                
                filename = os.path.basename(member.name)
                annotation = self.parse_filename(filename)
                if annotation:
                    annotations.append(annotation)
                    extracted_count += 1
        
        print(f"Extracted {extracted_count} samples to {output_dir}")
        return annotations