import cv2
import numpy as np
from pathlib import Path
import random
from torch.utils.data import Dataset


class CCPDDataLoader:
    """CCPD dataset loader for license plate recognition"""

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.chinese_chars = [
            "京",
            "沪",
            "津",
            "渝",
            "冀",
            "晋",
            "蒙",
            "辽",
            "吉",
            "黑",
            "苏",
            "浙",
            "皖",
            "闽",
            "赣",
            "鲁",
            "豫",
            "鄂",
            "湘",
            "粤",
            "桂",
            "琼",
            "川",
            "贵",
            "云",
            "藏",
            "陕",
            "甘",
            "青",
            "宁",
            "新",
        ]
        self.alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.all_chars = self.chinese_chars + list(self.alphanumeric)

    def parse_filename(self, filename):
        """Parse CCPD filename to extract bounding box and characters"""
        try:
            basename = filename.split(".")[0]
            parts = basename.split("-")

            # Extract bounding box coordinates
            bbox_str = parts[2]
            bbox_parts = bbox_str.split("_")
            bbox = []
            for part in bbox_parts:
                x, y = map(int, part.split("&"))
                bbox.append([x, y])

            # Extract character labels
            chars_str = parts[4]
            char_indices = list(map(int, chars_str.split("_")))
            characters = [self.all_chars[i] for i in char_indices]
            plate_text = "".join(characters)

            return {
                "bbox": np.array(bbox),
                "characters": characters,
                "plate_text": plate_text,
                "filename": filename,
            }
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return None

    def load_image(self, image_path):
        """Load image from path"""
        return cv2.imread(str(image_path))

    def load_dataset(self, max_samples=None):
        """Load dataset from directory"""
        image_files = list(self.data_path.glob("*.jpg"))

        if max_samples:
            image_files = image_files[:max_samples]

        annotations = []
        for image_file in image_files:
            # Check if file actually exists and is readable
            if image_file.exists() and image_file.stat().st_size > 0:
                annotation = self.parse_filename(image_file.name)
                if annotation:
                    annotation["image_path"] = str(image_file)
                    annotations.append(annotation)

        print(f"Loaded {len(annotations)} samples from {self.data_path}")
        return annotations

    def get_train_val_split(self, annotations, val_ratio=0.2, seed=42):
        """Split dataset into train and validation"""
        random.seed(seed)
        random.shuffle(annotations)

        split_idx = int(len(annotations) * (1 - val_ratio))
        train_data = annotations[:split_idx]
        val_data = annotations[split_idx:]

        return train_data, val_data


class CCPDDataset(Dataset):
    """PyTorch Dataset for CCPD data"""

    def __init__(self, annotations, transform=None, target_size=(128, 64)):
        self.annotations = annotations
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # Load image
        image = cv2.imread(annotation["image_path"])
        if image is None:
            print(f"Failed to load image: {annotation['image_path']}")
            # Create dummy image if loading fails
            image = np.ones((64, 128, 3), dtype=np.uint8) * 128
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, self.target_size)

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0

        if self.transform:
            image = self.transform(image)

        # Convert characters to indices for classification
        char_indices = []
        loader = CCPDDataLoader("")
        for char in annotation["characters"]:
            if char in loader.all_chars:
                char_indices.append(loader.all_chars.index(char))
            else:
                char_indices.append(0)  # unknown character

        return {
            "image": image,
            "chars": np.array(char_indices),
            "text": annotation["plate_text"],
            "bbox": annotation["bbox"],
        }
