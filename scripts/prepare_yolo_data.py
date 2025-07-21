#!/usr/bin/env python3
"""
Convert CCPD dataset to YOLO format for license plate detection
"""

import sys
import cv2
import shutil
from pathlib import Path

sys.path.append("src")

from data.data_loader import CCPDDataLoader


def ccpd_to_yolo_bbox(bbox, img_width, img_height):
    """Convert CCPD bbox format to YOLO format"""
    # CCPD bbox: [[x1,y1], [x2,y2]] (two corner points)
    # YOLO format: center_x, center_y, width, height (normalized 0-1)

    x1, y1 = bbox[0]
    x2, y2 = bbox[1]

    # Calculate center and dimensions
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # Normalize to 0-1
    norm_center_x = center_x / img_width
    norm_center_y = center_y / img_height
    norm_width = width / img_width
    norm_height = height / img_height

    return norm_center_x, norm_center_y, norm_width, norm_height


def convert_ccpd_to_yolo(ccpd_path, output_dir, train_split=0.8):
    """Convert CCPD dataset to YOLO format"""

    print("Loading CCPD dataset...")
    loader = CCPDDataLoader(ccpd_path)
    annotations = loader.load_dataset(max_samples=1000)  # Limit for demo

    if not annotations:
        print("No annotations found!")
        return

    print(f"Found {len(annotations)} samples")

    output_dir = Path(output_dir)

    # Create directory structure
    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Split dataset
    split_idx = int(len(annotations) * train_split)
    train_data = annotations[:split_idx]
    val_data = annotations[split_idx:]

    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Process train split
    print("Processing training data...")
    for i, annotation in enumerate(train_data):
        try:
            # Load image to get dimensions
            image = cv2.imread(annotation["image_path"])
            if image is None:
                continue

            h, w = image.shape[:2]

            # Convert bbox to YOLO format
            yolo_bbox = ccpd_to_yolo_bbox(annotation["bbox"], w, h)

            # Copy image
            img_filename = f"train_{i:05d}.jpg"
            img_dst = output_dir / "train" / "images" / img_filename
            shutil.copy2(annotation["image_path"], img_dst)

            # Create YOLO label file
            label_filename = f"train_{i:05d}.txt"
            label_path = output_dir / "train" / "labels" / label_filename

            # YOLO format: class_id center_x center_y width height
            with open(label_path, "w") as f:
                f.write(
                    f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                )

        except Exception as e:
            print(f"Error processing train sample {i}: {e}")
            continue

    # Process validation split
    print("Processing validation data...")
    for i, annotation in enumerate(val_data):
        try:
            # Load image to get dimensions
            image = cv2.imread(annotation["image_path"])
            if image is None:
                continue

            h, w = image.shape[:2]

            # Convert bbox to YOLO format
            yolo_bbox = ccpd_to_yolo_bbox(annotation["bbox"], w, h)

            # Copy image
            img_filename = f"val_{i:05d}.jpg"
            img_dst = output_dir / "val" / "images" / img_filename
            shutil.copy2(annotation["image_path"], img_dst)

            # Create YOLO label file
            label_filename = f"val_{i:05d}.txt"
            label_path = output_dir / "val" / "labels" / label_filename

            # YOLO format: class_id center_x center_y width height
            with open(label_path, "w") as f:
                f.write(
                    f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                )

        except Exception as e:
            print(f"Error processing val sample {i}: {e}")
            continue

    # Create dataset.yaml
    yaml_content = f"""# YOLO dataset config for license plate detection

path: {output_dir.absolute()}
train: train/images
val: val/images

nc: 1
names: ['license_plate']
"""

    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print("YOLO dataset conversion completed!")
    print(f"Output directory: {output_dir}")


def main():
    ccpd_path = "data/processed/ccpd_subset"
    output_dir = "data/yolo"

    if not Path(ccpd_path).exists():
        print(f"CCPD data not found at {ccpd_path}")
        return

    convert_ccpd_to_yolo(ccpd_path, output_dir)


if __name__ == "__main__":
    main()
