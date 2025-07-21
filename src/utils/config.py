"""
Configuration settings for the project
"""

import torch
from pathlib import Path

# Model configuration
MODEL_CONFIG = {
    "num_chars": 67,  # Chinese chars + alphanumeric
    "sequence_length": 7,  # Chinese plate length
    "img_height": 64,
    "img_width": 128,
    "input_channels": 3,
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 16,
    "learning_rate": 0.001,
    "epochs": 10,
    "device": "cpu",  # For VM compatibility
    "num_workers": 0,  # For CPU training
    "train_samples": 1000,
    "val_samples": 200,
    "test_samples": 100,
}

# Data paths
DATA_PATHS = {
    "raw_data": Path("data/raw"),
    "processed_data": Path("data/processed"),
    "models": Path("models"),
    "demo_images": Path("demo_images"),
}

# Chinese province characters
CHINESE_PROVINCES = [
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

# Alphanumeric characters
ALPHANUMERIC = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# All characters
ALL_CHARS = CHINESE_PROVINCES + list(ALPHANUMERIC)


def get_device():
    """Get available device for training"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def ensure_dirs():
    """Ensure all necessary directories exist"""
    for path in DATA_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
