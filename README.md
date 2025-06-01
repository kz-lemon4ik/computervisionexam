# Car Plate Recognition and Reconstruction

**Computer Vision Project (A.A. 2024-2025)**

Two-stage deep learning pipeline for license plate recognition using YOLOv5 for detection and PDLPR for character recognition.

## Project Description

Automatic license plate recognition system solving two key tasks:
1. **License plate detection** in car images (YOLOv5)
2. **Character recognition** on detected plates (PDLPR)

**Dataset:** CCPD (Chinese City Parking Dataset)  
**Framework:** PyTorch  
**Architecture:** YOLOv5 + PDLPR Pipeline

## Quick Start

### Dependencies Installation
```bash
pip install -r requirements.txt
```

### Usage
```bash
# Training models
python src/models/baseline.py --train

# Evaluation  
python src/models/baseline.py --evaluate
```

## Project Structure

```
computervisionexam/
├── README.md                    # Project description
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── 
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model architectures
│   ├── detection/              # YOLOv5 detection
│   └── recognition/            # PDLPR recognition
│
├── data/                       # Datasets
│   ├── raw/                    # Raw CCPD dataset
│   └── processed/              # Processed data
│
├── models/                     # Saved models
├── notebooks/                  # Jupyter notebooks
├── scripts/                    # Utility scripts
└── tests/                      # Tests
```

## Methodology

### Stage 1: License Plate Detection (YOLOv5)
- Train YOLOv5 on CCPD dataset
- Localize license plates in images
- Extract ROI (Region of Interest)

### Stage 2: Character Recognition (PDLPR)
- PDLPR architecture for sequence prediction
- Recognize alphanumeric characters
- Integration with detection results

### Baseline for Comparison
- Simple CNN for end-to-end recognition
- Performance comparison with proposed method

## Results

| Model | Detection Accuracy | Recognition Accuracy | End-to-End Accuracy |
|-------|-------------------|---------------------|-------------------|
| Baseline CNN | - | - | ~65% |
| YOLOv5 + PDLPR | >90% mAP@0.5 | >85% | >80% |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Performance evaluation
python scripts/evaluate_pipeline.py --test_dir data/test/

# Compare with baseline
python scripts/compare_models.py
```

## References

1. **Main Paper:** Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). A Real-Time License Plate Detection and Recognition Model in Unconstrained Scenarios. Sensors, 24(9), 2791

2. **Dataset:** Xu, Z.; Yang, W.; Meng, A.; Lu, N.; Huang, H.; Ying, C.; Huang, L. CCPD Dataset for End-to-End License Plate Detection and Recognition

3. **YOLOv5:** Ultralytics YOLOv5 for object detection

## License

This project is created for educational purposes for Computer Vision course.