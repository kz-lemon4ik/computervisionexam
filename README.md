# License Plate Recognition System

**Computer Vision Project (A.A. 2024-2025)**

A comprehensive two-stage deep learning pipeline for automatic license plate recognition combining YOLOv5 object detection with PDLPR character recognition, trained and evaluated on the CCPD dataset.

## Project Overview

This project implements an end-to-end automatic license plate recognition system with:

1. **License Plate Detection**: YOLOv5-based object detection to locate plates in vehicle images
2. **Character Recognition**: PDLPR (Parallel Detection and Language Parsing Recognition) for extracting text from detected plates
3. **Baseline Comparison**: CNN baseline model for performance comparison

**Dataset**: CCPD (Chinese City Parking Dataset)  
**Framework**: PyTorch  
**Target Environment**: CPU-optimized for VM deployment

## Architecture

```
Input Image → YOLOv5 Detection → Plate ROI → PDLPR Recognition → Output Text
                                    ↓
                            Baseline CNN (fallback)
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kz-lemon4ik/computervisionexam.git
cd computervisionexam

# Install dependencies
pip install -r requirements.txt

# Validate setup
python scripts/validate_setup.py
```

### System Demonstration

```bash
# Complete system demonstration (recommended)
python final_demo.py --demo

# Batch evaluation on dataset
python final_demo.py --evaluate data/processed/ccpd_subset --max_images 50

# Process specific images
python final_demo.py --images path/to/image1.jpg path/to/image2.jpg

# Integrated pipeline with visualization
python integrated_pipeline.py --demo

# Comprehensive evaluation
python scripts/comprehensive_evaluation.py
```

## Training

### Baseline Model

```bash
# Train baseline CNN
python src/models/baseline.py --train --epochs 10

# Evaluate baseline
python src/models/baseline.py --evaluate
```

### PDLPR Model

```bash
# Train PDLPR model
python src/recognition/train_pdlpr.py --epochs 20

# Train with custom parameters
python src/recognition/train_pdlpr.py --epochs 50 --batch_size 32
```

### YOLOv5 Setup

```bash
# Prepare YOLO data format
python scripts/prepare_yolo_data.py

# Setup YOLOv5 training environment
python src/detection/yolo_model.py
```

## Evaluation and Testing

### Run All Tests

```bash
# Run complete test suite
python run_tests.py

# Run specific test module
python run_tests.py test_models
python run_tests.py test_pipeline
```

### Performance Evaluation

```bash
# Evaluate pipeline performance
python scripts/evaluate_pipeline.py

# Optimize models for deployment
python scripts/optimize_models.py
```

## Project Structure

```
computervisionexam/
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── final_demo.py            # Main demonstration script
├── integrated_pipeline.py   # Complete pipeline implementation
├── simple_pipeline.py       # Simplified pipeline version
│
├── src/                     # Source code
│   ├── data/                # Data loading and preprocessing
│   │   ├── data_loader.py   # CCPD dataset loader
│   │   └── __init__.py
│   ├── models/              # Model implementations
│   │   ├── baseline.py      # Baseline CNN model
│   │   └── __init__.py
│   ├── detection/           # YOLOv5 detection module
│   │   ├── yolo_model.py    # YOLOv5 wrapper
│   │   └── __init__.py
│   ├── recognition/         # PDLPR recognition module
│   │   ├── pdlpr_model.py   # PDLPR architecture
│   │   ├── train_pdlpr.py   # PDLPR training script
│   │   └── __init__.py
│   ├── utils/               # Utilities and helpers
│   │   ├── config.py        # Configuration settings
│   │   ├── metrics.py       # Evaluation metrics
│   │   ├── optimization.py  # Performance optimization
│   │   └── __init__.py
│   ├── pipeline.py          # Integrated pipeline
│   └── __init__.py
│
├── scripts/                 # Utility scripts
│   ├── prepare_yolo_data.py # YOLO data preparation
│   ├── comprehensive_evaluation.py # Complete system evaluation
│   ├── cleanup_results.py   # Clean demo files and results
│   └── __init__.py
│
├── tests/                   # Test suite
│   ├── test_data_loader.py  # Data loader tests
│   ├── test_models.py       # Model tests
│   ├── test_pipeline.py     # Pipeline tests
│   └── __init__.py
│
├── data/                    # Data directory
│   ├── raw/                 # Raw CCPD dataset
│   └── processed/           # Processed data
│
├── models/                  # Saved models
│   ├── baseline_model.pth   # Trained baseline
│   ├── pdlpr_best.pth      # Best PDLPR model
│   └── optimized/          # Optimized models
│
└── demo_images/            # Demo images
    ├── input/              # Input examples
    └── output/             # Output results
```

## Dataset

The project uses the **CCPD (Chinese City Parking Dataset)** which contains:

- 250,000+ vehicle images with license plates
- Chinese license plate annotations
- Bounding box coordinates
- Character-level labels

Download the dataset and place `CCPD2019.tar.xz` in the `data/raw/` directory.

## Model Performance

### Current Results

| Component           | Metric             | Performance     |
| ------------------- | ------------------ | --------------- |
| YOLOv5 Detection    | mAP@0.5            | 99.5%           |
| PDLPR Recognition   | Character Accuracy | 75.9%           |
| PDLPR Recognition   | Sequence Accuracy  | 10.6%           |
| End-to-end Pipeline | Processing Time    | ~67ms per image |

_Results obtained on CCPD validation set with real license plate images._

## Development Workflow

### Adding New Features

1. Create feature branch
2. Implement in appropriate module
3. Add tests in `tests/`
4. Update documentation
5. Run validation: `python scripts/validate_setup.py`
6. Run tests: `python run_tests.py`

### Optimization Guidelines

- Models are optimized for CPU deployment
- Use `src/utils/optimization.py` for performance tools
- Profile with integrated performance tracker
- Apply quantization for production deployment

## Troubleshooting

### Common Issues

**Import Errors**: Ensure `src/` is in Python path

```bash
export PYTHONPATH="${PYTHONPATH}:src"
```

**Memory Issues**: Reduce batch size in training scripts

**Missing Dependencies**:

```bash
pip install -r requirements.txt
```

**CUDA Errors**: Project is designed for CPU, ensure device='cpu'

### Getting Help

1. Check `python scripts/validate_setup.py`
2. Run `python run_tests.py` to identify issues
3. Review error logs in terminal output

## Academic Context

This project fulfills requirements for computer vision coursework:

- **Dataset**: CCPD (academic standard)
- **Architecture**: Two-stage detection + recognition pipeline
- **Evaluation**: Comprehensive metrics and baseline comparison
- **Documentation**: Complete technical documentation

## Technical Specifications

- **Input**: RGB vehicle images (various sizes)
- **Output**: License plate text strings
- **Supported Formats**: JPG, PNG
- **Character Set**: Chinese provinces + alphanumeric (67 classes)
- **Sequence Length**: 7 characters (Chinese standard)

## License

Academic project for educational purposes.

## Acknowledgments

- CCPD Dataset authors
- YOLOv5 implementation (Ultralytics)
- PyTorch framework
- OpenCV library
