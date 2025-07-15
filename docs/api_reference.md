# API Reference

## Core Classes

### IntegratedPipeline

Main pipeline class for license plate recognition.

```python
from pipeline import create_pipeline

pipeline = create_pipeline(
    detection_model_path=None,
    recognition_model_path=None,
    baseline_model_path='models/baseline_model.pth'
)
```

#### Methods

**detect_and_recognize(image_path, use_pdlpr=True)**

Process image and return recognition results.

- **Parameters**:
  - `image_path` (str|Path|np.ndarray): Input image
  - `use_pdlpr` (bool): Use PDLPR model if available
- **Returns**: List of detection results
- **Example**:
```python
results = pipeline.detect_and_recognize('car.jpg')
for result in results:
    print(f"Plate: {result['plate_text']}")
```

**process_batch(image_paths, output_dir=None)**

Process multiple images.

- **Parameters**:
  - `image_paths` (list): List of image paths
  - `output_dir` (str, optional): Output directory for results
- **Returns**: Dictionary of results per image

### BaselineCNN

Simple CNN model for license plate recognition.

```python
from models.baseline import BaselineCNN, Trainer

model = BaselineCNN(num_chars=67, sequence_length=7)
trainer = Trainer(model, device='cpu')
```

#### Methods

**forward(x)**

Forward pass through the network.

- **Parameters**: `x` (torch.Tensor): Input tensor [B, 3, 64, 128]
- **Returns**: Output tensor [B, 7, 67]

### PDLPRModel

Advanced character recognition model.

```python
from recognition.pdlpr_model import create_pdlpr_model, PDLPRTrainer

model = create_pdlpr_model()
trainer = PDLPRTrainer(model, device='cpu')
```

#### Methods

**forward(x)**

Forward pass through PDLPR network.

- **Parameters**: `x` (torch.Tensor): Input tensor [B, 3, 64, 128]
- **Returns**: Output tensor [B, 7, 67]

### YOLOv5Detector

License plate detection wrapper.

```python
from detection.yolo_model import YOLOv5Detector

detector = YOLOv5Detector(model_path=None, device='cpu')
```

#### Methods

**detect_plates(image)**

Detect license plates in image.

- **Parameters**: `image` (np.ndarray|str): Input image
- **Returns**: List of detection dictionaries

**crop_plates(image, detections)**

Extract plate regions from detections.

- **Parameters**:
  - `image` (np.ndarray): Input image
  - `detections` (list): Detection results
- **Returns**: List of cropped plate dictionaries

## Utility Functions

### Metrics

```python
from utils.metrics import character_accuracy, sequence_accuracy, MetricsTracker

# Calculate accuracies
char_acc = character_accuracy(predictions, targets)
seq_acc = sequence_accuracy(predictions, targets)

# Track metrics during training
tracker = MetricsTracker()
tracker.update(outputs, labels, loss)
metrics = tracker.compute()
```

### Configuration

```python
from utils.config import MODEL_CONFIG, TRAIN_CONFIG, ALL_CHARS

# Access configuration
input_size = MODEL_CONFIG['img_height'], MODEL_CONFIG['img_width']
vocab_size = len(ALL_CHARS)
```

### Optimization

```python
from utils.optimization import ModelOptimizer, benchmark_operations

# Optimize model for CPU
optimized_model = ModelOptimizer.optimize_for_cpu(model)

# Benchmark performance
stats = benchmark_operations()
```

## Data Loading

### CCPDDataLoader

```python
from data.data_loader import CCPDDataLoader

loader = CCPDDataLoader('data/raw/CCPD2019.tar.xz')
result = loader.parse_filename(filename)
```

#### Methods

**parse_filename(filename)**

Parse CCPD filename to extract annotations.

- **Parameters**: `filename` (str): CCPD filename
- **Returns**: Dictionary with plate text, characters, and bbox

### SyntheticDataset

```python
from models.baseline import SyntheticDataset
from torch.utils.data import DataLoader

dataset = SyntheticDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=16)
```

## Training Scripts

### Baseline Training

```python
python src/models/baseline.py --train --epochs 10 --batch_size 16
```

**Arguments**:
- `--train`: Enable training mode
- `--evaluate`: Enable evaluation mode
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size

### PDLPR Training

```python
python src/recognition/train_pdlpr.py --epochs 20 --batch_size 16
```

**Arguments**:
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--save_dir`: Model save directory

## Command Line Interface

### Main Pipeline

```bash
python main.py [OPTIONS]
```

**Options**:
- `--input PATH`: Input image path
- `--output PATH`: Output text file path
- `--demo`: Run demonstration mode
- `--use_pipeline`: Use integrated pipeline
- `--model PATH`: Model path (default: models/baseline_model.pth)

### Demo Script

```bash
python demo.py
```

Runs simple demonstration of the recognition pipeline.

### Test Runner

```bash
python run_tests.py [TEST_MODULE]
```

**Arguments**:
- `TEST_MODULE` (optional): Specific test module to run

## Error Handling

### Common Exceptions

**ModelNotFoundError**: Raised when model file is not found
**InvalidImageError**: Raised when image cannot be loaded
**RecognitionError**: Raised when recognition fails

### Return Formats

**Detection Result**:
```python
{
    'bbox': [x, y, width, height],
    'confidence': 0.85,
    'class': 'license_plate'
}
```

**Recognition Result**:
```python
{
    'plate_text': 'ABC1234',
    'bbox': [x, y, width, height],
    'detection_confidence': 0.85,
    'recognition_confidence': 0.90,
    'method': 'pdlpr',
    'plate_id': 0
}
```

**Performance Stats**:
```python
{
    'detection': {
        'count': 10,
        'avg_time': 0.015,
        'min_time': 0.012,
        'max_time': 0.020
    },
    'recognition': {
        'count': 10,
        'avg_time': 0.025,
        'min_time': 0.020,
        'max_time': 0.035
    }
}
```