# Dataset Directory

This directory contains binary dataset files for training and testing the federated learning model.

## Directory Structure

```
datasets/
└── plant-leaf-disease-224x224/
    ├── train/
    │   └── train_data.bin       (1750 samples)
    └── test/
        └── test_data.bin         (500 samples)
```

## Binary Data Format

Each sample in the binary file contains:
- **1 byte**: Label (class index, typically 0-3 for 4-class classification)
- **150,528 bytes**: Image data (224×224×3 RGB image in row-major order)

**Total bytes per sample**: 150,529 bytes

## Example Data

The default dataset expects apple leaf disease classification with 4 classes:
- Class 0: Gray Spot Disease (灰斑病)
- Class 1: Rust Disease (锈病)
- Class 2: Large Spot Disease (大斑病)
- Class 3: Healthy (健康)

However, **you can use any image classification dataset** with the same format.

## Data Preparation Instructions

### 1. Convert Images to Binary Format

```python
import numpy as np
from PIL import Image

def create_binary_dataset(image_paths, labels, output_file):
    with open(output_file, 'wb') as f:
        for img_path, label in zip(image_paths, labels):
            # Read and resize image to 224x224
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            img_array = np.array(img, dtype=np.uint8)
            
            # Write label (1 byte)
            f.write(bytes([label]))
            
            # Write image data (150528 bytes)
            f.write(img_array.tobytes())
```

### 2. Dataset Requirements

- **Image resolution**: 224×224 pixels, RGB format
- **Training samples**: 1750 (adjustable)
- **Test samples**: 500 (adjustable)
- **Classes**: 4 (or modify code to support different number of classes)

### 3. Place Files

Place the binary files in the appropriate directories:
- `datasets/plant-leaf-disease-224x224/train/train_data.bin`
- `datasets/plant-leaf-disease-224x224/test/test_data.bin`

## Notes

- The actual dataset files are not included in this repository due to size constraints.
- Prepare your own dataset using the format specified above.
- Ensure the number of samples matches the code expectations (1750 for train, 500 for test).
