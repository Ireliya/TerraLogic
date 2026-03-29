# Infrared Detection Tool

This directory contains the **InfraredDetection** tool for the SpatialreasonAgent project, which implements infrared small target detection using the DMIST (Deep Multi-scale Infrared Small Target) framework.

## Overview

The InfraredDetection tool is designed to detect small infrared targets in satellite/aerial imagery using advanced deep learning techniques. It leverages the DMIST framework's LASNet model, which uses temporal sequence analysis to improve detection accuracy for small moving targets.

## Features

- **DMIST Framework Integration**: Uses the state-of-the-art LASNet model for infrared target detection
- **Temporal Sequence Analysis**: Processes 5 consecutive frames for improved motion-based detection
- **Small Target Specialization**: Optimized for detecting small infrared targets in satellite/aerial imagery
- **GPU Acceleration**: Supports CUDA devices for fast inference
- **Visualization Generation**: Creates annotated images showing detection results
- **Standardized Output**: Returns JSON format compatible with SpatialreasonAgent architecture

## Architecture

### Files Structure
```
spatialreason/tools/IR/
├── InfraredDetection.py          # Main tool implementation
├── DMIST/                        # DMIST framework (cloned from GitHub)
│   ├── nets/                     # Neural network architectures
│   ├── utils/                    # Utility functions
│   ├── model_data/               # Model configuration
│   └── ...
├── pre-trained_weights_DMIST-100.pth  # Pre-trained model weights
├── test_infrared_detection.py    # Test script
├── integration_example.py        # Integration demonstration
└── README.md                     # This file
```

### Key Components

1. **InfraredDetectionTool**: Main tool class inheriting from LangChain's BaseTool
2. **DMIST Framework**: Deep learning framework for infrared small target detection
3. **LASNet Model**: Linking-Aware Sliced Network for temporal sequence processing
4. **Visualization Engine**: Creates annotated detection results

## Usage

### Basic Usage

```python
from InfraredDetection import InfraredDetectionTool

# Initialize the tool
tool = InfraredDetectionTool(device="cuda:2")

# Run detection
result = tool._run(
    image_path="path/to/infrared_image.png",
    confidence_threshold=0.5,
    nms_iou_threshold=0.3,
    device="cuda:2"
)

# Parse results
import json
result_data = json.loads(result)
print(f"Detected {result_data['output']['num_detections']} targets")
```

### Input Parameters

- **image_path** (str): Path to infrared image file (PNG, JPG, or BMP)
- **confidence_threshold** (float, default=0.5): Detection confidence threshold (0.0-1.0)
- **nms_iou_threshold** (float, default=0.3): Non-maximum suppression IoU threshold
- **device** (str, default="cuda:2"): Device for inference (cuda:0, cuda:1, etc.)

### Output Format

The tool returns a JSON string with the following structure:

```json
{
  "tool": "infrared_detection",
  "input": {
    "image_path": "path/to/image.png",
    "confidence_threshold": 0.5,
    "nms_iou_threshold": 0.3,
    "device": "cuda:2"
  },
  "output": {
    "detections": [
      {
        "class": "target",
        "confidence": 0.955,
        "polygon": [[112, 21], [118, 21], [118, 27], [112, 27], [112, 21]],
        "center": [115, 24],
        "area": 36
      }
    ],
    "num_detections": 1,
    "image_shape": [256, 256],
    "processed_images": 5,
    "visualization_path": "temp/infrared_detection/result.png",
    "method": "DMIST"
  },
  "execution_id": "abc123",
  "success": true,
  "summary": "Detected 1 infrared target with confidence 0.955."
}
```

## Technical Details

### DMIST Framework

The tool uses the DMIST (Deep Multi-scale Infrared Small Target) framework, which includes:

- **LASNet**: Linking-Aware Sliced Network for temporal processing
- **RDIAN**: Residual Dense Infrared Attention Network as backbone
- **Motion Analysis**: Uses 5 consecutive frames for motion-based detection
- **Multi-scale Processing**: Handles targets at different scales

### Temporal Sequence Processing

The tool requires 5 consecutive frames for optimal performance:
- Current frame (target frame)
- 4 previous frames for motion analysis
- If previous frames don't exist, the current frame is duplicated

### GPU Memory Management

- Uses hardcoded GPU allocation (cuda:2 by default)
- Supports DataParallel for multi-GPU inference
- Automatic fallback to CPU if CUDA unavailable

## Testing

### Run Tests

```bash
cd spatialreason/tools/IR
python test_infrared_detection.py
```

### Run Integration Example

```bash
cd spatialreason/tools/IR
python integration_example.py
```

## Performance

### Test Results

On the provided infrared test images:
- **Detection Accuracy**: Successfully detects small infrared targets
- **Confidence Scores**: Typically 0.6-0.95 for valid targets
- **Processing Speed**: Fast inference with GPU acceleration
- **Temporal Consistency**: Improved accuracy with sequence analysis

### Typical Performance Metrics
- **Input Size**: 512x512 pixels (resized automatically)
- **Target Size**: 8-24 pixels (small targets)
- **Inference Time**: ~100-200ms per sequence (5 frames)
- **Memory Usage**: ~2-4GB GPU memory

## Integration with SpatialreasonAgent

The tool follows SpatialreasonAgent architecture patterns:

1. **Tool Interface**: Inherits from LangChain BaseTool
2. **Standardized I/O**: JSON input/output format
3. **Error Handling**: Comprehensive error reporting
4. **Logging**: Detailed execution information
5. **Visualization**: Automatic result visualization
6. **Device Management**: Configurable GPU allocation

## Dependencies

- PyTorch
- torchvision
- PIL (Pillow)
- OpenCV
- NumPy
- LangChain
- Pydantic

## Limitations

1. **Temporal Requirement**: Needs 5 consecutive frames for optimal performance
2. **Small Targets Only**: Optimized for small infrared targets (8-24 pixels)
3. **GPU Memory**: Requires significant GPU memory for inference
4. **Model Dependency**: Requires pre-trained DMIST weights

## Future Enhancements

1. **Multi-class Detection**: Support for different target types
2. **Real-time Processing**: Streaming video analysis
3. **Adaptive Thresholding**: Dynamic confidence adjustment
4. **Model Optimization**: TensorRT/ONNX optimization for speed
5. **Cloud Integration**: Support for cloud-based inference
