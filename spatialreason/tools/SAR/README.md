# SAR (Synthetic Aperture Radar) Tools for Spatial Reasoning

This directory contains comprehensive SAR analysis tools integrated into the spatial reasoning toolkit, based on the SARATR-X foundation model using HiViT (Hierarchical Vision Transformer) architecture.

## Overview

The SAR tools provide specialized capabilities for analyzing Synthetic Aperture Radar imagery, particularly focused on maritime surveillance and target detection. The implementation wraps the MMDetection framework with simplified interfaces that integrate seamlessly with the spatial reasoning agent architecture.

## Components

### 1. SAR Detection Tool (`sar_detection.py`)
- **Purpose**: Detect objects in SAR imagery, particularly ships in maritime scenes
- **Model**: HiViT-based SARATR-X trained on SSDD (SAR Ship Detection Dataset)
- **Input**: SAR images (PNG, JPG, TIFF)
- **Output**: Bounding boxes, confidence scores, object coordinates
- **Specialization**: Maritime surveillance, ship detection

### 2. SAR Classification Tool (`sar_classification.py`)
- **Purpose**: Classify SAR imagery into different categories
- **Classification Types**:
  - **Scene**: urban, rural, coastal, desert, forest, mountain, agricultural, industrial, water, mixed
  - **Target**: ship, aircraft, vehicle, building, bridge, tank, helicopter, fighter_jet, cargo_ship, fishing_boat
  - **Fine-grained**: T62_tank, T72_tank, BMP2_vehicle, BTR60_vehicle, Boeing_aircraft, cargo_ship, etc.
- **Model**: HiViT-based classifier with task-specific heads
- **Input**: SAR images (PNG, JPG, TIFF)
- **Output**: Class predictions with confidence scores

### 3. Configuration Management (`sar_config.py`)
- **Purpose**: Simplified configuration abstraction over MMDetection's complex config system
- **Features**:
  - Model configuration (HiViT parameters)
  - Detection parameters (thresholds, NMS, etc.)
  - Classification parameters (class mappings, confidence)
  - System configuration (GPU, paths, logging)
- **Benefits**: Easy parameter tuning without MMDetection expertise

### 4. Model Management (`sar_model_manager.py`)
- **Purpose**: Unified model loading, initialization, and GPU management
- **Features**:
  - Model caching for efficient reuse
  - GPU validation and allocation
  - Error handling and fallback mechanisms
  - Memory management and cleanup
- **GPU Strategy**: Follows established pattern (GPU 2 for RemoteSAM tools)

### 5. Test Suite (`test_sar_tools.py`)
- **Purpose**: Comprehensive testing of all SAR components
- **Coverage**:
  - Configuration management
  - Model loading and validation
  - Tool initialization and execution
  - Integration with spatial reasoning toolkit
  - Error handling and edge cases

## Installation and Setup

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install pathlib

# MMDetection framework (required for SAR detection)
pip install mmdet
pip install mmengine
pip install mmcv
```

### Model Weights
The SAR tools require the pretrained HiViT model weights:
- **Location**: `spatialreason/tools/SAR/codelab/model/mae_hivit_base_1600ep.pth`
- **Source**: SARATR-X foundation model
- **Size**: ~500MB
- **Training**: Pre-trained on large-scale SAR datasets

### Configuration Files
- **Detection Config**: `spatialreason/tools/SAR/codelab/code/detection/configs/_hivit_/hivit_base_SSDD.py`
- **Model Architecture**: HiViT with embed_dim=512, depths=[2, 2, 20], num_heads=8

## Usage Examples

### SAR Detection
```python
from spatialreason.tools.SAR import create_sar_detection_tool

# Create detection tool
detector = create_sar_detection_tool(device="cuda:2")

# Run detection
result = detector._run(
    image_path="path/to/sar_image.jpg",
    confidence_threshold=0.3
)

# Parse results
import json
detection_results = json.loads(result)
print(f"Found {detection_results['total_detections']} ships")
```

### SAR Classification
```python
from spatialreason.tools.SAR import create_sar_classification_tool

# Create classification tool
classifier = create_sar_classification_tool(device="cuda:2")

# Run scene classification
result = classifier._run(
    image_path="path/to/sar_image.jpg",
    classification_type="scene",
    confidence_threshold=0.5
)

# Parse results
import json
classification_results = json.loads(result)
top_prediction = classification_results['classification_results']['top_prediction']
print(f"Scene classified as: {top_prediction['class']} ({top_prediction['confidence']:.2%})")
```

### Configuration Management
```python
from spatialreason.tools.SAR.sar_config import SARConfigManager

# Create configuration manager
config = SARConfigManager()

# Update detection parameters
config.update_detection_params(
    confidence_threshold=0.4,
    max_detections=50
)

# Update system parameters
config.update_system_params(
    device="cuda:1",
    output_dir="custom/output/path"
)

# Save configuration
config.save_config("custom_sar_config.json")
```

## Integration with Spatial Reasoning Toolkit

### Tool Registration
The SAR tools are automatically registered in the spatial reasoning toolkit:
- **Tool Names**: `sar_detection`, `sar_classification`
- **Category**: `perception`
- **Registry**: Included in default perception toolkit
- **Selector**: Available through unified selector system

### Parameter Schemas
Both tools follow the toolkit's standardized parameter schema:
```python
{
    "image_path": {"type": "string", "description": "Path to SAR image"},
    "confidence_threshold": {"type": "number", "default": 0.3},
    "device": {"type": "string", "default": "cuda:2"}
}
```

### Output Standardization
All SAR tools produce JSON outputs with standardized structure:
```json
{
    "success": true,
    "image_path": "path/to/image.jpg",
    "total_detections": 3,
    "detections": [...],
    "model_info": {...},
    "summary": "Natural language summary of results"
}
```

## GPU Management

### Device Allocation
- **Default Device**: `cuda:2` (following established pattern)
- **Validation**: Automatic GPU availability checking
- **Fallback**: Error handling for unavailable devices
- **Memory**: Automatic cleanup and cache management

### Error Handling
- **Device Validation**: Checks GPU availability before model loading
- **Model Loading**: Graceful handling of missing weights or configs
- **Inference**: Robust error recovery during processing
- **Memory**: Automatic GPU memory cleanup

## Performance Considerations

### Model Caching
- Models are cached after first load for efficiency
- Cache keys include device and model type
- Manual cache clearing available for memory management

### Preprocessing
- **Detection**: Images resized to 800x800 pixels
- **Classification**: Images resized to 224x224 pixels
- **Normalization**: SAR-specific (mean=[0,0,0], std=[1,1,1])

### Batch Processing
- Single image processing (following single-tool execution architecture)
- Efficient memory usage with automatic cleanup
- Progress tracking for long-running operations

## Testing

### Running Tests
```bash
# Run all SAR tests
python -m pytest spatialreason/tools/SAR/test_sar_tools.py -v

# Run specific test categories
python -m pytest spatialreason/tools/SAR/test_sar_tools.py::TestSARConfiguration -v
python -m pytest spatialreason/tools/SAR/test_sar_tools.py::TestSARDetectionTool -v
```

### Test Coverage
- Configuration management and validation
- Model loading and GPU allocation
- Tool initialization and execution
- Error handling and edge cases
- Integration with spatial reasoning toolkit

## Troubleshooting

### Common Issues

1. **MMDetection Import Errors**
   - Ensure MMDetection is properly installed
   - Check Python path includes SAR codebase directories
   - Verify compatible versions of mmdet, mmengine, mmcv

2. **GPU Memory Issues**
   - Use `clear_cache()` method to free GPU memory
   - Reduce batch size or image resolution
   - Check GPU memory availability

3. **Model Loading Failures**
   - Verify model weights exist at specified path
   - Check file permissions and accessibility
   - Ensure sufficient disk space

4. **Configuration Errors**
   - Validate configuration file format (JSON)
   - Check parameter names and types
   - Use default configuration as reference

### Debug Mode
Enable detailed logging by setting log level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Adding New SAR Capabilities
1. Follow the established tool interface pattern
2. Implement proper error handling and GPU management
3. Add comprehensive tests
4. Update documentation and examples
5. Ensure integration with spatial reasoning toolkit

### Code Style
- Follow PEP 8 conventions
- Use type hints for all functions
- Include comprehensive docstrings
- Add logging for debugging and monitoring

## References

- **SARATR-X Paper**: Foundation model for SAR target recognition
- **HiViT Architecture**: Hierarchical Vision Transformer
- **SSDD Dataset**: SAR Ship Detection Dataset
- **MMDetection**: Object detection framework
- **Spatial Reasoning Toolkit**: Parent framework documentation
