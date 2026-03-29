"""
Utility functions for spatial reasoning tools.
Provides common validation, processing, and helper functions.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


def validate_ground_resolution(meters_per_pixel: float) -> Dict[str, Any]:
    """
    Validate ground resolution parameter.
    
    Args:
        meters_per_pixel: Ground resolution in meters per pixel
        
    Returns:
        Dictionary with validation result and message
    """
    if not isinstance(meters_per_pixel, (int, float)):
        return {
            "valid": False,
            "message": f"Ground resolution must be a number, got {type(meters_per_pixel).__name__}"
        }
    
    if meters_per_pixel <= 0:
        return {
            "valid": False,
            "message": f"Ground resolution must be positive, got {meters_per_pixel}"
        }
    
    # Reasonable range for satellite imagery (0.1m to 100m per pixel)
    if meters_per_pixel < 0.1 or meters_per_pixel > 100.0:
        return {
            "valid": False,
            "message": f"Ground resolution {meters_per_pixel} m/px is outside reasonable range (0.1-100.0 m/px)"
        }
    
    return {
        "valid": True,
        "message": "Ground resolution is valid"
    }


def validate_image_path(image_path: str) -> Dict[str, Any]:
    """
    Validate image file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with validation result and message
    """
    if not isinstance(image_path, str):
        return {
            "valid": False,
            "message": f"Image path must be a string, got {type(image_path).__name__}"
        }
    
    if not image_path.strip():
        return {
            "valid": False,
            "message": "Image path cannot be empty"
        }
    
    path = Path(image_path)
    
    if not path.exists():
        return {
            "valid": False,
            "message": f"Image file not found: {image_path}"
        }
    
    if not path.is_file():
        return {
            "valid": False,
            "message": f"Path is not a file: {image_path}"
        }
    
    # Check file extension
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    if path.suffix.lower() not in valid_extensions:
        return {
            "valid": False,
            "message": f"Unsupported image format: {path.suffix}. Supported: {', '.join(valid_extensions)}"
        }
    
    return {
        "valid": True,
        "message": "Image path is valid"
    }


def validate_confidence_threshold(threshold: float) -> Dict[str, Any]:
    """
    Validate confidence threshold parameter.
    
    Args:
        threshold: Confidence threshold value
        
    Returns:
        Dictionary with validation result and message
    """
    if not isinstance(threshold, (int, float)):
        return {
            "valid": False,
            "message": f"Confidence threshold must be a number, got {type(threshold).__name__}"
        }
    
    if not 0.0 <= threshold <= 1.0:
        return {
            "valid": False,
            "message": f"Confidence threshold must be between 0.0 and 1.0, got {threshold}"
        }
    
    return {
        "valid": True,
        "message": "Confidence threshold is valid"
    }


def validate_text_prompt(text_prompt: str) -> Dict[str, Any]:
    """
    Validate text prompt parameter.
    
    Args:
        text_prompt: Natural language text prompt
        
    Returns:
        Dictionary with validation result and message
    """
    if not isinstance(text_prompt, str):
        return {
            "valid": False,
            "message": f"Text prompt must be a string, got {type(text_prompt).__name__}"
        }
    
    if not text_prompt.strip():
        return {
            "valid": False,
            "message": "Text prompt cannot be empty"
        }
    
    # Check for reasonable length
    if len(text_prompt.strip()) < 2:
        return {
            "valid": False,
            "message": "Text prompt is too short (minimum 2 characters)"
        }
    
    if len(text_prompt) > 500:
        return {
            "valid": False,
            "message": f"Text prompt is too long ({len(text_prompt)} characters, maximum 500)"
        }
    
    return {
        "valid": True,
        "message": "Text prompt is valid"
    }


def create_output_directory(base_dir: str, tool_name: str) -> Path:
    """
    Create output directory for tool results.
    
    Args:
        base_dir: Base directory for outputs
        tool_name: Name of the tool
        
    Returns:
        Path object for the created directory
    """
    output_dir = Path(base_dir) / tool_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json_result(data: Dict[str, Any], output_path: str) -> bool:
    """
    Save dictionary data as JSON file.

    Args:
        data: Dictionary to save
        output_path: Path to save the JSON file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {output_path}: {e}")
        return False


def save_perception_tool_output(image_path: str, tool_name: str, tool_result: str, dataset_dir: str = "results") -> Dict[str, str]:
    """
    Save perception tool outputs with consistent naming scheme for cross-validation and error analysis.

    Args:
        image_path: Path to the input image
        tool_name: Name of the perception tool (detection, segmentation, classification)
        tool_result: JSON string result from the tool
        dataset_dir: Directory to save outputs (default: "results")

    Returns:
        Dictionary with saved file paths
    """
    try:
        import json
        import shutil
        from pathlib import Path

        # Extract image name without extension
        image_name = Path(image_path).stem

        # Create dataset directory
        dataset_path = Path(dataset_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Parse tool result to extract file paths
        try:
            result_data = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Could not parse tool result for {tool_name}")
            return {}

        saved_files = {}

        # Save main result JSON with consistent naming
        result_filename = f"{image_name}_{tool_name}_result.json"
        result_path = dataset_path / result_filename

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        saved_files['result_json'] = str(result_path)

        # Copy and rename tool-specific output files
        if tool_name == "segmentation":
            # Copy segmentation mask
            if "segmented_mask_path" in result_data:
                source_mask = result_data["segmented_mask_path"]
                if Path(source_mask).exists():
                    mask_filename = f"{image_name}_{tool_name}.png"
                    mask_dest = dataset_path / mask_filename
                    shutil.copy2(source_mask, mask_dest)
                    saved_files['mask'] = str(mask_dest)

        elif tool_name == "detection":
            # Copy detection masks and visualization
            if "detection_masks" in result_data:
                for class_name, mask_path in result_data["detection_masks"].items():
                    if Path(mask_path).exists():
                        mask_filename = f"{image_name}_{tool_name}_{class_name}.png"
                        mask_dest = dataset_path / mask_filename
                        shutil.copy2(mask_path, mask_dest)
                        saved_files[f'mask_{class_name}'] = str(mask_dest)

            # Copy visualization if available
            if "visualization_path" in result_data:
                viz_source = result_data["visualization_path"]
                if Path(viz_source).exists():
                    viz_filename = f"{image_name}_{tool_name}_visualization.png"
                    viz_dest = dataset_path / viz_filename
                    shutil.copy2(viz_source, viz_dest)
                    saved_files['visualization'] = str(viz_dest)

        elif tool_name == "classification":
            # Copy classification visualization if available
            if "visualization_path" in result_data:
                viz_source = result_data["visualization_path"]
                if Path(viz_source).exists():
                    viz_filename = f"{image_name}_{tool_name}_visualization.png"
                    viz_dest = dataset_path / viz_filename
                    shutil.copy2(viz_source, viz_dest)
                    saved_files['visualization'] = str(viz_dest)

        print(f"✅ Saved {tool_name} outputs for {image_name}: {len(saved_files)} files")
        return saved_files

    except Exception as e:
        print(f"❌ Error saving perception tool output: {e}")
        return {}


def load_json_result(input_path: str) -> Optional[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Dictionary with loaded data, or None if failed
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {input_path}: {e}")
        return None


def calculate_pixel_area(meters_per_pixel: float) -> float:
    """
    Calculate area of a single pixel in square meters.
    
    Args:
        meters_per_pixel: Ground resolution in meters per pixel
        
    Returns:
        Area of one pixel in square meters
    """
    return meters_per_pixel ** 2


def pixels_to_meters(pixels: int, meters_per_pixel: float) -> float:
    """
    Convert pixel distance to meters.
    
    Args:
        pixels: Distance in pixels
        meters_per_pixel: Ground resolution in meters per pixel
        
    Returns:
        Distance in meters
    """
    return pixels * meters_per_pixel


def meters_to_pixels(meters: float, meters_per_pixel: float) -> int:
    """
    Convert meter distance to pixels.
    
    Args:
        meters: Distance in meters
        meters_per_pixel: Ground resolution in meters per pixel
        
    Returns:
        Distance in pixels (rounded to nearest integer)
    """
    return int(round(meters / meters_per_pixel))


def calculate_area_statistics(areas: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures for a list of areas.
    
    Args:
        areas: List of area values
        
    Returns:
        Dictionary with statistical measures
    """
    if not areas:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0
        }
    
    areas_array = np.array(areas)
    
    return {
        "count": len(areas),
        "total": float(np.sum(areas_array)),
        "mean": float(np.mean(areas_array)),
        "median": float(np.median(areas_array)),
        "min": float(np.min(areas_array)),
        "max": float(np.max(areas_array)),
        "std": float(np.std(areas_array))
    }


def format_area(area_m2: float, precision: int = 2) -> str:
    """
    Format area value with appropriate units.
    
    Args:
        area_m2: Area in square meters
        precision: Number of decimal places
        
    Returns:
        Formatted area string with units
    """
    if area_m2 < 1.0:
        return f"{area_m2 * 10000:.{precision}f} cm²"
    elif area_m2 < 10000:  # Less than 1 hectare
        return f"{area_m2:.{precision}f} m²"
    else:  # 1 hectare or more
        hectares = area_m2 / 10000
        return f"{hectares:.{precision}f} ha"


def format_distance(distance_m: float, precision: int = 2) -> str:
    """
    Format distance value with appropriate units.
    
    Args:
        distance_m: Distance in meters
        precision: Number of decimal places
        
    Returns:
        Formatted distance string with units
    """
    if distance_m < 1.0:
        return f"{distance_m * 100:.{precision}f} cm"
    elif distance_m < 1000:
        return f"{distance_m:.{precision}f} m"
    else:
        return f"{distance_m / 1000:.{precision}f} km"


def create_error_response(error_message: str, **additional_fields) -> str:
    """
    Create standardized error response JSON.
    
    Args:
        error_message: Error message to include
        **additional_fields: Additional fields to include in response
        
    Returns:
        JSON string with error response
    """
    response = {
        "success": False,
        "error": error_message,
        **additional_fields
    }
    return json.dumps(response, indent=2)


def create_success_response(data: Dict[str, Any]) -> str:
    """
    Create standardized success response JSON.

    Args:
        data: Data to include in response

    Returns:
        JSON string with success response
    """
    response = {
        "success": True,
        **data
    }
    return json.dumps(response, indent=2)


def suggest_optimal_parameters(image_path: str, meters_per_pixel: float = 0.3) -> Dict[str, Any]:
    """
    Suggest optimal parameters for image processing based on image characteristics.

    Args:
        image_path: Path to the input image
        meters_per_pixel: Ground resolution in meters per pixel

    Returns:
        Dictionary with suggested parameters
    """
    try:
        # Load image to analyze
        image = Image.open(image_path)
        width, height = image.size

        # Calculate image area in square meters
        pixel_area = meters_per_pixel ** 2
        total_area_m2 = width * height * pixel_area

        # Suggest buffer distance based on image resolution
        if meters_per_pixel <= 0.5:  # High resolution
            suggested_buffer_m = 10.0
        elif meters_per_pixel <= 1.0:  # Medium resolution
            suggested_buffer_m = 20.0
        else:  # Lower resolution
            suggested_buffer_m = 50.0

        suggested_buffer_pixels = int(suggested_buffer_m / meters_per_pixel)

        # Suggest confidence thresholds based on image size
        if total_area_m2 < 100000:  # Small area, use higher confidence
            detection_confidence = 0.7
            classification_confidence = 0.75
        elif total_area_m2 < 1000000:  # Medium area
            detection_confidence = 0.6
            classification_confidence = 0.65
        else:  # Large area, use lower confidence to catch more objects
            detection_confidence = 0.5
            classification_confidence = 0.55

        return {
            "image_dimensions": {"width": width, "height": height},
            "total_area_m2": round(total_area_m2, 2),
            "suggested_buffer_distance_m": suggested_buffer_m,
            "suggested_buffer_distance_pixels": suggested_buffer_pixels,
            "suggested_confidence_thresholds": {
                "detection": detection_confidence,
                "classification": classification_confidence
            },
            "processing_recommendations": {
                "chunk_processing": total_area_m2 > 10000000,  # Process in chunks if very large
                "memory_optimization": width * height > 10000 * 10000  # Optimize memory for large images
            }
        }

    except Exception as e:
        return {
            "error": f"Failed to analyze image: {str(e)}",
            "suggested_buffer_distance_m": 30.0,
            "suggested_buffer_distance_pixels": int(30.0 / meters_per_pixel),
            "suggested_confidence_thresholds": {
                "detection": 0.6,
                "classification": 0.65
            }
        }
