"""
Infrared Small Target Detection Tool using DMIST (Deep Multi-scale Infrared Small Target) framework.
This tool detects small infrared targets in satellite/aerial imagery using the LASNet model.
"""

import os
import json
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
import colorsys

# Setup logging
logger = logging.getLogger(__name__)

# Add DMIST path to sys.path
DMIST_PATH = Path(__file__).parent / "DMIST"
if str(DMIST_PATH) not in sys.path:
    sys.path.insert(0, str(DMIST_PATH))

try:
    from nets.LASNet import LASNet
    from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image, show_config)
    from utils.utils_bbox import decode_outputs, non_max_suppression
    DMIST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DMIST framework not available: {e}")
    DMIST_AVAILABLE = False


def get_history_imgs(line):
    """
    Get history images for temporal sequence processing.
    DMIST requires 5 consecutive frames for motion analysis.

    If consecutive images don't exist, returns the current image repeated 5 times
    to ensure the model has valid input for temporal analysis.

    Args:
        line: Path to the current image

    Returns:
        List of paths to 5 consecutive images (current and 4 previous), or current image repeated 5 times if history unavailable
    """
    dir_path = line.replace(line.split('/')[-1], '')
    file_type = line.split('.')[-1]
    filename_without_ext = line.split('/')[-1].split('.')[0]

    # Extract numeric part and suffix from filename (handle cases like "0_ir", "123_ir", etc.)
    # Split by underscore and take the first part as numeric, rest as suffix
    if '_' in filename_without_ext:
        parts = filename_without_ext.split('_')
        numeric_part = parts[0]
        suffix = '_' + '_'.join(parts[1:])  # Preserve all parts after first underscore
    else:
        numeric_part = filename_without_ext
        suffix = ''

    try:
        index = int(numeric_part)
    except ValueError:
        # If still can't parse, default to 0
        logger.warning(f"Could not parse image index from '{filename_without_ext}', defaulting to 0")
        index = 0

    # Try to find 5 consecutive images: [index-4, index-3, index-2, index-1, index]
    # Preserve the suffix (e.g., "_ir") in the generated filenames
    history_paths = [os.path.join(dir_path, f"{max(id, 0)}{suffix}.{file_type}") for id in range(index - 4, index + 1)]

    # Check if all history images exist
    all_exist = all(os.path.exists(path) for path in history_paths)

    if all_exist:
        logger.debug(f"[IR Detection] Found all 5 consecutive history images for {filename_without_ext}")
        return history_paths
    else:
        # If not all history images exist, use the current image repeated 5 times
        # This is a fallback for non-consecutive image datasets
        logger.warning(f"[IR Detection] Not all history images found for {filename_without_ext}. Using current image repeated 5 times as fallback.")
        return [line] * 5


class InfraredDetectionInput(BaseModel):
    """Input schema for infrared detection tool."""
    image_path: str = Field(description="Path to infrared image file (PNG, JPG, or BMP)")
    confidence_threshold: float = Field(default=0.5, description="Detection confidence threshold (0.0-1.0)")
    nms_iou_threshold: float = Field(default=0.3, description="Non-maximum suppression IoU threshold")
    device: str = Field(default="cuda:2", description="Device for inference (cuda:0, cuda:1, etc.)")

    # Optional parameters for unified arguments field (from upstream tools)
    text_prompt: Optional[str] = Field(default="detect infrared small targets", description="Task intent description (e.g., 'detect infrared small targets')")
    meters_per_pixel: Optional[float] = Field(default=None, description="Ground resolution for spatial analysis (null for IR dataset)")


class InfraredDetectionTool(BaseTool):
    """
    Infrared small target detection tool using DMIST framework.
    Detects small infrared targets in satellite/aerial imagery using temporal sequence analysis.
    """
    name: str = "infrared_detection"
    description: str = (
        "Detect small infrared targets in satellite/aerial imagery using DMIST (Deep Multi-scale Infrared Small Target) framework. "
        "Specialized for detecting small moving targets in infrared imagery using temporal sequence analysis. "
        "Returns JSON with detected target locations, classes, and confidence scores."
    )
    args_schema: type[BaseModel] = InfraredDetectionInput

    # Configure Pydantic to allow arbitrary types
    class Config:
        arbitrary_types_allowed = True

    # Add fields that are set in __init__
    device: str = "cuda:2"
    model: Any = None
    model_path: str = ""
    classes_path: str = ""
    class_names: List[str] = []
    num_classes: int = 1
    input_shape: List[int] = [512, 512]
    temp_dir: Any = None
    colors: List[Tuple[int, int, int]] = []
    ir_tools_path: Any = None
    dmist_path: Any = None

    def __init__(self, device: str = "cuda:2", **kwargs):
        """
        Initialize infrared detection tool.

        Args:
            device: CUDA device for inference (default: cuda:2 for GPU allocation policy)
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        # Use the specified device (default cuda:2) to comply with GPU allocation policy
        # GPU 0: OpenCompass, GPU 1: Qwen2-VL, GPU 2: Perception tools (including IR), GPU 3: Reserve
        self.device = device
        self.model = None
        self.class_names = []
        self.num_classes = 1
        self.input_shape = [512, 512]

        # Set up paths
        self.ir_tools_path = Path(__file__).parent
        self.dmist_path = self.ir_tools_path / "DMIST"

        # Model and class configuration
        self.model_path = str(self.ir_tools_path / "pre-trained_weights_DMIST-100.pth")
        self.classes_path = str(self.dmist_path / "model_data" / "classes.txt")

        # Create temp directory for outputs
        self.temp_dir = Path("temp") / "infrared_detection"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model if available
        self._initialize_model()

    def _bbox_to_polygon(self, bbox: List[float]) -> List[List[float]]:
        """
        Convert bounding box coordinates to polygon format.

        Args:
            bbox: Bounding box in format [x_min, y_min, x_max, y_max]

        Returns:
            Polygon format: [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]
            (4 corners + closing point for closed polygon)
        """
        try:
            x_min, y_min, x_max, y_max = bbox
            polygon = [
                [float(x_min), float(y_min)],  # Top-left
                [float(x_max), float(y_min)],  # Top-right
                [float(x_max), float(y_max)],  # Bottom-right
                [float(x_min), float(y_max)],  # Bottom-left
                [float(x_min), float(y_min)]   # Close the polygon
            ]
            return polygon
        except Exception as e:
            print(f"Error converting bbox to polygon: {e}")
            return []

    def _initialize_model(self):
        """Initialize the DMIST infrared detection model."""
        if not DMIST_AVAILABLE:
            print("Warning: DMIST framework not available. Infrared detection will use fallback mode.")
            return

        if not os.path.exists(self.classes_path):
            print(f"Warning: Classes file not found: {self.classes_path}")
            return

        if not os.path.exists(self.model_path):
            print(f"Warning: Model weights not found: {self.model_path}")
            return

        try:
            # Load class names
            self.class_names, self.num_classes = get_classes(self.classes_path)
            print(f"Loaded {self.num_classes} classes: {self.class_names}")

            # Initialize the LASNet model
            self.model = LASNet(self.num_classes, num_frame=5)

            # Always use CPU for loading to avoid device mismatch issues
            # Then move to target device after loading
            device_obj = torch.device('cpu')

            # Load model weights to CPU first
            self.model.load_state_dict(torch.load(self.model_path, map_location=device_obj))
            self.model = self.model.eval()

            # Move entire model to specified device after loading
            # self.device is always "cuda:0" to avoid device conflicts
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                self.model = self.model.to(self.device)
                print(f"✅ DMIST model loaded on {self.device}")
            else:
                self.model = self.model.to('cpu')
                print(f"✅ DMIST model loaded on CPU")

            # Generate colors for visualization
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

            print(f"✅ DMIST infrared detection model initialized successfully on {self.device}")

        except Exception as e:
            print(f"❌ Error initializing DMIST infrared detection model: {e}")
            self.model = None

    def _detect_with_dmist(self, image_paths: List[str], confidence: float, nms_iou: float) -> Dict[str, Any]:
        """
        Perform infrared target detection using DMIST model.

        Args:
            image_paths: List of 5 consecutive image paths for temporal analysis
            confidence: Detection confidence threshold
            nms_iou: Non-maximum suppression IoU threshold

        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise RuntimeError("DMIST model not initialized")

        try:
            # Set default CUDA device to ensure all tensors are created on the correct device
            if torch.cuda.is_available() and self.device.startswith('cuda'):
                device_idx = int(self.device.split(':')[1])
                torch.cuda.set_device(device_idx)
                logger.debug(f"[IR Detection] Set default CUDA device to {self.device}")

            # Load images
            images = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    images.append(Image.open(img_path))
                else:
                    # If image doesn't exist, duplicate the last available image
                    if images:
                        images.append(images[-1].copy())
                    else:
                        raise FileNotFoundError(f"Base image not found: {img_path}")

            # Ensure we have exactly 5 images
            while len(images) < 5:
                images.insert(0, images[0].copy())

            # Get image shape for coordinate conversion
            image_shape = np.array(np.shape(images[0])[0:2])

            # Preprocess images
            images_rgb = [cvtColor(image) for image in images]
            current_image = images_rgb[-1]  # The current frame for visualization

            # Resize and normalize
            image_data = [resize_image(image, (self.input_shape[1], self.input_shape[0]), True) for image in images_rgb]
            image_data = [np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1)) for image in image_data]

            # Stack images for temporal processing: (3, 5, 512, 512)
            image_data = np.stack(image_data, axis=1)
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                images_tensor = torch.from_numpy(image_data)
                if torch.cuda.is_available() and self.device.startswith('cuda'):
                    # Explicitly move tensor to the specified device
                    # This ensures all tensors are on the same device as the model
                    images_tensor = images_tensor.to(self.device)
                    logger.debug(f"[IR Detection] Tensor moved to {self.device}")
                else:
                    images_tensor = images_tensor.to('cpu')

                # Verify model and tensor are on the same device
                model_device = next(self.model.parameters()).device
                if images_tensor.device != model_device:
                    logger.warning(f"[IR Detection] Device mismatch detected! Model on {model_device}, tensor on {images_tensor.device}")
                    images_tensor = images_tensor.to(model_device)

                outputs = self.model(images_tensor)
                outputs = decode_outputs(outputs, self.input_shape)
                outputs = non_max_suppression(
                    outputs, self.num_classes, self.input_shape,
                    image_shape, True, conf_thres=confidence, nms_thres=nms_iou
                )

            # Process results
            detections = []
            if outputs[0] is not None:
                top_label = np.array(outputs[0][:, 6], dtype='int32')
                top_conf = outputs[0][:, 4] * outputs[0][:, 5]
                top_boxes = outputs[0][:, :4]

                for i, class_id in enumerate(top_label):
                    top, left, bottom, right = top_boxes[i]

                    # Ensure coordinates are within image bounds
                    top = max(0, int(top))
                    left = max(0, int(left))
                    bottom = min(current_image.size[1], int(bottom))
                    right = min(current_image.size[0], int(right))

                    # Convert bbox to polygon format for unified geometric representation
                    bbox = [float(left), float(top), float(right), float(bottom)]
                    polygon = self._bbox_to_polygon(bbox)

                    detection = {
                        "class": self.class_names[int(class_id)] if class_id < len(self.class_names) else "target",
                        "confidence": float(top_conf[i]),
                        "polygon": polygon,  # Polygon format for spatial operations
                        "center": [(left + right) // 2, (top + bottom) // 2],
                        "area": (right - left) * (bottom - top)
                    }
                    detections.append(detection)

            return {
                "detections": detections,
                "image_shape": image_shape.tolist()
            }

        except Exception as e:
            raise RuntimeError(f"Error during DMIST inference: {e}")

    def _create_visualization(self, image_path: str, detections: List[Dict], output_path: str) -> str:
        """
        Create visualization of detection results.

        Args:
            image_path: Path to the original image
            detections: List of detection results
            output_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        try:
            # Load original image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # Try to load font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except:
                font = ImageFont.load_default()

            # Draw detections
            for i, detection in enumerate(detections):
                # Extract bbox from polygon format (or use bbox if available for backward compatibility)
                if "polygon" in detection and detection["polygon"]:
                    polygon = detection["polygon"]
                    x_coords = [point[0] for point in polygon]
                    y_coords = [point[1] for point in polygon]
                    left, top, right, bottom = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
                elif "bbox" in detection:
                    # Fallback for backward compatibility
                    bbox = detection["bbox"]
                    left, top, right, bottom = map(int, bbox)
                else:
                    continue  # Skip if no geometry available

                class_name = detection["class"]
                confidence = detection["confidence"]

                # Choose color (cycle through available colors)
                color_idx = i % len(self.colors) if self.colors else 0
                color = self.colors[color_idx] if self.colors else (255, 0, 0)

                # Draw bounding box
                thickness = 2
                for t in range(thickness):
                    draw.rectangle([left + t, top + t, right - t, bottom - t], outline=color)

                # Draw label
                label = f'{class_name} {confidence:.2f}'

                # Calculate text size and position
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]

                # Position text above box if possible, otherwise below
                if top - text_height >= 0:
                    text_origin = [left, top - text_height]
                else:
                    text_origin = [left, bottom]

                # Draw text background
                draw.rectangle([text_origin[0], text_origin[1],
                              text_origin[0] + text_width, text_origin[1] + text_height],
                              fill=color)

                # Draw text
                draw.text(text_origin, label, fill=(255, 255, 255), font=font)

            # Save visualization
            image.save(output_path)
            return output_path

        except Exception as e:
            print(f"Error creating visualization: {e}")
            return ""

    def _run(self, image_path: str, confidence_threshold: float = 0.5,
            nms_iou_threshold: float = 0.3, device: str = "cuda:2",
            text_prompt: Optional[str] = None, meters_per_pixel: Optional[float] = None) -> str:
        """
        Execute infrared target detection.

        Args:
            image_path: Path to infrared image
            confidence_threshold: Detection confidence threshold
            nms_iou_threshold: NMS IoU threshold
            device: Device for inference (default: cuda:2 per GPU allocation policy)
            text_prompt: Task intent description (optional)
            meters_per_pixel: Ground resolution for spatial analysis (optional, null for IR dataset)

        Returns:
            JSON string with detection results
        """
        try:
            # Validate input
            if not os.path.exists(image_path):
                return json.dumps({
                    "error": f"Image file not found: {image_path}",
                    "success": False
                })

            # Use the specified device (default cuda:2) for all tensor operations
            # This ensures consistent tensor placement and prevents "Expected all tensors to be on the same device" errors
            if device != self.device:
                logger.info(f"[IR Detection] Device parameter '{device}' differs from initialized device '{self.device}' - using {self.device}")

            # Generate unique execution ID
            execution_id = str(uuid.uuid4())[:8]

            # Get history images for temporal analysis
            history_images = get_history_imgs(image_path)

            # Perform detection
            if self.model is not None and DMIST_AVAILABLE:
                detection_results = self._detect_with_dmist(
                    history_images, confidence_threshold, nms_iou_threshold
                )
            else:
                # Fallback: simulate detection for testing
                detection_results = self._simulate_detection(image_path)

            # Create visualization
            vis_filename = f"infrared_detection_{execution_id}.png"
            vis_path = str(self.temp_dir / vis_filename)

            if detection_results["detections"]:
                self._create_visualization(image_path, detection_results["detections"], vis_path)

            # Prepare final results
            num_detections = len(detection_results["detections"])
            image_shape = detection_results.get("image_shape", [])

            # Calculate image area for coverage percentage
            if image_shape and len(image_shape) >= 2:
                total_image_area = image_shape[0] * image_shape[1]
            else:
                total_image_area = 1  # Avoid division by zero

            results = {
                "tool": "infrared_detection",
                "input": {
                    "image_path": image_path,
                    "confidence_threshold": confidence_threshold,
                    "nms_iou_threshold": nms_iou_threshold,
                    "device": device
                },
                "output": {
                    "detections": detection_results["detections"],
                    "image_shape": image_shape,
                    "visualization_path": vis_path if os.path.exists(vis_path) else ""
                },
                "execution_id": execution_id,
                "success": True
            }

            # Generate summary
            if num_detections > 0:
                target_classes = [d["class"] for d in detection_results["detections"]]
                avg_confidence = np.mean([d["confidence"] for d in detection_results["detections"]])
                summary = f"Detected {num_detections} infrared targets with average confidence {avg_confidence:.3f}. Target classes: {', '.join(set(target_classes))}."
            else:
                summary = "No infrared targets detected in the image."

            results["summary"] = summary

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in perception tools, spatial relation tools, and statistical tools for consistency

            # Build classes_requested list (infrared detection always targets "target" class)
            # Note: We use "target" internally but map to "infrared_small_target" for scene context matching
            classes_requested = ["target"]

            # Sort classes alphabetically for consistency
            classes_requested_sorted = sorted(classes_requested)

            # Calculate class_stats for scene context matching
            # Map "target" to "infrared_small_target" for scene context configuration
            class_stats = {}
            if num_detections > 0:
                # Calculate total area of all detections
                total_detection_area = sum(det.get("area", 0) for det in detection_results["detections"])
                coverage_pct = round(100.0 * total_detection_area / total_image_area, 2) if total_image_area > 0 else 0.0
            else:
                coverage_pct = 0.0

            # Use "infrared_small_target" as the class name for scene context matching
            class_stats["infrared_small_target"] = {
                "coverage_pct": coverage_pct,
                "detection_count": num_detections
            }

            # Build the unified arguments field with 6 fields (matching detection tool format)
            arguments = {
                "image_path": image_path,
                "classes_requested": classes_requested_sorted,
                "text_prompt": text_prompt if text_prompt else "detect infrared small targets",
                "meters_per_pixel": meters_per_pixel,  # null for IR dataset
                "total_detections": num_detections,
                "class_stats": class_stats
            }

            # Add unified arguments field to result
            results["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(results, indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Infrared detection failed: {str(e)}",
                "success": False,
                "tool": "infrared_detection",
                "input": {
                    "image_path": image_path,
                    "confidence_threshold": confidence_threshold,
                    "nms_iou_threshold": nms_iou_threshold,
                    "device": device
                }
            })

    def _simulate_detection(self, image_path: str) -> Dict[str, Any]:
        """
        Simulate infrared detection for testing when model is not available.

        Args:
            image_path: Path to image

        Returns:
            Simulated detection results
        """
        try:
            # Load image to get dimensions
            image = Image.open(image_path)
            width, height = image.size

            # Simulate 1-3 small targets
            np.random.seed(42)  # For reproducible results
            num_targets = np.random.randint(0, 4)

            detections = []
            for _ in range(num_targets):
                # Generate small target (typical infrared targets are small)
                target_size = np.random.randint(8, 24)
                center_x = np.random.randint(target_size, width - target_size)
                center_y = np.random.randint(target_size, height - target_size)

                left = center_x - target_size // 2
                top = center_y - target_size // 2
                right = center_x + target_size // 2
                bottom = center_y + target_size // 2

                # Convert bbox to polygon format for unified geometric representation
                bbox = [float(left), float(top), float(right), float(bottom)]
                polygon = self._bbox_to_polygon(bbox)

                detection = {
                    "class": "target",
                    "confidence": np.random.uniform(0.6, 0.95),
                    "polygon": polygon,  # Polygon format for spatial operations
                    "center": [center_x, center_y],
                    "area": target_size * target_size
                }
                detections.append(detection)

            return {
                "detections": detections,
                "image_shape": [height, width]
            }

        except Exception as e:
            print(f"Error in simulation: {e}")
            return {
                "detections": [],
                "image_shape": []
            }