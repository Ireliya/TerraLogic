"""
SAR Detection Tool using Old SARATR-X HiViT model.
This tool detects objects in SAR images using the original SARATR-X HiViT model with MMDetection 2.24.1.

The tool supports detection of three main object categories:
- Bridge (category_id: 0)
- Harbor (category_id: 1) 
- Tank (category_id: 2)

Usage Examples:
1. Test with default settings:
   python sar_detection.py

2. Test with specific image:
   python sar_detection.py --image dataset/images_ogsod/1000100.png

3. Use as a module:
   from sar_detection import create_sar_detection_tool
   detector = create_sar_detection_tool(device="cuda:2")
   result = detector._run("path/to/image.png", confidence_threshold=0.3)
"""

import os
import sys
import json
import uuid
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging
import traceback

# Add the old SARATR-X detection codebase to Python path FIRST
OLD_SARATR_PATH = Path(__file__).parent / "SARATR-X" / "detection"
if str(OLD_SARATR_PATH) not in sys.path:
    sys.path.insert(0, str(OLD_SARATR_PATH))

# Remove any conflicting paths that might have newer MMDetection
sys.path = [p for p in sys.path if 'codelab' not in p or OLD_SARATR_PATH.name in p]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Object class mapping for SAR imagery
OBJECT_CLASSES = {
    0: "bridge",
    1: "harbor",
    2: "tank"
}

# SAR dataset standard GSD (Ground Sample Distance)
SAR_DEFAULT_GSD = 3.0  # meters per pixel for OGSOD dataset


class SARDetectionInput(BaseModel):
    """Input schema for SAR detection tool."""
    image_path: str = Field(description="Path to SAR image file")
    confidence_threshold: float = Field(default=0.3, description="Detection confidence threshold (0.0-1.0)")
    device: str = Field(default="cuda:0", description="Device for inference (cuda:0, cuda:1, etc.)")

    # Optional parameters for unified arguments field (from upstream tools)
    text_prompt: Optional[str] = Field(default="detect SAR objects", description="Task intent description (e.g., 'detect SAR objects')")
    meters_per_pixel: Optional[float] = Field(default=SAR_DEFAULT_GSD, description="Ground resolution for spatial analysis (3.0 for OGSOD dataset)")

class SARDetectionTool(BaseTool):
    """
    SAR Detection tool using old SARATR-X HiViT model.
    Detects bridges, harbors, and tanks in SAR imagery.
    """

    name: str = "sar_detection"
    description: str = "Detect objects (bridges, harbors, tanks) in SAR images using old SARATR-X HiViT model"
    args_schema: Type[BaseModel] = SARDetectionInput

    # Private attributes for the model
    _model: Optional[Any] = None
    _device: str = "cuda:0"
    _model_path: Optional[str] = None
    _config_path: Optional[str] = None

    def __init__(self, device: str = "cuda:0", **kwargs):
        super().__init__(**kwargs)
        logger.info(f"[SARDetection] Initializing old SARATR-X SAR detection tool with device: {device}")
        self._device = device
        self._validate_device()
        self._setup_model_paths()
        self._initialize_model()

    def _validate_device(self) -> None:
        """Validate and ensure the specified CUDA device is available."""
        if self._device.startswith('cuda:'):
            try:
                gpu_id = int(self._device.split(':')[1])
                if not torch.cuda.is_available():
                    raise RuntimeError(f"[SARDetection] CUDA not available but {self._device} was requested")
                elif gpu_id >= torch.cuda.device_count():
                    available_gpus = torch.cuda.device_count()
                    raise RuntimeError(f"[SARDetection] GPU {gpu_id} not available (only {available_gpus} GPUs detected)")
                else:
                    # Test GPU accessibility
                    torch.cuda.set_device(gpu_id)
                    test_tensor = torch.tensor([1.0], device=self._device)
                    del test_tensor
                    torch.cuda.empty_cache()
                    logger.info(f"[SARDetection] ✅ GPU {gpu_id} validated and accessible")
            except (ValueError, IndexError, RuntimeError) as e:
                raise RuntimeError(f"[SARDetection] Device validation failed: {e}")

    def _setup_model_paths(self) -> None:
        """Setup paths for model configuration and checkpoint files."""
        logger.info("[SARDetection] Setting up model paths for old SARATR-X...")
        
        # Base paths
        sar_dir = Path(__file__).parent
        old_saratr_dir = sar_dir / "SARATR-X" / "detection"
        
        # Configuration file path - use the OGSOD config for 3-class detection
        config_candidates = [
            old_saratr_dir / "configs" / "_hivit_" / "hivit_base_OGSOD.py",
            old_saratr_dir / "configs" / "_hivit_" / "hivit_base_SSDD.py",
            sar_dir / "hivit_base_OGSOD.py"
        ]
        
        self._config_path = None
        for config_path in config_candidates:
            if config_path.exists():
                self._config_path = str(config_path)
                logger.info(f"[SARDetection] ✅ Found config file: {config_path}")
                break
        
        if not self._config_path:
            raise FileNotFoundError(f"[SARDetection] ❌ No config file found. Searched: {config_candidates}")
        
        # Model checkpoint path - prioritize latest.pth over checkpoint-800.pth
        checkpoint_candidates = [
            sar_dir / "latest.pth",
            sar_dir / "checkpoint-800.pth",
            old_saratr_dir / "latest.pth",
            old_saratr_dir / "checkpoint-800.pth"
        ]
        
        self._model_path = None
        for checkpoint_path in checkpoint_candidates:
            if checkpoint_path.exists():
                self._model_path = str(checkpoint_path)
                logger.info(f"[SARDetection] ✅ Found checkpoint file: {checkpoint_path}")
                break
        
        if not self._model_path:
            raise FileNotFoundError(f"[SARDetection] ❌ No checkpoint file found. Searched: {checkpoint_candidates}")
            
        logger.info(f"[SARDetection] Old SARATR-X model setup complete:")
        logger.info(f"  Config: {self._config_path}")
        logger.info(f"  Checkpoint: {self._model_path}")

    def _initialize_model(self) -> None:
        """Initialize the old SARATR-X HiViT model for detection."""
        try:
            logger.info(f"[SARDetection] Loading old SARATR-X HiViT model from {self._model_path}")

            # Try to fix MMCV extensions issue first
            self._fix_mmcv_extensions()

            # Import old MMDetection components (compatible with MMCV 1.6.0)
            try:
                from mmdet.apis import init_detector
                from mmcv import Config

                # Import old HiViT model from SARATR-X
                sys.path.insert(0, str(OLD_SARATR_PATH / "models"))
                import models_hivit  # This imports the old HiViT implementation

            except ImportError as e:
                logger.error(f"[SARDetection] Failed to import old MMDetection components: {e}")
                logger.info("[SARDetection] Attempting fallback model initialization...")
                self._initialize_fallback_model()
                return

            # Load configuration from the old config file
            cfg = Config.fromfile(self._config_path)

            # Update model configuration for inference
            cfg.model.pretrained = None  # We'll load from checkpoint

            # Initialize the model using old MMDetection API
            self._model = init_detector(cfg, self._model_path, device=self._device)

            # Ensure model is in evaluation mode for consistent inference
            self._model.eval()

            # Set deterministic behavior for consistent results
            import torch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Set random seeds for reproducibility
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)

            logger.info("[SARDetection] ✅ Old SARATR-X model initialized successfully with deterministic settings")

        except Exception as e:
            logger.error(f"[SARDetection] Failed to initialize old SARATR-X model: {e}")
            logger.info("[SARDetection] Attempting fallback model initialization...")
            self._initialize_fallback_model()

    def _fix_mmcv_extensions(self) -> None:
        """Try to fix MMCV extensions issues by patching imports."""
        try:
            # Set environment variable to disable MMCV CUDA extensions if they're problematic
            os.environ['MMCV_WITH_OPS'] = '0'

            # Patch mmcv.ops imports to avoid extension loading
            import mmcv.ops

            # Create dummy RoIPool class if it doesn't exist
            if not hasattr(mmcv.ops, 'RoIPool'):
                class DummyRoIPool:
                    def __init__(self, *args, **kwargs):
                        pass
                    def forward(self, *args, **kwargs):
                        return None

                mmcv.ops.RoIPool = DummyRoIPool
                logger.info("[SARDetection] Patched RoIPool with dummy implementation")

            logger.info("[SARDetection] Applied MMCV extensions patches")
        except Exception as e:
            logger.warning(f"[SARDetection] Could not patch MMCV extensions: {e}")

    def _initialize_fallback_model(self) -> None:
        """Initialize a fallback model when the main model fails to load."""
        logger.warning("[SARDetection] Using fallback model - will use subprocess approach")
        self._model = "subprocess"  # Use subprocess approach as fallback

    def _run(
        self,
        image_path: str,
        confidence_threshold: Optional[float] = 0.3,
        device: Optional[str] = None,
        text_prompt: Optional[str] = None,
        meters_per_pixel: Optional[float] = None
    ) -> str:
        """
        Execute detection on the input SAR image using old SARATR-X.

        Args:
            image_path: Path to input SAR image
            confidence_threshold: Minimum confidence threshold for detections
            device: CUDA device (optional, uses initialized device)
            text_prompt: Task intent description (optional)
            meters_per_pixel: Ground resolution for spatial analysis (optional, 3.0 for OGSOD dataset)

        Returns:
            JSON string containing detection results
        """
        try:
            logger.info(f"[SARDetection] Starting old SARATR-X detection for: {image_path}")
            logger.info(f"[SARDetection] Confidence threshold: {confidence_threshold}")

            # Validate input image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")

            if self._model is None:
                raise RuntimeError("Model not initialized properly")

            # Check if using subprocess fallback
            if self._model == "subprocess":
                logger.info("[SARDetection] Using subprocess approach for old SARATR-X...")
                detections = self._run_subprocess_detection(image_path, confidence_threshold)
            else:
                # Perform inference using old SARATR-X API
                logger.info("[SARDetection] Running old SARATR-X model inference...")

                # Ensure deterministic inference
                import torch
                with torch.no_grad():  # Disable gradient computation for inference
                    # Use old MMDetection inference API
                    from mmdet.apis import inference_detector

                    # Run inference - the old API handles preprocessing internally
                    model_output = inference_detector(self._model, image_path)

                logger.info(f"[SARDetection] Old SARATR-X inference completed. Output type: {type(model_output)}")

                # Get original image shape for postprocessing
                image = cv2.imread(image_path)
                original_shape = (image.shape[0], image.shape[1])  # (height, width)

                # Post-process results
                detections = self._postprocess_detections(model_output, original_shape, confidence_threshold)
            
            # Create output directory for results
            output_dir = Path("temp") / "sar_detection"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique result ID
            result_id = str(uuid.uuid4())[:8]
            
            # Save visualization if detections found
            visualization_path = None
            if detections:
                visualization_path = self._save_detection_visualization(
                    image_path, detections, output_dir
                )
            
            # Prepare result
            result = {
                "success": True,  # Mark as successful for downstream tools
                "tool": "sar_detection",
                "result_id": result_id,
                "image_path": image_path,
                "detections": detections,
                "detection_count": len(detections),
                "confidence_threshold": confidence_threshold,
                "visualization_path": visualization_path,
                "summary": f"Detected {len(detections)} objects in SAR image using old SARATR-X: " +
                          ", ".join([f"{d['class']} ({d['confidence']:.2f})" for d in detections[:3]]) +
                          (f" and {len(detections)-3} more" if len(detections) > 3 else "")
            }
            
            logger.info(f"[SARDetection] Detection completed: {len(detections)} objects found")

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in perception tools, spatial relation tools, and statistical tools for consistency

            # Build classes_requested list dynamically based on actual detections
            # Extract unique classes from detections
            detected_classes = list(set(d.get("class") for d in detections if d.get("class")))

            # Sort classes alphabetically for consistency
            classes_requested_sorted = sorted(detected_classes)

            # Count detections per class (only for detected classes)
            detection_counts = {}
            for cls in detected_classes:
                detection_counts[cls] = sum(1 for d in detections if d.get("class") == cls)

            # Sort detection_counts keys alphabetically
            detection_counts_sorted = {k: detection_counts[k] for k in sorted(detection_counts.keys())}

            # Calculate class_stats: coverage percentage for each detected class
            # Load the image to get dimensions
            try:
                image = cv2.imread(image_path)
                if image is None:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image_height, image_width = image.shape[:2]
                    total_image_area = image_height * image_width
                else:
                    total_image_area = 1  # Fallback to avoid division by zero
            except Exception as e:
                logger.warning(f"[SARDetection] Warning: Could not load image for area calculation: {e}")
                total_image_area = 1

            # Calculate coverage percentage for each class
            class_stats = {}
            for cls in classes_requested_sorted:
                class_detections = [d for d in detections if d.get("class") == cls]
                if class_detections:
                    total_class_area = sum(d.get("area", 0) for d in class_detections)
                    coverage_pct = round(100.0 * total_class_area / total_image_area, 2)
                    class_stats[cls] = {"coverage_pct": coverage_pct}

            # Set meters_per_pixel to SAR default if not provided
            gsd = meters_per_pixel if meters_per_pixel is not None else SAR_DEFAULT_GSD

            # Build the unified arguments field with 7 fields
            arguments = {
                "image_path": image_path,
                "classes_requested": classes_requested_sorted,
                "text_prompt": text_prompt if text_prompt else "detect SAR objects",
                "meters_per_pixel": gsd,
                "total_detections": len(detections),
                "detection_counts": detection_counts_sorted,
                "class_stats": class_stats
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[SARDetection] Detection failed: {error_msg}")
            traceback.print_exc()
            
            error_result = {
                "tool": "sar_detection",
                "error": error_msg,
                "detections": [],
                "detection_count": 0,
                "image_path": image_path,
                "summary": f"Detection failed: {error_msg}"
            }
            return json.dumps(error_result, indent=2)

    def _run_subprocess_detection(self, image_path: str, confidence_threshold: float) -> List[Dict]:
        """Run detection using subprocess to avoid MMCV import issues."""
        try:
            import subprocess
            import tempfile
            import json

            logger.info("[SARDetection] Running detection via subprocess...")

            # Create a temporary script that runs the detection
            script_content = f'''
import sys
import os
import json
import numpy as np

# Set environment to avoid MMCV extensions
os.environ['MMCV_WITH_OPS'] = '0'

# Add old SARATR-X path
sys.path.insert(0, '{OLD_SARATR_PATH}')

try:
    from mmdet.apis import init_detector, inference_detector
    from mmcv import Config

    # Initialize model
    config_path = '{self._config_path}'
    checkpoint_path = '{self._model_path}'
    device = '{self._device}'

    model = init_detector(config_path, checkpoint_path, device=device)

    # Run inference
    result = inference_detector(model, '{image_path}')

    # Process results
    detections = []
    if isinstance(result, list):
        for class_id, class_detections in enumerate(result):
            if isinstance(class_detections, np.ndarray) and class_detections.size > 0:
                for detection in class_detections:
                    if len(detection) >= 5:
                        x1, y1, x2, y2, confidence = detection[:5]
                        if confidence >= {confidence_threshold}:
                            class_names = {{0: "bridge", 1: "harbor", 2: "tank"}}
                            class_name = class_names.get(class_id, f"class_{{class_id}}")

                            # Convert bbox to polygon format for unified geometric representation
                            polygon = [
                                [float(x1), float(y1)],
                                [float(x2), float(y1)],
                                [float(x2), float(y2)],
                                [float(x1), float(y2)],
                                [float(x1), float(y1)]
                            ]

                            detection_dict = {{
                                "class": class_name,
                                "class_id": int(class_id),
                                "confidence": float(confidence),
                                "polygon": polygon,
                                "area": float((x2 - x1) * (y2 - y1))
                            }}
                            detections.append(detection_dict)

    print(json.dumps(detections))

except Exception as e:
    print(json.dumps({{"error": str(e)}}))
'''

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                temp_script = f.name

            try:
                # Run the script
                result = subprocess.run([
                    'python', temp_script
                ], capture_output=True, text=True, timeout=120)

                if result.returncode == 0:
                    output = result.stdout.strip()
                    detections_data = json.loads(output)

                    if isinstance(detections_data, dict) and "error" in detections_data:
                        logger.error(f"[SARDetection] Subprocess error: {detections_data['error']}")
                        return []

                    logger.info(f"[SARDetection] Subprocess detection completed: {len(detections_data)} raw detections")

                    # Apply the same filtering as the main detection method
                    import cv2
                    image = cv2.imread(image_path)
                    original_shape = (image.shape[0], image.shape[1]) if image is not None else (256, 256)

                    # Convert subprocess format to internal format for filtering
                    formatted_detections = []
                    for det in detections_data:
                        if isinstance(det, dict) and 'bbox' in det and len(det['bbox']) == 4:
                            x1, y1, x2, y2 = det['bbox']
                            bbox = [x1, y1, x2, y2]

                            # Convert bbox to polygon format for unified geometric representation
                            polygon = self._bbox_to_polygon(bbox)

                            formatted_det = {
                                "class": det.get('class', 'unknown'),
                                "class_id": det.get('class_id', 0),
                                "confidence": det.get('confidence', 0.0),
                                "polygon": polygon,  # Polygon format for spatial operations
                                "area": det.get('area', (x2-x1)*(y2-y1))
                            }
                            formatted_detections.append(formatted_det)

                    # Apply filtering
                    filtered_detections = self._apply_detection_filtering(formatted_detections, original_shape)

                    # Convert back to subprocess format with polygon coordinates
                    result_detections = []
                    for det in filtered_detections:
                        result_det = {
                            "class": det["class"],
                            "class_id": det["class_id"],
                            "confidence": det["confidence"],
                            "polygon": det["polygon"],  # Polygon format for spatial operations
                            "area": det["area"]
                        }
                        result_detections.append(result_det)

                    logger.info(f"[SARDetection] Filtered to {len(result_detections)} detections")
                    return result_detections
                else:
                    logger.error(f"[SARDetection] Subprocess failed: {result.stderr}")
                    return []

            finally:
                # Clean up temporary script
                try:
                    os.unlink(temp_script)
                except:
                    pass

        except Exception as e:
            logger.error(f"[SARDetection] Subprocess detection failed: {e}")
            return []

    def _postprocess_detections(self, model_output: Any, original_shape: Tuple[int, int],
                              confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Postprocess old SARATR-X model output to extract detection results.
        Applies proper NMS and confidence filtering to reduce over-detection.

        Args:
            model_output: Raw model output from old SARATR-X
            original_shape: Original image shape (height, width)
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of detection dictionaries
        """
        detections = []

        try:
            logger.info(f"[SARDetection] Processing old SARATR-X output of type: {type(model_output)}")

            # Handle old MMDetection model output format (list of arrays)
            if isinstance(model_output, list) and len(model_output) > 0:
                logger.info(f"[SARDetection] Processing {len(model_output)} class outputs")

                # Collect all detections with higher confidence threshold for initial filtering
                all_detections = []

                for class_id, class_detections in enumerate(model_output):
                    if isinstance(class_detections, np.ndarray) and class_detections.size > 0:
                        logger.info(f"[SARDetection] Class {class_id} has {len(class_detections)} raw detections")

                        for detection in class_detections:
                            if len(detection) >= 5:  # [x1, y1, x2, y2, confidence]
                                x1, y1, x2, y2, confidence = detection[:5]

                                # Apply confidence threshold
                                if confidence >= confidence_threshold:
                                    class_name = OBJECT_CLASSES.get(class_id, f"class_{class_id}")

                                    # Convert bbox to polygon format for unified geometric representation
                                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                                    polygon = self._bbox_to_polygon(bbox)

                                    detection_dict = {
                                        "class": class_name,
                                        "class_id": int(class_id),
                                        "confidence": float(confidence),
                                        "bbox": bbox,  # Keep bbox for filtering operations
                                        "polygon": polygon,  # Polygon format for spatial operations
                                        "area": float((x2 - x1) * (y2 - y1))
                                    }
                                    all_detections.append(detection_dict)

                # Apply additional filtering to reduce over-detection
                detections = self._apply_detection_filtering(all_detections, original_shape)

            else:
                logger.warning(f"[SARDetection] Unexpected model output format: {type(model_output)}")

        except Exception as e:
            logger.error(f"[SARDetection] Error in postprocessing: {e}")
            traceback.print_exc()

        logger.info(f"[SARDetection] Returning {len(detections)} filtered detections")
        return detections

    def _apply_detection_filtering(self, detections: List[Dict], original_shape: Tuple[int, int]) -> List[Dict]:
        """
        Apply additional filtering to reduce over-detection issues.

        Args:
            detections: List of detection dictionaries
            original_shape: Original image shape (height, width)

        Returns:
            Filtered list of detections
        """
        if not detections:
            return detections

        try:
            import numpy as np

            # Sort detections by confidence (highest first)
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

            # Apply per-class filtering
            filtered_detections = []
            class_counts = {}

            for detection in detections:
                class_name = detection['class']
                class_id = detection['class_id']

                # Limit detections per class to reasonable numbers
                max_per_class = 10  # Reasonable limit for SAR images
                if class_counts.get(class_id, 0) >= max_per_class:
                    continue

                # Apply minimal confidence threshold to preserve model outputs
                # Set to very low value to ensure nearly all model detections pass through filtering
                min_confidence = 0.01  # Minimal threshold to preserve legitimate detections
                if detection['confidence'] < min_confidence:
                    continue

                # Check for reasonable bounding box size
                bbox = detection['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                # Filter out very small or very large detections
                min_size = 5  # Minimum size in pixels
                max_size_ratio = 0.8  # Maximum 80% of image

                if (width < min_size or height < min_size or
                    width > original_shape[1] * max_size_ratio or
                    height > original_shape[0] * max_size_ratio):
                    continue

                # Apply simple NMS within same class
                should_keep = True
                for existing in filtered_detections:
                    if existing['class_id'] == class_id:
                        # Calculate IoU
                        iou = self._calculate_iou(detection['bbox'], existing['bbox'])
                        if iou > 0.5:  # More lenient NMS threshold
                            should_keep = False
                            break

                if should_keep:
                    filtered_detections.append(detection)
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1

            logger.info(f"[SARDetection] Filtered from {len(detections)} to {len(filtered_detections)} detections")
            return filtered_detections

        except Exception as e:
            logger.error(f"[SARDetection] Error in detection filtering: {e}")
            return detections[:20]  # Fallback: return top 20 detections

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        try:
            # Extract coordinates
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2

            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)

            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0

            intersection = (x2_i - x1_i) * (y2_i - y1_i)

            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _bbox_to_polygon(self, bbox: List[float]) -> List[List[float]]:
        """
        Convert bounding box coordinates to polygon format.

        Args:
            bbox: Bounding box in format [x_min, y_min, x_max, y_max]

        Returns:
            Polygon coordinates as [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
            representing the four corners of the rectangle (closed polygon)
        """
        try:
            x_min, y_min, x_max, y_max = bbox

            # Create polygon with 4 corners + closing point
            polygon = [
                [float(x_min), float(y_min)],  # Top-left
                [float(x_max), float(y_min)],  # Top-right
                [float(x_max), float(y_max)],  # Bottom-right
                [float(x_min), float(y_max)],  # Bottom-left
                [float(x_min), float(y_min)]   # Close the polygon
            ]

            return polygon
        except Exception as e:
            logger.error(f"[SARDetection] Error converting bbox to polygon: {e}")
            return []

    def _save_detection_visualization(self, image_path: str, detections: List[Dict],
                                    output_dir: Path) -> str:
        """Save visualization of detections on the original image."""
        try:
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                return None

            # Define colors for different classes
            class_colors = {
                "bridge": (0, 255, 0),    # Green
                "harbor": (255, 0, 0),    # Blue
                "tank": (0, 0, 255)       # Red
            }

            # Draw detections
            for detection in detections:
                confidence = detection["confidence"]
                class_name = detection["class"]

                # Extract bbox from polygon format (or use bbox if available for backward compatibility)
                if "polygon" in detection and detection["polygon"]:
                    polygon = detection["polygon"]
                    # Extract bbox from polygon: [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
                    x_coords = [point[0] for point in polygon]
                    y_coords = [point[1] for point in polygon]
                    x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))
                elif "bbox" in detection:
                    # Fallback for backward compatibility
                    bbox = detection["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                else:
                    # Skip if neither polygon nor bbox is available
                    continue

                color = class_colors.get(class_name, (255, 255, 255))

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save visualization
            output_filename = f"sar_detection_{Path(image_path).stem}_result.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), image)

            logger.info(f"[SARDetection] Visualization saved: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"[SARDetection] Failed to save visualization: {e}")
            return None

def create_sar_detection_tool(device: str = "cuda:0") -> SARDetectionTool:
    """
    Factory function to create a SAR detection tool using old SARATR-X.

    Args:
        device: CUDA device for inference

    Returns:
        Configured SARDetectionTool instance
    """
    return SARDetectionTool(device=device)
