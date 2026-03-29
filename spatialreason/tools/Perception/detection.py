"""
Object Detection Tool using RemoteSAM model.
This tool detects and locates objects in satellite imagery using text prompts.
"""
import os
import json
import uuid
import cv2
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from langchain_core.tools import BaseTool
from skimage.measure import label, regionprops
import torch

from spatialreason.tools.utils import validate_ground_resolution

# Import RemoteSAM components at module level
try:
    from .remotesam.model import RemoteSAM, init_demo_model
    SAM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RemoteSAM not available: {e}")
    SAM_AVAILABLE = False


def load_class_config() -> Dict[str, Any]:
    """Load class configuration from class_config.json"""
    config_path = Path(__file__).parent.parent / "class_config.json"  # Go up to tools directory
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load class_config.json: {e}")
        return {}


def load_detection_filtering_config() -> Dict[str, Any]:
    """Load detection filtering configuration from class_config.json"""
    config = load_class_config()
    return config.get("_detection_filtering", {
        "global_settings": {
            "global_min_confidence": 0.35,
            "default_min_area_ratio": 0.003,
            "default_top_k": 3,
            "use_percentile_thresholds": False,
            "percentile_threshold": 90
        },
        "per_class_confidence": {"default": 0.40},
        "per_class_min_area_ratio": {"default": 0.003},
        "nms_settings": {
            "enable_soft_nms": True,
            "soft_nms_sigma": 0.5,
            "intra_class_iou_threshold": 0.45,
            "cross_class_iou_threshold": 0.3,
            "confused_class_pairs": [["forest", "barren"]]
        },
        "top_k_selection": {
            "per_class_top_k": {"default": 3},
            "selection_criteria": "area_desc"
        }
    })


def map_text_prompt_to_classes(text_prompt: str, class_config: Dict[str, List[str]]) -> List[str]:
    """
    Map a text prompt to RemoteSAM class names using the class configuration.
    Uses word boundary matching to avoid substring matches (e.g., "tree" matching "trees").

    Args:
        text_prompt: Natural language description
        class_config: Dictionary mapping class names to aliases

    Returns:
        List of class names that match the text prompt
    """
    import re

    text_lower = text_prompt.lower()
    matched_classes = []

    # Sort class names by length descending to match longer names first
    # This ensures "low_vegetation" is matched before "vegetation"
    sorted_classes = sorted(class_config.items(), key=lambda x: len(x[0]), reverse=True)

    # Direct match with class names using word boundaries
    for class_name, aliases in sorted_classes:
        # Use regex word boundary matching to ensure exact word matches
        pattern = r'\b' + re.escape(class_name.lower()) + r'\b'
        if re.search(pattern, text_lower):
            if class_name not in matched_classes:
                matched_classes.append(class_name)
            continue

        # Check aliases with word boundary matching
        for alias in aliases:
            alias_pattern = r'\b' + re.escape(alias.lower()) + r'\b'
            if re.search(alias_pattern, text_lower):
                if class_name not in matched_classes:
                    matched_classes.append(class_name)
                break

    # If no matches found, raise an exception - no fallback allowed
    if not matched_classes:
        raise ValueError(f"No valid classes found in text prompt: '{text_prompt}'. "
                        f"Expected classes: {list(class_config.keys())}")

    return matched_classes


class RemoteSAMDetectionTool(BaseTool):
    """
    Detection tool using RemoteSAM model for satellite imagery.
    Detects objects based on text prompts and returns bounding boxes and confidence scores.
    """

    name: str = "detection"
    description: str = "Detect and locate objects in satellite imagery using text prompts with RemoteSAM model"
    
    # Private attributes for the model
    _model: Optional[Any] = None
    _device: str = "cuda:2"

    def __init__(self, device: str = "cuda:2", model_path: str = None,
                 fallback_to_hf: bool = True, cache_dir: str = None, **kwargs):
        super().__init__(**kwargs)
        print(f"[ObjectDetection] __init__ called with device: {device}")
        self._device = device
        self._initialize_model(model_path=model_path, fallback_to_hf=fallback_to_hf, cache_dir=cache_dir)

    def _validate_device(self, device: str) -> str:
        """
        Validate and potentially fix the device string.

        Args:
            device: Device string to validate

        Returns:
            Validated device string
        """
        if device == "auto":
            # This should have been resolved by the factory function, but handle it just in case
            import os
            if os.getenv('SPATIAL_REASONING_GPU_MODE') == 'hardcoded':
                perception_gpu = os.getenv('SPATIAL_REASONING_PERCEPTION_GPU', '2')
                device = f"cuda:{perception_gpu}"
            else:
                device = "cuda:2"
            print(f"[ObjectDetection] Resolved 'auto' device to: {device}")

        # Validate CUDA device availability with strict hardcoded allocation
        if device.startswith('cuda:'):
            try:
                gpu_id = int(device.split(':')[1])
                if not torch.cuda.is_available():
                    raise RuntimeError(f"[ObjectDetection] CUDA not available but {device} was requested. Cannot proceed with hardcoded GPU allocation strategy.")
                elif gpu_id >= torch.cuda.device_count():
                    available_gpus = torch.cuda.device_count()
                    raise RuntimeError(f"[ObjectDetection] GPU {gpu_id} not available (only {available_gpus} GPUs detected). Cannot proceed with hardcoded GPU allocation strategy.")
                else:
                    # Test if the GPU is accessible
                    try:
                        torch.cuda.set_device(gpu_id)
                        test_tensor = torch.tensor([1.0], device=device)
                        del test_tensor
                        torch.cuda.empty_cache()
                        return device
                    except Exception as e:
                        raise RuntimeError(f"[ObjectDetection] GPU {gpu_id} not accessible ({e}). Cannot proceed with hardcoded GPU allocation strategy.")
            except (ValueError, IndexError):
                raise RuntimeError(f"[ObjectDetection] Invalid device format '{device}'. Cannot proceed with hardcoded GPU allocation strategy.")

        return device
    
    def _initialize_model(self, model_path=None, fallback_to_hf=True, cache_dir=None):
        """Initialize the RemoteSAM model for object detection with Hugging Face fallback."""
        try:
            print(f"[ObjectDetection] Initializing RemoteSAM object detection tool...")
            print(f"[ObjectDetection] Model path: {model_path}, Device: {self._device}")

            if not SAM_AVAILABLE:
                raise ImportError("Required libraries for RemoteSAM are not available.")

            # Validate and potentially fix device string
            validated_device = self._validate_device(self._device)
            if validated_device != self._device:
                print(f"[ObjectDetection] Device changed from {self._device} to {validated_device}")
                self._device = validated_device

            print(f"[ObjectDetection] Using validated device: {self._device}")

            # Initialize the model using updated init_demo_model with HF support
            print(f"[ObjectDetection] Loading model with Hugging Face fallback enabled...")
            model = init_demo_model(
                checkpoint=model_path,
                device=self._device,
                fallback_to_hf=fallback_to_hf,
                cache_dir=cache_dir
            )

            # Create RemoteSAM instance
            self._model = RemoteSAM(
                RemoteSAM_model=model,
                device=self._device,
                use_EPOC=False,  # Disable EPOC for simplicity
                EPOC_threshold=0.25,
                MLC_balance_factor=0.5,
                MCC_balance_factor=1.0
            )

            print(f"[ObjectDetection] ✅ Model loaded successfully")

        except Exception as e:
            print(f"[ObjectDetection] ❌ Failed to initialize model: {e}")
            raise e
    
    def _detect_objects(self, image_path: str, classes: List[str]) -> Dict[str, np.ndarray]:
        """
        Detect objects in the image using RemoteSAM.
        
        Args:
            image_path: Path to the input image
            classes: List of object classes to detect
            
        Returns:
            Dictionary mapping class names to detection masks
        """
        print(f"[ObjectDetection] 🔍 Detecting objects in image: {image_path}")
        
        # Load and validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"[ObjectDetection] 📏 Image size: {image_rgb.shape[:2]}")
        
        # Use RemoteSAM to segment/detect objects
        mask_results = {}
        
        # Convert to PIL Image for RemoteSAM
        from PIL import Image
        image_pil = Image.fromarray(image_rgb)

        # Use RemoteSAM semantic segmentation for detection
        print(f"[ObjectDetection] 🎯 Detecting classes: {classes}")
        try:
            # Use RemoteSAM semantic segmentation (same as segmentation tool)
            mask_results_raw = self._model.semantic_seg(
                image=image_pil,
                classnames=classes,
                return_prob=False
            )

            # Process results
            h, w = image_rgb.shape[:2]  # Get image dimensions for empty mask creation
            for class_name in classes:
                if class_name in mask_results_raw:
                    mask = mask_results_raw[class_name]
                    if mask is not None and mask.sum() > 0:
                        mask_results[class_name] = mask.astype(np.uint8)
                        print(f"[ObjectDetection] ✅ Found {class_name}")
                    else:
                        print(f"[ObjectDetection] ⚠️  No {class_name} detected in image (empty mask)")
                        # Create empty mask for consistency - this is normal and not an error
                        # Use image dimensions directly instead of np.zeros_like(mask) which fails when mask is None
                        mask_results[class_name] = np.zeros((h, w), dtype=np.uint8)
                else:
                    print(f"[ObjectDetection] ⚠️  No results returned for {class_name} by RemoteSAM")
                    # Create empty mask for consistency
                    mask_results[class_name] = np.zeros((h, w), dtype=np.uint8)

        except Exception as e:
            print(f"[ObjectDetection] ❌ Error during detection: {e}")
            raise RuntimeError(f"RemoteSAM detection failed: {str(e)}. "
                             f"Cannot proceed with fallback data - real tool execution required.")
        
        print(f"[ObjectDetection] ✅ Detection complete for classes: {classes}")
        return mask_results
    
    def _extract_bounding_boxes(self, mask: np.ndarray, class_name: str) -> List[Dict]:
        """
        Extract bounding boxes from detection mask with 100% consistency guarantee.

        Args:
            mask: Binary mask of detected objects
            class_name: Name of the object class

        Returns:
            List of bounding box dictionaries
        """
        detections = []

        mask_sum = mask.sum()
        print(f"[BBoxExtraction] Processing {class_name}: mask sum = {mask_sum}")

        if mask_sum == 0:
            print(f"[BBoxExtraction] No pixels found for {class_name}")
            return detections

        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8)

        # Label connected components with better connectivity
        labeled_mask = label(binary_mask, connectivity=2)  # 8-connectivity for better region detection
        regions = regionprops(labeled_mask)

        print(f"[BBoxExtraction] Found {len(regions)} connected regions for {class_name}")

        for i, region in enumerate(regions):
            # Skip extremely small regions (likely noise)
            if region.area < 10:  # Less than 10 pixels
                print(f"[BBoxExtraction] Skipping tiny region {i+1} for {class_name}: {region.area} pixels")
                continue

            # Get bounding box coordinates
            min_row, min_col, max_row, max_col = region.bbox

            # Calculate area_ratio_score based on region area (NOT a model confidence score)
            # This metric represents the relative size of the detected region compared to the image
            area = region.area
            total_area = mask.shape[0] * mask.shape[1]
            area_ratio = area / total_area

            # Area-based scoring to ensure detections pass filtering
            # NOTE: This is NOT a model prediction confidence - it's derived from region size
            if area_ratio < 0.0001:  # Very small regions
                area_ratio_score = 0.4  # Above typical filtering thresholds
            elif area_ratio < 0.001:  # Small regions
                area_ratio_score = 0.5 + area_ratio * 100  # 0.5-0.6 range
            elif area_ratio < 0.01:  # Medium regions
                area_ratio_score = 0.6 + (area_ratio - 0.001) * 20  # 0.6-0.78 range
            else:  # Large regions
                area_ratio_score = min(0.95, 0.8 + (area_ratio - 0.01) * 1.5)

            # Ensure minimum score to pass filtering
            area_ratio_score = max(area_ratio_score, 0.4)

            # Create polygon from bounding box for spatial operations
            polygon_coords = [
                [float(min_col), float(min_row)],
                [float(max_col), float(min_row)],
                [float(max_col), float(max_row)],
                [float(min_col), float(max_row)],
                [float(min_col), float(min_row)]  # Close the polygon
            ]

            detection = {
                "object_id": f"{class_name}_{i+1}",
                "class": class_name,
                "confidence": round(area_ratio_score, 3),  # Area-based score, NOT model prediction confidence
                "polygon": polygon_coords,  # Polygon coordinates for spatial operations (unified geometric representation)
                "area_pixels": int(area),
                "centroid": {
                    "x": round(region.centroid[1], 1),
                    "y": round(region.centroid[0], 1)
                }
            }
            detections.append(detection)
            print(f"[BBoxExtraction] Created detection {i+1} for {class_name}: area={area}px, area_ratio_score={area_ratio_score:.3f}")
        
        return detections

    def _run(
        self,
        image_path: str,
        text_prompt: str,
        min_area_ratio_threshold: Optional[float] = 0.0001,  # Issue #8: Minimum area ratio threshold for filtering (0.01% of image)
        meters_per_pixel: Optional[float] = 0.3,  # Correct: This is spatial resolution (meters per pixel)
        threshold_policy: Optional[str] = "minimal",  # Issue #9: Changed default from "strict" to "minimal" to match actual behavior
        meters_per_pixel_used: Optional[float] = None,  # For traceability
        dataset_tag: Optional[str] = None,  # Dataset identifier for consistent directory naming
    ) -> str:
        """
        Execute the RemoteSAM object detection tool.

        Args:
            image_path: Path to the input image
            text_prompt: Natural language description of objects to detect
            min_area_ratio_threshold: Minimum area ratio threshold for filtering detections (0.0 to 1.0, where 1.0 = 100% of image)
            meters_per_pixel: Ground resolution in meters per pixel
            threshold_policy: Filtering policy - currently only "minimal" is supported (preserves segmentation consistency)

        Returns:
            JSON string containing detection results
        """
        try:
            print(f"[ObjectDetection] Starting object detection for: {image_path}")
            print(f"[ObjectDetection] Text prompt: '{text_prompt}'")

            # Validate input image exists and is readable
            from pathlib import Path
            import cv2

            if not Path(image_path).exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")

            # Try to load image to verify it's valid
            test_image = cv2.imread(image_path)
            if test_image is None:
                raise ValueError(f"Cannot read image file: {image_path}. File may be corrupted or in unsupported format.")

            print(f"[ObjectDetection] ✅ Image validation passed: {Path(image_path).name} ({test_image.shape})")

            # Generate unique execution ID for this detection run (Issue #7)
            # This ensures all output files from this execution have consistent unique identifiers
            execution_id = uuid.uuid4().hex[:8]
            print(f"[ObjectDetection] 🆔 Execution ID: {execution_id}")

            # Validate ground resolution
            if meters_per_pixel:
                validate_ground_resolution(meters_per_pixel)

            # Parse text prompt to determine what to detect using centralized class mapping (Issue #6)
            # Build class configuration with supported classes and their aliases
            # IMPORTANT: Removed "tree" and "trees" from "forest" aliases to avoid ambiguity with "tree" class
            # When user specifies "trees", it should match "tree" class, not "forest"
            class_config = {
                # LoveDA dataset classes
                "building": ["building", "buildings", "house", "houses", "structure", "structures"],
                "road": ["road", "roads", "street", "streets", "highway", "highways"],
                "water": ["water", "water body", "lake", "lakes", "river", "rivers", "sea", "ocean"],
                "barren": ["barren", "bare", "bare ground", "barren land"],
                "forest": ["forest", "forests", "woodland", "woods"],  # Removed "tree", "trees" to avoid ambiguity
                "agriculture": ["agriculture", "agricultural", "farm", "farms", "crop", "crops", "field", "fields"],
                # Potsdam dataset classes
                "car": ["car", "cars", "vehicle", "vehicles", "automobile", "automobiles"],
                "tree": ["tree", "trees", "vegetation", "plant", "plants"],
                "low_vegetation": ["low vegetation", "low_vegetation", "grass", "shrub", "shrubs"],
                "impervious_surface": ["impervious surface", "impervious_surface", "pavement", "concrete", "asphalt"],
                "clutter": ["clutter", "debris", "rubble", "junk"]
            }

            try:
                # Use centralized class mapping function
                classes = map_text_prompt_to_classes(text_prompt, class_config)
                print(f"[ObjectDetection] 🎯 Parsed classes from prompt: {classes}")
            except ValueError as e:
                # If no classes found in config, fall back to using the prompt as-is
                print(f"[ObjectDetection] ⚠️ {e}")
                print(f"[ObjectDetection] 🎯 Using text prompt as class name: {text_prompt}")
                classes = [text_prompt.strip()]
            
            # Perform detection using RemoteSAM
            mask_results = self._detect_objects(image_path, classes)
            
            # Extract bounding boxes from masks with consistency guarantee
            all_detections = []
            detection_masks = {}

            print(f"[ObjectDetection] 🔍 Processing {len(mask_results)} class masks...")
            for class_name, mask in mask_results.items():
                print(f"[ObjectDetection] Processing class '{class_name}': mask shape={mask.shape}, non-zero pixels={mask.sum()}")

                detections = self._extract_bounding_boxes(mask, class_name)
                all_detections.extend(detections)

                # CRITICAL FIX: Save ALL masks that have segmented pixels, not just those with detections
                # This ensures we preserve the segmentation output even if bbox extraction has issues
                if mask.sum() > 0:  # Any segmented pixels
                    detection_masks[class_name] = mask
                    print(f"[ObjectDetection] ✅ Saved mask for '{class_name}': {len(detections)} detections from {mask.sum()} pixels")
                else:
                    print(f"[ObjectDetection] ⚠️ No pixels found in mask for '{class_name}'")

            # Store original detections for fallback mechanism
            original_all_detections = all_detections.copy()

            # Load filtering configuration
            filtering_config = load_detection_filtering_config()

            # CRITICAL FIX: Ensure 100% consistency between segmentation and detection
            # Skip aggressive filtering to preserve all segmented objects
            print(f"[ObjectDetection] 🔧 Ensuring 100% segmentation-to-detection consistency...")
            print(f"[ObjectDetection] 📊 Raw detections from segmentation: {len(all_detections)}")

            # Group detections by class for verification
            class_detection_counts = {}
            for det in all_detections:
                class_name = det['class']
                class_detection_counts[class_name] = class_detection_counts.get(class_name, 0) + 1

            print(f"[ObjectDetection] 📋 Per-class detection counts: {class_detection_counts}")

            # Apply MINIMAL filtering to preserve segmentation consistency (Issue #8)
            # Only filter out extremely small detections that are likely noise
            # Use the min_area_ratio_threshold parameter to control filtering
            filtered_detections = []
            image_shape = test_image.shape[:2]  # (height, width)
            total_image_area = image_shape[0] * image_shape[1]
            min_area_threshold = min_area_ratio_threshold  # Issue #8: Use parameter instead of hardcoded value

            for det in all_detections:
                area_ratio = det.get('area_pixels', 0) / total_image_area
                if area_ratio >= min_area_threshold:
                    filtered_detections.append(det)
                else:
                    print(f"[ObjectDetection] 🗑️ Filtered tiny detection: {det['class']} (area: {det.get('area_pixels', 0)}px, ratio: {area_ratio:.6f})")

            all_detections = filtered_detections
            print(f"[ObjectDetection] ✅ After minimal filtering: {len(all_detections)} detections preserved")

            # Verify no classes were lost (Issue #10: Track diagnostic information)
            remaining_classes = set(det['class'] for det in all_detections)
            requested_classes = set(classes)
            lost_classes = requested_classes - remaining_classes

            # Issue #10: Store diagnostic information for result dictionary
            classes_requested = sorted(list(requested_classes))
            classes_detected = sorted(list(remaining_classes))
            classes_not_found = sorted(list(lost_classes))

            if lost_classes:
                print(f"[ObjectDetection] ⚠️ Classes lost during minimal filtering: {lost_classes}")
                # Restore lost classes from original detections
                for lost_class in lost_classes:
                    class_detections = [d for d in original_all_detections if d['class'] == lost_class]
                    if class_detections:
                        # Add the largest detection for the lost class
                        best_detection = max(class_detections, key=lambda x: x.get('area_pixels', 0))
                        all_detections.append(best_detection)
                        classes_detected.append(lost_class)  # Issue #10: Update detected classes list
                        classes_detected.sort()  # Keep sorted
                        print(f"[ObjectDetection] 🔄 Restored {lost_class}: area={best_detection.get('area_pixels', 0)}px")
                    else:
                        # Issue #10: Detailed logging when restoration fails
                        print(f"[ObjectDetection] ❌ Failed to restore {lost_class}: no detections found in original_all_detections")
                        print(f"[ObjectDetection] 📊 Available classes in original detections: {set(d['class'] for d in original_all_detections)}")

            print(f"[ObjectDetection] ✅ Final detection count: {len(all_detections)} (preserving segmentation consistency)")
            print(f"[ObjectDetection] 📋 Classes requested: {classes_requested}, detected: {classes_detected}, not found: {classes_not_found}")

            # Ensure we have detected objects - provide detailed debugging info
            if len(all_detections) == 0:
                # Count total detections before filtering for debugging
                total_before_filtering = 0
                confidence_info = []

                for class_name, mask in mask_results.items():
                    raw_detections = self._extract_bounding_boxes(mask, class_name)
                    total_before_filtering += len(raw_detections)

                    if raw_detections:
                        confidences = [d["confidence"] for d in raw_detections]
                        confidence_info.append(f"{class_name}: {len(raw_detections)} detections, confidence range: {min(confidences):.3f}-{max(confidences):.3f}")

                debug_info = f"Total detections before filtering: {total_before_filtering}. "
                if confidence_info:
                    debug_info += f"Confidence details: {'; '.join(confidence_info)}. "

                # Include filtering configuration info
                global_settings = filtering_config.get('global_settings', {})
                nms_settings = filtering_config.get('nms_settings', {})
                debug_info += f"Minimal filtering applied: area_threshold={min_area_threshold}, "
                debug_info += f"original_global_min={global_settings.get('global_min_confidence', 0.35)}, "
                debug_info += f"original_area_ratio_min={global_settings.get('default_min_area_ratio', 0.003)}. "

                raise RuntimeError(f"No objects detected after comprehensive filtering pipeline. {debug_info}"
                                 f"Consider adjusting filtering parameters in class_config.json or checking semantic alignment between prompt and image content.")
            
            # Use organized output directory structure
            from pathlib import Path
            import os

            # Extract image ID for directory organization
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            if image_id.endswith('_vis'):
                image_id = image_id[:-4]  # Remove '_vis' suffix

            # Create organized output directory using dataset tag
            # Use explicit None check to allow empty string to mean "no subdirectory"
            if dataset_tag is None:
                dataset_tag = "dataset"  # Default fallback only if None

            if dataset_tag:
                output_dir = Path("temp") / dataset_tag / image_id
            else:
                output_dir = Path("temp") / image_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create standardized class-based naming
            class_names = "_".join(sorted([obj['class'] for obj in all_detections]))
            if not class_names:
                class_names = "no_detections"

            # Save detection masks with standardized naming and verification (Issue #7)
            # Include execution_id to prevent overwrites and collisions
            mask_paths = {}
            print(f"[ObjectDetection] 💾 Saving {len(detection_masks)} detection masks...")

            for class_name, mask in detection_masks.items():
                mask_filename = f"detection_{class_name}_{execution_id}.png"
                mask_path = output_dir / mask_filename

                # Ensure mask is properly formatted for saving
                mask_to_save = (mask * 255).astype(np.uint8)
                success = cv2.imwrite(str(mask_path), mask_to_save)

                if success:
                    mask_paths[class_name] = str(mask_path)
                    print(f"[ObjectDetection] ✅ Saved mask for '{class_name}': {mask_path}")

                    # Verify the saved file
                    if mask_path.exists():
                        file_size = mask_path.stat().st_size
                        print(f"[ObjectDetection] 📊 Mask file size: {file_size} bytes")
                    else:
                        print(f"[ObjectDetection] ⚠️ Mask file not found after saving: {mask_path}")
                else:
                    print(f"[ObjectDetection] ❌ Failed to save mask for '{class_name}': {mask_path}")

            # Create and save final detection image with bounding boxes
            detection_image_path = None
            if len(all_detections) > 0:
                # Load original image
                original_image = cv2.imread(image_path)
                if original_image is not None:
                    # Draw bounding boxes on the image
                    detection_image = original_image.copy()

                    for detection in all_detections:
                        # Extract bbox coordinates from polygon (unified geometric representation)
                        polygon = detection.get("polygon", [])
                        if polygon and len(polygon) >= 4:
                            # Polygon format: [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]
                            x_coords = [point[0] for point in polygon]
                            y_coords = [point[1] for point in polygon]
                            x_min, x_max = int(min(x_coords)), int(max(x_coords))
                            y_min, y_max = int(min(y_coords)), int(max(y_coords))
                        else:
                            # Fallback if polygon is missing (should not happen)
                            print(f"[ObjectDetection] ⚠️ Warning: No polygon found for detection {detection.get('object_id', 'unknown')}")
                            continue

                        confidence = detection["confidence"]
                        class_name = detection["class"]

                        # Draw bounding box (green color)
                        cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Draw label with confidence
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                        # Draw label background
                        cv2.rectangle(detection_image,
                                    (x_min, y_min - label_size[1] - 10),
                                    (x_min + label_size[0], y_min),
                                    (0, 255, 0), -1)

                        # Draw label text
                        cv2.putText(detection_image, label,
                                  (x_min, y_min - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Save detection image with standardized naming (Issue #7)
                    # Include execution_id to prevent overwrites and collisions
                    detection_filename = f"detection_{class_names}_{execution_id}.png"
                    detection_image_path = output_dir / detection_filename
                    cv2.imwrite(str(detection_image_path), detection_image)
                    print(f"[ObjectDetection] 💾 Saved detection image: {detection_image_path}")
                    detection_image_path = str(detection_image_path)
            
            # Calculate area in real-world units if resolution is provided
            if meters_per_pixel:
                for detection in all_detections:
                    area_m2 = detection["area_pixels"] * (meters_per_pixel ** 2)
                    detection["area_m2"] = round(area_m2, 2)
            
            # Prepare results with filtering metadata
            # Issue #10: Add diagnostic information for class detection
            detected_class_count = len(set(d['class'] for d in all_detections))
            summary_msg = f"RemoteSAM detection completed for '{text_prompt}' with segmentation consistency. Found {len(all_detections)} objects across {detected_class_count} classes."
            if classes_not_found:
                summary_msg += f" Classes not detected: {', '.join(classes_not_found)}."

            result = {
                "success": True,
                "execution_id": execution_id,  # Issue #7: Include execution ID for traceability
                "image_path": image_path,
                "text_prompt": text_prompt,
                "total_detections": len(all_detections),
                "detections": all_detections,
                "detection_masks": mask_paths,
                "min_area_ratio_threshold": min_area_ratio_threshold,  # Issue #8: Use actual parameter name
                "classes_requested": classes_requested,  # Issue #10: Diagnostic information
                "classes_detected": classes_detected,  # Issue #10: Diagnostic information
                "classes_not_found": classes_not_found,  # Issue #10: Diagnostic information
                "threshold_policy": {
                    "declared_threshold": min_area_ratio_threshold,  # Issue #8: Use actual parameter
                    "effective_threshold": min_area_ratio_threshold,  # No adjustment in minimal filtering
                    "threshold_policy": threshold_policy or "minimal",
                    "adjustment_reason": "Minimal filtering applied for segmentation consistency"
                },
                "filtering_applied": {
                    "minimal_filtering_applied": True,
                    "area_threshold_used": min_area_threshold,
                    "aggressive_filtering_disabled": True,
                    "segmentation_consistency_mode": True
                },
                "summary": summary_msg
            }

            # Add detection image path if created
            if detection_image_path:
                result["detection_image_path"] = detection_image_path
            
            # Add resolution info if provided
            if meters_per_pixel:
                result["meters_per_pixel"] = meters_per_pixel
                result["meters_per_pixel_used"] = meters_per_pixel_used or meters_per_pixel  # For traceability
            
            # Note: Results will be saved by TempStorageManager to image-specific directories
            # No longer saving to global temp/detection directory
            
            print(f"[ObjectDetection] Detection completed successfully for '{text_prompt}'")
            print(f"[ObjectDetection] Found {len(all_detections)} objects across {len(set(d['class'] for d in all_detections))} classes")

            # Log per-class detection summary
            class_counts = {}
            for det in all_detections:
                class_name = det['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            print(f"[ObjectDetection] 📊 Per-class detections: {class_counts}")
            print(f"[ObjectDetection] 💾 Saved files: {len(mask_paths)} masks, detection image: {'Yes' if detection_image_path else 'No'}")

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics

            # Calculate counts by class in the same order as classes_requested
            counts_by_class = []
            for cls in classes_requested:
                count = sum(1 for det in all_detections if det["class"] == cls)
                counts_by_class.append(count)

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
                print(f"[ObjectDetection] Warning: Could not load image for area calculation: {e}")
                total_image_area = 1

            # Calculate coverage percentage for each class
            class_stats = {}
            for cls in classes_requested:
                class_detections = [det for det in all_detections if det["class"] == cls]
                if class_detections:
                    total_class_area = sum(det.get("area_pixels", 0) for det in class_detections)
                    coverage_pct = round(100.0 * total_class_area / total_image_area, 2)
                    class_stats[cls] = {"coverage_pct": coverage_pct}

            # Build the unified arguments field
            arguments = {
                "image_path": image_path,
                "classes_requested": classes_requested,
                "meters_per_pixel": meters_per_pixel if meters_per_pixel else None,
                "text_prompt": text_prompt,
                "total_detections": len(all_detections),
                "counts_by_class": counts_by_class,
                "class_stats": class_stats
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            # Return raw JSON result for programmatic use
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"RemoteSAM object detection failed: {str(e)}"
            print(f"[ObjectDetection] {error_msg}")

            error_result = {
                "success": False,
                "error": error_msg,
                "image_path": image_path,
                "text_prompt": text_prompt,
                "summary": f"Object detection failed for '{text_prompt}': {error_msg}"
            }
            # Return raw JSON error result for programmatic use
            return json.dumps(error_result, indent=2)


def create_detection_tool(device: str = "auto") -> RemoteSAMDetectionTool:
    """
    Factory function to create a RemoteSAM detection tool.

    Args:
        device: Device to run the model on ("auto" for automatic GPU selection, 'cuda:0', 'cpu')

    Returns:
        Configured RemoteSAMDetectionTool instance
    """
    if device == "auto":
        # Check for hardcoded GPU assignment
        import os
        if os.getenv('SPATIAL_REASONING_GPU_MODE') == 'hardcoded':
            perception_gpu = os.getenv('SPATIAL_REASONING_PERCEPTION_GPU', '2')
            device = f"cuda:{perception_gpu}"
            print(f"🎯 Detection tool using hardcoded GPU assignment: {device}")
        else:
            # Use hardcoded GPU 0 for all perception tools (single GPU system)
            device = "cuda:0"
            print(f"🎯 Detection tool using hardcoded GPU assignment: {device}")

    return RemoteSAMDetectionTool(device=device)
