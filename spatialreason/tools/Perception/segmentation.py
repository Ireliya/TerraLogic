"""
RemoteSAM Segmentation Tool for Water and Building Detection
Uses RemoteSAMv1.pth model for better segmentation performance.
"""

from typing import Type, Optional, List, Dict, Any
import os
import json
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import torch
import numpy as np
from PIL import Image

import cv2
from skimage.measure import label, regionprops
# Removed circular import - GeometryValidator will be imported dynamically if needed


def load_class_config() -> Dict[str, List[str]]:
    """Load class configuration from class_config.json"""
    config_path = Path(__file__).parent.parent / "class_config.json"
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


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

try:
    # Import the RemoteSAM model from the extracted components
    from .remotesam.model import RemoteSAM, init_demo_model
    SAM_AVAILABLE = True
except ImportError as e:
    SAM_AVAILABLE = False
    print(f"Warning: RemoteSAM model not available: {e}")


class SegmentationInput(BaseModel):
    """
    Input arguments for RemoteSAM Segmentation Tool:
      - image_path: path to PNG/JPG image
      - text_prompt: natural language description of objects to segment
    """
    image_path: str = Field(..., description="Path to image file (PNG or JPG).")
    text_prompt: str = Field(..., description="Natural language description of objects to segment (e.g., 'water bodies', 'buildings', 'roads', 'trees').")


class RemoteSAMSegmentationTool(BaseTool):
    """
    Text-prompted segmentation tool using RemoteSAMv1.pth model.
    Segments objects based on natural language descriptions (e.g., 'water bodies', 'buildings', 'roads').
    """
    name: str = "segmentation"
    description: str = (
        "Segment objects in satellite imagery using text prompts with RemoteSAM model. "
        "Supports natural language descriptions like 'water bodies', 'buildings', 'roads', 'vegetation'. "
        "Returns JSON with segmented object locations."
    )
    args_schema: Type[BaseModel] = SegmentationInput

    # Configure Pydantic to allow arbitrary types
    class Config:
        arbitrary_types_allowed = True

    # Add all fields that are set in __init__
    device: Any = None
    remote_sam: Any = None
    class_config: Dict[str, List[str]] = None
    geometry_validator: Any = None

    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda:2',
        fallback_to_hf: bool = True,
        cache_dir: str = None
    ):
        super().__init__()

        if not SAM_AVAILABLE:
            raise ImportError("Required libraries for RemoteSAM are not available.")

        # Set device
        print(f"[Segmentation] __init__ called with device: {device}")
        self.device = device

        try:
            # Initialize the model using updated init_demo_model with HF support
            model = init_demo_model(
                checkpoint=model_path,
                device=device,
                fallback_to_hf=fallback_to_hf,
                cache_dir=cache_dir
            )

            # Create RemoteSAM instance using your RemoteSAM class
            self.remote_sam = RemoteSAM(
                RemoteSAM_model=model,
                device=device,
                use_EPOC=False,  # Disable EPOC for simplicity
                EPOC_threshold=0.5,
                MLC_balance_factor=0.5,
                MCC_balance_factor=1.0
            )

        except Exception as e:
            raise

        # Load class configuration
        self.class_config = load_class_config()

        # Initialize geometry validator (dynamic import to avoid circular imports)
        try:
            from create_data.generate_gt.validation import GeometryValidator
            self.geometry_validator = GeometryValidator()
        except ImportError:
            # Fallback if validation module is not available
            self.geometry_validator = None

    def _segment_image(self, image_path: str, classes: List[str]) -> Dict[str, np.ndarray]:
        """Segment image using RemoteSAM model"""
        # Load image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')

        # Use your RemoteSAM semantic segmentation
        mask_results = self.remote_sam.semantic_seg(
            image=image,
            classnames=classes,
            return_prob=False
        )

        return mask_results

    def _extract_segments_from_mask(self, mask: np.ndarray, class_name: str) -> List[Dict]:
        """
        Extract polygon geometries from segmentation mask.

        Args:
            mask: Binary mask of segmented objects
            class_name: Name of the object class

        Returns:
            List of segment dictionaries with polygon coordinates as primary geometry
        """
        detections = []

        mask_sum = mask.sum()

        if mask_sum == 0:
            return detections

        # Label connected components
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        for i, region in enumerate(regions):
            # Get bounding box coordinates
            min_row, min_col, max_row, max_col = region.bbox

            # Calculate confidence based on region area (same logic as detection tool)
            area = region.area
            total_area = mask.shape[0] * mask.shape[1]

            # More reasonable confidence calculation:
            area_ratio = area / total_area
            if area_ratio < 0.001:  # < 0.1% of image
                confidence = min(0.5, max(0.3, 0.3 + area_ratio * 200))
            elif area_ratio < 0.01:  # 0.1% - 1% of image
                confidence = min(0.8, max(0.5, 0.5 + (area_ratio - 0.001) * 33.33))
            else:  # > 1% of image
                confidence = min(0.95, max(0.8, 0.8 + (area_ratio - 0.01) * 1.5))

            # Extract polygon coordinates from the region
            polygon_coords = self._extract_polygon_from_region(labeled_mask, region.label)

            # Always ensure we have valid polygon coordinates
            if not polygon_coords or len(polygon_coords) < 3:
                print(f"⚠️ [Segmentation] No valid polygon extracted for {class_name}_{i+1}, creating bbox polygon")
                # Create polygon from bounding box as fallback
                polygon_coords = [
                    [float(min_col), float(min_row)],
                    [float(max_col), float(min_row)],
                    [float(max_col), float(max_row)],
                    [float(min_col), float(max_row)],
                    [float(min_col), float(min_row)]  # Close the polygon
                ]
            else:
                print(f"✅ [Segmentation] Successfully extracted polygon for {class_name}_{i+1} with {len(polygon_coords)} points")

            detection = {
                "object_id": f"{class_name}_{i+1}",
                "class": class_name,
                "confidence": round(confidence, 3),
                "area_pixels": int(area),
                "centroid": {
                    "x": round(region.centroid[1], 1),
                    "y": round(region.centroid[0], 1)
                },
                "polygon": polygon_coords
            }
            detections.append(detection)

        return detections

    def _extract_polygon_from_region(self, labeled_mask: np.ndarray, region_label: int) -> List[List[float]]:
        """
        Extract polygon coordinates from a labeled region in the segmentation mask.

        Args:
            labeled_mask: Labeled mask with connected components
            region_label: Label of the specific region to extract

        Returns:
            List of [x, y] coordinate pairs forming the polygon boundary
        """
        try:
            # Create binary mask for this specific region
            region_mask = (labeled_mask == region_label).astype(np.uint8)

            # Find contours of the region
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print(f"⚠️ [Segmentation] No contours found for region {region_label}")
                return []

            # Use the largest contour (should be the main region)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)

            if contour_area < 9:  # Less than 3x3 pixels
                print(f"⚠️ [Segmentation] Contour too small for region {region_label}: {contour_area} pixels")
                # Return a minimal bounding box polygon instead of empty list
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > 0 and h > 0:
                    return [
                        [float(x), float(y)],
                        [float(x + w), float(y)],
                        [float(x + w), float(y + h)],
                        [float(x), float(y + h)],
                        [float(x), float(y)]  # Close the polygon
                    ]
                else:
                    return []

            # Try to get bounding rectangle first as fallback
            x, y, w, h = cv2.boundingRect(largest_contour)
            bbox_polygon = [
                [float(x), float(y)],
                [float(x + w), float(y)],
                [float(x + w), float(y + h)],
                [float(x), float(y + h)],
                [float(x), float(y)]  # Close the polygon
            ]

            # Try to extract actual contour polygon
            try:
                # Simplify the contour to reduce number of points while preserving shape
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # 2% of perimeter
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Convert contour to list of [x, y] coordinates
                polygon_coords = []
                for point in simplified_contour:
                    try:
                        # Handle different OpenCV contour formats
                        if isinstance(point, (list, tuple)) and len(point) > 0:
                            if isinstance(point[0], (list, tuple, np.ndarray)) and len(point[0]) >= 2:
                                # Format: [[[x, y]]] or [[x, y]]
                                x_pt, y_pt = point[0][0], point[0][1]
                            elif len(point) >= 2:
                                # Format: [[x, y]]
                                x_pt, y_pt = point[0], point[1]
                            else:
                                continue
                        elif isinstance(point, np.ndarray):
                            if point.shape == (1, 2):
                                # Format: [[x, y]]
                                x_pt, y_pt = point[0, 0], point[0, 1]
                            elif point.shape == (2,):
                                # Format: [x, y]
                                x_pt, y_pt = point[0], point[1]
                            else:
                                continue
                        else:
                            continue

                        polygon_coords.append([float(x_pt), float(y_pt)])
                    except (IndexError, ValueError, TypeError) as e:
                        print(f"⚠️ [Segmentation] Failed to extract point from contour: {e}, point shape: {getattr(point, 'shape', 'no shape')}")
                        continue

                # Ensure we have at least 3 points for a valid polygon
                if len(polygon_coords) >= 3:
                    # Close polygon if not already closed
                    if polygon_coords[0] != polygon_coords[-1]:
                        polygon_coords.append(polygon_coords[0])

                    # Apply geometry validation to clean the polygon (if available)
                    if self.geometry_validator is not None:
                        try:
                            # Use the process method for full validation with report
                            validation_result = self.geometry_validator.process(
                                polygon_coords,
                                validation_type="geometry_validity"
                            )

                            if validation_result.success and validation_result.data:
                                validation_report = validation_result.data

                                # Check if geometry was fixed
                                if "fixed_geometry" in validation_report and validation_report["fixed_geometry"] is not None:
                                    cleaned_polygon = validation_result.data["fixed_geometry"]
                                    geometry_validated = True
                                    repairs_applied = validation_report.get("warnings", [])
                                else:
                                    # Use original polygon if no fixes were needed
                                    cleaned_polygon = polygon_coords
                                    geometry_validated = validation_report.get("is_valid", True)
                                    repairs_applied = []
                            else:
                                # Fallback to simple cleaning
                                cleaned_polygon = self.geometry_validator.clean_geometry(polygon_coords)
                                geometry_validated = True
                                repairs_applied = []

                        except Exception as validation_error:
                            print(f"⚠️ [Segmentation] Geometry validation error: {validation_error}")
                            # Fallback to simple cleaning
                            cleaned_polygon = self.geometry_validator.clean_geometry(polygon_coords)
                            geometry_validated = True
                            repairs_applied = []
                    else:
                        # Fallback: use original polygon without validation
                        cleaned_polygon = polygon_coords
                        geometry_validated = True
                        repairs_applied = []

                    if cleaned_polygon is not None and geometry_validated:
                        # Convert back to coordinate list format
                        if hasattr(cleaned_polygon, 'exterior'):
                            coords = list(cleaned_polygon.exterior.coords)
                            validated_coords = [[float(x), float(y)] for x, y in coords]
                            if repairs_applied:
                                print(f"✅ [Segmentation] Applied geometry repairs: {repairs_applied}")
                            return validated_coords
                        else:
                            # cleaned_polygon is already in coordinate list format
                            if isinstance(cleaned_polygon, list) and len(cleaned_polygon) >= 3:
                                return cleaned_polygon
                            else:
                                print(f"⚠️ [Segmentation] Geometry validation failed, using original polygon")
                                return polygon_coords
                    else:
                        print(f"⚠️ [Segmentation] Geometry validation failed, using bbox polygon")
                        return bbox_polygon
                else:
                    print(f"⚠️ [Segmentation] Simplified contour has too few points ({len(polygon_coords)}), using bbox")
                    return bbox_polygon

            except Exception as contour_error:
                print(f"⚠️ [Segmentation] Contour processing failed for region {region_label}: {contour_error}, using bbox")
                return bbox_polygon

        except Exception as e:
            print(f"⚠️ [Segmentation] Failed to extract polygon for region {region_label}: {e}")
            return []

    def _run(
        self,
        image_path: str,
        text_prompt: str,
        confidence_threshold: Optional[float] = 0.5,
        meters_per_pixel: Optional[float] = None
    ) -> str:
        """
        Execute the RemoteSAM segmentation tool.

        Args:
            image_path: Path to the input image
            text_prompt: Natural language description of objects to segment
            confidence_threshold: Minimum confidence threshold for segments
            meters_per_pixel: Ground resolution in meters per pixel (optional, echoed back in response)

        Returns:
            JSON string containing segmentation results
        """
        try:
            # Validate input image exists and is readable
            from pathlib import Path
            import cv2

            if not Path(image_path).exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")

            # Try to load image to verify it's valid
            test_image = cv2.imread(image_path)
            if test_image is None:
                raise ValueError(f"Cannot read image file: {image_path}. File may be corrupted or in unsupported format.")

            print(f"[Segmentation] ✅ Image validation passed: {Path(image_path).name} ({test_image.shape})")

            # Parse text prompt to determine what to segment
            # Extract individual class names from the prompt
            import re

            # Look for patterns like "water and building", "building and forest", etc.
            # Common classes to look for (LoveDA, Potsdam, Vaihingen datasets - both singular and plural forms)
            loveda_classes = [
                "building", "buildings", "road", "roads", "water", "barren", "forest", "agriculture",
                "low_vegetation", "tree", "trees", "car", "cars"
            ]

            classes = []
            text_lower = text_prompt.lower()

            # Use word boundary matching to avoid substring matches (e.g., "tree" matching "trees")
            # Sort by length descending to match longer class names first (e.g., "low_vegetation" before "vegetation")
            for class_name in sorted(loveda_classes, key=len, reverse=True):
                # Use regex word boundary matching to ensure exact word matches
                pattern = r'\b' + re.escape(class_name) + r'\b'
                if re.search(pattern, text_lower):
                    # Keep the exact class name as it appears in the loveda_classes list
                    # This preserves the distinction between "tree" and "trees", "car" and "cars", etc.
                    if class_name not in classes:
                        classes.append(class_name)

            # If no specific classes found, fall back to the original mapping
            if not classes:
                classes = map_text_prompt_to_classes(text_prompt, self.class_config)

            # Perform segmentation using RemoteSAM
            mask_results = self._segment_image(image_path, classes)

            # Extract segments from masks and apply confidence filtering
            all_detections = []

            for class_name, mask in mask_results.items():
                if mask is not None and mask.sum() > 0:
                    detections = self._extract_segments_from_mask(mask, class_name)

                    # Filter by confidence threshold
                    if confidence_threshold:
                        detections = [d for d in detections if d["confidence"] >= confidence_threshold]

                    all_detections.extend(detections)
                else:
                    print(f"⚠️  Warning: RemoteSAM segmentation found no mask for {class_name} in image")
                    print(f"   This could indicate: (1) class not present in image, (2) model limitations, or (3) semantic mismatch")
                    # Continue processing other classes instead of failing immediately

            # Ensure we have detected objects - provide detailed debugging info
            if len(all_detections) == 0:
                # Count total detections before filtering for debugging
                total_before_filtering = 0
                confidence_info = []

                for class_name, mask in mask_results.items():
                    if mask is not None and mask.sum() > 0:
                        raw_detections = self._extract_segments_from_mask(mask, class_name)
                        total_before_filtering += len(raw_detections)

                        if raw_detections:
                            confidences = [d["confidence"] for d in raw_detections]
                            confidence_info.append(f"{class_name}: {len(raw_detections)} detections, confidence range: {min(confidences):.3f}-{max(confidences):.3f}")

                debug_info = f"Total detections before filtering: {total_before_filtering}. "
                if confidence_info:
                    debug_info += f"Confidence details: {'; '.join(confidence_info)}. "
                debug_info += f"Current threshold: {confidence_threshold}. "

                raise RuntimeError(f"No objects detected after confidence filtering. {debug_info}"
                                 f"Consider lowering confidence threshold or checking semantic alignment between prompt and image content.")

            # Use organized output directory structure
            import os

            # Extract image ID for directory organization
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            if image_id.endswith('_vis'):
                image_id = image_id[:-4]  # Remove '_vis' suffix

            # Create organized output directory
            output_dir = Path("temp") / "loveda" / image_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create standardized class-based naming (avoid duplication)
            unique_classes = sorted(set([obj['class'] for obj in all_detections]))
            class_names = "_".join(unique_classes)
            if not class_names:
                class_names = "no_detections"

            # Create and save segmentation mask visualizations
            segmentation_mask_paths = {}
            segmentation_image_path = None

            if len(all_detections) > 0:
                # Load original image for visualization
                original_image = cv2.imread(image_path)
                if original_image is not None:
                    # Create combined segmentation mask visualization
                    segmentation_image = original_image.copy()
                    combined_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)

                    # Color map for different classes
                    class_colors = {
                        'building': (0, 0, 255),    # Red
                        'road': (0, 255, 0),       # Green
                        'water': (255, 0, 0),      # Blue
                        'barren': (0, 255, 255),   # Yellow
                        'forest': (255, 0, 255),   # Magenta
                        'agriculture': (255, 255, 0) # Cyan
                    }

                    # Process each class mask from the segmentation results
                    for class_name, mask in mask_results.items():
                        if mask is not None and mask.sum() > 0:
                            # Save individual class mask with standardized naming
                            class_mask_filename = f"segmentation_{class_name}.png"
                            class_mask_path = output_dir / class_mask_filename
                            cv2.imwrite(str(class_mask_path), mask * 255)
                            segmentation_mask_paths[class_name] = str(class_mask_path)

                            # Add to combined mask with class-specific color
                            color = class_colors.get(class_name, (128, 128, 128))  # Default gray
                            mask_colored = np.zeros_like(original_image)
                            mask_colored[mask > 0] = color

                            # Blend with original image
                            alpha = 0.6
                            segmentation_image = cv2.addWeighted(segmentation_image, 1-alpha, mask_colored, alpha, 0)

                            # Add to combined mask
                            combined_mask[mask > 0] = 255

                    # Save combined segmentation visualization with standardized naming
                    segmentation_filename = f"segmentation_{class_names}.png"
                    segmentation_image_path = output_dir / segmentation_filename
                    cv2.imwrite(str(segmentation_image_path), segmentation_image)
                    segmentation_image_path = str(segmentation_image_path)

                    # Save combined mask with stable naming format
                    combined_mask_filename = f"combined_{class_names}.png"
                    combined_mask_path = output_dir / combined_mask_filename
                    cv2.imwrite(str(combined_mask_path), combined_mask)
                    segmentation_mask_paths['combined'] = str(combined_mask_path)

                    print(f"[Segmentation] 💾 Saved segmentation visualization: {segmentation_image_path}")
                    print(f"[Segmentation] 💾 Saved {len(segmentation_mask_paths)} mask files")

            # Prepare segmentation results with segments and visualization paths
            result = {
                "success": True,
                "image_path": image_path,
                "text_prompt": text_prompt,
                "total_segments": len(all_detections),
                "segments": all_detections,
                "segmentation_masks": segmentation_mask_paths,
                "confidence_threshold": confidence_threshold,
                "summary": f"RemoteSAM segmentation completed for '{text_prompt}'. Found {len(all_detections)} segments."
            }

            # Add segmentation image path if created
            if segmentation_image_path:
                result["segmentation_image_path"] = segmentation_image_path

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in the detection tool for consistency

            # Sort classes alphabetically for consistency
            classes_requested = sorted(classes)

            # Calculate segments by class in the same order as classes_requested
            segments_by_class = []
            for cls in classes_requested:
                count = sum(1 for seg in all_detections if seg["class"] == cls)
                segments_by_class.append(count)

            # Calculate class_stats: coverage percentage for each segmented class
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
                print(f"[Segmentation] Warning: Could not load image for area calculation: {e}")
                total_image_area = 1

            # Calculate coverage percentage for each class
            class_stats = {}
            for cls in classes_requested:
                class_segments = [seg for seg in all_detections if seg["class"] == cls]
                if class_segments:
                    total_class_area = sum(seg.get("area_pixels", 0) for seg in class_segments)
                    coverage_pct = round(100.0 * total_class_area / total_image_area, 2)
                    class_stats[cls] = {"coverage_pct": coverage_pct}

            # Build the unified arguments field
            arguments = {
                "image_path": image_path,
                "classes_requested": classes_requested,
                "meters_per_pixel": meters_per_pixel,  # Echo back the meters_per_pixel parameter from the tool call
                "text_prompt": text_prompt,
                "total_segments": len(all_detections),
                "segments_by_class": segments_by_class,
                "class_stats": class_stats
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            # Note: Results will be saved by TempStorageManager to image-specific directories
            # No longer saving to global temp/segmentation directory

            # Return raw JSON result for programmatic use
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"RemoteSAM segmentation failed: {str(e)}"
            error_result = {
                "success": False,
                "error": error_msg,
                "image_path": image_path,
                "text_prompt": text_prompt,
                "summary": f"Segmentation failed for '{text_prompt}': {error_msg}"
            }
            # Return raw JSON error result for programmatic use
            return json.dumps(error_result, indent=2)




# Create convenience function for backward compatibility
def create_segmentation_tool(model_path: str = "pretrained_weights/RemoteSAMv1.pth", device: str = "auto") -> RemoteSAMSegmentationTool:
    """
    Create a RemoteSAM segmentation tool with specified parameters.

    Args:
        model_path: Path to RemoteSAMv1.pth model file
        device: Device to run the model on ("auto" for automatic GPU selection, 'cuda:0', 'cpu')

    Returns:
        Configured RemoteSAMSegmentationTool instance
    """
    if device == "auto":
        # Check for hardcoded GPU assignment
        import os
        if os.getenv('SPATIAL_REASONING_GPU_MODE') == 'hardcoded':
            perception_gpu = os.getenv('SPATIAL_REASONING_PERCEPTION_GPU', '2')
            device = f"cuda:{perception_gpu}"
            print(f"🎯 Segmentation tool using hardcoded GPU assignment: {device}")
        else:
            # Use hardcoded GPU 2 for all perception tools
            device = "cuda:2"
            print(f"🎯 Segmentation tool using hardcoded GPU assignment: {device}")

    return RemoteSAMSegmentationTool(model_path=model_path, device=device)
