"""
Region-based Scene Classification Tool using RemoteSAM model.
This tool performs region-based classification in satellite imagery by first identifying distinct spatial regions
based on visual similarity and spatial coherence, then classifying each region into one of 6 primary scene categories:
Urban land, Agriculture land, Rangeland, Forest land, Water, and Barren land.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from langchain_core.tools import BaseTool
from PIL import Image
import logging
import cv2
from skimage import measure, morphology

# Import RemoteSAM components for semantic segmentation
try:
    from .remotesam.model import init_demo_model, RemoteSAM
    SAM_AVAILABLE = True
except ImportError:
    print("⚠️  RemoteSAM not available for semantic segmentation classification")
    SAM_AVAILABLE = False


# Primary Scene Classification Categories (6 fundamental scene categories)
HIERARCHICAL_CATEGORIES = {
    0: "Urban land",
    1: "Agriculture land",
    2: "Rangeland",
    3: "Forest land",
    4: "Water",
    5: "Barren land"
}

# Mapping from fine-grained classes to primary scene categories
# Focus on 10 fundamental land-cover classes
CLASS_TO_CATEGORY_MAPPING = {
    # Urban land category
    "building": "Urban land",
    "cars": "Urban land",
    "car": "Urban land",
    "road": "Urban land",
    "bridge": "Urban land",
    "impervious surfaces": "Urban land",
    "small vehicle": "Urban land",
    "large vehicle": "Urban land",

    # Agriculture land category
    "agriculture": "Agriculture land",

    # Rangeland category
    "low vegetation": "Rangeland",
    "low_vegetation": "Rangeland",

    # Forest land category
    "forest": "Forest land",
    "tree": "Forest land",
    "trees": "Forest land",

    # Water category
    "water": "Water",

    # Barren land category
    "barren": "Barren land"
}

# 10 fundamental land-cover classes for scene classification
REMOTE_SENSING_CLASSES = [
    "agriculture", "barren", "bridge", "building", "cars",
    "forest", "low_vegetation", "road", "trees", "water"
]

# Aliases for scene category matching
CATEGORY_ALIASES = {
    "Urban land": ["urban", "residential", "city", "town", "developed", "settlement", "buildings", "infrastructure", "roads"],
    "Agriculture land": ["agriculture", "agricultural", "crop", "farming", "farmland"],
    "Rangeland": ["rangeland", "grassland", "pasture", "low vegetation", "shrub"],
    "Forest land": ["forest", "woodland", "trees", "vegetation", "forested"],
    "Water": ["water", "water body", "lake", "river", "ocean", "sea", "aquatic"],
    "Barren land": ["barren", "bare", "desert", "rock", "exposed", "unvegetated"]
}


class HierarchicalRegionClassifier:
    """
    RemoteSAM-based scene classifier for spatial region classification.
    First identifies distinct spatial regions using 10 fundamental land-cover classes,
    then maps objects to 6 primary scene categories: Urban land, Agriculture land, Rangeland,
    Forest land, Water, and Barren land. Groups objects by spatial proximity and category.
    """

    def __init__(self, device: str = "cuda:0", cache_dir: str = None):
        """
        Initialize the semantic segmentation classifier.

        Args:
            device: Device to run the model on
            cache_dir: Directory to cache downloaded models
        """
        self.device = device
        self.cache_dir = cache_dir
        self.remote_sam = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the RemoteSAM model for semantic segmentation."""
        if not SAM_AVAILABLE:
            raise ImportError("RemoteSAM not available for semantic segmentation classification")

        try:
            # Initialize RemoteSAM model
            model = init_demo_model(
                checkpoint=None,  # Use HF fallback
                device=self.device,
                fallback_to_hf=True,
                cache_dir=self.cache_dir
            )

            # Create RemoteSAM instance
            self.remote_sam = RemoteSAM(
                RemoteSAM_model=model,
                device=self.device,
                use_EPOC=False,
                EPOC_threshold=0.25,
                MLC_balance_factor=0.5,
                MCC_balance_factor=1.0
            )

            print(f"✅ RemoteSAM semantic segmentation classifier initialized on {self.device}")

        except Exception as e:
            print(f"❌ Failed to initialize RemoteSAM semantic segmentation classifier: {e}")
            raise e

    def classify_image_hierarchical(self, image_path: str, target_classes: List[str] = None) -> Dict[str, Any]:
        """
        Perform scene-based region classification.

        Args:
            image_path: Path to the input image
            target_classes: List of target land-cover classes to detect (if None, uses all 10 classes)

        Returns:
            Dictionary containing scene classification results and regional groupings
        """
        if target_classes is None:
            target_classes = REMOTE_SENSING_CLASSES

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Stage 1: Perform semantic segmentation for each of the 10 land-cover classes
        segmentation_results = {}
        detected_objects = []

        for class_name in target_classes:
            try:
                # Use RemoteSAM semantic segmentation
                mask_results = self.remote_sam.semantic_seg(
                    image=image,
                    classnames=[class_name],
                    return_prob=False
                )

                mask = mask_results.get(class_name)
                if mask is not None and mask.sum() > 0:
                    # Process the segmentation mask to extract regions
                    regions = self._extract_regions_from_mask(mask, class_name)
                    segmentation_results[class_name] = {
                        'regions': regions,
                        'total_pixels': int(mask.sum()),
                        'coverage_percentage': float(mask.sum() / (mask.shape[0] * mask.shape[1]) * 100)
                    }
                    detected_objects.extend(regions)

            except Exception as e:
                print(f"⚠️  Failed to segment {class_name}: {e}")
                continue

        # Stage 2: Map detected objects to hierarchical categories
        hierarchical_regions = self._group_objects_by_category(detected_objects)

        return {
            'segmentation_results': segmentation_results,
            'detected_objects': detected_objects,
            'hierarchical_regions': hierarchical_regions,
            'total_classes_found': len(segmentation_results),
            'total_categories_found': len(hierarchical_regions),
            'image_size': image.size
        }

    def _extract_regions_from_mask(self, mask: np.ndarray, class_name: str) -> List[Dict[str, Any]]:
        """
        Extract distinct regions from a segmentation mask.

        Args:
            mask: Binary segmentation mask
            class_name: Name of the land cover class
            image_size: Size of the original image (width, height)

        Returns:
            List of region dictionaries with bounding boxes, polygon coordinates, and properties
        """
        regions = []

        # Clean up the mask using morphological operations
        cleaned_mask = morphology.remove_small_objects(mask.astype(bool), min_size=100)
        cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=50)

        # Label connected components
        labeled_mask = measure.label(cleaned_mask)
        region_props = measure.regionprops(labeled_mask)

        for i, region in enumerate(region_props):
            # Calculate region properties
            area_pixels = region.area
            bbox = region.bbox  # (min_row, min_col, max_row, max_col)
            centroid = region.centroid

            # Convert to standard bbox format
            min_row, min_col, max_row, max_col = bbox

            # Calculate confidence based on region size and compactness
            compactness = (4 * np.pi * area_pixels) / (region.perimeter ** 2) if region.perimeter > 0 else 0
            size_factor = min(area_pixels / 1000, 1.0)  # Normalize by expected region size
            confidence = min(0.5 + 0.3 * size_factor + 0.2 * compactness, 1.0)

            # Extract polygon coordinates from the region
            polygon_coords = self._extract_polygon_from_region(labeled_mask, region.label)

            region_dict = {
                "object_id": f"{class_name}_{i+1}",
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x_min": int(min_col),
                    "y_min": int(min_row),
                    "x_max": int(max_col),
                    "y_max": int(max_row),
                    "width": int(max_col - min_col),
                    "height": int(max_row - min_row)
                },
                "area_pixels": int(area_pixels),
                "centroid": {
                    "x": round(centroid[1], 1),
                    "y": round(centroid[0], 1)
                },
                "coverage_percentage": round(area_pixels / (mask.shape[0] * mask.shape[1]) * 100, 2),
                "compactness": round(compactness, 3),
                "polygon": polygon_coords  # Add polygon coordinates for spatial analysis
            }
            regions.append(region_dict)

        # Sort regions by area (largest first)
        regions.sort(key=lambda x: x["area_pixels"], reverse=True)

        return regions

    def _extract_polygon_from_region(self, labeled_mask: np.ndarray, region_label: int) -> List[List[float]]:
        """
        Extract polygon coordinates from a labeled region in the segmentation mask.
        Returns polygon in standard format: [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]

        Args:
            labeled_mask: Labeled mask with connected components
            region_label: Label of the specific region to extract

        Returns:
            List of [x, y] coordinate pairs forming the polygon boundary (closed polygon)
        """
        try:
            # Create binary mask for this specific region
            region_mask = (labeled_mask == region_label).astype(np.uint8)

            # Find contours of the region
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return []

            # Use the largest contour (should be the main region)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rectangle as the standard polygon format
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Create polygon in standard format: [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]
            polygon_coords = [
                [float(x), float(y)],
                [float(x + w), float(y)],
                [float(x + w), float(y + h)],
                [float(x), float(y + h)],
                [float(x), float(y)]  # Close the polygon
            ]

            return polygon_coords

        except Exception as e:
            print(f"⚠️  Failed to extract polygon for region {region_label}: {e}")
            return []

    def _group_objects_by_category(self, detected_objects: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group detected objects into hierarchical categories and calculate regional statistics.

        Args:
            detected_objects: List of detected object regions

        Returns:
            Dictionary with hierarchical categories as keys and regional statistics as values
        """
        hierarchical_regions = {}

        # Initialize category groups
        for category in HIERARCHICAL_CATEGORIES.values():
            hierarchical_regions[category] = {
                'objects': [],
                'total_objects': 0,
                'total_area_pixels': 0,
                'coverage_percentage': 0.0,
                'confidence_scores': [],
                'average_confidence': 0.0,
                'bounding_box': None,
                'centroid': None
            }

        # Group objects by their hierarchical category
        for obj in detected_objects:
            class_name = obj.get('class', '')
            category = CLASS_TO_CATEGORY_MAPPING.get(class_name)

            if category and category in hierarchical_regions:
                hierarchical_regions[category]['objects'].append(obj)
                hierarchical_regions[category]['total_objects'] += 1
                hierarchical_regions[category]['total_area_pixels'] += obj.get('area_pixels', 0)
                hierarchical_regions[category]['confidence_scores'].append(obj.get('confidence', 0.0))

        # Calculate statistics for each category
        image_total_pixels = 0
        if detected_objects:
            # Estimate total image pixels from first object's coverage calculation
            first_obj = detected_objects[0]
            if 'coverage_percentage' in first_obj and first_obj['coverage_percentage'] > 0:
                image_total_pixels = int(first_obj['area_pixels'] / (first_obj['coverage_percentage'] / 100))

        for category, data in hierarchical_regions.items():
            if data['total_objects'] > 0:
                # Calculate average confidence
                data['average_confidence'] = sum(data['confidence_scores']) / len(data['confidence_scores'])

                # Calculate coverage percentage
                if image_total_pixels > 0:
                    data['coverage_percentage'] = (data['total_area_pixels'] / image_total_pixels) * 100

                # Calculate combined bounding box for the category
                data['bounding_box'] = self._calculate_category_bounding_box(data['objects'])

                # Calculate category centroid
                data['centroid'] = self._calculate_category_centroid(data['objects'])

        # Remove empty categories
        hierarchical_regions = {k: v for k, v in hierarchical_regions.items() if v['total_objects'] > 0}

        return hierarchical_regions

    def _calculate_category_bounding_box(self, objects: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate combined bounding box for objects in a category."""
        if not objects:
            return None

        min_x = min(obj['bbox']['x_min'] for obj in objects)
        min_y = min(obj['bbox']['y_min'] for obj in objects)
        max_x = max(obj['bbox']['x_max'] for obj in objects)
        max_y = max(obj['bbox']['y_max'] for obj in objects)

        return {
            'x_min': min_x,
            'y_min': min_y,
            'x_max': max_x,
            'y_max': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }

    def _calculate_category_centroid(self, objects: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate weighted centroid for objects in a category."""
        if not objects:
            return None

        total_area = sum(obj.get('area_pixels', 0) for obj in objects)
        if total_area == 0:
            return None

        weighted_x = sum(obj['centroid']['x'] * obj.get('area_pixels', 0) for obj in objects)
        weighted_y = sum(obj['centroid']['y'] * obj.get('area_pixels', 0) for obj in objects)

        return {
            'x': round(weighted_x / total_area, 1),
            'y': round(weighted_y / total_area, 1)
        }


class RemoteSAMClassificationTool(BaseTool):
    """
    Scene classification tool using RemoteSAM model.
    Performs region-based classification by first detecting objects from 10 fundamental land-cover classes,
    then mapping them to 6 primary scene categories: Urban land, Agriculture land, Rangeland,
    Forest land, Water, and Barren land. Groups objects by spatial proximity and category.
    """

    name: str = "classification"
    description: str = "Perform scene classification mapping 10 land-cover classes to 6 primary scene categories"

    # Private attributes for the model
    _classifier: Optional[HierarchicalRegionClassifier] = None
    _device: str = "cuda:2"
    _logger: logging.Logger = None

    def __init__(self, device: str = "cuda:2", cache_dir: str = None, **kwargs):
        super().__init__(**kwargs)
        print(f"[Classification] __init__ called with device: {device}")
        self._device = device
        self._logger = logging.getLogger(__name__)
        self._initialize_model(cache_dir=cache_dir)

    def run(self, tool_input: Union[Dict[str, Any], str]) -> str:
        """
        Run the classification tool with dictionary input.

        Args:
            tool_input: Dictionary containing tool parameters or JSON string

        Returns:
            JSON string containing classification results
        """
        try:
            # Handle both dictionary and JSON string input
            if isinstance(tool_input, str):
                import json
                params = json.loads(tool_input)
            else:
                params = tool_input

            # Extract parameters
            image_path = params.get("image_path")
            text_prompt = params.get("text_prompt", "")
            confidence_threshold = params.get("confidence_threshold", 0.5)
            meters_per_pixel = params.get("meters_per_pixel", 0.3)
            classes_requested = params.get("classes_requested", None)

            # Call the internal _run method
            return self._run(
                image_path=image_path,
                text_prompt=text_prompt,
                confidence_threshold=confidence_threshold,
                meters_per_pixel=meters_per_pixel,
                classes_requested=classes_requested
            )

        except Exception as e:
            error_result = {
                "success": False,
                "tool_name": "classification",
                "error": str(e),
                "summary": f"Classification failed: {str(e)}"
            }
            return json.dumps(error_result, indent=2)

    def _initialize_model(self, cache_dir=None):
        """Initialize the RemoteSAM hierarchical region classification model."""
        try:
            self._logger.info("Loading RemoteSAM hierarchical region classification model...")

            # Initialize the hierarchical region classifier
            self._classifier = HierarchicalRegionClassifier(
                device=self._device,
                cache_dir=cache_dir
            )

            self._logger.info(f"✅ RemoteSAM hierarchical region classifier initialized on {self._device}")

        except Exception as e:
            self._logger.error(f"❌ Failed to initialize RemoteSAM hierarchical region classifier: {e}")
            self._classifier = None
    
    def _classify_image_hierarchical(self, image_path: str, target_classes: List[str] = None) -> Dict[str, Any]:
        """
        Perform scene-based region classification.

        Args:
            image_path: Path to the input image
            target_classes: List of target land-cover classes to detect (if None, uses all 10 classes)

        Returns:
            Dictionary containing scene classification results
        """
        # Load and validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Two-stage approach for improved reliability:
            # Stage 1: Run segmentation to identify all class regions in the image
            print(f"   🔍 Stage 1: Running segmentation to identify class regions...")

            try:
                results = self._classifier.classify_image_hierarchical(image_path, target_classes)
                segmentation_results = results['segmentation_results']
                detected_objects = results['detected_objects']
                hierarchical_regions = results['hierarchical_regions']

            except Exception:
                # Fallback: Try with all available classes if target classes failed
                try:
                    results = self._classifier.classify_image_hierarchical(image_path, None)
                    segmentation_results = results['segmentation_results']
                    detected_objects = results['detected_objects']
                    hierarchical_regions = results['hierarchical_regions']
                except Exception as fallback_error:
                    raise RuntimeError(f"Hierarchical classification failed: {fallback_error}")

            if not detected_objects:
                # Return result indicating no detections rather than failing
                return {
                    "predicted_category": "Unknown",
                    "confidence": 0.0,
                    "segmentation_results": segmentation_results,
                    "detected_objects": detected_objects,
                    "hierarchical_regions": hierarchical_regions,
                    "total_classes_found": 0,
                    "total_categories_found": 0,
                    "coverage_stats": {},
                    "image_size": results.get('image_size', [0, 0])
                }

            # Calculate coverage statistics for hierarchical categories

            # Coverage stats for individual classes (for compatibility)
            coverage_stats = {}

            for class_name, class_data in segmentation_results.items():
                coverage_stats[class_name] = {
                    'total_pixels': class_data['total_pixels'],
                    'coverage_percentage': class_data['coverage_percentage'],
                    'region_count': len(class_data['regions']),
                    'confidence': class_data['coverage_percentage'] / 100.0  # Normalize to 0-1
                }

            # Coverage stats for hierarchical categories
            category_stats = {}
            for category, region_data in hierarchical_regions.items():
                category_stats[category] = {
                    'total_objects': region_data['total_objects'],
                    'total_area_pixels': region_data['total_area_pixels'],
                    'coverage_percentage': region_data['coverage_percentage'],
                    'average_confidence': region_data['average_confidence']
                }

            if not category_stats:
                return {
                    "predicted_category": "Unknown",
                    "confidence": 0.0,
                    "segmentation_results": segmentation_results,
                    "detected_objects": detected_objects,
                    "hierarchical_regions": hierarchical_regions,
                    "total_classes_found": 0,
                    "total_categories_found": 0,
                    "coverage_stats": coverage_stats,
                    "category_stats": category_stats,
                    "image_size": results.get('image_size', [0, 0])
                }

            # Find dominant category (highest coverage percentage)
            dominant_category = max(category_stats.keys(), key=lambda x: category_stats[x]['coverage_percentage'])
            dominant_confidence = category_stats[dominant_category]['average_confidence']

            return {
                "predicted_category": dominant_category,
                "confidence": dominant_confidence,
                "segmentation_results": segmentation_results,
                "detected_objects": detected_objects,
                "hierarchical_regions": hierarchical_regions,
                "total_classes_found": results['total_classes_found'],
                "total_categories_found": results['total_categories_found'],
                "coverage_stats": coverage_stats,
                "category_stats": category_stats,
                "image_size": results['image_size']
            }

        except Exception as e:
            error_msg = f"Classification failed: {e}"
            self._logger.error(error_msg)
            raise RuntimeError(f"Classification tool execution failed: {error_msg}")

    def _determine_hierarchical_categories(self, category_stats: Dict[str, Dict]) -> Tuple[str, str, float, float]:
        """
        Determine primary and secondary scene categories based on coverage percentages.

        Args:
            category_stats: Dictionary with category names as keys and coverage statistics as values

        Returns:
            Tuple of (primary_category, secondary_category, primary_coverage, secondary_coverage)
            secondary_category is None if no category has >20% coverage
        """
        if not category_stats:
            return "Water", None, 0.0, 0.0

        # Sort categories by coverage percentage (highest first)
        sorted_categories = sorted(
            category_stats.items(),
            key=lambda x: x[1].get('coverage_percentage', 0.0),
            reverse=True
        )

        # Primary category (highest coverage)
        primary_category = sorted_categories[0][0]
        primary_coverage = sorted_categories[0][1].get('coverage_percentage', 0.0)

        # Secondary category (if coverage >20% and different from primary)
        secondary_category = None
        secondary_coverage = 0.0

        if len(sorted_categories) > 1:
            second_coverage = sorted_categories[1][1].get('coverage_percentage', 0.0)
            if second_coverage > 20.0:  # Significant coverage threshold
                secondary_category = sorted_categories[1][0]
                secondary_coverage = second_coverage

        return primary_category, secondary_category, primary_coverage, secondary_coverage

    def _ensure_json_serializable(self, data):
        """
        Recursively convert numpy arrays and other non-serializable objects to JSON-serializable formats.

        Args:
            data: Data structure that may contain numpy arrays

        Returns:
            JSON-serializable version of the data
        """
        import numpy as np

        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, dict):
            return {key: self._ensure_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._ensure_json_serializable(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._ensure_json_serializable(item) for item in data)
        else:
            return data

    # Note: _parse_target_classes_from_prompt method removed since we now use
    # fixed land cover classification that analyzes all LoveDA classes regardless of query

    def _match_text_to_hierarchical_category(self, text_prompt: str) -> str:
        """
        Match text prompt to the most relevant scene category.

        Args:
            text_prompt: Natural language description

        Returns:
            Most relevant scene category name
        """
        text_lower = text_prompt.lower()

        # Direct category name matching
        for category in HIERARCHICAL_CATEGORIES.values():
            if category.lower() in text_lower:
                return category

        # Alias matching for categories
        for category, aliases in CATEGORY_ALIASES.items():
            for alias in aliases:
                if alias in text_lower:
                    return category

        # Check for specific class mentions and map to category
        for class_name, category in CLASS_TO_CATEGORY_MAPPING.items():
            if class_name in text_lower:
                return category

        # Default to general classification if no specific match
        return "Natural"

    def _run(
        self,
        image_path: str,
        text_prompt: str,
        confidence_threshold: Optional[float] = 0.5,  # Fixed: Use proper classification confidence threshold from config
        meters_per_pixel: Optional[float] = 0.3,  # Correct: This is spatial resolution (meters per pixel)
        classes_requested: Optional[List[str]] = None,  # Classes requested in the tool call (used to filter which categories to classify)
    ) -> str:
        """
        Execute the RemoteSAM scene classification tool.

        Args:
            image_path: Path to the input image
            text_prompt: Natural language description (comma-separated list of scene categories to classify)
            confidence_threshold: Minimum confidence threshold for region detection
            meters_per_pixel: Ground resolution in meters per pixel (for metadata)
            classes_requested: List of scene categories requested in the tool call (filters which categories to classify)

        Returns:
            JSON string containing scene classification results for the requested scene categories
        """
        try:
            # Check if model is available
            if self._classifier is None:
                raise RuntimeError("RemoteSAM scene classifier not initialized")

            # Determine which scene categories to classify based on text_prompt or classes_requested
            # Parse text_prompt to extract requested scene categories
            requested_categories = []
            if classes_requested:
                # Use classes_requested if provided (from LLM or generate_gt_all.py)
                requested_categories = classes_requested
            elif text_prompt:
                # Parse text_prompt to extract category names
                # Split by comma and strip whitespace
                requested_categories = [cat.strip() for cat in text_prompt.split(",")]

            # If no categories specified, default to all 6 hierarchical categories
            if not requested_categories:
                requested_categories = sorted(list(HIERARCHICAL_CATEGORIES.values()))
                print(f"   ℹ️  No specific categories requested, classifying all 6 scene categories")
            else:
                print(f"   🎯 Requested scene categories: {requested_categories}")

            # Normalize requested categories to match HIERARCHICAL_CATEGORIES values
            # Handle both "Agriculture land" and "agriculture_land" formats
            normalized_requested = set()
            for cat in requested_categories:
                cat_normalized = cat.strip()
                # Try exact match first
                if cat_normalized in HIERARCHICAL_CATEGORIES.values():
                    normalized_requested.add(cat_normalized)
                else:
                    # Try case-insensitive match with space/underscore flexibility
                    cat_lower = cat_normalized.lower().replace("_", " ")
                    for hierarchical_cat in HIERARCHICAL_CATEGORIES.values():
                        if hierarchical_cat.lower() == cat_lower:
                            normalized_requested.add(hierarchical_cat)
                            break

            # If no valid categories found, default to all
            if not normalized_requested:
                print(f"   ⚠️  No valid scene categories found in request, defaulting to all 6 categories")
                normalized_requested = set(HIERARCHICAL_CATEGORIES.values())

            # Scene classification: Always analyze all 10 fundamental land-cover classes
            # Then filter results to only the requested scene categories
            remote_sensing_classes = REMOTE_SENSING_CLASSES

            print(f"   🎯 Scene classification approach: Analyzing all {len(remote_sensing_classes)} land-cover classes")
            print(f"   📋 Land-cover classes: {remote_sensing_classes}")
            print(f"   🏗️  Filtering to {len(normalized_requested)} requested scene categories: {sorted(normalized_requested)}")
            print(f"   ℹ️  Note: Classification maps land-cover classes to scene categories")

            # Perform comprehensive scene classification for all land-cover classes
            classification_result = self._classify_image_hierarchical(image_path, remote_sensing_classes)

            # Extract detected objects and scene regions
            detected_objects = classification_result["detected_objects"]
            hierarchical_regions = classification_result["hierarchical_regions"]

            # Filter objects by confidence threshold
            filtered_objects = [obj for obj in detected_objects
                              if obj["confidence"] >= confidence_threshold]

            # Handle case where no objects meet threshold
            if not filtered_objects and detected_objects:
                print(f"⚠️  No objects meet confidence threshold {confidence_threshold}")
                print(f"   Found {len(detected_objects)} objects with lower confidence")
                print(f"   Proceeding with all detected objects for data generation purposes")
                filtered_objects = detected_objects

            # Match text prompt to expected category (for context)
            expected_category = self._match_text_to_hierarchical_category(text_prompt)

            # Use filtered objects as detections for compatibility with spatial analysis pipeline
            # Ensure all data is JSON-serializable
            detections = self._ensure_json_serializable(filtered_objects)

            # Calculate overall statistics
            total_objects = len(filtered_objects)
            total_categories = len(hierarchical_regions)
            meets_threshold = total_objects > 0

            # Implement scene category-based classification logic
            coverage_stats = classification_result.get("coverage_stats", {})
            category_stats = classification_result.get("category_stats", {})

            # Determine primary and secondary categories based on coverage percentages
            primary_category, secondary_category, primary_confidence, secondary_confidence = self._determine_hierarchical_categories(category_stats)

            # Display ONLY requested categories (if classes_requested is provided)
            # Otherwise display all detected categories
            print(f"   📊 Scene classification results:")

            if classes_requested:
                # Filter to only show requested categories
                # Normalize requested class names to match category_stats keys
                requested_categories_normalized = {
                    cls.lower().replace("_", " "): cls for cls in classes_requested
                }

                categories_displayed = 0
                for category_name, stats in category_stats.items():
                    # Check if this category matches any requested class (case-insensitive, space/underscore flexible)
                    category_normalized = category_name.lower()
                    if category_normalized in requested_categories_normalized:
                        coverage_pct = stats.get('coverage_percentage', 0.0)
                        original_requested_name = requested_categories_normalized[category_normalized]
                        print(f"      - {category_name}: {coverage_pct:.1f}% coverage (requested as '{original_requested_name}')")
                        categories_displayed += 1

                # Check if any requested classes were not found
                found_categories = {cat.lower() for cat in category_stats.keys()}
                missing_categories = [
                    cls for cls in classes_requested
                    if cls.lower().replace("_", " ") not in found_categories
                ]

                if missing_categories:
                    print(f"      ⚠️  Requested but not detected: {missing_categories}")

                if categories_displayed == 0:
                    print(f"      - No requested categories detected")
            else:
                # Display all categories sorted by coverage percentage (highest first)
                all_categories_sorted = sorted(
                    category_stats.items(),
                    key=lambda x: x[1].get('coverage_percentage', 0.0),
                    reverse=True
                )

                # Display all categories with non-zero coverage
                categories_displayed = 0
                for idx, (category_name, stats) in enumerate(all_categories_sorted):
                    coverage_pct = stats.get('coverage_percentage', 0.0)
                    if coverage_pct > 0.0:
                        if idx == 0:
                            print(f"      - Primary category: {category_name} ({coverage_pct:.1f}% coverage)")
                        else:
                            print(f"      - Category {idx + 1}: {category_name} ({coverage_pct:.1f}% coverage)")
                        categories_displayed += 1

                if categories_displayed == 0:
                    print(f"      - No categories detected")

            # Use primary category as dominant
            dominant_category = primary_category
            dominant_confidence = primary_confidence / 100.0  # Convert percentage to 0-1 scale

            # Safely extract results with proper error handling and JSON serialization
            segmentation_results_raw = classification_result.get("segmentation_results", {})
            coverage_stats = classification_result.get("coverage_stats", {})
            category_stats = classification_result.get("category_stats", {})
            total_classes_found = classification_result.get("total_classes_found", 0)
            total_categories_found = classification_result.get("total_categories_found", 0)
            image_size = classification_result.get("image_size", [0, 0])

            # Convert segmentation results to JSON-serializable format (remove numpy arrays)
            segmentation_results = {}
            for class_name, class_data in segmentation_results_raw.items():
                segmentation_results[class_name] = {
                    'regions': class_data.get('regions', []),
                    'total_pixels': class_data.get('total_pixels', 0),
                    'coverage_percentage': class_data.get('coverage_percentage', 0.0)
                    # Note: Excluding 'mask' field as it contains numpy arrays
                }



            # Prepare results with scene classification approach
            result = {
                "success": True,
                "image_path": image_path,
                "text_prompt": text_prompt,
                "total_detections": total_objects,
                "detections": detections,
                "expected_category": expected_category,  # From query (for context)
                "predicted_category": dominant_category,  # Primary scene category
                "confidence": round(dominant_confidence, 4),
                "meets_threshold": meets_threshold,
                "confidence_threshold": confidence_threshold,
                "segmentation_results": segmentation_results,
                "hierarchical_regions": self._ensure_json_serializable(hierarchical_regions),
                "coverage_stats": coverage_stats,
                "category_stats": category_stats,
                "total_classes_found": total_classes_found,
                "total_categories_found": total_categories_found,
                "analyzed_classes": remote_sensing_classes,  # All 10 land-cover classes
                "image_size": image_size,
                "scene_classification": {
                    "primary_category": primary_category,
                    "primary_coverage": round(primary_confidence, 2),
                    "secondary_category": secondary_category,
                    "secondary_coverage": round(secondary_confidence, 2) if secondary_category else 0.0,
                    "approach": "scene_based_classification",
                    "categories": list(HIERARCHICAL_CATEGORIES.values()),
                    "classes_analyzed": remote_sensing_classes
                },
                "model_info": {
                    "model_type": "RemoteSAM-Scene-Classification",
                    "approach": "scene-based-classification",
                    "num_classes": len(remote_sensing_classes),
                    "num_categories": len(HIERARCHICAL_CATEGORIES),
                    "classes": remote_sensing_classes,
                    "categories": list(HIERARCHICAL_CATEGORIES.values())
                },
                "summary": f"Scene classification: Analyzed {len(remote_sensing_classes)} land-cover classes mapped to {len(HIERARCHICAL_CATEGORIES)} scene categories. Primary: {primary_category} ({primary_confidence:.1f}%), Secondary: {secondary_category or 'None'} ({secondary_confidence:.1f}% if secondary_category else 0.0%). Total objects: {total_objects}, Categories: {total_categories}."
            }

            # Add resolution info if provided
            if meters_per_pixel:
                result["meters_per_pixel"] = meters_per_pixel

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in detection and segmentation tools for consistency

            # Use only the requested categories (from normalized_requested set)
            # Sort for consistent output
            classes_for_response = sorted(list(normalized_requested))

            # Calculate class_stats: coverage percentage for ONLY the requested categories
            # Note: Need to handle format mismatch between classes_for_response (may be lowercase with underscores)
            # and category_stats keys (title case with spaces from HIERARCHICAL_CATEGORIES)
            class_stats = {}

            # Create a mapping from normalized names to actual category_stats keys
            category_stats_normalized = {
                cat.lower().replace(" ", "_"): cat
                for cat in category_stats.keys()
            }

            for cls in classes_for_response:
                # Normalize the requested class name for lookup
                cls_normalized = cls.lower().replace(" ", "_")

                # Check if this normalized class exists in category_stats
                if cls_normalized in category_stats_normalized:
                    # Get the actual category name from category_stats
                    actual_category = category_stats_normalized[cls_normalized]
                    coverage_pct = round(category_stats[actual_category].get('coverage_percentage', 0.0), 2)
                    # Use the actual category name (title case) as the key in class_stats
                    class_stats[actual_category] = {"coverage_pct": coverage_pct}
                else:
                    # Class was requested but not detected in the image - include it with 0% coverage
                    # Use the original requested class name (preserve format from request)
                    class_stats[cls] = {"coverage_pct": 0.0}

            # Build the unified arguments field
            arguments = {
                "image_path": image_path,
                "classes_requested": classes_for_response,  # Only the requested categories
                "meters_per_pixel": meters_per_pixel if meters_per_pixel else None,
                "text_prompt": text_prompt,
                "predicted_class": dominant_category,
                "class_stats": class_stats
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            # Note: Results will be saved by TempStorageManager to image-specific directories
            # No longer creating global temp/classification directory

            # Return structured data
            return json.dumps(result, indent=2)

        except Exception as e:
            error_msg = f"RemoteSAM scene classification failed: {str(e)}"
            print(f"   ❌ Scene classification tool error: {error_msg}")

            # Provide detailed error information for debugging
            import traceback
            traceback.print_exc()

            error_result = {
                "success": False,
                "error": error_msg,
                "image_path": image_path,
                "text_prompt": text_prompt,
                "analyzed_classes": remote_sensing_classes if 'remote_sensing_classes' in locals() else REMOTE_SENSING_CLASSES,
                "summary": f"Scene classification failed for '{text_prompt}': {error_msg}"
            }
            # Return structured error data
            return json.dumps(error_result, indent=2)


def create_classification_tool(device: str = "auto") -> RemoteSAMClassificationTool:
    """
    Factory function to create a RemoteSAM scene classification tool.

    Args:
        device: Device to run the model on ("auto" for automatic GPU selection, 'cuda:0', 'cpu')

    Returns:
        Configured RemoteSAMClassificationTool instance with scene classification capabilities
    """
    if device == "auto":
        # Check for hardcoded GPU assignment
        import os
        if os.getenv('SPATIAL_REASONING_GPU_MODE') == 'hardcoded':
            perception_gpu = os.getenv('SPATIAL_REASONING_PERCEPTION_GPU', '2')
            device = f"cuda:{perception_gpu}"
            print(f"🎯 Classification tool using hardcoded GPU assignment: {device}")
        else:
            # Use hardcoded GPU 0 for all perception tools (single GPU system)
            device = "cuda:0"
            print(f"🎯 Classification tool using hardcoded GPU assignment: {device}")

    return RemoteSAMClassificationTool(device=device)
