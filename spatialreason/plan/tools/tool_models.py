"""
Tool data models for the spatial reasoning agent.

This module contains the core data models for representing tools, toolkits,
and toolkit collections without external dependencies.
"""

from typing import List, Optional


class Tool:
    """Tool representation without embedding dependencies."""

    def __init__(self, name: str, category: str, description: str, dependencies: Optional[List[str]] = None):
        self.api_dest = {
            "type_name": category,
            "name": name,
            "desc": description,
            "package_name": f"{category}_{name}"
        }

        # Define proper parameter schemas based on tool type
        self.api_doc = self._get_tool_parameters(name, category)

        # Store tool dependencies (e.g., ["perception"] for spatial relation tools)
        # This is used by DependencyValidator to enforce tool execution order
        self.dependencies = dependencies or []

    def _get_tool_parameters(self, name: str, category: str) -> dict:
        """Get proper parameter definitions for each tool type."""

        # Special parameters for change detection tool
        if name == "change_detection":
            required_params = [
                {"name": "image_path_t1", "type": "STRING", "description": "Path to pre-change (T1) image file", "default": ""},
                {"name": "image_path_t2", "type": "STRING", "description": "Path to post-change (T2) image file", "default": ""}
            ]
            optional_params = [
                {"name": "num_classes", "type": "NUMBER", "description": "Number of semantic classes (default: 6)", "default": "6"},
                {"name": "confidence_threshold", "type": "NUMBER", "description": "Confidence threshold for change detection (0.0-1.0)", "default": "0.5"},
                {"name": "text_prompt", "type": "STRING", "description": "Task intent description", "default": "detect semantic changes"},
                {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "null"}
            ]
            return {
                "required_parameters": required_params,
                "optional_parameters": optional_params
            }

        # Common parameters for perception tools
        if category == "perception" or name in ["segmentation", "detection", "classification"]:
            required_params = [
                {"name": "image_path", "type": "STRING", "description": "Path to image file (PNG or JPG)", "default": ""},
                {"name": "text_prompt", "type": "STRING", "description": "Comma-separated list of class names to detect/segment/classify (e.g., 'building, road' or 'Agriculture land', 'Rangeland')", "default": ""}
            ]
            optional_params = []

            # Add classes_requested for all perception tools
            # This is a list of class names extracted from text_prompt
            if name in ["segmentation", "detection", "classification"]:
                optional_params.append({
                    "name": "classes_requested",
                    "type": "ARRAY",
                    "description": "List of class names to detect/segment/classify. Should match the classes mentioned in text_prompt (e.g., ['building', 'road'] or ['Agriculture land', 'Rangeland'])",
                    "default": "[]"
                })

            # Add meters_per_pixel only for detection and classification tools, not segmentation
            if name in ["detection", "classification"]:
                optional_params.append({"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"})

            # Add confidence_threshold for all perception tools (segmentation, detection, classification)
            if name in ["segmentation", "detection", "classification"]:
                optional_params.append({
                    "name": "confidence_threshold",
                    "type": "NUMBER",
                    "description": "Minimum confidence threshold as a decimal value between 0.0 and 1.0 (e.g., 0.5 for 50% confidence)",
                    "default": "0.5" if name in ["segmentation", "classification"] else "0.6"
                })

            return {
                "required_parameters": required_params,
                "optional_parameters": optional_params
            }

        # Parameters for spatial relation tools
        elif category == "spatial_relations" or name in ["buffer", "overlap", "containment"]:
            if name == "buffer":
                # Buffer tool specific parameters - updated to match actual buffer tool interface
                required_params = [
                    {"name": "buffer_distance_meters", "type": "NUMBER", "description": "Buffer distance in meters", "default": "30.0"},
                    {"name": "image_path", "type": "STRING", "description": "Path to input image", "default": ""}
                ]
                optional_params = [
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"},
                    {"name": "query_text", "type": "STRING", "description": "Query text for role assignment", "default": ""}
                ]
                optional_params = [
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"},
                    {"name": "geometry_type", "type": "STRING", "description": "Type of input geometry (points, polygons, bboxes)", "default": "bboxes"}
                ]
            elif name == "overlap":
                # Overlap tool specific parameters - CRITICAL: Use source_class and target_class to match benchmark format
                required_params = [
                    {"name": "source_class", "type": "STRING", "description": "Name of the first class to compare (e.g., 'bridge')", "default": ""},
                    {"name": "target_class", "type": "STRING", "description": "Name of the second class to compare (e.g., 'harbor')", "default": ""}
                ]
                optional_params = [
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"},
                    {"name": "source_polygon_count", "type": "NUMBER", "description": "Number of source class polygons", "default": ""},
                    {"name": "target_polygon_count", "type": "NUMBER", "description": "Number of target class polygons", "default": ""}
                ]
            elif name == "containment":
                # Containment tool specific parameters - semantic interface matching benchmark format
                # The tool accepts class names and internally extracts geometries from perception results
                required_params = []
                optional_params = [
                    {"name": "container_class", "type": "STRING", "description": "Name of the container class (e.g., 'water', 'forest')", "default": ""},
                    {"name": "contained_class", "type": "STRING", "description": "Name of the contained class (e.g., 'building', 'cars')", "default": ""},
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"},
                    {"name": "threshold_pct", "type": "NUMBER", "description": "Optional containment threshold percentage (0-100)", "default": ""}
                ]
            elif name == "area_measurement":
                # Area measurement tool specific parameters
                required_params = []
                optional_params = [
                    {"name": "area_class", "type": "STRING", "description": "Name of the class to measure area for (e.g., 'water', 'forest', 'agriculture')", "default": ""},
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"}
                ]
            elif name == "distance_calculation":
                # Distance calculation tool specific parameters
                required_params = []
                optional_params = [
                    {"name": "set_a_class", "type": "STRING", "description": "Name of the first class for distance calculation (e.g., 'building', 'road')", "default": ""},
                    {"name": "set_b_class", "type": "STRING", "description": "Name of the second class for distance calculation (e.g., 'water', 'forest')", "default": ""},
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"}
                ]
            else:
                # Generic spatial relation parameters (fallback)
                required_params = [
                    {"name": "geometry1", "type": "STRING", "description": "First geometry (JSON format)", "default": ""},
                    {"name": "geometry2", "type": "STRING", "description": "Second geometry (JSON format)", "default": ""}
                ]
                optional_params = [
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"}
                ]

            return {
                "required_parameters": required_params,
                "optional_parameters": optional_params
            }

        # Parameters for spatial statistics tools
        elif category == "spatial_statistics":
            if name == "object_count_aoi":
                # object_count_aoi specific parameters
                required_params = []
                optional_params = [
                    {"name": "object_class", "type": "STRING", "description": "Class name of objects to count (e.g., 'bridge', 'target')", "default": ""},
                    {"name": "aoi_class", "type": "STRING", "description": "Class name of AOI region or special value ('query_region', 'full_image')", "default": ""},
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel (null for IR dataset)", "default": "0.3"}
                ]
            else:
                # Generic spatial statistics parameters
                required_params = [
                    {"name": "geometry", "type": "STRING", "description": "Geometry data (JSON format)", "default": ""}
                ]
                optional_params = [
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel", "default": "0.3"}
                ]

            return {
                "required_parameters": required_params,
                "optional_parameters": optional_params
            }

        # Parameters for IR tools
        elif category == "ir_tools" or name == "infrared_detection":
            required_params = [
                {"name": "image_path", "type": "STRING", "description": "Path to infrared image file (PNG, JPG, or BMP)", "default": ""}
            ]
            optional_params = [
                {"name": "confidence_threshold", "type": "NUMBER", "description": "Detection confidence threshold as a decimal value between 0.0 and 1.0 (e.g., 0.5 for 50% confidence)", "default": "0.5"},
                {"name": "nms_iou_threshold", "type": "NUMBER", "description": "Non-maximum suppression IoU threshold as a decimal value between 0.0 and 1.0", "default": "0.3"}
            ]

            return {
                "required_parameters": required_params,
                "optional_parameters": optional_params
            }

        # Parameters for SAR tools
        elif category == "sar_tools" or name in ["sar_detection", "sar_classification"]:
            if name == "sar_detection":
                required_params = [
                    {"name": "image_path", "type": "STRING", "description": "Path to SAR image file", "default": ""}
                ]
                optional_params = [
                    {"name": "confidence_threshold", "type": "NUMBER", "description": "Detection confidence threshold as a decimal value between 0.0 and 1.0 (e.g., 0.3 for 30% confidence)", "default": "0.3"},
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel (3.0 for OGSOD dataset)", "default": "3.0"}
                ]
            else:  # sar_classification
                required_params = [
                    {"name": "image_path", "type": "STRING", "description": "Path to SAR image file", "default": ""}
                ]
                optional_params = [
                    {"name": "confidence_threshold", "type": "NUMBER", "description": "Classification confidence threshold as a decimal value between 0.0 and 1.0 (e.g., 0.5 for 50% confidence)", "default": "0.5"},
                    {"name": "meters_per_pixel", "type": "NUMBER", "description": "Ground resolution in meters per pixel (3.0 for OGSOD dataset)", "default": "3.0"}
                ]

            return {
                "required_parameters": required_params,
                "optional_parameters": optional_params
            }

        # Fallback for unknown tools
        else:
            return {
                "required_parameters": [
                    {"name": "input", "type": "STRING", "description": "Input data", "default": ""}
                ],
                "optional_parameters": [
                    {"name": "threshold", "type": "NUMBER", "description": "Processing threshold", "default": "0.5"}
                ]
            }


class Toolkit:
    """Toolkit representation without embedding dependencies."""

    def __init__(self, name: str, tools: List[Tool]):
        self.name = name
        self.tool_lists = tools

    def toolkit_exp(self) -> str:
        """Generate toolkit description."""
        tool_names = [tool.api_dest["name"] for tool in self.tool_lists]
        return f"Toolkit '{self.name}': Available tools - {', '.join(tool_names)}. " \
               f"This toolkit provides {len(self.tool_lists)} tools for spatial analysis and processing."


class ToolkitList:
    """Toolkit list without embedding dependencies."""

    def __init__(self, toolkit_num: int = 1):
        self.toolkit_num = toolkit_num
        self.tool_kits = []
        self._create_default_toolkits()

    def _create_default_toolkits(self):
        """Create default toolkits using ToolRegistry."""
        # Import here to avoid circular imports
        from .tool_registry import ToolRegistry

        registry = ToolRegistry()
        toolkit_list = registry.create_default_toolkits(self.toolkit_num)
        self.tool_kits = toolkit_list.tool_kits
