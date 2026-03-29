"""
Unified tool interface that consolidates tool definitions, metadata, and execution logic.
Eliminates duplication between chat_model.py, step_executor.py, and configuration files.
"""

from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import re


class ToolInterface(ABC):
    """
    Abstract interface for all tools in the spatial reasoning agent.
    Provides unified access to tool metadata, definitions, and execution.
    """
    
    def __init__(self, tool_id: str):
        """
        Initialize tool interface.

        Args:
            tool_id: Unique identifier for the tool
        """
        self.tool_id = tool_id
        self.metadata = self._load_tool_metadata()

    def _load_tool_metadata(self) -> Dict[str, Any]:
        """Load tool metadata from hardcoded definitions."""
        # Simplified metadata without config manager
        metadata_map = {
            "segmentation": {
                "tool_id": "segmentation",
                "name": "RemoteSAM Segmentation Tool",
                "description": "Segment objects in satellite imagery using text prompts",
                "keywords": ["segment", "segmentation", "water", "building", "objects"],
                "use_cases": [
                    "segment buildings in urban areas",
                    "segment water bodies and rivers",
                    "segment vegetation and forests",
                    "segment roads and infrastructure",
                    "delineate object boundaries"
                ]
            },
            "detection": {
                "tool_id": "detection",
                "name": "RemoteSAM Detection Tool",
                "description": "Detect and locate objects in satellite imagery",
                "keywords": ["detect", "detection", "locate", "find", "objects", "building", "water"],
                "use_cases": [
                    "detect buildings and structures",
                    "find water bodies and lakes",
                    "locate vehicles and aircraft",
                    "identify infrastructure elements",
                    "count objects in images"
                ]
            },
            "classification": {
                "tool_id": "classification",
                "name": "RemoteSAM Classification Tool",
                "description": "Classify objects and regions in satellite imagery using 25 remote sensing categories",
                "keywords": ["classify", "classification", "identify", "type", "objects", "infrastructure"],
                "use_cases": [
                    "classify infrastructure objects (ships, storage tanks, bridges, vehicles)",
                    "identify sports facilities (baseball diamonds, tennis courts, basketball courts)",
                    "categorize transportation infrastructure (roads, roundabouts, harbors)",
                    "analyze land cover types (water, forest, agriculture, barren land)",
                    "detect aircraft and vehicles (planes, helicopters, cars)"
                ]
            },
            "area_measurement": {
                "tool_id": "area_measurement",
                "name": "Area Measurement Tool",
                "description": "Measure areas of polygonal geometries",
                "keywords": ["area", "measure", "size", "coverage"],
                "use_cases": [
                    "measure building footprint areas",
                    "calculate water body coverage",
                    "land use area analysis"
                ]
            },

            "object_count_aoi": {
                "tool_id": "object_count_aoi",
                "name": "Object Count in AOI Tool",
                "description": "Count objects within areas of interest",
                "keywords": ["count", "objects", "aoi", "statistics"],
                "use_cases": [
                    "count buildings in flood zones",
                    "count objects within boundaries",
                    "spatial object statistics"
                ]
            },
            "buffer": {
                "tool_id": "buffer",
                "name": "Buffer Tool",
                "description": "Create buffer zones around geometries",
                "keywords": ["buffer", "zone", "proximity", "distance"],
                "use_cases": [
                    "create buffer zones around water bodies",
                    "analyze proximity to infrastructure",
                    "flood risk zone creation"
                ]
            },
            "distance_calculation": {
                "tool_id": "distance_calculation",
                "name": "Distance Calculation Tool",
                "description": "Calculate distances between geometries",
                "keywords": ["distance", "measure", "proximity", "separation"],
                "use_cases": [
                    "measure distance between buildings and water",
                    "calculate proximity to infrastructure",
                    "spatial relationship analysis"
                ]
            },
            "overlap": {
                "tool_id": "overlap",
                "name": "Overlap Tool",
                "description": "Calculate overlap ratios between geometries",
                "keywords": ["overlap", "intersection", "ratio", "coverage"],
                "use_cases": [
                    "calculate building overlap with flood zones",
                    "analyze land use overlap",
                    "intersection analysis"
                ]
            },
            "containment": {
                "tool_id": "containment",
                "name": "Containment Tool",
                "description": "Analyze containment relationships between geometries",
                "keywords": ["containment", "within", "inside", "boundary"],
                "use_cases": [
                    "check if buildings are within flood zones",
                    "analyze objects within boundaries",
                    "spatial containment analysis"
                ]
            },
            "sar_detection": {
                "tool_id": "sar_detection",
                "name": "SAR Detection Tool",
                "description": "Detect objects in SAR (Synthetic Aperture Radar) imagery using HiViT-based SARATR-X model",
                "keywords": ["sar", "radar", "detection", "ship", "vessel", "maritime", "synthetic aperture radar"],
                "use_cases": [
                    "detect ships in SAR maritime imagery",
                    "locate vessels in radar images",
                    "maritime surveillance and monitoring",
                    "SAR-based object detection",
                    "radar image analysis"
                ]
            },
            "sar_classification": {
                "tool_id": "sar_classification",
                "name": "SAR Classification Tool",
                "description": "Classify SAR imagery into scene types, target categories, or fine-grained classes using SARATR-X model",
                "keywords": ["sar", "radar", "classification", "scene", "target", "fine-grained", "synthetic aperture radar"],
                "use_cases": [
                    "classify SAR scenes (urban, rural, coastal, etc.)",
                    "identify target types in radar images",
                    "fine-grained classification of military vehicles",
                    "SAR-based scene understanding",
                    "radar image categorization"
                ]
            }
        }

        if self.tool_id in metadata_map:
            return metadata_map[self.tool_id]

        raise ValueError(f"Tool metadata not found for {self.tool_id}")
    
    @property
    def name(self) -> str:
        """Get tool name."""
        return self.metadata["name"]
    
    @property
    def description(self) -> str:
        """Get tool description."""
        return self.metadata["description"]
    
    @property
    def category(self) -> str:
        """Get tool category."""
        # The tool_id is the category (segmentation, detection, classification)
        return self.tool_id
    
    @property
    def keywords(self) -> List[str]:
        """Get tool keywords for selection."""
        return self.metadata.get("keywords", [])
    
    @property
    def use_cases(self) -> List[str]:
        """Get tool use cases."""
        return self.metadata.get("use_cases", [])
    
    def get_langchain_definition(self) -> Dict[str, Any]:
        """
        Get LangChain-compatible tool definition.
        Replaces hardcoded definitions in chat_model.py.
        """
        base_definition = {
            "type": "function",
            "function": {
                "name": self.tool_id,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self._get_parameter_schema(),
                    "required": self._get_required_parameters()
                }
            }
        }
        
        return base_definition
    
    @abstractmethod
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for the tool. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_required_parameters(self) -> List[str]:
        """Get required parameters for the tool. Must be implemented by subclasses."""
        pass
    
    def extract_text_prompt(self, user_input: str, context: Optional[str] = None) -> str:
        """
        Extract appropriate text prompt from user input.
        Consolidates logic from chat_model.py and step_executor.py.
        """
        input_lower = user_input.lower()
        context_lower = (context or "").lower()
        combined_text = f"{input_lower} {context_lower}".strip()
        
        # Use tool-specific keywords and use cases for better extraction
        for use_case in self.use_cases:
            use_case_words = set(use_case.lower().split())
            input_words = set(combined_text.split())
            
            # If significant overlap with use case, extract relevant terms
            if len(use_case_words.intersection(input_words)) >= 2:
                return self._extract_from_use_case(use_case, combined_text)
        
        # Fallback to keyword-based extraction
        return self._extract_from_keywords(combined_text)
    
    def _extract_from_use_case(self, use_case: str, input_text: str) -> str:
        """Extract text prompt based on matching use case."""
        use_case_lower = use_case.lower()
        
        # Extract object mentions from use case
        if "building" in use_case_lower:
            return "buildings"
        elif "water" in use_case_lower:
            return "water bodies"
        elif "vegetation" in use_case_lower or "forest" in use_case_lower:
            return "vegetation"
        elif "road" in use_case_lower:
            return "roads"
        elif "land cover" in use_case_lower:
            return "buildings, water, vegetation"
        
        # Default based on category
        return self._get_default_prompt()
    
    def _extract_from_keywords(self, input_text: str) -> str:
        """Extract text prompt using keyword matching."""
        # Look for specific object mentions
        if any(word in input_text for word in ["building", "structure", "house"]):
            return "buildings"
        elif any(word in input_text for word in ["water", "river", "lake", "pond"]):
            return "water bodies"
        elif any(word in input_text for word in ["vegetation", "forest", "tree", "plant"]):
            return "vegetation"
        elif any(word in input_text for word in ["road", "highway", "street"]):
            return "roads"
        elif any(word in input_text for word in ["land cover", "land use"]):
            return "buildings, water, vegetation"
        
        return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default text prompt based on tool category."""
        if self.category == "segmentation":
            return "buildings"
        elif self.category == "detection":
            return "objects"
        elif self.category == "classification":
            return "remote sensing objects and land cover"
        else:
            return "objects"
    
    def matches_query(self, query: str, threshold: float = 0.3) -> float:
        """
        Calculate how well this tool matches a query.
        Returns confidence score between 0.0 and 1.0.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Check keyword matches - improved scoring
        keyword_matches = sum(1 for keyword in self.keywords if keyword in query_lower)
        keyword_score = min(keyword_matches / max(len(self.keywords), 1), 1.0) if self.keywords else 0.0

        # Boost score if any primary keywords match
        primary_keywords = self.keywords[:2] if len(self.keywords) >= 2 else self.keywords
        primary_matches = sum(1 for keyword in primary_keywords if keyword in query_lower)
        if primary_matches > 0:
            keyword_score = max(keyword_score, 0.5)  # Minimum 0.5 if primary keyword matches

        # Check use case matches
        use_case_score = 0.0
        for use_case in self.use_cases:
            use_case_words = set(use_case.lower().split())
            overlap = len(query_words.intersection(use_case_words))
            case_score = overlap / max(len(use_case_words), 1)
            use_case_score = max(use_case_score, case_score)

        # Combined score with better weighting
        if use_case_score > 0:
            combined_score = (keyword_score * 0.3) + (use_case_score * 0.7)
        else:
            combined_score = keyword_score  # Use only keyword score if no use cases match

        return combined_score


class SpatialToolInterface(ToolInterface):
    """
    Specialized interface for spatial analysis tools (spatial_relations and spatial_statistics).
    Provides basic parameter schema for geometric operations.
    """

    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get common parameter schema for spatial analysis tools."""
        schema = {
            "image_path": {
                "type": "string",
                "description": "Path to input satellite image"
            },
            "meters_per_pixel": {
                "type": "number",
                "description": "Ground resolution in meters per pixel",
                "default": 0.3
            }
        }

        # Add tool-specific parameters based on tool_id
        if self.tool_id == "buffer":
            schema.update({
                "buffer_distance_meters": {
                    "type": "number",
                    "description": "Buffer distance in meters"
                },
                "meters_per_pixel": {
                    "type": "number",
                    "description": "Ground resolution in meters per pixel"
                },
                "query_text": {
                    "type": "string",
                    "description": "Query text for role assignment"
                }
            })
        elif self.tool_id == "distance_calculation":
            schema.update({
                "geometry_set_1": {
                    "type": "array",
                    "description": "First set of geometries"
                },
                "geometry_set_2": {
                    "type": "array",
                    "description": "Second set of geometries"
                }
            })
        elif self.tool_id == "area_measurement":
            schema.update({
                "polygons": {
                    "type": "array",
                    "description": "List of polygon coordinates"
                }
            })

        elif self.tool_id == "object_count_aoi":
            schema.update({
                "object_geometries": {
                    "type": "array",
                    "description": "List of object geometries to count"
                },
                "aoi_geometries": {
                    "type": "array",
                    "description": "Areas of interest geometries"
                }
            })

        return schema

    def _get_required_parameters(self) -> List[str]:
        """Get required parameters for spatial analysis tools."""
        base_params = ["image_path"]

        if self.tool_id == "buffer":
            return base_params + ["buffer_distance_meters"]
        elif self.tool_id == "distance_calculation":
            return base_params + ["geometry_set_1", "geometry_set_2"]
        elif self.tool_id == "area_measurement":
            return base_params + ["polygons"]

        elif self.tool_id == "object_count_aoi":
            return base_params + ["object_geometries", "aoi_geometries"]
        else:
            return base_params

    def create_tool_args(self,
                        image_path: str,
                        user_input: str,
                        context: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Create tool arguments for spatial analysis tools.
        Enhanced to match benchmark_mini.json argument structures.
        """
        if self.tool_id == "buffer":
            return self._create_buffer_args(image_path, user_input, context, **kwargs)
        elif self.tool_id == "overlap":
            return self._create_overlap_args(image_path, user_input, context, **kwargs)
        elif self.tool_id == "containment":
            return self._create_containment_args(image_path, user_input, context, **kwargs)
        else:
            # Default spatial tool arguments
            return {
                "image_path": image_path,
                "meters_per_pixel": kwargs.get("meters_per_pixel", 0.3)
            }

    def _create_buffer_args(self, image_path: str, user_input: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create buffer tool arguments matching benchmark format."""
        # Extract buffer distance from user input or use default
        buffer_distance = self._extract_buffer_distance(user_input, **kwargs)

        # Get geometry coordinates from previous segmentation results or kwargs
        geometry_coordinates = kwargs.get("geometry_coordinates", [])

        # If no geometry coordinates provided, try to get from tool context
        if not geometry_coordinates:
            geometry_coordinates = _tool_context.get_geometry_coordinates(image_path)

        # CRITICAL FIX: Do NOT generate mock coordinates - this breaks data integrity
        # If no coordinates are available, return empty list and let the tool handle the error
        if not geometry_coordinates:
            planner_logger.warning("No geometry coordinates available for buffer tool - this indicates missing perception results")
            geometry_coordinates = []

        return {
            "geometry_coordinates": geometry_coordinates,
            "buffer_distance_meters": buffer_distance,
            "image_path": image_path,
            "geometry_type": "polygons"
        }

    def _create_overlap_args(self, image_path: str, user_input: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create overlap tool arguments matching benchmark format."""
        # Extract classes from user input or kwargs
        classes_used = kwargs.get("classes_used", [])

        # If no classes provided, try to get from tool context or extract from input
        if not classes_used:
            classes_used = _tool_context.get_last_classes()

        if not classes_used:
            classes_used = self._extract_classes_from_input(user_input, **kwargs)

        return {
            "classes_used": classes_used,
            "image_path": image_path
        }

    def _create_containment_args(self, image_path: str, user_input: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Create containment tool arguments matching benchmark format."""
        # Extract classes from user input or kwargs
        classes_used = kwargs.get("classes_used", [])

        # If no classes provided, try to get from tool context or extract from input
        if not classes_used:
            classes_used = _tool_context.get_last_classes()

        if not classes_used:
            classes_used = self._extract_classes_from_input(user_input, **kwargs)

        return {
            "classes_used": classes_used,
            "image_path": image_path
        }

    def _extract_buffer_distance(self, user_input: str, **kwargs) -> float:
        """Extract buffer distance from user input or kwargs."""
        import re

        # Check if explicitly provided in kwargs
        if "buffer_distance_meters" in kwargs:
            return float(kwargs["buffer_distance_meters"])

        # Try to extract from user input using regex patterns
        # Look for patterns like "50 m", "100 meters", "20m", etc.
        distance_patterns = [
            r'(\d+(?:\.\d+)?)\s*m(?:eters?)?(?:\s|$)',  # "50 m", "100 meters"
            r'(\d+(?:\.\d+)?)\s*meter(?:s)?(?:\s|$)',   # "50 meter", "100 meters"
            r'buffer.*?(\d+(?:\.\d+)?)',                # "buffer 50", "buffer by 100"
            r'within\s+(\d+(?:\.\d+)?)',                # "within 50"
        ]

        for pattern in distance_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return float(match.group(1))

        # Default buffer distances based on common use cases
        if "road" in user_input.lower():
            return 50.0  # Default road buffer
        elif "agriculture" in user_input.lower():
            return 100.0  # Default agriculture buffer
        else:
            return 30.0  # General default

    def _extract_classes_from_input(self, user_input: str, **kwargs) -> List[str]:
        """Extract class names from user input or kwargs."""
        # Check if explicitly provided in kwargs
        if "classes_used" in kwargs:
            return kwargs["classes_used"]

        # Extract from text_prompt if available
        if "text_prompt" in kwargs:
            text_prompt = kwargs["text_prompt"]
        else:
            text_prompt = user_input

        # Common class mappings from benchmark
        class_mappings = {
            "building": "buildings",
            "buildings": "buildings",
            "road": "roads",
            "roads": "roads",
            "water": "water",
            "forest": "forests",
            "forests": "forests",
            "agriculture": "agriculture",
            "barren": "barren"
        }

        # Extract classes from comma-separated text
        classes = []
        if text_prompt:
            # Split by common separators
            parts = re.split(r'[,\s]+', text_prompt.lower())
            for part in parts:
                part = part.strip()
                if part in class_mappings:
                    mapped_class = class_mappings[part]
                    if mapped_class not in classes:
                        classes.append(mapped_class)

        return classes if classes else ["buildings", "roads"]  # Default classes


class ToolExecutionContext:
    """
    Context manager for tool execution pipeline.
    Maintains state between tool calls to enable proper argument chaining.
    """

    def __init__(self):
        self.segmentation_results = {}  # Store segmentation outputs by image_path
        self.last_segmentation_classes = []  # Track last segmented classes
        self.geometry_coordinates = []  # Store geometry coordinates for buffer tools

    def store_segmentation_result(self, image_path: str, result: Dict[str, Any], classes: List[str]):
        """Store segmentation results for later use by spatial tools."""
        self.segmentation_results[image_path] = result
        self.last_segmentation_classes = classes

        # CRITICAL FIX: Do NOT generate mock coordinates - extract actual coordinates from segmentation results
        # This ensures data integrity in the tool pipeline
        self.geometry_coordinates = self._extract_actual_coordinates_from_segmentation(result, classes)

    def get_geometry_coordinates(self, image_path: str) -> List[List[List[float]]]:
        """Get geometry coordinates for buffer operations."""
        return self.geometry_coordinates

    def get_last_classes(self) -> List[str]:
        """Get classes from last segmentation for overlap/containment tools."""
        return self.last_segmentation_classes

    def _extract_actual_coordinates_from_segmentation(self, segmentation_result: Dict[str, Any], classes: List[str]) -> List[List[List[float]]]:
        """
        Extract actual geometry coordinates from segmentation results.
        CRITICAL FIX: No mock data generation - only real coordinates from tool outputs.
        """
        actual_coordinates = []

        try:
            # Try to extract coordinates from segmentation result structure
            if isinstance(segmentation_result, dict):
                # Check for coordinates_by_class structure
                coordinates_by_class = segmentation_result.get("coordinates_by_class", {})
                for class_name, class_coords in coordinates_by_class.items():
                    if isinstance(class_coords, list):
                        for coord_set in class_coords:
                            if isinstance(coord_set, list) and coord_set:
                                actual_coordinates.append(coord_set)

                # Check for other possible coordinate fields
                for field in ["polygons", "geometries", "coordinates", "masks"]:
                    if field in segmentation_result:
                        field_data = segmentation_result[field]
                        if isinstance(field_data, list):
                            actual_coordinates.extend(field_data)

            print(f"[ToolContext] Extracted {len(actual_coordinates)} actual coordinates from segmentation results")

        except Exception as e:
            print(f"[ToolContext] Failed to extract coordinates from segmentation results: {e}")

        return actual_coordinates

    def _generate_mock_coordinates(self, segmentation_result: Dict[str, Any], classes: List[str]) -> List[List[List[float]]]:
        """
        DEPRECATED: Mock coordinate generation is disabled to ensure data integrity.
        This method now returns empty list to prevent placeholder data generation.
        """
        print(f"[ToolContext] WARNING: Mock coordinate generation is disabled. No coordinates available.")
        return []


# Global context instance for tool chaining
_tool_context = ToolExecutionContext()


def update_segmentation_context(image_path: str, result: Dict[str, Any], text_prompt: str):
    """
    Update the global tool context with segmentation results.
    Should be called after segmentation tool execution.
    """
    # Extract classes from text_prompt
    classes = []
    if text_prompt:
        # Split by common separators and clean up
        parts = re.split(r'[,\s]+', text_prompt.lower())
        for part in parts:
            part = part.strip()
            if part and len(part) > 2:  # Filter out very short words
                classes.append(part)

    _tool_context.store_segmentation_result(image_path, result, classes)


def get_tool_context() -> ToolExecutionContext:
    """Get the global tool execution context."""
    return _tool_context


def reset_tool_context():
    """Reset the global tool execution context."""
    global _tool_context
    _tool_context = ToolExecutionContext()


class RemoteSAMToolInterface(ToolInterface):
    """
    Specialized interface for RemoteSAM tools.
    Consolidates common functionality for segmentation, detection, and classification.
    """
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get common parameter schema for RemoteSAM tools."""
        schema = {
            "image_path": {
                "type": "string",
                "description": "Path to image file (PNG or JPG)"
            },
            "text_prompt": {
                "type": "string",
                "description": f"Natural language description of objects to {self.category}"
            }
        }

        # Add meters_per_pixel only for detection and classification tools, not segmentation
        if self.category in ["detection", "classification"]:
            schema["meters_per_pixel"] = {
                "type": "number",
                "description": "Ground resolution in meters per pixel"
            }
            schema["confidence_threshold"] = {
                "type": "number",
                "description": "Minimum confidence threshold (0.0-1.0)"
            }
        
        if self.category == "segmentation":
            schema["buffer_distance_pixels"] = {
                "type": "number",
                "description": "Buffer distance in pixels"
            }
            schema["buffer_distance_m"] = {
                "type": "number",
                "description": "Buffer distance in meters (legacy)"
            }
        
        return schema
    
    def _get_required_parameters(self) -> List[str]:
        """Get required parameters for RemoteSAM tools."""
        return ["image_path", "text_prompt"]
    
    def create_tool_args(self,
                        image_path: str,
                        user_input: str,
                        context: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Create tool arguments from user input.
        Enhanced to match benchmark_mini.json argument structures.
        """
        text_prompt = self.extract_text_prompt(user_input, context)

        # Base arguments for all RemoteSAM tools - match benchmark format exactly
        args = {
            "image_path": image_path,
            "text_prompt": text_prompt
        }

        # Add meters_per_pixel only for detection and classification tools, not segmentation
        if self.category in ["detection", "classification"]:
            args["meters_per_pixel"] = kwargs.get("meters_per_pixel", 0.3)
            # Use remote sensing best practices: 0.6 for detection, 0.65 for classification
            default_threshold = 0.6 if self.category == "detection" else 0.65
            args["confidence_threshold"] = kwargs.get("confidence_threshold", default_threshold)

        if self.category == "segmentation":
            # Segmentation tools get confidence_threshold but NOT meters_per_pixel
            confidence_threshold = kwargs.get("confidence_threshold", 0.5)
            # Cap confidence threshold for segmentation to prevent failures due to overly high thresholds
            if confidence_threshold > 0.8:
                print(f"[ToolInterface] Warning: Capping high confidence threshold {confidence_threshold} to 0.5 for segmentation")
                confidence_threshold = 0.5
            args["confidence_threshold"] = confidence_threshold
            if "buffer_distance_pixels" in kwargs:
                args["buffer_distance_pixels"] = kwargs["buffer_distance_pixels"]
            if "buffer_distance_m" in kwargs:
                args["buffer_distance_m"] = kwargs["buffer_distance_m"]

        return args


class SARToolInterface(ToolInterface):
    """
    Tool interface for SAR (Synthetic Aperture Radar) analysis tools.
    Handles SARATR-X-based detection and classification tools.
    """

    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema for SAR tools."""
        base_schema = {
            "image_path": {
                "type": "string",
                "description": "Path to SAR image file (PNG, JPG, or TIFF)"
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Confidence threshold (0.0-1.0)",
                "default": 0.3 if self.tool_id == "sar_detection" else 0.5
            },
            "device": {
                "type": "string",
                "description": "CUDA device for inference",
                "default": "cuda:2"
            }
        }

        if self.tool_id == "sar_classification":
            base_schema["classification_type"] = {
                "type": "string",
                "description": "Type of classification: 'scene', 'target', or 'fine_grained'",
                "default": "scene"
            }

        return base_schema

    def _get_required_parameters(self) -> List[str]:
        """Get required parameters for SAR tools."""
        return ["image_path"]

    def create_tool_args(self,
                        image_path: str,
                        user_input: str,
                        context: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Create tool arguments for SAR analysis tools.

        Args:
            image_path: Path to SAR image
            user_input: User query/input
            context: Optional context information
            **kwargs: Additional arguments

        Returns:
            Dictionary of tool arguments
        """
        # Base arguments for all SAR tools
        args = {
            "image_path": image_path
        }

        if self.tool_id == "sar_detection":
            # SAR detection specific arguments
            args["confidence_threshold"] = kwargs.get("confidence_threshold", 0.3)
            args["device"] = kwargs.get("device", "cuda:2")

        elif self.tool_id == "sar_classification":
            # SAR classification specific arguments
            args["classification_type"] = self._extract_classification_type(user_input, context)
            args["confidence_threshold"] = kwargs.get("confidence_threshold", 0.5)
            args["device"] = kwargs.get("device", "cuda:2")

        return args

    def _extract_classification_type(self, user_input: str, context: Optional[str] = None) -> str:
        """
        Extract classification type from user input.

        Args:
            user_input: User query
            context: Optional context

        Returns:
            Classification type ('scene', 'target', or 'fine_grained')
        """
        input_text = (user_input + " " + (context or "")).lower()

        # Check for fine-grained classification keywords
        fine_grained_keywords = [
            "fine-grained", "fine_grained", "detailed", "specific", "precise",
            "tank", "t62", "t72", "bmp2", "btr60", "btr70", "zsu234",
            "airbus", "boeing", "comac", "cargo", "fishing", "tanker"
        ]

        # Check for target classification keywords
        target_keywords = [
            "target", "vehicle", "ship", "aircraft", "building", "bridge",
            "truck", "car", "helicopter", "fighter", "jet"
        ]

        # Check for scene classification keywords
        scene_keywords = [
            "scene", "terrain", "land", "cover", "urban", "rural", "coastal",
            "desert", "forest", "mountain", "agricultural", "industrial"
        ]

        if any(keyword in input_text for keyword in fine_grained_keywords):
            return "fine_grained"
        elif any(keyword in input_text for keyword in target_keywords):
            return "target"
        elif any(keyword in input_text for keyword in scene_keywords):
            return "scene"
        else:
            # Default to scene classification
            return "scene"


# Factory function for creating tool interfaces
def create_tool_interface(tool_id: str) -> ToolInterface:
    """
    Factory function to create appropriate tool interface.

    Args:
        tool_id: Tool identifier

    Returns:
        Appropriate tool interface instance
    """
    if tool_id in ["segmentation", "detection", "classification"]:
        return RemoteSAMToolInterface(tool_id)
    elif tool_id in ["sar_detection", "sar_classification"]:
        return SARToolInterface(tool_id)
    elif tool_id in ["buffer", "overlap", "containment"]:
        # For spatial relations tools, use spatial interface
        return SpatialToolInterface(tool_id)
    elif tool_id in ["distance_calculation", "area_measurement", "object_count_aoi"]:
        # For spatial statistics tools, use spatial interface
        return SpatialToolInterface(tool_id)
    else:
        # For unknown tools, raise an error
        raise ValueError(f"Unknown tool ID: {tool_id}")


# Global tool interface registry
_tool_interfaces = {}

def get_tool_interface(tool_id: str) -> ToolInterface:
    """Get cached tool interface instance."""
    if tool_id not in _tool_interfaces:
        _tool_interfaces[tool_id] = create_tool_interface(tool_id)
    return _tool_interfaces[tool_id]
