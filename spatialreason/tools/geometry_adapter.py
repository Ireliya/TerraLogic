"""
Geometry Format Adapter for Spatial Reasoning Agent

This module provides robust geometry format conversion between perception tools
and spatial relation tools. It handles the conversion from detection tool outputs
(bounding boxes) to spatial relation tool inputs (polygon coordinates).

Key Features:
- Converts bounding box format [x1, y1, x2, y2] to polygon coordinates [[x1,y1], [x2,y2], ...]
- Handles various geometry formats from perception tools
- Provides validation and error handling for geometry conversions
- Maintains coordinate system consistency across tools
"""

import json
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
from shapely.geometry import Polygon, Point, box
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


class GeometryFormatAdapter:
    """
    Adapter class for converting between different geometry formats used by
    perception tools and spatial relation tools.
    """
    
    def __init__(self):
        """Initialize the geometry format adapter."""
        self.supported_input_formats = ["bbox", "polygon", "point", "centroid"]
        self.supported_output_formats = ["polygon_coords", "bbox_coords", "point_coords"]
    
    def convert_perception_to_spatial(self, 
                                    perception_results: Dict[str, Any],
                                    target_format: str = "polygon_coords") -> Dict[str, List[List[float]]]:
        """
        Convert perception tool results to spatial relation tool format.
        
        Args:
            perception_results: Results from detection/segmentation tools
            target_format: Target format for spatial tools ("polygon_coords", "bbox_coords", "point_coords")
            
        Returns:
            Dictionary mapping class names to converted coordinate lists
        """
        converted_geometries = {}
        
        try:
            # Handle different perception result formats
            if "detections" in perception_results:
                # Detection tool format
                detections = perception_results["detections"]
                for detection in detections:
                    class_name = detection.get("class", "unknown")
                    if class_name not in converted_geometries:
                        converted_geometries[class_name] = []
                    
                    # Convert detection geometry
                    converted_geom = self._convert_detection_geometry(detection, target_format)
                    if converted_geom:
                        converted_geometries[class_name].append(converted_geom)
            
            elif "segmentations" in perception_results:
                # Segmentation tool format
                segmentations = perception_results["segmentations"]
                for segmentation in segmentations:
                    class_name = segmentation.get("class", "unknown")
                    if class_name not in converted_geometries:
                        converted_geometries[class_name] = []
                    
                    # Convert segmentation geometry
                    converted_geom = self._convert_segmentation_geometry(segmentation, target_format)
                    if converted_geom:
                        converted_geometries[class_name].append(converted_geom)
            
            logger.info(f"Converted {len(converted_geometries)} geometry classes to {target_format}")
            return converted_geometries
            
        except Exception as e:
            logger.error(f"Failed to convert perception results: {e}")
            return {}
    
    def _convert_detection_geometry(self, detection: Dict[str, Any], target_format: str) -> Optional[List[float]]:
        """Convert a single detection geometry to target format.

        Prioritizes polygon format (unified geometric representation) over bbox.
        Falls back to centroid if polygon is not available.
        """
        try:
            # Prioritize polygon format (unified geometric representation)
            if "polygon" in detection:
                polygon = detection["polygon"]
                return self._convert_polygon_to_target(polygon, target_format)
            # Fallback to bbox if polygon is not available
            elif "bbox" in detection:
                bbox = detection["bbox"]
                return self._convert_bbox_to_target(bbox, target_format)
            # Last resort: use centroid
            elif "centroid" in detection:
                centroid = detection["centroid"]
                return self._convert_centroid_to_target(centroid, target_format)
            else:
                logger.warning(f"No supported geometry found in detection: {list(detection.keys())}")
                return None

        except Exception as e:
            logger.error(f"Failed to convert detection geometry: {e}")
            return None
    
    def _convert_segmentation_geometry(self, segmentation: Dict[str, Any], target_format: str) -> Optional[List[float]]:
        """Convert a single segmentation geometry to target format."""
        try:
            if "polygon" in segmentation:
                polygon = segmentation["polygon"]
                return self._convert_polygon_to_target(polygon, target_format)
            elif "bbox" in segmentation:
                bbox = segmentation["bbox"]
                return self._convert_bbox_to_target(bbox, target_format)
            else:
                logger.warning(f"No supported geometry found in segmentation: {list(segmentation.keys())}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to convert segmentation geometry: {e}")
            return None
    
    def _convert_bbox_to_target(self, bbox: Dict[str, Any], target_format: str) -> List[float]:
        """Convert bounding box to target format."""
        if isinstance(bbox, dict):
            x_min = bbox.get("x_min", 0)
            y_min = bbox.get("y_min", 0)
            x_max = bbox.get("x_max", 0)
            y_max = bbox.get("y_max", 0)
        elif isinstance(bbox, list) and len(bbox) >= 4:
            x_min, y_min, x_max, y_max = bbox[:4]
        else:
            raise ValueError(f"Invalid bbox format: {bbox}")
        
        if target_format == "polygon_coords":
            # Convert bbox to polygon coordinate list
            return [
                [float(x_min), float(y_min)],
                [float(x_max), float(y_min)],
                [float(x_max), float(y_max)],
                [float(x_min), float(y_max)],
                [float(x_min), float(y_min)]  # Close the polygon
            ]
        elif target_format == "bbox_coords":
            return [float(x_min), float(y_min), float(x_max), float(y_max)]
        elif target_format == "point_coords":
            # Return centroid of bbox
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            return [float(center_x), float(center_y)]
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def _convert_centroid_to_target(self, centroid: Dict[str, Any], target_format: str) -> List[float]:
        """Convert centroid to target format."""
        if isinstance(centroid, dict):
            x = centroid.get("x", 0)
            y = centroid.get("y", 0)
        elif isinstance(centroid, list) and len(centroid) >= 2:
            x, y = centroid[:2]
        else:
            raise ValueError(f"Invalid centroid format: {centroid}")
        
        if target_format == "point_coords":
            return [float(x), float(y)]
        elif target_format == "polygon_coords":
            # Create a small square around the point
            buffer = 1.0
            return [
                [float(x - buffer), float(y - buffer)],
                [float(x + buffer), float(y - buffer)],
                [float(x + buffer), float(y + buffer)],
                [float(x - buffer), float(y + buffer)],
                [float(x - buffer), float(y - buffer)]
            ]
        elif target_format == "bbox_coords":
            # Create a small bbox around the point
            buffer = 1.0
            return [float(x - buffer), float(y - buffer), float(x + buffer), float(y + buffer)]
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def _convert_polygon_to_target(self, polygon: List[List[float]], target_format: str) -> List[float]:
        """Convert polygon coordinates to target format."""
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"Invalid polygon format: {polygon}")
        
        if target_format == "polygon_coords":
            # Ensure polygon is closed
            if polygon[0] != polygon[-1]:
                polygon = polygon + [polygon[0]]
            return [[float(x), float(y)] for x, y in polygon]
        elif target_format == "bbox_coords":
            # Calculate bounding box of polygon
            x_coords = [point[0] for point in polygon]
            y_coords = [point[1] for point in polygon]
            return [float(min(x_coords)), float(min(y_coords)), 
                   float(max(x_coords)), float(max(y_coords))]
        elif target_format == "point_coords":
            # Calculate centroid of polygon
            shapely_poly = Polygon(polygon)
            centroid = shapely_poly.centroid
            return [float(centroid.x), float(centroid.y)]
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    def convert_planner_coordinates(self, coordinates: List[Dict[str, Any]], 
                                  target_geometry_type: str) -> List[List[float]]:
        """
        Convert planner coordinate format to spatial tool format.
        
        Args:
            coordinates: List of coordinate dictionaries from planner
            target_geometry_type: Target geometry type ("points", "polygons", "bboxes")
            
        Returns:
            List of converted coordinates in target format
        """
        converted_coords = []
        
        for coord_entry in coordinates:
            try:
                if isinstance(coord_entry, dict):
                    entry_type = coord_entry.get("type", "")
                    entry_coords = coord_entry.get("coordinates", [])
                    
                    if entry_type == "bbox":
                        if target_geometry_type == "polygons":
                            # Convert bbox to polygon
                            if len(entry_coords) >= 4:
                                x_min, y_min, x_max, y_max = entry_coords[:4]
                                polygon_coords = [
                                    [float(x_min), float(y_min)],
                                    [float(x_max), float(y_min)],
                                    [float(x_max), float(y_max)],
                                    [float(x_min), float(y_max)],
                                    [float(x_min), float(y_min)]
                                ]
                                converted_coords.append(polygon_coords)
                        elif target_geometry_type == "bboxes":
                            converted_coords.append(entry_coords)
                        elif target_geometry_type == "points":
                            # Convert bbox to centroid
                            if len(entry_coords) >= 4:
                                x_min, y_min, x_max, y_max = entry_coords[:4]
                                center_x = (x_min + x_max) / 2
                                center_y = (y_min + y_max) / 2
                                converted_coords.append([float(center_x), float(center_y)])
                    
                    elif entry_type == "polygon":
                        if target_geometry_type == "polygons":
                            converted_coords.append(entry_coords)
                        elif target_geometry_type == "bboxes":
                            # Convert polygon to bbox
                            if len(entry_coords) >= 3:
                                x_coords = [point[0] for point in entry_coords]
                                y_coords = [point[1] for point in entry_coords]
                                bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                                converted_coords.append(bbox)
                        elif target_geometry_type == "points":
                            # Convert polygon to centroid
                            if len(entry_coords) >= 3:
                                shapely_poly = Polygon(entry_coords)
                                centroid = shapely_poly.centroid
                                converted_coords.append([float(centroid.x), float(centroid.y)])
                    
                    elif entry_type == "point":
                        if target_geometry_type == "points":
                            converted_coords.append(entry_coords)
                        elif target_geometry_type == "polygons":
                            # Convert point to small polygon
                            if len(entry_coords) >= 2:
                                x, y = entry_coords[:2]
                                buffer = 1.0
                                polygon_coords = [
                                    [float(x - buffer), float(y - buffer)],
                                    [float(x + buffer), float(y - buffer)],
                                    [float(x + buffer), float(y + buffer)],
                                    [float(x - buffer), float(y + buffer)],
                                    [float(x - buffer), float(y - buffer)]
                                ]
                                converted_coords.append(polygon_coords)
                        elif target_geometry_type == "bboxes":
                            # Convert point to small bbox
                            if len(entry_coords) >= 2:
                                x, y = entry_coords[:2]
                                buffer = 1.0
                                bbox = [float(x - buffer), float(y - buffer), 
                                       float(x + buffer), float(y + buffer)]
                                converted_coords.append(bbox)
                
                else:
                    # Handle direct coordinate lists
                    if target_geometry_type == "polygons" and isinstance(coord_entry, list):
                        converted_coords.append(coord_entry)
                    elif target_geometry_type == "bboxes" and isinstance(coord_entry, list) and len(coord_entry) >= 4:
                        converted_coords.append(coord_entry[:4])
                    elif target_geometry_type == "points" and isinstance(coord_entry, list) and len(coord_entry) >= 2:
                        converted_coords.append(coord_entry[:2])
                        
            except Exception as e:
                logger.error(f"Failed to convert coordinate entry {coord_entry}: {e}")
                continue
        
        logger.info(f"Converted {len(converted_coords)} coordinates to {target_geometry_type} format")
        return converted_coords
