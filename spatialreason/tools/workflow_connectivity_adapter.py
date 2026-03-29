"""
Workflow Connectivity Adapter for Spatial Reasoning Agent

This module provides comprehensive data flow connectivity between perception tools,
spatial relationship tools, and statistical tools. It handles format conversion,
data extraction, and parameter mapping to enable seamless multi-step workflows.

Key Features:
- Converts perception tool outputs to statistical tool inputs
- Converts spatial relationship tool outputs to statistical tool inputs  
- Handles geometry format conversions (bbox to polygon, etc.)
- Provides parameter mapping and validation
- Maintains coordinate system consistency across tools

Usage Examples:
    # Direct perception → statistics workflow
    adapter = WorkflowConnectivityAdapter()
    
    # Convert segmentation results to area measurement input
    area_params = adapter.convert_perception_to_area_measurement(
        perception_results=segmentation_results,
        image_path="image.tif"
    )
    
    # Convert detection results to object count input
    count_params = adapter.convert_perception_to_object_count(
        perception_results=detection_results,
        aoi_results=segmentation_results,  # AOI from another perception tool
        image_path="image.tif"
    )
    
    # Convert spatial relation results to statistics
    distance_params = adapter.convert_spatial_relation_to_distance(
        spatial_results=buffer_results,
        target_results=detection_results,
        image_path="image.tif"
    )
"""

import json
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path

class WorkflowConnectivityAdapter:
    """
    Comprehensive adapter for connecting perception, spatial relations, and statistical tools.
    Handles data format conversion and parameter mapping between different tool types.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_geometries_from_perception(self, perception_results: Dict[str, Any], 
                                         tool_type: str = "auto") -> Dict[str, List[List[float]]]:
        """
        Extract geometry coordinates from perception tool results.
        
        Args:
            perception_results: Results from perception tools (detection, segmentation, classification)
            tool_type: Type of perception tool ("detection", "segmentation", "classification", "auto")
            
        Returns:
            Dictionary mapping geometry types to coordinate lists
        """
        try:
            if isinstance(perception_results, str):
                perception_results = json.loads(perception_results)
            
            # Auto-detect tool type if not specified
            if tool_type == "auto":
                if "segments" in perception_results:
                    tool_type = "segmentation"
                elif "detections" in perception_results:
                    # Check if detections have polygon data (segmentation) or just bbox (detection)
                    if perception_results["detections"] and "polygon" in perception_results["detections"][0]:
                        tool_type = "segmentation"
                    else:
                        tool_type = "detection"
                elif "hierarchical_regions" in perception_results:
                    tool_type = "classification"
                else:
                    tool_type = "detection"  # Default fallback

            geometries = {
                "polygons": [],
                "bboxes": [],
                "points": []
            }

            if tool_type == "segmentation":
                # Extract polygon coordinates from segmentation (check both "segments" and "detections" for backward compatibility)
                segments = perception_results.get("segments", perception_results.get("detections", []))
                for segment in segments:
                    if "polygon" in segment and segment["polygon"]:
                        geometries["polygons"].append(segment["polygon"])
            
            elif tool_type == "detection" and "detections" in perception_results:
                # Extract geometries from detection (prioritize polygon format - unified geometric representation)
                for detection in perception_results["detections"]:
                    # Prioritize polygon format (unified geometric representation)
                    if "polygon" in detection and detection["polygon"]:
                        geometries["polygons"].append(detection["polygon"])
                    # Fallback to bbox if polygon is not available
                    elif "bbox" in detection:
                        bbox = detection["bbox"]
                        bbox_coords = [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
                        geometries["bboxes"].append(bbox_coords)

                        # Also create polygon version from bbox
                        polygon = self._bbox_to_polygon(bbox)
                        geometries["polygons"].append(polygon)

                    # Also extract centroid point if available
                    centroid = detection.get("centroid", {})
                    if centroid:
                        geometries["points"].append([centroid["x"], centroid["y"]])
            
            elif tool_type == "classification" and "hierarchical_regions" in perception_results:
                # Extract regions from classification results
                for category, regions in perception_results["hierarchical_regions"].items():
                    for region in regions:
                        if "bbox" in region:
                            bbox = region["bbox"]
                            polygon = self._bbox_to_polygon(bbox)
                            geometries["polygons"].append(polygon)
                        if "centroid" in region:
                            centroid = region["centroid"]
                            geometries["points"].append([centroid["x"], centroid["y"]])
            
            return geometries
            
        except Exception as e:
            self.logger.error(f"Error extracting geometries from perception results: {e}")
            return {"polygons": [], "bboxes": [], "points": []}
    
    def _bbox_to_polygon(self, bbox: Dict[str, float]) -> List[List[float]]:
        """Convert bounding box to polygon coordinates."""
        x_min = bbox["x_min"]
        y_min = bbox["y_min"] 
        x_max = bbox["x_max"]
        y_max = bbox["y_max"]
        
        return [
            [x_min, y_min],
            [x_max, y_min], 
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min]  # Close the polygon
        ]
    
    def convert_perception_to_area_measurement(self, perception_results: Dict[str, Any],
                                             image_path: str,
                                             meters_per_pixel: float,
                                             tool_type: str = "auto") -> Dict[str, Any]:
        """
        Convert perception tool results to area measurement tool parameters.
        
        Args:
            perception_results: Results from perception tools
            image_path: Path to input image
            meters_per_pixel: Ground resolution
            tool_type: Type of perception tool
            
        Returns:
            Parameters for AreaMeasurementTool
        """
        geometries = self.extract_geometries_from_perception(perception_results, tool_type)
        
        # Area measurement requires polygons
        polygons = geometries["polygons"]
        if not polygons:
            raise ValueError("No polygon geometries found for area measurement")
        
        return {
            "polygons": polygons,
            "image_path": image_path,
            "meters_per_pixel": meters_per_pixel
        }
    
    def convert_perception_to_object_count(self, perception_results: Dict[str, Any],
                                         aoi_results: Dict[str, Any],
                                         image_path: str,
                                         meters_per_pixel: float,
                                         object_tool_type: str = "auto",
                                         aoi_tool_type: str = "auto") -> Dict[str, Any]:
        """
        Convert perception tool results to object count AOI tool parameters.
        
        Args:
            perception_results: Results from perception tools (objects to count)
            aoi_results: Results from perception tools (areas of interest)
            image_path: Path to input image
            meters_per_pixel: Ground resolution
            object_tool_type: Type of perception tool for objects
            aoi_tool_type: Type of perception tool for AOI
            
        Returns:
            Parameters for ObjectCountInAOITool
        """
        object_geometries = self.extract_geometries_from_perception(perception_results, object_tool_type)
        aoi_geometries = self.extract_geometries_from_perception(aoi_results, aoi_tool_type)
        
        # Object count requires object geometries (any type) and AOI polygons
        objects = object_geometries["points"] + object_geometries["bboxes"] + object_geometries["polygons"]
        aoi_polygons = aoi_geometries["polygons"]
        
        if not objects:
            raise ValueError("No object geometries found for counting")
        if not aoi_polygons:
            raise ValueError("No AOI polygons found for counting")
        
        return {
            "object_geometries": objects,
            "aoi_geometries": aoi_polygons,
            "image_path": image_path,
            "meters_per_pixel": meters_per_pixel
        }
    
    def convert_perception_to_distance_calculation(self, perception_results_1: Dict[str, Any],
                                                 perception_results_2: Dict[str, Any],
                                                 image_path: str,
                                                 meters_per_pixel: float,
                                                 tool_type_1: str = "auto",
                                                 tool_type_2: str = "auto") -> Dict[str, Any]:
        """
        Convert two perception tool results to distance calculation tool parameters.
        
        Args:
            perception_results_1: First set of perception results
            perception_results_2: Second set of perception results
            image_path: Path to input image
            meters_per_pixel: Ground resolution
            tool_type_1: Type of first perception tool
            tool_type_2: Type of second perception tool
            
        Returns:
            Parameters for DistanceCalculationTool
        """
        geometries_1 = self.extract_geometries_from_perception(perception_results_1, tool_type_1)
        geometries_2 = self.extract_geometries_from_perception(perception_results_2, tool_type_2)
        
        # Use the most appropriate geometry type available
        geom_set_1 = geometries_1["polygons"] or geometries_1["bboxes"] or geometries_1["points"]
        geom_set_2 = geometries_2["polygons"] or geometries_2["bboxes"] or geometries_2["points"]
        
        if not geom_set_1:
            raise ValueError("No geometries found in first perception results")
        if not geom_set_2:
            raise ValueError("No geometries found in second perception results")
        
        # Determine geometry types
        geom_type_1 = "polygons" if geometries_1["polygons"] else ("bboxes" if geometries_1["bboxes"] else "points")
        geom_type_2 = "polygons" if geometries_2["polygons"] else ("bboxes" if geometries_2["bboxes"] else "points")
        
        return {
            "geometry_set_1": geom_set_1,
            "geometry_set_2": geom_set_2,
            "measurement_unit": "meters",
            "image_path": image_path,
            "meters_per_pixel": meters_per_pixel,
            "geometry_type_1": geom_type_1,
            "geometry_type_2": geom_type_2
        }

    def extract_geometries_from_spatial_relation(self, spatial_results: Dict[str, Any],
                                                tool_type: str = "auto") -> Dict[str, List[List[float]]]:
        """
        Extract geometry coordinates from spatial relation tool results.

        Args:
            spatial_results: Results from spatial relation tools (buffer, overlap, containment)
            tool_type: Type of spatial relation tool ("buffer", "overlap", "containment", "auto")

        Returns:
            Dictionary mapping geometry types to coordinate lists
        """
        try:
            if isinstance(spatial_results, str):
                spatial_results = json.loads(spatial_results)

            # Auto-detect tool type if not specified
            if tool_type == "auto":
                if "buffer_zones" in spatial_results:
                    tool_type = "buffer"
                elif "overlap_analysis" in spatial_results:
                    tool_type = "overlap"
                elif "containment_analysis" in spatial_results:
                    tool_type = "containment"
                else:
                    tool_type = "buffer"  # Default fallback

            geometries = {
                "polygons": [],
                "bboxes": [],
                "points": []
            }

            if tool_type == "buffer" and "buffer_zones" in spatial_results:
                # Extract buffer zone polygons
                for buffer_zone in spatial_results["buffer_zones"]:
                    if "buffer_polygon" in buffer_zone:
                        geometries["polygons"].append(buffer_zone["buffer_polygon"])

            elif tool_type == "overlap":
                # Extract geometries from overlap analysis
                if "overlap_analysis" in spatial_results:
                    analysis = spatial_results["overlap_analysis"]
                    if "intersection_polygon" in analysis:
                        geometries["polygons"].append(analysis["intersection_polygon"])

            elif tool_type == "containment":
                # Extract geometries from containment analysis
                if "containment_analysis" in spatial_results:
                    analysis = spatial_results["containment_analysis"]
                    if "container_polygon" in analysis:
                        geometries["polygons"].append(analysis["container_polygon"])
                    if "contained_geometries" in analysis:
                        for geom in analysis["contained_geometries"]:
                            if isinstance(geom, list) and len(geom) == 2:
                                geometries["points"].append(geom)
                            elif isinstance(geom, list) and len(geom) > 2:
                                geometries["polygons"].append(geom)

            return geometries

        except Exception as e:
            self.logger.error(f"Error extracting geometries from spatial relation results: {e}")
            return {"polygons": [], "bboxes": [], "points": []}

    def convert_spatial_relation_to_area_measurement(self, spatial_results: Dict[str, Any],
                                                   image_path: str,
                                                   meters_per_pixel: float,
                                                   tool_type: str = "auto") -> Dict[str, Any]:
        """
        Convert spatial relation tool results to area measurement tool parameters.

        Args:
            spatial_results: Results from spatial relation tools
            image_path: Path to input image
            meters_per_pixel: Ground resolution
            tool_type: Type of spatial relation tool

        Returns:
            Parameters for AreaMeasurementTool
        """
        geometries = self.extract_geometries_from_spatial_relation(spatial_results, tool_type)

        # Area measurement requires polygons
        polygons = geometries["polygons"]
        if not polygons:
            raise ValueError("No polygon geometries found in spatial relation results for area measurement")

        return {
            "polygons": polygons,
            "image_path": image_path,
            "meters_per_pixel": meters_per_pixel
        }

    def convert_spatial_relation_to_distance_calculation(self, spatial_results: Dict[str, Any],
                                                       target_results: Dict[str, Any],
                                                       image_path: str,
                                                       meters_per_pixel: float,
                                                       spatial_tool_type: str = "auto",
                                                       target_tool_type: str = "auto") -> Dict[str, Any]:
        """
        Convert spatial relation results and target results to distance calculation parameters.

        Args:
            spatial_results: Results from spatial relation tools
            target_results: Target geometries (from perception or spatial tools)
            image_path: Path to input image
            meters_per_pixel: Ground resolution
            spatial_tool_type: Type of spatial relation tool
            target_tool_type: Type of target tool

        Returns:
            Parameters for DistanceCalculationTool
        """
        spatial_geometries = self.extract_geometries_from_spatial_relation(spatial_results, spatial_tool_type)

        # Try to extract from target results (could be perception or spatial relation results)
        try:
            target_geometries = self.extract_geometries_from_perception(target_results, target_tool_type)
        except:
            target_geometries = self.extract_geometries_from_spatial_relation(target_results, target_tool_type)

        # Use the most appropriate geometry type available
        geom_set_1 = spatial_geometries["polygons"] or spatial_geometries["bboxes"] or spatial_geometries["points"]
        geom_set_2 = target_geometries["polygons"] or target_geometries["bboxes"] or target_geometries["points"]

        if not geom_set_1:
            raise ValueError("No geometries found in spatial relation results")
        if not geom_set_2:
            raise ValueError("No geometries found in target results")

        # Determine geometry types
        geom_type_1 = "polygons" if spatial_geometries["polygons"] else ("bboxes" if spatial_geometries["bboxes"] else "points")
        geom_type_2 = "polygons" if target_geometries["polygons"] else ("bboxes" if target_geometries["bboxes"] else "points")

        return {
            "geometry_set_1": geom_set_1,
            "geometry_set_2": geom_set_2,
            "measurement_unit": "meters",
            "image_path": image_path,
            "meters_per_pixel": meters_per_pixel,
            "geometry_type_1": geom_type_1,
            "geometry_type_2": geom_type_2
        }

    def convert_spatial_relation_to_object_count(self, spatial_results: Dict[str, Any],
                                               object_results: Dict[str, Any],
                                               image_path: str,
                                               meters_per_pixel: float,
                                               spatial_tool_type: str = "auto",
                                               object_tool_type: str = "auto") -> Dict[str, Any]:
        """
        Convert spatial relation results and object results to object count AOI parameters.

        Args:
            spatial_results: Results from spatial relation tools (used as AOI)
            object_results: Object geometries to count (from perception tools)
            image_path: Path to input image
            meters_per_pixel: Ground resolution
            spatial_tool_type: Type of spatial relation tool
            object_tool_type: Type of object tool

        Returns:
            Parameters for ObjectCountInAOITool
        """
        spatial_geometries = self.extract_geometries_from_spatial_relation(spatial_results, spatial_tool_type)

        # Try to extract from object results (could be perception results)
        try:
            object_geometries = self.extract_geometries_from_perception(object_results, object_tool_type)
        except:
            object_geometries = self.extract_geometries_from_spatial_relation(object_results, object_tool_type)

        # Object count requires object geometries (any type) and AOI polygons
        objects = object_geometries["points"] + object_geometries["bboxes"] + object_geometries["polygons"]
        aoi_polygons = spatial_geometries["polygons"]

        if not objects:
            raise ValueError("No object geometries found for counting")
        if not aoi_polygons:
            raise ValueError("No AOI polygons found in spatial relation results")

        return {
            "object_geometries": objects,
            "aoi_geometries": aoi_polygons,
            "image_path": image_path,
            "meters_per_pixel": meters_per_pixel
        }

    def validate_workflow_connectivity(self, source_results: Dict[str, Any],
                                     target_tool: str,
                                     source_tool_type: str = "auto") -> Dict[str, Any]:
        """
        Validate that source results can be connected to target tool.

        Args:
            source_results: Results from source tool
            target_tool: Target tool name ("area_measurement", "object_count_aoi", "distance_calculation")
            source_tool_type: Type of source tool

        Returns:
            Validation result with success status and details
        """
        try:
            # Try to extract geometries from source
            if source_tool_type in ["detection", "segmentation", "classification"] or source_tool_type == "auto":
                geometries = self.extract_geometries_from_perception(source_results, source_tool_type)
            else:
                geometries = self.extract_geometries_from_spatial_relation(source_results, source_tool_type)

            # Check compatibility with target tool
            validation_result = {
                "success": True,
                "source_tool_type": source_tool_type,
                "target_tool": target_tool,
                "available_geometries": {
                    "polygons": len(geometries["polygons"]),
                    "bboxes": len(geometries["bboxes"]),
                    "points": len(geometries["points"])
                },
                "compatibility": {},
                "recommendations": []
            }

            if target_tool == "area_measurement":
                if geometries["polygons"]:
                    validation_result["compatibility"]["area_measurement"] = "Compatible - polygons available"
                else:
                    validation_result["compatibility"]["area_measurement"] = "Incompatible - no polygons found"
                    validation_result["success"] = False
                    validation_result["recommendations"].append("Use segmentation tool to get polygon geometries")

            elif target_tool == "object_count_aoi":
                total_objects = len(geometries["polygons"]) + len(geometries["bboxes"]) + len(geometries["points"])
                if total_objects > 0:
                    validation_result["compatibility"]["object_count_aoi"] = f"Compatible - {total_objects} objects available"
                else:
                    validation_result["compatibility"]["object_count_aoi"] = "Incompatible - no object geometries found"
                    validation_result["success"] = False
                    validation_result["recommendations"].append("Use perception tools to detect objects first")

            elif target_tool == "distance_calculation":
                total_geometries = len(geometries["polygons"]) + len(geometries["bboxes"]) + len(geometries["points"])
                if total_geometries > 0:
                    validation_result["compatibility"]["distance_calculation"] = f"Compatible - {total_geometries} geometries available"
                else:
                    validation_result["compatibility"]["distance_calculation"] = "Incompatible - no geometries found"
                    validation_result["success"] = False
                    validation_result["recommendations"].append("Use perception tools to detect geometries first")

            return validation_result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "source_tool_type": source_tool_type,
                "target_tool": target_tool,
                "recommendations": ["Check source tool results format and try again"]
            }
