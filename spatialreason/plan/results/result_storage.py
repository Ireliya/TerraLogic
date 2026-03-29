"""
Result storage module for spatialreason.plan

This module contains the ResultStorage class which handles storing and
tracking execution results.

Classes:
    ResultStorage: Stores and tracks execution results
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List

# Setup module logger
logger = logging.getLogger(__name__)


class ResultStorage:
    """
    Stores and tracks execution results from tool execution.
    
    This class handles:
    - Storing tool execution results
    - Tracking perception results
    - Managing result metadata
    - Extracting results for downstream tools
    """
    
    def __init__(self):
        """Initialize the result storage."""
        self.tool_results_storage = {}
        self.perception_results = {}
        self.step_results = {}
        self.current_image_path = None
    
    def store_tool_result(self, tool_name: str, result: Any, action_input: Dict[str, Any]) -> None:
        """
        Store tool execution result for automatic parameter extraction.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result (JSON string or dict)
            action_input: Input parameters that were passed to the tool
        """
        try:
            # Parse result if it's a JSON string
            if isinstance(result, str) and result.startswith('{'):
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    parsed_result = {"raw_output": result}
            else:
                parsed_result = result if isinstance(result, dict) else {"raw_output": str(result)}
            
            # Store the tool result with metadata
            if tool_name not in self.tool_results_storage:
                self.tool_results_storage[tool_name] = []
            
            self.tool_results_storage[tool_name].append({
                "result": parsed_result,
                "input_params": action_input,
                "timestamp": time.time(),
                "image_path": action_input.get("image_path", self.current_image_path)
            })
            
            logger.info(f"🔧 Stored {tool_name} result for automatic parameter extraction")
        
        except Exception as e:
            logger.error(f"Failed to store tool result for {tool_name}: {e}")
    
    def store_perception_result(self, tool_name: str, result: Any, image_path: str) -> None:
        """
        Store perception tool results for coordinate extraction by spatial tools.

        Args:
            tool_name: Name of the perception tool (segmentation, detection, classification, sar_detection, sar_classification, infrared_detection)
            result: JSON result string from the perception tool
            image_path: Path to the processed image
        """
        try:
            # Parse the result JSON
            result_data = json.loads(result) if isinstance(result, str) else result

            if not result_data.get("success", False):
                logger.warning(f"Perception tool {tool_name} failed, not storing results")
                return

            # Store the current image path
            self.current_image_path = image_path

            # Extract and store coordinates from perception results
            # Handle different field names used by different perception tools:
            # - segmentation: uses "segments" field
            # - detection, classification, sar_detection: use "detections" field
            # - infrared_detection: uses "detections" nested inside "output" object
            # - sar_classification: uses "classification_results" field

            detections = None

            # Try different field names in order of priority
            if "segments" in result_data:
                # Segmentation tool uses "segments"
                detections = result_data.get("segments", [])
                logger.debug(f"Found {len(detections)} segments from segmentation tool")
            elif "detections" in result_data:
                # Detection, classification, sar_detection use "detections"
                detections = result_data.get("detections", [])
                logger.debug(f"Found {len(detections)} detections from detection/classification tool")
            elif "output" in result_data and isinstance(result_data["output"], dict):
                # Infrared detection nests detections inside "output" object
                detections = result_data["output"].get("detections", [])
                logger.debug(f"Found {len(detections)} detections from infrared_detection tool (nested in output)")
            elif "classification_results" in result_data:
                # SAR classification uses "classification_results"
                # This is a different structure - it's a dict with class predictions, not a list of detections
                # We'll handle this separately below
                detections = []
                logger.debug(f"Found classification_results from sar_classification tool")
            else:
                # Fallback: empty list
                detections = []
                logger.warning(f"No detections/segments/classification_results found in {tool_name} result. Available keys: {list(result_data.keys())}")

            coordinates_by_class = {}
            total_detections = 0

            # Handle standard detections/segments format (list of objects)
            if detections:
                for detection in detections:
                    class_name = detection.get("class", "unknown")

                    # Extract different coordinate formats
                    coordinate_data = {
                        "object_id": detection.get("object_id", f"{class_name}_{total_detections + 1}"),
                        "bbox": detection.get("bbox", {}),
                        "centroid": detection.get("centroid", {}),
                        "polygon": detection.get("polygon", []),
                        "area_pixels": detection.get("area_pixels", 0),
                        "confidence": detection.get("confidence", 0.0)
                    }

                    if class_name not in coordinates_by_class:
                        coordinates_by_class[class_name] = []
                    coordinates_by_class[class_name].append(coordinate_data)
                    total_detections += 1

            # Handle SAR classification results (different structure)
            elif "classification_results" in result_data:
                # SAR classification returns a dict with class predictions
                # We don't extract coordinates from classification results
                # Just store the raw result for downstream tools
                classification_results = result_data.get("classification_results", {})
                logger.debug(f"Stored SAR classification results: {classification_results}")

            # Store in perception results
            self.perception_results[tool_name] = {
                "coordinates_by_class": coordinates_by_class,
                "total_detections": total_detections,
                "image_path": image_path,
                "raw_result": result_data
            }

            logger.info(f"✅ Stored {tool_name} perception results: {total_detections} detections")

        except Exception as e:
            logger.error(f"Failed to store perception result for {tool_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def store_step_result(self, step_id: int, result: Any) -> None:
        """
        Store result from a single step execution.
        
        Args:
            step_id: ID of the step
            result: Result from step execution
        """
        self.step_results[step_id] = result
        logger.debug(f"Stored result for step {step_id}")
    
    def get_tool_result(self, tool_name: str) -> Optional[Any]:
        """
        Get the most recent result for a tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Most recent tool result or None
        """
        if tool_name in self.tool_results_storage:
            results = self.tool_results_storage[tool_name]
            if results:
                return results[-1]["result"]
        return None
    
    def get_perception_result(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get perception result for a tool.
        
        Args:
            tool_name: Name of the perception tool
        
        Returns:
            Perception result dictionary or None
        """
        return self.perception_results.get(tool_name)
    
    def get_all_tool_results(self, tool_name: str) -> List[Dict[str, Any]]:
        """
        Get all results for a tool.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            List of all tool results
        """
        return self.tool_results_storage.get(tool_name, [])
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.tool_results_storage.clear()
        self.perception_results.clear()
        self.step_results.clear()
        logger.info("Cleared all stored results")
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of stored results.
        
        Returns:
            Dictionary with storage summary
        """
        return {
            "tools_stored": list(self.tool_results_storage.keys()),
            "perception_tools_stored": list(self.perception_results.keys()),
            "steps_stored": list(self.step_results.keys()),
            "total_tool_results": sum(len(v) for v in self.tool_results_storage.values()),
            "current_image_path": self.current_image_path
        }

