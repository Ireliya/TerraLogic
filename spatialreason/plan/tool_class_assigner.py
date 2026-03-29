"""
Tool Class Assigner for reconstructing tool-class assignments from agent execution.

This module analyzes the agent's execution trace to reconstruct the semantic relationships
between tools and classes, creating a tool_class_assignment field that reflects what the
agent ACTUALLY did during execution, not what it SHOULD have done.
"""

import logging
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger("tool_class_assigner")


class ToolClassAssigner:
    """
    Reconstructs tool-class assignments from agent execution metadata and results.
    
    This class analyzes the execution trace to determine:
    - Which classes were detected by perception tools
    - Which classes were used by spatial tools (buffer, overlap, etc.)
    - Which classes were counted by object_count_aoi
    - GSD and sensor information from the query
    """
    
    def __init__(self):
        """Initialize the tool class assigner."""
        self.logger = logging.getLogger("tool_class_assigner")
    
    def generate_assignment(
        self,
        execution_metadata: List[Dict[str, Any]],
        perception_results: Dict[str, Any],
        tool_results_storage: Dict[str, List[Dict[str, Any]]],
        input_query: str
    ) -> Dict[str, Any]:
        """
        Generate tool_class_assignment from execution trace.
        
        Args:
            execution_metadata: List of execution metadata for each step
            perception_results: Perception tool results with detected classes
            tool_results_storage: Storage of all tool results
            input_query: Original user query
            
        Returns:
            Dictionary with tool_class_assignment structure
        """
        assignment = {}
        
        # Extract classes from perception tools
        detection_classes = self._extract_detection_classes(perception_results, tool_results_storage)
        if detection_classes:
            assignment["detection_classes"] = detection_classes
        
        # Extract classes used by spatial tools from execution metadata
        buffer_classes = self._extract_buffer_classes(execution_metadata, tool_results_storage)
        if buffer_classes:
            assignment["buffer_classes"] = buffer_classes
        
        overlap_classes = self._extract_overlap_classes(execution_metadata, tool_results_storage)
        if overlap_classes:
            assignment["overlap_classes"] = overlap_classes
        
        object_count_aoi_classes = self._extract_object_count_aoi_classes(execution_metadata, tool_results_storage)
        if object_count_aoi_classes:
            assignment["object_count_aoi_classes"] = object_count_aoi_classes
        
        containment_classes = self._extract_containment_classes(execution_metadata, tool_results_storage)
        if containment_classes:
            assignment["containment_classes"] = containment_classes
        
        # Extract parameters from query
        params = self._extract_parameters(input_query)
        if params:
            assignment["params"] = params
        
        return assignment
    
    def _extract_detection_classes(
        self,
        perception_results: Dict[str, Any],
        tool_results_storage: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Extract classes detected by perception tools."""
        classes = []
        
        # Try perception_results first (backward compatibility)
        if perception_results:
            for tool_name in ["detection", "segmentation", "classification"]:
                if tool_name in perception_results:
                    detected = perception_results[tool_name].get("classes_detected", [])
                    if detected:
                        classes.extend(detected)
        
        # Fallback to tool_results_storage
        if not classes and tool_results_storage:
            for tool_name in ["detection", "segmentation", "classification"]:
                if tool_name in tool_results_storage and tool_results_storage[tool_name]:
                    latest_result = tool_results_storage[tool_name][-1].get("result", {})
                    detected = latest_result.get("classes_detected", [])
                    if detected:
                        classes.extend(detected)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_classes = []
        for c in classes:
            if c not in seen:
                seen.add(c)
                unique_classes.append(c)
        
        return unique_classes
    
    def _extract_buffer_classes(
        self,
        execution_metadata: List[Dict[str, Any]],
        tool_results_storage: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Extract classes used by buffer tool."""
        classes = []
        
        for metadata in execution_metadata:
            if metadata.get("tool_name") == "buffer":
                tool_args = metadata.get("tool_args", {})
                if "buffer_class" in tool_args:
                    classes.append(tool_args["buffer_class"])
        
        return list(set(classes))  # Remove duplicates
    
    def _extract_overlap_classes(
        self,
        execution_metadata: List[Dict[str, Any]],
        tool_results_storage: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Extract classes used by overlap tool."""
        classes = []
        
        for metadata in execution_metadata:
            if metadata.get("tool_name") == "overlap":
                tool_args = metadata.get("tool_args", {})
                if "source_class" in tool_args:
                    classes.append(tool_args["source_class"])
                if "target_class" in tool_args:
                    classes.append(tool_args["target_class"])
        
        return list(set(classes))  # Remove duplicates
    
    def _extract_object_count_aoi_classes(
        self,
        execution_metadata: List[Dict[str, Any]],
        tool_results_storage: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Extract classes used by object_count_aoi tool (the object_class to count)."""
        classes = []
        
        for metadata in execution_metadata:
            if metadata.get("tool_name") == "object_count_aoi":
                tool_args = metadata.get("tool_args", {})
                if "object_class" in tool_args:
                    classes.append(tool_args["object_class"])
        
        return list(set(classes))  # Remove duplicates
    
    def _extract_containment_classes(
        self,
        execution_metadata: List[Dict[str, Any]],
        tool_results_storage: Dict[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """Extract classes used by containment tool."""
        classes = []
        
        for metadata in execution_metadata:
            if metadata.get("tool_name") == "containment":
                tool_args = metadata.get("tool_args", {})
                if "container_class" in tool_args:
                    classes.append(tool_args["container_class"])
                if "contained_class" in tool_args:
                    classes.append(tool_args["contained_class"])
        
        return list(set(classes))  # Remove duplicates
    
    def _extract_parameters(self, input_query: str) -> Dict[str, Any]:
        """Extract GSD and sensor information from query."""
        params = {}
        
        # Extract GSD
        gsd_match = re.search(r'gsd\s*[=:]\s*([0-9.]+)\s*m\s*/\s*px', input_query.lower())
        if gsd_match:
            params["gsd"] = float(gsd_match.group(1))
        else:
            # Try other GSD patterns
            gsd_patterns = [
                r'\(([0-9.]+)\s*m\s*/\s*px\)',
                r'([0-9.]+)\s*m\s*/\s*pixel',
                r'resolution[:\s]*([0-9.]+)\s*m'
            ]
            for pattern in gsd_patterns:
                match = re.search(pattern, input_query.lower())
                if match:
                    params["gsd"] = float(match.group(1))
                    break
        
        # Extract sensor type
        if "optical" in input_query.lower():
            params["sensor"] = "optical"
        elif "sar" in input_query.lower():
            params["sensor"] = "SAR"
        elif "infrared" in input_query.lower() or "ir" in input_query.lower():
            params["sensor"] = "infrared"
        else:
            params["sensor"] = "optical"  # Default
        
        return params if params else {}

