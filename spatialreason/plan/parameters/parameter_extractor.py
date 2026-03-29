"""
Parameter extraction module for spatialreason.plan

This module contains the ParameterExtractor class which handles extraction of
tool parameters from query context and execution results.

Classes:
    ParameterExtractor: Extracts tool parameters from various sources
"""

import json
import logging
import os
import re
from typing import Optional, Dict, Any, List

# Setup module logger
logger = logging.getLogger(__name__)


class ParameterExtractor:
    """
    Extracts tool parameters from query context and execution results.
    
    This class handles parameter extraction for:
    - Perception tools (detection, segmentation, classification)
    - Spatial analysis tools (buffer, overlap, containment, etc.)
    - Analysis tools
    """
    
    def __init__(self, input_query: str = "", perception_results: Dict = None, 
                 tool_results_storage: Dict = None, current_image_path: str = ""):
        """
        Initialize the parameter extractor.
        
        Args:
            input_query: The input query string
            perception_results: Dictionary of perception tool results
            tool_results_storage: Dictionary of all tool results
            current_image_path: Path to the current image being processed
        """
        self.input_query = input_query
        self.perception_results = perception_results or {}
        self.tool_results_storage = tool_results_storage or {}
        self.current_image_path = current_image_path
    
    def extract_distance_from_query(self, query_text: str = None) -> Optional[float]:
        """
        Extract distance value from natural language query text.
        
        Args:
            query_text: Natural language query string (uses self.input_query if None)
        
        Returns:
            Distance in meters, or None if no distance found
        """
        query_text = query_text or self.input_query
        if not query_text:
            return None
        
        # Convert to lowercase for pattern matching
        query_lower = query_text.lower()
        
        # Regex patterns for distance extraction
        distance_patterns = [
            r'within\s+(\d+(?:\.\d+)?)\s*(?:m|meters?)\s+(?:of|from)',
            r'(\d+(?:\.\d+)?)\s*(?:m|meters?)\s+(?:of|from)',
            r'(\d+(?:\.\d+)?)\s*(?:m|meters?)\s+(?:away|distance)',
            r'within\s+(\d+(?:\.\d+)?)\s*(?:m|meters?)',
            r'(\d+(?:\.\d+)?)\s*(?:m|meter|meters?)\s+buffer'
        ]
        
        # Try each pattern in order
        for pattern in distance_patterns:
            match = re.search(pattern, query_lower)
            if match:
                try:
                    distance = float(match.group(1))
                    logger.debug(f"Extracted distance {distance}m using pattern: {pattern}")
                    return distance
                except (ValueError, IndexError):
                    continue
        
        logger.debug(f"No distance pattern matched in query: {query_text[:100]}...")
        return None
    
    def extract_image_path_from_context(self) -> Optional[str]:
        """Extract image path from the input query context."""
        # First, try to use the stored image path
        if self.current_image_path:
            logger.info(f"✅ Using stored image path: {self.current_image_path}")
            return self.current_image_path
        
        # Try to extract from query
        if self.input_query:
            # Look for common image path patterns
            patterns = [
                r'(?:image|file|path):\s*([^\s,\]]+\.(?:jpg|png|tif|tiff))',
                r'([^\s,\]]+\.(?:jpg|png|tif|tiff))',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, self.input_query, re.IGNORECASE)
                if match:
                    path = match.group(1)
                    if os.path.exists(path):
                        logger.info(f"Extracted image path from query: {path}")
                        return path
        
        logger.warning("No image path found in context")
        return None
    
    def extract_text_prompt_from_context(self, tool_name: str = "") -> Optional[str]:
        """Extract a text prompt from the query context."""
        if not self.input_query:
            return None
        
        # Extract keywords from query
        keywords = re.findall(r'\b[a-z]+(?:\s+[a-z]+)?\b', self.input_query.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'is', 'are', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'must', 'can', 'detect',
                     'segment', 'classify', 'find', 'identify', 'locate', 'count'}
        
        meaningful_keywords = [kw for kw in keywords if kw not in stop_words and len(kw) > 2]
        
        if meaningful_keywords:
            prompt = ' '.join(meaningful_keywords[:3])
            logger.info(f"Extracted text prompt: {prompt}")
            return prompt
        
        return None
    
    def extract_enhanced_tool_arguments(self, tool_name: str, step_plan: str) -> Dict[str, Any]:
        """
        Extract enhanced tool arguments from tool name and step plan.
        
        Args:
            tool_name: Name of the tool
            step_plan: The step plan description
        
        Returns:
            Dictionary of tool arguments
        """
        arguments = {
            "image_path": self.extract_image_path_from_context(),
            "text_prompt": self.extract_text_prompt_from_context(tool_name),
            "meters_per_pixel": 0.3,
        }
        
        # Add tool-specific parameters
        if tool_name in ["detection", "segmentation", "classification"]:
            arguments["confidence_threshold"] = 0.5
        
        if "buffer" in tool_name.lower():
            distance = self.extract_distance_from_query()
            if distance:
                arguments["distance"] = distance
        
        return arguments
    
    def extract_class_from_perception_results(self, tool_name: str = None) -> Optional[str]:
        """
        Extract class name from perception results.
        
        Args:
            tool_name: Optional tool name to filter results
        
        Returns:
            Class name or None
        """
        if not self.perception_results:
            return None
        
        # Get first available class from perception results
        for tool, result in self.perception_results.items():
            if isinstance(result, dict) and "classes_detected" in result:
                classes = result.get("classes_detected", [])
                if classes:
                    return classes[0]
        
        return None
    
    def extract_geometries_from_sources(self, source_tools: List[str]) -> List[List[List[float]]]:
        """
        Extract geometries from multiple tool sources in priority order.
        
        Args:
            source_tools: List of tool names to extract geometries from
        
        Returns:
            List of geometry coordinates
        """
        geometries = []
        
        for tool_name in source_tools:
            if tool_name in self.tool_results_storage:
                result = self.tool_results_storage[tool_name]
                if isinstance(result, dict):
                    geoms = self._extract_geometries_from_tool_result(tool_name, result)
                    geometries.extend(geoms)
        
        return geometries
    
    def _extract_geometries_from_tool_result(self, tool_name: str, result: Dict) -> List[List[List[float]]]:
        """
        Extract geometry coordinates from a specific tool result.
        
        Args:
            tool_name: Name of the tool
            result: Tool result dictionary
        
        Returns:
            List of geometry coordinates
        """
        geometries = []
        
        # Try different geometry field names
        geometry_fields = ["geometries", "segments", "detections", "polygons", "coordinates"]
        
        for field in geometry_fields:
            if field in result:
                geoms = result[field]
                if isinstance(geoms, list):
                    geometries.extend(geoms)
        
        return geometries

