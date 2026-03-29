"""
Tool registry for the spatial reasoning agent.

This module provides classes for tool registration, discovery, and toolkit management,
separating these concerns from the main planning logic.
"""

import logging
from typing import List, Dict, Any, Optional
from .tool_models import Tool, Toolkit, ToolkitList

# Setup logger
logger = logging.getLogger("tool_registry")


class ToolRegistry:
    """
    Manages tool registration, discovery, and toolkit creation.
    
    This class provides a centralized way to manage available tools,
    create toolkits, and handle tool discovery without embedding
    this logic in the main planner.
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self._registered_tools = {}
        self._toolkits = {}
        self._default_toolkit_definitions = self._get_default_toolkit_definitions()
        
    def _get_default_toolkit_definitions(self) -> List[Dict[str, Any]]:
        """Get default toolkit definitions for all 5 complete toolkits with dependency metadata."""
        return [
            {
                "name": "perception",
                "tools": [
                    {
                        "name": "segmentation",
                        "category": "perception",
                        "description": "PERCEPTION TOOL: Extracts object boundaries from satellite imagery using text prompts. Converts semantic descriptions into polygon coordinate arrays for spatial analysis tools.",
                        "dependencies": []  # Perception tools have no dependencies
                    },
                    {
                        "name": "detection",
                        "category": "perception",
                        "description": "PERCEPTION TOOL: Locates objects in satellite imagery using text prompts. Converts semantic descriptions into bounding box coordinate arrays for spatial analysis tools.",
                        "dependencies": []  # Perception tools have no dependencies
                    },
                    {
                        "name": "classification",
                        "category": "perception",
                        "description": "PERCEPTION TOOL: Classifies regions in satellite imagery using text prompts. Converts semantic descriptions into classified region coordinate arrays for spatial analysis tools.",
                        "dependencies": []  # Perception tools have no dependencies
                    },
                    {
                        "name": "change_detection",
                        "category": "perception",
                        "description": "PERCEPTION TOOL: Detects semantic changes between bi-temporal satellite images. Identifies changed regions with semantic class transitions using Change3D model.",
                        "dependencies": []  # Perception tools have no dependencies
                    }
                ]
            },
            {
                "name": "spatial_relations",
                "tools": [
                    {
                        "name": "buffer",
                        "category": "spatial_relations",
                        "description": "SPATIAL RELATION TOOL: Creates buffer zones around geometries. Works best with coordinate arrays from perception tools. Flexible input handling.",
                        "dependencies": ["perception"]  # Requires perception results
                    },
                    {
                        "name": "overlap",
                        "category": "spatial_relations",
                        "description": "SPATIAL RELATION TOOL: Calculates overlap metrics between polygons. Works best with coordinate arrays from perception tools. Flexible input handling.",
                        "dependencies": ["perception"]  # Requires perception results
                    },
                    {
                        "name": "containment",
                        "category": "spatial_relations",
                        "description": "SPATIAL RELATION TOOL: Determines containment relationships between geometries. Works best with coordinate arrays from perception tools. Flexible input handling.",
                        "dependencies": ["perception"]  # Requires perception results
                    }
                ]
            },
            {
                "name": "spatial_statistics",
                "tools": [
                    {
                        "name": "distance_calculation",
                        "category": "spatial_statistics",
                        "description": "STATISTICAL TOOL: Calculates distances between geometries. Works best with coordinate arrays from perception tools for quantitative analysis.",
                        "dependencies": ["perception"]  # Requires perception results
                    },
                    {
                        "name": "area_measurement",
                        "category": "spatial_statistics",
                        "description": "STATISTICAL TOOL: Computes area of polygons from coordinate data. Works best with coordinate arrays from perception tools for quantitative analysis.",
                        "dependencies": ["perception"]  # Requires perception results
                    },
                    {
                        "name": "object_count_aoi",
                        "category": "spatial_statistics",
                        "description": "STATISTICAL TOOL: Counts objects within areas from coordinate data. Works best with coordinate arrays from perception tools for quantitative analysis.",
                        "dependencies": ["perception"]  # Requires perception results
                    }
                ]
            },
            {
                "name": "sar",
                "tools": [
                    {
                        "name": "sar_detection",
                        "category": "sar_tools",
                        "description": "SAR TOOL: Detects objects in SAR (Synthetic Aperture Radar) imagery using HiViT-based SARATR-X model. Specialized for maritime surveillance and ship detection in radar imagery.",
                        "dependencies": []  # SAR detection is a perception tool
                    },
                    {
                        "name": "sar_classification",
                        "category": "sar_tools",
                        "description": "SAR TOOL: Classifies SAR imagery into scene types, target categories, or fine-grained classes using SARATR-X model. Supports scene, target, and fine-grained classification for radar data.",
                        "dependencies": []  # SAR classification is a perception tool
                    }
                ]
            },
            {
                "name": "ir",
                "tools": [
                    {
                        "name": "infrared_detection",
                        "category": "ir_tools",
                        "description": "IR TOOL: Detects small infrared targets in satellite/aerial imagery using DMIST (Deep Multi-scale Infrared Small Target) framework. Specialized for detecting small moving targets in infrared imagery using temporal sequence analysis.",
                        "dependencies": []  # IR detection is a perception tool
                    }
                ]
            }
        ]
    
    def register_tool(self, name: str, category: str, description: str, dependencies: Optional[List[str]] = None) -> Tool:
        """
        Register a new tool.

        Args:
            name: Tool name
            category: Tool category
            description: Tool description
            dependencies: Optional list of tool dependencies (e.g., ["perception"])

        Returns:
            Created Tool instance
        """
        tool = Tool(name, category, description, dependencies=dependencies)
        tool_key = f"{category}_{name}"
        self._registered_tools[tool_key] = tool
        logger.info(f"Registered tool: {tool_key} with dependencies: {dependencies or []}")
        return tool
    
    def get_tool(self, name: str, category: str = None) -> Optional[Tool]:
        """
        Get a registered tool by name and optional category.
        
        Args:
            name: Tool name
            category: Optional tool category
            
        Returns:
            Tool instance if found, None otherwise
        """
        if category:
            tool_key = f"{category}_{name}"
            return self._registered_tools.get(tool_key)
        else:
            # Search across all categories
            for tool_key, tool in self._registered_tools.items():
                if tool.api_dest["name"] == name:
                    return tool
            return None
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Tool category
            
        Returns:
            List of tools in the category
        """
        return [tool for tool_key, tool in self._registered_tools.items() 
                if tool.api_dest["type_name"] == category]
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._registered_tools.values())
    
    def create_toolkit(self, name: str, tool_names: List[str] = None, 
                      categories: List[str] = None) -> Toolkit:
        """
        Create a toolkit with specified tools.
        
        Args:
            name: Toolkit name
            tool_names: Optional list of specific tool names to include
            categories: Optional list of categories to include all tools from
            
        Returns:
            Created Toolkit instance
        """
        tools = []
        
        if tool_names:
            for tool_name in tool_names:
                tool = self.get_tool(tool_name)
                if tool:
                    tools.append(tool)
                else:
                    logger.warning(f"Tool '{tool_name}' not found in registry")
        
        if categories:
            for category in categories:
                category_tools = self.get_tools_by_category(category)
                tools.extend(category_tools)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in tools:
            tool_key = f"{tool.api_dest['type_name']}_{tool.api_dest['name']}"
            if tool_key not in seen:
                seen.add(tool_key)
                unique_tools.append(tool)
        
        toolkit = Toolkit(name, unique_tools)
        self._toolkits[name] = toolkit
        logger.info(f"Created toolkit '{name}' with {len(unique_tools)} tools")
        return toolkit
    
    def create_default_toolkits(self, toolkit_num: int = 5) -> ToolkitList:
        """
        Create default toolkits based on predefined definitions.

        Args:
            toolkit_num: Number of toolkits to create (default: 5 for all complete toolkits)

        Returns:
            ToolkitList instance with default toolkits
        """
        # First, register all default tools with their dependencies
        for toolkit_def in self._default_toolkit_definitions:
            for tool_def in toolkit_def["tools"]:
                self.register_tool(
                    tool_def["name"],
                    tool_def["category"],
                    tool_def["description"],
                    dependencies=tool_def.get("dependencies", [])  # Pass dependencies from definition
                )
        
        # Create toolkits
        toolkits = []
        for i in range(min(toolkit_num, len(self._default_toolkit_definitions))):
            toolkit_def = self._default_toolkit_definitions[i]
            # Map toolkit names to their actual category names
            # "sar" toolkit uses "sar_tools" category, "ir" toolkit uses "ir_tools" category
            category_name = toolkit_def["name"]
            if category_name == "sar":
                category_name = "sar_tools"
            elif category_name == "ir":
                category_name = "ir_tools"

            toolkit = self.create_toolkit(
                toolkit_def["name"],
                categories=[category_name]
            )
            toolkits.append(toolkit)
        
        # If toolkit_num is greater than available definitions, repeat the last toolkit
        while len(toolkits) < toolkit_num:
            last_def = self._default_toolkit_definitions[-1]
            toolkit_name = f"{last_def['name']}_{len(toolkits)}"
            toolkit = self.create_toolkit(
                toolkit_name,
                categories=[last_def["name"]]
            )
            toolkits.append(toolkit)
        
        toolkit_list = ToolkitList.__new__(ToolkitList)  # Create without calling __init__
        toolkit_list.toolkit_num = toolkit_num
        toolkit_list.tool_kits = toolkits
        
        logger.info(f"Created {len(toolkits)} default toolkits")
        return toolkit_list
    
    def get_toolkit(self, name: str) -> Optional[Toolkit]:
        """Get a toolkit by name."""
        return self._toolkits.get(name)
    
    def list_toolkits(self) -> List[str]:
        """Get list of all toolkit names."""
        return list(self._toolkits.keys())
    
    def get_tool_count(self) -> int:
        """Get total number of registered tools."""
        return len(self._registered_tools)
    
    def get_category_counts(self) -> Dict[str, int]:
        """Get count of tools by category."""
        counts = {}
        for tool in self._registered_tools.values():
            category = tool.api_dest["type_name"]
            counts[category] = counts.get(category, 0) + 1
        return counts
