"""
Workflow state management for the spatial reasoning agent.

This module provides classes for tracking workflow state and execution history
for informational and analysis purposes, without enforcing any execution constraints.
"""

import logging
from typing import Dict, List, Any, Optional

# Setup logger
logger = logging.getLogger("workflow_state")


class WorkflowStateManager:
    """
    Manages workflow state tracking for informational purposes.
    
    This class tracks tool execution history, results, and workflow phases
    without enforcing any execution constraints or dependencies.
    """
    
    def __init__(self):
        """Initialize workflow state manager."""
        self.reset_state()
        
    def reset_state(self):
        """Reset workflow state for a new planning session."""
        try:
            self.state = {
                "perception_results": {},  # Store results from perception tools
                "executed_tools": [],      # Track execution order for analysis
                "tool_execution_history": []  # Track detailed execution history
            }
            logger.info("🔄 Workflow state reset for new planning session")
        except Exception as e:
            logger.error(f"Failed to reset workflow state: {e}")
    
    def update_state(self, tool_name: str, execution_result: str, success: bool):
        """
        Update workflow state after tool execution for informational tracking.
        
        Args:
            tool_name: Name of the executed tool
            execution_result: Result from tool execution
            success: Whether the tool executed successfully
        """
        try:
            # Add to executed tools list
            self.state["executed_tools"].append(tool_name)
            
            # Add detailed execution history
            execution_entry = {
                "tool_name": tool_name,
                "success": success,
                "timestamp": len(self.state["tool_execution_history"]) + 1
            }
            self.state["tool_execution_history"].append(execution_entry)
            
            # Define tool categories for informational logging
            perception_tools = {"segmentation", "detection", "classification"}
            spatial_relation_tools = {"buffer", "overlap", "containment", "distance_calculation"}
            
            if tool_name in perception_tools and success:
                # Store perception results for informational purposes
                self.state["perception_results"][tool_name] = execution_result
                logger.info(f"📊 Perception tool '{tool_name}' completed successfully")
            elif tool_name in perception_tools and not success:
                logger.info(f"📊 Perception tool '{tool_name}' execution failed")
            elif tool_name in spatial_relation_tools:
                logger.info(f"📊 Spatial relation tool '{tool_name}' executed with success: {success}")
            
            # Log current workflow state for informational purposes
            executed_perception = [t for t in self.state["executed_tools"] if t in perception_tools]
            executed_spatial = [t for t in self.state["executed_tools"] if t in spatial_relation_tools]
            logger.debug(f"📊 Workflow tracking: {len(executed_perception)} perception tools, {len(executed_spatial)} spatial tools executed")
            
        except Exception as e:
            logger.error(f"Failed to update workflow state: {e}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current workflow state for informational purposes.
        
        Returns:
            Dictionary containing workflow state information
        """
        try:
            perception_tools = {"segmentation", "detection", "classification"}
            spatial_relation_tools = {"buffer", "overlap", "containment", "distance_calculation"}
            
            executed_tools = self.state.get("executed_tools", [])
            perception_results = self.state.get("perception_results", {})
            execution_history = self.state.get("tool_execution_history", [])
            
            executed_perception = [t for t in executed_tools if t in perception_tools]
            executed_spatial = [t for t in executed_tools if t in spatial_relation_tools]
            
            summary = {
                "total_executed_tools": len(executed_tools),
                "perception_tools_executed": len(executed_perception),
                "spatial_relation_tools_executed": len(executed_spatial),
                "perception_results_count": len(perception_results),
                "executed_perception_tools": executed_perception,
                "executed_spatial_tools": executed_spatial,
                "execution_history_length": len(execution_history),
                "workflow_phase": self._determine_workflow_phase()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get workflow state summary: {e}")
            return {"error": str(e)}
    
    def _determine_workflow_phase(self) -> str:
        """
        Determine the current phase of the workflow based on executed tools for informational purposes.
        
        Returns:
            String indicating the current workflow phase
        """
        try:
            perception_tools = {"segmentation", "detection", "classification"}
            spatial_relation_tools = {"buffer", "overlap", "containment", "distance_calculation"}
            
            executed_tools = self.state.get("executed_tools", [])
            
            executed_perception = any(t in perception_tools for t in executed_tools)
            executed_spatial = any(t in spatial_relation_tools for t in executed_tools)
            
            if not executed_perception and not executed_spatial:
                return "initialization"
            elif executed_perception and not executed_spatial:
                return "perception_phase"
            elif executed_perception and executed_spatial:
                return "mixed_analysis_phase"
            elif not executed_perception and executed_spatial:
                return "spatial_analysis_phase"
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Failed to determine workflow phase: {e}")
            return "error"
    
    def get_executed_tools(self) -> List[str]:
        """Get list of executed tools."""
        return self.state.get("executed_tools", [])
    
    def get_perception_results(self) -> Dict[str, str]:
        """Get perception tool results."""
        return self.state.get("perception_results", {})
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get detailed execution history."""
        return self.state.get("tool_execution_history", [])
    
    def has_executed_tool(self, tool_name: str) -> bool:
        """Check if a specific tool has been executed."""
        return tool_name in self.state.get("executed_tools", [])
    
    def get_tool_category_counts(self) -> Dict[str, int]:
        """Get counts of executed tools by category."""
        perception_tools = {"segmentation", "detection", "classification"}
        spatial_relation_tools = {"buffer", "overlap", "containment"}
        spatial_statistics_tools = {"distance_calculation", "area_measurement", "object_count_aoi"}

        executed_tools = self.state.get("executed_tools", [])

        return {
            "perception": len([t for t in executed_tools if t in perception_tools]),
            "spatial_relations": len([t for t in executed_tools if t in spatial_relation_tools]),
            "spatial_statistics": len([t for t in executed_tools if t in spatial_statistics_tools]),
            "other": len([t for t in executed_tools if t not in (perception_tools | spatial_relation_tools | spatial_statistics_tools)])
        }
