"""
Workflow-aware filtering for the spatial reasoning agent.

This module provides workflow-aware filtering that analyzes tool execution
patterns for informational purposes without enforcing any constraints.
"""

import logging
from typing import List, Dict, Any, Optional
from spatialreason.plan.workflow import WorkflowStateManager

# Setup logger
logger = logging.getLogger("workflow_filter")


class WorkflowAwareFilter:
    """
    Workflow-aware filter that analyzes tool execution patterns.
    
    This class provides informational analysis of tool usage patterns
    without enforcing any execution constraints or dependencies.
    """
    
    def __init__(self, workflow_state_manager: Optional[WorkflowStateManager] = None):
        """
        Initialize workflow-aware filter.
        
        Args:
            workflow_state_manager: Optional workflow state manager for analysis
        """
        self.workflow_state_manager = workflow_state_manager
        
        # Define tool categories for analysis
        self.perception_tools = {"segmentation", "detection", "classification"}
        self.spatial_relation_tools = {"buffer", "overlap", "containment"}
        # DISABLED FOR EVALUATION: Spatial statistics tools excluded from current evaluation round
        # self.spatial_statistics_tools = {"distance_calculation", "area_measurement", "length_measurement", "object_count_aoi"}
    
    def analyze_tool_distribution(self, retrieved_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of tools by category for informational purposes.
        
        Args:
            retrieved_tools: List of retrieved tool dictionaries from semantic search
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Separate tools by category for analysis
            perception_candidates = []
            spatial_relation_candidates = []
            # DISABLED FOR EVALUATION: Spatial statistics tools excluded from current evaluation round
            # spatial_statistics_candidates = []
            other_candidates = []

            # Define disabled spatial statistics tools for filtering
            disabled_spatial_stats = {"distance_calculation", "area_measurement", "length_measurement", "object_count_aoi"}

            for tool in retrieved_tools:
                tool_name = tool.get("tool_name", "").lower()
                if tool_name in self.perception_tools:
                    perception_candidates.append(tool)
                elif tool_name in self.spatial_relation_tools:
                    spatial_relation_candidates.append(tool)
                elif tool_name in disabled_spatial_stats:
                    # Skip spatial statistics tools - they are disabled for this evaluation
                    logger.debug(f"Skipping disabled spatial statistics tool: {tool_name}")
                    continue
                else:
                    other_candidates.append(tool)

            analysis = {
                "total_tools": len(retrieved_tools),
                "perception_count": len(perception_candidates),
                "spatial_relation_count": len(spatial_relation_candidates),
                # "spatial_statistics_count": len(spatial_statistics_candidates),
                "other_count": len(other_candidates),
                "perception_tools": [t.get("tool_name", "") for t in perception_candidates],
                "spatial_relation_tools": [t.get("tool_name", "") for t in spatial_relation_candidates],
                # "spatial_statistics_tools": [t.get("tool_name", "") for t in spatial_statistics_candidates],
                "other_tools": [t.get("tool_name", "") for t in other_candidates]
            }

            # Log analysis for informational purposes
            logger.debug(f"📊 Tool distribution analysis: {analysis['perception_count']} perception, "
                        f"{analysis['spatial_relation_count']} spatial relation, "
                        f"{analysis['other_count']} other tools (spatial statistics disabled)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze tool distribution: {e}")
            return {"error": str(e)}
    
    def analyze_execution_history(self) -> Dict[str, Any]:
        """
        Analyze the execution history for informational purposes.
        
        Returns:
            Dictionary containing execution history analysis
        """
        if not self.workflow_state_manager:
            return {"error": "No workflow state manager available"}
        
        try:
            executed_tools = self.workflow_state_manager.get_executed_tools()
            
            # Categorize executed tools
            executed_perception = [t for t in executed_tools if t in self.perception_tools]
            executed_spatial_relation = [t for t in executed_tools if t in self.spatial_relation_tools]
            # DISABLED FOR EVALUATION: Spatial statistics tools excluded from current evaluation round
            # executed_spatial_statistics = [t for t in executed_tools if t in self.spatial_statistics_tools]
            disabled_spatial_stats = {"distance_calculation", "area_measurement", "length_measurement", "object_count_aoi"}
            executed_other = [t for t in executed_tools if t not in (
                self.perception_tools | self.spatial_relation_tools | disabled_spatial_stats
            )]

            analysis = {
                "total_executed": len(executed_tools),
                "perception_executed": len(executed_perception),
                "spatial_relation_executed": len(executed_spatial_relation),
                # "spatial_statistics_executed": len(executed_spatial_statistics),
                "other_executed": len(executed_other),
                "executed_perception_tools": executed_perception,
                "executed_spatial_relation_tools": executed_spatial_relation,
                # "executed_spatial_statistics_tools": executed_spatial_statistics,
                "executed_other_tools": executed_other,
                "execution_order": executed_tools
            }

            # Log execution history analysis
            logger.debug(f"📊 Execution history: {analysis['perception_executed']} perception, "
                        f"{analysis['spatial_relation_executed']} spatial relation tools executed "
                        f"(spatial statistics disabled for evaluation)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze execution history: {e}")
            return {"error": str(e)}
    
    def apply_informational_filtering(self, retrieved_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply workflow-aware filtering for informational purposes only.
        No longer enforces dependencies - just logs tool categories for analysis.
        
        Args:
            retrieved_tools: List of retrieved tool dictionaries from semantic search
            
        Returns:
            List of tools in original semantic relevance order
        """
        try:
            # Analyze tool distribution
            tool_analysis = self.analyze_tool_distribution(retrieved_tools)
            
            # Analyze execution history if available
            if self.workflow_state_manager:
                execution_analysis = self.analyze_execution_history()
                
                # Log combined analysis
                logger.debug(f"📊 Combined analysis: Available tools - {tool_analysis.get('perception_count', 0)} perception, "
                           f"{tool_analysis.get('spatial_relation_count', 0)} spatial relation; "
                           f"Executed tools - {execution_analysis.get('perception_executed', 0)} perception, "
                           f"{execution_analysis.get('spatial_relation_executed', 0)} spatial relation")
            
            # Return tools in original semantic relevance order (no dependency-based reordering)
            return retrieved_tools
            
        except Exception as e:
            logger.warning(f"Workflow-aware filtering analysis failed: {e}. Using original order.")
            return retrieved_tools
    
    def get_workflow_recommendations(self) -> Dict[str, Any]:
        """
        Get workflow recommendations based on current state (informational only).
        
        Returns:
            Dictionary containing workflow recommendations
        """
        if not self.workflow_state_manager:
            return {"error": "No workflow state manager available"}
        
        try:
            state_summary = self.workflow_state_manager.get_state_summary()
            
            recommendations = {
                "current_phase": state_summary.get("workflow_phase", "unknown"),
                "tools_executed": state_summary.get("total_executed_tools", 0),
                "perception_completed": state_summary.get("perception_tools_executed", 0),
                "spatial_analysis_completed": state_summary.get("spatial_relation_tools_executed", 0),
                "suggestions": []
            }
            
            # Generate informational suggestions (not constraints)
            if recommendations["perception_completed"] == 0:
                recommendations["suggestions"].append("Consider starting with perception tools to extract object coordinates")
            
            if recommendations["perception_completed"] > 0 and recommendations["spatial_analysis_completed"] == 0:
                recommendations["suggestions"].append("Perception data available - spatial relation tools can now utilize coordinate data")
            
            if recommendations["tools_executed"] == 0:
                recommendations["suggestions"].append("Beginning analysis - consider the logical flow of your spatial reasoning task")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate workflow recommendations: {e}")
            return {"error": str(e)}
