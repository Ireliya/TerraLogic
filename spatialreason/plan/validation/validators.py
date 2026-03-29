"""
Validation module for spatialreason.plan

This module contains validation and error checking functionality for the planner.

Classes:
    EvaluationModeValidator: Validates evaluation mode and prevents mock fallbacks
    DependencyValidator: Validates tool dependencies and enforces execution order constraints
"""

import json
import logging
import os
import sys
from typing import Optional, Dict, Any, List

# Setup module logger
logger = logging.getLogger(__name__)


class EvaluationModeValidator:
    """
    Validator for evaluation mode detection and response validation.
    
    This class handles:
    - Detection of evaluation mode based on environment indicators
    - Validation of tool responses
    - Fallback toolkit prompt generation
    """
    
    def __init__(self):
        """Initialize the evaluation mode validator."""
        self.evaluation_mode = None
    
    def detect_evaluation_mode(self, evaluation_mode: Optional[bool] = None) -> bool:
        """
        Detect if we're running in evaluation mode to control mock fallbacks.
        
        Args:
            evaluation_mode: Explicit mode setting (True/False/None)
        
        Returns:
            Boolean indicating if evaluation mode is active
        """
        if evaluation_mode is not None:
            self.evaluation_mode = evaluation_mode
            return evaluation_mode
        
        # Auto-detection based on environment indicators
        evaluation_indicators = [
            'OPENCOMPASS_EVAL' in os.environ,
            'spatial_reasoning_eval' in os.getcwd(),
            'opencompass' in ' '.join(sys.argv),
            any('eval' in arg.lower() for arg in sys.argv),
            'benchmark' in os.getcwd().lower(),
            'evaluation' in os.getcwd().lower()
        ]
        
        is_evaluation = any(evaluation_indicators)
        self.evaluation_mode = is_evaluation
        
        if is_evaluation:
            logger.info("🔍 Auto-detected EVALUATION mode based on environment indicators")
        else:
            logger.info("🔍 Auto-detected DEVELOPMENT mode - mock fallbacks available")
        
        return is_evaluation
    
    def is_response_successful(self, response: str) -> bool:
        """
        Reliable method to check if a response indicates success.
        Uses only JSON parsing for accurate error detection.
        
        Args:
            response: JSON response string to check
        
        Returns:
            bool: True if response is successful, False if error occurred
        """
        try:
            parsed_response = json.loads(response)
            error_value = parsed_response.get("error")
            # Consider response successful if error is None, empty string, or False
            return not error_value
        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, consider it an error
            return False
    
    def generate_fallback_toolkit_prompt(self, toolkit_list) -> str:
        """
        Generate fallback toolkit prompt with consistent toolkit ID mapping.
        
        Args:
            toolkit_list: The toolkit list object containing available toolkits
        
        Returns:
            str: Formatted fallback toolkit prompt
        """
        fallback_prompt = "Available toolkits and their tools:\n"
        
        # Use consistent toolkit ID mapping
        for i, toolkit in enumerate(toolkit_list.tool_kits):
            toolkit_name = "Perception" if i == 0 else "Spatial Relations" if i == 1 else "Analysis"
            tool_names = [tool.api_dest["name"] for tool in toolkit.tool_lists]
            fallback_prompt += f"Toolkit {i} ({toolkit_name}): {', '.join(tool_names)}\n"
        
        return fallback_prompt
    
    def is_evaluation_mode_active(self) -> bool:
        """
        Check if evaluation mode is currently active.

        Returns:
            bool: True if evaluation mode is active, False otherwise
        """
        if self.evaluation_mode is None:
            self.detect_evaluation_mode()
        return self.evaluation_mode


class DependencyValidator:
    """
    Validator for tool dependency constraints.

    This class enforces that spatial relation tools and statistical tools
    cannot execute without prior perception tool execution. It implements
    Solution 1 (Dependency Metadata) of the hybrid dependency constraint
    enforcement strategy.

    Dependency Rules:
    - Spatial relation tools (buffer, overlap, containment) require perception results
    - Statistical tools (count, area_measurement, distance_calculation, object_count_aoi) require perception results
    - Perception tools (segmentation, detection, classification, sar_detection, sar_classification, infrared_detection) have no dependencies
    """

    # Define which tools require perception results
    TOOLS_REQUIRING_PERCEPTION = {
        # Spatial relation tools
        "buffer", "overlap", "containment",
        # Statistical tools
        "count", "area_measurement", "distance_calculation", "object_count_aoi",
        "distance_tool"  # Alternative name for distance_calculation
    }

    # Define which tools are perception tools
    PERCEPTION_TOOLS = {
        # Optical perception tools
        "segmentation", "detection", "classification", "change_detection",
        # SAR perception tools
        "sar_detection", "sar_classification",
        # IR perception tools
        "infrared_detection"
    }

    def __init__(self):
        """Initialize the dependency validator."""
        self.logger = logging.getLogger(__name__)

    def validate_tool_dependencies(self, tool_name: str, perception_results: Dict[str, Any]) -> None:
        """
        Validate that a tool's dependencies are satisfied before execution.

        Args:
            tool_name: Name of the tool to validate
            perception_results: Dictionary of stored perception results

        Raises:
            RuntimeError: If tool dependencies are not satisfied
        """
        # DEBUG: Log what we're checking
        self.logger.info(f"🔍 DEBUG [validate_tool_dependencies]: Checking dependencies for '{tool_name}'")
        self.logger.info(f"   Tool requires perception: {tool_name in self.TOOLS_REQUIRING_PERCEPTION}")
        self.logger.info(f"   Current perception_results keys: {list(perception_results.keys()) if perception_results else 'None'}")
        self.logger.info(f"   Number of perception results: {len(perception_results) if perception_results else 0}")

        # Check if this tool requires perception results
        if tool_name in self.TOOLS_REQUIRING_PERCEPTION:
            # Check if perception results are available
            if not perception_results or len(perception_results) == 0:
                error_msg = (
                    f"❌ DEPENDENCY VIOLATION: Tool '{tool_name}' requires perception results "
                    f"but none are available. Perception tools must execute BEFORE spatial relation "
                    f"or statistical tools. Available perception tools: {', '.join(self.PERCEPTION_TOOLS)}"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.logger.info(
                f"✅ Dependency check passed for '{tool_name}': "
                f"Perception results available ({len(perception_results)} result(s))"
            )

    def validate_tool_sequence(self, tool_sequence: List[str]) -> bool:
        """
        Validate that a tool sequence respects dependency constraints.

        Args:
            tool_sequence: List of tool names in execution order

        Returns:
            bool: True if sequence is valid, False otherwise
        """
        perception_executed = False

        for i, tool_name in enumerate(tool_sequence):
            # Check if this is a perception tool
            if tool_name in self.PERCEPTION_TOOLS:
                perception_executed = True
                self.logger.debug(f"Step {i}: Perception tool '{tool_name}' - OK")

            # Check if this tool requires perception
            elif tool_name in self.TOOLS_REQUIRING_PERCEPTION:
                if not perception_executed:
                    error_msg = (
                        f"❌ INVALID SEQUENCE: Tool '{tool_name}' at position {i} requires "
                        f"perception tool execution first. Perception tools must come before "
                        f"spatial relation or statistical tools."
                    )
                    self.logger.error(error_msg)
                    return False
                self.logger.debug(f"Step {i}: Spatial/Statistical tool '{tool_name}' - OK (perception executed)")

            else:
                self.logger.debug(f"Step {i}: Unknown tool '{tool_name}' - skipping validation")

        return True

    def get_dependency_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get dependency information for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dictionary with dependency information
        """
        if tool_name in self.PERCEPTION_TOOLS:
            return {
                "tool_name": tool_name,
                "tool_type": "perception",
                "dependencies": [],
                "requires_perception": False
            }
        elif tool_name in self.TOOLS_REQUIRING_PERCEPTION:
            return {
                "tool_name": tool_name,
                "tool_type": "spatial_relation" if tool_name in ["buffer", "overlap", "containment"] else "statistical",
                "dependencies": ["perception"],
                "requires_perception": True
            }
        else:
            return {
                "tool_name": tool_name,
                "tool_type": "unknown",
                "dependencies": [],
                "requires_perception": False
            }

