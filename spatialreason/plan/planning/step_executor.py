"""
Step execution module for spatialreason.plan

This module contains the StepExecutor class which handles execution of
individual steps and manages tool invocation.

Classes:
    StepExecutor: Executes individual steps and manages tool invocation
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple

# Setup module logger
logger = logging.getLogger(__name__)


class StepExecutor:
    """
    Executes individual steps and manages tool invocation.
    
    This class handles:
    - Step execution with semantic filtering
    - Tool invocation and result handling
    - Execution metadata tracking
    - Step consistency analysis
    """
    
    def __init__(self, semantic_filter=None, toolkit_list=None):
        """
        Initialize the step executor.
        
        Args:
            semantic_filter: Semantic filter for tool selection
            toolkit_list: Available toolkits
        """
        self.semantic_filter = semantic_filter
        self.toolkit_list = toolkit_list
    
    def execute_step(self, step: List[Any], step_index: int, 
                    process_steps_func=None) -> Tuple[str, int, Dict[str, Any]]:
        """
        Execute a single step with semantic filtering.
        
        Args:
            step: The step to execute [step_id, step_plan]
            step_index: Index of the step
            process_steps_func: Function to process the step
        
        Returns:
            Tuple of (json_response, status_code, execution_metadata)
        """
        try:
            logger.info(f"ProcessSingleStep {step_index}: {step[1][:50]}...")
            
            # Apply semantic filtering specific to this step
            step_id, step_plan = step[0], step[1]
            toolkit = self.toolkit_list.tool_kits[int(step_id)].tool_lists
            
            # Get semantically filtered tools for this step
            filtered_tools = self.semantic_filter.filter_tools_by_relevance(
                query=step_plan,
                available_tools=toolkit,
                top_k=2
            )
            
            # Execute the step using filtered tools
            if process_steps_func:
                json_response, status_code, actual_tool_args = process_steps_func(step, filtered_tools)
            else:
                json_response, status_code, actual_tool_args = "", 1, {}
            
            # Extract tool information from response
            tool_name = "unknown"
            tool_args = actual_tool_args
            try:
                if json_response:
                    response_data = json.loads(json_response) if isinstance(json_response, str) else json_response
                    tool_name = response_data.get("tool_name", response_data.get("api_name", "unknown"))
                    if not tool_args and "arguments" in response_data:
                        tool_args = response_data["arguments"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            
            # Create execution metadata
            execution_metadata = {
                "step_index": step_index,
                "step_description": step_plan,
                "tools_available": len(toolkit),
                "tools_filtered": len(filtered_tools),
                "semantic_filtering_applied": True,
                "execution_successful": status_code == 0,
                "tool_name": tool_name,
                "tool_args": tool_args
            }
            
            return json_response, status_code, execution_metadata
        
        except Exception as e:
            logger.error(f"ProcessSingleStep failed: {e}")
            error_response = json.dumps({"error": f"Step processing failed: {e}", "response": ""})
            return error_response, 1, {"error": str(e)}
    
    def analyze_step_consistency(self, step_responses: List[str]) -> Dict[str, Any]:
        """
        Analyze consistency between step results.
        
        Args:
            step_responses: List of step response JSON strings
        
        Returns:
            Dictionary with consistency analysis results
        """
        consistency_report = {
            "total_steps": len(step_responses),
            "inconsistencies_detected": [],
            "spatial_analysis_results": [],
            "perception_results": []
        }
        
        try:
            for i, response_str in enumerate(step_responses):
                try:
                    response_data = json.loads(response_str) if isinstance(response_str, str) else response_str
                    
                    # Check for errors
                    if response_data.get("error"):
                        consistency_report["inconsistencies_detected"].append({
                            "step": i,
                            "tool": response_data.get("tool_name", "unknown"),
                            "warning": "Tool execution error",
                            "summary": response_data.get("error", "Unknown error")
                        })
                    
                    # Track results by type
                    if "count" in response_data:
                        consistency_report["spatial_analysis_results"].append({
                            "step": i,
                            "count": response_data["count"]
                        })
                    
                    if "detections" in response_data:
                        consistency_report["perception_results"].append({
                            "step": i,
                            "detections": len(response_data.get("detections", []))
                        })
                
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
        
        except Exception as e:
            logger.error(f"Failed to analyze step consistency: {e}")
        
        return consistency_report
    
    def validate_step_execution(self, response: str, tool_name: str) -> bool:
        """
        Validate if step execution was successful.
        
        Args:
            response: Tool response string
            tool_name: Name of the tool
        
        Returns:
            bool: True if execution was successful
        """
        try:
            if isinstance(response, str):
                response_data = json.loads(response)
            else:
                response_data = response
            
            # Check for success indicators
            if response_data.get("error"):
                logger.warning(f"Tool {tool_name} returned error: {response_data.get('error')}")
                return False
            
            if response_data.get("success") is False:
                logger.warning(f"Tool {tool_name} execution failed")
                return False
            
            return True
        
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(f"Failed to validate response for tool {tool_name}")
            return False
    
    def extract_execution_results(self, response: str) -> Dict[str, Any]:
        """
        Extract key results from tool execution response.
        
        Args:
            response: Tool response string
        
        Returns:
            Dictionary of extracted results
        """
        try:
            if isinstance(response, str):
                response_data = json.loads(response)
            else:
                response_data = response
            
            results = {}
            
            # Extract common result fields
            for field in ["count", "percentage", "distance", "area", "detections", "classes_detected"]:
                if field in response_data:
                    results[field] = response_data[field]
            
            return results
        
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Failed to extract execution results")
            return {}

