"""
Result formatting module for spatialreason.plan

This module contains the ResultFormatter class which handles formatting
execution results for evaluation and output.

Classes:
    ResultFormatter: Formats results for evaluation and output
"""

import json
import logging
from typing import Dict, Any, List, Optional

# Setup module logger
logger = logging.getLogger(__name__)


class ResultFormatter:
    """
    Formats execution results for evaluation and output.

    This class handles:
    - Generating structured dialog format
    - Creating tool responses
    - Synthesizing final answers
    - Formatting results for benchmark compatibility
    """

    def __init__(self, planner_llm=None):
        """
        Initialize the result formatter.

        Args:
            planner_llm: Optional LLM instance for generating dynamic thoughts
        """
        self.planner_llm = planner_llm
    
    def generate_structured_dialog_format(self, original_query: str, generated_plan: List[str],
                                         step_responses: List[str], execution_metadata: List[Dict]) -> str:
        """
        Generate structured dialog format compatible with evaluation framework.
        
        Args:
            original_query: The original user query
            generated_plan: The generated plan steps
            step_responses: List of responses from step execution
            execution_metadata: Metadata about step execution
        
        Returns:
            str: JSON string representing structured dialog format
        """
        try:
            dialogs = []
            
            # Add user query
            dialogs.append({
                "role": "user",
                "content": original_query
            })
            
            # Process each executed step
            for i, (step_response, metadata) in enumerate(zip(step_responses, execution_metadata)):
                if not metadata.get("execution_successful", False):
                    continue
                
                tool_name = metadata.get("tool_name", "unknown")
                tool_args = metadata.get("tool_args", {})
                
                # Create assistant response with tool_call
                thought = self._generate_realistic_thought(tool_name, tool_args, original_query, i)
                assistant_response = {
                    "role": "assistant",
                    "thought": thought,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": tool_args
                            }
                        }
                    ]
                }
                dialogs.append(assistant_response)
                
                # Add tool response
                if step_response:
                    structured_content = self._create_structured_tool_response(
                        step_response, tool_name, tool_args
                    )
                    
                    tool_response = {
                        "role": "tool",
                        "name": tool_name,
                        "content": structured_content
                    }
                    dialogs.append(tool_response)
            
            # Add final answer
            final_answer = self._synthesize_final_answer(original_query, step_responses, execution_metadata)

            # CRITICAL: Do NOT add a final assistant completion message
            # The dialogs array should end with the last tool result message
            # The synthesized answer is stored in the 'answer' field for evaluation

            # Create final response structure
            # The 'answer' field contains the actual synthesized answer for evaluation
            result = {
                "dialogs": dialogs,
                "answer": final_answer  # Store the actual answer value separately
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Failed to generate structured dialog format: {e}")
            # Fallback to simple format - no final assistant message
            return json.dumps({
                "dialogs": [
                    {"role": "user", "content": original_query}
                ],
                "answer": f"Analysis completed using {len(step_responses)} steps."
            })
    
    def _generate_realistic_thought(self, tool_name: str, tool_args: Dict[str, Any],
                                   original_query: str, step_index: int) -> str:
        """
        Generate a realistic thought for tool execution using LLM.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            original_query: Original user query
            step_index: Index of the step

        Returns:
            str: Realistic thought text
        """
        # Try LLM-based generation if available
        if self.planner_llm:
            try:
                prompt = f"""You are a remote sensing expert explaining your reasoning for executing a spatial analysis tool.

QUERY: {original_query}
TOOL_TO_EXECUTE: {tool_name}
TOOL_ARGUMENTS: {json.dumps(tool_args, indent=2)}

Generate a brief, natural thought (1-2 sentences) that explains:
1. Why this tool is being called
2. What it will accomplish in the context of the query
3. How it contributes to answering the question

Be specific and reference the actual classes/parameters being used. Keep it concise and professional.
Return ONLY the thought text, no quotes or explanations."""

                self.planner_llm._reset_and_feed("You are a remote sensing expert.", prompt)
                thought = self.planner_llm.predict()

                if thought and thought.strip():
                    logger.debug(f"Generated LLM-based thought for {tool_name}: {thought[:80]}...")
                    return thought.strip()
            except Exception as e:
                logger.warning(f"Failed to generate LLM-based thought: {e}")

        # Fallback to generic thought if LLM not available or fails
        logger.debug(f"Using fallback thought for {tool_name}")
        return f"I will use {tool_name} tool to process the spatial data with the specified parameters."
    
    def _create_structured_tool_response(self, step_response: str, tool_name: str,
                                        tool_args: Dict[str, Any]) -> str:
        """
        Create structured tool response for benchmark compatibility.
        
        Args:
            step_response: Raw tool response
            tool_name: Name of the tool
            tool_args: Tool arguments used
        
        Returns:
            str: Structured tool response
        """
        try:
            # Parse response if it's JSON
            if isinstance(step_response, str):
                try:
                    response_data = json.loads(step_response)
                except json.JSONDecodeError:
                    response_data = {"raw_output": step_response}
            else:
                response_data = step_response
            
            # Create structured response
            structured = {
                "tool": tool_name,
                "status": "success" if response_data.get("success", True) else "failed",
                "result": response_data
            }
            
            return json.dumps(structured, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to create structured tool response: {e}")
            return json.dumps({"error": str(e)})
    
    def _synthesize_final_answer(self, original_query: str, step_responses: List[str],
                                execution_metadata: List[Dict]) -> str:
        """
        Synthesize final answer from step responses.
        
        Args:
            original_query: The original user query
            step_responses: List of responses from step execution
            execution_metadata: Metadata about step execution
        
        Returns:
            str: Final synthesized answer
        """
        try:
            # Extract key results from step responses
            results_summary = []
            
            for i, (response, metadata) in enumerate(zip(step_responses, execution_metadata)):
                tool_name = metadata.get("tool_name", "unknown")
                
                try:
                    if isinstance(response, str):
                        response_data = json.loads(response)
                    else:
                        response_data = response
                    
                    # Extract key information
                    if "count" in response_data:
                        results_summary.append(f"{tool_name}: {response_data['count']} objects")
                    elif "percentage" in response_data:
                        results_summary.append(f"{tool_name}: {response_data['percentage']}%")
                    elif "distance" in response_data:
                        results_summary.append(f"{tool_name}: {response_data['distance']}m")
                
                except (json.JSONDecodeError, KeyError):
                    pass
            
            # Generate final answer
            if results_summary:
                return "Analysis completed. " + " ".join(results_summary)
            else:
                return "Analysis completed successfully."
        
        except Exception as e:
            logger.error(f"Failed to synthesize final answer: {e}")
            return "Analysis completed."
    
    def format_for_benchmark(self, result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Format result for benchmark compatibility.
        
        Args:
            result: Raw result dictionary
            tool_name: Name of the tool
        
        Returns:
            Formatted result dictionary
        """
        formatted = {
            "tool": tool_name,
            "success": result.get("success", True),
            "timestamp": result.get("timestamp"),
        }
        
        # Add tool-specific fields
        if tool_name in ["detection", "segmentation", "classification"]:
            formatted["classes_detected"] = result.get("classes_detected", [])
            formatted["total_detections"] = result.get("total_detections", 0)
        
        return formatted

