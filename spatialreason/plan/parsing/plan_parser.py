"""
Plan parsing utilities for the spatial reasoning agent.

This module provides classes for parsing JSON plans from LLM responses,
handling various response formats and extracting structured plan data.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

# Setup logger
logger = logging.getLogger("plan_parser")


class PlanParser:
    """
    Handles parsing of JSON plans from LLM responses.
    
    This class provides robust JSON extraction capabilities that can handle
    various response formats including explanatory text, code blocks, and
    malformed JSON structures.
    """
    
    def __init__(self, evaluation_mode: bool = False):
        """
        Initialize plan parser.
        
        Args:
            evaluation_mode: Whether to enable strict evaluation mode logging
        """
        self.evaluation_mode = evaluation_mode
    
    def extract_json_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Extract JSON from response text that may contain extra content.
        Handles cases where the model generates explanatory text after JSON.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            List of parsed JSON objects representing plan steps
        """
        # First try to parse as complete JSON array
        try:
            parsed = json.loads(response_text.strip())
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # Try to find JSON array first (most common for plans)
        array_pattern = r'\[.*?\]'
        array_matches = re.findall(array_pattern, response_text, re.DOTALL)
        for match in array_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                continue

        # Handle comma-separated JSON objects (common planner output format)
        # Look for multiple JSON objects separated by commas
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        object_matches = re.findall(object_pattern, response_text, re.DOTALL)

        if len(object_matches) > 1:
            # Multiple objects found, try to parse them as an array
            try:
                array_text = '[' + ','.join(object_matches) + ']'
                parsed = json.loads(array_text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: try individual JSON objects
        for match in object_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                continue

        # If no valid JSON found, try to extract from code blocks
        code_block_pattern = r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```'
        code_matches = re.findall(code_block_pattern, response_text, re.DOTALL)
        for match in code_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                continue

        # Last resort: try to parse the entire response as JSON
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

        # If all else fails, return empty list
        logger.warning(f"Could not extract valid JSON from response: {response_text[:200]}...")

        # In evaluation mode, provide more detailed logging
        if self.evaluation_mode:
            logger.error(f"❌ JSON EXTRACTION FAILED in evaluation mode")
            logger.error(f"Full response text: {response_text}")
            logger.error(f"Response length: {len(response_text)} characters")

        return []
    
    def validate_plan_structure(self, plan_json: List[Dict[str, Any]]) -> bool:
        """
        Validate the structure of a parsed plan.
        
        Args:
            plan_json: Parsed plan JSON
            
        Returns:
            True if plan structure is valid, False otherwise
        """
        if not isinstance(plan_json, list):
            logger.error("Plan must be a list of steps")
            return False
        
        if len(plan_json) == 0:
            logger.warning("Plan is empty")
            return False
        
        for i, step in enumerate(plan_json):
            if not isinstance(step, dict):
                logger.error(f"Step {i} must be a dictionary")
                return False
            
            if len(step.keys()) != 1:
                logger.warning(f"Step {i} should have exactly one key-value pair")
            
            # Check if step has at least one key
            if len(step.keys()) == 0:
                logger.error(f"Step {i} is empty")
                return False
        
        return True
    
    def parse_plan(self, response_text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse a complete plan from LLM response text.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Parsed plan as list of dictionaries, or None if parsing failed
        """
        try:
            # Use robust JSON extraction
            plan_json = self.extract_json_from_response(response_text)
            
            if not plan_json:
                logger.error("No valid JSON plan found in response")
                
                # Check evaluation mode configuration for strict validation
                if self.evaluation_mode:
                    try:
                        from spatialreason.config.configuration_loader import ConfigurationLoader
                        config_loader = ConfigurationLoader()
                        eval_config = config_loader.get_evaluation_config()

                        if eval_config.get('log_raw_planner_output', True):
                            logger.error(f"Raw planner output for debugging: {response_text[:500]}...")

                        if eval_config.get('fail_on_no_plan', True):
                            error_msg = f"❌ EVALUATION ERROR: Planner failed to generate valid JSON plan. Raw output: {response_text[:200]}..."
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                    except ImportError:
                        # Fallback if config loader not available
                        if self.evaluation_mode:
                            error_msg = f"❌ EVALUATION ERROR: Planner failed to generate valid JSON plan. Raw output: {response_text[:200]}..."
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                
                return None
            
            # Validate plan structure
            if not self.validate_plan_structure(plan_json):
                logger.error("Plan structure validation failed")
                return None
            
            logger.info(f"Successfully parsed plan with {len(plan_json)} steps")
            return plan_json
            
        except Exception as e:
            logger.error(f"Plan parsing error: {e}")
            return None
    
    def get_plan_summary(self, plan_json: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of the parsed plan.
        
        Args:
            plan_json: Parsed plan JSON
            
        Returns:
            Dictionary containing plan summary information
        """
        if not plan_json:
            return {"error": "No plan provided"}
        
        summary = {
            "total_steps": len(plan_json),
            "step_keys": [],
            "unique_toolkits": set(),
            "step_descriptions": []
        }
        
        for i, step in enumerate(plan_json):
            if isinstance(step, dict) and len(step.keys()) > 0:
                key = list(step.keys())[0]
                value = step[key]
                
                summary["step_keys"].append(key)
                summary["step_descriptions"].append(value)
                
                # Extract toolkit information from key
                if "toolkit" in key.lower() or "tool" in key.lower():
                    # Try to extract toolkit ID
                    import re
                    toolkit_match = re.search(r'(\d+)', key)
                    if toolkit_match:
                        summary["unique_toolkits"].add(toolkit_match.group(1))
        
        summary["unique_toolkits"] = list(summary["unique_toolkits"])
        return summary
