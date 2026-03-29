"""
Step parsing utilities for the spatial reasoning agent.

This module provides classes for parsing individual plan steps,
extracting toolkit IDs, and validating step parameters.
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

# Setup logger
logger = logging.getLogger("step_parser")


class StepParser:
    """
    Handles parsing of individual plan steps including toolkit ID extraction
    and step validation.
    """
    
    def __init__(self, available_toolkits: int = 3):
        """
        Initialize step parser.
        
        Args:
            available_toolkits: Number of available toolkits for validation
        """
        self.available_toolkits = available_toolkits
        
        # Define toolkit ID extraction patterns
        # Order patterns from most specific to least specific to avoid false matches
        self.toolkit_id_patterns = [
            r'tool_(\d+)\.id',      # {tool_0.id}, {tool_1.id}, {tool_2.id}
            r'toolkit_(\d+)',       # toolkit_0, toolkit_1, etc.
            r'tool_(\d+)',          # {tool_0}
            r'(\d+)\.id',           # {0.id}
            r'(\d+)$'               # 0, 1, 2 (end of string to avoid partial matches)
        ]
    
    def extract_toolkit_id(self, step_key: str) -> str:
        """
        Extract toolkit ID from step key, handling various formats and template variables.
        
        Args:
            step_key: The step key containing toolkit ID information
            
        Returns:
            Extracted toolkit ID as string, defaults to "0" if extraction fails
        """
        # Extract toolkit ID and handle template variables
        toolkit_id_raw = step_key.split(" ")[-1]
        logger.debug(f"Parsing toolkit ID from key: '{step_key}' → raw ID: '{toolkit_id_raw}'")
        
        toolkit_id = "0"  # Default fallback
        for pattern in self.toolkit_id_patterns:
            match = re.search(pattern, toolkit_id_raw)
            if match:
                toolkit_id = match.group(1)
                logger.debug(f"Extracted toolkit ID '{toolkit_id}' from '{toolkit_id_raw}' using pattern '{pattern}'")
                break
        
        if not any(re.search(pattern, toolkit_id_raw) for pattern in self.toolkit_id_patterns):
            logger.warning(f"Could not parse toolkit ID from '{toolkit_id_raw}', using default 0")
        
        return toolkit_id
    
    def validate_toolkit_id(self, toolkit_id: str) -> str:
        """
        Validate and normalize toolkit ID to ensure it's within valid range.
        
        Args:
            toolkit_id: Raw toolkit ID string
            
        Returns:
            Validated and normalized toolkit ID string
        """
        try:
            toolkit_idx = int(toolkit_id)
            max_toolkit_id = self.available_toolkits - 1
            
            if toolkit_idx >= self.available_toolkits:
                # Expose the root cause instead of masking it with automatic fixes
                error_msg = f"❌ STEP PARSER ERROR: Invalid toolkit ID {toolkit_idx} (max: {max_toolkit_id})"
                logger.error(error_msg)
                valid_ids = ", ".join(str(i) for i in range(self.available_toolkits))
                logger.error(f"❌ Available toolkit IDs: {valid_ids} (total: {self.available_toolkits} toolkits)")
                logger.error(f"❌ The LLM planner generated an out-of-bounds toolkit reference")
                logger.error(f"❌ This indicates the planning prompt or toolkit definitions are incorrect")

                # Raise error to expose the root cause
                raise ValueError(f"Invalid toolkit ID {toolkit_idx}. Valid range: 0-{max_toolkit_id}. "
                               f"The LLM planner should not generate toolkit IDs outside this range.")
            elif toolkit_idx < 0:
                # Handle negative IDs
                logger.warning(f"⚠️ Negative toolkit ID {toolkit_idx}, using default 0")
                toolkit_id = "0"
            else:
                # Valid toolkit ID
                logger.debug(f"✅ Valid toolkit ID: {toolkit_id}")
                
        except ValueError:
            # If still not a valid integer, default to 0
            toolkit_id = "0"
            logger.warning(f"⚠️ Invalid toolkit ID (not numeric), using default 0")
        
        return toolkit_id
    
    def parse_step(self, step_dict: Dict[str, Any]) -> Tuple[str, str]:
        """
        Parse a single step dictionary to extract toolkit ID and method.
        
        Args:
            step_dict: Dictionary representing a single plan step
            
        Returns:
            Tuple of (toolkit_id, method) strings
        """
        if not isinstance(step_dict, dict) or len(step_dict) == 0:
            logger.error("Invalid step format: must be non-empty dictionary")
            return "0", ""
        
        # Get the first (and typically only) key-value pair
        key = list(step_dict.keys())[0]
        method = step_dict[key]
        
        # Extract and validate toolkit ID
        raw_toolkit_id = self.extract_toolkit_id(key)
        validated_toolkit_id = self.validate_toolkit_id(raw_toolkit_id)
        
        return validated_toolkit_id, method
    
    def parse_steps(self, plan_json: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """
        Parse multiple plan steps to extract toolkit IDs and methods.
        
        Args:
            plan_json: List of step dictionaries
            
        Returns:
            List of (toolkit_id, method) tuples
        """
        if not isinstance(plan_json, list):
            logger.error("Plan must be a list of steps")
            return []
        
        steps = []
        for i, step_dict in enumerate(plan_json):
            try:
                toolkit_id, method = self.parse_step(step_dict)
                steps.append((toolkit_id, method))
                logger.debug(f"Parsed step {i}: toolkit_id='{toolkit_id}', method='{method[:50]}...'")
            except Exception as e:
                logger.error(f"Error parsing step {i}: {e}")
                # Add default step to maintain step count
                steps.append(("0", ""))
        
        logger.info(f"Successfully parsed {len(steps)} steps")
        return steps
    
    def validate_step_method(self, method: str) -> bool:
        """
        Validate that a step method is not empty and contains meaningful content.
        
        Args:
            method: The method/description string from a step
            
        Returns:
            True if method is valid, False otherwise
        """
        if not isinstance(method, str):
            logger.warning("Method must be a string")
            return False
        
        if not method.strip():
            logger.warning("Method cannot be empty")
            return False
        
        # Check for minimum meaningful content
        if len(method.strip()) < 3:
            logger.warning("Method too short to be meaningful")
            return False
        
        return True
    
    def get_step_statistics(self, steps: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Generate statistics about parsed steps.
        
        Args:
            steps: List of (toolkit_id, method) tuples
            
        Returns:
            Dictionary containing step statistics
        """
        if not steps:
            return {"error": "No steps provided"}
        
        toolkit_usage = {}
        valid_methods = 0
        empty_methods = 0
        
        for toolkit_id, method in steps:
            # Count toolkit usage
            toolkit_usage[toolkit_id] = toolkit_usage.get(toolkit_id, 0) + 1
            
            # Count method validity
            if self.validate_step_method(method):
                valid_methods += 1
            else:
                empty_methods += 1
        
        return {
            "total_steps": len(steps),
            "toolkit_usage": toolkit_usage,
            "unique_toolkits": len(toolkit_usage),
            "valid_methods": valid_methods,
            "empty_methods": empty_methods,
            "most_used_toolkit": max(toolkit_usage.items(), key=lambda x: x[1])[0] if toolkit_usage else None
        }
    
    def update_available_toolkits(self, count: int):
        """
        Update the number of available toolkits for validation.
        
        Args:
            count: New number of available toolkits
        """
        self.available_toolkits = count
        logger.debug(f"Updated available toolkits to {count}")


class StepExecutionParser:
    """
    Handles parsing of step execution results and metadata.
    """
    
    def __init__(self):
        """Initialize step execution parser."""
        pass
    
    def parse_execution_result(self, result: Any) -> Dict[str, Any]:
        """
        Parse and normalize step execution results.
        
        Args:
            result: Raw execution result from tool
            
        Returns:
            Normalized result dictionary
        """
        if isinstance(result, dict):
            return result
        elif isinstance(result, str):
            return {"result": result, "type": "string"}
        elif isinstance(result, (int, float)):
            return {"result": result, "type": "numeric"}
        elif isinstance(result, list):
            return {"result": result, "type": "list", "count": len(result)}
        else:
            return {"result": str(result), "type": "unknown"}
    
    def extract_error_info(self, error: Exception) -> Dict[str, Any]:
        """
        Extract structured information from execution errors.
        
        Args:
            error: Exception that occurred during execution
            
        Returns:
            Dictionary containing error information
        """
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "is_cuda_oom": "cuda out of memory" in str(error).lower() or "out of memory" in str(error).lower()
        }
