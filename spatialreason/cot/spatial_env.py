"""
SpatialReason Environment for Chain-of-Thought reasoning.
Bridges the CoT framework with SpatialreasonAgent tools.
Configuration-driven architecture using tools_config.yaml.
"""

import json
import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path


class SpatialReasonEnvironment:
    """
    Environment class that integrates SpatialreasonAgent tools with CoT reasoning.
    Maps our spatial analysis tools to the CoT function calling interface.
    Configuration-driven architecture using tools_config.yaml.
    """

    def __init__(self, tools_dict: Dict[str, Any], image_path: Optional[str] = None):
        """
        Initialize the spatial reasoning environment.

        Args:
            tools_dict: Dictionary of available tools (segmentation, detection, classification)
            image_path: Path to the image being analyzed
        """
        self.tools_dict = tools_dict
        self.image_path = image_path
        self.execution_history = []
        self.success_state = False
        self.final_result = None

        # Load configuration
        self.config = self._load_tools_config()

        # CoT interface requirements
        self.task_description = "Analyze satellite imagery using remote sensing tools"
        self.input_description = ""
        self.tool_names = list(tools_dict.keys())
        self.functions = self._create_function_schemas_from_config()

    def _load_tools_config(self) -> Dict[str, Any]:
        """Load tools configuration from YAML file."""
        config_path = Path(__file__).parent.parent / "config" / "tools_config.yaml"

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Tools configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing tools configuration: {e}")

    def _create_function_schemas_from_config(self) -> List[Dict[str, Any]]:
        """Create OpenAI function schemas dynamically from configuration."""
        schemas = []

        # Process each tool from configuration
        for tool_id, tool_config in self.config.get("tools", {}).items():
            # Only include tools that are available in tools_dict
            if tool_id in self.tool_names:
                schema = self._build_tool_schema(tool_id, tool_config)
                schemas.append(schema)

        # Add the special Finish function
        schemas.append(self._create_finish_schema())

        return schemas

    def _build_tool_schema(self, tool_id: str, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build function schema for a single tool from configuration."""
        schema = {
            "name": tool_id,
            "description": tool_config.get("description", f"{tool_id} tool"),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        # Get parameter schema from config
        param_schema = tool_config.get("parameter_schema", {})
        required_params = tool_config.get("parameters", {}).get("required", [])
        optional_params = tool_config.get("parameters", {}).get("optional", [])

        # Build properties for each parameter
        for param_name, param_config in param_schema.items():
            # Skip image_path as it will be automatically injected
            if param_name == "image_path":
                continue

            property_def = {
                "type": param_config.get("type", "string"),
                "description": param_config.get("description", f"{param_name} parameter")
            }

            # Add default value if specified
            if "default" in param_config:
                property_def["default"] = param_config["default"]

            # Add constraints if specified
            if "min" in param_config:
                property_def["minimum"] = param_config["min"]
            if "max" in param_config:
                property_def["maximum"] = param_config["max"]
            if "enum" in param_config:
                property_def["enum"] = param_config["enum"]

            schema["parameters"]["properties"][param_name] = property_def

            # Add to required list if it's required and not image_path
            if param_name in required_params and param_name != "image_path":
                schema["parameters"]["required"].append(param_name)

        # Add note about automatic image_path injection
        if "image_path" in required_params:
            schema["description"] += " (image_path is automatically injected by the environment)"

        return schema

    def _create_finish_schema(self) -> Dict[str, Any]:
        """Create the special Finish function schema."""
        return {
            "name": "Finish",
            "description": "Complete the analysis and provide final answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "return_type": {
                        "type": "string",
                        "enum": ["give_answer", "give_up"],
                        "description": "Type of completion"
                    },
                    "final_answer": {
                        "type": "string",
                        "description": "Final analysis result or conclusion"
                    }
                },
                "required": ["return_type", "final_answer"]
            }
        }
    
    def restart(self):
        """Restart the environment to initial state."""
        self.execution_history = []
        self.success_state = False
        self.final_result = None
    
    def step(self, action_name: str, action_input: str) -> tuple[str, int]:
        """
        Execute a tool action and return observation.

        Args:
            action_name: Name of the tool to execute
            action_input: JSON string with tool arguments

        Returns:
            tuple: (observation_string, status_code)
                status_code: 0=success, 1=error, 4=terminal_success
        """
        try:
            # Parse action input
            if isinstance(action_input, str):
                args = json.loads(action_input)
            else:
                args = action_input

            # Handle completion
            if action_name == "Finish":
                return_type = args.get("return_type", "give_answer")
                final_answer = args.get("final_answer", "")

                if return_type == "give_answer":
                    self.success_state = True
                    self.final_result = final_answer
                    observation = f"Analysis completed successfully. Final result: {final_answer}"
                    return observation, 4  # Terminal success
                else:
                    observation = "Analysis abandoned."
                    return observation, 4  # Terminal failure

            # Execute spatial analysis tool
            if action_name in self.tools_dict:
                tool = self.tools_dict[action_name]

                # Prepare tool arguments with configuration-driven defaults
                tool_args = self._prepare_tool_arguments(action_name, args)

                # Execute tool
                result = tool.invoke(tool_args)

                # Record execution
                self.execution_history.append({
                    "action": action_name,
                    "args": args,
                    "result": result
                })

                # Return natural language observation
                observation = f"Successfully executed {action_name} analysis. Result: {result}"
                return observation, 0  # Success

            else:
                # Unknown tool
                observation = f"Unknown tool: {action_name}. Available tools: {self.tool_names}"
                return observation, 1  # Error

        except Exception as e:
            observation = f"Error executing {action_name}: {str(e)}"
            return observation, 1  # Error

    def _prepare_tool_arguments(self, tool_name: str, user_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare tool arguments with automatic context injection and configuration-driven defaults.

        Args:
            tool_name: Name of the tool being executed
            user_args: Arguments provided by the LLM

        Returns:
            Dict with complete tool arguments including injected context and defaults
        """
        tool_args = {}

        # Get tool configuration
        tool_config = self.config.get("tools", {}).get(tool_name, {})
        param_schema = tool_config.get("parameter_schema", {})

        # Automatically inject image_path (implicit context parameter)
        tool_args["image_path"] = self.image_path

        # Process each parameter from the schema
        for param_name, param_config in param_schema.items():
            if param_name == "image_path":
                # Already injected above
                continue

            if param_name in user_args:
                # Use user-provided value
                tool_args[param_name] = user_args[param_name]
            elif "default" in param_config:
                # Use default from configuration
                tool_args[param_name] = param_config["default"]
            else:
                # Check global defaults
                default_value = self._get_global_default(param_name)
                if default_value is not None:
                    tool_args[param_name] = default_value

        # Add any additional user arguments not in schema (for flexibility)
        for param_name, value in user_args.items():
            if param_name not in tool_args:
                tool_args[param_name] = value

        return tool_args

    def _get_global_default(self, param_name: str) -> Any:
        """Get global default value for a parameter from configuration."""
        defaults = self.config.get("defaults", {})

        # Map common parameter names to configuration paths
        param_mappings = {
            "confidence_threshold": "confidence_thresholds",
            "meters_per_pixel": "spatial.meters_per_pixel",
            "buffer_distance_meters": "spatial.buffer_distance_meters"
        }

        if param_name in param_mappings:
            config_path = param_mappings[param_name]
            # Navigate nested configuration
            value = defaults
            for key in config_path.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value

        return None
    
    def check_success(self) -> int:
        """
        Check if the reasoning has reached a successful conclusion.
        
        Returns:
            int: 1 if successful, 0 if still in progress, -1 if failed
        """
        if self.success_state:
            return 1
        return 0
    
    def get_score(self) -> float:
        """Get current state score (for search algorithms)."""
        if self.success_state:
            return 1.0
        return len(self.execution_history) * 0.1  # Progress score
    
    def to_json(self) -> Dict[str, Any]:
        """Export environment state to JSON."""
        return {
            "image_path": self.image_path,
            "execution_history": self.execution_history,
            "success_state": self.success_state,
            "final_result": self.final_result,
            "available_tools": self.tool_names
        }
