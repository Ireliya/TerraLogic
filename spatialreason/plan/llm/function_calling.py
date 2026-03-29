"""
Function calling utilities for the spatial reasoning agent.

This module provides utilities for function schema generation, function calling,
and response parsing for LLM-based tool execution.
"""

import json
import logging
from typing import List, Dict, Any, Optional

# Setup logger
logger = logging.getLogger("function_calling")

# Import utilities with fallback
try:
    from spatialreason.plan.utils import standardize, change_name
except ImportError:
    logger.warning("Utils not available, using fallback implementations")
    
    def standardize(text: str) -> str:
        """Fallback standardize function."""
        return text.lower().replace(" ", "_")
    
    def change_name(text: str) -> str:
        """Fallback change_name function."""
        return text.replace("_", "")


class FunctionCallHandler:
    """
    Handles function calling operations including schema generation,
    function formatting, and response parsing.
    """
    
    def __init__(self):
        """Initialize function call handler."""
        self.type_mapping = {
            "NUMBER": "number",  # Changed from "integer" to "number" to support floating-point values like 0.5
            "STRING": "string",
            "BOOLEAN": "boolean",
            "ARRAY": "array"
        }
    
    def fetch_api_json(self, tool) -> Dict[str, Any]:
        """
        Extract API JSON from tool object.
        
        Args:
            tool: Tool object with api_dest and api_doc attributes
            
        Returns:
            Dictionary containing tool API information
        """
        api_dest = tool.api_dest
        api_doc = tool.api_doc
        
        output_json = {
            "category_name": api_dest["type_name"],
            "api_name": api_dest["name"],
            "api_description": api_dest["desc"],
            "required_parameters": api_doc["required_parameters"],
            "optional_parameters": api_doc["optional_parameters"],
            "tool_name": api_dest["package_name"]
        }
        
        return output_json
    
    def build_parameter_schema(self, para: Dict[str, Any], description_max_length: int = 256) -> tuple:
        """
        Build parameter schema for both required and optional parameters.
        
        Args:
            para: Parameter definition dictionary
            description_max_length: Maximum length for parameter descriptions
            
        Returns:
            tuple: (parameter_name, parameter_schema_dict)
        """
        # Standardize parameter name
        name = change_name(standardize(para["name"]))

        # Determine parameter type
        param_type = self.type_mapping.get(para["type"], "string")

        # Build base parameter schema
        prompt = {
            "type": param_type,
            "description": para["description"][:description_max_length]
        }

        # Add items field for array types
        if param_type == "array":
            prompt["items"] = {"type": "string"}  # Array of strings for class names

        # Add example value only if it exists and is meaningful
        default_value = para.get('default', '')
        if default_value and str(default_value).strip():
            prompt["example_value"] = default_value

        return name, prompt
    
    def api_json_to_function_json(self, api_json: Dict[str, Any], standard_tool_name: str) -> tuple:
        """
        Convert API JSON to function calling JSON schema.
        
        Args:
            api_json: API information dictionary
            standard_tool_name: Standardized tool name
            
        Returns:
            tuple: (function_schema, category_name, pure_api_name)
        """
        description_max_length = 256
        template = {
            "name": "",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "optional": [],
            }
        }
        
        pure_api_name = change_name(standardize(api_json["api_name"]))
        template["name"] = pure_api_name + f"_for_{standard_tool_name}"
        template["name"] = template["name"][-64:]  # Limit name length
        
        # Simplified description - detailed formatting will be handled by _format_functions()
        if api_json["api_description"].strip() != "":
            template["description"] = api_json['api_description'].strip()[:description_max_length]
        else:
            template["description"] = f"Tool for {standard_tool_name}"
        
        # Process required parameters using common helper method
        if "required_parameters" in api_json.keys() and len(api_json["required_parameters"]) > 0:
            for para in api_json["required_parameters"]:
                name, prompt = self.build_parameter_schema(para, description_max_length)
                template["parameters"]["properties"][name] = prompt
                template["parameters"]["required"].append(name)
        
        # Process optional parameters using the same helper method
        for para in api_json["optional_parameters"]:
            name, prompt = self.build_parameter_schema(para, description_max_length)
            template["parameters"]["properties"][name] = prompt
            template["parameters"]["optional"].append(name)
        
        return template, api_json["category_name"], pure_api_name
    
    def format_functions(self, functions: List[Dict[str, Any]]) -> str:
        """
        Format function definitions for the prompt.
        
        Args:
            functions: List of function definitions
            
        Returns:
            Formatted functions text
        """
        formatted_functions = []
        for func in functions:
            func_text = f"- {func['name']}: {func.get('description', 'No description')}"
            if 'parameters' in func:
                params = func['parameters']
                if 'properties' in params:
                    param_list = []
                    for param_name, param_info in params['properties'].items():
                        param_desc = param_info.get('description', 'No description')
                        param_type = param_info.get('type', 'string')
                        param_list.append(f"{param_name} ({param_type}): {param_desc}")
                    if param_list:
                        func_text += f"\n  Parameters: {', '.join(param_list)}"
            formatted_functions.append(func_text)
        
        return "\n".join(formatted_functions)
    
    def parse_function_call(self, response: str, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse function call from model response.
        
        Args:
            response: Model response text
            functions: Available functions
            
        Returns:
            Function call dictionary
        """
        try:
            # Look for function call pattern
            lines = response.split('\n')
            function_name = None
            arguments_str = "{}"
            
            for i, line in enumerate(lines):
                if line.startswith("Function:"):
                    function_name = line.replace("Function:", "").strip()
                elif line.startswith("Arguments:"):
                    arguments_str = line.replace("Arguments:", "").strip()
                    # Try to get multi-line JSON if needed
                    if not arguments_str.endswith('}') and i + 1 < len(lines):
                        for j in range(i + 1, len(lines)):
                            arguments_str += lines[j]
                            if lines[j].strip().endswith('}'):
                                break
            
            # Validate function name
            if function_name:
                valid_names = [f["name"] for f in functions]
                if function_name not in valid_names:
                    # Try to find closest match
                    for valid_name in valid_names:
                        if valid_name in function_name or function_name in valid_name:
                            function_name = valid_name
                            break
                    else:
                        # Use first available function as fallback
                        function_name = valid_names[0] if valid_names else "unknown"
            else:
                # Use first available function as fallback
                function_name = functions[0]["name"] if functions else "unknown"
            
            # Validate JSON arguments
            try:
                json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments_str = "{}"
            
            return {
                "name": function_name,
                "arguments": arguments_str
            }
            
        except Exception as e:
            logger.error(f"Error parsing function call: {e}")
            return {
                "name": functions[0]["name"] if functions else "error",
                "arguments": "{}"
            }
    
    def create_function_calling_prompt(self, base_prompt: str, functions: List[Dict[str, Any]]) -> str:
        """
        Create enhanced prompt with function information for function calling.

        Args:
            base_prompt: Base conversation prompt
            functions: List of available functions

        Returns:
            Enhanced prompt with function calling instructions
        """
        functions_text = self.format_functions(functions)

        # Import CLASS_VOCABULARY constraint for perception tools
        try:
            from spatialreason.config.class_vocabulary import get_class_vocabulary_prompt
            class_vocab = get_class_vocabulary_prompt()

            # Check if any perception tools are in the function list
            perception_tools = ['detection', 'segmentation', 'classification',
                              'sar_detection', 'sar_classification', 'infrared_detection']
            has_perception_tool = any(
                any(tool_name in func.get('name', '') for tool_name in perception_tools)
                for func in functions
            )

            if has_perception_tool:
                class_vocab_reminder = f"""
CRITICAL REMINDER - CLASS VOCABULARY CONSTRAINT:
For perception tools (detection, segmentation, classification), the text_prompt parameter MUST use ONLY these exact class names:
{class_vocab}

Do NOT use any variations, plurals, or synonyms. Use the exact names listed above.

IMPORTANT CLASS NAMING RULES:
1. For detection and segmentation tools: Use lowercase names WITHOUT "_land" suffix
   - CORRECT: "forest", "barren", "agriculture"
   - INCORRECT: "forest_land", "barren_land", "agriculture_land"
2. For classification tool: Use title case WITH space (not underscore)
   - CORRECT: "Forest land", "Barren land", "Agriculture land"
   - INCORRECT: "forest_land", "barren_land", "agriculture_land"

CRITICAL REMINDER - PARAMETER FORMAT:
For perception tools (detection, segmentation, classification):
1. text_prompt: Should be a simple comma-separated list of class names (e.g., "building, forest" or "Agriculture land, Rangeland")
   - Do NOT use full sentences like "Classify regions into Agriculture land and Rangeland"
   - Do NOT use descriptive text, just the class names separated by commas
2. classes_requested: Should be an array of the same class names from text_prompt (e.g., ["building", "forest"] or ["Agriculture land", "Rangeland"])
   - Do NOT put the entire text_prompt as a single element in the array
   - Each class name should be a separate element in the array
   - The array should contain ONLY the classes mentioned in the query, not all possible classes

Example CORRECT format for segmentation:
{{
  "text_prompt": "building, forest",
  "classes_requested": ["building", "forest"]
}}

Example CORRECT format for classification:
{{
  "text_prompt": "Agriculture land, Rangeland",
  "classes_requested": ["Agriculture land", "Rangeland"]
}}

Example INCORRECT format:
{{
  "text_prompt": "building, forest_land",
  "classes_requested": ["building", "forest_land"]
}}
"""
            else:
                class_vocab_reminder = ""
        except ImportError:
            class_vocab_reminder = ""

        # Add guidance for spatial relation tools
        spatial_tools = ['overlap', 'containment', 'buffer', 'distance_calculation', 'area_measurement']
        has_spatial_tool = any(
            any(tool_name in func.get('name', '') for tool_name in spatial_tools)
            for func in functions
        )

        if has_spatial_tool:
            spatial_tool_reminder = """
CRITICAL REMINDER - SPATIAL RELATION TOOLS:
For spatial relation tools (overlap, containment, buffer, distance_calculation, area_measurement):
- ALWAYS provide class names as parameters so geometries can be extracted from perception results
- For overlap tool: Provide BOTH source_class AND target_class (e.g., "bridge" and "harbor"), plus source_polygon_count and target_polygon_count
- For containment tool: Provide BOTH container_class AND contained_class (e.g., "water" and "building")
- For area_measurement tool: Provide area_class (e.g., "agriculture")
- For distance_calculation tool: Provide BOTH set_a_class AND set_b_class (e.g., "road" and "water")
- For buffer tool: Provide buffer_distance_meters and query_text

Example CORRECT format for overlap:
{{
  "source_class": "bridge",
  "target_class": "harbor",
  "meters_per_pixel": 3.0,
  "source_polygon_count": 1,
  "target_polygon_count": 3
}}

Example CORRECT format for containment:
{{
  "container_class": "water",
  "contained_class": "building",
  "meters_per_pixel": 0.3
}}
"""
        else:
            spatial_tool_reminder = ""

        # Add guidance for confidence_threshold parameter
        confidence_threshold_reminder = """
CRITICAL REMINDER - CONFIDENCE THRESHOLD PARAMETER:
For perception tools (detection, segmentation, classification, infrared_detection, sar_detection, sar_classification):
- confidence_threshold MUST be a DECIMAL value between 0.0 and 1.0
- CORRECT: 0.5 (means 50% confidence)
- INCORRECT: 50 (this is a percentage, not a decimal)
- CORRECT: 0.3 (means 30% confidence)
- INCORRECT: 30 (this is a percentage, not a decimal)
- Always use decimal notation: 0.0, 0.1, 0.2, ..., 0.9, 1.0
"""

        enhanced_prompt = f"{base_prompt}\n\nAvailable Functions:\n{functions_text}{class_vocab_reminder}{spatial_tool_reminder}{confidence_threshold_reminder}\n\nPlease respond with a function call in the format:\nFunction: <function_name>\nArguments: <json_arguments>\n\nAssistant:"

        return enhanced_prompt
