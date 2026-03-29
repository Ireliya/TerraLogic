"""
Chain-of-Thought helper methods for the enhanced chat model.
Contains utility functions for CoT reasoning processing.
"""

import re
import json
from typing import Dict, Any, Tuple, Optional
from copy import deepcopy


class CoTHelpers:
    """Helper methods for Chain-of-Thought reasoning."""
    
    @staticmethod
    def contains_thought(response: str) -> bool:
        """Check if response contains a thought."""
        return "Thought:" in response or "thought:" in response.lower()
    
    @staticmethod
    def extract_thought(response: str) -> str:
        """Extract thought content from response."""
        # Look for "Thought:" pattern
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n(?:Action:|$))', response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            return thought_match.group(1).strip()
        return response.strip()
    
    @staticmethod
    def contains_function_call(response: str) -> bool:
        """Check if response contains a function call."""
        return "Action:" in response and "Action Input:" in response
    
    @staticmethod
    def extract_function_call(response: str) -> Tuple[str, Dict[str, Any]]:
        """Extract function name and arguments from response."""
        try:
            # Extract action name
            action_match = re.search(r'Action:\s*(\w+)', response, re.IGNORECASE)
            if not action_match:
                raise ValueError("No action found")
            
            action_name = action_match.group(1).strip()
            
            # Extract action input
            input_match = re.search(r'Action Input:\s*({.*?})', response, re.DOTALL | re.IGNORECASE)
            if not input_match:
                # Try without braces
                input_match = re.search(r'Action Input:\s*(.*?)(?=\n|$)', response, re.IGNORECASE)
                if input_match:
                    input_str = input_match.group(1).strip()
                    # Try to parse as JSON, fallback to simple text
                    try:
                        action_args = json.loads(input_str)
                    except:
                        action_args = {"text_prompt": input_str}
                else:
                    action_args = {}
            else:
                action_args = json.loads(input_match.group(1))
            
            return action_name, action_args
            
        except Exception as e:
            print(f"[CoT] Error extracting function call: {e}")
            return "segmentation", {"text_prompt": "objects"}
    
    @staticmethod
    def create_thought_node(parent_node, thought_content: str):
        """Create a new thought node."""
        from spatialreason.cot.simple_tree import create_thought_node
        return create_thought_node(parent_node, thought_content)
    
    @staticmethod
    def create_action_node(parent_node, action_name: str):
        """Create a new action node."""
        from spatialreason.cot.simple_tree import create_action_node
        return create_action_node(parent_node, action_name)
    
    @staticmethod
    def create_action_input_node(parent_node, action_args: Dict[str, Any]):
        """Create a new action input node."""
        from spatialreason.cot.simple_tree import create_action_input_node
        return create_action_input_node(parent_node, action_args)
    
    @staticmethod
    def generate_cot_response(messages, model_func) -> str:
        """Generate CoT response using the model."""
        try:
            # Format messages for the model
            formatted_messages = []
            for msg in messages:
                if msg["role"] in ["system", "user", "assistant"]:
                    formatted_messages.append(msg)
            
            # Generate response
            response = model_func(formatted_messages)
            return response
            
        except Exception as e:
            print(f"[CoT] Error generating response: {e}")
            return "Thought: I need to analyze the image. Action: segmentation Action Input: {\"text_prompt\": \"objects\"}"


def add_cot_methods_to_model(model_instance):
    """Add CoT helper methods to the enhanced chat model instance."""
    
    def _contains_thought(self, response: str) -> bool:
        return CoTHelpers.contains_thought(response)
    
    def _extract_thought(self, response: str) -> str:
        return CoTHelpers.extract_thought(response)
    
    def _contains_function_call(self, response: str) -> bool:
        return CoTHelpers.contains_function_call(response)
    
    def _extract_function_call(self, response: str) -> Tuple[str, Dict[str, Any]]:
        return CoTHelpers.extract_function_call(response)
    
    def _create_thought_node(self, parent_node, thought_content: str):
        return CoTHelpers.create_thought_node(parent_node, thought_content)
    
    def _create_action_node(self, parent_node, action_name: str):
        return CoTHelpers.create_action_node(parent_node, action_name)
    
    def _create_action_input_node(self, parent_node, action_args: Dict[str, Any]):
        return CoTHelpers.create_action_input_node(parent_node, action_args)
    
    def _generate_cot_response(self, messages) -> str:
        return CoTHelpers.generate_cot_response(messages, self._generate_with_qwen)
    
    # Bind methods to the model instance
    import types
    model_instance._contains_thought = types.MethodType(_contains_thought, model_instance)
    model_instance._extract_thought = types.MethodType(_extract_thought, model_instance)
    model_instance._contains_function_call = types.MethodType(_contains_function_call, model_instance)
    model_instance._extract_function_call = types.MethodType(_extract_function_call, model_instance)
    model_instance._create_thought_node = types.MethodType(_create_thought_node, model_instance)
    model_instance._create_action_node = types.MethodType(_create_action_node, model_instance)
    model_instance._create_action_input_node = types.MethodType(_create_action_input_node, model_instance)
    model_instance._generate_cot_response = types.MethodType(_generate_cot_response, model_instance)
    
    print("[DEBUG] CoT helper methods added to enhanced chat model")
