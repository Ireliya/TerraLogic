"""
Model-agnostic LLM client for the spatial reasoning agent planner.

This module provides the PlannerLLM class for language model integration,
supporting multiple models (GPT-4o, Claude, Llama, Qwen, etc.) through a unified interface.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from enum import Enum

# Setup logger
logger = logging.getLogger("planner_llm")


class ModelType(Enum):
    """Supported model types for the planner."""
    LOCAL_QWEN = "local_qwen"
    REMOTE_GPT4O = "remote_gpt4o"
    REMOTE_CLAUDE = "remote_claude"
    REMOTE_LLAMA = "remote_llama"
    CUSTOM = "custom"


class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from the model."""
        pass
    
    @abstractmethod
    def generate_with_conversation(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from conversation history."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model backend is available."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class LocalQwenBackend(ModelBackend):
    """Backend for local Qwen models."""
    
    def __init__(self, model_path: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda:1",
                 shared_model=None, shared_tokenizer=None):
        self.model_path = model_path
        self.device = device
        self.shared_model = shared_model
        self.shared_tokenizer = shared_tokenizer
        self.model = None
        self.tokenizer = None
        
        # Generation configuration optimized for planning tasks
        self.generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": None
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load Qwen model and tokenizer."""
        try:
            if self.shared_model is not None and self.shared_tokenizer is not None:
                logger.info("Using shared Qwen model instance for planner")
                self.model = self.shared_model
                self.tokenizer = self.shared_tokenizer
            else:
                logger.info(f"Loading Qwen model for planner on {self.device}")
                from spatialreason.utils.qwen_utils import load_qwen_model
                self.tokenizer, self.model = load_qwen_model(
                    model_path=self.model_path,
                    device=self.device
                )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            logger.info("Local Qwen backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise e
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using local Qwen model.

        Args:
            prompt: Input prompt text
            **kwargs: Optional parameters including 'temperature'

        Returns:
            Generated response text
        """
        try:
            import torch

            # Create a copy of generation config and update with kwargs
            gen_config = dict(self.generation_config)
            if 'temperature' in kwargs:
                gen_config['temperature'] = kwargs['temperature']

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            return response

        except Exception as e:
            logger.error(f"Local Qwen generation failed: {e}")
            raise e

    def generate_with_conversation(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response from conversation history.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Optional parameters including 'temperature'

        Returns:
            Generated response text
        """
        # Format conversation for Qwen
        conversation_text = ""
        for msg in messages:
            role = msg["role"].title()
            content = msg["content"]
            conversation_text += f"{role}: {content}\n\n"

        conversation_text += "Assistant: "
        return self.generate_response(conversation_text, **kwargs)
    
    def is_available(self) -> bool:
        """Check if Qwen model is available."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Qwen model information."""
        return {
            "model_type": "local_qwen",
            "model_path": self.model_path,
            "device": self.device,
            "available": self.is_available()
        }


class RemoteBackend(ModelBackend):
    """Backend for remote models (GPT-4o, Claude, etc.)."""
    
    def __init__(self, model_manager, model_name: str = "gpt-4o"):
        self.model_manager = model_manager
        self.model_name = model_name
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using remote model."""
        try:
            temperature = kwargs.get('temperature', 0.1)
            return self.model_manager.generate_text_response(prompt, temperature=temperature)
        except Exception as e:
            logger.error(f"Remote model generation failed: {e}")
            raise e
    
    def generate_with_conversation(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from conversation history."""
        try:
            temperature = kwargs.get('temperature', 0.1)
            return self.model_manager.generate_chat_response(messages, temperature=temperature)
        except Exception as e:
            logger.error(f"Remote model conversation generation failed: {e}")
            raise e
    
    def is_available(self) -> bool:
        """Check if remote model is available."""
        return self.model_manager is not None and self.model_manager.is_available()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get remote model information."""
        return {
            "model_type": "remote",
            "model_name": self.model_name,
            "available": self.is_available()
        }


class PlannerLLM:
    """
    Model-agnostic LLM for planning tasks supporting multiple language models.
    Provides unified interface for GPT-4o, Claude, Llama, Qwen, and other models.
    """
    
    # Class-level cache for model instances
    _instances = {}
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached instances."""
        logger.info(f"Clearing PlannerLLM cache ({len(cls._instances)} instances)")
        cls._instances.clear()
    
    def __init__(self, model_type: Union[ModelType, str] = ModelType.LOCAL_QWEN,
                 model_path: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = "cuda:1",
                 shared_model=None, shared_tokenizer=None, remote_model_manager=None,
                 model_name: str = "gpt-4o", **kwargs):
        """
        Initialize model-agnostic PlannerLLM.
        
        Args:
            model_type: Type of model to use (local_qwen, remote_gpt4o, etc.)
            model_path: Path for local models
            device: Device for local models
            shared_model: Pre-loaded model instance for local models
            shared_tokenizer: Pre-loaded tokenizer for local models
            remote_model_manager: Model manager for remote models
            model_name: Name of remote model
        """
        # Convert string to enum if needed
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        self.model_type = model_type
        self.conversation_history = []
        self.system_message = ""
        
        # Initialize appropriate backend
        if model_type == ModelType.LOCAL_QWEN:
            self.backend = LocalQwenBackend(
                model_path=model_path,
                device=device,
                shared_model=shared_model,
                shared_tokenizer=shared_tokenizer
            )
        elif model_type in [ModelType.REMOTE_GPT4O, ModelType.REMOTE_CLAUDE, ModelType.REMOTE_LLAMA]:
            if remote_model_manager is None:
                raise ValueError(f"remote_model_manager required for {model_type}")
            self.backend = RemoteBackend(remote_model_manager, model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"PlannerLLM initialized with {model_type.value} backend")
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.system_message = ""
    
    def _reset_and_feed(self, system: str, user: str):
        """Reset conversation and add system/user messages."""
        self.clear_conversation()
        self.system_message = system
        self.conversation_history = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    
    def _format_conversation(self) -> str:
        """Format conversation history for generation."""
        conversation_text = ""
        for msg in self.conversation_history:
            role = msg["role"].title()
            content = msg["content"]
            conversation_text += f"{role}: {content}\n\n"
        
        conversation_text += "Assistant: "
        return conversation_text
    
    def predict(self, temperature: float = None) -> str:
        """
        Generate response using the configured model backend.

        Args:
            temperature: Optional temperature parameter for generation. If not provided,
                        uses backend default (0.1 for local Qwen, 0.1 for remote).

        Returns:
            Generated response text
        """
        try:
            kwargs = {}
            if temperature is not None:
                kwargs['temperature'] = temperature

            if self.conversation_history:
                response = self.backend.generate_with_conversation(self.conversation_history, **kwargs)
            else:
                response = self.backend.generate_response("Please provide a response.", **kwargs)

            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        except Exception as e:
            logger.error(f"PlannerLLM prediction failed: {e}")
            raise e
    
    def predict_fun(self, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response with function calling capability."""
        try:
            from .function_calling import FunctionCallHandler
            handler = FunctionCallHandler()
            
            # Create enhanced prompt with function information
            base_prompt = self._format_conversation()
            enhanced_prompt = handler.create_function_calling_prompt(base_prompt, functions)
            
            # Generate response
            response = self.backend.generate_response(enhanced_prompt)
            
            # Parse function call from response
            function_call = handler.parse_function_call(response, functions)
            return function_call
            
        except Exception as e:
            logger.error(f"PlannerLLM function calling failed: {e}")
            return {"name": "error", "arguments": "{}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model backend."""
        info = self.backend.get_model_info()
        info["planner_model_type"] = self.model_type.value
        return info
    
    def is_available(self) -> bool:
        """Check if the model backend is available."""
        return self.backend.is_available()


# Backward compatibility aliases
QwenPlannerLLM = PlannerLLM


class RemotePlannerLLM(PlannerLLM):
    """Specialized remote planner LLM for backward compatibility."""

    def __init__(self, remote_model_manager, model_name: str = "gpt-4o"):
        super().__init__(
            model_type=ModelType.REMOTE_GPT4O,
            remote_model_manager=remote_model_manager,
            model_name=model_name
        )


def create_remote_planner_llm(remote_model_manager, model_name: str = "gpt-4o") -> PlannerLLM:
    """Convenience function to create remote PlannerLLM."""
    return PlannerLLM(
        model_type=ModelType.REMOTE_GPT4O,
        remote_model_manager=remote_model_manager,
        model_name=model_name
    )


def create_local_qwen_planner_llm(model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
                                  device: str = "cuda:1",
                                  shared_model=None, shared_tokenizer=None) -> PlannerLLM:
    """Convenience function to create local Qwen PlannerLLM."""
    return PlannerLLM(
        model_type=ModelType.LOCAL_QWEN,
        model_path=model_path,
        device=device,
        shared_model=shared_model,
        shared_tokenizer=shared_tokenizer
    )


class PlannerModelConfig:
    """Configuration class for easy model selection and comparison."""

    @staticmethod
    def get_gpt4o_config(remote_model_manager) -> Dict[str, Any]:
        """Get configuration for GPT-4o model."""
        return {
            "model_type": ModelType.REMOTE_GPT4O,
            "remote_model_manager": remote_model_manager,
            "model_name": "gpt-4o"
        }

    @staticmethod
    def get_claude_config(remote_model_manager) -> Dict[str, Any]:
        """Get configuration for Claude model."""
        return {
            "model_type": ModelType.REMOTE_CLAUDE,
            "remote_model_manager": remote_model_manager,
            "model_name": "claude-3-sonnet"
        }

    @staticmethod
    def get_qwen_config(model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
                       device: str = "cuda:1",
                       shared_model=None, shared_tokenizer=None) -> Dict[str, Any]:
        """Get configuration for local Qwen model."""
        return {
            "model_type": ModelType.LOCAL_QWEN,
            "model_path": model_path,
            "device": device,
            "shared_model": shared_model,
            "shared_tokenizer": shared_tokenizer
        }

    @staticmethod
    def create_planner_from_config(config: Dict[str, Any]) -> PlannerLLM:
        """Create PlannerLLM from configuration dictionary."""
        return PlannerLLM(**config)
