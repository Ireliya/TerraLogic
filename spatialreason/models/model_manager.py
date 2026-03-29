"""
Unified Model Manager for the Spatial Reasoning Agent.
Provides a single interface for model operations across all components.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from .model_factory import ModelFactory, ModelConfig, ModelType, ModelInterface


class UnifiedModelManager:
    """
    Unified model manager that provides a consistent interface for all model operations.
    Handles both local Qwen2-VL and remote GPT-4o models transparently.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize unified model manager.
        
        Args:
            config: Model configuration (defaults to local Qwen2-VL)
        """
        self.logger = logging.getLogger(__name__)
        self.factory = ModelFactory()
        
        # Use default local config if none provided
        if config is None:
            config = ModelConfig.local_config()
        
        self.config = config
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on configuration."""
        try:
            self.model = self.factory.create_model(
                self.config.model_type,
                **self.config.model_kwargs
            )
            self.logger.info(f"✅ Initialized {self.config.model_type.value} model")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model: {e}")
            raise
    
    def generate_text_response(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate text response using the configured model."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        return self.model.generate_text_response(prompt, temperature)
    
    def generate_with_image(self, prompt: str, image: Union[str, Image.Image], temperature: float = 0.1) -> str:
        """Generate response with image input using the configured model."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        return self.model.generate_with_image(prompt, image, temperature)
    
    def generate_chat_response(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        """Generate response from chat messages using the configured model."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        return self.model.generate_chat_response(messages, temperature)
    
    def get_model(self) -> ModelInterface:
        """Get the underlying model instance."""
        if not self.model:
            raise RuntimeError("Model not initialized")
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer if available (for local models)."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        # For local models, try to get tokenizer
        if hasattr(self.model, 'tokenizer'):
            return self.model.tokenizer

        # For remote models, tokenizer is not applicable
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.model:
            return {"status": "not_initialized"}
        return self.model.get_model_info()

    def is_available(self) -> bool:
        """Check if the model is available and ready."""
        if not self.model:
            return False
        return self.model.is_available()
    
    def switch_model(self, new_config: ModelConfig):
        """Switch to a different model configuration."""
        try:
            self.logger.info(f"Switching from {self.config.model_type.value} to {new_config.model_type.value} model")
            self.config = new_config
            self._initialize_model()
            self.logger.info("✅ Model switch completed successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to switch model: {e}")
            raise
    
    def get_current_config(self) -> ModelConfig:
        """Get the current model configuration."""
        return self.config
    
    def supports_vision(self) -> bool:
        """Check if the current model supports vision capabilities."""
        info = self.get_model_info()
        return info.get("supports_vision", False)
    
    def supports_chat(self) -> bool:
        """Check if the current model supports chat capabilities."""
        info = self.get_model_info()
        return info.get("supports_chat", False)


class ModelManagerBuilder:
    """Builder class for creating configured model managers."""
    
    def __init__(self):
        self.config = None
    
    def with_local_model(self, 
                        model_path: str = "model/Qwen2-VL-7B-Instruct",
                        device: str = "cuda:1",
                        shared_model: Any = None,
                        shared_tokenizer: Any = None,
                        shared_processor: Any = None) -> 'ModelManagerBuilder':
        """Configure for local Qwen2-VL model."""
        self.config = ModelConfig.local_config(
            model_path=model_path,
            device=device,
            shared_model=shared_model,
            shared_tokenizer=shared_tokenizer,
            shared_processor=shared_processor
        )
        return self
    
    def with_remote_model(self,
                         api_key: str = "sk-1rqo5RNpB8RxQ6t5946490AbCcFa417dA170475492A0F34c",
                         base_url: str = "https://api.gpt.ge/v1/",
                         model_name: str = "gpt-4o") -> 'ModelManagerBuilder':
        """Configure for remote GPT-4o model."""
        self.config = ModelConfig.remote_config(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )
        return self
    
    def with_config(self, config: ModelConfig) -> 'ModelManagerBuilder':
        """Configure with a custom ModelConfig."""
        self.config = config
        return self
    
    def build(self) -> UnifiedModelManager:
        """Build the model manager."""
        return UnifiedModelManager(self.config)


# Convenience functions
def create_local_model_manager(model_path: str = "model/Qwen2-VL-7B-Instruct",
                              device: str = "cuda:1",
                              shared_model: Any = None,
                              shared_tokenizer: Any = None,
                              shared_processor: Any = None) -> UnifiedModelManager:
    """Create a model manager with local Qwen2-VL model."""
    return (ModelManagerBuilder()
            .with_local_model(model_path, device, shared_model, shared_tokenizer, shared_processor)
            .build())


def create_remote_model_manager(api_key: str = "sk-1rqo5RNpB8RxQ6t5946490AbCcFa417dA170475492A0F34c",
                               base_url: str = "https://api.gpt.ge/v1/",
                               model_name: str = "gpt-4o") -> UnifiedModelManager:
    """Create a model manager with remote GPT-4o model."""
    return (ModelManagerBuilder()
            .with_remote_model(api_key, base_url, model_name)
            .build())


def create_model_manager_from_config(config: ModelConfig) -> UnifiedModelManager:
    """Create a model manager from a configuration."""
    return UnifiedModelManager(config)
