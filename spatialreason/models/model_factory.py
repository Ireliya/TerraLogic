"""
Model Factory for dynamic selection between local and remote models.
Provides runtime selection between Qwen2-VL-7B-Instruct and GPT-4o models.
"""

import logging
from typing import Any, Dict, Optional, Union
from enum import Enum

from .base_llm import ModelInterface
from .local_model import LocalQwenModel
from .remote_model import RemoteGPT4oModel


class ModelType(Enum):
    """Enumeration of supported model types."""
    LOCAL = "local"
    REMOTE = "remote"


class ModelFactory:
    """
    Factory class for creating and managing model instances.
    Supports runtime selection between local Qwen2-VL and remote GPT-4o models.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._model_cache = {}
    
    def create_model(self, 
                    model_type: Union[ModelType, str],
                    **kwargs) -> ModelInterface:
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create (local or remote)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            ModelInterface: Initialized model instance
        """
        # Convert string to enum if necessary
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())
        
        # Create cache key
        cache_key = f"{model_type.value}_{hash(frozenset(kwargs.items()))}"
        
        # Return cached instance if available
        if cache_key in self._model_cache:
            self.logger.info(f"Returning cached {model_type.value} model instance")
            return self._model_cache[cache_key]
        
        # Create new model instance
        if model_type == ModelType.LOCAL:
            model = self._create_local_model(**kwargs)
        elif model_type == ModelType.REMOTE:
            model = self._create_remote_model(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Cache the instance
        self._model_cache[cache_key] = model
        self.logger.info(f"Created and cached new {model_type.value} model instance")
        
        return model
    
    def _create_local_model(self, **kwargs) -> LocalQwenModel:
        """Create local Qwen2-VL model instance."""
        try:
            # Set default values for local model
            defaults = {
                "model_path": "model/Qwen2-VL-7B-Instruct",
                "device": "cuda:1"
            }

            # Filter out parameters that are not relevant for local models
            local_relevant_params = {
                'model_path', 'local_model_path', 'device', 'shared_model',
                'shared_tokenizer', 'shared_processor'
            }

            # Only include parameters that are relevant for local models
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in local_relevant_params}

            # Handle local_model_path -> model_path mapping
            if 'local_model_path' in filtered_kwargs and 'model_path' not in filtered_kwargs:
                filtered_kwargs['model_path'] = filtered_kwargs.pop('local_model_path')
            elif 'local_model_path' in filtered_kwargs:
                # Remove local_model_path if model_path is already present
                filtered_kwargs.pop('local_model_path')

            defaults.update(filtered_kwargs)

            self.logger.info(f"Creating local Qwen2-VL model with path: {defaults['model_path']}")
            return LocalQwenModel(**defaults)

        except Exception as e:
            self.logger.error(f"❌ Failed to create local model: {e}")
            raise
    
    def _create_remote_model(self, **kwargs) -> RemoteGPT4oModel:
        """Create remote GPT-4o model instance."""
        try:
            # Import secure configuration
            from ..config import get_remote_config

            # Set default values from secure configuration
            defaults = get_remote_config()
            # Remove api_key_env key if present (not needed for model init)
            defaults.pop("api_key_env", None)
            defaults.pop("default_temperature", None)

            # Filter out parameters that are not relevant for remote models
            remote_relevant_params = {
                'api_key', 'base_url', 'model_name'
            }

            # Only include parameters that are relevant for remote models
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in remote_relevant_params}
            defaults.update(filtered_kwargs)

            self.logger.info(f"Creating remote GPT-4o model with base_url: {defaults['base_url']}")
            return RemoteGPT4oModel(**defaults)

        except Exception as e:
            self.logger.error(f"❌ Failed to create remote model: {e}")
            raise
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        return {
            "local": {
                "name": "Qwen2-VL-7B-Instruct",
                "type": "local",
                "supports_vision": True,
                "supports_chat": True,
                "description": "Local Qwen2-VL model with direct imports"
            },
            "remote": {
                "name": "GPT-4o",
                "type": "remote", 
                "supports_vision": True,
                "supports_chat": True,
                "description": "Remote GPT-4o model via API"
            }
        }
    
    def clear_cache(self):
        """Clear the model cache."""
        self.logger.info(f"Clearing model cache ({len(self._model_cache)} instances)")
        self._model_cache.clear()
    
    def get_cached_models(self) -> Dict[str, str]:
        """Get information about cached models."""
        return {key: type(model).__name__ for key, model in self._model_cache.items()}


class ModelConfig:
    """Configuration class for model selection and parameters."""
    
    def __init__(self,
                 model_type: Union[ModelType, str] = ModelType.LOCAL,
                 **model_kwargs):
        """
        Initialize model configuration.

        Args:
            model_type: Type of model to use
            **model_kwargs: Additional model-specific parameters
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type.lower())

        self.model_type = model_type
        self.model_kwargs = model_kwargs

    @property
    def device(self) -> str:
        """Get the device setting from model kwargs."""
        return self.model_kwargs.get('device', 'cpu')

    @property
    def model_name(self) -> str:
        """Get the model name from model kwargs."""
        return self.model_kwargs.get('model_name', 'unknown')

    @property
    def local_model_path(self) -> Optional[str]:
        """Get the local model path from model kwargs."""
        return self.model_kwargs.get('local_model_path', None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_type": self.model_type.value,
            **self.model_kwargs
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        model_type = config_dict.pop("model_type", "local")
        return cls(model_type=model_type, **config_dict)
    
    @classmethod
    def local_config(cls, 
                    model_path: str = "model/Qwen2-VL-7B-Instruct",
                    device: str = "cuda:1",
                    **kwargs) -> 'ModelConfig':
        """Create configuration for local model."""
        return cls(
            model_type=ModelType.LOCAL,
            model_path=model_path,
            device=device,
            **kwargs
        )
    
    @classmethod
    def remote_config(cls,
                     api_key: str = None,
                     base_url: str = "https://api.gpt.ge/v1/",
                     model_name: str = "gpt-4o",
                     **kwargs) -> 'ModelConfig':
        """Create configuration for remote model."""
        # Use secure configuration if no API key provided
        if api_key is None:
            from ..config import get_api_key
            api_key = get_api_key()

        return cls(
            model_type=ModelType.REMOTE,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            **kwargs
        )


# Global factory instance
_factory_instance = None

def get_model_factory() -> ModelFactory:
    """Get global model factory instance."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = ModelFactory()
    return _factory_instance


def create_model(model_type: Union[ModelType, str], **kwargs) -> ModelInterface:
    """Convenience function to create a model using the global factory."""
    factory = get_model_factory()
    return factory.create_model(model_type, **kwargs)


# Import UnifiedModelManager for convenience functions
from .model_manager import UnifiedModelManager


# Convenience functions
def create_local_model_manager(model_path: str = "model/Qwen2-VL-7B-Instruct",
                              device: str = "cuda:1",
                              shared_model: Any = None,
                              shared_tokenizer: Any = None,
                              shared_processor: Any = None) -> UnifiedModelManager:
    """Create a model manager with local Qwen2-VL model."""
    from .model_manager import ModelManagerBuilder
    return (ModelManagerBuilder()
            .with_local_model(model_path, device, shared_model, shared_tokenizer, shared_processor)
            .build())


def create_remote_model_manager(api_key: str = None,
                               base_url: str = "https://api.gpt.ge/v1/",
                               model_name: str = "gpt-4o") -> UnifiedModelManager:
    """Create a model manager with remote GPT-4o model."""
    from .model_manager import ModelManagerBuilder
    return (ModelManagerBuilder()
            .with_remote_model(api_key, base_url, model_name)
            .build())


def create_model_manager_from_config(config: ModelConfig) -> UnifiedModelManager:
    """Create a model manager from a configuration."""
    return UnifiedModelManager(config)
