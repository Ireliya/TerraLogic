"""
Model configuration for the spatial reasoning agent.
Handles secure API key management and model settings.
"""

import os
from typing import Dict, Any, Optional


class ModelConfiguration:
    """Configuration class for model settings and API keys."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        "remote_model": {
            "model_name": "gpt-4o",
            "base_url": "https://api.gpt.ge/v1/",
            "api_key_env": "SPATIAL_REASONING_API_KEY",
            "default_temperature": 0.1
        },
        "local_model": {
            "model_path": "model/Qwen2-VL-7B-Instruct",
            "device": "cuda:1",
            "default_temperature": 0.1
        }
    }
    
    @classmethod
    def get_api_key(cls) -> str:
        """
        Get API key from environment variable or return default for development.
        
        Returns:
            str: API key for remote model access
        """
        # Try to get from environment variable first
        api_key = os.getenv("SPATIAL_REASONING_API_KEY")
        
        if api_key:
            return api_key
        
        # Fallback to development key (should be replaced in production)
        development_key = "sk-1rqo5RNpB8RxQ6t5946490AbCcFa417dA170475492A0F34c"
        
        # Log warning about using development key
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Using development API key. Set SPATIAL_REASONING_API_KEY environment variable for production.")
        
        return development_key
    
    @classmethod
    def get_remote_model_config(cls) -> Dict[str, Any]:
        """Get configuration for remote GPT-4o model."""
        config = cls.DEFAULT_CONFIG["remote_model"].copy()
        config["api_key"] = cls.get_api_key()
        return config
    
    @classmethod
    def get_local_model_config(cls) -> Dict[str, Any]:
        """Get configuration for local Qwen2-VL model."""
        return cls.DEFAULT_CONFIG["local_model"].copy()
    
    @classmethod
    def get_default_model_type(cls) -> str:
        """Get default model type from environment or return 'remote'."""
        return os.getenv("SPATIAL_REASONING_MODEL_TYPE", "remote")
    
    @classmethod
    def is_remote_model_preferred(cls) -> bool:
        """Check if remote model should be used by default."""
        return cls.get_default_model_type().lower() == "remote"


# Convenience functions
def get_api_key() -> str:
    """Get API key for remote model access."""
    return ModelConfiguration.get_api_key()


def get_remote_config() -> Dict[str, Any]:
    """Get remote model configuration."""
    return ModelConfiguration.get_remote_model_config()


def get_local_config() -> Dict[str, Any]:
    """Get local model configuration."""
    return ModelConfiguration.get_local_model_config()


def get_default_model_type() -> str:
    """Get default model type."""
    return ModelConfiguration.get_default_model_type()
