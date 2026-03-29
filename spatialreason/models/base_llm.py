"""
Base LLM classes for unified model management across the spatial reasoning agent.
Provides model abstraction layer supporting both local Qwen2-VL and remote GPT-4o models.
"""

import torch
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from ..utils.qwen_utils import load_qwen_model, get_qwen_embedding


class ModelInterface(ABC):
    """
    Abstract interface for all model implementations in the spatial reasoning agent.
    Provides unified methods for both local and remote models.
    """

    @abstractmethod
    def generate_text_response(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate text response from the model."""
        pass

    @abstractmethod
    def generate_with_image(self, prompt: str, image: Union[str, Image.Image], temperature: float = 0.1) -> str:
        """Generate response with image input."""
        pass

    @abstractmethod
    def generate_chat_response(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        """Generate response from chat messages."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and ready."""
        pass


class BaseLLM(ABC):
    """
    Abstract base class for all LLM wrappers in the spatial reasoning agent.
    Provides common functionality for model loading, conversation management, and embedding generation.
    """

    def __init__(self, model=None, tokenizer=None, device="auto", model_path="Qwen/Qwen2-VL-7B-Instruct"):
        """
        Initialize base LLM with common configuration.

        Args:
            model: Pre-loaded Qwen2-VL model instance
            tokenizer: Pre-loaded Qwen tokenizer instance
            device: Device for model operations ("auto" for automatic GPU selection, "cuda:0", "cpu")
            model_path: Hugging Face model ID for loading
        """
        self.model = model
        self.tokenizer = tokenizer

        # Handle automatic GPU selection with hardcoded assignments
        if device == "auto":
            # Check for hardcoded GPU assignment
            import os
            if os.getenv('SPATIAL_REASONING_GPU_MODE') == 'hardcoded':
                lang_gpu = os.getenv('SPATIAL_REASONING_LANG_GPU', '1')
                self.device = f"cuda:{lang_gpu}"
                print(f"🎯 {self.__class__.__name__} using hardcoded GPU assignment: {self.device}")
            else:
                # Use hardcoded GPU 1 for language models
                self.device = "cuda:1"
                print(f"🎯 {self.__class__.__name__} using hardcoded GPU assignment: {self.device}")
        else:
            self.device = device

        self.model_path = model_path
        self.conversation_history = []
        self._generation_config = {
            "max_tokens": 1024,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.9
        }

        # Load model if not provided
        if model is None or tokenizer is None:
            self._load_model()
    
    def _load_model(self):
        """Load Qwen2-VL model from Hugging Face Hub using centralized utility."""
        try:
            print(f"🌐 Loading {self.__class__.__name__} model from Hugging Face Hub...")
            self.tokenizer, self.model = load_qwen_model(
                model_path=self.model_path,
                device=self.device
            )
            print(f"✅ {self.__class__.__name__} model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load model for {self.__class__.__name__}: {e}")
            raise e
    
    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding for text using centralized utility.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding tensor
        """
        return get_qwen_embedding(text, self.model, self.tokenizer, self.device)
    
    def set_generation_config(self, **kwargs):
        """Update generation configuration."""
        self._generation_config.update(kwargs)
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def add_system_message(self, content: str):
        """Add system message to conversation."""
        self.conversation_history.append({"role": "system", "content": content})
    
    def add_user_message(self, content: str):
        """Add user message to conversation."""
        self.conversation_history.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content: str):
        """Add assistant message to conversation."""
        self.conversation_history.append({"role": "assistant", "content": content})
    
    @abstractmethod
    def generate_response(self, max_tokens: int = None) -> str:
        """Generate response using the model. Must be implemented by subclasses."""
        pass
    
    def _format_conversation_for_generation(self) -> str:
        """Format conversation history for model generation."""
        conversation_text = ""
        for msg in self.conversation_history:
            role = msg["role"].title()
            content = msg["content"]
            conversation_text += f"{role}: {content}\n\n"
        
        conversation_text += "Assistant: "
        return conversation_text



