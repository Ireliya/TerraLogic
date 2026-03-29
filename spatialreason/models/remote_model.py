"""
Remote Model Wrapper for GPT-4o.
Uses existing client2 module infrastructure with GPT-4o model.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import base64
import io

from .base_llm import ModelInterface


class RemoteGPT4oModel(ModelInterface):
    """
    Remote model wrapper using existing client2 infrastructure.
    Supports various remote models including GPT-4o, Claude Sonnet 4, etc.
    """
    
    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://api.gpt.ge/v1/",
                 model_name: str = "gpt-4o"):
        """
        Initialize remote model using existing client2 infrastructure.

        Args:
            api_key: API key for remote service (if None, will use secure config)
            base_url: Base URL for API relay
            model_name: Model name (e.g., gpt-4o, claude-sonnet-4-20250514, etc.)
        """
        # Use secure configuration if no API key provided
        if api_key is None:
            from ..config import get_api_key
            api_key = get_api_key()

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client with existing configuration
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers={"x-foo": "true"}
            )
            self.logger.info(f"✅ Remote model client initialized: {model_name} via {base_url}")
        except ImportError as e:
            self.logger.error(f"❌ Failed to import OpenAI client: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize remote model client ({model_name}): {e}")
            raise
    
    def generate_text_response(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate text response from remote model."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                temperature=temperature
            )

            result = response.choices[0].message.content.strip()
            self.logger.info(f"✅ {self.model_name} text generation successful")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {self.model_name} text generation failed: {e}")
            raise
    
    def generate_with_image(self, prompt: str, image: Union[str, Image.Image], temperature: float = 0.1) -> str:
        """Generate response from remote model with image input."""
        try:
            # Convert image to base64
            if isinstance(image, str):
                # Image path provided
                image_url = self._image_path_to_base64(image)
            else:
                # PIL Image provided
                image_url = self._pil_image_to_base64(image)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': image_url}}
                    ],
                }],
                temperature=temperature
            )
            
            result = response.choices[0].message.content.strip()
            self.logger.info(f"✅ {self.model_name} image generation successful")
            return result

        except Exception as e:
            self.logger.error(f"❌ {self.model_name} image generation failed: {e}")
            raise
    
    def generate_chat_response(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        """Generate response from chat messages using GPT-4o."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            
            result = response.choices[0].message.content.strip()
            self.logger.info(f"✅ GPT-4o chat generation successful")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ GPT-4o chat generation failed: {e}")
            raise
    
    def _image_path_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 data URL."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine image format
            image_format = image_path.lower().split('.')[-1]
            if image_format == 'jpg':
                image_format = 'jpeg'
            
            return f"data:image/{image_format};base64,{encoded_string}"
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert image path to base64: {e}")
            raise
    
    def _pil_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            buffer.seek(0)
            
            # Encode to base64
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert PIL image to base64: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get remote model information."""
        return {
            "model_type": "remote",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "supports_vision": True,
            "supports_chat": True
        }
    
    def is_available(self) -> bool:
        """Check if remote model is available by testing a simple request."""
        try:
            # Test with a simple prompt
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0.1
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            self.logger.warning(f"GPT-4o availability check failed: {e}")
            return False

    def to(self, device):
        """
        Compatibility method for PyTorch model interface.
        Remote models don't need device assignment, so this is a no-op.

        Args:
            device: Target device (ignored for remote models)

        Returns:
            self: Returns self for method chaining compatibility
        """
        self.logger.debug(f"Remote model .to({device}) called - no-op for remote models")
        return self


# Backward compatibility alias
GPT4oClient = RemoteGPT4oModel
