"""
Local Model Wrapper for Qwen2-VL-7B-Instruct.
Maintains existing local-only functionality with direct imports and trust_remote_code=True.
"""

import torch
import logging
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from pathlib import Path
import sys

from .base_llm import ModelInterface


class LocalQwenModel(ModelInterface):
    """
    Local Qwen2-VL model wrapper that maintains existing functionality.
    Uses direct imports from local model files with trust_remote_code=True.
    """
    
    def __init__(self, 
                 model_path: str = "model/Qwen2-VL-7B-Instruct",
                 device: str = "cuda:1",
                 shared_model: Any = None,
                 shared_tokenizer: Any = None,
                 shared_processor: Any = None):
        """
        Initialize local Qwen2-VL model.
        
        Args:
            model_path: Path to local model directory
            device: Device to run model on
            shared_model: Pre-loaded model instance (prevents OOM)
            shared_tokenizer: Pre-loaded tokenizer instance
            shared_processor: Pre-loaded processor instance
        """
        self.model_path = Path(model_path).resolve()
        self.device = device
        self.model = shared_model
        self.tokenizer = shared_tokenizer
        self.processor = shared_processor
        self.logger = logging.getLogger(__name__)
        
        # Generation configuration
        self.generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.1,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": None  # Will be set after tokenizer loading
        }
        
        # Load model if not provided
        if self.model is None or self.tokenizer is None:
            self._load_model()
        else:
            self.logger.info("Using shared model instances to prevent CUDA OOM")
            # Set pad token for shared tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
    
    def _load_model(self):
        """Load Qwen2-VL model using direct imports from local files."""
        try:
            self.logger.info(f"Loading local Qwen2-VL model from {self.model_path}")
            
            # Add model path to sys.path for direct imports
            if str(self.model_path) not in sys.path:
                sys.path.insert(0, str(self.model_path))
            
            # Import model classes directly from local files
            from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
            from configuration_qwen2_vl import Qwen2VLConfig
            from transformers import AutoTokenizer, AutoProcessor
            
            # Load configuration
            config = Qwen2VLConfig.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Load model with explicit device assignment
            device_id = int(self.device.split(':')[1]) if self.device.startswith('cuda:') else 0
            with torch.cuda.device(device_id):
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    str(self.model_path),
                    config=config,
                    torch_dtype="auto",
                    device_map=self.device,
                    low_cpu_mem_usage=True
                )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
            
            self.logger.info(f"✅ Local Qwen2-VL model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load local Qwen2-VL model: {e}")
            raise
    
    def generate_text_response(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate text response from local Qwen2-VL model."""
        try:
            # Prepare messages for chat template
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate
            generation_config = self.generation_config.copy()
            generation_config["temperature"] = temperature
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Local model text generation failed: {e}")
            raise
    
    def generate_with_image(self, prompt: str, image: Union[str, Image.Image], temperature: float = 0.1) -> str:
        """Generate response with image input using local Qwen2-VL model."""
        try:
            # Load image if path provided
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            
            # Prepare messages with image
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Apply chat template and process
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            generation_config = self.generation_config.copy()
            generation_config["temperature"] = temperature
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Local model image generation failed: {e}")
            raise
    
    def generate_chat_response(self, messages: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        """Generate response from chat messages using local Qwen2-VL model."""
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate
            generation_config = self.generation_config.copy()
            generation_config["temperature"] = temperature
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"❌ Local model chat generation failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get local model information."""
        return {
            "model_type": "local",
            "model_name": "Qwen2-VL-7B-Instruct",
            "model_path": str(self.model_path),
            "device": self.device,
            "supports_vision": True,
            "supports_chat": True
        }
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        return (self.model is not None and 
                self.tokenizer is not None and 
                self.processor is not None)
