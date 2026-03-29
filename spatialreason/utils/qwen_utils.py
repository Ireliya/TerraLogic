"""
Qwen model utilities for spatial reasoning agent.
Provides centralized model loading and embedding functions.
"""

import torch
from typing import Tuple, Any
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, Qwen2VLForConditionalGeneration


def load_qwen_model(model_path: str, device: str = "cuda:1", use_direct_imports: bool = False) -> Tuple[Any, Any]:
    """
    Load Qwen2-VL model and tokenizer from Hugging Face Hub or local path.

    Args:
        model_path: Hugging Face model ID or local path
        device: Device to load model on
        use_direct_imports: Whether to use direct imports from local model files

    Returns:
        Tuple of (tokenizer, model)
    """
    try:
        if use_direct_imports:
            print(f"🚀 Loading Qwen model using direct imports from {model_path}...")
            return _load_qwen_with_direct_imports(model_path, device)
        else:
            print(f"🚀 Loading Qwen model using standard transformers from {model_path}...")
            return _load_qwen_with_transformers(model_path, device)

    except Exception as e:
        print(f"❌ Failed to load Qwen model: {e}")
        raise e


def _load_qwen_with_direct_imports(model_path: str, device: str) -> Tuple[Any, Any]:
    """Load Qwen model using direct imports from local files."""
    import sys
    from pathlib import Path

    # Add the local model path to sys.path
    model_path = Path(model_path).resolve()
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))

    try:
        # Import model classes directly
        from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        from configuration_qwen2_vl import Qwen2VLConfig
        from transformers import AutoTokenizer

        # Load configuration and tokenizer
        config = Qwen2VLConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # Load model with explicit device assignment (no auto device mapping)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype="auto",
            device_map=device,  # Use explicit device instead of "auto"
            low_cpu_mem_usage=True
        )

        return tokenizer, model

    except ImportError as e:
        print(f"[ERROR] Direct import failed: {e}")
        print("Falling back to transformers approach...")
        return _load_qwen_with_transformers(str(model_path), device)


def _load_qwen_with_transformers(model_path: str, device: str) -> Tuple[Any, Any]:
    """Load Qwen model using standard transformers approach."""
    from transformers import AutoConfig, AutoTokenizer, Qwen2VLForConditionalGeneration

    # First, explicitly load the config with trust_remote_code=True
    print(f"🔧 Loading config for {model_path} with trust_remote_code=True...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"[DEBUG] Config loaded successfully: {config.model_type}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    # Load model using Qwen2VLForConditionalGeneration with explicit device assignment
    print(f"[DEBUG] Using Qwen2VLForConditionalGeneration for {config.model_type} model on {device}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,  # Pass the pre-loaded config
        trust_remote_code=True,
        torch_dtype="auto",  # Use "auto" for better compatibility
        device_map=device,   # Use explicit device instead of "auto"
        low_cpu_mem_usage=True
    )

    return tokenizer, model


def get_qwen_embedding(text: str, model: Any, tokenizer: Any) -> torch.Tensor:
    """
    Get embedding for text using Qwen model.
    
    Args:
        text: Input text
        model: Qwen model instance
        tokenizer: Qwen tokenizer instance
        
    Returns:
        Text embedding tensor
    """
    try:
        # Tokenize text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to model device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use last hidden state mean as embedding
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Failed to get embedding: {e}")
        # Return zero embedding as fallback
        return torch.zeros(1, 768)  # Default embedding size
