"""
Simple GPU utilities for spatial reasoning agent.
Provides basic GPU management and environment setup with hardcoded assignments.
"""

import torch
import os
from typing import List, Dict, Optional

# Custom GPU assignments as requested - Updated for single GPU system
HARDCODED_GPU_ASSIGNMENTS = {
    'opencompass': 'cuda:0',      # OpenCompass uses GPU 0
    'chat_model': 'cuda:0',       # Qwen2-VL language model uses GPU 0 (shared)
    'perception_tools': 'cuda:0', # All RemoteSAM tools share GPU 0 (shared)
    'reserve': 'cuda:0'           # GPU 0 reserved (shared)
}
from typing import List, Dict, Any
from pathlib import Path


# Removed SimpleGPUManager class - replaced with hardcoded assignments only


def setup_gpu_environment():
    """Setup GPU environment for optimal performance."""
    if torch.cuda.is_available():
        # Set memory fraction to avoid OOM - more conservative for single GPU
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Additional memory management for multi-GPU usage
        torch.cuda.empty_cache()
        num_gpus = torch.cuda.device_count()
        print(f"🔧 GPU environment configured for {num_gpus} GPU(s) with hardcoded allocation strategy")
    else:
        print("⚠️  CUDA not available, using CPU")


def load_prompts_from_file(prompt_file: str) -> Dict[str, str]:
    """Load prompts from a text file."""
    prompts = {}
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple parsing - look for sections marked with [SECTION_NAME]
        sections = content.split('[')
        for section in sections[1:]:  # Skip first empty section
            if ']' in section:
                name, text = section.split(']', 1)
                prompts[name.strip()] = text.strip()
        
        # If no sections found, use entire content as default
        if not prompts:
            prompts['WATER_FLOOD_ASSISTANT'] = content.strip()
            
    except FileNotFoundError:
        print(f"⚠️  Prompt file not found: {prompt_file}")
        # Provide default prompt
        prompts['WATER_FLOOD_ASSISTANT'] = """You are a spatial reasoning assistant specialized in remote sensing analysis. 
You can analyze satellite imagery to detect objects, segment regions, and perform spatial analysis tasks.
Use the available tools to help users with their spatial analysis needs."""
    
    return prompts


def get_hardcoded_device(component: str) -> str:
    """
    Get hardcoded device assignment for a specific component.

    Args:
        component: Component name ('chat_model', 'perception_tools', etc.)

    Returns:
        Device string (e.g., 'cuda:1')
    """
    return HARDCODED_GPU_ASSIGNMENTS.get(component, 'cpu')


def clear_gpu_cache(device: str) -> None:
    """
    Clear GPU cache for a specific device.

    Args:
        device: Device string (e.g., 'cuda:1')
    """
    if device.startswith('cuda:') and torch.cuda.is_available():
        try:
            gpu_id = int(device.split(':')[1])
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            print(f"✅ Cleared cache on {device}")
        except Exception as e:
            print(f"⚠️  Failed to clear cache on {device}: {e}")


def print_gpu_assignments() -> None:
    """Print the hardcoded GPU assignments."""
    print("🎯 Hardcoded GPU Assignments:")
    for component, device in HARDCODED_GPU_ASSIGNMENTS.items():
        print(f"   - {component}: {device}")
