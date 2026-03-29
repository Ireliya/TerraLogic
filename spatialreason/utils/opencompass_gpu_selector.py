"""
OpenCompass GPU selector for handling GPU failures and failover.
"""

import torch
from typing import Optional


def handle_opencompass_gpu_failure(failed_device: str) -> str:
    """
    Handle GPU failure during OpenCompass evaluation with adaptive fallback.

    Args:
        failed_device: The device that failed (e.g., 'auto', 'cuda:1')

    Returns:
        Fallback device string or raises RuntimeError if no fallback possible
    """
    print(f"❌ GPU failure detected for device: {failed_device}")

    # Check if CUDA is available at all
    if not torch.cuda.is_available():
        print("🔄 CUDA not available, falling back to CPU")
        return "cpu"

    num_gpus = torch.cuda.device_count()
    print(f"🔍 {num_gpus} GPUs visible in current CUDA context")

    # If the failed device was "auto", try to find a working GPU
    if failed_device == "auto":
        # Test GPUs in order of preference
        for gpu_id in range(min(num_gpus, 3)):  # Test up to 3 GPUs
            test_device = f"cuda:{gpu_id}"
            if _test_gpu_accessibility(test_device):
                print(f"🔄 Found working GPU: {test_device}")
                return test_device

        # If no GPU works, fall back to CPU
        print("🔄 No working GPUs found, falling back to CPU")
        return "cpu"

    # If a specific CUDA device failed, check if it's due to invalid device ordinal
    if failed_device.startswith('cuda:'):
        gpu_id = int(failed_device.split(':')[1])
        if gpu_id >= num_gpus:
            print(f"🔄 Device {failed_device} not available (only {num_gpus} GPUs visible)")
            # Try to find an alternative GPU
            for alt_gpu_id in range(num_gpus):
                alt_device = f"cuda:{alt_gpu_id}"
                if _test_gpu_accessibility(alt_device):
                    print(f"🔄 Using alternative GPU: {alt_device}")
                    return alt_device

            # No working GPU found, fall back to CPU
            print("🔄 No alternative GPUs available, falling back to CPU")
            return "cpu"

    # For other failures, provide CPU fallback
    print(f"🔄 GPU failover failed: Device {failed_device} failed. Under hardcoded allocation strategy, OpenCompass must use cuda:0. Please check device assignments.")
    return "cpu"


def _test_gpu_accessibility(device: str) -> bool:
    """
    Test if a GPU device is accessible.
    
    Args:
        device: Device string (e.g., 'cuda:0')
        
    Returns:
        True if device is accessible, False otherwise
    """
    try:
        if device.startswith('cuda:'):
            gpu_id = int(device.split(':')[1])
            
            # Check if GPU ID is valid
            if gpu_id >= torch.cuda.device_count():
                return False
                
            # Test basic operations on the device
            torch.cuda.set_device(gpu_id)
            test_tensor = torch.tensor([1.0], device=device)
            _ = test_tensor + 1  # Simple operation
            del test_tensor
            torch.cuda.empty_cache()
            return True
            
    except Exception as e:
        print(f"⚠️  GPU {device} not accessible: {e}")
        return False
    
    return False


def get_recommended_device_for_opencompass() -> str:
    """
    Get recommended device for OpenCompass evaluation with adaptive fallback.

    Returns:
        Device string (cuda:0 preferred, or best available alternative)
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, OpenCompass will use CPU")
        return "cpu"

    num_gpus = torch.cuda.device_count()
    print(f"🔍 {num_gpus} GPUs visible for OpenCompass")

    # Prefer GPU 0 according to hardcoded allocation strategy
    if num_gpus >= 1 and _test_gpu_accessibility("cuda:0"):
        print("✅ Using preferred GPU 0 for OpenCompass")
        return "cuda:0"

    # If GPU 0 not available, try other GPUs
    for gpu_id in range(num_gpus):
        test_device = f"cuda:{gpu_id}"
        if _test_gpu_accessibility(test_device):
            print(f"🔄 Using alternative GPU {gpu_id} for OpenCompass")
            return test_device

    # If no GPU works, fall back to CPU
    print("⚠️  No accessible GPUs found, OpenCompass will use CPU")
    return "cpu"


def print_gpu_status():
    """Print current GPU status for debugging."""
    print("🔍 GPU Status Report:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"   Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            device = f"cuda:{i}"
            accessible = _test_gpu_accessibility(device)
            status = "✅ Working" if accessible else "❌ Not accessible"
            
            try:
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB) - {status}")
            except Exception:
                print(f"   GPU {i}: Unknown - {status}")
    else:
        print("   No CUDA GPUs available")
