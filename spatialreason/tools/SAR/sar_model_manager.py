"""
Model management system for SAR tools.
Handles model loading, initialization, and GPU management for SARATR-X models.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

try:
    from .sar_config import SARConfigManager, get_sar_config
except ImportError:
    from sar_config import SARConfigManager, get_sar_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SARModelManager:
    """
    Manages SAR model loading, initialization, and GPU allocation.
    Provides unified interface for both detection and classification models.
    """
    
    def __init__(self, config_manager: Optional[SARConfigManager] = None):
        """
        Initialize model manager.
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config = config_manager or get_sar_config()
        self._models = {}  # Cache for loaded models
        self._device_map = {}  # Track device assignments
        
        # Add SAR codebase to Python path
        self._setup_python_path()
        
        logger.info(f"[SARModelManager] Initialized with device: {self.config.system_config.device}")
    
    def _setup_python_path(self) -> None:
        """Add SAR codebase directories to Python path."""
        sar_base = Path(__file__).parent / "codelab" / "code"
        
        paths_to_add = [
            str(sar_base / "detection"),
            str(sar_base / "classification"),
            str(sar_base / "detection" / "mmdet"),
            str(sar_base)
        ]
        
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        logger.info(f"[SARModelManager] Added {len(paths_to_add)} paths to Python path")
    
    def validate_device(self, device: str) -> bool:
        """
        Validate that the specified device is available.
        
        Args:
            device: Device string (e.g., 'cuda:2', 'cpu')
            
        Returns:
            True if device is valid and available
        """
        try:
            if device.startswith('cuda:'):
                gpu_id = int(device.split(':')[1])
                
                if not torch.cuda.is_available():
                    logger.error(f"[SARModelManager] CUDA not available but {device} was requested")
                    return False
                
                if gpu_id >= torch.cuda.device_count():
                    available_gpus = torch.cuda.device_count()
                    logger.error(f"[SARModelManager] GPU {gpu_id} not available (only {available_gpus} GPUs detected)")
                    return False
                
                # Test GPU accessibility
                torch.cuda.set_device(gpu_id)
                test_tensor = torch.tensor([1.0], device=device)
                del test_tensor
                torch.cuda.empty_cache()
                
                logger.info(f"[SARModelManager] ✅ Device {device} validated and accessible")
                return True
                
            elif device == 'cpu':
                logger.info(f"[SARModelManager] ✅ CPU device validated")
                return True
            else:
                logger.error(f"[SARModelManager] Unsupported device: {device}")
                return False
                
        except Exception as e:
            logger.error(f"[SARModelManager] Device validation failed for {device}: {e}")
            return False
    
    def load_detection_model(self, device: Optional[str] = None) -> torch.nn.Module:
        """
        Load and initialize the SAR detection model.
        
        Args:
            device: Optional device override
            
        Returns:
            Loaded detection model
        """
        device = device or self.config.system_config.device
        model_key = f"detection_{device}"
        
        # Return cached model if available
        if model_key in self._models:
            logger.info(f"[SARModelManager] Using cached detection model on {device}")
            return self._models[model_key]
        
        # Validate device
        if not self.validate_device(device):
            raise RuntimeError(f"Device validation failed: {device}")
        
        try:
            logger.info(f"[SARModelManager] Loading detection model on {device}")
            
            # Import MMDetection components
            from mmengine.config import Config
            from mmdet.registry import MODELS
            
            # Import local HiViT model
            import models  # This imports the local HiViT implementation
            
            # Load configuration
            config_path = self.config.system_config.config_path
            cfg = Config.fromfile(config_path)
            
            # Update model configuration
            cfg.model.backbone.init_cfg.checkpoint = self.config.system_config.model_path
            
            # Build model
            model = MODELS.build(cfg.model)
            
            # Load pretrained weights
            checkpoint_path = self.config.system_config.model_path
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict with proper key mapping
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"[SARModelManager] Missing keys in checkpoint: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"[SARModelManager] Unexpected keys in checkpoint: {len(unexpected_keys)}")
            
            # Move to device and set to eval mode
            model = model.to(device)
            model.eval()
            
            # Cache the model
            self._models[model_key] = model
            self._device_map[model_key] = device
            
            logger.info(f"[SARModelManager] ✅ Detection model loaded successfully on {device}")
            return model
            
        except Exception as e:
            logger.error(f"[SARModelManager] Failed to load detection model: {e}")
            raise RuntimeError(f"Detection model loading failed: {e}")
    
    def load_classification_model(self, classification_type: str = "scene", 
                                device: Optional[str] = None) -> torch.nn.Module:
        """
        Load and initialize the SAR classification model.
        
        Args:
            classification_type: Type of classification ('scene', 'target', 'fine_grained')
            device: Optional device override
            
        Returns:
            Loaded classification model
        """
        device = device or self.config.system_config.device
        model_key = f"classification_{classification_type}_{device}"
        
        # Return cached model if available
        if model_key in self._models:
            logger.info(f"[SARModelManager] Using cached classification model ({classification_type}) on {device}")
            return self._models[model_key]
        
        # Validate device
        if not self.validate_device(device):
            raise RuntimeError(f"Device validation failed: {device}")
        
        try:
            logger.info(f"[SARModelManager] Loading classification model ({classification_type}) on {device}")
            
            # Import local HiViT model for classification
            try:
                from model.hivit import hivit_base
            except ImportError:
                # Fallback to creating a simple HiViT model
                logger.warning("[SARModelManager] Could not import hivit_base, using fallback implementation")
                model = self._create_fallback_classification_model(classification_type)
            else:
                # Get number of classes for the classification type
                class_mappings = self.config.classification_config.classification_types
                num_classes = len(class_mappings.get(classification_type, class_mappings["scene"]))
                
                # Create model
                model = hivit_base(num_classes=num_classes)
            
            # Load pretrained weights
            checkpoint_path = self.config.system_config.model_path
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict (classification head may be randomly initialized)
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                logger.info(f"[SARModelManager] Loaded pretrained weights (classification head may be random)")
            except Exception as e:
                logger.warning(f"[SARModelManager] Could not load all weights: {e}")
                logger.info(f"[SARModelManager] Using randomly initialized model")
            
            # Move to device and set to eval mode
            model = model.to(device)
            model.eval()
            
            # Cache the model
            self._models[model_key] = model
            self._device_map[model_key] = device
            
            logger.info(f"[SARModelManager] ✅ Classification model ({classification_type}) loaded successfully on {device}")
            return model
            
        except Exception as e:
            logger.error(f"[SARModelManager] Failed to load classification model: {e}")
            raise RuntimeError(f"Classification model loading failed: {e}")
    
    def _create_fallback_classification_model(self, classification_type: str) -> torch.nn.Module:
        """Create a fallback classification model when HiViT import fails."""
        logger.info("[SARModelManager] Creating fallback classification model")
        
        # Get number of classes
        class_mappings = self.config.classification_config.classification_types
        num_classes = len(class_mappings.get(classification_type, class_mappings["scene"]))
        
        # Create a simple CNN-based classifier as fallback
        class FallbackClassifier(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 7 * 7, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        return FallbackClassifier(num_classes)
    
    def get_model(self, model_type: str, **kwargs) -> torch.nn.Module:
        """
        Get a model by type.
        
        Args:
            model_type: Type of model ('detection' or 'classification')
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model
        """
        if model_type == "detection":
            return self.load_detection_model(**kwargs)
        elif model_type == "classification":
            return self.load_classification_model(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def clear_cache(self, model_type: Optional[str] = None) -> None:
        """
        Clear model cache.
        
        Args:
            model_type: Optional model type to clear ('detection' or 'classification')
                       If None, clears all cached models
        """
        if model_type is None:
            # Clear all models
            for model_key in list(self._models.keys()):
                del self._models[model_key]
                if model_key in self._device_map:
                    del self._device_map[model_key]
            logger.info("[SARModelManager] Cleared all cached models")
        else:
            # Clear specific model type
            keys_to_remove = [key for key in self._models.keys() if key.startswith(model_type)]
            for key in keys_to_remove:
                del self._models[key]
                if key in self._device_map:
                    del self._device_map[key]
            logger.info(f"[SARModelManager] Cleared {len(keys_to_remove)} {model_type} models from cache")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "loaded_models": list(self._models.keys()),
            "device_map": self._device_map.copy(),
            "config_summary": self.config.get_summary()
        }


# Global model manager instance
_global_model_manager = None

def get_sar_model_manager() -> SARModelManager:
    """Get global SAR model manager instance."""
    global _global_model_manager
    if _global_model_manager is None:
        _global_model_manager = SARModelManager()
    return _global_model_manager

def set_sar_model_manager(model_manager: SARModelManager) -> None:
    """Set global SAR model manager instance."""
    global _global_model_manager
    _global_model_manager = model_manager
