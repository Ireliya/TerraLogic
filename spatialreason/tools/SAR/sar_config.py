"""
Configuration management for SAR (Synthetic Aperture Radar) tools.
Provides simplified configuration abstraction over MMDetection's complex config system.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class SARModelConfig:
    """Configuration for SAR model parameters."""
    model_type: str = "HiViT"
    embed_dim: int = 512
    depths: List[int] = None
    num_heads: int = 8
    patch_size: int = 16
    img_size: int = 224
    use_checkpoint: bool = True
    
    def __post_init__(self):
        if self.depths is None:
            self.depths = [2, 2, 20]


@dataclass
class SARDetectionConfig:
    """Configuration for SAR detection parameters."""
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.5
    max_detections: int = 100
    input_size: int = 800
    dataset_type: str = "SSDD"  # SAR Ship Detection Dataset
    num_classes: int = 1  # Ships only for SSDD
    
    # Data preprocessing
    mean: List[float] = None
    std: List[float] = None
    
    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.0, 0.0, 0.0]  # SAR-specific normalization
        if self.std is None:
            self.std = [1.0, 1.0, 1.0]


@dataclass
class SARClassificationConfig:
    """Configuration for SAR classification parameters."""
    confidence_threshold: float = 0.5
    input_size: int = 224
    classification_types: Dict[str, List[str]] = None
    
    # Data preprocessing
    mean: List[float] = None
    std: List[float] = None
    
    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.0, 0.0, 0.0]  # SAR-specific normalization
        if self.std is None:
            self.std = [1.0, 1.0, 1.0]
        
        if self.classification_types is None:
            self.classification_types = {
                "scene": [
                    "urban", "rural", "coastal", "desert", "forest", "mountain", 
                    "agricultural", "industrial", "water", "mixed"
                ],
                "target": [
                    "ship", "aircraft", "vehicle", "building", "bridge", "tank",
                    "helicopter", "fighter_jet", "cargo_ship", "fishing_boat"
                ],
                "fine_grained": [
                    "T62_tank", "T72_tank", "BMP2_vehicle", "BTR60_vehicle", 
                    "BTR70_vehicle", "ZSU234_vehicle", "Boeing_aircraft", 
                    "Airbus_aircraft", "COMAC_aircraft", "cargo_ship", 
                    "fishing_boat", "tanker_ship", "naval_vessel"
                ]
            }


@dataclass
class SARSystemConfig:
    """System-level configuration for SAR tools."""
    device: str = "cuda:0"
    model_path: str = ""
    config_path: str = ""
    output_dir: str = "temp/detection"
    log_level: str = "INFO"
    
    # GPU management
    gpu_memory_fraction: float = 0.8
    allow_growth: bool = True
    
    def __post_init__(self):
        # Set default paths relative to SAR directory
        if not self.model_path:
            sar_dir = Path(__file__).parent
            self.model_path = str(sar_dir / "codelab" / "model" / "mae_hivit_base_1600ep.pth")
        
        if not self.config_path:
            sar_dir = Path(__file__).parent
            self.config_path = str(sar_dir / "codelab" / "code" / "detection" / "configs" / "_hivit_" / "hivit_base_SSDD.py")


class SARConfigManager:
    """
    Configuration manager for SAR tools.
    Provides simplified interface to complex MMDetection configurations.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        self.config_file = config_file
        self.model_config = SARModelConfig()
        self.detection_config = SARDetectionConfig()
        self.classification_config = SARClassificationConfig()
        self.system_config = SARSystemConfig()
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'model' in config_data:
                self.model_config = SARModelConfig(**config_data['model'])
            
            if 'detection' in config_data:
                self.detection_config = SARDetectionConfig(**config_data['detection'])
            
            if 'classification' in config_data:
                self.classification_config = SARClassificationConfig(**config_data['classification'])
            
            if 'system' in config_data:
                self.system_config = SARSystemConfig(**config_data['system'])
            
            print(f"[SARConfig] Loaded configuration from {config_file}")
            
        except Exception as e:
            print(f"[SARConfig] Warning: Failed to load config from {config_file}: {e}")
            print(f"[SARConfig] Using default configuration")
    
    def save_config(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        try:
            config_data = {
                'model': asdict(self.model_config),
                'detection': asdict(self.detection_config),
                'classification': asdict(self.classification_config),
                'system': asdict(self.system_config)
            }
            
            # Ensure output directory exists
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"[SARConfig] Saved configuration to {config_file}")
            
        except Exception as e:
            print(f"[SARConfig] Error saving config to {config_file}: {e}")
    
    def get_mmdet_config(self, task_type: str = "detection") -> Dict[str, Any]:
        """
        Generate MMDetection-compatible configuration.
        
        Args:
            task_type: Type of task ('detection' or 'classification')
            
        Returns:
            MMDetection configuration dictionary
        """
        if task_type == "detection":
            return self._get_detection_mmdet_config()
        elif task_type == "classification":
            return self._get_classification_mmdet_config()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _get_detection_mmdet_config(self) -> Dict[str, Any]:
        """Generate MMDetection config for detection task."""
        return {
            'model': {
                'type': 'FasterRCNN',
                'backbone': {
                    'type': 'HiViT',
                    'img_size': self.model_config.img_size,
                    'patch_size': self.model_config.patch_size,
                    'embed_dim': self.model_config.embed_dim,
                    'depths': self.model_config.depths,
                    'num_heads': self.model_config.num_heads,
                    'use_checkpoint': self.model_config.use_checkpoint,
                    'init_cfg': {
                        'type': 'Pretrained',
                        'checkpoint': self.system_config.model_path
                    }
                },
                'neck': {
                    'type': 'FPN',
                    'in_channels': [128, 256, 512, 512],
                    'out_channels': 256,
                    'num_outs': 5
                }
            },
            'data_preprocessor': {
                'type': 'DetDataPreprocessor',
                'mean': self.detection_config.mean,
                'std': self.detection_config.std,
                'bgr_to_rgb': True,
                'pad_size_divisor': 32
            },
            'test_cfg': {
                'rcnn': {
                    'score_thr': self.detection_config.confidence_threshold,
                    'nms': {
                        'type': 'nms',
                        'iou_threshold': self.detection_config.nms_threshold
                    },
                    'max_per_img': self.detection_config.max_detections
                }
            }
        }
    
    def _get_classification_mmdet_config(self) -> Dict[str, Any]:
        """Generate configuration for classification task."""
        return {
            'model': {
                'type': 'ImageClassifier',
                'backbone': {
                    'type': 'HiViT',
                    'img_size': self.model_config.img_size,
                    'patch_size': self.model_config.patch_size,
                    'embed_dim': self.model_config.embed_dim,
                    'depths': self.model_config.depths,
                    'num_heads': self.model_config.num_heads,
                    'init_cfg': {
                        'type': 'Pretrained',
                        'checkpoint': self.system_config.model_path
                    }
                },
                'head': {
                    'type': 'LinearClsHead',
                    'num_classes': len(self.classification_config.classification_types['scene']),
                    'in_channels': self.model_config.embed_dim
                }
            },
            'data_preprocessor': {
                'type': 'ClsDataPreprocessor',
                'mean': self.classification_config.mean,
                'std': self.classification_config.std,
                'to_rgb': True
            }
        }
    
    def update_detection_params(self, **kwargs) -> None:
        """Update detection configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.detection_config, key):
                setattr(self.detection_config, key, value)
            else:
                print(f"[SARConfig] Warning: Unknown detection parameter: {key}")
    
    def update_classification_params(self, **kwargs) -> None:
        """Update classification configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.classification_config, key):
                setattr(self.classification_config, key, value)
            else:
                print(f"[SARConfig] Warning: Unknown classification parameter: {key}")
    
    def update_system_params(self, **kwargs) -> None:
        """Update system configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.system_config, key):
                setattr(self.system_config, key, value)
            else:
                print(f"[SARConfig] Warning: Unknown system parameter: {key}")
    
    def validate_paths(self) -> bool:
        """Validate that required model and config paths exist."""
        model_path = Path(self.system_config.model_path)
        config_path = Path(self.system_config.config_path)
        
        if not model_path.exists():
            print(f"[SARConfig] Error: Model weights not found: {model_path}")
            return False
        
        if not config_path.exists():
            print(f"[SARConfig] Error: Config file not found: {config_path}")
            return False
        
        print(f"[SARConfig] ✅ All required paths validated")
        return True
    
    def get_summary(self) -> str:
        """Get configuration summary."""
        return f"""
SAR Configuration Summary:
- Model: {self.model_config.model_type} (embed_dim={self.model_config.embed_dim})
- Device: {self.system_config.device}
- Detection threshold: {self.detection_config.confidence_threshold}
- Classification threshold: {self.classification_config.confidence_threshold}
- Model path: {self.system_config.model_path}
- Output directory: {self.system_config.output_dir}
        """.strip()


# Global configuration instance
_global_config = None

def get_sar_config() -> SARConfigManager:
    """Get global SAR configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = SARConfigManager()
    return _global_config

def set_sar_config(config_manager: SARConfigManager) -> None:
    """Set global SAR configuration instance."""
    global _global_config
    _global_config = config_manager
