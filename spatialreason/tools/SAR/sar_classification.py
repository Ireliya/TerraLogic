"""
SAR (Synthetic Aperture Radar) Classification Tool using SARATR-X model.
This tool classifies SAR imagery into different scene types and target categories.
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add the classification codebase to Python path
SAR_CODEBASE_PATH = Path(__file__).parent / "SARATR-X" / "classification"
if str(SAR_CODEBASE_PATH) not in sys.path:
    sys.path.insert(0, str(SAR_CODEBASE_PATH))

# SAR dataset standard GSD (Ground Sample Distance)
SAR_DEFAULT_GSD = 3.0  # meters per pixel for OGSOD dataset


class SARClassificationInput(BaseModel):
    """Input schema for SAR classification tool."""
    image_path: str = Field(description="Path to SAR image file")
    classification_type: str = Field(default="scene", description="Type of classification ('scene', 'target', 'fine_grained')")
    confidence_threshold: float = Field(default=0.5, description="Minimum confidence threshold (0.0-1.0)")
    device: str = Field(default="cuda:0", description="Device for inference (cuda:0, cuda:1, etc.)")

    # Optional parameters for unified arguments field (from upstream tools)
    text_prompt: Optional[str] = Field(default="classify SAR scene", description="Task intent description (e.g., 'classify SAR scene')")
    meters_per_pixel: Optional[float] = Field(default=SAR_DEFAULT_GSD, description="Ground resolution for spatial analysis (3.0 for OGSOD dataset)")


class SARClassificationTool(BaseTool):
    """
    SAR Classification tool using SARATR-X model for scene and target classification.
    Supports multiple classification types: scene, target, and fine-grained.
    """

    name: str = "sar_classification"
    description: str = "Classify SAR imagery into scene types, target categories, or fine-grained classes"
    args_schema: Type[BaseModel] = SARClassificationInput

    # Private attributes for the model
    _model: Optional[Any] = None
    _device: str = "cuda:2"
    _model_path: Optional[str] = None
    _class_mappings: Dict[str, List[str]] = {}

    def __init__(self, device: str = "cuda:0", **kwargs):
        super().__init__(**kwargs)
        print(f"[SARClassification] Initializing SAR classification tool with device: {device}")
        self._device = device
        self._validate_device()
        self._setup_model_paths()
        self._setup_class_mappings()
        self._initialize_model()

    def _validate_device(self) -> None:
        """Validate and ensure the specified CUDA device is available."""
        if self._device.startswith('cuda:'):
            try:
                gpu_id = int(self._device.split(':')[1])
                if not torch.cuda.is_available():
                    raise RuntimeError(f"[SARClassification] CUDA not available but {self._device} was requested")
                elif gpu_id >= torch.cuda.device_count():
                    available_gpus = torch.cuda.device_count()
                    raise RuntimeError(f"[SARClassification] GPU {gpu_id} not available (only {available_gpus} GPUs detected)")
                else:
                    # Test GPU accessibility
                    torch.cuda.set_device(gpu_id)
                    test_tensor = torch.tensor([1.0], device=self._device)
                    del test_tensor
                    torch.cuda.empty_cache()
                    print(f"[SARClassification] ✅ GPU {gpu_id} validated and accessible")
            except (ValueError, IndexError, RuntimeError) as e:
                raise RuntimeError(f"[SARClassification] Device validation failed: {e}")

    def _setup_model_paths(self) -> None:
        """Setup paths for model weights and configuration files."""
        # Try to find available model files in order of preference
        possible_models = [
            Path(__file__).parent / "latest.pth",
            Path(__file__).parent / "checkpoint-800.pth",
            Path(__file__).parent / "codelab" / "model" / "mae_hivit_base_1600ep.pth"
        ]

        self._model_path = None
        for model_path in possible_models:
            if model_path.exists():
                self._model_path = model_path
                break

        if self._model_path is None:
            print(f"[SARClassification] Warning: No model weights found. Available files:")
            sar_dir = Path(__file__).parent
            for file in sar_dir.glob("*.pth"):
                print(f"  - {file}")
            # Use the first available .pth file as fallback
            pth_files = list(sar_dir.glob("*.pth"))
            if pth_files:
                self._model_path = pth_files[0]
                print(f"[SARClassification] Using fallback model: {self._model_path}")
            else:
                raise FileNotFoundError(f"[SARClassification] No .pth model files found in {sar_dir}")

        print(f"[SARClassification] Model path: {self._model_path}")

    def _setup_class_mappings(self) -> None:
        """Setup class mappings for different classification types."""
        self._class_mappings = {
            "scene": [
                "urban", "coastal"
            ],
            "target": [
                "ship", "aircraft", "vehicle", "building", "bridge", "tank",
                "helicopter", "fighter_jet", "cargo_ship", "fishing_boat"
            ],
            "fine_grained": [
                # Military vehicles
                "T62_tank", "T72_tank", "BMP2_vehicle", "BTR60_vehicle", 
                "BTR70_vehicle", "ZSU234_vehicle",
                # Aircraft
                "Boeing_aircraft", "Airbus_aircraft", "COMAC_aircraft",
                # Ships
                "cargo_ship", "fishing_boat", "tanker_ship", "naval_vessel"
            ]
        }

    def _initialize_model(self) -> None:
        """Initialize the HiViT model for SAR classification."""
        try:
            print(f"[SARClassification] Loading HiViT model from {self._model_path}")
            
            # Import local HiViT model
            try:
                sys.path.insert(0, str(Path(__file__).parent / "SARATR-X" / "classification"))
                from model.hivit import hivit_base
            except ImportError as e:
                raise ImportError(f"[SARClassification] Failed to import HiViT model: {e}")
            
            # Create model with appropriate number of classes
            # Use scene classification as default (can be adapted for other types)
            num_classes = len(self._class_mappings["scene"])
            model = hivit_base(num_classes=num_classes)
            
            # Load pretrained weights
            checkpoint = torch.load(str(self._model_path), map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict (may need adaptation for classification head)
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"[SARClassification] Loaded pretrained weights (some layers may be randomly initialized)")
            except Exception as e:
                print(f"[SARClassification] Warning: Could not load all weights: {e}")
                print(f"[SARClassification] Using randomly initialized classification head")
            
            # Move to device and set to eval mode
            model = model.to(self._device)
            model.eval()
            
            self._model = model
            print(f"[SARClassification] ✅ Model loaded successfully on {self._device}")
            
        except Exception as e:
            print(f"[SARClassification] ❌ Failed to initialize model: {e}")
            raise RuntimeError(f"SAR classification model initialization failed: {e}")

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess SAR image for classification.
        
        Args:
            image_path: Path to input SAR image
            
        Returns:
            Preprocessed tensor ready for model inference
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to RGB (SAR images are typically grayscale, but model expects 3 channels)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (224x224 for classification)
        target_size = 224
        image_resized = cv2.resize(image, (target_size, target_size))
        
        # Normalize for SAR data
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self._device)
        
        return image_tensor

    def _classify_image(self, image_tensor: torch.Tensor, classification_type: str,
                       confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Classify the preprocessed image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor
            classification_type: Type of classification ('scene', 'target', 'fine_grained')
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Classification results dictionary
        """
        with torch.no_grad():
            # Forward pass
            logits = self._model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(5, probabilities.size(1)), dim=1)
            
            # Convert to numpy
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
            
            # Get class names for the classification type
            class_names = self._class_mappings.get(classification_type, self._class_mappings["scene"])
            
            # Build results - STRICT confidence threshold enforcement
            predictions = []
            for prob, idx in zip(top_probs, top_indices):
                if idx < len(class_names) and prob >= confidence_threshold:
                    predictions.append({
                        "class": class_names[idx],
                        "confidence": round(float(prob), 3),
                        "class_index": int(idx)
                    })

            # CRITICAL FIX: Do NOT include predictions below confidence threshold
            # This ensures semantic consistency with queries that reference classification results
            # If no predictions meet threshold, return empty results to indicate classification failure

            return {
                "classification_type": classification_type,
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None,
                "confidence_threshold_met": len(predictions) > 0
            }

    def _run(
        self,
        image_path: str,
        classification_type: Optional[str] = "scene",
        confidence_threshold: Optional[float] = 0.5,
        device: Optional[str] = None,
        text_prompt: Optional[str] = None,
        meters_per_pixel: Optional[float] = None
    ) -> str:
        """
        Execute SAR classification on the input image.

        Args:
            image_path: Path to input SAR image
            classification_type: Type of classification ('scene', 'target', 'fine_grained')
            confidence_threshold: Minimum confidence threshold
            device: CUDA device (optional, uses initialized device)
            text_prompt: Task intent description (optional)
            meters_per_pixel: Ground resolution for spatial analysis (optional, 3.0 for OGSOD dataset)

        Returns:
            JSON string containing classification results
        """
        try:
            print(f"[SARClassification] Starting SAR classification for: {image_path}")
            print(f"[SARClassification] Classification type: {classification_type}")
            print(f"[SARClassification] Confidence threshold: {confidence_threshold}")
            
            # Validate input image
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")
            
            # Validate classification type
            if classification_type not in self._class_mappings:
                classification_type = "scene"
                print(f"[SARClassification] Invalid classification type, defaulting to 'scene'")
            
            # Preprocess image
            image_tensor = self._preprocess_image(image_path)
            print(f"[SARClassification] Image preprocessed: {image_tensor.shape}")
            
            # Classify image
            classification_results = self._classify_image(
                image_tensor, classification_type, confidence_threshold
            )
            
            # Create output directory
            image_id = Path(image_path).stem
            output_dir = Path("temp") / "detection" / image_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare results
            result = {
                "success": True,
                "image_path": image_path,
                "classification_results": classification_results,
                "confidence_threshold": confidence_threshold,
                "model_info": {
                    "model_type": "HiViT-SARATR-X",
                    "classification_type": classification_type,
                    "device": self._device,
                    "available_classes": self._class_mappings[classification_type]
                }
            }
            
            # Add natural language summary
            if classification_results["top_prediction"]:
                top_pred = classification_results["top_prediction"]
                result["summary"] = (f"SAR classification completed. "
                                   f"Image classified as '{top_pred['class']}' "
                                   f"with {top_pred['confidence']:.1%} confidence.")
            else:
                result["summary"] = "SAR classification completed but no confident predictions found."
            
            print(f"[SARClassification] ✅ Classification completed successfully")

            # ========== UNIFIED ARGUMENTS FIELD ==========
            # Construct unified arguments field combining input configuration and output statistics
            # This matches the format used in perception tools, spatial relation tools, and statistical tools for consistency

            # Build classes_requested list from the classification type
            classes_requested = self._class_mappings.get(classification_type, self._class_mappings["scene"])

            # Sort classes alphabetically for consistency
            classes_requested_sorted = sorted(classes_requested)

            # Get top prediction class
            top_prediction_class = None
            if classification_results["top_prediction"]:
                top_prediction_class = classification_results["top_prediction"]["class"]

            # Calculate class_stats: coverage percentage for each predicted class
            # For classification, we use confidence scores as a proxy for coverage
            # since classification produces probabilities rather than spatial masks
            class_stats = {}
            predictions = classification_results.get("predictions", [])
            for pred in predictions:
                class_name = pred.get("class")
                confidence = pred.get("confidence", 0.0)
                if class_name:
                    # Convert confidence (0-1) to coverage percentage (0-100)
                    coverage_pct = round(confidence * 100.0, 2)
                    class_stats[class_name] = {"coverage_pct": coverage_pct}

            # Set meters_per_pixel to SAR default if not provided
            gsd = meters_per_pixel if meters_per_pixel is not None else SAR_DEFAULT_GSD

            # Build the unified arguments field with 6 fields
            arguments = {
                "image_path": image_path,
                "classes_requested": classes_requested_sorted,
                "text_prompt": text_prompt if text_prompt else "classify SAR scene",
                "meters_per_pixel": gsd,
                "top_prediction_class": top_prediction_class,
                "class_stats": class_stats
            }

            # Add unified arguments field to result
            result["arguments"] = arguments
            # ========== END UNIFIED ARGUMENTS FIELD ==========

            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"SAR classification failed: {str(e)}"
            print(f"[SARClassification] ❌ {error_msg}")
            
            error_result = {
                "success": False,
                "error": error_msg,
                "image_path": image_path,
                "summary": f"SAR classification failed: {error_msg}"
            }
            return json.dumps(error_result, indent=2)


def create_sar_classification_tool(device: str = "cuda:2") -> SARClassificationTool:
    """
    Factory function to create a SAR classification tool.
    
    Args:
        device: CUDA device for inference
        
    Returns:
        Configured SARClassificationTool instance
    """
    return SARClassificationTool(device=device)
