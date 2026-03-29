"""
Semantic Change Detection Tool using Change3D model.
This tool detects semantic changes between bi-temporal satellite images.
"""
import os
import sys
import json
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add Change3D to path
CHANGE3D_PATH = Path(__file__).parent.parent.parent.parent / "Change3D"
if str(CHANGE3D_PATH) not in sys.path:
    sys.path.insert(0, str(CHANGE3D_PATH))

# Import Change3D components
try:
    from model.trainer import Trainer
    CHANGE3D_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Change3D not available: {e}")
    CHANGE3D_AVAILABLE = False


class ChangeDetectionInput(BaseModel):
    """Input schema for change detection tool."""
    image_path_t1: str = Field(description="Path to pre-change (T1) image file")
    image_path_t2: str = Field(description="Path to post-change (T2) image file")
    num_classes: int = Field(default=6, description="Number of semantic classes (default: 6 for HRSCD/SECOND)")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold for change detection (0.0-1.0)")
    device: str = Field(default="cuda:0", description="Device for inference (cuda:0, cuda:1, etc.)")
    
    # Optional parameters for unified arguments field
    text_prompt: Optional[str] = Field(default="detect semantic changes", description="Task intent description")
    meters_per_pixel: Optional[float] = Field(default=None, description="Ground resolution for spatial analysis")


class ChangeDetectionTool(BaseTool):
    """
    Semantic Change Detection tool using Change3D model.
    Detects semantic changes between bi-temporal satellite images.
    """

    name: str = "change_detection"
    description: str = "Detect semantic changes between bi-temporal satellite images using Change3D model"
    args_schema: type[BaseModel] = ChangeDetectionInput

    # Private attributes for the model
    _model: Optional[Any] = None
    _device: str = "cuda:0"
    _model_path: Optional[str] = None
    _temp_dir: Optional[Path] = None

    def __init__(self, device: str = "cuda:0", num_classes: int = 6, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"[ChangeDetection] Initializing Change3D change detection tool with device: {device}")
        self._device = device
        self._num_classes = num_classes
        self._validate_device()
        self._setup_paths()
        self._initialize_model()

    def _validate_device(self):
        """Validate CUDA device availability."""
        if "cuda" in self._device:
            if not torch.cuda.is_available():
                logger.warning(f"CUDA not available, falling back to CPU")
                self._device = "cpu"
            else:
                device_id = int(self._device.split(":")[-1]) if ":" in self._device else 0
                if device_id >= torch.cuda.device_count():
                    logger.warning(f"Device {self._device} not available, using cuda:0")
                    self._device = "cuda:0"

    def _setup_paths(self):
        """Setup model and output paths."""
        # Pretrained backbone weights path (X3D)
        self._pretrained_path = CHANGE3D_PATH / "X3D_L.pyth"

        # Trained model weights path (full model trained on HRSCD)
        self._model_path = CHANGE3D_PATH / "best_model.pth"

        # Create temp directory for outputs
        self._temp_dir = Path(__file__).parent / "temp" / "change_detection"
        self._temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[ChangeDetection] Pretrained backbone path: {self._pretrained_path}")
        logger.info(f"[ChangeDetection] Trained model path: {self._model_path}")
        logger.info(f"[ChangeDetection] Temp directory: {self._temp_dir}")

    def _initialize_model(self):
        """Initialize the Change3D model with new architecture."""
        if not CHANGE3D_AVAILABLE:
            logger.error("[ChangeDetection] Change3D not available - cannot initialize model")
            return

        # Check if pretrained backbone exists
        if not self._pretrained_path.exists():
            logger.error(f"[ChangeDetection] Pretrained backbone not found at {self._pretrained_path}")
            return

        try:
            # Initialize model with new simple API
            # NEW: Trainer(model_type='x3d', num_class=6)
            logger.info(f"[ChangeDetection] Initializing Trainer with model_type='x3d', num_class={self._num_classes}")
            self._model = Trainer(model_type='x3d', num_class=self._num_classes)

            # Load X3D backbone weights manually (NEW: not automatic anymore)
            logger.info(f"[ChangeDetection] Loading X3D backbone from {self._pretrained_path}")
            x3d_state = torch.load(self._pretrained_path, map_location='cpu')

            # X3D_L.pyth contains 'model_state' key
            if 'model_state' in x3d_state:
                self._model.encoder.x3d.load_state_dict(x3d_state['model_state'], strict=True)
                logger.info(f"[ChangeDetection] X3D backbone loaded successfully")
            else:
                logger.error(f"[ChangeDetection] X3D_L.pyth missing 'model_state' key")
                self._model = None
                return

            # Load trained model weights if available
            if self._model_path.exists():
                logger.info(f"[ChangeDetection] Loading trained model weights from {self._model_path}")
                state_dict = torch.load(self._model_path, map_location='cpu')

                # Handle DataParallel 'module.' prefix
                if list(state_dict.keys())[0].startswith('module.'):
                    logger.info(f"[ChangeDetection] Removing 'module.' prefix from state_dict keys")
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove 'module.' prefix
                        new_state_dict[name] = v
                    state_dict = new_state_dict

                self._model.load_state_dict(state_dict, strict=True)
                logger.info(f"[ChangeDetection] Successfully loaded trained model weights")
            else:
                logger.warning(f"[ChangeDetection] Trained model weights not found at {self._model_path}")
                logger.warning(f"[ChangeDetection] Using only pretrained X3D backbone - results may be suboptimal")

            # Move to device and set to eval mode
            self._model = self._model.to(self._device)
            self._model.eval()

            logger.info(f"[ChangeDetection] Model initialized successfully on {self._device}")

        except Exception as e:
            logger.error(f"[ChangeDetection] Failed to initialize model: {e}")
            import traceback
            traceback.print_exc()
            self._model = None

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for model input following Change3D's preprocessing pipeline.

        The preprocessing matches SCDTransforms.normalize() from Change3D/data/transforms.py:
        1. Scale to [0, 1] by dividing by 255
        2. Normalize using mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
           This transforms the range from [0, 1] to [-1, 1]

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 256x256 if needed
        if image.shape[:2] != (256, 256):
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Scale to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Normalize using Change3D's default normalization parameters
        # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        # This transforms [0, 1] to [-1, 1]
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std

        # Convert to tensor [H, W, C] -> [C, H, W]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)

        # Add batch dimension [C, H, W] -> [1, C, H, W]
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def _postprocess_masks(
        self,
        pre_mask: torch.Tensor,
        post_mask: torch.Tensor,
        change_mask: torch.Tensor,
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess model outputs to numpy arrays.

        Args:
            pre_mask: Pre-change semantic mask [B, C, H, W]
            post_mask: Post-change semantic mask [B, C, H, W]
            change_mask: Binary change mask [B, 1, H, W]
            confidence_threshold: Threshold for binary change mask

        Returns:
            Tuple of (pre_semantic, post_semantic, binary_change) as numpy arrays
        """
        # Convert to numpy and remove batch dimension
        pre_semantic = torch.argmax(pre_mask, dim=1).squeeze(0).cpu().numpy()
        post_semantic = torch.argmax(post_mask, dim=1).squeeze(0).cpu().numpy()
        binary_change = (change_mask.squeeze().cpu().numpy() > confidence_threshold).astype(np.uint8)

        return pre_semantic, post_semantic, binary_change

    def _extract_change_regions(
        self,
        pre_semantic: np.ndarray,
        post_semantic: np.ndarray,
        binary_change: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Extract change regions with semantic information.

        Args:
            pre_semantic: Pre-change semantic segmentation [H, W]
            post_semantic: Post-change semantic segmentation [H, W]
            binary_change: Binary change mask [H, W]

        Returns:
            List of change region dictionaries
        """
        from skimage.measure import label, regionprops

        # Label connected components in change mask
        labeled_changes = label(binary_change)
        regions = regionprops(labeled_changes)

        change_regions = []
        for region in regions:
            # Get bounding box
            min_row, min_col, max_row, max_col = region.bbox

            # Get polygon coordinates (bounding box format)
            polygon = [
                [min_col, min_row],
                [max_col, min_row],
                [max_col, max_row],
                [min_col, max_row],
                [min_col, min_row]
            ]

            # Get dominant classes in this region
            region_mask = labeled_changes == region.label
            pre_classes = pre_semantic[region_mask]
            post_classes = post_semantic[region_mask]

            # Find most common class in pre and post
            pre_class = int(np.bincount(pre_classes.flatten()).argmax())
            post_class = int(np.bincount(post_classes.flatten()).argmax())

            change_regions.append({
                "polygon": polygon,
                "area": int(region.area),
                "centroid": [int(region.centroid[1]), int(region.centroid[0])],  # [x, y]
                "pre_class": pre_class,
                "post_class": post_class,
                "change_type": f"class_{pre_class}_to_class_{post_class}"
            })

        return change_regions

    def _create_color_palette(self) -> List[List[int]]:
        """
        Create a color palette for semantic classes.

        Returns:
            List of RGB colors for each class
        """
        palette = [
            [128, 128, 128],  # Class 0: Gray (Background/No info)
            [255, 0, 0],      # Class 1: Red (Artificial surfaces/Buildings)
            [0, 255, 0],      # Class 2: Green (Agricultural areas)
            [0, 0, 255],      # Class 3: Blue (Forests)
            [255, 255, 0],    # Class 4: Yellow (Wetlands/Vegetation)
            [0, 255, 255],    # Class 5: Cyan (Water)
        ]
        return palette

    def _class_to_rgb(self, class_mask: np.ndarray, palette: List[List[int]]) -> np.ndarray:
        """
        Convert class indices to RGB image.

        Args:
            class_mask: 2D array of class indices
            palette: List of RGB colors for each class

        Returns:
            RGB image
        """
        h, w = class_mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx, color in enumerate(palette):
            rgb[class_mask == cls_idx] = color
        return rgb

    def _create_visualization(
        self,
        image_path_t1: str,
        image_path_t2: str,
        pre_semantic: np.ndarray,
        post_semantic: np.ndarray,
        binary_change: np.ndarray,
        output_path: str
    ) -> str:
        """
        Create visualization of change detection results using direct image composition.
        This avoids matplotlib's interpolation artifacts.

        Args:
            image_path_t1: Path to T1 image
            image_path_t2: Path to T2 image
            pre_semantic: Pre-change semantic segmentation [H, W] with class indices
            post_semantic: Post-change semantic segmentation [H, W] with class indices
            binary_change: Binary change mask [H, W] with 0/1 values
            output_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        # Load original images
        img_t1 = cv2.imread(image_path_t1)
        img_t2 = cv2.imread(image_path_t2)

        if img_t1 is not None:
            img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
        if img_t2 is not None:
            img_t2 = cv2.cvtColor(img_t2, cv2.COLOR_BGR2RGB)

        # Get color palette
        palette = self._create_color_palette()

        # Convert semantic masks to RGB
        pre_semantic_rgb = self._class_to_rgb(pre_semantic, palette)
        post_semantic_rgb = self._class_to_rgb(post_semantic, palette)

        # Create change visualization (white for changes, black for no change)
        change_rgb = np.zeros_like(img_t1)
        change_rgb[binary_change > 0] = [255, 255, 255]

        # Create semantic change visualization (only show semantics in changed regions)
        semantic_change = np.zeros_like(pre_semantic)
        semantic_change[binary_change > 0] = post_semantic[binary_change > 0]
        semantic_change_rgb = self._class_to_rgb(semantic_change, palette)

        # Create a 2x3 grid visualization
        h, w = img_t1.shape[:2]

        # Add padding between images
        pad = 10
        pad_color = [255, 255, 255]  # White padding

        # Create rows
        row1 = np.hstack([
            img_t1,
            np.full((h, pad, 3), pad_color, dtype=np.uint8),
            img_t2,
            np.full((h, pad, 3), pad_color, dtype=np.uint8),
            change_rgb
        ])

        row2 = np.hstack([
            pre_semantic_rgb,
            np.full((h, pad, 3), pad_color, dtype=np.uint8),
            post_semantic_rgb,
            np.full((h, pad, 3), pad_color, dtype=np.uint8),
            semantic_change_rgb
        ])

        # Stack rows
        grid = np.vstack([
            row1,
            np.full((pad, row1.shape[1], 3), pad_color, dtype=np.uint8),
            row2
        ])

        # Add text labels using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        font_color = (0, 0, 0)  # Black text

        # Calculate text positions
        col_width = w + pad
        text_y = 20

        labels = [
            ('T1 (Pre-change)', pad),
            ('T2 (Post-change)', col_width + pad),
            ('Binary Change', 2 * col_width + pad),
            ('T1 Semantic', pad),
            ('T2 Semantic', col_width + pad),
            ('Semantic Changes', 2 * col_width + pad)
        ]

        for i, (label, x_offset) in enumerate(labels):
            y_pos = text_y if i < 3 else h + pad + text_y
            cv2.putText(grid, label, (x_offset, y_pos), font, font_scale, font_color, font_thickness)

        # Save using OpenCV (no interpolation)
        grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, grid_bgr)

        return output_path

    def _run(
        self,
        image_path_t1: str,
        image_path_t2: str,
        num_classes: Optional[int] = None,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        text_prompt: Optional[str] = None,
        meters_per_pixel: Optional[float] = None,
        classes_requested: Optional[list] = None
    ) -> str:
        """
        Execute semantic change detection on bi-temporal images.

        Args:
            image_path_t1: Path to pre-change (T1) image
            image_path_t2: Path to post-change (T2) image
            num_classes: Number of semantic classes (optional, uses initialized value)
            confidence_threshold: Threshold for binary change detection
            device: CUDA device (optional, uses initialized device)
            text_prompt: Task intent description (optional)
            meters_per_pixel: Ground resolution for spatial analysis (optional)
            classes_requested: List of class names being analyzed (optional, for benchmark compatibility)

        Returns:
            JSON string with detection results
        """
        try:
            # Validate inputs
            if not os.path.exists(image_path_t1):
                raise FileNotFoundError(f"T1 image not found: {image_path_t1}")
            if not os.path.exists(image_path_t2):
                raise FileNotFoundError(f"T2 image not found: {image_path_t2}")

            if self._model is None:
                raise RuntimeError("Model not initialized. Check if Change3D is available and weights are loaded.")

            # Extract base filename from input image for visualization naming
            # e.g., "dataset/hrscd_images/t1/cd_0.png" -> "cd_0"
            base_filename = Path(image_path_t1).stem

            # Generate execution ID (kept for backward compatibility in output JSON)
            execution_id = str(uuid.uuid4())[:8]

            logger.info(f"[ChangeDetection] Processing image pair: {image_path_t1}, {image_path_t2}")

            # Preprocess images
            img_t1 = self._preprocess_image(image_path_t1).to(self._device)
            img_t2 = self._preprocess_image(image_path_t2).to(self._device)

            # Run inference
            # NEW: Use forward() instead of update_scd()
            # Returns: scd_pred1 (pre), scd_pred2 (post), bcd_pred (change)
            with torch.no_grad():
                pre_mask, post_mask, change_mask = self._model(img_t1, img_t2)

            # Postprocess outputs
            pre_semantic, post_semantic, binary_change = self._postprocess_masks(
                pre_mask, post_mask, change_mask, confidence_threshold
            )

            # Extract change regions
            change_regions = self._extract_change_regions(
                pre_semantic, post_semantic, binary_change
            )

            # Create visualization with filename based on input image name
            vis_filename = f"change_detection_{base_filename}.png"
            vis_path = str(self._temp_dir / vis_filename)
            self._create_visualization(
                image_path_t1, image_path_t2,
                pre_semantic, post_semantic, binary_change,
                vis_path
            )

            # Save segmentation masks for downstream use
            t1_mask_filename = f"t1_segmentation_{base_filename}.npy"
            t2_mask_filename = f"t2_segmentation_{base_filename}.npy"
            change_mask_filename = f"change_mask_{base_filename}.npy"

            t1_mask_path = str(self._temp_dir / t1_mask_filename)
            t2_mask_path = str(self._temp_dir / t2_mask_filename)
            change_mask_path = str(self._temp_dir / change_mask_filename)

            np.save(t1_mask_path, pre_semantic)
            np.save(t2_mask_path, post_semantic)
            np.save(change_mask_path, binary_change)

            logger.info(f"[ChangeDetection] Saved T1 segmentation mask to: {t1_mask_path}")
            logger.info(f"[ChangeDetection] Saved T2 segmentation mask to: {t2_mask_path}")
            logger.info(f"[ChangeDetection] Saved change mask to: {change_mask_path}")

            # Calculate statistics
            total_pixels = binary_change.size
            changed_pixels = int(binary_change.sum())
            change_percentage = (changed_pixels / total_pixels) * 100

            # Prepare results
            results = {
                "tool": "change_detection",
                "input": {
                    "image_path_t1": image_path_t1,
                    "image_path_t2": image_path_t2,
                    "num_classes": num_classes or self._num_classes,
                    "confidence_threshold": confidence_threshold,
                    "device": device or self._device
                },
                "output": {
                    "change_regions": change_regions,
                    "num_changes": len(change_regions),
                    "total_changed_pixels": changed_pixels,
                    "change_percentage": round(change_percentage, 2),
                    "image_shape": list(binary_change.shape),
                    "visualization_path": vis_path if os.path.exists(vis_path) else "",
                    "t1_segmentation_mask_path": t1_mask_path,
                    "t2_segmentation_mask_path": t2_mask_path,
                    "change_mask_path": change_mask_path
                },
                "arguments": {
                    "image_path_t1": image_path_t1,
                    "image_path_t2": image_path_t2,
                    "num_classes": num_classes or self._num_classes,
                    "confidence_threshold": confidence_threshold,
                    "text_prompt": text_prompt or "detect semantic changes",
                    "meters_per_pixel": meters_per_pixel,
                    "total_changes": len(change_regions),
                    "change_percentage": round(change_percentage, 2)
                },
                "execution_id": execution_id,
                "success": True
            }

            # Generate summary
            summary = self._generate_summary(results)
            results["summary"] = summary

            logger.info(f"[ChangeDetection] Detection complete: {len(change_regions)} change regions detected")

            return json.dumps(results, indent=2)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[ChangeDetection] Detection failed: {error_msg}")
            import traceback
            traceback.print_exc()

            error_result = {
                "tool": "change_detection",
                "error": error_msg,
                "change_regions": [],
                "num_changes": 0,
                "image_path_t1": image_path_t1,
                "image_path_t2": image_path_t2,
                "summary": f"Change detection failed: {error_msg}",
                "success": False
            }
            return json.dumps(error_result, indent=2)

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate natural language summary of change detection results.

        Args:
            results: Detection results dictionary

        Returns:
            Natural language summary string
        """
        output = results.get("output", {})
        num_changes = output.get("num_changes", 0)
        change_percentage = output.get("change_percentage", 0)

        if num_changes == 0:
            return "No semantic changes detected between the two images."

        # Build summary
        summary_parts = []
        summary_parts.append(f"Detected {num_changes} change region{'s' if num_changes != 1 else ''}")
        summary_parts.append(f"covering {change_percentage}% of the image area.")

        # Add change type information if available
        change_regions = output.get("change_regions", [])
        if change_regions:
            change_types = {}
            for region in change_regions:
                change_type = region.get("change_type", "unknown")
                change_types[change_type] = change_types.get(change_type, 0) + 1

            if len(change_types) > 0:
                type_summary = ", ".join([f"{count} {ctype}" for ctype, count in change_types.items()])
                summary_parts.append(f"Change types: {type_summary}.")

        return " ".join(summary_parts)


def create_change_detection_tool(device: str = "auto", num_classes: int = 6) -> ChangeDetectionTool:
    """
    Factory function to create a Change Detection tool.

    Args:
        device: Device to run the model on ("auto" for automatic GPU selection, 'cuda:0', 'cpu')
        num_classes: Number of semantic classes (default: 6 for HRSCD/SECOND)

    Returns:
        Configured ChangeDetectionTool instance
    """
    if device == "auto":
        # Check for hardcoded GPU assignment
        import os
        if os.getenv('SPATIAL_REASONING_GPU_MODE') == 'hardcoded':
            perception_gpu = os.getenv('SPATIAL_REASONING_PERCEPTION_GPU', '2')
            device = f"cuda:{perception_gpu}"
            logger.info(f"🎯 Change detection tool using hardcoded GPU assignment: {device}")
        else:
            # Use hardcoded GPU 0 for all perception tools (single GPU system)
            device = "cuda:0"
            logger.info(f"🎯 Change detection tool using hardcoded GPU assignment: {device}")

    return ChangeDetectionTool(device=device, num_classes=num_classes)


# Main execution for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Change Detection Tool")
    parser.add_argument("--image_t1", type=str, help="Path to T1 (pre-change) image")
    parser.add_argument("--image_t2", type=str, help="Path to T2 (post-change) image")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of semantic classes")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")

    args = parser.parse_args()

    # Create tool
    tool = create_change_detection_tool(device=args.device, num_classes=args.num_classes)

    # Run detection if images provided
    if args.image_t1 and args.image_t2:
        result = tool._run(
            image_path_t1=args.image_t1,
            image_path_t2=args.image_t2,
            confidence_threshold=args.threshold
        )
        print(result)
    else:
        print("Change Detection Tool initialized successfully!")
        print(f"Device: {tool._device}")
        print(f"Model available: {tool._model is not None}")
        print("\nUsage: python change_detection.py --image_t1 <path_t1> --image_t2 <path_t2>")

