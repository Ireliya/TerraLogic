"""
RemoteSAM model implementation for spatial reasoning agent.
"""

import sys
import os
import torch
import torchvision
import numpy as np
import transformers
from PIL import Image
from pathlib import Path

# Import local modules
from . import utils
from . import segmentation
from . import transforms as T
from .RuleBasedCaptioning import single_captioning
from .args import get_default_args

# Hugging Face integration
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    print("⚠️  huggingface_hub not available. Install with: pip install huggingface_hub")
    HF_HUB_AVAILABLE = False


def download_remotesam_model(model_name: str = "RemoteSAMv1.pth",
                            cache_dir: str = None,
                            force_download: bool = False) -> str:
    """
    Download RemoteSAM model from Hugging Face Hub.

    Args:
        model_name: Name of the model file to download
        cache_dir: Directory to cache the model (default: ~/.cache/huggingface)
        force_download: Whether to force re-download even if cached

    Returns:
        Path to the downloaded model file

    Raises:
        ImportError: If huggingface_hub is not available
        Exception: If download fails
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for downloading models. "
            "Install with: pip install huggingface_hub"
        )

    try:
        print(f"🔄 Downloading {model_name} from Hugging Face Hub...")

        # Download the model file
        model_path = hf_hub_download(
            repo_id="1e12Leon/RemoteSAM",
            filename=model_name,
            cache_dir=cache_dir,
            force_download=force_download
        )

        print(f"✅ Model downloaded successfully to: {model_path}")
        return model_path

    except Exception as e:
        print(f"❌ Failed to download model from Hugging Face: {e}")
        raise e


def _validate_model_file(file_path: str) -> bool:
    """
    Validate that a model file exists and has proper integrity.

    Args:
        file_path: Path to the model file

    Returns:
        True if file is valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False

        # Check file size (RemoteSAM model should be > 100MB)
        file_size = os.path.getsize(file_path)
        if file_size < 100 * 1024 * 1024:  # 100MB minimum
            print(f"⚠️  Model file too small ({file_size} bytes): {file_path}")
            return False

        # Check file permissions
        if not os.access(file_path, os.R_OK):
            print(f"⚠️  Model file not readable: {file_path}")
            return False

        return True

    except Exception as e:
        print(f"⚠️  Error validating model file {file_path}: {e}")
        return False


def _get_hf_cache_path(cache_dir: str = None) -> str:
    """
    Get the expected Hugging Face cache path for RemoteSAM model.

    Args:
        cache_dir: Custom cache directory (if None, uses default HF cache)

    Returns:
        Expected path to cached model file
    """
    if cache_dir is None:
        # Use default Hugging Face cache directory
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    # Expected path structure for RemoteSAM model in HF cache
    expected_path = os.path.join(
        cache_dir,
        "models--1e12Leon--RemoteSAM",
        "snapshots"
    )

    # Find the latest snapshot directory
    if os.path.exists(expected_path):
        try:
            snapshots = [d for d in os.listdir(expected_path) if os.path.isdir(os.path.join(expected_path, d))]
            if snapshots:
                # Use the most recent snapshot
                latest_snapshot = max(snapshots, key=lambda x: os.path.getctime(os.path.join(expected_path, x)))
                model_path = os.path.join(expected_path, latest_snapshot, "RemoteSAMv1.pth")
                return model_path
        except Exception as e:
            print(f"⚠️  Error finding HF cache snapshot: {e}")

    return None


def get_model_path(model_path: str = None,
                  fallback_to_hf: bool = True,
                  cache_dir: str = None) -> str:
    """
    Get the path to RemoteSAM model, with fallback to Hugging Face download.

    Args:
        model_path: Local path to model file (if None, uses default)
        fallback_to_hf: Whether to fallback to Hugging Face download if local file not found
        cache_dir: Directory to cache downloaded models

    Returns:
        Path to the model file

    Raises:
        FileNotFoundError: If model not found and fallback disabled
        Exception: If download fails
    """
    # Use default path if not provided
    if model_path is None:
        model_path = "pretrained_weights/RemoteSAMv1.pth"

    # Check if local file exists and is valid
    if _validate_model_file(model_path):
        print(f"✅ Using local model: {model_path}")
        return model_path

    # Check if model exists in Hugging Face cache before attempting download
    if fallback_to_hf:
        hf_cache_path = _get_hf_cache_path(cache_dir)
        if hf_cache_path and _validate_model_file(hf_cache_path):
            print(f"✅ Using cached model from Hugging Face: {hf_cache_path}")
            return hf_cache_path

    # Try fallback to Hugging Face download only if not found locally or in cache
    if fallback_to_hf:
        print(f"⚠️  Local model not found at {model_path}")
        print("🔄 Attempting to download from Hugging Face...")
        try:
            downloaded_path = download_remotesam_model(cache_dir=cache_dir)
            # Validate the downloaded file
            if _validate_model_file(downloaded_path):
                return downloaded_path
            else:
                raise Exception("Downloaded model file failed validation")
        except Exception as e:
            print(f"❌ Hugging Face download failed: {e}")
            raise FileNotFoundError(
                f"Model not found locally at {model_path} and Hugging Face download failed: {e}"
            )
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")


def get_transform():
    """Get image transformation pipeline."""
    transforms = [
        T.Resize(896, 896),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return T.Compose(transforms)


def embed_sentences(sentences, max_tokens=20):
    """
    Embed sentences using BERT tokenizer.

    Args:
        sentences: List of sentences to embed
        max_tokens: Maximum number of tokens

    Returns:
        Tuple of (inputs, attentions) tensors
    """
    # Initialize BERT tokenizer using local files only
    import os
    from pathlib import Path

    # Try to use local BERT model with multiple path strategies
    # Strategy 1: Relative to current working directory
    local_bert_path = Path('bert-base-uncased')

    # Strategy 2: Relative to current working directory (explicit)
    if not local_bert_path.exists():
        local_bert_path = Path.cwd() / 'bert-base-uncased'

    # Strategy 3: Absolute path to the SpatialreasonAgent directory
    if not local_bert_path.exists():
        # Find the SpatialreasonAgent root directory
        current_file = Path(__file__).resolve()
        # Go up from spatialreason/tools/Perception/remotesam/model.py to find the root
        spatialreason_root = current_file.parent.parent.parent.parent
        local_bert_path = spatialreason_root / 'bert-base-uncased'

    # Strategy 4: Check common locations
    if not local_bert_path.exists():
        common_paths = [
            Path('/home/yuhang/Downloads/SpatialreasonAgent/bert-base-uncased'),
            Path.home() / 'Downloads/SpatialreasonAgent/bert-base-uncased',
        ]
        for path in common_paths:
            if path.exists():
                local_bert_path = path
                break

    if local_bert_path.exists():
        print(f"🔧 Using local BERT tokenizer from: {local_bert_path}")
        bert_model = transformers.BertTokenizer.from_pretrained(
            str(local_bert_path),
            local_files_only=True,
            do_lower_case=True
        )
    else:
        print("⚠️ Local BERT model not found, attempting online download...")
        try:
            bert_model = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        except Exception as e:
            print(f"❌ Failed to load BERT tokenizer: {e}")
            print("💡 Please ensure bert-base-uncased directory exists with tokenizer files")
            print(f"💡 Searched paths:")
            print(f"   - {Path('bert-base-uncased').absolute()}")
            print(f"   - {Path.cwd() / 'bert-base-uncased'}")
            if 'spatialreason_root' in locals():
                print(f"   - {spatialreason_root / 'bert-base-uncased'}")
            raise RuntimeError(f"BERT tokenizer loading failed: {e}")

    inputs = []
    attentions = []

    for sentence in sentences:
        sentence_tokenized = bert_model.encode(text=sentence, add_special_tokens=True)
        sentence_tokenized = sentence_tokenized[:max_tokens]  
        
        # Pad the tokenized sentence
        padded_sent_toks = [0] * max_tokens
        padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
        
        # Create a sentence token mask: 1 for real words; 0 for padded tokens
        attention_mask = [0] * max_tokens
        attention_mask[:len(sentence_tokenized)] = [1] * len(sentence_tokenized)
        
        inputs.append([padded_sent_toks])
        attentions.append([attention_mask])
    
    return torch.tensor(inputs), torch.tensor(attentions)


def _validate_device_string(device: str) -> str:
    """
    Validate device string with strict hardcoded allocation enforcement.

    Args:
        device: Device string to validate

    Returns:
        Validated device string or raises error
    """
    if device == "auto":
        # This should have been resolved earlier, but handle it
        if torch.cuda.is_available():
            device = "cuda:2"  # Hardcoded assignment for perception tools
        else:
            raise RuntimeError("CUDA not available but auto device selection requested. Cannot proceed with hardcoded GPU allocation strategy.")

    if device.startswith('cuda:'):
        try:
            gpu_id = int(device.split(':')[1])
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA not available but {device} was requested. Cannot proceed with hardcoded GPU allocation strategy.")
            elif gpu_id >= torch.cuda.device_count():
                available_gpus = torch.cuda.device_count()
                raise RuntimeError(f"GPU {gpu_id} not available (only {available_gpus} GPUs detected). Cannot proceed with hardcoded GPU allocation strategy.")
            else:
                # Test basic CUDA operations on the device
                try:
                    test_tensor = torch.tensor([1.0], device=device)
                    del test_tensor
                    torch.cuda.empty_cache()
                    return device
                except Exception as e:
                    raise RuntimeError(f"GPU {gpu_id} not accessible ({e}). Cannot proceed with hardcoded GPU allocation strategy.")
        except (ValueError, IndexError):
            raise RuntimeError(f"Invalid device format '{device}'. Cannot proceed with hardcoded GPU allocation strategy.")
    elif device == "cpu":
        return device
    else:
        raise RuntimeError(f"Unknown device '{device}'. Cannot proceed with hardcoded GPU allocation strategy.")


def init_demo_model(checkpoint=None, device='cuda:2', fallback_to_hf=True, cache_dir=None):
    """
    Initialize the RemoteSAM model from checkpoint with Hugging Face fallback.

    Args:
        checkpoint: Path to model checkpoint (if None, uses default with HF fallback)
        device: Device to load model on
        fallback_to_hf: Whether to fallback to Hugging Face download if local file not found
        cache_dir: Directory to cache downloaded models

    Returns:
        Loaded model

    Raises:
        FileNotFoundError: If model not found and fallback disabled
        Exception: If model loading fails
    """
    try:
        # Get model path with Hugging Face fallback
        if checkpoint is None:
            checkpoint = get_model_path(fallback_to_hf=fallback_to_hf, cache_dir=cache_dir)
        else:
            # If specific checkpoint provided, validate it first
            if not _validate_model_file(checkpoint) and fallback_to_hf:
                print(f"⚠️  Specified checkpoint not found or invalid: {checkpoint}")
                checkpoint = get_model_path(fallback_to_hf=True, cache_dir=cache_dir)
            elif not _validate_model_file(checkpoint):
                raise FileNotFoundError(f"Model checkpoint not found or invalid: {checkpoint}")

        print(f"🔄 Loading RemoteSAM model from: {checkpoint}")

        # Validate device before using it
        validated_device = _validate_device_string(device)
        if validated_device != device:
            print(f"🔄 Device changed from {device} to {validated_device}")
            device = validated_device

        # Get default arguments
        args = get_default_args()
        args.device = device
        args.window12 = True

        # Load model architecture
        model = segmentation.__dict__["lavt_one"](pretrained='', args=args)

        # Load checkpoint with version-compatible parameters
        try:
            # Try with weights_only parameter (PyTorch >= 1.13)
            model_ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            model_ckpt = torch.load(checkpoint, map_location='cpu')

        model.load_state_dict(model_ckpt['model'], strict=False)

        # Move model to device with strict error handling
        try:
            model = model.to(device)
            print(f"✅ RemoteSAM model loaded successfully on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to move RemoteSAM model to {device}: {e}. Cannot proceed with hardcoded GPU allocation strategy.")

        return model

    except Exception as e:
        print(f"❌ Failed to initialize RemoteSAM model: {e}")
        raise e


class RemoteSAM:
    """
    RemoteSAM model wrapper for segmentation tasks.
    """

    def __init__(self, RemoteSAM_model, device, *, use_EPOC=False, EPOC_threshold=0.25, 
                 MLC_balance_factor=0.5, MCC_balance_factor=1.0):
        """
        Initialize RemoteSAM wrapper.
        
        Args:
            RemoteSAM_model: The loaded RemoteSAM model
            device: Device to run on
            use_EPOC: Whether to use EPOC post-processing
            EPOC_threshold: Threshold for EPOC
            MLC_balance_factor: Balance factor for multi-label classification
            MCC_balance_factor: Balance factor for multi-class classification
        """
        self.RemoteSAM_model = RemoteSAM_model
        self.RemoteSAM_model.eval()
        self.device = device
        self.EPOC_threshold = EPOC_threshold
        self.MLC_balance_factor = MLC_balance_factor
        self.MCC_balance_factor = MCC_balance_factor
        
        # Pooling operations
        self.GMP = torch.nn.AdaptiveMaxPool2d(1)
        self.GAP = torch.nn.AdaptiveAvgPool2d(1)
        
        # Initialize EPOC if requested
        self.EPOC_model = None
        self.image_processor = None
        
        if use_EPOC:
            try:
                EPOC_checkpoint = "chendelong/DirectSAM-EntitySeg-1024px-0501"
                self.image_processor = transformers.AutoImageProcessor.from_pretrained(
                    EPOC_checkpoint, reduce_labels=True
                )
                EPOC_model = transformers.AutoModelForSemanticSegmentation.from_pretrained(
                    EPOC_checkpoint, num_labels=1, ignore_mismatched_sizes=True
                )
                self.EPOC_model = EPOC_model.to(device).eval()
            except Exception as e:
                print(f"Warning: Failed to load EPOC model: {e}")
                self.EPOC_model = None

    def check_input(self, image, classnames):
        """
        Check and normalize input format.
        
        Args:
            image: Input image (PIL Image or numpy array)
            classnames: Class names (string or list)
            
        Returns:
            Tuple of (image, classnames) in normalized format
        """
        assert isinstance(image, Image.Image) or isinstance(image, np.ndarray), \
            "image should be PIL Image or ndarray"
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if isinstance(classnames, str):
            classnames = [classnames]
        else:
            assert isinstance(classnames, list), "classnames should be str or list"

        return image, classnames

    def semantic_seg(self, *, image, classnames, return_prob=False):
        """
        Perform semantic segmentation.
        
        Args:
            image: Input image
            classnames: List of class names to segment
            return_prob: Whether to return probability maps
            
        Returns:
            Dictionary of segmentation masks (and probabilities if requested)
        """
        image, classnames = self.check_input(image, classnames)

        mask_result = {}
        prob_result = {}

        origin_image = image
        image, _ = get_transform()(image, image)
        image = image.unsqueeze(0).to(self.device)

        inputs, attentions = embed_sentences([f"{classname} in the image" for classname in classnames])
        inputs = inputs.to(self.device)
        attentions = attentions.to(self.device)

        for j in range(inputs.shape[0]):
            output = self.RemoteSAM_model(image, inputs[j], l_mask=attentions[j])

            mask = output.cpu().argmax(1, keepdim=True)  # (1, 1, resized_shape)
            mask = torch.nn.functional.interpolate(mask.float(), origin_image.size[::-1])  # (1, 1, origin_shape)
            mask = mask.squeeze().data.numpy().astype(np.uint8)  # np(origin_shape)
            
            prob = torch.softmax(output, dim=1)[:, [1], :, :].cpu()  # (1, 1, resized_shape)
            prob = torch.nn.functional.interpolate(prob.float(), origin_image.size[::-1])  # (1, 1, origin_shape)
            prob = prob.squeeze().data.numpy()  # np(origin_shape)

            mask_result[classnames[j]] = mask
            prob_result[classnames[j]] = prob
            
        if return_prob:
            return mask_result, prob_result
        else:
            return mask_result

    def referring_seg(self, *, image, sentence, return_prob=False):
        """
        Perform referring segmentation.
        
        Args:
            image: Input image
            sentence: Referring expression
            return_prob: Whether to return probability map
            
        Returns:
            Segmentation mask (and probability if requested)
        """
        image, sentence = self.check_input(image, sentence)
        assert len(sentence) == 1

        origin_image = image
        image, _ = get_transform()(image, image)
        image = image.unsqueeze(0).to(self.device)

        inputs, attentions = embed_sentences(sentence)
        inputs = inputs[0].to(self.device)
        attentions = attentions[0].to(self.device)

        output = self.RemoteSAM_model(image, inputs, l_mask=attentions)

        mask = output.cpu().argmax(1, keepdim=True)  # (1, 1, resized_shape)
        mask = torch.nn.functional.interpolate(mask.float(), origin_image.size[::-1])  # (1, 1, origin_shape)
        mask = mask.squeeze().data.numpy().astype(np.uint8)  # np(origin_shape)

        prob = torch.softmax(output, dim=1)[:, [1], :, :].cpu()  # (1, 1, resized_shape)
        prob = torch.nn.functional.interpolate(prob.float(), origin_image.size[::-1])  # (1, 1, origin_shape)
        prob = prob.squeeze().data.numpy()  # np(origin_shape)

        if return_prob:
            return mask, prob
        else:
            return mask

    def detection(self, *, image, classnames):
        """
        Perform object detection.
        
        Args:
            image: Input image
            classnames: List of class names to detect
            
        Returns:
            Dictionary of detected bounding boxes
        """
        image, classnames = self.check_input(image, classnames)

        masks, probs = self.semantic_seg(image=image, classnames=classnames, return_prob=True)

        result = {}

        origin_image = image
        image, _ = get_transform()(image, image)
        image = image.unsqueeze(0).to(self.device)

        if self.EPOC_model:
            contour = utils.EPOC(origin_image, self.image_processor, self.EPOC_model)
            binarized = (contour > self.EPOC_threshold).astype(np.uint8)

        for classname in classnames:
            result[classname] = None

            if self.EPOC_model:
                refined_mask = masks[classname] * (1 - binarized)
                boxes = utils.M2B(masks[classname], probs[classname], new_mask=refined_mask, box_type='hbb')
            else:
                boxes = utils.M2B(masks[classname], probs[classname], box_type='hbb')

            if len(boxes) > 0:
                indices = torchvision.ops.nms(
                    torch.tensor([[a[0], a[1], a[2], a[3]] for a in boxes]).float(), 
                    torch.tensor([a[4] for a in boxes]).float(), 
                    0.5
                )
                boxes = [boxes[i] for i in indices]
                result[classname] = boxes
        
        return result

    def counting(self, *, image, classnames):
        """
        Count objects of specified classes.
        
        Args:
            image: Input image
            classnames: List of class names to count
            
        Returns:
            Dictionary of object counts
        """
        boxes = self.detection(image=image, classnames=classnames)
        result = {}
        for classname in classnames:
            if boxes[classname]:
                result[classname] = len(boxes[classname])
            else:
                result[classname] = 0
        return result

    def captioning(self, *, image, classnames, region_split=4):
        """
        Generate caption based on detected objects.
        
        Args:
            image: Input image
            classnames: List of class names
            region_split: Number of regions to split image into
            
        Returns:
            Generated caption string
        """
        temp = self.detection(image=image, classnames=classnames)

        # Filter out empty boxes
        boxes = {}
        for classname in classnames:
            if temp[classname]:
                boxes[classname] = temp[classname]

        if isinstance(image, np.ndarray):
            shape = image.shape[:2]
        elif isinstance(image, Image.Image):
            shape = image.size[::-1]

        caption = single_captioning(boxes, shape, region_split)

        return caption
