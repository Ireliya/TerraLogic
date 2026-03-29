"""
Agent factory for creating and initializing the spatial reasoning agent.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import torch
from contextlib import nullcontext
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.tools import BaseTool
from spatialreason.agent import Agent
from spatialreason.tools.Perception.segmentation import RemoteSAMSegmentationTool
from spatialreason.tools.Perception.detection import RemoteSAMDetectionTool
from spatialreason.tools.Perception.classification import RemoteSAMClassificationTool
from spatialreason.tools.Perception.change_detection import ChangeDetectionTool
from spatialreason.tools.SAR.sar_detection import SARDetectionTool
from spatialreason.tools.SAR.sar_classification import SARClassificationTool
from spatialreason.tools.IR.InfraredDetection import InfraredDetectionTool
from spatialreason.tools.SpatialRelations.buffer_tool import BufferTool
from spatialreason.tools.SpatialStatistics.distance_tool import DistanceCalculationTool
from spatialreason.tools.SpatialRelations.overlap_tool import OverlapRatioTool
from spatialreason.tools.SpatialRelations.containment_tool import ContainmentTool
from spatialreason.tools.SpatialStatistics.area_measurement import AreaMeasurementTool
from spatialreason.tools.SpatialStatistics.object_count_aoi import ObjectCountInAOITool
# NOTE: SpatialQueryAnswerTool was removed - proper ReAct termination comes from
# tools returning actionable results (like MedRAX), not from a special synthesis tool.

from spatialreason.utils.simple_gpu_utils import load_prompts_from_file
# Removed MultiStepPlanner import - planner module removed
# Removed FloodRiskAssessmentTool and GroundResolutionDetector imports
from spatialreason.models.enhanced_chat_model import EnhancedChatModel

# Custom GPU assignments as requested
HARDCODED_GPU_ASSIGNMENTS = {
    'opencompass': 'cuda:0',      # OpenCompass uses GPU 0
    'chat_model': 'cuda:1',       # Qwen2-VL language model uses GPU 1
    'perception_tools': 'cuda:2', # All RemoteSAM tools share GPU 2
    'reserve': 'cuda:3'           # GPU 3 reserved
}


def get_failover_device(failed_device: str) -> str:
    """Get a failover device when the current device fails during runtime."""
    global _global_gpu_manager
    if _global_gpu_manager is None:
        return "cpu"

    failover = _global_gpu_manager.get_failover_device(failed_device)
    return failover if failover else "cpu"


def _load_model_with_direct_imports(model_path: str, device: str):
    """
    Load Qwen2-VL model using direct imports from local model files.
    This bypasses the CONFIG_MAPPING registry entirely.
    """
    import sys
    import os
    from pathlib import Path

    # Add the local model path to sys.path to import model classes
    model_path = Path(model_path).resolve()
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))

    print(f"[DEBUG] Added {model_path} to sys.path for direct imports")

    try:
        # Import model classes directly from the local model files
        print("[DEBUG] Importing Qwen2VL classes directly from local files...")

        # Try to import the main model classes
        try:
            from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
            from configuration_qwen2_vl import Qwen2VLConfig
            print("[DEBUG] Successfully imported Qwen2VLForConditionalGeneration and Qwen2VLConfig")
        except ImportError as e:
            print(f"[DEBUG] Failed to import from modeling_qwen2_vl: {e}")
            # Try alternative import names
            try:
                from qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLConfig
                print("[DEBUG] Successfully imported from qwen2_vl module")
            except ImportError as e2:
                print(f"[DEBUG] Failed to import from qwen2_vl: {e2}")
                raise ImportError(f"Could not import Qwen2VL classes from {model_path}")

        # Import tokenizer and processor classes
        try:
            from transformers import AutoTokenizer
            # Try to import processor from local files first
            try:
                from processing_qwen2_vl import Qwen2VLProcessor
                print("[DEBUG] Successfully imported Qwen2VLProcessor from local files")
            except ImportError:
                # Fallback to transformers AutoProcessor
                from transformers import AutoProcessor as Qwen2VLProcessor
                print("[DEBUG] Using AutoProcessor as fallback")
        except ImportError as e:
            print(f"[DEBUG] Failed to import tokenizer/processor: {e}")
            raise

        # Load configuration directly
        print(f"[DEBUG] Loading configuration from {model_path}/config.json...")
        config = Qwen2VLConfig.from_pretrained(model_path)
        print(f"[DEBUG] Configuration loaded: {config.model_type}")

        # Load tokenizer
        print(f"[DEBUG] Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Load processor
        print(f"[DEBUG] Loading processor from {model_path}...")
        try:
            processor = Qwen2VLProcessor.from_pretrained(model_path)
        except Exception as e:
            print(f"[DEBUG] Processor loading failed: {e}, creating fallback")
            processor = None

        # Load model with direct class
        print(f"[DEBUG] Loading model using direct Qwen2VLForConditionalGeneration...")
        device_id = int(device.split(':')[1]) if device.startswith('cuda:') else 0

        with torch.cuda.device(device_id):
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                config=config,
                torch_dtype="auto",
                device_map=device,  # Use explicit device assignment instead of "auto"
                low_cpu_mem_usage=True
            )

        print(f"[DEBUG] ✓ Successfully loaded model using direct imports")
        return model, tokenizer, processor

    except Exception as e:
        print(f"[ERROR] Direct import loading failed: {e}")
        raise


def _load_model_with_direct_imports_only(model_path: str, device: str):
    """
    Load Qwen2-VL model using ONLY direct imports from local model files.
    This function has NO fallbacks and will fail if direct imports don't work.
    Used when force_local_only=True to ensure no network access.
    """
    import sys
    import os
    from pathlib import Path

    model_path = Path(model_path).resolve()

    # Validate that the local model directory exists and contains required files
    if not model_path.exists():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")

    required_files = ["config.json", "modeling_qwen2_vl.py", "configuration_qwen2_vl.py"]
    missing_files = []
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {model_path}: {missing_files}\n"
            f"Required files: {required_files}"
        )

    # Add the local model path to sys.path for direct imports
    if str(model_path) not in sys.path:
        sys.path.insert(0, str(model_path))

    print(f"[LOCAL-ONLY] Added {model_path} to sys.path for direct imports")
    print(f"[LOCAL-ONLY] Importing Qwen2VL classes directly from local files...")

    try:
        # Import model classes directly from the local model files - NO FALLBACKS
        from modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        from configuration_qwen2_vl import Qwen2VLConfig
        print("[LOCAL-ONLY] ✓ Successfully imported Qwen2VLForConditionalGeneration and Qwen2VLConfig")

        # Import tokenizer (this should work with standard transformers)
        from transformers import AutoTokenizer

        # Try to import processor from local files
        processor = None
        try:
            from processing_qwen2_vl import Qwen2VLProcessor
            print("[LOCAL-ONLY] ✓ Successfully imported Qwen2VLProcessor from local files")
        except ImportError:
            print("[LOCAL-ONLY] ⚠️ Qwen2VLProcessor not found in local files, will be None")

        # Load configuration directly from local files
        print(f"[LOCAL-ONLY] Loading configuration from {model_path}/config.json...")
        config = Qwen2VLConfig.from_pretrained(str(model_path))
        print(f"[LOCAL-ONLY] ✓ Configuration loaded: {config.model_type}")

        # Load tokenizer from local files
        print(f"[LOCAL-ONLY] Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        print("[LOCAL-ONLY] ✓ Tokenizer loaded successfully")

        # Load processor from local files if available
        if processor:
            try:
                processor_instance = processor.from_pretrained(str(model_path))
                print("[LOCAL-ONLY] ✓ Processor loaded successfully")
            except Exception as e:
                print(f"[LOCAL-ONLY] ⚠️ Processor loading failed: {e}, setting to None")
                processor_instance = None
        else:
            processor_instance = None

        # Load model with direct class - NO FALLBACKS
        print(f"[LOCAL-ONLY] Loading model using direct Qwen2VLForConditionalGeneration...")
        device_id = int(device.split(':')[1]) if device.startswith('cuda:') else 0

        with torch.cuda.device(device_id):
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                str(model_path),
                config=config,
                torch_dtype="auto",
                device_map=device,  # Use explicit device assignment instead of "auto"
                low_cpu_mem_usage=True
            )

        print(f"[LOCAL-ONLY] ✅ Successfully loaded model using DIRECT IMPORTS ONLY")
        print(f"[LOCAL-ONLY] Model device: {device}")
        print(f"[LOCAL-ONLY] Model dtype: {model.dtype}")

        return model, tokenizer, processor_instance

    except ImportError as e:
        raise ImportError(
            f"[LOCAL-ONLY] FAILED to import Qwen2VL classes from {model_path}: {e}\n"
            f"force_local_only=True requires all model files to be present locally.\n"
            f"Please ensure your local model directory contains:\n"
            f"- modeling_qwen2_vl.py\n"
            f"- configuration_qwen2_vl.py\n"
            f"- config.json\n"
            f"- Model weight files (.bin or .safetensors)\n"
            f"- Tokenizer files"
        )
    except Exception as e:
        raise RuntimeError(
            f"[LOCAL-ONLY] FAILED to load model from {model_path}: {e}\n"
            f"This error occurred in force_local_only mode with no fallbacks available."
        )


def _load_model_with_transformers(model_path: str, device: str):
    """
    Load Qwen2-VL model using standard transformers approach with trust_remote_code=True.
    Supports both local paths and Hugging Face Hub models.
    """
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import os
    import torch

    # Validate device parameter
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda:N'")

    if device.startswith("cuda:"):
        device_id = int(device.split(':')[1])
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA not available but device {device} was requested. Cannot proceed with hardcoded GPU allocation strategy.")
        elif device_id >= torch.cuda.device_count():
            raise RuntimeError(f"GPU {device_id} not available (only {torch.cuda.device_count()} GPUs detected). Cannot proceed with hardcoded GPU allocation strategy.")

    # Check if this is a local path
    is_local_path = os.path.exists(model_path)
    if is_local_path:
        print(f"🔧 Loading from local path: {model_path}")
    else:
        print(f"🔧 Loading from Hugging Face Hub: {model_path}")

    print(f"🎯 Target device: {device}")

    # Load processor, model, and tokenizer for Qwen2-VL
    # Try different approaches for compatibility with older transformers
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=is_local_path  # Only use local files if it's a local path
        )
    except Exception as e:
        print(f"[DEBUG] AutoProcessor failed: {e}")
        # Create a simple processor fallback
        try:
            from transformers import AutoImageProcessor, AutoTokenizer
            image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )
            tokenizer_temp = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )

            # Create a simple processor-like object
            class SimpleProcessor:
                def __init__(self, image_processor, tokenizer):
                    self.image_processor = image_processor
                    self.tokenizer = tokenizer

                def __call__(self, *args, **kwargs):
                    return self.tokenizer(*args, **kwargs)

            processor = SimpleProcessor(image_processor, tokenizer_temp)
            print("[DEBUG] Created fallback processor")
        except Exception as e2:
            print(f"[DEBUG] Fallback processor also failed: {e2}")

    # First, explicitly load the config with trust_remote_code=True to ensure proper propagation
    print(f"🔧 Loading config for {model_path} with trust_remote_code=True...")
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=is_local_path
    )
    print(f"[DEBUG] Config loaded successfully: {config.model_type}")

    # Determine device mapping based on device
    if device == 'cpu':
        device_map = 'cpu'
        torch_dtype = torch.float32  # Use float32 for CPU
        device_context = nullcontext()
        print(f"[DEBUG] Using CPU device mapping")
    else:
        # Use explicit device assignment instead of "auto" to prevent GPU conflicts
        device_map = device  # Direct device assignment (e.g., "cuda:1")
        torch_dtype = "auto"
        device_id = int(device.split(':')[1]) if device.startswith('cuda:') else 0
        device_context = torch.cuda.device(device_id)
        print(f"[DEBUG] Using explicit CUDA device mapping for {device}")

    # Clear GPU cache before loading model
    if torch.cuda.is_available() and device.startswith("cuda:"):
        torch.cuda.empty_cache()
        print(f"🧹 Cleared CUDA cache before loading model")

    # Use appropriate device context
    with device_context:
        # Add memory optimization for GPU loading
        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "local_files_only": is_local_path
        }

        # Add memory optimization for GPU
        if device != "cpu":
            model_kwargs["low_cpu_mem_usage"] = True
            # Set max memory per device to prevent OOM
            if device.startswith("cuda:"):
                device_id = int(device.split(':')[1])
                model_kwargs["max_memory"] = {device_id: "20GiB"}

        try:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
            print("[DEBUG] Loaded with Qwen2VLForConditionalGeneration")
        except (ImportError, Exception) as e:
            print(f"[DEBUG] Qwen2VLForConditionalGeneration failed: {e}")
            # Fallback for older transformers versions
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            print("[DEBUG] Loaded with AutoModelForCausalLM")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=is_local_path
        )

        # Final memory cleanup after loading all components
        if torch.cuda.is_available() and device.startswith("cuda:"):
            torch.cuda.empty_cache()
            device_id = int(device.split(':')[1])
            memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3   # GB
            print(f"🧹 Final GPU {device_id} memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

    return model, tokenizer, processor


def initialize_agent(
    prompt_file: str,
    tools_to_use: Optional[List[str]] = None,
    model_name: str = "gpt-4o",
    device: str = "auto",
    local_model_path: Optional[str] = None,
    force_local_only: bool = False,
    use_model_abstraction: bool = True,
    model_type: str = "remote",
    disable_planner: bool = False,
) -> Tuple[Agent, Dict[str, BaseTool]]:
    """
    Initialize the spatial reasoning agent with specified tools and model.

    Args:
        prompt_file: Path to the system prompts file
        tools_to_use: List of tool names to use (default: all available tools)
        model_name: Name of the model to load (default: gpt-4o)
        device: Device to run the model on ("auto" for automatic GPU selection, only used for local models)
        local_model_path: Path to local model directory (only used when model_type="local")
        force_local_only: Force local-only loading (deprecated, use model_type="local")
        use_model_abstraction: Use the new model abstraction layer (default: True)
        model_type: Type of model to use ("local" or "remote", default: "remote")
        disable_planner: Disable planner for ablation study (ReAct-style execution, default: False)

    Returns:
        Tuple of (Agent, tools_dict, planner)
    """
    print(f"🔧 initialize_agent called with tools_to_use: {tools_to_use}")

    # 1. Setup GPU environment with hardcoded assignments
    from spatialreason.utils.simple_gpu_utils import setup_gpu_environment
    setup_gpu_environment()

    # Resolve device assignments with your specified GPU allocation
    import torch
    import os

    if device == "auto":
        # Use hardcoded GPU allocation with OpenCompass coordination support
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"🔍 Detected {num_gpus} CUDA devices")

            # Check if OpenCompass wrapper has set coordination environment variables
            gpu_mode = os.environ.get('SPATIAL_REASONING_GPU_MODE', '')
            if gpu_mode == 'coordinated':
                # Use coordinated GPU assignments from OpenCompass wrapper
                lang_gpu = os.environ.get('SPATIAL_REASONING_LANG_GPU', '1')
                perception_gpu = os.environ.get('SPATIAL_REASONING_PERCEPTION_GPU', '2')

                # Validate that the assigned GPUs actually exist in the current CUDA context
                # Strict enforcement of hardcoded GPU allocation - no silent fallbacks
                if int(lang_gpu) >= num_gpus:
                    raise RuntimeError(f"❌ Coordinated language GPU {lang_gpu} not available (only {num_gpus} GPUs visible). "
                                     f"Cannot proceed with hardcoded GPU allocation strategy. "
                                     f"Please ensure OpenCompass allocates sufficient GPUs (num_gpus >= {max(int(lang_gpu), int(perception_gpu)) + 1}).")
                if int(perception_gpu) >= num_gpus:
                    raise RuntimeError(f"❌ Coordinated perception GPU {perception_gpu} not available (only {num_gpus} GPUs visible). "
                                     f"Cannot proceed with hardcoded GPU allocation strategy. "
                                     f"Please ensure OpenCompass allocates sufficient GPUs (num_gpus >= {max(int(lang_gpu), int(perception_gpu)) + 1}).")

                device_assignments = {
                    'segmentation': f'cuda:{perception_gpu}',
                    'detection': f'cuda:{perception_gpu}',
                    'classification': f'cuda:{perception_gpu}',
                    'chat_model': f'cuda:{lang_gpu}'
                }
                print(f"🎯 Coordinated GPU allocation: Language=GPU{lang_gpu}, Perception=GPU{perception_gpu}")
                print("   (Coordinated by OpenCompass wrapper to prevent memory conflicts)")

            elif os.environ.get('CUDA_VISIBLE_DEVICES', ''):
                # OpenCompass environment detection with CUDA_VISIBLE_DEVICES
                cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                print(f"🔧 OpenCompass environment detected: CUDA_VISIBLE_DEVICES={cuda_visible}")

                # CRITICAL FIX: Adapt to the actual number of visible GPUs
                # The GPU indices in device assignments refer to the restricted CUDA context
                if num_gpus >= 3:
                    # Ideal case: GPU0=OpenCompass, GPU1=Language, GPU2=Perception
                    device_assignments = {
                        'segmentation': 'cuda:2',      # GPU 2 for perception tools
                        'detection': 'cuda:2',
                        'classification': 'cuda:2',
                        'chat_model': 'cuda:1'         # GPU 1 for language model
                    }
                    print(f"🎯 OpenCompass GPU allocation: OpenCompass=GPU0, Language=GPU1, Perception=GPU2")
                elif num_gpus >= 2:
                    # Two GPUs available: GPU0=OpenCompass, GPU1=Language+Perception
                    device_assignments = {
                        'segmentation': 'cuda:1',      # Share GPU 1 for perception
                        'detection': 'cuda:1',
                        'classification': 'cuda:1',
                        'chat_model': 'cuda:1'         # GPU 1 for language model
                    }
                    print(f"🎯 OpenCompass GPU allocation (2 GPUs): OpenCompass=GPU0, Language+Perception=GPU1")
                else:
                    # Single GPU available: All components share GPU0
                    device_assignments = {
                        'segmentation': 'cuda:0',      # Shared GPU
                        'detection': 'cuda:0',
                        'classification': 'cuda:0',
                        'chat_model': 'cuda:0'         # Shared GPU
                    }
                    print(f"🎯 Single GPU mode: All components share GPU0")
            else:
                # Standard environment - adapt to available GPUs
                if num_gpus >= 3:
                    # Use original hardcoded assignments
                    device_assignments = {
                        'segmentation': 'cuda:2',      # Perception tools on GPU 2
                        'detection': 'cuda:2',
                        'classification': 'cuda:2',
                        'chat_model': 'cuda:1'         # Language model on GPU 1
                    }
                    print(f"🎯 Standard GPU allocation: OpenCompass=GPU0, Language=GPU1, Perception=GPU2")
                elif num_gpus >= 2:
                    # Two GPUs: GPU0=OpenCompass, GPU1=Language+Perception
                    device_assignments = {
                        'segmentation': 'cuda:1',
                        'detection': 'cuda:1',
                        'classification': 'cuda:1',
                        'chat_model': 'cuda:1'
                    }
                    print(f"🎯 Standard GPU allocation (2 GPUs): OpenCompass=GPU0, Language+Perception=GPU1")
                else:
                    # Single GPU: All components share GPU0
                    device_assignments = {
                        'segmentation': 'cuda:0',
                        'detection': 'cuda:0',
                        'classification': 'cuda:0',
                        'chat_model': 'cuda:0'
                    }
                    print(f"🎯 Standard GPU allocation (1 GPU): All components share GPU0")

            # Set environment variables for tools (only if not already coordinated)
            if gpu_mode != 'coordinated':
                os.environ['SPATIAL_REASONING_GPU_MODE'] = 'hardcoded'
                chat_gpu_id = device_assignments['chat_model'].split(':')[1]
                perception_gpu_id = device_assignments['segmentation'].split(':')[1]
                os.environ['SPATIAL_REASONING_LANG_GPU'] = chat_gpu_id
                os.environ['SPATIAL_REASONING_PERCEPTION_GPU'] = perception_gpu_id


        else:
            # CUDA not available
            device_assignments = {
                'segmentation': 'cpu',
                'detection': 'cpu',
                'classification': 'cpu',
                'chat_model': 'cpu'
            }
            print(f"🎯 CUDA not available: All components use CPU")

        # Clear cache on assigned CUDA GPUs with validation
        if torch.cuda.is_available():
            cuda_devices = set([d for d in device_assignments.values() if d.startswith('cuda:')])
            for gpu_device in cuda_devices:
                gpu_id = int(gpu_device.split(':')[1])
                # Validate GPU ID is within available range
                if gpu_id >= num_gpus:
                    print(f"   ⚠️  Skipping cache clear for {gpu_device}: GPU not available (only {num_gpus} GPUs visible)")
                    continue
                try:
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    print(f"   ✅ Cleared cache on {gpu_device}")
                except Exception as e:
                    print(f"   ⚠️  Failed to clear cache on {gpu_device}: {e}")

    else:
        # Use specified device for all tools
        device_assignments = {
            'segmentation': device,
            'detection': device,
            'classification': device,
            'chat_model': device
        }
        print(f"🎯 Using specified device for all tools: {device}")

    # Debug: Print final device assignments
    print(f"📋 Final device assignments: {device_assignments}")

    # Debug: Print environment variables that tools will see
    print(f"🔍 Environment variables for tools:")
    print(f"   SPATIAL_REASONING_GPU_MODE: {os.environ.get('SPATIAL_REASONING_GPU_MODE', 'Not set')}")
    print(f"   SPATIAL_REASONING_LANG_GPU: {os.environ.get('SPATIAL_REASONING_LANG_GPU', 'Not set')}")
    print(f"   SPATIAL_REASONING_PERCEPTION_GPU: {os.environ.get('SPATIAL_REASONING_PERCEPTION_GPU', 'Not set')}")

    # Extract chat model device for later use
    chat_model_device = device_assignments['chat_model']
    print(f"🔧 Using chat model device: {chat_model_device}")

    # Prepare the assigned device for model loading
    if chat_model_device.startswith('cuda:'):
        import torch  # Ensure torch is available in this scope
        device_id = int(chat_model_device.split(':')[1])
        print(f"🧹 Preparing {chat_model_device} for model loading...")
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 2. Load system prompt
    load_dotenv()
    prompts = load_prompts_from_file(prompt_file)
    system_prompt = prompts.get("WATER_FLOOD_ASSISTANT", "")

    # 3. Instantiate tools with simplified, canonical names
    # Fix lambda closure issue by capturing device assignments in local variables
    segmentation_device = device_assignments['segmentation']
    detection_device = device_assignments['detection']
    classification_device = device_assignments['classification']

    print(f"🔧 Tool device assignments:")
    print(f"   Segmentation: {segmentation_device}")
    print(f"   Detection: {detection_device}")
    print(f"   Classification: {classification_device}")

    all_tools = {
        # Perception tools (RemoteSAM-based)
        "segmentation": lambda dev=segmentation_device: RemoteSAMSegmentationTool(device=dev, fallback_to_hf=True),     # RemoteSAM segmentation tool
        "detection": lambda dev=detection_device: RemoteSAMDetectionTool(device=dev),               # RemoteSAM detection tool
        "classification": lambda dev=classification_device: RemoteSAMClassificationTool(device=dev), # RemoteSAM 25-class remote sensing classification tool
        "change_detection": lambda dev=detection_device: ChangeDetectionTool(device=dev),           # Change3D semantic change detection tool

        # SAR analysis tools (Enhanced SARATR-X-based with newer implementation)
        "sar_detection": lambda dev=detection_device: SARDetectionTool(device=dev),           # Enhanced SAR object detection using SARATR-X with MMDetection 3.x support
        "sar_classification": lambda dev=classification_device: SARClassificationTool(device=dev), # Enhanced SAR scene/target classification using improved HiViT architecture

        # IR analysis tools (DMIST-based infrared detection)
        "infrared_detection": lambda dev=detection_device: InfraredDetectionTool(device=dev), # Infrared small target detection using DMIST framework

        # Spatial relations tools (CPU-based geometric operations)
        "buffer": lambda: BufferTool(),                           # Buffer zone creation
        "distance_calculation": lambda: DistanceCalculationTool(), # Distance measurement
        "overlap": lambda: OverlapRatioTool(),                    # IoU and overlap analysis
        "containment": lambda: ContainmentTool(),                 # Containment analysis

        # Spatial statistics tools (CPU-based statistical operations)
        "area_measurement": lambda: AreaMeasurementTool(),         # Area calculation
        "object_count_aoi": lambda: ObjectCountInAOITool(),       # Object counting in AOI
        # NOTE: No special "answer synthesis" tool needed - proper ReAct termination
        # comes from tools returning actionable results (like MedRAX does).
    }


    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 tools_to_use parameter: {tools_to_use}")
    logger.info(f"🔧 all_tools keys: {list(all_tools.keys())}")

    # Debug: Write to file in current directory
    import os
    debug_file = os.path.join(os.getcwd(), 'agent_factory_debug.log')
    try:
        with open(debug_file, 'a') as f:
            f.write(f"tools_to_use = {tools_to_use}\n")
            f.write(f"all_tools keys = {list(all_tools.keys())}\n")
            f.write(f"infrared_detection in all_tools = {'infrared_detection' in all_tools}\n")
            if tools_to_use:
                f.write(f"infrared_detection in tools_to_use = {'infrared_detection' in tools_to_use}\n")
            f.write("---\n")
    except Exception as e:
        logger.error(f"Failed to write debug log: {e}")

    tools_dict: Dict[str, BaseTool] = {}
    for name in (tools_to_use or all_tools.keys()):
        if name in all_tools:
            try:
                print(f"Initializing tool: {name}...")
                tools_dict[name] = all_tools[name]()
                print(f"✅ Successfully initialized: {name}")
            except Exception as e:
                print(f"❌ Failed to initialize tool {name}: {e}")
                print(f"   Skipping {name} and continuing with other tools...")
                continue

    print(f"Initialized {len(tools_dict)} tools: {list(tools_dict.keys())}")

    if len(tools_dict) == 0:
        print("⚠️ Warning: No tools were successfully initialized!")
        print("   This may be expected in testing environments.")
        print("   Please check if this is intentional:")
        print("   - Model files exist (e.g., pretrained_weights/RemoteSAMv1.pth)")
        print("   - CUDA is available if using GPU")
        print("   - All required dependencies are installed")
        print("   Continuing with empty tool set for testing purposes...")

    # 3. Load model using abstraction layer (default) or legacy approach
    if use_model_abstraction:
        print(f"🚀 Using model abstraction layer with {model_type} model...")
        from spatialreason.models import ModelConfig, create_model_manager_from_config

        # Create model configuration based on type
        if model_type.lower() == "local":
            model_path = local_model_path or "model/Qwen2-VL-7B-Instruct"
            config = ModelConfig.local_config(
                model_path=model_path,
                device=chat_model_device
            )
            print(f"   📁 Local model path: {model_path}")
            print(f"   🎯 Device: {chat_model_device}")
        elif model_type.lower() == "remote":
            # Detect Ollama models and use appropriate endpoint
            # Check if this is an Ollama model (contains ':' or '/' or matches known Ollama patterns)
            is_ollama_model = (
                ":" in model_name or  # Direct Ollama format: qwen:7b-chat, llama3.1:8b
                "/" in model_name or  # Ollama namespace format: internlm/internlm3-8b-instruct
                model_name == "qwen2.5-7b-instruct" or  # Legacy format
                model_name.startswith("ollama_")  # Explicit Ollama prefix
            )

            if is_ollama_model:
                # Use Ollama endpoint for Ollama models
                ollama_model_name = model_name

                # Convert legacy format if needed: qwen2.5-7b-instruct -> qwen2.5:7b-instruct
                if model_name == "qwen2.5-7b-instruct":
                    ollama_model_name = model_name.replace("-", ":", 1)  # Replace first hyphen with colon

                # Remove ollama_ prefix if present
                if ollama_model_name.startswith("ollama_"):
                    ollama_model_name = ollama_model_name[7:]  # Remove "ollama_" prefix

                config = ModelConfig.remote_config(
                    model_name=ollama_model_name,
                    base_url="http://127.0.0.1:11434/v1/",
                    api_key="ollama"  # Ollama doesn't require a real API key
                )
                print(f"   🌐 Remote model: {model_name} (Ollama)")
                print(f"   🔗 API endpoint: http://127.0.0.1:11434/v1/")
                print(f"   📝 Ollama model name: {ollama_model_name}")
            else:
                # Use default GPT endpoint for other models
                config = ModelConfig.remote_config(model_name=model_name)
                print(f"   🌐 Remote model: {model_name}")
                print(f"   🔗 API endpoint configured")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Create unified model manager
        model_manager = create_model_manager_from_config(config)

        # For compatibility with existing code, we'll extract the underlying model components
        # when using local models, or use the manager directly for remote models
        if model_type.lower() == "local":
            # Extract the actual model components for compatibility
            local_model = model_manager.model
            model = local_model.model
            tokenizer = local_model.tokenizer
            processor = local_model.processor
            print(f"✅ Model abstraction layer initialized with local {model_name}")
        else:
            # For remote models, use the manager directly
            # The EnhancedChatModel will handle the remote model interface
            model = model_manager
            tokenizer = None
            processor = None
            print(f"✅ Model abstraction layer initialized with remote {model_name}")

            # Skip GPU memory management for remote models
            print("   ⏭️  Skipping GPU memory management (remote model)")
    else:
        # 3. Load Qwen2-VL model with memory management (legacy approach)
        print(f"Loading {model_name} model on device: {chat_model_device}...")

        # Clear GPU cache before loading model to prevent OOM
        if torch.cuda.is_available() and 'cuda' in chat_model_device:
            torch.cuda.empty_cache()
            print(f"🧹 Cleared CUDA cache before model loading")

        # Legacy model loading approach
        try:
            # Force local-only mode if specified
            if force_local_only:
                if not local_model_path:
                    raise ValueError(
                        "force_local_only=True requires local_model_path to be specified. "
                        "Please provide the path to your local Qwen2-VL model directory."
                    )
                print(f"🔒 FORCE LOCAL-ONLY MODE: Using local model from {local_model_path}")
                print(f"🚫 All Hugging Face Hub fallbacks are DISABLED")
                # Use standard transformers loading for local model directory
                model, tokenizer, processor = _load_model_with_transformers(
                    local_model_path, chat_model_device
                )
            else:
                # Determine model path - use local path if provided, otherwise HuggingFace Hub
                if local_model_path:
                    model_path = local_model_path
                    print(f"[DEBUG] Using local model path: {model_path}")
                    use_local_imports = True
                else:
                    # Always use Hugging Face Hub model path
                    if model_name == "Qwen2-VL-7B-Instruct":
                        model_path = "Qwen/Qwen2-VL-7B-Instruct"
                        print(f"[DEBUG] Using Hugging Face Hub model: {model_path}")
                    elif model_name == "Qwen/Qwen2-VL-7B-Instruct":
                        model_path = model_name  # Already in correct format
                        print(f"[DEBUG] Using Hugging Face Hub model: {model_path}")
                    else:
                        model_path = model_name
                    use_local_imports = False

                # Load model using direct imports if local path is provided
                if use_local_imports:
                    print(f"🚀 Loading {model_name} using direct imports from local path...")
                    model, tokenizer, processor = _load_model_with_direct_imports(
                        model_path, chat_model_device
                    )
                else:
                    print(f"🚀 Loading {model_name} using standard transformers approach...")
                    model, tokenizer, processor = _load_model_with_transformers(
                        model_path, chat_model_device
                    )

                print(f"[DEBUG] ✓ Successfully loaded {model_name}")
                print(f"[DEBUG] Target device: {chat_model_device}")
                print(f"[DEBUG] Model dtype: {model.dtype}")

                # Verify model is actually on assigned device
                if hasattr(model, 'device'):
                    actual_device = str(model.device)
                    print(f"[DEBUG] Actual model device: {actual_device}")
                    if chat_model_device not in actual_device:
                        print(f"[WARNING] Model may not be fully on {chat_model_device}. Check device mapping.")
                else:
                    print(f"[DEBUG] Model device verification: Checking first parameter...")
                    first_param_device = next(model.parameters()).device
                    print(f"[DEBUG] First parameter device: {first_param_device}")
                    if chat_model_device not in str(first_param_device):
                        print(f"[WARNING] Model parameters may not be on {chat_model_device}!")

                # Additional memory management after model loading
                if torch.cuda.is_available() and 'cuda' in chat_model_device:
                    torch.cuda.empty_cache()
                    device_id = int(chat_model_device.split(':')[1])
                    memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3   # GB
                    print(f"[DEBUG] GPU {device_id} memory after model loading: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        except Exception as e:
            print(f"[ERROR] Failed to load {model_name} from Hugging Face Hub: {e}")
            raise

    # 4. Initialize Multi-Step Planner with semantic tool retrieval (CONDITIONAL - can be disabled for ablation)
    if disable_planner:
        print("⚠️ ABLATION MODE: Planner initialization skipped")
        print("   Agent will use ReAct-style iterative tool execution")
        planner = None
    else:
        print("🧠 Initializing Multi-Step Planner with semantic tool retrieval...")
        try:
            from spatialreason.plan.plan import PlannerProcessor

            # Initialize model-agnostic planner for both local and remote models
            # Supports GPT-4o, Claude, Llama, Qwen, and other models through unified interface
            print("   🧠 Initializing model-agnostic planner for structured planning and execution...")

            # Resolve corpus.tsv path relative to the agent directory
            import os
            agent_root = Path(__file__).parent.parent  # Go up to SpatialreasonAgent root
            corpus_path = agent_root / "spatialreason" / "plan" / "corpus.tsv"

            if use_model_abstraction and model_type.lower() == "remote":
                print("   🌐 Using remote model for planner with semantic tool retrieval")
                print(f"   🤖 Model: {model_name} (model-agnostic interface)")
                # For remote models, create planner with model-agnostic PlannerLLM
                planner = PlannerProcessor(
                    input_query="",  # Will be set dynamically per query
                    device="cpu",  # Remote models don't need GPU device
                    model_path="remote",  # Indicate remote model usage
                    corpus_tsv_path=str(corpus_path),
                    retrieval_model_path="sentence-transformers/all-MiniLM-L6-v2",
                    tools_dict=tools_dict,  # Pass actual tools for real execution
                    shared_model=None,  # No shared model for remote
                    shared_tokenizer=None,  # No shared tokenizer for remote
                    evaluation_mode=None,  # Auto-detect based on environment
                    remote_model_manager=model_manager if use_model_abstraction else None  # Pass remote model manager
                )
                print(f"✅ Model-agnostic planner initialized with remote {model_name} model")
            else:
                # Local model initialization (Qwen, Llama, etc.)
                planner_model_path = local_model_path if local_model_path else model_name
                print(f"   📁 Using local model for planner: {planner_model_path}")
                print("   🤖 Backend: Local model with model-agnostic interface")

                # Initialize planner with proper device and model configuration
                # Pass shared model instances to prevent CUDA OOM
                # Auto-detect evaluation mode to control mock fallbacks
                planner = PlannerProcessor(
                    input_query="",  # Will be set dynamically per query
                    device=chat_model_device,
                    model_path=planner_model_path,
                    corpus_tsv_path=str(corpus_path),
                    retrieval_model_path="sentence-transformers/all-MiniLM-L6-v2",
                    tools_dict=tools_dict,  # Pass actual tools for real execution
                    shared_model=model if hasattr(model, 'parameters') else None,     # Only pass if it's a torch model
                    shared_tokenizer=tokenizer,  # Share the already-loaded tokenizer
                    evaluation_mode=None    # Auto-detect based on environment
                )
                print("✅ Model-agnostic planner initialized with local model")
        except Exception as e:
            print(f"⚠️ Failed to initialize planner: {e}")
            print("   Falling back to direct tool selection")
            planner = None

    # 5. Wrap into EnhancedChatModel with unified interfaces
    print("🤖 Initializing Enhanced Chat Model with dynamic tool loading...")
    llm = EnhancedChatModel(
        model=model,
        tokenizer=tokenizer,
        device=chat_model_device,  # Use dynamically assigned device
        planner=planner  # Pass planner for intelligent tool selection
    )

    # 6. Create Agent
    agent = Agent(
        llm,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=system_prompt,
        checkpointer=MemorySaver(),
        max_turn=10,  # Maximum iterations for ReAct-style execution
    )

    return agent, tools_dict, planner
