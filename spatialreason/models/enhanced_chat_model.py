"""
Enhanced chat model that uses the unified tool interface and execution engine.
Eliminates duplication from the original MergedLlamaChatModel.
"""

import os
import re
import json
import uuid
import importlib
from typing import Any, List, Optional, Dict, ClassVar
from pydantic import PrivateAttr
import torch
from PIL import Image
from copy import deepcopy

from langchain_core.tools import BaseTool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage
from langchain.schema import ChatMessage, ChatGeneration, LLMResult
from langchain.chat_models.base import BaseChatModel

from .response import Response
from .base_llm import BaseLLM
from ..tools.tool_interface import get_tool_interface


class EnhancedChatModel(BaseChatModel):
    """
    Enhanced chat model with Chain-of-Thought reasoning for spatial analysis.
    Focused architecture for multi-step spatial reasoning tasks with dynamic tool loading.
    """

    # CoT reasoning constants
    DEFAULT_MAX_STEPS: ClassVar[int] = 5
    DEFAULT_NUM_PASSES: ClassVar[int] = 1

    # Default device for chat model (hardcoded assignment)
    DEFAULT_DEVICE: ClassVar[str] = "cuda:1"

    # Tool devices (hardcoded assignments)
    TOOL_DEVICES: ClassVar[Dict[str, str]] = {
        "segmentation": "cuda:2",
        "detection": "cuda:2",
        "classification": "cuda:2"
    }

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _device: str = PrivateAttr()
    _planner: Any = PrivateAttr()

    # Dynamic tool loading attributes
    _tool_cache: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _available_tools: List[str] = PrivateAttr(default_factory=lambda: ["segmentation", "detection", "classification"])

    def __init__(self,
                 model: Any,
                 tokenizer: Any,
                 device: str = None,
                 planner: Any = None,
                 **kwargs):
        """
        Initialize enhanced chat model for CoT reasoning with dynamic device assignment.

        Args:
            model: Model instance (Qwen2-VL model or UnifiedModelManager for remote models)
            tokenizer: Qwen tokenizer (None for remote models)
            device: Device to use for chat model (default: cuda:0, ignored for remote models)
            planner: Optional planner for intelligent tool selection
        """
        super().__init__(**kwargs)

        # Detect if we're using a remote model (UnifiedModelManager)
        from .model_manager import UnifiedModelManager
        self._is_remote_model = isinstance(model, UnifiedModelManager)

        if self._is_remote_model:
            print("🌐 Enhanced Chat Model: Using remote model interface")
            self._model_manager = model
            self._model = None
            self._tokenizer = None
            self._device = "remote"  # Special device indicator for remote models
            # Skip device validation for remote models
        else:
            print("🖥️ Enhanced Chat Model: Using local model interface")
            # Set device (use provided device or default)
            self._device = device or self.DEFAULT_DEVICE

            # Validate the assigned device is available
            self._validate_chat_model_device()

            # Validate tool devices are available
            self._validate_tool_devices()

            # Move model and tokenizer to assigned device
            self._model = model.to(self._device)
            self._tokenizer = tokenizer
            self._model_manager = None

        self._planner = planner

        # Initialize dynamic tool loading
        self._tool_cache = {}
        self._available_tools = ["segmentation", "detection", "classification"]
        self._bound_tools = []  # Tools bound for ReAct-style execution

        # CoT reasoning is always enabled in this simplified architecture
        # Initialize CoT helper methods
        self._initialize_cot_helpers()

    def _validate_chat_model_device(self) -> None:
        """
        Validate that the assigned chat model device is available.

        Raises:
            RuntimeError: If the assigned device is not available
        """
        # Skip validation for remote models
        if self._device == 'remote':
            print(f"🌐 Remote model device validation skipped")
            return

        # Allow CPU usage for testing and environments without CUDA
        if self._device == 'cpu':
            print(f"[DEBUG] Using CPU device for enhanced chat model")
            return

        # For CUDA devices, perform validation
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system, but CUDA device was requested.")

        # Extract device ID from device string
        if self._device.startswith('cuda:'):
            device_id = int(self._device.split(':')[1])
        else:
            raise RuntimeError(f"Invalid device format: {self._device}. Expected 'cuda:X' or 'cpu' format.")

        if torch.cuda.device_count() <= device_id:
            raise RuntimeError(f"CUDA device {device_id} is not available. Only {torch.cuda.device_count()} CUDA devices found.")

        # Test device accessibility
        try:
            torch.cuda.set_device(device_id)
            # Test memory allocation on the device
            test_tensor = torch.tensor([1.0], device=self._device)
            del test_tensor
            torch.cuda.empty_cache()
            print(f"✅ Chat model device {self._device} validated successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to access CUDA device {device_id}: {str(e)}")

    def _validate_tool_devices(self) -> None:
        """
        Validate that tool devices are available.

        Raises:
            RuntimeError: If required tool devices are not available
        """
        # Skip tool device validation for remote models
        if self._device == 'remote':
            print(f"🌐 Tool device validation skipped for remote model")
            return

        required_devices = set(self.TOOL_DEVICES.values())

        for device in required_devices:
            # Skip validation for "auto" devices - they will be resolved later
            if device == "auto":
                print(f"[INFO] Tool device '{device}' will be resolved automatically")
                continue

            # Skip validation for CPU devices
            if device == "cpu":
                print(f"[INFO] Tool device '{device}' validated (CPU)")
                continue

            # Validate CUDA devices
            if device.startswith('cuda:'):
                try:
                    device_id = int(device.split(':')[1])
                    if torch.cuda.device_count() <= device_id:
                        print(f"[WARNING] Tool device {device} not available. Only {torch.cuda.device_count()} CUDA devices found.")
                        print(f"[WARNING] Tools will fall back to available devices.")
                        # Update tool devices to use available devices
                        self._fallback_tool_devices()
                        break
                    else:
                        # Test device accessibility
                        try:
                            torch.cuda.set_device(device_id)
                            test_tensor = torch.tensor([1.0], device=device)
                            del test_tensor
                            torch.cuda.empty_cache()
                            print(f"[INFO] Tool device {device} validated successfully")
                        except Exception as e:
                            print(f"[WARNING] Failed to access tool device {device}: {str(e)}")
                            self._fallback_tool_devices()
                            break
                except (ValueError, IndexError) as e:
                    print(f"[WARNING] Invalid device format '{device}': {e}")
                    self._fallback_tool_devices()
                    break
            else:
                print(f"[WARNING] Unknown device format '{device}', skipping validation")

    def _fallback_tool_devices(self) -> None:
        """
        Fallback tool devices to available GPUs or CPU when hardcoded assignments fail.
        """
        print("[INFO] Applying tool device fallback strategy...")

        if not torch.cuda.is_available():
            # No CUDA available - use CPU for all tools
            for tool_name in self.TOOL_DEVICES:
                self.TOOL_DEVICES[tool_name] = "cpu"
            print("[INFO] CUDA not available - all tools assigned to CPU")
            return

        num_gpus = torch.cuda.device_count()
        print(f"[INFO] {num_gpus} GPUs available for tool fallback")

        # Assign tools to available GPUs, preferring GPU 0 for single GPU systems
        if num_gpus >= 1:
            # Use GPU 0 for all tools in single/limited GPU scenarios
            fallback_device = "cuda:0"
            for tool_name in self.TOOL_DEVICES:
                self.TOOL_DEVICES[tool_name] = fallback_device
            print(f"[INFO] All tools assigned to {fallback_device} (fallback mode)")
        else:
            # No GPUs available - use CPU
            for tool_name in self.TOOL_DEVICES:
                self.TOOL_DEVICES[tool_name] = "cpu"
            print("[INFO] No GPUs available - all tools assigned to CPU")

    def _setup_automatic_tool_devices(self) -> None:
        """Setup hardcoded tool device assignments."""
        # Hardcoded GPU assignments according to allocation strategy
        hardcoded_assignments = {
            'segmentation': 'cuda:2',      # All perception tools use GPU 2
            'detection': 'cuda:2',
            'classification': 'cuda:2',
            'containment': 'cpu',          # Spatial tools use CPU
            'overlap': 'cpu',
            'buffer': 'cpu',
            'area_measurement': 'cpu',
            'length_measurement': 'cpu',
            'object_count_aoi': 'cpu'
        }

        # Update tool devices with hardcoded assignments
        for tool_name, device in hardcoded_assignments.items():
            if tool_name in self.TOOL_DEVICES:
                self.TOOL_DEVICES[tool_name] = device

        # Verify assigned CUDA devices are available
        import torch
        for tool_name, device in hardcoded_assignments.items():
            if device.startswith('cuda:'):
                device_id = int(device.split(':')[1])
                if not torch.cuda.is_available():
                    raise RuntimeError(f"CUDA not available but {device} was assigned to {tool_name}. Cannot proceed with hardcoded GPU allocation strategy.")
                elif device_id >= torch.cuda.device_count():
                    raise RuntimeError(f"GPU {device_id} not available for {tool_name} (only {torch.cuda.device_count()} GPUs detected). Cannot proceed with hardcoded GPU allocation strategy.")

        print(f"[INFO] Hardcoded tool device assignments: {self.TOOL_DEVICES}")
        print("[INFO] GPU allocation: OpenCompass=GPU0, LanguageModel=GPU1, PerceptionTools=GPU2")

    def _initialize_cot_helpers(self):
        """Initialize Chain-of-Thought helper methods."""
        try:
            from spatialreason.models.cot_helpers import add_cot_methods_to_model
            add_cot_methods_to_model(self)
        except ImportError:
            # CoT helpers not available - CoT functionality will be disabled
            pass

    def _generate_with_qwen(self, messages) -> str:
        """
        Generate response using either local Qwen model or remote GPT-4o model.

        Args:
            messages: List of chat messages

        Returns:
            str: Generated response
        """
        try:
            if self._is_remote_model:
                # Use remote model manager for generation
                return self._model_manager.generate_chat_response(messages, temperature=0.1)
            else:
                # Use local Qwen model for generation
                import torch

                # Apply chat template
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Tokenize
                inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

                # Generate
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self._tokenizer.pad_token_id
                    )

                # Decode response
                response = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                return response

        except Exception as e:
            print(f"[ERROR] Model generation failed: {e}")
            # Return a fallback response
            return "Thought: I need to analyze the image. Action: segmentation Action Input: {\"text_prompt\": \"objects\"}"


    def _get_tool_class_mapping(self) -> Dict[str, str]:
        """
        Get mapping of tool names to their class paths for dynamic loading.

        Returns:
            Dict mapping tool names to importable class paths
        """
        return {
            "segmentation": "spatialreason.tools.Perception.segmentation.RemoteSAMSegmentationTool",
            "detection": "spatialreason.tools.Perception.detection.RemoteSAMDetectionTool",
            "classification": "spatialreason.tools.Perception.classification.RemoteSAMClassificationTool"
        }

    def _load_tool(self, tool_name: str) -> Any:
        """
        Dynamically load and instantiate a tool by name.

        Args:
            tool_name: Name of the tool to load

        Returns:
            Instantiated tool object

        Raises:
            ValueError: If tool name is not recognized
            ImportError: If tool class cannot be imported
        """
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        tool_mapping = self._get_tool_class_mapping()
        if tool_name not in tool_mapping:
            raise ValueError(f"Unknown tool: {tool_name}. Available tools: {list(tool_mapping.keys())}")

        class_path = tool_mapping[tool_name]
        try:
            # Split module path and class name
            module_path, class_name = class_path.rsplit('.', 1)

            # Import module and get class
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)

            # Instantiate tool with appropriate device (separate from chat model)
            tool_device = self.TOOL_DEVICES.get(tool_name, "cuda:1")  # Default to cuda:1
            print(f"[Dynamic Loading] Loading {tool_name} tool on {tool_device} (chat model on {self._device})")

            if tool_name in ["segmentation", "detection", "classification"]:
                tool_instance = tool_class(device=tool_device)
            else:
                # Fallback for unknown tools
                tool_instance = tool_class()

            # Cache the tool instance
            self._tool_cache[tool_name] = tool_instance

            return tool_instance

        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load tool '{tool_name}' from '{class_path}': {str(e)}")

    def _get_available_tool_names(self) -> List[str]:
        """
        Get list of available tool names.

        Returns:
            List of available tool names
        """
        return self._available_tools.copy()


    def _execute_single_tool(self, user_text: str, image_path: Optional[str]) -> Response:
        """
        Execute a single tool based on user query using dynamic tool loading.

        Args:
            user_text: User's query text
            image_path: Optional path to image file

        Returns:
            Response object with tool execution result
        """
        selected_tool_name = self.select_tool_for_query(user_text)
        if selected_tool_name:
            try:
                # Dynamically load the tool
                tool_instance = self._load_tool(selected_tool_name)

                # Get tool interface for argument creation
                from spatialreason.tools.tool_interface import get_tool_interface
                tool_interface = get_tool_interface(selected_tool_name)
                actual_image_path = image_path if image_path else "IMAGE"
                args = tool_interface.create_tool_args(
                    image_path=actual_image_path,
                    user_input=user_text
                )

                # Execute tool
                result = tool_instance.invoke(args)

                # Update tool context if this was a segmentation tool
                if selected_tool_name == "segmentation":
                    from spatialreason.tools.tool_interface import update_segmentation_context
                    text_prompt = args.get("text_prompt", user_text)
                    update_segmentation_context(actual_image_path, result, text_prompt)

                # Return simple response with just content (no tool_calls to avoid format issues)
                return Response({
                    "content": result,
                    "tool_calls": []
                })

            except (ValueError, ImportError) as e:
                return Response({
                    "content": f"Failed to load or execute tool '{selected_tool_name}': {str(e)}",
                    "tool_calls": []
                })
            except Exception as e:
                # Catch any other errors during tool execution
                import traceback
                error_details = traceback.format_exc()
                print(f"[ERROR] Tool execution error for '{selected_tool_name}': {str(e)}")
                print(f"[ERROR] Full traceback:\n{error_details}")

                return Response({
                    "content": f"Tool execution failed for '{selected_tool_name}': {str(e)}. Check console for details.",
                    "tool_calls": []
                })

        return Response({
            "content": "I wasn't able to process your request. Please try again with a more specific query.",
            "tool_calls": []
        })

    def _create_cot_prompt(self, user_text: str) -> tuple[str, str]:
        """
        Create Chain-of-Thought prompts for spatial analysis.

        Args:
            user_text: User's query text

        Returns:
            tuple: (system_prompt, user_prompt)
        """
        # Use spatial-specific ReAct prompts for CoT
        from spatialreason.cot.spatial_prompts import get_spatial_prompt
        return get_spatial_prompt(user_text, "Analyze satellite imagery using remote sensing tools")


    @property
    def _llm_type(self) -> str:
        return "enhanced-qwen2vl-chat-model"

    def _generate_react_tool_calls(self, user_text: str, image_path: Optional[str], messages: List[Any]) -> AIMessage:
        """
        Generate tool calls for ReAct-style execution when planner is disabled.

        This method uses the remote model (GPT-4o) to decide which tools to call based on the query.

        Args:
            user_text: User query
            image_path: Path to image
            messages: Conversation history

        Returns:
            AIMessage with tool_calls or final answer
        """
        import json

        # Check if this is a continuation (has tool results in history)
        has_tool_results = any(hasattr(msg, 'type') and msg.type == 'tool' for msg in messages)

        if has_tool_results:
            # This is a continuation - decide whether to continue reasoning or generate final answer
            print("[DEBUG] ReAct: Deciding next action after tool execution")

            # Build conversation context with tool results for the decision
            conversation = []
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    # User message
                    if 'user' in type(msg).__name__.lower() or (hasattr(msg, 'type') and msg.type == 'human'):
                        conversation.append({"role": "user", "content": str(msg.content)})
                    # Assistant message
                    elif 'ai' in type(msg).__name__.lower():
                        conversation.append({"role": "assistant", "content": str(msg.content)})
                # Tool result
                elif hasattr(msg, 'type') and msg.type == 'tool':
                    tool_name = getattr(msg, 'name', 'unknown')
                    tool_content = str(getattr(msg, 'content', ''))[:300]  # Limit length
                    conversation.append({
                        "role": "system",
                        "content": f"Tool '{tool_name}' returned: {tool_content}..."
                    })

            # Get available tool names
            available_tools = [t.name for t in self._bound_tools] if self._bound_tools else []
            tools_list = ", ".join(available_tools)

            # Ask model to decide next action
            decision_prompt = f"""You are performing iterative spatial analysis using the ReAct framework (Reasoning + Acting).

Original Query: {user_text}

You have executed some tools and received their results (shown above in the conversation history).

Now decide your NEXT ACTION:

Option 1: CONTINUE REASONING - If you need more information to fully answer the query
   - Select another tool to execute
   - Respond with JSON: {{"action": "continue", "tool_name": "tool_name", "arguments": {{...}}}}

Option 2: FINISH - If you have enough information to provide a complete answer
   - Respond with JSON: {{"action": "finish", "answer": "your final answer"}}

Available tools: {tools_list}

Guidelines:
- For complex queries requiring multiple steps (e.g., "area of X within Y meters of Z"), you likely need multiple tools
- For simple queries (e.g., "segment water bodies"), one tool may be sufficient
- Consider if you have all the data needed to answer the original query

IMPORTANT TOOL DEPENDENCIES:
- Spatial relation tools (buffer, overlap, containment) require perception_results from perception tools
- If you haven't run a perception tool yet, you must run one first (segmentation/detection/classification)
- Spatial statistics tools (area_measurement, distance_calculation, object_count_aoi) also require perception_results

PARAMETER FORMAT:
- Use simplified parameters (e.g., buffer_class, geometry_count for buffer tool)
- The system will automatically provide perception_results to spatial tools
- For buffer tool: use buffer_class (not classes_used), buffer_distance_meters, meters_per_pixel, geometry_count

Respond with ONLY a JSON object, no other text or markdown formatting."""

            conversation.append({"role": "user", "content": decision_prompt})

            # Get decision from model
            if self._is_remote_model:
                response_text = self._model_manager.generate_chat_response(
                    conversation,
                    temperature=0.1
                )
            else:
                response_text = self._generate_text(conversation)

            # Parse the decision
            try:
                # Clean up response (remove markdown code blocks if present)
                response_text = response_text.strip()
                if response_text.startswith("```"):
                    lines = response_text.split("\n")
                    response_text = "\n".join([l for l in lines if not l.startswith("```")])
                    response_text = response_text.strip()

                decision = json.loads(response_text)
                action = decision.get("action", "finish")

                if action == "continue":
                    # Continue reasoning - generate another tool call
                    tool_name = decision.get("tool_name")
                    tool_args = decision.get("arguments", {})

                    # Add image_path to arguments if not present and tool needs it
                    if image_path and "image_path" not in tool_args:
                        tool_args["image_path"] = image_path

                    # Create AIMessage with tool_calls
                    tool_call_id = str(uuid.uuid4())
                    msg = AIMessage(
                        content="",
                        tool_calls=[{
                            "id": tool_call_id,
                            "name": tool_name,
                            "args": tool_args
                        }]
                    )

                    print(f"[DEBUG] ReAct: Continuing reasoning - next tool: {tool_name}")
                    return msg

                else:  # action == "finish"
                    # Generate final answer
                    final_answer = decision.get("answer", "Analysis complete.")
                    print(f"[DEBUG] ReAct: Finishing - generating final answer")
                    return AIMessage(content=final_answer)

            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARNING] Failed to parse decision from model: {e}")
                print(f"[WARNING] Model response was: {response_text}")

                # Fallback: Try to generate final answer
                synthesis_prompt = f"{user_text}\n\nBased on the tool execution results above, provide a concise final answer."

                if self._is_remote_model:
                    final_answer = self._model_manager.generate_chat_response(
                        [{"role": "user", "content": synthesis_prompt}],
                        temperature=0.1
                    )
                else:
                    final_answer = self._generate_text([{"role": "user", "content": synthesis_prompt}])

                print(f"[DEBUG] ReAct: Using fallback final answer")
                return AIMessage(content=final_answer)

        # First call - generate tool calls
        print("[DEBUG] ReAct: Generating initial tool calls")

        # Create detailed tool descriptions with parameter schemas
        tool_details = []
        for tool in self._bound_tools:
            # Get parameter schema from the tool
            params_info = []
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                properties = schema.get('properties', {})
                required = schema.get('required', [])

                for param_name, param_info in properties.items():
                    param_desc = param_info.get('description', '')
                    is_required = param_name in required
                    req_str = "REQUIRED" if is_required else "optional"
                    params_info.append(f"    - {param_name} ({req_str}): {param_desc}")

            tool_detail = f"- {tool.name}: {tool.description}\n  Parameters:\n" + "\n".join(params_info)
            tool_details.append(tool_detail)

        tool_descriptions = "\n\n".join(tool_details)

        planning_prompt = f"""You are a spatial reasoning assistant. Analyze the following query and decide which tool(s) to use.

Available tools:
{tool_descriptions}

User query: {user_text}
Image path: {image_path}

IMPORTANT TOOL DEPENDENCIES:
- Spatial relation tools (buffer, overlap, containment) require perception_results from perception tools (segmentation, detection, classification)
- Always start with a perception tool (segmentation/detection/classification) to identify objects
- Then use spatial relation tools or spatial statistics tools to analyze relationships

Based on the query, select the FIRST tool to use and provide its arguments in JSON format.
IMPORTANT: Use simplified parameter format (the system will automatically provide perception_results to spatial tools).

For perception tools (segmentation, detection, classification):
- image_path: the path to the image
- text_prompt: what objects to identify (extracted from the user query)

For buffer tool (use AFTER perception tool):
- buffer_class: the class to create buffers around (e.g., "trees", "buildings")
- buffer_distance_meters: buffer distance in meters
- meters_per_pixel: ground resolution (extract from query, e.g., "GSD = 0.05 m/px" means 0.05)
- geometry_count: number of objects of buffer_class (from perception results)

For overlap/containment tools (use AFTER perception tool):
- Use simplified parameters (the system will provide perception_results automatically)

Respond with ONLY a JSON object in this exact format:
{{
    "tool_name": "name_of_tool",
    "arguments": {{
        "image_path": "path/to/image",
        "text_prompt": "objects to segment",
        "other_param": "value"
    }}
}}

Do not include any other text, explanations, or markdown formatting."""

        # Get tool selection from model
        if self._is_remote_model:
            response_text = self._model_manager.generate_chat_response(
                [{"role": "user", "content": planning_prompt}],
                temperature=0.1
            )
        else:
            response_text = self._generate_text([{"role": "user", "content": planning_prompt}])

        # Parse the response to extract tool call
        try:
            # Clean up response (remove markdown code blocks if present)
            response_text = response_text.strip()
            if response_text.startswith("```"):
                # Remove markdown code blocks
                lines = response_text.split("\n")
                response_text = "\n".join([l for l in lines if not l.startswith("```")])
                response_text = response_text.strip()

            tool_call_data = json.loads(response_text)
            tool_name = tool_call_data.get("tool_name")
            tool_args = tool_call_data.get("arguments", {})

            # Add image_path to arguments if not present and tool needs it
            if image_path and "image_path" not in tool_args:
                tool_args["image_path"] = image_path

            # Create AIMessage with tool_calls
            tool_call_id = str(uuid.uuid4())
            msg = AIMessage(
                content="",
                tool_calls=[{
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": tool_args
                }]
            )

            print(f"[DEBUG] ReAct: Generated tool call - {tool_name} with args: {tool_args}")
            return msg

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print(f"[ERROR] Failed to parse tool call from model response: {e}")
            print(f"[ERROR] Model response was: {response_text}")

            # Fallback: Return a message without tool calls (will end the workflow)
            return AIMessage(content="I need more information to proceed with this analysis.")

    def bind_tools(self, tools: List[BaseTool]) -> "EnhancedChatModel":
        """
        Bind tools to the model (compatibility method for LangChain Agent).
        Stores tools for ReAct-style tool calling when planner is disabled.
        """
        # Store tools for ReAct-style execution
        self._bound_tools = tools
        print(f"[DEBUG] bind_tools called with {len(tools)} tools - stored for ReAct execution")
        return self

    def select_tool_for_query(self, query: str) -> Optional[str]:
        """
        Select appropriate tool for query using planner or dynamic tool name matching.
        Works with tool names instead of pre-instantiated tool objects.
        """
        # Use planner if available
        if self._planner:
            try:
                # Use semantic tool filtering to select best tool
                from spatialreason.plan.tools.tool_models import Tool, ToolkitList

                # Create simple tools for semantic filtering - use all 3 toolkits to prevent ID range errors
                toolkit_list = ToolkitList(3)
                all_tools = []
                for toolkit in toolkit_list.tool_kits:
                    all_tools.extend(toolkit.tool_lists)

                # Apply semantic filtering
                filtered_tools = self._planner.semantic_filter.filter_tools_by_relevance(
                    query=query,
                    available_tools=all_tools,
                    top_k=1
                )

                if filtered_tools:
                    # Map to actual tool name (use the keys from agent factory)
                    tool_name_mapping = {
                        "segmentation": "segmentation",
                        "detection": "detection",
                        "classification": "classification",
                        "buffer": "buffer",
                        "overlap": "overlap",
                        "containment": "containment"
                    }
                    selected_tool_name = filtered_tools[0].api_dest["name"]
                    return tool_name_mapping.get(selected_tool_name)

            except Exception as e:
                print(f"[ERROR] Semantic tool selection failed: {e}")
                print(f"[ERROR] Research integrity requires semantic-only tool selection")
                raise RuntimeError(f"Semantic tool selection failure compromises research validity: {e}")

        # RESEARCH INTEGRITY: No fallback mechanisms allowed
        print(f"[ERROR] No planner available for semantic tool selection")
        raise RuntimeError("Semantic tool selection requires planner with semantic filtering capabilities")

    def use_planner_for_query(self, query: str, image_path: str = None) -> Optional[str]:
        """
        Use the planner for full planning and execution instead of single tool selection.

        Args:
            query: User query
            image_path: Optional image path

        Returns:
            Planner result or None if planner not available
        """
        if not self._planner:
            return None

        try:
            # Use the planner's plan_and_execute method
            result = self._planner.plan_and_execute(query, image_path)
            return result
        except Exception as e:
            print(f"[DEBUG] Planner execution failed: {e}")
            return None

    def _normalize_messages(self, messages: List[Any]) -> tuple[str, Any]:
        """Extract text and image from messages."""
        last = messages[-1]

        # Extract text
        if isinstance(last, dict) and "text" in last:
            text = last["text"]
        elif isinstance(last, dict) and "content" in last:
            text = last["content"]
        elif hasattr(last, 'content'):
            text = last.content
        else:
            text = str(last)

        # Extract image
        if isinstance(last, dict):
            image = last.get('additional_kwargs', {}).get('image')
        else:
            image = getattr(last, 'additional_kwargs', {}).get('image')

        return text, image

    def __call__(self, messages: List[Any], image: Optional[Any] = None,
                 max_steps: Optional[int] = None, num_passes: Optional[int] = None, **kwargs) -> LLMResult:
        """
        Main entry point for Chain-of-Thought spatial analysis.

        Args:
            messages: Input messages
            image: Optional image data
            max_steps: Maximum reasoning steps per pass (overrides default)
            num_passes: Number of reasoning attempts (overrides default)
        """
        # Extract user text and image (single normalization call)
        user_text, extracted_image = self._normalize_messages(messages)
        # Use provided image if available, otherwise use extracted image
        if image is None:
            image = extracted_image

        # Keep track of original image path for tool arguments
        original_image_path = None
        if image is not None and isinstance(image, str):
            original_image_path = image
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                # Failed to load image, continue without it
                image = None

        try:
            # Check if we should use ReAct-style tool calling (planner disabled + tools bound)
            if self._planner is None and self._bound_tools:
                # ReAct mode: Generate tool calls for the agent workflow
                print("[DEBUG] ReAct mode: Generating tool calls for agent workflow")
                tool_call_msg = self._generate_react_tool_calls(user_text, original_image_path, messages)
                return LLMResult(generations=[[ChatGeneration(message=tool_call_msg)]])

            # Otherwise use Chain-of-Thought reasoning (with planner if available)
            resp = self._execute_cot_reasoning(user_text, original_image_path, max_steps, num_passes)
        except Exception as e:
            # Enhanced error reporting for debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Enhanced Chat Model Error: {str(e)}")
            print(f"[ERROR] Full traceback:\n{error_details}")

            resp = Response({
                "content": f"I encountered an error while processing your request: {str(e)}. Please check the console for detailed error information.",
                "tool_calls": []
            })

        # Convert Response to LLMResult
        if isinstance(resp, str):
            msg = AIMessage(content=resp)
            return LLMResult(generations=[[ChatGeneration(message=msg)]])

        # Always create simple AIMessage with just content (avoid tool_calls format issues)
        content = resp.get("content", "") if hasattr(resp, 'get') else str(resp)
        msg = AIMessage(content=content)

        return LLMResult(generations=[[ChatGeneration(message=msg)]])

    def invoke(self, messages: List[Any], **kwargs) -> AIMessage:
        """
        Invoke method for Agent compatibility - returns AIMessage directly.

        Args:
            messages: Input messages
            **kwargs: Additional arguments

        Returns:
            AIMessage: The response message for the agent workflow
        """
        # Call our main method to get LLMResult
        llm_result = self.__call__(messages, **kwargs)

        # Extract the AIMessage from the LLMResult
        if llm_result.generations and llm_result.generations[0]:
            return llm_result.generations[0][0].message
        else:
            # Fallback if something goes wrong
            return AIMessage(content="I encountered an error processing your request.")

    def _execute_cot_reasoning(self, user_text: str, image_path: Optional[str],
                              max_steps: Optional[int] = None, num_passes: Optional[int] = None) -> Response:
        """
        Execute Chain-of-Thought reasoning following the provided pseudocode algorithm.

        Args:
            user_text: Original user query
            image_path: Path to the image being analyzed
            max_steps: Maximum reasoning steps per pass (overrides default)
            num_passes: Number of reasoning attempts (overrides default)

        Returns:
            Response with final reasoning result
        """
        try:
            # Try planner-based execution first if available
            if self._planner:
                try:
                    planner_result = self.use_planner_for_query(user_text, image_path)
                    if planner_result:
                        return Response({
                            "content": planner_result,
                            "tool_calls": [{
                                "id": str(uuid.uuid4()),
                                "name": "planner_execution",
                                "args": {"query": user_text, "image_path": image_path},
                                "result": planner_result
                            }]
                        })
                except Exception as e:
                    print(f"[DEBUG] Planner execution failed, falling back to CoT: {e}")

            # Fallback to CoT reasoning
            # Import CoT components
            from spatialreason.cot.spatial_env import SpatialReasonEnvironment

            # Initialize environment with dynamically loaded tools
            tools_dict = {}
            for tool_name in self._get_available_tool_names():
                try:
                    tool_instance = self._load_tool(tool_name)
                    tools_dict[tool_name] = tool_instance
                except (ValueError, ImportError) as e:
                    # Skip tools that fail to load
                    continue

            spatial_env = SpatialReasonEnvironment(tools_dict, image_path)
            spatial_env.input_description = user_text

            # Determine parameters (use passed values or class defaults)
            actual_max_steps = max_steps if max_steps is not None else self.DEFAULT_MAX_STEPS
            actual_num_passes = num_passes if num_passes is not None else self.DEFAULT_NUM_PASSES

            # Execute multi-pass reasoning as per pseudocode
            for pass_num in range(actual_num_passes):

                # Initialize tree with query and image data
                from spatialreason.cot.simple_tree import SimpleTree, memory_efficient_copy
                tree = SimpleTree()
                tree.root.node_type = "Action Input"
                # CRITICAL FIX: Use memory-efficient copying to prevent CUDA OOM
                tree.root.io_state = memory_efficient_copy(spatial_env)

                # Execute reasoning chain
                terminal_node = self._execute_reasoning_chain(tree.root, actual_max_steps)

                # Check if reasoning was successful
                if terminal_node and terminal_node.io_state.check_success() == 1:
                    final_answer = terminal_node.io_state.final_result

                    return Response({
                        "content": final_answer,
                        "tool_calls": [{
                            "id": str(uuid.uuid4()),
                            "name": "cot_reasoning",
                            "args": {"query": user_text, "image_path": image_path},
                            "result": final_answer
                        }]
                    })

            # If all passes failed
            return Response({
                "content": "I wasn't able to complete the analysis after multiple reasoning attempts. Could you please rephrase your request or try a more specific query?",
                "tool_calls": []
            })

        except Exception as e:
            # Fallback to direct tool execution using unified method
            return self._execute_single_tool(user_text, image_path)

    def _execute_reasoning_chain(self, current_node, max_steps: int):
        """
        Execute the reasoning chain following the pseudocode algorithm.

        Args:
            current_node: Current tree node
            max_steps: Maximum reasoning steps

        Returns:
            Terminal node with final result
        """
        # Initialize system prompts in current_node.messages
        if not current_node.messages:
            # Use unified prompt construction method
            system_prompt, user_prompt = self._create_cot_prompt(current_node.io_state.input_description)

            current_node.messages.append({"role": "system", "content": system_prompt})
            current_node.messages.append({"role": "user", "content": user_prompt})

        step_count = 0
        while step_count < max_steps:
            # Call LLM with current messages
            llm_response = self._generate_cot_response(current_node.messages)

            # Parse response for thoughts and actions
            if self._contains_thought(llm_response):
                # Create new "Thought" node
                thought_content = self._extract_thought(llm_response)
                new_node = self._create_thought_node(current_node, thought_content)
                current_node = new_node

            if self._contains_function_call(llm_response):
                # Create new "Action" node
                action_name, action_args = self._extract_function_call(llm_response)
                action_node = self._create_action_node(current_node, action_name)

                # Create "Action Input" node and execute
                input_node = self._create_action_input_node(action_node, action_args)
                observation, status = input_node.io_state.step(action_name, action_args)
                input_node.observation = observation
                input_node.observation_code = status

                current_node = input_node

                # Add function observation to messages
                current_node.messages.append({
                    "role": "function",
                    "name": action_name,
                    "content": observation
                })

            # Check termination conditions (following pseudocode)
            if (current_node.get_depth() >= max_steps or
                current_node.is_terminal or
                current_node.io_state.check_success() == 1):
                return current_node

            step_count += 1

        return current_node


    def _generate(self, messages: List[ChatMessage], **kwargs) -> LLMResult:
        """Generate method required by BaseChatModel."""
        return self.__call__(messages, **kwargs)

    async def _agenerate(self, messages: List[ChatMessage], **kwargs) -> LLMResult:
        return self.__call__(messages, **kwargs)


# Alias for backward compatibility
EnhancedChatModel200 = EnhancedChatModel
