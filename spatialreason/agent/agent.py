import json
import operator
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import SceneContextAnalyzer for scene-enhanced answer synthesis
try:
    from create_data.scene_context_analyzer import SceneContextAnalyzer
    SCENE_CONTEXT_AVAILABLE = True
except ImportError:
    SCENE_CONTEXT_AVAILABLE = False
    print("⚠️ SceneContextAnalyzer not available - scene context enhancement disabled")

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests using multi-step planning
    and execution following the algorithmic pattern with semantic tool retrieval.

    Workflow: process → plan → execute_steps → synthesize → END
    Behavior: One user query = Multi-step plan = Multiple tool executions = Synthesized result

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
        planner (Any): Multi-step planner for algorithmic execution.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
        max_turn: int = 10,
    ):
        """
        Initialize the Agent with multi-step planning capabilities.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
            max_turn (int, optional): Maximum number of iterations for ReAct-style execution. Defaults to 10.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools
        self.max_turn = max_turn

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Extract planner from the enhanced chat model if available
        self.planner = getattr(model, '_planner', None) if hasattr(model, '_planner') else None

        # Storage for perception results (for ReAct-style execution)
        # This stores results from perception tools to provide to spatial relation tools
        self.perception_results = {}
        self.current_image_path = None
        self.current_query = None  # Store current user query for text_prompt fallback

        # Track recent tool calls to detect redundant operations
        self.recent_tool_calls = []  # List of (tool_name, args_hash) tuples

        # Track tool execution steps for answer synthesis (matches generate_gt_all.py structure)
        self.execution_steps = []  # List of {tool, arguments, output} dicts
        self.tool_sequence = []  # List of tool names in execution order

        # Initialize SceneContextAnalyzer for scene-enhanced answer synthesis
        self.scene_analyzer = None
        if SCENE_CONTEXT_AVAILABLE:
            try:
                self.scene_analyzer = SceneContextAnalyzer(
                    config_path="benchmark/scene_context_config.yaml"
                )
                print("✅ SceneContextAnalyzer initialized for answer synthesis")
            except Exception as e:
                print(f"⚠️ Failed to initialize SceneContextAnalyzer: {e}")

        # Define the agent workflow - MULTI-STEP EXECUTION WITH PLANNING OR REACT-STYLE ITERATION
        # With planner: process → plan_and_execute → END
        # Without planner (ablation): process → execute_tools → process → ... → END (ReAct-style)
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("plan_and_execute", self.plan_and_execute_steps)
        workflow.add_node("execute_tools", self.execute_tools)  # NEW: For ReAct-style execution

        # Unified routing: planner → tool execution → end
        workflow.add_conditional_edges(
            "process",
            self.route_after_process,
            {
                "plan_and_execute": "plan_and_execute",
                "execute_tools": "execute_tools",
                END: END
            }
        )

        # Loop back to process after tool execution (ReAct-style iterative reasoning)
        workflow.add_edge("execute_tools", "process")

        # Planner execution ends workflow (one-shot planning)
        workflow.add_edge("plan_and_execute", END)
        workflow.set_entry_point("process")

        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        self._original_model = model  # Keep reference to original model for scratchpad access

    def reset_for_new_query(self):
        """
        Reset agent state for a new query.

        CRITICAL: This prevents cross-query perception_results pollution.
        Without this reset, perception results from the previous sample carry over,
        causing geometry extraction to fail when the new query asks about different classes.
        """
        self.perception_results = {}
        self.current_image_path = None
        self.current_query = None
        self.recent_tool_calls = []
        # Reset execution tracking for answer synthesis
        self.execution_steps = []
        self.tool_sequence = []
        print("🔄 Reset agent state for new query (perception_results, image_path, tool_calls, execution_steps)")

    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """
        messages = state["messages"]

        # CRITICAL FIX: Detect if this is a new query (first message in workflow)
        # If messages only contains the initial user query (no tool calls/responses yet),
        # reset agent state to prevent cross-query pollution
        is_new_query = len(messages) == 1 and hasattr(messages[0], 'type') and messages[0].type == 'human'
        if is_new_query:
            self.reset_for_new_query()

        # Extract and store query and image path from user query (for ReAct-style execution)
        # Now we always extract since we reset on new query
        user_query, image_path = self._extract_query_and_image(messages)
        if user_query and user_query != self.current_query:
            self.current_query = user_query
            print(f"📝 Stored user query: {user_query[:100]}...")
        if image_path and image_path != self.current_image_path:
            self.current_image_path = image_path
            print(f"📁 Stored image path: {image_path}")

        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def route_after_process(self, state: AgentState) -> str:
        """
        Route to planner, tool execution, or end based on agent state.

        This is the unified routing function that determines the next step:
        1. Use planner if available (multi-step planning)
        2. Execute tools if model requested them (ReAct-style)
        3. End if no actions needed
        4. NEW: Force answer synthesis if sufficient tools have been executed

        Args:
            state (AgentState): Current agent state

        Returns:
            str: Next node to execute ("plan_and_execute", "execute_tools", or END)
        """
        # Check iteration limit to prevent infinite loops
        tool_count = sum(1 for msg in state["messages"]
                        if hasattr(msg, 'type') and msg.type == 'tool')

        if tool_count >= self.max_turn:
            print(f"⚠️ Max iterations ({self.max_turn}) reached, ending workflow")

            # CRITICAL FIX: Add a final message explaining the failure
            # Extract the original query
            user_query = None
            for msg in state["messages"]:
                if hasattr(msg, 'type') and msg.type == 'human':
                    user_query = msg.content
                    break

            # Create a proper failure message
            failure_msg = (
                f"I was unable to complete the analysis after {self.max_turn} iterations. "
                f"The spatial reasoning task encountered repeated tool execution issues. "
                f"Original query: {user_query if user_query else 'Unknown'}"
            )

            # Add the failure message to the state
            from langchain_core.messages import SystemMessage
            state["messages"].append(SystemMessage(content=failure_msg))

            return END

        # NOTE: Forced synthesis workaround removed in favor of proper ReAct termination.
        # With improved tool outputs (objects_within_buffer counts, actionable summaries),
        # the LLM should naturally decide to stop calling tools and provide answers.
        # See REACT_IMPLEMENTATION_IMPROVEMENTS.md for details.

        # Route based on planner availability and tool calls
        if self.should_use_planner(state):
            return "plan_and_execute"
        elif self.has_tool_calls(state):
            return "execute_tools"
        else:
            return END

    # NOTE: _should_force_synthesis, _synthesize_and_add_answer, and _compute_answer_from_results
    # have been removed as part of the proper ReAct implementation.
    # The LLM should now naturally terminate when it receives actionable tool results.
    # See REACT_IMPLEMENTATION_IMPROVEMENTS.md for details on the architectural changes.

    def should_use_planner(self, state: AgentState) -> bool:
        """
        Determine if the planner should be used for multi-step execution.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if planner is available and should be used, False otherwise.
        """
        # Use planner if available and the query seems complex enough
        if not self.planner:
            return False

        # Extract user query from messages
        messages = state["messages"]
        if not messages:
            return False

        # Get the last user message
        last_message = messages[-1]
        user_text = ""
        if hasattr(last_message, 'content'):
            user_text = last_message.content or ""
        elif isinstance(last_message, dict):
            user_text = last_message.get('content', '')

        # Use planner for any non-empty query (let the planner decide complexity)
        return bool(user_text.strip())

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls (legacy method for compatibility).

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(getattr(response, 'tool_calls', [])) > 0

    def plan_and_execute_steps(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Execute multi-step planning and execution using the algorithmic pattern.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing execution results.
        """
        messages = state["messages"]

        # Extract user query and image path from messages
        user_query, image_path = self._extract_query_and_image(messages)

        if not user_query:
            return {"messages": [SystemMessage(content="No valid query found.")]}

        try:
            print(f"🚀 Starting multi-step planning and execution for query: '{user_query[:50]}...'")

            # Use the planner's plan_and_execute method (algorithm-compliant)
            final_result = self.planner.plan_and_execute(user_query, image_path)

            # Create response message
            response_message = SystemMessage(content=final_result)

            # Log the multi-step execution
            if self.log_tools:
                self._log_multi_step_execution(user_query, image_path, final_result)

            print("✅ Multi-step execution completed successfully")
            return {"messages": [response_message]}

        except Exception as e:
            print(f"❌ Multi-step execution failed: {e}")
            error_message = f"Multi-step planning failed: {str(e)}. Falling back to direct processing."
            return {"messages": [SystemMessage(content=error_message)]}

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response (ReAct-style execution).

        This method:
        1. Adapts tool arguments (converts simplified params to full schema)
        2. Executes tools
        3. Stores perception results for use by spatial tools
        4. Returns tool results to the conversation

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = getattr(state["messages"][-1], 'tool_calls', [])
        results = []

        for call in tool_calls:
            print(f"🔧 Executing tool: {call['name']}")
            print(f"📝 Original args: {call['args']}")

            # CRITICAL FIX: Detect redundant tool calls
            redundancy_warning = self._check_redundant_tool_call(call["name"], call["args"])

            if call["name"] not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                try:
                    # CRITICAL: Adapt tool arguments before execution
                    # Convert simplified parameters (buffer_class, geometry_count) to full schema (perception_results, classes_used)
                    adapted_args = self._adapt_tool_arguments(call["name"], call["args"])
                    print(f"📝 Adapted args: {adapted_args}")

                    # Execute tool with adapted arguments
                    tool = self.tools[call["name"]]

                    # CRITICAL FIX: Handle Pydantic BaseModel tools that expect tool_input parameter
                    # Spatial relation tools (buffer, overlap, containment) use Pydantic models with tool_input
                    import inspect
                    run_signature = inspect.signature(tool._run)
                    valid_params = set(run_signature.parameters.keys()) - {'self'}
                    print(f"🔍 Valid params for {call['name']}: {valid_params}")
                    print(f"🔍 Adapted args keys: {set(adapted_args.keys())}")

                    # Check if tool expects a single 'tool_input' parameter (Pydantic BaseModel)
                    if 'tool_input' in valid_params and len(valid_params) == 1:
                        # Spatial relation/statistics tools: wrap all args into tool_input dict
                        print(f"🔧 Tool expects 'tool_input' parameter - wrapping all args")
                        result = tool._run(tool_input=adapted_args)
                    elif call["name"] in ["segmentation", "detection", "classification", "change_detection",
                                       "sar_detection", "sar_classification", "infrared_detection"]:
                        # Perception tools: pass args as kwargs, filter to valid params
                        filtered_args = {k: v for k, v in adapted_args.items() if k in valid_params}
                        print(f"📝 Filtered args (kept only valid params): {filtered_args}")
                        result = tool._run(**filtered_args)
                    else:
                        # Fallback: try to pass as dict if not perception tool
                        print(f"⚠️ Unknown tool type, attempting to pass as dict")
                        result = tool._run(adapted_args)

                    # Store perception results for use by spatial tools
                    if call["name"] in ["segmentation", "detection", "classification"]:
                        self._store_perception_result(call["name"], result)

                    # CRITICAL: Track execution step for answer synthesis
                    # This matches generate_gt_all.py's structure for scene context detection
                    execution_step = {
                        "tool": call["name"],
                        "arguments": adapted_args,
                        "output": self._parse_tool_result_for_tracking(result)
                    }
                    self.execution_steps.append(execution_step)
                    if call["name"] not in self.tool_sequence:
                        self.tool_sequence.append(call["name"])
                    print(f"📊 Tracked execution step: {call['name']}")

                    # CRITICAL FIX: Detect tool failures and provide explicit error messages
                    failure_detected = self._detect_tool_failure(call["name"], result)
                    if failure_detected:
                        error_msg = failure_detected
                        print(f"⚠️ Tool failure detected: {error_msg}")
                        # Append error message to result so LLM can adjust strategy
                        result = f"{result}\n\nERROR: {error_msg}\nPlease try a different approach or verify the input parameters."

                    # Add redundancy warning if detected
                    if redundancy_warning:
                        result = f"{result}\n\nWARNING: {redundancy_warning}"

                except Exception as e:
                    print(f"❌ Tool execution failed: {e}")
                    import traceback
                    traceback.print_exc()
                    result = f"Tool execution failed: {str(e)}\nPlease verify the input parameters and try again."

            tool_message = ToolMessage(
                tool_call_id=call["id"],
                name=call["name"],
                args=call["args"],  # Keep original args for logging
                content=str(result),
            )
            results.append(tool_message)

        self._save_tool_calls(results)
        print("✅ Tool execution complete, returning to model processing!")

        return {"messages": results}

    def _normalize_class_name(self, class_name: str) -> str:
        """
        Normalize class names to handle singular/plural and formatting variations.

        This mirrors the logic in generate_gt_all.py lines 1441-1451.
        """
        if not class_name:
            return ""
        normalized = class_name.lower().strip().replace(" ", "_")

        # Handle common plural forms (but preserve words that end in 's' naturally)
        if normalized.endswith('s') and normalized not in ['barren', 'water', 'grass', 'class']:
            singular = normalized[:-1]
            return singular

        return normalized

    def _find_matching_class(self, requested_class: str, available_classes: list) -> str:
        """
        Find a matching class from available classes using fuzzy matching.

        This mirrors the logic in generate_gt_all.py lines 1453-1490 and
        uses CLASS_ALIASES similar to spatialops.py lines 568-582.

        Args:
            requested_class: Class name requested by LLM (e.g., "water_bodies")
            available_classes: Classes available in perception_results (e.g., ["water", "forest"])

        Returns:
            Matched class name from available_classes, or empty string if no match
        """
        if not requested_class or not available_classes:
            return ""

        # Common class name aliases for semantic variations
        # This is critical for handling LLM-generated class names that differ from perception output
        CLASS_ALIASES = {
            # Water variations
            'water_bodies': 'water',
            'water_body': 'water',
            'waterbody': 'water',
            'waterbodies': 'water',
            'lake': 'water',
            'river': 'water',
            'pond': 'water',
            # Tree/forest variations
            'tree_canopy': 'trees',
            'tree_canopies': 'trees',
            'tree': 'trees',
            'forest_area': 'forest',
            'forest_areas': 'forest',
            'forests': 'forest',
            'woodland': 'forest',
            # Agriculture variations
            'agriculture_parcel': 'agriculture',
            'agriculture_parcels': 'agriculture',
            'agricultural': 'agriculture',
            'agricultural_area': 'agriculture',
            'agricultural_areas': 'agriculture',
            'farmland': 'agriculture',
            'cropland': 'agriculture',
            'crop': 'agriculture',
            # Building variations
            'building_footprint': 'building',
            'building_footprints': 'buildings',
            'structure': 'building',
            'structures': 'buildings',
            # Vehicle variations
            'car_parking': 'car',
            'parked_cars': 'cars',
            'parked_car': 'car',
            'vehicle': 'car',
            'vehicles': 'cars',
            # Road variations
            'road_network': 'road',
            'road_networks': 'roads',
            'street': 'road',
            'streets': 'roads',
            # Vegetation variations
            'low_vegetation': 'low_vegetation',
            'vegetation': 'low_vegetation',
            'grass': 'low_vegetation',
        }

        requested_norm = self._normalize_class_name(requested_class)

        # 1. Try exact match after normalization
        for available_class in available_classes:
            available_norm = self._normalize_class_name(available_class)
            if requested_norm == available_norm:
                return available_class

        # 2. Try alias mapping
        if requested_norm in CLASS_ALIASES:
            alias_target = CLASS_ALIASES[requested_norm]
            for available_class in available_classes:
                available_norm = self._normalize_class_name(available_class)
                if available_norm == alias_target or available_norm == alias_target + 's':
                    print(f"🔍 [Fuzzy Match] Alias: '{requested_class}' → '{alias_target}' → '{available_class}'")
                    return available_class

        # 3. Try substring matching (bidirectional)
        for available_class in available_classes:
            available_lower = available_class.lower()
            # Check if requested contains available or vice versa
            if available_lower in requested_norm or requested_norm in available_lower:
                print(f"🔍 [Fuzzy Match] Substring: '{requested_class}' ↔ '{available_class}'")
                return available_class

        # 4. Try keyword matching for common patterns
        keywords_to_classes = {
            'tree': ['trees', 'tree', 'forest'],
            'car': ['cars', 'car', 'vehicle'],
            'water': ['water', 'lake', 'river'],
            'building': ['buildings', 'building', 'structure'],
            'road': ['roads', 'road', 'street'],
            'vegetation': ['low_vegetation', 'vegetation', 'grass'],
            'forest': ['forest', 'trees', 'woodland'],
            'agriculture': ['agriculture', 'farmland', 'crop'],
        }

        for keyword, targets in keywords_to_classes.items():
            if keyword in requested_norm:
                for available_class in available_classes:
                    available_lower = available_class.lower()
                    for target in targets:
                        if target in available_lower or available_lower in target:
                            print(f"🔍 [Fuzzy Match] Keyword '{keyword}': '{requested_class}' → '{available_class}'")
                            return available_class

        print(f"⚠️ Could not match '{requested_class}' with any available classes: {available_classes}")
        return ""

    def _adapt_tool_arguments(self, tool_name: str, tool_args: dict) -> dict:
        """
        Adapt tool arguments from simplified format to full schema.

        This method converts the simplified parameters that the LLM generates
        (e.g., buffer_class, geometry_count) to the full schema that the tools expect
        (e.g., perception_results, classes_used).

        This is similar to what the planner does in plan.py lines 2572-2602.

        Args:
            tool_name: Name of the tool
            tool_args: Original tool arguments from LLM

        Returns:
            dict: Adapted tool arguments ready for tool execution
        """
        adapted_args = tool_args.copy()

        # Perception tools: map common parameter name variations
        # EXTENDED: Include all perception tools (optical, SAR, IR)
        perception_tools = ["segmentation", "detection", "classification",
                           "sar_detection", "sar_classification", "infrared_detection"]
        if tool_name in perception_tools:
            # Map common variations of text_prompt
            # CRITICAL: class_names is commonly used by LLMs like Qwen2.5 instead of text_prompt
            text_prompt_aliases = ["class_names", "object_type", "class", "classes", "classes_of_interest",
                                  "classes_requested", "target", "objects", "model", "sensor_type",
                                  "query", "detection_query", "detection_target"]
            for alias in text_prompt_aliases:
                if alias in adapted_args and "text_prompt" not in adapted_args:
                    value = adapted_args.pop(alias)
                    # Handle list values (e.g., classes_of_interest: ['tree', 'car'])
                    if isinstance(value, list):
                        value = ", ".join(value)
                    adapted_args["text_prompt"] = value
                    print(f"🔧 Mapped {alias} -> text_prompt: {value}")

            # BUG 5 FIX: If text_prompt is still missing after alias checking, extract from user query
            if "text_prompt" not in adapted_args:
                if self.current_query:
                    # Extract text_prompt from the user query
                    extracted_prompt = self._extract_text_prompt_from_query(self.current_query)
                    if extracted_prompt:
                        adapted_args["text_prompt"] = extracted_prompt
                        print(f"🔧 Extracted text_prompt from user query: {extracted_prompt}")
                    else:
                        # Fallback: use the full query as text_prompt
                        adapted_args["text_prompt"] = self.current_query
                        print(f"🔧 Using full query as text_prompt fallback: {self.current_query[:100]}...")
                else:
                    print(f"⚠️ No text_prompt available and no current_query stored for {tool_name}")

            # Fix image_path if it's invalid or missing
            if "image_path" in adapted_args:
                # Check if the image path is valid (exists or is a proper path)
                import os
                if not os.path.exists(adapted_args["image_path"]):
                    # LLM generated an invalid path, replace with actual image path
                    if self.current_image_path:
                        print(f"🔧 Replacing invalid image_path '{adapted_args['image_path']}' with actual path: {self.current_image_path}")
                        adapted_args["image_path"] = self.current_image_path
            elif self.current_image_path:
                # image_path is missing, add it
                adapted_args["image_path"] = self.current_image_path
                print(f"🔧 Added missing image_path: {self.current_image_path}")

        # ALL spatial tools (relations + statistics) need perception_results and classes_used
        # This includes: buffer, overlap, containment, distance, area_measurement, object_count_aoi
        spatial_tools = ["buffer", "overlap", "containment", "distance", "area_measurement", "object_count_aoi"]
        if tool_name in spatial_tools:
            # Provide stored perception results (for spatial relation tools)
            if tool_name in ["buffer", "overlap", "containment"]:
                if self.perception_results:
                    adapted_args["perception_results"] = self.perception_results
                    print(f"🔧 Providing stored perception results to {tool_name} tool")
                else:
                    print(f"⚠️ No stored perception results available for {tool_name} tool")

            # CRITICAL FIX: Extract ALL classes from perception_results
            # All spatial tools need ALL detected classes to properly analyze spatial relationships
            # e.g., for "cars within 7m of trees", we need both 'tree' (buffer source) and 'car' (target)
            all_detected_classes = set()
            if self.perception_results:
                for tool_result in self.perception_results.values():
                    if isinstance(tool_result, dict):
                        # Extract from segments (segmentation tool)
                        if 'segments' in tool_result:
                            for seg in tool_result['segments']:
                                if 'class' in seg:
                                    all_detected_classes.add(seg['class'].lower())
                        # Extract from detections (detection tool)
                        if 'detections' in tool_result:
                            for det in tool_result['detections']:
                                if 'class' in det:
                                    all_detected_classes.add(det['class'].lower())
                        # Extract from classes_detected
                        if 'classes_detected' in tool_result:
                            for cls in tool_result['classes_detected']:
                                all_detected_classes.add(cls.lower())
                        # Extract from classes_requested
                        if 'classes_requested' in tool_result:
                            for cls in tool_result['classes_requested']:
                                all_detected_classes.add(cls.lower())

            # Convert buffer_class to classes_used, but include ALL detected classes
            if "buffer_class" in adapted_args:
                buffer_class = adapted_args.pop("buffer_class").lower()
                # Include buffer_class first, then all other detected classes
                classes_used = [buffer_class]
                for cls in all_detected_classes:
                    if cls != buffer_class:
                        classes_used.append(cls)
                adapted_args["classes_used"] = classes_used
                print(f"🔧 Converted buffer_class to classes_used (including all detected): {adapted_args['classes_used']}")
            elif all_detected_classes and "classes_used" not in adapted_args:
                # No buffer_class specified, use all detected classes
                adapted_args["classes_used"] = list(all_detected_classes)
                print(f"🔧 Auto-populated classes_used from perception results: {adapted_args['classes_used']}")

            # Ensure image_path is provided
            if self.current_image_path and "image_path" not in adapted_args:
                adapted_args["image_path"] = self.current_image_path
                print(f"🔧 Added image_path: {self.current_image_path}")

            # CRITICAL FIX: Spatial tools need actual polygon coordinates, not class names
            # The LLM sends class_a, class_b, from_class, to_class, etc. but tools expect set_a, set_b, polygons, etc.
            # This includes overlap and containment which also need polygon extraction
            if tool_name in ["distance", "area_measurement", "object_count_aoi", "overlap", "containment"]:
                adapted_args = self._adapt_spatial_statistics_args(tool_name, adapted_args)

        return adapted_args

    def _adapt_spatial_statistics_args(self, tool_name: str, adapted_args: dict) -> dict:
        """
        Adapt arguments for spatial statistics tools (distance, area_measurement, object_count_aoi).

        These tools expect actual polygon coordinates, but the LLM sends class names.
        This method extracts geometries from perception_results and maps them to the correct format.

        Args:
            tool_name: Name of the spatial statistics tool
            adapted_args: Arguments after initial adaptation

        Returns:
            dict: Arguments with polygon coordinates extracted from perception_results
        """
        from spatialreason.tools.spatialops import preprocess_all_geometries_for_spatial_relations

        if not self.perception_results:
            print(f"⚠️ No perception results available for {tool_name} geometry extraction")
            return adapted_args

        # Extract class names from LLM input (various parameter name patterns)
        class_a = None
        class_b = None

        # CRITICAL FIX: Include all possible aliases - LLM sends various parameter names
        # Also include set_a/set_b when they're sent as strings (class names) instead of polygon coordinates
        # Added containment-specific aliases: container_class, containers_class, contained_class
        # Added overlap-specific aliases: reference_layer, overlay_layer, base_class, overlay_class
        # The LLM uses different parameter name variations depending on context
        class_a_aliases = ["class_a", "from_class", "class_1", "source_class", "set_a_class", "buffer_class", "set_a",
                          "container_class", "containers_class", "reference_layer", "base_class",
                          "object_class", "aoi_class_a"]
        class_b_aliases = ["class_b", "to_class", "class_2", "target_class", "set_b_class", "set_b",
                          "contained_class", "overlay_layer", "overlay_class",
                          "aoi_class", "aoi_class_b"]

        # Also check geometry_count - LLM sometimes puts class name there by mistake
        if "geometry_count" in adapted_args:
            gc = adapted_args.get("geometry_count")
            if isinstance(gc, str) and not gc.isdigit():
                # geometry_count contains a class name, treat it as class_b
                print(f"🔧 Found class name in geometry_count: {gc}")
                if class_b is None:
                    class_b = gc.lower()
                adapted_args.pop("geometry_count")

        for alias in class_a_aliases:
            if alias in adapted_args:
                value = adapted_args.get(alias)
                # Only extract if it's a string (class name), not a list (polygon coordinates)
                if isinstance(value, str):
                    class_a = adapted_args.pop(alias)
                    class_a = class_a.lower()
                    print(f"🔧 Extracted class_a from '{alias}': {class_a}")
                    break
                elif isinstance(value, list) and len(value) > 0:
                    # Check if it's a list of strings (class names) or list of coordinates
                    if all(isinstance(item, str) for item in value):
                        # List of class names - take first one
                        class_a = value[0].lower()
                        adapted_args.pop(alias)
                        print(f"🔧 Extracted class_a from '{alias}' (list): {class_a}")
                        break
                    else:
                        # Already polygon coordinates - don't extract
                        print(f"🔧 '{alias}' already contains polygon coordinates, skipping extraction")
                        break

        for alias in class_b_aliases:
            if alias in adapted_args:
                value = adapted_args.get(alias)
                # Only extract if it's a string (class name), not a list (polygon coordinates)
                if isinstance(value, str):
                    class_b = adapted_args.pop(alias)
                    class_b = class_b.lower()
                    print(f"🔧 Extracted class_b from '{alias}': {class_b}")
                    break
                elif isinstance(value, list) and len(value) > 0:
                    # Check if it's a list of strings (class names) or list of coordinates
                    if all(isinstance(item, str) for item in value):
                        # List of class names - take first one
                        class_b = value[0].lower()
                        adapted_args.pop(alias)
                        print(f"🔧 Extracted class_b from '{alias}' (list): {class_b}")
                        break
                    else:
                        # Already polygon coordinates - don't extract
                        print(f"🔧 '{alias}' already contains polygon coordinates, skipping extraction")
                        break

        # CRITICAL FIX: Handle meters_per_pixel null values
        # LLM sometimes sends meters_per_pixel: null which causes Pydantic validation errors
        if "meters_per_pixel" in adapted_args:
            mpp = adapted_args["meters_per_pixel"]
            if mpp is None or (isinstance(mpp, str) and mpp.lower() == "null"):
                adapted_args["meters_per_pixel"] = 0.3  # Default GSD
                print(f"🔧 Fixed null meters_per_pixel, using default: 0.3")

        # CRITICAL FIX: Validate that class names are not empty strings
        # LLM sometimes generates empty strings for class parameters, causing validation errors
        if class_a is not None and (not class_a or class_a.strip() == ""):
            print(f"⚠️ Empty source_class for {tool_name}, treating as None")
            class_a = None
        if class_b is not None and (not class_b or class_b.strip() == ""):
            print(f"⚠️ Empty target_class for {tool_name}, treating as None")
            class_b = None

        # Build classes_used list from extracted class names
        classes_to_extract = []
        if class_a:
            classes_to_extract.append(class_a)
        if class_b:
            classes_to_extract.append(class_b)

        # If no classes extracted, use all available classes
        if not classes_to_extract and "classes_used" in adapted_args:
            classes_to_extract = adapted_args.get("classes_used", [])

        if not classes_to_extract:
            print(f"⚠️ No classes specified for {tool_name}, cannot extract geometries")
            return adapted_args

        # CRITICAL: Get available classes from perception_results for fuzzy matching
        available_classes = set()
        for tool_result in self.perception_results.values():
            if isinstance(tool_result, dict):
                # Extract from segments
                if 'segments' in tool_result:
                    for seg in tool_result['segments']:
                        if 'class' in seg:
                            available_classes.add(seg['class'].lower())
                # Extract from detections
                if 'detections' in tool_result:
                    for det in tool_result['detections']:
                        if 'class' in det:
                            available_classes.add(det['class'].lower())
                # Extract from classes_detected
                if 'classes_detected' in tool_result:
                    for cls in tool_result['classes_detected']:
                        available_classes.add(cls.lower())

        available_classes_list = list(available_classes)
        print(f"🔍 Available classes in perception_results: {available_classes_list}")

        # CRITICAL FIX: Use fuzzy matching to map LLM class names to perception classes
        # This handles variations like "water_bodies" → "water", "agriculture_parcel" → "agriculture"
        matched_classes = []
        class_mapping = {}  # Map original LLM class name to matched perception class
        for requested_class in classes_to_extract:
            matched = self._find_matching_class(requested_class, available_classes_list)
            if matched:
                matched_classes.append(matched)
                class_mapping[requested_class] = matched
            else:
                # Keep original class name as fallback (preprocessing might still find it)
                matched_classes.append(requested_class)
                class_mapping[requested_class] = requested_class

        print(f"🔍 Class mapping: {class_mapping}")

        # Update class_a and class_b with matched names for later use
        if class_a and class_a in class_mapping:
            class_a = class_mapping[class_a]
        if class_b and class_b in class_mapping:
            class_b = class_mapping[class_b]

        # Use preprocessing function to extract geometries with matched classes
        preprocessing_result = preprocess_all_geometries_for_spatial_relations(
            self.perception_results, matched_classes
        )

        if not preprocessing_result.get("success", False):
            print(f"⚠️ Geometry preprocessing failed: {preprocessing_result.get('error', 'Unknown error')}")
            return adapted_args

        all_geometries = preprocessing_result.get("all_geometries", {})
        print(f"🔧 Extracted geometries for classes: {list(all_geometries.keys())}")

        # Convert geometry dicts to polygon coordinate lists
        def extract_polygon_coords(geometry_list):
            """Extract polygon coordinates from geometry dictionaries."""
            polygons = []
            for geom in geometry_list:
                # Try various coordinate field names
                coords = None
                if "polygon" in geom:
                    coords = geom["polygon"]
                elif "coordinates" in geom:
                    coords = geom["coordinates"]
                elif "mask_polygon" in geom:
                    coords = geom["mask_polygon"]
                elif "bbox" in geom:
                    # Convert bbox to polygon
                    bbox = geom["bbox"]
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        coords = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]

                if coords:
                    polygons.append(coords)
            return polygons

        # Adapt arguments based on tool type
        if tool_name == "distance":
            # Distance tool expects set_a and set_b (lists of polygons)
            if class_a and class_a in all_geometries:
                adapted_args["set_a"] = extract_polygon_coords(all_geometries[class_a])
                print(f"🔧 Mapped {class_a} to set_a: {len(adapted_args['set_a'])} polygons")
            if class_b and class_b in all_geometries:
                adapted_args["set_b"] = extract_polygon_coords(all_geometries[class_b])
                print(f"🔧 Mapped {class_b} to set_b: {len(adapted_args['set_b'])} polygons")
            # Update classes_used
            adapted_args["classes_used"] = classes_to_extract

        elif tool_name == "area_measurement":
            # Area measurement tool expects polygons (list of all polygons)
            all_polygons = []
            for class_name, geom_list in all_geometries.items():
                all_polygons.extend(extract_polygon_coords(geom_list))
            adapted_args["polygons"] = all_polygons
            adapted_args["classes_used"] = classes_to_extract
            print(f"🔧 Mapped all classes to polygons: {len(all_polygons)} total polygons")

        elif tool_name == "object_count_aoi":
            # Object count AOI tool expects objects and aois
            # Extract object_class and aoi_class from adapted_args or LLM input
            object_class = adapted_args.pop("object_class", None) or class_a
            aoi_class = adapted_args.pop("aoi_class", None) or class_b

            if object_class and object_class in all_geometries:
                adapted_args["objects"] = extract_polygon_coords(all_geometries[object_class])
                print(f"🔧 Mapped {object_class} to objects: {len(adapted_args['objects'])} polygons")
            if aoi_class and aoi_class in all_geometries:
                adapted_args["aois"] = extract_polygon_coords(all_geometries[aoi_class])
                print(f"🔧 Mapped {aoi_class} to aois: {len(adapted_args['aois'])} polygons")
            adapted_args["classes_used"] = classes_to_extract

        elif tool_name == "overlap":
            # Overlap tool expects class_a_polygons and class_b_polygons (lists of polygon coords)
            if class_a and class_a in all_geometries:
                adapted_args["class_a_polygons"] = extract_polygon_coords(all_geometries[class_a])
                adapted_args["class_a_name"] = class_a
                print(f"🔧 Mapped {class_a} to class_a_polygons: {len(adapted_args['class_a_polygons'])} polygons")
            else:
                adapted_args["class_a_polygons"] = []
                print(f"⚠️ No geometries found for class_a: {class_a}")
            if class_b and class_b in all_geometries:
                adapted_args["class_b_polygons"] = extract_polygon_coords(all_geometries[class_b])
                adapted_args["class_b_name"] = class_b
                print(f"🔧 Mapped {class_b} to class_b_polygons: {len(adapted_args['class_b_polygons'])} polygons")
            else:
                adapted_args["class_b_polygons"] = []
                print(f"⚠️ No geometries found for class_b: {class_b}")
            adapted_args["classes_used"] = classes_to_extract

        elif tool_name == "containment":
            # Containment tool expects 'containers' and 'contained' (not container_polygons/contained_polygons)
            # class_a = container class, class_b = contained class
            if class_a and class_a in all_geometries:
                adapted_args["containers"] = extract_polygon_coords(all_geometries[class_a])
                adapted_args["container_class"] = class_a
                print(f"🔧 Mapped {class_a} to containers: {len(adapted_args['containers'])} polygons")
            else:
                adapted_args["containers"] = []
                print(f"⚠️ No geometries found for container class: {class_a}")
            if class_b and class_b in all_geometries:
                adapted_args["contained"] = extract_polygon_coords(all_geometries[class_b])
                adapted_args["contained_class"] = class_b
                print(f"🔧 Mapped {class_b} to contained: {len(adapted_args['contained'])} polygons")
            else:
                adapted_args["contained"] = []
                print(f"⚠️ No geometries found for contained class: {class_b}")
            adapted_args["classes_used"] = classes_to_extract

        return adapted_args

    def _store_perception_result(self, tool_name: str, result: str):
        """
        Store perception tool results for use by spatial relation tools.

        Args:
            tool_name: Name of the perception tool
            result: Tool result (JSON string)
        """
        try:
            # Parse result if it's a JSON string
            if isinstance(result, str) and result.strip().startswith('{'):
                result_dict = json.loads(result)
            else:
                result_dict = {"result": str(result)}

            # Store the result
            self.perception_results[tool_name] = result_dict
            print(f"✅ Stored perception results from {tool_name}")

        except Exception as e:
            print(f"⚠️ Failed to store perception result from {tool_name}: {e}")

    def _detect_tool_failure(self, tool_name: str, result: str) -> str:
        """
        Detect if a tool returned a failure response (zeros, empty arrays, errors).

        Args:
            tool_name: Name of the tool
            result: Tool result (JSON string or error message)

        Returns:
            str: Error message if failure detected, empty string otherwise
        """
        try:
            # Check for explicit error messages
            if "failed" in result.lower() or "error" in result.lower():
                return "Tool returned an error response"

            # Parse JSON result
            if isinstance(result, str) and result.strip().startswith('{'):
                result_dict = json.loads(result)

                # Check for buffer tool failures (zero values)
                if tool_name == "buffer":
                    total_area = result_dict.get("total_buffered_area_sqm", -1)
                    buffer_distance = result_dict.get("buffer_distance_meters", -1)
                    classes_buffered = result_dict.get("classes_buffered", [])

                    if total_area == 0 and buffer_distance == 0 and len(classes_buffered) == 0:
                        return "Buffer tool returned zero values - likely received empty parameters"
                    elif total_area == 0:
                        return "Buffer tool returned zero buffered area - check if objects were detected"

                # Check for overlap tool failures
                elif tool_name == "overlap":
                    if result_dict.get("success") is False:
                        return f"Overlap tool failed: {result_dict.get('error', 'Unknown error')}"

                # Check for containment tool failures
                elif tool_name == "containment":
                    if result_dict.get("success") is False:
                        return f"Containment tool failed: {result_dict.get('error', 'Unknown error')}"

                # Check for perception tool failures (empty detections)
                elif tool_name in ["segmentation", "detection", "classification"]:
                    # Check if no objects were detected
                    if "detections" in result_dict and len(result_dict["detections"]) == 0:
                        return "No objects detected - try adjusting the text prompt or confidence threshold"

        except Exception as e:
            # If we can't parse the result, it might be an error message
            if "exception" in result.lower() or "traceback" in result.lower():
                return "Tool execution encountered an exception"

        return ""  # No failure detected

    def _check_redundant_tool_call(self, tool_name: str, tool_args: dict) -> str:
        """
        Check if this tool call is redundant (same tool with similar args called recently).

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments

        Returns:
            str: Warning message if redundancy detected, empty string otherwise
        """
        import hashlib

        # Create a hash of the tool arguments (simplified version)
        # Only hash key parameters to detect similar calls
        key_params = {}
        if tool_name in ["segmentation", "detection", "classification"]:
            key_params = {
                "text_prompt": tool_args.get("text_prompt", ""),
                "image_path": tool_args.get("image_path", "")
            }
        elif tool_name == "buffer":
            key_params = {
                "buffer_distance_meters": tool_args.get("buffer_distance_meters", 0),
                "buffer_class": tool_args.get("buffer_class", ""),
                "classes_used": str(tool_args.get("classes_used", []))
            }
        else:
            key_params = tool_args

        args_str = str(sorted(key_params.items()))
        args_hash = hashlib.md5(args_str.encode()).hexdigest()

        # Check recent tool calls (last 5)
        recent_calls = [(name, hash_val) for name, hash_val in self.recent_tool_calls[-5:]]

        # Count how many times this exact call was made recently
        same_calls = sum(1 for name, hash_val in recent_calls if name == tool_name and hash_val == args_hash)

        # Add to recent calls
        self.recent_tool_calls.append((tool_name, args_hash))

        # Warn if called more than twice with same parameters
        if same_calls >= 2:
            return (f"This tool ({tool_name}) has been called {same_calls + 1} times with similar parameters. "
                   f"Consider trying a different approach or different parameters.")

        return ""

    def _generate_realistic_thought(self, tool_name: str, tool_args: dict,
                                    original_query: str, conversation_history: List[AnyMessage]) -> str:
        """
        Generate realistic thought using LLM based on conversation context.

        This method generates context-aware thoughts similar to how the planner does it,
        using the LLM to create reasoning text that explains why a tool is being used.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments for the tool
            original_query: Original user query
            conversation_history: Previous messages in the conversation

        Returns:
            str: Context-aware thought text
        """
        import json

        # Extract previous tool results for context
        previous_observations = []
        for msg in conversation_history:
            if hasattr(msg, 'type') and msg.type == 'tool':
                tool_name_prev = getattr(msg, 'name', 'unknown')
                content_preview = str(getattr(msg, 'content', ''))[:100]
                previous_observations.append(f"- {tool_name_prev}: {content_preview}...")

        context = "\n".join(previous_observations) if previous_observations else "No previous observations"
        step_number = len([m for m in conversation_history if hasattr(m, 'type') and m.type == 'tool']) + 1

        prompt = f"""You are a remote sensing expert performing iterative spatial analysis using the ReAct framework.

Original Query: {original_query}

Previous Observations:
{context}

Next Tool to Execute (Step {step_number}): {tool_name}
Tool Arguments: {json.dumps(tool_args, indent=2)}

Generate a brief thought (1-2 sentences) explaining WHY you are using this specific tool with these parameters at this step.
Reference the original query and any previous observations if relevant. Keep it concise and professional.
Return ONLY the thought text, no quotes or explanations."""

        # Use the model to generate thought
        try:
            # Access the model manager from the original model
            if hasattr(self._original_model, '_model_manager') and self._original_model._model_manager:
                thought = self._original_model._model_manager.generate_chat_response(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                if thought and thought.strip():
                    print(f"✅ Generated LLM-based thought for {tool_name}: {thought[:80]}...")
                    return thought.strip()
        except Exception as e:
            print(f"⚠️ Failed to generate LLM-based thought: {e}")

        # Fallback to generic thought if LLM fails
        print(f"⚠️ Using fallback thought for {tool_name}")
        return f"I will use {tool_name} tool to process the spatial data with the specified parameters."

    def generate_structured_response(self, state: AgentState) -> str:
        """
        Generate structured response in benchmark.json format from ReAct execution trace.

        This method is called when the workflow ends in ReAct mode (no planner).
        It converts the conversation history into the same structured format that
        the planner generates: {"dialogs": [...], "answer": "..."}

        Args:
            state (AgentState): Final agent state with complete message history

        Returns:
            str: JSON string with dialogs and answer fields
        """
        import json

        try:
            messages = state["messages"]
            dialogs = []
            final_answer = ""

            # Extract user query from first message
            user_query = ""
            if messages and len(messages) > 0:
                first_msg = messages[0]
                if hasattr(first_msg, 'content'):
                    user_query = first_msg.content
                elif isinstance(first_msg, dict):
                    user_query = first_msg.get('content', '')

            # Add user query to dialogs
            if user_query:
                dialogs.append({
                    "role": "user",
                    "content": user_query
                })

            # Process messages to build dialogs
            for i, msg in enumerate(messages[1:], 1):  # Skip first message (user query)
                # Handle AIMessage with tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        # Generate realistic thought using LLM
                        thought = self._generate_realistic_thought(
                            tool_call['name'],
                            tool_call.get('args', {}),
                            user_query,
                            messages[:i+1]  # Conversation history up to this point
                        )

                        # Transform arguments to match benchmark format
                        structured_args = self._create_structured_tool_arguments(
                            tool_call.get('args', {}), tool_call['name']
                        )

                        # Add assistant message with tool call
                        dialogs.append({
                            "role": "assistant",
                            "thought": thought,
                            "tool_calls": [{
                                "type": "function",
                                "function": {
                                    "name": tool_call['name'],
                                    "arguments": structured_args
                                }
                            }]
                        })

                # Handle ToolMessage (tool results)
                elif hasattr(msg, 'type') and msg.type == 'tool':
                    tool_name = getattr(msg, 'name', 'unknown')
                    tool_content = getattr(msg, 'content', '')

                    # Parse tool content if it's a JSON string
                    try:
                        if isinstance(tool_content, str) and tool_content.strip().startswith('{'):
                            tool_result = json.loads(tool_content)
                        else:
                            tool_result = {"result": str(tool_content)}
                    except json.JSONDecodeError:
                        tool_result = {"result": str(tool_content)}

                    # Create structured tool response matching planner format
                    # Extract key metrics and arguments field, exclude polygon/mask data
                    structured_content = self._create_structured_tool_response(
                        tool_result, tool_name
                    )

                    # Add tool response to dialogs
                    dialogs.append({
                        "role": "tool",
                        "name": tool_name,
                        "content": structured_content
                    })

                # Handle final AIMessage (answer)
                elif hasattr(msg, 'content') and msg.content:
                    # CRITICAL FIX: Only extract final answer from AIMessage, not SystemMessage
                    # SystemMessage is used for failure notifications when max iterations is reached
                    is_ai_message = hasattr(msg, 'type') and msg.type == 'ai'
                    has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0

                    if is_ai_message and not has_tool_calls:
                        # This is the final answer from the model
                        final_answer = msg.content

            # If no final answer was found, use a default message
            if not final_answer:
                final_answer = "I wasn't able to complete the analysis after multiple reasoning attempts. Could you please rephrase your request or try a more specific query?"

            # ==================== ENHANCED ANSWER SYNTHESIS ====================
            # Apply two-mode answer synthesis (basic + scene-enhanced) if execution steps available
            # This matches generate_gt_all.py's answer quality
            activated_scene = None
            if self.execution_steps:
                try:
                    # Extract query content for answer synthesis
                    query_content = ""
                    for msg in messages:
                        if hasattr(msg, 'type') and msg.type == 'human':
                            query_content = msg.content
                            break

                    if query_content:
                        print(f"🔧 Applying enhanced answer synthesis with {len(self.execution_steps)} execution steps")
                        enhanced_answer, activated_scene = self.get_enhanced_final_answer(query_content)

                        if enhanced_answer and enhanced_answer.strip():
                            # Use enhanced answer if it's better than raw LLM output
                            # Check if enhanced answer contains actual analysis results
                            if any(keyword in enhanced_answer.lower() for keyword in
                                   ["analysis", "detected", "found", "shows", "measured", "%", "meters", "objects"]):
                                print(f"✅ Using enhanced answer (scene: {activated_scene})")
                                final_answer = enhanced_answer
                            else:
                                print("⚠️ Enhanced answer lacks analysis content, keeping original")
                        else:
                            print("⚠️ Enhanced answer empty, keeping original")
                except Exception as e:
                    print(f"⚠️ Answer synthesis failed: {e}, keeping original answer")

            # Create structured response
            # NOTE: activated_scene is NOT included in output (used internally for synthesis only)
            result = {
                "dialogs": dialogs,
                "answer": final_answer
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            print(f"❌ Failed to generate structured response: {e}")
            # Fallback to simple format
            return json.dumps({
                "dialogs": [{"role": "user", "content": "Query processing failed"}],
                "answer": "Error generating structured response"
            })

    def _create_structured_tool_arguments(self, tool_args: dict, tool_name: str) -> dict:
        """
        Transform tool_calls arguments to match benchmark.json format.

        The LLM may send complex nested structures, but benchmark expects flat structures
        with specific field names.

        Args:
            tool_args: Raw tool arguments from LLM
            tool_name: Name of the tool

        Returns:
            dict: Transformed arguments matching benchmark format
        """
        if tool_name == "overlap":
            # Benchmark expects: source_class, target_class, meters_per_pixel, source_polygon_count, target_polygon_count
            # LLM may send: class_a_polygons, class_b_polygons with nested geometry_count and classes_used
            # Or may send direct class names, or have classes_used at top level
            transformed = {}
            source_class = ""
            target_class = ""
            source_polygon_count = 0
            target_polygon_count = 0

            # Try extracting from nested class_a_polygons structure
            class_a_data = tool_args.get("class_a_polygons", {})
            if isinstance(class_a_data, dict):
                classes_used = class_a_data.get("classes_used", [])
                if classes_used:
                    source_class = classes_used[0] if isinstance(classes_used, list) else classes_used
                source_polygon_count = class_a_data.get("geometry_count", 0)

            # Try extracting from nested class_b_polygons structure
            class_b_data = tool_args.get("class_b_polygons", {})
            if isinstance(class_b_data, dict):
                classes_used = class_b_data.get("classes_used", [])
                if classes_used:
                    target_class = classes_used[0] if isinstance(classes_used, list) else classes_used
                target_polygon_count = class_b_data.get("geometry_count", 0)

            # Try direct class names from multiple aliases
            if not source_class:
                source_class = (
                    tool_args.get("source_class") or
                    tool_args.get("class_a") or
                    tool_args.get("class_a_name") or
                    ""
                )
            if not target_class:
                target_class = (
                    tool_args.get("target_class") or
                    tool_args.get("class_b") or
                    tool_args.get("class_b_name") or
                    ""
                )

            # Try top-level classes_used list
            top_classes_used = tool_args.get("classes_used", [])
            if isinstance(top_classes_used, list) and len(top_classes_used) >= 2:
                if not source_class:
                    source_class = top_classes_used[0]
                if not target_class:
                    target_class = top_classes_used[1]

            # Get polygon counts from direct args if not from nested
            if source_polygon_count == 0:
                source_polygon_count = tool_args.get("source_polygon_count", 0)
            if target_polygon_count == 0:
                target_polygon_count = tool_args.get("target_polygon_count", 0)

            transformed["source_class"] = source_class
            transformed["target_class"] = target_class
            transformed["source_polygon_count"] = source_polygon_count
            transformed["target_polygon_count"] = target_polygon_count
            transformed["meters_per_pixel"] = tool_args.get("meters_per_pixel", 0.3)

            return transformed

        elif tool_name == "buffer":
            # Benchmark expects: buffer_class, buffer_distance_meters, meters_per_pixel
            transformed = {}
            transformed["buffer_class"] = tool_args.get("buffer_class", tool_args.get("class_name", ""))
            transformed["buffer_distance_meters"] = tool_args.get("buffer_distance_meters", tool_args.get("distance", 0))
            transformed["meters_per_pixel"] = tool_args.get("meters_per_pixel", 0.3)
            return transformed

        elif tool_name == "distance":
            # Benchmark expects: source_class, target_class, meters_per_pixel
            # LLM may send: source_class, target_class, class_a, class_b, set_a, set_b, from_class, to_class
            # Or may send empty strings but have classes_used list
            transformed = {}

            # Try multiple aliases for source class
            source_class = (
                tool_args.get("source_class") or
                tool_args.get("class_a") or
                tool_args.get("set_a") or
                tool_args.get("from_class") or
                ""
            )

            # Try multiple aliases for target class
            target_class = (
                tool_args.get("target_class") or
                tool_args.get("class_b") or
                tool_args.get("set_b") or
                tool_args.get("to_class") or
                ""
            )

            # If source/target still empty, try to extract from classes_used list
            classes_used = tool_args.get("classes_used", [])
            if isinstance(classes_used, list) and len(classes_used) >= 2:
                if not source_class:
                    source_class = classes_used[0]
                if not target_class:
                    target_class = classes_used[1]
            elif isinstance(classes_used, list) and len(classes_used) == 1:
                # Only one class in list - use it for whichever is missing
                if not source_class:
                    source_class = classes_used[0]
                elif not target_class:
                    target_class = classes_used[0]

            transformed["source_class"] = source_class
            transformed["target_class"] = target_class
            transformed["meters_per_pixel"] = tool_args.get("meters_per_pixel", 0.3)
            return transformed

        elif tool_name == "containment":
            # Benchmark expects: container_class, contained_class, meters_per_pixel
            # LLM may send: buffer_class, geometry_count, image_path
            transformed = {}
            # container_class: the class that contains other objects (e.g., water)
            transformed["container_class"] = tool_args.get("container_class", tool_args.get("buffer_class", ""))
            # contained_class: the class being contained (e.g., agriculture)
            transformed["contained_class"] = tool_args.get("contained_class", tool_args.get("geometry_count", ""))
            transformed["meters_per_pixel"] = tool_args.get("meters_per_pixel", 0.3)
            return transformed

        # Default: return original args for other tools
        return tool_args

    def _create_structured_tool_response(self, tool_result: dict, tool_name: str) -> dict:
        """
        Create structured tool response matching benchmark.json format.

        Extracts key metrics and arguments field from tool results,
        excluding detailed polygon/mask data to match planner output format.

        Args:
            tool_result: Raw tool response data
            tool_name: Name of the tool that generated the response

        Returns:
            dict: Structured content in benchmark format {"type": "json", "content": {...}}
        """
        import json

        try:
            # Extract key metrics based on tool type
            content = {}

            if tool_name == "detection":
                # Detection tool response format
                total_objects = tool_result.get("total_detections", 0)
                detections = tool_result.get("detections", [])
                classes_detected = list(set(d.get("class") for d in detections if d.get("class")))

                content = {
                    "total_objects": total_objects,
                    "classes_detected": classes_detected
                }

                # Add arguments field if present in tool output
                if "arguments" in tool_result:
                    content["arguments"] = tool_result["arguments"]

            elif tool_name == "segmentation":
                # Segmentation tool response format
                total_objects = tool_result.get("total_segments", 0)
                segments = tool_result.get("segments", [])
                classes_detected = list(set(s.get("class") for s in segments if s.get("class")))

                content = {
                    "total_objects": total_objects,
                    "classes_detected": classes_detected
                }

                # Add arguments field if present in tool output
                if "arguments" in tool_result:
                    content["arguments"] = tool_result["arguments"]

            elif tool_name == "classification":
                # Classification tool response format
                content = {
                    "predicted_category": tool_result.get("predicted_category", ""),
                    "confidence": tool_result.get("confidence", 0),
                    "total_detections": tool_result.get("total_detections", 0)
                }

                # Add arguments field if present in tool output
                if "arguments" in tool_result:
                    content["arguments"] = tool_result["arguments"]

            elif tool_name == "buffer":
                # Buffer tool response format - match benchmark exactly
                # Benchmark only includes: buffer_distance_meters, geometry_count, total_buffered_area_sqm
                unified_buffer = tool_result.get("unified_buffer", {})
                if isinstance(unified_buffer, dict):
                    total_buffered_area = unified_buffer.get("area_sqm", 0)
                else:
                    total_buffered_area = tool_result.get("total_buffered_area_sqm", 0.0)

                content = {
                    "buffer_distance_meters": tool_result.get("buffer_distance_meters", 0),
                    "geometry_count": tool_result.get("geometry_count", 0),
                    "total_buffered_area_sqm": total_buffered_area
                }

            elif tool_name == "overlap":
                # Overlap tool response format - match benchmark exactly
                # Benchmark only includes: overlap_percentage
                content = {
                    "overlap_percentage": tool_result.get("overlap_percentage", 0)
                }

            elif tool_name == "containment":
                # Containment tool response format - match benchmark exactly
                # Benchmark only includes: containment_percentage, contained_count
                content = {
                    "containment_percentage": tool_result.get("containment_percentage", 0),
                    "contained_count": tool_result.get("objects_fully_contained", 0)
                }

            elif tool_name == "object_count_aoi":
                # Object count AOI tool response format
                # Extract object count from nested output structure
                output_data = tool_result.get("output", {})
                summary = output_data.get("summary", {})
                # Benchmark format uses 'object_count' field
                object_count = summary.get("objects_in_any_aoi",
                               tool_result.get("objects_in_aoi",
                               tool_result.get("object_count", 0)))
                content = {
                    "object_count": object_count
                }

            elif tool_name == "area_measurement":
                # Area measurement tool response format
                output_data = tool_result.get("output", {})
                union_summary = output_data.get("union_summary", {})
                total_area_sqm = union_summary.get("union_area_sqm", 0)

                content = {
                    "total_area_sqm": total_area_sqm
                }

            elif tool_name in ["distance_calculation", "distance_tool"]:
                # Distance calculation tool response format
                output_data = tool_result.get("output", {})
                stats = output_data.get("stats", {})
                mean_distance = stats.get("meters", {}).get("mean", 0)

                content = {
                    "mean_distance_meters": mean_distance
                }

            elif tool_name == "change_detection":
                # Change detection tool response format
                total_changes = tool_result.get("total_changes", 0)
                change_percentage = tool_result.get("change_percentage", 0)

                content = {
                    "total_changes": total_changes,
                    "change_percentage": change_percentage
                }

                # Add arguments field if present in tool output
                if "arguments" in tool_result:
                    content["arguments"] = tool_result["arguments"]

            else:
                # Generic tool response - extract common fields
                for key in ["total_objects", "total_count", "success", "error", "summary"]:
                    if key in tool_result:
                        content[key] = tool_result[key]

                # If no common fields found, include the full response
                if not content:
                    content = tool_result

            # Return in benchmark.json format
            return {
                "type": "json",
                "content": content
            }

        except Exception as e:
            print(f"⚠️  Failed to structure tool response for {tool_name}: {e}")
            # Fallback to simple format
            return {
                "type": "json",
                "content": {"raw_response": str(tool_result)}
            }

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)

    def _extract_query_and_image(self, messages: List[AnyMessage]) -> tuple[str, Optional[str]]:
        """
        Extract user query and image path from messages.

        Args:
            messages: List of messages from the conversation

        Returns:
            tuple: (user_query, image_path)
        """
        user_query = ""
        image_path = None

        print(f"🔍 DEBUG: Extracting from {len(messages)} messages")
        for i, message in enumerate(messages):
            print(f"🔍 Message {i}: type={type(message)}, content={getattr(message, 'content', 'NO_CONTENT')[:100]}...")
            if hasattr(message, 'additional_kwargs'):
                print(f"🔍 Message {i} additional_kwargs: {message.additional_kwargs}")

        # CRITICAL FIX: Look for the last USER message only (exclude AI responses)
        # Use pattern-based detection instead of hardcoded type checking
        for message in reversed(messages):
            message_content = None
            message_role = None

            # Extract content dynamically
            if hasattr(message, 'content'):
                message_content = message.content
            elif isinstance(message, dict) and 'content' in message:
                message_content = message['content']

            # Determine message role/type dynamically
            if isinstance(message, dict) and 'role' in message:
                message_role = message['role']
            else:
                # Infer role from class name patterns (flexible approach)
                class_name = type(message).__name__.lower()
                if 'human' in class_name or 'user' in class_name:
                    message_role = 'user'
                elif 'ai' in class_name or 'assistant' in class_name or 'bot' in class_name:
                    message_role = 'assistant'
                elif 'system' in class_name:
                    message_role = 'system'
                else:
                    # For unknown types, check if content starts with AI response patterns
                    if message_content and isinstance(message_content, str):
                        # Common AI response patterns that indicate recycled responses
                        ai_patterns = [
                            "The detection tool successfully identified",
                            "The segmentation tool successfully",
                            "The classification tool successfully",
                            "Based on the analysis",
                            "I can see that",
                            "The spatial analysis shows"
                        ]
                        if any(pattern in message_content for pattern in ai_patterns):
                            message_role = 'assistant'  # Treat as AI response

            # Only process user messages, skip AI/assistant responses
            if message_role == 'user' and message_content:
                user_query = message_content
                print(f"🔍 Found user query: {user_query[:100]}...")
                break
            elif message_role in ['assistant', 'system']:
                print(f"🔍 Skipping {message_role} message to prevent recycling: {(message_content or '')[:50]}...")
                continue
            elif message_content:
                print(f"🔍 Skipping unknown message type {type(message).__name__}: {message_content[:50]}...")
                continue

        # Look for image in additional_kwargs from USER messages only
        for message in reversed(messages):
            # Determine if this is a user message using the same flexible approach
            message_role = None
            if isinstance(message, dict) and 'role' in message:
                message_role = message['role']
            else:
                class_name = type(message).__name__.lower()
                if 'human' in class_name or 'user' in class_name:
                    message_role = 'user'
                elif 'ai' in class_name or 'assistant' in class_name or 'bot' in class_name:
                    message_role = 'assistant'
                elif 'system' in class_name:
                    message_role = 'system'

            # Only extract image from user messages
            if message_role == 'user':
                if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                    image_path = message.additional_kwargs.get('image')
                    if image_path:
                        print(f"🔍 Found image path in user message additional_kwargs: {image_path}")
                        break
                elif isinstance(message, dict) and 'additional_kwargs' in message:
                    image_path = message['additional_kwargs'].get('image')
                    if image_path:
                        print(f"🔍 Found image path in user dict: {image_path}")
                        break
            elif message_role in ['assistant', 'system']:
                print(f"🔍 Skipping image extraction from {message_role} message")
                continue

        # CRITICAL FIX: Extract image path from OpenCompass format in query content
        if user_query:
            import re
            print(f"🔍 Parsing OpenCompass format from: {user_query}")

            # Look for "Image: /path/to/file.ext" pattern in the query
            image_match = re.search(r'Image:\s*([^\n\r]+\.(jpg|jpeg|png|tiff|tif))', user_query, re.IGNORECASE)
            if image_match:
                image_path = image_match.group(1).strip()
                print(f"🔍 ✅ Extracted image path from query content: {image_path}")

                # Remove the image line from the query to get clean query text
                user_query = re.sub(r'Image:\s*[^\n\r]+\.(jpg|jpeg|png|tiff|tif)\s*\n?', '', user_query, flags=re.IGNORECASE).strip()
                # Also remove "Query:" prefix if present
                user_query = re.sub(r'^Query:\s*', '', user_query, flags=re.IGNORECASE).strip()
                print(f"🔍 ✅ Cleaned user query: {user_query}")
            else:
                print(f"🔍 ❌ No image path found in query content")

        print(f"🔍 Final extraction result: query='{user_query}', image_path='{image_path}'")
        print(f"🔍 AGENT VERSION CHECK: Using FLEXIBLE agent code with pattern-based filtering - v4.0")
        print(f"🔍 MESSAGE TYPE FILTERING: AI response recycling prevention ACTIVE (flexible approach)")
        return user_query, image_path

    def _extract_text_prompt_from_query(self, query: str) -> Optional[str]:
        """
        Extract relevant text_prompt from user query for perception tools.

        BUG 5 FIX: When the LLM doesn't provide text_prompt, extract target objects
        from the user query to use as text_prompt for perception tools.

        Args:
            query: The user's natural language query

        Returns:
            str: Extracted text prompt containing target objects, or None if extraction fails
        """
        import re

        if not query:
            return None

        # Clean up the query - remove resolution info like "(0.3 m/px)" or "(3 m/px)"
        cleaned_query = re.sub(r'\(\d+\.?\d*\s*m/?px\)', '', query).strip()

        # Common patterns for extracting target objects from spatial reasoning queries
        # Pattern 1: "How many X are detected/located/found..."
        match = re.search(r'how many\s+([a-zA-Z\s,]+)\s+(?:are|is|were)\s+(?:detected|located|found|visible|present|within|inside|near)',
                         cleaned_query, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 2: "Detect/Find/Identify X in the image"
        match = re.search(r'(?:detect|find|identify|locate|segment)\s+([a-zA-Z\s,]+)\s+(?:in|within|from|on)',
                         cleaned_query, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 3: "Are any X located/present/visible..."
        match = re.search(r'are\s+(?:any|there|some)\s+([a-zA-Z\s,]+)\s+(?:located|present|visible|detected|within|near)',
                         cleaned_query, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 4: "What percentage of X overlaps/covers..."
        match = re.search(r'(?:what|which)\s+(?:percentage|portion|area)\s+(?:of\s+)?([a-zA-Z\s,]+)\s+(?:overlaps|covers|is covered)',
                         cleaned_query, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 5: "X within/near/adjacent to Y" - extract both classes
        match = re.search(r'([a-zA-Z\s]+)\s+(?:within|near|adjacent to|close to)\s+(?:the\s+)?([a-zA-Z\s]+)',
                         cleaned_query, re.IGNORECASE)
        if match:
            class_a = match.group(1).strip()
            class_b = match.group(2).strip()
            # Return both classes for comprehensive detection
            return f"{class_a}, {class_b}"

        # Pattern 6: "infrared small targets" or "SAR targets" - domain-specific
        if 'infrared' in cleaned_query.lower() or 'ir' in cleaned_query.lower():
            return "infrared small targets"
        if 'sar' in cleaned_query.lower():
            # Try to extract SAR-specific objects
            match = re.search(r'(ships?|vessels?|bridges?|tanks?|aircraft|buildings?|vehicles?)',
                             cleaned_query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return "targets"

        # Pattern 7: Extract common remote sensing objects
        common_objects = ['building', 'buildings', 'road', 'roads', 'tree', 'trees', 'car', 'cars',
                         'vehicle', 'vehicles', 'ship', 'ships', 'aircraft', 'airplane', 'bridge',
                         'water', 'forest', 'agriculture', 'industrial', 'residential', 'commercial',
                         'tank', 'tanks', 'container', 'containers', 'parking', 'lot']
        found_objects = []
        for obj in common_objects:
            if obj.lower() in cleaned_query.lower():
                found_objects.append(obj)
        if found_objects:
            # Remove duplicates and join
            unique_objects = list(dict.fromkeys(found_objects))
            return ", ".join(unique_objects[:5])  # Limit to 5 objects

        # Fallback: return None to indicate extraction failed (will use full query)
        return None

    def _log_multi_step_execution(self, query: str, image_path: Optional[str], result: str) -> None:
        """
        Log multi-step execution details.

        Args:
            query: User query
            image_path: Path to image (if any)
            result: Final execution result
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"multi_step_execution_{timestamp}.json"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "image_path": image_path,
            "execution_type": "multi_step_planning",
            "final_result": result,
            "planner_available": self.planner is not None
        }

        with open(filename, "w") as f:
            json.dump(log_entry, f, indent=4)

    # ==================== ANSWER SYNTHESIS METHODS ====================
    # These methods implement the two-mode answer synthesis from generate_gt_all.py:
    # 1. Basic Mode: Direct answer synthesis from tool outputs
    # 2. Scene-Enhanced Mode: Threshold-based scene context integration

    def _parse_tool_result_for_tracking(self, result: Any) -> Dict[str, Any]:
        """
        Parse tool result into structured format for execution tracking.

        This matches the structure expected by SceneContextAnalyzer for
        extracting class_stats from perception tool outputs.

        Args:
            result: Raw tool execution result (string or dict)

        Returns:
            Structured dict with success flag and parsed output
        """
        try:
            # If result is already a dict, wrap it
            if isinstance(result, dict):
                return {"success": True, "output": result, "arguments": result}

            # Try to parse JSON string
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        return {"success": True, "output": parsed, "arguments": parsed}
                except json.JSONDecodeError:
                    pass

            # Fallback: return as content string
            return {"success": True, "content": str(result)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _detect_modality(self, tool_sequence: List[str]) -> str:
        """
        Detect the query modality based on tool sequence.

        Args:
            tool_sequence: List of tool names in execution order

        Returns:
            Modality type: "optical", "sar", or "ir"
        """
        for tool in tool_sequence:
            if tool.startswith("sar_"):
                return "sar"
            elif tool.startswith("ir_") or tool == "infrared_detection":
                return "ir"

        # Default to optical if no specific modality detected
        return "optical"

    def _generate_answer(self, query_content: str, execution_steps: List[Dict[str, Any]]) -> str:
        """
        Generate natural language answer based on tool results using LLM.

        This is the BASIC MODE answer synthesis - directly synthesizes from tool outputs.
        Matches the approach in generate_gt_all.py lines 1908-2108.

        Args:
            query_content: Original user query
            execution_steps: List of tool execution steps with outputs

        Returns:
            Natural language answer string
        """
        if not execution_steps:
            return "Unable to complete analysis - no tool execution data available."

        # Detect query type for proper answer formatting
        query_lower = query_content.lower()
        is_percentage_query = any(keyword in query_lower for keyword in ["percentage", "percent", "%"])
        is_area_query = any(keyword in query_lower for keyword in ["area", "square meters", "sqm", "m²"])
        is_count_query = any(keyword in query_lower for keyword in ["how many", "count", "number of"])
        is_distance_query = any(keyword in query_lower for keyword in ["distance", "how far", "meters away"])

        # Find target step based on query type
        target_step = None
        if is_percentage_query:
            # Search for overlap or containment tool
            for step in reversed(execution_steps):
                step_tool = step.get("tool")
                if step_tool in ["overlap", "containment"]:
                    target_step = step
                    break

        # If no specific tool found, use final step
        if target_step is None:
            target_step = execution_steps[-1] if execution_steps else {}

        tool_output = target_step.get("output", {})
        tool_name = target_step.get("tool", "unknown")

        # Check success
        if isinstance(tool_output, dict) and not tool_output.get("success", True):
            return "Unable to complete analysis due to tool execution errors."

        # Extract numerical results based on tool type
        numerical_result = None
        result_type = None

        # Parse output from different structures
        output_data = tool_output
        if isinstance(tool_output, dict):
            output_data = tool_output.get("output", tool_output)

        if tool_name == "object_count_aoi":
            if isinstance(output_data, dict):
                numerical_result = output_data.get("total_objects", 0)
            result_type = "count"
        elif tool_name == "area_measurement":
            if isinstance(output_data, dict):
                numerical_result = output_data.get("total_area_sqm", 0)
            result_type = "area"
        elif tool_name == "overlap":
            if isinstance(output_data, dict):
                if is_area_query and not is_percentage_query:
                    numerical_result = output_data.get("intersection_area_sqm", 0)
                    result_type = "area"
                else:
                    numerical_result = output_data.get("overlap_percentage", 0)
                    result_type = "percentage"
        elif tool_name == "containment":
            if isinstance(output_data, dict):
                numerical_result = output_data.get("containment_percentage", 0)
                result_type = "percentage"
        elif tool_name in ["distance_calculation", "distance_tool"]:
            if isinstance(output_data, dict):
                stats = output_data.get("stats", {})
                numerical_result = stats.get("meters", {}).get("mean", 0)
            result_type = "mean_distance"
        elif tool_name in ["detection", "segmentation"]:
            if isinstance(output_data, dict):
                numerical_result = output_data.get("total_detections",
                                    output_data.get("total_segments", 0))
            result_type = "count"

        # Use LLM to generate natural language answer
        if numerical_result is not None:
            try:
                # Build additional context based on tool type
                additional_context = ""

                if tool_name == "containment":
                    contained_count = 0
                    if isinstance(output_data, dict):
                        contained_count = output_data.get("contained_count", 0)
                    additional_context = f"\nCONTAINED_COUNT: {contained_count}"

                    # Check if yes/no question
                    is_yes_no = any(query_lower.startswith(prefix) for prefix in ["are ", "is ", "do ", "does "])
                    if is_yes_no:
                        additional_context += "\nNOTE: This is a yes/no question. Start with 'Yes' or 'No'."

                elif tool_name == "overlap":
                    if is_percentage_query and not is_area_query:
                        additional_context = f"\nNOTE: The NUMERICAL_RESULT ({numerical_result}) is already a percentage."
                    elif is_area_query and not is_percentage_query:
                        additional_context = f"\nNOTE: The NUMERICAL_RESULT ({numerical_result}) is area in square meters."

                elif tool_name in ["distance_calculation", "distance_tool"]:
                    additional_context = f"\nNOTE: The NUMERICAL_RESULT ({numerical_result}) is the mean distance in meters."

                prompt = f"""You are a remote sensing expert providing analysis results to a user.

QUERY: {query_content}
ANALYSIS_TOOL: {tool_name}
RESULT_TYPE: {result_type}
NUMERICAL_RESULT: {numerical_result}{additional_context}

CRITICAL INSTRUCTIONS:
1. START your answer by directly addressing what the query asks for
2. Include the specific numerical value in your answer
3. Keep the answer natural and conversational (2-3 sentences)
4. Do NOT confuse count with area, or area with percentage

Return ONLY the answer text, no quotes or explanations."""

                # Use model to generate answer
                if hasattr(self._original_model, '_model_manager') and self._original_model._model_manager:
                    answer = self._original_model._model_manager.generate_chat_response(
                        [{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    if answer and answer.strip():
                        return answer.strip()
            except Exception as e:
                print(f"⚠️ Failed to generate LLM answer: {e}")

        # Fallback to template-based answers
        if tool_name == "object_count_aoi":
            return f"The analysis found {numerical_result} objects within the specified area of interest."
        elif tool_name == "area_measurement":
            return f"The total area measured is {numerical_result:.2f} square meters."
        elif tool_name == "overlap":
            if result_type == "area":
                return f"The overlap analysis shows {numerical_result:.2f} square meters of intersection area."
            else:
                return f"The overlap analysis shows {numerical_result:.2f}% overlap between the geometries."
        elif tool_name == "containment":
            if numerical_result > 0:
                return f"Yes, the containment analysis shows {numerical_result:.2f}% containment."
            else:
                return f"No, the containment analysis shows {numerical_result:.2f}% containment."
        elif tool_name in ["detection", "segmentation"]:
            return f"The analysis detected {numerical_result} objects in the image."
        elif tool_name in ["distance_calculation", "distance_tool"]:
            return f"The mean distance between the objects is {numerical_result:.2f} meters."

        return "Analysis completed successfully."

    def _generate_answer_with_scene(self, query_content: str, tool_sequence: List[str],
                                   execution_steps: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        """
        Generate answer with scene detection and context enhancement.

        This is the SCENE-ENHANCED MODE - uses threshold-based scene activation
        to enrich answers with domain context. Matches generate_gt_all.py lines 2110-2147.

        Uses SceneContextAnalyzer to:
        1. Detect modality (optical/sar/ir)
        2. Extract class coverage statistics from perception tool outputs
        3. Match against scene activation patterns (e.g., "forest ≥ 40% AND barren ≥ 20%")
        4. Synthesize unified answer with scene interpretation and recommendations

        Args:
            query_content: Original query text
            tool_sequence: List of tool names executed
            execution_steps: List of tool execution steps

        Returns:
            Tuple of (enhanced_answer, activated_scene_name)
        """
        # Generate base answer using basic mode
        base_answer = self._generate_answer(query_content, execution_steps)

        # Check if scene context analyzer is available
        if not self.scene_analyzer:
            print("⚠️ SceneContextAnalyzer not available - returning base answer")
            return base_answer, None

        # Detect activated scene using threshold-based pattern matching
        try:
            scene_name, scene_context = self.scene_analyzer.detect_activated_scene(
                tool_sequence, execution_steps
            )

            if scene_name and scene_context:
                print(f"🎯 Scene activated: {scene_name}")
                # Synthesize unified answer with scene context
                enhanced_answer = self._synthesize_unified_answer_with_scene(
                    query_content, base_answer, scene_name, scene_context
                )
                return enhanced_answer, scene_name
            else:
                print("📊 No scene activated - using base answer")
                return base_answer, None

        except Exception as e:
            print(f"⚠️ Error during scene detection: {e}")
            return base_answer, None

    def _synthesize_unified_answer_with_scene(self, query_content: str, base_answer: str,
                                             scene_name: str, scene_context: Dict[str, Any]) -> str:
        """
        Synthesize a unified, cohesive answer that integrates tool outputs with scene context.

        Uses LLM to weave together:
        1. Quantitative tool outputs (e.g., "92.6% of agriculture_land lies within 60m buffer")
        2. Scene context interpretation (e.g., desertification front analysis)
        3. Recommended actions naturally

        Result reads as a single flowing paragraph from an expert analyst.
        Matches generate_gt_all.py lines 2149-2236.

        Args:
            query_content: Original query text
            base_answer: Base answer from tool outputs
            scene_name: Name of activated scene
            scene_context: Scene context dictionary with interpretation and recommendations

        Returns:
            Unified, synthesized answer text
        """
        try:
            # Extract scene context components
            description = scene_context.get("description", "").strip()
            interpretation = scene_context.get("interpretation_context", "").strip()
            response_enhancement = scene_context.get("response_enhancement", "").strip()

            # Build synthesis prompt
            synthesis_prompt = f"""You are an expert remote sensing analyst. Your task is to synthesize a single, cohesive, flowing paragraph that naturally integrates quantitative analysis results with scene context interpretation and recommendations.

**Original Query:** {query_content}

**Quantitative Analysis Result:** {base_answer}

**Scene Context Identified:** {scene_name}

**Scene Description:** {description}

**Scene Interpretation:** {interpretation}

**Recommended Actions:** {response_enhancement}

**CRITICAL INSTRUCTIONS:**
1. PRESERVE the direct answer from the quantitative results - do NOT change or reframe it
2. The base answer already directly addresses the query (count, area, percentage, etc.)
3. Enhance it by seamlessly integrating scene context and implications
4. Structure: [Direct Answer from Quantitative Results] + [Scene Context] + [Implications/Recommendations]
5. Keep it natural and conversational (2-4 sentences total)
6. Do NOT replace the direct answer with scene context - only enhance it

**Task:** Create a unified expert analysis paragraph that:
1. Starts with the quantitative results (the direct answer to the query)
2. Naturally weaves in the scene context interpretation
3. Integrates the recommended actions as logical next steps
4. Reads as a single flowing analysis, NOT as separate concatenated sections
5. Avoids section headers, bullet points, or visible concatenation markers

**Output:** A single paragraph (2-4 sentences) that flows naturally as expert analysis."""

            # Use LLM to synthesize unified answer
            if hasattr(self._original_model, '_model_manager') and self._original_model._model_manager:
                print(f"🔧 Synthesizing unified answer for scene: {scene_name}")
                synthesized_answer = self._original_model._model_manager.generate_chat_response(
                    [{"role": "user", "content": synthesis_prompt}],
                    temperature=0.3  # Lower temperature for consistent synthesis
                )

                if synthesized_answer and synthesized_answer.strip():
                    # Clean up the response
                    synthesized_answer = synthesized_answer.strip()

                    # Remove any markdown formatting if present
                    if synthesized_answer.startswith("**") or synthesized_answer.startswith("##"):
                        lines = synthesized_answer.split("\n")
                        content_lines = [line.strip() for line in lines
                                        if line.strip() and not line.startswith("#") and not line.startswith("**")]
                        synthesized_answer = " ".join(content_lines)

                    print(f"✅ Synthesized answer: {synthesized_answer[:100]}...")
                    return synthesized_answer

        except Exception as e:
            print(f"⚠️ Error synthesizing unified answer: {e}")

        # Fallback to simple enhancement if LLM synthesis fails
        print("⚠️ Falling back to simple scene enhancement")
        if self.scene_analyzer:
            return self.scene_analyzer.enhance_answer_with_scene_context(base_answer, scene_context)
        return base_answer

    def get_enhanced_final_answer(self, query_content: str) -> Tuple[str, Optional[str]]:
        """
        Public method to get the enhanced final answer with scene context.

        This is the main entry point for answer synthesis. It:
        1. Uses tracked execution_steps and tool_sequence from ReAct execution
        2. Applies two-mode answer synthesis (basic + scene-enhanced)
        3. Returns both the answer and the activated scene name

        Args:
            query_content: Original user query

        Returns:
            Tuple of (final_answer, activated_scene_name)
        """
        if not self.execution_steps:
            return "Unable to complete analysis - no tool execution data available.", None

        # Use scene-enhanced mode if scene analyzer is available
        if self.scene_analyzer:
            return self._generate_answer_with_scene(
                query_content, self.tool_sequence, self.execution_steps
            )
        else:
            # Fallback to basic mode
            return self._generate_answer(query_content, self.execution_steps), None