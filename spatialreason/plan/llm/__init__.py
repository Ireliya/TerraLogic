"""
Language model integration modules for the spatial reasoning agent.

This package contains classes for integrating with language models,
handling function calling, and managing model interactions.
"""

# LLM modules
from .planner_llm import (
    PlannerLLM, QwenPlannerLLM, RemotePlannerLLM,
    PlannerModelConfig, ModelType,
    create_remote_planner_llm, create_local_qwen_planner_llm
)
from .function_calling import FunctionCallHandler

# Note: Legacy qwen_client.py has been removed - all functionality moved to model-agnostic PlannerLLM

__all__ = [
    "PlannerLLM",
    "QwenPlannerLLM",  # Backward compatibility alias
    "RemotePlannerLLM",
    "PlannerModelConfig",
    "ModelType",
    "create_remote_planner_llm",
    "create_local_qwen_planner_llm",
    "FunctionCallHandler",
]
