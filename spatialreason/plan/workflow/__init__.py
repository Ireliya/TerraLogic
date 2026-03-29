"""
Workflow management modules for the spatial reasoning agent.

This package contains classes for tracking workflow state and managing
the execution flow of spatial reasoning tasks.
"""

# Workflow modules
from .state_manager import WorkflowStateManager
# from .phase_detector import WorkflowPhaseDetector

__all__ = [
    "WorkflowStateManager",
    # Will be populated as more modules are created
]
