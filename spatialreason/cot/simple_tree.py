"""
Simplified tree structure for SpatialreasonAgent Chain-of-Thought reasoning.
Contains only the essential functionality needed for our spatial analysis CoT.
"""

from typing import List, Optional, Any, Dict
from copy import deepcopy
import torch


def memory_efficient_copy(obj: Any) -> Any:
    """
    Memory-efficient copying that avoids deep copying large PyTorch tensors and models.

    This function prevents CUDA OOM errors during CoT tree node creation by:
    1. Skipping deep copy of PyTorch tensors and models
    2. Creating shallow references for tool dictionaries containing models
    3. Only deep copying lightweight data structures
    4. Clearing GPU cache when needed

    Args:
        obj: Object to copy (typically io_state from spatial environment)

    Returns:
        Memory-efficient copy of the object
    """
    if obj is None:
        return None

    # Clear GPU cache before processing large objects to free up memory
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore cache clearing errors

    # Handle PyTorch tensors - return reference instead of copy
    if isinstance(obj, torch.Tensor):
        return obj  # Shallow reference to avoid memory duplication

    # Handle PyTorch modules/models - return reference instead of copy
    if isinstance(obj, torch.nn.Module):
        return obj  # Shallow reference to avoid memory duplication

    # Handle dictionaries that might contain tools/models
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Check if value contains PyTorch objects
            if hasattr(value, '__dict__') and any(
                isinstance(getattr(value, attr, None), (torch.Tensor, torch.nn.Module))
                for attr in dir(value) if not attr.startswith('_')
            ):
                # Shallow copy for objects containing PyTorch components
                result[key] = value
            elif key == 'tools_dict':
                # Special handling for tools_dict - shallow copy to preserve tool instances
                result[key] = value  # Shallow reference
            else:
                # Safe to deep copy lightweight objects
                try:
                    result[key] = memory_efficient_copy(value)
                except (RuntimeError, TypeError):
                    # Fallback to shallow copy if deep copy fails
                    result[key] = value
        return result

    # Handle lists
    if isinstance(obj, list):
        return [memory_efficient_copy(item) for item in obj]

    # Handle objects with __dict__ (custom classes)
    if hasattr(obj, '__dict__'):
        # Check if object contains PyTorch components
        has_torch_components = any(
            isinstance(getattr(obj, attr, None), (torch.Tensor, torch.nn.Module))
            for attr in dir(obj) if not attr.startswith('_')
        )

        if has_torch_components:
            # Return shallow reference for objects with PyTorch components
            return obj
        else:
            # Safe to attempt deep copy for lightweight objects
            try:
                return deepcopy(obj)
            except (RuntimeError, TypeError):
                # Fallback to shallow copy
                return obj

    # For primitive types and other objects, attempt deep copy
    try:
        return deepcopy(obj)
    except (RuntimeError, TypeError):
        # Fallback to shallow copy for objects that can't be deep copied
        return obj


class SimpleTreeNode:
    """
    Simplified tree node for CoT reasoning.
    Contains only the essential functionality needed for spatial analysis.
    """
    
    def __init__(self):
        # Node state
        self.is_terminal = False
        self.pruned = False
        self.finished = False
        
        # Node content
        self.node_type: Optional[str] = None  # "Thought", "Action", "Action Input", "Final Answer"
        self.description: str = ""
        self.observation: str = ""
        self.observation_code: Optional[int] = None
        
        # Tree structure
        self.children: List['SimpleTreeNode'] = []
        self.father: Optional['SimpleTreeNode'] = None
        
        # State and messages
        self.io_state: Any = None
        self.messages: List[Dict[str, Any]] = []
    
    def get_depth(self) -> int:
        """Get the depth of this node from the root."""
        if self.father is None:
            return 0
        return self.father.get_depth() + 1
    
    def get_max_depth(self) -> int:
        """Get the maximum depth of the subtree rooted at this node."""
        if not self.children:
            return 1
        return 1 + max(child.get_max_depth() for child in self.children)
    
    def get_size(self) -> int:
        """Get the total number of nodes in the subtree rooted at this node."""
        size = 1
        for child in self.children:
            size += child.get_size()
        return size
    
    def add_child(self, child: 'SimpleTreeNode'):
        """Add a child node."""
        child.father = self
        self.children.append(child)
    
    def prune(self):
        """Mark this subtree as pruned."""
        self.pruned = True
        for child in self.children:
            child.prune()
    
    def make_finish(self, inter_val: int = 1):
        """Mark this node and ancestors as finished."""
        self.finished = True
        if self.node_type == "Action Input":
            inter_val -= 1
        if self.father is not None and inter_val >= 0:
            self.father.make_finish(inter_val)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_type": self.node_type,
            "description": self.description,
            "observation": self.observation,
            "depth": self.get_depth(),
            "is_terminal": self.is_terminal,
            "pruned": self.pruned,
            "finished": self.finished,
            "child_count": len(self.children)
        }
    
    def get_reasoning_chain(self) -> List[Dict[str, Any]]:
        """Get the reasoning chain from root to this node."""
        chain = []
        node = self
        
        # Traverse up to root
        path = []
        while node.father is not None:
            path.append(node)
            node = node.father
        
        # Build chain from root to current node
        for node in reversed(path):
            if node.node_type in ["Thought", "Action", "Action Input"]:
                chain.append(node.to_dict())
        
        return chain


class SimpleTree:
    """
    Simplified tree structure for CoT reasoning.
    Contains only the essential functionality needed for spatial analysis.
    """
    
    def __init__(self):
        self.root = SimpleTreeNode()
        self.current_node = self.root
    
    def get_size(self) -> int:
        """Get total number of nodes in the tree."""
        return self.root.get_size()
    
    def get_max_depth(self) -> int:
        """Get maximum depth of the tree."""
        return self.root.get_max_depth()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation."""
        return {
            "size": self.get_size(),
            "max_depth": self.get_max_depth(),
            "root": self._node_to_dict_recursive(self.root)
        }
    
    def _node_to_dict_recursive(self, node: SimpleTreeNode) -> Dict[str, Any]:
        """Recursively convert node and children to dictionary."""
        node_dict = node.to_dict()
        node_dict["children"] = [
            self._node_to_dict_recursive(child) 
            for child in node.children
        ]
        return node_dict
    
    def find_terminal_nodes(self) -> List[SimpleTreeNode]:
        """Find all terminal nodes in the tree."""
        terminal_nodes = []
        self._find_terminal_nodes_recursive(self.root, terminal_nodes)
        return terminal_nodes
    
    def _find_terminal_nodes_recursive(self, node: SimpleTreeNode, terminal_nodes: List[SimpleTreeNode]):
        """Recursively find terminal nodes."""
        if node.is_terminal or not node.children:
            terminal_nodes.append(node)
        else:
            for child in node.children:
                self._find_terminal_nodes_recursive(child, terminal_nodes)


def create_thought_node(parent_node: SimpleTreeNode, thought_content: str) -> SimpleTreeNode:
    """Create a new thought node with memory-efficient copying."""
    new_node = SimpleTreeNode()
    new_node.node_type = "Thought"
    new_node.description = thought_content
    new_node.io_state = memory_efficient_copy(parent_node.io_state) if parent_node.io_state else None
    new_node.messages = parent_node.messages.copy()

    parent_node.add_child(new_node)

    # Add thought to messages
    new_node.messages.append({"role": "assistant", "content": f"Thought: {thought_content}"})

    return new_node


def create_action_node(parent_node: SimpleTreeNode, action_name: str) -> SimpleTreeNode:
    """Create a new action node with memory-efficient copying."""
    action_node = SimpleTreeNode()
    action_node.node_type = "Action"
    action_node.description = action_name
    action_node.io_state = memory_efficient_copy(parent_node.io_state) if parent_node.io_state else None
    action_node.messages = parent_node.messages.copy()

    parent_node.add_child(action_node)

    return action_node


def create_action_input_node(parent_node: SimpleTreeNode, action_args: Dict[str, Any]) -> SimpleTreeNode:
    """Create a new action input node with memory-efficient copying to prevent CUDA OOM."""
    import json

    input_node = SimpleTreeNode()
    input_node.node_type = "Action Input"
    input_node.description = json.dumps(action_args)
    # CRITICAL FIX: Use memory-efficient copying to avoid CUDA OOM from PyTorch tensors
    input_node.io_state = memory_efficient_copy(parent_node.io_state) if parent_node.io_state else None
    input_node.messages = parent_node.messages.copy()

    parent_node.add_child(input_node)

    return input_node
