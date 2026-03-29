"""
Shared CLASS_VOCABULARY constant for consistent class naming across the agent system.

This module defines the authoritative list of valid class names that can be used
in perception tools (detection, segmentation, classification, SAR, IR).

The vocabulary is used by:
- Agent's planner: For tool parameter generation during query execution
- Evaluation scripts: For validating class names in predictions

IMPORTANT: This vocabulary MUST match the CLASS_VOCABULARY in generate_gt_all.py
to ensure the agent produces the same results as the ground truth generation.

NOTE: generate_gt_all.py maintains its own copy of CLASS_VOCABULARY since it was
already used to generate benchmark.json. This module ensures the agent uses the
same vocabulary without modifying the working ground truth generation script.
"""

# Class vocabulary constraint
# Includes both normalized names (lowercase with underscores) and proper classification category names (title case with spaces)
CLASS_VOCABULARY = {
    # Normalized names (for detection, segmentation, SAR, IR tools)
    # NOTE: Use base names without "_land" suffix for perception tools (e.g., "forest" not "forest_land")
    "agriculture", "barren", "bridge", "building", "cars", "forest", "harbor",
    "infrared_small_target", "low_vegetation", "rangeland", "road", "tank", "trees", "water",
    # Proper classification category names (title case with spaces - for classification tool ONLY)
    "Agriculture land", "Barren land", "Forest land", "Rangeland", "Urban land", "Water"
}


def get_class_vocabulary_prompt() -> str:
    """
    Get formatted CLASS_VOCABULARY string for LLM prompts.
    
    Returns:
        Formatted string listing all valid class names, sorted alphabetically
    """
    return ', '.join(sorted(CLASS_VOCABULARY))


def get_normalized_classes() -> set:
    """
    Get only the normalized class names (lowercase with underscores).

    Returns:
        Set of normalized class names for detection, segmentation, SAR, IR tools
    """
    return {
        "agriculture", "barren", "bridge", "building", "cars", "forest", "harbor",
        "infrared_small_target", "low_vegetation", "rangeland", "road", "tank", "trees", "water"
    }


def get_classification_categories() -> set:
    """
    Get only the classification category names (title case with spaces).
    
    Returns:
        Set of classification category names for classification tool
    """
    return {
        "Agriculture land", "Barren land", "Forest land", "Rangeland", "Urban land", "Water"
    }


def validate_class_name(class_name: str) -> bool:
    """
    Validate if a class name is in the vocabulary.
    
    Args:
        class_name: Class name to validate
        
    Returns:
        True if class name is valid, False otherwise
    """
    return class_name in CLASS_VOCABULARY


def filter_valid_classes(class_list: list, allow_buffer_prefix: bool = False) -> list:
    """
    Filter a list of class names to only include valid ones.
    
    Args:
        class_list: List of class names to filter
        allow_buffer_prefix: If True, allow "buffer_<class>" format for spatial tools
        
    Returns:
        Filtered list containing only valid class names
    """
    if allow_buffer_prefix:
        return [c for c in class_list if c in CLASS_VOCABULARY or c.startswith("buffer_")]
    else:
        return [c for c in class_list if c in CLASS_VOCABULARY]

