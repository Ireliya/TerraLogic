"""
Simple response formatter for spatial reasoning tools.
Provides conversational formatting for tool responses.
"""

import json
from typing import Dict, Any, Optional


def format_conversational_response(tool_name: str, json_response: str, 
                                 include_technical: bool = False) -> str:
    """
    Format tool response in a conversational, human-readable way.
    
    Args:
        tool_name: Name of the tool that generated the response
        json_response: JSON response string from the tool
        include_technical: Whether to include technical details
        
    Returns:
        Formatted conversational response string
    """
    try:
        # Parse the JSON response
        data = json.loads(json_response)
        
        # Handle error responses
        if not data.get("success", True):
            error_msg = data.get("error", "Unknown error occurred")
            return f"I encountered an issue while running {tool_name}: {error_msg}"
        
        # Format based on tool type
        if tool_name == "detection":
            return _format_detection_response(data, include_technical)
        elif tool_name == "segmentation":
            return _format_segmentation_response(data, include_technical)
        elif tool_name == "classification":
            return _format_classification_response(data, include_technical)
        elif tool_name in ["overlap", "buffer", "containment"]:
            return _format_spatial_analysis_response(data, tool_name, include_technical)
        else:
            # Generic formatting
            return _format_generic_response(data, tool_name, include_technical)
            
    except json.JSONDecodeError:
        # If JSON parsing fails, return the raw response
        return f"Tool {tool_name} completed. Raw response: {json_response}"
    except Exception as e:
        return f"Error formatting response from {tool_name}: {str(e)}"


def _format_detection_response(data: Dict[str, Any], include_technical: bool = False) -> str:
    """Format detection tool response conversationally."""
    detections = data.get("detections", [])
    text_prompt = data.get("text_prompt", "objects")
    
    if not detections:
        return f"I didn't find any {text_prompt} in the image."
    
    count = len(detections)
    response = f"I found {count} {text_prompt} in the image."
    
    if include_technical:
        # Add confidence scores and locations
        high_conf = [d for d in detections if d.get("confidence", 0) > 0.8]
        if high_conf:
            response += f" {len(high_conf)} of them have high confidence scores (>80%)."
    
    return response


def _format_segmentation_response(data: Dict[str, Any], include_technical: bool = False) -> str:
    """Format segmentation tool response conversationally."""
    text_prompt = data.get("text_prompt", "objects")
    masks_found = data.get("masks_found", 0)
    
    if masks_found == 0:
        return f"I couldn't segment any {text_prompt} in the image."
    
    response = f"I successfully segmented {masks_found} {text_prompt} regions in the image."
    
    if include_technical:
        total_pixels = data.get("total_segmented_pixels", 0)
        if total_pixels > 0:
            response += f" The segmented areas cover {total_pixels:,} pixels."
    
    return response


def _format_classification_response(data: Dict[str, Any], include_technical: bool = False) -> str:
    """Format classification tool response conversationally."""
    classifications = data.get("classifications", [])
    
    if not classifications:
        return "I couldn't classify any regions in the image."
    
    # Get the most common class
    class_counts = {}
    for cls in classifications:
        class_name = cls.get("class", "unknown")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        most_common = max(class_counts.items(), key=lambda x: x[1])
        response = f"I classified the image regions. The most common class is '{most_common[0]}' with {most_common[1]} instances."
        
        if len(class_counts) > 1:
            other_classes = [f"{cls} ({count})" for cls, count in class_counts.items() if cls != most_common[0]]
            response += f" Other classes found: {', '.join(other_classes)}."
    else:
        response = "I completed the classification analysis."
    
    return response


def _format_spatial_analysis_response(data: Dict[str, Any], tool_name: str, 
                                    include_technical: bool = False) -> str:
    """Format spatial analysis tool response conversationally."""
    result_value = data.get("result_value", False)
    analysis = data.get("analysis", "")
    
    # Tool-specific formatting
    if tool_name == "overlap":
        percentage = data.get("overlap_percentage", 0)
        if result_value:
            response = f"Yes, there is significant spatial overlap ({percentage:.1f}%) between the analyzed areas."
        else:
            response = f"No, there is minimal spatial overlap ({percentage:.1f}%) between the analyzed areas."
    
    elif tool_name == "buffer":
        distance = data.get("buffer_distance", 0)
        within_buffer = data.get("within_buffer", False)
        if within_buffer:
            response = f"Yes, the areas are within the buffer zone (distance: {distance} meters)."
        else:
            response = f"No, the areas are outside the buffer zone (distance: {distance} meters)."
    
    elif tool_name == "containment":
        ratio = data.get("containment_ratio", 0)
        is_contained = data.get("is_contained", False)
        if is_contained:
            response = f"Yes, there is spatial containment with a ratio of {ratio:.2f}."
        else:
            response = f"No, there is no significant spatial containment (ratio: {ratio:.2f})."
    
    else:
        response = f"Spatial analysis completed. Result: {'Yes' if result_value else 'No'}."
    
    # Add analysis if available
    if analysis and include_technical:
        response += f" Analysis: {analysis}"
    
    return response


def _format_generic_response(data: Dict[str, Any], tool_name: str, 
                           include_technical: bool = False) -> str:
    """Format generic tool response conversationally."""
    summary = data.get("summary", "")
    if summary:
        return summary
    
    # Try to extract meaningful information
    if "result" in data:
        return f"Tool {tool_name} completed successfully. Result: {data['result']}"
    elif "output" in data:
        return f"Tool {tool_name} completed. Output: {data['output']}"
    else:
        return f"Tool {tool_name} completed successfully."


# Utility function for backward compatibility
def format_tool_response(tool_name: str, response_data: Any) -> str:
    """
    Backward compatibility function for formatting tool responses.
    
    Args:
        tool_name: Name of the tool
        response_data: Response data (dict or string)
        
    Returns:
        Formatted response string
    """
    if isinstance(response_data, str):
        return format_conversational_response(tool_name, response_data)
    elif isinstance(response_data, dict):
        return format_conversational_response(tool_name, json.dumps(response_data))
    else:
        return f"Tool {tool_name} completed with result: {str(response_data)}"
