"""
Rule-based captioning for RemoteSAM detection results.
"""

import numpy as np
from typing import Dict, List, Tuple


def single_captioning(boxes: Dict[str, List], shape: Tuple[int, int], region_split: int = 4) -> str:
    """
    Generate a caption based on detected objects and their spatial relationships.
    
    Args:
        boxes: Dictionary of detected boxes for each class
        shape: Image shape (height, width)
        region_split: Number of regions to split the image into
        
    Returns:
        Generated caption string
    """
    if not boxes:
        return "No objects detected in the image."
    
    height, width = shape
    caption_parts = []
    
    # Count total objects
    total_objects = sum(len(box_list) for box_list in boxes.values() if box_list)
    
    if total_objects == 0:
        return "No objects detected in the image."
    
    # Generate basic counts
    count_descriptions = []
    for class_name, box_list in boxes.items():
        if box_list and len(box_list) > 0:
            count = len(box_list)
            if count == 1:
                count_descriptions.append(f"1 {class_name}")
            else:
                count_descriptions.append(f"{count} {class_name}s")
    
    if count_descriptions:
        caption_parts.append(f"The image contains {', '.join(count_descriptions)}.")
    
    # Analyze spatial distribution
    spatial_desc = analyze_spatial_distribution(boxes, shape, region_split)
    if spatial_desc:
        caption_parts.append(spatial_desc)
    
    # Analyze object relationships
    relationship_desc = analyze_object_relationships(boxes, shape)
    if relationship_desc:
        caption_parts.append(relationship_desc)
    
    # Combine all parts
    if caption_parts:
        return " ".join(caption_parts)
    else:
        return f"Detected {total_objects} objects in the image."


def analyze_spatial_distribution(boxes: Dict[str, List], shape: Tuple[int, int], region_split: int = 4) -> str:
    """
    Analyze the spatial distribution of objects in the image.
    
    Args:
        boxes: Dictionary of detected boxes
        shape: Image shape (height, width)
        region_split: Number of regions to split the image into
        
    Returns:
        Spatial distribution description
    """
    height, width = shape
    region_height = height // region_split
    region_width = width // region_split
    
    # Define region names
    region_names = {
        (0, 0): "top-left",
        (0, 1): "top-center",
        (0, 2): "top-right",
        (1, 0): "center-left",
        (1, 1): "center",
        (1, 2): "center-right",
        (2, 0): "bottom-left",
        (2, 1): "bottom-center",
        (2, 2): "bottom-right"
    }
    
    # Count objects in each region
    region_counts = {}
    
    for class_name, box_list in boxes.items():
        if not box_list:
            continue
            
        for box in box_list:
            x1, y1, x2, y2 = box[:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Determine region
            region_row = min(int(center_y // region_height), region_split - 1)
            region_col = min(int(center_x // region_width), region_split - 1)
            
            if region_split == 3:
                region_key = (region_row, region_col)
                if region_key in region_names:
                    region_name = region_names[region_key]
                    if region_name not in region_counts:
                        region_counts[region_name] = {}
                    if class_name not in region_counts[region_name]:
                        region_counts[region_name][class_name] = 0
                    region_counts[region_name][class_name] += 1
    
    # Generate spatial description
    if not region_counts:
        return ""
    
    spatial_parts = []
    for region, class_counts in region_counts.items():
        if class_counts:
            class_desc = []
            for class_name, count in class_counts.items():
                if count == 1:
                    class_desc.append(f"1 {class_name}")
                else:
                    class_desc.append(f"{count} {class_name}s")
            
            if class_desc:
                spatial_parts.append(f"{', '.join(class_desc)} in the {region}")
    
    if spatial_parts:
        return f"Spatially, there are {', '.join(spatial_parts)}."
    
    return ""


def analyze_object_relationships(boxes: Dict[str, List], shape: Tuple[int, int]) -> str:
    """
    Analyze relationships between different object classes.
    
    Args:
        boxes: Dictionary of detected boxes
        shape: Image shape (height, width)
        
    Returns:
        Object relationship description
    """
    if len(boxes) < 2:
        return ""
    
    height, width = shape
    relationships = []
    
    # Get all class names with detected objects
    classes_with_objects = [class_name for class_name, box_list in boxes.items() if box_list]
    
    if len(classes_with_objects) < 2:
        return ""
    
    # Analyze pairwise relationships
    for i, class1 in enumerate(classes_with_objects):
        for j, class2 in enumerate(classes_with_objects):
            if i >= j:
                continue
            
            boxes1 = boxes[class1]
            boxes2 = boxes[class2]
            
            # Calculate average distances
            distances = []
            for box1 in boxes1:
                for box2 in boxes2:
                    dist = calculate_box_distance(box1, box2)
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                min_distance = np.min(distances)
                
                # Normalize by image diagonal
                image_diagonal = np.sqrt(height**2 + width**2)
                normalized_distance = min_distance / image_diagonal
                
                # Generate relationship description
                if normalized_distance < 0.1:
                    relationships.append(f"{class1}s and {class2}s are very close together")
                elif normalized_distance < 0.3:
                    relationships.append(f"{class1}s and {class2}s are nearby")
                elif normalized_distance > 0.7:
                    relationships.append(f"{class1}s and {class2}s are far apart")
    
    if relationships:
        return f"In terms of spatial relationships, {', '.join(relationships)}."
    
    return ""


def calculate_box_distance(box1: List, box2: List) -> float:
    """
    Calculate the minimum distance between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2, ...]
        box2: Second bounding box [x1, y1, x2, y2, ...]
        
    Returns:
        Minimum distance between the boxes
    """
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate center points
    center1_x = (x1_1 + x2_1) / 2
    center1_y = (y1_1 + y2_1) / 2
    center2_x = (x1_2 + x2_2) / 2
    center2_y = (y1_2 + y2_2) / 2
    
    # Calculate Euclidean distance between centers
    distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    return distance


def get_size_description(box: List, shape: Tuple[int, int]) -> str:
    """
    Get size description for a bounding box.
    
    Args:
        box: Bounding box [x1, y1, x2, y2, ...]
        shape: Image shape (height, width)
        
    Returns:
        Size description string
    """
    x1, y1, x2, y2 = box[:4]
    box_area = (x2 - x1) * (y2 - y1)
    image_area = shape[0] * shape[1]
    
    area_ratio = box_area / image_area
    
    if area_ratio > 0.5:
        return "large"
    elif area_ratio > 0.2:
        return "medium"
    elif area_ratio > 0.05:
        return "small"
    else:
        return "very small"


def get_position_description(box: List, shape: Tuple[int, int]) -> str:
    """
    Get position description for a bounding box.
    
    Args:
        box: Bounding box [x1, y1, x2, y2, ...]
        shape: Image shape (height, width)
        
    Returns:
        Position description string
    """
    height, width = shape
    x1, y1, x2, y2 = box[:4]
    
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Determine horizontal position
    if center_x < width / 3:
        h_pos = "left"
    elif center_x > 2 * width / 3:
        h_pos = "right"
    else:
        h_pos = "center"
    
    # Determine vertical position
    if center_y < height / 3:
        v_pos = "top"
    elif center_y > 2 * height / 3:
        v_pos = "bottom"
    else:
        v_pos = "middle"
    
    if h_pos == "center" and v_pos == "middle":
        return "center"
    elif h_pos == "center":
        return v_pos
    elif v_pos == "middle":
        return h_pos
    else:
        return f"{v_pos}-{h_pos}"
