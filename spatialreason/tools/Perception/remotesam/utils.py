"""
Utility functions for RemoteSAM model.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
from scipy import ndimage
from skimage.measure import label, regionprops


def EPOC(image, image_processor, model):
    """
    EPOC (Entity-based Post-processing for Object Contours) function.
    This is a placeholder implementation.
    """
    # Convert PIL image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Simple edge detection as placeholder
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Normalize to 0-1 range
    return edges.astype(np.float32) / 255.0


def M2B(mask, prob, new_mask=None, box_type='hbb'):
    """
    Mask to Bounding Box conversion.
    
    Args:
        mask: Binary mask
        prob: Probability map
        new_mask: Optional refined mask
        box_type: Type of bounding box ('hbb' for horizontal)
        
    Returns:
        List of bounding boxes with confidence scores
    """
    if new_mask is not None:
        mask = new_mask
    
    # Find connected components
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    boxes = []
    
    for region in regions:
        # Get bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox
        
        # Calculate confidence score from probability map
        if prob is not None:
            region_prob = prob[min_row:max_row, min_col:max_col]
            confidence = np.mean(region_prob)
        else:
            confidence = 1.0
        
        # Filter out very small regions or low confidence
        if region.area > 100 and confidence > 0.1:
            boxes.append([min_col, min_row, max_col, max_row, confidence])
    
    return boxes


def resize_image(image, target_size):
    """
    Resize image to target size.
    
    Args:
        image: PIL Image or numpy array
        target_size: Tuple of (width, height)
        
    Returns:
        Resized image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    return image.resize(target_size, Image.LANCZOS)


def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize image with given mean and std.
    
    Args:
        image: numpy array or PIL Image
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to float and normalize to 0-1
    image = image.astype(np.float32) / 255.0
    
    # Apply normalization
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    normalized = (image - mean) / std
    
    return normalized


def post_process_mask(mask, min_area=100):
    """
    Post-process segmentation mask by removing small components.
    
    Args:
        mask: Binary mask
        min_area: Minimum area threshold
        
    Returns:
        Cleaned mask
    """
    # Label connected components
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    # Create cleaned mask
    cleaned_mask = np.zeros_like(mask)
    
    for region in regions:
        if region.area >= min_area:
            # Keep this region
            coords = region.coords
            cleaned_mask[coords[:, 0], coords[:, 1]] = 1
    
    return cleaned_mask


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def apply_morphological_operations(mask, operation='close', kernel_size=3):
    """
    Apply morphological operations to clean up mask.
    
    Args:
        mask: Binary mask
        operation: Type of operation ('open', 'close', 'erode', 'dilate')
        kernel_size: Size of morphological kernel
        
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'open':
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    elif operation == 'erode':
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    elif operation == 'dilate':
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    else:
        return mask


def smooth_mask(mask, sigma=1.0):
    """
    Smooth mask using Gaussian filter.
    
    Args:
        mask: Binary mask
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed mask
    """
    smoothed = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (smoothed > 0.5).astype(np.uint8)


def extract_largest_component(mask):
    """
    Extract the largest connected component from mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with only the largest component
    """
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    if not regions:
        return mask
    
    # Find largest region
    largest_region = max(regions, key=lambda r: r.area)
    
    # Create mask with only largest component
    result_mask = np.zeros_like(mask)
    coords = largest_region.coords
    result_mask[coords[:, 0], coords[:, 1]] = 1
    
    return result_mask
