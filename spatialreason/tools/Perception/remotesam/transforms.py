"""
Image transforms for RemoteSAM model.
"""

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for t in self.transforms:
            if target is not None:
                image, target = t(image, target)
            else:
                image = t(image, target) if hasattr(t, '__call__') and len(t.__code__.co_varnames) > 1 else t(image)
        
        if target is not None:
            return image, target
        return image


class Resize:
    """Resize image to specified size."""
    
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def __call__(self, image, target=None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize image
        resized_image = image.resize((self.width, self.height), Image.LANCZOS)
        
        if target is not None:
            # Resize target if provided
            if isinstance(target, np.ndarray):
                target = Image.fromarray(target)
            resized_target = target.resize((self.width, self.height), Image.NEAREST)
            return resized_image, resized_target
        
        return resized_image


class ToTensor:
    """Convert PIL Image or numpy array to tensor."""
    
    def __call__(self, image, target=None):
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            image = T.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            # Convert numpy to tensor
            if image.ndim == 3:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            else:
                image = torch.from_numpy(image).float()
            
            # Normalize to 0-1 if needed
            if image.max() > 1.0:
                image = image / 255.0
        
        if target is not None:
            if isinstance(target, Image.Image):
                target = torch.from_numpy(np.array(target)).long()
            elif isinstance(target, np.ndarray):
                target = torch.from_numpy(target).long()
            return image, target
        
        return image


class Normalize:
    """Normalize tensor with mean and std."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor):
            # Apply normalization
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            image = (image - mean) / std
        
        if target is not None:
            return image, target
        return image


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target=None):
        if torch.rand(1) < self.p:
            if isinstance(image, torch.Tensor):
                image = torch.flip(image, [-1])
            elif isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif isinstance(image, np.ndarray):
                image = np.fliplr(image)
            
            if target is not None:
                if isinstance(target, torch.Tensor):
                    target = torch.flip(target, [-1])
                elif isinstance(target, Image.Image):
                    target = target.transpose(Image.FLIP_LEFT_RIGHT)
                elif isinstance(target, np.ndarray):
                    target = np.fliplr(target)
        
        if target is not None:
            return image, target
        return image


class RandomVerticalFlip:
    """Randomly flip image vertically."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target=None):
        if torch.rand(1) < self.p:
            if isinstance(image, torch.Tensor):
                image = torch.flip(image, [-2])
            elif isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif isinstance(image, np.ndarray):
                image = np.flipud(image)
            
            if target is not None:
                if isinstance(target, torch.Tensor):
                    target = torch.flip(target, [-2])
                elif isinstance(target, Image.Image):
                    target = target.transpose(Image.FLIP_TOP_BOTTOM)
                elif isinstance(target, np.ndarray):
                    target = np.flipud(target)
        
        if target is not None:
            return image, target
        return image


class ColorJitter:
    """Randomly change brightness, contrast, saturation and hue."""
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, image, target=None):
        if isinstance(image, Image.Image):
            image = self.color_jitter(image)
        
        if target is not None:
            return image, target
        return image


class RandomRotation:
    """Randomly rotate image."""
    
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, image, target=None):
        angle = torch.rand(1) * 2 * self.degrees - self.degrees
        
        if isinstance(image, Image.Image):
            image = image.rotate(angle.item(), expand=False, fillcolor=0)
        elif isinstance(image, np.ndarray):
            # Use OpenCV for rotation
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle.item(), 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        if target is not None:
            if isinstance(target, Image.Image):
                target = target.rotate(angle.item(), expand=False, fillcolor=0)
            elif isinstance(target, np.ndarray):
                h, w = target.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle.item(), 1.0)
                target = cv2.warpAffine(target, M, (w, h))
            return image, target
        
        return image


class Pad:
    """Pad image to specified size."""
    
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill
    
    def __call__(self, image, target=None):
        if isinstance(image, Image.Image):
            w, h = image.size
            target_w, target_h = self.size
            
            if w < target_w or h < target_h:
                # Calculate padding
                pad_w = max(0, target_w - w)
                pad_h = max(0, target_h - h)
                
                # Pad image
                padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
                image = T.Pad(padding, fill=self.fill)(image)
        
        if target is not None:
            if isinstance(target, Image.Image):
                w, h = target.size
                target_w, target_h = self.size
                
                if w < target_w or h < target_h:
                    pad_w = max(0, target_w - w)
                    pad_h = max(0, target_h - h)
                    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
                    target = T.Pad(padding, fill=0)(target)
            
            return image, target
        
        return image


class CenterCrop:
    """Center crop image to specified size."""
    
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image, target=None):
        if isinstance(image, Image.Image):
            image = T.CenterCrop(self.size)(image)
        
        if target is not None:
            if isinstance(target, Image.Image):
                target = T.CenterCrop(self.size)(target)
            return image, target
        
        return image
