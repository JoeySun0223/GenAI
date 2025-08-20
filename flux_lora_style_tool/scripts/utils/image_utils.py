#!/usr/bin/env python3
"""
Image Processing Utilities Module
Provides image loading, saving, directory creation and other utilities
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def load_images(directory: Path, exclude_dirs: Optional[List[str]] = None) -> List[Path]:
    """
    Load image files from directory
    
    Args:
        directory: Image directory path
        exclude_dirs: List of subdirectories to exclude
    
    Returns:
        List of image file paths
    """
    if exclude_dirs is None:
        exclude_dirs = []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return image_files
    
    # Check if directory is empty
    all_files = list(directory.rglob('*'))
    if not all_files:
        logger.warning(f"Directory is empty: {directory}")
        return image_files
    
    for file_path in directory.rglob('*'):
        # Skip excluded directories
        if any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
            continue
        
        # Check file extension
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    # Sort by filename
    image_files.sort(key=lambda x: x.name)
    
    if image_files:
        logger.info(f"Found {len(image_files)} images in {directory}")
    else:
        logger.warning(f"No supported image files found in {directory}")
        logger.info(f"Supported formats: {', '.join(image_extensions)}")
    
    return image_files

def save_image(image, output_path: Path, quality: int = 95):
    """
    Save image
    
    Args:
        image: PIL image object or numpy array
        output_path: Output path
        quality: JPEG quality (1-100)
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy array to PIL image
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Save image
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            image.save(output_path, 'JPEG', quality=quality)
        else:
            image.save(output_path)
        
        logger.debug(f"Image saved: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save image {output_path}: {e}")
        raise

def create_output_dirs() -> dict:
    """
    Create output directory structure
    
    Returns:
        Output directory dictionary
    """
    base_dir = Path(__file__).parent.parent.parent
    
    output_dirs = {
        'output': base_dir / 'output',
        'debug': base_dir / 'debug'
    }
    
    # Create directories
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")
    
    return output_dirs

def resize_image(image: Image.Image, target_size: int, keep_aspect_ratio: bool = True) -> Image.Image:
    """
    Resize image
    
    Args:
        image: Input image
        target_size: Target size
        keep_aspect_ratio: Whether to maintain aspect ratio
    
    Returns:
        Resized image
    """
    if keep_aspect_ratio:
        # Calculate scale ratio
        ratio = min(target_size / image.width, target_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_image = image.resize(new_size, Image.LANCZOS)
        
        # Create target size canvas
        canvas = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        
        # Center the resized image
        x = (target_size - new_size[0]) // 2
        y = (target_size - new_size[1]) // 2
        canvas.paste(resized_image, (x, y))
        
        return canvas
    else:
        return image.resize((target_size, target_size), Image.LANCZOS)

def create_image_grid(images: List[Image.Image], grid_size: tuple = (2, 2), 
                     spacing: int = 10) -> Image.Image:
    """
    Create image grid
    
    Args:
        images: List of images
        grid_size: Grid size (rows, cols)
        spacing: Image spacing
    
    Returns:
        Grid image
    """
    if not images:
        raise ValueError("Image list is empty")
    
    rows, cols = grid_size
    num_images = min(len(images), rows * cols)
    
    # Get single image dimensions
    img_width, img_height = images[0].size
    
    # Calculate grid dimensions
    grid_width = cols * img_width + (cols - 1) * spacing
    grid_height = rows * img_height + (rows - 1) * spacing
    
    # Create grid canvas
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Place images
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        x = col * (img_width + spacing)
        y = row * (img_height + spacing)
        
        grid_image.paste(images[i], (x, y))
    
    return grid_image

def validate_image(image_path: Path) -> bool:
    """
    Validate if image file is valid
    
    Args:
        image_path: Image file path
    
    Returns:
        Whether valid
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def get_image_info(image_path: Path) -> dict:
    """
    Get image information
    
    Args:
        image_path: Image file path
    
    Returns:
        Image information dictionary
    """
    try:
        with Image.open(image_path) as img:
            info = {
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'file_size': image_path.stat().st_size,
                'valid': True
            }
        return info
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }
