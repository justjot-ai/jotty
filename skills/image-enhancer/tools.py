"""
Image Enhancer Skill - Improve image quality.

Enhances resolution, sharpness, and clarity for screenshots,
presentations, and social media posts.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available, image enhancement will be limited")


async def enhance_image_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance image quality.
    
    Args:
        params:
            - image_path (str): Path to image
            - output_path (str, optional): Output path
            - target_resolution (str, optional): Target resolution
            - enhancements (list, optional): Enhancements to apply
            - use_case (str, optional): Use case
            - preserve_original (bool, optional): Keep original
    
    Returns:
        Dictionary with enhanced image path and improvements
    """
    image_path = params.get('image_path', '')
    output_path = params.get('output_path', None)
    target_resolution = params.get('target_resolution', '2x')
    enhancements = params.get('enhancements', ['all'])
    use_case = params.get('use_case', 'web')
    preserve_original = params.get('preserve_original', True)
    
    if not image_path:
        return {
            'success': False,
            'error': 'image_path is required'
        }
    
    if not PIL_AVAILABLE:
        return {
            'success': False,
            'error': 'PIL/Pillow not available. Install with: pip install Pillow numpy'
        }
    
    try:
        image_file = Path(os.path.expanduser(image_path))
        if not image_file.exists():
            return {
                'success': False,
                'error': f'Image file not found: {image_path}'
            }
        
        # Load image
        img = Image.open(image_file)
        original_size = img.size
        
        improvements = {
            'original_size': original_size,
            'enhancements_applied': []
        }
        
        # Apply enhancements
        if 'all' in enhancements or 'sharpness' in enhancements:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # 50% sharper
            improvements['enhancements_applied'].append('sharpness')
        
        if 'all' in enhancements or 'clarity' in enhancements:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)  # 20% more contrast
            improvements['enhancements_applied'].append('clarity')
        
        if 'all' in enhancements or 'artifacts' in enhancements:
            # Reduce artifacts with slight blur then sharpen
            img = img.filter(ImageFilter.MedianFilter(size=3))
            improvements['enhancements_applied'].append('artifact_reduction')
        
        # Upscale if requested
        if target_resolution == '2x':
            new_size = (original_size[0] * 2, original_size[1] * 2)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            improvements['resolution'] = '2x (retina)'
        elif target_resolution == '4k':
            # Scale to 4K maintaining aspect ratio
            target_width = 3840
            aspect_ratio = original_size[1] / original_size[0]
            new_size = (target_width, int(target_width * aspect_ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            improvements['resolution'] = '4K'
        elif target_resolution == 'retina':
            # 2x for retina displays
            new_size = (original_size[0] * 2, original_size[1] * 2)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            improvements['resolution'] = 'retina (2x)'
        
        improvements['new_size'] = img.size
        
        # Determine output path
        if not output_path:
            stem = image_file.stem
            suffix = image_file.suffix
            output_path = image_file.parent / f"{stem}_enhanced{suffix}"
        else:
            output_path = Path(os.path.expanduser(output_path))
        
        # Save enhanced image
        img.save(output_path, quality=95, optimize=True)
        
        # Preserve original if requested
        if preserve_original:
            original_backup = image_file.parent / f"{image_file.stem}_original{image_file.suffix}"
            if not original_backup.exists():
                import shutil
                shutil.copy2(image_file, original_backup)
        
        return {
            'success': True,
            'original_path': str(image_file),
            'enhanced_path': str(output_path),
            'improvements': improvements
        }
        
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }
