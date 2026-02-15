"""
Slack GIF Creator Skill - Create Slack-optimized animated GIFs.

Creates animated GIFs optimized for Slack's size constraints
and requirements for both message GIFs and emoji GIFs.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import os
import math

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("slack-gif-creator")


logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    import imageio
    GIF_AVAILABLE = True
except ImportError:
    GIF_AVAILABLE = False
    logger.warning("PIL/imageio not available, GIF creation will be limited")


@async_tool_wrapper()
async def create_slack_gif_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a Slack-optimized animated GIF.
    
    Args:
        params:
            - description (str): Description of animation
            - gif_type (str, optional): 'message' or 'emoji'
            - output_path (str, optional): Output path
            - width (int, optional): Width
            - height (int, optional): Height
            - fps (int, optional): Frames per second
            - duration (float, optional): Duration in seconds
            - animation_type (str, optional): Animation type
    
    Returns:
        Dictionary with GIF path and validation results
    """
    status.set_callback(params.pop('_status_callback', None))

    description = params.get('description', '')
    gif_type = params.get('gif_type', 'message')
    output_path = params.get('output_path', None)
    width = params.get('width', None)
    height = params.get('height', None)
    fps = params.get('fps', None)
    duration = params.get('duration', 3.0)
    animation_type = params.get('animation_type', 'custom')
    
    if not description:
        return {
            'success': False,
            'error': 'description is required'
        }
    
    if not GIF_AVAILABLE:
        return {
            'success': False,
            'error': 'PIL/imageio not available. Install with: pip install Pillow imageio'
        }
    
    # Set defaults based on GIF type
    if gif_type == 'emoji':
        width = width or 128
        height = height or 128
        fps = fps or 12
        max_size_kb = 64
        max_duration = 2.0
    else:  # message
        width = width or 480
        height = height or 480
        fps = fps or 18
        max_size_kb = 2000
        max_duration = 5.0
    
    # Clamp duration
    duration = min(duration, max_duration)
    
    # Calculate frame count
    frame_count = int(fps * duration)
    
    # Limit frames for emoji GIFs
    if gif_type == 'emoji' and frame_count > 15:
        frame_count = 15
        duration = frame_count / fps
    
    try:
        # Generate frames based on animation type
        frames = await _generate_frames(
            description, width, height, frame_count, animation_type
        )
        
        # Determine output path
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"slack_{gif_type}_{timestamp}.gif"
            output_path = Path.home() / 'Downloads' / filename
        else:
            output_path = Path(os.path.expanduser(output_path))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create GIF with optimization
        num_colors = 48 if gif_type == 'emoji' else 256
        
        imageio.mimsave(
            str(output_path),
            frames,
            duration=1.0/fps,
            loop=0,
            palettesize=num_colors,
            subrectangles=True  # Optimize for smaller file size
        )
        
        # Check file size
        file_size_kb = output_path.stat().st_size / 1024
        meets_requirements = file_size_kb <= max_size_kb
        
        if not meets_requirements:
            logger.warning(
                f"GIF size ({file_size_kb:.1f}KB) exceeds limit ({max_size_kb}KB)"
            )
        
        return {
            'success': True,
            'gif_path': str(output_path),
            'file_size_kb': round(file_size_kb, 2),
            'meets_requirements': meets_requirements,
            'dimensions': f'{width}x{height}',
            'frame_count': frame_count,
            'duration': duration
        }
        
    except Exception as e:
        logger.error(f"GIF creation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def _generate_frames(
    description: str,
    width: int,
    height: int,
    frame_count: int,
    animation_type: str
) -> list:
    """Generate animation frames."""
    
    frames = []
    
    for i in range(frame_count):
        # Create base image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Calculate animation progress (0.0 to 1.0)
        progress = i / frame_count
        
        # Apply animation based on type
        if animation_type == 'shake':
            offset_x = int(5 * math.sin(progress * math.pi * 4))
            offset_y = int(5 * math.cos(progress * math.pi * 4))
            center_x = width // 2 + offset_x
            center_y = height // 2 + offset_y
        
        elif animation_type == 'bounce':
            bounce_height = abs(math.sin(progress * math.pi * 2)) * 30
            center_x = width // 2
            center_y = height // 2 - int(bounce_height)
        
        elif animation_type == 'pulse':
            scale = 0.8 + 0.2 * abs(math.sin(progress * math.pi * 2))
            center_x = width // 2
            center_y = height // 2
        
        elif animation_type == 'spin':
            angle = progress * 360
            center_x = width // 2
            center_y = height // 2
        
        else:  # custom/default
            center_x = width // 2
            center_y = height // 2
        
        # Draw a simple shape (circle) as placeholder
        radius = min(width, height) // 4
        if animation_type == 'pulse':
            radius = int(radius * scale)
        
        bbox = [
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius
        ]
        
        draw.ellipse(bbox, fill='blue', outline='darkblue', width=2)
        
        # Add text if there's space
        if width >= 200 and height >= 200:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except (OSError, IOError):
                # Font file not found, use default
                font = ImageFont.load_default()
            
            text = description[:20] if len(description) <= 20 else description[:17] + '...'
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = (width - text_width) // 2
            text_y = center_y + radius + 10
            
            draw.text((text_x, text_y), text, fill='black', font=font)
        
        frames.append(img)
    
    return frames
