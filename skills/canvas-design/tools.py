"""
Canvas Design Skill - Create visual art and designs using design philosophy.

Creates original visual designs by first establishing a design philosophy,
then expressing it visually through form, space, color, and composition.
"""
import asyncio
import logging
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import os
import random

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("canvas-design")


logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available, canvas design will be limited")

try:
    from reportlab.pdfgen import canvas as pdf_canvas
    from reportlab.lib.pagesizes import letter, A4
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not available, PDF creation will be limited")


@async_tool_wrapper()
async def create_design_artwork_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create visual artwork based on design philosophy.
    
    Args:
        params:
            - design_brief (str): Brief description or concept
            - output_format (str, optional): 'png' or 'pdf'
            - dimensions (tuple, optional): (width, height)
            - design_philosophy (str, optional): Pre-defined philosophy
            - style (str, optional): Style hint
    
    Returns:
        Dictionary with artwork path and design philosophy
    """
    status.set_callback(params.pop('_status_callback', None))

    design_brief = params.get('design_brief', '')
    output_format = params.get('output_format', 'png')
    dimensions = params.get('dimensions', (1920, 1080))
    design_philosophy = params.get('design_philosophy', None)
    style = params.get('style', 'minimalist')
    
    if not design_brief:
        return {
            'success': False,
            'error': 'design_brief is required'
        }
    
    if output_format == 'png' and not PIL_AVAILABLE:
        return {
            'success': False,
            'error': 'PIL/Pillow not available. Install with: pip install Pillow'
        }
    
    if output_format == 'pdf' and not REPORTLAB_AVAILABLE:
        return {
            'success': False,
            'error': 'reportlab not available. Install with: pip install reportlab'
        }
    
    try:
        # Generate design philosophy if not provided
        if not design_philosophy:
            design_philosophy = await _generate_design_philosophy(design_brief, style)
        
        # Create artwork based on philosophy
        if output_format == 'png':
            artwork_path = await _create_png_artwork(design_brief, design_philosophy, dimensions, style)
        else:
            artwork_path = await _create_pdf_artwork(design_brief, design_philosophy, dimensions, style)
        
        return {
            'success': True,
            'artwork_path': artwork_path,
            'design_philosophy': design_philosophy,
            'format': output_format
        }
        
    except Exception as e:
        logger.error(f"Canvas design creation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


async def _generate_design_philosophy(brief: str, style: str) -> str:
    """Generate design philosophy from brief."""
    
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill('claude-cli-llm')
        
        if claude_skill:
            generate_tool = claude_skill.tools.get('generate_text_tool')
            
            if generate_tool:
                prompt = f"""Create a visual design philosophy (4-6 paragraphs) for this brief:

**Brief:** {brief}
**Style Hint:** {style}

The philosophy should:
- Name the aesthetic movement (1-2 words)
- Describe how it manifests through space, form, color, composition
- Emphasize visual expression over text
- Stress meticulous craftsmanship and expert execution
- Leave creative room for visual interpretation

Output only the philosophy, no explanations."""
                
                if inspect.iscoroutinefunction(generate_tool):
                    result = await generate_tool({
                        'prompt': prompt,
                        'model': 'sonnet',
                        'timeout': 120
                    })
                else:
                    result = generate_tool({
                        'prompt': prompt,
                        'model': 'sonnet',
                        'timeout': 120
                    })
                
                if result.get('success'):
                    return result.get('text', '')
    except Exception as e:
        logger.debug(f"Design philosophy generation failed: {e}")
    
    # Fallback philosophy
    return f"""**{style.title()} Expression**

A {style} approach emphasizing visual communication through form, space, and color. 
The design expresses ideas through spatial relationships and composition rather than 
textual explanation. Every element is meticulously placed with expert craftsmanship, 
creating a cohesive visual language that communicates through visual weight, balance, 
and intentional restraint."""


async def _create_png_artwork(
    brief: str,
    philosophy: str,
    dimensions: Tuple[int, int],
    style: str
) -> str:
    """Create PNG artwork."""
    
    width, height = dimensions
    
    # Create image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Apply design based on style
    if style == 'minimalist':
        # Minimalist: lots of white space, subtle elements
        _draw_minimalist(draw, width, height, brief)
    
    elif style == 'bold':
        # Bold: strong colors, geometric shapes
        _draw_bold(draw, width, height, brief)
    
    elif style == 'organic':
        # Organic: flowing shapes, natural colors
        _draw_organic(draw, width, height, brief)
    
    else:  # geometric
        # Geometric: precise shapes, grid-based
        _draw_geometric(draw, width, height, brief)
    
    # Save artwork
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"canvas_design_{timestamp}.png"
    output_path = Path.home() / 'Downloads' / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    img.save(output_path, quality=95)
    
    return str(output_path)


def _draw_minimalist(draw: ImageDraw.Draw, width: int, height: int, brief: str):
    """Draw minimalist design."""
    
    # Large white space with subtle elements
    center_x, center_y = width // 2, height // 2
    
    # Subtle line
    draw.line([(width * 0.2, center_y), (width * 0.8, center_y)], fill='#e0e0e0', width=2)
    
    # Small text element (minimal)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    text = brief[:30] if len(brief) <= 30 else brief[:27] + '...'
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = center_x - text_width // 2
    text_y = center_y + 40
    
    draw.text((text_x, text_y), text, fill='#666666', font=font)


def _draw_bold(draw: ImageDraw.Draw, width: int, height: int, brief: str):
    """Draw bold design."""
    
    # Bold geometric shapes
    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3']
    
    # Large rectangle
    rect_width = width // 3
    rect_height = height // 2
    x1 = width // 2 - rect_width // 2
    y1 = height // 2 - rect_height // 2
    x2 = x1 + rect_width
    y2 = y1 + rect_height
    
    draw.rectangle([x1, y1, x2, y2], fill=random.choice(colors), outline='#333', width=4)
    
    # Bold text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    text = brief[:20] if len(brief) <= 20 else brief[:17] + '...'
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = width // 2 - text_width // 2
    text_y = y2 + 40
    
    draw.text((text_x, text_y), text, fill='#333', font=font)


def _draw_organic(draw: ImageDraw.Draw, width: int, height: int, brief: str):
    """Draw organic design."""
    
    # Organic flowing shapes
    center_x, center_y = width // 2, height // 2
    
    # Draw organic circles
    colors = ['#a8e6cf', '#dcedc1', '#ffd3a5', '#ffaaa5']
    
    for i in range(3):
        radius = (width // 8) + (i * 30)
        color = colors[i % len(colors)]
        
        # Offset circles for organic feel
        offset_x = center_x + random.randint(-50, 50)
        offset_y = center_y + random.randint(-50, 50)
        
        bbox = [
            offset_x - radius,
            offset_y - radius,
            offset_x + radius,
            offset_y + radius
        ]
        
        draw.ellipse(bbox, fill=color, outline='#888', width=2)


def _draw_geometric(draw: ImageDraw.Draw, width: int, height: int, brief: str):
    """Draw geometric design."""
    
    # Grid-based geometric shapes
    grid_size = width // 6
    
    colors = ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6']
    
    # Draw grid pattern
    for x in range(0, width, grid_size):
        for y in range(0, height, grid_size):
            if (x // grid_size + y // grid_size) % 2 == 0:
                draw.rectangle(
                    [x, y, x + grid_size, y + grid_size],
                    fill=random.choice(colors),
                    outline='#ecf0f1',
                    width=1
                )


async def _create_pdf_artwork(
    brief: str,
    philosophy: str,
    dimensions: Tuple[int, int],
    style: str
) -> str:
    """Create PDF artwork."""
    
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab not available")
    
    width, height = dimensions
    
    # Convert pixels to points (1 inch = 72 points, assuming 96 DPI)
    width_pt = width * 72 / 96
    height_pt = height * 72 / 96
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"canvas_design_{timestamp}.pdf"
    output_path = Path.home() / 'Downloads' / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create PDF
    c = pdf_canvas.Canvas(str(output_path), pagesize=(width_pt, height_pt))
    
    # Draw background
    c.setFillColorRGB(1, 1, 1)  # White
    c.rect(0, 0, width_pt, height_pt, fill=1)
    
    # Draw design elements (simplified for PDF)
    c.setFillColorRGB(0.2, 0.2, 0.2)  # Dark gray
    c.setFont("Helvetica-Bold", 24)
    
    text = brief[:50] if len(brief) <= 50 else brief[:47] + '...'
    text_width = c.stringWidth(text, "Helvetica-Bold", 24)
    c.drawString((width_pt - text_width) / 2, height_pt / 2, text)
    
    c.save()
    
    return str(output_path)
