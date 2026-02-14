"""
GIF Creator Skill

GIF creation toolkit using Pillow for creating animated GIFs,
text animations, loading spinners, and optimizing GIFs.
"""
import os
import logging
import math
from typing import Dict, Any, List, Tuple

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("gif-creator")


logger = logging.getLogger(__name__)


class GIFCreator:
    """Utility class for GIF creation operations."""

    @staticmethod
    def _ensure_pillow():
        """Check if Pillow is available and return the module."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            return Image, ImageDraw, ImageFont
        except ImportError:
            return None, None, None

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def _get_font(size: int = 24):
        """Get a font, falling back to default if necessary."""
        try:
            from PIL import ImageFont
            # Try common font paths
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]
            for path in font_paths:
                if os.path.exists(path):
                    return ImageFont.truetype(path, size)
            return ImageFont.load_default()
        except Exception:
            from PIL import ImageFont
            return ImageFont.load_default()


@tool_wrapper()
def create_gif_from_images_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an animated GIF from a list of image files.

    Args:
        params: Dictionary containing:
            - image_paths (list, required): List of image file paths
            - output_path (str, required): Output path for the GIF
            - duration (int, optional): Duration per frame in ms (default: 500)
            - loop (int, optional): Number of loops, 0 = infinite (default: 0)

    Returns:
        Dictionary with success status, file_path, and frame_count
    """
    status.set_callback(params.pop('_status_callback', None))

    Image, ImageDraw, ImageFont = GIFCreator._ensure_pillow()
    if Image is None:
        return {'success': False, 'error': 'Pillow not installed. Install with: pip install Pillow'}

    image_paths = params.get('image_paths', [])
    output_path = params.get('output_path')
    duration = params.get('duration', 500)
    loop = params.get('loop', 0)

    if not image_paths:
        return {'success': False, 'error': 'image_paths parameter is required'}
    if not output_path:
        return {'success': False, 'error': 'output_path parameter is required'}

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        frames = []
        for path in image_paths:
            if not os.path.exists(path):
                return {'success': False, 'error': f'Image file not found: {path}'}
            img = Image.open(path).convert('RGBA')
            frames.append(img)

        if not frames:
            return {'success': False, 'error': 'No valid images found'}

        # Convert to palette mode for GIF
        palette_frames = []
        for frame in frames:
            # Convert RGBA to RGB with white background
            if frame.mode == 'RGBA':
                background = Image.new('RGB', frame.size, (255, 255, 255))
                background.paste(frame, mask=frame.split()[3])
                frame = background
            palette_frames.append(frame.convert('P', palette=Image.ADAPTIVE, colors=256))

        # Save GIF
        palette_frames[0].save(
            output_path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=duration,
            loop=loop,
            optimize=True
        )

        logger.info(f"GIF created from {len(frames)} images: {output_path}")

        return {
            'success': True,
            'file_path': output_path,
            'frame_count': len(frames),
            'duration_per_frame': duration
        }

    except Exception as e:
        logger.error(f"GIF creation failed: {e}", exc_info=True)
        return {'success': False, 'error': f'GIF creation failed: {str(e)}'}


@tool_wrapper()
def create_text_animation_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an animated text GIF with various effects.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to animate
            - output_path (str, required): Output path for the GIF
            - animation (str, optional): 'fade', 'slide', 'bounce', 'typewriter'
            - size (tuple, optional): Image size (width, height)
            - bg_color (str, optional): Background color hex
            - text_color (str, optional): Text color hex

    Returns:
        Dictionary with success status and file_path
    """
    status.set_callback(params.pop('_status_callback', None))

    Image, ImageDraw, ImageFont = GIFCreator._ensure_pillow()
    if Image is None:
        return {'success': False, 'error': 'Pillow not installed. Install with: pip install Pillow'}

    text = params.get('text')
    output_path = params.get('output_path')
    animation = params.get('animation', 'fade')
    size = params.get('size', (400, 200))
    bg_color = params.get('bg_color', '#ffffff')
    text_color = params.get('text_color', '#000000')

    if not text:
        return {'success': False, 'error': 'text parameter is required'}
    if not output_path:
        return {'success': False, 'error': 'output_path parameter is required'}

    # Handle size as list or tuple
    if isinstance(size, list):
        size = tuple(size)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        bg_rgb = GIFCreator._hex_to_rgb(bg_color)
        text_rgb = GIFCreator._hex_to_rgb(text_color)

        frames = []
        n_frames = 20
        font = GIFCreator._get_font(32)

        if animation == 'fade':
            # Fade in animation
            for i in range(n_frames):
                img = Image.new('RGB', size, bg_rgb)
                draw = ImageDraw.Draw(img)
                alpha = int(255 * (i / (n_frames - 1)))
                # Blend text color with background based on alpha
                blended = tuple(
                    int(bg_rgb[j] + (text_rgb[j] - bg_rgb[j]) * (alpha / 255))
                    for j in range(3)
                )
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (size[0] - text_width) // 2
                y = (size[1] - text_height) // 2
                draw.text((x, y), text, fill=blended, font=font)
                frames.append(img)

        elif animation == 'slide':
            # Slide in from left
            bbox = ImageDraw.Draw(Image.new('RGB', size)).textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            y = (size[1] - text_height) // 2

            for i in range(n_frames):
                img = Image.new('RGB', size, bg_rgb)
                draw = ImageDraw.Draw(img)
                progress = i / (n_frames - 1)
                # Ease out cubic
                progress = 1 - pow(1 - progress, 3)
                x = int(-text_width + (text_width + (size[0] - text_width) // 2) * progress)
                draw.text((x, y), text, fill=text_rgb, font=font)
                frames.append(img)

        elif animation == 'bounce':
            # Bounce animation
            bbox = ImageDraw.Draw(Image.new('RGB', size)).textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (size[0] - text_width) // 2
            center_y = (size[1] - text_height) // 2

            for i in range(n_frames):
                img = Image.new('RGB', size, bg_rgb)
                draw = ImageDraw.Draw(img)
                # Bounce using sine wave
                offset = int(30 * abs(math.sin(i * math.pi / (n_frames / 2))))
                y = center_y - offset
                draw.text((x, y), text, fill=text_rgb, font=font)
                frames.append(img)

        elif animation == 'typewriter':
            # Typewriter effect
            bbox = ImageDraw.Draw(Image.new('RGB', size)).textbbox((0, 0), text, font=font)
            text_height = bbox[3] - bbox[1]
            y = (size[1] - text_height) // 2

            for i in range(len(text) + 5):  # +5 for pause at end
                img = Image.new('RGB', size, bg_rgb)
                draw = ImageDraw.Draw(img)
                partial_text = text[:min(i + 1, len(text))]
                partial_bbox = draw.textbbox((0, 0), partial_text, font=font)
                partial_width = partial_bbox[2] - partial_bbox[0]
                x = (size[0] - partial_width) // 2
                draw.text((x, y), partial_text, fill=text_rgb, font=font)
                frames.append(img)
        else:
            return {'success': False, 'error': f"Unknown animation type: {animation}. Use 'fade', 'slide', 'bounce', or 'typewriter'"}

        # Convert to palette mode and save
        palette_frames = [f.convert('P', palette=Image.ADAPTIVE, colors=256) for f in frames]

        duration = 100 if animation != 'typewriter' else 150

        palette_frames[0].save(
            output_path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )

        logger.info(f"Text animation GIF created: {output_path}")

        return {
            'success': True,
            'file_path': output_path,
            'animation': animation,
            'frame_count': len(frames)
        }

    except Exception as e:
        logger.error(f"Text animation creation failed: {e}", exc_info=True)
        return {'success': False, 'error': f'Text animation creation failed: {str(e)}'}


@tool_wrapper()
def create_loading_spinner_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a loading spinner GIF animation.

    Args:
        params: Dictionary containing:
            - output_path (str, required): Output path for the GIF
            - size (int, optional): Size in pixels (default: 64)
            - color (str, optional): Spinner color hex (default: '#0066cc')
            - style (str, optional): 'dots', 'circle', 'bars' (default: 'dots')

    Returns:
        Dictionary with success status and file_path
    """
    status.set_callback(params.pop('_status_callback', None))

    Image, ImageDraw, ImageFont = GIFCreator._ensure_pillow()
    if Image is None:
        return {'success': False, 'error': 'Pillow not installed. Install with: pip install Pillow'}

    output_path = params.get('output_path')
    size = params.get('size', 64)
    color = params.get('color', '#0066cc')
    style = params.get('style', 'dots')

    if not output_path:
        return {'success': False, 'error': 'output_path parameter is required'}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        color_rgb = GIFCreator._hex_to_rgb(color)
        frames = []
        n_frames = 12

        if style == 'dots':
            # Rotating dots spinner
            n_dots = 8
            for frame_idx in range(n_frames):
                img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
                draw = ImageDraw.Draw(img)
                center = size // 2
                radius = size // 3
                dot_radius = size // 16

                for dot_idx in range(n_dots):
                    angle = (2 * math.pi * dot_idx / n_dots) - (math.pi / 2)
                    angle += (2 * math.pi * frame_idx / n_frames)
                    x = center + int(radius * math.cos(angle))
                    y = center + int(radius * math.sin(angle))

                    # Fade based on position
                    alpha = int(255 * ((n_dots - dot_idx) / n_dots))
                    dot_color = (*color_rgb, alpha)

                    draw.ellipse(
                        [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                        fill=dot_color
                    )
                frames.append(img)

        elif style == 'circle':
            # Arc spinner
            for frame_idx in range(n_frames):
                img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
                draw = ImageDraw.Draw(img)
                padding = size // 8
                start_angle = int(360 * frame_idx / n_frames)

                draw.arc(
                    [padding, padding, size - padding, size - padding],
                    start=start_angle,
                    end=start_angle + 270,
                    fill=color_rgb,
                    width=size // 8
                )
                frames.append(img)

        elif style == 'bars':
            # Equalizer-style bars
            n_bars = 5
            bar_width = size // (n_bars * 2)
            spacing = size // n_bars

            for frame_idx in range(n_frames):
                img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
                draw = ImageDraw.Draw(img)

                for bar_idx in range(n_bars):
                    # Animate bar heights with phase offset
                    phase = (frame_idx / n_frames + bar_idx / n_bars) * 2 * math.pi
                    height_factor = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(phase))
                    bar_height = int(size * 0.8 * height_factor)

                    x = spacing // 2 + bar_idx * spacing
                    y = size - (size - bar_height) // 2 - bar_height

                    draw.rectangle(
                        [x, y, x + bar_width, y + bar_height],
                        fill=(*color_rgb, 255)
                    )
                frames.append(img)
        else:
            return {'success': False, 'error': f"Unknown style: {style}. Use 'dots', 'circle', or 'bars'"}

        # Convert RGBA to palette with transparency
        palette_frames = []
        for frame in frames:
            # Convert to RGB with white background for GIF
            background = Image.new('RGB', frame.size, (255, 255, 255))
            if frame.mode == 'RGBA':
                background.paste(frame, mask=frame.split()[3])
            else:
                background.paste(frame)
            palette_frames.append(background.convert('P', palette=Image.ADAPTIVE, colors=256))

        palette_frames[0].save(
            output_path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=80,
            loop=0,
            optimize=True,
            transparency=255
        )

        logger.info(f"Loading spinner GIF created: {output_path}")

        return {
            'success': True,
            'file_path': output_path,
            'style': style,
            'size': size,
            'frame_count': len(frames)
        }

    except Exception as e:
        logger.error(f"Loading spinner creation failed: {e}", exc_info=True)
        return {'success': False, 'error': f'Loading spinner creation failed: {str(e)}'}


@tool_wrapper()
def resize_gif_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resize a GIF for specific platforms like Slack.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Input GIF file path
            - output_path (str, required): Output path for resized GIF
            - size (str, optional): 'emoji' (128x128), 'message' (480x480), 'custom'
            - width (int, optional): Custom width (required if size='custom')
            - height (int, optional): Custom height (required if size='custom')

    Returns:
        Dictionary with success status, file_path, and new dimensions
    """
    status.set_callback(params.pop('_status_callback', None))

    Image, ImageDraw, ImageFont = GIFCreator._ensure_pillow()
    if Image is None:
        return {'success': False, 'error': 'Pillow not installed. Install with: pip install Pillow'}

    file_path = params.get('file_path')
    output_path = params.get('output_path')
    size_preset = params.get('size', 'emoji')
    width = params.get('width')
    height = params.get('height')

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}
    if not output_path:
        return {'success': False, 'error': 'output_path parameter is required'}
    if not os.path.exists(file_path):
        return {'success': False, 'error': f'Input file not found: {file_path}'}

    # Determine target size
    size_presets = {
        'emoji': (128, 128),
        'message': (480, 480)
    }

    if size_preset == 'custom':
        if not width or not height:
            return {'success': False, 'error': "width and height required when size='custom'"}
        target_size = (width, height)
    else:
        target_size = size_presets.get(size_preset)
        if not target_size:
            return {'success': False, 'error': f"Unknown size preset: {size_preset}. Use 'emoji', 'message', or 'custom'"}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with Image.open(file_path) as img:
            frames = []
            durations = []

            try:
                while True:
                    # Get frame duration
                    duration = img.info.get('duration', 100)
                    durations.append(duration)

                    # Resize frame
                    frame = img.copy()
                    frame = frame.convert('RGBA')
                    frame = frame.resize(target_size, Image.LANCZOS)

                    # Convert to palette
                    background = Image.new('RGB', frame.size, (255, 255, 255))
                    if frame.mode == 'RGBA':
                        background.paste(frame, mask=frame.split()[3])
                    else:
                        background.paste(frame)
                    frames.append(background.convert('P', palette=Image.ADAPTIVE, colors=256))

                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if not frames:
                return {'success': False, 'error': 'No frames found in GIF'}

            # Use first frame's duration if all same, otherwise use frame-specific
            avg_duration = sum(durations) // len(durations)

            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=avg_duration,
                loop=0,
                optimize=True
            )

        logger.info(f"GIF resized to {target_size}: {output_path}")

        return {
            'success': True,
            'file_path': output_path,
            'width': target_size[0],
            'height': target_size[1],
            'frame_count': len(frames)
        }

    except Exception as e:
        logger.error(f"GIF resize failed: {e}", exc_info=True)
        return {'success': False, 'error': f'GIF resize failed: {str(e)}'}


@tool_wrapper()
def optimize_gif_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize GIF file size by reducing colors and frame rate.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Input GIF file path
            - output_path (str, required): Output path for optimized GIF
            - colors (int, optional): Number of colors, 2-256 (default: 128)
            - fps (int, optional): Target frames per second (default: 10)

    Returns:
        Dictionary with success status, file sizes, and reduction percentage
    """
    status.set_callback(params.pop('_status_callback', None))

    Image, ImageDraw, ImageFont = GIFCreator._ensure_pillow()
    if Image is None:
        return {'success': False, 'error': 'Pillow not installed. Install with: pip install Pillow'}

    file_path = params.get('file_path')
    output_path = params.get('output_path')
    colors = params.get('colors', 128)
    fps = params.get('fps', 10)

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}
    if not output_path:
        return {'success': False, 'error': 'output_path parameter is required'}
    if not os.path.exists(file_path):
        return {'success': False, 'error': f'Input file not found: {file_path}'}

    # Validate colors
    colors = max(2, min(256, colors))

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        original_size = os.path.getsize(file_path)
        target_duration = 1000 // fps  # ms per frame

        with Image.open(file_path) as img:
            frames = []
            original_durations = []

            try:
                while True:
                    duration = img.info.get('duration', 100)
                    original_durations.append(duration)
                    frame = img.copy().convert('RGB')
                    frames.append(frame)
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if not frames:
                return {'success': False, 'error': 'No frames found in GIF'}

            # Skip frames to match target FPS
            if original_durations:
                avg_original_duration = sum(original_durations) / len(original_durations)
                skip_factor = max(1, int(target_duration / avg_original_duration))
            else:
                skip_factor = 1

            # Select frames
            selected_frames = frames[::skip_factor] if skip_factor > 1 else frames

            if not selected_frames:
                selected_frames = [frames[0]]

            # Convert to reduced palette
            palette_frames = [
                f.convert('P', palette=Image.ADAPTIVE, colors=colors)
                for f in selected_frames
            ]

            palette_frames[0].save(
                output_path,
                save_all=True,
                append_images=palette_frames[1:],
                duration=target_duration,
                loop=0,
                optimize=True
            )

        new_size = os.path.getsize(output_path)
        reduction = ((original_size - new_size) / original_size) * 100

        logger.info(f"GIF optimized: {original_size} -> {new_size} bytes ({reduction:.1f}% reduction)")

        return {
            'success': True,
            'file_path': output_path,
            'original_size': original_size,
            'new_size': new_size,
            'reduction_percent': round(reduction, 1),
            'original_frames': len(frames),
            'new_frames': len(palette_frames),
            'colors': colors,
            'fps': fps
        }

    except Exception as e:
        logger.error(f"GIF optimization failed: {e}", exc_info=True)
        return {'success': False, 'error': f'GIF optimization failed: {str(e)}'}


@tool_wrapper()
def extract_frames_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract individual frames from a GIF file.

    Args:
        params: Dictionary containing:
            - file_path (str, required): Input GIF file path
            - output_dir (str, required): Directory to save extracted frames

    Returns:
        Dictionary with success status, frame count, and list of frame paths
    """
    status.set_callback(params.pop('_status_callback', None))

    Image, ImageDraw, ImageFont = GIFCreator._ensure_pillow()
    if Image is None:
        return {'success': False, 'error': 'Pillow not installed. Install with: pip install Pillow'}

    file_path = params.get('file_path')
    output_dir = params.get('output_dir')

    if not file_path:
        return {'success': False, 'error': 'file_path parameter is required'}
    if not output_dir:
        return {'success': False, 'error': 'output_dir parameter is required'}
    if not os.path.exists(file_path):
        return {'success': False, 'error': f'Input file not found: {file_path}'}

    try:
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        frame_paths = []

        with Image.open(file_path) as img:
            frame_idx = 0
            try:
                while True:
                    frame = img.copy().convert('RGBA')
                    frame_path = os.path.join(output_dir, f"{base_name}_frame_{frame_idx:04d}.png")
                    frame.save(frame_path, 'PNG')
                    frame_paths.append(frame_path)
                    frame_idx += 1
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

        logger.info(f"Extracted {len(frame_paths)} frames from {file_path}")

        return {
            'success': True,
            'frame_count': len(frame_paths),
            'output_dir': output_dir,
            'frame_paths': frame_paths
        }

    except Exception as e:
        logger.error(f"Frame extraction failed: {e}", exc_info=True)
        return {'success': False, 'error': f'Frame extraction failed: {str(e)}'}
