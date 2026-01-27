# GIF Creator Skill

## Description
GIF creation toolkit using Pillow. Create animated GIFs from images, text animations, loading spinners, and optimize GIFs for various platforms like Slack.

## Tools

### create_gif_from_images_tool
Create an animated GIF from a list of image files.

**Parameters:**
- `image_paths` (list, required): List of image file paths to combine into GIF
- `output_path` (str, required): Output path for the GIF file
- `duration` (int, optional): Duration per frame in milliseconds (default: 500)
- `loop` (int, optional): Number of loops, 0 = infinite (default: 0)

### create_text_animation_tool
Create an animated text GIF with various effects.

**Parameters:**
- `text` (str, required): Text to animate
- `output_path` (str, required): Output path for the GIF file
- `animation` (str, optional): Animation type - 'fade', 'slide', 'bounce', 'typewriter' (default: 'fade')
- `size` (tuple, optional): Image size as (width, height) (default: (400, 200))
- `bg_color` (str, optional): Background color (default: '#ffffff')
- `text_color` (str, optional): Text color (default: '#000000')

### create_loading_spinner_tool
Create a loading spinner GIF animation.

**Parameters:**
- `output_path` (str, required): Output path for the GIF file
- `size` (int, optional): Size in pixels (default: 64)
- `color` (str, optional): Spinner color (default: '#0066cc')
- `style` (str, optional): Spinner style - 'dots', 'circle', 'bars' (default: 'dots')

### resize_gif_tool
Resize a GIF for specific platforms like Slack.

**Parameters:**
- `file_path` (str, required): Input GIF file path
- `output_path` (str, required): Output path for resized GIF
- `size` (str, optional): Preset size - 'emoji' (128x128), 'message' (480x480), 'custom' (default: 'emoji')
- `width` (int, optional): Custom width (required if size='custom')
- `height` (int, optional): Custom height (required if size='custom')

### optimize_gif_tool
Optimize GIF file size by reducing colors and frame rate.

**Parameters:**
- `file_path` (str, required): Input GIF file path
- `output_path` (str, required): Output path for optimized GIF
- `colors` (int, optional): Number of colors in palette, 2-256 (default: 128)
- `fps` (int, optional): Target frames per second (default: 10)

### extract_frames_tool
Extract individual frames from a GIF file.

**Parameters:**
- `file_path` (str, required): Input GIF file path
- `output_dir` (str, required): Directory to save extracted frames

## Requirements
- `Pillow` library

Install: `pip install Pillow`
