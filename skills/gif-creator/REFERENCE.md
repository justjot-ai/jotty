# GIF Creator Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`create_gif_from_images_tool`](#create_gif_from_images_tool) | Create an animated GIF from a list of image files. |
| [`create_text_animation_tool`](#create_text_animation_tool) | Create an animated text GIF with various effects. |
| [`create_loading_spinner_tool`](#create_loading_spinner_tool) | Create a loading spinner GIF animation. |
| [`resize_gif_tool`](#resize_gif_tool) | Resize a GIF for specific platforms like Slack. |
| [`optimize_gif_tool`](#optimize_gif_tool) | Optimize GIF file size by reducing colors and frame rate. |
| [`extract_frames_tool`](#extract_frames_tool) | Extract individual frames from a GIF file. |

---

## `create_gif_from_images_tool`

Create an animated GIF from a list of image files.

**Parameters:**

- **image_paths** (`list, required`): List of image file paths
- **output_path** (`str, required`): Output path for the GIF
- **duration** (`int, optional`): Duration per frame in ms (default: 500)
- **loop** (`int, optional`): Number of loops, 0 = infinite (default: 0)

**Returns:** Dictionary with success status, file_path, and frame_count

---

## `create_text_animation_tool`

Create an animated text GIF with various effects.

**Parameters:**

- **text** (`str, required`): Text to animate
- **output_path** (`str, required`): Output path for the GIF
- **animation** (`str, optional`): 'fade', 'slide', 'bounce', 'typewriter'
- **size** (`tuple, optional`): Image size (width, height)
- **bg_color** (`str, optional`): Background color hex
- **text_color** (`str, optional`): Text color hex

**Returns:** Dictionary with success status and file_path

---

## `create_loading_spinner_tool`

Create a loading spinner GIF animation.

**Parameters:**

- **output_path** (`str, required`): Output path for the GIF
- **size** (`int, optional`): Size in pixels (default: 64)
- **color** (`str, optional`): Spinner color hex (default: '#0066cc')
- **style** (`str, optional`): 'dots', 'circle', 'bars' (default: 'dots')

**Returns:** Dictionary with success status and file_path

---

## `resize_gif_tool`

Resize a GIF for specific platforms like Slack.

**Parameters:**

- **file_path** (`str, required`): Input GIF file path
- **output_path** (`str, required`): Output path for resized GIF
- **size** (`str, optional`): 'emoji' (128x128), 'message' (480x480), 'custom'
- **width** (`int, optional`): Custom width (required if size='custom')
- **height** (`int, optional`): Custom height (required if size='custom')

**Returns:** Dictionary with success status, file_path, and new dimensions

---

## `optimize_gif_tool`

Optimize GIF file size by reducing colors and frame rate.

**Parameters:**

- **file_path** (`str, required`): Input GIF file path
- **output_path** (`str, required`): Output path for optimized GIF
- **colors** (`int, optional`): Number of colors, 2-256 (default: 128)
- **fps** (`int, optional`): Target frames per second (default: 10)

**Returns:** Dictionary with success status, file sizes, and reduction percentage

---

## `extract_frames_tool`

Extract individual frames from a GIF file.

**Parameters:**

- **file_path** (`str, required`): Input GIF file path
- **output_dir** (`str, required`): Directory to save extracted frames

**Returns:** Dictionary with success status, frame count, and list of frame paths
