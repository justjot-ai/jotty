---
name: generating-openai-images
description: "Generate, edit, and create variations of images using OpenAI's DALL-E API. Uses DALL-E 3 for generation and DALL-E 2 for editing and variations. Use when the user wants to generate image, create image, image."
---

# OpenAI Image Generation Skill

## Description
Generate, edit, and create variations of images using OpenAI's DALL-E API. Uses DALL-E 3 for generation and DALL-E 2 for editing and variations.


## Type
base


## Capabilities
- media

## Tools

### generate_image_tool
Generate an image from a text prompt using DALL-E 3.

**Parameters:**
- `prompt` (str, required): Text description of the image to generate
- `size` (str, optional): Image size - '1024x1024', '1792x1024', '1024x1792' (default: '1024x1024')
- `quality` (str, optional): Image quality - 'standard' or 'hd' (default: 'standard')
- `style` (str, optional): Image style - 'vivid' or 'natural' (default: 'vivid')
- `output_path` (str, optional): Output directory (default: ~/jotty/images)

**Returns:**
- `image_path`: Path to the generated image
- `revised_prompt`: The revised prompt used by DALL-E 3

### edit_image_tool
Edit an existing image with a text prompt using DALL-E 2.

**Parameters:**
- `image_path` (str, required): Path to the image to edit (PNG, square, <4MB)
- `prompt` (str, required): Text description of the desired edit
- `mask_path` (str, optional): Path to mask image (transparent areas will be edited)
- `size` (str, optional): Output size - '256x256', '512x512', '1024x1024' (default: '1024x1024')
- `n` (int, optional): Number of images to generate (default: 1, max: 10)
- `output_path` (str, optional): Output directory (default: ~/jotty/images)

**Returns:**
- `image_paths`: List of paths to the edited images

### create_variation_tool
Create variations of an existing image using DALL-E 2.

**Parameters:**
- `image_path` (str, required): Path to the source image (PNG, square, <4MB)
- `size` (str, optional): Output size - '256x256', '512x512', '1024x1024' (default: '1024x1024')
- `n` (int, optional): Number of variations to generate (default: 1, max: 10)
- `output_path` (str, optional): Output directory (default: ~/jotty/images)

**Returns:**
- `image_paths`: List of paths to the variation images

## Requirements
- `requests` library
- `OPENAI_API_KEY` environment variable

## Example Usage

```python
# Generate a new image
result = generate_image_tool({
    "prompt": "A futuristic city skyline at sunset with flying cars",
    "size": "1792x1024",
    "quality": "hd",
    "style": "vivid"
})
# Returns: {"success": True, "image_path": "~/jotty/images/dalle3_20260126_123456.png", ...}

# Edit an existing image
result = edit_image_tool({
    "image_path": "/path/to/image.png",
    "prompt": "Add a rainbow in the sky",
    "size": "1024x1024"
})

# Create variations
result = create_variation_tool({
    "image_path": "/path/to/image.png",
    "n": 3,
    "size": "1024x1024"
})
```

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Triggers
- "openai image gen"
- "generate image"
- "create image"
- "image"

## Category
media-creation
