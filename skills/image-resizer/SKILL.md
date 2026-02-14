---
name: resizing-images
description: "Resize, crop, and convert images using Pillow. Use when the user wants to resize image, crop image, convert image format, thumbnail."
---

# Image Resizer Skill

Resize, crop, and convert images using Pillow. Use when the user wants to resize image, crop image, convert image format, thumbnail.

## Type
base

## Capabilities
- generate

## Reference
For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Parse input parameters
- [ ] Step 2: Execute operation
- [ ] Step 3: Return results
```

## Triggers
- "resize image"
- "crop image"
- "image thumbnail"
- "convert image"
- "scale image"

## Category
content-creation

## Tools

### resize_image_tool
Resize an image to specified dimensions.

**Parameters:**
- `input_path` (str, required): Path to source image
- `output_path` (str, optional): Path for resized image (default: adds _resized suffix)
- `width` (int, optional): Target width in pixels
- `height` (int, optional): Target height in pixels
- `maintain_aspect` (bool, optional): Maintain aspect ratio (default: true)
- `quality` (int, optional): JPEG quality 1-100 (default: 85)
- `format` (str, optional): Output format: JPEG, PNG, WEBP (default: same as input)

**Returns:**
- `success` (bool)
- `output_path` (str): Path to resized image
- `original_size` (dict): Original width and height
- `new_size` (dict): New width and height

## Dependencies
Pillow
