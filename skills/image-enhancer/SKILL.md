---
name: image-enhancer
description: "This skill takes images and screenshots and makes them look better—sharper, clearer, and more professional. Enhances resolution, improves sharpness, reduces artifacts, and optimizes for different use cases. Use when the user wants to generate image, create image, image."
---

# Image Enhancer Skill

Improves image quality by enhancing resolution, sharpness, and clarity. Perfect for screenshots, presentations, and social media.

## Description

This skill takes images and screenshots and makes them look better—sharper, clearer, and more professional. Enhances resolution, improves sharpness, reduces artifacts, and optimizes for different use cases.


## Type
base


## Capabilities
- media

## Tools

### `enhance_image_tool`

Enhance image quality.

**Parameters:**
- `image_path` (str, required): Path to image file
- `output_path` (str, optional): Output path (default: adds _enhanced suffix)
- `target_resolution` (str, optional): Target resolution - '2x', '4k', 'retina', 'original' (default: '2x')
- `enhancements` (list, optional): Enhancements to apply - 'sharpness', 'clarity', 'artifacts', 'all' (default: ['all'])
- `use_case` (str, optional): Use case - 'web', 'print', 'social_media', 'presentation' (default: 'web')
- `preserve_original` (bool, optional): Keep original file (default: True)

**Returns:**
- `success` (bool): Whether enhancement succeeded
- `original_path` (str): Path to original image
- `enhanced_path` (str): Path to enhanced image
- `improvements` (dict): Details of improvements made
- `error` (str, optional): Error message if failed

## Usage Examples

### Basic Enhancement

```python
result = await enhance_image_tool({
    'image_path': 'screenshot.png',
    'target_resolution': '2x'
})
```

### For Social Media

```python
result = await enhance_image_tool({
    'image_path': 'photo.jpg',
    'use_case': 'social_media',
    'target_resolution': '4k'
})
```

## Dependencies

- `PIL` (Pillow): For image processing
- `numpy`: For image manipulation

## Triggers
- "image enhancer"
- "generate image"
- "create image"
- "image"

## Category
media-creation
