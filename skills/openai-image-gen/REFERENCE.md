# OpenAI Image Generation Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`generate_image_tool`](#generate_image_tool) | Generate an image from a text prompt using DALL-E 3. |
| [`edit_image_tool`](#edit_image_tool) | Edit an existing image with a text prompt using DALL-E 2. |
| [`create_variation_tool`](#create_variation_tool) | Create variations of an existing image using DALL-E 2. |
| [`analyze_image_tool`](#analyze_image_tool) | Analyze an image using VLM (Vision Language Model). |
| [`describe_image_tool`](#describe_image_tool) | Get a detailed description of an image. |
| [`extract_brand_theme_tool`](#extract_brand_theme_tool) | Extract brand theme (colors, fonts, style) from a reference image. |
| [`generate_brand_image_tool`](#generate_brand_image_tool) | Generate an image following brand guidelines. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`generate_image`](#generate_image) | Generate an image using DALL-E 3. |
| [`edit_image`](#edit_image) | Edit an image using DALL-E 2. |
| [`create_variation`](#create_variation) | Create variations of an image using DALL-E 2. |
| [`get_instance`](#get_instance) | No description available. |
| [`analyze`](#analyze) | Analyze image with VLM via litellm. |
| [`describe`](#describe) | Get detailed image description. |
| [`extract_brand_theme`](#extract_brand_theme) | Extract brand theme as structured JSON. |

---

## `generate_image_tool`

Generate an image from a text prompt using DALL-E 3.

**Parameters:**

- **prompt** (`str, required`): Text description of the image to generate
- **size** (`str, optional`): Image size - '1024x1024', '1792x1024', or '1024x1792' (default: '1024x1024')
- **quality** (`str, optional`): Image quality - 'standard' or 'hd' (default: 'standard')
- **style** (`str, optional`): Image style - 'vivid' or 'natural' (default: 'vivid')
- **output_path** (`str, optional`): Output directory (default: ~/jotty/images)

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - image_path (str): Path to the generated image - revised_prompt (str): The revised prompt used by DALL-E 3 - error (str, optional): Error message if failed

---

## `edit_image_tool`

Edit an existing image with a text prompt using DALL-E 2.  Note: The input image must be a square PNG image less than 4MB. If a mask is provided, the transparent areas indicate where the image should be edited.

**Parameters:**

- **image_path** (`str, required`): Path to the image to edit (PNG, square, <4MB)
- **prompt** (`str, required`): Text description of the desired edit
- **mask_path** (`str, optional`): Path to mask image (transparent areas will be edited)
- **size** (`str, optional`): Output size - '256x256', '512x512', or '1024x1024' (default: '1024x1024')
- **n** (`int, optional`): Number of images to generate (default: 1, max: 10)
- **output_path** (`str, optional`): Output directory (default: ~/jotty/images)

**Returns:** Dictionary with: - success (bool): Whether edit succeeded - image_paths (list): Paths to the edited images - error (str, optional): Error message if failed

---

## `create_variation_tool`

Create variations of an existing image using DALL-E 2.  Note: The input image must be a square PNG image less than 4MB.

**Parameters:**

- **image_path** (`str, required`): Path to the source image (PNG, square, <4MB)
- **size** (`str, optional`): Output size - '256x256', '512x512', or '1024x1024' (default: '1024x1024')
- **n** (`int, optional`): Number of variations to generate (default: 1, max: 10)
- **output_path** (`str, optional`): Output directory (default: ~/jotty/images)

**Returns:** Dictionary with: - success (bool): Whether variation creation succeeded - image_paths (list): Paths to the variation images - error (str, optional): Error message if failed

---

## `analyze_image_tool`

Analyze an image using VLM (Vision Language Model).

**Parameters:**

- **image_path** (`str, required`): Path to image file
- **question** (`str, optional`): What to analyze (default: general description)
- **detail** (`str, optional`): 'low' or 'high' (default: 'high')

**Returns:** Dictionary with success, analysis, model

---

## `describe_image_tool`

Get a detailed description of an image.

**Parameters:**

- **image_path** (`str, required`): Path to image file

**Returns:** Dictionary with success, description, model

---

## `extract_brand_theme_tool`

Extract brand theme (colors, fonts, style) from a reference image.

**Parameters:**

- **image_path** (`str, required`): Path to reference image

**Returns:** Dictionary with success, theme dict

---

## `generate_brand_image_tool`

Generate an image following brand guidelines.

**Parameters:**

- **prompt** (`str, required`): What to generate
- **brand_theme** (`dict, optional`): Brand theme dict with colors/style
- **reference_image_path** (`str, optional`): Extract theme from this image
- **size** (`str, optional`): Image size (default: '1024x1024')
- **quality** (`str, optional`): 'standard' or 'hd'

**Returns:** Dictionary with success, image_path, theme_used, theme_source

---

## `generate_image`

Generate an image using DALL-E 3.

**Parameters:**

- **prompt** (`str`)
- **size** (`str`)
- **quality** (`str`)
- **style** (`str`)
- **model** (`str`)
- **n** (`int`)

**Returns:** `Dict[str, Any]`

---

## `edit_image`

Edit an image using DALL-E 2.

**Parameters:**

- **image_path** (`str`)
- **prompt** (`str`)
- **mask_path** (`Optional[str]`)
- **size** (`str`)
- **n** (`int`)

**Returns:** `Dict[str, Any]`

---

## `create_variation`

Create variations of an image using DALL-E 2.

**Parameters:**

- **image_path** (`str`)
- **size** (`str`)
- **n** (`int`)

**Returns:** `Dict[str, Any]`

---

## `get_instance`

No description available.

---

## `analyze`

Analyze image with VLM via litellm.

**Parameters:**

- **image_path** (`str`)
- **question** (`str`)
- **detail** (`str`)
- **max_tokens** (`int`)

**Returns:** `Dict[str, Any]`

---

## `describe`

Get detailed image description.

**Parameters:**

- **image_path** (`str`)

**Returns:** `Dict[str, Any]`

---

## `extract_brand_theme`

Extract brand theme as structured JSON.

**Parameters:**

- **image_path** (`str`)

**Returns:** `Dict[str, Any]`
