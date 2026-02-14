# Visual Inspector Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`visual_inspect_tool`](#visual_inspect_tool) | Analyze an image using a Vision Language Model to extract visual state. |
| [`inspect_file_visually_tool`](#inspect_file_visually_tool) | Open any file and visually inspect it using VLM. |
| [`inspect_pptx_slides_tool`](#inspect_pptx_slides_tool) | Convert PPTX slides to images and analyze ALL slides in parallel with VLM. |
| [`inspect_pdf_pages_tool`](#inspect_pdf_pages_tool) | Convert PDF pages to images and analyze in parallel with VLM. |
| [`inspect_code_for_errors_tool`](#inspect_code_for_errors_tool) | Inspect code file for indentation, syntax, formatting, and logic errors. |
| [`inspect_browser_state_tool`](#inspect_browser_state_tool) | Capture browser screenshot and analyze with VLM. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`analyze_image`](#analyze_image) | Call VLM with an image and prompt. |
| [`analyze_text`](#analyze_text) | Call VLM with text-only prompt (for code inspection). |
| [`analyze_batch`](#analyze_batch) | Analyze multiple images in parallel. |
| [`convert`](#convert) | Convert file to inspectable format. |
| [`pdf_pages_to_images`](#pdf_pages_to_images) | Convert PDF pages to PNG images using pdftoppm or ImageMagick. |
| [`pptx_to_images`](#pptx_to_images) | Convert PPTX slides to images. |
| [`get_visual_verification_guidance`](#get_visual_verification_guidance) | Return Visual Verification Protocol guidance for agent system prompts. |

---

## `visual_inspect_tool`

Analyze an image using a Vision Language Model to extract visual state.

**Parameters:**

- **image_path** (`str, required`): Path to image file
- **question** (`str, optional`): Analysis question
- **task_context** (`str, optional`): Current task
- **goal_context** (`str, optional`): Overall goal

**Returns:** Dictionary with visual_state, model, image_path

---

## `inspect_file_visually_tool`

Open any file and visually inspect it using VLM.

**Parameters:**

- **file_path** (`str, required`): Path to file
- **question** (`str, optional`): Analysis question
- **task_context** (`str, optional`): Current task context

**Returns:** Dictionary with visual_state, file_path, file_type

---

## `inspect_pptx_slides_tool`

Convert PPTX slides to images and analyze ALL slides in parallel with VLM.

**Parameters:**

- **pptx_path** (`str, required`): Path to .pptx file
- **question** (`str, optional`): Analysis question per slide
- **slides** (`list, optional`): Specific slide indices (0-based). None = all.

**Returns:** Dictionary with summary, slide_analyses, total_slides

---

## `inspect_pdf_pages_tool`

Convert PDF pages to images and analyze in parallel with VLM.

**Parameters:**

- **pdf_path** (`str, required`): Path to .pdf file
- **question** (`str, optional`): Analysis question per page
- **pages** (`list, optional`): Specific page indices (0-based). None = all.

**Returns:** Dictionary with summary, page_analyses, total_pages

---

## `inspect_code_for_errors_tool`

Inspect code file for indentation, syntax, formatting, and logic errors.

**Parameters:**

- **file_path** (`str, required`): Path to code file
- **error_context** (`str, optional`): Known error messages

**Returns:** Dictionary with issues, file_path, lines_analyzed, total_lines

---

## `inspect_browser_state_tool`

Capture browser screenshot and analyze with VLM.

**Parameters:**

- **question** (`str, optional`): What to analyze
- **task_context** (`str, optional`): Current task
- **goal_context** (`str, optional`): Overall goal

**Returns:** Dictionary with visual_state, screenshot_path

---

## `analyze_image`

Call VLM with an image and prompt.

**Parameters:**

- **image_path** (`str`)
- **prompt** (`str`)
- **detail** (`str`)
- **max_tokens** (`int`)

**Returns:** `Dict[str, Any]`

---

## `analyze_text`

Call VLM with text-only prompt (for code inspection).

**Parameters:**

- **prompt** (`str`)
- **max_tokens** (`int`)

**Returns:** `Dict[str, Any]`

---

## `analyze_batch`

Analyze multiple images in parallel.

**Parameters:**

- **image_paths** (`List[str]`)
- **prompt** (`str`)
- **detail** (`str`)
- **max_tokens** (`int`)

**Returns:** `List[Dict[str, Any]]`

---

## `convert`

Convert file to inspectable format.

**Parameters:**

- **file_path** (`str`)
- **save_path** (`Optional[str]`)

**Returns:** `Dict[str, Any]`

---

## `pdf_pages_to_images`

Convert PDF pages to PNG images using pdftoppm or ImageMagick.

**Parameters:**

- **pdf_path** (`str`)
- **output_dir** (`str`)

**Returns:** `List[str]`

---

## `pptx_to_images`

Convert PPTX slides to images. Tries LibreOffice, then PDF pipeline.

**Parameters:**

- **pptx_path** (`str`)

**Returns:** `List[str]`

---

## `get_visual_verification_guidance`

Return Visual Verification Protocol guidance for agent system prompts.  Append this to agent instructions when the agent has access to visual inspection tools (browser-automation, visual-inspector).

**Returns:** `str`
