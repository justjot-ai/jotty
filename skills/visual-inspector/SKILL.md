---
name: inspecting-visuals
description: "Provides on-demand visual inspection capabilities using Vision Language Models (VLM). Analyzes screenshots, code files, PPTX slides, PDF pages, and any image to extract rich state information. Uses litellm for unified VLM access (Claude Sonnet, GPT-4V, etc.)."
---

# Visual Inspector Skill

VLM-powered visual state extraction for screenshots, files, presentations, PDFs, and code.

## Description

Provides on-demand visual inspection capabilities using Vision Language Models (VLM).
Analyzes screenshots, code files, PPTX slides, PDF pages, and any image to extract
rich state information. Uses litellm for unified VLM access (Claude Sonnet, GPT-4V, etc.).


## Type
base


## Capabilities
- media
- code
- document

## Tools

### visual_inspect_tool
Analyze an image using VLM to extract visual state.

**Parameters:**
- `image_path` (str, required): Path to the image file (png, jpg, gif, webp, bmp)
- `question` (str, optional): Specific question about the visual state
- `task_context` (str, optional): Current task being performed
- `goal_context` (str, optional): Overall goal

**Returns:**
- `visual_state` (str): Detailed VLM analysis
- `model` (str): VLM model used
- `image_path` (str): Path analyzed

### inspect_file_visually_tool
Open any file and visually inspect it using VLM.

**Parameters:**
- `file_path` (str, required): Path to file (images, code, text, PDF, SVG)
- `question` (str, optional): Specific analysis question
- `task_context` (str, optional): Current task context

**Returns:**
- `visual_state` (str): Analysis result
- `file_path` (str): File analyzed
- `file_type` (str): Detected file type

### inspect_pptx_slides_tool
Convert PPTX slides to images and analyze ALL slides in parallel with VLM.

**Parameters:**
- `pptx_path` (str, required): Path to .pptx file
- `question` (str, optional): Analysis question for each slide
- `slides` (list, optional): Specific slide indices (0-based). None = all.

**Returns:**
- `summary` (str): Aggregated summary
- `slide_analyses` (list): Per-slide analysis
- `total_slides` (int): Number of slides

### inspect_pdf_pages_tool
Convert PDF pages to images and analyze in parallel with VLM.

**Parameters:**
- `pdf_path` (str, required): Path to .pdf file
- `question` (str, optional): Analysis question for each page
- `pages` (list, optional): Specific page indices (0-based). None = all.

**Returns:**
- `summary` (str): Aggregated summary
- `page_analyses` (list): Per-page analysis
- `total_pages` (int): Number of pages

### inspect_code_for_errors_tool
Inspect code file for indentation, syntax, formatting, and logic errors.

**Parameters:**
- `file_path` (str, required): Path to code file
- `error_context` (str, optional): Known error messages for targeted analysis

**Returns:**
- `issues` (str): Detailed issue report with line references
- `lines_analyzed` (int): Number of lines checked
- `total_lines` (int): Total file lines

### inspect_browser_state_tool
Capture browser screenshot and analyze with VLM.

**Parameters:**
- `question` (str, optional): What to analyze about browser state
- `task_context` (str, optional): Current task description
- `goal_context` (str, optional): Overall goal

**Returns:**
- `visual_state` (str): Analysis result
- `screenshot_path` (str): Path to captured screenshot

## Environment Variables

- `VLM_MODEL`: VLM model to use (default: claude-sonnet-4-5-20250929 via litellm)
- `LITELLM_API_KEY`: API key for litellm
- `LITELLM_BASE_URL`: Base URL for litellm
- `OPENAI_API_KEY`: Fallback API key

## Dependencies

- litellm
- Optional: pygmentize (code rendering), poppler/pdftoppm (PDF), LibreOffice (PPTX), ImageMagick (SVG)

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Capture target
- [ ] Step 2: Analyze visuals
- [ ] Step 3: Detect issues
- [ ] Step 4: Generate report
```

**Step 1: Capture target**
Take a screenshot or load the image to inspect.

**Step 2: Analyze visuals**
Use AI vision to identify UI elements, layout, and content.

**Step 3: Detect issues**
Find visual bugs, accessibility problems, or design inconsistencies.

**Step 4: Generate report**
Produce a structured inspection report with findings.

## Triggers
- "visual inspector"

## Category
media-creation
