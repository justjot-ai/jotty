# Brand Guidelines Skill

Applies Anthropic's official brand colors and typography to artifacts for consistent brand styling.

## Description

This skill applies Anthropic's brand identity (colors, typography) to documents, presentations, and other artifacts to maintain consistent brand styling.


## Type
derived

## Base Skills
- image-generator

## Tools

### `apply_brand_styling_tool`

Apply brand styling to a document or artifact.

**Parameters:**
- `input_file` (str, required): Path to input file
- `output_file` (str, optional): Output path (default: adds _branded suffix)
- `file_type` (str, optional): File type - 'pptx', 'docx', 'html', 'css' (auto-detected)
- `preserve_content` (bool, optional): Preserve original content (default: True)

**Returns:**
- `success` (bool): Whether styling succeeded
- `input_file` (str): Path to input file
- `output_file` (str): Path to styled output
- `styles_applied` (dict): Details of styles applied
- `error` (str, optional): Error message if failed

## Brand Colors

- **Dark:** `#141413` - Primary text and dark backgrounds
- **Light:** `#faf9f5` - Light backgrounds
- **Mid Gray:** `#b0aea5` - Secondary elements
- **Light Gray:** `#e8e6dc` - Subtle backgrounds
- **Orange:** `#d97757` - Primary accent
- **Blue:** `#6a9bcc` - Secondary accent
- **Green:** `#788c5d` - Tertiary accent

## Typography

- **Headings:** Poppins (Arial fallback)
- **Body Text:** Lora (Georgia fallback)

## Usage Examples

### Apply to Presentation

```python
result = await apply_brand_styling_tool({
    'input_file': 'presentation.pptx',
    'file_type': 'pptx'
})
```

## Dependencies

- `python-pptx`: For PowerPoint styling
- `python-docx`: For Word document styling
