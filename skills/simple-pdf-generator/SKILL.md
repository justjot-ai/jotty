---
name: simple-pdf-generator
description: "Template for converting text/markdown content to PDF. Use when the user wants to create pdf, generate pdf, convert to pdf."
---

# Simple PDF Generator

Template for converting text/markdown content to PDF.

## Type
derived

## Base Skills
- document-converter


## Capabilities
- document

## Use Cases

- Generate PDFs from text content
- Convert markdown documents to PDF
- Create reports programmatically
- Batch PDF generation

## Tools

### generate_pdf_tool

Generate PDF from text/markdown content.

**Parameters:**
- `content` (str, required): Text or markdown content
- `output_file` (str, optional): Output PDF path (auto-generated if not provided)
- `input_format` (str, optional): 'text' or 'markdown' (default: 'markdown')
- `page_size` (str, optional): 'a4', 'letter', etc. (default: 'a4')
- `cleanup_temp` (bool, optional): Delete temp files (default: True)

**Returns:**
- `success` (bool): Whether generation succeeded
- `pdf_path` (str): Path to generated PDF
- `file_size` (int): Size of PDF file in bytes
- `input_format` (str): Format used
- `page_size` (str): Page size used

## Example

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
skill = registry.get_skill('simple-pdf-generator')
result = await skill.tools['generate_pdf_tool']({
    'content': '# My Document\n\nThis is a test.',
    'output_file': 'output.pdf'
})

if result['success']:
    print(f"PDF created: {result['pdf_path']}")
```

## Customization

See `customization.md` for how to customize this template.

## Triggers
- "simple pdf generator"
- "create pdf"
- "generate pdf"
- "convert to pdf"
- "pdf"

## Category
document-creation
