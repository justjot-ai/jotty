# LaTeX Document Generator - StatQuest Style

Generates professional LaTeX documents with proper mathematical notation, styled in the StatQuest format.


## Type
base

## Tools

### `generate_latex_document_tool`

Generates a LaTeX document from structured content and optionally compiles it to PDF.

**Parameters:**
- `title` (str, required): Document title
- `subtitle` (str, optional): Document subtitle
- `author` (str, optional): Document author
- `sections` (list, required): List of section dictionaries with:
  - `title` (str): Section title
  - `level` (int): Section level (1, 2, or 3)
  - `content` (list): List of content blocks (text, equation, list, keybox, example, table, code)
- `output_file` (str, optional): Output .tex filename (default: auto-generated)
- `compile_pdf` (bool, optional): Whether to compile to PDF (default: True)
- `include_toc` (bool, optional): Include table of contents (default: True)
- `color_theme` (dict, optional): Custom color theme with `primary` and `secondary` RGB values

**Returns:**
- `success` (bool): Whether generation succeeded
- `tex_file` (str): Path to generated .tex file
- `pdf_file` (str, optional): Path to generated PDF if compiled
- `error` (str, optional): Error message if failed

**Example:**
```python
result = generate_latex_document_tool({
    'title': 'Machine Learning Basics',
    'subtitle': 'A Comprehensive Guide',
    'sections': [
        {
            'title': 'Introduction',
            'level': 1,
            'content': [
                {'type': 'text', 'content': 'Welcome to ML basics!'},
                {'type': 'keybox', 'title': 'Key Concept', 'content': 'ML is powerful!'}
            ]
        }
    ],
    'compile_pdf': True
})
```
