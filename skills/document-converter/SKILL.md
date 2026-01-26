# Document Converter Skill

## Description
Converts documents between various formats: Markdown, PDF, EPUB, DOCX, HTML. Uses Pandoc for conversions.

## Tools

### convert_to_pdf_tool
Converts a document to PDF format.

**Parameters:**
- `input_file` (str, required): Path to input file (markdown, html, docx, etc.)
- `output_file` (str, optional): Output PDF path (default: auto-generated)
- `page_size` (str, optional): Page size - 'a4', 'a5', 'a6', 'letter', 'remarkable', default: 'a4'
- `title` (str, optional): Document title
- `author` (str, optional): Document author

**Returns:**
- `success` (bool): Whether conversion succeeded
- `output_path` (str): Path to generated PDF
- `error` (str, optional): Error message if failed

### convert_to_epub_tool
Converts a document to EPUB format.

**Parameters:**
- `input_file` (str, required): Path to input file
- `output_file` (str, optional): Output EPUB path
- `title` (str, optional): Document title
- `author` (str, optional): Document author

**Returns:**
- `success` (bool): Whether conversion succeeded
- `output_path` (str): Path to generated EPUB
- `error` (str, optional): Error message if failed

### convert_to_docx_tool
Converts a document to DOCX format.

**Parameters:**
- `input_file` (str, required): Path to input file
- `output_file` (str, optional): Output DOCX path
- `title` (str, optional): Document title

**Returns:**
- `success` (bool): Whether conversion succeeded
- `output_path` (str): Path to generated DOCX
- `error` (str, optional): Error message if failed

### convert_to_html_tool
Converts a document to HTML format.

**Parameters:**
- `input_file` (str, required): Path to input file
- `output_file` (str, optional): Output HTML path
- `title` (str, optional): Document title
- `standalone` (bool, optional): Generate standalone HTML, default: True

**Returns:**
- `success` (bool): Whether conversion succeeded
- `output_path` (str): Path to generated HTML
- `error` (str, optional): Error message if failed

### convert_to_markdown_tool
Converts a document to Markdown format.

**Parameters:**
- `input_file` (str, required): Path to input file
- `output_file` (str, optional): Output Markdown path

**Returns:**
- `success` (bool): Whether conversion succeeded
- `output_path` (str): Path to generated Markdown
- `error` (str, optional): Error message if failed
