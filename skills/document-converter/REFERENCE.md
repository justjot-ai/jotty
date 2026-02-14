# Document Converter Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`convert_to_pdf_tool`](#convert_to_pdf_tool) | Convert a document to PDF format using Pandoc. |
| [`convert_to_epub_tool`](#convert_to_epub_tool) | Convert a document to EPUB format using Pandoc. |
| [`convert_to_docx_tool`](#convert_to_docx_tool) | Convert a document to DOCX format using Pandoc. |
| [`convert_to_html_tool`](#convert_to_html_tool) | Convert a document to HTML format using Pandoc. |
| [`convert_to_markdown_tool`](#convert_to_markdown_tool) | Convert a document to Markdown format using Pandoc. |

---

## `convert_to_pdf_tool`

Convert a document to PDF format using Pandoc.  IMPORTANT: Pandoc can convert FROM markdown, HTML, DOCX, EPUB, LaTeX, etc. but CANNOT convert FROM PDF. Always use the original source file (markdown/HTML/DOCX) as input, not a PDF file.  Supported input formats: markdown (.md), HTML (.html), DOCX (.docx), EPUB (.epub), LaTeX (.tex), reStructuredText (.rst), Textile, MediaWiki, etc.

**Parameters:**

- **input_file** (`str, required`): Path to input file (markdown, HTML, DOCX, etc. - NOT PDF)
- **output_file** (`str, optional`): Output PDF path
- **page_size** (`str, optional`): Page size (a4, a5, a6, letter, remarkable), default: 'a4'
- **title** (`str, optional`): Document title
- **author** (`str, optional`): Document author

**Returns:** Dictionary with: - success (bool): Whether conversion succeeded - output_path (str): Path to generated PDF - error (str, optional): Error message if failed

---

## `convert_to_epub_tool`

Convert a document to EPUB format using Pandoc.

**Parameters:**

- **input_file** (`str, required`): Path to input file
- **output_file** (`str, optional`): Output EPUB path
- **title** (`str, optional`): Document title
- **author** (`str, optional`): Document author

**Returns:** Dictionary with: - success (bool): Whether conversion succeeded - output_path (str): Path to generated EPUB - error (str, optional): Error message if failed

---

## `convert_to_docx_tool`

Convert a document to DOCX format using Pandoc.  IMPORTANT: Pandoc CANNOT convert FROM PDF. Use the original source file (markdown, HTML, etc.) as input, not a PDF file.  Supported input formats: markdown (.md), HTML (.html), EPUB (.epub), LaTeX (.tex), reStructuredText (.rst), Textile, MediaWiki, etc. NOT supported: PDF (.pdf) - Pandoc cannot read PDF files.

**Parameters:**

- **input_file** (`str, required`): Path to input file (markdown, HTML, etc. - NOT PDF)
- **output_file** (`str, optional`): Output DOCX path
- **title** (`str, optional`): Document title

**Returns:** Dictionary with: - success (bool): Whether conversion succeeded - output_path (str): Path to generated DOCX - error (str, optional): Error message if failed

---

## `convert_to_html_tool`

Convert a document to HTML format using Pandoc.  IMPORTANT: Pandoc CANNOT convert FROM PDF. Use the original source file (markdown, DOCX, etc.) as input, not a PDF file.  Supported input formats: markdown (.md), DOCX (.docx), EPUB (.epub), LaTeX (.tex), reStructuredText (.rst), Textile, MediaWiki, etc. NOT supported: PDF (.pdf) - Pandoc cannot read PDF files.

**Parameters:**

- **input_file** (`str, required`): Path to input file (markdown, DOCX, etc. - NOT PDF)
- **output_file** (`str, optional`): Output HTML path
- **title** (`str, optional`): Document title
- **standalone** (`bool, optional`): Generate standalone HTML, default: True

**Returns:** Dictionary with: - success (bool): Whether conversion succeeded - output_path (str): Path to generated HTML - error (str, optional): Error message if failed

---

## `convert_to_markdown_tool`

Convert a document to Markdown format using Pandoc.

**Parameters:**

- **input_file** (`str, required`): Path to input file
- **output_file** (`str, optional`): Output Markdown path

**Returns:** Dictionary with: - success (bool): Whether conversion succeeded - output_path (str): Path to generated Markdown - error (str, optional): Error message if failed
