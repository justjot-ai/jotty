# PDF Tools Skill

## Description
Comprehensive PDF toolkit for extracting, manipulating, and creating PDF files. Uses pdfplumber for extraction, pypdf for manipulation, and reportlab for creation.


## Type
base


## Capabilities
- document

## Tools

### extract_text_tool
Extract text content from PDF pages.

**Parameters:**
- `file_path` (str, required): Path to the PDF file
- `pages` (list, optional): List of page numbers (0-indexed) to extract
- `layout` (bool, optional): Preserve layout/formatting (default: False)

### extract_tables_tool
Extract tables from PDF as structured data.

**Parameters:**
- `file_path` (str, required): Path to the PDF file
- `pages` (list, optional): List of page numbers (0-indexed) to extract from
- `output_format` (str, optional): 'json', 'csv', or 'dataframe' (default: 'json')

### merge_pdfs_tool
Merge multiple PDF files into a single document.

**Parameters:**
- `file_paths` (list, required): List of PDF file paths to merge
- `output_path` (str, required): Path for the merged output PDF

### split_pdf_tool
Split a PDF into separate page files.

**Parameters:**
- `file_path` (str, required): Path to the PDF file to split
- `pages` (list/str, required): Page numbers (1-indexed) as list [1,3,5] or range "1-5"
- `output_dir` (str, required): Directory to save split pages

### get_metadata_tool
Get metadata and information from a PDF file.

**Parameters:**
- `file_path` (str, required): Path to the PDF file

### rotate_pages_tool
Rotate pages in a PDF file.

**Parameters:**
- `file_path` (str, required): Path to the PDF file
- `rotation` (int, required): Rotation angle (90, 180, or 270 degrees)
- `pages` (list, optional): Page numbers to rotate (1-indexed). Rotates all if not provided.
- `output_path` (str, optional): Output file path. Creates '_rotated' suffix if not provided.

### create_pdf_tool
Create a PDF from text or markdown content.

**Parameters:**
- `content` (str, required): Text or markdown content to convert
- `output_path` (str, required): Output file path for the PDF
- `title` (str, optional): Document title
- `page_size` (str, optional): 'A4', 'LETTER', 'LEGAL', 'A4-LANDSCAPE', 'LETTER-LANDSCAPE' (default: 'A4')

**Supported Markdown:**
- `# Heading 1`, `## Heading 2`, `### Heading 3`
- Bullet points with `-` or `*`
- `**bold**` and `*italic*` text
- `---` or `***` for page breaks

## Requirements
- `pdfplumber` - for text and table extraction
- `pypdf` - for PDF manipulation (merge, split, rotate, metadata)
- `reportlab` - for PDF creation

Install all dependencies:
```bash
pip install pdfplumber pypdf reportlab
```

## Triggers
- "pdf tools"
- "create pdf"
- "generate pdf"
- "convert to pdf"
- "pdf"

## Category
document-creation
