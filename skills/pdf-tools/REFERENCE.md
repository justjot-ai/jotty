# PDF Tools Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`extract_text_tool`](#extract_text_tool) | Extract text from a PDF file. |
| [`extract_tables_tool`](#extract_tables_tool) | Extract tables from a PDF file. |
| [`merge_pdfs_tool`](#merge_pdfs_tool) | Merge multiple PDF files into a single PDF. |
| [`split_pdf_tool`](#split_pdf_tool) | Split a PDF into separate page files. |
| [`get_metadata_tool`](#get_metadata_tool) | Get metadata from a PDF file. |
| [`rotate_pages_tool`](#rotate_pages_tool) | Rotate pages in a PDF file. |
| [`create_pdf_tool`](#create_pdf_tool) | Create a PDF from text or markdown content. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`extract_text`](#extract_text) | Extract text from PDF pages. |
| [`extract_tables`](#extract_tables) | Extract tables from PDF pages. |
| [`merge_pdfs`](#merge_pdfs) | Merge multiple PDF files into one. |
| [`split_pdf`](#split_pdf) | Split PDF into separate files. |
| [`rotate_pages`](#rotate_pages) | Rotate pages in a PDF. |
| [`get_metadata`](#get_metadata) | Get metadata from a PDF file. |
| [`create_pdf`](#create_pdf) | Create a PDF from text or markdown content. |

---

## `extract_text_tool`

Extract text from a PDF file.

**Parameters:**

- **file_path** (`str, required`): Path to the PDF file
- **pages** (`list, optional`): List of page numbers (0-indexed) to extract
- **layout** (`bool, optional`): Preserve layout/formatting (default: False)

**Returns:** Dictionary with success status, extracted text content by page

---

## `extract_tables_tool`

Extract tables from a PDF file.

**Parameters:**

- **file_path** (`str, required`): Path to the PDF file
- **pages** (`list, optional`): List of page numbers (0-indexed) to extract from
- **output_format** (`str, optional`): 'json', 'csv', or 'dataframe' (default: 'json')

**Returns:** Dictionary with success status, extracted tables

---

## `merge_pdfs_tool`

Merge multiple PDF files into a single PDF.

**Parameters:**

- **file_paths** (`list, required`): List of PDF file paths to merge
- **output_path** (`str, required`): Path for the merged output PDF

**Returns:** Dictionary with success status, output path, file count

---

## `split_pdf_tool`

Split a PDF into separate page files.

**Parameters:**

- **file_path** (`str, required`): Path to the PDF file to split
- **pages** (`list/str, required`): Page numbers to extract (1-indexed) Can be a list like [1, 3, 5] or a range string like "1-5"
- **output_dir** (`str, required`): Directory to save split pages

**Returns:** Dictionary with success status, output files list

---

## `get_metadata_tool`

Get metadata from a PDF file.

**Parameters:**

- **file_path** (`str, required`): Path to the PDF file

**Returns:** Dictionary with success status, page count, file size, metadata

---

## `rotate_pages_tool`

Rotate pages in a PDF file.

**Parameters:**

- **file_path** (`str, required`): Path to the PDF file
- **rotation** (`int, required`): Rotation angle (90, 180, or 270 degrees)
- **pages** (`list, optional`): Page numbers to rotate (1-indexed). If not provided, rotates all pages.
- **output_path** (`str, optional`): Output file path. If not provided, creates a new file with '_rotated' suffix.

**Returns:** Dictionary with success status, output path, rotation info

---

## `create_pdf_tool`

Create a PDF from text or markdown content.

**Parameters:**

- **content** (`str, required`): Text or markdown content to convert
- **output_path** (`str, required`): Output file path for the PDF
- **title** (`str, optional`): Document title
- **page_size** (`str, optional`): Page size - 'A4', 'LETTER', 'LEGAL', 'A4-LANDSCAPE', 'LETTER-LANDSCAPE' (default: 'A4')

**Returns:** Dictionary with success status, output path, file size Supported Markdown: - # Heading 1, ## Heading 2, ### Heading 3 - Bullet points with - or * - **bold** and *italic* text - --- or *** for page breaks

---

## `extract_text`

Extract text from PDF pages.

**Parameters:**

- **pages** (`Optional[List[int]]`)
- **layout** (`bool`)

**Returns:** `Dict[str, Any]`

---

## `extract_tables`

Extract tables from PDF pages.

**Parameters:**

- **pages** (`Optional[List[int]]`)
- **output_format** (`str`)

**Returns:** `Dict[str, Any]`

---

## `merge_pdfs`

Merge multiple PDF files into one.

**Parameters:**

- **file_paths** (`List[str]`)
- **output_path** (`str`)

**Returns:** `Dict[str, Any]`

---

## `split_pdf`

Split PDF into separate files.

**Parameters:**

- **file_path** (`str`)
- **pages** (`Union[List[int], str]`)
- **output_dir** (`str`)

**Returns:** `Dict[str, Any]`

---

## `rotate_pages`

Rotate pages in a PDF.

**Parameters:**

- **file_path** (`str`)
- **rotation** (`int`)
- **pages** (`Optional[List[int]]`)
- **output_path** (`Optional[str]`)

**Returns:** `Dict[str, Any]`

---

## `get_metadata`

Get metadata from a PDF file.

**Parameters:**

- **file_path** (`str`)

**Returns:** `Dict[str, Any]`

---

## `create_pdf`

Create a PDF from text or markdown content.

**Parameters:**

- **content** (`str`)
- **output_path** (`str`)
- **title** (`Optional[str]`)
- **page_size** (`str`)

**Returns:** `Dict[str, Any]`
