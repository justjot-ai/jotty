# DOCX Tools Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`read_docx_tool`](#read_docx_tool) | Read text content from a Word document. |
| [`create_docx_tool`](#create_docx_tool) | Create a new Word document with proper markdown formatting. |
| [`create_professional_checklist_tool`](#create_professional_checklist_tool) | Create a beautifully formatted professional checklist document. |
| [`add_paragraph_tool`](#add_paragraph_tool) | Add a paragraph to an existing Word document. |
| [`add_table_tool`](#add_table_tool) | Add a table to an existing Word document. |
| [`add_image_tool`](#add_image_tool) | Add an image to an existing Word document. |
| [`replace_text_tool`](#replace_text_tool) | Find and replace text in a Word document. |
| [`get_styles_tool`](#get_styles_tool) | List available styles in a Word document. |

---

## `read_docx_tool`

Read text content from a Word document.

**Parameters:**

- **file_path** (`str, required`): Path to the .docx file
- **include_tables** (`bool, optional`): Include table content (default: True)

**Returns:** Dictionary with: - success (bool): Whether reading succeeded - text (str): Extracted text content - paragraphs (list): List of paragraph texts - tables (list): List of tables (if include_tables is True) - error (str, optional): Error message if failed

---

## `create_docx_tool`

Create a new Word document with proper markdown formatting.  Supports markdown conversion: - # Header → Heading 1 - ## Header → Heading 2 - ### Header → Heading 3 - - [ ] item → Checkbox (unchecked) - - [x] item → Checkbox (checked) - - item or * item → Bullet point - 1. item → Numbered list - **bold** → Bold text - *italic* → Italic text

**Parameters:**

- **content** (`str or list, required`): Markdown content or list of paragraphs
- **output_path** (`str, required`): Output file path
- **title** (`str, optional`): Document title (added as Heading1)

**Returns:** Dictionary with: - success (bool): Whether creation succeeded - file_path (str): Path to created document - error (str, optional): Error message if failed

---

## `create_professional_checklist_tool`

Create a beautifully formatted professional checklist document.  Features: - Professional color scheme (deep blue headers, clean styling) - Table-based checklist with columns: Item, Reference, Status - Form fields at top (Name, Date, Reviewer, Period) - Section hierarchy (Parts, numbered sections) - Checkboxes with proper formatting

**Parameters:**

- **content** (`str, required`): Markdown content with checklist items
- **output_path** (`str, required`): Output file path
- **title** (`str, optional`): Main title
- **subtitle** (`str, optional`): Subtitle
- **organization** (`str, optional`): Organization name for header
- **include_form_fields** (`bool, optional`): Add form fields at top (default: True)

**Returns:** Dictionary with success status and file path

---

## `add_paragraph_tool`

Add a paragraph to an existing Word document.

**Parameters:**

- **file_path** (`str, required`): Path to the .docx file
- **text** (`str, required`): Paragraph text to add
- **style** (`str, optional`): Paragraph style (Normal, Heading1, Heading2, etc.)

**Returns:** Dictionary with: - success (bool): Whether addition succeeded - file_path (str): Path to modified document - error (str, optional): Error message if failed

---

## `add_table_tool`

Add a table to an existing Word document.

**Parameters:**

- **file_path** (`str, required`): Path to the .docx file
- **data** (`list, required`): List of lists representing table rows
- **headers** (`list, optional`): List of header strings (if provided, adds header row)

**Returns:** Dictionary with: - success (bool): Whether addition succeeded - file_path (str): Path to modified document - rows (int): Number of rows added - cols (int): Number of columns - error (str, optional): Error message if failed

---

## `add_image_tool`

Add an image to an existing Word document.

**Parameters:**

- **file_path** (`str, required`): Path to the .docx file
- **image_path** (`str, required`): Path to the image file
- **width** (`float, optional`): Image width in inches (default: 6.0)

**Returns:** Dictionary with: - success (bool): Whether addition succeeded - file_path (str): Path to modified document - image_path (str): Path to the image that was added - error (str, optional): Error message if failed

---

## `replace_text_tool`

Find and replace text in a Word document.

**Parameters:**

- **file_path** (`str, required`): Path to the .docx file
- **find** (`str, required`): Text to find
- **replace** (`str, required`): Text to replace with

**Returns:** Dictionary with: - success (bool): Whether replacement succeeded - file_path (str): Path to modified document - replacements (int): Number of replacements made - error (str, optional): Error message if failed

---

## `get_styles_tool`

List available styles in a Word document.

**Parameters:**

- **file_path** (`str, required`): Path to the .docx file

**Returns:** Dictionary with: - success (bool): Whether listing succeeded - styles (list): List of style names with their types - paragraph_styles (list): List of paragraph style names - character_styles (list): List of character style names - error (str, optional): Error message if failed
