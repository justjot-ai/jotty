---
name: processing-docx
description: "Word document toolkit using python-docx for reading, creating, and manipulating Word documents (.docx files)."
---

# DOCX Tools Skill

## Description
Word document toolkit using python-docx for reading, creating, and manipulating Word documents (.docx files).


## Type
base


## Capabilities
- document

## Tools

### read_docx_tool
Read text content from a Word document.

**Parameters:**
- `file_path` (str, required): Path to the .docx file
- `include_tables` (bool, optional): Include table content (default: True)

### create_docx_tool
Create a new Word document.

**Parameters:**
- `content` (str or list, required): Text content or list of paragraphs
- `output_path` (str, required): Output file path
- `title` (str, optional): Document title (added as Heading1)

### add_paragraph_tool
Add a paragraph to an existing Word document.

**Parameters:**
- `file_path` (str, required): Path to the .docx file
- `text` (str, required): Paragraph text to add
- `style` (str, optional): Paragraph style (Normal, Heading1, Heading2, etc.)

### add_table_tool
Add a table to an existing Word document.

**Parameters:**
- `file_path` (str, required): Path to the .docx file
- `data` (list, required): List of lists representing table rows
- `headers` (list, optional): List of header strings

### add_image_tool
Add an image to an existing Word document.

**Parameters:**
- `file_path` (str, required): Path to the .docx file
- `image_path` (str, required): Path to the image file
- `width` (float, optional): Image width in inches (default: 6.0)

### replace_text_tool
Find and replace text in a Word document.

**Parameters:**
- `file_path` (str, required): Path to the .docx file
- `find` (str, required): Text to find
- `replace` (str, required): Text to replace with

### get_styles_tool
List available styles in a Word document.

**Parameters:**
- `file_path` (str, required): Path to the .docx file

## Requirements
- `python-docx` library

Install with: `pip install python-docx`

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Load document
- [ ] Step 2: Inspect content
- [ ] Step 3: Modify content
- [ ] Step 4: Export document
```

**Step 1: Load document**
Open the target Word document for processing.

**Step 2: Inspect content**
List sections, paragraphs, tables, and images.

**Step 3: Modify content**
Update text, add tables, insert images, or change formatting.

**Step 4: Export document**
Save the modified document to the desired format.

## Triggers
- "docx tools"

## Category
document-creation
