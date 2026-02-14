# PowerPoint Editor Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`read_pptx_tool`](#read_pptx_tool) | Read presentation content from a PPTX file. |
| [`get_slide_layouts_tool`](#get_slide_layouts_tool) | Get available slide layouts from a PPTX file. |
| [`add_slide_tool`](#add_slide_tool) | Add a new slide to a presentation. |
| [`update_slide_tool`](#update_slide_tool) | Update content of an existing slide. |
| [`delete_slide_tool`](#delete_slide_tool) | Delete a slide from a presentation. |
| [`reorder_slides_tool`](#reorder_slides_tool) | Reorder slides in a presentation. |
| [`add_image_to_slide_tool`](#add_image_to_slide_tool) | Add an image to a slide. |
| [`extract_text_tool`](#extract_text_tool) | Extract all text from a presentation. |
| [`write_pptx_script_tool`](#write_pptx_script_tool) | Save an LLM-generated PptxGenJS Node. |
| [`execute_pptx_script_tool`](#execute_pptx_script_tool) | Execute a PptxGenJS script via Node. |
| [`read_pptx_to_markdown_tool`](#read_pptx_to_markdown_tool) | Read a PPTX file and convert to Markdown. |
| [`read_pptx_to_js_tool`](#read_pptx_to_js_tool) | Read a PPTX file and convert to PptxGenJS JavaScript code. |
| [`get_pptxgenjs_api_tool`](#get_pptxgenjs_api_tool) | Get PptxGenJS API reference for agent exploration. |
| [`get_chart_data_template_tool`](#get_chart_data_template_tool) | Get chart data template for a given chart type. |
| [`search_pptxgenjs_docs_tool`](#search_pptxgenjs_docs_tool) | Search PptxGenJS documentation for a feature. |
| [`save_file_to_path_tool`](#save_file_to_path_tool) | Copy a generated file to a user-specified location. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`write_script`](#write_script) | Save an LLM-generated PptxGenJS Node. |
| [`execute_script`](#execute_script) | Execute a PptxGenJS script via Node. |
| [`read_to_markdown`](#read_to_markdown) | Read PPTX file and convert to Markdown. |
| [`read_to_js`](#read_to_js) | Read PPTX and convert to PptxGenJS JavaScript code. |
| [`get_api_reference`](#get_api_reference) | Get PptxGenJS API reference for agents. |
| [`get_chart_template`](#get_chart_template) | Get chart data template for a given chart type. |
| [`search_docs`](#search_docs) | Search PptxGenJS docs for a feature. |
| [`save_file`](#save_file) | Copy a generated file to a user-specified location. |

---

## `read_pptx_tool`

Read presentation content from a PPTX file.

**Parameters:**

- **file_path** (`str, required`): Path to the PPTX file
- **include_notes** (`bool, optional`): Include slide notes (default: False)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - slides (list): List of slide content dicts - slide_count (int): Total number of slides - error (str, optional): Error message if failed

---

## `get_slide_layouts_tool`

Get available slide layouts from a PPTX file.

**Parameters:**

- **file_path** (`str, required`): Path to the PPTX file

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - layouts (list): List of layout dicts with index and name - error (str, optional): Error message if failed

---

## `add_slide_tool`

Add a new slide to a presentation.

**Parameters:**

- **file_path** (`str, required`): Path to the PPTX file
- **layout_index** (`int, optional`): Index of slide layout to use (default: 1)
- **title** (`str, optional`): Slide title
- **content** (`str or list, optional`): Slide content (text or list of bullets)
- **position** (`int, optional`): Position to insert slide (default: end)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - slide_index (int): Index of the new slide - slide_count (int): Total slides after addition - error (str, optional): Error message if failed

---

## `update_slide_tool`

Update content of an existing slide.

**Parameters:**

- **file_path** (`str, required`): Path to the PPTX file
- **slide_index** (`int, required`): Index of slide to update (0-based)
- **title** (`str, optional`): New slide title
- **content** (`str or list, optional`): New slide content

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - slide_index (int): Index of updated slide - error (str, optional): Error message if failed

---

## `delete_slide_tool`

Delete a slide from a presentation.

**Parameters:**

- **file_path** (`str, required`): Path to the PPTX file
- **slide_index** (`int, required`): Index of slide to delete (0-based)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - deleted_index (int): Index of deleted slide - slide_count (int): Total slides after deletion - error (str, optional): Error message if failed

---

## `reorder_slides_tool`

Reorder slides in a presentation.

**Parameters:**

- **file_path** (`str, required`): Path to the PPTX file
- **new_order** (`list, required`): List of slide indices in new order e.g., [2, 0, 1] moves slide 2 to first, slide 0 to second, etc.

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - new_order (list): The applied order - slide_count (int): Total number of slides - error (str, optional): Error message if failed

---

## `add_image_to_slide_tool`

Add an image to a slide.

**Parameters:**

- **file_path** (`str, required`): Path to the PPTX file
- **slide_index** (`int, required`): Index of slide to add image to (0-based)
- **image_path** (`str, required`): Path to the image file
- **position** (`dict, optional`): Position with 'left', 'top', 'width', 'height' in inches Default: {'left': 1, 'top': 2, 'width': 6, 'height': 4}

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - slide_index (int): Index of slide with image - image_path (str): Path to added image - error (str, optional): Error message if failed

---

## `extract_text_tool`

Extract all text from a presentation.

**Parameters:**

- **file_path** (`str, required`): Path to the PPTX file

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - text (str): All extracted text joined by newlines - slides_text (list): List of text per slide - total_characters (int): Total character count - error (str, optional): Error message if failed

---

## `write_pptx_script_tool`

Save an LLM-generated PptxGenJS Node.js script to disk.

**Parameters:**

- **script_content** (`str, required`): Full PptxGenJS Node.js script
- **filename** (`str, optional`): Script filename (default: custom_pptx.mjs)

**Returns:** Dictionary with success, script_path, script_size_bytes, next_step

---

## `execute_pptx_script_tool`

Execute a PptxGenJS script via Node.js.

**Parameters:**

- **script_path** (`str, required`): Path to .mjs script
- **timeout** (`int, optional`): Max seconds (default: 60)

**Returns:** Dictionary with success, output, pptx_path

---

## `read_pptx_to_markdown_tool`

Read a PPTX file and convert to Markdown.

**Parameters:**

- **pptx_path** (`str, required`): Path to .pptx file
- **include_notes** (`bool, optional`): Include speaker notes (default: True)
- **include_metadata** (`bool, optional`): Include file metadata (default: True)

**Returns:** Dictionary with success, markdown, slides_count, tables_count, images_count

---

## `read_pptx_to_js_tool`

Read a PPTX file and convert to PptxGenJS JavaScript code.

**Parameters:**

- **pptx_path** (`str, required`): Path to .pptx file
- **output_path** (`str, optional`): Path for generated script
- **preserve_images** (`bool, optional`): Extract images (default: True)

**Returns:** Dictionary with success, script, script_path, slides_extracted

---

## `get_pptxgenjs_api_tool`

Get PptxGenJS API reference for agent exploration.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with charts, text, tables, shapes, images API reference

---

## `get_chart_data_template_tool`

Get chart data template for a given chart type.

**Parameters:**

- **chart_type** (`str, required`): bar, line, pie, doughnut, area, radar

**Returns:** Dictionary with template data and options

---

## `search_pptxgenjs_docs_tool`

Search PptxGenJS documentation for a feature.

**Parameters:**

- **query** (`str, required`): Search query (e.g., 'radar chart', 'table border')

**Returns:** Dictionary with matching doc sections and URLs

---

## `save_file_to_path_tool`

Copy a generated file to a user-specified location.

**Parameters:**

- **source_path** (`str, required`): Path to source file
- **destination_path** (`str, required`): Where to save

**Returns:** Dictionary with success, saved_to, file_size_bytes

---

## `write_script`

Save an LLM-generated PptxGenJS Node.js script to disk.

**Parameters:**

- **script_content**
- **filename**

---

## `execute_script`

Execute a PptxGenJS script via Node.js.

**Parameters:**

- **script_path**
- **timeout**

---

## `read_to_markdown`

Read PPTX file and convert to Markdown.

**Parameters:**

- **pptx_path**
- **include_notes**
- **include_metadata**

---

## `read_to_js`

Read PPTX and convert to PptxGenJS JavaScript code.

**Parameters:**

- **pptx_path**
- **output_path**
- **preserve_images**

---

## `get_api_reference`

Get PptxGenJS API reference for agents.

---

## `get_chart_template`

Get chart data template for a given chart type.

**Parameters:**

- **chart_type**

---

## `search_docs`

Search PptxGenJS docs for a feature.

**Parameters:**

- **query**

---

## `save_file`

Copy a generated file to a user-specified location.

**Parameters:**

- **source_path**
- **destination_path**
