# PowerPoint Editor Skill

## Description
Editing toolkit for existing PowerPoint (.pptx) files using python-pptx. This skill is focused on modifying existing presentations, unlike slide-generator which creates new presentations from scratch.

## Tools

### read_pptx_tool
Read presentation content from a PPTX file.

**Parameters:**
- `file_path` (str, required): Path to the PPTX file
- `include_notes` (bool, optional): Include slide notes (default: False)

**Returns:**
- `slides`: List of slide content with title, content, shapes_count
- `slide_count`: Total number of slides

### get_slide_layouts_tool
Get available slide layouts from a PPTX file.

**Parameters:**
- `file_path` (str, required): Path to the PPTX file

**Returns:**
- `layouts`: List of layout dicts with index and name
- `layout_count`: Number of available layouts

### add_slide_tool
Add a new slide to a presentation.

**Parameters:**
- `file_path` (str, required): Path to the PPTX file
- `layout_index` (int, optional): Index of slide layout to use (default: 1)
- `title` (str, optional): Slide title
- `content` (str or list, optional): Slide content (text or list of bullets)
- `position` (int, optional): Position to insert slide (default: end)

**Returns:**
- `slide_index`: Index of the new slide
- `slide_count`: Total slides after addition

### update_slide_tool
Update content of an existing slide.

**Parameters:**
- `file_path` (str, required): Path to the PPTX file
- `slide_index` (int, required): Index of slide to update (0-based)
- `title` (str, optional): New slide title
- `content` (str or list, optional): New slide content

**Returns:**
- `slide_index`: Index of updated slide

### delete_slide_tool
Delete a slide from a presentation.

**Parameters:**
- `file_path` (str, required): Path to the PPTX file
- `slide_index` (int, required): Index of slide to delete (0-based)

**Returns:**
- `deleted_index`: Index of deleted slide
- `slide_count`: Total slides after deletion

### reorder_slides_tool
Reorder slides in a presentation.

**Parameters:**
- `file_path` (str, required): Path to the PPTX file
- `new_order` (list, required): List of slide indices in new order (e.g., [2, 0, 1] moves slide 2 to first position)

**Returns:**
- `new_order`: The applied order
- `slide_count`: Total number of slides

### add_image_to_slide_tool
Add an image to a slide.

**Parameters:**
- `file_path` (str, required): Path to the PPTX file
- `slide_index` (int, required): Index of slide to add image to (0-based)
- `image_path` (str, required): Path to the image file
- `position` (dict, optional): Position with 'left', 'top', 'width', 'height' in inches (default: left=1, top=2, width=6, height=4)

**Returns:**
- `slide_index`: Index of slide with image
- `image_path`: Path to added image

### extract_text_tool
Extract all text from a presentation.

**Parameters:**
- `file_path` (str, required): Path to the PPTX file

**Returns:**
- `text`: All extracted text joined by newlines
- `slides_text`: List of text per slide
- `total_characters`: Total character count

## Requirements
- `python-pptx` library

Install with:
```bash
pip install python-pptx
```

## Examples

### Read a presentation
```python
result = read_pptx_tool({'file_path': '/path/to/presentation.pptx', 'include_notes': True})
```

### Add a new slide
```python
result = add_slide_tool({
    'file_path': '/path/to/presentation.pptx',
    'layout_index': 1,
    'title': 'New Section',
    'content': ['First point', 'Second point', 'Third point']
})
```

### Update slide content
```python
result = update_slide_tool({
    'file_path': '/path/to/presentation.pptx',
    'slide_index': 2,
    'title': 'Updated Title',
    'content': ['Updated bullet 1', 'Updated bullet 2']
})
```

### Add image to slide
```python
result = add_image_to_slide_tool({
    'file_path': '/path/to/presentation.pptx',
    'slide_index': 1,
    'image_path': '/path/to/image.png',
    'position': {'left': 2, 'top': 3, 'width': 5, 'height': 3}
})
```

### Reorder slides
```python
# Move last slide to first position
result = reorder_slides_tool({
    'file_path': '/path/to/presentation.pptx',
    'new_order': [3, 0, 1, 2]  # For a 4-slide presentation
})
```
