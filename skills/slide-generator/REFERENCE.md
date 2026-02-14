# Slide Generator Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`generate_slides_pdf_tool`](#generate_slides_pdf_tool) | Generate PDF slides directly using reportlab (slide-style pages). |
| [`generate_slides_tool`](#generate_slides_tool) | Generate PowerPoint slides from structured content. |
| [`generate_slides_from_topic_tool`](#generate_slides_from_topic_tool) | Generate PowerPoint slides from a topic using AI. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`wrap_pdf_text`](#wrap_pdf_text) | Wrap text for PDF slides. |
| [`draw_background`](#draw_background) | No description available. |
| [`wrap_text`](#wrap_text) | Wrap long text to multiple lines. |

---

## `generate_slides_pdf_tool`

Generate PDF slides directly using reportlab (slide-style pages).

**Parameters:**

- **title** (`str, required`): Presentation title
- **slides** (`list, required`): List of slide dicts with 'title' and 'bullets'
- **subtitle** (`str, optional`): Subtitle for title slide
- **output_path** (`str, optional`): Output directory
- **template** (`str, optional`): Color theme - 'dark', 'light', 'blue'

**Returns:** Dictionary with success, file_path, slide_count

---

## `generate_slides_tool`

Generate PowerPoint slides from structured content.

**Parameters:**

- **title** (`str, required`): Presentation title
- **slides** (`list, required`): List of slide dicts with 'title' and 'bullets'
- **subtitle** (`str, optional`): Subtitle for title slide
- **author** (`str, optional`): Author name
- **output_path** (`str, optional`): Output file path
- **template** (`str, optional`): Color theme - 'dark', 'light', 'blue' (default: 'dark')

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - file_path (str): Path to generated PPTX file - slide_count (int): Number of slides created - error (str, optional): Error message if failed

---

## `generate_slides_from_topic_tool`

Generate PowerPoint slides from a topic using AI.

**Parameters:**

- **topic** (`str, required`): Topic to create slides about
- **n_slides** (`int, optional`): Number of content slides (default: 10)
- **template** (`str, optional`): Color theme - 'dark', 'light', 'blue'
- **output_path** (`str, optional`): Output directory
- **export_as** (`str, optional`): 'pptx', 'pdf', or 'both' (default: 'pptx')
- **send_telegram** (`bool, optional`): Send to Telegram (default: True)

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - file_path (str): Path to generated PPTX file - pdf_path (str): Path to PDF version (if export_as includes pdf) - slide_count (int): Number of slides - telegram_sent (bool): Whether sent to Telegram

---

## `wrap_pdf_text`

Wrap text for PDF slides.

**Parameters:**

- **text**
- **max_width_chars**

---

## `draw_background`

No description available.

---

## `wrap_text`

Wrap long text to multiple lines.

**Parameters:**

- **text**
- **max_chars**
