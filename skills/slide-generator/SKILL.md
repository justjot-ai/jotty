# Slide Generator Skill

## Description
Generates actual PowerPoint (.pptx) slides using python-pptx. Creates professional presentations with proper slide layouts, not just markdown documents.

## Tools

### generate_slides_tool
Generate PowerPoint slides from structured content.

**Parameters:**
- `title` (str, required): Presentation title
- `slides` (list, required): List of slide dicts with 'title' and 'bullets'
- `subtitle` (str, optional): Subtitle for title slide
- `template` (str, optional): Color theme - 'dark', 'light', 'blue' (default: 'dark')
- `output_path` (str, optional): Output directory

### generate_slides_from_topic_tool
Generate PowerPoint slides from a topic using AI.

**Parameters:**
- `topic` (str, required): Topic to create slides about
- `n_slides` (int, optional): Number of content slides (default: 10)
- `template` (str, optional): Color theme
- `send_telegram` (bool, optional): Send to Telegram (default: True)

## Requirements
- `python-pptx` library
- `claude-cli-llm` skill (for topic-based generation)
- `telegram-sender` skill (optional, for sending)
