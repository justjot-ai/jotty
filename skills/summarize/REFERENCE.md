# Summarize Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`summarize_text_tool`](#summarize_text_tool) | Summarize text content using Claude CLI. |
| [`summarize_url_tool`](#summarize_url_tool) | Summarize webpage content from a URL using Claude CLI. |
| [`summarize_file_tool`](#summarize_file_tool) | Summarize file content (txt, md, pdf) using Claude CLI. |
| [`extract_key_points_tool`](#extract_key_points_tool) | Extract key points from text using Claude CLI. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`fetch_url_content`](#fetch_url_content) | Fetch and extract text content from a URL. |
| [`read_file_content`](#read_file_content) | Read and extract text content from a file. |
| [`summarize`](#summarize) | Summarize text content. |
| [`extract_key_points`](#extract_key_points) | Extract key points from text. |

---

## `summarize_text_tool`

Summarize text content using Claude CLI.

**Parameters:**

- **text** (`str, required`): Text content to summarize
- **length** (`str, optional`): Summary length - 'short', 'medium', 'long' (default: 'medium')
- **style** (`str, optional`): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
- **model** (`str, optional`): Claude model - 'sonnet', 'opus', 'haiku' (default: 'sonnet')

**Returns:** Dictionary with: - success (bool): Whether summarization succeeded - summary (str): Generated summary - length (str): Summary length used - style (str): Output style used - error (str, optional): Error message if failed

---

## `summarize_url_tool`

Summarize webpage content from a URL using Claude CLI.

**Parameters:**

- **url** (`str, required`): URL to fetch and summarize
- **length** (`str, optional`): Summary length - 'short', 'medium', 'long' (default: 'medium')
- **style** (`str, optional`): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
- **model** (`str, optional`): Claude model (default: 'sonnet')
- **timeout** (`int, optional`): URL fetch timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether summarization succeeded - summary (str): Generated summary - url (str): Source URL - title (str): Page title - error (str, optional): Error message if failed

---

## `summarize_file_tool`

Summarize file content (txt, md, pdf) using Claude CLI.

**Parameters:**

- **file_path** (`str, required`): Path to the file to summarize
- **length** (`str, optional`): Summary length - 'short', 'medium', 'long' (default: 'medium')
- **style** (`str, optional`): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
- **model** (`str, optional`): Claude model (default: 'sonnet')

**Returns:** Dictionary with: - success (bool): Whether summarization succeeded - summary (str): Generated summary - file_path (str): Source file path - file_type (str): Detected file type - error (str, optional): Error message if failed

---

## `extract_key_points_tool`

Extract key points from text using Claude CLI.

**Parameters:**

- **text** (`str, required`): Text content to analyze
- **max_points** (`int, optional`): Maximum number of key points (default: 5, max: 20)
- **model** (`str, optional`): Claude model (default: 'sonnet')

**Returns:** Dictionary with: - success (bool): Whether extraction succeeded - key_points (str): Extracted key points as numbered list - max_points (int): Maximum points requested - error (str, optional): Error message if failed

---

## `fetch_url_content`

Fetch and extract text content from a URL.

**Parameters:**

- **url** (`str`)
- **timeout** (`int`)

**Returns:** `Dict[str, Any]`

---

## `read_file_content`

Read and extract text content from a file.

**Parameters:**

- **file_path** (`str`)

**Returns:** `Dict[str, Any]`

---

## `summarize`

Summarize text content.

**Parameters:**

- **text** (`str`)
- **length** (`str`)
- **style** (`str`)
- **model** (`str`)
- **timeout** (`int`)

**Returns:** Dictionary with success status and summary

---

## `extract_key_points`

Extract key points from text.

**Parameters:**

- **text** (`str`)
- **max_points** (`int`)
- **model** (`str`)
- **timeout** (`int`)

**Returns:** Dictionary with success status and key points
