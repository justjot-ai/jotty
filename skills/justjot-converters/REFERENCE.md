# JustJot Converters Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`arxiv_to_markdown_tool`](#arxiv_to_markdown_tool) | Convert an arXiv paper to formatted markdown or PDF. |
| [`youtube_to_markdown_tool`](#youtube_to_markdown_tool) | Convert a YouTube video transcript to formatted markdown or PDF. |
| [`html_to_markdown_tool`](#html_to_markdown_tool) | Convert an HTML page to clean markdown. |
| [`send_to_kindle_tool`](#send_to_kindle_tool) | Send a document to Kindle via email. |
| [`kindle_configure_tool`](#kindle_configure_tool) | Configure Kindle email delivery settings. |
| [`kindle_status_tool`](#kindle_status_tool) | Check Kindle email configuration status. |
| [`sync_to_remarkable_tool`](#sync_to_remarkable_tool) | Upload a document to reMarkable device via cloud. |
| [`remarkable_register_tool`](#remarkable_register_tool) | Register device with reMarkable cloud. |
| [`remarkable_status_tool`](#remarkable_status_tool) | Check reMarkable cloud registration status. |

---

## `arxiv_to_markdown_tool`

Convert an arXiv paper to formatted markdown or PDF.

**Parameters:**

- **arxiv_id** (`str, required`): ArXiv paper ID or URL (e.g., "1706.03762" or "https://arxiv.org/abs/1706.03762")
- **output_format** (`str, optional`): "markdown", "pdf", "remarkable", "epub", "kindle" (default: "remarkable")
- **output_dir** (`str, optional`): Output directory path

**Returns:** Dictionary with success, output_path, title, error

---

## `youtube_to_markdown_tool`

Convert a YouTube video transcript to formatted markdown or PDF.

**Parameters:**

- **url** (`str, required`): YouTube video URL
- **output_format** (`str, optional`): "markdown", "pdf", "remarkable", "epub" (default: "markdown")
- **include_timestamps** (`bool, optional`): Include timestamps (default: true)
- **summarize** (`bool, optional`): Generate AI summary (default: false)
- **summary_type** (`str, optional`): "short", "medium", "comprehensive", "study_guide" (default: "comprehensive")
- **output_dir** (`str, optional`): Output directory
- **webshare_username** (`str, optional`): Proxy username
- **webshare_password** (`str, optional`): Proxy password

**Returns:** Dictionary with success, output_path, title, author, duration, error

---

## `html_to_markdown_tool`

Convert an HTML page to clean markdown.

**Parameters:**

- **url** (`str, optional`): URL to fetch and convert
- **content** (`str, optional`): Direct HTML content to convert
- **title** (`str, optional`): Document title (default: "HTML Document")
- **output_path** (`str, optional`): Output file path

**Returns:** Dictionary with success, content, output_path, error

---

## `send_to_kindle_tool`

Send a document to Kindle via email.

**Parameters:**

- **file_path** (`str, required`): Path to document (PDF or EPUB)
- **subject** (`str, optional`): Email subject
- **convert** (`bool, optional`): Request Kindle conversion (default: true)

**Returns:** Dictionary with success, kindle_email, error

---

## `kindle_configure_tool`

Configure Kindle email delivery settings.

**Parameters:**

- **kindle_email** (`str, required`): Your Kindle email (xxxxx@kindle.com)
- **smtp_email** (`str, required`): Your email address
- **smtp_password** (`str, required`): SMTP password (use App Password for Gmail)
- **provider** (`str, optional`): "gmail", "outlook", "yahoo", "custom" (default: "gmail")
- **smtp_server** (`str, optional`): Custom SMTP server (required if provider="custom")
- **smtp_port** (`int, optional`): SMTP port (default: 587)

**Returns:** Dictionary with success, message, error

---

## `kindle_status_tool`

Check Kindle email configuration status.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with configured, kindle_email, smtp_email, provider

---

## `sync_to_remarkable_tool`

Upload a document to reMarkable device via cloud.

**Parameters:**

- **file_path** (`str, required`): Path to PDF file
- **folder_name** (`str, optional`): Folder name on reMarkable (creates if doesn't exist)
- **document_name** (`str, optional`): Document name (uses filename if not provided)

**Returns:** Dictionary with success, document_name, error

---

## `remarkable_register_tool`

Register device with reMarkable cloud.

**Parameters:**

- **one_time_code** (`str, required`): 8-character code from https://my.remarkable.com/device/browser/connect

**Returns:** Dictionary with success, message, error

---

## `remarkable_status_tool`

Check reMarkable cloud registration status.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with registered, rmapy_installed, config_file
