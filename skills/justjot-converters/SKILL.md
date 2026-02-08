# JustJot Converters Skill

## Description
Utility converters for transforming content between formats and distributing documents to devices.
Wraps JustJot.ai's specialized converters for arXiv papers, YouTube videos, HTML pages,
and device sync utilities (Kindle, reMarkable).


## Type
derived

## Base Skills
- document-converter


## Capabilities
- document

## Tools

### arxiv_to_markdown_tool
Convert an arXiv paper to formatted markdown or PDF.

**Parameters:**
- `arxiv_id` (str, required): ArXiv paper ID (e.g., "1706.03762") or URL
- `output_format` (str, optional): Output format - "markdown", "pdf", "remarkable", "epub", "kindle" (default: "remarkable")
- `output_dir` (str, optional): Output directory path

**Returns:**
- `success` (bool): Whether conversion succeeded
- `output_path` (str): Path to generated file
- `title` (str): Paper title
- `error` (str, optional): Error message if failed

### youtube_to_markdown_tool
Convert a YouTube video transcript to formatted markdown or PDF.

**Parameters:**
- `url` (str, required): YouTube video URL
- `output_format` (str, optional): Output format - "markdown", "pdf", "remarkable", "epub" (default: "markdown")
- `include_timestamps` (bool, optional): Include timestamps in transcript (default: true)
- `summarize` (bool, optional): Generate AI summary (default: false)
- `summary_type` (str, optional): Summary type - "short", "medium", "comprehensive", "study_guide" (default: "comprehensive")
- `output_dir` (str, optional): Output directory path
- `webshare_username` (str, optional): Webshare proxy username (for cloud deployments)
- `webshare_password` (str, optional): Webshare proxy password (for cloud deployments)

**Returns:**
- `success` (bool): Whether conversion succeeded
- `output_path` (str): Path to generated file
- `title` (str): Video title
- `author` (str): Channel name
- `duration` (str): Video duration
- `error` (str, optional): Error message if failed

### html_to_markdown_tool
Convert an HTML page to clean markdown.

**Parameters:**
- `url` (str, optional): URL to fetch and convert
- `content` (str, optional): Direct HTML content to convert
- `title` (str, optional): Document title
- `output_path` (str, optional): Output file path

**Returns:**
- `success` (bool): Whether conversion succeeded
- `content` (str): Converted markdown content
- `output_path` (str, optional): Path if saved to file
- `error` (str, optional): Error message if failed

### send_to_kindle_tool
Send a document to Kindle via email.

**Parameters:**
- `file_path` (str, required): Path to document (PDF or EPUB)
- `subject` (str, optional): Email subject (uses filename if not provided)
- `convert` (bool, optional): Request Kindle conversion (default: true)

**Requires Configuration:**
Run `kindle_configure_tool` first to set up email credentials.

**Returns:**
- `success` (bool): Whether email was sent
- `kindle_email` (str): Destination Kindle email
- `error` (str, optional): Error message if failed

### kindle_configure_tool
Configure Kindle email delivery settings.

**Parameters:**
- `kindle_email` (str, required): Your Kindle email (xxxxx@kindle.com)
- `smtp_email` (str, required): Your email address to send from
- `smtp_password` (str, required): SMTP password (use App Password for Gmail)
- `provider` (str, optional): Email provider - "gmail", "outlook", "yahoo", "custom" (default: "gmail")
- `smtp_server` (str, optional): Custom SMTP server (required if provider="custom")
- `smtp_port` (int, optional): SMTP port (default: 587)

**Returns:**
- `success` (bool): Whether configuration was saved
- `message` (str): Instructions for completing setup
- `error` (str, optional): Error message if failed

### kindle_status_tool
Check Kindle email configuration status.

**Parameters:** None

**Returns:**
- `configured` (bool): Whether Kindle email is configured
- `kindle_email` (str, optional): Configured Kindle email
- `smtp_email` (str, optional): Configured sender email
- `provider` (str, optional): Email provider

### sync_to_remarkable_tool
Upload a document to reMarkable device via cloud.

**Parameters:**
- `file_path` (str, required): Path to PDF file
- `folder_name` (str, optional): Folder name on reMarkable (creates if doesn't exist)
- `document_name` (str, optional): Document name (uses filename if not provided)

**Requires Registration:**
Run `remarkable_register_tool` first with a one-time code.

**Returns:**
- `success` (bool): Whether upload succeeded
- `document_name` (str): Name on reMarkable
- `error` (str, optional): Error message if failed

### remarkable_register_tool
Register device with reMarkable cloud.

**Parameters:**
- `one_time_code` (str, required): 8-character code from https://my.remarkable.com/device/browser/connect

**Returns:**
- `success` (bool): Whether registration succeeded
- `message` (str): Registration status
- `error` (str, optional): Error message if failed

### remarkable_status_tool
Check reMarkable cloud registration status.

**Parameters:** None

**Returns:**
- `registered` (bool): Whether device is registered
- `rmapy_installed` (bool): Whether rmapy library is installed
- `config_file` (str): Path to config file

## Examples

### Convert arXiv paper for reMarkable
```python
result = arxiv_to_markdown_tool({
    'arxiv_id': '1706.03762',
    'output_format': 'remarkable'
})
# Creates optimized PDF for reMarkable E Ink display
```

### Get YouTube transcript with AI summary
```python
result = youtube_to_markdown_tool({
    'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'summarize': True,
    'summary_type': 'study_guide'
})
```

### Send document to Kindle
```python
# First configure (once)
kindle_configure_tool({
    'kindle_email': 'your_kindle@kindle.com',
    'smtp_email': 'you@gmail.com',
    'smtp_password': 'your-app-password',
    'provider': 'gmail'
})

# Then send
send_to_kindle_tool({
    'file_path': '/path/to/document.pdf'
})
```

### Upload to reMarkable
```python
# First register (once)
remarkable_register_tool({
    'one_time_code': 'abcd1234'
})

# Then sync
sync_to_remarkable_tool({
    'file_path': '/path/to/document.pdf',
    'folder_name': 'Research Papers'
})
```

## Dependencies
- JustJot.ai utility modules (auto-imported if available)
- Optional: youtube-transcript-api, yt-dlp for YouTube
- Optional: rmapy, rmapi for reMarkable
- Optional: html2text for HTML conversion
