# Summarize Skill

Text summarization skill using Claude CLI for summarizing text, URLs, and files.

## Description

This skill provides comprehensive text summarization capabilities using Claude CLI. It can summarize plain text, fetch and summarize web pages, and extract content from various file formats including txt, md, pdf, and html.


## Type
base


## Capabilities
- analyze

## Tools

### summarize_text_tool

Summarize text content directly.

**Parameters:**
- `text` (str, required): Text content to summarize
- `length` (str, optional): Summary length - 'short', 'medium', 'long' (default: 'medium')
- `style` (str, optional): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
- `model` (str, optional): Claude model - 'sonnet', 'opus', 'haiku' (default: 'sonnet')

**Returns:**
- `success` (bool): Whether summarization succeeded
- `summary` (str): Generated summary
- `length` (str): Summary length used
- `style` (str): Output style used
- `input_length` (int): Character count of input text

### summarize_url_tool

Fetch and summarize webpage content from a URL.

**Parameters:**
- `url` (str, required): URL to fetch and summarize
- `length` (str, optional): Summary length - 'short', 'medium', 'long' (default: 'medium')
- `style` (str, optional): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
- `model` (str, optional): Claude model (default: 'sonnet')
- `timeout` (int, optional): URL fetch timeout in seconds (default: 30)

**Returns:**
- `success` (bool): Whether summarization succeeded
- `summary` (str): Generated summary
- `url` (str): Source URL
- `title` (str): Page title
- `content_length` (int): Character count of fetched content

### summarize_file_tool

Summarize file content from txt, md, pdf, or html files.

**Parameters:**
- `file_path` (str, required): Path to the file to summarize
- `length` (str, optional): Summary length - 'short', 'medium', 'long' (default: 'medium')
- `style` (str, optional): Output style - 'bullet', 'paragraph', 'numbered' (default: 'paragraph')
- `model` (str, optional): Claude model (default: 'sonnet')

**Returns:**
- `success` (bool): Whether summarization succeeded
- `summary` (str): Generated summary
- `file_path` (str): Source file path
- `file_type` (str): Detected file type
- `content_length` (int): Character count of extracted content
- `page_count` (int, optional): Number of pages (for PDFs)

### extract_key_points_tool

Extract key points from text content.

**Parameters:**
- `text` (str, required): Text content to analyze
- `max_points` (int, optional): Maximum number of key points (default: 5, max: 20)
- `model` (str, optional): Claude model (default: 'sonnet')

**Returns:**
- `success` (bool): Whether extraction succeeded
- `key_points` (str): Extracted key points as numbered list
- `max_points` (int): Maximum points requested
- `input_length` (int): Character count of input text

## Usage

```python
from skills.summarize.tools import summarize_text_tool, summarize_url_tool

# Summarize text
result = summarize_text_tool({
    'text': 'Long text content to summarize...',
    'length': 'short',
    'style': 'bullet'
})
print(result['summary'])

# Summarize a URL
result = summarize_url_tool({
    'url': 'https://example.com/article',
    'length': 'medium',
    'style': 'paragraph'
})
print(result['summary'])

# Summarize a file
result = summarize_file_tool({
    'file_path': '/path/to/document.pdf',
    'length': 'long'
})
print(result['summary'])

# Extract key points
result = extract_key_points_tool({
    'text': 'Text to analyze...',
    'max_points': 10
})
print(result['key_points'])
```

## Length Options

- **short**: 2-3 sentences, approximately 50-75 words
- **medium**: 1-2 paragraphs, approximately 150-250 words
- **long**: Detailed summary covering all main points, approximately 400-600 words

## Style Options

- **bullet**: Bullet points with clear, concise items
- **paragraph**: Flowing prose paragraphs
- **numbered**: Numbered list of key points

## Supported File Types

- `.txt` - Plain text files
- `.md`, `.markdown` - Markdown files
- `.rst` - reStructuredText files
- `.pdf` - PDF documents (requires PyPDF2)
- `.html` - HTML files

## Requirements

- Claude CLI installed and authenticated (`claude auth login`)
- `claude-cli-llm` skill available in registry
- `requests` library for URL fetching
- `beautifulsoup4` and `html2text` for HTML parsing
- `PyPDF2` for PDF file support (optional)

## Dependencies

This skill uses the `claude-cli-llm` skill from the registry for LLM calls.

## Architecture

The skill uses a service class pattern (`SummarizationService`) for code organization and reusability. The service lazily loads the Claude CLI skill from the registry on first use.

```
SummarizationService
    ├── summarize()           - Core summarization method
    ├── extract_key_points()  - Key point extraction
    ├── fetch_url_content()   - URL content fetching
    └── read_file_content()   - File content extraction

Tool Functions (exported)
    ├── summarize_text_tool
    ├── summarize_url_tool
    ├── summarize_file_tool
    └── extract_key_points_tool
```
