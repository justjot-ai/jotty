# NotebookLM PDF Generator Setup Guide

## Overview

The NotebookLM PDF generator skill uses browser automation to upload markdown/text content to Google NotebookLM and generate PDFs. Since NotebookLM doesn't have a public API yet, we use Playwright for browser automation.

## Installation

### 1. Install Playwright

```bash
pip install playwright
playwright install chromium
```

### 2. First-Time Setup (Sign In)

NotebookLM requires Google authentication. On first use, you need to sign in manually:

```python
from skills.notebooklm_pdf.tools import notebooklm_pdf_tool

# Run with headless=False to see the browser and sign in
result = await notebooklm_pdf_tool({
    'content': '# My Document\n\nContent here...',
    'title': 'My Document',
    'headless': False  # Show browser window
})
```

1. The browser will open
2. Sign in to your Google account when prompted
3. The session will be saved for future use

### 3. Persistent Sessions

After signing in once, your session is saved in `~/.notebooklm_browser`. You can:

- Use `headless=True` for automated runs (default)
- Customize the session directory:
  ```bash
  export NOTEBOOKLM_USER_DATA_DIR=~/.my_notebooklm_session
  ```

## Usage

### Basic Usage

```python
from skills.notebooklm_pdf.tools import notebooklm_pdf_tool

# From markdown content
result = await notebooklm_pdf_tool({
    'content': '# My Document\n\nContent here...',
    'title': 'My Document',
    'author': 'Author Name'
})

# From markdown file
result = await notebooklm_pdf_tool({
    'content_file': '/path/to/document.md',
    'title': 'Document Title',
    'output_dir': '/custom/output/path'
})
```

### Parameters

- `content` (str, optional): Markdown or text content to upload
- `content_file` (str, optional): Path to markdown/text file (alternative to content)
- `title` (str, optional): Document title
- `author` (str, optional): Document author
- `output_file` (str, optional): Output PDF path
- `output_dir` (str, optional): Output directory (defaults to `stock_market/outputs`)
- `headless` (bool, optional): Run browser in headless mode (default: True)
- `use_browser` (bool, optional): Use browser automation (default: True)
- `user_data_dir` (str, optional): Browser user data directory for persistent sessions

### Output

PDFs are saved to `/var/www/sites/personal/stock_market/outputs/` by default.

## How It Works

1. **Browser Automation**: Uses Playwright to control Chromium browser
2. **Content Upload**: Pastes content directly into NotebookLM or uploads file
3. **PDF Export**: Clicks export button and downloads PDF
4. **Fallback**: If browser automation fails, falls back to Pandoc conversion

## Troubleshooting

### "NotebookLM requires sign-in"

**Solution**: Run with `headless=False` first to sign in manually:

```python
result = await notebooklm_pdf_tool({
    'content': '...',
    'headless': False
})
```

After signing in once, future runs can use `headless=True`.

### "Could not find export option"

**Possible causes**:
- NotebookLM interface changed
- Content wasn't uploaded successfully
- Export button not visible

**Solutions**:
1. Run with `headless=False` to see what's happening
2. Check if content was actually uploaded
3. Manually export from NotebookLM if needed

### Browser automation fails

**Fallback**: The skill automatically falls back to Pandoc conversion if browser automation fails. This ensures PDFs are always generated, though without NotebookLM's AI features.

## Environment Variables

- `NOTEBOOKLM_USER_DATA_DIR`: Custom browser session directory
- `NOTEBOOKLM_API_KEY`: (Future) API key when NotebookLM API becomes available

## Limitations

- Requires manual sign-in on first use
- Depends on NotebookLM's web interface (may break if interface changes)
- Browser automation is slower than API would be
- Requires Playwright and Chromium installed

## Future Improvements

When NotebookLM releases an API:
1. The skill will automatically use the API instead of browser automation
2. Set `NOTEBOOKLM_API_KEY` environment variable
3. Browser automation will become a fallback
