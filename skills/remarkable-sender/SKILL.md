# reMarkable Sender Skill

Send PDFs to reMarkable tablet via cloud API.

## Description

This skill uploads PDFs to reMarkable tablet using rmapi (reMarkable cloud API client).

## Features

- Upload PDFs to reMarkable cloud
- Support for folders
- Automatic registration check
- Uses existing rmapi configuration

## Usage

```python
from skills.remarkable_sender.tools import send_to_remarkable_tool

result = await send_to_remarkable_tool({
    'file_path': '/path/to/document.pdf',
    'folder': '/Reports',  # Optional
    'document_name': 'My Report'  # Optional
})
```

## Parameters

- `file_path` (str, required): Path to PDF file
- `folder` (str, optional): reMarkable folder path (default: '/')
- `document_name` (str, optional): Document name (default: filename)

## Requirements

- rmapi installed and configured
- Device registered with reMarkable cloud
- See: https://github.com/juruen/rmapi

## Setup

1. Install rmapi: `go install github.com/juruen/rmapi@latest`
2. Register device: Get code from https://my.remarkable.com/device/browser/connect
3. Run: `rmapi` (will prompt for code)
