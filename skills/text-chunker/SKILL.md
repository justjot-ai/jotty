# Text Chunker Skill

## Description
Splits text into semantic chunks for RAG (Retrieval Augmented Generation) systems. Uses intelligent chunking that preserves context and structure.

## Tools

### chunk_text_tool
Splits text into chunks with configurable size and overlap.

**Parameters:**
- `text` (str, required): Text to chunk
- `chunk_size` (int, optional): Maximum chunk size in characters, default: 500
- `chunk_overlap` (int, optional): Overlap between chunks in characters, default: 100
- `separators` (list, optional): Custom separators for chunking (default: markdown-aware)
- `preserve_headers` (bool, optional): Include headers in chunk context, default: True
- `token_limit` (int, optional): Use token limit instead of character limit (requires tiktoken)

**Returns:**
- `success` (bool): Whether chunking succeeded
- `chunks` (list): List of chunk dicts with text, index, char_count, token_count
- `count` (int): Number of chunks created
- `error` (str, optional): Error message if failed
