# Text Utilities Skill

## Description
Provides text manipulation, encoding/decoding, formatting, and text processing utilities.


## Type
base


## Capabilities
- analyze

## Tools

### encode_text_tool
Encodes text using various encodings (base64, URL, HTML entities).

**Parameters:**
- `text` (str, required): Text to encode
- `encoding` (str, required): Encoding type - 'base64', 'url', 'html', 'hex'

**Returns:**
- `success` (bool): Whether encoding succeeded
- `encoded` (str): Encoded text
- `error` (str, optional): Error message if failed

### decode_text_tool
Decodes text from various encodings.

**Parameters:**
- `text` (str, required): Encoded text to decode
- `encoding` (str, required): Encoding type - 'base64', 'url', 'html', 'hex'

**Returns:**
- `success` (bool): Whether decoding succeeded
- `decoded` (str): Decoded text
- `error` (str, optional): Error message if failed

### format_text_tool
Formats text in various ways (uppercase, lowercase, title case, etc.).

**Parameters:**
- `text` (str, required): Text to format
- `format` (str, required): Format type - 'upper', 'lower', 'title', 'capitalize', 'swapcase', 'strip', 'reverse'

**Returns:**
- `success` (bool): Whether formatting succeeded
- `formatted` (str): Formatted text
- `error` (str, optional): Error message if failed

### extract_text_tool
Extracts text from various formats (JSON, HTML, markdown, etc.).

**Parameters:**
- `content` (str, required): Content to extract text from
- `format` (str, required): Format type - 'json', 'html', 'markdown', 'plain'

**Returns:**
- `success` (bool): Whether extraction succeeded
- `extracted` (str): Extracted text
- `error` (str, optional): Error message if failed

### count_text_tool
Counts words, characters, lines, etc. in text.

**Parameters:**
- `text` (str, required): Text to analyze
- `count_type` (str, optional): What to count - 'words', 'chars', 'lines', 'sentences', 'paragraphs' (default: 'words')

**Returns:**
- `success` (bool): Whether counting succeeded
- `count` (int): Count result
- `count_type` (str): Type of count performed
- `error` (str, optional): Error message if failed

### replace_text_tool
Finds and replaces text patterns.

**Parameters:**
- `text` (str, required): Text to process
- `find` (str, required): Text to find
- `replace` (str, required): Replacement text
- `case_sensitive` (bool, optional): Case-sensitive matching (default: True)
- `regex` (bool, optional): Use regex pattern (default: False)

**Returns:**
- `success` (bool): Whether replacement succeeded
- `result` (str): Text with replacements
- `replacements` (int): Number of replacements made
- `error` (str, optional): Error message if failed
