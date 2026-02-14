# Text Utilities Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`encode_text_tool`](#encode_text_tool) | Encode text using various encodings. |
| [`decode_text_tool`](#decode_text_tool) | Decode text from various encodings. |
| [`format_text_tool`](#format_text_tool) | Format text in various ways. |
| [`extract_text_tool`](#extract_text_tool) | Extract text from various formats. |
| [`count_text_tool`](#count_text_tool) | Count words, characters, lines, etc. |
| [`replace_text_tool`](#replace_text_tool) | Find and replace text patterns. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`extract_strings`](#extract_strings) | No description available. |

---

## `encode_text_tool`

Encode text using various encodings.

**Parameters:**

- **text** (`str, required`): Text to encode
- **encoding** (`str, required`): Encoding type - 'base64', 'url', 'html', 'hex'

**Returns:** Dictionary with: - success (bool): Whether encoding succeeded - encoded (str): Encoded text - error (str, optional): Error message if failed

---

## `decode_text_tool`

Decode text from various encodings.

**Parameters:**

- **text** (`str, required`): Encoded text to decode
- **encoding** (`str, required`): Encoding type - 'base64', 'url', 'html', 'hex'

**Returns:** Dictionary with: - success (bool): Whether decoding succeeded - decoded (str): Decoded text - error (str, optional): Error message if failed

---

## `format_text_tool`

Format text in various ways.

**Parameters:**

- **text** (`str, required`): Text to format
- **format** (`str, required`): Format type - 'upper', 'lower', 'title', 'capitalize', 'swapcase', 'strip', 'reverse'

**Returns:** Dictionary with: - success (bool): Whether formatting succeeded - formatted (str): Formatted text - error (str, optional): Error message if failed

---

## `extract_text_tool`

Extract text from various formats.

**Parameters:**

- **content** (`str, required`): Content to extract text from
- **format** (`str, required`): Format type - 'json', 'html', 'markdown', 'plain'

**Returns:** Dictionary with: - success (bool): Whether extraction succeeded - extracted (str): Extracted text - error (str, optional): Error message if failed

---

## `count_text_tool`

Count words, characters, lines, etc. in text.

**Parameters:**

- **text** (`str, required`): Text to analyze
- **count_type** (`str, optional`): What to count - 'words', 'chars', 'lines', 'sentences', 'paragraphs' (default: 'words')

**Returns:** Dictionary with: - success (bool): Whether counting succeeded - count (int): Count result - count_type (str): Type of count performed - error (str, optional): Error message if failed

---

## `replace_text_tool`

Find and replace text patterns.

**Parameters:**

- **text** (`str, required`): Text to process
- **find** (`str, required`): Text to find
- **replace** (`str, required`): Replacement text
- **case_sensitive** (`bool, optional`): Case-sensitive matching (default: True)
- **regex** (`bool, optional`): Use regex pattern (default: False)

**Returns:** Dictionary with: - success (bool): Whether replacement succeeded - result (str): Text with replacements - replacements (int): Number of replacements made - error (str, optional): Error message if failed

---

## `extract_strings`

No description available.

**Parameters:**

- **obj**
