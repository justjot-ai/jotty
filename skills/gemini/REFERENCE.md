# Google Gemini Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`generate_text_tool`](#generate_text_tool) | Generate text using Google Gemini API. |
| [`generate_with_image_tool`](#generate_with_image_tool) | Generate text from an image and prompt using Gemini vision capabilities. |
| [`chat_tool`](#chat_tool) | Multi-turn chat conversation using Google Gemini API. |
| [`list_models_tool`](#list_models_tool) | List available Gemini models and their capabilities. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`generate_content`](#generate_content) | Generate content using Gemini API. |
| [`chat`](#chat) | Multi-turn chat conversation. |
| [`extract_text`](#extract_text) | Extract text content from Gemini API response. |

---

## `generate_text_tool`

Generate text using Google Gemini API.

**Parameters:**

- **prompt** (`str, required`): Text generation prompt
- **model** (`str, optional`): Gemini model - 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash' (default: 'gemini-1.5-flash')
- **temperature** (`float, optional`): Sampling temperature 0.0-1.0 (default: 0.7)
- **max_tokens** (`int, optional`): Maximum output tokens

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - text (str): Generated text - model (str): Model used - error (str, optional): Error message if failed

---

## `generate_with_image_tool`

Generate text from an image and prompt using Gemini vision capabilities.

**Parameters:**

- **prompt** (`str, required`): Text prompt describing what to do with the image
- **image_path** (`str, required`): Path to the image file
- **model** (`str, optional`): Gemini model (default: 'gemini-1.5-flash')
- **temperature** (`float, optional`): Sampling temperature 0.0-1.0 (default: 0.7)
- **max_tokens** (`int, optional`): Maximum output tokens

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - text (str): Generated text - model (str): Model used - image_path (str): Path to the processed image - error (str, optional): Error message if failed

---

## `chat_tool`

Multi-turn chat conversation using Google Gemini API.

**Parameters:**

- **messages** (`list, required`): List of message dicts with 'role' and 'content'
- **role**: 'user' or 'assistant'/'model'
- **content**: Message text
- **model** (`str, optional`): Gemini model (default: 'gemini-1.5-flash')
- **temperature** (`float, optional`): Sampling temperature 0.0-1.0 (default: 0.7)
- **max_tokens** (`int, optional`): Maximum output tokens

**Returns:** Dictionary with: - success (bool): Whether chat succeeded - text (str): Assistant's response - model (str): Model used - message_count (int): Number of messages in conversation - error (str, optional): Error message if failed

---

## `list_models_tool`

List available Gemini models and their capabilities.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - success (bool): Always True - models (list): List of available models with details

---

## `generate_content`

Generate content using Gemini API.

**Parameters:**

- **prompt** (`str`)
- **model** (`str`)
- **temperature** (`float`)
- **max_tokens** (`Optional[int]`)
- **image_data** (`Optional[str]`)
- **image_mime_type** (`str`)

**Returns:** API response with generated content

---

## `chat`

Multi-turn chat conversation.

**Parameters:**

- **messages** (`List[Dict[str, str]]`)
- **model** (`str`)
- **temperature** (`float`)
- **max_tokens** (`Optional[int]`)

**Returns:** API response with generated content

---

## `extract_text`

Extract text content from Gemini API response.

**Parameters:**

- **response** (`Dict[str, Any]`)

**Returns:** `str`
