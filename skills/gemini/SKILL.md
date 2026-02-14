---
name: calling-gemini
description: "This skill provides LLM capabilities using Google's Gemini models for text generation, image understanding (vision), and multi-turn chat conversations. Supports multiple Gemini models including gemini-1.5-flash, gemini-1.5-pro, and gemini-2.0-flash."
---

# Google Gemini Skill

Text generation, vision, and chat capabilities using Google Gemini API.

## Description

This skill provides LLM capabilities using Google's Gemini models for text generation,
image understanding (vision), and multi-turn chat conversations. Supports multiple
Gemini models including gemini-1.5-flash, gemini-1.5-pro, and gemini-2.0-flash.


## Type
base


## Capabilities
- analyze
- code

## Usage

```python
from skills.gemini.tools import generate_text_tool, generate_with_image_tool, chat_tool

# Text generation
result = generate_text_tool({
    'prompt': 'Write a haiku about coding',
    'model': 'gemini-1.5-flash',
    'temperature': 0.8
})

# Vision - analyze an image
result = generate_with_image_tool({
    'prompt': 'Describe what you see in this image',
    'image_path': '/path/to/image.jpg',
    'model': 'gemini-1.5-pro'
})

# Multi-turn chat
result = chat_tool({
    'messages': [
        {'role': 'user', 'content': 'Hello, who are you?'},
        {'role': 'assistant', 'content': 'I am Gemini, an AI assistant.'},
        {'role': 'user', 'content': 'Can you help me with Python?'}
    ],
    'model': 'gemini-1.5-flash'
})
```

## Tools

### generate_text_tool

Generate text using Google Gemini API.

**Parameters:**
- `prompt` (str, required): Text generation prompt
- `model` (str, optional): Gemini model (default: 'gemini-1.5-flash')
  - `gemini-1.5-flash` - Fast, efficient for quick tasks
  - `gemini-1.5-pro` - Advanced model for complex tasks
  - `gemini-2.0-flash` - Latest generation fast model
- `temperature` (float, optional): Sampling temperature 0.0-1.0 (default: 0.7)
- `max_tokens` (int, optional): Maximum output tokens

**Returns:**
- `success` (bool): Whether generation succeeded
- `text` (str): Generated text
- `model` (str): Model used
- `error` (str, optional): Error message if failed

### generate_with_image_tool

Generate text from an image and prompt using Gemini vision capabilities.

**Parameters:**
- `prompt` (str, required): Text prompt describing what to do with the image
- `image_path` (str, required): Path to the image file
- `model` (str, optional): Gemini model (default: 'gemini-1.5-flash')
- `temperature` (float, optional): Sampling temperature 0.0-1.0 (default: 0.7)
- `max_tokens` (int, optional): Maximum output tokens

**Supported Image Formats:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)
- BMP (.bmp)

**Returns:**
- `success` (bool): Whether generation succeeded
- `text` (str): Generated text
- `model` (str): Model used
- `image_path` (str): Path to the processed image
- `error` (str, optional): Error message if failed

### chat_tool

Multi-turn chat conversation using Google Gemini API.

**Parameters:**
- `messages` (list, required): List of message dicts with 'role' and 'content'
  - `role`: 'user' or 'assistant'/'model'
  - `content`: Message text
- `model` (str, optional): Gemini model (default: 'gemini-1.5-flash')
- `temperature` (float, optional): Sampling temperature 0.0-1.0 (default: 0.7)
- `max_tokens` (int, optional): Maximum output tokens

**Returns:**
- `success` (bool): Whether chat succeeded
- `text` (str): Assistant's response
- `model` (str): Model used
- `message_count` (int): Number of messages in conversation
- `error` (str, optional): Error message if failed

### list_models_tool

List available Gemini models and their capabilities.

**Parameters:**
- None required

**Returns:**
- `success` (bool): Always True
- `models` (list): List of available models with details
- `default_model` (str): Default model identifier

## Requirements

- Google API key set as environment variable:
  - `GOOGLE_API_KEY` or
  - `GEMINI_API_KEY`
- No additional Python dependencies required (uses urllib)

## API Endpoint

Base URL: `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`

## Models Comparison

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| gemini-1.5-flash | Fast | Good | Quick tasks, simple queries |
| gemini-1.5-pro | Moderate | Excellent | Complex reasoning, detailed analysis |
| gemini-2.0-flash | Fast | Very Good | Latest capabilities, general use |

## Architecture

Uses direct HTTP requests to Google Generative Language API via urllib.
No external dependencies required beyond Python standard library.
The GeminiAPIClient class handles authentication, request formatting, and response parsing.

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Triggers
- "gemini"

## Category
workflow-automation
