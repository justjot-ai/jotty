# Presenton AI Presentation Generator Skill

## Description
Integrates with [Presenton](https://github.com/presenton/presenton), an open-source AI presentation generator that runs locally via Docker. Generate professional presentations from text prompts, with support for multiple LLM providers and export to PPTX/PDF.

## Features
- **AI-Powered Generation**: Create presentations from text descriptions
- **Docker Integration**: Automatic container management (start/stop)
- **Multiple LLM Support**: OpenAI, Anthropic Claude, Google Gemini, Ollama
- **Research Integration**: Generate presentations from web research
- **Export Formats**: PPTX and PDF output
- **Template Support**: Various design templates available

## Tools

### check_presenton_status_tool
Check if Presenton Docker container is running and API is accessible.

**Parameters:**
- `container_name` (str, optional): Docker container name (default: 'presenton')
- `base_url` (str, optional): Presenton API URL (default: 'http://localhost:5000')

**Returns:**
- `success` (bool): Whether check succeeded
- `container_running` (bool): Whether Docker container is running
- `api_accessible` (bool): Whether API is responding

---

### start_presenton_tool
Start the Presenton Docker container.

**Parameters:**
- `container_name` (str, optional): Docker container name (default: 'presenton')
- `port` (int, optional): Port to expose (default: 5000)
- `data_dir` (str, optional): Directory for app data (default: '~/presenton_data')
- `image` (str, optional): Docker image (default: 'ghcr.io/presenton/presenton:latest')
- `llm` (str, optional): LLM provider - 'openai', 'google', 'anthropic', 'ollama' (default: 'anthropic')
- `openai_api_key` (str, optional): OpenAI API key
- `google_api_key` (str, optional): Google API key
- `anthropic_api_key` (str, optional): Anthropic API key
- `image_provider` (str, optional): Image provider - 'pexels', 'pixabay', 'dalle3'
- `force_recreate` (bool, optional): Remove and recreate container if exists (default: False)

**Returns:**
- `success` (bool): Whether container started successfully
- `url` (str): URL to access Presenton
- `container_name` (str): Container name

---

### stop_presenton_tool
Stop the Presenton Docker container.

**Parameters:**
- `container_name` (str, optional): Docker container name (default: 'presenton')
- `remove` (bool, optional): Remove container after stopping (default: False)

**Returns:**
- `success` (bool): Whether container stopped successfully
- `removed` (bool): Whether container was removed

---

### generate_presentation_tool
Generate an AI-powered presentation using Presenton.

**Parameters:**
- `content` (str, required): Topic or content for the presentation
- `n_slides` (int, optional): Number of slides (default: 8)
- `language` (str, optional): Target language (default: 'English')
- `template` (str, optional): Design template name
- `export_as` (str, optional): Export format - 'pptx' or 'pdf' (default: 'pptx')
- `tone` (str, optional): Content tone - 'professional', 'casual', 'funny', 'formal', 'inspirational'
- `web_search` (bool, optional): Enable web grounding for research (default: False)
- `base_url` (str, optional): Presenton API URL (default: 'http://localhost:5000')
- `auto_start` (bool, optional): Auto-start container if not running (default: True)
- `timeout` (int, optional): Request timeout in seconds (default: 300)

**Returns:**
- `success` (bool): Whether generation succeeded
- `presentation_id` (str): Unique presentation ID
- `file_path` (str): Path to generated presentation file
- `edit_url` (str): URL to edit presentation in browser
- `download_url` (str): URL to download presentation

---

### list_templates_tool
List available presentation templates in Presenton.

**Parameters:**
- `base_url` (str, optional): Presenton API URL (default: 'http://localhost:5000')

**Returns:**
- `success` (bool): Whether request succeeded
- `templates` (list): List of available templates

---

### generate_presentation_from_research_tool
Generate a presentation from web research on a topic. Combines web search with presentation generation for data-backed presentations.

**Parameters:**
- `topic` (str, required): Research topic for the presentation
- `n_slides` (int, optional): Number of slides (default: 10)
- `language` (str, optional): Target language (default: 'English')
- `template` (str, optional): Design template name
- `export_as` (str, optional): Export format - 'pptx' or 'pdf' (default: 'pptx')
- `tone` (str, optional): Content tone (default: 'professional')
- `max_search_results` (int, optional): Max search results to use (default: 10)

**Returns:**
- `success` (bool): Whether generation succeeded
- `presentation_id` (str): Unique presentation ID
- `file_path` (str): Path to generated presentation
- `research_summary` (str): Summary of research used

---

### download_presentation_tool
Download a generated presentation file.

**Parameters:**
- `presentation_id` (str, required): Presentation ID to download
- `output_path` (str, optional): Path to save file (default: ~/presentations/)
- `format` (str, optional): Download format - 'pptx' or 'pdf' (default: 'pptx')
- `base_url` (str, optional): Presenton API URL

**Returns:**
- `success` (bool): Whether download succeeded
- `file_path` (str): Path to downloaded file

## Usage

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('presenton')

# Check status
status_tool = skill.tools['check_presenton_status_tool']
status = status_tool({})

# Start container if needed
if not status.get('api_accessible'):
    start_tool = skill.tools['start_presenton_tool']
    start_tool({'anthropic_api_key': 'your-api-key'})

# Generate presentation
gen_tool = skill.tools['generate_presentation_tool']
result = gen_tool({
    'content': 'Introduction to Machine Learning: Key concepts, algorithms, and applications',
    'n_slides': 12,
    'tone': 'professional',
    'export_as': 'pptx'
})

if result['success']:
    print(f"Presentation created: {result['file_path']}")
    print(f"Edit at: {result['edit_url']}")
```

## Workflow Example

1. **Start Presenton** (if not running):
   ```python
   start_tool({'llm': 'anthropic', 'anthropic_api_key': os.environ['ANTHROPIC_API_KEY']})
   ```

2. **Generate Presentation**:
   ```python
   gen_tool({'content': 'AI in Healthcare', 'n_slides': 10, 'web_search': True})
   ```

3. **Download Result**:
   ```python
   download_tool({'presentation_id': 'abc123', 'output_path': '~/presentations'})
   ```

4. **Stop Container** (when done):
   ```python
   stop_tool({'remove': False})
   ```

## Requirements

- Docker installed and running
- API key for LLM provider (Anthropic, OpenAI, or Google)
- `requests` Python library
- Optional: `web-search` skill for research-based presentations

## Environment Variables

The skill will automatically use these environment variables if set:
- `ANTHROPIC_API_KEY`: For Claude models
- `OPENAI_API_KEY`: For OpenAI models
- `GOOGLE_API_KEY`: For Google Gemini models

## Supported LLM Providers

| Provider | Models | Environment Variable |
|----------|--------|---------------------|
| Anthropic | claude-3-5-sonnet | ANTHROPIC_API_KEY |
| OpenAI | gpt-4.1 | OPENAI_API_KEY |
| Google | gemini-2.0-flash | GOOGLE_API_KEY |
| Ollama | Various local models | (local, no key needed) |

## Image Providers

- `pexels` - Free stock images (default)
- `pixabay` - Free stock images
- `dalle3` - OpenAI DALL-E 3
- `gemini_flash` - Google Gemini image generation
