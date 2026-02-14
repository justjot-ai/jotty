# Presenton AI Presentation Generator Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`check_presenton_status_tool`](#check_presenton_status_tool) | Check if Presenton Docker container is running and API is accessible. |
| [`start_presenton_tool`](#start_presenton_tool) | Start the Presenton Docker container. |
| [`stop_presenton_tool`](#stop_presenton_tool) | Stop the Presenton Docker container. |
| [`generate_presentation_tool`](#generate_presentation_tool) | Generate an AI-powered presentation using Presenton. |
| [`list_templates_tool`](#list_templates_tool) | List available presentation templates in Presenton. |
| [`generate_presentation_from_research_tool`](#generate_presentation_from_research_tool) | Generate a presentation from web research on a topic. |
| [`download_presentation_tool`](#download_presentation_tool) | Download a generated presentation file. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`health_check`](#health_check) | Check if Presenton API is accessible. |
| [`generate_presentation`](#generate_presentation) | Generate a presentation via API. |
| [`get_templates`](#get_templates) | Get available templates. |
| [`get_presentation`](#get_presentation) | Get presentation details. |

---

## `check_presenton_status_tool`

Check if Presenton Docker container is running and API is accessible.

**Parameters:**

- **container_name** (`str, optional`): Docker container name (default: 'presenton')
- **base_url** (`str, optional`): Presenton API URL (default: 'http://localhost:5000')

**Returns:** Dictionary with: - success (bool): Whether check succeeded - container_running (bool): Whether Docker container is running - api_accessible (bool): Whether API is responding - container_name (str): Container name checked - base_url (str): API URL checked - error (str, optional): Error message if check failed

---

## `start_presenton_tool`

Start the Presenton Docker container.

**Parameters:**

- **container_name** (`str, optional`): Docker container name (default: 'presenton')
- **port** (`int, optional`): Port to expose (default: 5000)
- **data_dir** (`str, optional`): Directory for app data (default: './presenton_data')
- **image** (`str, optional`): Docker image (default: 'ghcr.io/presenton/presenton:latest')
- **llm** (`str, optional`): LLM provider - 'openai', 'google', 'anthropic', 'ollama' (default: 'anthropic')
- **openai_api_key** (`str, optional`): OpenAI API key
- **google_api_key** (`str, optional`): Google API key
- **anthropic_api_key** (`str, optional`): Anthropic API key
- **image_provider** (`str, optional`): Image provider - 'pexels', 'pixabay', 'dalle3', etc.
- **force_recreate** (`bool, optional`): Remove and recreate container if exists (default: False)

**Returns:** Dictionary with: - success (bool): Whether container started successfully - container_name (str): Container name - url (str): URL to access Presenton - error (str, optional): Error message if failed

---

## `stop_presenton_tool`

Stop the Presenton Docker container.

**Parameters:**

- **container_name** (`str, optional`): Docker container name (default: 'presenton')
- **remove** (`bool, optional`): Remove container after stopping (default: False)

**Returns:** Dictionary with: - success (bool): Whether container stopped successfully - container_name (str): Container name - removed (bool): Whether container was removed - error (str, optional): Error message if failed

---

## `generate_presentation_tool`

Generate an AI-powered presentation using Presenton.

**Parameters:**

- **content** (`str, required`): Topic or content for the presentation
- **n_slides** (`int, optional`): Number of slides (default: 8)
- **language** (`str, optional`): Target language (default: 'English')
- **template** (`str, optional`): Design template name
- **export_as** (`str, optional`): Export format - 'pptx' or 'pdf' (default: 'pptx')
- **tone** (`str, optional`): Content tone - 'professional', 'casual', 'funny', 'formal', 'inspirational'
- **web_search** (`bool, optional`): Enable web grounding for research (default: False)
- **base_url** (`str, optional`): Presenton API URL (default: 'http://localhost:5000')
- **auto_start** (`bool, optional`): Auto-start container if not running (default: True)
- **timeout** (`int, optional`): Request timeout in seconds (default: 300)

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - presentation_id (str): Unique presentation ID - file_path (str): Path to generated presentation file - edit_url (str): URL to edit presentation in browser - error (str, optional): Error message if failed

---

## `list_templates_tool`

List available presentation templates in Presenton.

**Parameters:**

- **base_url** (`str, optional`): Presenton API URL (default: 'http://localhost:5000')

**Returns:** Dictionary with: - success (bool): Whether request succeeded - templates (list): List of available templates - error (str, optional): Error message if failed

---

## `generate_presentation_from_research_tool`

Generate a presentation from web research on a topic.  This tool combines web search research with presentation generation to create informative, data-backed presentations.

**Parameters:**

- **topic** (`str, required`): Research topic for the presentation
- **n_slides** (`int, optional`): Number of slides (default: 10)
- **language** (`str, optional`): Target language (default: 'English')
- **template** (`str, optional`): Design template name
- **export_as** (`str, optional`): Export format - 'pptx' or 'pdf' (default: 'pptx')
- **tone** (`str, optional`): Content tone (default: 'professional')
- **max_search_results** (`int, optional`): Max search results to use (default: 10)
- **base_url** (`str, optional`): Presenton API URL

**Returns:** Dictionary with: - success (bool): Whether generation succeeded - presentation_id (str): Unique presentation ID - file_path (str): Path to generated presentation - edit_url (str): URL to edit presentation - research_summary (str): Summary of research used - error (str, optional): Error message if failed

---

## `download_presentation_tool`

Download a generated presentation file.

**Parameters:**

- **presentation_id** (`str, required`): Presentation ID to download
- **output_path** (`str, optional`): Path to save file (default: ~/presentations/)
- **format** (`str, optional`): Download format - 'pptx' or 'pdf' (default: 'pptx')
- **base_url** (`str, optional`): Presenton API URL

**Returns:** Dictionary with: - success (bool): Whether download succeeded - file_path (str): Path to downloaded file - error (str, optional): Error message if failed

---

## `health_check`

Check if Presenton API is accessible.

**Returns:** `bool`

---

## `generate_presentation`

Generate a presentation via API.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** `Dict[str, Any]`

---

## `get_templates`

Get available templates.

**Returns:** `Dict[str, Any]`

---

## `get_presentation`

Get presentation details.

**Parameters:**

- **presentation_id** (`str`)

**Returns:** `Dict[str, Any]`
