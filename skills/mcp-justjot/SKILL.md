# MCP JustJot.ai Skill

Integrates JustJot.ai MCP tools as Jotty skills.

## Description

This skill wraps JustJot.ai MCP (Model Context Protocol) tools, making them available as Jotty skills. This allows you to use JustJot.ai features (ideas, templates, sections, tags) from within Jotty workflows.

## Available Tools

### Ideas Management
- `list_ideas_tool`: List all ideas
- `create_idea_tool`: Create a new idea
- `update_idea_tool`: Update an existing idea
- `delete_idea_tool`: Delete an idea
- `get_idea_tool`: Get idea by ID

### Templates Management
- `list_templates_tool`: List templates
- `create_template_tool`: Create template
- `get_template_tool`: Get template by ID

### Sections Management
- `add_section_tool`: Add section to idea
- `update_section_tool`: Update section
- `delete_section_tool`: Delete section

### Tags Management
- `list_tags_tool`: List tags
- `create_tag_tool`: Create tag

## Usage

```python
from skills.mcp_justjot.tools import list_ideas_tool, create_idea_tool

# List ideas
ideas = await list_ideas_tool({})

# Create idea
idea = await create_idea_tool({
    'title': 'My New Idea',
    'description': 'Description here',
    'templateName': 'default'
})
```

## Architecture

Uses MCP tool executor to call JustJot.ai API endpoints.
MCP tools are automatically wrapped as Jotty skills.

## Configuration

Set environment variables:
- `JUSTJOT_API_URL`: JustJot.ai API URL (default: http://localhost:3000)
- `JUSTJOT_AUTH_TOKEN`: Authentication token (if required)
