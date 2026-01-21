"""
Section Tools - LLM Tool Definitions

Auto-generates LLM tool definitions from section schemas.
Enables LLM to choose ANY section type based on user query.

This is the DRY way - no hardcoded logic per section type!
"""

from typing import List, Dict, Any
from ..ui.schema_validator import schema_registry


def generate_section_tools() -> List[Dict[str, Any]]:
    """
    Auto-generate LLM tool definitions from section schemas.

    Returns list of tool definitions that can be used with Claude/OpenAI
    function calling.

    Returns:
        List of tool definitions in Anthropic format

    Example output:
        [
            {
                "name": "return_kanban",
                "description": "Return kanban board for visual task tracking",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "columns": {...},
                        "title": {...}
                    }
                }
            },
            ...
        ]
    """
    tools = []

    # Get all available section types from schema registry
    for section_type in schema_registry.list_sections():
        schema = schema_registry.get_schema(section_type)
        if not schema:
            continue

        # Convert section type to tool name (kebab-case → snake_case)
        tool_name = f"return_{section_type.replace('-', '_')}"

        # Build tool definition
        tool = {
            "name": tool_name,
            "description": schema.get('description', f'Return {section_type} section'),
        }

        # Add input schema if available
        if 'schema' in schema:
            section_schema = schema['schema']

            # Wrap section schema in parameters
            tool["input_schema"] = {
                "type": "object",
                "properties": {
                    "content": section_schema,  # The main content
                    "title": {
                        "type": "string",
                        "description": "Optional title for the section"
                    }
                },
                "required": ["content"]
            }

        tools.append(tool)

    return tools


# Lightweight tool hints for system prompt (instead of full schemas)
def generate_tool_hints() -> str:
    """
    Generate concise tool hints for LLM system prompt.

    Returns compact list of available tools (~2KB instead of ~200KB schemas).

    Example output:
        \"\"\"
        Available section helpers:

        Project Management:
        - return_kanban(columns=[{id, title, items}], title) - Visual task boards
        - return_gantt(tasks=[{start, end}], title) - Timeline charts

        Data Visualization:
        - return_chart(type='bar|line', data={}, title) - Charts
        ...
        \"\"\"
    """
    hints = ["Available section helpers:", ""]

    # Group by category
    by_category = {}

    for section_type in schema_registry.list_sections():
        schema = schema_registry.get_schema(section_type)
        if not schema:
            continue

        # Get hint (or generate default)
        hint = schema.get('llmHint', f"return_{section_type.replace('-', '_')}(content, title)")
        description = schema.get('description', '')

        # TODO: Get category from section registry
        category = 'General'

        if category not in by_category:
            by_category[category] = []

        by_category[category].append(f"- {hint}  # {description}")

    # Format by category
    for category, items in by_category.items():
        hints.append(f"{category}:")
        hints.extend(items)
        hints.append("")

    return "\n".join(hints)


# Example system prompt using tool hints
SYSTEM_PROMPT_TEMPLATE = """
You are a task management assistant with access to rich visualizations.

When responding to queries about tasks, choose the BEST section format:
- Visual overview → use return_kanban
- Text summary → use return_text
- Progress over time → use return_chart
- etc.

{tool_hints}

Guidelines:
1. Choose the most appropriate section type for the query
2. Generate well-formatted content matching the schema
3. Use titles to provide context
4. Be concise but informative
"""


def get_system_prompt() -> str:
    """Get complete system prompt with tool hints."""
    tool_hints = generate_tool_hints()
    return SYSTEM_PROMPT_TEMPLATE.format(tool_hints=tool_hints)
