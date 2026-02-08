# LIDA to JustJot Skill

## Description

Multi-agent skill that transforms data and natural language questions into AI-generated visualizations saved as JustJot ideas. Uses LIDA for visualization generation and JustJot's registry-driven section system for maximum compatibility with all 47+ section types.


## Type
composite

## Base Skills
- image-generator
- mcp-justjot

## Execution
sequential


## Capabilities
- visualize
- data-fetch

## Workflow

1. **Input**: DataFrame + Natural language question
2. **Visualization**: LIDA VisualizationLayer generates charts
3. **Transform**: Registry-driven SectionTransformer creates sections
4. **Output**: JustJot idea via MCP client

## Features

- **Full Section Type Support**: Works with ALL 47+ JustJot section types via registry
- **LIDA Integration**: AI-powered visualization from natural language
- **Multiple Output Types**: Single visualization, multi-chart dashboards, custom ideas
- **Smart Section Selection**: LLM context for optimal section type choice
- **Flexible Input**: Accepts DataFrame, CSV, JSON, file paths

## Tools

### `visualize_to_idea_tool`

Generate LIDA visualization and create JustJot idea.

```python
from skills.lida_to_justjot.tools import visualize_to_idea_tool

result = await visualize_to_idea_tool({
    'data': df,  # or CSV string, file path
    'question': 'Show total sales by region as a bar chart',
    'title': 'Regional Sales Analysis',
    'tags': ['sales', 'analytics'],
    'userId': 'user_xxx',
    'include_data': True,
    'include_chart': True,
    'include_code': True,
    'include_insights': True,
})
```

### `create_dashboard_tool`

Create multi-chart dashboard idea.

```python
from skills.lida_to_justjot.tools import create_dashboard_tool

result = await create_dashboard_tool({
    'data': df,
    'request': 'Analyze sales performance across all dimensions',
    'num_charts': 4,
    'title': 'Sales Dashboard Q4 2024',
    'tags': ['dashboard', 'sales'],
    'userId': 'user_xxx',
})
```

### `create_custom_idea_tool`

Create idea with custom sections using ANY section type.

```python
from skills.lida_to_justjot.tools import create_custom_idea_tool

result = await create_custom_idea_tool({
    'title': 'Business Strategy Plan',
    'sections': [
        {'type': 'text', 'title': 'Executive Summary', 'content': '# Overview\n\n...'},
        {'type': 'chart', 'title': 'Market Analysis', 'content': '{"type":"bar",...}'},
        {'type': 'swot', 'title': 'SWOT Analysis', 'content': '{"strengths":[...]}'},
        {'type': 'kanban-board', 'title': 'Implementation Tasks', 'content': '{"columns":[...]}'},
        {'type': 'timeline', 'title': 'Milestones', 'content': '{"events":[...]}'},
    ],
    'tags': ['strategy', 'planning'],
    'userId': 'user_xxx',
})
```

### `get_section_types_tool`

Get all available JustJot section types.

```python
from skills.lida_to_justjot.tools import get_section_types_tool

result = get_section_types_tool()
# Returns: {
#     'success': True,
#     'count': 47,
#     'types': [{'id': 'chart', 'label': 'Chart', 'category': 'Visualization'}, ...],
#     'by_category': {'Visualization': [...], 'Content': [...], ...}
# }
```

### `get_section_context_tool`

Get LLM context for section type selection.

```python
from skills.lida_to_justjot.tools import get_section_context_tool

result = get_section_context_tool()
# Returns context string to include in LLM prompts
```

## Parameters

### `visualize_to_idea_tool`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| data | DataFrame/str/list | Yes | - | Source data |
| question | str | Yes | - | Natural language visualization question |
| title | str | No | Auto | Idea title |
| description | str | No | Auto | Idea description |
| tags | list | No | [] | Tags for the idea |
| userId | str | No | None | Clerk user ID for assignment |
| author | str | No | None | Author name |
| include_data | bool | No | True | Include data-table section |
| include_chart | bool | No | True | Include chart section |
| include_code | bool | No | True | Include visualization code |
| include_insights | bool | No | True | Include AI insights |
| interactive | bool | No | True | Use interactive charts |

## Section Types

The skill supports ALL JustJot section types via the registry:

**Visualization:**
- `chart` - Charts (bar, line, pie, etc.)
- `data-table` - Data tables
- `html` - Interactive HTML (Plotly/Altair)

**Content:**
- `text` - Markdown text
- `code` - Code with syntax highlighting
- `todos` - Todo lists
- `notes` - Notes

**Diagrams:**
- `mermaid` - Mermaid diagrams
- `flowchart` - Flowcharts
- `mindmap` - Mind maps
- `network-graph` - Network graphs

**Business:**
- `swot` - SWOT analysis
- `kanban-board` - Kanban boards
- `timeline` - Timelines
- `decision-matrix` - Decision matrices

**And 30+ more...**

## Dependencies

- `core.semantic.visualization.VisualizationLayer` - LIDA integration
- `core.semantic.visualization.justjot` - Section registry and transformers
- `mcp-justjot-mcp-client` skill - Idea creation via MCP
- `mcp-justjot` skill - Fallback HTTP API

## Example Usage

```python
import pandas as pd
import asyncio
from skills.lida_to_justjot.tools import (
    visualize_to_idea_tool,
    create_custom_idea_tool,
    get_section_types_tool,
)

# Sample data
df = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West'],
    'Q1_Sales': [120000, 85000, 150000, 95000],
    'Q2_Sales': [135000, 92000, 165000, 102000],
})

# Create visualization idea
async def main():
    result = await visualize_to_idea_tool({
        'data': df,
        'question': 'Compare Q1 vs Q2 sales by region',
        'title': 'Quarterly Sales Comparison',
        'tags': ['sales', 'quarterly'],
        'userId': 'user_xxx',
    })
    print(f"Created idea: {result.get('idea_id')}")

asyncio.run(main())
```

## Architecture

```
                    +------------------+
                    |   User Request   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   DataFrame +    |
                    |   NL Question    |
                    +--------+---------+
                             |
              +--------------v--------------+
              |   LIDA VisualizationLayer   |
              |  (AI Chart Generation)      |
              +--------------+--------------+
                             |
              +--------------v--------------+
              |   JustJotIdeaBuilder        |
              |  - Registry-driven          |
              |  - SectionTransformer       |
              |  - ChartTransformer         |
              +--------------+--------------+
                             |
              +--------------v--------------+
              |   MCP Client                |
              |  (mcp-justjot-mcp-client)   |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |   JustJot.ai     |
                    |   (MongoDB)      |
                    +------------------+
```
