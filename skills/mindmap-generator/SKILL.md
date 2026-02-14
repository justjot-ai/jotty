---
name: generating-mindmaps
description: "Generates visual mindmap diagrams from text content using LLM analysis. Creates Mermaid mindmap syntax that can be rendered or converted to PDF. Use when the user wants to mind map, mindmap, brainstorm diagram."
---

# Mindmap Generator Skill

## Description
Generates visual mindmap diagrams from text content using LLM analysis. Creates Mermaid mindmap syntax that can be rendered or converted to PDF.


## Type
base


## Capabilities
- visualize

## Tools

### generate_mindmap_tool
Generates a Mermaid mindmap diagram from text content.

**Parameters:**
- `content` (str, required): Text content to analyze and convert to mindmap
- `title` (str, optional): Mindmap title, default: 'Mindmap'
- `style` (str, optional): Mindmap style - 'hierarchical' or 'radial', default: 'hierarchical'
- `max_branches` (int, optional): Maximum number of main branches (3-10), default: 7
- `max_depth` (int, optional): Maximum depth of branches, default: 3

**Returns:**
- `success` (bool): Whether generation succeeded
- `mindmap_code` (str): Mermaid mindmap code
- `title` (str): Mindmap title
- `output_format` (str): Output format (mermaid)
- `error` (str, optional): Error message if failed

## Triggers
- "mindmap generator"
- "mind map"
- "mindmap"
- "brainstorm diagram"
- "generate"
- "create"

## Category
workflow-automation
