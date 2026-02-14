---
name: notion-research-documentation
description: "This skill enables comprehensive research workflows: search for information across Notion workspace, fetch and analyze relevant pages, synthesize findings, and create well-structured documentation with proper citations. Use when the user wants to save to notion, notion page, create notion."
---

# Notion Research Documentation Skill

Searches across Notion workspace, synthesizes findings, and creates comprehensive research documentation.

## Description

This skill enables comprehensive research workflows: search for information across Notion workspace, fetch and analyze relevant pages, synthesize findings, and create well-structured documentation with proper citations.


## Type
composite

## Base Skills
- web-search
- notion

## Execution
sequential


## Capabilities
- research
- document

## Tools

### `research_and_document_tool`

Research topic in Notion and create documentation.

**Parameters:**
- `research_topic` (str, required): Topic to research
- `output_format` (str, optional): Format - 'summary', 'comprehensive', 'brief' (default: 'summary')
- `search_queries` (list, optional): Additional search queries
- `parent_page_id` (str, optional): Parent page for documentation
- `include_citations` (bool, optional): Include citations (default: True)

**Returns:**
- `success` (bool): Whether research succeeded
- `documentation_page_id` (str): Created documentation page ID
- `sources_found` (int): Number of sources found
- `sources` (list): List of source page IDs
- `error` (str, optional): Error message if failed

## Usage Examples

### Research Summary

```python
result = await research_and_document_tool({
    'research_topic': 'Q4 Product Strategy',
    'output_format': 'summary'
})
```

### Comprehensive Report

```python
result = await research_and_document_tool({
    'research_topic': 'Market Analysis',
    'output_format': 'comprehensive',
    'include_citations': True
})
```

## Dependencies

- `notion`: For Notion API integration
- `claude-cli-llm`: For synthesis and analysis

## Triggers
- "notion research documentation"
- "save to notion"
- "notion page"
- "create notion"
- "search for"
- "look up"
- "find information"
- "research"

## Category
document-creation
