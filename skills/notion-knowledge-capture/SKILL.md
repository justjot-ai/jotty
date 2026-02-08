# Notion Knowledge Capture Skill

Transforms conversations and discussions into structured documentation pages in Notion.

## Description

This skill captures insights, decisions, and knowledge from chat context, formats them appropriately, and saves to Notion wikis or databases with proper organization and linking for easy discovery.


## Type
composite

## Base Skills
- web-search
- notion

## Execution
sequential


## Capabilities
- research
- data-fetch

## Tools

### `capture_knowledge_to_notion_tool`

Capture knowledge from conversation to Notion.

**Parameters:**
- `content` (str, required): Content to capture from conversation
- `content_type` (str, optional): Type - 'faq', 'how_to', 'decision', 'concept', 'meeting_summary' (default: 'concept')
- `title` (str, optional): Page title (auto-generated if not provided)
- `parent_page_id` (str, optional): Notion parent page ID
- `database_id` (str, optional): Notion database ID
- `tags` (list, optional): Tags for categorization
- `link_to_pages` (list, optional): List of page IDs to link to

**Returns:**
- `success` (bool): Whether capture succeeded
- `page_id` (str): Created Notion page ID
- `page_url` (str): URL to created page
- `error` (str, optional): Error message if failed

## Usage Examples

### Capture FAQ

```python
result = await capture_knowledge_to_notion_tool({
    'content': 'Q: How do I reset my password? A: Go to settings...',
    'content_type': 'faq',
    'database_id': 'faq-db-id'
})
```

### Capture Decision

```python
result = await capture_knowledge_to_notion_tool({
    'content': 'Decision: Use React for frontend. Rationale: ...',
    'content_type': 'decision',
    'parent_page_id': 'project-page-id'
})
```

## Dependencies

- `notion`: For Notion API integration
