---
name: managing-notion
description: "Integrates with Notion API to search, read, create, and update pages and databases. Supports full CRUD operations on Notion workspace content. Use when the user wants to save to notion, notion page, create notion."
---

# Notion Skill

## Description
Integrates with Notion API to search, read, create, and update pages and databases. Supports full CRUD operations on Notion workspace content.


## Type
base


## Capabilities
- data-fetch
- document

## Configuration
API key is loaded from:
1. Environment variable: `NOTION_API_KEY`
2. Config file: `~/.config/notion/api_key`

## Tools

### search_pages_tool
Search for pages and databases in Notion.

**Parameters:**
- `query` (str, required): Search query text
- `filter` (dict, optional): Filter by object type - `{"property": "object", "value": "page"}` or `"database"`
- `sort` (dict, optional): Sort results - `{"direction": "ascending", "timestamp": "last_edited_time"}`
- `page_size` (int, optional): Number of results (default: 10, max: 100)
- `start_cursor` (str, optional): Pagination cursor

### get_page_tool
Get a page's properties and optionally its content blocks.

**Parameters:**
- `page_id` (str, required): The Notion page ID
- `include_content` (bool, optional): Also fetch page content blocks (default: True)

### create_page_tool
Create a new page in Notion.

**Parameters:**
- `parent_id` (str, required): Parent page or database ID
- `parent_type` (str, optional): `page_id` or `database_id` (default: `page_id`)
- `title` (str, required): Page title
- `properties` (dict, optional): Additional properties (for database pages)
- `content` (list, optional): List of block objects for page content
- `icon` (dict, optional): Page icon
- `cover` (dict, optional): Cover image

### update_page_tool
Update a page's properties in Notion.

**Parameters:**
- `page_id` (str, required): The Notion page ID
- `properties` (dict, optional): Properties to update
- `archived` (bool, optional): Set to True to archive the page
- `icon` (dict, optional): New page icon
- `cover` (dict, optional): New cover image

### query_database_tool
Query a Notion database with filters and sorts.

**Parameters:**
- `database_id` (str, required): The Notion database ID
- `filter` (dict, optional): Filter conditions
- `sorts` (list, optional): Sort conditions
- `page_size` (int, optional): Number of results (default: 100, max: 100)
- `start_cursor` (str, optional): Pagination cursor

**Filter Examples:**
```json
{"property": "Status", "select": {"equals": "Done"}}
{"property": "Due Date", "date": {"before": "2024-01-01"}}
{"and": [filter1, filter2]}
```

**Sort Examples:**
```json
[{"property": "Created", "direction": "descending"}]
[{"timestamp": "last_edited_time", "direction": "ascending"}]
```

### create_database_item_tool
Add a new item (page) to a Notion database.

**Parameters:**
- `database_id` (str, required): The Notion database ID
- `properties` (dict, required): Properties for the new item
- `content` (list, optional): List of block objects for page content
- `icon` (dict, optional): Item icon
- `cover` (dict, optional): Cover image

**Property Value Examples:**
```json
{"title": [{"text": {"content": "My Title"}}]}
{"rich_text": [{"text": {"content": "Some text"}}]}
{"number": 42}
{"select": {"name": "Option A"}}
{"multi_select": [{"name": "Tag1"}, {"name": "Tag2"}]}
{"date": {"start": "2024-01-01"}}
{"checkbox": true}
{"url": "https://example.com"}
```

## Requirements
- `requests` library
- Notion API integration token with appropriate permissions

## Usage Examples

**Search for pages:**
```python
result = search_pages_tool({'query': 'project notes', 'page_size': 5})
```

**Get page content:**
```python
result = get_page_tool({'page_id': 'abc123', 'include_content': True})
```

**Create a page:**
```python
result = create_page_tool({
    'parent_id': 'parent-page-id',
    'title': 'New Page Title',
    'icon': {'type': 'emoji', 'emoji': '📝'}
})
```

**Query database:**
```python
result = query_database_tool({
    'database_id': 'db-id',
    'filter': {'property': 'Status', 'select': {'equals': 'In Progress'}},
    'sorts': [{'property': 'Due Date', 'direction': 'ascending'}]
})
```

**Add database item:**
```python
result = create_database_item_tool({
    'database_id': 'db-id',
    'properties': {
        'Name': {'title': [{'text': {'content': 'New Task'}}]},
        'Status': {'select': {'name': 'Todo'}},
        'Priority': {'select': {'name': 'High'}}
    }
})
```

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Connect to workspace
- [ ] Step 2: Search and navigate
- [ ] Step 3: Create or update pages
- [ ] Step 4: Organize content
```

**Step 1: Connect to workspace**
Authenticate with the Notion workspace via API.

**Step 2: Search and navigate**
Search pages and databases to find relevant content.

**Step 3: Create or update pages**
Create new pages or update existing ones with structured content.

**Step 4: Organize content**
Manage databases, properties, and page relationships.

## Triggers
- "notion"
- "save to notion"
- "notion page"
- "create notion"

## Category
communication
