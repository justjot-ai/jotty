# Notion Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`search_pages_tool`](#search_pages_tool) | Search for pages and databases in Notion. |
| [`get_page_tool`](#get_page_tool) | Get a page's properties and optionally its content blocks. |
| [`create_page_tool`](#create_page_tool) | Create a new page in Notion. |
| [`update_page_tool`](#update_page_tool) | Update a page's properties in Notion. |
| [`query_database_tool`](#query_database_tool) | Query a Notion database with filters and sorts. |
| [`create_database_item_tool`](#create_database_item_tool) | Add a new item (page) to a Notion database. |

---

## `search_pages_tool`

Search for pages and databases in Notion.

**Parameters:**

- **query** (`str, required`): Search query text
- **filter** (`dict, optional`): Filter by object type - {"property": "object", "value": "page"} or "database"
- **sort** (`dict, optional`): Sort results - {"direction": "ascending" or "descending", "timestamp": "last_edited_time"}
- **page_size** (`int, optional`): Number of results (default: 10, max: 100)
- **start_cursor** (`str, optional`): Pagination cursor
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether request succeeded - results (list): List of matching pages/databases - has_more (bool): Whether more results exist - next_cursor (str): Cursor for next page - error (str, optional): Error message if failed

---

## `get_page_tool`

Get a page's properties and optionally its content blocks.

**Parameters:**

- **page_id** (`str, required`): The Notion page ID
- **include_content** (`bool, optional`): Also fetch page content blocks (default: True)
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether request succeeded - page (dict): Page properties - content (list, optional): List of content blocks - error (str, optional): Error message if failed

---

## `create_page_tool`

Create a new page in Notion.

**Parameters:**

- **parent_id** (`str, required`): Parent page or database ID
- **parent_type** (`str, optional`): 'page_id' or 'database_id' (default: 'page_id')
- **title** (`str, required`): Page title
- **properties** (`dict, optional`): Additional properties (for database pages)
- **content** (`list, optional`): List of block objects for page content
- **icon** (`dict, optional`): Page icon - {"type": "emoji", "emoji": "..."} or {"type": "external", "external": {"url": "..."}}
- **cover** (`dict, optional`): Cover image - {"type": "external", "external": {"url": "..."}}
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether creation succeeded - page_id (str): ID of created page - url (str): URL of created page - page (dict): Full page object - error (str, optional): Error message if failed

---

## `update_page_tool`

Update a page's properties in Notion.

**Parameters:**

- **page_id** (`str, required`): The Notion page ID
- **properties** (`dict, optional`): Properties to update
- **archived** (`bool, optional`): Set to True to archive the page
- **icon** (`dict, optional`): New page icon
- **cover** (`dict, optional`): New cover image
- **timeout** (`int, optional`): Request timeout in seconds (default: 30)

**Returns:** Dictionary with: - success (bool): Whether update succeeded - page (dict): Updated page object - error (str, optional): Error message if failed

---

## `query_database_tool`

Query a Notion database with filters and sorts.

**Parameters:**

- **database_id** (`str, required`): The Notion database ID
- **filter** (`dict, optional`): Filter conditions
- **sorts** (`list, optional`): Sort conditions
- **page_size** (`int, optional`): Number of results (default: 100, max: 100)
- **start_cursor** (`str, optional`): Pagination cursor
- **timeout** (`int, optional`): Request timeout in seconds (default: 30) Filter examples: {"property": "Status", "select": {"equals": "Done"}} {"property": "Due Date", "date": {"before": "2024-01-01"}} {"and": [filter1, filter2]} {"or": [filter1, filter2]} Sort examples: [{"property": "Created", "direction": "descending"}] [{"timestamp": "last_edited_time", "direction": "ascending"}]

**Returns:** Dictionary with: - success (bool): Whether query succeeded - results (list): List of database items - has_more (bool): Whether more results exist - next_cursor (str): Cursor for next page - error (str, optional): Error message if failed

---

## `create_database_item_tool`

Add a new item (page) to a Notion database.

**Parameters:**

- **database_id** (`str, required`): The Notion database ID
- **properties** (`dict, required`): Properties for the new item
- **content** (`list, optional`): List of block objects for page content
- **icon** (`dict, optional`): Item icon
- **cover** (`dict, optional`): Cover image
- **timeout** (`int, optional`): Request timeout in seconds (default: 30) Property value examples:
- **Title**: {"title": [{"text": {"content": "My Title"}}]} Rich text: {"rich_text": [{"text": {"content": "Some text"}}]}
- **Number**: {"number": 42}
- **Select**: {"select": {"name": "Option A"}} Multi-select: {"multi_select": [{"name": "Tag1"}, {"name": "Tag2"}]}
- **Date**: {"date": {"start": "2024-01-01", "end": "2024-01-02"}}
- **Checkbox**: {"checkbox": True}
- **URL**: {"url": "https://example.com"}
- **Email**: {"email": "test@example.com"}
- **Phone**: {"phone_number": "+1234567890"}

**Returns:** Dictionary with: - success (bool): Whether creation succeeded - item_id (str): ID of created item - url (str): URL of created item - item (dict): Full item object - error (str, optional): Error message if failed
