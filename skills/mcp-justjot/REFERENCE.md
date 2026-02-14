# mcp-justjot - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`list_ideas_tool`](#list_ideas_tool) | List all ideas. |
| [`create_idea_tool`](#create_idea_tool) | Create a new idea. |
| [`get_idea_tool`](#get_idea_tool) | Get idea by ID. |
| [`update_idea_tool`](#update_idea_tool) | Update an existing idea. |
| [`delete_idea_tool`](#delete_idea_tool) | Delete an idea. |
| [`list_templates_tool`](#list_templates_tool) | List all templates. |
| [`get_template_tool`](#get_template_tool) | Get template by ID or name. |
| [`add_section_tool`](#add_section_tool) | Add section to idea. |
| [`update_section_tool`](#update_section_tool) | Update section in idea. |
| [`list_tags_tool`](#list_tags_tool) | List all tags. |
| [`get_ideas_by_tag_tool`](#get_ideas_by_tag_tool) | Get ideas filtered by tag. |

---

## `list_ideas_tool`

List all ideas.

**Parameters:**

- **full** (`bool, optional`): Include full content (default: False)
- **search** (`str, optional`): Search query

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - ideas (list): List of ideas - error (str, optional): Error message if failed

---

## `create_idea_tool`

Create a new idea.

**Parameters:**

- **title** (`str, required`): Idea title
- **description** (`str, optional`): Idea description
- **templateName** (`str, optional`): Template name
- **tags** (`list, optional`): List of tags
- **status** (`str, optional`): Idea status

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - idea (dict): Created idea - error (str, optional): Error message if failed

---

## `get_idea_tool`

Get idea by ID.

**Parameters:**

- **idea_id** (`str, required`): Idea ID

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - idea (dict): Idea data - error (str, optional): Error message if failed

---

## `update_idea_tool`

Update an existing idea.

**Parameters:**

- **idea_id** (`str, required`): Idea ID
- **title** (`str, optional`): New title
- **description** (`str, optional`): New description
- **status** (`str, optional`): New status
- **tags** (`list, optional`): New tags

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - idea (dict): Updated idea - error (str, optional): Error message if failed

---

## `delete_idea_tool`

Delete an idea.

**Parameters:**

- **idea_id** (`str, required`): Idea ID

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - error (str, optional): Error message if failed

---

## `list_templates_tool`

List all templates.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** `Dict[str, Any]`

---

## `get_template_tool`

Get template by ID or name.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** `Dict[str, Any]`

---

## `add_section_tool`

Add section to idea.

**Parameters:**

- **idea_id** (`str, required`): Idea ID
- **title** (`str, required`): Section title
- **type** (`str, required`): Section type
- **content** (`str, optional`): Section content

**Returns:** `Dict[str, Any]`

---

## `update_section_tool`

Update section in idea.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** `Dict[str, Any]`

---

## `list_tags_tool`

List all tags.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** `Dict[str, Any]`

---

## `get_ideas_by_tag_tool`

Get ideas filtered by tag.

**Parameters:**

- **tag** (`str, required`): Tag to filter by
- **limit** (`int, optional`): Maximum results

**Returns:** Dictionary with filtered ideas list
