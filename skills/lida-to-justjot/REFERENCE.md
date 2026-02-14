# LIDA to JustJot Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`visualize_to_idea_tool`](#visualize_to_idea_tool) | Generate LIDA visualization and create JustJot idea. |
| [`create_dashboard_tool`](#create_dashboard_tool) | Create multi-chart dashboard idea. |
| [`create_custom_idea_tool`](#create_custom_idea_tool) | Create idea with custom sections using any section type. |
| [`get_section_types_tool`](#get_section_types_tool) | Get all available JustJot section types. |
| [`get_section_context_tool`](#get_section_context_tool) | Get LLM context for section type selection. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`registry`](#registry) | Lazy load skills registry. |
| [`visualize_to_idea`](#visualize_to_idea) | Generate visualization and create JustJot idea. |
| [`create_dashboard_idea`](#create_dashboard_idea) | Create multi-chart dashboard idea. |
| [`create_custom_idea`](#create_custom_idea) | Create idea with custom sections using any section type from registry. |
| [`get_available_section_types`](#get_available_section_types) | Get all available section types from JustJot registry. |
| [`get_section_types_context`](#get_section_types_context) | Get LLM context for section type selection. |

---

## `visualize_to_idea_tool`

Generate LIDA visualization and create JustJot idea.

**Parameters:**

- **data** (`required`): DataFrame, CSV string, CSV path, or list of dicts
- **question** (`str, required`): Natural language visualization question
- **title** (`str, optional`): Idea title
- **description** (`str, optional`): Idea description
- **tags** (`list, optional`): Tags for the idea
- **userId** (`str, optional`): Clerk user ID
- **author** (`str, optional`): Author name
- **include_data** (`bool, optional`): Include data table section (default: True)
- **include_chart** (`bool, optional`): Include chart section (default: True)
- **include_code** (`bool, optional`): Include code section (default: True)
- **include_insights** (`bool, optional`): Include insights section (default: True)
- **interactive** (`bool, optional`): Use interactive charts (default: True)

**Returns:** Dictionary with success status, idea_id, and details

---

## `create_dashboard_tool`

Create multi-chart dashboard idea.

**Parameters:**

- **data** (`required`): DataFrame, CSV string, CSV path, or list of dicts
- **request** (`str, required`): High-level analysis request
- **num_charts** (`int, optional`): Number of charts (default: 4)
- **title** (`str, optional`): Dashboard title
- **tags** (`list, optional`): Tags
- **userId** (`str, optional`): Clerk user ID
- **author** (`str, optional`): Author name

**Returns:** Dictionary with success status and details

---

## `create_custom_idea_tool`

Create idea with custom sections using any section type.

**Parameters:**

- **title** (`str, required`): Idea title
- **sections** (`list, required`): List of section dicts with type, title, content
- **description** (`str, optional`): Idea description
- **tags** (`list, optional`): Tags
- **userId** (`str, optional`): Clerk user ID
- **author** (`str, optional`): Author name

**Returns:** Dictionary with success status and details

---

## `get_section_types_tool`

Get all available JustJot section types.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with available section types and their info

---

## `get_section_context_tool`

Get LLM context for section type selection.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with context string for LLM prompts

---

## `registry`

Lazy load skills registry.

---

## `visualize_to_idea`

Generate visualization and create JustJot idea.

**Parameters:**

- **df** (`pd.DataFrame`)
- **question** (`str`)
- **title** (`str`)
- **description** (`str`)
- **tags** (`List[str]`)
- **user_id** (`str`)
- **author** (`str`)
- **include_data** (`bool`)
- **include_chart** (`bool`)
- **include_code** (`bool`)
- **include_insights** (`bool`)
- **interactive** (`bool`)

**Returns:** Dict with success status, idea_id, and details

---

## `create_dashboard_idea`

Create multi-chart dashboard idea.

**Parameters:**

- **df** (`pd.DataFrame`)
- **user_request** (`str`)
- **num_charts** (`int`)
- **title** (`str`)
- **tags** (`List[str]`)
- **user_id** (`str`)
- **author** (`str`)

**Returns:** Dict with success status and details

---

## `create_custom_idea`

Create idea with custom sections using any section type from registry.

**Parameters:**

- **sections** (`List[Dict[str, Any]]`)
- **title** (`str`)
- **description** (`str`)
- **tags** (`List[str]`)
- **user_id** (`str`)
- **author** (`str`)

**Returns:** Dict with success status and details

---

## `get_available_section_types`

Get all available section types from JustJot registry.

**Returns:** `List[Dict[str, str]]`

---

## `get_section_types_context`

Get LLM context for section type selection.

**Returns:** `str`
