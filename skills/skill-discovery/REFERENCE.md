# Skill Discovery - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`list_categories_tool`](#list_categories_tool) | List all skill categories. |
| [`list_skills_tool`](#list_skills_tool) | List skills with optional filtering. |
| [`get_skill_info_tool`](#get_skill_info_tool) | Get detailed information about a specific skill. |
| [`get_discovery_summary_tool`](#get_discovery_summary_tool) | Get a summary of all skills for agent discovery. |
| [`refresh_manifest_tool`](#refresh_manifest_tool) | Refresh the skills manifest (reload + discover new skills). |
| [`categorize_skill_tool`](#categorize_skill_tool) | Categorize a skill (move to a category). |
| [`find_skills_for_task_tool`](#find_skills_for_task_tool) | Find relevant skills for a given task description. |

---

## `list_categories_tool`

List all skill categories.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - categories (list): List of category info

---

## `list_skills_tool`

List skills with optional filtering.

**Parameters:**

- **category** (`str, optional`): Filter by category
- **tag** (`str, optional`): Filter by tag
- **search** (`str, optional`): Search query
- **include_uncategorized** (`bool, optional`): Include uncategorized skills

**Returns:** Dictionary with: - success (bool): Whether operation succeeded - skills (list): List of skill info

---

## `get_skill_info_tool`

Get detailed information about a specific skill.

**Parameters:**

- **skill_name** (`str, required`): Name of the skill

**Returns:** Dictionary with skill details and available tools

---

## `get_discovery_summary_tool`

Get a summary of all skills for agent discovery.

**Parameters:**

- **format** (`str, optional`): 'json' or 'markdown' (default: 'json')

**Returns:** Dictionary with skill summary

---

## `refresh_manifest_tool`

Refresh the skills manifest (reload + discover new skills).

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with refresh status

---

## `categorize_skill_tool`

Categorize a skill (move to a category).

**Parameters:**

- **skill_name** (`str, required`): Name of the skill
- **category** (`str, required`): Target category

**Returns:** Dictionary with operation status

---

## `find_skills_for_task_tool`

Find relevant skills for a given task description.

**Parameters:**

- **task** (`str, required`): Description of the task
- **max_results** (`int, optional`): Maximum skills to return (default: 10)
- **use_llm** (`bool, optional`): Use LLM for semantic matching (default: True)

**Returns:** Dictionary with recommended skills
