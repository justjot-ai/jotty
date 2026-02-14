---
name: discovering-skills
description: "Meta-skill for discovering and understanding available Jotty skills. Helps agents find the right tools for tasks."
---

# Skill Discovery

## Description
Meta-skill for discovering and understanding available Jotty skills. Helps agents find the right tools for tasks.


## Type
derived

## Base Skills
- file-operations


## Capabilities
- research

## Tools

### list_categories_tool
List all skill categories with descriptions.

### list_skills_tool
List skills with optional filtering by category, tag, or search query.

**Parameters:**
- `category` (str, optional): Filter by category name
- `tag` (str, optional): Filter by tag
- `search` (str, optional): Search query
- `include_uncategorized` (bool, optional): Include uncategorized skills (default: true)

### get_skill_info_tool
Get detailed information about a specific skill including available tools.

**Parameters:**
- `skill_name` (str, required): Name of the skill

### get_discovery_summary_tool
Get a summary of all skills for agent discovery.

**Parameters:**
- `format` (str, optional): 'json' or 'markdown' (default: 'json')

### refresh_manifest_tool
Refresh the skills manifest to discover newly added skills.

### categorize_skill_tool
Move a skill to a different category.

**Parameters:**
- `skill_name` (str, required): Name of the skill
- `category` (str, required): Target category

### find_skills_for_task_tool
Find relevant skills for a given task description.

**Parameters:**
- `task` (str, required): Description of the task
- `max_results` (int, optional): Maximum skills to return (default: 10)

## Usage

```python
# Find skills for a task
result = find_skills_for_task_tool({'task': 'create a presentation about AI'})
# Returns: slide-generator, pptx-editor, presenton

# List all document skills
result = list_skills_tool({'category': 'documents'})

# Get info about a skill
result = get_skill_info_tool({'skill_name': 'slide-generator'})
```

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Search skills catalog
- [ ] Step 2: Analyze capabilities
- [ ] Step 3: Rank matches
- [ ] Step 4: Return recommendations
```

**Step 1: Search skills catalog**
Query the registry for skills matching the task description.

**Step 2: Analyze capabilities**
Evaluate skill capabilities, tools, and compatibility.

**Step 3: Rank matches**
Score and rank skills by relevance to the task.

**Step 4: Return recommendations**
Provide the best-matching skills with usage examples.

## Triggers
- "skill discovery"

## Category
workflow-automation
