---
name: skill-writing
description: "Creates new Jotty skills from natural language descriptions. Generates SKILL.md and tools.py files. Use when the user wants to create a skill, build a tool, or generate a new capability."
---

# Skill Writer

## Description
Creates new Jotty skills from natural language descriptions. Generates complete
skill packages (SKILL.md + tools.py) that are immediately loadable by the registry.
Can also improve existing skills based on feedback.

## Type
base

## Capabilities
- generate

## Triggers
- "create a skill"
- "build a tool"
- "generate a skill"
- "write a skill"
- "make a new skill"
- "create a new capability"

## Category
workflow-automation

## Tools

### create_skill_tool
Creates a new Jotty skill from a description.

**Parameters:**
- `name` (str, required): Skill name in kebab-case (e.g., "pdf-merger", "stock-screener")
- `description` (str, required): What the skill should do, in plain English
- `requirements` (list[str], optional): Python packages needed (e.g., ["requests", "pandas"])
- `examples` (list[str], optional): Example usage scenarios to guide generation

**Returns:**
- `success` (bool): Whether skill was created
- `skill_name` (str): Normalized skill name
- `skill_path` (str): Path to created skill directory
- `tools` (list[str]): Tool function names generated
- `error` (str, optional): Error message if failed

### improve_skill_tool
Improves an existing skill based on feedback.

**Parameters:**
- `name` (str, required): Name of the existing skill to improve
- `feedback` (str, required): What to change or improve

**Returns:**
- `success` (bool): Whether improvement was applied
- `skill_name` (str): Skill that was improved
- `changes` (str): Summary of changes made
- `error` (str, optional): Error message if failed

### list_skills_tool
Lists all available skills in the registry.

**Parameters:**
- `category` (str, optional): Filter by category
- `search` (str, optional): Search term to filter skills

**Returns:**
- `success` (bool): Always true
- `skills` (list[dict]): List of skill info dicts with name, description
- `total` (int): Total number of skills found
