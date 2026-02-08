# Skill Creator Skill

Helps create new Jotty skills by generating templates, validating structure, and providing guidance.

## Description

This skill assists in creating new Jotty skills by generating SKILL.md templates, validating skill structure, and providing best practices for skill development.


## Type
derived

## Base Skills
- file-operations

## Tools

### `create_skill_template_tool`

Create a new skill template with proper structure.

**Parameters:**
- `skill_name` (str, required): Name of the skill (kebab-case)
- `description` (str, required): Brief description of what the skill does
- `output_directory` (str, optional): Where to create the skill (default: skills/)
- `include_tools` (bool, optional): Include tools.py template (default: True)
- `include_requirements` (bool, optional): Include requirements.txt (default: True)

**Returns:**
- `success` (bool): Whether creation succeeded
- `skill_path` (str): Path to created skill directory
- `files_created` (list): List of files created
- `error` (str, optional): Error message if failed

### `validate_skill_tool`

Validate an existing skill's structure and metadata.

**Parameters:**
- `skill_path` (str, required): Path to skill directory

**Returns:**
- `success` (bool): Whether validation succeeded
- `valid` (bool): Whether skill is valid
- `issues` (list): List of validation issues found
- `warnings` (list): List of warnings

## Usage Examples

### Create New Skill

```python
result = await create_skill_template_tool({
    'skill_name': 'my-new-skill',
    'description': 'Does something useful',
    'include_tools': True
})
```

### Validate Skill

```python
result = await validate_skill_tool({
    'skill_path': 'skills/my-skill'
})
```

## Dependencies

- `file-operations`: For creating files and directories
