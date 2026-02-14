---
name: dev-workflow
description: "This composite skill combines: 1. **Changelog Generation** (Source): changelog-generator 2. **Skill Creation** (Processor): skill-creator 3. **Webapp Testing** (Sink): webapp-testing."
---

# Development Workflow Composite Skill

Complete development workflow: changelog generation → skill creation → webapp testing.

## Description

This composite skill combines:
1. **Changelog Generation** (Source): changelog-generator
2. **Skill Creation** (Processor): skill-creator
3. **Webapp Testing** (Sink): webapp-testing


## Type
composite

## Base Skills
- github
- shell-exec
- file-operations

## Execution
sequential


## Capabilities
- code

## Usage

```python
from skills.dev_workflow.tools import dev_workflow_tool

# Full workflow
result = await dev_workflow_tool({
    'workflow_type': 'full',
    'changelog_version': '2.0.0',
    'skill_name': 'my-new-skill',
    'skill_description': 'Does something useful',
    'app_url': 'http://localhost:3000'
})

# Just changelog
result = await dev_workflow_tool({
    'workflow_type': 'changelog',
    'changelog_since': 'last-release',
    'changelog_version': '2.0.0'
})
```

## Parameters

- `workflow_type` (str, required): 'changelog', 'skill_creation', 'testing', or 'full'
- `changelog_since` (str, optional): Git reference (default: 'HEAD~10')
- `changelog_version` (str, optional): Version for changelog
- `skill_name` (str, optional): Name for new skill
- `skill_description` (str, optional): Description for new skill
- `app_url` (str, optional): URL for testing (default: 'http://localhost:3000')
- `test_type` (str, optional): Test type (default: 'screenshot')
- `generate_changelog` (bool, optional): Generate changelog
- `create_skill` (bool, optional): Create skill
- `test_app` (bool, optional): Test app

## Architecture

Source → Processor → Sink pattern:
- **Source**: Changelog generator
- **Processor**: Skill creator
- **Sink**: Webapp testing

No code duplication - reuses existing skills.

## Triggers
- "dev workflow"

## Category
development
