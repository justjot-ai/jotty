---
name: building-ci-cd-pipelines
description: "Generate GitHub Actions workflow YAML configurations. Pure Python. Use when the user wants to create CI/CD pipeline, GitHub Actions, workflow yaml."
---

# Ci Cd Pipeline Builder Skill

Generate GitHub Actions workflow YAML configurations. Pure Python. Use when the user wants to create CI/CD pipeline, GitHub Actions, workflow yaml.

## Type
base

## Capabilities
- code
- devops

## Reference
For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Parse input parameters
- [ ] Step 2: Execute operation
- [ ] Step 3: Return results
```

## Triggers
- "ci/cd"
- "github actions"
- "workflow"
- "pipeline"
- "continuous integration"

## Category
development

## Tools

### github_actions_tool
Generate GitHub Actions workflow YAML.

**Parameters:**
- `language` (str, required): Programming language (python, node, go, rust, java)
- `features` (list, optional): Features: test, lint, build, deploy, docker (default: [test])
- `name` (str, optional): Workflow name
- `branches` (list, optional): Trigger branches (default: [main])

**Returns:**
- `success` (bool)
- `yaml` (str): GitHub Actions workflow YAML
- `file_path` (str): Suggested file path

## Dependencies
None
