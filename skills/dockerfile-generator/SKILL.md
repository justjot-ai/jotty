---
name: generating-dockerfiles
description: "Generate Dockerfiles from language and framework specifications. Pure Python. Use when the user wants to generate Dockerfile, containerize, Docker image."
---

# Dockerfile Generator Skill

Generate Dockerfiles from language and framework specifications. Pure Python. Use when the user wants to generate Dockerfile, containerize, Docker image.

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
- "dockerfile"
- "docker"
- "containerize"
- "docker image"
- "container"

## Category
development

## Tools

### generate_dockerfile_tool
Generate a Dockerfile for a project.

**Parameters:**
- `language` (str, required): Language (python, node, go, rust, java)
- `framework` (str, optional): Framework (fastapi, flask, express, nextjs, gin)
- `port` (int, optional): Exposed port (default: auto-detected)
- `multi_stage` (bool, optional): Use multi-stage build (default: true)

**Returns:**
- `success` (bool)
- `dockerfile` (str): Dockerfile content
- `dockerignore` (str): .dockerignore content

## Dependencies
None
