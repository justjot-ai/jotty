---
name: managing-env-configs
description: "Manage .env files — validate, diff, merge, detect missing variables. Use when the user wants to compare env files, validate env, find missing env vars."
---

# Env Config Manager Skill

Manage .env files — validate, diff, merge, detect missing variables. Use when the user wants to compare env files, validate env, find missing env vars.

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
- "env"
- "dotenv"
- ".env"
- "environment variables"
- "env diff"

## Category
development

## Tools

### parse_env_tool
Parse a .env file and return key-value pairs.

**Parameters:**
- `file_path` (str, required): Path to .env file

**Returns:**
- `success` (bool)
- `variables` (dict): Key-value pairs
- `count` (int): Number of variables

## Dependencies
None
