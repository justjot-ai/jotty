---
name: managing-semver
description: "Parse, compare, and bump semantic versions (major.minor.patch). Check version constraints."
---

# Semver Manager Skill

Parse, compare, and bump semantic versions (major.minor.patch). Check version constraints.

## Type
base

## Capabilities
- code

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
- "semver"
- "semantic version"
- "bump version"
- "version compare"

## Category
development

## Tools

### semver_tool
Parse, compare, or bump semantic versions.

**Parameters:**
- `action` (str, required): parse, bump, compare, satisfies
- `version` (str, required): Semantic version string
- `part` (str): major/minor/patch (for bump)
- `other` (str): Second version (for compare)
- `constraint` (str): e.g. >=1.0.0 (for satisfies)

## Dependencies
None
