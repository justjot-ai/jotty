---
name: yaml-validator
description: "Validate YAML syntax, check structure, and convert between YAML and JSON"
---

# Yaml Validator Skill

Validate YAML syntax, check structure, and convert between YAML and JSON

## Type
base

## Capabilities
- Validate YAML syntax
- Convert YAML to JSON
- Convert JSON to YAML

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
- "validate yaml"
- "convert yaml to json"
- "check yaml syntax"

## Category
data/validation

## Tools

### validate_yaml
Validate and optionally convert YAML.
**Params:** yaml_text (str), to_json (bool)

## Dependencies
PyYAML
