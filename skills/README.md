# Jotty Skills Directory

This directory contains dynamically loaded skills for Jotty.

Skills are loaded automatically by SkillsRegistry.

## Skill Types

Every skill has a **type** that defines its composition pattern:

| Type | Description | Example |
|------|-------------|---------|
| **base** | Fundamental atomic skill. Does one thing, no dependency on other skills. | `web-search`, `calculator`, `telegram-sender` |
| **derived** | Extends ONE base skill for a specific domain. | `stock-research` (from `web-search`), `investing-commodities` (from `web-scraper`) |
| **composite** | Combines 2+ base/derived skills in a workflow (sequential, parallel, or mixed). | `search-summarize-pdf-telegram` (web-search + claude-cli-llm + document-converter + telegram-sender) |

## Structure

Each skill should have:
- `SKILL.md` - Skill metadata and documentation
- `tools.py` - Tool implementations

### SKILL.md Format

```markdown
# Skill Name

Description text here.

## Type
base

## Base Skills
- web-search

## Execution
sequential

## Capabilities
- research
- data-fetch

## Description
Detailed description...

## Tools
### tool_name
...
```

### Type Sections

- `## Type` - One of: `base`, `derived`, `composite`
- `## Base Skills` - List of skills this depends on (required for derived/composite)
- `## Execution` - Execution mode for composite skills: `sequential`, `parallel`, `mixed`
- `## Capabilities` - List of capability tags for discovery (e.g., `research`, `data-fetch`, `communicate`)

## Adding New Skills

### New Base Skill

```
skills/
  my-skill/
    SKILL.md    # Type: base
    tools.py
```

### New Derived Skill

```
skills/
  my-domain-skill/
    SKILL.md    # Type: derived, Base Skills: [parent-skill]
    tools.py
```

### New Composite Skill

```
skills/
  my-pipeline/
    SKILL.md    # Type: composite, Base Skills: [skill-a, skill-b], Execution: sequential
    tools.py
```

## Querying Skills by Type

```python
from Jotty.core.registry import get_skills_registry, SkillType

registry = get_skills_registry()

# List all base skills
base_skills = registry.list_skills_by_type(SkillType.BASE)

# List all derived skills
derived_skills = registry.list_skills_by_type(SkillType.DERIVED)

# List all composite skills
composite_skills = registry.list_skills_by_type(SkillType.COMPOSITE)

# Get type counts
summary = registry.get_skill_type_summary()
# {"base": 50, "derived": 40, "composite": 35}
```
