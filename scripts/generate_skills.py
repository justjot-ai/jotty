"""
Skill Batch Generator — creates SKILL.md + tools.py + eval.json for each skill.

Usage:
    python scripts/generate_skills.py
"""

import json
import os
from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent / "skills"


def create_skill(
    name: str,
    frontmatter_name: str,
    description: str,
    category: str,
    capabilities: list,
    triggers: list,
    tools_code: str,
    tool_docs: str,
    eval_tool: str,
    eval_input: dict,
    deps: str = "None",
):
    """Create a complete skill directory with all files."""
    skill_dir = SKILLS_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # --- SKILL.md ---
    caps = "\n".join(f"- {c}" for c in capabilities)
    trigs = "\n".join(f'- "{t}"' for t in triggers)
    title = name.replace("-", " ").title()

    skill_md = f"""---
name: {frontmatter_name}
description: "{description}"
---

# {title} Skill

{description}

## Type
base

## Capabilities
{caps}

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
{trigs}

## Category
{category}

## Tools

{tool_docs}

## Dependencies
{deps}
"""
    (skill_dir / "SKILL.md").write_text(skill_md)

    # --- tools.py ---
    (skill_dir / "tools.py").write_text(tools_code)

    # --- eval.json ---
    eval_data = {
        "skill": name,
        "version": "1.0",
        "scenarios": [
            {
                "type": "triggering",
                "prompt": f"I need to {frontmatter_name.replace('-', ' ')}",
                "expected_skill": name,
                "should_match": True,
                "notes": f"Should discover {name} in top-3 results",
            },
            {
                "type": "functional",
                "tool": eval_tool,
                "input": eval_input,
                "expected_output_keys": ["success"],
                "notes": f"Basic invocation of {eval_tool}",
            },
            {
                "type": "edge_case",
                "prompt": f"Use {name} with missing required parameters",
                "expected_behavior": "graceful_error",
                "notes": "Should handle invalid input gracefully",
            },
        ],
    }
    (skill_dir / "eval.json").write_text(json.dumps(eval_data, indent=2) + "\n")

    print(f"  Created: {name}")


if __name__ == "__main__":
    print("Generating skills...")
    # Skills are defined in generate_skills_batch_*.py files
    # This module provides the create_skill() helper
    print("Import create_skill from this module and call it with skill definitions.")
