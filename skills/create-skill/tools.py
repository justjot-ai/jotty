"""create-skill — Meta-skill that generates new skills from natural language.

Given a description, generates:
1. tools.py — Tool function implementations
2. SKILL.md — Skill metadata and documentation

Writes output to skills/user/{skill_name}/ for auto-discovery.

Usage::

    result = await create_skill(
        description="Check Hacker News for top stories",
        skill_name="hacker-news",
    )
    # {"created": "hacker-news", "path": "skills/user/hacker-news/"}
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any


_SKILLS_USER_DIR = Path(__file__).resolve().parents[1] / "user"

_SKILL_TEMPLATE = textwrap.dedent(
    """\
    ---
    description: "{description}"
    ---

    # {name}

    ## Description
    {description}

    ## Type
    derived

    ## Capabilities
    {capabilities}

    ## Tools
    {tools}

    ## Category
    user-generated
"""
)

_TOOLS_TEMPLATE = textwrap.dedent(
    '''\
    """{name} — Auto-generated skill.

    {description}
    """

    from __future__ import annotations
    from typing import Any


    {functions}
'''
)

_SAFE_NAME = re.compile(r"[^a-z0-9_-]")


async def create_skill(
    description: str,
    skill_name: str,
    *,
    tool_names: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a new skill from natural language description.

    Args:
        description: What the skill should do (natural language).
        skill_name: Identifier for the skill (kebab-case).
        tool_names: Optional explicit tool function names.

    Returns:
        Dict with created skill info: name, path, files.
    """
    # Sanitize skill name
    safe_name = _SAFE_NAME.sub("-", skill_name.lower().strip())
    if not safe_name:
        return {"error": "Invalid skill name"}

    skill_dir = _SKILLS_USER_DIR / safe_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Generate tool function names from description if not provided
    if not tool_names:
        tool_names = [_derive_function_name(safe_name)]

    # Try LLM generation first, fall back to template
    functions_code = await _generate_with_llm(description, safe_name, tool_names)
    if not functions_code:
        functions_code = _generate_template(description, safe_name, tool_names)

    # Write tools.py
    tools_content = _TOOLS_TEMPLATE.format(
        name=safe_name,
        description=description,
        functions=functions_code,
    )
    (skill_dir / "tools.py").write_text(tools_content, encoding="utf-8")

    # Write SKILL.md
    caps_str = "\n".join(f"- {t}" for t in tool_names)
    tools_str = "\n".join(f"- {t}: {description}" for t in tool_names)
    skill_md = _SKILL_TEMPLATE.format(
        name=safe_name,
        description=description,
        capabilities=caps_str,
        tools=tools_str,
    )
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    return {
        "created": safe_name,
        "path": str(skill_dir),
        "files": ["tools.py", "SKILL.md"],
        "tool_names": tool_names,
    }


async def _generate_with_llm(
    description: str,
    name: str,
    tool_names: list[str],
) -> str | None:
    """Try to generate skill code using LLM. Returns None if unavailable."""
    try:
        import dspy

        class SkillGenSignature(dspy.Signature):
            """Generate a Python tool function for a Jotty skill."""

            skill_name: str = dspy.InputField(desc="Skill identifier")
            skill_description: str = dspy.InputField(desc="What the skill should do")
            function_names: str = dspy.InputField(desc="Comma-separated function names")
            code: str = dspy.OutputField(
                desc="Python code with async function definitions. "
                "Each function takes keyword arguments and returns a dict. "
                "Include type hints and docstrings. "
                "Do NOT include imports or module docstring."
            )

        gen = dspy.ChainOfThought(SkillGenSignature)
        result = gen(
            skill_name=name,
            skill_description=description,
            function_names=", ".join(tool_names),
        )
        code = result.code.strip()
        # Basic validation: must contain 'def ' and 'return'
        if "def " in code and "return" in code:
            return code
    except Exception:
        pass
    return None


def _generate_template(
    description: str,
    name: str,
    tool_names: list[str],
) -> str:
    """Generate a simple template function when LLM is unavailable."""
    funcs = []
    for fn in tool_names:
        funcs.append(
            textwrap.dedent(
                f"""\
            async def {fn}(**kwargs: Any) -> dict[str, Any]:
                \"\"\"{description}\"\"\"
                # TODO: Implement {fn}
                return {{"status": "not_implemented", "skill": "{name}"}}
        """
            )
        )
    return "\n\n".join(funcs)


def _derive_function_name(skill_name: str) -> str:
    """Convert kebab-case skill name to snake_case function name."""
    return skill_name.replace("-", "_")
