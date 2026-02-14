#!/usr/bin/env python3
"""
Create REFERENCE.md files for complex skills (5+ tools).

Per Anthropic's progressive disclosure guide:
- SKILL.md stays concise (overview + navigation)
- Detailed API documentation goes in REFERENCE.md (one level deep)
- Claude loads REFERENCE.md on demand when working with a skill

Uses ast to parse tools.py and extract:
- Public function names (not starting with _)
- Docstrings (full text, parsed into description/params/returns)
- Parameter annotations and type hints
"""

import ast
import os
import re
import textwrap
from pathlib import Path
from typing import Optional


SKILLS_DIR = Path("/var/www/sites/personal/stock_market/Jotty/skills")
MIN_TOOLS = 5


def parse_docstring(docstring: Optional[str]) -> dict:
    """Parse a docstring into description, params, and returns sections."""
    result = {"description": "", "params": [], "returns": ""}

    if not docstring:
        return result

    lines = textwrap.dedent(docstring).strip().split("\n")

    # State machine: description -> args -> returns
    state = "description"
    desc_lines = []
    param_lines = []
    return_lines = []
    current_param = None

    for line in lines:
        stripped = line.strip()

        # Detect section transitions
        if stripped.lower() in ("args:", "parameters:", "params:"):
            state = "args"
            continue
        if stripped.lower() in ("returns:", "return:", "yields:"):
            state = "returns"
            continue
        if stripped.lower() in ("raises:", "note:", "notes:", "examples:", "example:"):
            state = "other"
            continue

        if state == "description":
            desc_lines.append(stripped)
        elif state == "args":
            # Check if this is a new parameter line (e.g., "- path (str, required): ...")
            param_match = re.match(
                r"[-*]\s+(\w+)\s*\(([^)]*)\)\s*:\s*(.*)", stripped
            )
            if param_match:
                name, type_info, desc = param_match.groups()
                current_param = {
                    "name": name,
                    "type": type_info.strip(),
                    "description": desc.strip(),
                }
                param_lines.append(current_param)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                # Simple param line without type: "- name: description"
                simple_match = re.match(r"[-*]\s+(\w+)\s*:\s*(.*)", stripped)
                if simple_match:
                    name, desc = simple_match.groups()
                    current_param = {
                        "name": name,
                        "type": "",
                        "description": desc.strip(),
                    }
                    param_lines.append(current_param)
                else:
                    # Continuation or non-param list item
                    if current_param:
                        current_param["description"] += " " + stripped.lstrip("- *")
            elif stripped.startswith("params:") or stripped.startswith("params "):
                # "params: Dictionary containing:" - skip this meta-line
                continue
            elif current_param and stripped:
                # Continuation of previous param description
                current_param["description"] += " " + stripped
        elif state == "returns":
            if stripped:
                return_lines.append(stripped)

    # Clean up description
    desc_text = " ".join(desc_lines).strip()
    # Remove trailing "Args:" or "Parameters:" that might have been included
    desc_text = re.sub(r"\s*(Args|Parameters|Params)\s*:?\s*$", "", desc_text).strip()
    # If the description references "params: Dictionary containing:" skip that part
    desc_text = re.sub(
        r"\s*params\s*:\s*Dictionary containing\s*:?\s*$", "", desc_text
    ).strip()

    result["description"] = desc_text
    result["params"] = param_lines
    result["returns"] = " ".join(return_lines).strip()

    return result


def extract_tool_functions(tools_path: Path) -> list:
    """Extract public tool functions from a tools.py file using ast."""
    with open(tools_path, "r") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    seen_names = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        # Skip duplicates (e.g., same name defined in different classes)
        if node.name in seen_names:
            continue
        seen_names.add(node.name)

        # Extract docstring
        docstring = ast.get_docstring(node)
        parsed = parse_docstring(docstring)

        # Extract parameter info from function signature
        sig_params = []
        args = node.args

        # Get annotations
        for arg in args.args:
            if arg.arg in ("self", "cls"):
                continue

            annotation = ""
            if arg.annotation:
                annotation = ast.unparse(arg.annotation) if hasattr(ast, "unparse") else ""

            sig_params.append(
                {"name": arg.arg, "annotation": annotation}
            )

        # Extract return annotation
        return_annotation = ""
        if node.returns:
            return_annotation = (
                ast.unparse(node.returns) if hasattr(ast, "unparse") else ""
            )

        # Check for decorators (tool_wrapper, etc.)
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)

        is_tool = any(
            "tool" in d.lower() or node.name.endswith("_tool") for d in decorators
        ) or node.name.endswith("_tool")

        functions.append(
            {
                "name": node.name,
                "description": parsed["description"],
                "first_line": parsed["description"].split(".")[0].strip() + "."
                if parsed["description"]
                else "",
                "params_from_docstring": parsed["params"],
                "params_from_sig": sig_params,
                "returns_description": parsed["returns"],
                "return_annotation": return_annotation,
                "is_tool": is_tool,
                "decorators": decorators,
                "lineno": node.lineno,
            }
        )

    # Sort: tool functions first (ending in _tool), then helpers, by line number
    functions.sort(key=lambda f: (0 if f["is_tool"] else 1, f["lineno"]))

    return functions


def get_skill_name(skill_dir: Path) -> str:
    """Extract human-readable skill name from SKILL.md frontmatter or directory name."""
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        with open(skill_md) as f:
            content = f.read()
        # Look for "# Title" after frontmatter
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

    # Fallback: convert directory name
    return skill_dir.name.replace("-", " ").title()


def generate_reference_md(skill_name: str, functions: list) -> str:
    """Generate REFERENCE.md content for a skill."""
    lines = []
    lines.append(f"# {skill_name} - API Reference\n")

    # Table of contents: separate tools from helper functions
    tool_funcs = [f for f in functions if f["is_tool"]]
    helper_funcs = [f for f in functions if not f["is_tool"]]

    lines.append("## Contents\n")

    if tool_funcs:
        lines.append("### Tools\n")
        lines.append("| Function | Description |")
        lines.append("|----------|-------------|")
        for func in tool_funcs:
            brief = func["first_line"] if func["first_line"] else "No description available."
            # Truncate long descriptions for the table
            if len(brief) > 80:
                brief = brief[:77] + "..."
            lines.append(f"| [`{func['name']}`](#{func['name']}) | {brief} |")
        lines.append("")

    if helper_funcs:
        lines.append("### Helper Functions\n")
        lines.append("| Function | Description |")
        lines.append("|----------|-------------|")
        for func in helper_funcs:
            brief = func["first_line"] if func["first_line"] else "No description available."
            if len(brief) > 80:
                brief = brief[:77] + "..."
            lines.append(f"| [`{func['name']}`](#{func['name']}) | {brief} |")
        lines.append("")

    lines.append("---\n")

    # Detailed documentation for each function
    all_funcs = tool_funcs + helper_funcs

    for i, func in enumerate(all_funcs):
        lines.append(f"## `{func['name']}`\n")

        if func["description"]:
            lines.append(f"{func['description']}\n")
        else:
            lines.append("No description available.\n")

        # Parameters section
        doc_params = func["params_from_docstring"]
        sig_params = func["params_from_sig"]

        if doc_params:
            lines.append("**Parameters:**\n")
            for p in doc_params:
                type_str = f" (`{p['type']}`)" if p["type"] else ""
                desc = p["description"] if p["description"] else ""
                lines.append(f"- **{p['name']}**{type_str}: {desc}")
            lines.append("")
        elif sig_params:
            # Fall back to signature params if no docstring params
            lines.append("**Parameters:**\n")
            for p in sig_params:
                ann = f" (`{p['annotation']}`)" if p["annotation"] else ""
                lines.append(f"- **{p['name']}**{ann}")
            lines.append("")

        # Returns section
        if func["returns_description"]:
            lines.append(f"**Returns:** {func['returns_description']}\n")
        elif func["return_annotation"]:
            lines.append(f"**Returns:** `{func['return_annotation']}`\n")

        # Add separator between functions (but not after the last one)
        if i < len(all_funcs) - 1:
            lines.append("---\n")

    return "\n".join(lines)


def update_skill_md(skill_dir: Path) -> bool:
    """Insert ## Reference section into SKILL.md before ## Triggers."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return False

    with open(skill_md, "r") as f:
        content = f.read()

    # Already has reference section
    if "## Reference" in content:
        return False

    # Insert before ## Triggers
    reference_block = (
        "## Reference\n\n"
        "For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).\n\n"
    )

    # Try inserting before ## Triggers
    triggers_match = re.search(r"^## Triggers", content, re.MULTILINE)
    if triggers_match:
        insert_pos = triggers_match.start()
        new_content = content[:insert_pos] + reference_block + content[insert_pos:]
    else:
        # If no ## Triggers, insert before ## Category or at the end
        category_match = re.search(r"^## Category", content, re.MULTILINE)
        if category_match:
            insert_pos = category_match.start()
            new_content = content[:insert_pos] + reference_block + content[insert_pos:]
        else:
            # Append at end
            new_content = content.rstrip() + "\n\n" + reference_block

    with open(skill_md, "w") as f:
        f.write(new_content)

    return True


def main():
    skills_processed = 0
    total_tools_documented = 0
    skipped_existing = 0

    print(f"Scanning skills in: {SKILLS_DIR}")
    print(f"Minimum tools threshold: {MIN_TOOLS}\n")

    qualifying_skills = []

    for skill_name_dir in sorted(os.listdir(SKILLS_DIR)):
        skill_dir = SKILLS_DIR / skill_name_dir
        tools_path = skill_dir / "tools.py"
        skill_md_path = skill_dir / "SKILL.md"
        reference_path = skill_dir / "REFERENCE.md"

        if not tools_path.is_file():
            continue

        if not skill_md_path.is_file():
            continue

        # Check exclusion conditions
        if reference_path.exists():
            skipped_existing += 1
            continue

        with open(skill_md_path) as f:
            if "## Reference" in f.read():
                skipped_existing += 1
                continue

        # Extract functions
        functions = extract_tool_functions(tools_path)

        if len(functions) < MIN_TOOLS:
            continue

        qualifying_skills.append((skill_name_dir, skill_dir, functions))

    print(f"Found {len(qualifying_skills)} skills with {MIN_TOOLS}+ tools:\n")

    for skill_name_dir, skill_dir, functions in qualifying_skills:
        skill_name = get_skill_name(skill_dir)
        tool_count = len([f for f in functions if f["is_tool"]])
        helper_count = len([f for f in functions if not f["is_tool"]])

        # Generate REFERENCE.md
        reference_content = generate_reference_md(skill_name, functions)
        reference_path = skill_dir / "REFERENCE.md"

        with open(reference_path, "w") as f:
            f.write(reference_content)

        # Update SKILL.md
        updated = update_skill_md(skill_dir)

        skills_processed += 1
        total_tools_documented += len(functions)

        md_status = "updated" if updated else "skipped (no insertion point)"
        print(
            f"  [{skills_processed:2d}] {skill_name_dir:40s} "
            f"| {len(functions):2d} functions ({tool_count} tools, {helper_count} helpers) "
            f"| REFERENCE.md created | SKILL.md {md_status}"
        )

    print(f"\n{'=' * 70}")
    print(f"Skills processed:        {skills_processed}")
    print(f"Total tools documented:  {total_tools_documented}")
    print(f"Skipped (existing):      {skipped_existing}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
