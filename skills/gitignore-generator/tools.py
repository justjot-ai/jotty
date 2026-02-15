"""Generate .gitignore files for languages/frameworks."""

from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("gitignore-generator")

TEMPLATES: Dict[str, List[str]] = {
    "python": [
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        "*.so",
        "dist/",
        "build/",
        "*.egg-info/",
        ".eggs/",
        "*.egg",
        ".venv/",
        "venv/",
        "env/",
        ".env",
        ".pytest_cache/",
        ".mypy_cache/",
        "*.pyo",
        "htmlcov/",
        ".coverage",
        ".tox/",
    ],
    "node": [
        "node_modules/",
        "npm-debug.log*",
        "yarn-debug.log*",
        "yarn-error.log*",
        ".npm",
        ".yarn-integrity",
        "dist/",
        "build/",
        ".env",
        ".env.local",
        ".env.*.local",
        "coverage/",
        ".next/",
        ".nuxt/",
    ],
    "java": [
        "*.class",
        "*.jar",
        "*.war",
        "*.ear",
        "target/",
        ".gradle/",
        "build/",
        ".settings/",
        ".project",
        ".classpath",
        "*.iml",
        ".idea/",
        "out/",
    ],
    "go": [
        "*.exe",
        "*.exe~",
        "*.dll",
        "*.so",
        "*.dylib",
        "*.test",
        "*.out",
        "vendor/",
        "go.sum",
        ".env",
    ],
    "rust": [
        "target/",
        "**/*.rs.bk",
        "Cargo.lock",
    ],
    "c": [
        "*.o",
        "*.obj",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.a",
        "*.lib",
        "*.exe",
        "*.out",
        "build/",
        "cmake-build-*/",
    ],
    "general": [
        ".DS_Store",
        "Thumbs.db",
        "*.swp",
        "*.swo",
        "*~",
        ".idea/",
        ".vscode/",
        "*.log",
    ],
}


@tool_wrapper(required_params=["languages"])
def generate_gitignore(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a .gitignore file for given languages/frameworks.

    Params:
        languages: list of language keys (python, node, java, go, rust, c, general)
        extras: list of additional patterns to include
    """
    status.set_callback(params.pop("_status_callback", None))
    langs = params["languages"]
    if isinstance(langs, str):
        langs = [l.strip() for l in langs.split(",")]
    extras = params.get("extras", [])

    lines: List[str] = ["# Auto-generated .gitignore", ""]
    for lang in langs:
        key = lang.lower().strip()
        patterns = TEMPLATES.get(key, [])
        if not patterns:
            lines.append(f"# Unknown language: {key}")
            continue
        lines.append(f"# --- {key.title()} ---")
        lines.extend(patterns)
        lines.append("")

    if extras:
        lines.append("# --- Custom ---")
        lines.extend(extras)
        lines.append("")

    content = "\n".join(lines)
    return tool_response(gitignore=content, languages=langs)


__all__ = ["generate_gitignore"]
