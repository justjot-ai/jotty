"""Generate README.md files from project info."""

from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("readme-generator")


@tool_wrapper(required_params=["name", "description"])
def generate_readme(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a README.md from project metadata.

    Params:
        name: project name
        description: short description
        features: list of feature strings
        installation: install instructions string
        usage: usage example string
        license: license name (e.g. MIT)
        badges: list of {label, url} badge dicts
        contributing: contributing guidelines
        author: author name
    """
    status.set_callback(params.pop("_status_callback", None))
    name = params["name"]
    desc = params["description"]
    sections = []

    # Title + description
    sections.append(f"# {name}\n\n{desc}")

    # Badges
    for b in params.get("badges", []):
        sections.append(f"![{b.get('label', '')}]({b.get('url', '')})")

    # Features
    features = params.get("features", [])
    if features:
        items = "\n".join(f"- {f}" for f in features)
        sections.append(f"## Features\n\n{items}")

    # Installation
    install = params.get("installation", "")
    if install:
        sections.append(f"## Installation\n\n```bash\n{install}\n```")

    # Usage
    usage = params.get("usage", "")
    if usage:
        sections.append(f"## Usage\n\n```\n{usage}\n```")

    # Contributing
    contrib = params.get("contributing", "")
    if contrib:
        sections.append(f"## Contributing\n\n{contrib}")

    # Author
    author = params.get("author", "")
    if author:
        sections.append(f"## Author\n\n{author}")

    # License
    lic = params.get("license", "")
    if lic:
        sections.append(f"## License\n\n{lic}")

    readme = "\n\n".join(sections) + "\n"
    return tool_response(readme=readme, sections=len(sections))


__all__ = ["generate_readme"]
