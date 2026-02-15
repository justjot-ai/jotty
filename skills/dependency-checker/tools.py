"""Parse requirements.txt/package.json and check for issues."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("dependency-checker")

# Known packages with security advisories or deprecations
_KNOWN_ISSUES: Dict[str, str] = {
    "requests": "Ensure >= 2.31.0 (CVE-2023-32681)",
    "urllib3": "Ensure >= 2.0.7 (CVE-2023-45803)",
    "cryptography": "Ensure >= 41.0.6 (multiple CVEs)",
    "pillow": "Ensure >= 10.2.0 (CVE-2023-50447)",
    "django": "Ensure >= 4.2.8 (security updates)",
    "flask": "Ensure >= 3.0.0 for latest security patches",
    "jinja2": "Ensure >= 3.1.3 (CVE-2024-22195)",
    "setuptools": "Ensure >= 70.0.0 (CVE-2024-6345)",
    "certifi": "Ensure >= 2024.2.2 for updated CA bundle",
    "axios": "Ensure >= 1.6.0 (CVE-2023-45857)",
    "lodash": "Ensure >= 4.17.21 (prototype pollution)",
    "express": "Ensure >= 4.19.2 (CVE-2024-29041)",
    "jsonwebtoken": "Ensure >= 9.0.0 (CVE-2022-23529)",
}


def _parse_requirements(content: str) -> List[Dict[str, Any]]:
    deps = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = re.match(r"^([a-zA-Z0-9_.-]+)\s*([><=!~]+)?\s*([\d.*]+)?", line)
        if match:
            name = match.group(1).lower()
            op = match.group(2) or ""
            ver = match.group(3) or ""
            issue = _KNOWN_ISSUES.get(name)
            deps.append(
                {
                    "name": name,
                    "operator": op,
                    "version": ver,
                    "pinned": op == "==",
                    "advisory": issue,
                }
            )
    return deps


def _parse_package_json(content: str) -> List[Dict[str, Any]]:
    deps = []
    try:
        pkg = json.loads(content)
    except json.JSONDecodeError:
        return deps
    for section in ("dependencies", "devDependencies"):
        for name, ver in pkg.get(section, {}).items():
            clean = re.sub(r"[^\d.]", "", ver)
            issue = _KNOWN_ISSUES.get(name)
            deps.append(
                {
                    "name": name,
                    "version": ver,
                    "clean_version": clean,
                    "pinned": not ver.startswith("^") and not ver.startswith("~"),
                    "dev": section == "devDependencies",
                    "advisory": issue,
                }
            )
    return deps


@tool_wrapper(required_params=["content"])
def check_dependencies(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and check dependencies for issues.

    Params:
        content: file content (requirements.txt or package.json)
        file_type: 'requirements' or 'package_json' (auto-detected if omitted)
    """
    status.set_callback(params.pop("_status_callback", None))
    content = params["content"]
    ftype = params.get("file_type", "")

    if not ftype:
        ftype = "package_json" if content.strip().startswith("{") else "requirements"

    if ftype == "package_json":
        deps = _parse_package_json(content)
    else:
        deps = _parse_requirements(content)

    advisories = [d for d in deps if d.get("advisory")]
    unpinned = [d for d in deps if not d.get("pinned")]

    return tool_response(
        dependencies=deps,
        total=len(deps),
        advisories=advisories,
        advisory_count=len(advisories),
        unpinned=unpinned,
        unpinned_count=len(unpinned),
        file_type=ftype,
    )


__all__ = ["check_dependencies"]
