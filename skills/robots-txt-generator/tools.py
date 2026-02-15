"""Generate robots.txt files with user-agent rules."""
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("robots-txt-generator")


@tool_wrapper(required_params=["rules"])
def generate_robots_txt(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a robots.txt file.

    Params:
        rules: list of rule dicts:
            - user_agent: bot name or "*" (default "*")
            - allow: list of allowed paths
            - disallow: list of disallowed paths
            - crawl_delay: delay in seconds (optional)
        sitemaps: list of sitemap URLs
        host: preferred host (optional)
    """
    status.set_callback(params.pop("_status_callback", None))
    rules = params["rules"]
    sitemaps = params.get("sitemaps", [])
    host = params.get("host", "")

    lines: List[str] = ["# robots.txt", "# Auto-generated", ""]

    for rule in rules:
        ua = rule.get("user_agent", "*")
        lines.append(f"User-agent: {ua}")

        for path in rule.get("allow", []):
            lines.append(f"Allow: {path}")
        for path in rule.get("disallow", []):
            lines.append(f"Disallow: {path}")

        delay = rule.get("crawl_delay")
        if delay is not None:
            lines.append(f"Crawl-delay: {delay}")
        lines.append("")

    for sm in sitemaps:
        lines.append(f"Sitemap: {sm}")

    if host:
        lines.append(f"Host: {host}")

    if sitemaps or host:
        lines.append("")

    content = "\n".join(lines)
    return tool_response(robots_txt=content, rule_count=len(rules))


__all__ = ["generate_robots_txt"]
