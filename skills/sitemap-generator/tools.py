"""Generate XML sitemaps from URL lists."""

from datetime import datetime
from typing import Any, Dict, List
from xml.sax.saxutils import escape

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("sitemap-generator")


@tool_wrapper(required_params=["urls"])
def generate_sitemap(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an XML sitemap from a list of URLs.

    Params:
        urls: list of URL strings or list of dicts:
            - url: the URL (required)
            - lastmod: last modification date (YYYY-MM-DD)
            - changefreq: always|hourly|daily|weekly|monthly|yearly|never
            - priority: 0.0 to 1.0
        default_changefreq: default changefreq for simple URL strings
        default_priority: default priority for simple URL strings (default 0.5)
    """
    status.set_callback(params.pop("_status_callback", None))
    urls = params["urls"]
    default_cf = params.get("default_changefreq", "weekly")
    default_pri = float(params.get("default_priority", 0.5))

    valid_freqs = {"always", "hourly", "daily", "weekly", "monthly", "yearly", "never"}

    lines = [
        """<?xml version="1.0" encoding="UTF-8"?>""",
        """<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">""",
    ]

    count = 0
    for entry in urls:
        if isinstance(entry, str):
            entry = {"url": entry}
        url = entry.get("url", "")
        if not url:
            continue
        lastmod = entry.get("lastmod", datetime.now().strftime("%Y-%m-%d"))
        cf = entry.get("changefreq", default_cf)
        if cf not in valid_freqs:
            cf = default_cf
        pri = float(entry.get("priority", default_pri))
        pri = max(0.0, min(1.0, pri))

        lines.append("  <url>")
        lines.append(f"    <loc>{escape(url)}</loc>")
        lines.append(f"    <lastmod>{escape(lastmod)}</lastmod>")
        lines.append(f"    <changefreq>{cf}</changefreq>")
        lines.append(f"    <priority>{pri:.1f}</priority>")
        lines.append("  </url>")
        count += 1

    lines.append("</urlset>")
    xml = "\n".join(lines)
    return tool_response(sitemap_xml=xml, url_count=count)


__all__ = ["generate_sitemap"]
