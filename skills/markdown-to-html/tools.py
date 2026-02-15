"""Markdown to HTML Skill — convert markdown to styled HTML."""

import re
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("markdown-to-html")

CSS = """<style>
body{font-family:system-ui,sans-serif;line-height:1.6;max-width:800px;margin:0 auto;padding:20px;color:#333}
h1,h2,h3{color:#1a1a1a;border-bottom:1px solid #eee;padding-bottom:0.3em}
code{background:#f4f4f4;padding:2px 6px;border-radius:3px;font-size:0.9em}
pre code{display:block;padding:16px;overflow-x:auto}
blockquote{border-left:4px solid #ddd;margin:0;padding:0 16px;color:#666}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:8px;text-align:left}
th{background:#f4f4f4}
a{color:#0366d6}
</style>"""


def _md_to_html(md: str) -> str:
    html = md
    # Code blocks
    html = re.sub(
        r"```(\w*)\n(.*?)```",
        lambda m: f'<pre><code class="{m.group(1)}">{m.group(2)}</code></pre>',
        html,
        flags=re.DOTALL,
    )
    # Inline code
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    # Headers
    html = re.sub(r"^######\s+(.+)$", r"<h6>\1</h6>", html, flags=re.MULTILINE)
    html = re.sub(r"^#####\s+(.+)$", r"<h5>\1</h5>", html, flags=re.MULTILINE)
    html = re.sub(r"^####\s+(.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
    html = re.sub(r"^###\s+(.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^##\s+(.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^#\s+(.+)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
    # Bold and italic
    html = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", html)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    # Links and images
    html = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1">', html)
    html = re.sub(r"\[([^\]]*)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)
    # Lists
    html = re.sub(r"^[-*]\s+(.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
    # Blockquotes
    html = re.sub(r"^>\s+(.+)$", r"<blockquote>\1</blockquote>", html, flags=re.MULTILINE)
    # Horizontal rules
    html = re.sub(r"^---+$", r"<hr>", html, flags=re.MULTILINE)
    # Paragraphs
    html = re.sub(r"\n\n+", r"\n</p><p>\n", html)
    html = f"<p>{html}</p>"

    return html


@tool_wrapper(required_params=["markdown"])
def markdown_to_html_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Markdown text to HTML."""
    status.set_callback(params.pop("_status_callback", None))
    md = params["markdown"]
    include_style = params.get("include_style", True)

    html = _md_to_html(md)
    if include_style:
        html = CSS + html

    return tool_response(html=html, char_count=len(html))


__all__ = ["markdown_to_html_tool"]
