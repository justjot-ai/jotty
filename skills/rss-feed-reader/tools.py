"""RSS Feed Reader Skill - fetch and parse RSS/Atom feeds."""

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("rss-feed-reader")

# Common RSS/Atom namespaces
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "media": "http://search.yahoo.com/mrss/",
}


def _text(el: Optional[ET.Element], default: str = "") -> str:
    if el is None:
        return default
    return (el.text or "").strip()


def _parse_rss(root: ET.Element, limit: int) -> dict:
    channel = root.find("channel")
    if channel is None:
        return {"title": "", "items": []}
    title = _text(channel.find("title"))
    description = _text(channel.find("description"))
    items = []
    for item in channel.findall("item")[:limit]:
        items.append(
            {
                "title": _text(item.find("title")),
                "link": _text(item.find("link")),
                "description": _text(item.find("description"))[:500],
                "pub_date": _text(item.find("pubDate")),
                "author": _text(item.find("dc:creator", NS)) or _text(item.find("author")),
            }
        )
    return {"title": title, "description": description, "items": items}


def _parse_atom(root: ET.Element, limit: int) -> dict:
    ns = NS["atom"]
    title = _text(root.find(f"{{{ns}}}title"))
    items = []
    for entry in root.findall(f"{{{ns}}}entry")[:limit]:
        link_el = entry.find(f"{{{ns}}}link")
        link = link_el.get("href", "") if link_el is not None else ""
        summary = _text(entry.find(f"{{{ns}}}summary")) or _text(entry.find(f"{{{ns}}}content"))
        items.append(
            {
                "title": _text(entry.find(f"{{{ns}}}title")),
                "link": link,
                "description": summary[:500],
                "pub_date": _text(entry.find(f"{{{ns}}}updated"))
                or _text(entry.find(f"{{{ns}}}published")),
                "author": _text(entry.find(f"{{{ns}}}author/{{{ns}}}name")),
            }
        )
    return {"title": title, "description": "", "items": items}


@tool_wrapper(required_params=["url"])
def fetch_rss_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch and parse an RSS or Atom feed."""
    status.set_callback(params.pop("_status_callback", None))
    url = params["url"]
    limit = min(max(int(params.get("limit", 10)), 1), 100)

    try:
        import requests

        resp = requests.get(url, timeout=15, headers={"User-Agent": "JottyRSSReader/1.0"})
        resp.raise_for_status()
        xml_text = resp.text
    except Exception as e:
        return tool_error(f"Failed to fetch feed: {e}")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return tool_error(f"Failed to parse XML: {e}")

    tag = root.tag.lower().split("}")[-1] if "}" in root.tag else root.tag.lower()
    if tag == "rss" or root.find("channel") is not None:
        result = _parse_rss(root, limit)
    elif "feed" in tag:
        result = _parse_atom(root, limit)
    else:
        return tool_error(f"Unrecognized feed format: {root.tag}")

    return tool_response(
        feed_title=result["title"],
        feed_description=result.get("description", ""),
        items=result["items"],
        item_count=len(result["items"]),
        feed_url=url,
    )


__all__ = ["fetch_rss_tool"]
