"""Slug Generator Skill — create URL-friendly slugs from text."""
import re
import unicodedata
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("slug-generator")

# Common unicode transliteration replacements
_REPLACEMENTS = {
    "\u00e4": "ae", "\u00f6": "oe", "\u00fc": "ue",
    "\u00c4": "Ae", "\u00d6": "Oe", "\u00dc": "Ue",
    "\u00df": "ss", "\u00e9": "e", "\u00e8": "e",
    "\u00e0": "a", "\u00e2": "a", "\u00f4": "o",
    "\u00e7": "c", "\u00f1": "n", "\u00ee": "i", "\u00f9": "u",
}


def _transliterate(text: str) -> str:
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@tool_wrapper(required_params=["text"])
def slugify_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a URL-friendly slug from text."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    sep = params.get("separator", "-")
    max_len = params.get("max_length", 200)
    lower = params.get("lowercase", True)

    slug = _transliterate(text)
    if lower:
        slug = slug.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", sep, slug).strip(sep)
    if max_len and len(slug) > max_len:
        slug = slug[:max_len].rstrip(sep)
    return tool_response(slug=slug, original=text, length=len(slug))


__all__ = ["slugify_tool"]
