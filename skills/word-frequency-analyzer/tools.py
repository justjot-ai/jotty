"""Word frequency analyzer — count, rank, cloud data."""

import re
from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("word-frequency-analyzer")

_STOP = frozenset(
    "a an the and or but in on at to for of is it its this that was were be "
    "been being have has had do does did will would shall should may might can "
    "could am are not no nor so if then than too very just about above after "
    "again all also any because before between both by during each few from "
    "further get got he her here hers herself him himself his how i into me "
    "more most my myself off once only other our ours ourselves out over own "
    "same she some such them themselves there these they those through under "
    "until up us we what when where which while who whom why with you your "
    "yours yourself yourselves".split()
)


@tool_wrapper(required_params=["text"])
def word_freq_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Count word frequencies excluding stop words."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    top_n = int(params.get("top_n", 20))
    include_stop = params.get("include_stop_words", False)
    words = re.findall(r"[a-zA-Z\']+", text.lower())
    total = len(words)
    freq: dict = {}
    for w in words:
        if not include_stop and w in _STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    top = ranked[:top_n]
    max_count = top[0][1] if top else 1
    cloud = [{"word": w, "count": c, "weight": round(c / max_count, 3)} for w, c in top]
    return tool_response(
        total_words=total,
        unique_words=len(freq),
        top_words=[{"word": w, "count": c} for w, c in top],
        word_cloud_data=cloud,
        stop_words_excluded=not include_stop,
    )


__all__ = ["word_freq_tool"]
