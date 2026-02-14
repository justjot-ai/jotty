"""Text statistics — word count, reading time, Flesch-Kincaid."""
import re, math
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus
status = SkillStatus("text-statistics")


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 2:
        return 1
    word = re.sub(r"(?:es|ed|e)$", "", word) or word
    vowels = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowels))


@tool_wrapper(required_params=["text"])
def analyze_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return word count, char count, sentence count, reading time, FK grade."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    words = re.findall(r"\b[a-zA-Z0-9\']+\b", text)
    word_count = len(words)
    char_count = len(text)
    char_no_spaces = len(text.replace(" ", ""))
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    reading_time_min = round(word_count / 238, 2)
    total_syllables = sum(_count_syllables(w) for w in words)
    if word_count > 0:
        fk = (0.39 * (word_count / sentence_count)
              + 11.8 * (total_syllables / word_count)
              - 15.59)
        fk = round(fk, 2)
    else:
        fk = 0.0
    return tool_response(
        word_count=word_count, char_count=char_count,
        char_count_no_spaces=char_no_spaces,
        sentence_count=len(sentences), reading_time_minutes=reading_time_min,
        flesch_kincaid_grade=fk, syllable_count=total_syllables,
    )


__all__ = ["analyze_text_tool"]
