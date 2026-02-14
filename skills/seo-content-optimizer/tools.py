"""SEO Content Optimizer Skill - analyze text for SEO metrics."""
import re
import math
from typing import Dict, Any, List
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("seo-content-optimizer")


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:es|ed|e)$", "", word) or word
    vowels = re.findall(r"[aeiouy]+", word)
    return max(1, len(vowels))


def _flesch_reading_ease(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    if not sentences or not words:
        return 0.0
    total_syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = total_syllables / len(words)
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return round(max(0, min(100, score)), 1)


def _reading_level(score: float) -> str:
    if score >= 90:
        return "5th grade (very easy)"
    elif score >= 80:
        return "6th grade (easy)"
    elif score >= 70:
        return "7th grade (fairly easy)"
    elif score >= 60:
        return "8th-9th grade (standard)"
    elif score >= 50:
        return "10th-12th grade (fairly difficult)"
    elif score >= 30:
        return "College (difficult)"
    return "College graduate (very difficult)"


@tool_wrapper(required_params=["text"])
def analyze_seo_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze text for SEO: keyword density, readability, suggestions."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    keywords = params.get("keywords", [])
    title = params.get("title", "")

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Readability
    flesch = _flesch_reading_ease(text)
    avg_sentence_len = round(word_count / max(sentence_count, 1), 1)
    avg_word_len = round(sum(len(w) for w in words) / max(word_count, 1), 1)

    # Keyword density
    kw_density = {}
    for kw in keywords:
        kw_lower = kw.lower()
        count = text.lower().count(kw_lower)
        density = round((count / max(word_count, 1)) * 100, 2)
        kw_density[kw] = {"count": count, "density_pct": density}

    # Suggestions
    suggestions = []
    if word_count < 300:
        suggestions.append("Content is short. Aim for 300+ words for better SEO.")
    if avg_sentence_len > 25:
        suggestions.append("Sentences are long. Aim for under 20 words per sentence.")
    if flesch < 60:
        suggestions.append("Readability is low. Simplify language for broader audience.")
    if title and len(title) > 60:
        suggestions.append("Title is over 60 characters. Keep under 60 for search results.")
    if title and len(title) < 30:
        suggestions.append("Title is short. Aim for 30-60 characters.")
    for kw, info in kw_density.items():
        if info["density_pct"] < 0.5:
            suggestions.append(f"Keyword '{kw}' density is low ({info['density_pct']}%). Aim for 1-2%.")
        elif info["density_pct"] > 3.0:
            suggestions.append(f"Keyword '{kw}' density is high ({info['density_pct']}%). May be seen as stuffing.")
    if not any(s.startswith(("# ", "## ")) for s in text.split("\n")):
        suggestions.append("No headings detected. Use headings (H1, H2) to structure content.")

    return tool_response(
        word_count=word_count, sentence_count=sentence_count,
        paragraph_count=len(paragraphs),
        readability={"flesch_score": flesch, "level": _reading_level(flesch),
                     "avg_sentence_length": avg_sentence_len,
                     "avg_word_length": avg_word_len},
        keyword_density=kw_density,
        suggestions=suggestions,
    )


__all__ = ["analyze_seo_tool"]
