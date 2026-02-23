"""
Response Analyzer - Heuristic quality signal extraction from LLM responses.

Pure functions for analyzing response content quality. No LLM calls,
no state, only regex-based signal detection.

Extracted from learning_service.py for modularity.
"""

from __future__ import annotations

import re
from typing import Any, Dict

# Response quality signal detectors
_STRUCTURE_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:"
    r"#{1,4}\s|"  # Markdown headings
    r"\d+\.\s|"  # Numbered lists
    r"[A-Z]\.\s|"  # Lettered sections
    r"\*\*[A-Z]|"  # Bold section headers
    r"```|"  # Code blocks
    r"\|.*\|.*\|"  # Tables
    r")",
    re.MULTILINE,
)

# Match fenced code blocks -- also handle unclosed blocks (LLMs sometimes omit
# the closing ```) by making the closing fence optional at end-of-string.
_CODE_BLOCK_RE = re.compile(r"```\w*\n.*?(?:```|$)", re.DOTALL)
_CITATION_RE = re.compile(r"\b(?:et al\.?|(?:19|20)\d{2}[a-z]?)\b")


def analyze_response(content: str, goal: str) -> Dict[str, Any]:
    """
    Analyze LLM response to extract quality signals. Pure heuristics,
    no LLM calls. Returns a dict of features for learning records.

    Scoring breakdown (max 1.0):
      0.20 -- Goal coverage (keyword overlap with goal)
      0.15 -- Structure (headings, lists, formatting)
      0.15 -- Depth (word count, paragraphs)
      0.15 -- Code quality (when code expected: blocks, classes, tests)
      0.10 -- Explanation quality (reasoning markers, examples)
      0.10 -- Completeness (intro + conclusion, multiple sections)
      0.05 -- Math/formulas
      0.05 -- Citations/references
      0.05 -- Tables/comparisons
    """
    if not content:
        return {"empty": True, "quality_score": 0.0}

    content_lower = content.lower()
    goal_lower = goal.lower()

    # -- Structure analysis --
    structure_hits = len(_STRUCTURE_MARKERS.findall(content))
    has_headings = bool(re.search(r"(?:^|\n)#{1,4}\s", content))
    heading_count = len(re.findall(r"(?:^|\n)#{1,4}\s", content))
    has_numbered_list = bool(re.search(r"(?:^|\n)\s*\d+\.\s", content))
    has_bullet_list = bool(re.search(r"(?:^|\n)\s*[-*]\s", content))
    has_code_blocks = bool(_CODE_BLOCK_RE.search(content))
    code_blocks = _CODE_BLOCK_RE.findall(content)
    code_block_count = len(code_blocks)
    has_table = bool(re.search(r"\|.*\|.*\|", content))

    # -- Depth signals --
    word_count = len(content.split())
    paragraph_count = len([p for p in content.split("\n\n") if p.strip()])
    has_citations = bool(_CITATION_RE.search(content))
    has_math = bool(
        re.search(
            r"[=\u2211\u222b\u2202\u2207\u03bb\u03b1\u03b2\u03b3]|\\frac|O\(n|O\(log", content
        )
    )

    # -- Goal coverage --
    goal_keywords = set(re.findall(r"\b[a-z]{4,}\b", goal_lower))
    response_keywords = set(re.findall(r"\b[a-z]{4,}\b", content_lower))
    if goal_keywords:
        coverage = len(goal_keywords & response_keywords) / len(goal_keywords)
    else:
        coverage = 0.5

    # -- Code quality signals (deeper than just counting blocks) --
    _impl_keywords = {
        "implement",
        "code",
        "class",
        "function",
        "method",
        "test",
        "write a",
        "build a",
        "create a",
        "algorithm",
        "program",
        "develop",
        "script",
        "module",
    }
    expects_code = any(kw in goal_lower for kw in _impl_keywords)

    # Analyze code block content for actual substance
    code_lines_total = 0
    has_class_def = False
    has_function_def = False
    has_assertions = False
    has_test_func = False
    assertion_count = 0
    for block in code_blocks:
        code_lines_total += block.count("\n")
        if re.search(r"\bclass\s+\w+", block):
            has_class_def = True
        if re.search(r"\bdef\s+\w+", block):
            has_function_def = True
        if re.search(r"\bassert\b", block):
            has_assertions = True
            assertion_count += len(re.findall(r"\bassert\b", block))
        if re.search(r"\bdef\s+test_", block):
            has_test_func = True

    # -- Explanation quality signals --
    _reasoning_markers = re.findall(
        r"\b(?:because|therefore|since|however|although|"
        r"this means|in other words|for example|specifically|"
        r"the reason|note that|importantly|consequently|thus|"
        r"consider|observe that|intuitively)\b",
        content_lower,
    )
    reasoning_density = len(_reasoning_markers) / max(word_count / 100, 1)

    # Examples and illustrations
    example_count = len(
        re.findall(
            r"(?:for example|e\.g\.|for instance|such as|consider|"
            r"suppose|let's say|imagine|here's an example)",
            content_lower,
        )
    )

    # -- Completeness signals --
    sections = re.split(r"\n#{1,4}\s", content)
    has_intro = bool(sections and len(sections[0].strip()) > 50)
    has_conclusion = bool(
        re.search(
            r"(?:in (?:summary|conclusion)|to summarize|overall|" r"in short|key takeaway|final)",
            content_lower[-500:] if len(content) > 500 else content_lower,
        )
    )
    multi_section = heading_count >= 3

    # -- Error/incompleteness detection --
    has_todo = bool(re.search(r"\bTODO\b|\bFIXME\b|\bXXX\b|\.{3}\s*$", content))
    has_error_msg = bool(
        re.search(
            r"(?:Error|Exception|Traceback|undefined|not implemented)",
            content,
        )
    )
    has_truncation = content.rstrip().endswith("...") or content.rstrip().endswith("\u2026")

    # ===================================================================
    # SCORING -- calibrated weights summing to 1.0
    # ===================================================================
    quality = 0.0

    # 1. Goal coverage (0.20)
    quality += coverage * 0.20

    # 2. Structure (0.15)
    struct_score = 0.0
    struct_score += min(0.06, structure_hits * 0.01)
    struct_score += 0.03 if has_headings else 0.0
    struct_score += 0.03 if has_numbered_list or has_bullet_list else 0.0
    struct_score += 0.03 if multi_section else 0.0
    quality += min(0.15, struct_score)

    # 3. Depth (0.15)
    depth_score = 0.0
    depth_score += min(0.08, word_count / 3000 * 0.08)  # 3000 words = max
    depth_score += min(0.07, paragraph_count / 8 * 0.07)  # 8 paragraphs = max
    quality += min(0.15, depth_score)

    # 4. Code quality (0.15) -- only when code expected
    if expects_code:
        code_score = 0.0
        code_score += min(0.04, code_block_count * 0.02)  # multiple blocks
        code_score += min(0.03, code_lines_total / 50 * 0.03)  # substantial code
        code_score += 0.02 if has_class_def else 0.0  # proper OOP
        code_score += 0.02 if has_function_def else 0.0  # proper functions
        code_score += 0.02 if has_test_func else 0.0  # test functions
        code_score += min(0.02, assertion_count / 5 * 0.02)  # assertions
        quality += min(0.15, code_score)
        if code_block_count == 0:
            quality *= 0.5  # No code at all when expected = major penalty
    else:
        quality += 0.05 if has_code_blocks else 0.0
        quality += 0.05 if word_count >= 400 else 0.0  # non-code needs depth

    # 5. Explanation quality (0.10)
    explain_score = 0.0
    explain_score += min(0.05, reasoning_density * 0.025)  # reasoning words
    explain_score += min(0.05, example_count * 0.02)  # examples/illustrations
    quality += min(0.10, explain_score)

    # 6. Completeness (0.10)
    complete_score = 0.0
    complete_score += 0.03 if has_intro else 0.0
    complete_score += 0.03 if has_conclusion else 0.0
    complete_score += 0.04 if multi_section else 0.0
    quality += min(0.10, complete_score)

    # 7. Math/formulas (0.05)
    quality += 0.05 if has_math else 0.0

    # 8. Citations/references (0.05)
    quality += 0.05 if has_citations else 0.0

    # 9. Tables/comparisons (0.05)
    quality += 0.05 if has_table else 0.0

    # -- Penalties for errors/incompleteness --
    if has_todo:
        quality *= 0.85
    if has_error_msg and expects_code:
        quality *= 0.90
    if has_truncation:
        quality *= 0.90

    quality = round(min(1.0, max(0.0, quality)), 3)

    return {
        "quality_score": quality,
        "word_count": word_count,
        "paragraph_count": paragraph_count,
        "structure_score": min(1.0, structure_hits / 10),
        "goal_coverage": round(coverage, 3),
        "has_code": has_code_blocks,
        "code_block_count": code_block_count,
        "code_lines": code_lines_total,
        "has_class": has_class_def,
        "has_functions": has_function_def,
        "has_tests": has_test_func,
        "assertion_count": assertion_count,
        "has_headings": has_headings,
        "heading_count": heading_count,
        "has_numbered_list": has_numbered_list,
        "has_table": has_table,
        "has_citations": has_citations,
        "has_math": has_math,
        "reasoning_density": round(reasoning_density, 2),
        "example_count": example_count,
        "has_conclusion": has_conclusion,
        "content_length": len(content),
    }
