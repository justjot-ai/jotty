"""Topic Research Swarm — Robust LLM output parsers.

DSPy/LLM outputs are unpredictable — they may return:
  - Strict pipe format: "CLAIM: x | EVIDENCE: y | CONFIDENCE: 0.8 || CLAIM: z ..."
  - JSON: [{"claim": "x", "evidence": "y"}]
  - Numbered lists: "1. x\n2. y"
  - Markdown bullets: "- x\n- y"
  - Mixed formats

These parsers handle all variants gracefully.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def parse_findings(raw: Any, sub_topic: str = "") -> List[Dict[str, Any]]:
    """Parse findings from LLM output — handles pipe format, JSON, and free text.

    Returns list of dicts with: claim, evidence, confidence, sub_topic
    """
    if not raw:
        return []

    # If it's already a list (LLM returned structured)
    if isinstance(raw, list):
        return _normalize_finding_list(raw, sub_topic)

    # If it's a dict (single finding)
    if isinstance(raw, dict):
        return _normalize_finding_list([raw], sub_topic)

    text = str(raw)

    # Try JSON parse first
    findings = _try_json_parse(text, sub_topic)
    if findings:
        return findings

    # Try pipe-delimited format: "CLAIM: x | EVIDENCE: y || CLAIM: z"
    findings = _try_pipe_parse(text, sub_topic)
    if findings:
        return findings

    # Try numbered list: "1. CLAIM: x, EVIDENCE: y\n2. CLAIM: z"
    findings = _try_numbered_parse(text, sub_topic)
    if findings:
        return findings

    # Fallback: treat each non-empty line as a claim
    findings = _try_line_parse(text, sub_topic)
    return findings


def parse_key_numbers(raw: Any) -> Dict[str, str]:
    """Parse key numbers/data points from LLM output."""
    if not raw:
        return {}

    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if v}

    if isinstance(raw, list):
        result = {}
        for item in raw:
            if isinstance(item, dict):
                for k, v in item.items():
                    result[str(k)] = str(v)
            elif isinstance(item, str) and ":" in item:
                k, v = item.split(":", 1)
                result[k.strip()] = v.strip()
        return result

    text = str(raw)

    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items() if v}
    except (json.JSONDecodeError, ValueError):
        pass

    # Pipe-delimited key:value pairs
    result = {}
    for pair in re.split(r"\||\n", text):
        pair = pair.strip()
        if not pair:
            continue
        # Remove bullet markers
        pair = re.sub(r"^[-*•]\s*", "", pair)
        if ":" in pair:
            k, v = pair.split(":", 1)
            k = k.strip().strip('"').strip("'")
            v = v.strip().strip('"').strip("'")
            if k and v and len(k) < 100:
                result[k] = v
    return result


def parse_comparison_matrix(raw: Any, entities: List[str]) -> List[Dict[str, Any]]:
    """Parse comparison matrix from LLM output."""
    if not raw:
        return []

    if isinstance(raw, list):
        return _normalize_comparison_list(raw, entities)

    if isinstance(raw, dict):
        return _normalize_comparison_list([raw], entities)

    text = str(raw)

    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _normalize_comparison_list(parsed, entities)
    except (json.JSONDecodeError, ValueError):
        pass

    # Pipe-delimited: "DIMENSION: x | ENTITY1: val | ENTITY2: val | WINNER: w || ..."
    matrix = []
    for dim_str in text.split("||"):
        parts = {}
        for part in re.split(r"\|", dim_str):
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                parts[key.strip().upper()] = val.strip()

        if parts.get("DIMENSION"):
            entry = {
                "dimension": parts.get("DIMENSION", ""),
                "entity_scores": {},
                "winner": parts.get("WINNER", ""),
                "analysis": parts.get("ANALYSIS", ""),
            }
            for entity in entities:
                entry["entity_scores"][entity] = parts.get(entity.upper(), "")
            matrix.append(entry)

    return matrix


def parse_verified_claims(raw: Any) -> Tuple[List[Dict[str, Any]], float, List[str]]:
    """Parse fact-check results. Returns (verified_claims, reliability, issues)."""
    if not raw:
        return [], 0.5, []

    if isinstance(raw, dict):
        claims = raw.get("verified_claims", raw.get("claims", []))
        reliability = _safe_float(raw.get("overall_reliability", 0.5))
        issues = raw.get("issues", [])
        if isinstance(issues, str):
            issues = [i.strip() for i in issues.split("|") if i.strip()]
        return _normalize_claims_list(claims), reliability, issues

    if isinstance(raw, list):
        return _normalize_claims_list(raw), 0.5, []

    text = str(raw)

    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parse_verified_claims(parsed)
        if isinstance(parsed, list):
            return _normalize_claims_list(parsed), 0.5, []
    except (json.JSONDecodeError, ValueError):
        pass

    # Pipe-delimited
    verified = []
    for claim_str in text.split("||"):
        parts = {}
        for part in re.split(r"\|", claim_str):
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                parts[key.strip().lower()] = val.strip()
        if parts.get("claim"):
            verified.append(
                {
                    "claim": parts.get("claim", ""),
                    "status": parts.get("status", "unverified"),
                    "confidence": _safe_float(parts.get("confidence", 0.5)),
                    "note": parts.get("note", ""),
                }
            )

    return verified, 0.5, []


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_float(val: Any, default: float = 0.5) -> float:
    """Safely convert to float."""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        # Handle strings like "0.8" or "80%"
        s = str(val).strip().rstrip("%")
        f = float(s)
        if f > 1.0 and "%" not in str(val):
            return f
        if f > 1.0:
            return f / 100.0
        return f
    except (ValueError, TypeError):
        return default


def _try_json_parse(text: str, sub_topic: str) -> List[Dict[str, Any]]:
    """Try parsing as JSON array or object."""
    # Find JSON array in text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return _normalize_finding_list(parsed, sub_topic)
        except json.JSONDecodeError:
            pass

    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict):
                return _normalize_finding_list([parsed], sub_topic)
        except json.JSONDecodeError:
            pass

    return []


def _try_pipe_parse(text: str, sub_topic: str) -> List[Dict[str, Any]]:
    """Try parsing pipe-delimited format."""
    findings = []
    for finding_str in text.split("||"):
        parts = {}
        for part in finding_str.split("|"):
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                parts[key.strip().lower()] = val.strip()
        if parts.get("claim"):
            findings.append(
                {
                    "claim": parts["claim"],
                    "evidence": parts.get("evidence", ""),
                    "confidence": _safe_float(parts.get("confidence", 0.5)),
                    "sub_topic": sub_topic,
                }
            )
    return findings


def _try_numbered_parse(text: str, sub_topic: str) -> List[Dict[str, Any]]:
    """Try parsing numbered list format."""
    findings = []
    for match in re.finditer(r"(?:^|\n)\s*\d+[.)]\s*(.+?)(?=\n\s*\d+[.)]|\Z)", text, re.DOTALL):
        line = match.group(1).strip()
        if not line or len(line) < 10:
            continue

        # Try to extract claim/evidence/confidence from within the line
        parts = {}
        for part in re.split(r"[|;]", line):
            part = part.strip()
            if ":" in part:
                key, val = part.split(":", 1)
                k = key.strip().lower()
                if k in ("claim", "evidence", "confidence", "finding"):
                    parts[k] = val.strip()

        if parts.get("claim") or parts.get("finding"):
            findings.append(
                {
                    "claim": parts.get("claim", parts.get("finding", "")),
                    "evidence": parts.get("evidence", ""),
                    "confidence": _safe_float(parts.get("confidence", 0.6)),
                    "sub_topic": sub_topic,
                }
            )
        else:
            # Whole line is the claim
            findings.append(
                {
                    "claim": line[:300],
                    "evidence": "",
                    "confidence": 0.5,
                    "sub_topic": sub_topic,
                }
            )

    return findings


def _try_line_parse(text: str, sub_topic: str) -> List[Dict[str, Any]]:
    """Fallback: treat each line as a finding."""
    findings = []
    for line in text.split("\n"):
        line = line.strip()
        # Remove bullet markers
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line and len(line) > 15:
            findings.append(
                {
                    "claim": line[:300],
                    "evidence": "",
                    "confidence": 0.5,
                    "sub_topic": sub_topic,
                }
            )
    return findings


def _normalize_finding_list(items: List[Any], sub_topic: str) -> List[Dict[str, Any]]:
    """Normalize a list of findings into standard format."""
    findings = []
    for item in items:
        if isinstance(item, dict):
            claim = (
                item.get("claim")
                or item.get("finding")
                or item.get("title")
                or item.get("summary")
                or ""
            )
            if claim:
                findings.append(
                    {
                        "claim": str(claim),
                        "evidence": str(
                            item.get("evidence") or item.get("source") or item.get("details") or ""
                        ),
                        "confidence": _safe_float(
                            item.get("confidence") or item.get("score") or 0.6
                        ),
                        "sub_topic": sub_topic,
                    }
                )
        elif isinstance(item, str) and len(item) > 10:
            findings.append(
                {
                    "claim": item[:300],
                    "evidence": "",
                    "confidence": 0.5,
                    "sub_topic": sub_topic,
                }
            )
    return findings


def _normalize_comparison_list(items: List[Any], entities: List[str]) -> List[Dict[str, Any]]:
    """Normalize comparison items."""
    matrix = []
    for item in items:
        if isinstance(item, dict):
            dim = item.get("dimension") or item.get("name") or item.get("category") or ""
            if dim:
                entry = {
                    "dimension": str(dim),
                    "entity_scores": {},
                    "winner": str(item.get("winner", "")),
                    "analysis": str(item.get("analysis", item.get("notes", ""))),
                }
                # Look for entity scores
                scores = item.get("entity_scores", item.get("scores", {}))
                if isinstance(scores, dict):
                    entry["entity_scores"] = {str(k): str(v) for k, v in scores.items()}
                else:
                    for ent in entities:
                        entry["entity_scores"][ent] = str(item.get(ent, item.get(ent.lower(), "")))
                matrix.append(entry)
    return matrix


def _normalize_claims_list(items: List[Any]) -> List[Dict[str, Any]]:
    """Normalize claim verification list."""
    claims = []
    for item in items:
        if isinstance(item, dict):
            claim = item.get("claim", item.get("text", ""))
            if claim:
                claims.append(
                    {
                        "claim": str(claim),
                        "status": str(item.get("status", "unverified")),
                        "confidence": _safe_float(item.get("confidence", 0.5)),
                        "note": str(item.get("note", item.get("notes", ""))),
                    }
                )
        elif isinstance(item, str):
            claims.append(
                {
                    "claim": item,
                    "status": "unverified",
                    "confidence": 0.5,
                    "note": "",
                }
            )
    return claims
