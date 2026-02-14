"""Data Anonymizer Skill — mask PII in text."""
import re
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("data-anonymizer")

PATTERNS = {
    "email": (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
    "phone": (r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", "[PHONE]"),
    "ip": (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]"),
    "credit_card": (r"\b(?:\d{4}[- ]?){3}\d{4}\b", "[CREDIT_CARD]"),
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
}


@tool_wrapper(required_params=["text"])
def anonymize_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize PII in text by replacing with placeholders."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    detections = {}

    mask_map = {
        "email": params.get("mask_emails", True),
        "phone": params.get("mask_phones", True),
        "ip": params.get("mask_ips", True),
        "credit_card": params.get("mask_credit_cards", True),
        "ssn": params.get("mask_ssn", True),
    }

    for pii_type, (pattern, replacement) in PATTERNS.items():
        if not mask_map.get(pii_type, True):
            continue
        matches = re.findall(pattern, text)
        if matches:
            detections[pii_type] = len(matches)
            text = re.sub(pattern, replacement, text)

    return tool_response(anonymized=text, detections=detections,
                         total_redactions=sum(detections.values()))


__all__ = ["anonymize_text_tool"]
