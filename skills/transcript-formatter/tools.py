"""Transcript Formatter Skill - clean raw transcripts."""
import re
from typing import Dict, Any, List, Optional, Tuple
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("transcript-formatter")

TS_PATTERNS = [
    re.compile(r"^(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*[-:]?\s*(.*)"),
    re.compile(r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.*)"),
    re.compile(r"^(\d{1,2}:\d{2})\s+(.*)"),
]

SPEAKER_PATTERN = re.compile(r"^([A-Za-z][A-Za-z0-9 _.'-]{0,40}):\s+(.*)")


def _parse_line(line: str) -> Tuple[Optional[str], Optional[str], str]:
    timestamp = None
    speaker = None
    text = line.strip()
    if not text:
        return None, None, ""
    for pat in TS_PATTERNS:
        m = pat.match(text)
        if m:
            timestamp = m.group(1)
            text = m.group(2).strip()
            break
    m = SPEAKER_PATTERN.match(text)
    if m:
        speaker = m.group(1).strip()
        text = m.group(2).strip()
    return timestamp, speaker, text


@tool_wrapper(required_params=["text"])
def format_transcript_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Format a raw transcript with speaker labels and timestamps."""
    status.set_callback(params.pop("_status_callback", None))
    raw = params["text"]
    merge = params.get("merge_speakers", True)
    show_ts = params.get("include_timestamps", True)

    lines = raw.strip().split("\n")
    entries = []
    for line in lines:
        ts, speaker, text = _parse_line(line)
        if text:
            entries.append({"timestamp": ts, "speaker": speaker, "text": text})

    if not entries:
        return tool_error("No transcript content found")

    if merge:
        merged = []
        for entry in entries:
            if (merged and entry["speaker"] and
                    merged[-1]["speaker"] == entry["speaker"]):
                merged[-1]["text"] += " " + entry["text"]
                if not merged[-1]["timestamp"] and entry["timestamp"]:
                    merged[-1]["timestamp"] = entry["timestamp"]
            else:
                merged.append(dict(entry))
        entries = merged

    speakers = sorted(set(e["speaker"] for e in entries if e["speaker"]))

    output_lines = []
    for e in entries:
        parts = []
        if show_ts and e["timestamp"]:
            parts.append(f"[{e['timestamp']}]")
        if e["speaker"]:
            parts.append(f"{e['speaker']}:")
        parts.append(e["text"])
        output_lines.append(" ".join(parts))

    first_ts = next((e["timestamp"] for e in entries if e["timestamp"]), None)
    last_ts = next((e["timestamp"] for e in reversed(entries) if e["timestamp"]), None)

    return tool_response(
        formatted="\n\n".join(output_lines),
        speakers=speakers, speaker_count=len(speakers),
        line_count=len(entries),
        first_timestamp=first_ts, last_timestamp=last_ts,
    )


__all__ = ["format_transcript_tool"]
