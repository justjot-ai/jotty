"""
Meeting Insights Analyzer Skill - Analyze meeting transcripts for communication patterns.

Identifies conflict avoidance, filler words, speaking ratios, and provides
actionable feedback for communication improvement.
"""

import asyncio
import inspect
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

# Status emitter for progress updates
status = SkillStatus("meeting-insights-analyzer")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def analyze_meeting_insights_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze meeting transcripts for communication patterns and insights.

    Args:
        params:
            - transcript_files (list): List of transcript file paths
            - user_name (str, optional): Your name/identifier
            - analysis_types (list, optional): Types of analysis
            - output_file (str, optional): Path to save report

    Returns:
        Dictionary with insights, statistics, recommendations
    """
    status.set_callback(params.pop("_status_callback", None))

    transcript_files = params.get("transcript_files", [])
    user_name = params.get("user_name", "")
    analysis_types = params.get("analysis_types", ["all"])
    output_file = params.get("output_file", None)

    if not transcript_files:
        return {"success": False, "error": "transcript_files is required"}

    # Read transcripts
    transcripts = []
    for file_path in transcript_files:
        try:
            path = Path(file_path)
            if path.exists():
                content = path.read_text(encoding="utf-8")
                transcripts.append({"file": str(path), "content": content, "name": path.stem})
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

    if not transcripts:
        return {"success": False, "error": "No valid transcript files found"}

    # Perform analysis
    insights = {}
    statistics = {}

    if "all" in analysis_types or "conflict_avoidance" in analysis_types:
        insights["conflict_avoidance"] = await _analyze_conflict_avoidance(transcripts, user_name)

    if "all" in analysis_types or "speaking_ratios" in analysis_types:
        stats = await _analyze_speaking_ratios(transcripts, user_name)
        statistics.update(stats)

    if "all" in analysis_types or "filler_words" in analysis_types:
        insights["filler_words"] = await _analyze_filler_words(transcripts, user_name)

    if "all" in analysis_types or "active_listening" in analysis_types:
        insights["active_listening"] = await _analyze_active_listening(transcripts, user_name)

    if "all" in analysis_types or "leadership" in analysis_types:
        insights["leadership"] = await _analyze_leadership(transcripts, user_name)

    # Generate recommendations
    recommendations = await _generate_recommendations(insights, statistics)

    # Generate report
    report_content = _generate_report(insights, statistics, recommendations, transcripts)

    # Save report if requested
    if output_file:
        Path(output_file).write_text(report_content, encoding="utf-8")

    return {
        "success": True,
        "insights": insights,
        "statistics": statistics,
        "recommendations": recommendations,
        "output_file": output_file,
        "report": report_content,
    }


async def _analyze_conflict_avoidance(transcripts: List[Dict], user_name: str) -> Dict:
    """Analyze conflict avoidance patterns."""

    # Pattern matching for hedging language
    hedging_patterns = [
        r"\b(maybe|perhaps|kind of|sort of|I think|I guess|possibly|probably)\b",
        r"\b(yeah,?\s+but|well,?\s+but|I mean)\b",
        r"\b(whatever you think|up to you|if you want|if that\'s okay)\b",
    ]

    instances = []
    for transcript in transcripts:
        content = transcript["content"]
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if user_name.lower() in line.lower():
                for pattern in hedging_patterns:
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        instances.append(
                            {
                                "meeting": transcript["name"],
                                "line": i + 1,
                                "text": line.strip(),
                                "pattern": pattern,
                            }
                        )

    return {
        "total_instances": len(instances),
        "instances": instances[:10],  # Top 10
        "pattern": "Hedging language and indirect communication detected",
    }


async def _analyze_speaking_ratios(transcripts: List[Dict], user_name: str) -> Dict:
    """Analyze speaking ratios and turn-taking."""

    total_words = 0
    user_words = 0
    total_turns = 0
    user_turns = 0

    for transcript in transcripts:
        content = transcript["content"]
        lines = content.split("\n")

        for line in lines:
            if line.strip():
                total_turns += 1
                words = len(line.split())
                total_words += words

                if user_name.lower() in line.lower():
                    user_turns += 1
                    user_words += words

    speaking_ratio = (user_words / total_words * 100) if total_words > 0 else 0
    turn_ratio = (user_turns / total_turns * 100) if total_turns > 0 else 0

    return {
        "speaking_ratio": round(speaking_ratio, 1),
        "turn_ratio": round(turn_ratio, 1),
        "total_words": total_words,
        "user_words": user_words,
        "total_turns": total_turns,
        "user_turns": user_turns,
    }


async def _analyze_filler_words(transcripts: List[Dict], user_name: str) -> Dict:
    """Analyze filler word usage."""

    filler_words = ["um", "uh", "like", "you know", "actually", "basically", "literally"]

    instances = {}
    total_fillers = 0

    for transcript in transcripts:
        content = transcript["content"]
        lines = content.split("\n")

        for line in lines:
            if user_name.lower() in line.lower():
                line_lower = line.lower()
                for filler in filler_words:
                    count = len(re.findall(rf"\b{filler}\b", line_lower))
                    if count > 0:
                        instances[filler] = instances.get(filler, 0) + count
                        total_fillers += count

    return {
        "total_fillers": total_fillers,
        "by_word": instances,
        "pattern": "Filler word usage detected",
    }


async def _analyze_active_listening(transcripts: List[Dict], user_name: str) -> Dict:
    """Analyze active listening indicators."""

    listening_indicators = [
        r"\b(what do you think|how do you feel|can you explain|tell me more)\b",
        r"\b(so you\'re saying|if I understand|let me make sure)\b",
        r"\b(building on|expanding on|following up on)\b",
    ]

    instances = []
    for transcript in transcripts:
        content = transcript["content"]
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if user_name.lower() in line.lower():
                for pattern in listening_indicators:
                    if re.search(pattern, line, re.IGNORECASE):
                        instances.append(
                            {"meeting": transcript["name"], "line": i + 1, "text": line.strip()}
                        )

    return {
        "total_instances": len(instances),
        "instances": instances[:10],
        "pattern": "Active listening indicators",
    }


async def _analyze_leadership(transcripts: List[Dict], user_name: str) -> Dict:
    """Analyze leadership and facilitation style."""

    # Use AI for deeper analysis
    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill("claude-cli-llm")

        if claude_skill:
            generate_tool = claude_skill.tools.get("generate_text_tool")

            if generate_tool:
                # Prepare transcript summary
                transcript_summary = "\n\n".join(
                    [
                        f"Meeting: {t['name']}\n{t['content'][:1000]}"
                        for t in transcripts[:3]  # Limit for prompt size
                    ]
                )

                prompt = f"""Analyze the leadership and facilitation style in these meeting transcripts.

**User:** {user_name}

**Transcripts:**
{transcript_summary}

Provide analysis on:
1. Decision-making approach (directive vs collaborative)
2. How disagreements are handled
3. Inclusion of quieter participants
4. Time management and agenda control
5. Follow-up clarity

Return JSON format with specific examples."""

                if inspect.iscoroutinefunction(generate_tool):
                    result = await generate_tool(
                        {"prompt": prompt, "model": "sonnet", "timeout": 90}
                    )
                else:
                    result = generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 90})

                if result.get("success"):
                    return {
                        "analysis": result.get("text", ""),
                        "pattern": "Leadership style analysis",
                    }
    except Exception as e:
        logger.debug(f"AI leadership analysis failed: {e}")

    return {"analysis": "Analysis pending - requires AI processing", "pattern": "Leadership style"}


async def _generate_recommendations(insights: Dict, statistics: Dict) -> List[str]:
    """Generate actionable recommendations."""

    recommendations = []

    # Conflict avoidance recommendations
    if insights.get("conflict_avoidance", {}).get("total_instances", 0) > 5:
        recommendations.append(
            "Practice direct communication: Name issues directly in the first sentence, "
            "remove hedging words like 'maybe' and 'kind of'"
        )

    # Speaking ratio recommendations
    speaking_ratio = statistics.get("speaking_ratio", 0)
    if speaking_ratio > 60:
        recommendations.append(
            f"Your speaking ratio is {speaking_ratio}% - aim for 40-50% to encourage "
            "more team participation"
        )
    elif speaking_ratio < 20:
        recommendations.append(
            f"Your speaking ratio is {speaking_ratio}% - consider speaking up more "
            "to share your expertise"
        )

    # Filler words recommendations
    total_fillers = insights.get("filler_words", {}).get("total_fillers", 0)
    if total_fillers > 10:
        recommendations.append(
            f"Reduce filler words (found {total_fillers} instances) - practice pausing "
            "instead of using 'um' or 'uh'"
        )

    return recommendations


def _generate_report(
    insights: Dict, statistics: Dict, recommendations: List[str], transcripts: List[Dict]
) -> str:
    """Generate markdown report."""

    lines = [
        "# Meeting Insights Analysis",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Meetings Analyzed:** {len(transcripts)}",
        "",
        "## Summary Statistics",
        "",
        f"- **Speaking Ratio:** {statistics.get('speaking_ratio', 0)}%",
        f"- **Turn Ratio:** {statistics.get('turn_ratio', 0)}%",
        f"- **Total Words:** {statistics.get('total_words', 0)}",
        "",
        "## Insights",
        "",
    ]

    # Add insights sections
    for insight_type, insight_data in insights.items():
        lines.append(f"### {insight_type.replace('_', ' ').title()}")
        lines.append("")
        if isinstance(insight_data, dict):
            lines.append(f"**Pattern:** {insight_data.get('pattern', 'N/A')}")
            lines.append(f"**Instances:** {insight_data.get('total_instances', 0)}")
            if insight_data.get("instances"):
                lines.append("")
                lines.append("**Examples:**")
                for instance in insight_data["instances"][:3]:
                    lines.append(f"- {instance.get('text', '')[:100]}")
        lines.append("")

    # Recommendations
    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    return "\n".join(lines)
