"""
Competitive Ads Extractor Skill - Extract and analyze competitor ads.

Helps understand competitor advertising strategies by extracting ads from
ad libraries and analyzing messaging, creative approaches, and targeting.
"""

import asyncio
import inspect
import logging
import os
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
status = SkillStatus("competitive-ads-extractor")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def extract_competitive_ads_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and analyze competitor ads from ad libraries.

    Args:
        params:
            - competitor_name (str): Name of competitor
            - platforms (list, optional): Platforms to search
            - max_ads (int, optional): Maximum ads to extract
            - analysis_depth (str, optional): Analysis depth
            - output_file (str, optional): Path to save report

    Returns:
        Dictionary with ads, insights, analysis
    """
    status.set_callback(params.pop("_status_callback", None))

    competitor_name = params.get("competitor_name", "")
    platforms = params.get("platforms", ["facebook", "google"])
    max_ads = params.get("max_ads", 20)
    analysis_depth = params.get("analysis_depth", "detailed")
    output_file = params.get("output_file", None)

    if not competitor_name:
        return {"success": False, "error": "competitor_name is required"}

    # Search for ads
    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()
        web_search_skill = registry.get_skill("web-search")

        if not web_search_skill:
            return {"success": False, "error": "web-search skill not available"}

        search_tool = web_search_skill.tools.get("search_web_tool")
        if not search_tool:
            return {"success": False, "error": "search_web_tool not found"}

        # Search for ad libraries
        ads = []
        for platform in platforms:
            query = f"{competitor_name} ads {platform} ad library"

            if inspect.iscoroutinefunction(search_tool):
                result = await search_tool({"query": query, "max_results": 10})
            else:
                result = search_tool({"query": query, "max_results": 10})

            if result.get("success"):
                for res in result.get("results", [])[:5]:
                    ads.append(
                        {
                            "platform": platform,
                            "title": res.get("title", ""),
                            "url": res.get("url", ""),
                            "snippet": res.get("snippet", ""),
                        }
                    )

        # Analyze ads using AI
        insights = await _analyze_ads(ads, competitor_name, analysis_depth)

        # Generate report
        report_content = _generate_ads_report(ads, insights, competitor_name)

        # Save report if requested
        if output_file:
            Path(output_file).write_text(report_content, encoding="utf-8")

        return {
            "success": True,
            "ads": ads[:max_ads],
            "insights": insights,
            "output_file": output_file,
            "report": report_content,
        }

    except Exception as e:
        logger.error(f"Ad extraction failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def _analyze_ads(ads: List[Dict], competitor_name: str, depth: str) -> Dict:
    """Analyze ads for messaging, creative, and targeting insights."""

    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill("claude-cli-llm")

        if not claude_skill:
            return {}

        generate_tool = claude_skill.tools.get("generate_text_tool")
        if not generate_tool:
            return {}

        # Prepare ads summary
        ads_text = "\n".join(
            [
                f"{i+1}. [{ad['platform']}] {ad['title']}\n   {ad['snippet'][:200]}"
                for i, ad in enumerate(ads[:15])  # Limit for prompt size
            ]
        )

        prompt = f"""Analyze these competitor ads for {competitor_name} and provide insights:

**Ads Found:**
{ads_text}

Provide analysis on:
1. **Messaging Themes**: What messages and value propositions are they using?
2. **Creative Approaches**: What types of visuals, copy styles, formats?
3. **Targeting Signals**: Who are they targeting based on ad content?
4. **Call-to-Actions**: What CTAs are they using?
5. **Trends**: Any patterns or trends across ads?

Return JSON format with structured insights."""

        if inspect.iscoroutinefunction(generate_tool):
            result = await generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 120})
        else:
            result = generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 120})

        if result.get("success"):
            import json
            import re

            text = result.get("text", "")
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"analysis": text}
    except Exception as e:
        logger.debug(f"AI ad analysis failed: {e}")

    return {
        "messaging_themes": "Analysis pending",
        "creative_approaches": "Analysis pending",
        "targeting_signals": "Analysis pending",
    }


def _generate_ads_report(ads: List[Dict], insights: Dict, competitor_name: str) -> str:
    """Generate markdown report."""

    lines = [
        f"# Competitive Ads Analysis: {competitor_name}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Ads Found:** {len(ads)}",
        "",
        "## Summary",
        "",
    ]

    # Add insights
    if insights:
        for key, value in insights.items():
            lines.append(f"### {key.replace('_', ' ').title()}")
            lines.append("")
            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"- **{k}**: {v}")
            elif isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(str(value))
            lines.append("")

    # Add ads list
    lines.append("## Ads Found")
    lines.append("")
    for i, ad in enumerate(ads, 1):
        lines.append(f"### Ad {i}: {ad.get('platform', 'Unknown')}")
        lines.append(f"**Title:** {ad.get('title', 'N/A')}")
        lines.append(f"**URL:** {ad.get('url', 'N/A')}")
        lines.append(f"**Snippet:** {ad.get('snippet', 'N/A')}")
        lines.append("")

    return "\n".join(lines)
