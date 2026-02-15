"""
Content Research Writer Skill - Assist in writing high-quality content.

Provides research, citations, hook improvement, and section-by-section feedback
while maintaining the writer's unique voice.
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
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

# Status emitter for progress updates
status = SkillStatus("content-research-writer")


logger = logging.getLogger(__name__)


@tool_wrapper()
async def write_content_with_research_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assist in writing content with research and feedback.

    Args:
        params:
            - topic (str): Topic or title
            - content_type (str, optional): Type of content
            - target_audience (str, optional): Audience description
            - target_length (str, optional): Length target
            - writing_style (str, optional): Writing style
            - draft_content (str, optional): Existing draft
            - action (str, optional): Action to perform
            - section_to_review (str, optional): Section to review
            - research_topics (list, optional): Topics to research

    Returns:
        Dictionary with outline, research, feedback, etc.
    """
    status.set_callback(params.pop("_status_callback", None))

    topic = params.get("topic", "")
    content_type = params.get("content_type", "article")
    target_audience = params.get("target_audience", "general audience")
    target_length = params.get("target_length", "medium")
    writing_style = params.get("writing_style", "conversational")
    draft_content = params.get("draft_content", "")
    action = params.get("action", "outline")
    section_to_review = params.get("section_to_review", "")
    research_topics = params.get("research_topics", [])

    if not topic:
        return {"success": False, "error": "topic is required"}

    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill("claude-cli-llm")

        if not claude_skill:
            return {"success": False, "error": "claude-cli-llm skill not available"}

        generate_tool = claude_skill.tools.get("generate_text_tool")
        if not generate_tool:
            return {"success": False, "error": "generate_text_tool not found"}

        result = {}

        if action == "outline":
            result = await _create_outline(
                generate_tool, topic, content_type, target_audience, target_length, writing_style
            )

        elif action == "research":
            result = await _conduct_research(generate_tool, topic, research_topics)

        elif action == "improve_hook":
            result = await _improve_hook(generate_tool, topic, draft_content, writing_style)

        elif action == "review_section":
            result = await _review_section(
                generate_tool, topic, section_to_review or draft_content, writing_style
            )

        elif action == "full_review":
            result = await _full_review(generate_tool, topic, draft_content, writing_style)

        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}. Valid actions: outline, research, improve_hook, review_section, full_review",
            }

        # Check if helper returned an error
        if result.get("error"):
            return {"success": False, "error": result.get("error")}

        return {"success": True, **result}

    except Exception as e:
        logger.error(f"Content writing failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def _create_outline(
    generate_tool,
    topic: str,
    content_type: str,
    target_audience: str,
    target_length: str,
    writing_style: str,
) -> Dict:
    """Create content outline."""

    prompt = f"""Create a detailed outline for a {content_type} about "{topic}".

**Target Audience:** {target_audience}
**Target Length:** {target_length}
**Writing Style:** {writing_style}

Create a comprehensive outline with:
1. Hook/Introduction ideas
2. Main sections with key points
3. Supporting evidence needed
4. Research gaps to fill
5. Conclusion approach

Format as markdown with clear structure."""

    if inspect.iscoroutinefunction(generate_tool):
        result = await generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 120})
    else:
        result = generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 120})

    if result.get("success"):
        return {"outline": result.get("text", "")}

    return {"error": result.get("error", "Outline generation failed")}


async def _conduct_research(generate_tool, topic: str, research_topics: List[str]) -> Dict:
    """Conduct research on topics."""

    # Use web search if available
    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()
        web_search_skill = registry.get_skill("web-search")

        research_findings = {}
        citations = []

        if web_search_skill:
            search_tool = web_search_skill.tools.get("search_web_tool")

            for research_topic in research_topics or [topic]:
                query = f"{topic} {research_topic}"

                if inspect.iscoroutinefunction(search_tool):
                    search_result = await search_tool({"query": query, "max_results": 5})
                else:
                    search_result = search_tool({"query": query, "max_results": 5})

                if search_result.get("success"):
                    results = search_result.get("results", [])
                    research_findings[research_topic] = results

                    for res in results:
                        citations.append(
                            {
                                "title": res.get("title", ""),
                                "url": res.get("url", ""),
                                "snippet": res.get("snippet", ""),
                            }
                        )

        # Use AI to synthesize research
        research_summary = "\n".join(
            [
                f"**{topic}:**\n"
                + "\n".join(
                    [f"- {r.get('title', '')}: {r.get('snippet', '')[:200]}" for r in findings[:3]]
                )
                for topic, findings in research_findings.items()
            ]
        )

        prompt = f"""Synthesize this research into key findings with citations:

{research_summary}

Provide:
1. Key findings (numbered)
2. Supporting evidence
3. Proper citations in format: [1] Title (URL)
4. Expert quotes if available"""

        if inspect.iscoroutinefunction(generate_tool):
            result = await generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 120})
        else:
            result = generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 120})

        if result.get("success"):
            return {
                "research": {"findings": result.get("text", ""), "sources": research_findings},
                "citations": citations,
            }

    except Exception as e:
        logger.debug(f"Research failed: {e}")

    return {"research": {"findings": "Research pending"}, "citations": []}


async def _improve_hook(generate_tool, topic: str, draft_content: str, writing_style: str) -> Dict:
    """Improve content hook/introduction."""

    prompt = f"""Analyze and improve the hook/introduction for this content:

**Topic:** {topic}
**Writing Style:** {writing_style}

**Current Hook:**
{draft_content[:500]}

Provide:
1. Analysis of what works and what could be stronger
2. 3 alternative hook options with explanations
3. Why each option works
4. Recommendation for best option"""

    if inspect.iscoroutinefunction(generate_tool):
        result = await generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 90})
    else:
        result = generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 90})

    if result.get("success"):
        return {
            "improved_content": result.get("text", ""),
            "feedback": {"type": "hook_improvement", "analysis": result.get("text", "")},
        }

    return {"error": result.get("error", "Hook improvement failed")}


async def _review_section(
    generate_tool, topic: str, section_content: str, writing_style: str
) -> Dict:
    """Review a specific section."""

    prompt = f"""Review this section of content about "{topic}":

**Writing Style:** {writing_style}

**Section Content:**
{section_content}

Provide detailed feedback:
1. What works well (strengths)
2. Suggestions for improvement (clarity, flow, evidence)
3. Specific line edits with explanations
4. Questions to consider"""

    if inspect.iscoroutinefunction(generate_tool):
        result = await generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 120})
    else:
        result = generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 120})

    if result.get("success"):
        return {"feedback": {"type": "section_review", "analysis": result.get("text", "")}}

    return {"error": result.get("error", "Section review failed")}


async def _full_review(generate_tool, topic: str, draft_content: str, writing_style: str) -> Dict:
    """Full draft review."""

    prompt = f"""Provide comprehensive review of this complete draft about "{topic}":

**Writing Style:** {writing_style}

**Full Draft:**
{draft_content[:8000]}

Provide:
1. Overall assessment (strengths and impact)
2. Structure & flow analysis
3. Content quality evaluation
4. Technical quality (grammar, consistency)
5. Readability assessment
6. Final polish suggestions
7. Pre-publish checklist"""

    if inspect.iscoroutinefunction(generate_tool):
        result = await generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 180})
    else:
        result = generate_tool({"prompt": prompt, "model": "sonnet", "timeout": 180})

    if result.get("success"):
        return {"feedback": {"type": "full_review", "analysis": result.get("text", "")}}

    return {"error": result.get("error", "Full review failed")}
