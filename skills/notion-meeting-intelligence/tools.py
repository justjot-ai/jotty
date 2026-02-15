"""
Notion Meeting Intelligence Skill - Prepare meeting materials from Notion.

Gathers context from Notion, enriches with research, and creates
comprehensive pre-reads and agendas for meetings.
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
status = SkillStatus("notion-meeting-intelligence")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def prepare_meeting_materials_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare meeting materials from Notion context.

    Args:
        params:
            - meeting_topic (str): Meeting topic
            - meeting_type (str, optional): Type of meeting
            - attendees (list, optional): Attendee names
            - related_project (str, optional): Related project
            - search_queries (list, optional): Notion search queries
            - create_pre_read (bool, optional): Create pre-read
            - create_agenda (bool, optional): Create agenda

    Returns:
        Dictionary with created pages and context
    """
    status.set_callback(params.pop("_status_callback", None))

    meeting_topic = params.get("meeting_topic", "")
    meeting_type = params.get("meeting_type", "status_update")
    attendees = params.get("attendees", [])
    related_project = params.get("related_project", None)
    search_queries = params.get("search_queries", [])
    create_pre_read = params.get("create_pre_read", True)
    create_agenda = params.get("create_agenda", True)

    if not meeting_topic:
        return {"success": False, "error": "meeting_topic is required"}

    try:
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        registry = get_skills_registry()
        registry.init()
        notion_skill = registry.get_skill("notion")

        if not notion_skill:
            return {"success": False, "error": "notion skill not available"}

        context_found = {}

        # Search Notion for context
        if search_queries or related_project:
            search_tool = notion_skill.tools.get("search_tool") or notion_skill.tools.get(
                "search_pages_tool"
            )

            if search_tool:
                queries = search_queries or [meeting_topic]
                if related_project:
                    queries.append(related_project)

                for query in queries:
                    if inspect.iscoroutinefunction(search_tool):
                        result = await search_tool({"query": query})
                    else:
                        result = search_tool({"query": query})

                    if result.get("success"):
                        context_found[query] = result.get("results", [])

        # Generate pre-read if requested
        pre_read_page_id = None
        if create_pre_read:
            pre_read_content = _generate_pre_read(
                meeting_topic, meeting_type, context_found, attendees
            )

            create_tool = notion_skill.tools.get("create_page_tool") or notion_skill.tools.get(
                "create_pages_tool"
            )
            if create_tool:
                create_params = {
                    "title": f"{meeting_topic} - Pre-Read (Internal)",
                    "content": pre_read_content,
                }

                if inspect.iscoroutinefunction(create_tool):
                    result = await create_tool(create_params)
                else:
                    result = create_tool(create_params)

                if result.get("success"):
                    pre_read_page_id = result.get("page_id", "")

        # Generate agenda if requested
        agenda_page_id = None
        if create_agenda:
            agenda_content = _generate_agenda(meeting_topic, meeting_type, attendees)

            create_tool = notion_skill.tools.get("create_page_tool") or notion_skill.tools.get(
                "create_pages_tool"
            )
            if create_tool:
                create_params = {"title": f"{meeting_topic} - Agenda", "content": agenda_content}

                if inspect.iscoroutinefunction(create_tool):
                    result = await create_tool(create_params)
                else:
                    result = create_tool(create_params)

                if result.get("success"):
                    agenda_page_id = result.get("page_id", "")

        return {
            "success": True,
            "pre_read_page_id": pre_read_page_id,
            "agenda_page_id": agenda_page_id,
            "context_found": context_found,
        }

    except Exception as e:
        logger.error(f"Meeting preparation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _generate_pre_read(topic: str, meeting_type: str, context: Dict, attendees: List[str]) -> str:
    """Generate pre-read content."""

    content = f"""# {topic} - Pre-Read

**Meeting Type:** {meeting_type.replace('_', ' ').title()}
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Attendees:** {', '.join(attendees) if attendees else 'TBD'}

## Background Context

{_format_context(context)}

## Key Topics

- [Topic 1]
- [Topic 2]
- [Topic 3]

## Preparation

- Review related documents
- Come prepared with questions
- Think about desired outcomes

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    return content


def _generate_agenda(topic: str, meeting_type: str, attendees: List[str]) -> str:
    """Generate agenda content."""

    content = f"""# {topic} - Agenda

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Attendees:** {', '.join(attendees) if attendees else 'TBD'}

## Agenda

1. **Opening & Introductions** (5 min)
   - Review agenda
   - Set objectives

2. **Main Discussion** (30 min)
   - [Topic 1]
   - [Topic 2]
   - [Topic 3]

3. **Action Items & Next Steps** (10 min)
   - Assign owners
   - Set deadlines

4. **Closing** (5 min)
   - Summary
   - Next meeting

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    return content


def _format_context(context: Dict) -> str:
    """Format context from Notion search results."""

    if not context:
        return "No additional context found in Notion."

    formatted = ""
    for query, results in context.items():
        if results:
            formatted += f"### Related to: {query}\n\n"
            for result in results[:3]:  # Top 3 results
                title = result.get("title", "Untitled")
                formatted += f"- {title}\n"
            formatted += "\n"

    return formatted or "No additional context found."
