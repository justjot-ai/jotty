"""
Notion Knowledge Capture Skill - Transform conversations into Notion documentation.

Captures insights, decisions, and knowledge from chat context and saves
to Notion with proper organization and linking.
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
status = SkillStatus("notion-knowledge-capture")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def capture_knowledge_to_notion_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture knowledge from conversation to Notion.

    Args:
        params:
            - content (str): Content to capture
            - content_type (str, optional): Type of content
            - title (str, optional): Page title
            - parent_page_id (str, optional): Parent page ID
            - database_id (str, optional): Database ID
            - tags (list, optional): Tags
            - link_to_pages (list, optional): Page IDs to link

    Returns:
        Dictionary with created page details
    """
    status.set_callback(params.pop("_status_callback", None))

    content = params.get("content", "")
    content_type = params.get("content_type", "concept")
    title = params.get("title", None)
    parent_page_id = params.get("parent_page_id", None)
    database_id = params.get("database_id", None)
    tags = params.get("tags", [])
    link_to_pages = params.get("link_to_pages", [])

    if not content:
        return {"success": False, "error": "content is required"}

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

        # Format content based on type
        formatted_content = _format_content(content, content_type, link_to_pages)

        # Generate title if not provided
        if not title:
            title = _generate_title(content, content_type)

        # Use Notion skill to create page
        create_tool = notion_skill.tools.get("create_page_tool") or notion_skill.tools.get(
            "create_pages_tool"
        )

        if not create_tool:
            return {"success": False, "error": "Notion create page tool not found"}

        create_params = {"title": title, "content": formatted_content}

        if parent_page_id:
            create_params["parent_page_id"] = parent_page_id

        if database_id:
            create_params["database_id"] = database_id

        if tags:
            create_params["tags"] = tags

        if inspect.iscoroutinefunction(create_tool):
            result = await create_tool(create_params)
        else:
            result = create_tool(create_params)

        if result.get("success"):
            return {
                "success": True,
                "page_id": result.get("page_id", ""),
                "page_url": result.get("url", ""),
                "title": title,
            }

        return {"success": False, "error": result.get("error", "Failed to create Notion page")}

    except Exception as e:
        logger.error(f"Knowledge capture failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _format_content(content: str, content_type: str, link_to_pages: List[str]) -> str:
    """Format content based on type."""

    formatted = content

    # Add type-specific formatting
    if content_type == "faq":
        formatted = (
            f"# FAQ Entry\n\n{content}\n\n---\n*Captured: {datetime.now().strftime('%Y-%m-%d')}*"
        )

    elif content_type == "how_to":
        formatted = f"# How-To Guide\n\n{content}\n\n---\n*Last updated: {datetime.now().strftime('%Y-%m-%d')}*"

    elif content_type == "decision":
        formatted = (
            f"# Decision Record\n\n{content}\n\n---\n*Date: {datetime.now().strftime('%Y-%m-%d')}*"
        )

    elif content_type == "meeting_summary":
        formatted = (
            f"# Meeting Summary\n\n{content}\n\n---\n*Date: {datetime.now().strftime('%Y-%m-%d')}*"
        )

    # Add links if provided
    if link_to_pages:
        formatted += "\n\n## Related Pages\n"
        for page_id in link_to_pages:
            formatted += f"- [[{page_id}]]\n"

    return formatted


def _generate_title(content: str, content_type: str) -> str:
    """Generate title from content."""

    # Extract first line or first sentence
    lines = content.split("\n")
    first_line = lines[0].strip()

    # Clean up title
    title = first_line[:100]  # Limit length

    # Remove markdown formatting
    title = title.replace("#", "").strip()
    title = title.replace("**", "").strip()

    # Add type prefix if needed
    if content_type == "faq":
        if not title.lower().startswith("q:"):
            title = f"FAQ: {title}"
    elif content_type == "decision":
        if "decision" not in title.lower():
            title = f"Decision: {title}"

    return (
        title or f"{content_type.replace('_', ' ').title()} - {datetime.now().strftime('%Y-%m-%d')}"
    )
