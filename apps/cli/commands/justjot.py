"""
JustJot Command
===============

/J command for creating ideas on JustJot.ai
"""

import os
import logging
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_EMAIL = "setia.naveen@gmail.com"
JUSTJOT_API_URL = os.environ.get("JUSTJOT_API_URL", "https://api.justjot.ai")


class JustJotCommand(BaseCommand):
    """
    /J - Create idea on JustJot.ai

    Quick command to submit ideas to JustJot.ai platform.
    Uses the last generated output or specified text as the idea content.
    """

    name = "J"
    aliases = ["justjot", "jj", "idea"]
    description = "Create idea on JustJot.ai"
    usage = "/J [idea text] or /J (uses last output)"
    category = "integrations"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute JustJot idea creation."""

        # Get idea content
        if args.raw.strip():
            # User provided idea text
            idea_content = args.raw.strip()
            idea_title = idea_content[:100] + "..." if len(idea_content) > 100 else idea_content
        else:
            # Use last output from conversation
            if hasattr(cli, '_output_history') and cli._output_history:
                idea_content = cli._output_history[-1]
                # Extract title from first line or heading
                lines = idea_content.strip().split('\n')
                first_line = lines[0].strip().lstrip('#').strip()
                idea_title = first_line[:100] if first_line else "Idea from Jotty"
            else:
                cli.renderer.warning("No content to submit. Provide idea text or run a query first.")
                return CommandResult.fail("No content available")

        # Get email from config or default
        email = args.flags.get('email', DEFAULT_EMAIL)

        # Show preview
        cli.renderer.panel(
            f"**Title:** {idea_title}\n\n"
            f"**Content preview:** {idea_content[:200]}{'...' if len(idea_content) > 200 else ''}\n\n"
            f"**Email:** {email}",
            title="JustJot.ai Idea",
            style="cyan"
        )

        # Submit to JustJot
        try:
            result = await self._submit_to_justjot(
                title=idea_title,
                content=idea_content,
                email=email,
                cli=cli
            )

            if result.get('success'):
                cli.renderer.success(f"Idea submitted to JustJot.ai!")
                if result.get('url'):
                    cli.renderer.print(f"  View: [cyan]{result['url']}[/cyan]")
                return CommandResult.ok(output="Idea created successfully")
            else:
                cli.renderer.error(f"Failed: {result.get('error', 'Unknown error')}")
                return CommandResult.fail(result.get('error', 'Submission failed'))

        except Exception as e:
            logger.error(f"JustJot submission error: {e}")
            cli.renderer.error(f"Error: {e}")
            return CommandResult.fail(str(e))

    async def _submit_to_justjot(
        self,
        title: str,
        content: str,
        email: str,
        cli: "JottyCLI"
    ) -> dict:
        """
        Submit idea to JustJot.ai.

        Tries multiple methods:
        1. JustJot MCP server (if configured)
        2. JustJot HTTP API
        3. JustJot skill (if available)
        """

        # Method 1: Try MCP server
        mcp_result = await self._try_mcp_submission(title, content, email, cli)
        if mcp_result:
            return mcp_result

        # Method 2: Try HTTP API
        api_result = await self._try_api_submission(title, content, email)
        if api_result:
            return api_result

        # Method 3: Try skill
        skill_result = await self._try_skill_submission(title, content, email, cli)
        if skill_result:
            return skill_result

        return {
            'success': False,
            'error': 'JustJot not configured. Set JUSTJOT_API_URL or configure MCP server.'
        }

    async def _try_mcp_submission(
        self,
        title: str,
        content: str,
        email: str,
        cli: "JottyCLI"
    ) -> dict:
        """Try submitting via MCP server."""
        try:
            # Check if MCP client is available
            if hasattr(cli, 'mcp_client') and cli.mcp_client:
                # Call JustJot MCP tool
                result = await cli.mcp_client.call_tool(
                    'justjot',
                    'create_idea',
                    {
                        'title': title,
                        'content': content,
                        'email': email
                    }
                )
                return result
        except Exception as e:
            logger.debug(f"MCP submission not available: {e}")
        return None

    async def _try_api_submission(
        self,
        title: str,
        content: str,
        email: str
    ) -> dict:
        """Try submitting via HTTP API."""
        import aiohttp

        api_url = os.environ.get("JUSTJOT_API_URL")
        api_key = os.environ.get("JUSTJOT_API_KEY")

        if not api_url:
            return None

        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "title": title,
                "content": content,
                "email": email,
                "source": "jotty-cli"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{api_url}/ideas",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200 or response.status == 201:
                        data = await response.json()
                        return {
                            'success': True,
                            'id': data.get('id'),
                            'url': data.get('url')
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f"API error {response.status}: {error_text}"
                        }
        except Exception as e:
            logger.debug(f"API submission failed: {e}")
            return None

    async def _try_skill_submission(
        self,
        title: str,
        content: str,
        email: str,
        cli: "JottyCLI"
    ) -> dict:
        """Try submitting via mcp-justjot skill."""
        try:
            registry = cli.get_skills_registry()

            skill = registry.get_skill('mcp-justjot')
            if skill:
                tool = skill.tools.get('create_idea_tool') or skill.tools.get('create_idea')
                if tool:
                    import asyncio

                    params = {
                        'title': title,
                        'description': content,
                        'tags': ['jotty', 'ai-generated']
                    }

                    if asyncio.iscoroutinefunction(tool):
                        result = await tool(params)
                    else:
                        result = tool(params)

                    if result.get('success'):
                        idea_id = result.get('idea_id') or result.get('id')
                        return {
                            'success': True,
                            'id': idea_id,
                            'url': f"https://justjot.ai/ideas/{idea_id}" if idea_id else None
                        }
                    return result
        except Exception as e:
            logger.debug(f"Skill submission not available: {e}")
        return None

    def get_completions(self, partial: str) -> list:
        """Get completions for JustJot command."""
        if partial.startswith("--"):
            return ["--email="]
        return []
