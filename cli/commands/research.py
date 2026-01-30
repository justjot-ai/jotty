"""
Research Command
================

/research - Research topics from Reddit + X (last 30 days)
Wrapper for the last30days Claude Code skill.
"""

import logging
import subprocess
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)

# Path to last30days skill
LAST30DAYS_SCRIPT = Path.home() / ".claude" / "skills" / "last30days" / "scripts" / "last30days.py"


class ResearchCommand(BaseCommand):
    """
    /research - Research a topic from Reddit + X (last 30 days).

    Uses the last30days skill to gather recent discussions.
    """

    name = "research"
    aliases = ["last30days", "l30d", "trending"]
    description = "Research topic from Reddit + X (last 30 days)"
    usage = "/research <topic> [--quick|--deep] [--sources reddit|x|both]"
    category = "research"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute research command."""

        if not args.positional:
            cli.renderer.error("Topic required")
            cli.renderer.info("Usage: /research <topic> [--quick|--deep]")
            cli.renderer.info("Examples:")
            cli.renderer.info("  /research paytm")
            cli.renderer.info("  /research 'AI agents' --deep")
            cli.renderer.info("  /research bitcoin --sources reddit")
            return CommandResult.fail("Topic required")

        topic = " ".join(args.positional)

        # Check if skill exists
        if not LAST30DAYS_SCRIPT.exists():
            cli.renderer.error("last30days skill not found")
            cli.renderer.info(f"Expected at: {LAST30DAYS_SCRIPT}")
            cli.renderer.info("Install: git clone <last30days-repo> ~/.claude/skills/last30days")
            return CommandResult.fail("Skill not found")

        # Build command
        cmd = ["python3", str(LAST30DAYS_SCRIPT), topic]

        # Add flags
        if args.flags.get("quick"):
            cmd.append("--quick")
        elif args.flags.get("deep"):
            cmd.append("--deep")

        sources = args.flags.get("sources", "auto")
        if sources != "auto":
            cmd.append(f"--sources={sources}")

        # Output mode
        emit = args.flags.get("emit", "compact")
        cmd.append(f"--emit={emit}")

        cli.renderer.header(f"Researching: {topic}")
        cli.renderer.info("Gathering from Reddit + X (last 30 days)...")

        try:
            # Run the script
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=LAST30DAYS_SCRIPT.parent
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300  # 5 minute timeout
            )

            output = stdout.decode('utf-8', errors='replace')
            errors = stderr.decode('utf-8', errors='replace')

            if process.returncode != 0:
                cli.renderer.error(f"Research failed: {errors}")
                return CommandResult.fail(errors)

            # Check if web search is needed and clean up output
            if "WEBSEARCH REQUIRED" in output:
                cli.renderer.info("Performing web search...")
                web_results = await self._do_web_search(topic, cli)

                # Clean up the output - remove the "WEBSEARCH REQUIRED" section
                # and any references to "Claude" or external AI
                clean_output = self._clean_research_output(output)

                if web_results:
                    output = clean_output + "\n\n## Web Search Results\n\n" + web_results
                else:
                    output = clean_output

            # Display output
            cli.renderer.newline()
            if emit == "md":
                cli.renderer.markdown(output)
            else:
                print(output)

            # Store in history for export
            if not hasattr(cli, '_output_history'):
                cli._output_history = []
            cli._output_history.append(output)

            cli.renderer.newline()
            cli.renderer.success("Research complete")

            return CommandResult.ok(output=output)

        except asyncio.TimeoutError:
            cli.renderer.error("Research timed out (5 minutes)")
            return CommandResult.fail("Timeout")
        except Exception as e:
            cli.renderer.error(f"Research failed: {e}")
            return CommandResult.fail(str(e))

    def _clean_research_output(self, output: str) -> str:
        """Clean research output - remove AI references and unnecessary sections."""
        import re

        # Remove the WEBSEARCH REQUIRED section entirely
        output = re.sub(
            r'={50,}.*?### WEBSEARCH REQUIRED ###.*?={50,}',
            '',
            output,
            flags=re.DOTALL
        )

        # Remove references to Claude/GPT/AI assistants
        replacements = [
            (r'Claude:?\s*', ''),
            (r'GPT:?\s*', ''),
            (r'Use your WebSearch tool.*?\n', ''),
            (r'After searching, synthesize.*?\n', ''),
            (r'WebSearch items should rank.*?\n', ''),
        ]
        for pattern, replacement in replacements:
            output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)

        # Clean up extra whitespace
        output = re.sub(r'\n{3,}', '\n\n', output)

        return output.strip()

    async def _do_web_search(self, topic: str, cli: "JottyCLI") -> str:
        """Perform web search using Jotty's web-search skill."""
        try:
            registry = cli.get_skills_registry()
            skill = registry.get_skill('web-search')

            if not skill:
                cli.renderer.warning("web-search skill not available")
                return ""

            search_tool = skill.tools.get('search_web_tool')
            if not search_tool:
                return ""

            # Search for recent news/articles
            queries = [
                f"{topic} news 2025",
                f"{topic} latest updates",
            ]

            results_text = []
            for query in queries:
                cli.renderer.status(f"Searching: {query}")
                result = search_tool({'query': query, 'max_results': 5})

                if result.get('success'):
                    for r in result.get('results', [])[:5]:
                        title = r.get('title', '')
                        snippet = r.get('snippet', '')
                        url = r.get('url', '')
                        results_text.append(f"**{title}**\n{snippet}\n[{url}]\n")

            if results_text:
                return "\n".join(results_text)
            return ""

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return ""

    def get_completions(self, partial: str) -> list:
        """Get flag completions."""
        flags = ["--quick", "--deep", "--sources", "--emit"]
        if partial.startswith("-"):
            return [f for f in flags if f.startswith(partial)]
        return []
