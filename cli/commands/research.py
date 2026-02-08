"""
Research Command
================

/research - Intelligent research with LLM synthesis.
Uses UnifiedExecutor for smart analysis and output formatting.
"""

import logging
import asyncio
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class ResearchCommand(BaseCommand):
    """
    /research - Intelligent research with LLM synthesis.

    Uses UnifiedExecutor to:
    1. Search for recent information
    2. Synthesize findings with LLM
    3. Output in requested format (text, pdf, docx, slides, telegram)
    """

    name = "research"
    aliases = ["r", "search"]
    description = "Research topic with LLM synthesis (supports --pdf, --docx, --slides, --telegram output)"
    usage = "/research <topic> [--pdf|--docx|--slides|--telegram|--deep]"
    category = "research"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute research command using UnifiedExecutor."""

        if not args.positional:
            cli.renderer.error("Topic required")
            cli.renderer.info("Usage: /research <topic> [options]")
            cli.renderer.info("")
            cli.renderer.info("Examples:")
            cli.renderer.info("  /research paytm")
            cli.renderer.info("  /research paytm --pdf")
            cli.renderer.info("  /research paytm --slides")
            cli.renderer.info("  /research paytm --slides --pdf")
            cli.renderer.info("  /research 'AI agents' --deep --docx")
            cli.renderer.info("  /research bitcoin and send to telegram")
            cli.renderer.info("")
            cli.renderer.info("Options:")
            cli.renderer.info("  --pdf        Save as PDF")
            cli.renderer.info("  --docx       Save as Word document")
            cli.renderer.info("  --slides     Generate PowerPoint presentation")
            cli.renderer.info("  --slides --pdf  Generate slides as PDF")
            cli.renderer.info("  --telegram   Send to Telegram")
            cli.renderer.info("  --deep       More comprehensive research")
            return CommandResult.fail("Topic required")

        # Build the natural language task
        topic = " ".join(args.positional)

        # Determine output format from flags
        # Handle --slides --pdf combination
        output_format = "text"
        is_slides = args.flags.get("slides")
        is_pdf = args.flags.get("pdf")

        if is_slides and is_pdf:
            output_format = "slides_pdf"  # Slides exported as PDF
        elif is_slides:
            output_format = "slides"  # PPTX format
        elif is_pdf:
            output_format = "pdf"
        elif args.flags.get("docx"):
            output_format = "docx"
        elif args.flags.get("telegram"):
            output_format = "telegram"

        # Build intelligent task description
        depth = "comprehensive and detailed" if args.flags.get("deep") else "concise"

        task = f"Research '{topic}' - find recent news, updates, and developments. "
        task += f"Provide a {depth} synthesis with key findings, trends, and insights. "

        if output_format == "slides":
            task += "Create a PowerPoint presentation (PPTX) with the research findings."
        elif output_format == "slides_pdf":
            task += "Create a presentation with the research findings and export as PDF slides."
        elif output_format == "pdf":
            task += "Save the research report as a PDF file."
        elif output_format == "docx":
            task += "Save the research report as a Word document."
        elif output_format == "telegram":
            task += "Send the research summary to Telegram."

        cli.renderer.header(f"Researching: {topic}")
        cli.renderer.info(f"Output: {output_format}")

        try:
            # Execute via UnifiedExecutor directly (peer to SwarmManager, not child)
            from core.orchestration.v2.unified_executor import UnifiedExecutor

            # Check if renderer supports async status (Telegram)
            has_async_status = hasattr(cli.renderer, 'send_status_async')

            async def async_status_callback(stage: str, detail: str = ""):
                """Async callback for real-time status streaming."""
                msg = f"{stage}: {detail}" if detail else stage
                if has_async_status:
                    await cli.renderer.send_status_async(msg)
                else:
                    cli.renderer.status(msg)

            def sync_status_callback(stage: str, detail: str = ""):
                """Sync callback (for CLI)."""
                cli.renderer.status(f"{stage}: {detail}" if detail else stage)

            # Use async callback if available (Telegram), else sync (CLI)
            status_cb = async_status_callback if has_async_status else sync_status_callback

            executor = UnifiedExecutor(status_callback=status_cb)
            result = await executor.execute(task)

            # Clear status message after completion (Telegram)
            if has_async_status and hasattr(cli.renderer, 'clear_status_message'):
                await cli.renderer.clear_status_message()

            if result.success:
                # Display the synthesized content
                cli.renderer.newline()

                if result.content:
                    cli.renderer.markdown(result.content)

                # Store in history for export
                if not hasattr(cli, '_output_history'):
                    cli._output_history = []
                cli._output_history.append(result.content or "")

                cli.renderer.newline()
                cli.renderer.success("Research complete")

                if result.output_path:
                    cli.renderer.info(f"Saved to: {result.output_path}")

                return CommandResult.ok(
                    output=result.content,
                    data={
                        "topic": topic,
                        "output_format": output_format,
                        "output_path": result.output_path,
                    }
                )
            else:
                cli.renderer.error(f"Research failed: {result.error}")
                return CommandResult.fail(result.error or "Unknown error")

        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            cli.renderer.error(f"Research failed: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get flag completions."""
        flags = ["--pdf", "--docx", "--slides", "--telegram", "--deep"]
        if partial.startswith("-"):
            return [f for f in flags if f.startswith(partial)]
        return []
