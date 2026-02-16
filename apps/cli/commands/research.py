"""
Research Command
================

/research - Intelligent research with LLM synthesis.
Uses TierExecutor for smart analysis and output formatting.
"""

import logging
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class ResearchCommand(BaseCommand):
    """
    /research - Intelligent research with LLM synthesis.

    Uses TierExecutor to:
    1. Search for recent information
    2. Synthesize findings with LLM
    3. Output in requested format (text, pdf, docx, slides, telegram)
    """

    name = "research"
    aliases = ["r", "search"]
    description = (
        "Research topic with LLM synthesis (supports --pdf, --docx, --slides, --telegram output)"
    )
    usage = "/research <topic> [--pdf|--docx|--slides|--telegram|--deep]"
    category = "research"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute research command using TierExecutor."""

        if not args.positional:
            cli.renderer.error("Topic required")  # type: ignore[attr-defined]
            cli.renderer.info("Usage: /research <topic> [options]")  # type: ignore[attr-defined]
            cli.renderer.info("")  # type: ignore[attr-defined]
            cli.renderer.info("Examples:")  # type: ignore[attr-defined]
            cli.renderer.info("  /research paytm")  # type: ignore[attr-defined]
            cli.renderer.info("  /research paytm --pdf")  # type: ignore[attr-defined]
            cli.renderer.info("  /research paytm --slides")  # type: ignore[attr-defined]
            cli.renderer.info("  /research paytm --slides --pdf")  # type: ignore[attr-defined]
            cli.renderer.info("  /research 'AI agents' --deep --docx")  # type: ignore[attr-defined]
            cli.renderer.info("  /research bitcoin and send to telegram")  # type: ignore[attr-defined]
            cli.renderer.info("")  # type: ignore[attr-defined]
            cli.renderer.info("Options:")  # type: ignore[attr-defined]
            cli.renderer.info("  --pdf        Save as PDF")  # type: ignore[attr-defined]
            cli.renderer.info("  --docx       Save as Word document")  # type: ignore[attr-defined]
            cli.renderer.info("  --slides     Generate PowerPoint presentation")  # type: ignore[attr-defined]
            cli.renderer.info("  --slides --pdf  Generate slides as PDF")  # type: ignore[attr-defined]
            cli.renderer.info("  --telegram   Send to Telegram")  # type: ignore[attr-defined]
            cli.renderer.info("  --deep       More comprehensive research")  # type: ignore[attr-defined]
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

        cli.renderer.header(f"Researching: {topic}")  # type: ignore[attr-defined]
        cli.renderer.info(f"Output: {output_format}")  # type: ignore[attr-defined]

        try:
            # Execute via SDK (clean architecture)
            # Get SDK client from CLI app
            sdk_client = cli._get_sdk_client()  # type: ignore[attr-defined]

            # Check if renderer supports async status (Telegram)
            has_async_status = hasattr(cli.renderer, "send_status_async")  # type: ignore[attr-defined]

            # Register event listeners for status updates
            async def on_skill_start(event) -> None:
                """Handle skill start events."""
                skill_name = event.data.get("skill", "unknown") if event.data else "unknown"
                msg = f"Using skill: {skill_name}"
                if has_async_status:
                    await cli.renderer.send_status_async(msg)  # type: ignore[attr-defined]
                else:
                    cli.renderer.status(msg)  # type: ignore[attr-defined]

            async def on_thinking(event) -> None:
                """Handle thinking events."""
                if has_async_status:
                    await cli.renderer.send_status_async("Thinking...")  # type: ignore[attr-defined]
                else:
                    cli.renderer.status("Thinking...")  # type: ignore[attr-defined]

            # Import SDK types
            from Jotty.sdk import SDKEventType

            # Register event handlers
            sdk_client.on(SDKEventType.SKILL_START, on_skill_start)
            sdk_client.on(SDKEventType.THINKING, on_thinking)

            # Execute via SDK
            result = await sdk_client.chat(task)

            # Clear status message after completion (Telegram)
            if has_async_status and hasattr(cli.renderer, "clear_status_message"):  # type: ignore[attr-defined]
                await cli.renderer.clear_status_message()  # type: ignore[attr-defined]

            if result.success:
                # Display the synthesized content
                cli.renderer.newline()  # type: ignore[attr-defined]

                if result.content:
                    cli.renderer.markdown(result.content)  # type: ignore[attr-defined]

                # Store in history for export
                if not hasattr(cli, "_output_history"):
                    cli._output_history = []
                cli._output_history.append(result.content or "")

                cli.renderer.newline()  # type: ignore[attr-defined]
                cli.renderer.success("Research complete")  # type: ignore[attr-defined]

                output_path = result.metadata.get("output_path") if result.metadata else None
                if output_path:
                    cli.renderer.info(f"Saved to: {output_path}")  # type: ignore[attr-defined]

                return CommandResult.ok(
                    output=result.content,
                    data={
                        "topic": topic,
                        "output_format": output_format,
                        "output_path": output_path,
                    },
                )
            else:
                cli.renderer.error(f"Research failed: {result.error}")  # type: ignore[attr-defined]
                return CommandResult.fail(result.error or "Unknown error")

        except Exception as e:
            logger.error(f"Research failed: {e}", exc_info=True)
            cli.renderer.error(f"Research failed: {e}")  # type: ignore[attr-defined]
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get flag completions."""
        flags = ["--pdf", "--docx", "--slides", "--telegram", "--deep"]
        if partial.startswith("-"):
            return [f for f in flags if f.startswith(partial)]
        return []
