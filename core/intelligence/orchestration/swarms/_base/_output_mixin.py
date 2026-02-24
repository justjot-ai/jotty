"""
Output mixin for learning swarms.

Consolidates the duplicated PDF generation, HTML generation, and Telegram
delivery logic shared by OlympiadLearningSwarm and PerspectiveLearningSwarm.

Each concrete swarm provides domain-specific pieces:
- ``_build_telegram_summary()``: builds the Markdown summary message
- PDF/HTML generator functions (passed as callables)

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Telegram support — imported once, shared by all swarms using this mixin.
try:
    import sys

    _telegram_skill_path = str(
        Path(__file__).parent.parent.parent.parent / "skills" / "telegram-sender"
    )
    if _telegram_skill_path not in sys.path:
        sys.path.insert(0, _telegram_skill_path)
    from tools import send_telegram_file_tool, send_telegram_message_tool

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    send_telegram_message_tool = None  # type: ignore[assignment]
    send_telegram_file_tool = None  # type: ignore[assignment]


class OutputMixin:
    """Mixin providing shared PDF, HTML, and Telegram output for learning swarms.

    Concrete swarms inherit this mixin and override:
    - ``_build_telegram_summary()`` to produce a domain-specific Markdown message

    The mixin provides:
    - ``_generate_output_pdf()`` — calls a generator function with error handling
    - ``_generate_output_html()`` — calls a generator function with error handling
    - ``_deliver_to_telegram()`` — sends summary + PDF/HTML + markdown fallback
    """

    @staticmethod
    def is_telegram_available() -> bool:
        """Check if Telegram delivery is available."""
        return TELEGRAM_AVAILABLE

    async def _generate_output_pdf(
        self,
        generator_fn: Callable[..., Awaitable[Optional[str]]],
        topic: str,
        student_name: str,
        prefix: str,
        **generator_kwargs: Any,
    ) -> Optional[str]:
        """Generate a PDF using the provided generator function.

        Args:
            generator_fn: Async callable that generates the PDF (e.g. generate_lesson_pdf).
            topic: Topic name for the filename.
            student_name: Student name for the filename.
            prefix: Filename prefix (e.g. "olympiad", "perspective").
            **generator_kwargs: Extra kwargs passed to generator_fn.

        Returns:
            Path to the generated PDF, or None on failure.
        """
        try:
            safe_topic = topic.replace(" ", "_").replace(":", "").replace(",", "")[:50]
            output_path = f"/tmp/{prefix}_{safe_topic}_{student_name.replace(' ', '_')}_lesson.pdf"

            pdf_path = await generator_fn(output_path=output_path, **generator_kwargs)

            if pdf_path:
                logger.info(f"Generated PDF: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.warning(f"PDF generation failed: {e}")
            return None

    async def _generate_output_html(
        self,
        generator_fn: Callable[..., Awaitable[Optional[str]]],
        topic: str,
        student_name: str,
        prefix: str,
        **generator_kwargs: Any,
    ) -> Optional[str]:
        """Generate an HTML file using the provided generator function.

        Args:
            generator_fn: Async callable that generates the HTML.
            topic: Topic name for the filename.
            student_name: Student name for the filename.
            prefix: Filename prefix (e.g. "olympiad", "perspective").
            **generator_kwargs: Extra kwargs passed to generator_fn.

        Returns:
            Path to the generated HTML, or None on failure.
        """
        try:
            safe_topic = topic.replace(" ", "_").replace(":", "").replace(",", "")[:50]
            output_path = f"/tmp/{prefix}_{safe_topic}_{student_name.replace(' ', '_')}_slides.html"

            html_path = await generator_fn(output_path=output_path, **generator_kwargs)

            if html_path:
                logger.info(f"Generated HTML: {html_path}")
            return html_path

        except Exception as e:
            logger.warning(f"HTML generation failed: {e}")
            return None

    async def _deliver_to_telegram(
        self,
        topic: str,
        student_name: str,
        summary_message: str,
        full_content: str,
        prefix: str,
        pdf_path: Optional[str] = None,
        html_path: Optional[str] = None,
        md_path: Optional[str] = None,
    ) -> Any:
        """Send summary message + PDF/HTML/MD files to Telegram.

        This is the shared delivery logic. Each swarm builds its own
        ``summary_message`` via ``_build_telegram_summary()`` and passes
        it here.

        Args:
            topic: Topic name.
            student_name: Student name.
            summary_message: Pre-built Markdown summary message.
            full_content: Full markdown text (used as fallback if no PDF/MD).
            prefix: File prefix for temp markdown file.
            pdf_path: Path to generated PDF (if any).
            html_path: Path to generated HTML (if any).
            md_path: Path to generated Markdown (if any).
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram tools not available")
            return

        try:
            # Truncate message to Telegram's 4096 char limit
            message = summary_message
            if len(message) > 4000:
                message = message[:3950] + "\n..."

            result = await send_telegram_message_tool(
                {"message": message, "parse_mode": "Markdown"}
            )
            if result.get("success"):
                logger.info(f"Sent summary to Telegram: message_id {result.get('message_id')}")
            else:
                logger.error(f"Telegram message failed: {result.get('error')}")

            has_pdf = pdf_path and Path(pdf_path).exists()
            has_html = html_path and Path(html_path).exists()

            # Send PDF
            if has_pdf:
                file_result = await send_telegram_file_tool(
                    {
                        "file_path": pdf_path,
                        "caption": f"{topic} - Learning Guide for {student_name}",
                    }
                )
                if file_result.get("success"):
                    logger.info("Sent PDF to Telegram")
                else:
                    logger.error(f"PDF send failed: {file_result.get('error')}")

            # Send HTML
            if has_html:
                file_result = await send_telegram_file_tool(
                    {
                        "file_path": html_path,
                        "caption": f"{topic} - Interactive Slides for {student_name}",
                    }
                )
                if file_result.get("success"):
                    logger.info("Sent HTML to Telegram")
                else:
                    logger.error(f"HTML send failed: {file_result.get('error')}")

            # Send Markdown file (saved .md or temp fallback)
            has_md = md_path and Path(md_path).exists()
            if has_md:
                file_result = await send_telegram_file_tool(
                    {
                        "file_path": md_path,
                        "caption": f"{topic} - Lesson for {student_name} (Markdown)",
                    }
                )
                if file_result.get("success"):
                    logger.info("Sent markdown to Telegram")
                else:
                    logger.error(f"Markdown send failed: {file_result.get('error')}")
            elif not has_pdf and full_content:
                # Fallback: create temp markdown if no PDF and no saved .md
                safe_topic = topic.replace(" ", "_").replace(":", "")[:40]
                temp_path = Path(f"/tmp/{prefix}_{safe_topic}_{student_name.replace(' ', '_')}.md")
                temp_path.write_text(full_content, encoding="utf-8")

                file_result = await send_telegram_file_tool(
                    {
                        "file_path": str(temp_path),
                        "caption": f"{topic} - Lesson for {student_name} (Markdown)",
                    }
                )
                if file_result.get("success"):
                    logger.info("Sent markdown fallback to Telegram")

                try:
                    temp_path.unlink()
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            import traceback

            traceback.print_exc()


__all__ = ["OutputMixin", "TELEGRAM_AVAILABLE"]
