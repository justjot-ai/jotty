"""Tests for OutputMixin — shared output infrastructure for learning swarms."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from Jotty.core.intelligence.orchestration.swarms._base._output_mixin import (
    OutputMixin,
)


class ConcreteOutputUser(OutputMixin):
    """Minimal class using the mixin."""

    pass


class TestGenerateOutputPdf:
    """Tests for _generate_output_pdf()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calls_generator_fn_with_output_path(self) -> None:
        """Generator function is called with constructed output_path."""
        user = ConcreteOutputUser()

        mock_generator = AsyncMock(return_value="/tmp/olympiad_test_topic_Student_lesson.pdf")

        result = await user._generate_output_pdf(
            generator_fn=mock_generator,
            topic="Test Topic",
            student_name="Student",
            prefix="olympiad",
            content="fake_content",
            celebration_word="Wow",
        )

        assert result == "/tmp/olympiad_test_topic_Student_lesson.pdf"
        mock_generator.assert_called_once()
        call_kwargs = mock_generator.call_args[1]
        assert "output_path" in call_kwargs
        assert call_kwargs["output_path"].startswith("/tmp/olympiad_")
        assert call_kwargs["content"] == "fake_content"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self) -> None:
        """Returns None when generator raises."""
        user = ConcreteOutputUser()
        mock_generator = AsyncMock(side_effect=RuntimeError("WeasyPrint exploded"))

        result = await user._generate_output_pdf(
            generator_fn=mock_generator,
            topic="Topic",
            student_name="Name",
            prefix="test",
        )
        assert result is None


class TestGenerateOutputHtml:
    """Tests for _generate_output_html()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_calls_generator_fn_with_output_path(self) -> None:
        user = ConcreteOutputUser()
        mock_generator = AsyncMock(return_value="/tmp/test_Topic_Name_slides.html")

        result = await user._generate_output_html(
            generator_fn=mock_generator,
            topic="Topic",
            student_name="Name",
            prefix="test",
            content="fake",
        )

        assert result is not None
        mock_generator.assert_called_once()
        call_kwargs = mock_generator.call_args[1]
        assert call_kwargs["output_path"].endswith("_slides.html")


class TestDeliverToTelegram:
    """Tests for _deliver_to_telegram()."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_skips_when_telegram_unavailable(self) -> None:
        """Does nothing when TELEGRAM_AVAILABLE is False."""
        user = ConcreteOutputUser()

        with patch(
            "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.TELEGRAM_AVAILABLE",
            False,
        ):
            # Should not raise even with invalid paths
            await user._deliver_to_telegram(
                topic="Topic",
                student_name="Name",
                summary_message="Hello",
                full_content="# Full",
                prefix="test",
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sends_summary_and_pdf(self, tmp_path: Path) -> None:
        """Sends summary message and PDF when available."""
        user = ConcreteOutputUser()

        # Create a fake PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("fake pdf")

        mock_msg = AsyncMock(return_value={"success": True, "message_id": 42})
        mock_file = AsyncMock(return_value={"success": True})

        with (
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.TELEGRAM_AVAILABLE",
                True,
            ),
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.send_telegram_message_tool",
                mock_msg,
            ),
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.send_telegram_file_tool",
                mock_file,
            ),
        ):
            await user._deliver_to_telegram(
                topic="Topic",
                student_name="Name",
                summary_message="*Summary*",
                full_content="# Full content",
                prefix="test",
                pdf_path=str(pdf_file),
            )

        # Summary was sent
        mock_msg.assert_called_once()
        # PDF was sent (but no markdown fallback since PDF exists)
        assert mock_file.call_count == 1
        assert mock_file.call_args[0][0]["file_path"] == str(pdf_file)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_markdown_fallback_when_no_pdf(self) -> None:
        """Sends markdown file as fallback when no PDF exists."""
        user = ConcreteOutputUser()

        mock_msg = AsyncMock(return_value={"success": True})
        mock_file = AsyncMock(return_value={"success": True})

        with (
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.TELEGRAM_AVAILABLE",
                True,
            ),
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.send_telegram_message_tool",
                mock_msg,
            ),
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.send_telegram_file_tool",
                mock_file,
            ),
        ):
            await user._deliver_to_telegram(
                topic="Topic",
                student_name="Name",
                summary_message="*Summary*",
                full_content="# Markdown content",
                prefix="test",
                pdf_path=None,
            )

        # Markdown file was sent as fallback
        assert mock_file.call_count == 1
        sent_path = mock_file.call_args[0][0]["file_path"]
        assert sent_path.endswith(".md")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_truncates_long_messages(self) -> None:
        """Messages longer than 4000 chars are truncated."""
        user = ConcreteOutputUser()

        long_message = "x" * 5000
        mock_msg = AsyncMock(return_value={"success": True})
        mock_file = AsyncMock(return_value={"success": True})

        with (
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.TELEGRAM_AVAILABLE",
                True,
            ),
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.send_telegram_message_tool",
                mock_msg,
            ),
            patch(
                "Jotty.core.intelligence.orchestration.swarms._base._output_mixin.send_telegram_file_tool",
                mock_file,
            ),
        ):
            await user._deliver_to_telegram(
                topic="T",
                student_name="S",
                summary_message=long_message,
                full_content="md",
                prefix="t",
            )

        sent_msg = mock_msg.call_args[0][0]["message"]
        assert len(sent_msg) <= 4000
