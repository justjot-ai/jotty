"""
Tests for document-tools and messaging-tools skill integration.

Verifies that document_tools and messaging_tools are registered and their tools
run correctly (with mocks where external services are used).

Execution tests import the tool modules as packages (skills.document_tools.tools,
skills.messaging_tools.tools) so that relative imports in those modules work.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure repo root is on path so "skills.document_tools" and "skills.messaging_tools" resolve
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.mark.unit
class TestDocumentToolsMessagingToolsRegistry:
    """Registry loads document-tools and messaging-tools; excludes _* support packages."""

    def test_document_tools_loaded(self):
        from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        reg = get_skills_registry()
        reg.init()
        skill = reg.get_skill("document_tools")
        assert skill is not None
        assert skill.name == "document_tools"
        tools = skill.tools
        assert "generate_pdf_tool" in tools
        assert "generate_epub_tool" in tools
        assert "generate_html_tool" in tools
        assert "generate_docx_tool" in tools
        assert "generate_presentation_tool" in tools
        assert "generate_all_formats_tool" in tools
        assert "generate_epub_with_chapters_tool" in tools

    def test_messaging_tools_loaded(self):
        from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        reg = get_skills_registry()
        reg.init()
        skill = reg.get_skill("messaging_tools")
        assert skill is not None
        assert skill.name == "messaging_tools"
        tools = skill.tools
        assert "send_to_telegram_tool" in tools
        assert "send_to_whatsapp_tool" in tools
        assert "send_to_all_channels_tool" in tools

    def test_support_packages_excluded(self):
        from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

        reg = get_skills_registry()
        reg.init()
        assert "_infrastructure" not in reg.loaded_skills
        assert "_providers" not in reg.loaded_skills
        assert "_tools" not in reg.loaded_skills


@pytest.mark.unit
class TestDocumentToolsExecution:
    """document-tools tools return expected shape and handle errors."""

    def test_generate_pdf_tool_missing_markdown_path(self):
        from skills.document_tools import tools as doc_tools

        out = doc_tools.generate_pdf_tool({})
        assert out.get("success") is False
        assert (
            "markdown_path" in (out.get("error") or "").lower()
            or "required" in (out.get("error") or "").lower()
        )

    def test_generate_pdf_tool_with_mock_manager(self):
        from skills.document_tools import tools as doc_tools

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.format = "pdf"
        mock_result.file_path = "/tmp/out.pdf"
        mock_result.error = None
        mock_result.metadata = {}

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            f.write(b"# Test\n\nHello.")
            f.flush()
            path = f.name
        try:
            with patch.object(doc_tools, "_get_manager") as get_mgr:
                mock_mgr = MagicMock()
                mock_mgr.generate_pdf.return_value = mock_result
                get_mgr.return_value = mock_mgr
                doc_tools._manager = None
                out = doc_tools.generate_pdf_tool({"markdown_path": path})
                assert out.get("success") is True
                assert out.get("file_path") == "/tmp/out.pdf"
                mock_mgr.generate_pdf.assert_called_once()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_generate_all_formats_tool_missing_params(self):
        from skills.document_tools import tools as doc_tools

        out = doc_tools.generate_all_formats_tool({})
        assert out.get("success") is False
        assert (
            "markdown_path" in (out.get("error") or "").lower()
            or "required" in (out.get("error") or "").lower()
        )

        out2 = doc_tools.generate_all_formats_tool({"markdown_path": "/tmp/x.md", "title": "T"})
        assert out2.get("success") is False
        assert (
            "formats" in (out2.get("error") or "").lower()
            or "required" in (out2.get("error") or "").lower()
        )


@pytest.mark.unit
@pytest.mark.asyncio
class TestMessagingToolsExecution:
    """messaging-tools tools return expected shape and handle errors."""

    async def test_send_to_telegram_tool_missing_both_file_and_message(self):
        from skills.messaging_tools import tools as msg_tools

        out = await msg_tools.send_to_telegram_tool({})
        assert out.get("success") is False
        assert (
            "file_path" in (out.get("error") or "").lower()
            or "message" in (out.get("error") or "").lower()
        )

    async def test_send_to_whatsapp_tool_missing_to(self):
        from skills.messaging_tools import tools as msg_tools

        out = await msg_tools.send_to_whatsapp_tool({"message": "hi"})
        assert out.get("success") is False
        assert (
            "to" in (out.get("error") or "").lower()
            or "required" in (out.get("error") or "").lower()
        )

    async def test_send_to_telegram_tool_with_mock_manager(self):
        from skills.messaging_tools import tools as msg_tools

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.channel = "telegram"
        mock_result.message_id = "123"
        mock_result.error = None
        mock_result.metadata = {}

        with patch.object(msg_tools, "_get_manager") as get_mgr:
            mock_mgr = MagicMock()
            mock_mgr.send_to_telegram = AsyncMock(return_value=mock_result)
            get_mgr.return_value = mock_mgr
            msg_tools._manager = None
            out = await msg_tools.send_to_telegram_tool({"message": "Hello"})
            assert out.get("success") is True
            assert out.get("channel") == "telegram"
            mock_mgr.send_to_telegram.assert_called_once()

    async def test_send_to_all_channels_tool_missing_channels(self):
        from skills.messaging_tools import tools as msg_tools

        out = await msg_tools.send_to_all_channels_tool({})
        assert out.get("success") is False
        assert (
            "channels" in (out.get("error") or "").lower()
            or "required" in (out.get("error") or "").lower()
        )
