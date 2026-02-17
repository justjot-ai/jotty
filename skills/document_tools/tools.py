"""
Document Tools Skill - Multi-format document generation.

Exposes OutputFormatManager as registry tools for PDF, EPUB, HTML, DOCX, and presentation generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    result_to_tool_dict,
    tool_error,
    tool_wrapper,
)

status = SkillStatus("document-tools")

# Lazy import to avoid loading manager (and registry) until first tool use
_manager: Optional[Any] = None


def _get_manager() -> Any:
    global _manager
    if _manager is None:
        from .manager import OutputFormatManager

        _manager = OutputFormatManager()
    return _manager


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert OutputFormatResult to dict for tool response."""
    return result_to_tool_dict(result, include=("format", "file_path"))


@tool_wrapper(required_params=["markdown_path"])
def generate_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate PDF from a markdown file."""
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Creating", "Generating PDF...")
    manager = _get_manager()
    result = manager.generate_pdf(
        markdown_path=params["markdown_path"],
        title=params.get("title"),
        author=params.get("author"),
        page_size=params.get("page_size", "a4"),
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


@tool_wrapper(required_params=["markdown_path"])
def generate_epub_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate EPUB from a markdown file."""
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Creating", "Generating EPUB...")
    manager = _get_manager()
    result = manager.generate_epub(
        markdown_path=params["markdown_path"],
        title=params.get("title") or "Untitled",
        author=params.get("author") or "Unknown",
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


@tool_wrapper(required_params=["chapters"])
def generate_epub_with_chapters_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate rich EPUB with chapters (epub-builder)."""
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Creating", "Generating EPUB with chapters...")
    manager = _get_manager()
    result = manager.generate_epub_with_chapters(
        chapters=params["chapters"],
        title=params.get("title") or "Untitled",
        author=params.get("author") or "Unknown",
        description=params.get("description"),
        language=params.get("language", "en"),
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


@tool_wrapper(required_params=["markdown_path"])
def generate_html_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate HTML from a markdown file."""
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Creating", "Generating HTML...")
    manager = _get_manager()
    result = manager.generate_html(
        markdown_path=params["markdown_path"],
        title=params.get("title"),
        standalone=params.get("standalone", True),
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


@tool_wrapper(required_params=["markdown_path"])
def generate_docx_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate DOCX from a markdown file."""
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Creating", "Generating DOCX...")
    manager = _get_manager()
    result = manager.generate_docx(
        markdown_path=params["markdown_path"],
        title=params.get("title"),
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


@tool_wrapper(required_params=["content"])
def generate_presentation_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate presentation (PPTX/PDF) from content."""
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Creating", "Generating presentation...")
    manager = _get_manager()
    result = manager.generate_presentation(
        content=params["content"],
        title=params.get("title") or "Presentation",
        n_slides=params.get("n_slides", 10),
        export_as=params.get("export_as", "pptx"),
        tone=params.get("tone", "professional"),
    )
    return _result_to_dict(result)


@tool_wrapper(required_params=["markdown_path", "formats"])
def generate_all_formats_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate multiple formats from one markdown file."""
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Creating", "Generating multiple formats...")
    manager = _get_manager()
    results = manager.generate_all(
        markdown_path=params["markdown_path"],
        formats=params["formats"],
        title=params.get("title") or "Document",
        author=params.get("author"),
    )
    summary = manager.get_summary(results)
    results_dict = {fmt: _result_to_dict(res) for fmt, res in results.items()}
    return {
        "success": summary.get("successful", 0) > 0,
        "summary": summary,
        "results": results_dict,
    }
