"""
Document Tools Skill - Multi-format document generation.

Exposes OutputFormatManager as registry tools for PDF, EPUB, HTML, DOCX, and presentation generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

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
    return {
        "success": result.success,
        "format": getattr(result, "format", ""),
        "file_path": getattr(result, "file_path", None),
        "error": getattr(result, "error", None),
        "metadata": getattr(result, "metadata", None) or {},
    }


def generate_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate PDF from a markdown file.

    Args:
        params: markdown_path (required), title, author, page_size (default a4), output_path

    Returns:
        success, file_path, error, format, metadata
    """
    manager = _get_manager()
    markdown_path = params.get("markdown_path")
    if not markdown_path:
        return {"success": False, "error": "markdown_path is required"}
    result = manager.generate_pdf(
        markdown_path=markdown_path,
        title=params.get("title"),
        author=params.get("author"),
        page_size=params.get("page_size", "a4"),
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


def generate_epub_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate EPUB from a markdown file.

    Args:
        params: markdown_path (required), title (required), author (required), output_path

    Returns:
        success, file_path, error, format, metadata
    """
    manager = _get_manager()
    markdown_path = params.get("markdown_path")
    title = params.get("title") or "Untitled"
    author = params.get("author") or "Unknown"
    if not markdown_path:
        return {"success": False, "error": "markdown_path is required"}
    result = manager.generate_epub(
        markdown_path=markdown_path,
        title=title,
        author=author,
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


def generate_epub_with_chapters_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate rich EPUB with chapters (epub-builder).

    Args:
        params: chapters (list of {title, content}), title, author, description, language, output_path

    Returns:
        success, file_path, error, format, metadata
    """
    manager = _get_manager()
    chapters = params.get("chapters") or []
    title = params.get("title") or "Untitled"
    author = params.get("author") or "Unknown"
    if not chapters:
        return {"success": False, "error": "chapters (list of {title, content}) is required"}
    result = manager.generate_epub_with_chapters(
        chapters=chapters,
        title=title,
        author=author,
        description=params.get("description"),
        language=params.get("language", "en"),
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


def generate_html_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate HTML from a markdown file.

    Args:
        params: markdown_path (required), title, standalone (default True), output_path

    Returns:
        success, file_path, error, format, metadata
    """
    manager = _get_manager()
    markdown_path = params.get("markdown_path")
    if not markdown_path:
        return {"success": False, "error": "markdown_path is required"}
    result = manager.generate_html(
        markdown_path=markdown_path,
        title=params.get("title"),
        standalone=params.get("standalone", True),
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


def generate_docx_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate DOCX from a markdown file.

    Args:
        params: markdown_path (required), title, output_path

    Returns:
        success, file_path, error, format, metadata
    """
    manager = _get_manager()
    markdown_path = params.get("markdown_path")
    if not markdown_path:
        return {"success": False, "error": "markdown_path is required"}
    result = manager.generate_docx(
        markdown_path=markdown_path,
        title=params.get("title"),
        output_path=params.get("output_path"),
    )
    return _result_to_dict(result)


def generate_presentation_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate presentation (PPTX/PDF) from content.

    Args:
        params: content (required, markdown/text), title (required), n_slides, export_as (pptx/pdf), tone

    Returns:
        success, file_path, error, format, metadata
    """
    manager = _get_manager()
    content = params.get("content")
    title = params.get("title") or "Presentation"
    if not content:
        return {"success": False, "error": "content is required"}
    result = manager.generate_presentation(
        content=content,
        title=title,
        n_slides=params.get("n_slides", 10),
        export_as=params.get("export_as", "pptx"),
        tone=params.get("tone", "professional"),
    )
    return _result_to_dict(result)


def generate_all_formats_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate multiple formats from one markdown file.

    Args:
        params: markdown_path (required), formats (required, list e.g. ["pdf","epub","html"]),
                title (required), author

    Returns:
        success (true if any succeeded), summary (total, successful, failed, file_paths), results per format
    """
    manager = _get_manager()
    markdown_path = params.get("markdown_path")
    formats_list = params.get("formats") or []
    title = params.get("title") or "Document"
    if not markdown_path:
        return {"success": False, "error": "markdown_path is required"}
    if not formats_list:
        return {
            "success": False,
            "error": 'formats (list) is required, e.g. ["pdf", "epub", "html"]',
        }
    results = manager.generate_all(
        markdown_path=markdown_path,
        formats=formats_list,
        title=title,
        author=params.get("author"),
    )
    summary = manager.get_summary(results)
    # Serialize per-format results for the response
    results_dict = {fmt: _result_to_dict(res) for fmt, res in results.items()}
    return {
        "success": summary.get("successful", 0) > 0,
        "summary": summary,
        "results": results_dict,
    }
