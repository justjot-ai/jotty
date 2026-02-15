"""
Content Generator Skill

Generate professional documents and presentations:
- Document generation (PDF, DOCX)
- Slide deck creation (PPTX, HTML)
- Template-based content
"""

from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("content-generator")


@tool_wrapper(required_params=["content", "format"])
def generate_document(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a document from content.

    Params:
        content: Document content (markdown or text)
        format: Output format (pdf, docx, html)
        title: Document title
        output_path: Output file path

    Returns:
        {
            "success": True,
            "file_path": "...",
            "format": "pdf"
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .document import DocumentGenerator

        content = params["content"]
        format = params["format"]
        title = params.get("title", "Document")
        output_path = params.get("output_path")

        status.update(f"Generating {format} document...")

        generator = DocumentGenerator()
        file_path = generator.generate(
            content=content, format=format, title=title, output_path=output_path
        )

        status.complete("Document generated")

        return tool_response(file_path=file_path, format=format)

    except Exception as e:
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@tool_wrapper(required_params=["slides"])
def generate_presentation(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a presentation from slide content.

    Params:
        slides: List of slide dicts with title and content
        format: Output format (pptx, html) (default: pptx)
        theme: Presentation theme
        output_path: Output file path

    Returns:
        {
            "success": True,
            "file_path": "...",
            "slide_count": N
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .slides_generator import SlidesGenerator

        slides = params["slides"]
        format = params.get("format", "pptx")
        theme = params.get("theme", "default")
        output_path = params.get("output_path")

        status.update(f"Generating {len(slides)} slides...")

        generator = SlidesGenerator()
        file_path = generator.generate(
            slides=slides, format=format, theme=theme, output_path=output_path
        )

        status.complete(f"Presentation generated with {len(slides)} slides")

        return tool_response(file_path=file_path, slide_count=len(slides), format=format)

    except Exception as e:
        status.error(f"Failed: {e}")
        return tool_error(str(e))


__all__ = [
    "generate_document",
    "generate_presentation",
]
