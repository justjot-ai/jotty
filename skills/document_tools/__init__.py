"""
Document Tools - Document Generation & Format Conversion
========================================================

Comprehensive document generation and format conversion utilities.

Modules:
- manager: OutputFormatManager - Multi-format document generation (PDF, EPUB, HTML, DOCX, PPTX)

Individual skills in this package:
- document-converter: Core conversion between formats
- pdf-tools: PDF generation utilities
- epub-builder: EPUB generation with chapters
- presenton: Presentation generation

Usage:
    # Multi-format generation
    from skills.document_tools import OutputFormatManager, OutputFormat

    manager = OutputFormatManager()

    # Generate PDF
    result = manager.generate_pdf(
        markdown_path="content.md",
        title="My Report",
        author="Author Name"
    )

    # Generate multiple formats
    results = manager.generate_all(
        markdown_path="content.md",
        formats=["pdf", "epub", "html"],
        title="My Report"
    )
"""

from .manager import (
    OutputFormat,
    OutputFormatManager,
    OutputFormatResult,
)

__all__ = [
    "OutputFormat",
    "OutputFormatManager",
    "OutputFormatResult",
]
