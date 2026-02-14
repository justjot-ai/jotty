#!/usr/bin/env python3
"""
Output Sinks - Multi-Format Output Generation
==============================================

Generates outputs in multiple formats (PDF, EPUB, HTML, PPTX, DOCX)
using existing Jotty skills.

Uses:
- document-converter skill for PDF, EPUB, DOCX, HTML
- epub-builder skill for rich EPUB with chapters
- presenton skill for presentations
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""
    PDF = "pdf"
    EPUB = "epub"
    HTML = "html"
    DOCX = "docx"
    MARKDOWN = "markdown"
    PRESENTATION = "presentation"  # PPTX or PDF slides


@dataclass
class OutputSinkResult:
    """Result from output sink generation."""
    format: str
    success: bool
    file_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OutputSinkManager:
    """
    Manages multi-format output generation.

    Uses existing Jotty skills:
    - document-converter: PDF, EPUB, DOCX, HTML
    - epub-builder: Rich EPUB with chapters
    - presenton: Presentations (PPTX/PDF)

    Usage:
        manager = OutputSinkManager(output_dir="~/jotty/outputs")

        # Generate PDF
        result = manager.generate_pdf(
            markdown_path="content.md",
            title="My Research",
            author="Author Name"
        )

        # Generate multiple formats
        results = manager.generate_all(
            markdown_path="content.md",
            formats=["pdf", "epub", "html"],
            title="My Research"
        )
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        auto_load_skills: bool = True
    ):
        """
        Initialize output sink manager.

        Args:
            output_dir: Directory for outputs (default: ~/jotty/outputs)
            auto_load_skills: Auto-load Jotty skills registry
        """
        self.output_dir = Path(output_dir or os.path.expanduser("~/jotty/outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.registry = None
        self.skills = {}

        if auto_load_skills:
            self._load_skills()

    def _load_skills(self):
        """Load required skills from Jotty registry."""
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
            self.registry = get_skills_registry()
            self.registry.init()

            # Load document converter skill
            doc_skill = self.registry.get_skill('document-converter')
            if doc_skill and doc_skill.tools:
                self.skills['document-converter'] = doc_skill.tools
                logger.info("✅ Loaded document-converter skill")

            # Load epub builder skill
            epub_skill = self.registry.get_skill('epub-builder')
            if epub_skill and epub_skill.tools:
                self.skills['epub-builder'] = epub_skill.tools
                logger.info("✅ Loaded epub-builder skill")

            # Load presenton skill
            presenton_skill = self.registry.get_skill('presenton')
            if presenton_skill and presenton_skill.tools:
                self.skills['presenton'] = presenton_skill.tools
                logger.info("✅ Loaded presenton skill")

        except Exception as e:
            logger.warning(f"Could not load skills registry: {e}")
            logger.warning("Output sinks will generate markdown only")

    def generate_pdf(
        self,
        markdown_path: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        page_size: str = "a4",
        output_path: Optional[str] = None
    ) -> OutputSinkResult:
        """
        Generate PDF from markdown.

        Args:
            markdown_path: Path to markdown file
            title: Document title
            author: Document author
            page_size: Page size (a4, a5, letter, etc.)
            output_path: Custom output path

        Returns:
            OutputSinkResult with success status and file path
        """
        if 'document-converter' not in self.skills:
            return OutputSinkResult(
                format="pdf",
                success=False,
                error="document-converter skill not available"
            )

        try:
            convert_tool = self.skills['document-converter'].get('convert_to_pdf_tool')
            if not convert_tool:
                return OutputSinkResult(
                    format="pdf",
                    success=False,
                    error="convert_to_pdf_tool not found"
                )

            # Determine output path
            if output_path is None:
                base_name = Path(markdown_path).stem
                output_path = str(self.output_dir / f"{base_name}.pdf")

            # Convert
            result = convert_tool({
                'input_file': markdown_path,
                'output_file': output_path,
                'page_size': page_size,
                'title': title,
                'author': author
            })

            if result.get('success'):
                logger.info(f"✅ Generated PDF: {result.get('output_path')}")
                return OutputSinkResult(
                    format="pdf",
                    success=True,
                    file_path=result.get('output_path'),
                    metadata={'page_size': page_size}
                )
            else:
                return OutputSinkResult(
                    format="pdf",
                    success=False,
                    error=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return OutputSinkResult(
                format="pdf",
                success=False,
                error=str(e)
            )

    def generate_epub(
        self,
        markdown_path: str,
        title: str,
        author: str,
        output_path: Optional[str] = None
    ) -> OutputSinkResult:
        """Generate EPUB from markdown."""
        if 'document-converter' not in self.skills:
            return OutputSinkResult(
                format="epub",
                success=False,
                error="document-converter skill not available"
            )

        try:
            convert_tool = self.skills['document-converter'].get('convert_to_epub_tool')
            if not convert_tool:
                return OutputSinkResult(
                    format="epub",
                    success=False,
                    error="convert_to_epub_tool not found"
                )

            if output_path is None:
                base_name = Path(markdown_path).stem
                output_path = str(self.output_dir / f"{base_name}.epub")

            result = convert_tool({
                'input_file': markdown_path,
                'output_file': output_path,
                'title': title,
                'author': author
            })

            if result.get('success'):
                logger.info(f"✅ Generated EPUB: {result.get('output_path')}")
                return OutputSinkResult(
                    format="epub",
                    success=True,
                    file_path=result.get('output_path')
                )
            else:
                return OutputSinkResult(
                    format="epub",
                    success=False,
                    error=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"EPUB generation failed: {e}")
            return OutputSinkResult(
                format="epub",
                success=False,
                error=str(e)
            )

    def generate_epub_with_chapters(
        self,
        chapters: List[Dict[str, str]],
        title: str,
        author: str,
        description: Optional[str] = None,
        language: str = "en",
        output_path: Optional[str] = None
    ) -> OutputSinkResult:
        """
        Generate rich EPUB with chapters using epub-builder.

        Args:
            chapters: List of {'title': 'Chapter 1', 'content': 'markdown...'} dicts
            title: Book title
            author: Author name
            description: Book description
            language: Language code (default: en)
            output_path: Custom output path

        Returns:
            OutputSinkResult
        """
        if 'epub-builder' not in self.skills:
            return OutputSinkResult(
                format="epub",
                success=False,
                error="epub-builder skill not available"
            )

        try:
            build_tool = self.skills['epub-builder'].get('build_epub_tool')
            if not build_tool:
                return OutputSinkResult(
                    format="epub",
                    success=False,
                    error="build_epub_tool not found"
                )

            if output_path is None:
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_title = safe_title.replace(' ', '_')[:50]
                output_path = str(self.output_dir / f"{safe_title}.epub")

            result = build_tool({
                'title': title,
                'author': author,
                'chapters': chapters,
                'output_path': output_path,
                'language': language,
                'description': description
            })

            if result.get('success'):
                logger.info(f"✅ Generated EPUB with {result.get('chapter_count')} chapters: {result.get('output_path')}")
                return OutputSinkResult(
                    format="epub",
                    success=True,
                    file_path=result.get('output_path'),
                    metadata={'chapter_count': result.get('chapter_count')}
                )
            else:
                return OutputSinkResult(
                    format="epub",
                    success=False,
                    error=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"EPUB generation failed: {e}")
            return OutputSinkResult(
                format="epub",
                success=False,
                error=str(e)
            )

    def generate_html(
        self,
        markdown_path: str,
        title: Optional[str] = None,
        standalone: bool = True,
        output_path: Optional[str] = None
    ) -> OutputSinkResult:
        """Generate HTML from markdown."""
        if 'document-converter' not in self.skills:
            return OutputSinkResult(
                format="html",
                success=False,
                error="document-converter skill not available"
            )

        try:
            convert_tool = self.skills['document-converter'].get('convert_to_html_tool')
            if not convert_tool:
                return OutputSinkResult(
                    format="html",
                    success=False,
                    error="convert_to_html_tool not found"
                )

            if output_path is None:
                base_name = Path(markdown_path).stem
                output_path = str(self.output_dir / f"{base_name}.html")

            result = convert_tool({
                'input_file': markdown_path,
                'output_file': output_path,
                'title': title,
                'standalone': standalone
            })

            if result.get('success'):
                logger.info(f"✅ Generated HTML: {result.get('output_path')}")
                return OutputSinkResult(
                    format="html",
                    success=True,
                    file_path=result.get('output_path')
                )
            else:
                return OutputSinkResult(
                    format="html",
                    success=False,
                    error=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"HTML generation failed: {e}")
            return OutputSinkResult(
                format="html",
                success=False,
                error=str(e)
            )

    def generate_docx(
        self,
        markdown_path: str,
        title: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> OutputSinkResult:
        """Generate DOCX from markdown."""
        if 'document-converter' not in self.skills:
            return OutputSinkResult(
                format="docx",
                success=False,
                error="document-converter skill not available"
            )

        try:
            convert_tool = self.skills['document-converter'].get('convert_to_docx_tool')
            if not convert_tool:
                return OutputSinkResult(
                    format="docx",
                    success=False,
                    error="convert_to_docx_tool not found"
                )

            if output_path is None:
                base_name = Path(markdown_path).stem
                output_path = str(self.output_dir / f"{base_name}.docx")

            result = convert_tool({
                'input_file': markdown_path,
                'output_file': output_path,
                'title': title
            })

            if result.get('success'):
                logger.info(f"✅ Generated DOCX: {result.get('output_path')}")
                return OutputSinkResult(
                    format="docx",
                    success=True,
                    file_path=result.get('output_path')
                )
            else:
                return OutputSinkResult(
                    format="docx",
                    success=False,
                    error=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"DOCX generation failed: {e}")
            return OutputSinkResult(
                format="docx",
                success=False,
                error=str(e)
            )

    def generate_presentation(
        self,
        content: str,
        title: str,
        n_slides: int = 10,
        export_as: str = "pptx",
        tone: str = "professional",
        template: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> OutputSinkResult:
        """
        Generate presentation (PPTX or PDF) using presenton.

        Args:
            content: Presentation content/topic
            title: Presentation title
            n_slides: Number of slides
            export_as: Format (pptx or pdf)
            tone: Tone (professional, casual, funny, formal, inspirational)
            template: Design template name
            output_dir: Custom output directory

        Returns:
            OutputSinkResult
        """
        if 'presenton' not in self.skills:
            return OutputSinkResult(
                format="presentation",
                success=False,
                error="presenton skill not available"
            )

        try:
            gen_tool = self.skills['presenton'].get('generate_presentation_tool')
            if not gen_tool:
                return OutputSinkResult(
                    format="presentation",
                    success=False,
                    error="generate_presentation_tool not found"
                )

            result = gen_tool({
                'content': content,
                'n_slides': n_slides,
                'export_as': export_as,
                'tone': tone,
                'template': template,
                'auto_start': True
            })

            if result.get('success'):
                file_path = result.get('file_path')
                logger.info(f"✅ Generated presentation: {file_path}")
                return OutputSinkResult(
                    format="presentation",
                    success=True,
                    file_path=file_path,
                    metadata={
                        'presentation_id': result.get('presentation_id'),
                        'edit_url': result.get('edit_url'),
                        'n_slides': n_slides,
                        'format': export_as
                    }
                )
            else:
                return OutputSinkResult(
                    format="presentation",
                    success=False,
                    error=result.get('error', 'Unknown error')
                )

        except Exception as e:
            logger.error(f"Presentation generation failed: {e}")
            return OutputSinkResult(
                format="presentation",
                success=False,
                error=str(e)
            )

    def generate_all(
        self,
        markdown_path: str,
        formats: List[str],
        title: str,
        author: Optional[str] = None,
        **kwargs
    ) -> Dict[str, OutputSinkResult]:
        """
        Generate multiple output formats.

        Args:
            markdown_path: Path to markdown source
            formats: List of format names (pdf, epub, html, docx, presentation)
            title: Document title
            author: Document author
            **kwargs: Additional format-specific parameters

        Returns:
            Dict mapping format name to OutputSinkResult
        """
        results = {}

        for fmt in formats:
            fmt_lower = fmt.lower()

            if fmt_lower == "pdf":
                results["pdf"] = self.generate_pdf(
                    markdown_path, title, author,
                    page_size=kwargs.get('page_size', 'a4')
                )

            elif fmt_lower == "epub":
                results["epub"] = self.generate_epub(
                    markdown_path, title, author or "Unknown"
                )

            elif fmt_lower == "html":
                results["html"] = self.generate_html(
                    markdown_path, title,
                    standalone=kwargs.get('standalone', True)
                )

            elif fmt_lower == "docx":
                results["docx"] = self.generate_docx(
                    markdown_path, title
                )

            elif fmt_lower == "presentation":
                # Read markdown content for presentation
                try:
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    results["presentation"] = self.generate_presentation(
                        content=content,
                        title=title,
                        n_slides=kwargs.get('n_slides', 10),
                        export_as=kwargs.get('export_as', 'pptx'),
                        tone=kwargs.get('tone', 'professional')
                    )
                except Exception as e:
                    results["presentation"] = OutputSinkResult(
                        format="presentation",
                        success=False,
                        error=f"Could not read markdown: {e}"
                    )

            else:
                logger.warning(f"Unknown format: {fmt}")

        return results

    def get_summary(self, results: Dict[str, OutputSinkResult]) -> Dict[str, Any]:
        """
        Get summary of generation results.

        Args:
            results: Dict from generate_all()

        Returns:
            Summary dict with success counts and file paths
        """
        successful = [fmt for fmt, res in results.items() if res.success]
        failed = [fmt for fmt, res in results.items() if not res.success]

        file_paths = {
            fmt: res.file_path
            for fmt, res in results.items()
            if res.success and res.file_path
        }

        return {
            'total': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'successful_formats': successful,
            'failed_formats': failed,
            'file_paths': file_paths,
            'errors': {
                fmt: res.error
                for fmt, res in results.items()
                if not res.success and res.error
            }
        }
