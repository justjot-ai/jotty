#!/usr/bin/env python3
"""
Content Generation Tools for Jotty
Ported from JustJot.ai adapters/sinks/

Provides PDF, HTML, and Markdown generation with @jotty_method decorators
"""

from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import re
from datetime import datetime

from .document import Document, Section, SectionType


class ContentGenerators:
    """
    Content generation tools for Jotty agents

    Provides methods to create PDFs, HTML, and Markdown files from documents
    """

    def __init__(self):
        self.output_dir = Path("./outputs/research")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem"""
        # Remove/replace unsafe characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        # Limit length
        filename = filename[:200]
        return filename if filename else 'document'

    # =========================================================================
    # PDF Generation (via Pandoc + XeLaTeX)
    # =========================================================================

    def generate_pdf(
        self,
        document: Document,
        output_path: Optional[Path] = None,
        format: str = 'a4'
    ) -> Path:
        """
        Generate PDF from document using pandoc + XeLaTeX

        Args:
            document: Document to convert
            output_path: Output directory (default: ./outputs/research)
            format: PDF format (a4, a5, letter) - default: a4

        Returns:
            Path to generated PDF

        Raises:
            RuntimeError: If conversion fails
        """
        # Format mapping
        FORMAT_MAP = {
            'a4': 'a4',
            'a5': 'a5',
            'a6': 'a6',
            'letter': 'letter',
        }

        pdf_format = format.lower()
        if pdf_format not in FORMAT_MAP:
            raise ValueError(f"Unknown PDF format: {pdf_format}. "
                           f"Must be one of: {list(FORMAT_MAP.keys())}")

        pandoc_page_size = FORMAT_MAP[pdf_format]

        # Determine output path
        if output_path is None:
            output_path = self.output_dir

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = self._sanitize_filename(document.title)
        pdf_path = output_path / f"{filename}_{pdf_format}.pdf"

        # Get markdown content
        markdown_content = document.full_content
        if not markdown_content or not markdown_content.strip():
            raise ValueError("Document has no content to convert")

        # Save temporary markdown file
        temp_md = output_path / f"{filename}_temp.md"
        try:
            with open(temp_md, 'w', encoding='utf-8') as f:
                f.write(f"# {document.title}\n\n")
                if document.author:
                    f.write(f"**Author:** {document.author}\n\n")
                f.write(f"**Generated:** {document.created.strftime('%Y-%m-%d')}\n\n")
                f.write("---\n\n")
                f.write(markdown_content)

            print(f"      Converting to PDF ({pdf_format.upper()})...")

            # Use pandoc to convert to PDF
            cmd = [
                'pandoc',
                str(temp_md),
                '-f', 'markdown',
                '-t', 'pdf',
                '--pdf-engine=xelatex',
                f'--variable=papersize:{pandoc_page_size}',
                '-o', str(pdf_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                error_output = result.stderr if result.stderr else result.stdout
                raise RuntimeError(f"Pandoc failed with exit code {result.returncode}: {error_output}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Pandoc conversion timed out (>60 seconds)")
        except Exception as e:
            raise RuntimeError(f"PDF conversion failed: {e}")
        finally:
            # Always try to clean up temp file
            try:
                if temp_md.exists():
                    temp_md.unlink()
            except Exception as cleanup_error:
                print(f"      Warning: Failed to clean up temp file '{temp_md}': {cleanup_error}")

        # Verify output file was created
        if not pdf_path.exists():
            raise RuntimeError(
                f"PDF conversion completed but output file was not created at '{pdf_path}'. "
                f"This may indicate a silent failure in pandoc or the LaTeX engine."
            )

        # Verify output file has content
        file_size = pdf_path.stat().st_size
        if file_size == 0:
            pdf_path.unlink()
            raise RuntimeError(f"Generated PDF file is empty (0 bytes) - conversion failed")

        print(f"      ✅ PDF created: {pdf_path.name} ({file_size} bytes)")
        return pdf_path

    # =========================================================================
    # HTML Generation (via Pandoc)
    # =========================================================================

    def generate_html(
        self,
        document: Document,
        output_path: Optional[Path] = None,
        standalone: bool = True,
        include_toc: bool = True
    ) -> Path:
        """
        Generate HTML from document using pandoc

        Args:
            document: Document to convert
            output_path: Output directory (default: ./outputs/research)
            standalone: Create standalone HTML with CSS (default: True)
            include_toc: Include table of contents (default: True)

        Returns:
            Path to generated HTML
        """
        # Determine output path
        if output_path is None:
            output_path = self.output_dir

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = self._sanitize_filename(document.title)
        html_path = output_path / f"{filename}.html"

        # Get markdown content
        markdown_content = document.full_content

        # Save temporary markdown file
        temp_md = output_path / f"{filename}_temp.md"
        try:
            with open(temp_md, 'w', encoding='utf-8') as f:
                f.write(f"# {document.title}\n\n")
                if document.author:
                    f.write(f"**Author:** {document.author}\n\n")
                f.write(f"**Generated:** {document.created.strftime('%Y-%m-%d')}\n\n")
                f.write("---\n\n")
                f.write(markdown_content)

            print(f"      Converting to HTML...")

            # Use pandoc to convert markdown to HTML
            cmd = [
                'pandoc',
                str(temp_md),
                '-o', str(html_path),
                '--mathml',  # Use MathML for math
            ]

            # Add standalone HTML with CSS if requested
            if standalone:
                cmd.append('--standalone')
                cmd.append('--self-contained')

            # Add table of contents if requested
            if include_toc:
                cmd.extend(['--toc', '--toc-depth=3'])

            # Add metadata
            cmd.extend([
                '--metadata', f'title={document.title}',
            ])

            if document.author:
                cmd.extend(['--metadata', f'author={document.author}'])

            # Run pandoc
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                raise RuntimeError(f"Pandoc failed: {result.stderr}")

            # Clean up temp file
            temp_md.unlink()

            print(f"      ✅ HTML: {html_path.name}")

            return html_path

        except FileNotFoundError:
            raise RuntimeError("pandoc not found. Please install pandoc: https://pandoc.org/installing.html")
        except subprocess.TimeoutExpired:
            raise RuntimeError("HTML conversion timed out")
        except Exception as e:
            # Clean up temp file on error
            if temp_md.exists():
                temp_md.unlink()
            raise RuntimeError(f"HTML conversion failed: {e}")

    # =========================================================================
    # Markdown Export
    # =========================================================================

    def export_markdown(
        self,
        document: Document,
        output_path: Optional[Path] = None,
        include_metadata: bool = True
    ) -> Path:
        """
        Export document to Markdown file

        Args:
            document: Document to export
            output_path: Output directory (default: ./outputs/research)
            include_metadata: Include YAML frontmatter (default: True)

        Returns:
            Path to generated .md file
        """
        # Determine output path
        if output_path is None:
            output_path = self.output_dir

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = self._sanitize_filename(document.title)
        date_prefix = document.created.strftime('%Y-%m-%d')
        md_path = output_path / f"{date_prefix}-{filename}.md"

        print(f"      Exporting to Markdown...")

        # Build markdown content
        md_content = []

        if include_metadata:
            # Add frontmatter metadata
            md_content.append("---")
            yaml_title = document.title.replace('"', '\\"')
            md_content.append(f'title: "{yaml_title}"')
            if document.author:
                md_content.append(f"author: {document.author}")
            if document.topic:
                md_content.append(f"topic: {document.topic}")
            md_content.append(f"date: {document.created.strftime('%Y-%m-%d')}")
            md_content.append(f"source: {document.source_type}")
            md_content.append("---")
            md_content.append("")

        # Add title
        md_content.append(f"# {document.title}")
        md_content.append("")

        # Add content
        md_content.append(document.full_content)

        # Write file
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))

        print(f"      ✅ MD: {md_path.name}")

        return md_path
