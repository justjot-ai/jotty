"""
Enhanced Export Utilities for Jotty Web
=======================================

Professional document export with:
- Multiple PDF engines with fallback
- Custom templates for styling
- Code syntax highlighting
- Math equation support
- Error recovery and detailed logging
"""

import logging
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ExportConfig:
    """Export configuration and templates."""

    # PDF LaTeX template with modern styling
    PDF_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}

% Modern fonts
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

% Page layout
\usepackage[margin=1in]{geometry}
\usepackage{parskip}

% Code highlighting
\usepackage{listings}
\usepackage{xcolor}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10},
    keywordstyle=\color{blue}\bfseries,
    commentstyle=\color{green!60!black},
    stringstyle=\color{orange},
    numbers=left,
    numberstyle=\tiny\color{gray},
    numbersep=5pt
}

% Better tables
\usepackage{booktabs}
\usepackage{longtable}

% Images
\usepackage{graphicx}

% Links
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}

% Math
\usepackage{amsmath}
\usepackage{amssymb}

% Header/Footer
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\rhead{\thepage}
\lhead{$title$}

% Title styling
\usepackage{titling}
\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\par\end{center}\vskip 0.5em}
\predate{\begin{center}\small}
\postdate{\par\end{center}}

\title{$title$}
\date{$date$}

\begin{document}

$if(title)$
\maketitle
$endif$

$body$

\end{document}
"""

    # Beamer slides template
    SLIDES_TEMPLATE = r"""
\documentclass[aspectratio=169]{beamer}

\usetheme{metropolis}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage{listings}
\usepackage{booktabs}

\lstset{
    basicstyle=\ttfamily\scriptsize,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10}
}

\title{$title$}
\date{$date$}

\begin{document}

\begin{frame}
\titlepage
\end{frame}

$body$

\end{document}
"""


class DocumentExporter:
    """
    Professional document exporter with fallbacks and error handling.
    """

    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="jotty_export_"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _write_template(self, name: str, content: str) -> Path:
        """Write template to temp directory."""
        path = self.temp_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def _run_pandoc(self, args: list, timeout: int = 60) -> Tuple[bool, str]:
        """Run pandoc with error capture."""
        try:
            result = subprocess.run(
                ["pandoc"] + args, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                return False, result.stderr or "Unknown error"
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Conversion timed out"
        except FileNotFoundError:
            return False, "Pandoc not installed"
        except Exception as e:
            return False, str(e)

    def export_markdown(self, content: str, filename: str) -> Path:
        """Export as Markdown (passthrough)."""
        output = self.temp_dir / f"{filename}.md"
        output.write_text(content, encoding="utf-8")
        return output

    def export_html(self, content: str, filename: str, standalone: bool = True) -> Path:
        """Export as HTML with syntax highlighting."""
        md_file = self.temp_dir / f"{filename}.md"
        md_file.write_text(content, encoding="utf-8")

        output = self.temp_dir / f"{filename}.html"

        args = [
            str(md_file),
            "-o",
            str(output),
            "--highlight-style=pygments",
            "--metadata",
            f"title={filename}",
        ]
        if standalone:
            args.append("--standalone")

        success, error = self._run_pandoc(args)
        if not success:
            raise ExportError(f"HTML export failed: {error}")

        return output

    def export_pdf(self, content: str, filename: str, title: Optional[str] = None) -> Path:
        """
        Export as PDF with multiple engine fallback.

        Tries: xelatex -> pdflatex -> weasyprint (HTML->PDF)
        """
        md_file = self.temp_dir / f"{filename}.md"
        md_file.write_text(content, encoding="utf-8")

        output = self.temp_dir / f"{filename}.pdf"
        title = title or filename
        date = datetime.now().strftime("%Y-%m-%d")

        # Write custom template
        template_file = self._write_template("template.latex", ExportConfig.PDF_TEMPLATE)

        # Try xelatex first (best Unicode support)
        engines = ["xelatex", "pdflatex", "lualatex"]
        last_error = ""

        for engine in engines:
            args = [
                str(md_file),
                "-o",
                str(output),
                f"--pdf-engine={engine}",
                f"--template={template_file}",
                "-V",
                f"title={title}",
                "-V",
                f"date={date}",
                "--highlight-style=tango",
            ]

            success, error = self._run_pandoc(args, timeout=120)

            if success and output.exists():
                logger.info(f"PDF exported successfully with {engine}")
                return output
            else:
                last_error = error
                logger.warning(f"PDF export with {engine} failed: {error}")

        # Final fallback: HTML -> PDF via weasyprint
        try:
            import weasyprint

            html_file = self.export_html(content, filename, standalone=True)
            weasyprint.HTML(filename=str(html_file)).write_pdf(str(output))
            if output.exists():
                logger.info("PDF exported via WeasyPrint fallback")
                return output
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"WeasyPrint fallback failed: {e}")

        raise ExportError(f"PDF export failed with all engines. Last error: {last_error}")

    def export_docx(self, content: str, filename: str, title: Optional[str] = None) -> Path:
        """Export as DOCX with professional styling."""
        md_file = self.temp_dir / f"{filename}.md"
        md_file.write_text(content, encoding="utf-8")

        output = self.temp_dir / f"{filename}.docx"
        title = title or filename

        args = [
            str(md_file),
            "-o",
            str(output),
            "--metadata",
            f"title={title}",
            "--highlight-style=tango",
        ]

        success, error = self._run_pandoc(args)
        if not success:
            raise ExportError(f"DOCX export failed: {error}")

        return output

    def export_slides(self, content: str, filename: str, title: Optional[str] = None) -> Path:
        """Export as Beamer PDF slides."""
        # Preprocess content for slides (add slide breaks if not present)
        if "---" not in content and "# " in content:
            # Convert H1/H2 headers to slide breaks
            lines = content.split("\n")
            processed = []
            for line in lines:
                if line.startswith("## "):
                    processed.append("---")
                    processed.append("")
                processed.append(line)
            content = "\n".join(processed)

        md_file = self.temp_dir / f"{filename}.md"
        md_file.write_text(content, encoding="utf-8")

        output = self.temp_dir / f"{filename}_slides.pdf"
        title = title or filename
        date = datetime.now().strftime("%Y-%m-%d")

        # Check if metropolis theme is available, fallback to Madrid
        themes = ["metropolis", "Madrid", "default"]
        last_error = ""

        for theme in themes:
            args = [
                str(md_file),
                "-o",
                str(output),
                "-t",
                "beamer",
                "--pdf-engine=xelatex",
                "-V",
                f"theme:{theme}",
                "-V",
                f"title={title}",
                "-V",
                f"date={date}",
                "--highlight-style=tango",
            ]

            success, error = self._run_pandoc(args, timeout=120)

            if success and output.exists():
                logger.info(f"Slides exported with theme {theme}")
                return output
            else:
                last_error = error
                logger.warning(f"Slides export with {theme} failed: {error}")

        raise ExportError(f"Slides export failed. Last error: {last_error}")

    def export_epub(self, content: str, filename: str, title: Optional[str] = None) -> Path:
        """Export as EPUB ebook."""
        md_file = self.temp_dir / f"{filename}.md"
        md_file.write_text(content, encoding="utf-8")

        output = self.temp_dir / f"{filename}.epub"
        title = title or filename

        args = [
            str(md_file),
            "-o",
            str(output),
            "--metadata",
            f"title={title}",
            "--highlight-style=tango",
            "--toc",
        ]

        success, error = self._run_pandoc(args)
        if not success:
            raise ExportError(f"EPUB export failed: {error}")

        return output

    def cleanup(self):
        """Remove temporary files."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir: {e}")


class ExportError(Exception):
    """Export operation failed."""

    pass


def export_content(
    content: str, format: str, filename: str = "export", title: Optional[str] = None
) -> Tuple[Path, str]:
    """
    Export content to specified format.

    Returns:
        Tuple of (output_path, media_type)
    """
    exporter = DocumentExporter()

    media_types = {
        "md": "text/markdown",
        "html": "text/html",
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "epub": "application/epub+zip",
        "slides": "application/pdf",
    }

    format = format.lower()

    if format not in media_types:
        raise ExportError(f"Unsupported format: {format}. Supported: {list(media_types.keys())}")

    try:
        if format == "md":
            output = exporter.export_markdown(content, filename)
        elif format == "html":
            output = exporter.export_html(content, filename)
        elif format == "pdf":
            output = exporter.export_pdf(content, filename, title)
        elif format == "docx":
            output = exporter.export_docx(content, filename, title)
        elif format == "epub":
            output = exporter.export_epub(content, filename, title)
        elif format == "slides":
            output = exporter.export_slides(content, filename, title)

        return output, media_types[format]

    except ExportError:
        raise
    except Exception as e:
        raise ExportError(f"Export failed: {str(e)}")
