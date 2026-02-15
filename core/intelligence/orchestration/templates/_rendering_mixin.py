"""
Rendering mixin for ProfessionalMLReport.

Handles HTML template generation, Markdown building, Pandoc conversion,
and fallback PDF generation. Extracted from ml_report_generator.py
to reduce file size.
"""

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RenderingMixin:
    """Mixin providing rendering/output methods for ProfessionalMLReport.

    Methods in this mixin handle document assembly and PDF/HTML output generation.
    They expect to be mixed into ProfessionalMLReport which provides:
    - self.sections: list of markdown sections
    - self.figures: dict of figure paths
    - self.theme: str theme name
    - self.output_dir: str output directory
    """

    def _get_html_template(self) -> str:
        """Get base HTML template with inline CSS, sidebar nav, collapsible sections, print styles."""
        t = self.theme
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{title}}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root {{
    --primary: {t['primary']};
    --secondary: {t['secondary']};
    --accent: {t['accent']};
    --success: {t['success']};
    --warning: {t['warning']};
    --danger: {t['danger']};
    --text: {t['text']};
    --muted: {t['muted']};
    --bg: {t['background']};
    --table-header: {t['table_header']};
    --table-alt: {t['table_alt']};
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: {'"Georgia", "Times New Roman", serif' if self.theme_name == 'goldman' else '"Segoe UI", "Helvetica Neue", Arial, sans-serif'};
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
}}
.sidebar {{
    position: fixed;
    left: 0;
    top: 0;
    width: 260px;
    height: 100vh;
    background: var(--primary);
    color: white;
    overflow-y: auto;
    padding: 20px 0;
    z-index: 100;
}}
.sidebar h2 {{
    padding: 0 20px;
    margin-bottom: 15px;
    font-size: 16px;
    font-weight: 600;
    opacity: 0.9;
}}
.sidebar a {{
    display: block;
    padding: 8px 20px;
    color: rgba(255,255,255,0.8);
    text-decoration: none;
    font-size: 13px;
    transition: all 0.2s;
    border-left: 3px solid transparent;
}}
.sidebar a:hover, .sidebar a.active {{
    background: rgba(255,255,255,0.1);
    color: white;
    border-left-color: var(--accent);
}}
.main-content {{
    margin-left: 260px;
    padding: 40px;
    max-width: 1200px;
}}
.section {{
    background: white;
    border-radius: 8px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid #eee;
}}
.section-header {{
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.section-header h2 {{
    color: var(--primary);
    font-size: 22px;
    margin: 0;
}}
.section-header .toggle {{
    font-size: 18px;
    color: var(--muted);
    transition: transform 0.3s;
}}
.section-header .toggle.collapsed {{
    transform: rotate(-90deg);
}}
.section-body {{ padding-top: 15px; }}
.section-body.hidden {{ display: none; }}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 14px;
}}
th {{
    background: var(--table-header);
    color: white;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
}}
td {{
    padding: 8px 12px;
    border-bottom: 1px solid #eee;
}}
tr:nth-child(even) {{ background: var(--table-alt); }}
tr:hover {{ background: rgba(0,0,0,0.03); }}
.chart-container {{ margin: 20px 0; }}
.chart-container img {{ max-width: 100%; height: auto; border-radius: 4px; }}
.plotly-chart {{ width: 100%; min-height: 400px; }}
.metric-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin: 15px 0;
}}
.metric-card {{
    background: white;
    border: 1px solid #eee;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    border-top: 3px solid var(--accent);
}}
.metric-card .value {{
    font-size: 28px;
    font-weight: 700;
    color: var(--primary);
}}
.metric-card .label {{
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}}
blockquote {{
    border-left: 4px solid var(--accent);
    padding: 10px 15px;
    margin: 15px 0;
    background: rgba(49,130,206,0.05);
    border-radius: 0 4px 4px 0;
}}
.status-ok {{ color: var(--success); font-weight: 600; }}
.status-warn {{ color: var(--warning); font-weight: 600; }}
.status-alert {{ color: var(--danger); font-weight: 600; }}
.report-header {{
    text-align: center;
    padding: 40px 0;
    border-bottom: 2px solid var(--primary);
    margin-bottom: 30px;
}}
.report-header h1 {{
    font-size: 32px;
    color: var(--primary);
    margin-bottom: 8px;
}}
.report-header .subtitle {{
    font-size: 16px;
    color: var(--muted);
}}
@media print {{
    .sidebar {{ display: none; }}
    .main-content {{ margin-left: 0; padding: 20px; }}
    .section {{ break-inside: avoid; }}
    .section-body.hidden {{ display: block !important; }}
}}
</style>
</head>
<body>
{{sidebar}}
<div class="main-content">
{{header}}
{{content}}
<footer style="text-align:center;padding:30px;color:var(--muted);font-size:12px;">
    Generated by Jotty ML Report Generator | {{date}}
</footer>
</div>
<script>
document.querySelectorAll('.section-header').forEach(header => {{
    header.addEventListener('click', () => {{
        const body = header.nextElementSibling;
        const toggle = header.querySelector('.toggle');
        body.classList.toggle('hidden');
        toggle.classList.toggle('collapsed');
    }});
}});
// Sidebar active tracking
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.sidebar a');
window.addEventListener('scroll', () => {{
    let current = '';
    sections.forEach(section => {{
        const top = section.offsetTop - 100;
        if (window.scrollY >= top) current = section.id;
    }});
    navLinks.forEach(link => {{
        link.classList.toggle('active', link.getAttribute('href') === '#' + current);
    }});
}});
</script>
</body>
</html>"""

    def _build_html_document(self) -> str:
        """Build complete HTML document from section data and markdown content."""

        title = self._metadata.get("title", "ML Analysis Report")
        subtitle = self._metadata.get("subtitle", "")
        date = self._metadata.get("date", datetime.now().strftime("%B %d, %Y"))

        template = self._get_html_template()

        # Build sidebar navigation
        nav_items = []
        sections_html = []

        # Convert markdown content to HTML sections
        for i, content_block in enumerate(self._content):
            section_title = self._extract_section_title(content_block)
            section_id = f"section-{i}"

            nav_items.append(f'<a href="#{section_id}">{section_title}</a>')

            section_html = self._render_markdown_to_html(content_block)
            sections_html.append(
                f"""
<div class="section" id="{section_id}">
    <div class="section-header">
        <h2>{section_title}</h2>
        <span class="toggle">&#9660;</span>
    </div>
    <div class="section-body">
        {section_html}
    </div>
</div>"""
            )

        # Also render any stored section data with Plotly charts
        plotly_scripts = self._generate_plotly_scripts()

        sidebar_html = f"""
<div class="sidebar">
    <h2>{self.theme['header_brand']}</h2>
    {''.join(nav_items)}
</div>"""

        header_html = f"""
<div class="report-header">
    <h1>{title}</h1>
    <div class="subtitle">{subtitle}</div>
    <div class="subtitle">{date}</div>
</div>"""

        content_html = "\n".join(sections_html) + plotly_scripts

        html = template.replace("{title}", title)
        html = html.replace("{sidebar}", sidebar_html)
        html = html.replace("{header}", header_html)
        html = html.replace("{content}", content_html)
        html = html.replace("{date}", date)

        return html

    def _render_markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown content to HTML. Handles tables, images, bold, lists."""
        import re

        html = markdown_content.strip()

        # Remove H1 (already in section header)
        html = re.sub(r"^#\s+.+$", "", html, count=1, flags=re.MULTILINE)

        # H2, H3
        html = re.sub(r"^###\s+(.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^##\s+(.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

        # Italic
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)

        # Code
        html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)

        # Images - convert to embedded or chart container
        def replace_image(match: Any) -> Any:
            alt = match.group(1)
            src = match.group(2)
            full_path = self.output_dir / src
            return f'<div class="chart-container"><img src="{src}" alt="{alt}" loading="lazy"><p style="text-align:center;color:var(--muted);font-size:12px;">{alt}</p></div>'

        html = re.sub(r"!\[(.+?)\]\((.+?)\)", replace_image, html)

        # Tables
        html = self._convert_markdown_tables(html)

        # Blockquotes
        lines = html.split("\n")
        new_lines = []
        in_blockquote = False
        for line in lines:
            if line.strip().startswith(">"):
                if not in_blockquote:
                    new_lines.append("<blockquote>")
                    in_blockquote = True
                new_lines.append(line.strip().lstrip("> "))
            else:
                if in_blockquote:
                    new_lines.append("</blockquote>")
                    in_blockquote = False
                new_lines.append(line)
        if in_blockquote:
            new_lines.append("</blockquote>")
        html = "\n".join(new_lines)

        # Unordered lists
        html = re.sub(r"^- (.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)
        html = re.sub(r"(<li>.*</li>\n?)+", lambda m: f"<ul>{m.group(0)}</ul>", html)

        # Ordered lists
        html = re.sub(r"^\d+\.\s+(.+)$", r"<li>\1</li>", html, flags=re.MULTILINE)

        # Horizontal rules
        html = re.sub(r"^---+$", "<hr>", html, flags=re.MULTILINE)

        # Paragraphs (wrap remaining text)
        html = re.sub(r"\n\n+", "</p><p>", html)

        # Clean up empty paragraphs
        html = re.sub(r"<p>\s*</p>", "", html)

        return f"<div>{html}</div>"

    def _convert_markdown_tables(self, html: str) -> str:
        """Convert markdown tables to HTML tables."""
        import re

        def table_replacer(match: Any) -> Any:
            table_text = match.group(0)
            rows = [r.strip() for r in table_text.strip().split("\n") if r.strip()]

            if len(rows) < 2:
                return table_text

            # Parse header
            header_cells = [c.strip() for c in rows[0].split("|") if c.strip()]

            # Skip separator row
            data_rows = rows[2:] if len(rows) > 2 else []

            table_html = "<table><thead><tr>"
            for cell in header_cells:
                table_html += f"<th>{cell}</th>"
            table_html += "</tr></thead><tbody>"

            for row in data_rows:
                cells = [c.strip() for c in row.split("|") if c.strip()]
                table_html += "<tr>"
                for cell in cells:
                    # Color-code status cells
                    css_class = ""
                    if cell in ("OK", "Excellent", "Good", "PASS", "Homoscedastic"):
                        css_class = ' class="status-ok"'
                    elif cell in ("WARNING", "WARN", "Moderate", "Needs Improvement"):
                        css_class = ' class="status-warn"'
                    elif cell in ("ALERT", "FAIL", "Critical", "Unstable", "Heteroscedastic"):
                        css_class = ' class="status-alert"'
                    table_html += f"<td{css_class}>{cell}</td>"
                table_html += "</tr>"

            table_html += "</tbody></table>"
            return table_html

        # Match markdown table blocks (lines with |)
        pattern = r"(?:^\|.+\|$\n?){2,}"
        html = re.sub(pattern, table_replacer, html, flags=re.MULTILINE)
        return html

    def _build_markdown(self) -> str:
        """Build the full markdown document with theme-aware LaTeX formatting."""

        title = self._metadata.get("title", "ML Analysis Report")
        subtitle = self._metadata.get("subtitle", "")
        author = self._metadata.get("author", "Jotty SwarmMLComprehensive")
        date = self._metadata.get("date", datetime.now().strftime("%B %d, %Y"))
        t = self.theme

        # Convert theme colors to RGB
        primary_rgb = self._hex_to_rgb(t["primary"])
        secondary_rgb = self._hex_to_rgb(t["secondary"])
        accent_rgb = self._hex_to_rgb(t["accent"])
        success_rgb = self._hex_to_rgb(t["success"])
        muted_rgb = self._hex_to_rgb(t["muted"])
        table_alt_rgb = self._hex_to_rgb(t["table_alt"])

        # Theme-specific LaTeX settings
        if self.theme_name == "goldman":
            # Goldman: serif, uppercase headers, minimal lines, cool gray bg
            section_format = "\\\\titleformat{{\\\\section}}{{\\\\Large\\\\color{{Primary}}\\\\scshape}}{{\\\\thesection}}{{1em}}{{}}"
            subsection_format = "\\\\titleformat{{\\\\subsection}}{{\\\\large\\\\color{{Secondary}}}}{{\\\\thesubsection}}{{1em}}{{}}"
            font_pkg = "\\\\usepackage{{charter}}"
            extra_packages = """  - \\\\usepackage{{titlesec}}
  - {section_format}
  - {subsection_format}""".format(
                section_format=section_format, subsection_format=subsection_format
            )
        else:
            # Professional: sans-serif, bold headers
            font_pkg = ""
            extra_packages = ""

        header_brand = t["header_brand"]
        footer_text = t["footer_text"]

        title_page = f"""---
title: "{title}"
subtitle: "{subtitle}"
author: "{author}"
date: "{date}"
geometry: "margin=1in"
fontsize: 11pt
documentclass: article
colorlinks: true
linkcolor: Primary
urlcolor: Primary
toccolor: Primary
toc-depth: 3
numbersections: true
header-includes:
  - \\usepackage{{booktabs}}
  - \\usepackage{{longtable}}
  - \\usepackage{{array}}
  - \\usepackage{{multirow}}
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
  - \\usepackage{{colortbl}}
  - \\usepackage{{graphicx}}
  - \\usepackage{{xcolor}}
  - \\definecolor{{Primary}}{{RGB}}{{{primary_rgb}}}
  - \\definecolor{{Secondary}}{{RGB}}{{{secondary_rgb}}}
  - \\definecolor{{Accent}}{{RGB}}{{{accent_rgb}}}
  - \\definecolor{{Success}}{{RGB}}{{{success_rgb}}}
  - \\definecolor{{Muted}}{{RGB}}{{{muted_rgb}}}
  - \\definecolor{{TableAlt}}{{RGB}}{{{table_alt_rgb}}}
  - \\usepackage{{fancyhdr}}
  - \\pagestyle{{fancy}}
  - \\fancyhf{{}}
  - \\fancyhead[L]{{\\small\\textit{{{title[:35]}}}}}
  - \\fancyhead[R]{{\\small\\thepage}}
  - \\fancyfoot[L]{{\\small\\textcolor{{Muted}}{{{header_brand}}}}}
  - \\fancyfoot[R]{{\\small\\textcolor{{Muted}}{{{footer_text}}}}}
  - \\renewcommand{{\\headrulewidth}}{{0.4pt}}
  - \\renewcommand{{\\footrulewidth}}{{0.2pt}}
  - \\renewcommand{{\\arraystretch}}{{1.3}}
---

\\newpage

"""

        # Table of contents
        toc = """\\tableofcontents
\\newpage

"""

        # Combine all content
        body = "\n".join(self._content)

        # Append Environment & Reproducibility section
        env = self._metadata.get("environment", {})
        if env:
            libs = env.get("libraries", {})
            env_md = f"""
# Environment & Reproducibility

## Report Generator

| Property | Value |
|----------|-------|
| Report Version | {env.get('report_version', 'N/A')} |
| Generated At | {env.get('timestamp', 'N/A')} |

## System

| Property | Value |
|----------|-------|
| Python Version | {env.get('python_version', 'N/A').split(chr(10))[0]} |
| Platform | {env.get('platform', 'N/A')} |
| Machine | {env.get('machine', 'N/A')} |

## Library Versions

| Library | Version |
|---------|---------|
"""
            for lib_name, lib_ver in libs.items():
                env_md += f"| {lib_name} | {lib_ver} |\n"

            env_md += "\n---\n"
            body += env_md

            self._store_section_data(
                "environment",
                "Environment & Reproducibility",
                {
                    "report_version": env.get("report_version"),
                    "python_version": env.get("python_version", "").split("\n")[0],
                    "platform": env.get("platform"),
                    "libraries": libs,
                },
            )

        # Append report health summary if there were failures or warnings
        health = self.get_report_health()
        has_issues = self._failed_sections or self._failed_charts or health["total_warnings"] > 0
        if has_issues:
            health_md = f"""
# Report Health Summary

| Metric | Value |
|--------|-------|
| Total Sections Attempted | {health['total_sections']} |
| Succeeded | {health['succeeded']} |
| Failed Sections | {health['failed']} |
| Failed Charts | {health['total_charts_failed']} |
| Warnings | {health['total_warnings']} |

"""
            if health["failed_sections"]:
                health_md += """## Failed Sections

| Section | Error Type | Error Message |
|---------|-----------|---------------|
"""
                for fs in health["failed_sections"]:
                    health_md += (
                        f"| {fs['section']} | {fs['error_type']} | {fs['error_message'][:80]} |\n"
                    )
                health_md += "\n"

            if health["failed_charts"]:
                health_md += """## Failed Charts

| Chart | Error Type | Error Message |
|-------|-----------|---------------|
"""
                for fc in health["failed_charts"]:
                    health_md += (
                        f"| {fc['chart']} | {fc['error_type']} | {fc['error_message'][:80]} |\n"
                    )
                health_md += "\n"

            if health["warnings"]:
                health_md += """## Warnings

| Component | Message | Error Type |
|-----------|---------|-----------|
"""
                for w in health["warnings"]:
                    err_type = w.get("error_type") or ""
                    health_md += f"| {w['component']} | {w['message'][:80]} | {err_type} |\n"
                health_md += "\n"

            health_md += "---\n"
            body += health_md

        return title_page + toc + body

    def _convert_with_pandoc(self, md_path: Path, pdf_path: Path) -> bool:
        """Convert markdown to PDF using pandoc."""

        if not shutil.which("pandoc"):
            logger.warning("Pandoc not found")
            return False

        # Try pdflatex first (most compatible), then xelatex
        engines = ["pdflatex", "xelatex"]

        for engine in engines:
            # Build command with proper resource path
            cmd = [
                "pandoc",
                str(md_path.name),  # Just filename, we'll cd to the directory
                "-o",
                str(pdf_path.name),
                f"--pdf-engine={engine}",
                "--toc-depth=3",
                "--highlight-style=tango",
                "-V",
                "geometry:margin=1in",
            ]

            try:
                # Run from the output directory so relative paths work
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minutes timeout
                    cwd=str(self.output_dir),  # Run from output directory
                )

                if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                    logger.info(f"PDF generated with {engine}")
                    return True
                else:
                    # Log error for debugging
                    if result.stderr:
                        logger.debug(f"Pandoc {engine} stderr: {result.stderr[:500]}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Pandoc with {engine} timed out")
            except Exception as e:
                logger.warning(f"Pandoc with {engine} failed: {e}")

        return False

    def _fallback_pdf_generation(self, markdown: str, pdf_path: Path) -> Optional[str]:
        """Enhanced fallback PDF generation using reportlab with TOC, tables, images, and styling."""
        try:
            import re

            from reportlab.lib import colors as rl_colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import BaseDocTemplate, Frame
            from reportlab.platypus import Image as RLImage
            from reportlab.platypus import (
                PageBreak,
                PageTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
            )
            from reportlab.platypus.tableofcontents import TableOfContents

            t = self.theme
            primary_color = rl_colors.HexColor(t["primary"])
            accent_color = rl_colors.HexColor(t["accent"])
            text_color = rl_colors.HexColor(t["text"])
            muted_color = rl_colors.HexColor(t["muted"])
            table_header_color = rl_colors.HexColor(t["table_header"])
            table_alt_color = rl_colors.HexColor(t["table_alt"])

            # Custom styles
            styles = getSampleStyleSheet()
            styles.add(
                ParagraphStyle(
                    "ReportTitle",
                    parent=styles["Title"],
                    fontSize=24,
                    textColor=primary_color,
                    spaceAfter=20,
                    alignment=TA_CENTER,
                )
            )
            styles.add(
                ParagraphStyle(
                    "ReportH1",
                    parent=styles["Heading1"],
                    fontSize=18,
                    textColor=primary_color,
                    spaceBefore=24,
                    spaceAfter=10,
                )
            )
            styles.add(
                ParagraphStyle(
                    "ReportH2",
                    parent=styles["Heading2"],
                    fontSize=14,
                    textColor=accent_color,
                    spaceBefore=16,
                    spaceAfter=8,
                )
            )
            styles.add(
                ParagraphStyle(
                    "ReportH3",
                    parent=styles["Heading3"],
                    fontSize=12,
                    textColor=text_color,
                    spaceBefore=12,
                    spaceAfter=6,
                )
            )
            styles.add(
                ParagraphStyle(
                    "ReportBody",
                    parent=styles["Normal"],
                    fontSize=10,
                    textColor=text_color,
                    spaceBefore=3,
                    spaceAfter=6,
                    leading=14,
                )
            )
            styles.add(
                ParagraphStyle(
                    "ReportBullet",
                    parent=styles["Normal"],
                    fontSize=10,
                    textColor=text_color,
                    leftIndent=20,
                    bulletIndent=10,
                    spaceBefore=2,
                    spaceAfter=2,
                )
            )
            styles.add(
                ParagraphStyle(
                    "FooterStyle",
                    parent=styles["Normal"],
                    fontSize=8,
                    textColor=muted_color,
                    alignment=TA_CENTER,
                )
            )

            # Track headings for TOC
            heading_entries = []

            def _on_page(canvas: Any, doc_obj: Any) -> Any:
                """Add page number and brand to footer."""
                canvas.saveState()
                page_num = canvas.getPageNumber()
                footer_text = f"{t.get('footer_text', 'ML Report')}  |  Page {page_num}"
                canvas.setFont("Helvetica", 8)
                canvas.setFillColor(muted_color)
                canvas.drawCentredString(letter[0] / 2, 30, footer_text)
                canvas.restoreState()

            # Build document
            frame = Frame(72, 60, letter[0] - 144, letter[1] - 132, id="main")
            page_template = PageTemplate("main", frames=[frame], onPage=_on_page)
            doc = BaseDocTemplate(str(pdf_path), pagesize=letter, pageTemplates=[page_template])

            story = []

            # Title page
            story.append(Spacer(1, 2 * inch))
            story.append(
                Paragraph(self._metadata.get("title", "ML Analysis Report"), styles["ReportTitle"])
            )
            story.append(Spacer(1, 0.3 * inch))
            subtitle = self._metadata.get("subtitle", "")
            if subtitle:
                story.append(
                    Paragraph(
                        subtitle,
                        ParagraphStyle(
                            "SubTitle",
                            parent=styles["Normal"],
                            fontSize=13,
                            textColor=accent_color,
                            alignment=TA_CENTER,
                        ),
                    )
                )
            story.append(Spacer(1, 0.5 * inch))
            date_str = self._metadata.get("date", datetime.now().strftime("%Y-%m-%d"))
            story.append(
                Paragraph(
                    f"Generated: {date_str}",
                    ParagraphStyle(
                        "DateLine",
                        parent=styles["Normal"],
                        fontSize=10,
                        textColor=muted_color,
                        alignment=TA_CENTER,
                    ),
                )
            )
            story.append(PageBreak())

            # Table of Contents
            toc = TableOfContents()
            toc.levelStyles = [
                ParagraphStyle(
                    "TOC1", fontSize=12, leftIndent=20, spaceBefore=5, textColor=primary_color
                ),
                ParagraphStyle(
                    "TOC2", fontSize=10, leftIndent=40, spaceBefore=3, textColor=accent_color
                ),
            ]
            story.append(Paragraph("Table of Contents", styles["ReportH1"]))
            story.append(toc)
            story.append(PageBreak())

            def _render_markdown_inline(text: Any) -> Any:
                """Convert basic markdown inline formatting to reportlab XML."""
                text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
                text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
                text = re.sub(r"`([^`]+?)`", r'<font face="Courier">\1</font>', text)
                text = re.sub(r"<", "&lt;", text)
                # Undo our escaping of our own tags
                text = text.replace("&lt;b>", "<b>").replace("&lt;/b>", "</b>")
                text = text.replace("&lt;i>", "<i>").replace("&lt;/i>", "</i>")
                text = text.replace("&lt;font", "<font").replace("&lt;/font>", "</font>")
                return text

            def _parse_md_table(lines: Any) -> Any:
                """Parse markdown pipe table into list of rows."""
                rows = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("|") and not re.match(r"^\|[\s\-|]+\|$", line):
                        cells = [c.strip() for c in line.strip("|").split("|")]
                        rows.append(cells)
                return rows

            # Process markdown content
            lines = markdown.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                # Headings
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    heading_text = stripped[2:].strip()
                    story.append(Paragraph(heading_text, styles["ReportH1"]))
                    heading_entries.append((0, heading_text, len(story)))
                    i += 1
                    continue
                elif stripped.startswith("## "):
                    heading_text = stripped[3:].strip()
                    story.append(Paragraph(heading_text, styles["ReportH2"]))
                    heading_entries.append((1, heading_text, len(story)))
                    i += 1
                    continue
                elif stripped.startswith("### "):
                    heading_text = stripped[4:].strip()
                    story.append(Paragraph(heading_text, styles["ReportH3"]))
                    i += 1
                    continue

                # Horizontal rule / page break
                elif stripped == "---":
                    story.append(Spacer(1, 12))
                    i += 1
                    continue

                # Images
                elif re.match(r"^!\[.*?\]\((.+?)\)$", stripped):
                    img_match = re.match(r"^!\[.*?\]\((.+?)\)$", stripped)
                    img_path = img_match.group(1)
                    full_img_path = self.output_dir / img_path
                    if full_img_path.exists():
                        try:
                            img = RLImage(str(full_img_path), width=6 * inch, height=4 * inch)
                            img.hAlign = "CENTER"
                            story.append(img)
                            story.append(Spacer(1, 8))
                        except Exception:
                            pass
                    i += 1
                    continue

                # Tables (collect consecutive pipe lines)
                elif stripped.startswith("|"):
                    table_lines = []
                    while i < len(lines) and lines[i].strip().startswith("|"):
                        table_lines.append(lines[i])
                        i += 1
                    rows = _parse_md_table(table_lines)
                    if rows:
                        try:
                            # Convert to Paragraph cells for wrapping
                            table_data = []
                            for r_idx, row in enumerate(rows):
                                table_data.append(
                                    [
                                        Paragraph(
                                            _render_markdown_inline(cell), styles["ReportBody"]
                                        )
                                        for cell in row
                                    ]
                                )

                            col_count = max(len(r) for r in table_data)
                            avail_width = letter[0] - 144
                            col_widths = [avail_width / col_count] * col_count

                            tbl = Table(table_data, colWidths=col_widths)
                            tbl_style = [
                                ("BACKGROUND", (0, 0), (-1, 0), table_header_color),
                                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                                ("FONTSIZE", (0, 0), (-1, -1), 9),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#d0d0d0")),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ]
                            # Alternating row colors
                            for row_idx in range(1, len(table_data)):
                                if row_idx % 2 == 0:
                                    tbl_style.append(
                                        ("BACKGROUND", (0, row_idx), (-1, row_idx), table_alt_color)
                                    )

                            tbl.setStyle(TableStyle(tbl_style))
                            story.append(tbl)
                            story.append(Spacer(1, 8))
                        except Exception:
                            pass
                    continue

                # Bullet lists
                elif stripped.startswith("- ") or stripped.startswith("* "):
                    bullet_text = _render_markdown_inline(stripped[2:])
                    story.append(Paragraph(f"\u2022 {bullet_text}", styles["ReportBullet"]))
                    i += 1
                    continue

                # Numbered lists
                elif re.match(r"^\d+\.\s", stripped):
                    num_match = re.match(r"^(\d+)\.\s(.+)", stripped)
                    if num_match:
                        num = num_match.group(1)
                        text = _render_markdown_inline(num_match.group(2))
                        story.append(Paragraph(f"{num}. {text}", styles["ReportBullet"]))
                    i += 1
                    continue

                # Regular paragraph
                elif stripped:
                    rendered = _render_markdown_inline(stripped)
                    try:
                        story.append(Paragraph(rendered, styles["ReportBody"]))
                    except Exception:
                        pass
                    i += 1
                    continue

                else:
                    i += 1

            # Build with TOC notification
            class _TOCBuilder:
                """Handles TOC heading registration during multiBuild."""

                def __init__(self, toc_obj: Any) -> None:
                    self._toc = toc_obj

                def afterFlowable(self, flowable: Any) -> None:
                    if isinstance(flowable, Paragraph):
                        style_name = flowable.style.name
                        if style_name == "ReportH1":
                            self._toc.addEntry(0, flowable.getPlainText(), 0)
                        elif style_name == "ReportH2":
                            self._toc.addEntry(1, flowable.getPlainText(), 0)

            try:
                # Try multiBuild for TOC
                builder = _TOCBuilder(toc)
                original_afterFlowable = (
                    doc.afterFlowable if hasattr(doc, "afterFlowable") else None
                )
                doc.afterFlowable = builder.afterFlowable
                doc.multiBuild(story)
            except Exception:
                # Simple build fallback
                try:
                    doc2 = BaseDocTemplate(
                        str(pdf_path), pagesize=letter, pageTemplates=[page_template]
                    )
                    doc2.build(story)
                except Exception:
                    pass

            return str(pdf_path)

        except Exception as e:
            logger.error(f"Enhanced fallback PDF generation failed: {e}")
            # Minimal fallback — just dump text
            try:
                import re

                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

                doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
                styles = getSampleStyleSheet()
                story = [
                    Paragraph(self._metadata.get("title", "ML Report"), styles["Title"]),
                    Spacer(1, 12),
                ]

                text = re.sub(r"[#*_`\[\]!]", "", markdown)
                text = re.sub(r"\n{3,}", "\n\n", text)
                for para in text.split("\n\n"):
                    if para.strip():
                        try:
                            story.append(Paragraph(para.strip(), styles["Normal"]))
                            story.append(Spacer(1, 6))
                        except Exception:
                            pass
                doc.build(story)
                return str(pdf_path)
            except Exception as e2:
                logger.error(f"Minimal fallback PDF also failed: {e2}")
                return None
