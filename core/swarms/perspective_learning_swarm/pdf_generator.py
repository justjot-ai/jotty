"""Perspective Learning Swarm - PDF & HTML Generation.

Uses WeasyPrint (HTML->PDF) for professional, styled output.
Generates both interactive HTML and print-ready PDF.

Each perspective gets a distinct visual treatment with unique colors and icons.
Language sections use appropriate fonts (Noto Sans Devanagari, Noto Sans Kannada).
"""

import logging
import asyncio
import re
from typing import Optional
from pathlib import Path

from .types import (
    PerspectiveType, Language, LessonContent, PerspectiveSection,
    LanguageContent, PERSPECTIVE_LABELS, LANGUAGE_LABELS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# COLOR AND STYLE CONFIGURATION
# =============================================================================

PERSPECTIVE_COLORS = {
    PerspectiveType.INTUITIVE_VISUAL: {"primary": "#1565C0", "bg": "#E3F2FD", "border": "#42A5F5", "label": "See It Clearly"},
    PerspectiveType.STRUCTURED_FRAMEWORK: {"primary": "#2E7D32", "bg": "#E8F5E9", "border": "#66BB6A", "label": "Think It Through"},
    PerspectiveType.STORYTELLING: {"primary": "#6A1B9A", "bg": "#F3E5F5", "border": "#CE93D8", "label": "Feel the Story"},
    PerspectiveType.DEBATE_CRITICAL: {"primary": "#C62828", "bg": "#FFEBEE", "border": "#EF5350", "label": "Debate It"},
    PerspectiveType.HANDS_ON_PROJECT: {"primary": "#E65100", "bg": "#FFF3E0", "border": "#FF9800", "label": "Build It"},
    PerspectiveType.REAL_WORLD_APPLICATION: {"primary": "#00695C", "bg": "#E0F2F1", "border": "#4DB6AC", "label": "Live It"},
}

LANGUAGE_COLORS = {
    Language.HINDI: {"primary": "#E65100", "bg": "#FFF8E1", "border": "#FFB74D"},
    Language.KANNADA: {"primary": "#1565C0", "bg": "#E8EAF6", "border": "#7986CB"},
    Language.FRENCH: {"primary": "#283593", "bg": "#E8F5E9", "border": "#81C784"},
}


# =============================================================================
# HTML TEMPLATE
# =============================================================================

_PERSPECTIVE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title} - {student_name}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&family=Noto+Sans+Kannada:wght@400;700&display=swap');

@page {{
    size: A4;
    margin: 2cm 2.2cm;
    @bottom-center {{ content: counter(page) " / " counter(pages); font-size: 9pt; color: #888; }}
}}

@media screen {{
    body {{ background: #e0e0e0; padding: 20px 0; }}
    .page-container {{
        width: 210mm;
        min-height: 297mm;
        margin: 0 auto 20px;
        padding: 2cm 2.2cm;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
}}

@media print {{
    body {{ background: white; padding: 0; }}
    .page-container {{ margin: 0; padding: 0; box-shadow: none; page-break-after: always; }}
    .page-container:last-child {{ page-break-after: auto; }}
    .collapsible-content {{ display: block !important; }}
}}

* {{ box-sizing: border-box; }}
body {{
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 12pt;
    line-height: 1.65;
    color: #222;
    margin: 0;
    padding: 0;
}}
h1 {{
    font-size: 26pt;
    color: #1a237e;
    border-bottom: 3px solid #1a237e;
    padding-bottom: 10px;
    margin-top: 0;
    page-break-after: avoid;
}}
.subtitle {{ font-size: 14pt; color: #555; margin-top: -10px; margin-bottom: 5px; }}
.central-idea {{
    font-size: 13pt;
    font-style: italic;
    color: #333;
    margin-bottom: 20px;
    padding: 10px 16px;
    background: #f5f5f5;
    border-left: 4px solid #1a237e;
    border-radius: 0 8px 8px 0;
}}
h2 {{
    font-size: 18pt;
    border-bottom: 2px solid #c5cae9;
    padding-bottom: 5px;
    margin-top: 30px;
    page-break-after: avoid;
}}
h3 {{
    font-size: 14pt;
    margin-top: 20px;
    page-break-after: avoid;
}}

/* Stats bar */
.stats-bar {{
    width: 100%;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 15px 0;
    font-size: 11pt;
}}
.stat {{
    display: inline-block;
    text-align: center;
    min-width: 15%;
    margin-right: 2%;
    vertical-align: top;
}}
.stat-value {{ display: block; font-size: 18pt; font-weight: bold; color: #1a237e; margin-bottom: 4px; }}
.stat-label {{ display: block; font-size: 9pt; color: #888; text-transform: uppercase; }}

/* Table of contents */
.toc {{
    background: #f5f5f5;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 20px 0;
}}
.toc ol {{ padding-left: 20px; }}
.toc li {{ margin-bottom: 4px; font-size: 11pt; }}

/* Perspective cards */
.perspective-section {{
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    page-break-inside: avoid;
}}
.perspective-header {{
    font-size: 16pt;
    font-weight: bold;
    margin-bottom: 12px;
    padding-bottom: 8px;
}}

/* Specific perspective colors */
.perspective-intuitive {{ background: #E3F2FD; border-left: 6px solid #1565C0; }}
.perspective-intuitive .perspective-header {{ color: #1565C0; border-bottom: 2px solid #42A5F5; }}

.perspective-framework {{ background: #E8F5E9; border-left: 6px solid #2E7D32; }}
.perspective-framework .perspective-header {{ color: #2E7D32; border-bottom: 2px solid #66BB6A; }}

.perspective-story {{ background: #F3E5F5; border-left: 6px solid #6A1B9A; }}
.perspective-story .perspective-header {{ color: #6A1B9A; border-bottom: 2px solid #CE93D8; }}

.perspective-debate {{ background: #FFEBEE; border-left: 6px solid #C62828; }}
.perspective-debate .perspective-header {{ color: #C62828; border-bottom: 2px solid #EF5350; }}

.perspective-project {{ background: #FFF3E0; border-left: 6px solid #E65100; }}
.perspective-project .perspective-header {{ color: #E65100; border-bottom: 2px solid #FF9800; }}

.perspective-realworld {{ background: #E0F2F1; border-left: 6px solid #00695C; }}
.perspective-realworld .perspective-header {{ color: #00695C; border-bottom: 2px solid #4DB6AC; }}

/* Language sections */
.language-section {{
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    page-break-before: auto;
}}
.language-hindi {{
    background: #FFF8E1;
    border-left: 6px solid #E65100;
    font-family: 'Noto Sans Devanagari', 'Georgia', serif;
}}
.language-kannada {{
    background: #E8EAF6;
    border-left: 6px solid #1565C0;
    font-family: 'Noto Sans Kannada', 'Georgia', serif;
}}
.language-french {{
    background: #E8F5E9;
    border-left: 6px solid #283593;
}}

.language-header {{
    font-size: 16pt;
    font-weight: bold;
    margin-bottom: 12px;
    padding-bottom: 8px;
}}

/* Breakthrough moments */
.breakthrough {{
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border-left: 5px solid #f57f17;
    padding: 12px 18px;
    margin: 15px 0;
    border-radius: 0 8px 8px 0;
    font-weight: bold;
    font-size: 12pt;
}}
.breakthrough::before {{ content: "{celebration} "; }}

/* Activity boxes */
.activity-box {{
    background: #fff;
    border: 2px solid #FF9800;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    page-break-inside: avoid;
}}
.activity-box strong {{ color: #E65100; }}

/* Debate boxes */
.debate-for {{
    background: #E8F5E9;
    border-left: 4px solid #4CAF50;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
}}
.debate-against {{
    background: #FFEBEE;
    border-left: 4px solid #F44336;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
}}

/* Framework boxes */
.framework-card {{
    background: #fff;
    border: 2px solid #66BB6A;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    page-break-inside: avoid;
}}
.framework-card strong {{ color: #2E7D32; }}

/* Vocabulary */
.vocab-chip {{
    display: inline-block;
    background: #E3F2FD;
    color: #1565C0;
    padding: 4px 10px;
    border-radius: 16px;
    margin: 3px;
    font-size: 10pt;
}}

/* Slogan */
.slogan {{
    font-size: 13pt;
    font-weight: bold;
    font-style: italic;
    color: #E65100;
    padding: 8px 0;
}}

/* Parent guide */
.parent-guide {{
    background: #FFF8E1;
    border: 2px solid #FFB74D;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
    page-break-before: always;
}}
.parent-guide h2 {{ color: #E65100; border-bottom-color: #FFB74D; }}

/* Reflection */
.reflection-box {{
    background: #E8EAF6;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 20px 0;
}}

/* Socratic questions */
.pause-and-think {{
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    border-left: 5px solid #1565c0;
    padding: 12px 18px;
    margin: 15px 0;
    border-radius: 0 8px 8px 0;
    font-style: italic;
    font-size: 12pt;
    color: #0d47a1;
    page-break-inside: avoid;
}}

/* Collapsible sections (HTML interactive) */
details {{ margin: 8px 0; }}
summary {{
    cursor: pointer;
    font-weight: bold;
    color: #1565c0;
    font-size: 11pt;
}}

ul, ol {{ margin: 6px 0; padding-left: 22px; }}
li {{ margin-bottom: 4px; }}
</style>
</head>
<body>
<div class="page-container">

<h1>{title}</h1>
<div class="subtitle">A Multi-Perspective Learning Journey for {student_name}</div>
<div class="central-idea">{central_idea}</div>

<div class="stats-bar">
    <div class="stat"><div class="stat-value">{perspectives_count}</div><div class="stat-label">Perspectives</div></div>
    <div class="stat"><div class="stat-value">{languages_count}</div><div class="stat-label">Languages</div></div>
    <div class="stat"><div class="stat-value">{concepts_count}</div><div class="stat-label">Concepts</div></div>
    <div class="stat"><div class="stat-value">{vocab_count}</div><div class="stat-label">Vocabulary</div></div>
    <div class="stat"><div class="stat-value">{word_count}</div><div class="stat-label">Words</div></div>
</div>

{body_html}

</div>
</body>
</html>"""


# =============================================================================
# HTML RENDERER
# =============================================================================

class PerspectiveHTMLRenderer:
    """Renders PerspectiveLearning LessonContent into styled HTML."""

    def __init__(self, celebration_word: str = "Wonderful!"):
        self.celebration = celebration_word

    def render(self, content: LessonContent) -> str:
        """Render full lesson to HTML string."""
        body_parts = []

        # Table of contents
        body_parts.append(self._render_toc(content))

        # Why This Matters section
        body_parts.append('<h2>Why This Matters</h2>')
        if content.running_example:
            body_parts.append(f'<div class="breakthrough" style="border-left-color: #1a237e;">{self._md(content.running_example)}</div>')
        if content.learning_objectives:
            body_parts.append('<h3>Learning Objectives</h3><ol>')
            for obj in content.learning_objectives:
                body_parts.append(f'<li>{self._md_inline(obj)}</li>')
            body_parts.append('</ol>')

        # Perspective sections
        perspective_classes = {
            PerspectiveType.INTUITIVE_VISUAL: "perspective-intuitive",
            PerspectiveType.STRUCTURED_FRAMEWORK: "perspective-framework",
            PerspectiveType.STORYTELLING: "perspective-story",
            PerspectiveType.DEBATE_CRITICAL: "perspective-debate",
            PerspectiveType.HANDS_ON_PROJECT: "perspective-project",
            PerspectiveType.REAL_WORLD_APPLICATION: "perspective-realworld",
        }

        for section in content.perspectives:
            css_class = perspective_classes.get(section.perspective, "perspective-intuitive")
            label = PERSPECTIVE_LABELS.get(section.perspective, section.title)
            body_parts.append(f'<div class="perspective-section {css_class}">')
            body_parts.append(f'<div class="perspective-header">{label}</div>')
            body_parts.append(self._md(section.content))
            if section.key_takeaway:
                body_parts.append(f'<div class="breakthrough">{self._md_inline(section.key_takeaway)}</div>')
            if section.activity:
                body_parts.append(f'<div class="activity-box"><strong>Activity:</strong> {self._md_inline(section.activity)}</div>')
            body_parts.append('</div>')

        # Language sections
        language_classes = {
            Language.HINDI: "language-hindi",
            Language.KANNADA: "language-kannada",
            Language.FRENCH: "language-french",
        }

        for lang_section in content.language_sections:
            if lang_section.language == Language.ENGLISH:
                continue
            css_class = language_classes.get(lang_section.language, "")
            lang_label = LANGUAGE_LABELS.get(lang_section.language, lang_section.language.value.title())
            body_parts.append(f'<div class="language-section {css_class}">')
            body_parts.append(f'<div class="language-header">{lang_label} \u2014 {content.topic}</div>')
            if lang_section.summary:
                body_parts.append(self._md(lang_section.summary))
            if lang_section.key_vocabulary:
                body_parts.append('<p>')
                for term in lang_section.key_vocabulary:
                    body_parts.append(f'<span class="vocab-chip">{term}</span>')
                body_parts.append('</p>')
            if lang_section.reflection_prompts:
                body_parts.append('<h3>Reflection</h3><ul>')
                for prompt in lang_section.reflection_prompts:
                    body_parts.append(f'<li>{prompt}</li>')
                body_parts.append('</ul>')
            if lang_section.activity:
                body_parts.append(f'<div class="activity-box"><strong>Activity:</strong> {self._md_inline(lang_section.activity)}</div>')
            if lang_section.slogans:
                for slogan in lang_section.slogans:
                    body_parts.append(f'<p class="slogan">\u201c{slogan}\u201d</p>')
            body_parts.append('</div>')

        # Parent's Guide
        if content.parent_guide:
            body_parts.append('<div class="parent-guide">')
            body_parts.append("<h2>Parent's Guide</h2>")
            body_parts.append(self._md(content.parent_guide))
            body_parts.append('</div>')

        # Key Insights & Reflection
        if content.key_insights and any(content.key_insights):
            body_parts.append('<div class="reflection-box">')
            body_parts.append('<h2>Key Insights & Reflection</h2>')
            body_parts.append('<ol>')
            for ins in content.key_insights:
                if ins:
                    body_parts.append(f'<li>{self._md_inline(ins)}</li>')
            body_parts.append('</ol>')
            if content.socratic_questions:
                body_parts.append(f'<h3>Questions to Keep Thinking About, {content.student_name}</h3><ol>')
                for q in content.socratic_questions:
                    body_parts.append(f'<li>{self._md_inline(q)}</li>')
                body_parts.append('</ol>')
            body_parts.append('</div>')

        body_html = "\n".join(body_parts)

        return _PERSPECTIVE_HTML_TEMPLATE.format(
            title=content.topic,
            student_name=content.student_name,
            central_idea=content.central_idea or content.topic,
            celebration=self.celebration,
            perspectives_count=len(content.perspectives),
            languages_count=len(content.language_sections),
            concepts_count=len(content.key_concepts),
            vocab_count=len(content.vocabulary),
            word_count=content.total_words,
            body_html=body_html,
        )

    def _render_toc(self, content: LessonContent) -> str:
        """Render table of contents."""
        items = ['<div class="toc"><h3>Table of Contents</h3><ol>']
        items.append('<li>Why This Matters</li>')
        for section in content.perspectives:
            label = PERSPECTIVE_LABELS.get(section.perspective, section.title)
            items.append(f'<li>{label}</li>')
        for lang_section in content.language_sections:
            if lang_section.language != Language.ENGLISH:
                lang_label = LANGUAGE_LABELS.get(lang_section.language, lang_section.language.value.title())
                items.append(f'<li>{lang_label}</li>')
        items.append("<li>Parent's Guide</li>")
        items.append('<li>Key Insights & Reflection</li>')
        items.append('</ol></div>')
        return '\n'.join(items)

    def _md(self, text: str) -> str:
        """Convert markdown text to HTML."""
        if not text:
            return ""
        try:
            import markdown
            return markdown.markdown(text, extensions=['tables', 'fenced_code'])
        except Exception:
            text = text.replace('\n\n', '</p><p>')
            text = text.replace('\n', '<br>')
            return f"<p>{text}</p>"

    def _md_inline(self, text: str) -> str:
        """Convert markdown to HTML, stripping outer <p> wrapper."""
        html = self._md(text)
        stripped = html.strip()
        if stripped.startswith("<p>") and stripped.endswith("</p>"):
            inner = stripped[3:-4]
            if "<p>" not in inner:
                return inner
        return html


# =============================================================================
# PDF GENERATOR
# =============================================================================

async def generate_perspective_pdf(
    content: LessonContent,
    output_path: str,
    celebration_word: str = "Wonderful!",
) -> Optional[str]:
    """Generate a styled PDF from LessonContent using WeasyPrint.

    Returns:
        Path to generated PDF, or None on failure.
    """
    try:
        from weasyprint import HTML

        renderer = PerspectiveHTMLRenderer(celebration_word=celebration_word)
        html_str = renderer.render(content)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: HTML(string=html_str).write_pdf(output_path)
        )

        if Path(output_path).exists():
            size_kb = Path(output_path).stat().st_size / 1024
            logger.info(f"Generated PDF: {output_path} ({size_kb:.0f} KB)")
            return output_path
        return None

    except ImportError:
        logger.warning("WeasyPrint not installed, trying reportlab fallback")
        return await _generate_pdf_reportlab(content, output_path, celebration_word)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None


async def generate_perspective_html(
    content: LessonContent,
    output_path: str,
    celebration_word: str = "Wonderful!",
) -> Optional[str]:
    """Generate interactive HTML file from LessonContent.

    Returns:
        Path to generated HTML, or None on failure.
    """
    try:
        renderer = PerspectiveHTMLRenderer(celebration_word=celebration_word)
        html_str = renderer.render(content)

        Path(output_path).write_text(html_str, encoding='utf-8')

        if Path(output_path).exists():
            logger.info(f"Generated HTML: {output_path}")
            return output_path
        return None

    except Exception as e:
        logger.error(f"HTML generation failed: {e}")
        return None


async def _generate_pdf_reportlab(
    content: LessonContent,
    output_path: str,
    celebration_word: str,
) -> Optional[str]:
    """Fallback PDF generator using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

        loop = asyncio.get_running_loop()

        def _build() -> None:
            doc = SimpleDocTemplate(output_path, pagesize=A4,
                                    leftMargin=60, rightMargin=60,
                                    topMargin=50, bottomMargin=50)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('LessonTitle', parent=styles['Title'],
                                         fontSize=22, textColor=HexColor('#1a237e'))
            h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                                       fontSize=16, textColor=HexColor('#283593'))
            body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                         fontSize=11, leading=16)

            story = []
            story.append(Paragraph(content.topic, title_style))
            story.append(Paragraph(f"A Multi-Perspective Learning Journey for {content.student_name}", styles['Normal']))
            story.append(Spacer(1, 20))

            for section in content.perspectives:
                label = PERSPECTIVE_LABELS.get(section.perspective, section.title)
                story.append(Paragraph(label, h2_style))
                story.append(Spacer(1, 6))
                text = section.content.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
                try:
                    story.append(Paragraph(text[:2000], body_style))
                except Exception:
                    story.append(Paragraph(section.content[:500], body_style))
                story.append(Spacer(1, 12))

            doc.build(story)

        await loop.run_in_executor(None, _build)

        if Path(output_path).exists():
            logger.info(f"Generated PDF (reportlab): {output_path}")
            return output_path
        return None

    except Exception as e:
        logger.error(f"Reportlab PDF generation also failed: {e}")
        return None


__all__ = [
    'generate_perspective_pdf',
    'generate_perspective_html',
    'PerspectiveHTMLRenderer',
]
