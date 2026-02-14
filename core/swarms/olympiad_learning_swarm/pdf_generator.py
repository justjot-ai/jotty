"""Olympiad Learning Swarm - PDF & HTML Generation.

Uses WeasyPrint (HTML->PDF) for professional, styled output.
Generates both interactive HTML and print-ready PDF.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# HTML TEMPLATE
# =============================================================================

_LESSON_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title} - {student_name}</title>
<style>
/* Print/PDF styling */
@page {{
    size: A4;
    margin: 2cm 2.2cm;
    @bottom-center {{ content: counter(page) " / " counter(pages); font-size: 9pt; color: #888; }}
}}

/* Screen styling - A4 page preview */
@media screen {{
    body {{
        background: #e0e0e0;
        padding: 20px 0;
    }}
    .page-container {{
        width: 210mm;  /* A4 width */
        min-height: 297mm;  /* A4 height */
        margin: 0 auto 20px;
        padding: 2cm 2.2cm;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
}}

/* Print styling - remove page containers */
@media print {{
    body {{
        background: white;
        padding: 0;
    }}
    .page-container {{
        margin: 0;
        padding: 0;
        box-shadow: none;
        page-break-after: always;
    }}
    .page-container:last-child {{
        page-break-after: auto;
    }}
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
.subtitle {{
    font-size: 14pt;
    color: #555;
    margin-top: -10px;
    margin-bottom: 20px;
}}
h2 {{
    font-size: 18pt;
    color: #283593;
    border-bottom: 2px solid #c5cae9;
    padding-bottom: 5px;
    margin-top: 30px;
    page-break-after: avoid;
    page-break-before: auto;
}}
h3 {{
    font-size: 14pt;
    color: #3949ab;
    margin-top: 20px;
    page-break-after: avoid;
}}
.hook-box {{
    background: linear-gradient(135deg, #e8eaf6, #c5cae9);
    border-left: 5px solid #1a237e;
    padding: 15px 20px;
    margin: 20px 0;
    border-radius: 0 8px 8px 0;
    font-size: 13pt;
}}
.breakthrough {{
    background: linear-gradient(135deg, #fff8e1, #ffecb3);
    border-left: 5px solid #f57f17;
    padding: 12px 18px;
    margin: 15px 0;
    border-radius: 0 8px 8px 0;
    font-weight: bold;
    font-size: 12pt;
}}
.breakthrough::before {{
    content: "{celebration} ";
}}
.pattern-card {{
    background: #f3e5f5;
    border: 1px solid #ce93d8;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    page-break-inside: avoid;
}}
.pattern-card strong {{ color: #6a1b9a; }}
.strategy-card {{
    background: #e8f5e9;
    border: 1px solid #81c784;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    page-break-inside: avoid;
}}
.strategy-card strong {{ color: #2e7d32; }}
.problem-box {{
    background: #fff;
    border: 2px solid #42a5f5;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    page-break-inside: avoid;
}}
.problem-box.foundation {{ border-color: #66bb6a; }}
.problem-box.intermediate {{ border-color: #42a5f5; }}
.problem-box.advanced {{ border-color: #ff7043; }}
.problem-box.olympiad {{ border-color: #ab47bc; }}
.problem-header {{
    font-weight: bold;
    font-size: 11pt;
    margin-bottom: 8px;
    padding: 4px 10px;
    border-radius: 4px;
    display: inline-block;
}}
.problem-header.foundation {{ background: #e8f5e9; color: #2e7d32; }}
.problem-header.intermediate {{ background: #e3f2fd; color: #1565c0; }}
.problem-header.advanced {{ background: #fbe9e7; color: #d84315; }}
.problem-header.olympiad {{ background: #f3e5f5; color: #7b1fa2; }}
.hint-box {{
    background: #fffde7;
    border-left: 3px solid #fbc02d;
    padding: 8px 14px;
    margin: 8px 0;
    font-size: 11pt;
    page-break-inside: avoid;
}}
.solution-box {{
    background: #f1f8e9;
    border-left: 3px solid #7cb342;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 11pt;
    page-break-inside: avoid;
}}
.trap-box {{
    background: #fce4ec;
    border-left: 5px solid #e53935;
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
    page-break-inside: avoid;
}}
.trap-box strong {{ color: #c62828; }}
.connection-box {{
    background: #e0f7fa;
    border: 1px solid #4dd0e1;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 10px 0;
    page-break-inside: avoid;
}}
.next-topics {{
    background: #e8eaf6;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 20px 0;
}}
.next-topics ol {{ margin: 8px 0; padding-left: 20px; }}
.stats-bar {{
    width: 100%;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 15px 0;
    font-size: 11pt;
    border-collapse: separate;
}}
.stat {{
    display: inline-block;
    text-align: center;
    min-width: 18%;
    margin-right: 2%;
    vertical-align: top;
}}
.stat-value {{
    display: block;
    font-size: 18pt;
    font-weight: bold;
    color: #1a237e;
    margin-bottom: 4px;
}}
.stat-label {{
    display: block;
    font-size: 9pt;
    color: #888;
    text-transform: uppercase;
    font-weight: normal;
}}
code {{
    background: #f5f5f5;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 11pt;
}}
pre {{
    background: #263238;
    color: #eeffff;
    padding: 14px 18px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 10pt;
    line-height: 1.5;
}}
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
.pause-and-think::before {{
    content: "Pause and think: ";
    font-weight: bold;
    font-style: normal;
}}
.think-before-reveal {{
    background: #f3e5f5;
    border: 2px dashed #9c27b0;
    padding: 12px 18px;
    margin: 15px 0;
    border-radius: 8px;
    font-size: 11pt;
    color: #4a148c;
    page-break-inside: avoid;
}}
.transition-text {{
    font-style: italic;
    color: #555;
    margin: 20px 0 5px 0;
    font-size: 12pt;
}}
.rank-tips-section {{
    page-break-before: always;
}}
.rank-tip {{
    background: #fffde7;
    border-left: 4px solid #f9a825;
    padding: 8px 14px;
    margin: 6px 0;
    border-radius: 0 6px 6px 0;
    font-size: 11pt;
    page-break-inside: avoid;
}}
.rank-tip strong {{
    color: #e65100;
}}
details {{
    margin: 8px 0;
}}
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
<div class="subtitle">Personalized for {student_name}</div>

<div class="stats-bar">
    <div class="stat"><div class="stat-value">{concepts_count}</div><div class="stat-label">Concepts</div></div>
    <div class="stat"><div class="stat-value">{problems_count}</div><div class="stat-label">Problems</div></div>
    <div class="stat"><div class="stat-value">{patterns_count}</div><div class="stat-label">Patterns</div></div>
    <div class="stat"><div class="stat-value">{strategies_count}</div><div class="stat-label">Strategies</div></div>
    <div class="stat"><div class="stat-value">{learning_time}</div><div class="stat-label">Est. Time</div></div>
</div>

{body_html}

</div><!-- .page-container -->
</body>
</html>"""


# =============================================================================
# MARKDOWN TO HTML CONVERTER
# =============================================================================

class LessonHTMLRenderer:
    """Renders LessonContent into styled HTML."""

    def __init__(self, celebration_word: str = 'Brilliant!') -> None:
        self.celebration = celebration_word

    # Titles of sections rendered by dedicated structured blocks
    _DEDICATED_SECTION_TITLES = frozenset({
        "foundation check", "pattern library", "common traps",
        "strategy toolkit", "trap alert", "where this leads",
    })

    def _is_dedicated_section(self, title: str) -> bool:
        """Check if a section title is rendered by a dedicated structured block."""
        return title.lower().strip() in self._DEDICATED_SECTION_TITLES

    def render(self, content: 'LessonContent', learning_time: str = "") -> str:
        """Render full lesson to HTML string."""
        import markdown

        body_parts = []

        # Hook
        if content.sections and content.sections[0].content:
            body_parts.append(f'<div class="hook-box">{self._md(content.sections[0].content)}</div>')

        # Building blocks (structured rendering with headings + quick checks)
        if content.building_blocks:
            body_parts.append("<h2>Foundation Check</h2>")
            for block in content.building_blocks:
                body_parts.append(f"<h3>{block.name}</h3>")
                body_parts.append(f"<p>{self._md_inline(block.quick_review)}</p>")
                if block.check_question:
                    body_parts.append(f'<div class="hint-box"><strong>Quick check:</strong> {block.check_question}</div>')

        # Core sections (skip hook and sections that have dedicated renderers)
        for section in content.sections[1:]:
            # Skip sections already rendered by dedicated structured blocks
            if self._is_dedicated_section(section.title):
                continue

            if section.problems:
                # Problem section - render structured problems only (skip raw content)
                if section.transition_text:
                    body_parts.append(f'<p class="transition-text">{self._md_inline(section.transition_text)}</p>')
                body_parts.append(f"<h2>{section.title}</h2>")
                for i, prob in enumerate(section.problems, 1):
                    tier_class = prob.tier.value if hasattr(prob.tier, 'value') else 'intermediate'
                    body_parts.append(f'<div class="problem-box {tier_class}">')
                    body_parts.append(f'<div class="problem-header {tier_class}">Problem {i} ({prob.time_estimate_minutes} min)</div>')
                    body_parts.append(f"<p>{self._md_inline(prob.statement)}</p>")
                    if prob.hints:
                        body_parts.append('<div class="hint-box"><strong>Hints:</strong><ol>')
                        for hint in prob.hints:
                            body_parts.append(f"<li>{hint}</li>")
                        body_parts.append("</ol></div>")
                    if prob.solution:
                        body_parts.append(f'<div class="solution-box"><strong>Solution:</strong><br>{self._md(prob.solution)}</div>')
                    if prob.relates_to_pattern:
                        body_parts.append(f'<p><em>Pattern:</em> {prob.relates_to_pattern}</p>')
                    if prob.key_insight:
                        body_parts.append(f'<div class="breakthrough">{prob.key_insight}</div>')
                    body_parts.append("</div>")
            else:
                if section.transition_text:
                    body_parts.append(f'<p class="transition-text">{self._md_inline(section.transition_text)}</p>')
                body_parts.append(f"<h2>{section.title}</h2>")
                body_parts.append(self._md(section.content))
                if section.has_breakthrough_moment:
                    breakthrough_text = section.breakthrough_content if section.breakthrough_content else "Key insight from this section!"
                    body_parts.append(f'<div class="breakthrough">{self._md_inline(breakthrough_text)}</div>')

        # Patterns (structured card rendering)
        if content.patterns:
            body_parts.append("<h2>Pattern Library</h2>")
            for p in content.patterns:
                body_parts.append('<div class="pattern-card">')
                body_parts.append(f"<strong>{p.name}</strong>")
                body_parts.append(f"<p>{p.description}</p>")
                if p.when_to_use:
                    body_parts.append(f"<p><em>When to use:</em> {p.when_to_use}</p>")
                if p.example_trigger:
                    body_parts.append(f"<p><em>Look for:</em> {p.example_trigger}</p>")
                body_parts.append("</div>")

        # Strategies (structured card rendering)
        if content.strategies:
            body_parts.append("<h2>Strategy Toolkit</h2>")
            for s in content.strategies:
                body_parts.append('<div class="strategy-card">')
                body_parts.append(f"<strong>{s.name}</strong>")
                body_parts.append(f"<p>{s.description}</p>")
                if s.steps:
                    body_parts.append("<ol>")
                    for step in s.steps:
                        body_parts.append(f"<li>{step}</li>")
                    body_parts.append("</ol>")
                body_parts.append("</div>")

        # Mistakes (structured trap-box rendering)
        if content.mistakes:
            body_parts.append("<h2>Trap Alert!</h2>")
            for m in content.mistakes:
                body_parts.append('<div class="trap-box">')
                body_parts.append(f"<strong>{m.description}</strong>")
                body_parts.append(f"<p><em>Why:</em> {m.why_it_happens}</p>")
                body_parts.append(f"<p><em>Wrong:</em> {m.example_wrong}</p>")
                body_parts.append(f"<p><em>Correct:</em> {m.example_correct}</p>")
                body_parts.append(f"<p><em>How to avoid:</em> {m.how_to_avoid}</p>")
                body_parts.append("</div>")

        # Connections
        if content.connections:
            body_parts.append("<h2>Where This Leads</h2>")
            body_parts.append('<div class="connection-box"><ul>')
            for c in content.connections:
                body_parts.append(f"<li>{c}</li>")
            body_parts.append("</ul></div>")

        # Key insights summary
        if content.key_insights and any(content.key_insights):
            body_parts.append("<h2>Key Insights</h2>")
            body_parts.append("<ol>")
            for ins in content.key_insights:
                if ins:
                    body_parts.append(f'<li class="breakthrough" style="list-style: decimal;">{self._md_inline(ins)}</li>')
            body_parts.append("</ol>")

        # Rank tips (20-30 tips to secure #1)
        if content.rank_tips:
            body_parts.append('<div class="rank-tips-section">')
            body_parts.append(f"<h2>Tips to Secure #1 Rank in {content.topic}</h2>")
            body_parts.append(f"<p><em>{content.student_name}, here are the specific tips that separate the #1 scorer from everyone else:</em></p>")
            for i, tip in enumerate(content.rank_tips, 1):
                import re as _re
                clean_tip = _re.sub(r'^\d+[\.\)]\s*', '', tip)
                body_parts.append(f'<div class="rank-tip"><strong>{i}.</strong> {self._md_inline(clean_tip)}</div>')
            body_parts.append(f"<p><em>Master these {len(content.rank_tips)} tips and you won't just pass — you'll dominate, {content.student_name}!</em></p>")
            body_parts.append("</div>")

        # Next topics
        if content.next_topics:
            body_parts.append('<div class="next-topics">')
            body_parts.append(f"<h2>What's Next, {content.student_name}?</h2>")
            body_parts.append("<ol>")
            for t in content.next_topics[:6]:
                body_parts.append(f"<li>{t}</li>")
            body_parts.append("</ol></div>")

        body_html = "\n".join(body_parts)

        return _LESSON_HTML_TEMPLATE.format(
            title=f"{content.topic}",
            student_name=content.student_name,
            celebration=self.celebration,
            concepts_count=len(content.core_concepts),
            problems_count=len(content.problems),
            patterns_count=len(content.patterns),
            strategies_count=len(content.strategies),
            learning_time=learning_time or "~45 min",
            body_html=body_html,
        )

    def _md(self, text: str) -> str:
        """Convert markdown text to HTML (block-level, may produce <p>, <ul>, etc.)."""
        if not text:
            return ""
        try:
            import markdown
            return markdown.markdown(text, extensions=['tables', 'fenced_code'])
        except Exception:
            # Fallback: basic conversion
            text = text.replace('\n\n', '</p><p>')
            text = text.replace('\n', '<br>')
            return f"<p>{text}</p>"

    def _md_inline(self, text: str) -> str:
        """Convert markdown to HTML, stripping outer <p> wrapper to avoid nesting.

        Use this when the result will be placed inside an existing <p> or
        other inline context to prevent <p><p>...</p></p> nesting.
        """
        html = self._md(text)
        # Strip single wrapping <p>...</p> if present
        stripped = html.strip()
        if stripped.startswith("<p>") and stripped.endswith("</p>"):
            inner = stripped[3:-4]
            # Only strip if there's no nested <p> (single paragraph)
            if "<p>" not in inner:
                return inner
        return html


# =============================================================================
# PDF GENERATOR
# =============================================================================

async def generate_lesson_pdf(
    content: 'LessonContent',
    output_path: str,
    celebration_word: str = "Brilliant!",
    learning_time: str = "~45 min",
) -> Optional[str]:
    """Generate a styled PDF from LessonContent using WeasyPrint.

    Args:
        content: LessonContent dataclass with all lesson data
        output_path: Where to save the PDF
        celebration_word: Celebration word for breakthroughs
        learning_time: Estimated learning time string

    Returns:
        Path to generated PDF, or None on failure.
    """
    try:
        from weasyprint import HTML

        renderer = LessonHTMLRenderer(celebration_word=celebration_word)
        html_str = renderer.render(content, learning_time=learning_time)

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


async def generate_lesson_html(
    content: 'LessonContent',
    output_path: str,
    celebration_word: str = "Brilliant!",
    learning_time: str = "~45 min",
) -> Optional[str]:
    """Generate interactive HTML file from LessonContent.

    Args:
        content: LessonContent dataclass
        output_path: Where to save the HTML file
        celebration_word: Celebration word
        learning_time: Estimated time

    Returns:
        Path to generated HTML, or None on failure.
    """
    try:
        renderer = LessonHTMLRenderer(celebration_word=celebration_word)
        html_str = renderer.render(content, learning_time=learning_time)

        Path(output_path).write_text(html_str, encoding='utf-8')

        if Path(output_path).exists():
            logger.info(f"Generated HTML: {output_path}")
            return output_path
        return None

    except Exception as e:
        logger.error(f"HTML generation failed: {e}")
        return None


async def _generate_pdf_reportlab(
    content: 'LessonContent',
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
            highlight_style = ParagraphStyle('Highlight', parent=body_style,
                                              backColor=HexColor('#fff8e1'),
                                              borderPadding=8)

            story = []
            story.append(Paragraph(f"{content.topic}", title_style))
            story.append(Paragraph(f"Personalized for {content.student_name}", styles['Normal']))
            story.append(Spacer(1, 20))

            for section in content.sections:
                story.append(Paragraph(section.title, h2_style))
                story.append(Spacer(1, 6))

                # Clean text for reportlab (no markdown)
                text = section.content.replace('\n\n', '<br/><br/>').replace('\n', '<br/>')
                text = text.replace('**', '<b>').replace('*', '<i>')
                try:
                    story.append(Paragraph(text, body_style))
                except Exception:
                    story.append(Paragraph(section.content[:500], body_style))
                story.append(Spacer(1, 12))

                if section.has_breakthrough_moment:
                    story.append(Paragraph(f"{celebration_word} Key insight!", highlight_style))
                    story.append(Spacer(1, 8))

                for i, prob in enumerate(section.problems, 1):
                    story.append(Paragraph(f"Problem {i}: {prob.statement}", body_style))
                    if prob.solution:
                        story.append(Paragraph(f"Solution: {prob.solution[:300]}", body_style))
                    story.append(Spacer(1, 8))

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
    'generate_lesson_pdf',
    'generate_lesson_html',
    'LessonHTMLRenderer',
]
