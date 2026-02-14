from typing import Any
"""
Kids Book Templates
===================

Fun, colorful templates for children's books:
- Storybook: Colorful, large fonts, illustration-friendly
- Educational: Fun facts, quizzes, engaging layouts
- Activity Book: Interactive elements, puzzles, games
"""

from .base_template import BaseTemplate, TemplateColors, TemplateTypography, TemplateLayout, TemplateRegistry


class StorybookTemplate(BaseTemplate):
    """Storybook template - Colorful, whimsical, large fonts for children."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Storybook"
        self.description = "Colorful storybook design with large fonts and playful elements"
        self.category = "kids"

        self.colors = TemplateColors(
            primary="#7c3aed",        # Purple
            primary_dark="#5b21b6",
            primary_light="#a78bfa",
            secondary="#06b6d4",      # Cyan
            accent="#f59e0b",         # Amber
            success="#10b981",        # Green
            warning="#f59e0b",
            danger="#ef4444",
            text="#1f2937",
            text_light="#6b7280",
            background="#fefce8",     # Warm cream
            background_alt="#fef3c7",
            border="#fcd34d",
        )

        self.typography = TemplateTypography(
            heading_font="'Comic Sans MS', 'Chalkboard', cursive",
            body_font="'Georgia', serif",
            base_size="14pt",
            h1_size="36pt",
            h2_size="24pt",
            h3_size="18pt",
            line_height="1.8",
        )

        self.layout = TemplateLayout(
            page_size="A4",
            margin_top="2.5cm",
            margin_bottom="2.5cm",
            margin_left="2cm",
            margin_right="2cm",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Storybook Style */
@page {{
    background: linear-gradient(180deg, #fefce8 0%, #fef3c7 100%);
}}

body {{
    background: linear-gradient(180deg, {self.colors.background} 0%, {self.colors.background_alt} 100%);
}}

h1 {{
    font-family: {self.typography.heading_font};
    font-size: {self.typography.h1_size};
    color: {self.colors.primary};
    text-align: center;
    text-shadow: 3pt 3pt 0 {self.colors.accent};
    margin: 24pt 0;
}}

h2 {{
    font-family: {self.typography.heading_font};
    font-size: {self.typography.h2_size};
    color: {self.colors.secondary};
    text-align: center;
    margin: 20pt 0 16pt 0;
    page-break-after: avoid;
}}

h3 {{
    font-family: {self.typography.heading_font};
    font-size: {self.typography.h3_size};
    color: {self.colors.primary_light};
    margin: 16pt 0 12pt 0;
}}

p {{
    font-size: {self.typography.base_size};
    line-height: 2;
    text-align: justify;
    margin: 12pt 0;
}}

/* Story Box */
.story-box {{
    background: white;
    border: 4px solid {self.colors.accent};
    border-radius: 20pt;
    padding: 20pt;
    margin: 16pt 0;
    box-shadow: 8pt 8pt 0 {self.colors.primary_light};
}}

/* Character Box */
.character-box {{
    background: linear-gradient(135deg, {self.colors.primary_light} 0%, {self.colors.secondary} 100%);
    color: white;
    border-radius: 50%;
    width: 100pt;
    height: 100pt;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 48pt;
    margin: 12pt auto;
}}

/* Fun Fact Box */
.fun-fact {{
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 3px dashed {self.colors.accent};
    border-radius: 16pt;
    padding: 16pt;
    margin: 16pt 0;
    text-align: center;
}}

.fun-fact::before {{
    content: " Fun Fact! ";
    display: block;
    font-family: {self.typography.heading_font};
    font-size: 14pt;
    color: {self.colors.primary};
    margin-bottom: 8pt;
}}

/* Page Number */
.page-number {{
    text-align: center;
    font-family: {self.typography.heading_font};
    font-size: 18pt;
    color: {self.colors.primary};
    margin-top: 20pt;
}}

/* Illustration Placeholder */
.illustration {{
    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    border: 3px solid {self.colors.secondary};
    border-radius: 12pt;
    min-height: 200pt;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 20pt 0;
    font-family: {self.typography.heading_font};
    color: {self.colors.secondary};
}}

/* Decorative Elements */
.stars {{
    text-align: center;
    font-size: 24pt;
    color: {self.colors.accent};
    margin: 12pt 0;
}}

/* Image styling */
img {{
    border-radius: 12pt;
    border: 4px solid {self.colors.accent};
    box-shadow: 6pt 6pt 0 {self.colors.primary_light};
}}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>
<div class="stars"> </div>
{content}
<div class="stars"> The End </div>
</body>
</html>"""


class EducationalTemplate(BaseTemplate):
    """Educational template - Fun facts, quizzes, learning-focused."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Educational"
        self.description = "Learning-focused design with quizzes and fun facts"
        self.category = "kids"

        self.colors = TemplateColors(
            primary="#2563eb",        # Blue
            primary_dark="#1d4ed8",
            primary_light="#60a5fa",
            secondary="#16a34a",      # Green
            accent="#eab308",         # Yellow
            success="#16a34a",
            warning="#f59e0b",
            danger="#dc2626",
            text="#1e293b",
            text_light="#64748b",
            background="#f0f9ff",     # Light blue
            background_alt="#e0f2fe",
            border="#93c5fd",
        )

        self.typography = TemplateTypography(
            heading_font="'Verdana', sans-serif",
            body_font="'Arial', sans-serif",
            base_size="12pt",
            h1_size="28pt",
            h2_size="20pt",
            h3_size="14pt",
            line_height="1.7",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Educational Style */
body {{
    background: {self.colors.background};
}}

h1 {{
    font-size: {self.typography.h1_size};
    color: {self.colors.primary};
    text-align: center;
    background: linear-gradient(90deg, {self.colors.primary_light} 0%, {self.colors.secondary} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 20pt 0;
}}

h2 {{
    font-size: {self.typography.h2_size};
    color: white;
    background: {self.colors.primary};
    padding: 10pt 16pt;
    border-radius: 8pt;
    margin: 16pt 0 12pt 0;
    page-break-after: avoid;
}}

h3 {{
    font-size: {self.typography.h3_size};
    color: {self.colors.primary};
    border-left: 4px solid {self.colors.secondary};
    padding-left: 10pt;
    margin: 14pt 0 8pt 0;
}}

/* Learning Box */
.learning-box {{
    background: white;
    border: 2px solid {self.colors.primary_light};
    border-radius: 10pt;
    padding: 14pt;
    margin: 14pt 0;
    box-shadow: 0 4pt 8pt rgba(37, 99, 235, 0.1);
}}

/* Did You Know Box */
.did-you-know {{
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 5px solid {self.colors.accent};
    padding: 14pt;
    margin: 14pt 0;
    border-radius: 0 10pt 10pt 0;
}}

.did-you-know::before {{
    content: " Did You Know?";
    display: block;
    font-weight: 700;
    color: {self.colors.primary_dark};
    margin-bottom: 6pt;
}}

/* Quiz Box */
.quiz-box {{
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border: 2px solid {self.colors.secondary};
    border-radius: 10pt;
    padding: 14pt;
    margin: 14pt 0;
}}

.quiz-box::before {{
    content: " Quiz Time!";
    display: block;
    font-weight: 700;
    font-size: 14pt;
    color: {self.colors.secondary};
    margin-bottom: 10pt;
}}

/* Answer Options */
.option {{
    background: white;
    border: 2px solid {self.colors.border};
    border-radius: 6pt;
    padding: 8pt 12pt;
    margin: 6pt 0;
    cursor: pointer;
}}

.option:hover {{
    border-color: {self.colors.primary};
    background: {self.colors.background_alt};
}}

/* Key Points */
.key-point {{
    background: {self.colors.primary_light};
    color: white;
    padding: 4pt 10pt;
    border-radius: 20pt;
    display: inline-block;
    margin: 4pt;
    font-size: 10pt;
}}

/* Summary Box */
.summary {{
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_dark} 100%);
    color: white;
    border-radius: 10pt;
    padding: 16pt;
    margin: 16pt 0;
}}

.summary h3 {{
    color: white;
    border-left-color: {self.colors.accent};
}}

/* Tables */
th {{
    background: {self.colors.primary};
}}

/* Progress indicator */
.progress {{
    background: {self.colors.background_alt};
    border-radius: 10pt;
    height: 10pt;
    overflow: hidden;
}}

.progress-bar {{
    background: linear-gradient(90deg, {self.colors.secondary} 0%, {self.colors.primary_light} 100%);
    height: 100%;
}}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>{content}</body>
</html>"""


class ActivityBookTemplate(BaseTemplate):
    """Activity Book template - Interactive, puzzles, games."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Activity Book"
        self.description = "Interactive design with puzzles, games, and activities"
        self.category = "kids"

        self.colors = TemplateColors(
            primary="#e11d48",        # Rose
            primary_dark="#be123c",
            primary_light="#fb7185",
            secondary="#8b5cf6",      # Violet
            accent="#14b8a6",         # Teal
            success="#22c55e",
            warning="#f59e0b",
            danger="#ef4444",
            text="#1f2937",
            text_light="#6b7280",
            background="#fdf2f8",     # Pink tint
            background_alt="#fce7f3",
            border="#f9a8d4",
        )

        self.typography = TemplateTypography(
            heading_font="'Arial Rounded MT Bold', 'Helvetica Rounded', sans-serif",
            body_font="'Arial', sans-serif",
            base_size="12pt",
            h1_size="32pt",
            h2_size="22pt",
            h3_size="16pt",
            line_height="1.6",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Activity Book Style */
body {{
    background: {self.colors.background};
}}

h1 {{
    font-size: {self.typography.h1_size};
    color: {self.colors.primary};
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2pt;
    margin: 20pt 0;
}}

h2 {{
    font-size: {self.typography.h2_size};
    color: white;
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.secondary} 100%);
    padding: 12pt 16pt;
    border-radius: 0 30pt 30pt 0;
    margin: 16pt 0 12pt -20pt;
    padding-left: 36pt;
    page-break-after: avoid;
}}

h3 {{
    font-size: {self.typography.h3_size};
    color: {self.colors.secondary};
    margin: 14pt 0 8pt 0;
}}

/* Activity Box */
.activity-box {{
    background: white;
    border: 3px dashed {self.colors.primary};
    border-radius: 16pt;
    padding: 16pt;
    margin: 16pt 0;
    page-break-inside: avoid;
}}

.activity-box::before {{
    content: " Activity";
    display: inline-block;
    background: {self.colors.primary};
    color: white;
    padding: 4pt 12pt;
    border-radius: 12pt;
    font-weight: 700;
    margin-bottom: 10pt;
}}

/* Puzzle Box */
.puzzle-box {{
    background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
    border: 2px solid {self.colors.secondary};
    border-radius: 12pt;
    padding: 16pt;
    margin: 16pt 0;
}}

.puzzle-box::before {{
    content: " Puzzle";
    display: inline-block;
    background: {self.colors.secondary};
    color: white;
    padding: 4pt 12pt;
    border-radius: 12pt;
    font-weight: 700;
    margin-bottom: 10pt;
}}

/* Coloring Area */
.coloring-area {{
    background: white;
    border: 4px solid {self.colors.border};
    border-radius: 12pt;
    min-height: 150pt;
    margin: 14pt 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: {self.colors.text_light};
}}

/* Drawing Area */
.drawing-area {{
    background: white;
    border: 2px solid {self.colors.text_light};
    border-radius: 8pt;
    min-height: 200pt;
    margin: 14pt 0;
}}

/* Writing Lines */
.writing-lines {{
    background: repeating-linear-gradient(
        transparent,
        transparent 24pt,
        {self.colors.border} 24pt,
        {self.colors.border} 25pt
    );
    min-height: 100pt;
    margin: 14pt 0;
    padding: 4pt;
}}

/* Maze placeholder */
.maze {{
    background: {self.colors.background_alt};
    border: 2px solid {self.colors.primary};
    border-radius: 8pt;
    min-height: 200pt;
    margin: 14pt 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24pt;
}}

/* Sticker area */
.sticker-area {{
    border: 3px dotted {self.colors.accent};
    border-radius: 50%;
    width: 80pt;
    height: 80pt;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin: 8pt;
    background: white;
}}

/* Reward Stars */
.reward-stars {{
    text-align: center;
    font-size: 30pt;
    margin: 16pt 0;
}}

/* Instructions Box */
.instructions {{
    background: linear-gradient(135deg, #ccfbf1 0%, #99f6e4 100%);
    border-left: 5px solid {self.colors.accent};
    padding: 12pt;
    margin: 12pt 0;
    border-radius: 0 8pt 8pt 0;
}}

.instructions::before {{
    content: " Instructions:";
    display: block;
    font-weight: 700;
    color: {self.colors.accent};
    margin-bottom: 6pt;
}}

/* Checkbox List */
.checkbox-item {{
    display: flex;
    align-items: center;
    margin: 8pt 0;
}}

.checkbox {{
    width: 20pt;
    height: 20pt;
    border: 2px solid {self.colors.primary};
    border-radius: 4pt;
    margin-right: 10pt;
}}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>
{content}
<div class="reward-stars"> Great Job! </div>
</body>
</html>"""


# Register templates
TemplateRegistry.register(StorybookTemplate())
TemplateRegistry.register(EducationalTemplate())
TemplateRegistry.register(ActivityBookTemplate())
