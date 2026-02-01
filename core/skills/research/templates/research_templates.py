"""
Research Report Templates
=========================

Professional templates inspired by global investment banks:
- Goldman Sachs: Clean, minimal, navy blue
- Morgan Stanley: Modern, gradient headers, blue accent
- CLSA: Data-rich, detailed tables, Asian focus
- Motilal Oswal: Colorful, Indian market, orange accent
"""

from .base_template import BaseTemplate, TemplateColors, TemplateTypography, TemplateRegistry


class GoldmanSachsTemplate(BaseTemplate):
    """Goldman Sachs inspired template - Clean, minimal, professional."""

    def __init__(self):
        super().__init__()
        self.name = "Goldman Sachs"
        self.description = "Clean, minimal design with navy blue accents"
        self.category = "research"

        self.colors = TemplateColors(
            primary="#10294a",        # GS Navy
            primary_dark="#091830",
            primary_light="#1a4075",
            secondary="#6b7c93",
            accent="#7399c6",
            success="#2e7d32",
            warning="#f9a825",
            danger="#c62828",
            text="#1a1a1a",
            text_light="#666666",
            background="#ffffff",
            background_alt="#f5f7fa",
            border="#d0d7de",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Goldman Sachs Style */
@page {{
    @top-right {{
        content: "Goldman Sachs";
        font-family: 'Georgia', serif;
        font-size: 8pt;
        color: {self.colors.primary};
    }}
    @bottom-center {{
        content: counter(page);
        font-size: 8pt;
        color: {self.colors.secondary};
    }}
}}

h1 {{
    font-family: 'Georgia', serif;
    font-size: 28pt;
    font-weight: 400;
    color: {self.colors.primary};
    border-bottom: 2px solid {self.colors.primary};
    padding-bottom: 12pt;
    margin-bottom: 16pt;
}}

h2 {{
    font-family: 'Georgia', serif;
    font-size: 14pt;
    font-weight: 400;
    color: {self.colors.primary};
    text-transform: uppercase;
    letter-spacing: 1pt;
    border-bottom: 1px solid {self.colors.border};
    padding-bottom: 8pt;
    margin: 24pt 0 12pt 0;
    page-break-after: avoid;
}}

h3 {{
    font-size: 11pt;
    font-weight: 600;
    color: {self.colors.primary_dark};
    margin: 16pt 0 8pt 0;
}}

/* Rating Box */
.rating-buy {{ background: #e8f5e9; color: #1b5e20; padding: 8pt 16pt; display: inline-block; font-weight: 600; }}
.rating-hold {{ background: #fff8e1; color: #f57f17; padding: 8pt 16pt; display: inline-block; font-weight: 600; }}
.rating-sell {{ background: #ffebee; color: #b71c1c; padding: 8pt 16pt; display: inline-block; font-weight: 600; }}

/* Scenario Boxes */
.scenario-bull {{
    background: linear-gradient(135deg, #e8f5e9 0%, #ffffff 100%);
    border-left: 4px solid #2e7d32;
    padding: 12pt;
    margin: 12pt 0;
    page-break-inside: avoid;
}}

.scenario-base {{
    background: linear-gradient(135deg, #fff8e1 0%, #ffffff 100%);
    border-left: 4px solid #f9a825;
    padding: 12pt;
    margin: 12pt 0;
    page-break-inside: avoid;
}}

.scenario-bear {{
    background: linear-gradient(135deg, #ffebee 0%, #ffffff 100%);
    border-left: 4px solid #c62828;
    padding: 12pt;
    margin: 12pt 0;
    page-break-inside: avoid;
}}

/* Tables */
th {{
    background: {self.colors.primary};
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 0.5pt;
}}

/* Charts */
.chart-container {{
    text-align: center;
    margin: 16pt 0;
    padding: 12pt;
    background: {self.colors.background_alt};
    border: 1px solid {self.colors.border};
    page-break-inside: avoid;
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


class MorganStanleyTemplate(BaseTemplate):
    """Morgan Stanley inspired template - Modern, gradient headers."""

    def __init__(self):
        super().__init__()
        self.name = "Morgan Stanley"
        self.description = "Modern design with gradient headers and blue accents"
        self.category = "research"

        self.colors = TemplateColors(
            primary="#00205b",        # MS Blue
            primary_dark="#001640",
            primary_light="#003380",
            secondary="#5c6670",
            accent="#0077b6",
            success="#00875a",
            warning="#ff8c00",
            danger="#d32f2f",
            text="#1e2328",
            text_light="#5c6670",
            background="#ffffff",
            background_alt="#f4f6f8",
            border="#dde1e6",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Morgan Stanley Style */
@page {{
    @top-left {{
        content: "MORGAN STANLEY RESEARCH";
        font-size: 7pt;
        color: {self.colors.primary};
        letter-spacing: 1pt;
    }}
    @bottom-center {{
        content: "Page " counter(page) " of " counter(pages);
        font-size: 8pt;
        color: {self.colors.secondary};
    }}
}}

h1 {{
    font-size: 26pt;
    font-weight: 700;
    color: {self.colors.primary};
    margin: 0 0 8pt 0;
}}

h2 {{
    font-size: 14pt;
    font-weight: 600;
    color: white;
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_light} 100%);
    padding: 10pt 14pt;
    margin: 20pt 0 12pt 0;
    border-radius: 4pt;
    page-break-after: avoid;
}}

h3 {{
    font-size: 11pt;
    font-weight: 600;
    color: {self.colors.primary};
    border-left: 3px solid {self.colors.accent};
    padding-left: 10pt;
    margin: 14pt 0 8pt 0;
}}

/* Rating Badge */
.rating-buy {{
    background: linear-gradient(135deg, #00875a 0%, #00a86b 100%);
    color: white;
    padding: 10pt 20pt;
    border-radius: 4pt;
    font-weight: 700;
    font-size: 14pt;
    display: inline-block;
}}

.rating-hold {{
    background: linear-gradient(135deg, #ff8c00 0%, #ffa500 100%);
    color: white;
    padding: 10pt 20pt;
    border-radius: 4pt;
    font-weight: 700;
    font-size: 14pt;
    display: inline-block;
}}

.rating-sell {{
    background: linear-gradient(135deg, #d32f2f 0%, #ef5350 100%);
    color: white;
    padding: 10pt 20pt;
    border-radius: 4pt;
    font-weight: 700;
    font-size: 14pt;
    display: inline-block;
}}

/* Scenario Boxes */
.scenario-bull {{
    background: linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%);
    border: 1px solid #00875a;
    border-left: 4px solid #00875a;
    padding: 14pt;
    margin: 12pt 0;
    border-radius: 0 4pt 4pt 0;
    page-break-inside: avoid;
}}

.scenario-base {{
    background: linear-gradient(135deg, #fff3e0 0%, #ffffff 100%);
    border: 1px solid #ff8c00;
    border-left: 4px solid #ff8c00;
    padding: 14pt;
    margin: 12pt 0;
    border-radius: 0 4pt 4pt 0;
    page-break-inside: avoid;
}}

.scenario-bear {{
    background: linear-gradient(135deg, #ffebee 0%, #ffffff 100%);
    border: 1px solid #d32f2f;
    border-left: 4px solid #d32f2f;
    padding: 14pt;
    margin: 12pt 0;
    border-radius: 0 4pt 4pt 0;
    page-break-inside: avoid;
}}

/* Tables */
th {{
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_light} 100%);
}}

/* Key Metrics Card */
.metrics-card {{
    background: {self.colors.background_alt};
    border: 1px solid {self.colors.border};
    border-radius: 6pt;
    padding: 14pt;
    margin: 12pt 0;
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


class CLSATemplate(BaseTemplate):
    """CLSA inspired template - Data-rich, detailed, Asian market focus."""

    def __init__(self):
        super().__init__()
        self.name = "CLSA"
        self.description = "Data-rich design with detailed tables and red accents"
        self.category = "research"

        self.colors = TemplateColors(
            primary="#b71c1c",        # CLSA Red
            primary_dark="#7f0000",
            primary_light="#e53935",
            secondary="#424242",
            accent="#ff5722",
            success="#388e3c",
            warning="#ffa000",
            danger="#d32f2f",
            text="#212121",
            text_light="#757575",
            background="#ffffff",
            background_alt="#fafafa",
            border="#e0e0e0",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* CLSA Style */
@page {{
    @top-center {{
        content: "CLSA";
        font-size: 10pt;
        font-weight: 700;
        color: {self.colors.primary};
    }}
    @bottom-left {{
        content: "For important disclosures, refer to Disclaimers";
        font-size: 6pt;
        color: {self.colors.text_light};
    }}
    @bottom-right {{
        content: counter(page);
        font-size: 8pt;
    }}
}}

h1 {{
    font-size: 22pt;
    font-weight: 700;
    color: {self.colors.primary};
    border-bottom: 3px solid {self.colors.primary};
    padding-bottom: 8pt;
}}

h2 {{
    font-size: 13pt;
    font-weight: 700;
    color: white;
    background: {self.colors.primary};
    padding: 8pt 12pt;
    margin: 18pt 0 10pt 0;
    page-break-after: avoid;
}}

h3 {{
    font-size: 11pt;
    font-weight: 700;
    color: {self.colors.primary};
    margin: 12pt 0 6pt 0;
}}

/* Rating */
.rating-buy {{ background: #c8e6c9; color: #1b5e20; padding: 6pt 14pt; font-weight: 700; border: 2px solid #388e3c; }}
.rating-hold {{ background: #ffecb3; color: #e65100; padding: 6pt 14pt; font-weight: 700; border: 2px solid #ffa000; }}
.rating-sell {{ background: #ffcdd2; color: #b71c1c; padding: 6pt 14pt; font-weight: 700; border: 2px solid #d32f2f; }}

/* Scenario Boxes */
.scenario-bull {{
    background: #e8f5e9;
    border: 2px solid #388e3c;
    padding: 10pt;
    margin: 10pt 0;
    page-break-inside: avoid;
}}

.scenario-base {{
    background: #fff8e1;
    border: 2px solid #ffa000;
    padding: 10pt;
    margin: 10pt 0;
    page-break-inside: avoid;
}}

.scenario-bear {{
    background: #ffebee;
    border: 2px solid #d32f2f;
    padding: 10pt;
    margin: 10pt 0;
    page-break-inside: avoid;
}}

/* Tables - Data Rich */
table {{
    font-size: 8pt;
}}

th {{
    background: {self.colors.primary};
    font-size: 7pt;
    text-transform: uppercase;
    padding: 4pt 6pt;
}}

td {{
    padding: 4pt 6pt;
}}

/* Compact Charts */
.chart-container {{
    margin: 10pt 0;
    page-break-inside: avoid;
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


class MotilalOswalTemplate(BaseTemplate):
    """Motilal Oswal inspired template - Colorful, Indian market focus."""

    def __init__(self):
        super().__init__()
        self.name = "Motilal Oswal"
        self.description = "Colorful design with orange accents for Indian markets"
        self.category = "research"

        self.colors = TemplateColors(
            primary="#e65100",        # MOSL Orange
            primary_dark="#bf360c",
            primary_light="#ff9800",
            secondary="#37474f",
            accent="#ff6f00",
            success="#00c853",
            warning="#ffab00",
            danger="#ff1744",
            text="#263238",
            text_light="#607d8b",
            background="#ffffff",
            background_alt="#fff8f0",
            border="#ffcc80",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Motilal Oswal Style */
@page {{
    @top-left {{
        content: "Motilal Oswal Financial Services";
        font-size: 8pt;
        color: {self.colors.primary};
        font-weight: 600;
    }}
    @bottom-center {{
        content: counter(page);
        font-size: 9pt;
        color: {self.colors.primary};
        font-weight: 600;
    }}
}}

h1 {{
    font-size: 24pt;
    font-weight: 700;
    color: {self.colors.primary};
    border-left: 6px solid {self.colors.primary};
    padding-left: 12pt;
    margin-bottom: 12pt;
}}

h2 {{
    font-size: 14pt;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, {self.colors.primary} 0%, {self.colors.primary_light} 100%);
    padding: 10pt 14pt;
    margin: 18pt 0 12pt 0;
    border-radius: 0 20pt 20pt 0;
    page-break-after: avoid;
}}

h3 {{
    font-size: 11pt;
    font-weight: 600;
    color: {self.colors.primary};
    border-bottom: 2px solid {self.colors.primary_light};
    padding-bottom: 4pt;
    margin: 14pt 0 8pt 0;
}}

/* Rating Badges */
.rating-buy {{
    background: linear-gradient(135deg, #00c853 0%, #69f0ae 100%);
    color: white;
    padding: 12pt 24pt;
    border-radius: 25pt;
    font-weight: 700;
    font-size: 16pt;
    display: inline-block;
    box-shadow: 0 4pt 12pt rgba(0,200,83,0.3);
}}

.rating-hold {{
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_light} 100%);
    color: white;
    padding: 12pt 24pt;
    border-radius: 25pt;
    font-weight: 700;
    font-size: 16pt;
    display: inline-block;
    box-shadow: 0 4pt 12pt rgba(230,81,0,0.3);
}}

.rating-sell {{
    background: linear-gradient(135deg, #ff1744 0%, #ff5252 100%);
    color: white;
    padding: 12pt 24pt;
    border-radius: 25pt;
    font-weight: 700;
    font-size: 16pt;
    display: inline-block;
    box-shadow: 0 4pt 12pt rgba(255,23,68,0.3);
}}

/* Scenario Boxes - Colorful */
.scenario-bull {{
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border-left: 5px solid #00c853;
    padding: 14pt;
    margin: 12pt 0;
    border-radius: 0 8pt 8pt 0;
    page-break-inside: avoid;
}}

.scenario-base {{
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    border-left: 5px solid {self.colors.primary};
    padding: 14pt;
    margin: 12pt 0;
    border-radius: 0 8pt 8pt 0;
    page-break-inside: avoid;
}}

.scenario-bear {{
    background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
    border-left: 5px solid #ff1744;
    padding: 14pt;
    margin: 12pt 0;
    border-radius: 0 8pt 8pt 0;
    page-break-inside: avoid;
}}

/* Tables */
th {{
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_light} 100%);
    border-radius: 4pt 4pt 0 0;
}}

th:first-child {{
    border-radius: 4pt 0 0 0;
}}

th:last-child {{
    border-radius: 0 4pt 0 0;
}}

/* Charts */
.chart-container {{
    background: {self.colors.background_alt};
    border: 2px solid {self.colors.border};
    border-radius: 8pt;
    padding: 12pt;
    margin: 14pt 0;
    page-break-inside: avoid;
}}

/* Key Metrics */
.metric-highlight {{
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_light} 100%);
    color: white;
    padding: 8pt 14pt;
    border-radius: 6pt;
    font-weight: 700;
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


# Register templates
TemplateRegistry.register(GoldmanSachsTemplate())
TemplateRegistry.register(MorganStanleyTemplate())
TemplateRegistry.register(CLSATemplate())
TemplateRegistry.register(MotilalOswalTemplate())
