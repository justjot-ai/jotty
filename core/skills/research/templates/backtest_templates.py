"""
Backtest Report Templates
=========================

World-class templates for ML backtesting reports inspired by:
- Two Sigma (quantitative, data-driven)
- Renaissance Technologies (mathematical precision)
- AQR Capital (academic rigor)
- Man Group (institutional quality)
- Citadel (professional excellence)
"""

from .base_template import BaseTemplate, TemplateColors, TemplateTypography, TemplateLayout, TemplateRegistry


class TwoSigmaTemplate(BaseTemplate):
    """Two Sigma inspired template - clean, data-focused, modern."""

    def __init__(self):
        super().__init__()
        self.name = "Two Sigma"
        self.description = "Quantitative research style with clean data visualization"
        self.category = "backtest"

        self.colors = TemplateColors(
            primary="#0066CC",          # Two Sigma blue
            primary_dark="#004C99",
            primary_light="#3399FF",
            secondary="#4A5568",
            accent="#00B894",           # Teal for positive
            success="#00B894",
            warning="#FDCB6E",
            danger="#E17055",
            text="#2D3436",
            text_light="#636E72",
            background="#FFFFFF",
            background_alt="#F8F9FA",
            border="#DFE6E9",
        )

        self.typography = TemplateTypography(
            heading_font="'Roboto', 'Helvetica Neue', sans-serif",
            body_font="'Roboto', 'Helvetica Neue', sans-serif",
            mono_font="'JetBrains Mono', 'Fira Code', monospace",
            base_size="10pt",
            h1_size="26pt",
            h2_size="16pt",
            h3_size="12pt",
            line_height="1.5",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Two Sigma Template Styles */

body {{
    background: linear-gradient(180deg, {self.colors.background} 0%, {self.colors.background_alt} 100%);
}}

h1 {{
    color: {self.colors.primary_dark};
    font-size: {self.typography.h1_size};
    font-weight: 700;
    border-bottom: 3px solid {self.colors.primary};
    padding-bottom: 10pt;
    margin-top: 0;
    letter-spacing: -0.5pt;
}}

h2 {{
    color: {self.colors.primary};
    font-size: {self.typography.h2_size};
    font-weight: 600;
    margin-top: 20pt;
    padding-top: 12pt;
    border-top: 1px solid {self.colors.border};
}}

h3 {{
    color: {self.colors.secondary};
    font-size: {self.typography.h3_size};
    font-weight: 600;
    margin-top: 14pt;
}}

/* Metric Cards */
.metric-card {{
    background: {self.colors.background};
    border: 1px solid {self.colors.border};
    border-left: 4px solid {self.colors.primary};
    padding: 12pt;
    margin: 8pt 0;
    border-radius: 4pt;
}}

.metric-value {{
    font-size: 24pt;
    font-weight: 700;
    color: {self.colors.primary_dark};
}}

.metric-label {{
    font-size: 9pt;
    color: {self.colors.text_light};
    text-transform: uppercase;
    letter-spacing: 0.5pt;
}}

/* Tables */
table {{
    border: 1px solid {self.colors.border};
    border-radius: 4pt;
    overflow: hidden;
}}

th {{
    background: {self.colors.primary};
    color: white;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 0.5pt;
}}

tr:hover {{
    background: {self.colors.background_alt};
}}

/* Performance Indicators */
.positive {{
    color: {self.colors.success};
    font-weight: 600;
}}

.negative {{
    color: {self.colors.danger};
    font-weight: 600;
}}

.neutral {{
    color: {self.colors.secondary};
}}

/* Stats Box */
.stats-box {{
    display: inline-block;
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_dark} 100%);
    color: white;
    padding: 16pt;
    margin: 6pt;
    border-radius: 6pt;
    min-width: 120pt;
    text-align: center;
}}

.stats-box .value {{
    font-size: 20pt;
    font-weight: 700;
}}

.stats-box .label {{
    font-size: 8pt;
    opacity: 0.9;
    text-transform: uppercase;
}}

/* Charts */
.chart-container {{
    background: {self.colors.background};
    border: 1px solid {self.colors.border};
    border-radius: 8pt;
    padding: 12pt;
    margin: 16pt 0;
    page-break-inside: avoid;
}}

.chart-container img {{
    width: 100%;
    height: auto;
    border-radius: 4pt;
}}

/* Scenario Boxes */
.scenario-bull {{
    background: linear-gradient(135deg, rgba(0, 184, 148, 0.1) 0%, rgba(0, 184, 148, 0.05) 100%);
    border-left: 4px solid {self.colors.success};
    padding: 12pt;
    margin: 8pt 0;
}}

.scenario-base {{
    background: linear-gradient(135deg, rgba(0, 102, 204, 0.1) 0%, rgba(0, 102, 204, 0.05) 100%);
    border-left: 4px solid {self.colors.primary};
    padding: 12pt;
    margin: 8pt 0;
}}

.scenario-bear {{
    background: linear-gradient(135deg, rgba(225, 112, 85, 0.1) 0%, rgba(225, 112, 85, 0.05) 100%);
    border-left: 4px solid {self.colors.danger};
    padding: 12pt;
    margin: 8pt 0;
}}

/* Footer */
.report-footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 8pt 20pt;
    background: {self.colors.background_alt};
    border-top: 1px solid {self.colors.border};
    font-size: 8pt;
    color: {self.colors.text_light};
}}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>
    {content}
</body>
</html>"""


class RenTechTemplate(BaseTemplate):
    """Renaissance Technologies inspired template - mathematical, precise."""

    def __init__(self):
        super().__init__()
        self.name = "Renaissance"
        self.description = "Mathematical precision with dark theme for data focus"
        self.category = "backtest"

        self.colors = TemplateColors(
            primary="#6C5CE7",          # Purple
            primary_dark="#5849C4",
            primary_light="#A29BFE",
            secondary="#00CEC9",        # Cyan
            accent="#00B894",
            success="#00B894",
            warning="#FDCB6E",
            danger="#FF7675",
            text="#2D3436",
            text_light="#636E72",
            background="#FFFFFF",
            background_alt="#F5F6FA",
            border="#DFE6E9",
        )

        self.typography = TemplateTypography(
            heading_font="'Source Sans Pro', 'Segoe UI', sans-serif",
            body_font="'Source Sans Pro', 'Segoe UI', sans-serif",
            mono_font="'Source Code Pro', monospace",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Renaissance Template Styles */

h1 {{
    color: {self.colors.primary_dark};
    font-size: 28pt;
    font-weight: 700;
    background: linear-gradient(90deg, {self.colors.primary} 0%, {self.colors.secondary} 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 8pt;
    margin-bottom: 16pt;
}}

h2 {{
    color: {self.colors.primary};
    font-size: {self.typography.h2_size};
    font-weight: 600;
    margin-top: 24pt;
    position: relative;
}}

h2::after {{
    content: '';
    position: absolute;
    bottom: -4pt;
    left: 0;
    width: 40pt;
    height: 3pt;
    background: linear-gradient(90deg, {self.colors.primary} 0%, {self.colors.secondary} 100%);
    border-radius: 2pt;
}}

/* Data Tables */
table {{
    border: none;
    border-radius: 8pt;
    overflow: hidden;
    box-shadow: 0 2pt 8pt rgba(0,0,0,0.05);
}}

th {{
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_dark} 100%);
    color: white;
    font-weight: 600;
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.8pt;
    padding: 10pt 12pt;
}}

td {{
    padding: 10pt 12pt;
    border-bottom: 1px solid {self.colors.border};
}}

tr:last-child td {{
    border-bottom: none;
}}

/* Metrics Grid */
.metrics-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12pt;
    margin: 16pt 0;
}}

.metric-tile {{
    background: {self.colors.background};
    border-radius: 8pt;
    padding: 16pt;
    text-align: center;
    box-shadow: 0 2pt 8pt rgba(108, 92, 231, 0.1);
    border: 1px solid rgba(108, 92, 231, 0.1);
}}

.metric-tile .value {{
    font-size: 22pt;
    font-weight: 700;
    color: {self.colors.primary};
}}

.metric-tile .label {{
    font-size: 8pt;
    color: {self.colors.text_light};
    text-transform: uppercase;
    margin-top: 4pt;
}}

/* Performance Badge */
.performance-badge {{
    display: inline-block;
    padding: 4pt 12pt;
    border-radius: 20pt;
    font-size: 9pt;
    font-weight: 600;
}}

.performance-badge.positive {{
    background: rgba(0, 184, 148, 0.15);
    color: {self.colors.success};
}}

.performance-badge.negative {{
    background: rgba(255, 118, 117, 0.15);
    color: {self.colors.danger};
}}

/* Charts */
.chart-container {{
    background: {self.colors.background};
    border-radius: 12pt;
    padding: 16pt;
    margin: 20pt 0;
    box-shadow: 0 4pt 12pt rgba(0,0,0,0.05);
}}

/* Code/Data Blocks */
pre, code {{
    background: {self.colors.background_alt};
    border: 1px solid {self.colors.border};
    border-radius: 6pt;
    font-size: 9pt;
}}

/* Highlight Row */
tr.highlight {{
    background: rgba(108, 92, 231, 0.08) !important;
    font-weight: 600;
}}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>
    {content}
</body>
</html>"""


class AQRTemplate(BaseTemplate):
    """AQR Capital inspired template - academic, research-focused."""

    def __init__(self):
        super().__init__()
        self.name = "AQR"
        self.description = "Academic research style with rigorous data presentation"
        self.category = "backtest"

        self.colors = TemplateColors(
            primary="#1E3A5F",          # Navy blue
            primary_dark="#0D2137",
            primary_light="#2C5282",
            secondary="#C9302C",        # AQR red accent
            accent="#38A169",
            success="#38A169",
            warning="#D69E2E",
            danger="#C9302C",
            text="#1A202C",
            text_light="#4A5568",
            background="#FFFFFF",
            background_alt="#F7FAFC",
            border="#E2E8F0",
        )

        self.typography = TemplateTypography(
            heading_font="'Times New Roman', 'Georgia', serif",
            body_font="'Palatino', 'Book Antiqua', serif",
            mono_font="'Courier New', monospace",
            base_size="11pt",
            h1_size="24pt",
            h2_size="14pt",
            h3_size="12pt",
            line_height="1.7",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* AQR Academic Template Styles */

body {{
    font-family: {self.typography.body_font};
    text-align: justify;
}}

h1 {{
    font-family: {self.typography.heading_font};
    color: {self.colors.primary_dark};
    font-size: {self.typography.h1_size};
    font-weight: 700;
    text-align: center;
    border-bottom: 2px double {self.colors.primary};
    padding-bottom: 12pt;
    margin-bottom: 20pt;
}}

h2 {{
    font-family: {self.typography.heading_font};
    color: {self.colors.primary};
    font-size: {self.typography.h2_size};
    font-weight: 700;
    margin-top: 24pt;
    border-bottom: 1px solid {self.colors.border};
    padding-bottom: 6pt;
}}

h3 {{
    font-family: {self.typography.heading_font};
    color: {self.colors.primary_light};
    font-size: {self.typography.h3_size};
    font-weight: 600;
    font-style: italic;
}}

/* Academic Table Style */
table {{
    margin: 16pt auto;
    border: none;
    border-top: 2px solid {self.colors.primary};
    border-bottom: 2px solid {self.colors.primary};
}}

th {{
    background: transparent;
    color: {self.colors.primary_dark};
    font-weight: 700;
    border-bottom: 1px solid {self.colors.primary};
    text-align: center;
    font-size: 9pt;
}}

td {{
    border: none;
    border-bottom: 1px solid {self.colors.border};
    padding: 8pt 12pt;
    text-align: center;
}}

tr:last-child td {{
    border-bottom: none;
}}

/* Equation Box */
.equation {{
    background: {self.colors.background_alt};
    border: 1px solid {self.colors.border};
    padding: 16pt;
    margin: 12pt 40pt;
    text-align: center;
    font-family: {self.typography.mono_font};
    font-size: 11pt;
}}

/* Abstract Box */
.abstract {{
    background: {self.colors.background_alt};
    border-left: 3px solid {self.colors.primary};
    padding: 12pt 16pt;
    margin: 16pt 20pt;
    font-style: italic;
}}

/* Footnotes */
.footnote {{
    font-size: 8pt;
    color: {self.colors.text_light};
    border-top: 1px solid {self.colors.border};
    margin-top: 20pt;
    padding-top: 8pt;
}}

/* Statistics Table */
.stats-table {{
    width: 80%;
    margin: 16pt auto;
}}

.stats-table th {{
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 0.5pt;
}}

/* Significance Stars */
.sig-1 {{ color: {self.colors.secondary}; }}
.sig-5 {{ color: {self.colors.warning}; }}
.sig-10 {{ color: {self.colors.text_light}; }}

/* Charts */
.chart-container {{
    border: 1px solid {self.colors.border};
    padding: 12pt;
    margin: 16pt auto;
    max-width: 90%;
}}

.chart-caption {{
    text-align: center;
    font-style: italic;
    font-size: 9pt;
    color: {self.colors.text_light};
    margin-top: 8pt;
}}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>
    {content}
</body>
</html>"""


class ManGroupTemplate(BaseTemplate):
    """Man Group inspired template - institutional, sophisticated."""

    def __init__(self):
        super().__init__()
        self.name = "Man Group"
        self.description = "Institutional quality with sophisticated design"
        self.category = "backtest"

        self.colors = TemplateColors(
            primary="#003366",          # Man Group navy
            primary_dark="#002244",
            primary_light="#004C99",
            secondary="#CC9900",        # Gold accent
            accent="#009973",
            success="#009973",
            warning="#CC9900",
            danger="#CC3333",
            text="#1A1A2E",
            text_light="#4A4A6A",
            background="#FFFFFF",
            background_alt="#F5F7FA",
            border="#D1D5DB",
        )

        self.typography = TemplateTypography(
            heading_font="'Montserrat', 'Helvetica Neue', sans-serif",
            body_font="'Open Sans', 'Segoe UI', sans-serif",
            mono_font="'Roboto Mono', monospace",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Man Group Institutional Template Styles */

h1 {{
    color: {self.colors.primary_dark};
    font-size: 28pt;
    font-weight: 800;
    letter-spacing: -0.5pt;
    position: relative;
    padding-left: 16pt;
}}

h1::before {{
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 6pt;
    background: linear-gradient(180deg, {self.colors.secondary} 0%, {self.colors.primary} 100%);
}}

h2 {{
    color: {self.colors.primary};
    font-size: 16pt;
    font-weight: 700;
    margin-top: 28pt;
    padding-bottom: 8pt;
    border-bottom: 2px solid {self.colors.secondary};
}}

h3 {{
    color: {self.colors.primary_light};
    font-size: 12pt;
    font-weight: 600;
}}

/* Executive Summary Card */
.exec-summary {{
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_dark} 100%);
    color: white;
    padding: 24pt;
    border-radius: 8pt;
    margin: 16pt 0;
}}

.exec-summary h3 {{
    color: {self.colors.secondary};
    margin-top: 0;
}}

/* KPI Dashboard */
.kpi-row {{
    display: flex;
    justify-content: space-between;
    margin: 16pt 0;
}}

.kpi-card {{
    flex: 1;
    margin: 0 8pt;
    padding: 16pt;
    background: {self.colors.background};
    border: 1px solid {self.colors.border};
    border-top: 4px solid {self.colors.secondary};
    text-align: center;
}}

.kpi-card:first-child {{ margin-left: 0; }}
.kpi-card:last-child {{ margin-right: 0; }}

.kpi-value {{
    font-size: 28pt;
    font-weight: 800;
    color: {self.colors.primary_dark};
}}

.kpi-label {{
    font-size: 9pt;
    color: {self.colors.text_light};
    text-transform: uppercase;
    letter-spacing: 1pt;
}}

/* Tables */
table {{
    border: none;
    box-shadow: 0 2pt 8pt rgba(0, 51, 102, 0.08);
}}

th {{
    background: {self.colors.primary};
    color: white;
    font-weight: 600;
    padding: 12pt;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 0.5pt;
}}

td {{
    padding: 10pt 12pt;
}}

tr:nth-child(even) {{
    background: {self.colors.background_alt};
}}

/* Performance Pills */
.pill {{
    display: inline-block;
    padding: 3pt 10pt;
    border-radius: 12pt;
    font-size: 8pt;
    font-weight: 600;
}}

.pill-success {{
    background: rgba(0, 153, 115, 0.15);
    color: {self.colors.success};
}}

.pill-danger {{
    background: rgba(204, 51, 51, 0.15);
    color: {self.colors.danger};
}}

.pill-warning {{
    background: rgba(204, 153, 0, 0.15);
    color: {self.colors.warning};
}}

/* Charts */
.chart-container {{
    background: {self.colors.background};
    border: 1px solid {self.colors.border};
    border-radius: 8pt;
    padding: 16pt;
    margin: 20pt 0;
    box-shadow: 0 2pt 12pt rgba(0, 51, 102, 0.05);
}}

/* Footer */
@page {{
    @bottom-center {{
        content: "Man Group | Confidential";
        font-size: 8pt;
        color: {self.colors.text_light};
    }}
}}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>
    {content}
</body>
</html>"""


class CitadelTemplate(BaseTemplate):
    """Citadel inspired template - professional, executive-focused."""

    def __init__(self):
        super().__init__()
        self.name = "Citadel"
        self.description = "Executive-focused professional design"
        self.category = "backtest"

        self.colors = TemplateColors(
            primary="#1A1A2E",          # Dark navy/black
            primary_dark="#0F0F1A",
            primary_light="#2D2D44",
            secondary="#E94560",        # Citadel red
            accent="#16C79A",
            success="#16C79A",
            warning="#F5A623",
            danger="#E94560",
            text="#1A1A2E",
            text_light="#5C5C7A",
            background="#FFFFFF",
            background_alt="#F8F9FC",
            border="#E1E4E8",
        )

        self.typography = TemplateTypography(
            heading_font="'Playfair Display', 'Georgia', serif",
            body_font="'Lato', 'Helvetica', sans-serif",
            mono_font="'Fira Code', monospace",
            base_size="10pt",
            h1_size="30pt",
            h2_size="16pt",
            h3_size="12pt",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Citadel Executive Template Styles */

body {{
    background: {self.colors.background};
}}

h1 {{
    font-family: {self.typography.heading_font};
    color: {self.colors.primary};
    font-size: {self.typography.h1_size};
    font-weight: 700;
    text-align: center;
    margin-bottom: 8pt;
}}

h1 + h3 {{
    text-align: center;
    color: {self.colors.text_light};
    font-weight: 400;
    margin-top: 0;
}}

h2 {{
    font-family: {self.typography.heading_font};
    color: {self.colors.primary};
    font-size: {self.typography.h2_size};
    font-weight: 600;
    margin-top: 32pt;
    position: relative;
    padding-left: 20pt;
}}

h2::before {{
    content: '◆';
    position: absolute;
    left: 0;
    color: {self.colors.secondary};
}}

/* Hero Stats */
.hero-stats {{
    display: flex;
    justify-content: center;
    gap: 24pt;
    margin: 24pt 0;
    padding: 20pt;
    background: linear-gradient(135deg, {self.colors.primary} 0%, {self.colors.primary_light} 100%);
    border-radius: 8pt;
}}

.hero-stat {{
    text-align: center;
    color: white;
}}

.hero-stat .value {{
    font-size: 32pt;
    font-weight: 700;
    font-family: {self.typography.heading_font};
}}

.hero-stat .label {{
    font-size: 9pt;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1pt;
}}

/* Tables */
table {{
    border: none;
    border-radius: 8pt;
    overflow: hidden;
    box-shadow: 0 2pt 12pt rgba(26, 26, 46, 0.08);
}}

th {{
    background: {self.colors.primary};
    color: white;
    font-weight: 600;
    padding: 14pt 12pt;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 1pt;
}}

td {{
    padding: 12pt;
    border-bottom: 1px solid {self.colors.border};
}}

tr:hover {{
    background: rgba(233, 69, 96, 0.03);
}}

/* Verdict Badge */
.verdict {{
    display: inline-block;
    padding: 6pt 20pt;
    border-radius: 4pt;
    font-size: 11pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1pt;
}}

.verdict-outperform {{
    background: {self.colors.success};
    color: white;
}}

.verdict-underperform {{
    background: {self.colors.danger};
    color: white;
}}

.verdict-neutral {{
    background: {self.colors.text_light};
    color: white;
}}

/* Metric Comparison */
.comparison-row {{
    display: flex;
    align-items: center;
    padding: 12pt 0;
    border-bottom: 1px solid {self.colors.border};
}}

.comparison-label {{
    flex: 2;
    font-weight: 600;
}}

.comparison-strategy {{
    flex: 1;
    text-align: right;
    color: {self.colors.primary};
    font-weight: 700;
}}

.comparison-benchmark {{
    flex: 1;
    text-align: right;
    color: {self.colors.text_light};
}}

.comparison-alpha {{
    flex: 1;
    text-align: right;
    font-weight: 700;
}}

.alpha-positive {{ color: {self.colors.success}; }}
.alpha-negative {{ color: {self.colors.danger}; }}

/* Charts */
.chart-container {{
    background: {self.colors.background};
    border: 1px solid {self.colors.border};
    border-radius: 12pt;
    padding: 20pt;
    margin: 24pt 0;
    box-shadow: 0 4pt 16pt rgba(26, 26, 46, 0.06);
}}

.chart-title {{
    font-family: {self.typography.heading_font};
    font-size: 12pt;
    color: {self.colors.primary};
    margin-bottom: 12pt;
    padding-bottom: 8pt;
    border-bottom: 2px solid {self.colors.secondary};
}}

/* Risk Matrix */
.risk-matrix {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8pt;
    margin: 16pt 0;
}}

.risk-cell {{
    padding: 12pt;
    text-align: center;
    border-radius: 6pt;
}}

.risk-low {{
    background: rgba(22, 199, 154, 0.1);
    border: 1px solid {self.colors.success};
}}

.risk-medium {{
    background: rgba(245, 166, 35, 0.1);
    border: 1px solid {self.colors.warning};
}}

.risk-high {{
    background: rgba(233, 69, 96, 0.1);
    border: 1px solid {self.colors.danger};
}}

/* Disclaimer */
.disclaimer {{
    margin-top: 32pt;
    padding: 16pt;
    background: {self.colors.background_alt};
    border-radius: 8pt;
    font-size: 8pt;
    color: {self.colors.text_light};
    border-left: 3px solid {self.colors.secondary};
}}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>
    {content}
</body>
</html>"""


class QuantitativeTemplate(BaseTemplate):
    """Default quantitative research template - balanced, professional."""

    def __init__(self):
        super().__init__()
        self.name = "Quantitative"
        self.description = "Default quantitative research style"
        self.category = "backtest"

        self.colors = TemplateColors(
            primary="#2563EB",          # Modern blue
            primary_dark="#1D4ED8",
            primary_light="#3B82F6",
            secondary="#8B5CF6",        # Purple accent
            accent="#10B981",
            success="#10B981",
            warning="#F59E0B",
            danger="#EF4444",
            text="#1F2937",
            text_light="#6B7280",
            background="#FFFFFF",
            background_alt="#F9FAFB",
            border="#E5E7EB",
        )

    def get_css(self) -> str:
        return self.get_base_css() + f"""
/* Quantitative Default Template */

h1 {{
    color: {self.colors.primary_dark};
    font-size: 26pt;
    font-weight: 700;
    border-bottom: 3px solid {self.colors.primary};
    padding-bottom: 12pt;
}}

h2 {{
    color: {self.colors.primary};
    font-size: 16pt;
    font-weight: 600;
    margin-top: 24pt;
    padding-top: 16pt;
    border-top: 1px solid {self.colors.border};
}}

h3 {{
    color: {self.colors.text};
    font-size: 12pt;
    font-weight: 600;
}}

/* Metric Cards */
.metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12pt;
    margin: 16pt 0;
}}

.metric-card {{
    background: {self.colors.background_alt};
    border: 1px solid {self.colors.border};
    border-radius: 8pt;
    padding: 16pt;
    text-align: center;
}}

.metric-card .value {{
    font-size: 24pt;
    font-weight: 700;
    color: {self.colors.primary};
}}

.metric-card .label {{
    font-size: 9pt;
    color: {self.colors.text_light};
    text-transform: uppercase;
}}

/* Tables */
th {{
    background: {self.colors.primary};
}}

/* Charts */
.chart-container {{
    background: {self.colors.background};
    border: 1px solid {self.colors.border};
    border-radius: 8pt;
    padding: 16pt;
    margin: 16pt 0;
}}

/* Performance Colors */
.positive {{ color: {self.colors.success}; font-weight: 600; }}
.negative {{ color: {self.colors.danger}; font-weight: 600; }}
"""

    def get_html_wrapper(self, content: str, title: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{self.get_css()}</style>
</head>
<body>
    {content}
</body>
</html>"""


# Register all backtest templates
def register_backtest_templates() -> None:
    """Register all backtest templates with the registry."""
    templates = [
        TwoSigmaTemplate(),
        RenTechTemplate(),
        AQRTemplate(),
        ManGroupTemplate(),
        CitadelTemplate(),
        QuantitativeTemplate(),
    ]

    for template in templates:
        TemplateRegistry.register(template)


# Auto-register on import
register_backtest_templates()
