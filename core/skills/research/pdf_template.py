"""
Professional PDF Template for Research Reports
===============================================

World-class broker-grade PDF styling with:
- Proper page breaks
- Professional headers
- Color scheme matching institutional reports
- Clean typography
- Multiple template support
- Chart embedding
"""

from typing import List, Optional

# Professional Color Scheme (Inspired by Goldman Sachs / Morgan Stanley)
COLORS = {
    # Primary
    'primary_dark': '#1a365d',      # Deep Navy Blue
    'primary': '#2c5282',           # Navy Blue
    'primary_light': '#4299e1',     # Light Blue

    # Accent
    'accent_green': '#38a169',      # Buy Green
    'accent_yellow': '#d69e2e',     # Hold Yellow
    'accent_red': '#e53e3e',        # Sell Red

    # Neutrals
    'text_dark': '#1a202c',         # Almost Black
    'text': '#2d3748',              # Dark Gray
    'text_light': '#718096',        # Medium Gray
    'border': '#e2e8f0',            # Light Gray
    'background': '#f7fafc',        # Off White
    'white': '#ffffff',

    # Table
    'table_header': '#2c5282',      # Navy
    'table_row_alt': '#f7fafc',     # Light Gray
}

# CSS Template for WeasyPrint/HTML to PDF conversion
CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

@page {
    size: A4;
    margin: 2cm 1.5cm 2.5cm 1.5cm;

    @top-center {
        content: string(company-name);
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: #718096;
    }

    @bottom-left {
        content: "Jotty Research";
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #718096;
    }

    @bottom-center {
        content: counter(page) " of " counter(pages);
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #718096;
    }

    @bottom-right {
        content: "Confidential";
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #718096;
    }
}

/* First page - no header */
@page :first {
    @top-center {
        content: none;
    }
}

/* Page break utilities */
.page-break {
    page-break-before: always;
}

.avoid-break {
    page-break-inside: avoid;
}

.keep-together {
    page-break-inside: avoid;
}

/* Body */
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: #2d3748;
    background: #ffffff;
}

/* Headings */
h1 {
    font-size: 24pt;
    font-weight: 700;
    color: #1a365d;
    margin: 0 0 8pt 0;
    padding-bottom: 8pt;
    border-bottom: 3px solid #2c5282;
    string-set: company-name content();
}

h2 {
    font-size: 16pt;
    font-weight: 600;
    color: #2c5282;
    margin: 24pt 0 12pt 0;
    padding: 8pt 12pt;
    background: linear-gradient(90deg, #edf2f7 0%, #ffffff 100%);
    border-left: 4px solid #2c5282;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #1a365d;
    margin: 16pt 0 8pt 0;
    page-break-after: avoid;
}

h4 {
    font-size: 10pt;
    font-weight: 600;
    color: #2d3748;
    margin: 12pt 0 6pt 0;
}

/* Rating Box */
.rating-box {
    display: inline-block;
    padding: 4pt 12pt;
    border-radius: 4pt;
    font-weight: 700;
    font-size: 14pt;
    margin-right: 8pt;
}

.rating-buy {
    background: #c6f6d5;
    color: #22543d;
    border: 2px solid #38a169;
}

.rating-hold {
    background: #fefcbf;
    color: #744210;
    border: 2px solid #d69e2e;
}

.rating-sell {
    background: #fed7d7;
    color: #742a2a;
    border: 2px solid #e53e3e;
}

/* Cover Page Header */
.cover-header {
    text-align: center;
    padding: 20pt 0;
    margin-bottom: 20pt;
    border-bottom: 2px solid #2c5282;
}

.cover-header h1 {
    border-bottom: none;
    font-size: 28pt;
}

.cover-rating {
    font-size: 18pt;
    margin: 16pt 0;
}

.cover-metrics {
    display: flex;
    justify-content: center;
    gap: 24pt;
    margin: 16pt 0;
}

/* Key Metrics Box */
.metrics-box {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6pt;
    padding: 12pt;
    margin: 12pt 0;
    page-break-inside: avoid;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8pt;
}

.metric-item {
    display: flex;
    justify-content: space-between;
    padding: 4pt 0;
    border-bottom: 1px solid #e2e8f0;
}

.metric-label {
    color: #718096;
    font-size: 9pt;
}

.metric-value {
    font-weight: 600;
    color: #1a365d;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

thead {
    background: #2c5282;
    color: #ffffff;
}

th {
    padding: 8pt 6pt;
    text-align: left;
    font-weight: 600;
    border: none;
}

th:first-child {
    border-radius: 4pt 0 0 0;
}

th:last-child {
    border-radius: 0 4pt 0 0;
}

td {
    padding: 6pt;
    border-bottom: 1px solid #e2e8f0;
}

tr:nth-child(even) {
    background: #f7fafc;
}

tr:hover {
    background: #edf2f7;
}

/* Right-align numeric columns */
td:not(:first-child),
th:not(:first-child) {
    text-align: right;
}

/* Sensitivity Matrix */
.sensitivity-table th {
    background: #1a365d;
    font-size: 8pt;
}

.sensitivity-table td {
    font-size: 8pt;
    padding: 4pt;
}

.sensitivity-table td.highlight {
    background: #c6f6d5;
    font-weight: 600;
}

/* Football Field Chart */
.football-field {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4pt;
    padding: 12pt;
    margin: 12pt 0;
    font-family: monospace;
    font-size: 9pt;
    page-break-inside: avoid;
}

/* Code blocks (for ASCII charts) */
pre, code {
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 8pt;
    background: #f7fafc;
    padding: 8pt;
    border-radius: 4pt;
    overflow-x: auto;
    page-break-inside: avoid;
}

/* Lists */
ul, ol {
    margin: 8pt 0;
    padding-left: 20pt;
}

li {
    margin: 4pt 0;
}

/* Investment Thesis Box */
.thesis-box {
    background: linear-gradient(135deg, #ebf8ff 0%, #ffffff 100%);
    border: 1px solid #4299e1;
    border-radius: 6pt;
    padding: 12pt;
    margin: 12pt 0;
    page-break-inside: avoid;
}

.thesis-box h3 {
    color: #2c5282;
    margin-top: 0;
}

/* Risk Box */
.risk-box {
    background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
    border: 1px solid #fc8181;
    border-radius: 6pt;
    padding: 12pt;
    margin: 12pt 0;
    page-break-inside: avoid;
}

.risk-box h3 {
    color: #c53030;
    margin-top: 0;
}

/* Section Dividers */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 24pt 0;
}

/* Strong emphasis */
strong {
    font-weight: 600;
    color: #1a365d;
}

/* Blockquotes for callouts */
blockquote {
    border-left: 4px solid #4299e1;
    padding-left: 12pt;
    margin: 12pt 0;
    color: #4a5568;
    background: #f7fafc;
    padding: 8pt 12pt;
    border-radius: 0 4pt 4pt 0;
}

/* Disclaimer */
.disclaimer {
    font-size: 8pt;
    color: #718096;
    border-top: 1px solid #e2e8f0;
    padding-top: 12pt;
    margin-top: 24pt;
}

/* Charts container */
.chart-container {
    text-align: center;
    margin: 16pt 0;
    page-break-inside: avoid;
}

.chart-container img {
    max-width: 100%;
    height: auto;
}

/* Price target box */
.target-box {
    background: #1a365d;
    color: #ffffff;
    padding: 16pt;
    border-radius: 6pt;
    text-align: center;
    margin: 16pt 0;
    page-break-inside: avoid;
}

.target-box .price {
    font-size: 24pt;
    font-weight: 700;
}

.target-box .label {
    font-size: 10pt;
    opacity: 0.8;
}

/* Upside/Downside indicator */
.upside {
    color: #38a169;
    font-weight: 600;
}

.downside {
    color: #e53e3e;
    font-weight: 600;
}

/* Print-specific styles */
@media print {
    body {
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }

    .no-print {
        display: none;
    }
}

/* ==========================================
   WORLD-CLASS ENHANCEMENTS
   ========================================== */

/* Cover Page Rating Badge */
.rating-badge {
    display: inline-block;
    font-size: 24pt;
    font-weight: 700;
    padding: 12pt 24pt;
    border-radius: 8pt;
    margin: 16pt 0;
    text-transform: uppercase;
    letter-spacing: 2pt;
}

.rating-buy {
    background: linear-gradient(135deg, #38a169 0%, #276749 100%);
    color: #ffffff;
    box-shadow: 0 4pt 12pt rgba(56, 161, 105, 0.3);
}

.rating-hold {
    background: linear-gradient(135deg, #d69e2e 0%, #b7791f 100%);
    color: #ffffff;
    box-shadow: 0 4pt 12pt rgba(214, 158, 46, 0.3);
}

.rating-sell {
    background: linear-gradient(135deg, #e53e3e 0%, #c53030 100%);
    color: #ffffff;
    box-shadow: 0 4pt 12pt rgba(229, 62, 62, 0.3);
}

/* Cover Page Layout */
.cover-page {
    text-align: center;
    padding: 20pt 0;
}

.cover-page h1 {
    font-size: 32pt;
    border-bottom: none;
    margin-bottom: 0;
}

.cover-page h2 {
    background: none;
    border-left: none;
    color: #4a5568;
    font-size: 14pt;
    font-weight: 400;
    padding: 0;
    margin: 4pt 0 20pt 0;
}

/* Scenario Analysis Cards */
.scenario-card {
    border: 1px solid #e2e8f0;
    border-radius: 6pt;
    padding: 12pt;
    margin: 8pt 0;
    page-break-inside: avoid;
}

.scenario-bull {
    border-left: 4px solid #38a169;
    background: linear-gradient(90deg, #f0fff4 0%, #ffffff 50%);
}

.scenario-base {
    border-left: 4px solid #d69e2e;
    background: linear-gradient(90deg, #fffff0 0%, #ffffff 50%);
}

.scenario-bear {
    border-left: 4px solid #e53e3e;
    background: linear-gradient(90deg, #fff5f5 0%, #ffffff 50%);
}

/* Catalyst Timeline */
.catalyst-item {
    display: flex;
    align-items: center;
    padding: 8pt 0;
    border-bottom: 1px solid #e2e8f0;
}

.catalyst-timeline {
    background: #2c5282;
    color: #ffffff;
    padding: 2pt 8pt;
    border-radius: 4pt;
    font-size: 8pt;
    margin-right: 12pt;
    min-width: 60pt;
    text-align: center;
}

.catalyst-impact-high {
    background: #e53e3e;
}

.catalyst-impact-medium {
    background: #d69e2e;
}

.catalyst-impact-low {
    background: #38a169;
}

/* Industry Analysis Box */
.industry-box {
    background: linear-gradient(135deg, #ebf8ff 0%, #ffffff 100%);
    border: 1px solid #4299e1;
    border-radius: 6pt;
    padding: 12pt;
    margin: 12pt 0;
}

/* Earnings Table Highlight */
.earnings-table {
    border: 2px solid #2c5282;
}

.earnings-table thead {
    background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
}

.earnings-cagr {
    font-weight: 700;
    color: #38a169;
}

/* Valuation Summary Card */
.valuation-card {
    background: #1a365d;
    color: #ffffff;
    border-radius: 8pt;
    padding: 16pt;
    margin: 16pt 0;
    text-align: center;
}

.valuation-card .label {
    font-size: 9pt;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1pt;
}

.valuation-card .value {
    font-size: 28pt;
    font-weight: 700;
    margin: 8pt 0;
}

.valuation-card .subtext {
    font-size: 10pt;
    opacity: 0.9;
}

/* 52-Week Range Bar */
.range-bar {
    background: #e2e8f0;
    height: 8pt;
    border-radius: 4pt;
    position: relative;
    margin: 8pt 0;
}

.range-position {
    position: absolute;
    height: 16pt;
    width: 4pt;
    background: #2c5282;
    border-radius: 2pt;
    top: -4pt;
}

/* Key Metrics Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12pt;
    margin: 12pt 0;
}

.metric-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4pt;
    padding: 8pt;
    text-align: center;
}

.metric-card .value {
    font-size: 14pt;
    font-weight: 700;
    color: #1a365d;
}

.metric-card .label {
    font-size: 8pt;
    color: #718096;
    text-transform: uppercase;
}

/* Peer Comparison Highlight Row */
.peer-highlight {
    background: #ebf8ff !important;
    font-weight: 600;
}

/* Section Icons */
.section-icon {
    margin-right: 8pt;
    font-size: 14pt;
}

/* Footer Enhancement */
.report-footer {
    border-top: 2px solid #2c5282;
    padding-top: 12pt;
    margin-top: 24pt;
    text-align: center;
    color: #718096;
    font-size: 8pt;
}

/* Recommendation Box */
.recommendation-box {
    background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
    color: #ffffff;
    border-radius: 8pt;
    padding: 20pt;
    margin: 16pt 0;
    text-align: center;
    page-break-inside: avoid;
}

.recommendation-box h3 {
    color: #ffffff;
    margin: 0 0 12pt 0;
    font-size: 18pt;
}

.recommendation-box .rating {
    font-size: 32pt;
    font-weight: 700;
    margin: 8pt 0;
}

.recommendation-box .target {
    font-size: 24pt;
    font-weight: 600;
}

.recommendation-box .upside-box {
    background: rgba(255,255,255,0.2);
    border-radius: 4pt;
    padding: 8pt 16pt;
    display: inline-block;
    margin-top: 12pt;
}
"""

# HTML Template wrapper
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css}
    </style>
</head>
<body>
{content}
</body>
</html>
"""


def markdown_to_styled_html(markdown_content: str, title: str = "Research Report") -> str:
    """
    Convert markdown to professionally styled HTML.

    Args:
        markdown_content: Markdown content
        title: Report title

    Returns:
        Styled HTML string
    """
    import re

    # Pre-process: Remove div tags that interfere with markdown parsing
    # Save div classes for later restoration
    div_pattern = r'<div class="([^"]+)">'
    div_matches = re.findall(div_pattern, markdown_content)

    # Replace div tags with markers
    markdown_content = re.sub(r'<div class="cover-page">\s*', '', markdown_content)
    markdown_content = re.sub(r'<div class="metrics-grid">\s*', '', markdown_content)
    markdown_content = re.sub(r'<div class="rating-badge rating-\w+">\s*', '', markdown_content)
    markdown_content = re.sub(r'</div>', '', markdown_content)

    try:
        import markdown
        from markdown.extensions.tables import TableExtension
        from markdown.extensions.fenced_code import FencedCodeExtension
    except ImportError:
        # Fallback: basic conversion
        html_content = _basic_markdown_to_html(markdown_content)
        return HTML_TEMPLATE.format(title=title, css=CSS_TEMPLATE, content=html_content)

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'tables',
        'fenced_code',
        'nl2br',
        'sane_lists',
    ])

    html_content = md.convert(markdown_content)

    # Post-process HTML for better styling
    html_content = _enhance_html(html_content)

    return HTML_TEMPLATE.format(title=title, css=CSS_TEMPLATE, content=html_content)


def _enhance_html(html: str) -> str:
    """Enhance HTML with additional styling classes and page breaks."""
    import re

    # Add page breaks before major sections (h2)
    html = re.sub(
        r'<h2>',
        '<div class="page-break"></div>\n<h2>',
        html
    )

    # Remove first page break (before first h2)
    html = html.replace('<div class="page-break"></div>\n<h2>', '<h2>', 1)

    # Style cover page rating badge (large format)
    html = re.sub(
        r'<div class="rating-badge rating-buy">\s*🟢\s*BUY\s*</div>',
        '<div class="rating-badge rating-buy">✓ BUY</div>',
        html
    )
    html = re.sub(
        r'<div class="rating-badge rating-hold">\s*🟡\s*HOLD\s*</div>',
        '<div class="rating-badge rating-hold">● HOLD</div>',
        html
    )
    html = re.sub(
        r'<div class="rating-badge rating-sell">\s*🔴\s*SELL\s*</div>',
        '<div class="rating-badge rating-sell">✗ SELL</div>',
        html
    )

    # Style inline rating badges
    html = re.sub(
        r'🟢\s*BUY',
        '<span class="rating-box rating-buy">BUY</span>',
        html
    )
    html = re.sub(
        r'🟡\s*HOLD',
        '<span class="rating-box rating-hold">HOLD</span>',
        html
    )
    html = re.sub(
        r'🔴\s*SELL',
        '<span class="rating-box rating-sell">SELL</span>',
        html
    )

    # Style scenario analysis section headers with background boxes
    html = re.sub(
        r'<h3>Bull Case([^<]*)</h3>',
        r'</div><div class="scenario-bull"><h3 style="color:#276749;margin-top:0;">📈 Bull Case\1</h3>',
        html
    )
    html = re.sub(
        r'<h3>Base Case([^<]*)</h3>',
        r'</div><div class="scenario-base"><h3 style="color:#744210;margin-top:0;">📊 Base Case\1</h3>',
        html
    )
    html = re.sub(
        r'<h3>Bear Case([^<]*)</h3>',
        r'</div><div class="scenario-bear"><h3 style="color:#742a2a;margin-top:0;">📉 Bear Case\1</h3>',
        html
    )

    # Clean up any empty divs at the start
    html = re.sub(r'^</div>', '', html)

    # Ensure scenario divs are closed properly before next h2
    html = re.sub(r'(<div class="scenario-\w+">.*?)(<div class="page-break">)', r'\1</div>\2', html, flags=re.DOTALL)

    # Style upside/downside percentages (only in specific contexts, not globally)
    # This prevents color bleeding - we only style explicit "Upside:" and "Downside:" mentions
    html = re.sub(
        r'Upside:\s*\+(\d+\.?\d*%)',
        r'Upside: <span class="upside">+\1</span>',
        html
    )
    html = re.sub(
        r'Downside:\s*-(\d+\.?\d*%)',
        r'Downside: <span class="downside">-\1</span>',
        html
    )

    # Add avoid-break to tables
    html = re.sub(
        r'<table>',
        '<table class="avoid-break">',
        html
    )

    # Wrap investment thesis sections
    html = re.sub(
        r'(<h3>Investment Thesis</h3>)(.*?)(<h3>|<h2>|<div class="page-break">)',
        r'<div class="thesis-box">\1\2</div>\3',
        html,
        flags=re.DOTALL
    )

    # Wrap risk sections
    html = re.sub(
        r'(<h3>Key (?:Investment )?Risks</h3>)(.*?)(<h3>|<h2>|<div class="page-break">)',
        r'<div class="risk-box">\1\2</div>\3',
        html,
        flags=re.DOTALL
    )

    # Style impact badges in catalysts (contained styling)
    html = re.sub(r'🔴 High', '<strong style="color:#e53e3e">High</strong>', html)
    html = re.sub(r'🟡 Medium', '<strong style="color:#d69e2e">Medium</strong>', html)
    html = re.sub(r'🟢 Low', '<strong style="color:#38a169">Low</strong>', html)

    # Style checkmarks and warnings (contained styling)
    html = re.sub(r'✓ ', '<span style="color:#38a169;font-weight:bold">✓</span> ', html)
    html = re.sub(r'⚠ ', '<span style="color:#d69e2e;font-weight:bold">⚠</span> ', html)

    return html


def _basic_markdown_to_html(markdown_content: str) -> str:
    """Basic markdown to HTML conversion without external libraries."""
    import re

    html = markdown_content

    # Headers
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Code blocks
    html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)

    # Horizontal rules
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)

    # Lists
    html = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

    # Simple table conversion
    lines = html.split('\n')
    in_table = False
    new_lines = []

    for line in lines:
        if '|' in line and not line.strip().startswith('```'):
            if not in_table:
                new_lines.append('<table>')
                in_table = True

            if line.strip().startswith('|---') or line.strip().startswith('| ---'):
                continue  # Skip separator rows

            cells = [c.strip() for c in line.split('|')[1:-1]]
            if cells:
                row = '<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
                new_lines.append(row)
        else:
            if in_table:
                new_lines.append('</table>')
                in_table = False
            new_lines.append(line)

    if in_table:
        new_lines.append('</table>')

    html = '\n'.join(new_lines)

    # Paragraphs
    html = re.sub(r'\n\n+', r'</p>\n<p>', html)
    html = '<p>' + html + '</p>'

    return html


async def convert_md_to_pdf(
    md_path: str,
    output_path: str = None,
    template_name: str = None,
    chart_files: List[str] = None
) -> str:
    """
    Convert markdown file to professionally styled PDF.

    Args:
        md_path: Path to markdown file
        output_path: Optional output path for PDF
        template_name: Optional template name (goldman_sachs, morgan_stanley, clsa, motilal_oswal)
        chart_files: Optional list of chart image paths to embed

    Returns:
        Path to generated PDF
    """
    from pathlib import Path
    import base64

    md_file = Path(md_path)
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # Determine output path
    if output_path is None:
        output_path = str(md_file.with_suffix('.pdf'))

    # Convert markdown to HTML first
    title = md_file.stem.replace('_', ' ').title()

    # Get template if specified
    html_content = None
    if template_name:
        try:
            from .templates import TemplateRegistry
            template = TemplateRegistry.get(template_name)
            if template:
                inner_html = _markdown_to_html_content(markdown_content)
                html_content = template.get_html_wrapper(inner_html, title)
        except ImportError:
            pass

    # Fallback to default styling
    if html_content is None:
        html_content = markdown_to_styled_html(markdown_content, title)

    # Embed charts AFTER HTML conversion (avoids markdown parsing issues)
    if chart_files:
        charts_html = '<div class="page-break"></div>\n'
        charts_html += '<h2>Charts & Visualizations</h2>\n'

        valid_charts = 0
        for chart_path in chart_files:
            chart_file = Path(chart_path)
            if chart_file.exists() and chart_file.stat().st_size > 1000:  # Skip tiny/empty files
                with open(chart_file, 'rb') as f:
                    data = base64.b64encode(f.read()).decode('utf-8')
                chart_name = chart_file.stem.replace('_', ' ').title()

                charts_html += f'''<div class="chart-container" style="page-break-inside:avoid;margin:16pt 0;text-align:center;">
<img src="data:image/png;base64,{data}" alt="{chart_name}" style="max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:4pt;" />
<p style="text-align:center;font-size:9pt;color:#666;margin-top:6pt;">{chart_name}</p>
</div>\n'''
                valid_charts += 1

        # Only add charts section if we have valid charts
        if valid_charts > 0:
            # Insert before Technical Analysis section in HTML
            if '<h2>Technical Analysis</h2>' in html_content:
                html_content = html_content.replace(
                    '<h2>Technical Analysis</h2>',
                    charts_html + '\n<h2>Technical Analysis</h2>'
                )
            elif '<h2>' in html_content:
                # Find last h2 and insert after it
                last_h2_pos = html_content.rfind('</body>')
                if last_h2_pos > 0:
                    html_content = html_content[:last_h2_pos] + charts_html + html_content[last_h2_pos:]

    # Fallback check (in case above didn't set it)
    if html_content is None:
        title = md_file.stem.replace('_', ' ').title()
        html_content = markdown_to_styled_html(markdown_content, title)

    # Try WeasyPrint first (best quality)
    try:
        from weasyprint import HTML, CSS

        html_doc = HTML(string=html_content, base_url=str(md_file.parent))
        html_doc.write_pdf(output_path)
        return output_path

    except ImportError:
        pass

    # Try pdfkit (wkhtmltopdf wrapper)
    try:
        import pdfkit

        options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '15mm',
            'margin-bottom': '25mm',
            'margin-left': '15mm',
            'encoding': 'UTF-8',
            'enable-local-file-access': None,
        }

        pdfkit.from_string(html_content, output_path, options=options)
        return output_path

    except ImportError:
        pass

    # Fallback: save HTML for manual conversion
    html_path = str(Path(output_path).with_suffix('.html'))
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    raise ImportError(
        f"No PDF library available. Install weasyprint or pdfkit. "
        f"HTML saved to: {html_path}"
    )


def _markdown_to_html_content(markdown_content: str) -> str:
    """Convert markdown to HTML content only (no wrapper)."""
    import re

    # Pre-process: Remove div tags
    markdown_content = re.sub(r'<div class="[^"]*">\s*', '', markdown_content)
    markdown_content = re.sub(r'</div>', '', markdown_content)

    try:
        import markdown
        md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
        html_content = md.convert(markdown_content)
        return _enhance_html(html_content)
    except ImportError:
        return _basic_markdown_to_html(markdown_content)
