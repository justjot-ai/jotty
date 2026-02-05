"""
Professional PDF Template for Learning Content
===============================================

World-class educational PDF styling with:
- Clean, readable typography
- Color-coded sections
- Math-friendly formatting
- Progressive difficulty indicators
- Code block styling
- Bingo! highlight boxes
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import json

import dspy

logger = logging.getLogger(__name__)


class VisualizationSpecSignature(dspy.Signature):
    """Design a meaningful educational visualization for a concept.

    You are creating a chart that TEACHES something about the concept.
    The data should be illustrative and educational - showing how the concept works,
    not just labeling boxes.

    Choose chart_type wisely:
    - heatmap: for attention/correlation/matrix data
    - bar_chart: for comparing quantities or categories
    - line_chart: for trends, curves, or continuous relationships
    - scatter: for relationships between two variables
    - pie: for proportional breakdowns
    - none: if the concept doesn't benefit from a chart (e.g., too abstract)

    IMPORTANT LABEL RULES:
    - Keep ALL labels SHORT: max 2-3 words each. Use abbreviations (e.g., "Enc Layer 1" not "Encoder Layer 1 Output").
    - For line charts with string x-axis labels, use at most 6-8 labels.
    - For bar charts, use at most 8-10 categories.
    - Prefer numeric x-axes for line charts when possible (e.g., steps, epochs, positions).

    The data_json should contain realistic illustrative data that helps
    the reader understand the concept, not just placeholder values.
    """
    concept_name: str = dspy.InputField(desc="Name of the concept to visualize")
    concept_description: str = dspy.InputField(desc="Description of the concept including why it matters and how it works")

    chart_type: str = dspy.OutputField(desc="One of: bar_chart, line_chart, heatmap, scatter, pie, none")
    title: str = dspy.OutputField(desc="Chart title that explains what the visualization shows")
    x_label: str = dspy.OutputField(desc="X-axis label (empty string if not applicable)")
    y_label: str = dspy.OutputField(desc="Y-axis label (empty string if not applicable)")
    data_json: str = dspy.OutputField(desc='JSON string with chart data. For bar_chart: {"labels": [...], "values": [...]}. For line_chart: {"labels": [...], "series": [{"name": "...", "values": [...]}]}. For heatmap: {"x_labels": [...], "y_labels": [...], "values": [[...]]}. For scatter: {"x": [...], "y": [...], "labels": [...]}. For pie: {"labels": [...], "values": [...]}')
    annotation: str = dspy.OutputField(desc="Key insight to annotate on the chart - what should the reader notice?")

# Learning Color Scheme
LEARNING_COLORS = {
    # Primary (Knowledge/Learning)
    'primary_dark': '#1e3a5f',       # Deep Blue
    'primary': '#2563eb',            # Bright Blue
    'primary_light': '#60a5fa',      # Light Blue

    # Progress levels
    'level_basics': '#10b981',       # Green - Basics
    'level_intuition': '#3b82f6',    # Blue - Intuition
    'level_math': '#8b5cf6',         # Purple - Math
    'level_application': '#f59e0b',  # Orange - Application
    'level_deep': '#ef4444',         # Red - Deep Dive

    # Accents
    'bingo': '#fbbf24',              # Gold - Bingo moments!
    'insight': '#34d399',            # Mint - Key insights
    'code': '#1f2937',               # Dark - Code blocks

    # Neutrals
    'text_dark': '#111827',
    'text': '#374151',
    'text_light': '#6b7280',
    'border': '#e5e7eb',
    'background': '#f9fafb',
    'white': '#ffffff',
}

# CSS Template for Learning PDFs
LEARNING_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@page {
    size: A4;
    margin: 2cm 1.5cm 2.5cm 1.5cm;

    @top-center {
        content: string(paper-title);
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: #6b7280;
    }

    @bottom-left {
        content: "Jotty Learning";
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #6b7280;
    }

    @bottom-center {
        content: counter(page) " of " counter(pages);
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #6b7280;
    }

    @bottom-right {
        content: "ArXiv Paper Learning";
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #6b7280;
    }
}

@page :first {
    @top-center { content: none; }
}

.page-break { page-break-before: always; }
.avoid-break { page-break-inside: avoid; }

body {
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 10.5pt;
    line-height: 1.7;
    color: #374151;
    background: #ffffff;
}

/* Cover Page */
.cover-page {
    text-align: center;
    padding: 40pt 20pt;
    min-height: 80vh;
}

.cover-page h1 {
    font-size: 28pt;
    font-weight: 700;
    color: #1e3a5f;
    margin: 0 0 12pt 0;
    string-set: paper-title content();
    line-height: 1.3;
}

.cover-page .arxiv-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11pt;
    color: #2563eb;
    background: #eff6ff;
    padding: 4pt 12pt;
    border-radius: 4pt;
    display: inline-block;
    margin: 12pt 0;
}

.cover-page .authors {
    font-size: 11pt;
    color: #6b7280;
    margin: 16pt 0;
}

.cover-page .learning-badge {
    display: inline-block;
    background: linear-gradient(135deg, #2563eb 0%, #1e3a5f 100%);
    color: #ffffff;
    font-size: 14pt;
    font-weight: 600;
    padding: 12pt 24pt;
    border-radius: 8pt;
    margin: 24pt 0;
    box-shadow: 0 4pt 12pt rgba(37, 99, 235, 0.3);
}

.cover-page .stats {
    display: flex;
    justify-content: center;
    gap: 24pt;
    margin: 24pt 0;
}

.cover-page .stat-item {
    text-align: center;
}

.cover-page .stat-value {
    font-size: 24pt;
    font-weight: 700;
    color: #1e3a5f;
}

.cover-page .stat-label {
    font-size: 9pt;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1pt;
}

/* Section Headers */
h2 {
    font-size: 16pt;
    font-weight: 600;
    color: #1e3a5f;
    margin: 24pt 0 12pt 0;
    padding: 10pt 14pt;
    border-left: 4px solid #2563eb;
    background: linear-gradient(90deg, #eff6ff 0%, #ffffff 100%);
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    font-weight: 600;
    color: #1e3a5f;
    margin: 18pt 0 10pt 0;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    font-weight: 600;
    color: #374151;
    margin: 14pt 0 8pt 0;
}

/* Difficulty Level Indicators */
.level-badge {
    display: inline-block;
    font-size: 8pt;
    font-weight: 600;
    padding: 2pt 8pt;
    border-radius: 4pt;
    text-transform: uppercase;
    letter-spacing: 0.5pt;
    margin-left: 8pt;
}

.level-1 { background: #d1fae5; color: #065f46; }
.level-2 { background: #dbeafe; color: #1e40af; }
.level-3 { background: #ede9fe; color: #5b21b6; }
.level-4 { background: #fef3c7; color: #92400e; }
.level-5 { background: #fee2e2; color: #991b1b; }

/* Bingo Box - Key Insights */
.bingo-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
    border: 2px solid #fbbf24;
    border-radius: 8pt;
    padding: 14pt 16pt;
    margin: 16pt 0;
    page-break-inside: avoid;
}

.bingo-box::before {
    content: "🎯 Bingo!";
    display: block;
    font-weight: 700;
    font-size: 12pt;
    color: #92400e;
    margin-bottom: 8pt;
}

.bingo-box p {
    margin: 0;
    color: #78350f;
}

/* Hook Section */
.hook-section {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    color: #ffffff;
    padding: 20pt 24pt;
    border-radius: 8pt;
    margin: 20pt 0;
    page-break-inside: avoid;
}

.hook-section h3 {
    color: #ffffff;
    margin: 0 0 12pt 0;
    font-size: 14pt;
}

.hook-section p {
    color: rgba(255, 255, 255, 0.9);
    font-size: 11pt;
    line-height: 1.6;
    margin: 0;
}

/* Concept Card */
.concept-card {
    border: 1px solid #e5e7eb;
    border-radius: 8pt;
    padding: 14pt;
    margin: 12pt 0;
    page-break-inside: avoid;
}

.concept-card h4 {
    margin: 0 0 8pt 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.concept-card .difficulty {
    font-size: 9pt;
    color: #6b7280;
}

.concept-card .why-matters {
    font-style: italic;
    color: #6b7280;
    font-size: 10pt;
    margin: 8pt 0;
    padding-left: 12pt;
    border-left: 2px solid #dbeafe;
}

/* Math Block */
.math-block {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #8b5cf6;
    border-radius: 4pt;
    padding: 12pt 16pt;
    margin: 12pt 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10pt;
    overflow-x: auto;
    page-break-inside: avoid;
}

/* Code Block - Monokai Theme */
pre, code {
    font-family: 'JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', monospace;
    font-size: 9pt;
}

pre {
    /* Monokai background */
    background: #272822;
    color: #f8f8f2;
    padding: 14pt 16pt;
    border-radius: 6pt;
    border-left: 4px solid #a6e22e;
    line-height: 1.6;
    margin: 14pt 0;
    page-break-inside: avoid;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* Inline code */
code {
    background: #3e3d32;
    padding: 2pt 5pt;
    border-radius: 3pt;
    color: #e6db74;
    font-size: 9pt;
}

pre code {
    background: transparent;
    padding: 0;
    color: #f8f8f2;
}

/* Syntax highlighting hints in Monokai */
.code-keyword { color: #f92672; }
.code-string { color: #e6db74; }
.code-comment { color: #75715e; }
.code-function { color: #a6e22e; }
.code-number { color: #ae81ff; }
.code-class { color: #66d9ef; font-style: italic; }

/* Math Expressions */
.math-display {
    font-family: 'JetBrains Mono', 'Times New Roman', serif;
    font-size: 11pt;
    background: #fefce8;
    border: 1px solid #fef08a;
    border-left: 4px solid #eab308;
    padding: 12pt 16pt;
    margin: 14pt 0;
    border-radius: 4pt;
    text-align: center;
    color: #713f12;
    page-break-inside: avoid;
}

.math-inline {
    font-family: 'JetBrains Mono', 'Times New Roman', serif;
    font-size: 10pt;
    background: #fef9c3;
    padding: 1pt 4pt;
    border-radius: 3pt;
    color: #854d0e;
}

/* Insight List */
.insights-list {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8pt;
    padding: 14pt 16pt 14pt 20pt;
    margin: 16pt 0;
}

.insights-list h4 {
    color: #166534;
    margin: 0 0 10pt 0;
}

.insights-list li {
    color: #15803d;
    margin: 6pt 0;
}

/* Summary Box */
.summary-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8pt;
    padding: 16pt;
    margin: 20pt 0;
}

.summary-box h4 {
    color: #1e40af;
    margin: 0 0 10pt 0;
}

/* Next Steps */
.next-steps {
    background: #fefce8;
    border: 1px solid #fef08a;
    border-radius: 8pt;
    padding: 14pt 16pt;
    margin: 16pt 0;
}

.next-steps h4 {
    color: #854d0e;
    margin: 0 0 10pt 0;
}

.next-steps li {
    color: #713f12;
    margin: 6pt 0;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 9.5pt;
}

th {
    background: #1e3a5f;
    color: #ffffff;
    font-weight: 600;
    padding: 8pt 12pt;
    text-align: left;
}

td {
    padding: 8pt 12pt;
    border-bottom: 1px solid #e5e7eb;
}

tr:nth-child(even) {
    background: #f9fafb;
}

/* Progress bar */
.progress-bar {
    display: flex;
    height: 8pt;
    border-radius: 4pt;
    overflow: hidden;
    margin: 12pt 0;
}

.progress-segment {
    height: 100%;
}

/* Print styles */
@media print {
    body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .no-print { display: none; }
}

/* Visualization Container */
.visualization {
    margin: 16pt 0;
    text-align: center;
    page-break-inside: avoid;
}

.visualization img {
    max-width: 100%;
    height: auto;
    border: 1px solid #e5e7eb;
    border-radius: 6pt;
}

.visualization-caption {
    font-size: 9pt;
    color: #6b7280;
    margin-top: 8pt;
    font-style: italic;
}
"""


def _apply_syntax_highlighting(code: str) -> str:
    """Apply Monokai-style syntax highlighting to code."""
    import re

    # Escape HTML first
    code = code.replace('<', '&lt;').replace('>', '&gt;')

    # Process line by line
    lines = code.split('\n')
    result_lines = []

    for line in lines:
        # Check if line is a comment
        comment_match = re.match(r'^(\s*)(#.*)$', line)
        if comment_match:
            indent = comment_match.group(1)
            comment = comment_match.group(2)
            result_lines.append(f'{indent}<span class="code-comment">{comment}</span>')
            continue

        # Process strings first - replace with tokens
        string_tokens = {}
        token_counter = [0]

        def save_string(m):
            token = f"__XSTRX{token_counter[0]}XENDX__"
            string_tokens[token] = f'<span class="code-string">{m.group(0)}</span>'
            token_counter[0] += 1
            return token

        # Triple quotes
        line = re.sub(r'""".*?"""', save_string, line)
        line = re.sub(r"'''.*?'''", save_string, line)
        # F-strings and regular strings
        line = re.sub(r'f"[^"]*"', save_string, line)
        line = re.sub(r"f'[^']*'", save_string, line)
        line = re.sub(r'"[^"]*"', save_string, line)
        line = re.sub(r"'[^']*'", save_string, line)

        # Keywords - sorted by length (longest first) to avoid partial matches
        # Also use special markers to avoid matching inside our own output
        keywords = [
            ('finally', 'KW_FINALLY'), ('continue', 'KW_CONTINUE'),
            ('except', 'KW_EXCEPT'), ('return', 'KW_RETURN'),
            ('import', 'KW_IMPORT'), ('lambda', 'KW_LAMBDA'),
            ('while', 'KW_WHILE'), ('yield', 'KW_YIELD'),
            ('raise', 'KW_RAISE'), ('break', 'KW_BREAK'),
            ('class', 'KW_CLASS'), ('False', 'KW_FALSE'),
            ('async', 'KW_ASYNC'), ('await', 'KW_AWAIT'),
            ('elif', 'KW_ELIF'), ('else', 'KW_ELSE'),
            ('from', 'KW_FROM'), ('None', 'KW_NONE'),
            ('pass', 'KW_PASS'), ('True', 'KW_TRUE'),
            ('with', 'KW_WITH'), ('self', 'KW_SELF'),
            ('and', 'KW_AND'), ('def', 'KW_DEF'),
            ('for', 'KW_FOR'), ('not', 'KW_NOT'),
            ('try', 'KW_TRY'), ('cls', 'KW_CLS'),
            ('as', 'KW_AS'), ('if', 'KW_IF'),
            ('in', 'KW_IN'), ('is', 'KW_IS'), ('or', 'KW_OR')
        ]

        # Replace keywords with markers
        for kw, marker in keywords:
            line = re.sub(
                rf'(?<![a-zA-Z_])({kw})(?![a-zA-Z_0-9])',
                f'__X{marker}X__',
                line
            )

        # Numbers (before converting markers to HTML)
        line = re.sub(
            r'(?<![a-zA-Z_\dX])(\d+\.?\d*)(?![a-zA-Z_\d])',
            r'<span class="code-number">\1</span>',
            line
        )

        # Convert markers to HTML spans
        for kw, marker in keywords:
            line = line.replace(f'__X{marker}X__', f'<span class="code-keyword">{kw}</span>')

        # Function names after 'def '
        line = re.sub(
            r'(<span class="code-keyword">def</span>\s+)([a-zA-Z_]\w*)',
            r'\1<span class="code-function">\2</span>',
            line
        )

        # Class names after 'class '
        line = re.sub(
            r'(<span class="code-keyword">class</span>\s+)([a-zA-Z_]\w*)',
            r'\1<span class="code-class">\2</span>',
            line
        )

        # Restore string tokens
        for token, replacement in string_tokens.items():
            line = line.replace(token, replacement)

        result_lines.append(line)

    return '\n'.join(result_lines)


class VisualizationRenderer:
    """Renders LLM-generated visualization specs into matplotlib charts."""

    CHART_COLORS = {
        'primary': '#2563eb',
        'secondary': '#10b981',
        'accent': '#f59e0b',
        'danger': '#ef4444',
        'purple': '#8b5cf6',
        'pink': '#ec4899',
        'teal': '#06b6d4',
        'palette': ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'],
    }

    @classmethod
    def _auto_format_labels(cls, labels: list) -> dict:
        """Determine optimal label formatting based on content.

        Returns dict with keys: fontsize, rotation, ha, labels (possibly truncated).
        """
        if not labels:
            return {'fontsize': 8, 'rotation': 0, 'ha': 'center', 'labels': labels}

        # Check if labels are numeric (no rotation needed)
        all_numeric = all(
            isinstance(l, (int, float)) or (isinstance(l, str) and l.replace('.', '', 1).replace('-', '', 1).isdigit())
            for l in labels
        )
        if all_numeric:
            return {'fontsize': 8, 'rotation': 0, 'ha': 'center', 'labels': labels}

        # String labels — aggressive truncation based on density
        count = len(labels)
        # Tighter truncation when there are many labels
        if count > 10:
            max_chars = 8
        elif count > 6:
            max_chars = 12
        else:
            max_chars = 15

        truncated = []
        for l in labels:
            s = str(l)
            truncated.append(s[:max_chars - 1] + '…' if len(s) > max_chars else s)

        max_len = max(len(str(l)) for l in truncated)

        # For extremely dense label sets, show every other label
        if count > 12 and max_len > 5:
            spaced = [t if i % 2 == 0 else '' for i, t in enumerate(truncated)]
            return {'fontsize': 6, 'rotation': 60, 'ha': 'right', 'labels': spaced}
        elif max_len > 10 or count > 10:
            return {'fontsize': 6, 'rotation': 60, 'ha': 'right', 'labels': truncated}
        elif max_len > 6 or count > 6:
            return {'fontsize': 7, 'rotation': 45, 'ha': 'right', 'labels': truncated}
        else:
            return {'fontsize': 8, 'rotation': 0, 'ha': 'center', 'labels': truncated}

    @classmethod
    def _validate_data(cls, chart_type: str, data: dict) -> dict:
        """Sanitize LLM output to prevent rendering crashes.

        Returns cleaned data dict safe for rendering.
        """
        data = dict(data)  # shallow copy

        try:
            if chart_type in ('bar_chart', 'pie'):
                labels = data.get('labels', [])
                values = data.get('values', [])
                if labels and values:
                    min_len = min(len(labels), len(values))
                    data['labels'] = labels[:min_len]
                    # Ensure values are numeric
                    clean_values = []
                    for v in values[:min_len]:
                        try:
                            clean_values.append(float(v))
                        except (ValueError, TypeError):
                            clean_values.append(0.0)
                    data['values'] = clean_values

            elif chart_type == 'line_chart':
                labels = data.get('labels', [])
                series_list = data.get('series', [])
                if labels and series_list:
                    for s in series_list:
                        vals = s.get('values', [])
                        min_len = min(len(labels), len(vals))
                        s['values'] = vals[:min_len]
                    data['labels'] = labels[:min_len] if series_list else labels
                elif labels and data.get('values'):
                    min_len = min(len(labels), len(data['values']))
                    data['labels'] = labels[:min_len]
                    data['values'] = data['values'][:min_len]

            elif chart_type == 'heatmap':
                import numpy as np
                values = data.get('values', [])
                if values:
                    # Ensure rectangular: pad/truncate rows to same length
                    max_cols = max(len(row) for row in values) if values else 0
                    rect_values = []
                    for row in values:
                        clean_row = []
                        for v in row:
                            try:
                                clean_row.append(float(v))
                            except (ValueError, TypeError):
                                clean_row.append(0.0)
                        # Pad or truncate to max_cols
                        if len(clean_row) < max_cols:
                            clean_row.extend([0.0] * (max_cols - len(clean_row)))
                        else:
                            clean_row = clean_row[:max_cols]
                        rect_values.append(clean_row)
                    data['values'] = rect_values

                    # Sync labels with dimensions
                    x_labels = data.get('x_labels', data.get('labels', []))
                    y_labels = data.get('y_labels', [])
                    if x_labels and len(x_labels) != max_cols:
                        data['x_labels'] = x_labels[:max_cols] if len(x_labels) > max_cols else x_labels + [f'C{i}' for i in range(len(x_labels), max_cols)]
                    if y_labels and len(y_labels) != len(rect_values):
                        data['y_labels'] = y_labels[:len(rect_values)] if len(y_labels) > len(rect_values) else y_labels + [f'R{i}' for i in range(len(y_labels), len(rect_values))]

            elif chart_type == 'scatter':
                x = data.get('x', [])
                y = data.get('y', [])
                if x and y:
                    min_len = min(len(x), len(y))
                    data['x'] = x[:min_len]
                    data['y'] = y[:min_len]
                    if data.get('labels'):
                        data['labels'] = data['labels'][:min_len]

        except Exception as e:
            logger.warning(f"Data validation encountered error: {e}")

        return data

    @classmethod
    def render(cls, ax, chart_type: str, data: dict, title: str, x_label: str, y_label: str, annotation: str) -> bool:
        """Dispatch to the appropriate renderer. Returns True on success."""
        import matplotlib.pyplot as plt

        renderers = {
            'bar_chart': cls._render_bar_chart,
            'line_chart': cls._render_line_chart,
            'heatmap': cls._render_heatmap,
            'scatter': cls._render_scatter,
            'pie': cls._render_pie,
        }
        renderer = renderers.get(chart_type)
        if not renderer:
            return False

        # Validate data before rendering
        data = cls._validate_data(chart_type, data)

        fig = ax.figure
        # Figure-level styling
        fig.patch.set_facecolor('#fafbfc')
        ax.set_facecolor('#ffffff')

        renderer(ax, data, x_label, y_label)

        # Title with subtle separator line beneath
        ax.set_title(title, fontsize=13, fontweight='bold', pad=18, color='#1e3a5f')
        # Draw thin separator under title
        title_obj = ax.title
        fig.canvas.draw()
        try:
            bbox = title_obj.get_window_extent(renderer=fig.canvas.get_renderer())
            inv = fig.transFigure.inverted()
            title_bottom = inv.transform(bbox)[0][1]
            fig.add_artist(
                plt.Line2D([0.12, 0.88], [title_bottom - 0.01, title_bottom - 0.01],
                           transform=fig.transFigure, color='#e5e7eb', linewidth=0.8)
            )
        except Exception:
            pass  # Graceful fallback if renderer not ready

        # Consistent tick styling
        ax.tick_params(colors='#6b7280', labelsize=8)

        # Annotation using fig.text() — avoids collision with axis labels
        if annotation:
            fig.subplots_adjust(bottom=0.22)
            # Separator line above annotation area
            fig.add_artist(
                plt.Line2D([0.08, 0.92], [0.09, 0.09],
                           transform=fig.transFigure, color='#e5e7eb', linewidth=0.6)
            )
            fig.text(
                0.5, 0.035, annotation,
                ha='center', va='center',
                fontsize=7.5, fontstyle='italic', color='#6b7280',
                wrap=True
            )

        return True

    @classmethod
    def _render_bar_chart(cls, ax, data: dict, x_label: str, y_label: str):
        """Render a bar chart from spec data."""
        labels = data.get('labels', [])
        values = data.get('values', [])
        colors = data.get('colors', cls.CHART_COLORS['palette'][:len(labels)])

        if not labels or not values:
            return

        # Smart label formatting
        fmt = cls._auto_format_labels(labels)
        display_labels = fmt['labels']

        x_pos = range(len(display_labels))
        # Main bars with rounded appearance via edgecolor
        bars = ax.bar(x_pos, values, color=colors[:len(display_labels)],
                      edgecolor=[cls._lighten_color(c, 0.3) for c in colors[:len(display_labels)]],
                      linewidth=1.2, width=0.7, zorder=3)
        # Subtle highlight overlay on upper portion of bars for gradient effect
        max_val = max(values) if values else 1
        for bar, c in zip(bars, colors[:len(display_labels)]):
            lighter = cls._lighten_color(c, 0.4)
            ax.bar(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.35,
                   bottom=bar.get_height() * 0.65, width=bar.get_width(),
                   color=lighter, alpha=0.5, zorder=4)

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(display_labels, fontsize=fmt['fontsize'],
                           rotation=fmt['rotation'], ha=fmt['ha'])
        ax.set_xlabel(x_label, fontsize=9, color='#374151', labelpad=8)
        ax.set_ylabel(y_label, fontsize=9, color='#374151')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        ax.grid(axis='y', alpha=0.2, color='#d1d5db', linestyle='--')

        # Smart value label positioning — inside bar if tall enough, above otherwise
        for bar, val in zip(bars, values):
            bar_height = bar.get_height()
            if bar_height > max_val * 0.15:
                # Inside the bar
                ax.text(bar.get_x() + bar.get_width() / 2, bar_height * 0.85,
                        f'{val:.2g}', ha='center', va='top', fontsize=7,
                        color='white', fontweight='bold', zorder=5)
            else:
                # Above the bar
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar_height + 0.02 * max_val,
                        f'{val:.2g}', ha='center', va='bottom', fontsize=7,
                        color='#374151', zorder=5)

    @staticmethod
    def _lighten_color(hex_color: str, factor: float = 0.3) -> str:
        """Lighten a hex color by blending with white."""
        try:
            hex_color = hex_color.lstrip('#')
            r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            r = int(r + (255 - r) * factor)
            g = int(g + (255 - g) * factor)
            b = int(b + (255 - b) * factor)
            return f'#{r:02x}{g:02x}{b:02x}'
        except Exception:
            return '#d1d5db'

    @classmethod
    def _render_line_chart(cls, ax, data: dict, x_label: str, y_label: str):
        """Render a line chart from spec data."""
        labels = data.get('labels', [])
        series_list = data.get('series', [])

        if not labels or not series_list:
            # Fallback: single series with 'values' key
            values = data.get('values', [])
            if labels and values:
                series_list = [{'name': y_label or 'Value', 'values': values}]
            else:
                return

        # Smart label formatting — key fix for the overlap issue
        fmt = cls._auto_format_labels(labels)
        display_labels = fmt['labels']

        # Use numeric x-positions for string labels to allow smooth interpolation
        x_numeric = list(range(len(display_labels)))

        colors = cls.CHART_COLORS['palette']
        for i, series in enumerate(series_list):
            name = series.get('name', f'Series {i+1}')
            vals = series.get('values', [])
            if not vals:
                continue

            color = colors[i % len(colors)]
            plot_x = x_numeric[:len(vals)]

            # Try smooth interpolation if scipy available
            try:
                import numpy as np
                from scipy.interpolate import make_interp_spline
                if len(plot_x) >= 4:
                    x_arr = np.array(plot_x, dtype=float)
                    y_arr = np.array(vals, dtype=float)
                    x_smooth = np.linspace(x_arr.min(), x_arr.max(), 200)
                    spl = make_interp_spline(x_arr, y_arr, k=min(3, len(plot_x) - 1))
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color, linewidth=2.2, label=name, zorder=3)
                    ax.fill_between(x_smooth, y_smooth, alpha=0.08, color=color, zorder=2)
                    # Data point markers
                    ax.scatter(plot_x, vals, color=color, s=30, zorder=4,
                               edgecolors='white', linewidth=1.2)
                else:
                    raise ImportError  # Fall through to basic plot
            except (ImportError, Exception):
                ax.plot(plot_x, vals, color=color, linewidth=2.2,
                        marker='o', markersize=5, label=name, zorder=3,
                        markeredgecolor='white', markeredgewidth=1.2)
                ax.fill_between(plot_x, vals, alpha=0.08, color=color, zorder=2)

        # Apply formatted tick labels
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(display_labels, fontsize=fmt['fontsize'],
                           rotation=fmt['rotation'], ha=fmt['ha'])
        ax.set_xlabel(x_label, fontsize=9, color='#374151', labelpad=8)
        ax.set_ylabel(y_label, fontsize=9, color='#374151')

        # Legend outside plot for cleanliness when multiple series
        if len(series_list) > 1:
            ax.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(0, 1),
                      framealpha=0.9, edgecolor='#e5e7eb')
        else:
            ax.legend(fontsize=8, loc='best', framealpha=0.9, edgecolor='#e5e7eb')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        ax.grid(alpha=0.2, color='#d1d5db', linestyle='--')

    @classmethod
    def _render_heatmap(cls, ax, data: dict, x_label: str, y_label: str):
        """Render a heatmap from spec data."""
        import numpy as np

        x_labels = data.get('x_labels', data.get('labels', []))
        y_labels = data.get('y_labels', x_labels)
        values = data.get('values', [])

        if not values:
            return

        arr = np.array(values, dtype=float)
        # Perceptual colormap for educational clarity
        im = ax.imshow(arr, cmap='YlOrRd', aspect='auto', interpolation='nearest')

        # Format labels
        x_fmt = cls._auto_format_labels(x_labels)
        y_fmt = cls._auto_format_labels(y_labels)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_fmt['labels'], fontsize=x_fmt['fontsize'],
                           rotation=x_fmt['rotation'], ha=x_fmt['ha'])
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_fmt['labels'], fontsize=y_fmt['fontsize'])
        ax.set_xlabel(x_label, fontsize=9, color='#374151', labelpad=8)
        ax.set_ylabel(y_label, fontsize=9, color='#374151')

        # Add value text in cells if small enough — cleaner decimal display
        if arr.shape[0] <= 8 and arr.shape[1] <= 8:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    color = 'white' if val > arr.max() * 0.6 else '#1e3a5f'
                    # Cleaner formatting: integer if whole, 1 decimal otherwise
                    txt = f'{val:.0f}' if val == int(val) else f'{val:.1f}'
                    ax.text(j, i, txt, ha='center', va='center',
                            fontsize=7, fontweight='bold', color=color)

        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    @classmethod
    def _render_scatter(cls, ax, data: dict, x_label: str, y_label: str):
        """Render a scatter plot from spec data."""
        x = data.get('x', [])
        y = data.get('y', [])
        point_labels = data.get('labels', [])
        colors = data.get('colors', cls.CHART_COLORS['palette'][:len(x)])
        sizes = data.get('sizes', None)

        if not x or not y:
            return

        # Size variation if provided, otherwise default larger size
        if sizes and len(sizes) >= len(x):
            s = sizes[:len(x)]
        else:
            s = [80] * len(x)

        ax.scatter(x, y, c=colors[:len(x)], s=s,
                   edgecolors='white', linewidth=1.0, zorder=3, alpha=0.85)
        ax.set_xlabel(x_label, fontsize=9, color='#374151', labelpad=8)
        ax.set_ylabel(y_label, fontsize=9, color='#374151')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        ax.grid(alpha=0.2, color='#d1d5db', linestyle='--')

        # Label positioning with offset to reduce overlap
        if point_labels:
            offsets = [(7, 7), (-7, 7), (7, -7), (-7, -7)]
            for i, label in enumerate(point_labels[:len(x)]):
                offset = offsets[i % len(offsets)]
                ax.annotate(str(label)[:15], (x[i], y[i]), textcoords="offset points",
                            xytext=offset, fontsize=6.5, color='#4b5563',
                            fontweight='medium',
                            arrowprops=dict(arrowstyle='-', color='#d1d5db', lw=0.5)
                            if len(point_labels) <= 12 else None)

    @classmethod
    def _render_pie(cls, ax, data: dict, x_label: str, y_label: str):
        """Render a pie chart from spec data."""
        labels = data.get('labels', [])
        values = data.get('values', [])
        colors = data.get('colors', cls.CHART_COLORS['palette'][:len(labels)])

        if not labels or not values:
            return

        # Truncate long labels
        fmt = cls._auto_format_labels(labels)
        display_labels = fmt['labels']

        # Explode the largest slice for visual emphasis
        max_idx = values.index(max(values))
        explode = [0.05 if i == max_idx else 0 for i in range(len(values))]

        wedges, texts, autotexts = ax.pie(
            values, labels=display_labels, colors=colors[:len(display_labels)],
            autopct='%1.0f%%', pctdistance=0.78,
            textprops={'fontsize': 7.5}, startangle=90,
            explode=explode, shadow=True,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
        )
        for autotext in autotexts:
            autotext.set_fontsize(7)
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.axis('equal')


async def generate_concept_visualization(
    concept_name: str,
    concept_description: str,
    output_dir: str = "/tmp",
    **kwargs
) -> Optional[str]:
    """
    Generate an LLM-driven matplotlib visualization for a concept.

    Uses DSPy to ask the LLM what chart best illustrates the concept,
    then renders it with matplotlib. Returns None if the LLM decides
    no chart is appropriate (chart_type="none").

    Args:
        concept_name: Name of the concept
        concept_description: Description including why it matters
        output_dir: Directory to save the image

    Returns:
        Path to generated image or None
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import hashlib
        import time

        # Ask the LLM what visualization to produce
        spec_generator = dspy.ChainOfThought(VisualizationSpecSignature)
        spec = spec_generator(
            concept_name=concept_name,
            concept_description=concept_description
        )

        chart_type = str(spec.chart_type).strip().lower()

        # If LLM says no chart needed, skip
        if chart_type == 'none':
            logger.info(f"LLM decided no visualization needed for: {concept_name}")
            return None

        # Parse the data JSON
        try:
            data = json.loads(str(spec.data_json))
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Invalid data_json from LLM for {concept_name}, skipping visualization")
            return None

        # Create unique filename
        name_hash = hashlib.md5(f"{concept_name}_{time.time()}".encode()).hexdigest()[:8]
        output_path = Path(output_dir) / f"viz_{name_hash}.png"

        # Set up figure with breathing room
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        fig.patch.set_facecolor('#fafbfc')

        # Render the chart
        success = VisualizationRenderer.render(
            ax=ax,
            chart_type=chart_type,
            data=data,
            title=str(spec.title),
            x_label=str(spec.x_label),
            y_label=str(spec.y_label),
            annotation=str(spec.annotation)
        )

        if not success:
            plt.close()
            logger.warning(f"Unsupported chart type '{chart_type}' for {concept_name}")
            return None

        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.savefig(output_path, bbox_inches='tight', facecolor='#fafbfc', edgecolor='none')
        plt.close()

        logger.info(f"Generated {chart_type} visualization: {output_path}")
        return str(output_path)

    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return None
    except Exception as e:
        logger.error(f"Visualization generation failed for {concept_name}: {e}")
        return None


async def render_latex_to_image(
    latex_expr: str,
    output_dir: str = "/tmp"
) -> Optional[str]:
    """
    Render a LaTeX expression to an image using matplotlib.

    Args:
        latex_expr: LaTeX expression (without $ delimiters)
        output_dir: Directory to save the image

    Returns:
        Path to generated image or None
    """
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path
        import hashlib

        # Create unique filename
        expr_hash = hashlib.md5(latex_expr.encode()).hexdigest()[:8]
        output_path = Path(output_dir) / f"math_{expr_hash}.png"

        # Check if already exists
        if output_path.exists():
            return str(output_path)

        # Create figure for math rendering
        fig, ax = plt.subplots(figsize=(6, 1), dpi=150)
        ax.axis('off')

        # Render LaTeX
        ax.text(0.5, 0.5, f'${latex_expr}$',
                fontsize=14,
                ha='center', va='center',
                transform=ax.transAxes,
                usetex=False,  # Use mathtext instead of full LaTeX
                math_fontfamily='cm')

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', facecolor='white',
                    edgecolor='none', pad_inches=0.1, transparent=False)
        plt.close()

        return str(output_path)

    except Exception as e:
        logger.debug(f"LaTeX rendering failed: {e}")
        return None


def _convert_markdown_content(text: str) -> str:
    """Convert basic markdown to HTML with math and code support."""
    import re
    import uuid

    if not text:
        return ""

    # Store for preserved blocks (code, math) that shouldn't have <br/> added
    preserved = {}
    preserve_counter = [0]

    def preserve(content: str) -> str:
        """Store content and return placeholder that won't be matched by other patterns."""
        # Use a format that won't be matched by math/code patterns
        key = f"⟦BLOCK{preserve_counter[0]}⟧"
        preserve_counter[0] += 1
        preserved[key] = content
        return key

    # =================================================================
    # STEP 1: Clean up malformed code blocks
    # =================================================================
    # Remove standalone language identifiers that got separated from code fences
    lang_ids = r'python|javascript|js|typescript|ts|java|cpp|c\+\+|ruby|go|rust|bash|sh|sql|html|css'
    text = re.sub(rf'\n\s*({lang_ids})\s*\n', '\n', text)
    text = re.sub(rf'^\s*({lang_ids})\s*$', '', text, flags=re.MULTILINE)
    # Also handle "Code:\npython\n" pattern where language id follows a header
    text = re.sub(rf'(:\s*)\n\s*({lang_ids})\s*\n', r'\1\n', text)

    # =================================================================
    # STEP 2: Handle code blocks (preserve with placeholders)
    # =================================================================
    def replace_code_block(match):
        """Handle code blocks - extract and highlight code content."""
        full_match = match.group(0)
        # Extract code after optional language specifier
        code = re.sub(r'^```\w*\n?', '', full_match)
        code = re.sub(r'\n?```$', '', code)
        code = code.strip()
        if not code:
            return ''
        code = _apply_syntax_highlighting(code)
        return preserve(f'<pre><code>{code}</code></pre>')

    # Match fenced code blocks ```...```
    text = re.sub(r'```[\w]*\n.*?```', replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', replace_code_block, text, flags=re.DOTALL)

    # Also detect unfenced code patterns (4+ space indent or common code patterns)
    def detect_unfenced_code(match):
        """Detect code that wasn't in fences."""
        code = match.group(1).strip()
        if not code:
            return match.group(0)
        code = _apply_syntax_highlighting(code)
        return preserve(f'<pre><code>{code}</code></pre>')

    # Pattern: Lines starting with def/class/import followed by code-like content
    text = re.sub(
        r'^((?:def |class |import |from |@\w+).*(?:\n(?:    |\t).*)*)',
        detect_unfenced_code, text, flags=re.MULTILINE
    )

    # Additional code patterns to detect:
    # Pattern: Assignment with function call like "result = function(args)"
    text = re.sub(
        r'^(\s*[a-z_][a-z0-9_]*\s*=\s*[a-z_][a-z0-9_]*\([^)]*\).*)$',
        detect_unfenced_code, text, flags=re.MULTILINE | re.IGNORECASE
    )

    # Pattern: Multiple consecutive lines that look like code (4+ space indent)
    def detect_indented_block(match):
        code = match.group(0).strip()
        if len(code.split('\n')) >= 2:  # At least 2 lines
            code = _apply_syntax_highlighting(code)
            return preserve(f'<pre><code>{code}</code></pre>')
        return match.group(0)

    text = re.sub(
        r'(?:^    .+$\n?){2,}',
        detect_indented_block, text, flags=re.MULTILINE
    )

    # Pattern: Lines with common code syntax (list comprehension, dict, etc.)
    code_patterns = [
        r'^\s*\[.+\s+for\s+.+\s+in\s+.+\]',  # List comprehension
        r'^\s*\{.+:\s*.+\s+for\s+.+\}',  # Dict comprehension
        r'^\s*if\s+.+:$',  # if statement
        r'^\s*for\s+.+\s+in\s+.+:$',  # for loop
        r'^\s*while\s+.+:$',  # while loop
        r'^\s*return\s+.+$',  # return statement
        r'^\s*print\(.+\)$',  # print call
    ]
    for pattern in code_patterns:
        text = re.sub(
            f'({pattern}(?:\n(?:    |\t).*)*)',
            detect_unfenced_code, text, flags=re.MULTILINE
        )

    # =================================================================
    # STEP 3: Handle Math expressions (preserve with placeholders)
    # =================================================================
    # Display math: $$...$$ -> styled div
    def replace_display_math(match):
        math = match.group(1).strip()
        math = math.replace('<', '&lt;').replace('>', '&gt;')
        return preserve(f'<div class="math-display">{math}</div>')

    text = re.sub(r'\$\$(.+?)\$\$', replace_display_math, text, flags=re.DOTALL)

    # Inline math: $...$ -> styled span (but not currency like $100)
    def replace_inline_math(match):
        math = match.group(1).strip()
        # Skip if it looks like currency (just numbers)
        if re.match(r'^[\d,\.]+$', math):
            return match.group(0)
        math = math.replace('<', '&lt;').replace('>', '&gt;')
        return f'<span class="math-inline">{math}</span>'

    text = re.sub(r'\$([^$\n]+?)\$', replace_inline_math, text)

    # Also handle \( ... \) and \[ ... \] LaTeX delimiters
    text = re.sub(r'\\\[(.+?)\\\]', replace_display_math, text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', replace_inline_math, text)

    # Detect Unicode math and common equation patterns (not in LaTeX format)
    # Math symbols
    math_symbols = 'πΣ∇∫αβγδεθλμσωΔΩ∞≈≠≤≥±×÷√∂∈∉⊂⊃∪∩→←↔⇒⇐⇔'
    math_symbols_pattern = f'[{re.escape(math_symbols)}]'

    # Track what we've already wrapped
    def wrap_math_expr(match):
        eq = match.group(1) if match.lastindex and match.group(1) else match.group(0)
        # Don't wrap if already wrapped or is a placeholder
        if 'math-inline' in eq or '⟦BLOCK' in eq or '<span' in eq:
            return match.group(0)
        return preserve(f'<span class="math-inline">{eq}</span>')

    # PATTERN 1: Full equation with = sign and any math symbol
    # e.g., "G = Σ R(s,a) × π(a|s)"
    text = re.sub(
        rf'([A-Za-z_]+\s*=\s*[^<\n]*[{re.escape(math_symbols)}][^<\n]*)',
        wrap_math_expr, text
    )

    # PATTERN 2: Function notation with math symbols
    # e.g., "π(a|s)", "R(s,a)", "∇f(x)"
    text = re.sub(
        rf'({math_symbols_pattern}\s*\([^)]+\))',
        wrap_math_expr, text
    )
    text = re.sub(
        rf'([A-Za-z]_?\w*\([^)]*[,|][^)]*\))',  # Function with comma or pipe args
        wrap_math_expr, text
    )

    # PATTERN 3: Greek letters and math symbols standalone
    # e.g., "π", "Σ", "∇G"
    text = re.sub(
        rf'({math_symbols_pattern}[A-Za-z_]*)',
        wrap_math_expr, text
    )

    # PATTERN 4: Subscript/superscript notation
    # e.g., "x_i", "a_1", "π_new"
    text = re.sub(
        r'(?<![a-zA-Z])([a-zA-Z]+_(?:new|old|[a-z0-9]+))(?![a-zA-Z_])',
        wrap_math_expr, text
    )

    # PATTERN 5: Common math expressions without Greek letters
    # e.g., "R(s,a)", function notation
    text = re.sub(
        r'(?<![a-zA-Z])([A-Z]\([a-z],[a-z]\))(?![a-zA-Z])',
        wrap_math_expr, text
    )

    # =================================================================
    # STEP 4: Inline code (after code blocks to avoid conflicts)
    # =================================================================
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # =================================================================
    # STEP 5: Bold text before colon (Step 1:, Example:, etc.)
    # =================================================================
    # Pattern: Start of line or after break, word(s) followed by colon
    text = re.sub(r'^(\s*)([A-Z][^:\n]{0,40}):', r'\1<strong>\2:</strong>', text, flags=re.MULTILINE)
    # Also handle numbered steps like "1. Step name:" or "Step 1:"
    text = re.sub(r'(\d+[\.\)]\s*)([^:\n]{1,40}):', r'\1<strong>\2:</strong>', text)

    # =================================================================
    # STEP 6: Standard markdown formatting
    # =================================================================
    # Bold **text**
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)

    # Italic *text*
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)

    # Headers (h4, h5 for nested content)
    text = re.sub(r'^#### (.+)$', r'<h5>\1</h5>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)

    # Lists
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', text, flags=re.MULTILINE)

    # =================================================================
    # STEP 7: Paragraphs and line breaks
    # =================================================================
    # Paragraphs (double newline)
    text = re.sub(r'\n\n+', '</p><p>', text)

    # Single newlines to <br>
    text = text.replace('\n', '<br/>')

    # Clean up empty elements
    text = re.sub(r'<br/>\s*<br/>', '<br/>', text)
    text = re.sub(r'<p>\s*</p>', '', text)

    # Wrap in paragraph if not already
    if text and not text.strip().startswith('<'):
        text = f'<p>{text}</p>'

    # =================================================================
    # STEP 8: Restore preserved blocks
    # =================================================================
    for key, content in preserved.items():
        text = text.replace(key, content)

    return text


def generate_learning_html(
    paper_title: str,
    arxiv_id: str,
    authors: List[str],
    hook: str,
    concepts: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    key_insights: List[str],
    summary: str,
    next_steps: List[str],
    bingo_word: str = "Bingo!",
    learning_time: str = "20-30 minutes",
    total_words: int = 0
) -> str:
    """
    Generate professional HTML for learning content.

    Args:
        paper_title: Title of the paper
        arxiv_id: ArXiv ID
        authors: List of author names
        hook: Why should you care section
        concepts: List of concept dicts with name, description, difficulty, why_it_matters
        sections: List of section dicts with title, content, level, has_bingo_moment
        key_insights: List of key insight strings
        summary: Summary text
        next_steps: List of next step strings
        bingo_word: Celebration word
        learning_time: Estimated learning time
        total_words: Total word count

    Returns:
        Complete HTML string
    """

    # Build concepts list HTML
    concepts_html = ""
    for c in concepts[:8]:  # Limit to 8 concepts
        difficulty = c.get('difficulty', 3)
        concepts_html += f'''
        <div class="concept-card">
            <h4>{c.get('name', 'Concept')} <span class="level-badge level-{difficulty}">L{difficulty}</span></h4>
            <p>{c.get('description', '')}</p>
            <div class="why-matters">Why it matters: {c.get('why_it_matters', '')}</div>
        </div>
        '''

    # Build sections HTML
    sections_html = ""
    for s in sections:
        level = s.get('level', 1)
        level_names = {1: 'Basics', 2: 'Intuition', 3: 'Math', 4: 'Application', 5: 'Deep Dive'}
        level_name = level_names.get(level, 'Section')

        content = s.get('content', '')
        # Convert markdown to HTML
        content = _convert_markdown_content(content)

        section_html = f'''
        <div class="section-block avoid-break">
            <h3>{s.get('title', 'Section')} <span class="level-badge level-{level}">{level_name}</span></h3>
            <div class="section-content">{content}</div>
        '''

        if s.get('has_bingo_moment'):
            section_html += f'''
            <div class="bingo-box">
                <p>This is a key insight that connects everything together!</p>
            </div>
            '''

        if s.get('code_example'):
            code = s['code_example']
            # Strip markdown code block syntax if present
            import re
            code = re.sub(r'^```\w*\n?', '', code)  # Remove opening ```python
            code = re.sub(r'\n?```$', '', code)     # Remove closing ```
            code = code.strip()
            if code:
                # Apply syntax highlighting
                code = _apply_syntax_highlighting(code)
                section_html += f'''
                <pre><code>{code}</code></pre>
                '''

        # Add visualization if available
        if s.get('visualization_path'):
            # Use file:// protocol for local paths
            viz_path = s['visualization_path']
            if viz_path.startswith('/'):
                viz_path = f'file://{viz_path}'
            section_html += f'''
            <div class="visualization">
                <img src="{viz_path}" alt="Concept visualization" style="max-width: 100%; height: auto;"/>
            </div>
            '''

        section_html += '</div>'
        sections_html += section_html

    # Build insights HTML
    insights_html = ""
    if key_insights:
        insights_items = "\n".join([f"<li>{insight}</li>" for insight in key_insights[:6]])
        insights_html = f'''
        <div class="insights-list">
            <h4>✨ Key Insights ({bingo_word}!)</h4>
            <ol>{insights_items}</ol>
        </div>
        '''

    # Build next steps HTML
    next_html = ""
    if next_steps:
        next_items = "\n".join([f"<li>{step}</li>" for step in next_steps[:5]])
        next_html = f'''
        <div class="next-steps">
            <h4>📈 What to Learn Next</h4>
            <ul>{next_items}</ul>
        </div>
        '''

    # Build full HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{paper_title}</title>
    <style>
{LEARNING_CSS}
    </style>
</head>
<body>
    <!-- Cover Page -->
    <div class="cover-page">
        <h1>{paper_title}</h1>
        <div class="arxiv-id">arXiv:{arxiv_id}</div>
        <div class="authors">{', '.join(authors[:4])}</div>

        <div class="learning-badge">📚 Interactive Learning Guide</div>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{len(concepts)}</div>
                <div class="stat-label">Concepts</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(key_insights)}</div>
                <div class="stat-label">{bingo_word} Moments</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{learning_time}</div>
                <div class="stat-label">Learning Time</div>
            </div>
        </div>
    </div>

    <div class="page-break"></div>

    <!-- Hook Section -->
    <div class="hook-section">
        <h3>🎯 Why Should You Care?</h3>
        <p>{hook if hook else 'Discover cutting-edge research that could change how you think about AI.'}</p>
    </div>

    <!-- Concepts Overview -->
    <h2>🧠 Concepts You'll Learn</h2>
    {concepts_html}

    <div class="page-break"></div>

    <!-- Main Content Sections -->
    <h2>📖 Learning Journey</h2>
    {sections_html}

    <!-- Key Insights -->
    {insights_html}

    <!-- Summary -->
    <div class="summary-box">
        <h4>📝 Summary</h4>
        <p>{summary if summary else 'This paper introduces innovative concepts that build on existing research to push the boundaries of what is possible.'}</p>
    </div>

    <!-- Next Steps -->
    {next_html}

    <!-- Footer -->
    <div style="margin-top: 30pt; padding-top: 12pt; border-top: 1px solid #e5e7eb; font-size: 9pt; color: #6b7280; text-align: center;">
        Generated by Jotty ArxivLearningSwarm | {total_words} words | {learning_time} estimated learning time
    </div>
</body>
</html>'''

    return html


async def convert_learning_to_pdf(
    paper_title: str,
    arxiv_id: str,
    authors: List[str],
    hook: str,
    concepts: List[Dict[str, Any]],
    sections: List[Dict[str, Any]],
    key_insights: List[str],
    summary: str,
    next_steps: List[str],
    output_path: str,
    bingo_word: str = "Bingo!",
    learning_time: str = "20-30 minutes",
    total_words: int = 0
) -> str:
    """
    Convert learning content to PDF.

    Returns:
        Path to generated PDF
    """
    # Generate HTML
    html_content = generate_learning_html(
        paper_title=paper_title,
        arxiv_id=arxiv_id,
        authors=authors,
        hook=hook,
        concepts=concepts,
        sections=sections,
        key_insights=key_insights,
        summary=summary,
        next_steps=next_steps,
        bingo_word=bingo_word,
        learning_time=learning_time,
        total_words=total_words
    )

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Try WeasyPrint
    try:
        from weasyprint import HTML

        HTML(string=html_content).write_pdf(str(output_file))
        logger.info(f"✅ Generated PDF: {output_file}")
        return str(output_file)

    except ImportError:
        logger.warning("WeasyPrint not available, trying pdfkit...")

    # Try pdfkit
    try:
        import pdfkit

        options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '15mm',
            'margin-bottom': '25mm',
            'margin-left': '15mm',
            'encoding': 'UTF-8',
        }

        pdfkit.from_string(html_content, str(output_file), options=options)
        logger.info(f"✅ Generated PDF: {output_file}")
        return str(output_file)

    except ImportError:
        logger.warning("pdfkit not available, saving HTML...")

    # Fallback: Save HTML
    html_path = str(output_file.with_suffix('.html'))
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"📄 Saved HTML (PDF libraries not available): {html_path}")
    return html_path


__all__ = [
    'generate_learning_html',
    'convert_learning_to_pdf',
    'generate_concept_visualization',
    'VisualizationSpecSignature',
    'VisualizationRenderer',
    'LEARNING_CSS',
    'LEARNING_COLORS',
]
