"""
LIDA to JustJot Transformers

Registry-driven transformers that convert LIDA outputs to any JustJot section type.

Primary transformer:
- SectionTransformer: Generic transformer for any section type using registry

Specialized transformer (for complex LIDA code analysis):
- ChartTransformer: Extracts chart data from LIDA-generated visualization code
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd

from .section_types import (
    JustJotSection,
    ChartSectionContent,
    ChartType,
    ChartData,
    Dataset,
    ChartCustomization,
    TitleConfig,
    AxisConfig,
    LegendConfig,
    map_lida_chart_type,
    DEFAULT_COLORS,
    get_section_content_type,
    get_section_type_info,
)

logger = logging.getLogger(__name__)


class SectionTransformer:
    """
    Registry-driven transformer for any JustJot section type.

    Uses the section registry to determine content type and
    properly serialize content.

    Example:
        transformer = SectionTransformer()

        # Create any section type
        section = transformer.transform('kanban-board', kanban_data, 'My Board')
        section = transformer.transform('timeline', timeline_data, 'Events')
        section = transformer.transform('swot', swot_data, 'Analysis')
        section = transformer.transform('text', '# Hello', 'Intro')
    """

    def transform(
        self,
        section_type: str,
        content: Any,
        title: str = None
    ) -> JustJotSection:
        """
        Transform content to any JustJot section type.

        Args:
            section_type: Section type ID from registry (e.g., 'chart', 'kanban-board')
            content: Content data (dict, list, str, or DataFrame)
            title: Section title

        Returns:
            JustJotSection ready for use
        """
        # Get section info from registry
        type_info = get_section_type_info(section_type)
        content_type = get_section_content_type(section_type)

        # Generate default title if not provided
        if not title:
            title = type_info.label if type_info else section_type.replace('-', ' ').title()

        # Serialize content based on registry's contentType
        content_str = self._serialize_content(content, content_type)

        return JustJotSection(
            title=title,
            type=section_type,
            content=content_str
        )

    def _serialize_content(self, content: Any, content_type: str) -> str:
        """Serialize content based on section's contentType."""
        # Already a string
        if isinstance(content, str):
            return content

        # DataFrame to CSV
        if hasattr(content, 'to_csv'):
            return content.to_csv(index=False)

        # JSON content type
        if content_type == 'json':
            return json.dumps(content)

        # Default string conversion
        return str(content)

    def transform_dataframe(
        self,
        df: pd.DataFrame,
        section_type: str = 'data-table',
        title: str = 'Data',
        max_rows: int = 100
    ) -> JustJotSection:
        """
        Transform DataFrame to a data section (csv or data-table).

        Args:
            df: pandas DataFrame
            section_type: 'csv' or 'data-table'
            title: Section title
            max_rows: Maximum rows to include

        Returns:
            JustJotSection with CSV content
        """
        display_df = df.head(max_rows)
        return self.transform(section_type, display_df, title)

    def transform_text(
        self,
        text: str,
        title: str = 'Text',
        as_markdown: bool = True
    ) -> JustJotSection:
        """Transform text content to text section."""
        content = text
        if as_markdown and text and not text.startswith('#'):
            content = f"## {title}\n\n{text}"
        return self.transform('text', content, title)

    def transform_code(
        self,
        code: str,
        title: str = 'Code',
        language: str = 'python'
    ) -> JustJotSection:
        """Transform code to code section."""
        # Clean code
        lines = code.split('\n')
        cleaned = [line for line in lines if line.strip() or lines.index(line) > 0]
        clean_code = '\n'.join(cleaned).strip()

        return self.transform('code', clean_code, f"{title} ({language})")

    def transform_html(
        self,
        html: str,
        title: str = 'Interactive'
    ) -> JustJotSection:
        """Transform HTML to html section (for Plotly/Altair)."""
        return self.transform('html', html, title)


class ChartTransformer:
    """
    Specialized transformer for LIDA chart visualization output.

    Analyzes LIDA-generated code to extract:
    - Chart type (bar, line, pie, etc.)
    - Data (labels, values)
    - Axis labels
    - Title

    This is needed because LIDA generates visualization code,
    not structured chart data.
    """

    def __init__(self, colors: List[str] = None) -> None:
        self.colors = colors or DEFAULT_COLORS

    def transform(
        self,
        chart_result: Any,
        df: pd.DataFrame,
        title: str = None,
        chart_type: str = None
    ) -> JustJotSection:
        """
        Transform LIDA ChartResult to JustJot chart section.

        Args:
            chart_result: LIDA ChartResult object with .code attribute
            df: Source DataFrame for data extraction
            title: Chart title (extracted from code if not provided)
            chart_type: Chart type hint (inferred from code if not provided)

        Returns:
            JustJotSection with V2 chart content
        """
        code = getattr(chart_result, 'code', '') or ''

        # Extract info from LIDA code
        extracted_type = chart_type or self._extract_chart_type(code)
        extracted_title = title or self._extract_title(code)
        extracted_data = self._extract_data_from_code(code, df)

        # Map to JustJot chart type
        jj_chart_type = map_lida_chart_type(extracted_type)

        # Build datasets
        datasets = []
        if extracted_data:
            labels = extracted_data.get('labels', [])
            for i, ds_data in enumerate(extracted_data.get('datasets', [])):
                datasets.append(Dataset(
                    id=str(i + 1),
                    label=ds_data.get('label', f'Series {i + 1}'),
                    values=ds_data.get('values', []),
                    color=self.colors[i % len(self.colors)]
                ))
            chart_data = ChartData(labels=labels, datasets=datasets)
        else:
            chart_data = self._create_fallback_data(df, jj_chart_type)

        # Create customization
        x_label, y_label = self._extract_axis_labels(code)
        customization = ChartCustomization(
            colors=self.colors[:len(datasets)] if datasets else self.colors[:1],
            title=TitleConfig(text=extracted_title) if extracted_title else None,
            xAxis=AxisConfig(label=x_label, show=True) if x_label else None,
            yAxis=AxisConfig(label=y_label, show=True) if y_label else None,
            legend=LegendConfig(show=len(datasets) > 1, position="bottom")
        )

        # Build content
        content = ChartSectionContent(
            version=2,
            type=jj_chart_type,
            title=extracted_title,
            data=chart_data,
            customization=customization,
            metadata={"source": "lida"}
        )

        return JustJotSection(
            title=extracted_title or "Visualization",
            type="chart",
            content=content.to_json()
        )

    def _extract_chart_type(self, code: str) -> str:
        """Extract chart type from LIDA-generated code."""
        patterns = [
            (r'px\.bar\s*\(', 'bar'),
            (r'px\.line\s*\(', 'line'),
            (r'px\.scatter\s*\(', 'scatter'),
            (r'px\.pie\s*\(', 'pie'),
            (r'px\.area\s*\(', 'area'),
            (r'px\.histogram\s*\(', 'bar'),
            (r'px\.box\s*\(', 'bar'),
            (r'px\.funnel\s*\(', 'funnel'),
            (r'go\.Heatmap', 'heatmap'),
            (r'\.plot\s*\(.*kind\s*=\s*[\'"]bar', 'bar'),
            (r'\.plot\s*\(.*kind\s*=\s*[\'"]line', 'line'),
            (r'\.plot\s*\(.*kind\s*=\s*[\'"]pie', 'pie'),
            (r'sns\.barplot', 'bar'),
            (r'sns\.lineplot', 'line'),
            (r'sns\.scatterplot', 'scatter'),
            (r'sns\.heatmap', 'heatmap'),
            (r'plt\.bar\s*\(', 'bar'),
            (r'plt\.plot\s*\(', 'line'),
            (r'plt\.scatter\s*\(', 'scatter'),
            (r'plt\.pie\s*\(', 'pie'),
        ]

        for pattern, chart_type in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return chart_type

        return 'bar'

    def _extract_title(self, code: str) -> Optional[str]:
        """Extract chart title from code."""
        patterns = [
            r'title\s*=\s*[\'"]([^\'"]+)[\'"]',
            r'\.set_title\s*\(\s*[\'"]([^\'"]+)[\'"]',
            r'plt\.title\s*\(\s*[\'"]([^\'"]+)[\'"]',
        ]
        for pattern in patterns:
            match = re.search(pattern, code)
            if match:
                return match.group(1)
        return None

    def _extract_axis_labels(self, code: str) -> tuple:
        """Extract x and y axis labels from code."""
        x_patterns = [
            r'xaxis_title\s*=\s*[\'"]([^\'"]+)[\'"]',
            r'x\s*=\s*[\'"]([^\'"]+)[\'"]',
            r'plt\.xlabel\s*\(\s*[\'"]([^\'"]+)[\'"]',
        ]
        y_patterns = [
            r'yaxis_title\s*=\s*[\'"]([^\'"]+)[\'"]',
            r'y\s*=\s*[\'"]([^\'"]+)[\'"]',
            r'plt\.ylabel\s*\(\s*[\'"]([^\'"]+)[\'"]',
        ]

        x_label = y_label = None
        for pattern in x_patterns:
            if match := re.search(pattern, code):
                x_label = match.group(1)
                break
        for pattern in y_patterns:
            if match := re.search(pattern, code):
                y_label = match.group(1)
                break

        return x_label, y_label

    def _extract_data_from_code(self, code: str, df: pd.DataFrame) -> Optional[Dict]:
        """Extract data from LIDA code by analyzing groupby and aggregation."""
        try:
            # Find groupby + aggregation pattern
            groupby_match = re.search(r'groupby\s*\(\s*[\'"]([^\'"]+)[\'"]', code)
            agg_match = re.search(r'\[[\'"]([^\'"]+)[\'"]\]\s*\.\s*(sum|mean|count|max|min)', code)

            if groupby_match and agg_match:
                group_col = groupby_match.group(1)
                agg_col = agg_match.group(1)
                agg_func = agg_match.group(2)

                if group_col in df.columns and agg_col in df.columns:
                    grouped = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                    return {
                        'labels': grouped[group_col].astype(str).tolist(),
                        'datasets': [{
                            'label': f'{agg_func.title()} of {agg_col}',
                            'values': grouped[agg_col].tolist()
                        }]
                    }

            # Try x, y column extraction
            x_match = re.search(r'x\s*=\s*[\'"]([^\'"]+)[\'"]', code)
            y_match = re.search(r'y\s*=\s*[\'"]([^\'"]+)[\'"]', code)

            if x_match and y_match:
                x_col, y_col = x_match.group(1), y_match.group(1)
                if x_col in df.columns and y_col in df.columns:
                    if df[x_col].dtype == 'object':
                        grouped = df.groupby(x_col)[y_col].sum().reset_index()
                        labels = grouped[x_col].astype(str).tolist()
                        values = grouped[y_col].tolist()
                    else:
                        labels = df[x_col].astype(str).tolist()[:50]
                        values = df[y_col].tolist()[:50]

                    return {
                        'labels': labels,
                        'datasets': [{'label': y_col, 'values': values}]
                    }

        except Exception as e:
            logger.warning(f"Failed to extract data from code: {e}")

        return None

    def _create_fallback_data(self, df: pd.DataFrame, chart_type: ChartType) -> ChartData:
        """Create fallback chart data from DataFrame."""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        num_cols = df.select_dtypes(include=['number']).columns

        if len(cat_cols) > 0 and len(num_cols) > 0:
            grouped = df.groupby(cat_cols[0])[num_cols[0]].sum().reset_index()
            labels = grouped[cat_cols[0]].astype(str).tolist()[:20]
            values = grouped[num_cols[0]].tolist()[:20]
        elif len(num_cols) > 0:
            labels = [f"Row {i+1}" for i in range(min(20, len(df)))]
            values = df[num_cols[0]].tolist()[:20]
        else:
            labels = ["A", "B", "C", "D"]
            values = [10, 20, 30, 40]

        return ChartData(
            labels=labels,
            datasets=[Dataset(id="1", label="Value", values=values, color=self.colors[0])]
        )


# ============================================
# Convenience Function
# ============================================

def transform_to_section(section_type: str, content: Any, title: str = None, **kwargs: Any) -> JustJotSection:
    """
    Transform any content to a JustJot section.

    Args:
        section_type: Any valid section type from registry
        content: Content (dict, list, str, DataFrame)
        title: Section title
        **kwargs: Additional options

    Returns:
        JustJotSection ready for use

    Example:
        # Create a SWOT analysis
        section = transform_to_section('swot', {
            'strengths': ['Fast', 'Reliable'],
            'weaknesses': ['Limited features'],
            'opportunities': ['New market'],
            'threats': ['Competition']
        }, 'Strategic Analysis')

        # Create a timeline
        section = transform_to_section('timeline', {
            'events': [{'date': '2024-01', 'title': 'Launch'}]
        }, 'Project Timeline')
    """
    return SectionTransformer().transform(section_type, content, title)


