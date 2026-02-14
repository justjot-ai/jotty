"""
JustJot Section Type Definitions

Registry-driven section types that load dynamically from JustJot.ai's
section-registry.ts - the single source of truth.

This ensures LIDA can work with ALL JustJot section types and their schemas.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


# ============================================
# Registry-Driven Section Types
# ============================================

def get_section_registry():
    """Lazy import of section registry to avoid circular imports."""
    from .section_registry import get_registry
    return get_registry()


def get_all_section_types() -> Dict[str, Any]:
    """Get all section types from the JustJot registry."""
    return get_section_registry().get_all_types()


def get_section_type_info(type_id: str) -> Optional[Any]:
    """Get detailed info for a specific section type."""
    return get_section_registry().get_type(type_id)


def get_section_schema(type_id: str) -> Optional[str]:
    """Get the content schema for a section type."""
    return get_section_registry().get_schema(type_id)


def get_section_content_type(type_id: str) -> str:
    """Get the content type (json, text, markdown, code) for a section type."""
    return get_section_registry().get_content_type(type_id)


def get_sections_by_category(category: str) -> List[Any]:
    """Get all section types in a category."""
    return get_section_registry().get_types_by_category(category)


def get_all_categories() -> List[str]:
    """Get all available categories."""
    return get_section_registry().get_categories()


# ============================================
# Chart Types (from JustJot chartTypes.ts)
# ============================================

class ChartType(str, Enum):
    """JustJot chart types for the 'chart' section type."""
    BAR = "bar"
    COLUMN = "column"
    LINE = "line"
    AREA = "area"
    PIE = "pie"
    DOUGHNUT = "doughnut"
    SCATTER = "scatter"
    RADAR = "radar"
    COMBO = "combo"
    HEATMAP = "heatmap"
    FUNNEL = "funnel"
    GAUGE = "gauge"


LIDA_CHART_TYPE_MAP = {
    "bar": ChartType.BAR,
    "bar chart": ChartType.BAR,
    "column": ChartType.COLUMN,
    "column chart": ChartType.COLUMN,
    "line": ChartType.LINE,
    "line chart": ChartType.LINE,
    "area": ChartType.AREA,
    "area chart": ChartType.AREA,
    "pie": ChartType.PIE,
    "pie chart": ChartType.PIE,
    "doughnut": ChartType.DOUGHNUT,
    "donut": ChartType.DOUGHNUT,
    "scatter": ChartType.SCATTER,
    "scatter plot": ChartType.SCATTER,
    "scatterplot": ChartType.SCATTER,
    "radar": ChartType.RADAR,
    "heatmap": ChartType.HEATMAP,
    "heat map": ChartType.HEATMAP,
    "histogram": ChartType.BAR,
    "box": ChartType.BAR,
    "boxplot": ChartType.BAR,
    "funnel": ChartType.FUNNEL,
    "gauge": ChartType.GAUGE,
}


def map_lida_chart_type(lida_type: str) -> ChartType:
    """Map LIDA visualization type to JustJot ChartType."""
    if not lida_type:
        return ChartType.BAR
    normalized = lida_type.lower().strip()
    return LIDA_CHART_TYPE_MAP.get(normalized, ChartType.BAR)


# ============================================
# Chart Data Structures
# ============================================

@dataclass
class Dataset:
    """Chart dataset - single data series."""
    id: str
    label: str
    values: List[Union[float, int, None]]
    color: Optional[str] = None
    type: Optional[str] = None  # For combo charts

    def to_dict(self) -> Dict:
        # Normalize values: convert floats to int when they are whole numbers
        normalized_values = []
        for v in self.values:
            if v is None:
                normalized_values.append(None)
            elif isinstance(v, float) and v == int(v):
                normalized_values.append(int(v))
            else:
                normalized_values.append(v)

        result = {"id": self.id, "label": self.label, "values": normalized_values}
        if self.color:
            result["color"] = self.color
        if self.type:
            result["type"] = self.type
        return result


@dataclass
class ChartData:
    """Chart data structure with labels and datasets."""
    labels: List[str]
    datasets: List[Dataset]

    def to_dict(self) -> Dict:
        return {
            "labels": self.labels,
            "datasets": [ds.to_dict() for ds in self.datasets]
        }


@dataclass
class TitleConfig:
    text: str
    fontSize: int = 16
    color: Optional[str] = None

    def to_dict(self) -> Dict:
        result = {"text": self.text, "fontSize": self.fontSize}
        if self.color:
            result["color"] = self.color
        return result


@dataclass
class AxisConfig:
    label: str
    show: bool = True

    def to_dict(self) -> Dict:
        return {"label": self.label, "show": self.show}


@dataclass
class LegendConfig:
    show: bool = True
    position: str = "bottom"

    def to_dict(self) -> Dict:
        return {"show": self.show, "position": self.position}


@dataclass
class AnimationConfig:
    enabled: bool = True
    duration: int = 300

    def to_dict(self) -> Dict:
        return {"enabled": self.enabled, "duration": self.duration}


@dataclass
class ReferenceLine:
    """Reference/threshold line on chart."""
    id: str
    axis: str
    value: float
    label: Optional[str] = None
    color: Optional[str] = "#ff0000"
    strokeDasharray: Optional[str] = "5,5"

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "axis": self.axis, "value": self.value,
            "label": self.label, "color": self.color,
            "strokeDasharray": self.strokeDasharray
        }


@dataclass
class ChartAnnotations:
    """Chart annotations - reference lines, areas, text."""
    referenceLines: List[ReferenceLine] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"referenceLines": [rl.to_dict() for rl in self.referenceLines]}


@dataclass
class ChartCustomization:
    """Chart customization options."""
    colors: Optional[List[str]] = None
    useGradient: bool = False
    title: Optional[TitleConfig] = None
    xAxis: Optional[AxisConfig] = None
    yAxis: Optional[AxisConfig] = None
    legend: Optional[LegendConfig] = None
    animation: Optional[AnimationConfig] = None
    annotations: Optional[ChartAnnotations] = None

    def to_dict(self) -> Dict:
        result = {"useGradient": self.useGradient}
        if self.colors:
            result["colors"] = self.colors
        if self.title:
            result["title"] = self.title.to_dict()
        if self.xAxis:
            result["xAxis"] = self.xAxis.to_dict()
        if self.yAxis:
            result["yAxis"] = self.yAxis.to_dict()
        if self.legend:
            result["legend"] = self.legend.to_dict()
        if self.animation:
            result["animation"] = self.animation.to_dict()
        if self.annotations:
            result["annotations"] = self.annotations.to_dict()
        return result


# ============================================
# Section Content Classes
# ============================================

@dataclass
class ChartSectionContent:
    """JustJot Chart Section Content (V2 Schema)."""
    version: int = 2
    type: ChartType = ChartType.BAR
    title: Optional[str] = None
    data: ChartData = None
    customization: Optional[ChartCustomization] = None
    metadata: Optional[Dict] = None

    def to_json(self) -> str:
        result = {
            "version": self.version,
            "type": self.type.value if isinstance(self.type, ChartType) else self.type,
        }
        if self.title:
            result["title"] = self.title
        if self.data:
            result["data"] = self.data.to_dict()
        if self.customization:
            result["customization"] = self.customization.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return json.dumps(result)


@dataclass
class DataTableSectionContent:
    """JustJot Data Table Section Content (CSV format)."""
    csv_content: str = ""

    def to_content(self) -> str:
        return self.csv_content


@dataclass
class CodeSectionContent:
    """JustJot Code Section Content."""
    code: str = ""
    language: str = "python"

    def to_content(self) -> str:
        return self.code


@dataclass
class TextSectionContent:
    """JustJot Text/Markdown Section Content."""
    markdown: str = ""

    def to_content(self) -> str:
        return self.markdown


@dataclass
class HTMLSectionContent:
    """JustJot HTML Section Content for interactive visualizations."""
    html: str = ""

    def to_content(self) -> str:
        return self.html


# ============================================
# JustJot Section and Idea Structures
# ============================================

@dataclass
class JustJotSection:
    """
    A single JustJot section ready to be added to an idea.

    The type can be ANY valid section type from the registry.
    """
    title: str
    type: str  # Any section type from registry (e.g., 'chart', 'kanban-board', 'timeline')
    content: str  # JSON string or plain text depending on type's contentType

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "type": self.type,
            "content": self.content
        }

    @classmethod
    def create(cls, title: str, section_type: str, content: Any) -> 'JustJotSection':
        """
        Create a section with automatic content serialization based on type.

        Args:
            title: Section title
            section_type: Section type ID from registry
            content: Content (will be JSON serialized if needed)
        """
        content_type = get_section_content_type(section_type)

        if isinstance(content, str):
            content_str = content
        elif content_type == 'json':
            content_str = json.dumps(content) if not isinstance(content, str) else content
        else:
            content_str = str(content)

        return cls(title=title, type=section_type, content=content_str)


@dataclass
class JustJotIdea:
    """A complete JustJot idea with multiple sections."""
    title: str
    description: Optional[str] = None
    sections: List[JustJotSection] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    template_name: str = "Blank"
    status: str = "Draft"

    def add_section(self, section: JustJotSection) -> None:
        self.sections.append(section)

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "description": self.description,
            "sections": [s.to_dict() for s in self.sections],
            "tags": self.tags,
            "templateName": self.template_name,
            "status": self.status
        }


# ============================================
# Section Type Context for LLM
# ============================================

def get_section_types_context() -> str:
    """
    Generate a comprehensive context string about all JustJot section types.

    This can be included in LLM prompts to help LIDA choose the best
    section type for any visualization task.
    """
    registry = get_section_registry()
    all_types = registry.get_all_types()
    categories = registry.get_categories()

    context_lines = [
        "# JustJot Section Types Reference",
        "",
        "Available section types for creating ideas and visualizations:",
        ""
    ]

    for category in sorted(categories):
        types_in_cat = registry.get_types_by_category(category)
        if types_in_cat:
            context_lines.append(f"## {category}")
            context_lines.append("")
            for t in types_in_cat:
                context_lines.append(f"### {t.value} ({t.label})")
                context_lines.append(f"- **Icon**: {t.icon}")
                context_lines.append(f"- **Description**: {t.description}")
                context_lines.append(f"- **Content Type**: {t.content_type}")
                if t.content_schema:
                    schema_preview = t.content_schema[:200].replace('\n', '\\n')
                    context_lines.append(f"- **Schema Example**: `{schema_preview}...`")
                context_lines.append("")

    return "\n".join(context_lines)


def get_visualization_types_context() -> str:
    """
    Generate context specifically for visualization-relevant section types.
    """
    registry = get_section_registry()
    viz_types = registry.get_visualization_types()

    context_lines = [
        "# Visualization Section Types",
        "",
        "Section types most suitable for data visualization and analysis:",
        ""
    ]

    for t in viz_types:
        context_lines.append(f"## {t.value}")
        context_lines.append(f"- **Label**: {t.label}")
        context_lines.append(f"- **Category**: {t.category}")
        context_lines.append(f"- **Description**: {t.description}")
        context_lines.append(f"- **Content Type**: {t.content_type}")
        if t.content_schema:
            context_lines.append(f"- **Schema**:")
            context_lines.append(f"```")
            context_lines.append(t.content_schema[:500])
            context_lines.append(f"```")
        context_lines.append("")

    return "\n".join(context_lines)


def suggest_section_type(
    data_type: str = None,
    visualization_goal: str = None,
    is_interactive: bool = False,
    has_hierarchy: bool = False,
    has_timeline: bool = False,
    has_network: bool = False,
    is_tabular: bool = False,
    is_code: bool = False,
) -> str:
    """
    Suggest the best JustJot section type based on content characteristics.

    Args:
        data_type: Type of data (numeric, categorical, text, etc.)
        visualization_goal: Goal (compare, trend, distribution, relationship, etc.)
        is_interactive: Whether interactive visualization is needed
        has_hierarchy: Whether data has hierarchical structure
        has_timeline: Whether data has temporal component
        has_network: Whether data represents relationships/networks
        is_tabular: Whether data is best shown as a table
        is_code: Whether content is code

    Returns:
        Recommended section type ID
    """
    # Network/relationship data
    if has_network:
        return "network-graph"

    # Timeline data
    if has_timeline:
        return "timeline"

    # Hierarchical data
    if has_hierarchy:
        return "mindmap"

    # Code content
    if is_code:
        return "code"

    # Tabular data
    if is_tabular:
        return "data-table"

    # Interactive HTML (Plotly/Altair)
    if is_interactive:
        return "html"

    # Default to chart for numeric visualization
    if data_type in ['numeric', 'quantitative'] or visualization_goal:
        return "chart"

    # Text content
    return "text"


# ============================================
# Chart Data Normalization Utilities
# ============================================

def normalize_chart_value(value) -> None:
    """Normalize a chart value - convert float to int if whole number."""
    if value is None:
        return None
    if isinstance(value, float) and value == int(value):
        return int(value)
    return value


def normalize_chart_data(data: dict) -> dict:
    """
    Normalize chart data dict - ensures all values are properly typed.

    Converts floats like 520000.0 to integers 520000 for cleaner JSON.
    This is important for JustJot chart rendering compatibility.

    Args:
        data: Chart data dictionary with 'data.datasets[].values'

    Returns:
        Normalized chart data dictionary
    """
    if not isinstance(data, dict):
        return data

    if 'data' in data and 'datasets' in data['data']:
        for dataset in data['data']['datasets']:
            if 'values' in dataset:
                dataset['values'] = [normalize_chart_value(v) for v in dataset['values']]

    return data


# ============================================
# Default Colors
# ============================================

DEFAULT_COLORS = [
    "#3B82F6",  # Blue
    "#10B981",  # Green
    "#F59E0B",  # Amber
    "#EF4444",  # Red
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#F97316",  # Orange
]

DARK_THEME_COLORS = [
    "#60A5FA",  # Light Blue
    "#34D399",  # Light Green
    "#FBBF24",  # Light Amber
    "#F87171",  # Light Red
    "#A78BFA",  # Light Purple
    "#F472B6",  # Light Pink
]
