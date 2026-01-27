"""
JustJot Section Registry Integration

Loads section type definitions directly from JustJot.ai's
section-registry.ts file - the single source of truth.

This ensures LIDA integration always stays in sync with
JustJot's available section types and their schemas.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# Path to JustJot.ai section registry (relative to this package)
# From: Jotty/core/semantic/visualization/justjot/section_registry.py
# To:   JustJot.ai/src/lib/section-registry.ts
# Path traversal: justjot -> visualization -> semantic -> core -> Jotty -> stock_market (6 parents)
JUSTJOT_REGISTRY_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / \
    "JustJot.ai/src/lib/section-registry.ts"


@dataclass
class SectionTypeDefinition:
    """
    A section type definition from JustJot registry.

    Matches the structure in section-registry.ts
    """
    value: str  # Section type ID (e.g., 'chart', 'recharts')
    label: str  # Human-readable label
    icon: str   # Emoji icon
    description: str
    category: str
    has_own_ui: bool
    content_type: str  # 'json', 'text', 'markdown', 'code'
    content_schema: str  # Example/template content

    def get_parsed_schema(self) -> Optional[Any]:
        """Get parsed schema (JSON parsed if applicable)."""
        if not self.content_schema:
            return None
        if self.content_type == 'json':
            try:
                return json.loads(self.content_schema)
            except json.JSONDecodeError:
                return None
        return self.content_schema

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'value': self.value,
            'label': self.label,
            'icon': self.icon,
            'description': self.description,
            'category': self.category,
            'hasOwnUI': self.has_own_ui,
            'contentType': self.content_type,
            'contentSchema': self.content_schema
        }


class JustJotSectionRegistry:
    """
    Section registry that loads directly from JustJot.ai's section-registry.ts.

    This is the SINGLE SOURCE OF TRUTH for section types.

    Usage:
        registry = JustJotSectionRegistry()
        registry.load()  # Loads from JustJot.ai file

        # Get specific type
        chart_type = registry.get_type('chart')

        # Get visualization-relevant types
        viz_types = registry.get_visualization_types()
    """

    # Categories relevant for LIDA visualization output
    VISUALIZATION_CATEGORIES = ['Diagrams', 'Data', 'Content', 'Planning & Tracking']

    # Section types most useful for LIDA output (only types that exist in registry)
    LIDA_PREFERRED_TYPES = [
        'chart',         # Chart.js visualizations (bar, line, pie, etc.)
        'data-table',    # AG Grid data table with sorting/filtering
        'csv',           # CSV table data
        'code',          # Code block for generated visualization code
        'text',          # Markdown text for insights
        'html',          # Custom HTML (for Plotly/Altair interactive)
        'json',          # JSON data structures
        'mermaid',       # Mermaid diagrams (flowcharts, etc.)
        'timeline',      # Timeline visualizations
        'network-graph', # Network/graph visualizations
        'mindmap',       # Mind map diagrams
    ]

    def __init__(self, registry_path: Path = None):
        """
        Initialize the registry.

        Args:
            registry_path: Path to section-registry.ts.
                          Defaults to JustJot.ai location.
        """
        self.registry_path = registry_path or JUSTJOT_REGISTRY_PATH
        self._types: Dict[str, SectionTypeDefinition] = {}
        self._categories: List[str] = []
        self._loaded = False

    def load(self) -> bool:
        """
        Load section types from JustJot.ai's section-registry.ts.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.registry_path.exists():
            logger.warning(f"Registry file not found: {self.registry_path}")
            return False

        try:
            content = self.registry_path.read_text(encoding='utf-8')
            self._parse_registry(content)
            self._loaded = True
            logger.info(f"Loaded {len(self._types)} section types from {self.registry_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return False

    def _parse_registry(self, content: str) -> None:
        """Parse the TypeScript registry file."""
        # Extract the SECTION_REGISTRY array
        # Pattern matches the array contents between [ and ] as const;
        array_pattern = r'export const SECTION_REGISTRY\s*=\s*\[(.*?)\]\s*as\s*const;'
        match = re.search(array_pattern, content, re.DOTALL)

        if not match:
            logger.error("Could not find SECTION_REGISTRY in file")
            return

        array_content = match.group(1)

        # Parse each object in the array
        # Pattern matches { ... } blocks
        object_pattern = r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        objects = re.findall(object_pattern, array_content)

        categories_seen = set()

        for obj_content in objects:
            try:
                type_def = self._parse_type_object(obj_content)
                if type_def:
                    self._types[type_def.value] = type_def
                    categories_seen.add(type_def.category)
            except Exception as e:
                logger.debug(f"Failed to parse object: {e}")
                continue

        self._categories = sorted(list(categories_seen))

    def _parse_type_object(self, obj_content: str) -> Optional[SectionTypeDefinition]:
        """Parse a single type definition object."""
        def extract_value(pattern: str, default: str = '') -> str:
            match = re.search(pattern, obj_content)
            if match:
                return match.group(1)
            return default

        # Extract each field
        value = extract_value(r"value:\s*['\"]([^'\"]+)['\"]")
        if not value:
            return None

        label = extract_value(r"label:\s*['\"]([^'\"]+)['\"]")
        icon = extract_value(r"icon:\s*['\"]([^'\"]+)['\"]")
        description = extract_value(r"description:\s*['\"]([^'\"]+)['\"]")
        category = extract_value(r"category:\s*['\"]([^'\"]+)['\"]")

        # hasOwnUI is a boolean
        has_own_ui_match = re.search(r"hasOwnUI:\s*(true|false)", obj_content)
        has_own_ui = has_own_ui_match.group(1) == 'true' if has_own_ui_match else False

        # contentType has 'as ContentType' suffix
        content_type = extract_value(r"contentType:\s*['\"]([^'\"]+)['\"]")

        # contentSchema can be empty string or complex JSON
        schema_match = re.search(r"contentSchema:\s*['\"](.*)['\"]\s*(?:,|\}|$)", obj_content, re.DOTALL)
        if schema_match:
            content_schema = schema_match.group(1)
            # Unescape JSON strings
            content_schema = content_schema.replace("\\'", "'").replace('\\"', '"')
        else:
            content_schema = ''

        return SectionTypeDefinition(
            value=value,
            label=label,
            icon=icon,
            description=description,
            category=category,
            has_own_ui=has_own_ui,
            content_type=content_type,
            content_schema=content_schema
        )

    def ensure_loaded(self) -> None:
        """Ensure registry is loaded."""
        if not self._loaded:
            self.load()

    def get_type(self, type_id: str) -> Optional[SectionTypeDefinition]:
        """Get a specific section type by ID."""
        self.ensure_loaded()
        return self._types.get(type_id)

    def get_all_types(self) -> Dict[str, SectionTypeDefinition]:
        """Get all loaded section types."""
        self.ensure_loaded()
        return self._types.copy()

    def get_categories(self) -> List[str]:
        """Get all categories."""
        self.ensure_loaded()
        return self._categories.copy()

    def get_types_by_category(self, category: str) -> List[SectionTypeDefinition]:
        """Get all types in a category."""
        self.ensure_loaded()
        return [t for t in self._types.values() if t.category == category]

    def get_visualization_types(self) -> List[SectionTypeDefinition]:
        """Get types relevant for visualization output."""
        self.ensure_loaded()
        return [
            t for t in self._types.values()
            if t.value in self.LIDA_PREFERRED_TYPES
        ]

    def get_content_type(self, type_id: str) -> str:
        """Get the content type for a section type."""
        type_def = self.get_type(type_id)
        return type_def.content_type if type_def else 'text'

    def get_schema(self, type_id: str) -> Optional[str]:
        """Get the content schema for a section type."""
        type_def = self.get_type(type_id)
        return type_def.content_schema if type_def else None

    def suggest_type_for_lida(
        self,
        chart_type: str = None,
        has_interactive_html: bool = False,
        library: str = None,
        data_heavy: bool = False
    ) -> str:
        """
        Suggest the best JustJot section type for LIDA output.

        Args:
            chart_type: LIDA chart type (bar, line, pie, etc.)
            has_interactive_html: Whether interactive HTML is available
            library: LIDA library used (plotly, matplotlib, etc.)
            data_heavy: Whether the output is data-focused

        Returns:
            Recommended section type ID
        """
        # Interactive HTML from Plotly/Altair
        if has_interactive_html and library in ['plotly', 'altair']:
            return 'html'

        # Simple chart types work well with recharts
        simple_types = ['bar', 'line', 'area', 'pie']
        if chart_type and chart_type.lower() in simple_types:
            return 'recharts'

        # Data-heavy output
        if data_heavy:
            return 'data-table'

        # Default to full chart schema
        return 'chart'


# ============================================
# Global Registry Instance
# ============================================

_registry: Optional[JustJotSectionRegistry] = None


def get_registry() -> JustJotSectionRegistry:
    """Get the global section registry instance (lazy-loaded)."""
    global _registry
    if _registry is None:
        _registry = JustJotSectionRegistry()
        _registry.load()
    return _registry


@lru_cache(maxsize=1)
def get_all_section_types() -> Dict[str, SectionTypeDefinition]:
    """Get all section types (cached)."""
    return get_registry().get_all_types()


@lru_cache(maxsize=128)
def get_section_schema(type_id: str) -> Optional[str]:
    """Get schema for a section type (cached)."""
    return get_registry().get_schema(type_id)


# ============================================
# LIDA Chart Type Mappings
# ============================================

# JustJot chart types (from chartTypes.ts)
JUSTJOT_CHART_TYPES = [
    'bar', 'column', 'line', 'area', 'pie', 'doughnut',
    'scatter', 'radar', 'combo', 'heatmap', 'funnel', 'gauge'
]

# LIDA to JustJot chart type mapping
LIDA_TO_JUSTJOT_CHART_MAP = {
    'bar': 'bar',
    'bar chart': 'bar',
    'column': 'column',
    'column chart': 'column',
    'line': 'line',
    'line chart': 'line',
    'area': 'area',
    'area chart': 'area',
    'pie': 'pie',
    'pie chart': 'pie',
    'doughnut': 'doughnut',
    'donut': 'doughnut',
    'scatter': 'scatter',
    'scatter plot': 'scatter',
    'scatterplot': 'scatter',
    'radar': 'radar',
    'heatmap': 'heatmap',
    'heat map': 'heatmap',
    'histogram': 'bar',
    'box': 'bar',
    'boxplot': 'bar',
    'funnel': 'funnel',
    'gauge': 'gauge',
}


def map_lida_to_justjot_chart_type(lida_type: str) -> str:
    """Map LIDA chart type to JustJot chart type."""
    if not lida_type:
        return 'bar'
    normalized = lida_type.lower().strip()
    return LIDA_TO_JUSTJOT_CHART_MAP.get(normalized, 'bar')


# ============================================
# Content Schema Builders (using registry)
# ============================================

class ContentSchemaBuilder:
    """
    Build content that matches JustJot section schemas.

    Uses the registry to validate against expected formats.
    """

    def __init__(self, registry: JustJotSectionRegistry = None):
        self.registry = registry or get_registry()

    def build_chart_v2(
        self,
        chart_type: str,
        labels: List[str],
        datasets: List[Dict],
        title: str = None,
        customization: Dict = None
    ) -> str:
        """
        Build V2 chart section content.

        Schema from registry: chart contentSchema
        """
        content = {
            "version": 2,
            "type": map_lida_to_justjot_chart_type(chart_type),
            "data": {
                "labels": labels,
                "datasets": datasets
            }
        }
        if title:
            content["title"] = title
        if customization:
            content["customization"] = customization
        return json.dumps(content)

    def build_recharts(
        self,
        chart_type: str,
        data: List[Dict],
        colors: List[str] = None
    ) -> str:
        """
        Build Recharts section content.

        Schema: {"type":"line","data":[{"name":"Jan","value":400}]}
        """
        # Map to recharts supported types
        recharts_types = {'bar', 'line', 'area', 'pie'}
        normalized_type = chart_type.lower() if chart_type else 'bar'
        if normalized_type not in recharts_types:
            normalized_type = 'bar'

        content = {
            "type": normalized_type,
            "data": data
        }
        if colors:
            content["colors"] = colors
        return json.dumps(content)

    def build_kpi_dashboard(self, kpis: List[Dict]) -> str:
        """
        Build KPI dashboard section content.

        Schema: {"kpis":[{"id":"1","name":"KPI","value":100,"target":150,"unit":"$","trend":"up"}]}
        """
        normalized = []
        for i, kpi in enumerate(kpis):
            normalized.append({
                "id": kpi.get("id", str(i + 1)),
                "name": kpi.get("name", f"KPI {i + 1}"),
                "value": kpi.get("value", 0),
                "target": kpi.get("target", 100),
                "unit": kpi.get("unit", ""),
                "trend": kpi.get("trend", "stable"),
                "period": kpi.get("period", "current")
            })
        return json.dumps({"kpis": normalized})

    def build_table(
        self,
        columns: List[Dict],
        rows: List[Dict]
    ) -> str:
        """
        Build interactive table section content.

        Schema: {"columns":[...],"rows":[...],"sortColumn":null,"sortDirection":"asc"}
        """
        return json.dumps({
            "columns": columns,
            "rows": rows,
            "sortColumn": None,
            "sortDirection": "asc"
        })


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
