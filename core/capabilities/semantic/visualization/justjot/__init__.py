"""
LIDA-JustJot Integration Package

Transforms LIDA visualization outputs into JustJot section format,
enabling seamless integration of AI-generated visualizations into
JustJot ideas.

This package provides:
- Section registry loader (single source of truth from JustJot.ai)
- Section types and content classes for all JustJot renderers
- Transformers for chart, code, text, data sections
- Idea builder for creating complete visualization ideas

Example:
    from core.semantic.visualization import VisualizationLayer
    from core.semantic.visualization.justjot import (
        JustJotIdeaBuilder,
        get_all_section_types,
        get_section_types_context,
    )

    # Get available section types
    all_types = get_all_section_types()
    print(f"Available: {len(all_types)} section types")

    # Create visualization
    viz = VisualizationLayer.from_dataframe(df)
    builder = JustJotIdeaBuilder(viz)

    # Create idea with sections
    idea = builder.create_visualization_idea(
        question="Analyze sales by region",
        include_data=True,
        include_code=True,
        include_insights=True
    )

    # Save directly to JustJot via MCP
    idea_id = builder.save_to_justjot(idea, user_id="user_xxx")
"""

# Section Registry (Single Source of Truth)
from .section_registry import (
    JustJotSectionRegistry,
    SectionTypeDefinition,
    get_registry,
    get_all_section_types as registry_get_all_types,
    get_section_schema as registry_get_schema,
    ContentSchemaBuilder,
    JUSTJOT_CHART_TYPES,
    LIDA_TO_JUSTJOT_CHART_MAP,
    map_lida_to_justjot_chart_type,
    DEFAULT_COLORS,
    DARK_THEME_COLORS,
)

# Section Types (Registry-Driven)
from .section_types import (
    # Registry access functions
    get_all_section_types,
    get_section_type_info,
    get_section_schema,
    get_section_content_type,
    get_sections_by_category,
    get_all_categories,
    # Context generators for LLM
    get_section_types_context,
    get_visualization_types_context,
    suggest_section_type,
    # Chart types
    ChartType,
    map_lida_chart_type,
    # Data structures
    Dataset,
    ChartData,
    TitleConfig,
    AxisConfig,
    LegendConfig,
    AnimationConfig,
    ReferenceLine,
    ChartAnnotations,
    ChartCustomization,
    # Content classes
    ChartSectionContent,
    DataTableSectionContent,
    CodeSectionContent,
    TextSectionContent,
    HTMLSectionContent,
    # Section and Idea
    JustJotSection,
    JustJotIdea,
    # Normalization utilities
    normalize_chart_value,
    normalize_chart_data,
)

# Transformers (Registry-Driven)
from .transformers import (
    SectionTransformer,
    transform_to_section,
    ChartTransformer,  # Specialized for LIDA code analysis
)

# Builder
from .idea_builder import (
    JustJotIdeaBuilder,
    VisualizationIdeaConfig,
    create_quick_visualization_idea,
    create_quick_dashboard_idea,
)

__all__ = [
    # Registry
    'JustJotSectionRegistry',
    'SectionTypeDefinition',
    'get_registry',
    'ContentSchemaBuilder',
    'JUSTJOT_CHART_TYPES',
    'LIDA_TO_JUSTJOT_CHART_MAP',
    'map_lida_to_justjot_chart_type',
    'DEFAULT_COLORS',
    'DARK_THEME_COLORS',
    # Section Type Access
    'get_all_section_types',
    'get_section_type_info',
    'get_section_schema',
    'get_section_content_type',
    'get_sections_by_category',
    'get_all_categories',
    # LLM Context
    'get_section_types_context',
    'get_visualization_types_context',
    'suggest_section_type',
    # Chart Types
    'ChartType',
    'map_lida_chart_type',
    # Data Structures
    'Dataset',
    'ChartData',
    'TitleConfig',
    'AxisConfig',
    'LegendConfig',
    'AnimationConfig',
    'ReferenceLine',
    'ChartAnnotations',
    'ChartCustomization',
    # Content Classes
    'ChartSectionContent',
    'DataTableSectionContent',
    'CodeSectionContent',
    'TextSectionContent',
    'HTMLSectionContent',
    # Section and Idea
    'JustJotSection',
    'JustJotIdea',
    # Transformers
    'SectionTransformer',
    'transform_to_section',
    'ChartTransformer',
    # Builder
    'JustJotIdeaBuilder',
    'VisualizationIdeaConfig',
    'create_quick_visualization_idea',
    'create_quick_dashboard_idea',
]
