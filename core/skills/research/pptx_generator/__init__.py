"""
PowerPoint Generator for ArXiv Learning — package init.

All implementation lives in ``generator.py``.  This file re-exports
the public API so existing ``from ...pptx_generator import X`` still works.
"""

from .generator import (                           # noqa: F401
    generate_learning_pptx,
    convert_pptx_to_pdf,
    is_libreoffice_available,
    generate_intelligent_pptx,
    generate_and_improve_pptx,
    generate_learning_html_slides,
    generate_learning_html,
    generate_all_formats,
)

from .deck_judge import (                          # noqa: F401
    DiagramType,
    DiagramDecisionEngine,
    analyze_and_decide_diagrams,
    DeckJudge,
    DeckScore,
    AutoImprovementLoop,
)

from .visualization_planner import (               # noqa: F401
    LIDAStylePlanner,
    VisualizationSpec,
    convert_specs_to_pptx_data,
)

from .diagram_image_generator import (             # noqa: F401
    DiagramImageGenerator,
    MermaidDiagramGenerator,
)

from ..html_slide_generator import (               # noqa: F401
    HTMLSlideGenerator,
    LearningSlideBuilder,
    PresentationConfig,
    SlideType,
)

__all__ = [
    'generate_learning_pptx', 'convert_pptx_to_pdf', 'is_libreoffice_available',
    'generate_intelligent_pptx', 'generate_and_improve_pptx',
    'generate_learning_html_slides', 'generate_learning_html', 'generate_all_formats',
    'HTMLSlideGenerator', 'LearningSlideBuilder', 'PresentationConfig', 'SlideType',
    'analyze_and_decide_diagrams', 'DiagramDecisionEngine', 'DiagramType',
    'DeckJudge', 'DeckScore', 'AutoImprovementLoop',
]
