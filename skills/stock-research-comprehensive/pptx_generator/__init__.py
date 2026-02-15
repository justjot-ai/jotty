"""
PowerPoint Generator for ArXiv Learning — package init.

All implementation lives in ``generator.py``.  This file re-exports
the public API so existing ``from Jotty.core.pptx_generator import X`` still works.
"""

from ..html_slide_generator import (  # noqa: F401
    HTMLSlideGenerator,
    LearningSlideBuilder,
    PresentationConfig,
    SlideType,
)
from .deck_judge import (  # noqa: F401
    AutoImprovementLoop,
    DeckJudge,
    DeckScore,
    DiagramDecisionEngine,
    DiagramType,
    analyze_and_decide_diagrams,
)
from .diagram_image_generator import DiagramImageGenerator, MermaidDiagramGenerator  # noqa: F401
from .generator import (  # noqa: F401
    convert_pptx_to_pdf,
    generate_all_formats,
    generate_and_improve_pptx,
    generate_intelligent_pptx,
    generate_learning_html,
    generate_learning_html_slides,
    generate_learning_pptx,
    is_libreoffice_available,
)
from .visualization_planner import (  # noqa: F401
    LIDAStylePlanner,
    VisualizationSpec,
    convert_specs_to_pptx_data,
)

__all__ = [
    "generate_learning_pptx",
    "convert_pptx_to_pdf",
    "is_libreoffice_available",
    "generate_intelligent_pptx",
    "generate_and_improve_pptx",
    "generate_learning_html_slides",
    "generate_learning_html",
    "generate_all_formats",
    "HTMLSlideGenerator",
    "LearningSlideBuilder",
    "PresentationConfig",
    "SlideType",
    "analyze_and_decide_diagrams",
    "DiagramDecisionEngine",
    "DiagramType",
    "DeckJudge",
    "DeckScore",
    "AutoImprovementLoop",
]
