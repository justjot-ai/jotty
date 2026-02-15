"""
PDF Templates System
====================

Multiple professional templates for different use cases:

1. Research Reports:
   - Goldman Sachs style (clean, minimal, navy)
   - Morgan Stanley style (modern, gradient headers)
   - CLSA style (detailed, data-rich)
   - Motilal Oswal style (colorful, Indian market focus)

2. Kids Books:
   - Storybook (colorful, large fonts, illustrations)
   - Educational (fun facts, quizzes)
   - Activity Book (interactive elements)

3. ML Backtest Reports:
   - Two Sigma (quantitative, data-driven)
   - Renaissance Technologies (mathematical precision)
   - AQR Capital (academic rigor)
   - Man Group (institutional quality)
   - Citadel (professional excellence)
   - Quantitative (default balanced style)

4. Business Documents:
   - Corporate Report (professional, formal)
   - Pitch Deck (modern, visual)
   - White Paper (technical, detailed)
"""

from .backtest_templates import (
    AQRTemplate,
    CitadelTemplate,
    ManGroupTemplate,
    QuantitativeTemplate,
    RenTechTemplate,
    TwoSigmaTemplate,
)
from .base_template import BaseTemplate, TemplateRegistry
from .kids_templates import ActivityBookTemplate, EducationalTemplate, StorybookTemplate
from .research_templates import (
    CLSATemplate,
    GoldmanSachsTemplate,
    MorganStanleyTemplate,
    MotilalOswalTemplate,
)

__all__ = [
    "BaseTemplate",
    "TemplateRegistry",
    # Research templates
    "GoldmanSachsTemplate",
    "MorganStanleyTemplate",
    "CLSATemplate",
    "MotilalOswalTemplate",
    # Kids templates
    "StorybookTemplate",
    "EducationalTemplate",
    "ActivityBookTemplate",
    # Backtest templates
    "TwoSigmaTemplate",
    "RenTechTemplate",
    "AQRTemplate",
    "ManGroupTemplate",
    "CitadelTemplate",
    "QuantitativeTemplate",
]
