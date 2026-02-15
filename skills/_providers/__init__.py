"""
Skill Providers - Alternative Implementations

External integrations that provide skill capabilities through different backends:
- Streamlit, Morph - App building frameworks
- browser-use, OpenHands, Agent-S - Browser/terminal automation
- n8n, ActivePieces - Workflow automation

Moved from core/capabilities/skills/providers/ (Feb 2026) for better organization.
All skill-related code now lives in skills/.
"""

# Base classes come from infrastructure
from .._infrastructure import (
    CATEGORY_KEYWORDS,
    SKILL_CATEGORY_MAP,
    JottyDefaultProvider,
    ProviderCapability,
    ProviderRegistry,
    ProviderResult,
    ProviderSelector,
    SkillCategory,
    SkillProvider,
)

__all__ = [
    "ProviderCapability",
    "ProviderResult",
    "SkillCategory",
    "SkillProvider",
    "JottyDefaultProvider",
    "SKILL_CATEGORY_MAP",
    "CATEGORY_KEYWORDS",
    "ProviderRegistry",
    "ProviderSelector",
]
