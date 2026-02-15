"""
Skill Infrastructure - Provider Selection & Registry

Infrastructure for managing and selecting skill providers:
- Base classes for all providers
- Provider registry and selection logic
- Q-learning for provider optimization
- Skill category mapping

Moved from core/capabilities/skills/providers/ (Feb 2026) for better organization.
"""

from .base import (
    ProviderCapability,
    ProviderResult,
    SkillCategory,
    SkillProvider,
    JottyDefaultProvider,
    SKILL_CATEGORY_MAP,
    CATEGORY_KEYWORDS,
)

from .provider_registry import (
    ProviderRegistry,
    ProviderSelector,
    ProviderPerformance,
)

__all__ = [
    # Base classes
    "ProviderCapability",
    "ProviderResult",
    "SkillCategory",
    "SkillProvider",
    "JottyDefaultProvider",
    "SKILL_CATEGORY_MAP",
    "CATEGORY_KEYWORDS",
    # Registry
    "ProviderRegistry",
    "ProviderSelector",
    "ProviderPerformance",
]
