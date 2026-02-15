"""
Skill Providers for Jotty V2
============================

A pluggable provider system that allows multiple implementations for the same
skill category (browser, terminal, computer-use). The system learns which
provider works best for each use case using swarm intelligence.

Providers:
- jotty (default): Built-in implementations
- n8n: Workflows as skills (N8N_BASE_URL, N8N_API_KEY)
- activepieces: Flows as skills (ACTIVEPIECES_BASE_URL, ACTIVEPIECES_API_KEY)
- browser-use: Web automation via browser-use library
- openhands: Terminal/code via OpenHands SDK
- agent-s: GUI/computer control via Agent-S
- open-interpreter: Local code execution
- skyvern: RPA and form automation

Usage:
    from core.skills.providers import ProviderRegistry, SkillCategory

    registry = ProviderRegistry()
    provider = registry.get_best_provider(SkillCategory.BROWSER, task="scrape website")
    result = await provider.execute(task)
"""

from .agent_s_provider import AgentSProvider
from .base import (
    CATEGORY_KEYWORDS,
    SKILL_CATEGORY_MAP,
    ContributedSkill,
    JottyDefaultProvider,
    ProviderCapability,
    ProviderResult,
    SkillCategory,
    SkillProvider,
)
from .browser_use_provider import BrowserUseCompositeProvider, BrowserUseProvider
from .composite_provider import (
    AutomateWorkflowProvider,
    FullStackAgentProvider,
    ResearchAndAnalyzeProvider,
)
from .morph_provider import MorphProvider
from .open_interpreter_provider import OpenInterpreterProvider
from .openhands_provider import OpenHandsProvider
from .provider_registry import ProviderPerformance, ProviderRegistry, ProviderSelector
from .streamlit_provider import StreamlitProvider

__all__ = [
    # Base
    "SkillProvider",
    "SkillCategory",
    "ProviderCapability",
    "ProviderResult",
    "ContributedSkill",
    "JottyDefaultProvider",
    "SKILL_CATEGORY_MAP",
    "CATEGORY_KEYWORDS",
    # Registry
    "ProviderRegistry",
    "ProviderSelector",
    "ProviderPerformance",
    # Providers
    "BrowserUseProvider",
    "BrowserUseCompositeProvider",
    "OpenHandsProvider",
    "AgentSProvider",
    "OpenInterpreterProvider",
    # Composite
    "ResearchAndAnalyzeProvider",
    "AutomateWorkflowProvider",
    "FullStackAgentProvider",
    # App Building (Streamlit is default - fully open source)
    "StreamlitProvider",
    "MorphProvider",  # Requires cloud credentials
]
