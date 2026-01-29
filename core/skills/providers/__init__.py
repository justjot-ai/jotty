"""
Skill Providers for Jotty V2
============================

A pluggable provider system that allows multiple implementations for the same
skill category (browser, terminal, computer-use). The system learns which
provider works best for each use case using swarm intelligence.

Providers:
- jotty (default): Built-in implementations
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

from .base import (
    SkillProvider,
    SkillCategory,
    ProviderCapability,
    ProviderResult,
    JottyDefaultProvider,
    SKILL_CATEGORY_MAP,
    CATEGORY_KEYWORDS,
)
from .provider_registry import (
    ProviderRegistry,
    ProviderSelector,
    ProviderPerformance,
)
from .browser_use_provider import BrowserUseProvider, BrowserUseCompositeProvider
from .openhands_provider import OpenHandsProvider
from .agent_s_provider import AgentSProvider
from .open_interpreter_provider import OpenInterpreterProvider
from .composite_provider import (
    ResearchAndAnalyzeProvider,
    AutomateWorkflowProvider,
    FullStackAgentProvider,
)

__all__ = [
    # Base
    'SkillProvider',
    'SkillCategory',
    'ProviderCapability',
    'ProviderResult',
    'JottyDefaultProvider',
    'SKILL_CATEGORY_MAP',
    'CATEGORY_KEYWORDS',
    # Registry
    'ProviderRegistry',
    'ProviderSelector',
    'ProviderPerformance',
    # Providers
    'BrowserUseProvider',
    'BrowserUseCompositeProvider',
    'OpenHandsProvider',
    'AgentSProvider',
    'OpenInterpreterProvider',
    # Composite
    'ResearchAndAnalyzeProvider',
    'AutomateWorkflowProvider',
    'FullStackAgentProvider',
]
