"""
SwarmManager Provider Mixin
============================

Extracted from swarm_manager.py — handles skill provider registry
(browser-use, openhands, agent-s, etc.)
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ProviderMixin:
    """Mixin for skill provider registry management."""

    def _init_provider_registry(self):
        """Initialize the skill provider registry with all available providers."""
        # Lazy imports from swarm_manager module level
        from . import swarm_manager as _sm
        ProviderRegistry = _sm.ProviderRegistry
        BrowserUseProvider = _sm.BrowserUseProvider
        OpenHandsProvider = _sm.OpenHandsProvider
        AgentSProvider = _sm.AgentSProvider
        OpenInterpreterProvider = _sm.OpenInterpreterProvider
        ResearchAndAnalyzeProvider = _sm.ResearchAndAnalyzeProvider
        AutomateWorkflowProvider = _sm.AutomateWorkflowProvider
        FullStackAgentProvider = _sm.FullStackAgentProvider

        if not ProviderRegistry:
            logger.warning("Skill providers not available")
            return

        try:
            self.provider_registry = ProviderRegistry(
                swarm_intelligence=self.swarm_intelligence
            )

            # Register external providers
            for provider in [
                BrowserUseProvider({'headless': True}),
                OpenHandsProvider({'sandbox': True}),
                AgentSProvider({'safe_mode': True}),
                OpenInterpreterProvider({'auto_run': True}),
            ]:
                try:
                    self.provider_registry.register(provider)
                except Exception as e:
                    logger.debug(f"Could not register {provider.name}: {e}")

            # Register composite providers
            for provider in [
                ResearchAndAnalyzeProvider(),
                AutomateWorkflowProvider(),
                FullStackAgentProvider(),
            ]:
                try:
                    provider.set_registry(self.provider_registry)
                    if hasattr(provider, 'set_swarm_intelligence'):
                        provider.set_swarm_intelligence(self.swarm_intelligence)
                    self.provider_registry.register(provider)
                except Exception as e:
                    logger.debug(f"Could not register composite {provider.name}: {e}")

            # Load learned provider preferences
            provider_path = self._get_provider_registry_path()
            if provider_path.exists():
                self.provider_registry.load_state(str(provider_path))
                logger.info(f"Loaded provider learnings from {provider_path}")

            logger.info(f"✅ Provider registry initialized: {list(self.provider_registry.get_all_providers().keys())}")

        except Exception as e:
            logger.error(f"Failed to initialize provider registry: {e}")
            self.provider_registry = None

    def _get_provider_registry_path(self) -> Path:
        """Get path for provider registry persistence."""
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'provider_learnings.json'
        return Path.home() / '.jotty' / 'provider_learnings.json'

    async def execute_with_provider(
        self,
        category: str,
        task: str,
        context: Dict[str, Any] = None,
        provider_name: str = None
    ):
        """
        Execute a task using the skill provider system.

        Args:
            category: Skill category (browser, terminal, computer_use, etc.)
            task: Task description in natural language
            context: Additional context
            provider_name: Optional specific provider to use

        Returns:
            ProviderResult with execution output
        """
        from . import swarm_manager as _sm
        SkillCategory = _sm.SkillCategory

        if not self.provider_registry:
            logger.warning("Provider registry not available")
            return None

        try:
            cat_enum = SkillCategory(category) if isinstance(category, str) and SkillCategory else category

            result = await self.provider_registry.execute(
                category=cat_enum,
                task=task,
                context=context,
                provider_name=provider_name,
            )

            if result.success:
                self.swarm_intelligence.record_task_result(
                    agent_name=result.provider_name,
                    task_type=category,
                    success=result.success,
                    execution_time=result.execution_time,
                )

            return result

        except Exception as e:
            logger.error(f"Provider execution error: {e}")
            return None

    def get_provider_summary(self) -> Dict[str, Any]:
        """Get summary of provider registry state."""
        if not self.provider_registry:
            return {'available': False}
        return {
            'available': True,
            **self.provider_registry.get_registry_summary(),
        }
