"""
SwarmProviderGateway - Unified Provider Gateway

Unified gateway for all LLM providers (AI SDK, OpenRouter, OpenCode, Claude CLI, Cursor CLI, etc.).
Follows DRY: Reuses existing UnifiedLMProvider system.
"""
import logging
from typing import Optional, Dict, Any
import dspy
from dspy.clients.base_lm import BaseLM

logger = logging.getLogger(__name__)


class SwarmProviderGateway:
    """
    Unified provider gateway for all LLM providers.
    
    DRY Principle: Reuses existing UnifiedLMProvider and configure_dspy_lm.
    Provides consistent interface for all providers:
    - AI SDK providers (via JustJot.ai API)
    - OpenRouter
    - OpenCode (free GLM model)
    - Claude CLI
    - Cursor CLI
    - Anthropic API
    - OpenAI API
    - Google Gemini
    - Groq
    
    All providers accessible through single unified interface.
    """
    
    def __init__(self, config: Any = None, provider: Optional[str] = None) -> None:
        """
        Initialize SwarmProviderGateway.
        
        Args:
            config: Optional SwarmConfig
            provider: Optional preferred provider (auto-detect if None)
        """
        self.config = config
        self.provider = provider
        self._configured_lm: Optional[BaseLM] = None
        
        # DRY: Reuse existing UnifiedLMProvider
        from Jotty.core.foundation.unified_lm_provider import UnifiedLMProvider, configure_dspy_lm
        self._unified_provider = UnifiedLMProvider
        self._configure_fn = configure_dspy_lm
        
        # Auto-configure DSPy with best available provider
        # Check if we're in async context first
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # In async context - don't auto-configure yet, will configure on-demand
            logger.debug("SwarmProviderGateway: Detected async context, deferring auto-configuration")
        except RuntimeError:
            # Not in async context - safe to auto-configure
            self._auto_configure()
    
    def _auto_configure(self) -> Any:
        """Auto-configure DSPy with best available provider (DRY: reuse configure_dspy_lm)."""
        try:
            import asyncio
            import os
            import shutil
            # Check if we're in async context
            try:
                loop = asyncio.get_running_loop()
                in_async_context = True
            except RuntimeError:
                in_async_context = False

            # DRY: Reuse existing configuration function
            # In async contexts, don't call dspy.configure() - just create LM
            if in_async_context:
                # In async context, create LM but don't configure DSPy globally
                # Auto-detect best available provider (same logic as configure_dspy_lm)
                from Jotty.core.foundation.unified_lm_provider import UnifiedLMProvider

                if self.provider:
                    # User specified a provider
                    self._configured_lm = UnifiedLMProvider.create_lm(provider=self.provider)
                else:
                    # Auto-detect priority:
                    # 1. OpenCode Zen free models (no cost)
                    # 2. API providers (anthropic, openai, etc.)
                    # 3. CLI providers (claude-cli, cursor-cli)
                    # 4. Legacy OpenCode

                    # 1. Zen free models first (free, no cost)
                    if os.getenv('OPENCODE_ZEN_API_KEY'):
                        try:
                            self._configured_lm = UnifiedLMProvider.create_lm(provider='zen')
                            logger.info(" SwarmProviderGateway (async): Using OpenCode Zen (free)")
                        except Exception as e:
                            logger.debug(f"Zen failed: {e}")

                    # 2. API providers
                    if not self._configured_lm:
                        api_providers = [
                            ('anthropic', 'ANTHROPIC_API_KEY'),
                            ('openai', 'OPENAI_API_KEY'),
                            ('google', 'GOOGLE_API_KEY'),
                            ('groq', 'GROQ_API_KEY'),
                            ('openrouter', 'OPENROUTER_API_KEY'),
                        ]

                        for provider_name, env_key in api_providers:
                            if os.getenv(env_key):
                                try:
                                    self._configured_lm = UnifiedLMProvider.create_lm(provider=provider_name)
                                    logger.info(f" SwarmProviderGateway (async): Using {provider_name} API")
                                    break
                                except Exception:
                                    continue

                    # 3. JottyClaudeProvider (auto-manages wrapper)
                    if not self._configured_lm:
                        try:
                            from Jotty.core.foundation.jotty_claude_provider import JottyClaudeProvider, is_claude_available
                            if is_claude_available():
                                provider = JottyClaudeProvider(auto_start=True)
                                self._configured_lm = provider.get_lm()
                                logger.info(" SwarmProviderGateway (async): Using JottyClaudeProvider")
                        except Exception as e:
                            logger.debug(f"JottyClaudeProvider failed: {e}")

                    # 4. DirectClaudeCLI (simple subprocess, ~3s per call)
                    if not self._configured_lm and shutil.which('claude'):
                        try:
                            from Jotty.core.integration.direct_claude_cli_lm import DirectClaudeCLI
                            self._configured_lm = DirectClaudeCLI()
                            logger.info(" SwarmProviderGateway (async): Using DirectClaudeCLI")
                        except Exception as e:
                            logger.debug(f"DirectClaudeCLI failed: {e}")

                    # 5. Legacy OpenCode fallback
                    if not self._configured_lm:
                        try:
                            from Jotty.core.foundation.opencode_lm import OpenCodeLM
                            self._configured_lm = OpenCodeLM(model="glm-4")
                            logger.info(" SwarmProviderGateway (async): Using OpenCode GLM")
                        except Exception as e:
                            logger.debug(f"OpenCode failed: {e}")

                if self._configured_lm:
                    provider_name = getattr(self._configured_lm, 'provider', 'unknown')
                    model_name = getattr(self._configured_lm, 'model', 'unknown')
                    logger.info(f" SwarmProviderGateway configured (async-safe): {provider_name}/{model_name}")
            else:
                # In sync context, configure DSPy normally
                self._configured_lm = self._configure_fn(provider=self.provider)
                provider_name = getattr(self._configured_lm, 'provider', 'unknown')
                model_name = getattr(self._configured_lm, 'model', 'unknown')
                logger.info(f" SwarmProviderGateway configured: {provider_name}/{model_name}")
        except Exception as e:
            logger.warning(f" Provider auto-configuration failed: {e}, will use DSPy defaults")
            # Try to get current DSPy LM if configured
            try:
                import asyncio
                try:
                    asyncio.get_running_loop()
                    # In async context, use context manager
                    with dspy.context():
                        if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                            self._configured_lm = dspy.settings.lm
                            logger.info(" SwarmProviderGateway using existing DSPy LM configuration (async)")
                except RuntimeError:
                    # Not in async context
                    if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                        self._configured_lm = dspy.settings.lm
                        logger.info(" SwarmProviderGateway using existing DSPy LM configuration")
            except Exception:
                logger.warning(" No LM provider configured")
    
    def get_lm(self) -> Optional[BaseLM]:
        """
        Get configured DSPy LM instance.
        
        Returns:
            Configured BaseLM instance or None
        """
        # If already configured, return it
        if self._configured_lm:
            return self._configured_lm
        
        # If not configured yet, configure now (on-demand)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # In async context - configure without calling dspy.configure()
            logger.debug("SwarmProviderGateway.get_lm(): Configuring in async context")
            self._auto_configure()
        except RuntimeError:
            # Not in async context - safe to auto-configure normally
            self._auto_configure()
        
        if self._configured_lm:
            return self._configured_lm
        
        # Fallback to DSPy settings (with async context handling)
        try:
            import asyncio
            try:
                asyncio.get_running_loop()
                # In async context, use context manager
                with dspy.context():
                    if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                        return dspy.settings.lm
            except RuntimeError:
                # Not in async context
                if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                    return dspy.settings.lm
        except Exception:
            pass
        
        return None
    
    def configure_provider(self, provider: str, model: Optional[str] = None, **kwargs: Any) -> BaseLM:
        """
        Configure a specific provider.
        
        DRY: Reuses UnifiedLMProvider.create_lm()
        
        Args:
            provider: Provider name (opencode, openrouter, claude-cli, cursor-cli, anthropic, etc.)
            model: Optional model name
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Configured BaseLM instance
        """
        import asyncio
        # Check if we're in async context
        try:
            loop = asyncio.get_running_loop()
            in_async_context = True
        except RuntimeError:
            in_async_context = False
        
        # DRY: Reuse existing create_lm method
        lm = self._unified_provider.create_lm(provider=provider, model=model, **kwargs)
        
        # Only configure DSPy globally if not in async context
        if not in_async_context:
            dspy.configure(lm=lm)
        
        self._configured_lm = lm
        self.provider = provider
        logger.info(f" SwarmProviderGateway configured: {provider}/{model or 'default'}")
        return lm
    
    def list_available_providers(self) -> list:
        """
        List available providers.
        
        Returns:
            List of available provider names
        """
        providers = [
            'opencode',      # OpenCode (free GLM)
            'claude-cli',   # Claude CLI
            'cursor-cli',   # Cursor CLI
            'anthropic',    # Anthropic API
            'openai',       # OpenAI API
            'google',       # Google Gemini
            'groq',         # Groq
            'openrouter',   # OpenRouter
        ]
        return providers
    
    def is_provider_available(self, provider: str) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider: Provider name
            
        Returns:
            True if provider is available
        """
        try:
            # DRY: Try to create LM to check availability
            self._unified_provider.create_lm(provider=provider)
            return True
        except Exception:
            return False
