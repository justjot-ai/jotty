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
    
    def __init__(self, config=None, provider: Optional[str] = None):
        """
        Initialize SwarmProviderGateway.
        
        Args:
            config: Optional JottyConfig
            provider: Optional preferred provider (auto-detect if None)
        """
        self.config = config
        self.provider = provider
        self._configured_lm: Optional[BaseLM] = None
        
        # DRY: Reuse existing UnifiedLMProvider
        from ...foundation.unified_lm_provider import UnifiedLMProvider, configure_dspy_lm
        self._unified_provider = UnifiedLMProvider
        self._configure_fn = configure_dspy_lm
        
        # Auto-configure DSPy with best available provider
        self._auto_configure()
    
    def _auto_configure(self):
        """Auto-configure DSPy with best available provider (DRY: reuse configure_dspy_lm)."""
        try:
            # DRY: Reuse existing configuration function
            self._configured_lm = self._configure_fn(provider=self.provider)
            provider_name = getattr(self._configured_lm, 'provider', 'unknown')
            model_name = getattr(self._configured_lm, 'model', 'unknown')
            logger.info(f"🌐 SwarmProviderGateway configured: {provider_name}/{model_name}")
        except Exception as e:
            logger.warning(f"⚠️  Provider auto-configuration failed: {e}, will use DSPy defaults")
            # Try to get current DSPy LM if configured
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                self._configured_lm = dspy.settings.lm
                logger.info("🌐 SwarmProviderGateway using existing DSPy LM configuration")
            else:
                logger.warning("⚠️  No LM provider configured")
    
    def get_lm(self) -> Optional[BaseLM]:
        """
        Get configured DSPy LM instance.
        
        Returns:
            Configured BaseLM instance or None
        """
        if self._configured_lm:
            return self._configured_lm
        
        # Fallback to DSPy settings
        if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
            return dspy.settings.lm
        
        return None
    
    def configure_provider(self, provider: str, model: Optional[str] = None, **kwargs) -> BaseLM:
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
        # DRY: Reuse existing create_lm method
        lm = self._unified_provider.create_lm(provider=provider, model=model, **kwargs)
        dspy.configure(lm=lm)
        self._configured_lm = lm
        self.provider = provider
        logger.info(f"🌐 SwarmProviderGateway configured: {provider}/{model or 'default'}")
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
