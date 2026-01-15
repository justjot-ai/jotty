#!/usr/bin/env python3
"""
Unified DSPy LM Provider Registry
=================================

Consolidates ALL providers behind DSPy LM abstraction:
- Direct providers (OpenRouter, OpenCode)
- Vercel AI SDK providers (via AISDKProviderLM)
- CLI providers (Claude CLI, Cursor CLI)
- API providers (Anthropic, OpenAI, Google, Groq)

All providers accessible through single DSPy LM interface.
"""
import os
import dspy
from dspy.clients.base_lm import BaseLM
from typing import Optional, Dict, Any
from pathlib import Path


class UnifiedLMProvider:
    """
    Unified provider registry for all LM providers.
    All providers go through DSPy LM abstraction.
    """
    
    @staticmethod
    def create_lm(
        provider: str,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLM:
        """
        Create DSPy LM instance for any provider.
        
        Args:
            provider: Provider name:
                - 'opencode': OpenCode (free GLM model)
                - 'openrouter': OpenRouter
                - 'anthropic': Anthropic Claude
                - 'openai': OpenAI
                - 'google': Google Gemini
                - 'groq': Groq
                - 'claude-cli': Claude CLI
                - 'cursor-cli': Cursor CLI
            model: Model name (optional, provider-specific defaults)
            **kwargs: Additional provider-specific arguments
        
        Returns:
            DSPy LM instance
        """
        provider = provider.lower()
        
        # 1. OpenCode (direct DSPy LM provider)
        if provider == 'opencode':
            from .opencode_lm import OpenCodeLM
            return OpenCodeLM(model=model or None, **kwargs)
        
        # 2. AISDKProviderLM (for Vercel AI SDK providers)
        # This covers: anthropic, openai, google, groq, openrouter, claude-cli, cursor-cli
        try:
            from ..integration.ai_sdk_provider_adapter import AISDKProviderLM
            
            # Map provider names
            ai_sdk_provider = provider
            if provider == 'claude-cli':
                ai_sdk_provider = 'claude-cli'
            elif provider == 'cursor-cli':
                ai_sdk_provider = 'cursor-cli'
            elif provider == 'openrouter':
                ai_sdk_provider = 'openrouter'
            elif provider == 'anthropic':
                ai_sdk_provider = 'anthropic'
            elif provider == 'openai':
                ai_sdk_provider = 'openai'
            elif provider == 'google':
                ai_sdk_provider = 'google'
            elif provider == 'groq':
                ai_sdk_provider = 'groq'
            
            # Default models per provider
            default_models = {
                'claude-cli': 'sonnet',
                'cursor-cli': 'sonnet',
                'anthropic': 'claude-3-5-sonnet-20241022',
                'openai': 'gpt-4o',
                'google': 'gemini-2.0-flash-exp',
                'groq': 'llama-3.1-8b-instant',
                'openrouter': 'meta-llama/llama-3.3-70b-instruct:free',
            }
            
            model = model or default_models.get(ai_sdk_provider, 'sonnet')
            
            # Extract API key from kwargs or env
            api_key = kwargs.pop('api_key', None)
            if not api_key:
                env_keys = {
                    'anthropic': 'ANTHROPIC_API_KEY',
                    'openai': 'OPENAI_API_KEY',
                    'google': 'GOOGLE_API_KEY',
                    'groq': 'GROQ_API_KEY',
                    'openrouter': 'OPENROUTER_API_KEY',
                }
                env_key = env_keys.get(ai_sdk_provider)
                if env_key:
                    api_key = os.getenv(env_key)
            
            return AISDKProviderLM(
                provider=ai_sdk_provider,
                model=model,
                api_key=api_key,
                **kwargs
            )
        except ImportError:
            # Fallback to direct DSPy LM for API providers
            if provider in ('anthropic', 'openai', 'google', 'groq', 'openrouter'):
                return UnifiedLMProvider._create_direct_lm(provider, model, **kwargs)
            raise ValueError(f"Provider '{provider}' not available (AISDKProviderLM not found)")
    
    @staticmethod
    def _create_direct_lm(
        provider: str,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLM:
        """
        Create direct DSPy LM (fallback when AISDKProviderLM unavailable).
        """
        api_key = kwargs.pop('api_key', None)
        
        # Map provider to DSPy model format
        model_map = {
            'anthropic': f'anthropic/{model or "claude-3-5-sonnet-20241022"}',
            'openai': f'openai/{model or "gpt-4o"}',
            'google': f'google/{model or "gemini-2.0-flash-exp"}',
            'groq': f'groq/{model or "llama-3.1-8b-instant"}',
            'openrouter': f'openrouter/{model or "meta-llama/llama-3.3-70b-instruct:free"}',
        }
        
        if provider not in model_map:
            raise ValueError(f"Direct DSPy LM not supported for provider '{provider}'")
        
        if not api_key:
            env_keys = {
                'anthropic': 'ANTHROPIC_API_KEY',
                'openai': 'OPENAI_API_KEY',
                'google': 'GOOGLE_API_KEY',
                'groq': 'GROQ_API_KEY',
                'openrouter': 'OPENROUTER_API_KEY',
            }
            env_key = env_keys.get(provider)
            if env_key:
                api_key = os.getenv(env_key)
        
        if not api_key:
            raise ValueError(f"API key required for provider '{provider}'")
        
        return dspy.LM(model_map[provider], api_key=api_key, **kwargs)
    
    @staticmethod
    def configure_default_lm(provider: Optional[str] = None, **kwargs) -> BaseLM:
        """
        Configure DSPy with default LM provider.
        Auto-detects available providers in priority order:
        1. OpenCode (free)
        2. CLI providers (claude-cli, cursor-cli)
        3. API providers (anthropic, openrouter, groq)
        
        Args:
            provider: Optional provider name (auto-detect if None)
            **kwargs: Provider-specific arguments
        
        Returns:
            Configured DSPy LM instance
        """
        if provider:
            lm = UnifiedLMProvider.create_lm(provider, **kwargs)
            dspy.configure(lm=lm)
            return lm
        
        # Auto-detect provider priority:
        # 1. OpenCode (free)
        try:
            lm = UnifiedLMProvider.create_lm('opencode')
            dspy.configure(lm=lm)
            print("✅ DSPy configured with OpenCodeLM (free)", file=__import__('sys').stderr)
            return lm
        except Exception:
            pass
        
        # 2. CLI providers (no API key needed)
        if os.path.exists('/usr/local/bin/cursor') or os.path.exists('/home/coder/.local/bin/cursor-agent'):
            try:
                lm = UnifiedLMProvider.create_lm('cursor-cli')
                dspy.configure(lm=lm)
                print("✅ DSPy configured with Cursor CLI", file=__import__('sys').stderr)
                return lm
            except Exception:
                pass
        
        if os.path.exists('/usr/local/bin/claude') or os.path.exists('/usr/bin/claude'):
            try:
                lm = UnifiedLMProvider.create_lm('claude-cli')
                dspy.configure(lm=lm)
                print("✅ DSPy configured with Claude CLI", file=__import__('sys').stderr)
                return lm
            except Exception:
                pass
        
        # 3. API providers (require API keys)
        for provider_name in ['anthropic', 'openrouter', 'groq', 'openai', 'google']:
            try:
                lm = UnifiedLMProvider.create_lm(provider_name)
                dspy.configure(lm=lm)
                print(f"✅ DSPy configured with {provider_name}", file=__import__('sys').stderr)
                return lm
            except Exception:
                continue
        
        raise RuntimeError("No available LM providers found")


# Convenience function
def configure_dspy_lm(provider: Optional[str] = None, **kwargs) -> BaseLM:
    """
    Configure DSPy with unified LM provider.
    
    Usage:
        from Jotty.core.foundation.unified_lm_provider import configure_dspy_lm
        import dspy
        
        # Auto-detect best available provider
        configure_dspy_lm()
        
        # Use specific provider
        configure_dspy_lm('opencode')
        configure_dspy_lm('openrouter', model='meta-llama/llama-3.3-70b-instruct:free')
    """
    return UnifiedLMProvider.configure_default_lm(provider, **kwargs)
