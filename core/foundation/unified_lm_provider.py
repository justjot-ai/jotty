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
        
        # Priority: Use direct DSPy LM for API providers (faster, more reliable)
        # CLI providers still use AISDKProviderLM or direct CLI LM
        if provider in ('anthropic', 'openai', 'google', 'groq', 'openrouter'):
            # Use DSPy's native API support (faster than CLI, more reliable)
            return UnifiedLMProvider._create_direct_lm(provider, model, **kwargs)
        
        # CLI providers and OpenCode use AISDKProviderLM
        try:
            from ..integration.ai_sdk_provider_adapter import AISDKProviderLM
            
            # Default models per provider
            default_models = {
                'opencode': 'default',  # OpenCode free model (default)
                'claude-cli': 'sonnet',
                'cursor-cli': 'sonnet',
            }
            
            model = model or default_models.get(provider, 'sonnet')
            
            return AISDKProviderLM(
                provider=provider,
                model=model,
                **kwargs
            )
        except ImportError:
            # Fallback: Try direct CLI LM for CLI providers
            if provider == 'claude-cli':
                from .claude_cli_lm import ClaudeCLILM
                return ClaudeCLILM(model=model or 'sonnet', **kwargs)
            elif provider == 'cursor-cli':
                from .cursor_cli_lm import CursorCLILM
                return CursorCLILM(model=model or 'composer-1', **kwargs)
            raise ValueError(f"Provider '{provider}' not available")
    
    @staticmethod
    def _create_direct_lm(
        provider: str,
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLM:
        """
        Create direct DSPy LM (fallback when AISDKProviderLM unavailable).
        Uses DSPy's native provider support for faster, more reliable API access.
        """
        api_key = kwargs.pop('api_key', None)
        
        # Map provider to DSPy model format
        # Handle model aliases (e.g., "sonnet" -> full model name)
        model_aliases = {
            'anthropic': {
                'sonnet': 'claude-sonnet-4-20250514',  # Latest Sonnet 4
                'opus': 'claude-opus-4-20250514',
                'haiku': 'claude-3-5-haiku-20241022',
            },
            'openai': {
                'gpt4': 'gpt-4-turbo',
                'gpt4o': 'gpt-4o',
            },
        }
        
        # Resolve model alias if needed
        resolved_model = model
        if provider in model_aliases and model in model_aliases[provider]:
            resolved_model = model_aliases[provider][model]
        elif not resolved_model:
            # Default models
            defaults = {
                'anthropic': 'claude-sonnet-4-20250514',  # Latest Sonnet 4
                'openai': 'gpt-4o',
                'google': 'gemini-2.0-flash-exp',
                'groq': 'llama-3.1-8b-instant',
                'openrouter': 'meta-llama/llama-3.3-70b-instruct:free',
            }
            resolved_model = defaults.get(provider, model)
        
        model_map = {
            'anthropic': f'anthropic/{resolved_model}',
            'openai': f'openai/{resolved_model}',
            'google': f'google/{resolved_model}',
            'groq': f'groq/{resolved_model}',
            'openrouter': f'openrouter/{resolved_model}',
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
        
        # Use DSPy's native LM (faster than CLI, more reliable)
        # For Anthropic, enable structured output mode to enforce JSON
        if provider == 'anthropic':
            # Anthropic API supports response_format for structured output
            # This forces JSON output, preventing conversational responses
            kwargs.setdefault('response_format', {'type': 'json_object'})
        
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
        # 1. API providers first (faster, more reliable than CLI)
        # Check for API keys and use native DSPy support
        api_providers = [
            ('anthropic', 'ANTHROPIC_API_KEY', 'sonnet'),
            ('openai', 'OPENAI_API_KEY', 'gpt-4o'),
            ('google', 'GOOGLE_API_KEY', 'gemini-2.0-flash-exp'),
            ('groq', 'GROQ_API_KEY', 'llama-3.1-8b-instant'),
            ('openrouter', 'OPENROUTER_API_KEY', 'meta-llama/llama-3.3-70b-instruct:free'),
        ]
        
        for provider_name, env_key, default_model in api_providers:
            if os.getenv(env_key):
                try:
                    lm = UnifiedLMProvider.create_lm(provider_name, model=default_model)
                    dspy.configure(lm=lm)
                    print(f"✅ DSPy configured with {provider_name} API (native)", file=__import__('sys').stderr)
                    return lm
                except Exception as e:
                    print(f"⚠️  {provider_name} API failed: {e}", file=__import__('sys').stderr)
                    continue
        
        # 2. CLI providers (fallback if no API keys)
        # Use JottyClaudeProvider (auto-manages wrapper server)
        try:
            from .jotty_claude_provider import JottyClaudeProvider, is_claude_available
            if is_claude_available():
                provider = JottyClaudeProvider(auto_start=True)
                lm = provider.configure_dspy()
                print("✅ DSPy configured with JottyClaudeProvider", file=__import__('sys').stderr)
                return lm
        except Exception as e:
            print(f"⚠️  JottyClaudeProvider failed: {e}", file=__import__('sys').stderr)

        # Fallback to DirectClaudeCLI (simple subprocess, ~3s per call)
        import shutil
        claude_path = shutil.which('claude')
        if claude_path:
            try:
                from ..integration.direct_claude_cli_lm import DirectClaudeCLI
                lm = DirectClaudeCLI(model="sonnet")
                dspy.configure(lm=lm)
                print("✅ DSPy configured with DirectClaudeCLI", file=__import__('sys').stderr)
                return lm
            except Exception as e:
                print(f"⚠️  DirectClaudeCLI failed: {e}", file=__import__('sys').stderr)

        # Cursor CLI (composer-1 model, no on-demand needed)
        if os.path.exists('/usr/local/bin/cursor-agent'):
            try:
                from .cursor_cli_lm import CursorCLILM
                lm = CursorCLILM(model="composer-1")
                dspy.configure(lm=lm)
                print("✅ DSPy configured with Cursor CLI (direct, composer-1)", file=__import__('sys').stderr)
                return lm
            except Exception as e:
                print(f"⚠️  Cursor CLI failed: {e}", file=__import__('sys').stderr)

        # 3. OpenCode (GLM via remote execution for ARM64)
        try:
            from .opencode_lm import OpenCodeLM
            lm = OpenCodeLM(model="glm-4")  # Free GLM model
            dspy.configure(lm=lm)
            print("✅ DSPy configured with OpenCode GLM (remote)", file=__import__('sys').stderr)
            return lm
        except Exception as e:
            print(f"⚠️  OpenCode failed: {e}", file=__import__('sys').stderr)
        
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
