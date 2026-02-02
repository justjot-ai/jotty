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
Automatically injects current date/time context to all LLM calls.
"""
import os
from datetime import datetime
import dspy
from dspy.clients.base_lm import BaseLM
from typing import Optional, Dict, Any, List
from pathlib import Path


def get_current_context() -> str:
    """Get current date/time context for LLM."""
    now = datetime.now()
    return (
        f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A, %B %d, %Y')}). "
        f"Current time: {now.strftime('%H:%M:%S')}."
    )


class ContextAwareLM(BaseLM):
    """
    Wrapper LM that injects current date/time context to all calls.

    This ensures ALL LLM providers know the actual current date,
    preventing issues where the LLM thinks it's in its training cutoff year.
    """

    def __init__(self, wrapped_lm: BaseLM, **kwargs):
        """
        Wrap an existing LM with context injection.

        Args:
            wrapped_lm: The underlying LM to wrap
        """
        # Initialize with the wrapped LM's model name
        model_name = getattr(wrapped_lm, 'model', 'unknown')
        super().__init__(model=model_name, **kwargs)
        self._wrapped = wrapped_lm
        # Copy attributes from wrapped LM
        self.provider = getattr(wrapped_lm, 'provider', 'unknown')
        self.history = getattr(wrapped_lm, 'history', [])

    def _inject_context(self, prompt: str = None, messages: List[Dict] = None):
        """Inject current date context into prompt or messages."""
        context = f"[System Context: {get_current_context()}]"

        if prompt:
            return f"{context}\n\n{prompt}", messages

        if messages:
            # Prepend context to first user message or add as system message
            new_messages = list(messages)

            # Check if there's already a system message
            has_system = any(m.get('role') == 'system' for m in new_messages if isinstance(m, dict))

            if has_system:
                # Prepend context to existing system message
                for i, msg in enumerate(new_messages):
                    if isinstance(msg, dict) and msg.get('role') == 'system':
                        new_messages[i] = {
                            **msg,
                            'content': f"{context}\n\n{msg.get('content', '')}"
                        }
                        break
            else:
                # Add system message with context at the beginning
                new_messages.insert(0, {'role': 'system', 'content': context})

            return prompt, new_messages

        return prompt, messages

    def __call__(self, prompt: str = None, messages: List[Dict] = None, **kwargs):
        """Call the wrapped LM with injected context."""
        prompt, messages = self._inject_context(prompt, messages)
        return self._wrapped(prompt=prompt, messages=messages, **kwargs)

    def inspect_history(self, n: int = 1):
        """Delegate to wrapped LM."""
        if hasattr(self._wrapped, 'inspect_history'):
            return self._wrapped.inspect_history(n)
        return {'history': self.history[-n:] if self.history else []}

    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped LM."""
        return getattr(self._wrapped, name)


class UnifiedLMProvider:
    """
    Unified provider registry for all LM providers.
    All providers go through DSPy LM abstraction.
    """
    
    @staticmethod
    def create_lm(
        provider: str,
        model: Optional[str] = None,
        inject_context: bool = True,
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
            inject_context: Whether to inject current date/time context (default True)
            **kwargs: Additional provider-specific arguments

        Returns:
            DSPy LM instance (wrapped with ContextAwareLM if inject_context=True)
        """
        provider = provider.lower()

        # Priority: Use direct DSPy LM for API providers (faster, more reliable)
        # CLI providers still use AISDKProviderLM or direct CLI LM
        if provider in ('anthropic', 'openai', 'google', 'groq', 'openrouter'):
            # Use DSPy's native API support (faster than CLI, more reliable)
            lm = UnifiedLMProvider._create_direct_lm(provider, model, **kwargs)
            return ContextAwareLM(lm) if inject_context else lm
        
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

            lm = AISDKProviderLM(
                provider=provider,
                model=model,
                **kwargs
            )
            return ContextAwareLM(lm) if inject_context else lm
        except ImportError:
            # Fallback: Try direct CLI LM for CLI providers
            if provider == 'claude-cli':
                from .claude_cli_lm import ClaudeCLILM
                lm = ClaudeCLILM(model=model or 'sonnet', **kwargs)
                return ContextAwareLM(lm) if inject_context else lm
            elif provider == 'cursor-cli':
                from .cursor_cli_lm import CursorCLILM
                lm = CursorCLILM(model=model or 'composer-1', **kwargs)
                return ContextAwareLM(lm) if inject_context else lm
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
        # 1. CLI providers FIRST (Claude CLI is more reliable than API keys)
        # This avoids issues with invalid/expired API keys
        # Use JottyClaudeProvider (auto-manages wrapper server)
        try:
            from .jotty_claude_provider import JottyClaudeProvider, is_claude_available
            if is_claude_available():
                jotty_provider = JottyClaudeProvider(auto_start=True)
                raw_lm = jotty_provider.configure_dspy()
                # Wrap with context injection
                lm = ContextAwareLM(raw_lm)
                dspy.configure(lm=lm)
                print("✅ DSPy configured with JottyClaudeProvider", file=__import__('sys').stderr)
                return lm
        except Exception as e:
            print(f"⚠️  JottyClaudeProvider failed: {e}", file=__import__('sys').stderr)

        # 2. API providers (fallback if CLI not available)
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
        
        # 3. Fallback to DirectClaudeCLI (simple subprocess, ~3s per call)
        import shutil
        claude_path = shutil.which('claude')
        if claude_path:
            try:
                from ..integration.direct_claude_cli_lm import DirectClaudeCLI
                raw_lm = DirectClaudeCLI(model="sonnet")
                # Wrap with context injection
                lm = ContextAwareLM(raw_lm)
                dspy.configure(lm=lm)
                print("✅ DSPy configured with DirectClaudeCLI", file=__import__('sys').stderr)
                return lm
            except Exception as e:
                print(f"⚠️  DirectClaudeCLI failed: {e}", file=__import__('sys').stderr)

        # Cursor CLI (composer-1 model, no on-demand needed)
        if os.path.exists('/usr/local/bin/cursor-agent'):
            try:
                from .cursor_cli_lm import CursorCLILM
                raw_lm = CursorCLILM(model="composer-1")
                lm = ContextAwareLM(raw_lm)
                dspy.configure(lm=lm)
                print("✅ DSPy configured with Cursor CLI (direct, composer-1)", file=__import__('sys').stderr)
                return lm
            except Exception as e:
                print(f"⚠️  Cursor CLI failed: {e}", file=__import__('sys').stderr)

        # 3. OpenCode (GLM via remote execution for ARM64)
        try:
            from .opencode_lm import OpenCodeLM
            raw_lm = OpenCodeLM(model="glm-4")  # Free GLM model
            lm = ContextAwareLM(raw_lm)
            dspy.configure(lm=lm)
            print("✅ DSPy configured with OpenCode GLM (remote)", file=__import__('sys').stderr)
            return lm
        except Exception as e:
            print(f"⚠️  OpenCode failed: {e}", file=__import__('sys').stderr)
        
        raise RuntimeError("No available LM providers found")


# Convenience function
def configure_dspy_lm(provider: Optional[str] = None, **kwargs) -> BaseLM:
    """
    Configure DSPy with unified LM provider and JSONAdapter for structured output.

    Usage:
        from Jotty.core.foundation.unified_lm_provider import configure_dspy_lm
        import dspy

        # Auto-detect best available provider
        configure_dspy_lm()

        # Use specific provider
        configure_dspy_lm('opencode')
        configure_dspy_lm('openrouter', model='meta-llama/llama-3.3-70b-instruct:free')
    """
    lm = UnifiedLMProvider.configure_default_lm(provider, **kwargs)

    # Configure JSONAdapter for structured output support
    # This enables DSPy to enforce JSON output format for Pydantic models
    try:
        adapter = dspy.JSONAdapter()
        dspy.configure(lm=lm, adapter=adapter)
    except Exception:
        # Fall back to LM-only config if JSONAdapter fails
        pass

    return lm
