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
import logging
from datetime import datetime
import dspy
from dspy.clients.base_lm import BaseLM
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


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
        # Inherit kwargs from the wrapped LM (max_tokens, temperature, etc.)
        # so DSPy sees the correct settings on the wrapper, not BaseLM defaults.
        inherited_kwargs = dict(getattr(wrapped_lm, 'kwargs', {}))
        # Remove api_key from inherited kwargs — don't duplicate secrets
        inherited_kwargs.pop('api_key', None)
        inherited_kwargs.update(kwargs)  # Explicit overrides win
        super().__init__(model=model_name, **inherited_kwargs)
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

        if provider == 'zen':
            lm = UnifiedLMProvider._create_zen_lm(model, **kwargs)
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
        # Centralized in config_defaults — change model versions there.
        from Jotty.core.foundation.config_defaults import (
            MODEL_ALIASES, DEFAULTS,
        )
        model_aliases = {
            'anthropic': MODEL_ALIASES,
            'openai': {
                'gpt4': 'gpt-4-turbo',
                'gpt4o': DEFAULTS.MODEL_OPENAI_DEFAULT,
            },
        }
        
        # Resolve model alias if needed
        resolved_model = model
        if provider in model_aliases and model in model_aliases[provider]:
            resolved_model = model_aliases[provider][model]
        elif not resolved_model:
            # Default models — centralized in config_defaults
            defaults = {
                'anthropic': DEFAULTS.MODEL_SONNET,
                'openai': DEFAULTS.MODEL_OPENAI_DEFAULT,
                'google': DEFAULTS.MODEL_GEMINI_DEFAULT,
                'groq': DEFAULTS.MODEL_GROQ_DEFAULT,
                'openrouter': DEFAULTS.MODEL_OPENROUTER_DEFAULT,
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
        # Centralized default — DSPy's built-in 1024 truncates real-world output.
        from Jotty.core.foundation.config_defaults import LLM_MAX_OUTPUT_TOKENS
        kwargs.setdefault('max_tokens', LLM_MAX_OUTPUT_TOKENS)

        # For Anthropic, enable structured output mode to enforce JSON
        if provider == 'anthropic':
            # Anthropic API supports response_format for structured output
            # This forces JSON output, preventing conversational responses
            kwargs.setdefault('response_format', {'type': 'json_object'})
        
        return dspy.LM(model_map[provider], api_key=api_key, **kwargs)

    # =========================================================================
    # OPENCODE ZEN PROVIDER (Free and paid models via opencode.ai/zen)
    # =========================================================================

    # Zen model registry: model_id -> (endpoint_path, api_format)
    # api_format determines how DSPy should talk to the endpoint:
    #   'openai'     -> /v1/chat/completions (OpenAI-compatible)
    #   'anthropic'  -> /v1/messages (Anthropic-compatible)
    #   'responses'  -> /v1/responses (OpenAI Responses API)
    ZEN_MODELS = {
        # === FREE MODELS ===
        'glm-4.7-free':              ('chat/completions', 'openai'),
        'kimi-k2.5-free':            ('chat/completions', 'openai'),
        'minimax-m2.1-free':         ('messages',         'anthropic'),
        'big-pickle':                ('chat/completions', 'openai'),
        'gpt-5-nano':                ('responses',        'openai'),
        'trinity-large-preview-free': ('chat/completions', 'openai'),
        'alpha-g5':                  ('chat/completions', 'openai'),
        # === PAID - OpenAI-compatible ===
        'glm-4.7':            ('chat/completions', 'openai'),
        'glm-4.6':            ('chat/completions', 'openai'),
        'kimi-k2.5':          ('chat/completions', 'openai'),
        'kimi-k2-thinking':   ('chat/completions', 'openai'),
        'kimi-k2':            ('chat/completions', 'openai'),
        'minimax-m2.1':       ('chat/completions', 'openai'),
        'qwen3-coder':        ('chat/completions', 'openai'),
        # === PAID - Anthropic-compatible ===
        'claude-sonnet-4-5':  ('messages',         'anthropic'),
        'claude-sonnet-4':    ('messages',         'anthropic'),
        'claude-haiku-4-5':   ('messages',         'anthropic'),
        'claude-3-5-haiku':   ('messages',         'anthropic'),
        'claude-opus-4-6':    ('messages',         'anthropic'),
        'claude-opus-4-5':    ('messages',         'anthropic'),
        'claude-opus-4-1':    ('messages',         'anthropic'),
        # === PAID - OpenAI Responses API ===
        'gpt-5.2':            ('responses',        'openai'),
        'gpt-5.2-codex':      ('responses',        'openai'),
        'gpt-5.1':            ('responses',        'openai'),
        'gpt-5.1-codex':      ('responses',        'openai'),
        'gpt-5.1-codex-max':  ('responses',        'openai'),
        'gpt-5.1-codex-mini': ('responses',        'openai'),
        'gpt-5':              ('responses',        'openai'),
        'gpt-5-codex':        ('responses',        'openai'),
        # === PAID - Google ===
        'gemini-3-pro':       ('models/gemini-3-pro', 'google'),
        'gemini-3-flash':     ('models/gemini-3-flash', 'google'),
    }

    # Free Zen models (no credit card needed)
    ZEN_FREE_MODELS = [
        'glm-4.7-free', 'kimi-k2.5-free', 'minimax-m2.1-free',
        'big-pickle', 'gpt-5-nano', 'trinity-large-preview-free', 'alpha-g5',
    ]

    ZEN_BASE_URL = 'https://opencode.ai/zen/v1'

    @staticmethod
    def _create_zen_lm(
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLM:
        """
        Create DSPy LM for OpenCode Zen provider.

        Zen provides curated, benchmarked models via standard API endpoints.
        Free models available: glm-4.7-free, kimi-k2.5-free, minimax-m2.1-free,
        big-pickle, gpt-5-nano.

        Args:
            model: Zen model ID (e.g., 'glm-4.7-free', 'big-pickle')
                   Defaults to 'glm-4.7-free' (free, no cost)
            **kwargs: Additional LM arguments

        Returns:
            DSPy LM instance configured for Zen endpoint

        Env:
            OPENCODE_ZEN_API_KEY: API key from https://opencode.ai/auth
        """
        api_key = kwargs.pop('api_key', None) or os.getenv('OPENCODE_ZEN_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenCode Zen API key required. "
                "Set OPENCODE_ZEN_API_KEY env var or pass api_key=. "
                "Get your key at https://opencode.ai/auth"
            )

        # Default to a free model
        if not model or model == 'free':
            model = UnifiedLMProvider.ZEN_FREE_MODELS[0]
            logger.debug(f"Zen: using default free model '{model}'")

        # Look up endpoint and API format
        if model not in UnifiedLMProvider.ZEN_MODELS:
            original_model = model
            # Try fuzzy match — prefer free models first
            free_matches = [
                m for m in UnifiedLMProvider.ZEN_FREE_MODELS if model in m
            ]
            all_matches = [
                m for m in UnifiedLMProvider.ZEN_MODELS if model in m
            ]
            matches = free_matches or all_matches
            if matches:
                model = matches[0]
                logger.debug(f"Zen: fuzzy matched '{original_model}' to '{model}'")
            else:
                # Fallback to default free model instead of raising error
                fallback = UnifiedLMProvider.ZEN_FREE_MODELS[0]
                logger.warning(
                    f"Unknown Zen model '{original_model}', "
                    f"falling back to free model '{fallback}'. "
                    f"Available free: {', '.join(UnifiedLMProvider.ZEN_FREE_MODELS)}"
                )
                model = fallback

        endpoint_path, api_format = UnifiedLMProvider.ZEN_MODELS[model]
        base_url = f"{UnifiedLMProvider.ZEN_BASE_URL}"

        from Jotty.core.foundation.config_defaults import LLM_MAX_OUTPUT_TOKENS

        # DSPy validates temperature/max_tokens as direct constructor args
        # (not **kwargs), so we must extract and pass them explicitly.
        # For reasoning models (gpt-5 family, o1/o3/o4/o5), DSPy requires
        # temperature=1.0 or None, and max_tokens >= 16000 or None.
        import re
        is_reasoning = bool(re.match(
            r"^(?:o[1345](?:-(?:mini|nano|pro))?|gpt-5(?!-chat)(?:-.*)?)$",
            model,
        ))

        if is_reasoning:
            temperature = kwargs.pop('temperature', 1.0)
            if temperature != 1.0:
                temperature = 1.0
            max_tokens = kwargs.pop('max_tokens', 16000)
            if max_tokens is not None and max_tokens < 16000:
                max_tokens = 16000
        else:
            temperature = kwargs.pop('temperature', None)
            max_tokens = kwargs.pop('max_tokens', LLM_MAX_OUTPUT_TOKENS)

        if api_format == 'anthropic':
            dspy_model = f"anthropic/{model}"
        elif api_format == 'openai':
            dspy_model = f"openai/{model}"
        elif api_format == 'google':
            dspy_model = f"google/{model}"
        else:
            raise ValueError(f"Unknown API format '{api_format}' for Zen model '{model}'")

        lm = dspy.LM(
            dspy_model,
            api_key=api_key,
            api_base=f"{base_url}/",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        is_free = model in UnifiedLMProvider.ZEN_FREE_MODELS
        logger.info(
            f"Zen LM created: {model} ({'FREE' if is_free else 'PAID'}) "
            f"via {base_url}/{endpoint_path}"
        )
        return lm

    @staticmethod
    def list_zen_models(free_only: bool = False) -> Dict[str, Dict]:
        """
        List available Zen models with metadata.

        Args:
            free_only: Only return free models

        Returns:
            Dict mapping model_id -> {'endpoint', 'format', 'free'}
        """
        result = {}
        for model_id, (endpoint, fmt) in UnifiedLMProvider.ZEN_MODELS.items():
            is_free = model_id in UnifiedLMProvider.ZEN_FREE_MODELS
            if free_only and not is_free:
                continue
            result[model_id] = {
                'endpoint': f"{UnifiedLMProvider.ZEN_BASE_URL}/{endpoint}",
                'format': fmt,
                'free': is_free,
            }
        return result

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
                logger.debug("DSPy configured with JottyClaudeProvider")
                return lm
        except Exception as e:
            logger.debug(f"JottyClaudeProvider not available: {e}")

        # 2. OpenCode Zen free models (if API key available)
        zen_key = os.getenv('OPENCODE_ZEN_API_KEY')
        if zen_key:
            try:
                lm = UnifiedLMProvider.create_lm('zen', model='glm-4.7-free')
                dspy.configure(lm=lm)
                logger.debug("DSPy configured with OpenCode Zen (free: glm-4.7-free)")
                return lm
            except Exception as e:
                logger.debug(f"OpenCode Zen not available: {e}")

        # 3. API providers (fallback if CLI not available)
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
                    logger.debug(f"DSPy configured with {provider_name} API")
                    return lm
                except Exception as e:
                    logger.debug(f"{provider_name} API not available: {e}")
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
                logger.debug("DSPy configured with DirectClaudeCLI")
                return lm
            except Exception as e:
                logger.debug(f"DirectClaudeCLI not available: {e}")

        # Cursor CLI (composer-1 model, no on-demand needed)
        if os.path.exists('/usr/local/bin/cursor-agent'):
            try:
                from .cursor_cli_lm import CursorCLILM
                raw_lm = CursorCLILM(model="composer-1")
                lm = ContextAwareLM(raw_lm)
                dspy.configure(lm=lm)
                logger.debug("DSPy configured with Cursor CLI")
                return lm
            except Exception as e:
                logger.debug(f"Cursor CLI not available: {e}")

        # 3. OpenCode (GLM via remote execution for ARM64)
        try:
            from .opencode_lm import OpenCodeLM
            raw_lm = OpenCodeLM(model="glm-4")  # Free GLM model
            lm = ContextAwareLM(raw_lm)
            dspy.configure(lm=lm)
            logger.debug("DSPy configured with OpenCode GLM")
            return lm
        except Exception as e:
            print(f"⚠️  OpenCode failed: {e}", file=__import__('sys').stderr)

        raise RuntimeError("No available LM providers found")

    @staticmethod
    def get_available_providers() -> Dict[str, Any]:
        """
        Get all available providers and their configurations.

        Returns:
            Dict with provider info for API consumption:
            {
                'providers': {
                    'anthropic': {'available': True, 'models': [...], 'default': '...'},
                    ...
                },
                'model_aliases': {...},
                'recommended': 'provider_name'
            }
        """
        import shutil

        providers = {}

        # API providers
        api_providers = {
            'anthropic': {
                'env_key': 'ANTHROPIC_API_KEY',
                'models': ['claude-sonnet-4-20250514', 'claude-opus-4-20250514', 'claude-3-5-haiku-20241022'],
                'default': 'claude-sonnet-4-20250514',
                'aliases': {'sonnet': 'claude-sonnet-4-20250514', 'opus': 'claude-opus-4-20250514', 'haiku': 'claude-3-5-haiku-20241022'},
            },
            'openai': {
                'env_key': 'OPENAI_API_KEY',
                'models': ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                'default': 'gpt-4o',
                'aliases': {'gpt4': 'gpt-4-turbo', 'gpt4o': 'gpt-4o'},
            },
            'google': {
                'env_key': 'GOOGLE_API_KEY',
                'models': ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
                'default': 'gemini-2.0-flash-exp',
                'aliases': {},
            },
            'groq': {
                'env_key': 'GROQ_API_KEY',
                'models': ['llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'mixtral-8x7b-32768'],
                'default': 'llama-3.1-8b-instant',
                'aliases': {},
            },
            'openrouter': {
                'env_key': 'OPENROUTER_API_KEY',
                'models': ['meta-llama/llama-3.3-70b-instruct:free', 'anthropic/claude-3.5-sonnet'],
                'default': 'meta-llama/llama-3.3-70b-instruct:free',
                'aliases': {},
            },
        }

        for name, config in api_providers.items():
            has_key = bool(os.getenv(config['env_key']))
            providers[name] = {
                'type': 'api',
                'available': has_key,
                'models': config['models'],
                'default': config['default'],
                'aliases': config.get('aliases', {}),
                'requires': config['env_key'],
            }

        # CLI providers
        claude_available = bool(shutil.which('claude'))
        providers['claude-cli'] = {
            'type': 'cli',
            'available': claude_available,
            'models': ['sonnet', 'opus', 'haiku'],
            'default': 'sonnet',
            'aliases': {},
            'requires': 'claude CLI installed',
        }

        cursor_available = os.path.exists('/usr/local/bin/cursor-agent')
        providers['cursor-cli'] = {
            'type': 'cli',
            'available': cursor_available,
            'models': ['composer-1'],
            'default': 'composer-1',
            'aliases': {},
            'requires': 'cursor-agent installed',
        }

        # OpenCode (always available via remote)
        providers['opencode'] = {
            'type': 'remote',
            'available': True,
            'models': ['glm-4', 'default'],
            'default': 'default',
            'aliases': {},
            'requires': None,
            'note': 'Free GLM model via remote execution',
        }

        # OpenCode Zen (curated models, some free)
        zen_available = bool(os.getenv('OPENCODE_ZEN_API_KEY'))
        zen_free = UnifiedLMProvider.ZEN_FREE_MODELS
        zen_all = list(UnifiedLMProvider.ZEN_MODELS.keys())
        providers['zen'] = {
            'type': 'api',
            'available': zen_available,
            'models': zen_all,
            'free_models': zen_free,
            'default': 'glm-4.7-free',
            'aliases': {
                'free': 'glm-4.7-free',
                'kimi-free': 'kimi-k2.5-free',
                'minimax-free': 'minimax-m2.1-free',
                'pickle': 'big-pickle',
            },
            'requires': 'OPENCODE_ZEN_API_KEY',
            'note': f'{len(zen_free)} free models, {len(zen_all)} total. Get key at https://opencode.ai/auth',
        }

        # Determine recommended provider
        recommended = None
        priority = ['claude-cli', 'anthropic', 'openai', 'zen', 'groq', 'openrouter', 'opencode']
        for p in priority:
            if providers.get(p, {}).get('available'):
                recommended = p
                break

        return {
            'providers': providers,
            'recommended': recommended,
            'total_available': sum(1 for p in providers.values() if p.get('available')),
        }


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
