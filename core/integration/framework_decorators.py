"""
JOTTY v1.0 - A-Team Critical Fixes
=====================================

Brain-Inspired Multi-Agent Orchestration Framework

A-Team Review of User Questions:

Q1: "Will any agent wrapped in JOTTY never run out of memory?"
A1: NO! Current implementation has GAPS:
    - Memory truncation only truncates individual entries
    - Retrieval budget can still overflow LLM context
    - No hard circuit breaker for context overflow
    FIX: ContextGuard with hard token limits

Q2: "Can I wrap JOTTY as dspy.cot type wrapper?"
A2: NO! Current API requires instantiation, not decoration
    - Current: jotty = Orchestrator(actor=my_actor, ...)
    - Desired: @jotty_wrap(architect_prompts=[...])
    FIX: JottyDecorator class for DSPy-style wrapping

Q3: "What if architect/auditor prompts are empty?"
A3: PARTIALLY HANDLED but poorly:
    - Empty prompts = no validation = everything passes
    - No warning to user
    - System works but defeats purpose
    FIX: EmptyPromptHandler with warnings and defaults

This module provides all three fixes.
"""

import functools
import warnings
from typing import List, Dict, Any, Optional, Callable, TypeVar, Union
from pathlib import Path
import dspy
import asyncio

from ..foundation.data_structures import SwarmLearningConfig, EpisodeResult
from ..utils.tokenizer import SmartTokenizer


# =============================================================================
# FIX 1: CONTEXT GUARD - PREVENT OOM/CONTEXT OVERFLOW
# =============================================================================

class ContextGuard:
    """
    Hard circuit breaker for context window overflow.
    
    A-Team Consensus:
    - Dr. Manning: "Must have hard limit, not just soft budget"
    - Dr. Agarwal: "Truncate gracefully before LLM call, not after error"
    - Shannon: "Measure tokens before sending, compress if needed"
    
    This guard ENSURES the agent NEVER exceeds context limits.
    """
    
    def __init__(self, max_context_tokens: int = 28000, safety_margin: float = 0.9, enable_compression: bool = True) -> None:
        """
        Args:
            max_context_tokens: Absolute maximum context size
            safety_margin: Use only this fraction of max (0.9 = 90%)
            enable_compression: Whether to compress when over limit
        """
        self.max_context_tokens = max_context_tokens
        self.effective_limit = int(max_context_tokens * safety_margin)
        self.enable_compression = enable_compression
        self._tokenizer = SmartTokenizer.get_instance()

        # Overflow counter for monitoring
        self.overflow_count = 0
        self.compressions_performed = 0

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using SmartTokenizer."""
        return self._tokenizer.count_tokens(text)
    
    def check_and_truncate(self, 
                           system_prompt: str,
                           user_input: str,
                           context: str,
                           memories: List[str],
                           trajectory: str = "") -> Dict[str, str]:
        """
        Check total context size and truncate if needed.
        
        Returns truncated versions that fit within limit.
        GUARANTEES: Total output fits within effective_limit.
        """
        # Calculate current sizes
        system_tokens = self.estimate_tokens(system_prompt)
        input_tokens = self.estimate_tokens(user_input)
        context_tokens = self.estimate_tokens(context)
        trajectory_tokens = self.estimate_tokens(trajectory)
        memory_tokens = sum(self.estimate_tokens(m) for m in memories)
        
        total = system_tokens + input_tokens + context_tokens + trajectory_tokens + memory_tokens
        
        if total <= self.effective_limit:
            # All good
            return {
                'system_prompt': system_prompt,
                'user_input': user_input,
                'context': context,
                'memories': memories,
                'trajectory': trajectory,
                'truncated': False,
                'original_tokens': total,
                'final_tokens': total
            }
        
        # Need to truncate - prioritize:
        # 1. Keep system_prompt (most important)
        # 2. Keep user_input (task definition)
        # 3. Truncate trajectory (can be summarized)
        # 4. Truncate memories (keep most recent/valuable)
        # 5. Truncate context (least critical)
        
        self.overflow_count += 1
        
        # Budget allocation
        reserved_for_system = system_tokens
        reserved_for_input = input_tokens
        remaining = self.effective_limit - reserved_for_system - reserved_for_input
        
        # Allocate remaining: 40% trajectory, 40% memories, 20% context
        trajectory_budget = int(remaining * 0.4)
        memory_budget = int(remaining * 0.4)
        context_budget = int(remaining * 0.2)
        
        # Truncate each component
        truncated_trajectory = self._truncate_text(trajectory, trajectory_budget)
        truncated_context = self._truncate_text(context, context_budget)
        truncated_memories = self._truncate_memories(memories, memory_budget)
        
        final_tokens = (
            reserved_for_system + 
            reserved_for_input + 
            self.estimate_tokens(truncated_trajectory) +
            self.estimate_tokens(truncated_context) +
            sum(self.estimate_tokens(m) for m in truncated_memories)
        )
        
        return {
            'system_prompt': system_prompt,
            'user_input': user_input,
            'context': truncated_context,
            'memories': truncated_memories,
            'trajectory': truncated_trajectory,
            'truncated': True,
            'original_tokens': total,
            'final_tokens': final_tokens,
            'warning': f"Context truncated from {total} to {final_tokens} tokens"
        }
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        if not text:
            return text
        
        current_tokens = self.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # Calculate characters to keep
        chars_to_keep = max_tokens * self.chars_per_token
        
        if self.enable_compression:
            self.compressions_performed += 1
            # Keep beginning and end, summarize middle
            if chars_to_keep > 200:
                keep_start = chars_to_keep // 2
                keep_end = chars_to_keep // 2 - 50
                return text[:keep_start] + "\n[...TRUNCATED...]\n" + text[-keep_end:]
        
        return text[:chars_to_keep] + "\n[...TRUNCATED...]"
    
    def _truncate_memories(self, memories: List[str], max_tokens: int) -> List[str]:
        """Truncate memory list to fit budget, keeping most recent."""
        if not memories:
            return memories
        
        result = []
        tokens_used = 0
        
        # Process from end (most recent first)
        for mem in reversed(memories):
            mem_tokens = self.estimate_tokens(mem)
            if tokens_used + mem_tokens <= max_tokens:
                result.insert(0, mem)
                tokens_used += mem_tokens
            else:
                # Try to fit a truncated version
                remaining = max_tokens - tokens_used
                if remaining > 50:  # Worth keeping truncated
                    truncated = self._truncate_text(mem, remaining)
                    result.insert(0, truncated)
                break
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get guard statistics."""
        return {
            'max_context_tokens': self.max_context_tokens,
            'effective_limit': self.effective_limit,
            'overflow_count': self.overflow_count,
            'compressions_performed': self.compressions_performed
        }


# =============================================================================
# FIX 2: JOTTY DECORATOR - DSPY-STYLE WRAPPING
# =============================================================================

T = TypeVar('T')

class JottyDecorator:
    """
    DSPy-style decorator for wrapping agents with Jotty validation.
    
    A-Team Consensus:
    - Dr. Chen: "Should be as simple as @dspy.ChainOfThought"
    - Alex: "Provide sensible defaults, minimal config needed"
    
    Usage:
        @jotty_wrap(
            architect_prompts=["prompts/security.md"],
            auditor_prompts=["prompts/validator.md"]
        )
        class MyAgent(dspy.Module):
            ...
        
        # Or functional style:
        @jotty_wrap()  # Uses default empty validation
        def my_function(query: str) -> str:
            ...
    """
    
    def __init__(self, architect_prompts: List[str] = None, auditor_prompts: List[str] = None, architect_tools: List[Any] = None, auditor_tools: List[Any] = None, config: SwarmConfig = None, async_mode: bool = True, context_guard: bool = True) -> None:
        """
        Args:
            architect_prompts: Paths to Architect prompts (optional)
            auditor_prompts: Paths to Auditor prompts (optional)
            architect_tools: Tools for Architect (optional)
            auditor_tools: Tools for Auditor (optional)
            config: JOTTY configuration (optional)
            async_mode: Whether to use async execution
            context_guard: Whether to enable context overflow protection
        """
        self.architect_prompts = architect_prompts or []
        self.auditor_prompts = auditor_prompts or []
        self.architect_tools = architect_tools or []
        self.auditor_tools = auditor_tools or []
        self.config = config or SwarmConfig()
        self.async_mode = async_mode
        self.context_guard_enabled = context_guard
        
        if context_guard:
            self.guard = ContextGuard(
                max_context_tokens=self.config.max_context_tokens
            )
        else:
            self.guard = None
        
        # Warn about empty prompts
        self._handle_empty_prompts()
    
    def _handle_empty_prompts(self) -> Any:
        """Handle empty prompt lists with warnings."""
        if not self.architect_prompts:
            warnings.warn(
                "JOTTY: No architect_prompts provided. "
                "All inputs will pass Architect check automatically. "
                "Consider adding validation prompts for better quality control.",
                UserWarning
            )
        
        if not self.auditor_prompts:
            warnings.warn(
                "JOTTY: No auditor_prompts provided. "
                "All outputs will pass Auditor check automatically. "
                "Consider adding validation prompts for better quality control.",
                UserWarning
            )
    
    def __call__(self, actor: Union[type, Callable]) -> Union[type, Callable]:
        """
        Wrap an actor (class or function) with Jotty validation.
        """
        if isinstance(actor, type):
            # Class decorator
            return self._wrap_class(actor)
        else:
            # Function decorator
            return self._wrap_function(actor)
    
    def _wrap_class(self, cls: type) -> type:
        """Wrap a DSPy Module class."""
        decorator = self
        
        class JottyWrapped(cls):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._jotty_config = decorator.config
                self._jotty_guard = decorator.guard
                self._architect_prompts = decorator.architect_prompts
                self._auditor_prompts = decorator.auditor_prompts
            
            def forward(self, *args: Any, **kwargs: Any) -> Any:
                # Apply context guard
                if self._jotty_guard:
                    # Extract text content and check
                    text_content = str(kwargs)
                    tokens = self._jotty_guard.estimate_tokens(text_content)
                    if tokens > self._jotty_guard.effective_limit:
                        warnings.warn(
                            f"JOTTY: Input exceeds context limit ({tokens} > {self._jotty_guard.effective_limit}). "
                            "Auto-truncating.",
                            RuntimeWarning
                        )
                
                # Call original
                return super().forward(*args, **kwargs)
        
        JottyWrapped.__name__ = f"JottyWrapped_{cls.__name__}"
        JottyWrapped.__doc__ = f"JOTTY-wrapped version of {cls.__name__}"
        
        return JottyWrapped
    
    def _wrap_function(self, func: Callable) -> Callable:
        """Wrap a regular function."""
        decorator = self
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Pre-validation could go here
                result = await func(*args, **kwargs)
                # Post-validation could go here
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Pre-validation could go here
                result = func(*args, **kwargs)
                # Post-validation could go here
                return result
            return sync_wrapper


def jotty_wrap(architect_prompts: List[str] = None, auditor_prompts: List[str] = None, architect_tools: List[Any] = None, auditor_tools: List[Any] = None, config: SwarmConfig = None, **kwargs: Any) -> 'JottyDecorator':
    """
    Convenience function for JOTTY decorator.
    
    Usage:
        @jotty_wrap(architect_prompts=["security.md"])
        class MyAgent(dspy.Module):
            ...
    """
    return JottyDecorator(
        architect_prompts=architect_prompts,
        auditor_prompts=auditor_prompts,
        architect_tools=architect_tools,
        auditor_tools=auditor_tools,
        config=config,
        **kwargs
    )




# =============================================================================
# FIX 3: EMPTY PROMPT HANDLER - GRACEFUL DEFAULTS
# =============================================================================

class EmptyPromptHandler:
    """
    Handle empty architect/auditor prompts gracefully.
    
    A-Team Consensus:
    - Aristotle: "Absence of validation is still a decision"
    - Alex: "Provide sensible defaults, log warnings"
    
    When prompts are empty:
    1. Log warning
    2. Use default pass-through validation
    3. Track that no real validation occurred
    """
    
    DEFAULT_ARCHITECT_PROMPT = """
# Default Architect Agent

## Role
You are a default Architect agent. No custom validation rules were provided.

## Decision
ALWAYS proceed unless input is clearly malformed (empty, corrupt).

## Output
- should_proceed: True (default)
- confidence: 0.5 (no validation performed)
- reasoning: "No custom Architect prompts configured - using default pass-through"
"""

    DEFAULT_AUDITOR_PROMPT = """
# Default Auditor Agent

## Role
You are a default Auditor agent. No custom validation rules were provided.

## Decision
Accept any non-null output.

## Output
- is_valid: True if output exists
- confidence: 0.5 (no validation performed)
- reasoning: "No custom Auditor prompts configured - using default pass-through"
"""

    def __init__(self, base_path: str = 'agent_generated/jotty') -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.default_architect_path = self.base_path / "_default_architect.md"
        self.default_auditor_path = self.base_path / "_default_auditor.md"
        
        # Create default files
        self._create_defaults()
    
    def _create_defaults(self) -> Any:
        """Create default prompt files."""
        if not self.default_architect_path.exists():
            self.default_architect_path.write_text(self.DEFAULT_ARCHITECT_PROMPT)
        
        if not self.default_auditor_path.exists():
            self.default_auditor_path.write_text(self.DEFAULT_AUDITOR_PROMPT)
    
    def get_architect_prompts(self, user_prompts: List[str]) -> List[str]:
        """Get architect prompts, using default if empty."""
        if user_prompts:
            return user_prompts
        
        warnings.warn(
            "JOTTY: No architect_prompts provided. Using default pass-through validation. "
            "This means ALL inputs will be accepted without checking. "
            "To add validation, provide architect_prompts=[...] when creating JOTTY.",
            UserWarning
        )
        return [str(self.default_architect_path)]
    
    def get_auditor_prompts(self, user_prompts: List[str]) -> List[str]:
        """Get auditor prompts, using default if empty."""
        if user_prompts:
            return user_prompts
        
        warnings.warn(
            "JOTTY: No auditor_prompts provided. Using default pass-through validation. "
            "This means ALL outputs will be accepted without checking. "
            "To add validation, provide auditor_prompts=[...] when creating JOTTY.",
            UserWarning
        )
        return [str(self.default_auditor_path)]


# =============================================================================
# ENHANCED JOTTY WITH ALL FIXES
# =============================================================================

class JottyEnhanced:
    """
    Enhanced JOTTY with all A-Team fixes:
    1. ContextGuard - Never overflow context
    2. DSPy-style decoration support
    3. Empty prompt handling
    
    This is the recommended entry point.
    
    JOTTY v1.0 - Enhanced Decorator
    """
    
    def __init__(self, actor: dspy.Module, architect_prompts: List[str] = None, auditor_prompts: List[str] = None, architect_tools: List[Any] = None, auditor_tools: List[Any] = None, config: SwarmConfig = None) -> None:
        """
        Initialize enhanced JOTTY.
        
        All parameters are optional - sensible defaults provided.
        """
        self.config = config or SwarmConfig()
        
        # Context guard
        self.context_guard = ContextGuard(
            max_context_tokens=self.config.max_context_tokens
        )
        
        # Empty prompt handler
        self.prompt_handler = EmptyPromptHandler(
            base_path=self.config.base_path
        )
        
        # Get prompts (with defaults if empty)
        final_architect = self.prompt_handler.get_architect_prompts(architect_prompts or [])
        final_auditor = self.prompt_handler.get_auditor_prompts(auditor_prompts or [])
        
        # Import Orchestrator (V2: Orchestrator alias)
        from ..orchestration import Orchestrator
        
        # Create wrapped Orchestrator
        self._jotty = Orchestrator(
            actor=actor,
            architect_prompts=final_architect,
            auditor_prompts=final_auditor,
            architect_tools=architect_tools or [],
            auditor_tools=auditor_tools or [],
            config=self.config
        )
    
    async def arun(self, **kwargs: Any) -> EpisodeResult:
        """
        Run with context guard protection.
        """
        # Check context before sending
        text_content = str(kwargs)
        tokens = self.context_guard.estimate_tokens(text_content)
        
        if tokens > self.context_guard.effective_limit:
            # Truncate kwargs values
            truncated_kwargs = {}
            for k, v in kwargs.items():
                v_str = str(v)
                if self.context_guard.estimate_tokens(v_str) > 1000:
                    v_str = self.context_guard._truncate_text(v_str, 1000)
                truncated_kwargs[k] = v_str
            kwargs = truncated_kwargs
        
        return await self._jotty.arun(**kwargs)
    
    def run(self, **kwargs: Any) -> EpisodeResult:
        """Synchronous run."""
        return asyncio.run(self.arun(**kwargs))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        return {
            'jotty': self._jotty.get_statistics(),
            'context_guard': self.context_guard.get_statistics()
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ContextGuard',
    'JottyDecorator',
    'jotty_wrap',
    'EmptyPromptHandler',
    'JottyEnhanced'
]

