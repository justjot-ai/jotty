"""
Tool Execution Guard
=====================

Transparent wrapper applied at SkillDefinition.tools load time.
Integrates ToolInterceptor (call tracking) and ProviderHealthManager
(circuit breaker deferral) into every tool call.

Injection point: SkillDefinition.tools property in skills_registry.py.
This single point covers ALL 164+ skills across both execution paths:
- Path A: SkillPlanExecutor gets tools from registry
- Path B: Agent calls get tools from registry

Usage:
    from Jotty.core.infrastructure.integration.tool_execution_guard import (
        get_tool_execution_guard,
    )

    guard = get_tool_execution_guard()
    wrapped_tools = guard.wrap_skill_tools("web-search", raw_tools)
"""

import asyncio
import functools
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

from Jotty.core.infrastructure.integration.tool_interceptor import ToolCallRegistry
from Jotty.core.infrastructure.utils.provider_health import get_provider_health

logger = logging.getLogger(__name__)

# Provider inference from skill name
_SKILL_PROVIDER_MAP: Dict[str, str] = {
    "openai": "openai",
    "claude": "anthropic",
    "anthropic": "anthropic",
    "gemini": "google",
    "groq": "groq",
    "serper": "serper",
}


def _infer_provider(skill_name: str, result: Any) -> Optional[str]:
    """
    Infer the external provider from a tool result or skill name.

    Priority:
    1. Result dict has "provider" key → use it
    2. Skill name matches known provider → use mapping
    3. Default: None (no provider tracking)
    """
    # 1. Result dict with explicit provider
    if isinstance(result, dict) and "provider" in result:
        return result["provider"]

    # 2. Skill name heuristics (skip multi-provider skills like "web-search")
    lower_name = skill_name.lower().replace("-", "").replace("_", "")
    for keyword, provider in _SKILL_PROVIDER_MAP.items():
        if keyword in lower_name:
            return provider

    return None


class ToolExecutionGuard:
    """
    Transparent wrapper that integrates tool call tracking and provider health.

    Wraps tool functions at SkillDefinition.tools load time to:
    1. Check tool access policy (deny-wins cascade)
    2. Record call metrics in ToolCallRegistry (for learning)
    3. Record success/failure in ProviderHealthManager (for deferral)
    """

    def __init__(self, workspace_dir: str = "") -> None:
        self._registry = ToolCallRegistry()
        self._health = get_provider_health()
        self._workspace_dir = workspace_dir

    @property
    def interceptor_registry(self) -> ToolCallRegistry:
        """Access the underlying ToolCallRegistry."""
        return self._registry

    @property
    def health_manager(self):
        """Access the underlying ProviderHealthManager."""
        return self._health

    def _check_policy(self, tool_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if tool is allowed by policy chain. Returns True if allowed."""
        if not self._workspace_dir:
            return True
        try:
            from Jotty.core.infrastructure.integration.tool_policy import get_policy_chain

            chain = get_policy_chain(self._workspace_dir)
            channel = (context or {}).get("channel")
            group = (context or {}).get("group")
            return chain.evaluate(tool_name, channel=channel, group=group)
        except Exception:
            return True  # Fail open

    def wrap_skill_tools(self, skill_name: str, tools: Dict[str, Callable]) -> Dict[str, Callable]:
        """
        Wrap all tools from a skill with tracking and health recording.

        Args:
            skill_name: Name of the skill (for provider inference)
            tools: Dict of {tool_name: tool_function}

        Returns:
            Dict of {tool_name: wrapped_function}
        """
        wrapped: Dict[str, Callable] = {}
        interceptor = self._registry.get_or_create_interceptor(skill_name)

        for tool_name, tool_func in tools.items():
            if asyncio.iscoroutinefunction(tool_func):
                wrapped[tool_name] = self._wrap_async(skill_name, tool_name, tool_func, interceptor)
            else:
                wrapped[tool_name] = self._wrap_sync(skill_name, tool_name, tool_func, interceptor)

        return wrapped

    def _wrap_sync(
        self, skill_name: str, tool_name: str, tool_func: Callable, interceptor: Any
    ) -> Callable:
        """Create a sync wrapper with tracking and policy check."""

        @functools.wraps(tool_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Policy check: deny-wins cascade
            context = kwargs.get("_policy_context")
            if not self._check_policy(tool_name, context):
                return {"success": False, "error": f"Tool '{tool_name}' blocked by policy"}

            start = time.time()
            try:
                result = tool_func(*args, **kwargs)
                elapsed = time.time() - start

                # Determine success from result
                success = True
                if isinstance(result, dict) and not result.get("success", True):
                    success = False

                # Record in interceptor
                from Jotty.core.infrastructure.integration.tool_interceptor import ToolCall

                interceptor._calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=kwargs if kwargs else ({"positional": list(args)} if args else {}),
                        result=result,
                        success=success,
                        metadata={"skill": skill_name, "latency": elapsed},
                    )
                )

                # Record provider health
                provider = _infer_provider(skill_name, result)
                if provider:
                    if success:
                        self._health.record_success(provider)
                    else:
                        self._health.record_failure(provider)

                return result
            except Exception as e:
                elapsed = time.time() - start

                # Record failure in interceptor
                from Jotty.core.infrastructure.integration.tool_interceptor import ToolCall

                interceptor._calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=kwargs if kwargs else ({"positional": list(args)} if args else {}),
                        result=None,
                        success=False,
                        error=str(e),
                        metadata={"skill": skill_name, "latency": elapsed},
                    )
                )

                # Record provider failure
                provider = _infer_provider(skill_name, None)
                if provider:
                    self._health.record_failure(provider, e)

                raise

        return wrapper

    def _wrap_async(
        self, skill_name: str, tool_name: str, tool_func: Callable, interceptor: Any
    ) -> Callable:
        """Create an async wrapper with tracking and policy check."""

        @functools.wraps(tool_func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Policy check: deny-wins cascade
            context = kwargs.get("_policy_context")
            if not self._check_policy(tool_name, context):
                return {"success": False, "error": f"Tool '{tool_name}' blocked by policy"}

            start = time.time()
            try:
                result = await tool_func(*args, **kwargs)
                elapsed = time.time() - start

                success = True
                if isinstance(result, dict) and not result.get("success", True):
                    success = False

                from Jotty.core.infrastructure.integration.tool_interceptor import ToolCall

                interceptor._calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=kwargs if kwargs else ({"positional": list(args)} if args else {}),
                        result=result,
                        success=success,
                        metadata={"skill": skill_name, "latency": elapsed},
                    )
                )

                provider = _infer_provider(skill_name, result)
                if provider:
                    if success:
                        self._health.record_success(provider)
                    else:
                        self._health.record_failure(provider)

                return result
            except Exception as e:
                elapsed = time.time() - start

                from Jotty.core.infrastructure.integration.tool_interceptor import ToolCall

                interceptor._calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=kwargs if kwargs else ({"positional": list(args)} if args else {}),
                        result=None,
                        success=False,
                        error=str(e),
                        metadata={"skill": skill_name, "latency": elapsed},
                    )
                )

                provider = _infer_provider(skill_name, None)
                if provider:
                    self._health.record_failure(provider, e)

                raise

        return wrapper


# =============================================================================
# Singleton
# =============================================================================

_lock = threading.Lock()
_instance: Optional[ToolExecutionGuard] = None


def get_tool_execution_guard(workspace_dir: str = "") -> ToolExecutionGuard:
    """Get singleton ToolExecutionGuard instance."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                # Auto-detect workspace_dir from cwd if not provided
                if not workspace_dir:
                    import os

                    workspace_dir = os.getcwd()
                _instance = ToolExecutionGuard(workspace_dir=workspace_dir)
    return _instance
