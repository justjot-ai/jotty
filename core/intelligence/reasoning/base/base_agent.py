"""
BaseAgent - Abstract Base Class for All Jotty Agents

Provides unified infrastructure with lazy initialization:
- DSPy LM auto-configuration
- Memory integration (SwarmMemory)
- Context management (SharedContext)
- Cost tracking and monitoring
- Skills registry access
- Pre/post execution hooks
- Retry logic with exponential backoff

All agent types (DomainAgent, MetaAgent, AutonomousAgent) inherit from this.

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Re-export for backward compatibility — other modules import this lock
# from base_agent (e.g. swarm_learning.py line 165).
from Jotty.core.infrastructure.foundation.dspy_init import _dspy_lm_lock  # noqa: F401

# =============================================================================
# CONFIGURATION AND RESULT DATACLASSES
# =============================================================================


@dataclass
class AgentRuntimeConfig:
    """Unified runtime configuration for all agent types.

    All numeric defaults are 0 / 0.0 sentinels, resolved from
    ``Jotty.core.foundation.config_defaults`` in ``__post_init__``.
    This keeps a single source of truth for LLM settings.

    Note: This is the *runtime* config (model, temperature, enable_memory, etc.).
    For orchestration-level agent specification, see
    ``Jotty.core.foundation.agent_config.AgentConfig``.
    """

    name: str = ""
    model: str = ""  # "" → DEFAULT_MODEL_ALIAS
    llm_provider: str = ""  # "" → "anthropic" (auto-detect if empty)
    temperature: float = 0.0  # 0.0 → LLM_TEMPERATURE
    max_tokens: int = 0  # 0 → LLM_MAX_OUTPUT_TOKENS
    max_retries: int = 0  # 0 → MAX_RETRIES
    retry_delay: float = 0.0  # 0.0 → RETRY_BACKOFF_SECONDS
    timeout: float = 0.0  # 0.0 → LLM_TIMEOUT_SECONDS
    enable_memory: bool = True
    enable_context: bool = True
    enable_monitoring: bool = True
    enable_skills: bool = True
    enable_safety_gates: bool = True  # Safety validation (PII, cost, malicious input)
    system_prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Safety gate configuration
    safety_max_cost_usd: float = 1.0  # Maximum cost per execution
    safety_min_quality: float = 0.7  # Minimum output quality threshold
    safety_rate_limit_per_minute: int = 100  # Max API calls per minute

    def __post_init__(self) -> None:
        from Jotty.core.infrastructure.foundation.config_defaults import DEFAULTS

        if not self.name:
            self.name = self.__class__.__name__
        if not self.model:
            self.model = DEFAULTS.DEFAULT_MODEL_ALIAS
        if self.temperature == 0.0:
            self.temperature = DEFAULTS.LLM_TEMPERATURE
        if self.max_tokens <= 0:
            self.max_tokens = DEFAULTS.LLM_MAX_OUTPUT_TOKENS
        if self.max_retries <= 0:
            self.max_retries = DEFAULTS.MAX_RETRIES
        if self.retry_delay == 0.0:
            self.retry_delay = DEFAULTS.RETRY_BACKOFF_SECONDS
        if self.timeout == 0.0:
            self.timeout = float(DEFAULTS.LLM_TIMEOUT_SECONDS)


@dataclass
class AgentResult:
    """Standardized result from any agent execution."""

    success: bool
    output: Any
    agent_name: str = ""
    execution_time: float = 0.0
    retries: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output": self.output,
            "agent_name": self.agent_name,
            "execution_time": self.execution_time,
            "retries": self.retries,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# BASE AGENT ABSTRACT CLASS
# =============================================================================


class BaseAgent(ABC):
    """
    Abstract base class for all Jotty agents.

    Provides lazy-initialized infrastructure:
    - Memory: SwarmMemory for knowledge storage/retrieval
    - Context: SharedContext for cross-agent coordination
    - Monitoring: Cost tracking and metrics
    - Skills: SkillsRegistry for tool access
    - Collaboration: Agent-to-agent communication (optional mixin)

    Subclasses implement _execute_impl() for their specific logic.

    Usage:
        class MyAgent(BaseAgent):
            async def _execute_impl(self, **kwargs) -> Any:
                return {"result": "done"}

        agent = MyAgent(config=AgentRuntimeConfig(name="MyAgent"))
        result = await agent.execute(task="do something")
    """

    # Visual Verification Protocol prompt — appended to planning context
    # when visual skills (browser-automation, visual-inspector) are available.
    VVP_PROMPT = (
        "When visual-inspector or screenshot tools are available, use them to verify "
        "the results of UI-affecting actions (browser navigation, form fills, file creation). "
        "After taking an action, capture a screenshot and describe what you see to confirm success."
    )

    # Maximum exponential backoff delay (seconds) — caps retry waits
    MAX_RETRY_DELAY = 60.0

    # Error patterns that are never worth LLM-analyzing (just retry)
    _BLIND_RETRY_PATTERNS = ("rate limit", "429", "overloaded", "capacity")

    # Sentinel for "not yet set" — distinguishes from an explicit ``None``
    # injected by a swarm.  Lazy properties only create new instances when
    # the field is _UNSET; an explicit ``None`` or real value is respected.
    _UNSET: Any = object()

    def __init__(self, config: AgentRuntimeConfig = None) -> None:  # type: ignore[assignment]
        """
        Initialize BaseAgent with optional configuration.

        Args:
            config: Agent configuration. Defaults to AgentRuntimeConfig with class name.
        """
        self.config = config or AgentRuntimeConfig(name=self.__class__.__name__)
        if not self.config.name:
            self.config.name = self.__class__.__name__

        # Lazy-initialized infrastructure — _UNSET means "create on first access".
        # When a swarm injects resources via the property setters, the sentinel
        # is replaced and lazy creation is skipped.
        self._memory = self._UNSET
        self._memory_persistence = None
        self._context_manager = self._UNSET
        self._cost_tracker = None
        self._skills_registry = self._UNSET
        self._lm = None
        self._safety_validator = None  # Safety gates (ValidatorAgent)

        # Execution hooks
        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []

        # Agent collaboration infrastructure (optional)
        self._agent_directory: Dict[str, Any] = {}
        self._agent_slack: List[Dict[str, Any]] = []
        self._collaboration_history: List[Dict[str, Any]] = []

        # Metrics
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_retries": 0,
            "total_execution_time": 0.0,
        }

        self._initialized = False

    def set_jotty_config(self, config: Any) -> "BaseAgent":
        """Inject a shared SwarmConfig so lazy-loaded components (memory, etc.)
        use the same configuration as the rest of the system.

        Args:
            config: SwarmConfig instance from Orchestrator or CLI

        Returns:
            self (for chaining)
        """
        self._jotty_config = config
        return self

    # =========================================================================
    # LAZY INITIALIZATION
    # =========================================================================

    def _ensure_initialized(self) -> None:
        """Ensure all infrastructure is initialized (lazy loading)."""
        if self._initialized:
            return

        self._init_dspy_lm()
        self._init_safety_gates()
        self._initialized = True
        logger.debug(f"BaseAgent '{self.config.name}' initialized")

    def _load_api_keys(self) -> Any:
        """Load API keys from .env.anthropic and .env files if not in environment."""
        from Jotty.core.infrastructure.foundation.dspy_init import load_api_keys

        load_api_keys()

    def _init_dspy_lm(self) -> None:
        """Auto-configure DSPy LM if not already set.

        Thread-safe: delegates to the shared ``dspy_init.init_dspy_lm()``
        which uses a module-level lock.
        """
        from Jotty.core.infrastructure.foundation.dspy_init import init_dspy_lm

        provider = self.config.llm_provider or None  # None → auto-detect
        lm = init_dspy_lm(
            provider=provider,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=int(self.config.timeout),
        )
        if lm is not None:
            self._lm = lm

    @staticmethod
    def _auto_detect_provider() -> str:
        """Auto-detect available LM provider from environment."""
        from Jotty.core.infrastructure.foundation.dspy_init import auto_detect_provider

        return auto_detect_provider()

    def _init_safety_gates(self) -> None:
        """Initialize safety gates (ValidatorAgent) with default constraints.

        Safety gates protect against:
        - Malicious input (prompt injection, jailbreaks)
        - Cost overruns (budget limits)
        - PII leakage (SSN, credit cards, emails)
        - Low quality output
        - Rate limit violations
        """
        if not self.config.enable_safety_gates:
            logger.debug(f"Safety gates disabled for {self.config.name}")
            return

        try:
            from Jotty.core.infrastructure.monitoring.safety_gates import (  # type: ignore[import]
                CostBudgetConstraint,
                MaliciousInputConstraint,
                PIIConstraint,
                QualityThresholdConstraint,
                RateLimitConstraint,
                ValidatorAgent,
            )

            # Create default safety constraints
            constraints = [
                MaliciousInputConstraint(enabled=True),
                CostBudgetConstraint(
                    max_cost_usd=self.config.safety_max_cost_usd,
                    enabled=True,
                ),
                PIIConstraint(enabled=True),
                QualityThresholdConstraint(
                    min_quality=self.config.safety_min_quality,
                    enabled=True,
                ),
                RateLimitConstraint(
                    max_calls_per_minute=self.config.safety_rate_limit_per_minute,
                    enabled=True,
                ),
            ]

            self._safety_validator = ValidatorAgent(constraints=constraints)

            # Register safety hooks
            self.add_pre_hook(self._safety_pre_execution_gate)
            self.add_post_hook(self._safety_post_execution_gate)

            logger.info(
                f"✓ Safety gates enabled for {self.config.name} "
                f"(cost_limit=${self.config.safety_max_cost_usd}, "
                f"quality_min={self.config.safety_min_quality})"
            )
        except Exception as e:
            logger.warning(f"Could not initialize safety gates: {e}")
            self._safety_validator = None

    async def _safety_pre_execution_gate(self, agent: "BaseAgent", **kwargs: Any) -> None:
        """Pre-execution safety gate - validates input before execution.

        Args:
            agent: The agent instance (same as self when called as hook)
            **kwargs: Execution arguments

        Checks:
        - Malicious input (prompt injection, jailbreaks)
        - Rate limits (not calling APIs too fast)
        - Cost budget (have budget left)

        Raises:
            ValueError: If safety gate blocks execution
        """
        if not agent._safety_validator:
            return

        # Build validation context
        context = {
            "user_input": str(kwargs.get("task", kwargs.get("query", ""))),
            "cost_usd": 0.0,  # Pre-execution cost is 0
            "agent_name": agent.config.name,
            "timestamp": time.time(),
        }

        # Run pre-execution validators
        report = agent._safety_validator.validate_pre_execution(context)

        if not report.passed:
            # Collect blocking failures
            failures = [f.message for f in report.blocking_failures]
            error_msg = "; ".join(failures)
            logger.error(f"🚨 SAFETY GATE BLOCKED PRE-EXECUTION: {error_msg}")
            raise ValueError(f"Safety gate blocked execution: {error_msg}")

        # Log warnings (non-blocking)
        for warning in report.warnings:
            logger.warning(f"⚠️  Safety warning (pre-execution): {warning.message}")

    async def _safety_post_execution_gate(
        self, agent: "BaseAgent", result: AgentResult, **kwargs: Any
    ) -> None:
        """Post-execution safety gate - validates output after execution.

        Args:
            agent: The agent instance (same as self when called as hook)
            result: Execution result to validate
            **kwargs: Execution arguments

        Checks:
        - PII in output (SSN, credit cards, emails, phone numbers)
        - Quality threshold (minimum quality score)

        Modifies result in-place if safety gate blocks output.
        """
        if not agent._safety_validator:
            return

        # Build validation context
        context = {
            "output": str(result.output) if result.output else "",
            "quality_score": result.metadata.get("quality", 0.8),
            "agent_name": agent.config.name,
            "execution_time": result.execution_time,
            "cost_usd": result.metadata.get("cost_usd", 0.0),
            "timestamp": time.time(),
        }

        # Run post-execution validators
        report = agent._safety_validator.validate_post_execution(context)

        if not report.passed:
            # Collect blocking failures
            failures = [f.message for f in report.blocking_failures]
            error_msg = "; ".join(failures)
            logger.error(f"🚨 SAFETY GATE BLOCKED POST-EXECUTION: {error_msg}")

            # Modify result in-place to mark as failed
            result.success = False
            result.error = f"Safety gate blocked output: {error_msg}"
            result.output = None  # Clear potentially unsafe output

        # Log warnings (non-blocking)
        for warning in report.warnings:
            logger.warning(f"⚠️  Safety warning (post-execution): {warning.message}")

    @property
    def memory(self) -> Any:
        """Lazy-load SwarmMemory with disk persistence.

        Uses sentinel pattern: only creates a new instance when the field
        is ``_UNSET``.  If a swarm injects a shared memory instance via
        the setter, it is respected and lazy creation is skipped.
        """
        if self._memory is not self._UNSET:
            return self._memory
        if not self.config.enable_memory:
            return None
        try:
            from pathlib import Path

            from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
            from Jotty.core.intelligence.memory.cortex import SwarmMemory
            from Jotty.core.intelligence.memory.memory_persistence import (
                enable_memory_persistence,
            )

            jotty_config = getattr(self, "_jotty_config", None) or SwarmConfig()
            self._memory = SwarmMemory(config=jotty_config, agent_name=self.config.name)

            persistence_dir = Path.home() / "jotty" / "memory" / self.config.name
            self._memory_persistence = enable_memory_persistence(
                memory=self._memory,
                persistence_dir=persistence_dir,
            )
            logger.debug(f"Initialized SwarmMemory with persistence for {self.config.name}")
        except Exception as e:
            logger.warning(f"Could not initialize memory: {e}")
            self._memory = None
        return self._memory

    @memory.setter
    def memory(self, value: Any) -> None:
        """Allow swarms to inject a shared memory instance."""
        self._memory = value

    @property
    def context(self) -> Any:
        """Lazy-load context manager (SmartContextManager preferred, SharedContext fallback).

        Uses sentinel pattern: if a swarm injects a context manager via
        the setter, lazy creation is skipped.
        """
        if self._context_manager is not self._UNSET:
            return self._context_manager
        if not self.config.enable_context:
            return None
        try:
            from Jotty.core.infrastructure.context.facade import get_context_manager

            self._context_manager = get_context_manager()
            logger.debug(f"Initialized SmartContextManager for {self.config.name}")
        except Exception:
            try:
                from Jotty.core.infrastructure.persistence.shared_context import (
                    SharedContext,  # type: ignore[import]
                )

                self._context_manager = SharedContext()
                logger.debug(f"Initialized SharedContext for {self.config.name}")
            except Exception as e:
                logger.warning(f"Could not initialize context: {e}")
                self._context_manager = None
        return self._context_manager

    @context.setter
    def context(self, value: Any) -> None:
        """Allow swarms to inject a shared context manager."""
        self._context_manager = value

    @property
    def skills_registry(self) -> Any:
        """Lazy-load SkillsRegistry.

        Uses sentinel pattern: if a swarm injects a registry via the
        setter, lazy creation is skipped.
        """
        if self._skills_registry is not self._UNSET:
            return self._skills_registry
        if not self.config.enable_skills:
            return None
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

            self._skills_registry = get_skills_registry()
            if not self._skills_registry.initialized:  # type: ignore[attr-defined]
                self._skills_registry.init()  # type: ignore[attr-defined]
            logger.debug(f"Initialized SkillsRegistry for {self.config.name}")
        except Exception as e:
            logger.warning(f"Could not initialize skills registry: {e}")
            self._skills_registry = None
        return self._skills_registry

    @skills_registry.setter
    def skills_registry(self, value: Any) -> None:
        """Allow swarms to inject a shared skills registry."""
        self._skills_registry = value

    # =========================================================================
    # EXECUTION WITH RETRY LOGIC
    # =========================================================================

    async def execute(self, **kwargs: Any) -> AgentResult:
        """
        Execute the agent with intelligent retry and trajectory preservation.

        On failure:
        1. Classifies error type (infrastructure vs logic vs data)
        2. For non-trivial errors, uses LLM to analyze failure and suggest fix
        3. Injects analysis + trajectory into next attempt's kwargs
        4. Preserves what was tried across retries (trajectory)

        Args:
            **kwargs: Arguments passed to _execute_impl

        Returns:
            AgentResult with success status, output, and metadata
        """
        self._ensure_initialized()
        start_time = time.time()
        retries = 0
        last_error = None
        trajectory: List[Dict[str, Any]] = []  # What was tried and what happened

        # Run pre-execution hooks
        try:
            await self._run_pre_hooks(**kwargs)
        except Exception as e:
            logger.warning(f"Pre-hook failed: {e}")

        # Retry loop with LLM-analyzed recovery
        for attempt in range(self.config.max_retries):
            try:
                # Inject trajectory context for retries
                if trajectory:
                    kwargs["_retry_trajectory"] = trajectory
                    kwargs["_retry_guidance"] = trajectory[-1].get("guidance", "")

                # Execute the implementation
                if asyncio.iscoroutinefunction(self._execute_impl):
                    output = await asyncio.wait_for(
                        self._execute_impl(**kwargs), timeout=self.config.timeout
                    )
                else:
                    output = await asyncio.wait_for(
                        asyncio.to_thread(self._execute_impl, **kwargs), timeout=self.config.timeout
                    )

                execution_time = time.time() - start_time

                # Update metrics
                self._metrics["total_executions"] += 1
                self._metrics["successful_executions"] += 1
                self._metrics["total_retries"] += retries
                self._metrics["total_execution_time"] += execution_time

                result = AgentResult(
                    success=True,
                    output=output,
                    agent_name=self.config.name,
                    execution_time=execution_time,
                    retries=retries,
                )

                # Run post-execution hooks
                try:
                    await self._run_post_hooks(result, **kwargs)
                except Exception as e:
                    logger.warning(f"Post-hook failed: {e}")

                # Record to LearningService
                self._record_to_learning_service(
                    success=True,
                    execution_time=execution_time,
                    kwargs=kwargs,
                    output=output,
                    retries=retries,
                )

                return result

            except asyncio.TimeoutError:
                last_error = f"Execution timed out after {self.config.timeout}s"
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries} timed out")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries} failed: {e}")

            # Record trajectory entry for this failed attempt
            trajectory.append(
                {
                    "attempt": attempt + 1,
                    "error": last_error,
                    "guidance": "",  # Filled by LLM analysis below
                }
            )

            # Exponential backoff with max cap before retry
            if attempt < self.config.max_retries - 1:
                retries += 1
                delay = min(
                    self.config.retry_delay * (2**attempt),
                    self.MAX_RETRY_DELAY,
                )

                # LLM-analyzed retry: analyze failure for non-trivial errors
                guidance = self._analyze_failure(last_error, kwargs)
                if guidance:
                    trajectory[-1]["guidance"] = guidance
                    logger.info(f"Retry guidance: {guidance[:100]}")

                logger.info(
                    f"Retrying in {delay:.1f}s (attempt {attempt + 2}/{self.config.max_retries})"
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        execution_time = time.time() - start_time
        self._metrics["total_executions"] += 1
        self._metrics["failed_executions"] += 1
        self._metrics["total_retries"] += retries
        self._metrics["total_execution_time"] += execution_time

        # Record failure to LearningService
        self._record_to_learning_service(
            success=False,
            execution_time=execution_time,
            kwargs=kwargs,
            error=last_error,
            retries=retries,
        )

        return AgentResult(
            success=False,
            output=None,
            agent_name=self.config.name,
            execution_time=execution_time,
            retries=retries,
            error=last_error,
            metadata={"trajectory": trajectory},
        )

    def _record_to_learning_service(
        self,
        success: bool,
        execution_time: float,
        kwargs: Dict[str, Any],
        output: Any = None,
        error: Optional[str] = None,
        retries: int = 0,
    ) -> None:
        """Record execution outcome to the unified LearningService."""
        try:
            from Jotty.core.intelligence.learning.learning_service import LearningService

            service = LearningService.get_instance()
            task_str = str(kwargs.get("task", kwargs.get("query", kwargs.get("input", ""))))[:200]

            output_str = str(output) if output else ""
            agent_outcome: Dict[str, Any] = {
                "output_length": len(output_str),
            }
            if output_str:
                excerpt = (
                    output_str[:600].rsplit("\n", 1)[0] if len(output_str) > 600 else output_str
                )
                agent_outcome["response_excerpt"] = excerpt
            service.record(
                unit_name=self.config.name,
                unit_type="agent",
                domain=self.config.parameters.get("domain", "general"),
                task_type=self.__class__.__name__,
                context={"task": task_str},
                action={"model": self.config.model, "retries": retries},
                outcome=agent_outcome,
                success=success,
                quality=0.8 if success else 0.0,
                execution_time=execution_time,
                error_type=self._classify_error_type(error) if error else None,
                error_message=error,
            )
        except Exception as e:
            logger.debug(f"LearningService record failed: {e}")

    @staticmethod
    def _classify_error_type(error: str) -> str:
        """Classify an error string into a category for learning."""
        err_lower = (error or "").lower()
        if "timeout" in err_lower:
            return "timeout"
        if "rate limit" in err_lower or "429" in err_lower:
            return "rate_limit"
        if "api" in err_lower or "connection" in err_lower:
            return "infrastructure"
        if "permission" in err_lower or "auth" in err_lower:
            return "authentication"
        return "execution_failure"

    def _analyze_failure(self, error: str, kwargs: Dict[str, Any]) -> str:
        """Analyze a failure and return guidance for the next retry attempt.

        For rate-limit/capacity errors, returns empty (blind retry is fine).
        For logic/data errors, uses ErrorType classification to provide guidance.

        Returns guidance string or empty if blind retry is appropriate.
        """
        error_lower = error.lower()

        # Don't waste LLM call on rate limits — just retry
        if any(p in error_lower for p in self._BLIND_RETRY_PATTERNS):
            return ""

        try:
            from Jotty.core.intelligence.orchestration.execution.types import (
                ErrorType,  # type: ignore[import-not-found, import]
            )

            error_type = ErrorType.classify(error)

            guidance_map = {
                ErrorType.LOGIC: (
                    f"Previous attempt failed with a logic error: {error[:200]}. "
                    "Try a different approach — check parameter names, types, and formats."
                ),
                ErrorType.DATA: (
                    f"Previous attempt got bad data: {error[:200]}. "
                    "Validate inputs, try alternative data sources or formats."
                ),
                ErrorType.ENVIRONMENT: (
                    f"Environment issue detected: {error[:200]}. "
                    "Consider SSL bypass, proxy settings, or alternative endpoints."
                ),
                ErrorType.INFRASTRUCTURE: (
                    f"Infrastructure error: {error[:200]}. " "Retrying with same approach."
                ),
            }
            return guidance_map.get(error_type, "")
        except Exception:
            return ""

    @abstractmethod
    async def _execute_impl(self, **kwargs: Any) -> Any:
        """
        Implement the agent's core execution logic.

        Subclasses must override this method.

        Args:
            **kwargs: Execution arguments

        Returns:
            The execution output (any type)
        """
        pass

    # =========================================================================
    # HOOK MANAGEMENT
    # =========================================================================

    def add_pre_hook(self, hook: Callable) -> None:
        """Add a pre-execution hook."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable) -> None:
        """Add a post-execution hook."""
        self._post_hooks.append(hook)

    async def _run_pre_hooks(self, **kwargs: Any) -> Any:
        """Run all pre-execution hooks."""
        for hook in self._pre_hooks:
            if asyncio.iscoroutinefunction(hook):
                await hook(self, **kwargs)
            else:
                hook(self, **kwargs)

    async def _run_post_hooks(self, result: AgentResult, **kwargs: Any) -> Any:
        """Run all post-execution hooks."""
        for hook in self._post_hooks:
            if asyncio.iscoroutinefunction(hook):
                await hook(self, result, **kwargs)
            else:
                hook(self, result, **kwargs)

    # =========================================================================
    # MEMORY HELPERS
    # =========================================================================

    def store_memory(
        self,
        content: str,
        level: str = "episodic",
        context: Dict[str, Any] | None = None,
        goal: str = "",
    ) -> Any:
        """
        Store content in hierarchical memory.

        Args:
            content: Content to store
            level: Memory level (working, episodic, semantic, procedural, strategic)
            context: Additional context metadata
            goal: Goal associated with this memory
        """
        if self.memory is None:
            logger.debug("Memory not available, skipping store")
            return

        try:
            from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

            level_enum = MemoryLevel[level.upper()] if isinstance(level, str) else level
            self.memory.store(content=content, level=level_enum, context=context or {}, goal=goal)
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")

    def retrieve_memory(self, query: str, goal: str = "", budget_tokens: int = 1000) -> List[Any]:
        """
        Retrieve relevant memories.

        Args:
            query: Query for retrieval
            goal: Goal to guide retrieval
            budget_tokens: Maximum tokens to retrieve

        Returns:
            List of relevant memory entries
        """
        if self.memory is None:
            return []

        try:
            return self.memory.retrieve(query=query, goal=goal, budget_tokens=budget_tokens)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(f"Failed to retrieve memory: {e}")
            return []

    def save_memory(self) -> bool:
        """Persist memory to disk. Returns True if successful."""
        if self._memory_persistence:
            return self._memory_persistence.save()
        return False

    # =========================================================================
    # CONTEXT HELPERS
    # =========================================================================

    def register_context(self, key: str, value: Any) -> None:
        """Register data in shared context for other agents."""
        if self.context is not None:
            self.context.set(key, value)

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get data from shared context."""
        if self.context is not None:
            return self.context.get(key, default)
        return default

    def get_compressed_context(self, max_tokens: int = 2000) -> str:
        """Get compressed context summary for prompts."""
        if self.context is None:
            return ""

        try:
            # Get recent context and compress if needed
            all_context = {}
            for key in ["current_task", "current_goal", "recent_outputs", "agent_states"]:
                value = self.context.get(key)
                if value:
                    all_context[key] = value

            import json

            context_str = json.dumps(all_context, default=str)

            # Truncate if too long (rough token estimate: 4 chars per token)
            max_chars = max_tokens * 4
            if len(context_str) > max_chars:
                context_str = context_str[:max_chars] + "..."

            return context_str
        except Exception as e:
            logger.warning(f"Failed to get compressed context: {e}")
            return ""

    # =========================================================================
    # SKILL DISCOVERY
    # =========================================================================

    def discover_skills(self, task: str = "") -> List[Dict[str, Any]]:
        """
        Get all skills for LLM selection.

        Returns the full skill catalog. The LLM sees everything (~5K tokens)
        and picks the best match. No keyword pre-filtering - the LLM is
        better at semantic matching than any keyword heuristic.

        Args:
            task: Unused (kept for backward compatibility). Filtering
                  is done by the LLM in select_skills().

        Returns:
            List of all skill dicts from the registry
        """
        if self.skills_registry is None:
            logger.debug("Skills registry not available for discovery")
            return []

        return self.skills_registry.list_skills()  # type: ignore[no-any-return]

    # =========================================================================
    # SYSTEM CONTEXT & VISUAL VERIFICATION
    # =========================================================================

    def _get_system_context(self, discovered_skills: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Build the system prompt context for planning/execution.

        Auto-appends VVP_PROMPT when visual skills are discovered.
        Subclasses can override to add domain-specific context.

        Args:
            discovered_skills: Discovered skill dicts (checked for visual tools).

        Returns:
            System context string.
        """
        parts = []
        if self.config.system_prompt:
            parts.append(self.config.system_prompt)

        # Auto-append VVP when visual skills are available
        if discovered_skills:
            visual_names = {"visual-inspector", "browser-automation"}
            if any(s.get("name", "") in visual_names for s in discovered_skills):
                parts.append(self.VVP_PROMPT)

        return "\n\n".join(parts)

    # =========================================================================
    # LAZY CONFIG RESOLUTION
    # =========================================================================

    @staticmethod
    def resolve_config(key: str, *env_vars: str, default: str = "") -> str:
        """
        Resolve a config value at call-time via env var fallback chain.

        Resolves at call-time (not init-time), so env changes take effect
        without agent reinitialisation.

        Usage::

            model = BaseAgent.resolve_config("model", "JOTTY_MODEL", "ANTHROPIC_MODEL", default="sonnet")

        Args:
            key: Config key name (for logging).
            *env_vars: Environment variable names to check in order.
            default: Fallback value if no env var is set.

        Returns:
            First non-empty env var value, or default.
        """
        import os

        for var in env_vars:
            value = os.environ.get(var, "").strip()
            if value:
                return value
        return default

    # =========================================================================
    # AGENT COLLABORATION
    # =========================================================================

    def set_collaboration_context(
        self,
        agent_directory: Dict[str, Any],
        agent_slack: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Inject collaboration context from swarm/conductor.

        Enables agent-to-agent communication during execution:
        agents can request help or share knowledge mid-task.

        Args:
            agent_directory: Map of agent_name -> agent metadata/instance.
            agent_slack: Shared message queue for inter-agent requests.
        """
        self._agent_directory = agent_directory
        if agent_slack is not None:
            self._agent_slack = agent_slack

    def request_help(self, target_agent: str, query: str) -> None:
        """Post a help request to the agent slack for another agent."""
        self._agent_slack.append(
            {
                "from": self.config.name,
                "to": target_agent,
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._collaboration_history.append(
            {
                "type": "request",
                "to": target_agent,
                "query": query,
            }
        )

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get help requests addressed to this agent."""
        return [msg for msg in self._agent_slack if msg.get("to") == self.config.name]

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent execution metrics."""
        metrics = self._metrics.copy()
        if metrics["total_executions"] > 0:
            metrics["success_rate"] = metrics["successful_executions"] / metrics["total_executions"]
            metrics["avg_execution_time"] = (
                metrics["total_execution_time"] / metrics["total_executions"]
            )
        else:
            metrics["success_rate"] = 0.0
            metrics["avg_execution_time"] = 0.0
        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_retries": 0,
            "total_execution_time": 0.0,
        }

    # =========================================================================
    # I/O SCHEMA
    # =========================================================================

    def get_io_schema(self) -> Any:
        """Get typed AgentIOSchema for this agent.

        Returns a generic schema with ``task`` input and ``output``/``success``
        outputs.  Subclasses (DomainAgent) override with signature-derived schemas.
        Cached after first call.
        """
        if hasattr(self, "_io_schema") and self._io_schema is not None:  # type: ignore[has-type]
            return self._io_schema  # type: ignore[has-type]

        from Jotty.core.intelligence.reasoning.types.execution_types import (  # type: ignore[import-not-found]
            AgentIOSchema,
            ToolParam,
        )

        self._io_schema = AgentIOSchema(
            agent_name=getattr(self.config, "name", self.__class__.__name__),
            inputs=[ToolParam(name="task", description="Task to execute")],
            outputs=[
                ToolParam(name="output", description="Execution result"),
                ToolParam(
                    name="success", type_hint="bool", description="Whether execution succeeded"
                ),
            ],
        )
        return self._io_schema

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}')"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state."""
        return {
            "name": self.config.name,
            "class": self.__class__.__name__,
            "config": {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_retries": self.config.max_retries,
            },
            "metrics": self.get_metrics(),
            "initialized": self._initialized,
        }


__all__ = [
    "AgentRuntimeConfig",
    "AgentResult",
    "BaseAgent",
]
