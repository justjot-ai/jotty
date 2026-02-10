"""
BaseAgent - Abstract Base Class for All Jotty Agents

Provides unified infrastructure with lazy initialization:
- DSPy LM auto-configuration
- Memory integration (HierarchicalMemory)
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
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level lock for DSPy LM initialization.
# Prevents race conditions when multiple agents/swarms concurrently
# try to configure the global dspy.settings.lm.
_dspy_lm_lock = threading.Lock()


# =============================================================================
# CONFIGURATION AND RESULT DATACLASSES
# =============================================================================

@dataclass
class AgentConfig:
    """Unified configuration for all agent types.

    All numeric defaults are 0 / 0.0 sentinels, resolved from
    ``Jotty.core.foundation.config_defaults`` in ``__post_init__``.
    This keeps a single source of truth for LLM settings.
    """
    name: str = ""
    model: str = ""           # "" → DEFAULT_MODEL_ALIAS
    temperature: float = 0.0  # 0.0 → LLM_TEMPERATURE
    max_tokens: int = 0       # 0 → LLM_MAX_OUTPUT_TOKENS
    max_retries: int = 0      # 0 → MAX_RETRIES
    retry_delay: float = 0.0  # 0.0 → RETRY_BACKOFF_SECONDS
    timeout: float = 0.0      # 0.0 → LLM_TIMEOUT_SECONDS
    enable_memory: bool = True
    enable_context: bool = True
    enable_monitoring: bool = True
    enable_skills: bool = True
    system_prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from Jotty.core.foundation.config_defaults import DEFAULTS
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
    - Memory: HierarchicalMemory for knowledge storage/retrieval
    - Context: SharedContext for cross-agent coordination
    - Monitoring: Cost tracking and metrics
    - Skills: SkillsRegistry for tool access
    - Collaboration: Agent-to-agent communication (optional mixin)

    Subclasses implement _execute_impl() for their specific logic.

    Usage:
        class MyAgent(BaseAgent):
            async def _execute_impl(self, **kwargs) -> Any:
                return {"result": "done"}

        agent = MyAgent(config=AgentConfig(name="MyAgent"))
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

    def __init__(self, config: AgentConfig = None):
        """
        Initialize BaseAgent with optional configuration.

        Args:
            config: Agent configuration. Defaults to AgentConfig with class name.
        """
        self.config = config or AgentConfig(name=self.__class__.__name__)
        if not self.config.name:
            self.config.name = self.__class__.__name__

        # Lazy-initialized infrastructure
        self._memory = None
        self._context_manager = None
        self._cost_tracker = None
        self._skills_registry = None
        self._lm = None

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

    def set_jotty_config(self, config) -> 'BaseAgent':
        """Inject a shared JottyConfig so lazy-loaded components (memory, etc.)
        use the same configuration as the rest of the system.

        Args:
            config: JottyConfig instance from SwarmManager or CLI

        Returns:
            self (for chaining)
        """
        self._jotty_config = config
        return self

    # =========================================================================
    # LAZY INITIALIZATION
    # =========================================================================

    def _ensure_initialized(self):
        """Ensure all infrastructure is initialized (lazy loading)."""
        if self._initialized:
            return

        self._init_dspy_lm()
        self._initialized = True
        logger.debug(f"BaseAgent '{self.config.name}' initialized")

    def _load_api_keys(self):
        """Load API keys from .env.anthropic file if not in environment."""
        import os
        from pathlib import Path

        # Look for .env.anthropic in project root (4 levels up from this file)
        env_file = Path(__file__).parents[4] / '.env.anthropic'
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' in line:
                            key_name, key_val = line.split('=', 1)
                            key_name, key_val = key_name.strip(), key_val.strip()
                            if key_val and key_name not in os.environ:
                                os.environ[key_name] = key_val
                                logger.debug(f"Loaded {key_name} from {env_file}")
            except Exception as e:
                logger.warning(f"Failed to load API keys from {env_file}: {e}")

    def _init_dspy_lm(self):
        """Auto-configure DSPy LM if not already set.

        Thread-safe: uses _dspy_lm_lock to prevent races when multiple
        agents/swarms initialize concurrently.

        Priority:
        1. DirectAnthropicLM (if ANTHROPIC_API_KEY set) - fastest, ~0.5s
        2. PersistentClaudeCLI (uses claude CLI) - ~3s
        """
        try:
            import dspy
        except ImportError:
            logger.debug("DSPy not available, skipping LM configuration")
            return

        with _dspy_lm_lock:
            # Re-check inside lock — another thread may have configured it
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm is not None:
                self._lm = dspy.settings.lm
                return

            # Load API keys from .env.anthropic if not in environment
            import os
            if not os.environ.get('ANTHROPIC_API_KEY') or not os.environ.get('OPENROUTER_API_KEY'):
                self._load_api_keys()

            # Try direct API first (fastest)
            try:
                from Jotty.core.foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
                if is_api_key_available():
                    self._lm = DirectAnthropicLM(
                        model=self.config.model,
                        max_tokens=min(int(self.config.max_tokens), 8192),
                    )
                    dspy.configure(lm=self._lm)
                    logger.info(f"Auto-configured DSPy LM with DirectAnthropicLM ({self.config.model}) - fastest")
                    return
            except Exception as e:
                logger.debug(f"DirectAnthropicLM not available: {e}")

            # Fallback to Claude CLI
            try:
                from Jotty.core.foundation.persistent_claude_lm import PersistentClaudeCLI
                self._lm = PersistentClaudeCLI(model=self.config.model)
                dspy.configure(lm=self._lm)
                logger.info(f"Auto-configured DSPy LM with PersistentClaudeCLI ({self.config.model})")
            except Exception as e:
                logger.warning(f"Could not auto-configure DSPy LM: {e}")

    @property
    def memory(self):
        """Lazy-load HierarchicalMemory."""
        if self._memory is None and self.config.enable_memory:
            try:
                from Jotty.core.memory.cortex import HierarchicalMemory
                from Jotty.core.foundation.data_structures import JottyConfig
                # Use the shared _jotty_config if one was injected, otherwise
                # fall back to a default. This prevents creating N independent
                # JottyConfig instances with potentially different defaults.
                jotty_config = getattr(self, '_jotty_config', None) or JottyConfig()
                self._memory = HierarchicalMemory(
                    config=jotty_config,
                    agent_name=self.config.name
                )
                logger.debug(f"Initialized HierarchicalMemory for {self.config.name}")
            except Exception as e:
                logger.warning(f"Could not initialize memory: {e}")
        return self._memory

    @property
    def context(self):
        """Lazy-load SharedContext."""
        if self._context_manager is None and self.config.enable_context:
            try:
                from Jotty.core.persistence.shared_context import SharedContext
                self._context_manager = SharedContext()
                logger.debug(f"Initialized SharedContext for {self.config.name}")
            except Exception as e:
                logger.warning(f"Could not initialize context: {e}")
        return self._context_manager

    @property
    def skills_registry(self):
        """Lazy-load SkillsRegistry."""
        if self._skills_registry is None and self.config.enable_skills:
            try:
                from Jotty.core.registry.skills_registry import get_skills_registry
                self._skills_registry = get_skills_registry()
                if not self._skills_registry.initialized:
                    self._skills_registry.init()
                logger.debug(f"Initialized SkillsRegistry for {self.config.name}")
            except Exception as e:
                logger.warning(f"Could not initialize skills registry: {e}")
        return self._skills_registry

    # =========================================================================
    # EXECUTION WITH RETRY LOGIC
    # =========================================================================

    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute the agent with retry logic and hook support.

        Args:
            **kwargs: Arguments passed to _execute_impl

        Returns:
            AgentResult with success status, output, and metadata
        """
        self._ensure_initialized()
        start_time = time.time()
        retries = 0
        last_error = None

        # Run pre-execution hooks
        try:
            await self._run_pre_hooks(**kwargs)
        except Exception as e:
            logger.warning(f"Pre-hook failed: {e}")

        # Retry loop with exponential backoff
        for attempt in range(self.config.max_retries):
            try:
                # Execute the implementation
                if asyncio.iscoroutinefunction(self._execute_impl):
                    output = await asyncio.wait_for(
                        self._execute_impl(**kwargs),
                        timeout=self.config.timeout
                    )
                else:
                    output = await asyncio.wait_for(
                        asyncio.to_thread(self._execute_impl, **kwargs),
                        timeout=self.config.timeout
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

                return result

            except asyncio.TimeoutError:
                last_error = f"Execution timed out after {self.config.timeout}s"
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries} timed out")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{self.config.max_retries} failed: {e}")

            # Exponential backoff with max cap before retry
            if attempt < self.config.max_retries - 1:
                retries += 1
                delay = min(
                    self.config.retry_delay * (2 ** attempt),
                    self.MAX_RETRY_DELAY,
                )
                logger.info(f"Retrying in {delay:.1f}s (attempt {attempt + 2}/{self.config.max_retries})")
                await asyncio.sleep(delay)

        # All retries exhausted
        execution_time = time.time() - start_time
        self._metrics["total_executions"] += 1
        self._metrics["failed_executions"] += 1
        self._metrics["total_retries"] += retries
        self._metrics["total_execution_time"] += execution_time

        return AgentResult(
            success=False,
            output=None,
            agent_name=self.config.name,
            execution_time=execution_time,
            retries=retries,
            error=last_error,
        )

    @abstractmethod
    async def _execute_impl(self, **kwargs) -> Any:
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

    def add_pre_hook(self, hook: Callable):
        """Add a pre-execution hook."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable):
        """Add a post-execution hook."""
        self._post_hooks.append(hook)

    async def _run_pre_hooks(self, **kwargs):
        """Run all pre-execution hooks."""
        for hook in self._pre_hooks:
            if asyncio.iscoroutinefunction(hook):
                await hook(self, **kwargs)
            else:
                hook(self, **kwargs)

    async def _run_post_hooks(self, result: AgentResult, **kwargs):
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
        context: Dict[str, Any] = None,
        goal: str = ""
    ):
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
            from Jotty.core.foundation.data_structures import MemoryLevel
            level_enum = MemoryLevel[level.upper()] if isinstance(level, str) else level
            self.memory.store(
                content=content,
                level=level_enum,
                context=context or {},
                goal=goal
            )
        except Exception as e:
            logger.warning(f"Failed to store memory: {e}")

    def retrieve_memory(
        self,
        query: str,
        goal: str = "",
        budget_tokens: int = 1000
    ) -> List[Any]:
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
            return self.memory.retrieve(
                query=query,
                goal=goal,
                budget_tokens=budget_tokens
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve memory: {e}")
            return []

    # =========================================================================
    # CONTEXT HELPERS
    # =========================================================================

    def register_context(self, key: str, value: Any):
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
            for key in ['current_task', 'current_goal', 'recent_outputs', 'agent_states']:
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

        return self.skills_registry.list_skills()

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
            visual_names = {'visual-inspector', 'browser-automation'}
            if any(s.get('name', '') in visual_names for s in discovered_skills):
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
        self._agent_slack.append({
            'from': self.config.name,
            'to': target_agent,
            'query': query,
            'timestamp': datetime.now().isoformat(),
        })
        self._collaboration_history.append({
            'type': 'request',
            'to': target_agent,
            'query': query,
        })

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get help requests addressed to this agent."""
        return [
            msg for msg in self._agent_slack
            if msg.get('to') == self.config.name
        ]

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent execution metrics."""
        metrics = self._metrics.copy()
        if metrics["total_executions"] > 0:
            metrics["success_rate"] = (
                metrics["successful_executions"] / metrics["total_executions"]
            )
            metrics["avg_execution_time"] = (
                metrics["total_execution_time"] / metrics["total_executions"]
            )
        else:
            metrics["success_rate"] = 0.0
            metrics["avg_execution_time"] = 0.0
        return metrics

    def reset_metrics(self):
        """Reset all metrics to zero."""
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_retries": 0,
            "total_execution_time": 0.0,
        }

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
    'AgentConfig',
    'AgentResult',
    'BaseAgent',
]
