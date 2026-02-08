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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION AND RESULT DATACLASSES
# =============================================================================

@dataclass
class AgentConfig:
    """Unified configuration for all agent types."""
    name: str = ""
    model: str = "sonnet"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0
    enable_memory: bool = True
    enable_context: bool = True
    enable_monitoring: bool = True
    enable_skills: bool = True
    system_prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__


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

    Subclasses implement _execute_impl() for their specific logic.

    Usage:
        class MyAgent(BaseAgent):
            async def _execute_impl(self, **kwargs) -> Any:
                return {"result": "done"}

        agent = MyAgent(config=AgentConfig(name="MyAgent"))
        result = await agent.execute(task="do something")
    """

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

        # Metrics
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_retries": 0,
            "total_execution_time": 0.0,
        }

        self._initialized = False

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

    def _load_anthropic_key(self):
        """Load ANTHROPIC_API_KEY from .env.anthropic file if not in environment."""
        import os
        from pathlib import Path

        # Look for .env.anthropic in project root (4 levels up from this file)
        env_file = Path(__file__).parents[4] / '.env.anthropic'
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('ANTHROPIC_API_KEY='):
                            key = line.split('=', 1)[1].strip()
                            os.environ['ANTHROPIC_API_KEY'] = key
                            logger.debug(f"Loaded ANTHROPIC_API_KEY from {env_file}")
                            return
            except Exception as e:
                logger.warning(f"Failed to load API key from {env_file}: {e}")

    def _init_dspy_lm(self):
        """Auto-configure DSPy LM if not already set.

        Priority:
        1. DirectAnthropicLM (if ANTHROPIC_API_KEY set) - fastest, ~0.5s
        2. PersistentClaudeCLI (uses claude CLI) - ~3s
        """
        try:
            import dspy
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                # Load API key from .env.anthropic if not in environment
                import os
                if not os.environ.get('ANTHROPIC_API_KEY'):
                    self._load_anthropic_key()

                # Try direct API first (fastest)
                try:
                    from ...foundation.direct_anthropic_lm import DirectAnthropicLM, is_api_key_available
                    if is_api_key_available():
                        self._lm = DirectAnthropicLM(model=self.config.model)
                        dspy.configure(lm=self._lm)
                        logger.info(f"Auto-configured DSPy LM with DirectAnthropicLM ({self.config.model}) - fastest")
                        return
                except Exception as e:
                    logger.debug(f"DirectAnthropicLM not available: {e}")

                # Fallback to Claude CLI
                try:
                    from ...foundation.persistent_claude_lm import PersistentClaudeCLI
                    self._lm = PersistentClaudeCLI(model=self.config.model)
                    dspy.configure(lm=self._lm)
                    logger.info(f"Auto-configured DSPy LM with PersistentClaudeCLI ({self.config.model})")
                except Exception as e:
                    logger.warning(f"Could not auto-configure DSPy LM: {e}")
            else:
                self._lm = dspy.settings.lm
        except ImportError:
            logger.debug("DSPy not available, skipping LM configuration")

    @property
    def memory(self):
        """Lazy-load HierarchicalMemory."""
        if self._memory is None and self.config.enable_memory:
            try:
                from ...memory.cortex import HierarchicalMemory
                from ...foundation.data_structures import JottyConfig
                self._memory = HierarchicalMemory(
                    config=JottyConfig(),
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
                from ...persistence.shared_context import SharedContext
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
                from ...registry.skills_registry import get_skills_registry
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

            # Exponential backoff before retry
            if attempt < self.config.max_retries - 1:
                retries += 1
                delay = self.config.retry_delay * (2 ** attempt)
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
            from ...foundation.data_structures import MemoryLevel
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

    def discover_skills(self, task: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Discover relevant skills for a task using lightweight keyword matching.

        This is the fast path — no LLM calls. Uses stop-word filtering with
        name match (+3 score) and description match (+1 score).

        Available to ALL agents via BaseAgent inheritance.

        Args:
            task: Task description to match against
            max_results: Maximum number of skills to return

        Returns:
            List of skill dicts with name, description, category, tools,
            relevance_score — sorted by relevance descending
        """
        if self.skills_registry is None:
            logger.debug("Skills registry not available for discovery")
            return []

        task_lower = task.lower()

        stop_words = {
            'the', 'and', 'for', 'with', 'how', 'what', 'are', 'is',
            'to', 'of', 'in', 'on', 'a', 'an',
        }
        task_words = [
            w for w in task_lower.split()
            if len(w) > 2 and w not in stop_words
        ]

        # Simple stemming: also try without trailing 's' for plural matching
        task_words_stemmed = set(task_words)
        for w in task_words:
            if w.endswith('s') and len(w) > 3:
                task_words_stemmed.add(w[:-1])  # "slides" -> "slide"

        skills = []

        for skill_name, skill_def in self.skills_registry.loaded_skills.items():
            skill_name_lower = skill_name.lower()
            desc = getattr(skill_def, 'description', '') or ''
            desc_lower = desc.lower()

            score = 0
            for word in task_words_stemmed:
                if word in skill_name_lower:
                    score += 3
                if word in desc_lower:
                    score += 1

            if score > 0:
                tools = list(skill_def.tools.keys()) if hasattr(skill_def, 'tools') else []
                skills.append({
                    'name': skill_name,
                    'description': desc or skill_name,
                    'category': getattr(skill_def, 'category', 'general'),
                    'tools': tools,
                    'relevance_score': score,
                })

        skills.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Include ALL matched skills, then pad with unmatched skills up to max_results
        # This ensures LLM sees diverse options, not just keyword matches
        matched_skills = skills
        matched_names = {s['name'] for s in matched_skills}

        # Add unmatched skills to give LLM more options
        if len(matched_skills) < max_results:
            for skill_name, skill_def in self.skills_registry.loaded_skills.items():
                if skill_name not in matched_names:
                    desc = getattr(skill_def, 'description', '') or ''
                    tools = list(skill_def.tools.keys()) if hasattr(skill_def, 'tools') else []
                    matched_skills.append({
                        'name': skill_name,
                        'description': desc or skill_name,
                        'category': getattr(skill_def, 'category', 'general'),
                        'tools': tools,
                        'relevance_score': 0,
                    })
                if len(matched_skills) >= max_results:
                    break

        return matched_skills[:max_results]

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
