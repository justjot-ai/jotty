"""
Protocol Interfaces for Jotty

Provides type-safe interface contracts for:
- Skills
- Agents
- Memory systems
- LLM providers
- Swarms

Using Python's Protocol for structural subtyping.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol

from typing_extensions import runtime_checkable


@runtime_checkable
class SkillProtocol(Protocol):
    """
    Protocol for Jotty skills.

    Any class implementing this protocol can be used as a skill.
    """

    name: str
    description: str
    version: str

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the skill.

        Args:
            params: Skill parameters

        Returns:
            Result dict with 'success' key
        """
        ...

    def get_tools(self) -> Dict[str, Callable]:
        """
        Get tool functions for this skill.

        Returns:
            Dict mapping tool names to callables
        """
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol for Jotty agents.

    Any class implementing this protocol can be used as an agent.
    """

    name: str
    signature: Any  # DSPy signature

    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute agent task.

        Args:
            task: Task description
            context: Optional context

        Returns:
            Execution result
        """
        ...


@runtime_checkable
class MemorySystemProtocol(Protocol):
    """
    Protocol for memory systems.

    Ensures consistent interface across different memory backends.
    """

    def store(self, content: str, level: str, goal: str, metadata: Optional[Dict] = None) -> str:
        """
        Store memory.

        Args:
            content: Memory content
            level: Memory level (episodic, semantic, etc.)
            goal: Goal/context
            metadata: Optional metadata

        Returns:
            Memory ID
        """
        ...

    def retrieve(self, query: str, goal: str, top_k: int = 5) -> List[Any]:
        """
        Retrieve memories.

        Args:
            query: Search query
            goal: Goal/context
            top_k: Number of results

        Returns:
            List of memory results
        """
        ...

    def status(self) -> Dict[str, Any]:
        """
        Get memory system status.

        Returns:
            Status dict
        """
        ...


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """
    Protocol for LLM providers.

    Ensures consistent interface across OpenAI, Anthropic, Groq, etc.
    """

    def __call__(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call LLM with prompt.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific args

        Returns:
            LLM response
        """
        ...


@runtime_checkable
class SwarmProtocol(Protocol):
    """
    Protocol for swarms.

    Ensures consistent interface across different swarm types.
    """

    name: str
    domain: str

    def execute(self, goal: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute swarm workflow.

        Args:
            goal: Swarm goal
            context: Optional context

        Returns:
            Execution result
        """
        ...


@runtime_checkable
class ToolProtocol(Protocol):
    """
    Protocol for tools.

    Ensures consistent tool interface.
    """

    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool.

        Args:
            params: Tool parameters

        Returns:
            Result dict with 'success' and optional 'error'
        """
        ...


@runtime_checkable
class ObservabilityProtocol(Protocol):
    """
    Protocol for observability systems.

    Ensures consistent metrics/tracing interface.
    """

    def record_metric(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """Record metric value."""
        ...

    def start_span(self, name: str, attributes: Optional[Dict] = None) -> Any:
        """Start trace span."""
        ...


@runtime_checkable
class MetadataProvider(Protocol):
    """Protocol for objects that provide actor/swarm metadata context."""

    def get_context_for_actor(
        self, actor_name: str, query: str, previous_outputs: Any = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Return context for a specific actor."""
        ...

    def get_swarm_context(self, **kwargs: Any) -> Dict[str, Any]:
        """Return swarm-level context."""
        ...


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for objects that provide data retrieval and storage."""

    def retrieve(self, key: str, format: Optional[str] = None, **kwargs: Any) -> Any:
        """Retrieve data by key."""
        ...

    def store(self, key: str, value: Any, format: Optional[str] = None, **kwargs: Any) -> None:
        """Store data by key."""
        ...


@runtime_checkable
class ContextExtractor(Protocol):
    """Protocol for objects that extract context from content."""

    def extract(self, content: Any, query: str, max_tokens: int, **kwargs: Any) -> Any:
        """Extract relevant context from content."""
        ...


# Type aliases for common types
SkillParams = Dict[str, Any]
SkillResult = Dict[str, Any]
AgentContext = Dict[str, Any]
AgentResult = Dict[str, Any]
MemoryMetadata = Dict[str, Any]
LLMResponse = Any


def validate_skill(obj: Any) -> bool:
    """
    Validate if object implements SkillProtocol.

    Args:
        obj: Object to validate

    Returns:
        True if valid skill
    """
    return isinstance(obj, SkillProtocol)


def validate_agent(obj: Any) -> bool:
    """Validate if object implements AgentProtocol."""
    return isinstance(obj, AgentProtocol)


def validate_memory_system(obj: Any) -> bool:
    """Validate if object implements MemorySystemProtocol."""
    return isinstance(obj, MemorySystemProtocol)


def validate_llm_provider(obj: Any) -> bool:
    """Validate if object implements LLMProviderProtocol."""
    return isinstance(obj, LLMProviderProtocol)


def validate_swarm(obj: Any) -> bool:
    """Validate if object implements SwarmProtocol."""
    return isinstance(obj, SwarmProtocol)
