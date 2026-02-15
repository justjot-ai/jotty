"""
MemoryCapability - Mixin for Enhanced Memory Integration

Adds memory capabilities to any agent, swarm, or workflow:
- SwarmMemory integration
- Automatic memory storage and retrieval
- Context-aware memory queries
- Memory-enhanced execution

Usage:
    from Jotty.core.execution.base import BaseAgent
    from Jotty.core.execution.capabilities import MemoryCapability

    class ResearchAgent(BaseAgent, MemoryCapability):
        def __init__(self, config=None):
            BaseAgent.__init__(self, config)
            MemoryCapability.__init__(
                self,
                domain="research",
                auto_store=True
            )

        async def _execute_impl(self, query: str, **kwargs):
            # Retrieve relevant past research
            past_research = await self.retrieve_memories(query)

            # Use past research in current task
            result = await self._research_with_context(query, past_research)

            # Store result for future use
            await self.store_memory(result, query=query)

            return result
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryCapability:
    """
    Mixin that adds enhanced memory capabilities to any execution pattern.

    Provides:
    - Automatic memory storage after execution
    - Context-aware memory retrieval
    - Memory-enhanced prompts
    - Memory statistics and management

    Integrates with SwarmMemory for persistent storage across executions.
    """

    def __init__(
        self,
        domain: str,
        auto_store: bool = True,
        memory_levels: Optional[List[str]] = None,
        max_retrieval: int = 5,
    ) -> None:
        """
        Initialize memory capability.

        Args:
            domain: Domain name for memory categorization
            auto_store: Automatically store execution results
            memory_levels: Memory levels to use (default: episodic, procedural)
            max_retrieval: Maximum memories to retrieve per query
        """
        self.memory_domain = domain
        self.auto_store = auto_store
        self.memory_levels = memory_levels or ["episodic", "procedural"]
        self.max_retrieval = max_retrieval

        # Memory system (lazy-loaded)
        self._memory_system = None

        # Memory statistics
        self._memory_stats = {
            "memories_stored": 0,
            "memories_retrieved": 0,
            "last_storage": None,
            "last_retrieval": None,
        }

        logger.info(
            f"MemoryCapability initialized for domain '{domain}' "
            f"(auto_store={auto_store}, levels={memory_levels})"
        )

    @property
    def memory(self):
        """Lazy-load memory system."""
        if self._memory_system is None:
            try:
                from Jotty.core.intelligence.memory.facade import get_memory_system

                self._memory_system = get_memory_system()
                logger.debug(f"Memory system loaded for {self.memory_domain}")
            except Exception as e:
                logger.error(f"Failed to load memory system: {e}")
                self._memory_system = None

        return self._memory_system

    async def store_memory(
        self,
        content: Any,
        level: str = "episodic",
        goal: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        Store content in memory.

        Args:
            content: Content to store
            level: Memory level (episodic, semantic, procedural, meta, causal)
            goal: Associated goal/task
            metadata: Additional metadata
            **kwargs: Additional storage parameters

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.memory:
            logger.warning(f"Memory system not available for {self.memory_domain}")
            return None

        try:
            # Ensure content is string
            if not isinstance(content, str):
                content = str(content)

            # Add domain to metadata
            metadata = metadata or {}
            metadata["domain"] = self.memory_domain
            metadata["timestamp"] = datetime.now().isoformat()

            # Store in memory
            memory_id = self.memory.store(
                content=content,
                level=level,
                goal=goal or self.memory_domain,
                metadata=metadata,
                **kwargs,
            )

            # Update statistics
            self._memory_stats["memories_stored"] += 1
            self._memory_stats["last_storage"] = datetime.now().isoformat()

            logger.debug(f"Stored memory in {level} level for {self.memory_domain}: {memory_id}")

            return memory_id

        except Exception as e:
            logger.error(f"Failed to store memory for {self.memory_domain}: {e}")
            return None

    async def retrieve_memories(
        self,
        query: str,
        goal: Optional[str] = None,
        levels: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Any]:
        """
        Retrieve relevant memories.

        Args:
            query: Query for memory retrieval
            goal: Associated goal/task
            levels: Memory levels to search (default: self.memory_levels)
            top_k: Maximum memories to retrieve (default: self.max_retrieval)
            **kwargs: Additional retrieval parameters

        Returns:
            List of relevant memories
        """
        if not self.memory:
            logger.warning(f"Memory system not available for {self.memory_domain}")
            return []

        try:
            levels = levels or self.memory_levels
            top_k = top_k or self.max_retrieval

            # Retrieve memories
            memories = self.memory.retrieve(
                query=query, goal=goal or self.memory_domain, top_k=top_k, **kwargs
            )

            # Filter by domain if metadata available
            domain_memories = [
                m
                for m in memories
                if not hasattr(m, "metadata") or m.metadata.get("domain") == self.memory_domain
            ]

            # Update statistics
            self._memory_stats["memories_retrieved"] += len(domain_memories)
            self._memory_stats["last_retrieval"] = datetime.now().isoformat()

            logger.debug(f"Retrieved {len(domain_memories)} memories for {self.memory_domain}")

            return domain_memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories for {self.memory_domain}: {e}")
            return []

    def build_memory_context(
        self,
        query: str,
        memories: Optional[List[Any]] = None,
        max_tokens: int = 2000,
    ) -> str:
        """
        Build memory context string for prompts.

        Args:
            query: Current query
            memories: Memories to include (will retrieve if None)
            max_tokens: Maximum tokens for context

        Returns:
            Formatted memory context string
        """
        if memories is None:
            # This is async but we can't await in non-async method
            # Caller should retrieve memories first and pass them in
            return ""

        if not memories:
            return ""

        context_lines = [
            "# Relevant Past Experience",
            "",
        ]

        # Add memories with truncation if needed
        for i, memory in enumerate(memories[:10], 1):  # Max 10 memories
            content = getattr(memory, "content", str(memory))
            relevance = getattr(memory, "relevance", 0.0)
            level = getattr(memory, "level", "unknown")

            # Truncate long content
            if len(content) > 200:
                content = content[:197] + "..."

            context_lines.append(f"{i}. [{level.upper()}] (relevance: {relevance:.2f})")
            context_lines.append(f"   {content}")
            context_lines.append("")

        context = "\n".join(context_lines)

        # Simple token estimation (rough)
        estimated_tokens = len(context.split())
        if estimated_tokens > max_tokens:
            # Truncate if too long
            words = context.split()
            context = " ".join(words[:max_tokens]) + "\n... (truncated)"

        return context

    async def memory_enhanced_execution(
        self, execute_fn: callable, query: str, store_result: bool | None = None, **kwargs
    ) -> Any:
        """
        Execute function with memory enhancement.

        1. Retrieves relevant memories
        2. Adds memory context to execution
        3. Executes function
        4. Optionally stores result

        Args:
            execute_fn: Async function to execute
            query: Query for memory retrieval
            store_result: Store result in memory (default: self.auto_store)
            **kwargs: Arguments for execute_fn

        Returns:
            Execution result
        """
        store_result = store_result if store_result is not None else self.auto_store

        # Retrieve relevant memories
        memories = await self.retrieve_memories(query)

        # Add memory context to kwargs
        memory_context = self.build_memory_context(query, memories)
        if memory_context:
            kwargs["memory_context"] = memory_context

        # Execute function
        result = await execute_fn(query=query, **kwargs)

        # Store result if enabled
        if store_result and result:
            await self.store_memory(
                content=result, level="episodic", goal=self.memory_domain, metadata={"query": query}
            )

        return result

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = self._memory_stats.copy()

        # Add memory system stats if available
        if self.memory:
            try:
                memory_status = self.memory.status()
                stats["system_status"] = memory_status
            except Exception as e:
                logger.debug(f"Could not get memory system status: {e}")

        return stats

    def clear_domain_memories(self) -> bool:
        """
        Clear all memories for this domain.

        Warning: This is a destructive operation!

        Returns:
            True if successful, False otherwise
        """
        if not self.memory:
            logger.warning(f"Memory system not available for {self.memory_domain}")
            return False

        try:
            # This would need to be implemented in the memory system
            # For now, just log a warning
            logger.warning(f"Clear memories not fully implemented for domain: {self.memory_domain}")
            return False

        except Exception as e:
            logger.error(f"Failed to clear memories for {self.memory_domain}: {e}")
            return False
