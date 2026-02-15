"""
BaseSwarmAgent - Shared Base for All Swarm-Internal Agents
==========================================================

Deduplicates the identical BaseXxxAgent classes found across 10+ swarms.
Provides:
- Standard __init__ wiring (memory, context, bus, learned_context)
- _broadcast() for event bus communication

All swarm-internal agents
extend this class for domain-specific behavior.

Author: A-Team
Date: February 2026
"""

import logging
from typing import Any, Dict

from .domain_agent import DomainAgent, DomainAgentConfig

logger = logging.getLogger(__name__)


class BaseSwarmAgent(DomainAgent):
    """Shared base for all swarm-internal agents.

    Provides:
    - Standard __init__ wiring (memory, context, bus, learned_context)
    - _broadcast() for event bus communication
    """

    def __init__(
        self,
        memory: Any = None,
        context: Any = None,
        bus: Any = None,
        learned_context: str = "",
        signature: Any = None,
    ) -> None:
        config = DomainAgentConfig(
            name=self.__class__.__name__,
            enable_memory=memory is not None,
            enable_context=context is not None,
        )
        super().__init__(signature=signature, config=config)

        # Ensure LM is configured before child classes create DSPy modules
        self._ensure_initialized()

        # Domain-specific attributes (override lazy init if provided)
        if memory is not None:
            self._memory = memory
        if context is not None:
            self._context_manager = context
        self.bus = bus
        self.learned_context = learned_context

    def _broadcast(self, event: str, data: Dict[str, Any]) -> Any:
        """Emit a status event via the singleton AgentEventBroadcaster."""
        try:
            from Jotty.core.infrastructure.utils.async_utils import (
                AgentEvent,
                AgentEventBroadcaster,
            )

            broadcaster = AgentEventBroadcaster.get_instance()
            broadcaster.emit(
                AgentEvent(
                    type="status",
                    data={"event": event, **data},
                    agent_id=self.__class__.__name__,
                )
            )
        except Exception:
            pass
