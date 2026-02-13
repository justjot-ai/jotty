"""
BaseSwarmAgent - Shared Base for All Swarm-Internal Agents
==========================================================

Deduplicates the identical BaseXxxAgent classes found across 10+ swarms.
Provides:
- Standard __init__ wiring (memory, context, bus, learned_context)
- _broadcast() for event bus communication

All swarm-internal base agents (BaseTestAgent, BaseReviewAgent, etc.)
are aliased to this class or extend it for domain-specific extras.

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

    def __init__(self, memory=None, context=None, bus=None,
                 learned_context: str = "", signature=None):
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

    def _broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast event to other agents via the event bus."""
        if self.bus:
            try:
                from Jotty.core.agents.axon import Message
                msg = Message(
                    sender=self.__class__.__name__,
                    receiver="broadcast",
                    content={'event': event, **data}
                )
                self.bus.publish(msg)
            except Exception:
                pass
