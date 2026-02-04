"""
AgentLifecycleManager - Manages agent wrapping, initialization, and lifecycle.

Extracted from conductor.py to improve maintainability.
Handles agent wrapping decisions, annotation loading, and tool discovery coordination.
"""
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AgentLifecycleManager:
    """
    Centralized agent lifecycle management.

    Responsibilities:
    - Agent wrapping decisions (should_wrap_agent)
    - Annotation loading for validation enrichment
    - Tool discovery coordination
    - Agent initialization and setup
    """

    def __init__(self, config, tool_discovery_manager=None):
        """
        Initialize agent lifecycle manager.

        Args:
            config: JottyConfig
            tool_discovery_manager: ToolDiscoveryManager instance
        """
        self.config = config
        self.tool_discovery_manager = tool_discovery_manager
        self.wrapped_agents = set()
        self.annotations = {}

        logger.info("🎭 AgentLifecycleManager initialized")

    def should_wrap_agent(self, agent_config: Any) -> bool:
        """
        Determine if an agent needs to be wrapped with JOTTY.

        Wrapping is needed if:
        - Agent has validation prompts (architect_prompts or auditor_prompts)
        - Agent has tools (architect_tools or auditor_tools)

        Args:
            agent_config: ActorConfig instance

        Returns:
            True if agent should be wrapped
        """
        has_validation = bool(
            agent_config.architect_prompts or
            agent_config.auditor_prompts
        )
        has_tools = bool(
            agent_config.architect_tools or
            agent_config.auditor_tools
        )
        should_wrap = has_validation or has_tools

        if should_wrap:
            logger.debug(f"Agent '{agent_config.name}' needs wrapping (validation={has_validation}, tools={has_tools})")

        return should_wrap

    def load_annotations(self, path: Optional[str]) -> Dict[str, Any]:
        """
        Load annotations for validation enrichment.

        Args:
            path: Path to annotations JSON file

        Returns:
            Dict of annotations or empty dict if load fails
        """
        if not path:
            return {}

        try:
            path_obj = Path(path)
            if not path_obj.exists():
                logger.warning(f"Annotations file not found: {path}")
                return {}

            with open(path) as f:
                annotations = json.load(f)

            self.annotations = annotations
            logger.info(f"✅ Loaded {len(annotations)} annotations from {path}")
            return annotations

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse annotations JSON: {e}")
            return {}
        except Exception as e:
            logger.warning(f"Failed to load annotations: {e}")
            return {}

    def filter_tools_for_agent(
        self,
        all_tools: List[Any],
        agent_config: Any,
        role: str
    ) -> List[Any]:
        """
        Filter tools for a specific agent role (architect/auditor).

        Args:
            all_tools: All available tools
            agent_config: ActorConfig instance
            role: "architect" or "auditor"

        Returns:
            Filtered list of tools
        """
        if role == "architect":
            tools = agent_config.architect_tools
            if tools is None or len(tools) == 0:
                if self.tool_discovery_manager:
                    tools = self.tool_discovery_manager.filter_tools_for_planner(all_tools)
                else:
                    tools = []
            logger.debug(f"Filtered {len(tools)} architect tools for {agent_config.name}")
            return tools

        elif role == "auditor":
            tools = agent_config.auditor_tools
            if tools is None or len(tools) == 0:
                if self.tool_discovery_manager:
                    tools = self.tool_discovery_manager.filter_tools_for_reviewer(all_tools)
                else:
                    tools = []
            logger.debug(f"Filtered {len(tools)} auditor tools for {agent_config.name}")
            return tools

        else:
            logger.warning(f"Unknown role: {role}")
            return []

    def mark_agent_wrapped(self, agent_name: str):
        """
        Mark an agent as wrapped.

        Args:
            agent_name: Name of the agent
        """
        self.wrapped_agents.add(agent_name)
        logger.debug(f"Marked {agent_name} as wrapped")

    def is_agent_wrapped(self, agent_name: str) -> bool:
        """
        Check if an agent has been wrapped.

        Args:
            agent_name: Name of the agent

        Returns:
            True if agent is wrapped
        """
        return agent_name in self.wrapped_agents

    def get_annotations_for_agent(self, agent_name: str) -> Dict[str, Any]:
        """
        Get annotations for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dict of annotations for this agent
        """
        return self.annotations.get(agent_name, {})

    def wrap_agent_with_jotty(
        self,
        agent_config: Any,
        conductor_ref: Any
    ) -> Any:
        """
        Wrap an agent with Jotty wrapper for validation and tool support.

        NOTE: Delegates to conductor for now to use its context.
        Future enhancement: Make fully self-contained.

        Args:
            agent_config: ActorConfig instance
            conductor_ref: Reference to conductor (for accessing shared context, etc.)

        Returns:
            JottyCore wrapped agent
        """
        if not conductor_ref:
            logger.error("Cannot wrap agent without conductor reference")
            return agent_config.agent

        # Delegate to conductor's implementation for now
        if hasattr(conductor_ref, '_wrap_actor_with_jotty'):
            wrapped = conductor_ref._wrap_actor_with_jotty(agent_config)
            self.mark_agent_wrapped(agent_config.name)
            return wrapped

        logger.warning(f"Conductor does not have _wrap_actor_with_jotty method")
        return agent_config.agent

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent lifecycle statistics.

        Returns:
            Dict with lifecycle metrics
        """
        return {
            "total_wrapped_agents": len(self.wrapped_agents),
            "wrapped_agents": list(self.wrapped_agents),
            "annotations_loaded": len(self.annotations),
            "has_tool_discovery": self.tool_discovery_manager is not None
        }

    def reset_stats(self):
        """Reset lifecycle statistics."""
        self.wrapped_agents.clear()
        logger.debug("AgentLifecycleManager stats reset")


# ============================================================================
# BACKWARD COMPATIBILITY - DEPRECATED
# ============================================================================

import warnings


class ActorLifecycleManager(AgentLifecycleManager):
    """
    DEPRECATED: Use AgentLifecycleManager instead.

    This class is maintained for backward compatibility only.
    Will be removed in a future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ActorLifecycleManager is deprecated. Use AgentLifecycleManager instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

    # Provide deprecated method aliases
    def should_wrap_actor(self, actor_config):
        """DEPRECATED: Use should_wrap_agent() instead."""
        warnings.warn(
            "should_wrap_actor() is deprecated. Use should_wrap_agent() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.should_wrap_agent(actor_config)

    def filter_tools_for_actor(self, all_tools, actor_config, role):
        """DEPRECATED: Use filter_tools_for_agent() instead."""
        warnings.warn(
            "filter_tools_for_actor() is deprecated. Use filter_tools_for_agent() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.filter_tools_for_agent(all_tools, actor_config, role)

    def mark_actor_wrapped(self, actor_name):
        """DEPRECATED: Use mark_agent_wrapped() instead."""
        warnings.warn(
            "mark_actor_wrapped() is deprecated. Use mark_agent_wrapped() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.mark_agent_wrapped(actor_name)

    def is_actor_wrapped(self, actor_name):
        """DEPRECATED: Use is_agent_wrapped() instead."""
        warnings.warn(
            "is_actor_wrapped() is deprecated. Use is_agent_wrapped() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.is_agent_wrapped(actor_name)

    def get_annotations_for_actor(self, actor_name):
        """DEPRECATED: Use get_annotations_for_agent() instead."""
        warnings.warn(
            "get_annotations_for_actor() is deprecated. Use get_annotations_for_agent() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_annotations_for_agent(actor_name)

    def wrap_actor_with_jotty(self, actor_config, conductor_ref):
        """DEPRECATED: Use wrap_agent_with_jotty() instead."""
        warnings.warn(
            "wrap_actor_with_jotty() is deprecated. Use wrap_agent_with_jotty() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.wrap_agent_with_jotty(actor_config, conductor_ref)
