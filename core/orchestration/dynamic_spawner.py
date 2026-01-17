"""
Dynamic Agent Spawner for Jotty
================================

Enables agents to spawn new agents dynamically during execution.

Inspired by MegaAgent's agent recruitment pattern, adapted for Jotty's
DSPy-based architecture with proper DRY principles.

Key Features:
- Spawn new agents from DSPy signatures
- LLM-based decision making via tool calls
- Hierarchical supervisor-subordinate relationships
- Integration with existing MultiAgentsOrchestrator

Dr. Chen: "Agents should be able to recruit help when needed"
Dr. Agarwal: "Dynamic spawning enables true scalability"
"""

import logging
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from datetime import datetime

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ..foundation.agent_config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class SpawnedAgent:
    """Metadata for dynamically spawned agent"""
    name: str
    description: str
    spawned_by: str  # Parent agent name
    spawned_at: datetime = field(default_factory=datetime.now)
    signature_class: Optional[Type] = None
    config: Optional[AgentConfig] = None


class DynamicAgentSpawner:
    """
    Simple dynamic agent spawner (Approach 1 - MegaAgent style)

    Enables agents to spawn subordinates during runtime via tool calls.

    Usage:
        spawner = DynamicAgentSpawner()

        # Agent requests spawning via tool call
        new_agent = spawner.spawn_agent(
            name="researcher_3",
            description="Research specific sub-topic",
            signature=ResearcherSignature,
            spawned_by="planner"
        )

        # Add to conductor
        conductor.add_agent_dynamically(new_agent)

    Features:
    - Creates DSPy modules from signatures
    - Tracks parent-child relationships
    - Validates uniqueness
    - Provides spawning history
    """

    def __init__(self, max_spawned_per_agent: int = 5):
        """
        Initialize spawner

        Args:
            max_spawned_per_agent: Max subordinates per agent (prevents runaway spawning)
        """
        self.max_spawned_per_agent = max_spawned_per_agent

        # Track all spawned agents
        self.spawned_agents: Dict[str, SpawnedAgent] = {}

        # Track subordinates by parent
        self.subordinates: Dict[str, List[str]] = {}

        # Track total spawn count
        self.total_spawned = 0

        logger.info(f"🌱 DynamicAgentSpawner initialized (max {max_spawned_per_agent} per agent)")

    def can_spawn(self, parent_name: str) -> bool:
        """
        Check if parent agent can spawn more subordinates

        Args:
            parent_name: Parent agent name

        Returns:
            True if spawning is allowed
        """
        if parent_name not in self.subordinates:
            return True

        current_count = len(self.subordinates[parent_name])
        return current_count < self.max_spawned_per_agent

    def spawn_agent(
        self,
        name: str,
        description: str,
        signature: Type,
        spawned_by: str,
        architect_prompts: Optional[List[str]] = None,
        auditor_prompts: Optional[List[str]] = None,
        **config_kwargs
    ) -> AgentConfig:
        """
        Spawn a new agent dynamically

        Args:
            name: Unique agent name (e.g., "researcher_3")
            description: What this agent does
            signature: DSPy signature class for this agent
            spawned_by: Parent agent name
            architect_prompts: Optional validation prompts
            auditor_prompts: Optional validation prompts
            **config_kwargs: Additional AgentConfig parameters

        Returns:
            AgentConfig ready to be added to conductor

        Raises:
            ValueError: If spawning not allowed or name already exists
        """
        if not DSPY_AVAILABLE:
            raise RuntimeError("DSPy not available - cannot spawn agents")

        # Validate spawning allowed
        if not self.can_spawn(spawned_by):
            raise ValueError(
                f"Agent '{spawned_by}' has reached max subordinates "
                f"({self.max_spawned_per_agent}). Cannot spawn more."
            )

        # Validate unique name
        if name in self.spawned_agents:
            raise ValueError(f"Agent '{name}' already exists. Choose a different name.")

        # Create DSPy module from signature
        logger.info(f"🌱 Spawning agent '{name}' (parent: {spawned_by})")
        logger.info(f"   Description: {description}")
        logger.info(f"   Signature: {signature.__name__}")

        module = dspy.ChainOfThought(signature)

        # Create agent config
        config = AgentConfig(
            name=name,
            agent=module,
            architect_prompts=architect_prompts or [],
            auditor_prompts=auditor_prompts or [],
            metadata={
                "spawned_dynamically": True,
                "description": description,
                "spawned_by": spawned_by,
                "spawned_at": datetime.now().isoformat(),
                "signature": signature.__name__
            },
            **config_kwargs
        )

        # Record spawning
        spawned_agent = SpawnedAgent(
            name=name,
            description=description,
            spawned_by=spawned_by,
            signature_class=signature,
            config=config
        )

        self.spawned_agents[name] = spawned_agent

        # Track parent-child relationship
        if spawned_by not in self.subordinates:
            self.subordinates[spawned_by] = []
        self.subordinates[spawned_by].append(name)

        self.total_spawned += 1

        logger.info(f"✅ Agent '{name}' spawned successfully")
        logger.info(f"   Total spawned: {self.total_spawned}")
        logger.info(f"   Parent '{spawned_by}' now has {len(self.subordinates[spawned_by])} subordinates")

        return config

    def get_subordinates(self, parent_name: str) -> List[str]:
        """
        Get all subordinates spawned by a parent agent

        Args:
            parent_name: Parent agent name

        Returns:
            List of subordinate agent names
        """
        return self.subordinates.get(parent_name, [])

    def get_spawn_tree(self) -> Dict[str, Any]:
        """
        Get hierarchical spawn tree

        Returns:
            Nested dict showing parent-child relationships
        """
        tree = {}

        for parent, children in self.subordinates.items():
            tree[parent] = {
                "subordinates": children,
                "count": len(children)
            }

        return tree

    def get_agent_info(self, name: str) -> Optional[SpawnedAgent]:
        """
        Get metadata for a spawned agent

        Args:
            name: Agent name

        Returns:
            SpawnedAgent metadata or None
        """
        return self.spawned_agents.get(name)

    def reset(self):
        """Reset all spawning state"""
        self.spawned_agents.clear()
        self.subordinates.clear()
        self.total_spawned = 0
        logger.info("🔄 DynamicAgentSpawner reset")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get spawning statistics

        Returns:
            Dict with spawning stats
        """
        return {
            "total_spawned": self.total_spawned,
            "unique_parents": len(self.subordinates),
            "avg_subordinates_per_parent": (
                sum(len(subs) for subs in self.subordinates.values()) / len(self.subordinates)
                if self.subordinates else 0
            ),
            "max_subordinates": max(
                (len(subs) for subs in self.subordinates.values()),
                default=0
            ),
            "spawn_tree": self.get_spawn_tree()
        }

    def __repr__(self) -> str:
        return (
            f"DynamicAgentSpawner(total={self.total_spawned}, "
            f"parents={len(self.subordinates)}, "
            f"max_per_agent={self.max_spawned_per_agent})"
        )


# =============================================================================
# DSPY TOOL SIGNATURE FOR SPAWNING
# =============================================================================

if DSPY_AVAILABLE:
    class SpawnAgentSignature(dspy.Signature):
        """
        DSPy signature for spawn_agent tool

        Allows agents to request spawning new subordinates via tool calls
        """
        agent_name: str = dspy.InputField(
            desc="Unique name for the new agent (e.g., 'researcher_3'). "
            "Must not conflict with existing agents."
        )
        agent_description: str = dspy.InputField(
            desc="Clear description of what this agent will do. "
            "Be specific about its responsibilities."
        )
        signature_name: str = dspy.InputField(
            desc="DSPy signature class name for this agent "
            "(e.g., 'ResearcherSignature', 'ContentWriterSignature'). "
            "Must be a valid signature available in the system."
        )

        spawn_result: str = dspy.OutputField(
            desc="Result of spawn attempt: 'Success: Agent {name} spawned' or 'Error: {reason}'"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_spawn_tool(spawner: DynamicAgentSpawner, signature_registry: Dict[str, Type]):
    """
    Create a spawn_agent tool function for DSPy tool calling

    Args:
        spawner: DynamicAgentSpawner instance
        signature_registry: Dict mapping signature names to signature classes

    Returns:
        Callable tool function compatible with DSPy tools
    """
    def spawn_agent_tool(
        agent_name: str,
        agent_description: str,
        signature_name: str,
        spawned_by: str = "unknown"
    ) -> str:
        """
        Spawn a new agent dynamically

        Args:
            agent_name: Unique name for new agent
            agent_description: What the agent does
            signature_name: DSPy signature class name
            spawned_by: Parent agent name

        Returns:
            Success/error message
        """
        try:
            # Validate signature exists
            if signature_name not in signature_registry:
                available = ", ".join(signature_registry.keys())
                return (
                    f"Error: Unknown signature '{signature_name}'. "
                    f"Available signatures: {available}"
                )

            signature = signature_registry[signature_name]

            # Spawn agent
            config = spawner.spawn_agent(
                name=agent_name,
                description=agent_description,
                signature=signature,
                spawned_by=spawned_by
            )

            return f"Success: Agent '{agent_name}' spawned with signature {signature_name}"

        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Spawn failed: {e}", exc_info=True)
            return f"Error: Spawn failed - {str(e)}"

    # Set metadata for tool introspection
    spawn_agent_tool.__name__ = "spawn_agent"
    spawn_agent_tool.__doc__ = """
    Spawn a new subordinate agent to help with complex tasks.

    Use this when:
    - A task is too complex for one agent
    - You need specialized expertise
    - Parallel execution would speed up work

    Do NOT use this for simple tasks that you can handle yourself.
    """

    return spawn_agent_tool
