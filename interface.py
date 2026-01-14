"""
JOTTY Interface - Clean API for Multi-Agent Orchestration
============================================================

🎯 SINGLE ENTRY POINT: All other imports are internal implementation details.

A-TEAM CONSENSUS (2026-01-07):
- Use Jotty class OR Conductor directly
- AgentConfig comes from core/agent_config.py (uses 'agent' field)
- JottyConfig comes from core/data_structures.py
- NO DUPLICATE CLASS DEFINITIONS

Example Usage:
```python
from Jotty import Jotty, AgentConfig

jotty = Jotty(
    agents=[
        AgentConfig(
            name="SQLGenerator",
            agent=my_dspy_module,  # NOTE: 'agent', not 'actor'
            architect_prompts=["plan.md"],
            auditor_prompts=["validate.md"],
        )
    ]
)

result = await jotty.run("What was P2P count yesterday?")
```
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum, auto

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT REAL CLASSES (NOT DUPLICATES!)
# =============================================================================

# AgentConfig from core (uses 'agent' field - what pipeline uses)
from .core.foundation.agent_config import AgentConfig

# JottyConfig from core (what pipeline uses)
from .core.foundation.data_structures import JottyConfig

# SwarmResult from io_manager (what pipeline uses)
from .core.data.io_manager import SwarmResult


# =============================================================================
# ENUMS (These are unique to interface)
# =============================================================================

class ValidationMode(Enum):
    """When to run validation."""
    NONE = auto()           # No validation
    ARCHITECT = auto()      # Pre-execution planning only
    AUDITOR = auto()        # Post-execution validation only
    BOTH = auto()           # Full validation (default)


class LearningMode(Enum):
    """How the swarm learns."""
    DISABLED = auto()       # No learning
    CONTEXTUAL = auto()     # Update prompts/context (default)
    PERSISTENT = auto()     # Save Q-tables and memories


class CooperationMode(Enum):
    """How agents cooperate."""
    INDEPENDENT = auto()    # No cooperation (IPPO-style)
    SHARED_REWARD = auto()  # System reward (default)
    NASH = auto()           # Nash equilibrium communication


# =============================================================================
# METADATA PROTOCOL
# =============================================================================

class MetadataProtocol:
    """
    Protocol for metadata providers.
    
    Implement this to provide domain-specific metadata to agents.
    Jotty will auto-discover tools and context from this.
    """
    
    def get_all_tools(self) -> Dict[str, Any]:
        """Return all available tools as {name: callable}."""
        raise NotImplementedError
    
    def get_context(self) -> Dict[str, Any]:
        """Return all context data for agents."""
        raise NotImplementedError


# =============================================================================
# JOTTY CLASS (Thin wrapper to Conductor)
# =============================================================================

class Jotty:
    """
    JOTTY - Multi-Agent Orchestration with Game Theory
    
    🎯 This is a thin wrapper around Conductor for cleaner API.
    
    For most use cases, use Conductor directly:
    ```python
    from Jotty import Conductor, AgentConfig
    conductor = Conductor(actors=[AgentConfig(...)])
    result = await conductor.run(goal="...")
    ```
    
    This class provides a simpler interface:
    ```python
    from Jotty import Jotty, AgentConfig
    jotty = Jotty(agents=[AgentConfig(...)])
    result = await jotty.run("...")
    ```
    """
    
    def __init__(
        self,
        agents: Optional[List[AgentConfig]] = None,
        metadata_provider: Optional[MetadataProtocol] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Jotty.
        
        Parameters:
        -----------
        agents : List[AgentConfig]
            List of agent configurations.
            
        metadata_provider : MetadataProtocol, optional
            Provider for domain metadata and tools.
            
        config : Dict[str, Any], optional
            Configuration dict passed to Conductor.
        """
        self.agents = agents or []
        self.metadata_provider = metadata_provider
        self.config = config or {}
        
        # Lazy initialization
        self._conductor = None
        self._initialized = False
        
        logger.info(f"🚀 Jotty initialized with {len(self.agents)} agents")
    
    def add_agent(self, agent: AgentConfig) -> 'Jotty':
        """Add an agent. Returns self for chaining."""
        self.agents.append(agent)
        self._initialized = False
        return self
    
    def set_metadata_provider(self, provider: MetadataProtocol) -> 'Jotty':
        """Set metadata provider. Returns self for chaining."""
        self.metadata_provider = provider
        self._initialized = False
        return self
    
    async def run(self, goal: str, **kwargs) -> SwarmResult:
        """
        Run the swarm to achieve a goal.
        
        Parameters:
        -----------
        goal : str
            Natural language description of what to achieve.
            
        **kwargs : Any
            Additional context passed to agents.
        
        Returns:
        --------
        SwarmResult with final_output, agent_outputs, trajectory, etc.
        """
        # Initialize conductor if needed
        if not self._initialized:
            await self._initialize()
        
        # Run
        return await self._conductor.run(goal=goal, **kwargs)
    
    async def _initialize(self):
        """Initialize internal components."""
        from .core.orchestration.conductor import Conductor
        
        # Create conductor with agents
        self._conductor = Conductor(
            actors=self.agents,
            config=self.config,
            metadata_manager=self.metadata_provider,
        )
        
        self._initialized = True
        logger.info(f"✅ Jotty conductor initialized with {len(self.agents)} agents")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current learning state."""
        if self._conductor:
            return self._conductor.get_state()
        return {}
    
    def save_state(self, path: Optional[str] = None):
        """Save state to disk."""
        if self._conductor:
            self._conductor.save_state(path)
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Load state from disk."""
        if self._conductor:
            return self._conductor.load_state(path)
        return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes (imported from core)
    'Jotty',
    'AgentConfig',      # From core/agent_config.py
    'JottyConfig',    # From core/data_structures.py
    'SwarmResult',      # From core/io_manager.py
    
    # Enums (unique to interface)
    'ValidationMode',
    'LearningMode',
    'CooperationMode',
    
    # Protocol
    'MetadataProtocol',
]
