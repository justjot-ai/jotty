"""
JOTTY LAYERS
============

Layer-based facade for clean, organized imports.

Layers (top to bottom):
1. INTERFACE - External entry points (CLI, Gateway, SDK)
2. MODES - Execution modes (Chat, API, Workflow)
3. BRAIN - Coordination (Swarms, Agents, Intelligence)
4. REGISTRY - Capabilities (Skills/Hands, UI/Eyes)
5. MEMORY - Learning (SwarmMemory, TD-Lambda)
6. FOUNDATION - Cross-cutting (Exceptions, Config)

Usage:
    # Layer-based imports (new, clean)
    from Jotty.layers.brain import BaseSwarm, AutoAgent
    from Jotty.layers.registry import get_unified_registry
    from Jotty.layers.foundation import JottyError

    # Old imports still work
    from Jotty.core.swarms import BaseSwarm
    from Jotty.core.agents import AutoAgent
"""

# Re-export all layers
from . import interface
from . import modes
from . import brain
from . import registry
from . import memory
from . import foundation

__all__ = [
    "interface",
    "modes",
    "brain",
    "registry",
    "memory",
    "foundation",
]
