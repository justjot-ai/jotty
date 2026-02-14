"""
Workflows - Intent-Based Automation
====================================

High-level workflows that understand intent and automatically execute.
"""

from .auto_workflow import (
    AutoWorkflow,
    WorkflowIntent,
    build,
    research,
    develop,
)
from .smart_swarm_registry import (
    SmartSwarmRegistry,
    StageType,
    SwarmConfig,
    get_smart_registry,
)

__all__ = [
    "AutoWorkflow",
    "WorkflowIntent",
    "build",
    "research",
    "develop",
    "SmartSwarmRegistry",
    "StageType",
    "SwarmConfig",
    "get_smart_registry",
]
