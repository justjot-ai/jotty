"""
Autonomous Agent System - Unified Planning

Zero-configuration autonomous agent that handles complex workflows:
- Intent understanding (natural language → task graph)
- Agentic planning (fully LLM-based, no hardcoded logic)
- Autonomous execution (enhances AutoAgent)
- Workflow memory (enhances SwarmMemory)

Uses TaskPlanner for all planning (single source of truth).
"""

from .intent_parser import IntentParser, TaskGraph
from ..agent.agentic_planner import TaskPlanner, TaskPlan
from .enhanced_executor import AutonomousExecutor, EnhancedExecutionResult

__all__ = [
    'IntentParser',
    'TaskGraph',
    'TaskPlanner',  # Unified planner (replaces AutonomousPlanner)
    'TaskPlan',
    'AutonomousExecutor',
    'EnhancedExecutionResult',
]
