"""
Autonomous Agent System - Unified Planning

Zero-configuration autonomous agent that handles complex workflows:
- Intent understanding (natural language → task graph)
- Agentic planning (fully LLM-based, no hardcoded logic)
- Autonomous execution (enhances AutoAgent)
- Workflow memory (enhances HierarchicalMemory)

Uses AgenticPlanner for all planning (single source of truth).
"""

from .intent_parser import IntentParser, TaskGraph
from ..agents.agentic_planner import AgenticPlanner, ExecutionPlan
from .enhanced_executor import AutonomousExecutor, EnhancedExecutionResult

__all__ = [
    'IntentParser',
    'TaskGraph',
    'AgenticPlanner',  # Unified planner (replaces AutonomousPlanner)
    'ExecutionPlan',
    'AutonomousExecutor',
    'EnhancedExecutionResult',
]
