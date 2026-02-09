"""
Jotty V2 - True Agentic Assistant Framework
============================================

Quick Start:
------------

Zero-Config Mode (Recommended):
    from core.orchestration.v2 import SwarmManager
    from core.foundation.data_structures import JottyConfig
    
    # Natural language → automatic agent creation
    swarm = SwarmManager(agents="Research AI startups and create PDF report")
    result = await swarm.run(goal="Research AI startups")
    
    # Access results
    if result.success:
        print(result.output)

Traditional Mode:
    from core.orchestration.v2 import SwarmManager, AgentRunner
    from core.foundation.agent_config import AgentConfig
    from core.agents.auto_agent import AutoAgent
    
    agent = AutoAgent()
    swarm = SwarmManager(agents=AgentConfig(name="auto", agent=agent))
    result = await swarm.run(goal="Research topic")

Core Components:
----------------
- SwarmManager: Main orchestrator (zero-config support)
- SwarmTaskBoard: Task tracking and coordination
- SwarmPlanner: LLM-based planning
- SwarmMemory: Hierarchical memory system
- AgentRunner: Per-agent execution with validation

Autonomous Components (Zero-Config):
------------------------------------
- SwarmResearcher: Autonomous research (APIs, tools, solutions)
- SwarmInstaller: Auto-install dependencies (pip, npm, skills)
- SwarmConfigurator: Smart configuration management
- SwarmCodeGenerator: Glue code and integration code generation
- SwarmWorkflowLearner: Pattern learning and reuse
- SwarmIntegrator: Scheduling, monitoring, notifications
- SwarmProviderGateway: Unified provider management

Feature Components:
------------------
- SwarmUIRegistry: UI component generation
- SwarmProfiler: Performance profiling
- SwarmToolValidator: Tool validation
- SwarmToolRegistry: Tool management

Discovering Capabilities:
-------------------------
    swarm = SwarmManager(agents="test")
    
    # Check available providers
    providers = swarm.swarm_provider_gateway.list_available_providers()

Naming Convention:
------------------
- Swarm*: Swarm-level components (shared across agents)
- Agent*: Per-agent components (AgentRunner, AgentMemory, AgentLearner)
- All components use consistent, user-friendly names

For more information, see: V2_QUICK_START.md
"""

# Phase 1: Import V1 components and rename
from .swarm_roadmap import MarkovianTODO, SubtaskState, TaskStatus
from Jotty.core.agents.agentic_planner import AgenticPlanner, ExecutionPlan
from Jotty.core.memory.cortex import HierarchicalMemory

# Swarm-level User-Friendly Names (consistent naming)
SwarmTaskBoard = MarkovianTODO  # Swarm-level task tracking
SwarmPlanner = AgenticPlanner  # Swarm-level planning
SwarmMemory = HierarchicalMemory  # Swarm-level memory

# Phase 2: AgentRunner
from .agent_runner import AgentRunner, AgentRunnerConfig

# SwarmManager (unified orchestrator)
from .swarm_manager import SwarmManager

# Feature Components (Swarm-level, consistent naming)
from Jotty.core.registry.agui_component_registry import AGUIComponentRegistry
from Jotty.core.monitoring.profiler import PerformanceProfiler
from Jotty.core.registry.tool_validation import ToolValidator
from Jotty.core.registry.tools_registry import ToolsRegistry

# User-friendly aliases with Swarm prefix for consistency
SwarmUIRegistry = AGUIComponentRegistry  # Swarm-level UI component registry
SwarmProfiler = PerformanceProfiler  # Swarm-level performance profiler
SwarmToolValidator = ToolValidator  # Swarm-level tool validator
SwarmToolRegistry = ToolsRegistry  # Swarm-level tool registry

# Autonomous Components (Zero-Config, logical naming)
from .swarm_researcher import SwarmResearcher
from .swarm_installer import SwarmInstaller
from .swarm_configurator import SwarmConfigurator
from .swarm_code_generator import SwarmCodeGenerator
from .swarm_workflow_learner import SwarmWorkflowLearner
from .swarm_integrator import SwarmIntegrator
# Unified Provider Gateway (DRY: reuse existing provider system)
from .swarm_provider_gateway import SwarmProviderGateway
# State Management (V1 capabilities integrated)
from .swarm_state_manager import SwarmStateManager, AgentStateTracker
# Sandbox Manager (secure execution)
from .sandbox_manager import SandboxManager, TrustLevel, SandboxType, SandboxResult
# Auto Provider Discovery
from .auto_provider_discovery import AutoProviderDiscovery, DiscoveryResult

# Unified Executor (native LLM tool-calling execution)
from .unified_executor import UnifiedExecutor, ExecutionResult, ToolResult, StreamEvent, create_unified_executor
from .tool_generator import UnifiedToolGenerator, ToolDefinition

# SwarmLearner (extracted from v1 conductor for v2 independence)
from .swarm_learner import SwarmLearner, SwarmLearnerSignature

# SwarmLearningPipeline (extracted from SwarmManager)
from .learning_pipeline import SwarmLearningPipeline

# OptimizationPipeline (moved from v1)
from .optimization_pipeline import (
    OptimizationPipeline,
    OptimizationConfig,
    IterationResult,
    create_optimization_pipeline
)

# Core RL Components (essential for learning)
from .credit_assignment import CreditAssignment, ImprovementCredit
from .adaptive_learning import AdaptiveLearning, LearningState
from .policy_explorer import PolicyExplorer, PolicyExplorerSignature
from Jotty.core.learning.learning_coordinator import LearningCoordinator as LearningManager, LearningUpdate


# =========================================================================
# PIPELINE UTILITIES (AgentScope-inspired convenience functions)
# =========================================================================
# These extract common patterns from SwarmManager._execute_multi_agent
# into thin, reusable functions.  KISS: ~20 lines total.

import asyncio
from typing import List as _List
from .agent_runner import AgentRunner as _AgentRunner
from Jotty.core.foundation.data_structures import EpisodeResult as _EpisodeResult


async def sequential_pipeline(
    runners: _List[_AgentRunner],
    goal: str,
    **kwargs,
) -> _EpisodeResult:
    """
    Run agents sequentially, chaining output.

    Each agent receives the previous agent's output as additional context.
    Useful for: research → summarize → format pipelines.

    DRY: Reuses AgentRunner.run() for each step.

    Args:
        runners: List of AgentRunner instances
        goal: Initial goal
        **kwargs: Passed to each runner

    Returns:
        Final EpisodeResult (last agent's output)
    """
    result = None
    for runner in runners:
        enriched = goal
        if result and result.output:
            enriched = f"{goal}\n\nPrevious output:\n{str(result.output)[:2000]}"
        result = await runner.run(goal=enriched, **kwargs)
        if not result.success:
            break  # Stop pipeline on failure
    return result


async def fanout_pipeline(
    runners: _List[_AgentRunner],
    goal: str,
    **kwargs,
) -> _List[_EpisodeResult]:
    """
    Run agents in parallel on the same input.

    Useful for: getting multiple perspectives / ensemble approaches.

    DRY: Reuses AgentRunner.run() and asyncio.gather.

    Args:
        runners: List of AgentRunner instances
        goal: Goal for all agents
        **kwargs: Passed to each runner

    Returns:
        List of EpisodeResult (one per agent, preserving order)
    """
    return await asyncio.gather(
        *(r.run(goal=goal, **kwargs) for r in runners)
    )


# Export components
__all__ = [
    'SwarmTaskBoard',
    'SwarmPlanner',
    'SwarmMemory',
    'SubtaskState',
    'TaskStatus',
    'ExecutionPlan',
    'AgentRunner',
    'AgentRunnerConfig',
    'SwarmManager',
    # Feature components with Swarm prefix
    'SwarmUIRegistry',
    'SwarmProfiler',
    'SwarmToolValidator',
    'SwarmToolRegistry',
    # Autonomous components (zero-config)
    'SwarmResearcher',
    'SwarmInstaller',
    'SwarmConfigurator',
    'SwarmCodeGenerator',
    'SwarmWorkflowLearner',
    'SwarmIntegrator',
    # Unified Provider Gateway
    'SwarmProviderGateway',
    # State Management
    'SwarmStateManager',
    'AgentStateTracker',
    # Sandbox Manager (secure execution)
    'SandboxManager',
    'TrustLevel',
    'SandboxType',
    'SandboxResult',
    # Auto Provider Discovery
    'AutoProviderDiscovery',
    'DiscoveryResult',
    # Unified Executor (primary executor)
    'UnifiedExecutor',
    'ExecutionResult',
    'ToolResult',
    'StreamEvent',
    'create_unified_executor',
    'UnifiedToolGenerator',
    'ToolDefinition',
    # SwarmLearner (online prompt learning)
    'SwarmLearner',
    'SwarmLearnerSignature',
    # SwarmLearningPipeline (extracted learning)
    'SwarmLearningPipeline',
    # OptimizationPipeline
    'OptimizationPipeline',
    'OptimizationConfig',
    'IterationResult',
    'create_optimization_pipeline',
    # Core RL Components
    'CreditAssignment',
    'ImprovementCredit',
    'AdaptiveLearning',
    'LearningState',
    'PolicyExplorer',
    'PolicyExplorerSignature',
    'LearningManager',
    'LearningUpdate',
    # Pipeline utilities (AgentScope-inspired)
    'sequential_pipeline',
    'fanout_pipeline',
]
