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
    
    # List all capabilities
    capabilities = swarm.list_capabilities()
    
    # Get help for a component
    help_text = swarm.get_help("SwarmResearcher")
    
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
from ..roadmap import MarkovianTODO, SubtaskState, TaskStatus
from ...agents.agentic_planner import AgenticPlanner, ExecutionPlan
from ...memory.cortex import HierarchicalMemory

# Swarm-level User-Friendly Names (consistent naming)
SwarmTaskBoard = MarkovianTODO  # Swarm-level task tracking
SwarmPlanner = AgenticPlanner  # Swarm-level planning
SwarmMemory = HierarchicalMemory  # Swarm-level memory

# Phase 2: AgentRunner
from .agent_runner import AgentRunner, AgentRunnerConfig

# SwarmManager (unified orchestrator)
from .swarm_manager import SwarmManager

# Feature Components (Swarm-level, consistent naming)
from ...registry.agui_component_registry import AGUIComponentRegistry
from ...monitoring.profiler import PerformanceProfiler
from ...registry.tool_validation import ToolValidator
from ...registry.tools_registry import ToolsRegistry

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
]
