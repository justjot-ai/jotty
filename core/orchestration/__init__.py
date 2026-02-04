"""
Orchestration Layer - V2 Multi-Agent Coordination
==================================================

V2 is the ONLY implementation. All v1 code archived to _v1_archive/.

Usage:
    from core.orchestration import SwarmManager

    swarm = SwarmManager(agents="Research AI startups")
    result = await swarm.run(goal="Research AI startups")

Core Components:
---------------
- SwarmManager: Main orchestrator (handles N=1 and N>N agents)
- AgentRunner: Per-agent execution with validation
- SwarmTaskBoard: Task tracking (MarkovianTODO)
- SwarmLearner: Online prompt learning
- MASLearning: Cross-session learning persistence

RL Components:
-------------
- CreditAssignment: Multi-agent credit attribution
- AdaptiveLearning: Learning rate adaptation
- PolicyExplorer: Exploration when stuck
- OptimizationPipeline: Iterative optimization
"""

# V2 is the ONLY implementation
from .v2 import (
    # Main orchestrator
    SwarmManager,
    # Agent execution
    AgentRunner,
    AgentRunnerConfig,
    # Task management
    SwarmTaskBoard,
    MarkovianTODO,
    SubtaskState,
    TaskStatus,
    # Planning
    SwarmPlanner,
    ExecutionPlan,
    # Memory
    SwarmMemory,
    # Learning
    SwarmLearner,
    SwarmLearnerSignature,
    # RL Components (core features)
    CreditAssignment,
    ImprovementCredit,
    AdaptiveLearning,
    LearningState,
    PolicyExplorer,
    PolicyExplorerSignature,
    LearningManager,
    LearningUpdate,
    # OptimizationPipeline
    OptimizationPipeline,
    OptimizationConfig,
    IterationResult,
    create_optimization_pipeline,
    # Feature components
    SwarmUIRegistry,
    SwarmProfiler,
    SwarmToolValidator,
    SwarmToolRegistry,
    # Autonomous components
    SwarmResearcher,
    SwarmInstaller,
    SwarmConfigurator,
    SwarmCodeGenerator,
    SwarmWorkflowLearner,
    SwarmIntegrator,
    # Provider gateway
    SwarmProviderGateway,
    # State management
    SwarmStateManager,
    AgentStateTracker,
    # Sandbox
    SandboxManager,
    TrustLevel,
    SandboxType,
    SandboxResult,
    # Auto provider discovery
    AutoProviderDiscovery,
    DiscoveryResult,
    # Unified executor
    UnifiedExecutor,
    ExecutionResult,
    ToolResult,
    StreamEvent,
    create_unified_executor,
    UnifiedToolGenerator,
    ToolDefinition,
    # Legacy executor
    LeanExecutor,
)

__all__ = [
    # Main orchestrator
    'SwarmManager',
    # Agent execution
    'AgentRunner',
    'AgentRunnerConfig',
    # Task management
    'SwarmTaskBoard',
    'MarkovianTODO',
    'SubtaskState',
    'TaskStatus',
    # Planning
    'SwarmPlanner',
    'ExecutionPlan',
    # Memory
    'SwarmMemory',
    # Learning
    'SwarmLearner',
    'SwarmLearnerSignature',
    # RL Components (core features)
    'CreditAssignment',
    'ImprovementCredit',
    'AdaptiveLearning',
    'LearningState',
    'PolicyExplorer',
    'PolicyExplorerSignature',
    'LearningManager',
    'LearningUpdate',
    # OptimizationPipeline
    'OptimizationPipeline',
    'OptimizationConfig',
    'IterationResult',
    'create_optimization_pipeline',
    # Feature components
    'SwarmUIRegistry',
    'SwarmProfiler',
    'SwarmToolValidator',
    'SwarmToolRegistry',
    # Autonomous components
    'SwarmResearcher',
    'SwarmInstaller',
    'SwarmConfigurator',
    'SwarmCodeGenerator',
    'SwarmWorkflowLearner',
    'SwarmIntegrator',
    # Provider gateway
    'SwarmProviderGateway',
    # State management
    'SwarmStateManager',
    'AgentStateTracker',
    # Sandbox
    'SandboxManager',
    'TrustLevel',
    'SandboxType',
    'SandboxResult',
    # Auto provider discovery
    'AutoProviderDiscovery',
    'DiscoveryResult',
    # Unified executor
    'UnifiedExecutor',
    'ExecutionResult',
    'ToolResult',
    'StreamEvent',
    'create_unified_executor',
    'UnifiedToolGenerator',
    'ToolDefinition',
    # Legacy executor
    'LeanExecutor',
]
