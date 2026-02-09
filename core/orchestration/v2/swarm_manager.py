"""
SwarmManager - World-Class Orchestrator
========================================

Lazy-initialized, composable swarm orchestration.

All components are lazy-loaded via descriptors - only created when first
accessed. This means SwarmManager.__init__ is < 50ms regardless of how
many components are registered.

Architecture:
    SwarmManager
    ├── Core (always created)
    │   ├── config, agents, mode
    │   └── runners (AgentRunners for each agent)
    ├── Planning (lazy)
    │   ├── swarm_planner - AgenticPlanner
    │   ├── swarm_task_board - SwarmTaskBoard
    │   └── swarm_intent_parser - IntentParser
    ├── Memory (lazy)
    │   ├── swarm_memory - HierarchicalMemory
    │   └── swarm_state_manager - SwarmStateManager
    ├── Learning (lazy)
    │   ├── learning - SwarmLearningPipeline
    │   └── mas_learning - MASLearning
    ├── Autonomous (lazy, only on autonomous_setup)
    │   ├── swarm_researcher, swarm_installer
    │   ├── swarm_configurator, swarm_code_generator
    │   └── swarm_terminal
    └── Feature (lazy, only when needed)
        ├── swarm_ui_registry, swarm_profiler
        ├── swarm_tool_registry, provider_registry
        └── lotus (optimization layer)

Usage:
    sm = SwarmManager()  # Fast: ~10ms
    result = await sm.run("Research AI trends")  # Components init on demand
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from Jotty.core.foundation.data_structures import JottyConfig, EpisodeResult
from Jotty.core.foundation.agent_config import AgentConfig

from ._lazy import LazyComponent
from ._provider_mixin import ProviderMixin
from ._ensemble_mixin import EnsembleMixin
from ._learning_delegation_mixin import LearningDelegationMixin
from ._mas_zero_mixin import MASZeroMixin

logger = logging.getLogger(__name__)


# Skill Provider System - Lazy loaded
PROVIDERS_AVAILABLE = False
ProviderRegistry = None
SkillCategory = None
BrowserUseProvider = None
OpenHandsProvider = None
AgentSProvider = None
OpenInterpreterProvider = None
ResearchAndAnalyzeProvider = None
AutomateWorkflowProvider = None
FullStackAgentProvider = None

def _load_providers():
    """Lazy load providers to avoid circular imports."""
    global PROVIDERS_AVAILABLE, ProviderRegistry, SkillCategory
    global BrowserUseProvider, OpenHandsProvider, AgentSProvider
    global OpenInterpreterProvider, ResearchAndAnalyzeProvider
    global AutomateWorkflowProvider, FullStackAgentProvider

    if PROVIDERS_AVAILABLE:
        return True

    try:
        from Jotty.core.skills.providers import ProviderRegistry as PR, SkillCategory as SC
        from Jotty.core.skills.providers.browser_use_provider import BrowserUseProvider as BUP
        from Jotty.core.skills.providers.openhands_provider import OpenHandsProvider as OHP
        from Jotty.core.skills.providers.agent_s_provider import AgentSProvider as ASP
        from Jotty.core.skills.providers.open_interpreter_provider import OpenInterpreterProvider as OIP
        from Jotty.core.skills.providers.composite_provider import (
            ResearchAndAnalyzeProvider as RAP,
            AutomateWorkflowProvider as AWP,
            FullStackAgentProvider as FSP,
        )

        ProviderRegistry = PR
        SkillCategory = SC
        BrowserUseProvider = BUP
        OpenHandsProvider = OHP
        AgentSProvider = ASP
        OpenInterpreterProvider = OIP
        ResearchAndAnalyzeProvider = RAP
        AutomateWorkflowProvider = AWP
        FullStackAgentProvider = FSP
        PROVIDERS_AVAILABLE = True
        return True

    except ImportError as e:
        logger.debug(f"Skill providers not available: {e}")
        return False


# =========================================================================
# LAZY FACTORY FUNCTIONS - Called by LazyComponent descriptors
# =========================================================================

def _create_task_board():
    from Jotty.core.orchestration.v2.swarm_roadmap import MarkovianTODO
    return MarkovianTODO()

def _create_planner():
    from Jotty.core.agents.agentic_planner import AgenticPlanner
    return AgenticPlanner()

def _create_intent_parser(planner):
    from Jotty.core.autonomous.intent_parser import IntentParser
    return IntentParser(planner=planner)

def _create_memory(config):
    from Jotty.core.memory.cortex import HierarchicalMemory
    return HierarchicalMemory(config=config, agent_name="SwarmShared")

def _create_provider_gateway(config):
    from Jotty.core.orchestration.v2.swarm_provider_gateway import SwarmProviderGateway
    provider_preference = getattr(config, 'provider', None)
    return SwarmProviderGateway(config=config, provider=provider_preference)

def _create_researcher(config):
    from Jotty.core.orchestration.v2.swarm_researcher import SwarmResearcher
    return SwarmResearcher(config=config)

def _create_installer(config):
    from Jotty.core.orchestration.v2.swarm_installer import SwarmInstaller
    return SwarmInstaller(config=config)

def _create_configurator(config):
    from Jotty.core.orchestration.v2.swarm_configurator import SwarmConfigurator
    return SwarmConfigurator(config=config)

def _create_code_generator(config):
    from Jotty.core.orchestration.v2.swarm_code_generator import SwarmCodeGenerator
    return SwarmCodeGenerator(config=config)

def _create_workflow_learner(memory):
    from Jotty.core.orchestration.v2.swarm_workflow_learner import SwarmWorkflowLearner
    return SwarmWorkflowLearner(swarm_memory=memory)

def _create_integrator(config):
    from Jotty.core.orchestration.v2.swarm_integrator import SwarmIntegrator
    return SwarmIntegrator(config=config)

def _create_terminal(config):
    from Jotty.core.orchestration.v2.swarm_terminal import SwarmTerminal
    return SwarmTerminal(config=config, auto_fix=True, max_fix_attempts=3)

def _create_ui_registry():
    from Jotty.core.registry.agui_component_registry import get_agui_registry
    return get_agui_registry()

def _create_tool_validator():
    from Jotty.core.registry.tool_validation import ToolValidator
    return ToolValidator()

def _create_tool_registry():
    from Jotty.core.registry.tools_registry import get_tools_registry
    return get_tools_registry()

def _create_profiler(config):
    enable = getattr(config, 'enable_profiling', False)
    if not enable:
        return None
    from Jotty.core.monitoring.profiler import PerformanceProfiler
    return PerformanceProfiler(enable_cprofile=True)

def _create_state_manager(sm):
    from Jotty.core.orchestration.v2.swarm_state_manager import SwarmStateManager
    agents_dict = {a.name: a for a in sm.agents}
    return SwarmStateManager(
        swarm_task_board=sm.swarm_task_board,
        swarm_memory=sm.swarm_memory,
        io_manager=sm.io_manager,
        data_registry=sm.data_registry,
        shared_context=sm.shared_context,
        context_guard=sm.context_guard,
        config=sm.config,
        agents=agents_dict,
        agent_signatures={},
    )

def _create_shared_context():
    from Jotty.core.persistence.shared_context import SharedContext
    return SharedContext()

def _create_io_manager():
    from Jotty.core.data.io_manager import IOManager
    return IOManager()

def _create_data_registry():
    from Jotty.core.data.data_registry import DataRegistry
    return DataRegistry()

def _create_context_guard():
    from Jotty.core.context.context_guard import SmartContextGuard
    return SmartContextGuard()

def _create_learning_pipeline(config):
    from Jotty.core.orchestration.v2.learning_pipeline import SwarmLearningPipeline
    return SwarmLearningPipeline(config)

def _create_mas_learning(sm):
    from Jotty.core.orchestration.v2.mas_learning import MASLearning
    workspace_path = getattr(sm.config, 'base_path', None)
    return MASLearning(
        config=sm.config,
        workspace_path=workspace_path,
        swarm_intelligence=sm.swarm_intelligence,
        learning_manager=sm.learning_manager,
        transfer_learning=sm.transfer_learning,
    )


class SwarmManager(ProviderMixin, EnsembleMixin, LearningDelegationMixin, MASZeroMixin):
    """
    World-Class Swarm Orchestrator.

    All heavyweight components are lazy-loaded via descriptors.
    Init is fast (~10ms). Components are created on first access.

    MAS-ZERO enhancements (Ke et al., NeurIPS 2025):
        - Building blocks: multiple strategies run in parallel
        - Meta-feedback: solvability + completeness evaluation
        - Candidate verification: LLM-based best answer selection
        - Dynamic reduction: multi -> single when simpler is better
        - Iterative refinement: MAS-Evolve loop with experience library
        - TOO_HARD escalation: agents signal unsolvable sub-tasks

    AIOS-inspired scheduling (Mei et al., COLM 2025):
        - Concurrency semaphore: limits parallel LLM calls during multi-agent fan-out
        - Prevents API rate-limit errors and controls cost
        - Configurable via max_concurrent_agents (default 3)

    Modes:
        - single: 1 AutoAgent (default)
        - multi: N agents with SwarmTaskBoard coordination

    Integration:
        - ModeRouter.workflow() -> SwarmManager.run() for multi-step planning
        - ModeRouter.chat() -> UnifiedExecutor for direct tool-calling
    """

    # =========================================================================
    # LAZY COMPONENTS - Only created when first accessed
    # =========================================================================

    # Planning
    swarm_task_board = LazyComponent(lambda self: _create_task_board())
    swarm_planner = LazyComponent(lambda self: _create_planner())
    swarm_intent_parser = LazyComponent(lambda self: _create_intent_parser(self.swarm_planner))

    # Memory
    swarm_memory = LazyComponent(lambda self: _create_memory(self.config))

    # Provider Gateway
    swarm_provider_gateway = LazyComponent(lambda self: _create_provider_gateway(self.config))

    # Autonomous (only used in autonomous_setup)
    swarm_researcher = LazyComponent(lambda self: _create_researcher(self.config))
    swarm_installer = LazyComponent(lambda self: _create_installer(self.config))
    swarm_configurator = LazyComponent(lambda self: _create_configurator(self.config))
    swarm_code_generator = LazyComponent(lambda self: _create_code_generator(self.config))
    swarm_workflow_learner = LazyComponent(lambda self: _create_workflow_learner(self.swarm_memory))
    swarm_integrator = LazyComponent(lambda self: _create_integrator(self.config))
    swarm_terminal = LazyComponent(lambda self: _create_terminal(self.config))

    # Feature components
    swarm_ui_registry = LazyComponent(lambda self: _create_ui_registry())
    swarm_tool_validator = LazyComponent(lambda self: _create_tool_validator())
    swarm_tool_registry = LazyComponent(lambda self: _create_tool_registry())
    swarm_profiler = LazyComponent(lambda self: _create_profiler(self.config))

    # State management
    swarm_state_manager = LazyComponent(lambda self: _create_state_manager(self))
    shared_context = LazyComponent(lambda self: _create_shared_context())
    io_manager = LazyComponent(lambda self: _create_io_manager())
    data_registry = LazyComponent(lambda self: _create_data_registry())
    context_guard = LazyComponent(lambda self: _create_context_guard())

    # Learning (single pipeline, components accessed via .learning.xxx)
    learning = LazyComponent(lambda self: _create_learning_pipeline(self.config))
    mas_learning = LazyComponent(lambda self: _create_mas_learning(self))

    # =========================================================================
    # INIT - Fast, minimal, no I/O
    # =========================================================================

    def __init__(
        self,
        agents: Optional[Union[AgentConfig, List[AgentConfig], str]] = None,
        config: Optional[JottyConfig] = None,
        architect_prompts: Optional[List[str]] = None,
        auditor_prompts: Optional[List[str]] = None,
        enable_zero_config: bool = True,
        enable_lotus: bool = True,
        max_concurrent_agents: int = 3,
    ):
        """
        Initialize SwarmManager.

        Fast init (~10ms). All heavyweight components are lazy-loaded.

        Args:
            agents: AgentConfig, list of AgentConfigs, or natural language (zero-config)
            config: JottyConfig (defaults if None)
            architect_prompts: Architect prompt paths
            auditor_prompts: Auditor prompt paths
            enable_zero_config: Enable natural language -> agent conversion
            enable_lotus: Enable LOTUS optimization layer
            max_concurrent_agents: Max agents calling LLM concurrently (AIOS-inspired, default 3)
        """
        self.config = config or JottyConfig()
        self.enable_zero_config = enable_zero_config
        self.enable_lotus = enable_lotus
        self.episode_count = 0

        # AIOS-inspired: Concurrency control for multi-agent LLM fan-out.
        # Prevents API rate-limit errors when N agents fire in parallel.
        # DRY: Single semaphore, no wrapper classes needed.
        self.max_concurrent_agents = max_concurrent_agents
        self._agent_semaphore = asyncio.Semaphore(max_concurrent_agents)
        self._scheduling_stats: Dict[str, int] = {
            'total_scheduled': 0,
            'total_waited': 0,    # times an agent had to wait for a slot
            'peak_concurrent': 0,
            'current_concurrent': 0,
        }

        # Prompts
        self.architect_prompts = architect_prompts or ["configs/prompts/architect/base_architect.md"]
        self.auditor_prompts = auditor_prompts or ["configs/prompts/auditor/base_auditor.md"]

        # Zero-config: natural language -> agents
        if isinstance(agents, str) and enable_zero_config:
            logger.info("Zero-config mode: analyzing task for agent configuration")
            agents = self._create_zero_config_agents(agents)

        # Normalize agents
        if agents is None:
            from Jotty.core.agents.auto_agent import AutoAgent
            agents = [AgentConfig(name="auto", agent=AutoAgent())]
        elif isinstance(agents, AgentConfig):
            agents = [agents]

        self.agents = agents
        self.mode = "multi" if len(agents) > 1 else "single"

        # Runner and LOTUS state (created lazily in _ensure_runners)
        self.runners: Dict[str, 'AgentRunner'] = {}
        self._runners_built = False
        self.lotus = None
        self.lotus_optimizer = None

        # Provider registry (lazy)
        self.provider_registry = None

        # Setup cache
        self._setup_cache = {}

        # Optima-inspired efficiency tracking (Chen et al., 2024):
        # Track orchestration overhead vs. actual execution per run.
        # KISS: Just a dict, no new classes. Reset each run().
        self._efficiency_stats: Dict[str, float] = {}

        logger.info(f"SwarmManager: {self.mode} mode, {len(self.agents)} agents (lazy init)")

    # =========================================================================
    # LAZY RUNNER CREATION
    # =========================================================================

    def _ensure_runners(self):
        """Build AgentRunners on first run() call (not in __init__)."""
        if self._runners_built:
            return

        from Jotty.core.orchestration.v2.agent_runner import AgentRunner, AgentRunnerConfig

        for agent_config in self.agents:
            if agent_config.name in self.runners:
                continue

            runner_config = AgentRunnerConfig(
                architect_prompts=self.architect_prompts,
                auditor_prompts=self.auditor_prompts,
                config=self.config,
                agent_name=agent_config.name,
                enable_learning=True,
                enable_memory=True,
            )

            runner = AgentRunner(
                agent=agent_config.agent,
                config=runner_config,
                task_planner=self.swarm_planner,
                task_board=self.swarm_task_board,
                swarm_memory=self.swarm_memory,
                swarm_state_manager=self.swarm_state_manager,
                learning_manager=self.learning_manager,
                transfer_learning=self.transfer_learning,
                swarm_terminal=self.swarm_terminal,
                swarm_intelligence=self.swarm_intelligence,
            )
            self.runners[agent_config.name] = runner

        # Register agents with Axon for inter-agent communication
        self._register_agents_with_axon()

        # LOTUS optimization
        if self.enable_lotus:
            self._init_lotus_optimization()

        # Auto-load previous learnings + integrate MAS terminal in background
        # This saves ~3s by not blocking runner construction
        import threading
        def _bg_init():
            try:
                self.learning.auto_load()
                self.mas_learning.integrate_with_terminal(self.swarm_terminal)
            except Exception as e:
                logger.debug(f"Background learning init: {e}")
        threading.Thread(target=_bg_init, daemon=True).start()

        # Init providers if available
        if _load_providers():
            self._init_provider_registry()

        self._runners_built = True
        logger.info(f"Runners built: {list(self.runners.keys())}")

    # =========================================================================
    # LEARNING PIPELINE ACCESSORS (backward compat - delegate to learning)
    # =========================================================================

    @property
    def learning_manager(self):
        return self.learning.learning_manager

    @property
    def transfer_learning(self):
        return self.learning.transfer_learning

    @property
    def swarm_intelligence(self):
        return self.learning.swarm_intelligence

    @property
    def credit_weights(self):
        return self.learning.credit_weights

    @credit_weights.setter
    def credit_weights(self, value):
        self.learning.credit_weights = value

    @property
    def trajectory_predictor(self):
        return self.learning.trajectory_predictor

    @property
    def divergence_memory(self):
        return self.learning.divergence_memory

    @property
    def cooperative_credit(self):
        return self.learning.cooperative_credit

    @property
    def brain_state(self):
        return self.learning.brain_state

    @property
    def agent_abstractor(self):
        return self.learning.agent_abstractor

    @property
    def swarm_learner(self):
        return self.learning.swarm_learner

    @property
    def agent_slack(self):
        return self.learning.agent_slack

    @property
    def feedback_channel(self):
        return self.learning.feedback_channel
    
    # =========================================================================
    # LIFECYCLE MANAGEMENT
    # =========================================================================

    async def startup(self):
        """
        Async startup - prepare SwarmManager for execution.

        Call this before run() for controlled initialization.
        If not called, run() will auto-initialize (lazy).

        Returns:
            Self for chaining: `sm = await SwarmManager().startup()`
        """
        self._ensure_runners()
        logger.info("SwarmManager startup complete")
        return self

    async def shutdown(self):
        """
        Graceful shutdown - persist learnings and release resources.

        Should be called when SwarmManager is no longer needed.
        Safe to call multiple times.
        """
        try:
            # Persist all learnings
            if '_lazy_learning' in self.__dict__:
                self._auto_save_learnings()
                logger.info("Learnings saved on shutdown")

            # Clear runners
            self.runners.clear()
            self._runners_built = False

            logger.info("SwarmManager shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    async def __aenter__(self):
        """Async context manager: `async with SwarmManager() as sm:`"""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        await self.shutdown()
        return False

    # =========================================================================
    # INTROSPECTION & METRICS
    # =========================================================================

    def status(self) -> Dict[str, Any]:
        """
        Get full introspection of SwarmManager state.

        Shows which lazy components have been created, runner status,
        learning stats, and execution metrics.

        Returns:
            Dict with component status, metrics, and health info.
        """
        lazy_names = [
            'swarm_planner', 'swarm_task_board', 'swarm_memory',
            'swarm_intent_parser', 'swarm_provider_gateway',
            'swarm_researcher', 'swarm_installer', 'swarm_configurator',
            'swarm_code_generator', 'swarm_workflow_learner',
            'swarm_integrator', 'swarm_terminal',
            'swarm_ui_registry', 'swarm_tool_validator', 'swarm_tool_registry',
            'swarm_profiler', 'swarm_state_manager', 'shared_context',
            'io_manager', 'data_registry', 'context_guard',
            'learning', 'mas_learning',
        ]
        created = {n: f'_lazy_{n}' in self.__dict__ for n in lazy_names}
        created_count = sum(created.values())

        result = {
            'mode': self.mode,
            'agents': [a.name for a in self.agents],
            'runners_built': self._runners_built,
            'runners': list(self.runners.keys()),
            'episode_count': self.episode_count,
            'lotus_enabled': self.enable_lotus,
            'lotus_active': self.lotus is not None,
            'zero_config': self.enable_zero_config,
            'components': {
                'total': len(lazy_names),
                'created': created_count,
                'pending': len(lazy_names) - created_count,
                'detail': created,
            },
        }

        # Scheduling stats (AIOS-inspired)
        result['scheduling'] = {
            'max_concurrent_agents': self.max_concurrent_agents,
            'semaphore_available': self._agent_semaphore._value,
            **self._scheduling_stats,
        }

        # Add learning stats if pipeline is active
        if '_lazy_learning' in self.__dict__:
            try:
                result['learning'] = {
                    'episode_count': self.learning.episode_count,
                    'has_intelligence': self.swarm_intelligence is not None,
                }
            except Exception:
                result['learning'] = {'status': 'error'}

        # Add LOTUS stats if active
        if self.lotus:
            result['lotus_stats'] = self.get_lotus_stats()

        return result

    @property
    def metrics(self) -> Dict[str, Any]:
        """Quick access to execution metrics."""
        return {
            'episodes': self.episode_count,
            'agents': len(self.agents),
            'mode': self.mode,
            'runners_built': self._runners_built,
            'components_loaded': sum(
                1 for k in self.__dict__ if k.startswith('_lazy_')
            ),
        }

    # =========================================================================
    # LOTUS OPTIMIZATION
    # =========================================================================

    def _init_lotus_optimization(self):
        """
        Initialize LOTUS optimization layer.

        LOTUS-inspired optimizations:
        - Model Cascade: Use cheap models (Haiku) first, escalate to expensive (Opus) only when needed
        - Semantic Cache: Memoize semantic operations with content fingerprinting
        - Batch Executor: Batch LLM calls for throughput optimization
        - Adaptive Validator: Learn when to skip validation based on historical success

        DRY: Uses centralized LotusConfig for all optimization settings.
        """
        try:
            from Jotty.core.lotus.integration import LotusEnhancement, _enhance_agent_runner

            # Create LOTUS enhancement with default config
            self.lotus = LotusEnhancement(
                enable_cascade=True,
                enable_cache=True,
                enable_adaptive_validation=True,
            )
            self.lotus_optimizer = self.lotus.lotus_optimizer

            # Enhance all agent runners with adaptive validation
            for name, runner in self.runners.items():
                _enhance_agent_runner(runner, self.lotus)

                # Pre-warm the adaptive validator with initial trust
                # This allows validation skipping from the start
                # (simulates 15 successful validations per agent)
                for _ in range(15):
                    self.lotus.adaptive_validator.record_result(name, "architect", success=True)
                    self.lotus.adaptive_validator.record_result(name, "auditor", success=True)
                logger.debug(f"Pre-warmed LOTUS validator for agent: {name}")

            logger.info("LOTUS optimization layer initialized (pre-warmed validators)")

        except ImportError as e:
            logger.warning(f"LOTUS optimization not available: {e}")
            self.lotus = None
            self.lotus_optimizer = None

    def get_lotus_stats(self) -> Dict[str, Any]:
        """Get LOTUS optimization statistics."""
        if self.lotus:
            return self.lotus.get_stats()
        return {}

    def get_lotus_savings(self) -> Dict[str, float]:
        """Get estimated cost savings from LOTUS optimization."""
        if self.lotus:
            return self.lotus.get_savings()
        return {}

    # =========================================================================
    # ML Learning Bridge (for SkillOrchestrator / Swarm pipeline integration)
    # =========================================================================

    def get_ml_learning(self):
        """Get MASLearning instance for ML pipeline integration."""
        return self.mas_learning

    def record_report_section_outcome(self, section_name: str, success: bool, error: str = None):
        """Record report section outcome for cross-run learning."""
        try:
            if self.learning_manager and hasattr(self.learning_manager, 'record_experience'):
                self.learning_manager.record_experience(
                    agent_name='report_generator',
                    state={'section': section_name, 'error': (error or '')[:200]},
                    action={'type': 'generate_section', 'section': section_name},
                    reward=1.0 if success else -1.0,
                    domain='report_sections',
                )
        except Exception as e:
            logger.debug(f"Record section outcome failed: {e}")

    def should_skip_report_section(self, section_name: str) -> bool:
        """Check if a report section should be skipped based on learned failures."""
        try:
            if self.learning_manager and hasattr(self.learning_manager, 'get_learned_context'):
                context = self.learning_manager.get_learned_context(
                    state={'section': section_name, 'task': 'report_generation'},
                    action={'type': 'generate_section', 'section': section_name}
                )
                if context and 'negative reward' in context.lower():
                    return True
        except Exception:
            pass
        return False

    # =========================================================================
    # Provider, Ensemble, Learning methods — see _provider_mixin.py,
    # _ensemble_mixin.py, _learning_delegation_mixin.py
    # =========================================================================

    def _register_agents_with_axon(self):
        """Register all agents with SmartAgentSlack for inter-agent messaging."""
        from Jotty.core.agents.feedback_channel import FeedbackMessage, FeedbackType

        def _make_slack_callback(target_actor_name: str):
            def _callback(message):
                try:
                    fb = FeedbackMessage(
                        source_actor=message.from_agent,
                        target_actor=target_actor_name,
                        feedback_type=FeedbackType.RESPONSE,
                        content=str(message.data),
                        context={
                            'format': getattr(message, 'format', 'unknown'),
                            'size_bytes': getattr(message, 'size_bytes', None),
                            'metadata': getattr(message, 'metadata', {}) or {},
                            'timestamp': getattr(message, 'timestamp', None),
                        },
                        requires_response=False,
                        priority=2,
                    )
                    self.feedback_channel.send(fb)
                except Exception as e:
                    logger.warning(f"Slack callback failed for {target_actor_name}: {e}")
            return _callback

        for agent_config in self.agents:
            try:
                agent_obj = agent_config.agent
                signature_obj = getattr(agent_obj, 'signature', None)
                self.agent_slack.register_agent(
                    agent_name=agent_config.name,
                    signature=signature_obj if hasattr(signature_obj, 'input_fields') else None,
                    callback=_make_slack_callback(agent_config.name),
                    max_context=getattr(self.config, 'max_context_tokens', 16000),
                )
            except Exception as e:
                logger.warning(f"Could not register {agent_config.name} with SmartAgentSlack: {e}")

    # list_capabilities() and get_help() removed — see CLAUDE.md or docs/
    
    def parse_intent_to_agent_config(self, natural_language: str) -> AgentConfig:
        """
        Convert natural language to AgentConfig (zero-config).
        
        Public utility method for intent parsing.
        DRY: Reuses IntentParser and AutoAgent.
        
        Args:
            natural_language: Natural language request
            
        Returns:
            AgentConfig with AutoAgent
            
        Example:
            agent_config = swarm.parse_intent_to_agent_config("Research topic")
        """
        # Parse intent (DRY: reuse IntentParser)
        task_graph = self.swarm_intent_parser.parse(natural_language)
        
        # Create AutoAgent (DRY: reuse existing AutoAgent)
        agent = AutoAgent()
        
        # Create AgentConfig with parsed metadata
        agent_config = AgentConfig(
            name="auto",
            agent=agent,
            metadata={
                "original_request": natural_language,
                "task_type": task_graph.task_type.value,
                "workflow": task_graph.workflow,
                "operations": task_graph.operations,
                "integrations": task_graph.integrations,
                "requirements": task_graph.requirements,
            }
        )
        
        logger.info(f"✅ Converted natural language to AgentConfig: {task_graph.task_type.value}")
        return agent_config
    
    async def run(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Run task execution with full autonomy.

        Supports zero-config: natural language goal → autonomous execution.
        For simple tool-calling tasks, use UnifiedExecutor directly instead.

        Args:
            goal: Task goal/description (natural language supported)
            context: Optional ExecutionContext from ModeRouter
            skip_autonomous_setup: If True, skip research/install/configure (fast mode)
            status_callback: Optional callback(stage, detail) for progress updates
            ensemble: Enable prompt ensembling for multi-perspective analysis
            ensemble_strategy: Strategy for ensembling
            **kwargs: Additional arguments

        Returns:
            EpisodeResult with output and metadata
        """
        import time as _time
        run_start_time = _time.time()

        # Lazy init: Build runners on first run
        self._ensure_runners()

        # MAS-ZERO: Reset per-problem experience library
        self._reset_experience()
        self._efficiency_stats = {}  # Reset per-run

        # Extract ExecutionContext if provided by ModeRouter
        exec_context = kwargs.pop('context', None)

        # Extract special kwargs (pop ALL so they don't leak into **kwargs downstream)
        skip_autonomous_setup = kwargs.pop('skip_autonomous_setup', False)
        skip_validation = kwargs.pop('skip_validation', None)  # Explicit override
        status_callback = kwargs.pop('status_callback', None)
        ensemble = kwargs.pop('ensemble', None)  # None = auto-detect, True/False = explicit
        ensemble_strategy = kwargs.pop('ensemble_strategy', 'multi_perspective')

        # If ExecutionContext provided, extract callbacks from it
        if exec_context is not None:
            if status_callback is None and hasattr(exec_context, 'status_callback'):
                status_callback = exec_context.status_callback
            # Emit start event
            if hasattr(exec_context, 'emit_event'):
                from Jotty.core.foundation.types.sdk_types import SDKEventType
                exec_context.emit_event(SDKEventType.PLANNING, {
                    "goal": goal,
                    "mode": self.mode,
                    "agents": len(self.agents),
                })

        # Auto-detect ensemble for certain task types (if not explicitly set)
        # Optima-inspired: adaptive sizing returns (should_ensemble, max_perspectives)
        max_perspectives = 4  # default
        if ensemble is None:
            ensemble, max_perspectives = self._should_auto_ensemble(goal)
            if ensemble:
                logger.info(f"Auto-enabled ensemble: {max_perspectives} perspectives (use ensemble=False to override)")

        # Ensure DSPy LM is configured (critical for all agent operations)
        import dspy
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            lm = self.swarm_provider_gateway.get_lm()
            if lm:
                dspy.configure(lm=lm)
                logger.info(f"✅ DSPy LM configured: {getattr(lm, 'model', 'unknown')}")

        def _status(stage: str, detail: str = ""):
            """Report status if callback provided."""
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"📍 {stage}" + (f": {detail}" if detail else ""))

        # ── FAST PATH: Simple tasks bypass the entire agent pipeline ──
        # ValidationGate classifies task complexity using a cheap LLM (or heuristic).
        # For DIRECT tasks (Q&A, lookups, lists): skip zero-config, ensemble, and
        # AutonomousAgent overhead. Call the LLM directly. Target: <10s.
        # OPTIMIZATION: Skip gate for multi-agent mode — fast path only works
        # for single-agent, so running the gate wastes an LLM call.
        from Jotty.core.orchestration.v2.validation_gate import (
            ValidationGate, ValidationMode, get_validation_gate,
        )

        _gate_decision = None
        if self.mode == "single":
            _fast_gate = get_validation_gate()

            # Determine if user explicitly asked for skip/full validation
            _force_mode = None
            if skip_validation is True:
                _force_mode = ValidationMode.DIRECT
            elif skip_validation is False:
                _force_mode = ValidationMode.FULL

            _gate_decision = await _fast_gate.decide(
                goal=goal,
                agent_name=self.agents[0].name if self.agents else "auto",
                force_mode=_force_mode,
            )

        if _gate_decision and _gate_decision.mode == ValidationMode.DIRECT and self.mode == "single":
            _status("Fast path", f"DIRECT mode — bypassing agent pipeline ({_gate_decision.reason})")

            # Ensure DSPy LM is available (already done above)
            try:
                import dspy as _dspy
                lm = _dspy.settings.lm
                if lm is None:
                    raise RuntimeError("No LM configured")

                _fast_start = _time.time()
                response = lm(prompt=goal)
                if isinstance(response, list):
                    response = response[0] if response else ""
                elif hasattr(response, 'text'):
                    response = response.text
                response = str(response).strip()

                _fast_elapsed = _time.time() - _fast_start
                _status("Fast path complete", f"{_fast_elapsed:.1f}s")

                # Record outcome for gate learning
                _fast_gate.record_outcome(ValidationMode.DIRECT, bool(response))

                # Build minimal EpisodeResult
                fast_result = EpisodeResult(
                    output=response,
                    success=bool(response),
                    trajectory=[{'step': 1, 'action': 'direct_llm', 'output': response[:200]}],
                    tagged_outputs=[],
                    episode=self.episode_count,
                    execution_time=_fast_elapsed,
                    architect_results=[],
                    auditor_results=[],
                    agent_contributions={},
                )
                self.episode_count += 1

                # Save learning (lightweight)
                try:
                    self._save_learnings()
                except Exception:
                    pass

                total_elapsed = _time.time() - run_start_time
                _status("Complete", f"fast path success ({total_elapsed:.1f}s)")
                return fast_result

            except Exception as e:
                logger.info(f"Fast path failed ({e}), falling back to full pipeline")
                # Fall through to normal pipeline

        # Store gate decision for downstream use (AgentRunner will also gate architect/auditor)
        kwargs['_swarm_gate_decision'] = _gate_decision

        # Zero-config: LLM decides single vs multi-agent at RUN TIME (when goal is available)
        if self.enable_zero_config and self.mode == "single":
            _status("Analyzing task", "deciding single vs multi-agent")
            new_agents = self._create_zero_config_agents(goal, status_callback)
            if len(new_agents) > 1:
                # LLM detected parallel sub-goals - upgrade to multi-agent
                self.agents = new_agents
                self.mode = "multi"
                logger.info(f"🔄 Zero-config: Upgraded to {len(self.agents)} agents for parallel execution")

                # Create runners for new agents
                from Jotty.core.orchestration.v2.agent_runner import AgentRunner, AgentRunnerConfig
                for agent_config in self.agents:
                    if agent_config.name not in self.runners:
                        runner_config = AgentRunnerConfig(
                            architect_prompts=self.architect_prompts,
                            auditor_prompts=self.auditor_prompts,
                            config=self.config,
                            agent_name=agent_config.name,
                            enable_learning=True,
                            enable_memory=True
                        )
                        runner = AgentRunner(
                            agent=agent_config.agent,
                            config=runner_config,
                            task_planner=self.swarm_planner,
                            task_board=self.swarm_task_board,
                            swarm_memory=self.swarm_memory,
                            swarm_state_manager=self.swarm_state_manager,
                            learning_manager=self.learning_manager,
                            transfer_learning=self.transfer_learning,
                            swarm_terminal=self.swarm_terminal  # Shared intelligent terminal
                        )
                        self.runners[agent_config.name] = runner

        agent_info = f"{len(self.agents)} AutoAgent(s)" if len(self.agents) > 1 else "AutoAgent (zero-config)"
        _status("Starting", agent_info)

        # Profile execution if enabled
        if self.swarm_profiler:
            profile_context = self.swarm_profiler.profile("SwarmManager.run", metadata={"goal": goal, "mode": self.mode})
            profile_context.__enter__()
        else:
            profile_context = None

        try:
            # Store goal in shared context for state management
            if self.shared_context:
                self.shared_context.set('goal', goal)
                self.shared_context.set('query', goal)

            # Autonomous planning: Research, install, configure if needed
            # For multi-agent mode with zero-config: skip full setup (agents have specific sub-goals)
            # This reduces latency significantly for parallel agent execution
            if not skip_autonomous_setup and self.mode == "single":
                _status("Autonomous setup", "analyzing requirements")
                await self.autonomous_setup(goal, status_callback=status_callback)
            elif not skip_autonomous_setup and self.mode == "multi":
                _status("Fast mode", "multi-agent (agents configured with sub-goals)")
            else:
                _status("Fast mode", "skipping autonomous setup")

            # Set root task in SwarmTaskBoard
            self.swarm_task_board.root_task = goal

            # Record swarm-level step: goal received
            if self.swarm_state_manager:
                self.swarm_state_manager.record_swarm_step({
                    'step': 'goal_received',
                    'goal': goal,
                    'mode': self.mode,
                    'agent_count': len(self.agents)
                })

            # Ensemble mode: Multi-perspective analysis
            # For multi-agent: ensemble happens per-agent (each agent has different sub-goal)
            # For single-agent: ensemble happens at swarm level
            ensemble_result = None
            if ensemble and self.mode == "single":
                _status("Ensembling", f"strategy={ensemble_strategy}, perspectives={max_perspectives}")
                _ens_start = _time.time()
                ensemble_result = await self._execute_ensemble(
                    goal,
                    strategy=ensemble_strategy,
                    status_callback=status_callback,
                    max_perspectives=max_perspectives,
                )
                self._efficiency_stats['ensemble_time'] = _time.time() - _ens_start
                if ensemble_result.get('success'):
                    # Use synthesized perspective to guide execution
                    enriched_goal = f"{goal}\n\n[Multi-Perspective Analysis]:\n{ensemble_result.get('response', '')[:2000]}"

                    # Show quality scores if available
                    quality_scores = ensemble_result.get('quality_scores', {})
                    if quality_scores:
                        avg_quality = sum(quality_scores.values()) / len(quality_scores)
                        scores_str = ", ".join(f"{k}:{v:.0%}" for k, v in quality_scores.items())
                        _status("Ensemble quality", f"avg={avg_quality:.0%} ({scores_str})")
                    else:
                        _status("Ensemble complete", f"{len(ensemble_result.get('perspectives_used', []))} perspectives")

                    goal = enriched_goal  # Use enriched goal for execution
            elif ensemble and self.mode == "multi":
                # Multi-agent mode: SKIP per-agent ensemble (too expensive)
                # With N agents × 4 perspectives = 4N LLM calls - massive overkill
                # Each agent already has a specific sub-goal, no need for multi-perspective
                _status("Ensemble mode", "DISABLED for multi-agent (agents have specific sub-goals)")
                ensemble = False  # Disable for agents

            # Clean up internal flag before passing to agents
            kwargs.pop('_swarm_gate_decision', None)

            # Single-agent mode: Simple execution
            if self.mode == "single":
                agent_name = self.agents[0].name if self.agents else "auto"
                _status("Executing", f"agent '{agent_name}' with skill orchestration")
                # Architect → Actor → Auditor pipeline (now fast with max_eval_iters=2)
                # skip_validation: explicit kwarg wins, else derive from skip_autonomous_setup
                skip_val = skip_validation if skip_validation is not None else skip_autonomous_setup
                # Forward ensemble flag so AutoAgent doesn't auto-detect independently
                if ensemble is not None:
                    kwargs['ensemble'] = ensemble
                result = await self._execute_single_agent(
                    goal,
                    skip_validation=skip_val,
                    status_callback=status_callback,
                    ensemble_context=ensemble_result if ensemble else None,
                    **kwargs
                )

                # Optima-inspired efficiency summary
                total_elapsed = _time.time() - run_start_time
                ens_t = self._efficiency_stats.get('ensemble_time', 0)
                exec_t = getattr(result, 'execution_time', total_elapsed - ens_t)
                overhead = max(0, total_elapsed - exec_t)
                overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                self._efficiency_stats.update({'total_time': total_elapsed, 'overhead_pct': overhead_pct})
                _status("Complete", f"{'success' if result.success else 'failed'} ({total_elapsed:.1f}s, overhead={overhead_pct:.0f}%)")

                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)

                return result

            # Multi-agent mode: Use SwarmTaskBoard for coordination
            else:
                # MAS-ZERO: Dynamic reduction — check if multi-agent is overkill
                if self._mas_zero_should_reduce(goal):
                    _status("MAS-ZERO reduction", "reverting to single agent (simpler is better)")
                    self.mode = "single"
                    result = await self._execute_single_agent(
                        goal,
                        skip_validation=skip_validation if skip_validation is not None else skip_autonomous_setup,
                        status_callback=status_callback,
                        **kwargs
                    )
                else:
                    paradigm = kwargs.pop('discussion_paradigm', 'fanout')
                    _status("Executing", f"{len(self.agents)} agents — paradigm: {paradigm}")
                    _sv = skip_validation if skip_validation is not None else skip_autonomous_setup
                    result = await self._execute_multi_agent(
                        goal,
                        ensemble_context=ensemble_result if ensemble else None,
                        status_callback=status_callback,
                        ensemble=ensemble,
                        ensemble_strategy=ensemble_strategy,
                        discussion_paradigm=paradigm,
                        skip_validation=_sv,
                        **kwargs
                    )

                # Optima-inspired efficiency summary
                total_elapsed = _time.time() - run_start_time
                ens_t = self._efficiency_stats.get('ensemble_time', 0)
                exec_t = getattr(result, 'execution_time', total_elapsed - ens_t)
                overhead = max(0, total_elapsed - exec_t)
                overhead_pct = (overhead / total_elapsed * 100) if total_elapsed > 0 else 0
                self._efficiency_stats.update({'total_time': total_elapsed, 'overhead_pct': overhead_pct})
                _status("Complete", f"{'success' if result.success else 'failed'} ({total_elapsed:.1f}s, overhead={overhead_pct:.0f}%)")

                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)

                return result
        except Exception as e:
            _status("Error", str(e)[:50])
            # Exit profiling context on error
            if profile_context:
                profile_context.__exit__(type(e), e, None)
            raise
    
    # _execute_ensemble and _should_auto_ensemble — see _ensemble_mixin.py

    async def _execute_single_agent(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Execute single-agent mode.

        MAS-ZERO: For comparison/analysis tasks, runs building blocks
        (direct + ensemble) in parallel and verifies the best answer.
        For simple tasks, runs direct execution as before.

        Args:
            goal: Task goal
            **kwargs: Additional arguments

        Returns:
            EpisodeResult
        """
        # Remove ensemble_context from kwargs before passing to runner
        kwargs.pop('ensemble_context', None)

        agent_config = self.agents[0]
        runner = self.runners[agent_config.name]

        # NOTE: MAS-ZERO building blocks (MAS-Init) are NOT used in single-agent
        # mode. AutoAgent already has its own ensemble pipeline internally, so
        # building blocks would duplicate LLM calls. In multi-agent mode,
        # MAS-Verify + MAS-Evolve provide the cross-candidate selection instead.

        # Standard single-agent execution
        result = await runner.run(goal=goal, **kwargs)

        # Learn from execution (DRY: reuse workflow learner)
        if result.success:
            self._learn_from_result(result, agent_config)

        # Post-episode learning
        self._post_episode_learning(result, goal)

        # Auto-save learnings (persist across sessions)
        self._auto_save_learnings()

        return result

    async def _execute_multi_agent(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Execute multi-agent mode with configurable discussion paradigm.

        MALLM-inspired paradigms (Becker et al., EMNLP 2025):
            fanout      — All agents run in parallel on decomposed tasks (default)
            relay       — Sequential chain; each agent builds on previous output
            debate      — Agents critique each other's outputs in rounds
            refinement  — Iterative improve loop until quality stabilizes

        DRY: All paradigms reuse the same AgentRunner.run() and semaphore.
        """
        from Jotty.core.orchestration.v2.swarm_roadmap import TaskStatus
        from Jotty.core.learning.predictive_marl import ActualTrajectory
        from Jotty.core.agents.feedback_channel import FeedbackMessage, FeedbackType

        # Extract callbacks and ensemble params before passing to runners
        kwargs.pop('ensemble_context', None)
        status_callback = kwargs.pop('status_callback', None)
        ensemble = kwargs.pop('ensemble', False)
        ensemble_strategy = kwargs.pop('ensemble_strategy', 'multi_perspective')
        discussion_paradigm = kwargs.pop('discussion_paradigm', 'fanout')

        # MALLM-inspired: Dispatch to alternative paradigms before fan-out
        if discussion_paradigm == 'relay':
            return await self._paradigm_relay(goal, status_callback=status_callback, **kwargs)
        elif discussion_paradigm == 'debate':
            return await self._paradigm_debate(goal, status_callback=status_callback, **kwargs)
        elif discussion_paradigm == 'refinement':
            return await self._paradigm_refinement(goal, status_callback=status_callback, **kwargs)
        # else: 'fanout' — fall through to existing parallel execution below

        # Status update at method start
        if status_callback:
            try:
                status_callback("Multi-agent exec", f"starting {len(self.agents)} parallel agents")
            except Exception as e:
                logger.error(f"Status callback failed: {e}")

        max_attempts = getattr(self.config, 'max_task_attempts', 2)

        # Clear task board for fresh run (avoid stale tasks from previous runs)
        self.swarm_task_board.subtasks.clear()
        self.swarm_task_board.completed_tasks.clear()
        self.swarm_task_board.execution_order.clear()

        # Add tasks to SwarmTaskBoard
        # Zero-config agents from LLM are PARALLEL (independent sub-goals)
        # Only add dependencies if explicitly specified in agent config
        for i, agent_config in enumerate(self.agents):
            task_id = f"task_{i+1}"
            # Check if agent has explicit dependencies
            deps = getattr(agent_config, 'depends_on', []) or []

            # Use agent's sub-goal (from capabilities) as task description
            sub_goal = agent_config.capabilities[0] if agent_config.capabilities else f"{goal} (agent: {agent_config.name})"

            self.swarm_task_board.add_task(
                task_id=task_id,
                description=sub_goal,
                actor=agent_config.name,
                depends_on=deps  # Empty for parallel execution
            )
            logger.info(f"📋 Added task {task_id} for {agent_config.name}: {sub_goal[:50]}... (parallel: {len(deps)==0})")

        all_results = {}  # agent_name -> EpisodeResult
        attempt_counts = {}  # task_id -> attempts

        while True:
            # Collect all ready tasks (no unresolved dependencies)
            batch = []

            while True:
                next_task = self.swarm_task_board.get_next_task()
                if next_task is None:
                    break
                # Mark as IN_PROGRESS so it's not returned again
                next_task.status = TaskStatus.IN_PROGRESS
                batch.append(next_task)

            if not batch:
                break

            # Show batch info immediately with agent names for better UX
            if status_callback and len(batch) > 0:
                try:
                    agent_names = [t.actor for t in batch]
                    status_callback("Running batch", f"{len(batch)} agents: {', '.join(agent_names[:5])}")
                    # Show each agent's task for clarity
                    for task in batch:
                        agent_cfg = next((a for a in self.agents if a.name == task.actor), None)
                        sub_goal = agent_cfg.capabilities[0] if agent_cfg and agent_cfg.capabilities else task.description[:50]
                        status_callback(f"  {task.actor}", sub_goal[:60])
                except Exception:
                    pass

            # Pre-execution: trajectory prediction (non-blocking, run in background)
            predictions = {}
            # Skip trajectory prediction to reduce latency - agents start immediately
            # Prediction can happen asynchronously after execution starts
            if self.trajectory_predictor and len(batch) <= 2:  # Only for small batches
                for task in batch:
                    try:
                        prediction = self.trajectory_predictor.predict(
                            current_state=self.get_current_state(),
                            acting_agent=task.actor,
                            proposed_action={'task': task.description},
                            other_agents=[a.name for a in self.agents if a.name != task.actor],
                            goal=goal
                        )
                        predictions[task.actor] = prediction
                    except Exception as e:
                        logger.debug(f"Trajectory prediction skipped for {task.actor}: {e}")

            # Execute batch concurrently (status_callback already extracted at method start)
            # AIOS-inspired: Semaphore limits how many agents call LLM simultaneously.
            # Without this, N agents × (architect + agent + auditor) = 3N concurrent API calls.
            async def _run_task(task):
                # Check if we'll need to wait for a slot
                if self._agent_semaphore._value == 0:
                    self._scheduling_stats['total_waited'] += 1
                    if status_callback:
                        try:
                            status_callback(f"Agent {task.actor}", "waiting for LLM slot...")
                        except Exception:
                            pass

                async with self._agent_semaphore:
                    # Track concurrency stats
                    self._scheduling_stats['total_scheduled'] += 1
                    self._scheduling_stats['current_concurrent'] += 1
                    if self._scheduling_stats['current_concurrent'] > self._scheduling_stats['peak_concurrent']:
                        self._scheduling_stats['peak_concurrent'] = self._scheduling_stats['current_concurrent']

                    try:
                        # Show which agent is executing
                        agent_cfg = next((a for a in self.agents if a.name == task.actor), None)
                        sub_goal = agent_cfg.capabilities[0] if agent_cfg and agent_cfg.capabilities else task.description[:60]

                        if status_callback:
                            try:
                                status_callback(f"Agent {task.actor}", f"starting: {sub_goal}")
                            except Exception:
                                pass

                        # Create agent-specific status callback that prefixes with agent name
                        def agent_status_callback(stage: str, detail: str = ""):
                            if status_callback:
                                try:
                                    status_callback(f"  [{task.actor}] {stage}", detail)
                                except Exception:
                                    pass

                        runner = self.runners[task.actor]
                        # Pass the agent-specific callback and ensemble params
                        task_kwargs = dict(kwargs)
                        task_kwargs['status_callback'] = agent_status_callback
                        # Forward ensemble flag explicitly so AutoAgent doesn't auto-detect
                        task_kwargs['ensemble'] = ensemble
                        if ensemble:
                            task_kwargs['ensemble_strategy'] = ensemble_strategy
                        # Forward the swarm-level gate decision to skip redundant
                        # per-agent architect/auditor if the swarm already decided
                        if '_swarm_gate_decision' in task_kwargs:
                            task_kwargs.pop('_swarm_gate_decision')  # clean up internal flag
                        return task, await runner.run(goal=task.description, **task_kwargs)
                    finally:
                        self._scheduling_stats['current_concurrent'] -= 1

            coro_results = await asyncio.gather(
                *[_run_task(t) for t in batch],
                return_exceptions=True
            )

            # Process results
            for coro_result in coro_results:
                if isinstance(coro_result, Exception):
                    logger.error(f"Task execution exception: {coro_result}")
                    if status_callback:
                        try:
                            status_callback("Agent error", str(coro_result)[:60])
                        except Exception:
                            pass
                    continue

                task, result = coro_result
                attempt_counts[task.task_id] = attempt_counts.get(task.task_id, 0) + 1

                # Show agent completion status
                if status_callback:
                    status_icon = "✓" if result.success else "✗"
                    try:
                        status_callback(f"{status_icon} Agent {task.actor}", "completed" if result.success else "failed")
                    except Exception:
                        pass
                reward = 1.0 if result.success else -0.5

                # Post-execution: divergence learning
                if self.trajectory_predictor and task.actor in predictions:
                    try:
                        prediction = predictions[task.actor]
                        actual = ActualTrajectory(
                            steps=result.trajectory or [],
                            actual_reward=reward
                        )
                        divergence = self.trajectory_predictor.compute_divergence(prediction, actual)
                        self.divergence_memory.store(divergence)
                        self.trajectory_predictor.update_from_divergence(divergence)

                        # Use divergence as TD error weight for Q-update
                        divergence_penalty = 1.0 - min(1.0, divergence.total_divergence())
                        adjusted_reward = reward * divergence_penalty
                        state = {'query': goal, 'agent': task.actor}
                        action = {'actor': task.actor, 'task': task.description[:100]}
                        self.learning_manager.record_outcome(state, action, adjusted_reward)
                    except Exception as e:
                        logger.debug(f"Divergence learning skipped for {task.actor}: {e}")

                if result.success:
                    self.swarm_task_board.complete_task(task.task_id, result={'output': result.output})
                    all_results[task.actor] = result

                    agent_config = next((a for a in self.agents if a.name == task.actor), None)
                    if agent_config:
                        self._learn_from_result(result, agent_config)
                else:
                    # Retry with enriched context if attempts remain
                    if attempt_counts.get(task.task_id, 1) < max_attempts:
                        error_msg = str(getattr(result, 'error', None) or 'Execution failed')
                        fb = FeedbackMessage(
                            source_actor='swarm_manager',
                            target_actor=task.actor,
                            feedback_type=FeedbackType.ERROR,
                            content=f"Previous attempt failed: {error_msg}. Please try a different approach.",
                            context={'attempt': attempt_counts[task.task_id], 'error': error_msg},
                            requires_response=False,
                            priority=1,
                        )
                        self.feedback_channel.send(fb)
                        # Reset task to PENDING for retry
                        try:
                            self.swarm_task_board.add_task(
                                task_id=f"{task.task_id}_retry{attempt_counts[task.task_id]}",
                                description=task.description,
                                actor=task.actor
                            )
                        except Exception:
                            self.swarm_task_board.fail_task(task.task_id, error=error_msg)
                    else:
                        self.swarm_task_board.fail_task(
                            task.task_id,
                            error=str(getattr(result, 'error', None) or 'Execution failed')
                        )
                    all_results[task.actor] = result

        # MAS-ZERO: Handle TOO_HARD signals from agents
        # If any agent signaled TOO_HARD, try re-routing its task
        for agent_name, result in list(all_results.items()):
            output = result.output if hasattr(result, 'output') else result
            is_too_hard = (
                isinstance(output, dict) and output.get('too_hard')
            ) or (
                hasattr(result, 'output') and isinstance(result.output, dict)
                and result.output.get('too_hard')
            )
            if is_too_hard and not result.success:
                logger.info(f"MAS-ZERO: Agent '{agent_name}' signaled TOO_HARD, "
                           f"task may need re-decomposition")

        # MAS-ZERO: Meta-feedback evaluation (solvability + completeness)
        meta_feedback = self._mas_zero_evaluate(goal, all_results)

        # MAS-ZERO: Iterative refinement if meta-feedback says refine
        if meta_feedback.get('should_refine') and len(all_results) > 0:
            if status_callback:
                try:
                    status_callback("MAS-Evolve", "refining based on meta-feedback")
                except Exception:
                    pass
            all_results = await self._mas_zero_evolve(
                goal, all_results,
                max_iterations=2,
                status_callback=status_callback,
                **kwargs,
            )

        # Cooperative credit assignment
        self._assign_cooperative_credit(all_results, goal)

        # Post-episode learning
        combined_result = self._aggregate_results(all_results, goal)
        self._post_episode_learning(combined_result, goal)

        # Auto-save learnings (persist across sessions)
        self._auto_save_learnings()

        return combined_result

    # =========================================================================
    # DISCUSSION PARADIGMS (MALLM-inspired, Becker et al. EMNLP 2025)
    # =========================================================================

    async def _paradigm_run_agent(
        self,
        runner,
        sub_goal: str,
        agent_name: str,
        **kwargs,
    ) -> EpisodeResult:
        """
        Run a single agent within a paradigm, using fast-path when possible.

        LATENCY OPTIMIZATION: For simple sub-goals (no tool use needed),
        bypass the full AutoAgent pipeline (planner → skill select → execute)
        and make a direct LLM call. Saves 2-4 LLM calls per agent.

        Heuristic: If sub-goal doesn't need tools (no "search", "fetch",
        "create file", "send", etc.), go direct.

        Args:
            runner: AgentRunner instance
            sub_goal: The sub-task for this agent
            agent_name: Agent name for logging
            **kwargs: Forwarded to runner.run() if full pipeline needed

        Returns:
            EpisodeResult
        """
        # Heuristic: does this sub-goal need external tools?
        _tool_keywords = [
            'search', 'fetch', 'scrape', 'download', 'upload',
            'send', 'email', 'telegram', 'slack',
            'create file', 'save file', 'write file', 'read file',
            'execute', 'run code', 'compile', 'deploy',
            'database', 'sql', 'api call',
        ]
        _needs_tools = any(kw in sub_goal.lower() for kw in _tool_keywords)

        if not _needs_tools:
            # FAST PATH: Direct LLM call — ~1 LLM call instead of 3-5
            import dspy as _dspy
            lm = _dspy.settings.lm
            if lm:
                try:
                    import time as _time
                    _start = _time.time()

                    # Fire hooks if runner has them
                    hook_ctx = runner._run_hooks(
                        'pre_run', goal=sub_goal, agent_name=agent_name,
                        fast_path=True,
                    )
                    sub_goal = hook_ctx.get('goal', sub_goal)

                    response = lm(prompt=sub_goal)
                    if isinstance(response, list):
                        response = response[0] if response else ""
                    response = str(response).strip()

                    _elapsed = _time.time() - _start
                    logger.info(
                        f"⚡ Paradigm fast-path: {agent_name} "
                        f"({_elapsed:.1f}s, 1 LLM call)"
                    )

                    result = EpisodeResult(
                        output=response,
                        success=bool(response),
                        trajectory=[],
                        tagged_outputs={agent_name: response},
                        episode=0,
                        execution_time=_elapsed,
                        architect_results=[],
                        auditor_results=[],
                        agent_contributions={agent_name: response[:200]},
                    )

                    runner._run_hooks(
                        'post_run', goal=sub_goal, agent_name=agent_name,
                        result=result, success=result.success, elapsed=_elapsed,
                    )
                    return result
                except Exception as e:
                    logger.warning(f"Fast-path failed for {agent_name}: {e}, falling back to full pipeline")

        # FULL PATH: Use the complete AgentRunner pipeline
        return await runner.run(goal=sub_goal, **kwargs)

    async def _paradigm_relay(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Relay paradigm: agents execute sequentially, each building on prior output.

        MALLM's "relay" — like a relay race where context is passed along.
        DRY: Reuses existing runners. KISS: Simple loop.

        Best for: research → summarize → format pipelines.
        """
        status_callback = kwargs.pop('status_callback', None)
        # LATENCY: Disable per-agent ensemble — each sub-task is focused
        kwargs.setdefault('ensemble', False)

        all_results = {}
        enriched_goal = goal

        for agent_config in self.agents:
            runner = self.runners.get(agent_config.name)
            if not runner:
                continue

            sub_goal = agent_config.capabilities[0] if agent_config.capabilities else enriched_goal

            if status_callback:
                try:
                    status_callback(f"Relay → {agent_config.name}", sub_goal[:60])
                except Exception:
                    pass

            async with self._agent_semaphore:
                result = await self._paradigm_run_agent(
                    runner, sub_goal, agent_config.name, **kwargs
                )

            all_results[agent_config.name] = result

            # Chain: feed this agent's output into next agent's context
            if result.success and result.output:
                enriched_goal = (
                    f"{goal}\n\n"
                    f"[Previous agent '{agent_config.name}' output]:\n"
                    f"{str(result.output)[:2000]}"
                )
            else:
                logger.warning(f"Relay: {agent_config.name} failed, continuing with original goal")

        combined = self._aggregate_results(all_results, goal)
        self._post_episode_learning(combined, goal)
        self._auto_save_learnings()
        return combined

    async def _paradigm_debate(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Debate paradigm: agents produce drafts, then critique each other in rounds.

        MALLM's "debate" — adversarial refinement where agents challenge
        each other's solutions.  DRY: Reuses runners + FeedbackChannel.
        KISS: 2 rounds max (draft + 1 critique round).

        Best for: analysis tasks, controversial topics, quality-sensitive output.
        """
        status_callback = kwargs.pop('status_callback', None)
        max_debate_rounds = kwargs.pop('debate_rounds', 2)
        # LATENCY: Disable per-agent ensemble — debate itself IS multi-perspective
        kwargs.setdefault('ensemble', False)

        all_results: Dict[str, EpisodeResult] = {}

        # Round 1: All agents produce initial drafts (fan-out)
        if status_callback:
            try:
                status_callback("Debate round 1", "all agents drafting")
            except Exception:
                pass

        draft_tasks = []
        for agent_config in self.agents:
            runner = self.runners.get(agent_config.name)
            if not runner:
                continue
            sub_goal = agent_config.capabilities[0] if agent_config.capabilities else goal
            draft_tasks.append((agent_config.name, runner, sub_goal))

        async def _run_draft(name, runner, sub_goal):
            async with self._agent_semaphore:
                return name, await self._paradigm_run_agent(
                    runner, sub_goal, name, **kwargs
                )

        draft_results = await asyncio.gather(
            *[_run_draft(n, r, g) for n, r, g in draft_tasks],
            return_exceptions=True,
        )

        drafts = {}
        for item in draft_results:
            if not isinstance(item, tuple):
                # Exception or timeout — skip
                logger.warning(f"Debate draft failed: {item}")
                continue
            name, result = item
            all_results[name] = result
            if result.success and result.output:
                drafts[name] = str(result.output)[:1500]

        if len(drafts) < 2:
            # Not enough drafts to debate — fall back to fanout result
            combined = self._aggregate_results(all_results, goal)
            self._post_episode_learning(combined, goal)
            self._auto_save_learnings()
            return combined

        # Rounds 2+: Critique — each agent sees all other drafts and refines
        for round_num in range(2, max_debate_rounds + 1):
            if status_callback:
                try:
                    status_callback(f"Debate round {round_num}", "agents critiquing & refining")
                except Exception:
                    pass

            # Build critique context: each agent sees OTHER agents' drafts
            critique_tasks = []
            for agent_config in self.agents:
                runner = self.runners.get(agent_config.name)
                if not runner or agent_config.name not in drafts:
                    continue

                others = "\n\n".join(
                    f"[{name}'s draft]: {text}"
                    for name, text in drafts.items()
                    if name != agent_config.name
                )
                sub_goal = agent_config.capabilities[0] if agent_config.capabilities else goal
                critique_goal = (
                    f"{sub_goal}\n\n"
                    f"Other agents produced these solutions. "
                    f"Critique them and improve your answer:\n{others}"
                )
                critique_tasks.append((agent_config.name, runner, critique_goal))

            critique_results = await asyncio.gather(
                *[_run_draft(n, r, g) for n, r, g in critique_tasks],
                return_exceptions=True,
            )

            for item in critique_results:
                if not isinstance(item, tuple):
                    logger.warning(f"Debate critique failed: {item}")
                    continue
                name, result = item
                all_results[name] = result  # overwrite with refined version
                if result.success and result.output:
                    drafts[name] = str(result.output)[:1500]

        combined = self._aggregate_results(all_results, goal)
        self._post_episode_learning(combined, goal)
        self._auto_save_learnings()
        return combined

    async def _paradigm_refinement(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Collective refinement paradigm: iterative improvement until quality stabilizes.

        MALLM's "collective_refinement" — all agents see the same shared draft
        and iteratively improve it.  DRY: Reuses runners.
        KISS: Max 3 iterations, stop early if output stabilizes.

        Best for: writing tasks, code generation, report creation.
        """
        status_callback = kwargs.pop('status_callback', None)
        max_iterations = kwargs.pop('refinement_iterations', 3)
        # LATENCY: Disable per-agent ensemble — refinement itself IS iterative improvement
        kwargs.setdefault('ensemble', False)

        # Pick the first agent to produce initial draft
        first_agent = self.agents[0]
        runner = self.runners.get(first_agent.name)
        sub_goal = first_agent.capabilities[0] if first_agent.capabilities else goal

        if status_callback:
            try:
                status_callback("Refinement", f"initial draft by {first_agent.name}")
            except Exception:
                pass

        async with self._agent_semaphore:
            result = await self._paradigm_run_agent(
                runner, sub_goal, first_agent.name, **kwargs
            )

        current_draft = str(result.output)[:3000] if result.output else ""
        all_results = {first_agent.name: result}
        prev_draft = ""

        # Iterate: each remaining agent refines the shared draft
        for iteration in range(1, max_iterations + 1):
            # Early stop: draft didn't change meaningfully
            if current_draft and prev_draft and current_draft[:200] == prev_draft[:200]:
                logger.info(f"Refinement: converged at iteration {iteration}")
                break

            prev_draft = current_draft

            for agent_config in self.agents[1:]:  # skip first (already drafted)
                refiner = self.runners.get(agent_config.name)
                if not refiner:
                    continue

                refine_sub = agent_config.capabilities[0] if agent_config.capabilities else goal
                refine_goal = (
                    f"{refine_sub}\n\n"
                    f"Here is the current draft. Improve it:\n{current_draft}"
                )

                if status_callback:
                    try:
                        status_callback(
                            f"Refinement iter {iteration}",
                            f"{agent_config.name} improving",
                        )
                    except Exception:
                        pass

                async with self._agent_semaphore:
                    ref_result = await self._paradigm_run_agent(
                        refiner, refine_goal, agent_config.name, **kwargs
                    )

                all_results[agent_config.name] = ref_result
                if ref_result.success and ref_result.output:
                    current_draft = str(ref_result.output)[:3000]

        combined = self._aggregate_results(all_results, goal)
        self._post_episode_learning(combined, goal)
        self._auto_save_learnings()
        return combined

    def _aggregate_results(self, results: Dict[str, EpisodeResult], goal: str) -> EpisodeResult:
        """
        Combine all agent outputs into a single EpisodeResult.

        MAS-ZERO: Uses CandidateVerifier for intelligent selection
        instead of naive concatenation when multiple agents produce output.
        """
        if not results:
            return EpisodeResult(
                output=None,
                success=False,
                trajectory=[],
                tagged_outputs=[],
                episode=self.episode_count,
                execution_time=0.0,
                architect_results=[],
                auditor_results=[],
                agent_contributions={},
                alerts=["No tasks executed"],
            )

        if len(results) == 1:
            return list(results.values())[0]

        # MAS-ZERO: Use CandidateVerifier to pick the best output
        # Falls back to combined dict if verification fails or isn't applicable
        verified_output = self._mas_zero_verify(goal, results)

        # If verification selected a single best answer, use it
        # Otherwise fall back to combined output dict
        if verified_output is not None:
            combined_output = verified_output
        else:
            combined_output = {name: r.output for name, r in results.items()}

        all_success = all(r.success for r in results.values())

        # Merge trajectories
        merged_trajectory = []
        for name, r in results.items():
            for step in (r.trajectory or []):
                step_copy = dict(step)
                step_copy['agent'] = name
                merged_trajectory.append(step_copy)

        # Merge agent contributions
        merged_contributions = {}
        for r in results.values():
            if hasattr(r, 'agent_contributions') and r.agent_contributions:
                merged_contributions.update(r.agent_contributions)

        return EpisodeResult(
            output=combined_output,
            success=all_success,
            trajectory=merged_trajectory,
            tagged_outputs=[],
            episode=self.episode_count,
            execution_time=sum(getattr(r, 'execution_time', 0) for r in results.values()),
            architect_results=[],
            auditor_results=[],
            agent_contributions=merged_contributions
        )

    def _assign_cooperative_credit(self, results: Dict[str, EpisodeResult], goal: str):
        """
        Compute cooperative reward decomposition across agents.

        A-Team v8.0: Uses adaptive learned weights instead of hardcoded 0.3/0.4/0.3.
        Weights are updated based on episode success and persisted across sessions.
        """
        if not results or len(results) < 2:
            return

        # Track episode success for weight updates
        episode_success = all(r.success for r in results.values())

        for agent_name, result in results.items():
            base_reward = 1.0 if result.success else 0.0

            # Cooperation bonus: did this agent unblock downstream tasks?
            other_successes = sum(1 for n, r in results.items() if n != agent_name and r.success)
            total_others = len(results) - 1
            cooperation_bonus = other_successes / total_others if total_others > 0 else 0.0

            # Predictability bonus: was trajectory prediction accurate?
            predictability_bonus = 0.5  # Default neutral

            # A-Team v8.0: Use adaptive learned weights instead of hardcoded values
            cooperative_reward = (
                self.credit_weights.get('base_reward') * base_reward +
                self.credit_weights.get('cooperation_bonus') * cooperation_bonus +
                self.credit_weights.get('predictability_bonus') * predictability_bonus
            )

            try:
                state = {'query': goal, 'agent': agent_name, 'cooperative': True}
                action = {'actor': agent_name, 'task': goal[:100]}
                self.learning_manager.record_outcome(state, action, cooperative_reward, done=True)
            except Exception as e:
                logger.debug(f"Cooperative credit recording skipped for {agent_name}: {e}")

        # A-Team v8.0: Update weights based on episode outcome
        if episode_success:
            # Success - cooperation bonus was valuable, strengthen it
            self.credit_weights.update_from_feedback('cooperation_bonus', 0.1, reward=1.0)
        else:
            # Failure - maybe base reward should matter more
            self.credit_weights.update_from_feedback('base_reward', 0.05, reward=0.0)

    def _create_zero_config_agents(self, task: str, status_callback=None) -> List[AgentConfig]:
        """Delegate to ZeroConfigAgentFactory."""
        if not hasattr(self, '_zero_config_factory') or self._zero_config_factory is None:
            from Jotty.core.orchestration.v2.zero_config_factory import ZeroConfigAgentFactory
            self._zero_config_factory = ZeroConfigAgentFactory()
        return self._zero_config_factory.create_agents(task, status_callback)

    # _should_auto_ensemble — see _ensemble_mixin.py

    def _post_episode_learning(self, result: EpisodeResult, goal: str):
        """Delegate to SwarmLearningPipeline."""
        self.learning.post_episode(
            result=result,
            goal=goal,
            agents=self.agents,
            architect_prompts=self.architect_prompts,
            mas_learning=getattr(self, 'mas_learning', None),
            swarm_terminal=getattr(self, 'swarm_terminal', None),
        )
        self.episode_count = self.learning.episode_count

    def _learn_from_result(self, result: EpisodeResult, agent_config: AgentConfig):
        """Delegate to SwarmLearningPipeline."""
        self.learning.learn_from_result(
            result=result,
            agent_config=agent_config,
            workflow_learner=self.swarm_workflow_learner,
        )
    
    async def autonomous_setup(self, goal: str, status_callback=None):
        """
        Autonomous setup: Research, install, configure.

        Public method for manual autonomous setup.
        DRY: Reuses all autonomous components.

        Args:
            goal: Task goal
            status_callback: Optional callback(stage, detail) for progress

        Example:
            await swarm.autonomous_setup("Set up Reddit scraping")
        """
        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"📍 {stage}" + (f": {detail}" if detail else ""))

        # Cache check: skip if already set up for this goal
        cache_key = hash(goal)
        if cache_key in self._setup_cache:
            _status("Setup", "using cached")
            return

        # Parse intent to understand requirements
        _status("Parsing intent", goal)
        task_graph = self.swarm_intent_parser.parse(goal)

        # Research solutions if needed
        if task_graph.requirements or task_graph.integrations:
            # Filter out stop words and meaningless single-word requirements
            stop_words = {'existing', 'use', 'check', 'find', 'get', 'the', 'a', 'an', 'and', 'or', 'for', 'with'}
            meaningful_requirements = [
                req for req in task_graph.requirements
                if req.lower() not in stop_words and len(req.split()) > 1
            ]

            if meaningful_requirements:
                _status("Researching", f"{len(meaningful_requirements)} requirements")

            for i, requirement in enumerate(meaningful_requirements):
                if not requirement.strip():
                    continue
                _status("Research", f"[{i+1}/{len(meaningful_requirements)}] {requirement[:30]}")
                research_result = await self.swarm_researcher.research(requirement)
                if research_result.tools_found:
                    _status("Found tools", ", ".join(research_result.tools_found[:3]))
                    for tool in research_result.tools_found:
                        _status("Installing", tool)
                        await self.swarm_installer.install(tool)

        # Configure integrations
        if task_graph.integrations:
            _status("Configuring", f"{len(task_graph.integrations)} integrations")
            for integration in task_graph.integrations:
                _status("Configure", integration)
                await self.swarm_configurator.configure(integration)

        # Mark as cached
        self._setup_cache[cache_key] = True
        _status("Setup complete", "")

    # =====================================================================
    # State Management Methods (V1 capabilities integrated)
    # =====================================================================
    
    # State delegation — use self.swarm_state_manager directly.
    # Kept get_current_state as it's used internally by _execute_multi_agent.
    def get_current_state(self) -> Dict[str, Any]:
        """Get current swarm-level state."""
        if not self.swarm_state_manager:
            return {}
        return self.swarm_state_manager.get_current_state()

    # =====================================================================
    # Warmup — delegated to SwarmWarmup
    # =====================================================================

    async def warmup(self, **kwargs) -> Dict[str, Any]:
        """DrZero-inspired zero-data bootstrapping. See SwarmWarmup."""
        if not hasattr(self, '_warmup') or self._warmup is None:
            from Jotty.core.orchestration.v2.swarm_warmup import SwarmWarmup
            self._warmup = SwarmWarmup(self)
        return await self._warmup.warmup(**kwargs)

    def get_warmup_recommendation(self) -> Dict[str, Any]:
        """Check if warmup would be beneficial."""
        if not hasattr(self, '_warmup') or self._warmup is None:
            from Jotty.core.orchestration.v2.swarm_warmup import SwarmWarmup
            self._warmup = SwarmWarmup(self)
        return self._warmup.get_recommendation()

    # =====================================================================
    # DAG — delegated to SwarmDAGExecutor
    # =====================================================================

    async def run_with_dag(self, implementation_plan: str, **kwargs) -> EpisodeResult:
        """Execute via DAG-based orchestration. See SwarmDAGExecutor."""
        if not hasattr(self, '_dag_executor') or self._dag_executor is None:
            from Jotty.core.orchestration.v2.swarm_dag_executor import SwarmDAGExecutor
            self._dag_executor = SwarmDAGExecutor(self)
        return await self._dag_executor.run(implementation_plan, **kwargs)

    def get_dag_agents(self):
        """Get DAG agents for external use."""
        if not hasattr(self, '_dag_executor') or self._dag_executor is None:
            from Jotty.core.orchestration.v2.swarm_dag_executor import SwarmDAGExecutor
            self._dag_executor = SwarmDAGExecutor(self)
        return self._dag_executor.get_agents()
