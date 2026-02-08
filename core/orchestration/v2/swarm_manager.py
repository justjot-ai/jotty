"""
SwarmManager - Unified Orchestrator with User-Friendly Names

Uses user-friendly components:
- SwarmTaskBoard (replaces MarkovianTODO) - swarm-level task tracking
- SwarmPlanner (replaces AgenticPlanner) - swarm-level planning
- SwarmMemory (replaces HierarchicalMemory) - swarm-level memory
- AgentRunner (executes agents with validation)

Handles both single-agent (N=1) and multi-agent (N>1) cases.

Skills Integration:
- AutoAgent (used in AgentRunner) automatically discovers and executes skills
- Skills are integrated via AutoAgent's skill discovery and execution
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ...foundation.data_structures import JottyConfig, EpisodeResult
from ...foundation.agent_config import AgentConfig
from .agent_runner import AgentRunner, AgentRunnerConfig
# Import from v2 local modules (no v1 dependencies)
from .swarm_roadmap import MarkovianTODO as SwarmTaskBoard, TaskStatus
from ...agents.agentic_planner import AgenticPlanner as SwarmPlanner
from ...memory.cortex import HierarchicalMemory as SwarmMemory
# Feature components (using Swarm prefix for consistency)
from ...registry.agui_component_registry import get_agui_registry, AGUIComponentRegistry as SwarmUIRegistry
from ...monitoring.profiler import PerformanceProfiler as SwarmProfiler
from ...registry.tool_validation import ToolValidator as SwarmToolValidator
from ...registry.tools_registry import get_tools_registry, ToolsRegistry as SwarmToolRegistry
# Autonomous components (DRY: reuse existing, logical naming)
from ...autonomous.intent_parser import IntentParser
from ...agents.auto_agent import AutoAgent
from .swarm_researcher import SwarmResearcher
from .swarm_installer import SwarmInstaller
from .swarm_configurator import SwarmConfigurator
from .swarm_code_generator import SwarmCodeGenerator
from .swarm_workflow_learner import SwarmWorkflowLearner
from .swarm_integrator import SwarmIntegrator
from .swarm_terminal import SwarmTerminal
# Unified Provider Gateway (DRY: reuse existing provider system)
from .swarm_provider_gateway import SwarmProviderGateway
# State Management (V1 capabilities integrated)
from .swarm_state_manager import SwarmStateManager
# V2 Learning Pipeline (extracted to SwarmLearningPipeline)
from .learning_pipeline import SwarmLearningPipeline
from ...learning.predictive_marl import ActualTrajectory
from ...agents.feedback_channel import FeedbackChannel, FeedbackMessage, FeedbackType
from .swarm_intelligence import SyntheticTask, CurriculumGenerator
# MAS Learning - Persistent learning across sessions
from .mas_learning import MASLearning, get_mas_learning

# Skill Provider System (V2 World-Class) - Lazy imported to avoid circular deps
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
        from ...skills.providers import ProviderRegistry as PR, SkillCategory as SC
        from ...skills.providers.browser_use_provider import BrowserUseProvider as BUP
        from ...skills.providers.openhands_provider import OpenHandsProvider as OHP
        from ...skills.providers.agent_s_provider import AgentSProvider as ASP
        from ...skills.providers.open_interpreter_provider import OpenInterpreterProvider as OIP
        from ...skills.providers.composite_provider import (
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

logger = logging.getLogger(__name__)


class SwarmManager:
    """
    SwarmManager - Unified Orchestrator with User-Friendly Names
    
    Manages a swarm of agents with unified orchestration.
    Handles both single-agent and multi-agent cases.
    Uses intuitive component names (SwarmTaskBoard, SwarmPlanner, SwarmMemory).
    
    Skills Integration:
    - AutoAgent (used in AgentRunner) automatically discovers and executes skills
    - No manual skill-to-agent conversion needed for basic usage
    - Skills work seamlessly through AutoAgent
    
    AGUI Integration:
    - AGUI Component Registry for UI generation
    - Agents can generate UI components using registered adapters
    - Supports A2UI and AGUI protocols
    
    Feature Components (Swarm-level):
    - SwarmUIRegistry: UI component registry for agent-generated UI
    - SwarmProfiler: Performance profiler for identifying bottlenecks
    - SwarmToolValidator: Tool validation before registration
    - SwarmToolRegistry: Tool registration and discovery
    
    Autonomous Components (Zero-Config):
    - SwarmIntentParser: Natural language → AgentConfig conversion
    - SwarmResearcher: Autonomous research (APIs, tools, solutions)
    - SwarmInstaller: Auto-install dependencies
    - SwarmConfigurator: Smart configuration management
    - SwarmCodeGenerator: Glue code and integration code generation
    - SwarmWorkflowLearner: Pattern learning and reuse
    - SwarmIntegrator: Scheduling, monitoring, notifications
    
    Unified Provider Gateway:
    - SwarmProviderGateway: Unified provider gateway (reuses UnifiedLMProvider)
    - Auto-configures DSPy with best available provider
    - Supports all providers: OpenCode, OpenRouter, Claude CLI, Cursor CLI, Anthropic, OpenAI, Google, Groq
    """
    
    def __init__(
        self,
        agents: Optional[Union[AgentConfig, List[AgentConfig], str]] = None,
        config: Optional[JottyConfig] = None,
        architect_prompts: Optional[List[str]] = None,
        auditor_prompts: Optional[List[str]] = None,
        enable_zero_config: bool = True,
        enable_lotus: bool = True,  # LOTUS optimization (cascade, cache, adaptive validation)
    ):
        """
        Initialize SwarmManager.

        Args:
            agents: Single AgentConfig, list of AgentConfigs, or natural language string (zero-config)
            config: JottyConfig (defaults if None)
            architect_prompts: Architect prompt paths (optional)
            auditor_prompts: Auditor prompt paths (optional)
            enable_zero_config: Enable zero-config mode (natural language → AgentConfig)
            enable_lotus: Enable LOTUS optimization (model cascade, semantic cache, adaptive validation)
        """
        self.config = config or JottyConfig()
        self.enable_zero_config = enable_zero_config
        
        # Initialize core components first (needed for zero-config)
        self.swarm_task_board = SwarmTaskBoard()
        self.swarm_planner = SwarmPlanner()
        self.swarm_memory = SwarmMemory(
            config=self.config,
            agent_name="SwarmShared"
        )
        
        # Initialize intent parser early (needed for zero-config conversion)
        self.swarm_intent_parser = IntentParser(planner=self.swarm_planner)
        
        # Unified Provider Gateway (DRY: reuse existing UnifiedLMProvider)
        provider_preference = getattr(self.config, 'provider', None)
        self.swarm_provider_gateway = SwarmProviderGateway(
            config=self.config,
            provider=provider_preference
        )
        
        # Zero-config: Intelligently decide single vs multi-agent
        if isinstance(agents, str) and enable_zero_config:
            logger.info(f"🔮 Zero-config mode: Analyzing task for agent configuration")
            agents = self._create_zero_config_agents(agents)

        # Normalize agents to list
        if agents is None:
            # Default: Single AutoAgent
            agents = [AgentConfig(name="auto", agent=AutoAgent())]
        elif isinstance(agents, AgentConfig):
            agents = [agents]

        self.agents = agents
        self.mode = "multi" if len(agents) > 1 else "single"
        logger.info(f"🤖 Agent mode: {self.mode} ({len(self.agents)} agents)")
        
        # Default prompts - use evolved prompts from configs/prompts/ directory
        self.architect_prompts = architect_prompts or ["configs/prompts/architect/base_architect.md"]
        self.auditor_prompts = auditor_prompts or ["configs/prompts/auditor/base_auditor.md"]
        
        # Swarm-level Feature Components (consistent naming)
        # UI Component Registry
        self.swarm_ui_registry = get_agui_registry()
        logger.info(f"🎨 SwarmUIRegistry initialized - {len(self.swarm_ui_registry.list_section_types())} components available")
        
        # Performance Profiler
        enable_profiling = getattr(self.config, 'enable_profiling', False)
        self.swarm_profiler = SwarmProfiler(enable_cprofile=enable_profiling) if enable_profiling else None
        if self.swarm_profiler:
            logger.info("⏱️  SwarmProfiler enabled")
        
        # Tool Validation & Registry
        self.swarm_tool_validator = SwarmToolValidator()
        self.swarm_tool_registry = get_tools_registry()
        logger.info("✅ SwarmToolValidator initialized")
        
        # Autonomous Components (DRY: reuse existing, logical naming)
        self.swarm_researcher = SwarmResearcher(config=self.config)
        self.swarm_installer = SwarmInstaller(config=self.config)
        self.swarm_configurator = SwarmConfigurator(config=self.config)
        self.swarm_code_generator = SwarmCodeGenerator(config=self.config)
        self.swarm_workflow_learner = SwarmWorkflowLearner(swarm_memory=self.swarm_memory)
        self.swarm_integrator = SwarmIntegrator(config=self.config)

        # Intelligent Terminal (auto-fix, web search, skill generation)
        self.swarm_terminal = SwarmTerminal(
            config=self.config,
            auto_fix=True,
            max_fix_attempts=3
        )
        logger.info("🤖 Autonomous components initialized (including SwarmTerminal)")

        # MAS Learning initialization moved after _init_learning_pipeline() for DRY delegation
        
        # Initialize shared context and supporting components for state management
        from ...persistence.shared_context import SharedContext
        self.shared_context = SharedContext()
        
        # Initialize IOManager for output tracking (if not already present)
        from ...data.io_manager import IOManager
        self.io_manager = IOManager()
        
        # Initialize DataRegistry for artifact registration (if not already present)
        from ...data.data_registry import DataRegistry
        self.data_registry = DataRegistry()
        
        # Initialize ContextGuard for context management (if not already present)
        from ...context.context_guard import SmartContextGuard
        self.context_guard = SmartContextGuard()
        
        # Convert agents list to dict for state manager
        agents_dict = {agent.name: agent for agent in self.agents}
        
        # Initialize SwarmStateManager (V1 state management integrated)
        self.swarm_state_manager = SwarmStateManager(
            swarm_task_board=self.swarm_task_board,
            swarm_memory=self.swarm_memory,
            io_manager=self.io_manager,
            data_registry=self.data_registry,
            shared_context=self.shared_context,
            context_guard=self.context_guard,
            config=self.config,
            agents=agents_dict,
            agent_signatures={}  # Will be populated during execution
        )
        logger.info("📊 SwarmStateManager initialized (swarm + agent-level tracking)")

        # Learning Pipeline (extracted from SwarmManager)
        self.learning = SwarmLearningPipeline(self.config)

        # Expose learning components for backward compat (used by AgentRunner, callers, etc.)
        self.learning_manager = self.learning.learning_manager
        self.transfer_learning = self.learning.transfer_learning
        self.swarm_intelligence = self.learning.swarm_intelligence
        self.credit_weights = self.learning.credit_weights
        self.trajectory_predictor = self.learning.trajectory_predictor
        self.divergence_memory = self.learning.divergence_memory
        self.cooperative_credit = self.learning.cooperative_credit
        self.brain_state = self.learning.brain_state
        self.agent_abstractor = self.learning.agent_abstractor
        self.swarm_learner = self.learning.swarm_learner
        self.agent_slack = self.learning.agent_slack
        self.feedback_channel = self.learning.feedback_channel
        self.episode_count = 0
        logger.info("Learning pipeline initialized (SwarmLearningPipeline)")

        # Skill Provider Registry (lazy loaded)
        self.provider_registry = None
        if _load_providers():
            self._init_provider_registry()
        # Caching for autonomous_setup
        self._setup_cache = {}

        # MAS Learning - DRY: delegates to existing components
        workspace_path = getattr(self.config, 'base_path', None)
        self.mas_learning = MASLearning(
            config=self.config,
            workspace_path=workspace_path,
            swarm_intelligence=self.swarm_intelligence,
            learning_manager=self.learning_manager,
            transfer_learning=self.transfer_learning,
        )
        self.mas_learning.integrate_with_terminal(self.swarm_terminal)
        logger.info("🧠 MASLearning initialized")

        # Auto-load previous learnings
        self.learning.auto_load()

        # Create AgentRunners for each agent
        self.runners: Dict[str, AgentRunner] = {}
        for agent_config in self.agents:
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
                swarm_terminal=self.swarm_terminal,  # Shared intelligent terminal
                swarm_intelligence=self.swarm_intelligence,  # Curriculum feedback (Agent0)
            )

            self.runners[agent_config.name] = runner

        # Register agents with Axon (SmartAgentSlack) for inter-agent communication
        self._register_agents_with_axon()

        # LOTUS Optimization Layer (cascade, cache, adaptive validation)
        self.enable_lotus = enable_lotus
        self.lotus = None
        self.lotus_optimizer = None
        if enable_lotus:
            self._init_lotus_optimization()

        # Extracted components (lazy, lightweight references to self)
        from .zero_config_factory import ZeroConfigAgentFactory
        from .swarm_warmup import SwarmWarmup
        from .swarm_dag_executor import SwarmDAGExecutor
        self._zero_config_factory = ZeroConfigAgentFactory()
        self._warmup = SwarmWarmup(self)
        self._dag_executor = SwarmDAGExecutor(self)

        logger.info(f"SwarmManager initialized: {self.mode} mode, {len(self.agents)} agents")
    
    # _init_learning_pipeline removed — now handled by SwarmLearningPipeline

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
            from ...lotus.integration import LotusEnhancement, _enhance_agent_runner

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

    def _init_provider_registry(self):
        """Initialize the skill provider registry with all available providers."""
        global ProviderRegistry, BrowserUseProvider, OpenHandsProvider
        global AgentSProvider, OpenInterpreterProvider
        global ResearchAndAnalyzeProvider, AutomateWorkflowProvider, FullStackAgentProvider

        if not ProviderRegistry:
            logger.warning("Skill providers not available")
            return

        try:
            # Create registry with swarm intelligence for learning
            self.provider_registry = ProviderRegistry(
                swarm_intelligence=self.swarm_intelligence
            )

            # Register external providers
            providers_to_register = [
                BrowserUseProvider({'headless': True}),
                OpenHandsProvider({'sandbox': True}),
                AgentSProvider({'safe_mode': True}),
                OpenInterpreterProvider({'auto_run': True}),
            ]

            for provider in providers_to_register:
                try:
                    self.provider_registry.register(provider)
                except Exception as e:
                    logger.debug(f"Could not register {provider.name}: {e}")

            # Register composite providers
            composite_providers = [
                ResearchAndAnalyzeProvider(),
                AutomateWorkflowProvider(),
                FullStackAgentProvider(),
            ]

            for provider in composite_providers:
                try:
                    provider.set_registry(self.provider_registry)
                    if hasattr(provider, 'set_swarm_intelligence'):
                        provider.set_swarm_intelligence(self.swarm_intelligence)
                    self.provider_registry.register(provider)
                except Exception as e:
                    logger.debug(f"Could not register composite {provider.name}: {e}")

            # Load learned provider preferences
            provider_path = self._get_provider_registry_path()
            if provider_path.exists():
                self.provider_registry.load_state(str(provider_path))
                logger.info(f"Loaded provider learnings from {provider_path}")

            logger.info(f"✅ Provider registry initialized: {list(self.provider_registry.get_all_providers().keys())}")

        except Exception as e:
            logger.error(f"Failed to initialize provider registry: {e}")
            self.provider_registry = None

    def _get_provider_registry_path(self) -> Path:
        """Get path for provider registry persistence."""
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'provider_learnings.json'
        return Path.home() / '.jotty' / 'provider_learnings.json'

    async def execute_with_provider(
        self,
        category: str,
        task: str,
        context: Dict[str, Any] = None,
        provider_name: str = None
    ):
        """
        Execute a task using the skill provider system.

        Args:
            category: Skill category (browser, terminal, computer_use, etc.)
            task: Task description in natural language
            context: Additional context
            provider_name: Optional specific provider to use

        Returns:
            ProviderResult with execution output
        """
        global SkillCategory

        if not self.provider_registry:
            logger.warning("Provider registry not available")
            return None

        try:
            # Convert string category to enum
            cat_enum = SkillCategory(category) if isinstance(category, str) and SkillCategory else category

            # Execute via registry (uses learned selection)
            result = await self.provider_registry.execute(
                category=cat_enum,
                task=task,
                context=context,
                provider_name=provider_name,
            )

            # Record in swarm intelligence
            if result.success:
                self.swarm_intelligence.record_task_result(
                    agent_name=result.provider_name,
                    task_type=category,
                    success=result.success,
                    execution_time=result.execution_time,
                )

            return result

        except Exception as e:
            logger.error(f"Provider execution error: {e}")
            return None

    def get_provider_summary(self) -> Dict[str, Any]:
        """Get summary of provider registry state."""
        if not self.provider_registry:
            return {'available': False}

        return {
            'available': True,
            **self.provider_registry.get_registry_summary(),
        }

    # =========================================================================
    # Intelligent Terminal Methods (SwarmTerminal)
    # =========================================================================

    # terminal_* methods removed — use self.swarm_terminal directly

    # =========================================================================
    # Learning delegation (SwarmLearningPipeline handles persistence & hooks)
    # =========================================================================

    def _auto_load_learnings(self):
        """Delegate to SwarmLearningPipeline."""
        self.learning.auto_load()
        # Sync credit_weights reference after load (may have been replaced)
        self.credit_weights = self.learning.credit_weights
        # Log MAS stats
        if hasattr(self, 'mas_learning') and self.mas_learning:
            try:
                stats = self.mas_learning.get_statistics()
                logger.info(f"MAS Learning ready: {stats['fix_database']['total_fixes']} fixes, "
                           f"{stats['sessions']['total_sessions']} sessions")
            except Exception:
                pass

    def _auto_save_learnings(self):
        """Delegate to SwarmLearningPipeline."""
        self.learning.auto_save(
            mas_learning=getattr(self, 'mas_learning', None),
            swarm_terminal=getattr(self, 'swarm_terminal', None),
            provider_registry=getattr(self, 'provider_registry', None),
        )
        # Save HierarchicalMemory persistence
        if hasattr(self, 'memory_persistence') and self.memory_persistence:
            try:
                self.memory_persistence.save()
            except Exception as e:
                logger.debug(f"Could not auto-save memory: {e}")

    def load_relevant_learnings(self, task_description: str, agent_types: List[str] = None) -> Dict[str, Any]:
        """Load learnings relevant to the current task."""
        if not hasattr(self, 'mas_learning') or not self.mas_learning:
            return {}
        return self.mas_learning.load_relevant_learnings(
            task_description=task_description,
            agent_types=agent_types or [a.name for a in self.agents],
        )

    def record_agent_result(self, agent_name: str, task_type: str, success: bool,
                            time_taken: float, output_quality: float = 0.0):
        """Record an agent's task result for learning."""
        if hasattr(self, 'mas_learning') and self.mas_learning:
            self.mas_learning.record_agent_task(
                agent_type=agent_name, task_type=task_type,
                success=success, time_taken=time_taken, output_quality=output_quality,
            )

    def record_session_result(self, task_description: str,
                              agent_performances: Dict[str, Dict[str, Any]],
                              total_time: float, success: bool,
                              fixes_applied: List[Dict[str, Any]] = None,
                              stigmergy_signals: int = 0):
        """Record session results for future learning."""
        if hasattr(self, 'mas_learning') and self.mas_learning:
            self.mas_learning.record_session(
                task_description=task_description, agent_performances=agent_performances,
                fixes_applied=fixes_applied or [], stigmergy_signals=stigmergy_signals,
                total_time=total_time, success=success,
            )

    def get_transferable_context(self, query: str, agent: str = None) -> str:
        """Get transferable learnings as context for an agent."""
        return self.learning.get_transferable_context(query, agent)

    def get_swarm_wisdom(self, query: str) -> str:
        """Get collective swarm wisdom for a task."""
        return self.learning.get_swarm_wisdom(query)

    def get_agent_specializations(self) -> Dict[str, str]:
        """Get current specializations of all agents."""
        return self.learning.get_agent_specializations()

    def get_best_agent_for_task(self, query: str) -> Optional[str]:
        """Recommend the best agent for a task."""
        return self.learning.get_best_agent_for_task(query)

    def _register_agents_with_axon(self):
        """Register all agents with SmartAgentSlack for inter-agent messaging."""
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
            skip_autonomous_setup: If True, skip research/install/configure (fast mode)
            status_callback: Optional callback(stage, detail) for progress updates
            ensemble: Enable prompt ensembling for multi-perspective analysis
            ensemble_strategy: Strategy for ensembling
            **kwargs: Additional arguments

        Returns:
            EpisodeResult with output and metadata
        """
        # Extract special kwargs
        skip_autonomous_setup = kwargs.pop('skip_autonomous_setup', False)
        status_callback = kwargs.pop('status_callback', None)
        ensemble = kwargs.pop('ensemble', None)  # None = auto-detect, True/False = explicit
        ensemble_strategy = kwargs.pop('ensemble_strategy', 'multi_perspective')

        # Auto-detect ensemble for certain task types (if not explicitly set)
        if ensemble is None:
            ensemble = self._should_auto_ensemble(goal)
            if ensemble:
                logger.info(f"🔮 Auto-enabled ensemble for task type (use ensemble=False to override)")

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
                _status("Ensembling", f"strategy={ensemble_strategy} (swarm-level)")
                ensemble_result = await self._execute_ensemble(
                    goal,
                    strategy=ensemble_strategy,
                    status_callback=status_callback
                )
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

            # Single-agent mode: Simple execution
            if self.mode == "single":
                agent_name = self.agents[0].name if self.agents else "auto"
                _status("Executing", f"agent '{agent_name}' with skill orchestration")
                # Architect → Actor → Auditor pipeline (now fast with max_eval_iters=2)
                # skip_validation only when explicitly requested via skip_autonomous_setup
                skip_val = skip_autonomous_setup
                result = await self._execute_single_agent(
                    goal,
                    skip_validation=skip_val,
                    status_callback=status_callback,
                    ensemble_context=ensemble_result if ensemble else None,
                    **kwargs
                )

                _status("Complete", "success" if result.success else "failed")

                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)

                return result

            # Multi-agent mode: Use SwarmTaskBoard for coordination
            else:
                _status("Executing", f"{len(self.agents)} agents in parallel")
                result = await self._execute_multi_agent(
                    goal,
                    ensemble_context=ensemble_result if ensemble else None,
                    status_callback=status_callback,
                    ensemble=ensemble,  # Pass to agents for per-agent ensemble
                    ensemble_strategy=ensemble_strategy,
                    **kwargs
                )

                _status("Complete", "success" if result.success else "failed")

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
    
    async def _execute_ensemble(
        self,
        goal: str,
        strategy: str = 'multi_perspective',
        status_callback=None
    ) -> Dict[str, Any]:
        """
        Execute prompt ensembling for multi-perspective analysis.

        Uses the ensemble_prompt_tool from claude-cli-llm skill.

        Strategies:
        - self_consistency: Same prompt, N samples, synthesis
        - multi_perspective: Different expert personas (default)
        - gsa: Generative Self-Aggregation
        - debate: Multi-round argumentation

        Args:
            goal: The task/question to analyze
            strategy: Ensembling strategy
            status_callback: Optional progress callback

        Returns:
            Dict with ensemble results including synthesized response
        """
        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass

        try:
            # Try to use the skill
            try:
                from ...registry.skills_registry import get_skills_registry
                registry = get_skills_registry()
                registry.init()
                skill = registry.get_skill('claude-cli-llm')

                if skill:
                    ensemble_tool = skill.tools.get('ensemble_prompt_tool')
                    if ensemble_tool:
                        _status("Ensemble", f"using {strategy} strategy (domain-aware)")
                        result = ensemble_tool({
                            'prompt': goal,
                            'strategy': strategy,
                            'synthesis_style': 'structured',
                            'verbose': True  # Get quality scores and summaries
                        })
                        # Show individual perspective status
                        if result.get('success') and result.get('quality_scores'):
                            for name, score in result['quality_scores'].items():
                                _status(f"  {name}", f"quality={score:.0%}")
                        return result
            except ImportError:
                pass

            # Fallback: Use DSPy directly for multi-perspective
            import dspy
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                return {'success': False, 'error': 'No LLM configured'}

            lm = dspy.settings.lm

            # Simple multi-perspective implementation
            perspectives = [
                ("analytical", "Analyze this from a data-driven, logical perspective:"),
                ("creative", "Consider unconventional angles and innovative solutions:"),
                ("critical", "Play devil's advocate - identify risks and problems:"),
                ("practical", "Focus on feasibility and actionable steps:"),
            ]

            responses = {}
            for name, prefix in perspectives:
                _status(f"  {name}", "analyzing...")
                try:
                    prompt = f"{prefix}\n\n{goal}"
                    response = lm(prompt=prompt)
                    text = response[0] if isinstance(response, list) else str(response)
                    responses[name] = text
                except Exception as e:
                    logger.warning(f"Perspective '{name}' failed: {e}")

            if not responses:
                return {'success': False, 'error': 'All perspectives failed'}

            # Synthesize
            _status("Synthesizing", f"{len(responses)} perspectives")
            synthesis_prompt = f"""Synthesize these {len(responses)} expert perspectives into a comprehensive analysis:

Question: {goal}

{chr(10).join(f'**{k.upper()}:** {v[:600]}' for k, v in responses.items())}

Provide a structured synthesis with:
1. **Consensus**: Where perspectives agree
2. **Tensions**: Where they diverge
3. **Blind Spots**: Unique insights from each
4. **Recommendation**: Balanced conclusion"""

            synthesis = lm(prompt=synthesis_prompt)
            final_response = synthesis[0] if isinstance(synthesis, list) else str(synthesis)

            return {
                'success': True,
                'response': final_response,
                'perspectives_used': list(responses.keys()),
                'individual_responses': responses,
                'strategy': strategy,
                'confidence': len(responses) / len(perspectives)
            }

        except Exception as e:
            logger.error(f"Ensemble execution failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def _execute_single_agent(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Execute single-agent mode.

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
        Execute multi-agent mode with sequential execution based on agent order.

        Agents are executed in order (research → generate → notify) with each
        agent's output passed to the next. Uses retry on failure.
        """
        # Extract callbacks and ensemble params before passing to runners
        kwargs.pop('ensemble_context', None)
        status_callback = kwargs.pop('status_callback', None)
        ensemble = kwargs.pop('ensemble', False)
        ensemble_strategy = kwargs.pop('ensemble_strategy', 'multi_perspective')

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
            async def _run_task(task):
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
                            # Prefix with agent name so user knows which agent is doing what
                            status_callback(f"  [{task.actor}] {stage}", detail)
                        except Exception:
                            pass

                runner = self.runners[task.actor]
                # Pass the agent-specific callback and ensemble params
                task_kwargs = dict(kwargs)
                task_kwargs['status_callback'] = agent_status_callback
                # Each agent gets its own ensemble for its specific sub-goal
                if ensemble:
                    task_kwargs['ensemble'] = True
                    task_kwargs['ensemble_strategy'] = ensemble_strategy
                return task, await runner.run(goal=task.description, **task_kwargs)

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

        # Cooperative credit assignment
        self._assign_cooperative_credit(all_results, goal)

        # Post-episode learning
        combined_result = self._aggregate_results(all_results, goal)
        self._post_episode_learning(combined_result, goal)

        # Auto-save learnings (persist across sessions)
        self._auto_save_learnings()

        return combined_result

    def _aggregate_results(self, results: Dict[str, EpisodeResult], goal: str) -> EpisodeResult:
        """Combine all agent outputs into a single EpisodeResult."""
        if not results:
            return EpisodeResult(
                output=None,
                success=False,
                agent_name="swarm_manager",
                error="No tasks executed"
            )

        if len(results) == 1:
            return list(results.values())[0]

        # Combine outputs
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
        return self._zero_config_factory.create_agents(task, status_callback)

    def _should_auto_ensemble(self, goal: str) -> bool:
        """Delegate to swarm_ensemble module."""
        from .swarm_ensemble import should_auto_ensemble
        return should_auto_ensemble(goal)

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
        return await self._warmup.warmup(**kwargs)

    def get_warmup_recommendation(self) -> Dict[str, Any]:
        """Check if warmup would be beneficial."""
        return self._warmup.get_recommendation()

    # =====================================================================
    # DAG — delegated to SwarmDAGExecutor
    # =====================================================================

    async def run_with_dag(self, implementation_plan: str, **kwargs) -> EpisodeResult:
        """Execute via DAG-based orchestration. See SwarmDAGExecutor."""
        return await self._dag_executor.run(implementation_plan, **kwargs)

    def get_dag_agents(self):
        """Get DAG agents for external use."""
        return self._dag_executor.get_agents()
