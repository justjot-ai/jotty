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
# Import directly from source to avoid circular import
from ..roadmap import MarkovianTODO as SwarmTaskBoard
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
# Unified Provider Gateway (DRY: reuse existing provider system)
from .swarm_provider_gateway import SwarmProviderGateway
# State Management (V1 capabilities integrated)
from .swarm_state_manager import SwarmStateManager
# V1 Learning Pipeline (restored in V2)
from ..managers.learning_manager import LearningManager
from ...learning.predictive_marl import (
    LLMTrajectoryPredictor, DivergenceMemory,
    CooperativeCreditAssigner, ActualTrajectory
)
from ...memory.consolidation_engine import (
    BrainStateMachine, BrainModeConfig, AgentAbstractor
)
from ...agents.axon import SmartAgentSlack
from ...agents.feedback_channel import FeedbackChannel, FeedbackMessage, FeedbackType
from ..conductor import SwarmLearner
from ...learning.transfer_learning import TransferableLearningStore
from .swarm_intelligence import SwarmIntelligence
from ...foundation.robust_parsing import AdaptiveWeightGroup

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
    ):
        """
        Initialize SwarmManager.
        
        Args:
            agents: Single AgentConfig, list of AgentConfigs, or natural language string (zero-config)
            config: JottyConfig (defaults if None)
            architect_prompts: Architect prompt paths (optional)
            auditor_prompts: Auditor prompt paths (optional)
            enable_zero_config: Enable zero-config mode (natural language → AgentConfig)
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
        
        # Zero-config: Convert natural language to AgentConfig
        if isinstance(agents, str) and enable_zero_config:
            logger.info(f"🔮 Zero-config mode: Converting '{agents[:50]}...' to AgentConfig")
            agents = self.parse_intent_to_agent_config(agents)
        
        # Normalize agents to list
        if agents is None:
            # Default: AutoAgent for zero-config
            agents = AgentConfig(name="auto", agent=AutoAgent())
        
        if isinstance(agents, AgentConfig):
            self.agents = [agents]
            self.mode = "single"
        else:
            self.agents = agents
            self.mode = "multi" if len(agents) > 1 else "single"
        
        # Default prompts if not provided
        self.architect_prompts = architect_prompts or ["prompts/architect.md"]
        self.auditor_prompts = auditor_prompts or ["prompts/auditor.md"]
        
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
        logger.info("🤖 Autonomous components initialized")
        
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

        # Initialize V1 learning pipeline
        self._init_learning_pipeline()

        # Auto-load previous learnings (makes it truly self-learning across sessions)
        self._auto_load_learnings()

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
                transfer_learning=self.transfer_learning
            )

            self.runners[agent_config.name] = runner

        # Register agents with Axon (SmartAgentSlack) for inter-agent communication
        self._register_agents_with_axon()

        logger.info(f"SwarmManager initialized: {self.mode} mode, {len(self.agents)} agents")
        if self.swarm_profiler:
            logger.info("   ⏱️  SwarmProfiler enabled")
        logger.info("   ✅ SwarmToolValidator initialized")
        logger.info("   🎨 SwarmUIRegistry initialized")
        logger.info("   🌐 SwarmProviderGateway configured (unified provider system)")
        logger.info("   🤖 Autonomous components ready (zero-config enabled)")
        logger.info("   📊 SwarmStateManager initialized (swarm + agent-level tracking)")
    
    def _init_learning_pipeline(self):
        """Initialize V1 learning pipeline components for swarm-level learning."""
        # Core learning manager (wraps Q-learner)
        self.learning_manager = LearningManager(self.config)

        # Trajectory prediction (MARL)
        self.trajectory_predictor = None
        try:
            self.trajectory_predictor = LLMTrajectoryPredictor(self.config, horizon=5)
        except Exception as e:
            logger.warning(f"Trajectory predictor unavailable: {e}")

        # Divergence memory for storing prediction errors
        self.divergence_memory = DivergenceMemory(self.config)

        # Cooperative credit assignment
        self.cooperative_credit = CooperativeCreditAssigner(self.config)

        # Brain state machine for consolidation
        brain_config = BrainModeConfig()
        self.brain_state = BrainStateMachine(brain_config)

        # Agent abstractor for scalable role tracking
        self.agent_abstractor = AgentAbstractor(brain_config)

        # Inter-agent communication
        self.agent_slack = SmartAgentSlack(enable_cooperation=True)
        self.feedback_channel = FeedbackChannel()

        # Swarm learner for prompt evolution
        self.swarm_learner = SwarmLearner(self.config)

        # Transferable learning (cross-swarm, cross-goal)
        self.transfer_learning = TransferableLearningStore(self.config)

        # Swarm intelligence (emergent specialization, consensus, routing)
        self.swarm_intelligence = SwarmIntelligence(self.config)

        # A-Team v8.0: Adaptive credit assignment weights (replaces hardcoded 0.3/0.4/0.3)
        self.credit_weights = AdaptiveWeightGroup({
            'base_reward': 0.3,
            'cooperation_bonus': 0.4,
            'predictability_bonus': 0.3
        })

        # A-Team v8.0: Skill Provider Registry (browser-use, openhands, agent-s, etc.)
        self.provider_registry = None
        if _load_providers():
            self._init_provider_registry()

        # Caching for autonomous_setup
        self._setup_cache = {}

        # Episode counter
        self.episode_count = 0

        logger.info("V1 learning pipeline initialized")

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

    def _get_learning_path(self) -> Path:
        """Get default path for learning persistence."""
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'swarm_learnings.json'
        return Path.home() / '.jotty' / 'swarm_learnings.json'

    def _get_transfer_learning_path(self) -> Path:
        """Get path for transferable learning persistence."""
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'transfer_learnings.json'
        return Path.home() / '.jotty' / 'transfer_learnings.json'

    def _get_swarm_intelligence_path(self) -> Path:
        """Get path for swarm intelligence persistence."""
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'swarm_intelligence.json'
        return Path.home() / '.jotty' / 'swarm_intelligence.json'

    def _get_credit_weights_path(self) -> Path:
        """Get path for adaptive credit weights persistence."""
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'credit_weights.json'
        return Path.home() / '.jotty' / 'credit_weights.json'

    def _auto_load_learnings(self):
        """Auto-load previous learnings at startup (makes swarm truly self-learning)."""
        # Load Q-learner state
        learning_path = self._get_learning_path()
        if learning_path.exists():
            try:
                self.learning_manager.q_learner.load_state(str(learning_path))
                q_summary = self.learning_manager.get_q_table_summary()
                logger.info(f"Auto-loaded {q_summary['size']} Q-entries from {learning_path}")
            except Exception as e:
                logger.debug(f"Could not auto-load Q-learnings: {e}")

        # Load transferable learnings (cross-swarm, cross-goal)
        transfer_path = self._get_transfer_learning_path()
        if self.transfer_learning.load(str(transfer_path)):
            logger.info(f"Auto-loaded transferable learnings from {transfer_path}")

        # Load swarm intelligence (specializations, routing)
        si_path = self._get_swarm_intelligence_path()
        if self.swarm_intelligence.load(str(si_path)):
            specs = self.swarm_intelligence.get_specialization_summary()
            logger.info(f"Auto-loaded swarm intelligence: {len(specs)} agent profiles")

        # Load adaptive credit weights (A-Team v8.0: real learned weights)
        credit_path = self._get_credit_weights_path()
        if credit_path.exists():
            try:
                import json
                with open(credit_path, 'r') as f:
                    credit_data = json.load(f)
                self.credit_weights = AdaptiveWeightGroup.from_dict(credit_data)
                logger.info(f"Auto-loaded credit weights: {self.credit_weights}")
            except Exception as e:
                logger.debug(f"Could not auto-load credit weights: {e}")

    def _auto_save_learnings(self):
        """Auto-save learnings after execution (persists across sessions)."""
        # Save Q-learner state
        learning_path = self._get_learning_path()
        try:
            learning_path.parent.mkdir(parents=True, exist_ok=True)
            self.learning_manager.q_learner.save_state(str(learning_path))
        except Exception as e:
            logger.debug(f"Could not auto-save Q-learnings: {e}")

        # Save transferable learnings
        transfer_path = self._get_transfer_learning_path()
        try:
            self.transfer_learning.save(str(transfer_path))
        except Exception as e:
            logger.debug(f"Could not auto-save transfer learnings: {e}")

        # Save swarm intelligence
        si_path = self._get_swarm_intelligence_path()
        try:
            self.swarm_intelligence.save(str(si_path))
        except Exception as e:
            logger.debug(f"Could not auto-save swarm intelligence: {e}")

        # Save adaptive credit weights (A-Team v8.0: real learned weights)
        credit_path = self._get_credit_weights_path()
        try:
            import json
            credit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(credit_path, 'w') as f:
                json.dump(self.credit_weights.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Could not auto-save credit weights: {e}")

        # Save provider registry learnings (A-Team v8.0: which provider works best)
        if self.provider_registry:
            provider_path = self._get_provider_registry_path()
            try:
                self.provider_registry.save_state(str(provider_path))
            except Exception as e:
                logger.debug(f"Could not auto-save provider learnings: {e}")

    def get_transferable_context(self, query: str, agent: str = None) -> str:
        """
        Get transferable learnings as context for an agent.

        This provides learnings that transfer across:
        - Different agent combinations
        - Different goals/queries (via semantic similarity)
        - Different domains (via abstract patterns)
        """
        return self.transfer_learning.format_context_for_agent(query, agent)

    def get_swarm_wisdom(self, query: str) -> str:
        """
        Get collective swarm wisdom for a task.

        Combines:
        - Transferable learnings (patterns, role advice)
        - Swarm intelligence (specializations, routing)
        """
        task_type = self.transfer_learning.extractor.extract_task_type(query)

        # Get transferable context
        transfer_ctx = self.transfer_learning.format_context_for_agent(query)

        # Get swarm intelligence context
        swarm_ctx = self.swarm_intelligence.format_swarm_context(query, task_type)

        return f"{transfer_ctx}\n\n{swarm_ctx}"

    def get_agent_specializations(self) -> Dict[str, str]:
        """Get current specializations of all agents."""
        return self.swarm_intelligence.get_specialization_summary()

    def get_best_agent_for_task(self, query: str) -> Optional[str]:
        """
        Recommend the best agent for a task using swarm intelligence.

        Uses:
        - Emergent specialization tracking
        - Historical success rates
        - Trust scores
        - Transfer learning role profiles
        """
        task_type = self.transfer_learning.extractor.extract_task_type(query)
        available = [a.name for a in self.agents]

        # Primary: Use swarm intelligence routing
        best = self.swarm_intelligence.get_best_agent_for_task(task_type, available)
        if best:
            return best

        # Fallback: Use transfer learning role profiles
        best_role = self.transfer_learning.get_best_role_for_task(task_type)
        if best_role:
            for agent_config in self.agents:
                agent_role = self.transfer_learning.extractor.extract_role(agent_config.name)
                if agent_role == best_role:
                    return agent_config.name

        return available[0] if available else None

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

    def list_capabilities(self) -> Dict[str, List[str]]:
        """
        List all available capabilities for easy discovery.
        
        Returns:
            Dictionary mapping capability categories to method names
        """
        return {
                "orchestration": [
                "run(goal) - Execute task with full autonomy",
                "autonomous_setup(goal) - Manual autonomous setup (research, install, configure)",
                "parse_intent_to_agent_config(natural_language) - Convert natural language to AgentConfig"
            ],
            "research": [
                "swarm_researcher.research() - Research solutions, APIs, tools",
                "swarm_researcher.find_solutions() - Find solutions for requirements"
            ],
            "installation": [
                "swarm_installer.install() - Install dependencies (pip, npm, skills)",
                "swarm_installer.is_installed() - Check if package is installed"
            ],
            "configuration": [
                "swarm_configurator.configure() - Configure services with minimal prompts",
                "swarm_configurator.get_config() - Get saved configuration",
                "swarm_configurator.set_config() - Manually set configuration"
            ],
            "code_generation": [
                "swarm_code_generator.generate_glue_code() - Generate glue code between tools",
                "swarm_code_generator.generate_integration_code() - Generate integration code"
            ],
            "learning": [
                "swarm_workflow_learner.learn_from_execution() - Learn from workflow execution",
                "swarm_workflow_learner.find_similar_pattern() - Find similar workflow patterns",
                "swarm_workflow_learner.get_best_patterns() - Get best patterns by success rate"
            ],
            "integration": [
                "swarm_integrator.setup_scheduling() - Set up scheduled execution (cron, systemd)",
                "swarm_integrator.setup_monitoring() - Set up monitoring",
                "swarm_integrator.setup_notifications() - Set up error notifications"
            ],
            "provider": [
                "swarm_provider_gateway.get_lm() - Get configured DSPy LM instance",
                "swarm_provider_gateway.configure_provider() - Configure specific provider",
                "swarm_provider_gateway.list_available_providers() - List all supported providers",
                "swarm_provider_gateway.is_provider_available() - Check provider availability"
            ],
            "task_management": [
                "swarm_task_board.add_task() - Add task to board",
                "swarm_task_board.get_next_task() - Get next task to execute",
                "swarm_task_board.complete_task() - Mark task as complete",
                "swarm_task_board.fail_task() - Mark task as failed"
            ],
            "planning": [
                "swarm_planner.plan_execution() - Plan execution steps",
                "swarm_planner.infer_task_type() - Infer task type from description"
            ],
            "memory": [
                "swarm_memory.store() - Store information",
                "swarm_memory.retrieve() - Retrieve information"
            ],
        }
    
    def get_help(self, component_name: Optional[str] = None) -> str:
        """
        Get help information for a component or overall framework.
        
        Args:
            component_name: Optional component name (e.g., "SwarmResearcher")
            
        Returns:
            Help text string
        """
        if component_name:
            # Component-specific help
            help_texts = {
                "SwarmManager": """
SwarmManager - Main Orchestrator
--------------------------------
Manages swarm of agents with unified orchestration.

Key Methods:
- run(goal) - Execute task with full autonomy
- autonomous_setup(goal) - Manual autonomous setup
- parse_intent_to_agent_config(natural_language) - Convert natural language to AgentConfig
- list_capabilities() - List all available capabilities
- get_help(component) - Get help for component

Usage:
    swarm = SwarmManager(agents="Research topic")
    result = await swarm.run(goal="Research topic")
    
    # Manual setup
    await swarm.autonomous_setup("Set up Reddit scraping")
    
    # Parse intent manually
    agent_config = swarm.parse_intent_to_agent_config("Research AI startups")
""",
                "SwarmResearcher": """
SwarmResearcher - Autonomous Research
--------------------------------------
Researches solutions, APIs, tools, and best practices.

Key Methods:
- research(query, research_type) - Research a topic
- Returns: ResearchResult with findings, tools, APIs

Usage:
    result = await swarm.swarm_researcher.research("Reddit API")
    print(result.tools_found)  # ['praw', 'reddit-api']
""",
                "SwarmInstaller": """
SwarmInstaller - Dependency Installation
----------------------------------------
Automatically installs dependencies (pip, npm, skills).

Key Methods:
- install(package, package_type) - Install package
- is_installed(package) - Check if installed

Usage:
    result = await swarm.swarm_installer.install("praw")
    if result.success:
        print(f"Installed {result.package} via {result.method}")
""",
                "SwarmConfigurator": """
SwarmConfigurator - Smart Configuration
---------------------------------------
Handles configuration with minimal user prompts.

Key Methods:
- configure(service, config_template) - Configure service
- get_config(service) - Get saved configuration
- set_config(service, config) - Manually set configuration

Usage:
    result = await swarm.swarm_configurator.configure("reddit")
    if result.requires_user_input:
        # Prompt user for missing keys
        for prompt in result.prompts:
            print(prompt)
""",
                "SwarmCodeGenerator": """
SwarmCodeGenerator - Code Generation
-------------------------------------
Generates glue code and integration code.

Key Methods:
- generate_glue_code(source, destination) - Generate glue code
- generate_integration_code(service, operation) - Generate integration code

Usage:
    code = swarm.swarm_code_generator.generate_glue_code(
        source_tool="reddit_scraper",
        destination_tool="notion_client"
    )
    print(code.code)  # Generated Python code
""",
                "SwarmWorkflowLearner": """
SwarmWorkflowLearner - Pattern Learning
---------------------------------------
Learns workflow patterns and enables reuse.

Key Methods:
- learn_from_execution(...) - Learn from execution
- find_similar_pattern(...) - Find similar patterns
- get_best_patterns(limit) - Get best patterns

Usage:
    pattern = swarm.swarm_workflow_learner.find_similar_pattern(
        task_type="research",
        operations=["scrape", "summarize"],
        tools_available=["web-search", "pdf-generator"]
    )
""",
                "SwarmIntegrator": """
SwarmIntegrator - Integration Automation
----------------------------------------
Automates integration setup (scheduling, monitoring).

Key Methods:
- setup_scheduling(script, schedule) - Set up scheduling
- setup_monitoring(script) - Set up monitoring
- setup_notifications(channels, events) - Set up notifications

Usage:
    result = await swarm.swarm_integrator.setup_scheduling(
        script_path="/path/to/script.py",
        schedule="daily"
    )
""",
                "SwarmProviderGateway": """
SwarmProviderGateway - Provider Management
------------------------------------------
Unified gateway for all LLM providers.

Key Methods:
- get_lm() - Get configured DSPy LM
- configure_provider(provider, model) - Configure provider
- list_available_providers() - List all providers

Usage:
    providers = swarm.swarm_provider_gateway.list_available_providers()
    lm = swarm.swarm_provider_gateway.get_lm()
""",
            }
            
            return help_texts.get(component_name, f"No help available for {component_name}")
        
        # Overall framework help
        return """
Jotty V2 - True Agentic Assistant Framework
============================================

Quick Start:
    from core.orchestration.v2 import SwarmManager
    swarm = SwarmManager(agents="Research topic")
    result = await swarm.run(goal="Research topic")

Discover Capabilities:
    capabilities = swarm.list_capabilities()
    help_text = swarm.get_help("SwarmResearcher")

Components:
- SwarmManager: Main orchestrator
- SwarmResearcher: Research capability
- SwarmInstaller: Dependency installation
- SwarmConfigurator: Configuration management
- SwarmCodeGenerator: Code generation
- SwarmWorkflowLearner: Pattern learning
- SwarmIntegrator: Integration automation
- SwarmProviderGateway: Provider management

For component-specific help:
    swarm.get_help("ComponentName")
"""
    
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
        
        Args:
            goal: Task goal/description (natural language supported)
            **kwargs: Additional arguments
            
        Returns:
            EpisodeResult with output and metadata
        """
        logger.info(f"🚀 SwarmManager.run: {self.mode} mode - {goal[:50]}...")
        
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
                logger.info(f"✅ Stored 'goal' and 'query' in SharedContext: {goal[:100]}...")
            
            # Autonomous planning: Research, install, configure if needed
            await self.autonomous_setup(goal)
            
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
        
            # Single-agent mode: Simple execution
            if self.mode == "single":
                result = await self._execute_single_agent(goal, **kwargs)
                
                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)
                
                return result
        
            # Multi-agent mode: Use SwarmTaskBoard for coordination
            else:
                result = await self._execute_multi_agent(goal, **kwargs)
                
                # Exit profiling context
                if profile_context:
                    profile_context.__exit__(None, None, None)
                
                return result
        except Exception as e:
            # Exit profiling context on error
            if profile_context:
                profile_context.__exit__(type(e), e, None)
            raise
    
    async def _execute_single_agent(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Execute single-agent mode.
        
        Args:
            goal: Task goal
            **kwargs: Additional arguments
            
        Returns:
            EpisodeResult
        """
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
        Execute multi-agent mode with concurrent execution and retry.

        Groups tasks by dependency level (independent tasks run together),
        uses asyncio.gather for parallel execution, and retries failures
        with enriched context.
        """
        max_attempts = getattr(self.config, 'max_task_attempts', 2)

        # Add tasks to SwarmTaskBoard
        for i, agent_config in enumerate(self.agents):
            task_id = f"task_{i+1}"
            self.swarm_task_board.add_task(
                task_id=task_id,
                description=f"{goal} (agent: {agent_config.name})",
                actor=agent_config.name
            )

        all_results = {}  # agent_name -> EpisodeResult
        attempt_counts = {}  # task_id -> attempts

        while True:
            # Collect all ready tasks (no unresolved dependencies)
            batch = []
            while True:
                next_task = self.swarm_task_board.get_next_task()
                if next_task is None:
                    break
                batch.append(next_task)

            if not batch:
                break

            # Pre-execution: trajectory prediction for each task
            predictions = {}
            for task in batch:
                if self.trajectory_predictor:
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

            # Execute batch concurrently
            async def _run_task(task):
                runner = self.runners[task.actor]
                return task, await runner.run(goal=task.description, **kwargs)

            coro_results = await asyncio.gather(
                *[_run_task(t) for t in batch],
                return_exceptions=True
            )

            # Process results
            for coro_result in coro_results:
                if isinstance(coro_result, Exception):
                    logger.error(f"Task execution exception: {coro_result}")
                    continue

                task, result = coro_result
                attempt_counts[task.task_id] = attempt_counts.get(task.task_id, 0) + 1
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
    
    def _post_episode_learning(self, result: EpisodeResult, goal: str):
        """
        Post-episode learning: swarm learner, brain consolidation, NeuroChunk tiering.

        Called at end of both single-agent and multi-agent execution.
        """
        self.episode_count += 1

        # 1. SwarmLearner: record episode, conditionally update prompts
        try:
            trajectory = result.trajectory or []
            insights = []
            if hasattr(result, 'tagged_outputs') and result.tagged_outputs:
                insights = [str(t) for t in result.tagged_outputs[:5]]
            self.swarm_learner.record_episode(trajectory, result.success, insights)

            if self.swarm_learner.should_update_prompts():
                for prompt_path in self.architect_prompts:
                    try:
                        with open(prompt_path, 'r') as f:
                            current = f.read()
                        updated, changes = self.swarm_learner.update_prompt(prompt_path, current)
                        if changes:
                            logger.info(f"Prompt '{prompt_path}' evolved with {len(changes)} changes")
                    except Exception as e:
                        logger.debug(f"Prompt update skipped for {prompt_path}: {e}")
        except Exception as e:
            logger.debug(f"SwarmLearner recording skipped: {e}")

        # 2. Brain consolidation: process experience
        try:
            experience = {
                'content': str(result.output)[:500] if result.output else '',
                'context': {'goal': goal, 'episode': self.episode_count},
                'reward': 1.0 if result.success else 0.0,
                'agent': 'swarm',
            }
            # Use asyncio to run the async consolidation in a fire-and-forget manner
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.brain_state.process_experience(experience))
            else:
                loop.run_until_complete(self.brain_state.process_experience(experience))
        except Exception as e:
            logger.debug(f"Brain consolidation skipped: {e}")

        # 3. NeuroChunk tiering: promote/demote memories
        try:
            episode_reward = 1.0 if result.success else 0.0
            self.learning_manager.promote_demote_memories(episode_reward)
            self.learning_manager.prune_tier3()
        except Exception as e:
            logger.debug(f"NeuroChunk tiering skipped: {e}")

        # 4. Agent abstractor: update per-agent stats
        try:
            if hasattr(result, 'agent_contributions') and result.agent_contributions:
                for agent_name, contrib in result.agent_contributions.items():
                    success = getattr(contrib, 'decision_correct', result.success)
                    self.agent_abstractor.update_agent(agent_name, success)
            else:
                # Single agent case
                agent_name = getattr(result, 'agent_name', self.agents[0].name if self.agents else 'unknown')
                self.agent_abstractor.update_agent(agent_name, result.success)
        except Exception as e:
            logger.debug(f"Agent abstractor update skipped: {e}")

        # 5. Record into transferable learning store
        try:
            query = goal[:200] if goal else ''
            agent_name = getattr(result, 'agent_name', self.agents[0].name if self.agents else 'unknown')
            self.transfer_learning.record_experience(
                query=query,
                agent=agent_name,
                action=goal[:100],
                reward=episode_reward,
                success=result.success,
                error=str(getattr(result, 'error', None) or ''),
                context={'episode': self.episode_count}
            )
        except Exception as e:
            logger.debug(f"Transfer learning record skipped: {e}")

        # 6. Record into swarm intelligence (emergent specialization)
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(goal)
            execution_time = getattr(result, 'execution_time', 0.0)
            agent_name = getattr(result, 'agent_name', self.agents[0].name if self.agents else 'unknown')
            self.swarm_intelligence.record_task_result(
                agent_name=agent_name,
                task_type=task_type,
                success=result.success,
                execution_time=execution_time,
                context={'goal': goal[:100], 'episode': self.episode_count}
            )
        except Exception as e:
            logger.debug(f"Swarm intelligence record skipped: {e}")

        logger.debug(f"Post-episode learning complete (episode #{self.episode_count})")

    def _learn_from_result(self, result: EpisodeResult, agent_config: AgentConfig):
        """
        Learn from execution result (DRY: reuse workflow learner).
        
        Args:
            result: Execution result
            agent_config: Agent configuration
        """
        if not result.success:
            return
        
        metadata = getattr(agent_config, 'metadata', {}) or {}
        self.swarm_workflow_learner.learn_from_execution(
            task_type=metadata.get('task_type', 'unknown'),
            operations=metadata.get('operations', []),
            tools_used=metadata.get('integrations', []),
            success=True,
            execution_time=getattr(result, 'duration', 0.0),
            metadata={'agent': agent_config.name}
        )
    
    async def autonomous_setup(self, goal: str):
        """
        Autonomous setup: Research, install, configure.
        
        Public method for manual autonomous setup.
        DRY: Reuses all autonomous components.
        
        Args:
            goal: Task goal
            
        Example:
            await swarm.autonomous_setup("Set up Reddit scraping")
        """
        # Cache check: skip if already set up for this goal
        cache_key = hash(goal)
        if cache_key in self._setup_cache:
            return

        # Parse intent to understand requirements
        task_graph = self.swarm_intent_parser.parse(goal)
        
        # Research solutions if needed
        if task_graph.requirements or task_graph.integrations:
            logger.info("🔍 Researching solutions...")
            # Filter out stop words and meaningless single-word requirements
            stop_words = {'existing', 'use', 'check', 'find', 'get', 'the', 'a', 'an', 'and', 'or', 'for', 'with'}
            meaningful_requirements = [
                req for req in task_graph.requirements 
                if req.lower() not in stop_words and len(req.split()) > 1  # Multi-word or meaningful
            ]
            
            for requirement in meaningful_requirements:
                if not requirement.strip():
                    continue
                research_result = await self.swarm_researcher.research(requirement)
                if research_result.tools_found:
                    logger.info(f"✅ Found tools: {research_result.tools_found}")
                    # Install tools
                    for tool in research_result.tools_found:
                        await self.swarm_installer.install(tool)
        
        # Configure integrations
        if task_graph.integrations:
            logger.info(f"Configuring integrations: {task_graph.integrations}")
            for integration in task_graph.integrations:
                await self.swarm_configurator.configure(integration)

        # Mark as cached
        self._setup_cache[cache_key] = True

    # =====================================================================
    # State Management Methods (V1 capabilities integrated)
    # =====================================================================
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current swarm-level state for Q-prediction and introspection.
        
        Returns rich state including:
        - Task progress (completed, pending, failed)
        - Query/Goal context
        - Metadata context (tables, columns, filters)
        - Error patterns
        - Tool usage patterns
        - Actor outputs
        - Validation context
        - Agent states
        
        Returns:
            Dictionary with comprehensive state information
        """
        if not self.swarm_state_manager:
            return {}
        return self.swarm_state_manager.get_current_state()
    
    def get_agent_state(self, agent_name: str) -> Dict[str, Any]:
        """
        Get state for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary with agent-specific state (outputs, errors, tool usage, etc.)
        """
        if not self.swarm_state_manager:
            return {}
        return self.swarm_state_manager.get_agent_state(agent_name)
    
    def get_state_summary(self) -> str:
        """
        Get human-readable state summary.
        
        Returns:
            String summary of swarm and agent states
        """
        if not self.swarm_state_manager:
            return "State management not initialized"
        return self.swarm_state_manager.get_state_summary()
    
    def get_available_actions(self) -> List[Dict[str, Any]]:
        """
        Get available actions for exploration.
        
        Returns:
            List of available actions (agents that can be executed)
        """
        if not self.swarm_state_manager:
            return []
        return self.swarm_state_manager.get_available_actions()
    
    def save_state(self, file_path: Optional[Union[str, Path]] = None):
        """
        Save swarm and agent state to file.
        
        Args:
            file_path: Optional path to save state (defaults to config base_path)
        """
        if not self.swarm_state_manager:
            logger.warning("⚠️  SwarmStateManager not initialized, cannot save state")
            return
        
        if file_path is None:
            base_path = getattr(self.config, 'base_path', Path('/tmp/jotty_state'))
            file_path = Path(base_path) / 'swarm_state.json'
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.swarm_state_manager.save_state(file_path)
    
    def load_state(self, file_path: Union[str, Path]):
        """
        Load swarm and agent state from file.
        
        Args:
            file_path: Path to state file
        """
        if not self.swarm_state_manager:
            logger.warning("⚠️  SwarmStateManager not initialized, cannot load state")
            return
        
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"⚠️  State file not found: {file_path}")
            return
        
        self.swarm_state_manager.load_state(file_path)
        logger.info(f"State loaded from {file_path}")
