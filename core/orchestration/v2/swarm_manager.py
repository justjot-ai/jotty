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
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from collections import defaultdict

from ...foundation.data_structures import JottyConfig, EpisodeResult
from ...foundation.agent_config import AgentConfig
from .agent_runner import AgentRunner, AgentRunnerConfig
# Import directly from source to avoid circular import
from ..roadmap import MarkovianTODO as SwarmTaskBoard, TaskStatus
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
from .swarm_intelligence import SwarmIntelligence, SyntheticTask, CurriculumGenerator
from ...foundation.robust_parsing import AdaptiveWeightGroup
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

        # Intelligent Terminal (auto-fix, web search, skill generation)
        self.swarm_terminal = SwarmTerminal(
            config=self.config,
            auto_fix=True,
            max_fix_attempts=3
        )
        logger.info("🤖 Autonomous components initialized (including SwarmTerminal)")

        # MAS Learning - Persistent learning across sessions
        workspace_path = getattr(self.config, 'base_path', None)
        self.mas_learning = MASLearning(
            config=self.config,
            workspace_path=workspace_path
        )

        # Enable memory persistence for swarm_memory
        self.memory_persistence = self.mas_learning.enable_memory_persistence(
            self.swarm_memory,
            agent_name="SwarmShared"
        )

        # Integrate fix database with SwarmTerminal
        self.mas_learning.integrate_with_terminal(self.swarm_terminal)
        logger.info("🧠 MASLearning initialized (persistent learning across sessions)")
        
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
                transfer_learning=self.transfer_learning,
                swarm_terminal=self.swarm_terminal  # Shared intelligent terminal
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

    # =========================================================================
    # Intelligent Terminal Methods (SwarmTerminal)
    # =========================================================================

    async def terminal_execute(
        self,
        command: str,
        timeout: int = 60,
        auto_fix: bool = True,
        working_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute command with intelligent error handling.

        Uses SwarmTerminal for:
        - Automatic error detection
        - Web search for solutions
        - Auto-apply fixes
        - Learning from successful fixes

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds
            auto_fix: Automatically fix errors (default: True)
            working_dir: Working directory for command

        Returns:
            Dict with success, output, error, and fix info
        """
        result = await self.swarm_terminal.execute(
            command=command,
            timeout=timeout,
            auto_fix=auto_fix,
            working_dir=working_dir
        )

        return {
            'success': result.success,
            'command': result.command,
            'output': result.output,
            'error': result.error,
            'exit_code': result.exit_code,
            'fix_applied': result.fix_applied,
            'fix_description': result.fix_description
        }

    async def terminal_diagnose(self) -> Dict[str, Any]:
        """
        Run system diagnostics.

        Checks:
        - Internet connectivity
        - Disk space
        - Memory usage
        - Python environment
        - Common tools (git, node, docker)

        Returns:
            Dict with diagnostic results
        """
        return await self.swarm_terminal.diagnose_system()

    async def terminal_generate_skill(
        self,
        problem: str,
        error_context: str = "",
        skill_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a new skill to solve a problem.

        Uses LLM to:
        - Analyze the problem
        - Generate skill code
        - Install dependencies
        - Register skill

        Args:
            problem: Problem description
            error_context: Error context from failed attempts
            skill_name: Optional skill name

        Returns:
            Dict with skill info or None if failed
        """
        return await self.swarm_terminal.generate_skill(
            problem=problem,
            error_context=error_context,
            skill_name=skill_name
        )

    def terminal_get_fix_history(self) -> List[Dict[str, Any]]:
        """Get history of applied fixes from SwarmTerminal."""
        return self.swarm_terminal.get_fix_history()

    async def terminal_auto_remediate(
        self,
        error_message: str,
        context: Dict[str, Any] = None,
        auto_generate_skill: bool = False
    ) -> Dict[str, Any]:
        """
        Attempt to auto-remediate an error using SwarmTerminal.

        This method:
        1. Analyzes the error
        2. Searches web for solutions
        3. Attempts to apply fixes
        4. Optionally generates a new skill if needed

        Args:
            error_message: Error message to remediate
            context: Additional context (task, agent, etc.)
            auto_generate_skill: Generate skill if fix not found

        Returns:
            Dict with remediation results
        """
        context = context or {}

        # First, run diagnostics to check system state
        diagnostics = await self.swarm_terminal.diagnose_system()

        # Check if it's a network issue
        if not diagnostics['checks'].get('internet'):
            logger.warning("🌐 Network issue detected, attempting to diagnose...")
            net_result = await self.swarm_terminal.execute(
                "nmcli networking connectivity check || ping -c 3 8.8.8.8",
                auto_fix=True
            )
            if not net_result.success:
                return {
                    'success': False,
                    'error': 'Network connectivity issue',
                    'diagnostics': diagnostics,
                    'remediation': 'Network fix attempted but failed'
                }

        # Analyze error and search for solutions
        solution_found = False
        fix_applied = False
        fix_description = ""

        # Check if error looks like a command/package issue
        error_keywords = ['command not found', 'module', 'import', 'pip', 'npm', 'permission']
        is_command_error = any(kw in error_message.lower() for kw in error_keywords)

        if is_command_error:
            # Try to extract and fix the command
            import re
            cmd_match = re.search(r"'([^']+)'|`([^`]+)`|command:\s*(\S+)", error_message)
            if cmd_match:
                failed_cmd = cmd_match.group(1) or cmd_match.group(2) or cmd_match.group(3)
                logger.info(f"🔧 Attempting to fix command issue: {failed_cmd}")

                # Try running with auto-fix
                result = await self.swarm_terminal.execute(
                    f"which {failed_cmd} || type {failed_cmd}",
                    auto_fix=True
                )
                fix_applied = result.fix_applied
                fix_description = result.fix_description
                solution_found = result.success

        # If still not fixed and auto_generate_skill is enabled
        if not solution_found and auto_generate_skill:
            logger.info("✨ Attempting to generate new skill for problem...")
            skill_result = await self.swarm_terminal.generate_skill(
                problem=f"Error: {error_message}",
                error_context=str(context)
            )
            if skill_result:
                return {
                    'success': True,
                    'error': error_message,
                    'remediation': 'Generated new skill',
                    'skill_generated': skill_result,
                    'diagnostics': diagnostics
                }

        return {
            'success': solution_found,
            'error': error_message,
            'fix_applied': fix_applied,
            'fix_description': fix_description,
            'diagnostics': diagnostics,
            'remediation': 'Auto-remediation attempted'
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

        # Log MAS Learning statistics
        if hasattr(self, 'mas_learning') and self.mas_learning:
            stats = self.mas_learning.get_statistics()
            logger.info(f"MAS Learning ready: {stats['fix_database']['total_fixes']} fixes, "
                       f"{stats['agent_performance']['total_agents']} agents, "
                       f"{stats['sessions']['total_sessions']} sessions")

    def load_relevant_learnings(self, task_description: str, agent_types: List[str] = None) -> Dict[str, Any]:
        """
        Load learnings relevant to the current task.

        This is the key method for smart learning selection:
        - Matches task topics to past sessions
        - Suggests best agents for the task
        - Provides relevant fixes and strategies

        Args:
            task_description: Description of the current task
            agent_types: Agent types that will be used (optional)

        Returns:
            Dict with relevant learnings
        """
        if not hasattr(self, 'mas_learning') or not self.mas_learning:
            return {}

        return self.mas_learning.load_relevant_learnings(
            task_description=task_description,
            agent_types=agent_types or [a.name for a in self.agents]
        )

    def record_agent_result(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        time_taken: float,
        output_quality: float = 0.0
    ):
        """Record an agent's task result for learning."""
        if hasattr(self, 'mas_learning') and self.mas_learning:
            self.mas_learning.record_agent_task(
                agent_type=agent_name,
                task_type=task_type,
                success=success,
                time_taken=time_taken,
                output_quality=output_quality
            )

    def record_session_result(
        self,
        task_description: str,
        agent_performances: Dict[str, Dict[str, Any]],
        total_time: float,
        success: bool,
        fixes_applied: List[Dict[str, Any]] = None,
        stigmergy_signals: int = 0
    ):
        """Record session results for future learning."""
        if hasattr(self, 'mas_learning') and self.mas_learning:
            self.mas_learning.record_session(
                task_description=task_description,
                agent_performances=agent_performances,
                fixes_applied=fixes_applied or [],
                stigmergy_signals=stigmergy_signals,
                total_time=total_time,
                success=success
            )

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

        # Save MAS Learning (fix database, agent performance, sessions)
        if hasattr(self, 'mas_learning') and self.mas_learning:
            try:
                # Sync fixes from SwarmTerminal before saving
                self.mas_learning.sync_from_terminal(self.swarm_terminal)
                # Save all MAS learnings
                self.mas_learning.save_all()
            except Exception as e:
                logger.debug(f"Could not auto-save MAS learnings: {e}")

        # Save HierarchicalMemory persistence
        if hasattr(self, 'memory_persistence') and self.memory_persistence:
            try:
                self.memory_persistence.save()
            except Exception as e:
                logger.debug(f"Could not auto-save memory: {e}")

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
                "swarm_workflow_learner.get_best_patterns() - Get best patterns by success rate",
                "warmup(num_episodes) - DrZero-inspired zero-data bootstrapping",
                "get_warmup_recommendation() - Check if warmup would be beneficial"
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
            skip_autonomous_setup: If True, skip research/install/configure (fast mode)
            status_callback: Optional callback(stage, detail) for progress updates
            ensemble: Enable prompt ensembling for multi-perspective analysis
            ensemble_strategy: Strategy for ensembling:
                - 'self_consistency': Same prompt, N samples, synthesis
                - 'multi_perspective': Different expert personas (default)
                - 'gsa': Generative Self-Aggregation
                - 'debate': Multi-round argumentation
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
                # Skip validation for zero-config mode since AutoAgent has its own planning
                # Architect/Auditor validation is redundant and slow (10 ReAct iterations)
                skip_val = skip_autonomous_setup or self.enable_zero_config
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
        """
        Zero-config: LLM decides if task needs multiple agents.

        Uses LLM to analyze task:
        - Sequential workflow (A → B → C) → Single AutoAgent
        - Parallel/independent sub-goals → Multiple AutoAgents

        No hardcoded categories - purely LLM-driven decision.
        """
        from ...agents.auto_agent import AutoAgent
        import dspy

        def _status(stage: str, detail: str = ""):
            if status_callback:
                try:
                    status_callback(stage, detail)
                except Exception:
                    pass
            logger.info(f"📍 {stage}" + (f": {detail}" if detail else ""))

        # Try LLM-based decision
        try:
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                _status("LLM analyzing", "checking for parallel sub-goals")
                decision = self._llm_decide_agents(task)
                if decision and len(decision) > 1:
                    # Multiple independent sub-goals detected
                    agents = []
                    for i, sub_goal in enumerate(decision):
                        agent = AutoAgent()
                        # Derive logical name from sub-goal
                        agent_name = self._derive_agent_name(sub_goal, i)
                        agent_config = AgentConfig(
                            name=agent_name,
                            agent=agent,
                            capabilities=[sub_goal[:50]],
                            is_executor=True
                        )
                        agents.append(agent_config)
                        _status(f"  {agent_name}", sub_goal[:60])
                    _status("Multi-agent mode", f"{len(agents)} parallel agents")
                    return agents
                else:
                    _status("Single-agent mode", "sequential workflow detected")
        except Exception as e:
            logger.debug(f"LLM agent decision failed, using single agent: {e}")
            _status("Single-agent mode", "fallback (LLM decision failed)")

        # Default: Single AutoAgent handles sequential workflows well
        return [AgentConfig(name="auto", agent=AutoAgent())]

    def _llm_decide_agents(self, task: str) -> List[str]:
        """
        Use LLM to decide if task has parallel sub-goals.

        Returns:
            List of sub-goals if parallel work detected, else empty list
        """
        import dspy

        class AgentDecisionSignature(dspy.Signature):
            """Analyze if task has INDEPENDENT sub-goals that can run in PARALLEL.

            BE CONSERVATIVE - prefer single agent for most tasks.
            Only split into parallel agents when there are TRULY INDEPENDENT work streams.

            PARALLEL (rare): Sub-goals that produce SEPARATE outputs with NO dependencies.
            SEQUENTIAL (common): Steps that build on each other (research → synthesize → format).

            Examples:
            - "Create checklist for X" → SEQUENTIAL (single agent: research + create + format)
            - "Research X and generate PDF" → SEQUENTIAL (PDF needs research output)
            - "Compare A vs B AND compare C vs D" → PARALLEL ONLY if comparing different domains
            - "Analyze company X for multiple aspects" → SEQUENTIAL (one comprehensive analysis)

            AVOID creating multiple agents for:
            - Tasks that share the same domain/topic (even if phrased differently)
            - Tasks where one naturally leads to another
            - Tasks that will produce overlapping research
            """
            task: str = dspy.InputField(desc="The task to analyze")
            is_parallel: bool = dspy.OutputField(desc="True ONLY if task has truly independent sub-goals. Default to False.")
            sub_goals: str = dspy.OutputField(desc="If parallel, JSON list of 2-4 DISTINCT sub-goals (no duplicates). If sequential, empty list []")

        try:
            predictor = dspy.Predict(AgentDecisionSignature)
            result = predictor(task=task)

            logger.info(f"🤖 LLM decision: is_parallel={result.is_parallel}, sub_goals={result.sub_goals[:100]}")

            if result.is_parallel:
                import json
                sub_goals = json.loads(result.sub_goals)
                if isinstance(sub_goals, list) and len(sub_goals) > 1:
                    # Deduplicate similar sub-goals
                    sub_goals = self._deduplicate_sub_goals(sub_goals)
                    # Limit to max 4 parallel agents
                    sub_goals = sub_goals[:4]
                    if len(sub_goals) > 1:
                        logger.info(f"🔀 LLM detected {len(sub_goals)} parallel sub-goals: {sub_goals}")
                        return sub_goals
                    else:
                        logger.info(f"📝 After deduplication: single agent optimal")
            else:
                logger.info(f"📝 LLM detected sequential workflow - single agent optimal")
        except Exception as e:
            logger.debug(f"Agent decision parsing failed: {e}")

        return []  # Default: sequential/single agent

    def _deduplicate_sub_goals(self, sub_goals: List[str]) -> List[str]:
        """
        Remove duplicate or highly similar sub-goals.

        Uses word overlap to detect duplicates like:
        - "Identify oversight requirements for SPVs"
        - "Identify oversight requirements for Fund admins"
        These share too much overlap and should be merged.
        """
        if len(sub_goals) <= 1:
            return sub_goals

        def get_key_words(text: str) -> set:
            """Extract meaningful words from text."""
            stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'to', 'of', 'in', 'on', 'with', 'is', 'are', 'be', 'that', 'this'}
            words = set(w.lower() for w in text.split() if len(w) > 2 and w.lower() not in stop_words)
            return words

        def similarity(a: str, b: str) -> float:
            """Calculate Jaccard similarity between two texts."""
            words_a = get_key_words(a)
            words_b = get_key_words(b)
            if not words_a or not words_b:
                return 0.0
            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            return intersection / union if union > 0 else 0.0

        # Keep track of which goals to keep
        unique_goals = []
        for goal in sub_goals:
            is_duplicate = False
            for existing in unique_goals:
                sim = similarity(goal, existing)
                if sim > 0.6:  # 60% overlap = too similar
                    logger.debug(f"Merging duplicate sub-goal (sim={sim:.0%}): '{goal[:40]}...' with '{existing[:40]}...'")
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_goals.append(goal)

        if len(unique_goals) < len(sub_goals):
            logger.info(f"📝 Deduplicated: {len(sub_goals)} → {len(unique_goals)} sub-goals")

        return unique_goals

    def _derive_agent_name(self, sub_goal: str, index: int) -> str:
        """
        Derive a logical, descriptive agent name from sub-goal.

        Uses LLM to extract the most meaningful identifier from the sub-goal.

        Examples:
        - "Research BaFin KGAB framework requirements" → "bafin_kgab"
        - "Analyze technical indicators for TSLA" → "tsla_technicals"
        - "Compare regulatory approaches EU vs US" → "eu_us_comparison"
        """
        import dspy
        import re

        goal_lower = sub_goal.lower()

        # Try LLM-based name extraction first (more accurate)
        try:
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                class AgentNameSignature(dspy.Signature):
                    """Extract a short, descriptive agent name from task description.

                    The name should:
                    - Be 2-3 words max, joined by underscore
                    - Capture the MAIN TOPIC or ENTITY being worked on
                    - Be specific (not generic like "researcher" or "analyst")

                    Examples:
                    - "Research BaFin KGAB framework" → "bafin_kgab"
                    - "Analyze Tesla stock technicals" → "tesla_technicals"
                    - "Compare EU vs US regulations" → "eu_us_regs"
                    - "Generate summary of AI news" → "ai_news"
                    """
                    task: str = dspy.InputField(desc="Task description")
                    name: str = dspy.OutputField(desc="Short agent name (2-3 words, underscore separated, no generic words)")

                predictor = dspy.Predict(AgentNameSignature)
                result = predictor(task=sub_goal)
                name = result.name.strip().lower()
                # Clean: remove quotes, limit length, replace spaces
                name = re.sub(r'["\']', '', name)
                name = re.sub(r'\s+', '_', name)
                name = name[:20]  # Max 20 chars
                if name and len(name) >= 3:
                    logger.debug(f"LLM-derived agent name: {name}")
                    return name
        except Exception as e:
            logger.debug(f"LLM agent naming failed, using heuristics: {e}")

        # Fallback: Heuristic-based extraction
        # 1. Look for specific entity names (tickers, companies, frameworks)
        entities = re.findall(r'\b([A-Z]{2,6})\b', sub_goal)  # Stock tickers, acronyms
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', sub_goal)

        # 2. Common domain-specific patterns
        domain_patterns = [
            (r'fundamental', 'fundamentals'),
            (r'technical', 'technicals'),
            (r'sentiment', 'sentiment'),
            (r'regulatory|regulation|compliance', 'regulatory'),
            (r'risk\s*management', 'risk_mgmt'),
            (r'market\s*analysis', 'market'),
            (r'competitor', 'competitors'),
            (r'valuation', 'valuation'),
            (r'earnings', 'earnings'),
            (r'news|headline', 'news'),
        ]

        for pattern, name in domain_patterns:
            if re.search(pattern, goal_lower):
                # Try to prepend entity
                if entities:
                    return f"{entities[0].lower()}_{name}"
                return name

        # 3. Extract key topic from goal
        # Remove common verbs and articles
        cleaned = re.sub(
            r'^(research|analyze|generate|create|get|find|compare|evaluate|assess|review|summarize|identify)\s+',
            '', goal_lower
        )
        cleaned = re.sub(r'\s+(analysis|report|data|information|research|requirements|framework)$', '', cleaned)
        cleaned = re.sub(r'\b(the|a|an|and|or|for|with|from|to|of|on|in)\b', '', cleaned)

        # Get meaningful words
        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]

        if words:
            # Use first 2 meaningful words
            name_parts = words[:2]
            name = '_'.join(name_parts)[:18]
            return name

        # 4. Use entity if found
        if entities:
            return entities[0].lower()
        if proper_nouns:
            return proper_nouns[0].lower().replace(' ', '_')[:15]

        # Final fallback with index
        return f'task_{index+1}'

    def _should_auto_ensemble(self, goal: str) -> bool:
        """
        Determine if ensemble should be auto-enabled based on task type.

        BE CONSERVATIVE - ensemble adds significant latency (4x LLM calls).
        Only enable for tasks that genuinely benefit from multiple perspectives.

        Auto-enables ensemble for:
        - Comparison tasks (A vs B, compare X and Y)
        - Decision tasks (should I, which is better)

        DOES NOT auto-enable for:
        - Creation tasks (create, generate, write, build)
        - Simple research (research, find, search)
        - Document generation (checklist, report, summary)

        Args:
            goal: The task description

        Returns:
            True if ensemble should be auto-enabled
        """
        goal_lower = goal.lower()

        # EXCLUSION: Don't auto-ensemble for creation/generation tasks
        # These don't benefit from multi-perspective - they need execution
        creation_keywords = [
            'create ', 'generate ', 'write ', 'build ', 'make ',
            'checklist', 'template', 'document', 'report',
            'draft ', 'prepare ', 'compile ',
        ]
        for keyword in creation_keywords:
            if keyword in goal_lower:
                logger.debug(f"Auto-ensemble SKIPPED for creation task: {keyword}")
                return False

        # Comparison indicators (STRONG signal)
        comparison_keywords = [
            ' vs ', ' versus ', 'compare ',
            'difference between', 'differences between',
            'pros and cons', 'advantages and disadvantages',
        ]

        # Decision indicators (STRONG signal)
        decision_keywords = [
            'should i ', 'should we ',
            'which is better', 'what is best',
            'choose between', 'decide between',
        ]

        # Check for strong signals only
        for keyword in comparison_keywords:
            if keyword in goal_lower:
                logger.debug(f"Auto-ensemble triggered by comparison: {keyword}")
                return True

        for keyword in decision_keywords:
            if keyword in goal_lower:
                logger.debug(f"Auto-ensemble triggered by decision: {keyword}")
                return True

        # Analysis keywords - only if paired with comparison context
        # "fundamental analysis" alone is NOT enough, "compare fundamental analysis" is
        # This prevents false triggers
        return False

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

        # 7. MAS Learning: Record agent performance and session
        try:
            if hasattr(self, 'mas_learning') and self.mas_learning:
                task_type = self.transfer_learning.extractor.extract_task_type(goal) if hasattr(self, 'transfer_learning') else 'general'
                execution_time = getattr(result, 'execution_time', 0.0)

                # Record individual agent performance
                if hasattr(result, 'agent_contributions') and result.agent_contributions:
                    agent_performances = {}
                    for agent_name, contrib in result.agent_contributions.items():
                        success = getattr(contrib, 'decision_correct', result.success)
                        agent_time = getattr(contrib, 'execution_time', execution_time / len(result.agent_contributions))
                        self.mas_learning.record_agent_task(
                            agent_type=agent_name,
                            task_type=task_type,
                            success=success,
                            time_taken=agent_time
                        )
                        agent_performances[agent_name] = {
                            'success': success,
                            'success_rate': 1.0 if success else 0.0,
                            'avg_time': agent_time
                        }
                else:
                    agent_name = getattr(result, 'agent_name', self.agents[0].name if self.agents else 'unknown')
                    self.mas_learning.record_agent_task(
                        agent_type=agent_name,
                        task_type=task_type,
                        success=result.success,
                        time_taken=execution_time
                    )
                    agent_performances = {
                        agent_name: {
                            'success': result.success,
                            'success_rate': 1.0 if result.success else 0.0,
                            'avg_time': execution_time
                        }
                    }

                # Record session
                stigmergy_signals = len(self.swarm_intelligence.stigmergy.signals) if hasattr(self, 'swarm_intelligence') else 0
                self.mas_learning.record_session(
                    task_description=goal,
                    agent_performances=agent_performances,
                    fixes_applied=getattr(self.swarm_terminal, '_fix_history', []) if hasattr(self, 'swarm_terminal') else [],
                    stigmergy_signals=stigmergy_signals,
                    total_time=execution_time,
                    success=result.success
                )
        except Exception as e:
            logger.debug(f"MAS Learning record skipped: {e}")

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

    # =====================================================================
    # DrZero-Inspired Zero-Data Bootstrapping
    # =====================================================================

    async def warmup(
        self,
        num_episodes: int = 10,
        target_agent: Optional[str] = None,
        difficulty_range: Tuple[float, float] = (0.2, 0.6),
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        DrZero-inspired zero-data bootstrapping.

        Runs synthetic training episodes to bootstrap agent learning
        BEFORE real user tasks. This enables:
        1. COLD START MITIGATION: Agents have learned patterns before first real task
        2. SKILL DISCOVERY: Identify agent strengths/weaknesses
        3. BASELINE CALIBRATION: Establish HRPO group baselines
        4. MEMORY SEEDING: Populate memory with useful patterns

        DrZero insight: Self-generated curriculum allows agents to improve
        without external training data.

        Args:
            num_episodes: Number of synthetic training episodes
            target_agent: Optional specific agent to train (None = all)
            difficulty_range: (min, max) difficulty for curriculum
            verbose: Log progress

        Returns:
            Dict with warmup statistics:
            - episodes_run: Total episodes completed
            - success_rate: Fraction of successful episodes
            - agent_improvements: Per-agent performance changes
            - curriculum_stats: Curriculum generator statistics

        Example:
            # Warmup before real tasks
            swarm = SwarmManager(agents=[...])
            stats = await swarm.warmup(num_episodes=10)
            print(f"Warmup complete: {stats['success_rate']:.1%} success rate")

            # Now run real tasks with warmed-up agents
            result = await swarm.run("Real user task")
        """
        if verbose:
            logger.info(f"🔥 Starting DrZero warmup: {num_episodes} synthetic episodes")

        # Track warmup statistics
        stats = {
            'episodes_run': 0,
            'successes': 0,
            'failures': 0,
            'agent_results': defaultdict(lambda: {'success': 0, 'total': 0}),
            'task_type_results': defaultdict(lambda: {'success': 0, 'total': 0}),
            'initial_baselines': dict(self.swarm_intelligence.curriculum_generator.difficulty_by_type),
        }

        # Get curriculum generator
        curriculum = self.swarm_intelligence.curriculum_generator

        for episode in range(num_episodes):
            # 1. Generate synthetic task from curriculum
            task = curriculum.generate_training_task(
                profiles=self.swarm_intelligence.agent_profiles,
                target_agent=target_agent
            )

            # Ensure difficulty is within range
            if task.difficulty < difficulty_range[0]:
                task.difficulty = difficulty_range[0]
            elif task.difficulty > difficulty_range[1]:
                task.difficulty = difficulty_range[1]

            if verbose:
                logger.info(
                    f"  [{episode + 1}/{num_episodes}] "
                    f"Task: {task.task_type} (difficulty: {task.difficulty:.1%})"
                )

            # 2. Run synthetic episode
            try:
                result = await self._run_synthetic_episode(task, target_agent)
                success = result.get('success', False)
                execution_time = result.get('execution_time', 0.0)
            except Exception as e:
                logger.warning(f"  Warmup episode {episode + 1} failed: {e}")
                success = False
                execution_time = 0.0

            # 3. Update curriculum based on result
            curriculum.update_from_result(task, success, execution_time)

            # 4. Track statistics
            stats['episodes_run'] += 1
            if success:
                stats['successes'] += 1
            else:
                stats['failures'] += 1

            agent_name = task.target_agent or 'swarm'
            stats['agent_results'][agent_name]['total'] += 1
            if success:
                stats['agent_results'][agent_name]['success'] += 1

            stats['task_type_results'][task.task_type]['total'] += 1
            if success:
                stats['task_type_results'][task.task_type]['success'] += 1

        # Compute final statistics
        stats['success_rate'] = stats['successes'] / max(1, stats['episodes_run'])
        stats['final_baselines'] = dict(curriculum.difficulty_by_type)
        stats['curriculum_stats'] = curriculum.get_curriculum_stats()

        # Compute per-agent improvements
        stats['agent_improvements'] = {}
        for agent_name, results in stats['agent_results'].items():
            rate = results['success'] / max(1, results['total'])
            stats['agent_improvements'][agent_name] = rate

        # Convert defaultdicts to regular dicts for serialization
        stats['agent_results'] = dict(stats['agent_results'])
        stats['task_type_results'] = dict(stats['task_type_results'])

        if verbose:
            logger.info(f"🔥 Warmup complete: {stats['success_rate']:.1%} success rate")
            logger.info(f"   Episodes: {stats['episodes_run']}, Successes: {stats['successes']}")
            for task_type, results in stats['task_type_results'].items():
                rate = results['success'] / max(1, results['total'])
                logger.info(f"   {task_type}: {rate:.1%} ({results['total']} episodes)")

        # Auto-save learnings from warmup
        self._auto_save_learnings()

        return stats

    async def _run_synthetic_episode(
        self,
        task: 'SyntheticTask',
        target_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single synthetic training episode.

        This is an INTERNAL execution that:
        - Does NOT affect external systems
        - Updates agent learning/memory
        - Records in swarm intelligence

        Args:
            task: Synthetic task from curriculum
            target_agent: Specific agent to train

        Returns:
            Dict with episode results
        """
        import time as time_module
        start_time = time_module.time()

        # Determine which agent to use
        if target_agent:
            agent_name = target_agent
        elif self.mode == "single":
            agent_name = self.agents[0].name
        else:
            # Use swarm intelligence to route to best agent
            best = self.swarm_intelligence.get_best_agent_for_task(
                task.task_type,
                [a.name for a in self.agents]
            )
            agent_name = best or self.agents[0].name

        runner = self.runners.get(agent_name)
        if not runner:
            logger.warning(f"No runner for agent {agent_name}")
            return {'success': False, 'error': 'No runner', 'execution_time': 0.0}

        try:
            # Run with synthetic goal
            # Note: This is a simplified execution for warmup
            # Real execution would use full run() but warmup should be lightweight
            result = await runner.run(goal=task.description)

            success = getattr(result, 'success', False)
            execution_time = time_module.time() - start_time

            # Record in swarm intelligence
            self.swarm_intelligence.record_task_result(
                agent_name=agent_name,
                task_type=task.task_type,
                success=success,
                execution_time=execution_time,
                context={
                    'synthetic': True,
                    'warmup': True,
                    'difficulty': task.difficulty,
                    'curriculum_round': task.metadata.get('curriculum_round', 0)
                }
            )

            # Update transfer learning with synthetic experience
            try:
                self.transfer_learning.record_experience(
                    query=task.description[:200],
                    agent=agent_name,
                    action=task.task_type,
                    reward=1.0 if success else 0.0,
                    success=success,
                    error='',
                    context={'synthetic': True, 'warmup': True}
                )
            except Exception:
                pass  # Transfer learning is optional

            return {
                'success': success,
                'agent': agent_name,
                'task_type': task.task_type,
                'difficulty': task.difficulty,
                'execution_time': execution_time
            }

        except Exception as e:
            execution_time = time_module.time() - start_time
            logger.debug(f"Synthetic episode failed: {e}")

            # Still record the failure for learning
            self.swarm_intelligence.record_task_result(
                agent_name=agent_name,
                task_type=task.task_type,
                success=False,
                execution_time=execution_time,
                context={'synthetic': True, 'warmup': True, 'error': str(e)}
            )

            return {
                'success': False,
                'agent': agent_name,
                'task_type': task.task_type,
                'difficulty': task.difficulty,
                'execution_time': execution_time,
                'error': str(e)
            }

    def get_warmup_recommendation(self) -> Dict[str, Any]:
        """
        Get recommendation on whether warmup would be beneficial.

        Returns:
            Dict with:
            - should_warmup: bool
            - reason: str
            - recommended_episodes: int
            - weak_areas: List of task types needing attention
        """
        profiles = self.swarm_intelligence.agent_profiles
        curriculum = self.swarm_intelligence.curriculum_generator

        # Check if we have enough learning history
        total_tasks = sum(p.total_tasks for p in profiles.values())

        if total_tasks < 5:
            return {
                'should_warmup': True,
                'reason': 'Cold start - no learning history',
                'recommended_episodes': 15,
                'weak_areas': list(curriculum.task_templates.keys())
            }

        # Find weak task types
        weak_areas = []
        for agent_name, profile in profiles.items():
            for task_type, (success, total) in profile.task_success.items():
                if total >= 3 and success / total < 0.5:
                    weak_areas.append(task_type)

        weak_areas = list(set(weak_areas))

        if weak_areas:
            return {
                'should_warmup': True,
                'reason': f'Weak performance in: {", ".join(weak_areas)}',
                'recommended_episodes': len(weak_areas) * 5,
                'weak_areas': weak_areas
            }

        # Check average success rate
        total_success = sum(
            sum(s for s, t in p.task_success.values())
            for p in profiles.values()
        )
        total_attempts = sum(
            sum(t for s, t in p.task_success.values())
            for p in profiles.values()
        )

        if total_attempts > 0:
            avg_success = total_success / total_attempts
            if avg_success < 0.7:
                return {
                    'should_warmup': True,
                    'reason': f'Overall success rate low: {avg_success:.1%}',
                    'recommended_episodes': 10,
                    'weak_areas': weak_areas
                }

        return {
            'should_warmup': False,
            'reason': 'Learning state is healthy',
            'recommended_episodes': 0,
            'weak_areas': []
        }
