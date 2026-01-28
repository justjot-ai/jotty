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

import logging
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
                swarm_state_manager=self.swarm_state_manager  # Pass state manager for agent-level tracking
            )
            
            self.runners[agent_config.name] = runner
        
        logger.info(f"✅ SwarmManager initialized: {self.mode} mode, {len(self.agents)} agents")
        if self.swarm_profiler:
            logger.info("   ⏱️  SwarmProfiler enabled")
        logger.info("   ✅ SwarmToolValidator initialized")
        logger.info("   🎨 SwarmUIRegistry initialized")
        logger.info("   🌐 SwarmProviderGateway configured (unified provider system)")
        logger.info("   🤖 Autonomous components ready (zero-config enabled)")
        logger.info("   📊 SwarmStateManager initialized (swarm + agent-level tracking)")
    
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
        
        return result
    
    async def _execute_multi_agent(self, goal: str, **kwargs) -> EpisodeResult:
        """
        Execute multi-agent mode.
        
        Args:
            goal: Task goal
            **kwargs: Additional arguments
            
        Returns:
            EpisodeResult
        """
        # Add tasks to SwarmTaskBoard
        for i, agent_config in enumerate(self.agents):
            task_id = f"task_{i+1}"
            self.swarm_task_board.add_task(
                task_id=task_id,
                description=f"{goal} (agent: {agent_config.name})",
                actor=agent_config.name
            )
        
        # Execute tasks in order
        results = []
        while True:
            next_task = self.swarm_task_board.get_next_task()
            if next_task is None:
                break
            
            runner = self.runners[next_task.actor]
            result = await runner.run(goal=next_task.description, **kwargs)
            
            if result.success:
                self.swarm_task_board.complete_task(next_task.task_id, result={'output': result.output})
                
                # Learn from successful execution (DRY: reuse workflow learner)
                agent_config = next((a for a in self.agents if a.name == next_task.actor), None)
                if agent_config:
                    self._learn_from_result(result, agent_config)
            else:
                self.swarm_task_board.fail_task(next_task.task_id, error=str(result.error or "Execution failed"))
            
            results.append(result)
        
        # Return first result (or combine if needed)
        if results:
            return results[0]
        else:
            from ...foundation.data_structures import EpisodeResult
            return EpisodeResult(
                output=None,
                success=False,
                agent_name="swarm_manager",
                error="No tasks executed"
            )
    
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
            logger.info(f"⚙️  Configuring integrations: {task_graph.integrations}")
            for integration in task_graph.integrations:
                await self.swarm_configurator.configure(integration)
    
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
        
        # Check for similar workflows (DRY: reuse learned patterns)
        similar_pattern = self.swarm_workflow_learner.find_similar_pattern(
            task_type=task_graph.task_type.value,
            operations=task_graph.operations,
            tools_available=[]  # Will be populated from skills registry
        )
        
        if similar_pattern:
            logger.info(f"📚 Found similar workflow pattern: {similar_pattern.pattern_id}")
