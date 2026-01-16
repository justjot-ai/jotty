"""
JOTTY Conductor v2.3 - Brain-Inspired Multi-Actor Orchestration with Agentic Data Discovery
================================================================================================

SOTA Agentic Architecture with:
- Multi-actor orchestration with shared learning
- LLM-based Q-Predictor and Value Estimator (no static tables!)
- SmartContextGuard (NEVER out of context)
- PolicyExplorer with dynamic Roadmap
- Prompt updates as weight updates (online learning)
- Exploration until full reward (all Auditors pass)

BRAIN-INSPIRED ENHANCEMENTS (v2.1):
- Sharp Wave Ripple: Memory consolidation during "sleep"
- Hippocampal Extraction: What to remember (salience, novelty, relevance)
- Online/Offline Modes: Awake (learning) vs Sleep (consolidation)
- Agent Abstraction: Scalable swarm management
- Predictive MARL: Predict other agents, learn from divergence

A-TEAM AUTO-RESOLUTION (v2.2):
- Signature introspection: Auto-detects what each actor needs
- Parameter resolution: Auto-resolves from context providers and previous outputs
- Dependency graph: Auto-infers actor dependencies
- Zero hardcoding: Works with ANY agents, ANY domains

A-TEAM DATA REGISTRY (v2.3):
- Agentic data discovery: Agents can discover available data
- Semantic search: Find data by type, tags, fields
- Auto-registration: All outputs automatically indexed
- Type detection: HTML, DataFrames, files, predictions
- Non-breaking: Optional feature, backwards compatible

A-Team Design:
- Dr. Manning: Multi-agent TD(λ) with adaptive learning
- Dr. Chen: Inter-actor communication and coordination
- Dr. Agarwal: LLM-based everything (no embeddings)
- Aristotle: Causal learning across actors
- Shannon: Information-theoretic context management
- Alex: Robust persistence and health monitoring
"""

import asyncio
import inspect  # 🔑 NEW: For signature introspection
import json
import time
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging

# 🔑 NEW: Import Data Registry
from ..data.data_registry import DataRegistry, DataRegistryTool, DataArtifact
from ..data.io_manager import IOManager, ActorOutput, SwarmResult  # 🆕 A-TEAM: Typed output management
from ..data.data_transformer import SmartDataTransformer  # 🆕 A-TEAM: Intelligent data transformation
from ..agents.axon import SmartAgentSlack, AgentCapabilities, Message  # 🆕 A-TEAM: Intelligent agent communication

# 🤝 NEW: Import FeedbackChannel for agent coordination
from ..agents.feedback_channel import FeedbackChannel, FeedbackMessage, FeedbackType

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ..foundation.data_structures import JottyConfig, EpisodeResult, OutputTag, MemoryLevel
from ..foundation.exceptions import (
    JottyError,
    AgentExecutionError,
    ValidationError,
    PersistenceError,
    wrap_exception
)
from ..memory.cortex import HierarchicalMemory
from ..persistence.persistence import Vault
from .roadmap import MarkovianTODO, SubtaskState, TaskStatus  # Enhanced TODO with all methods
from ..foundation.agent_config import AgentConfig  # JOTTY v1.0: Unified agent config

# 🆕 REFACTORING: Components extracted for better separation of concerns
from .parameter_resolver import ParameterResolver
from .tool_manager import ToolManager
from .state_manager import StateManager

# 🆕 REFACTORING PHASE 1.2: Import canonical classes (remove duplicates from this file)
from ..learning.q_learning import LLMQPredictor
from ..context.context_guard import SmartContextGuard
from .policy_explorer import PolicyExplorer

# Import brain-inspired components
try:
    from ..memory.consolidation_engine import (
        BrainMode, BrainModeConfig, BrainStateMachine,
        HippocampalExtractor, SharpWaveRippleConsolidator,
        AgentAbstractor
    )
    BRAIN_MODES_AVAILABLE = True
except ImportError:
    BRAIN_MODES_AVAILABLE = False

# Import BrainInspiredMemoryManager (Neuralink-level neuroscience)
try:
    from ..memory.memory_orchestrator import BrainInspiredMemoryManager
    BRAIN_MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    BRAIN_MEMORY_MANAGER_AVAILABLE = False

# Import SimpleBrain (DEPRECATED - for backward compatibility only)
try:
    from ..memory.memory_orchestrator import Experience  # Only import Experience for type hints
    SIMPLE_BRAIN_AVAILABLE = True
except ImportError:
    SIMPLE_BRAIN_AVAILABLE = False

# Import predictive MARL
try:
    from .predictive_marl import (
        LLMTrajectoryPredictor, DivergenceMemory,
        CooperativeCreditAssigner, PredictedTrajectory, ActualTrajectory
    )
    PREDICTIVE_MARL_AVAILABLE = True
except ImportError:
    PREDICTIVE_MARL_AVAILABLE = False

# Import robust parsing utilities (A-Team Approved)
try:
    from ..foundation.robust_parsing import (
        parse_float_robust, AdaptiveThreshold, EpsilonGreedy, safe_hash
    )
    ROBUST_PARSING_AVAILABLE = True
except ImportError:
    ROBUST_PARSING_AVAILABLE = False

# Import Q-Learning with NeuroChunk Memory Management
try:
    from ..learning.q_learning import LLMQPredictor as NaturalLanguageQTable  # Use full Q-table implementation
    Q_LEARNING_AVAILABLE = True
except ImportError:
    Q_LEARNING_AVAILABLE = False

# Import TD(λ) Learning
try:
    from .learning import TDLambdaLearner
    TD_LAMBDA_AVAILABLE = True
except ImportError:
    TD_LAMBDA_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# MARKOVIAN TODO - Long-Horizon Task Management
# =============================================================================

@dataclass
class TodoItem:
    """A single TODO item with RL metadata."""
    id: str
    description: str
    actor: str  # Which actor handles this
    status: str  # pending, in_progress, completed, failed, blocked
    priority: float  # 0-1, learned priority
    estimated_reward: float  # Q-value estimate
    dependencies: List[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 5
    failure_reasons: List[str] = field(default_factory=list)
    completion_time: Optional[float] = None


# =============================================================================
# NOTE: MarkovianTODO is now imported from enhanced_state.py
# =============================================================================
# The enhanced MarkovianTODO has all the methods we need:
# - record_intermediary_values()
# - update_q_value()
# - predict_next()
# - get_state_summary()
# - Compatibility properties: .items, .completed
#
# Keeping TodoItem for backward compatibility with other components
# =============================================================================

# @dataclass
# class MarkovianTODO:  ← DEPRECATED: Use enhanced_state.MarkovianTODO
#     """DEPRECATED - Use MarkovianTODO from enhanced_state.py"""
#     root_task: str = ""
#     items: Dict[str, TodoItem] = field(default_factory=dict)
#     current_path: List[str] = field(default_factory=list)
#     completed: List[str] = field(default_factory=list)
#     failed: List[str] = field(default_factory=list)
#     estimated_remaining_steps: int = 0
#     
#     # RL state
#     state_history: List[Dict] = field(default_factory=list)
#     total_reward: float = 0.0
# 
# ALL METHODS REMOVED - Using enhanced_state.MarkovianTODO instead
# (which has all these methods plus more: record_intermediary_values, update_q_value, etc.)


# =============================================================================
# LLM-BASED Q-PREDICTOR - Imported from canonical location
# =============================================================================
# REFACTORING NOTE: LLMQPredictor now imported from core.learning.q_learning
# (See import section at top of file)


# =============================================================================
# SMART CONTEXT GUARD - Imported from canonical location
# =============================================================================
# REFACTORING NOTE: SmartContextGuard now imported from core.context.context_guard
# (See import section at top of file)


# =============================================================================
# POLICY EXPLORER - Imported from canonical location
# =============================================================================
# REFACTORING NOTE: PolicyExplorer now imported from core.orchestration.policy_explorer
# (See import section at top of file)


# =============================================================================
# SWARM LEARNER - Prompt Updates as Weight Updates
# =============================================================================

class SwarmLearnerSignature(dspy.Signature):
    """Update system prompts based on episode outcomes (online learning)."""
    
    current_prompt = dspy.InputField(desc="Current Architect/Auditor prompt")
    episode_trajectory = dspy.InputField(desc="What happened this episode")
    outcome = dspy.InputField(desc="Success/failure and why")
    patterns_observed = dspy.InputField(desc="Patterns that led to success/failure")
    
    updated_prompt = dspy.OutputField(desc="Updated prompt incorporating learnings")
    changes_made = dspy.OutputField(desc="List of specific changes made")
    learning_summary = dspy.OutputField(desc="What the system learned")


class SwarmLearner:
    """
    SOTA: Treats prompt updates as weight updates.
    
    Instead of static prompts:
    - Learns from each episode
    - Updates prompts with new patterns
    - Accumulates wisdom over time
    - Like fine-tuning but at prompt level
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.learner = dspy.ChainOfThought(SwarmLearnerSignature) if DSPY_AVAILABLE else None
        
        # Learning state
        self.learned_patterns: List[Dict] = []
        self.prompt_versions: Dict[str, List[str]] = {}  # prompt_name -> versions
        self.update_threshold = self.config.policy_update_threshold  # 🔧 STANFORD FIX: Episodes before updating
        self.episode_buffer: List[Dict] = []
    
    def record_episode(
        self, 
        trajectory: List[Dict], 
        outcome: bool, 
        insights: List[str]
    ):
        """Record episode for learning."""
        self.episode_buffer.append({
            'trajectory': trajectory,
            'outcome': outcome,
            'insights': insights,
            'timestamp': time.time()
        })
        
        # Extract patterns
        if outcome:
            self.learned_patterns.append({
                'type': 'success',
                'pattern': self._extract_pattern(trajectory),
                'timestamp': time.time()
            })
        else:
            self.learned_patterns.append({
                'type': 'failure',
                'pattern': self._extract_pattern(trajectory),
                'timestamp': time.time()
            })
    
    def should_update_prompts(self) -> bool:
        """Check if we should update prompts."""
        return len(self.episode_buffer) >= self.update_threshold
    
    def update_prompt(
        self, 
        prompt_name: str, 
        current_prompt: str
    ) -> Tuple[str, List[str]]:
        """
        Update a prompt based on learned patterns.
        
        Returns:
            (updated_prompt, changes_made)
        """
        if not self.learner or not self.episode_buffer:
            return current_prompt, []
        
        # Summarize episodes
        successes = [e for e in self.episode_buffer if e['outcome']]
        failures = [e for e in self.episode_buffer if not e['outcome']]
        
        patterns = []
        for p in self.learned_patterns[-20:]:
            patterns.append(f"{p['type']}: {p['pattern']}")
        
        try:
            result = self.learner(
                current_prompt=current_prompt[:3000],
                episode_trajectory=json.dumps({
                    'success_count': len(successes),
                    'failure_count': len(failures),
                    'recent_trajectories': [e['trajectory'][:3] for e in self.episode_buffer[-3:]]
                }, default=str)[:1500],
                outcome=f"Successes: {len(successes)}, Failures: {len(failures)}",
                patterns_observed="\n".join(patterns[-10:])[:1000]
            )
            
            updated = result.updated_prompt or current_prompt
            changes = result.changes_made.split('\n') if result.changes_made else []
            
            # Track versions
            if prompt_name not in self.prompt_versions:
                self.prompt_versions[prompt_name] = []
            self.prompt_versions[prompt_name].append(updated)
            
            # Clear buffer after update
            self.episode_buffer = []
            
            logger.info(f"📝 Prompt '{prompt_name}' updated with {len(changes)} changes")
            return updated, changes
            
        except Exception as e:
            logger.warning(f"Prompt update failed: {e}")
            return current_prompt, []
    
    def _extract_pattern(self, trajectory: List[Dict]) -> str:
        """Extract pattern from trajectory."""
        steps = [step.get('step', 'unknown') for step in trajectory[:5]]
        return " → ".join(steps)


# =============================================================================
# MAIN JOTTY CONDUCTOR CLASS
# =============================================================================

# Use AgentConfig from agent_config.py (JOTTY v1.0)
# Alias for backward compatibility in existing code
ActorConfig = AgentConfig


class MultiAgentsOrchestrator:
    """
    Multi-Actor Reinforced Validation Framework.

    Top-level multi-agent orchestrator (formerly Conductor).

    Orchestrates multiple actors with:
    - Shared hierarchical memory
    - LLM-based Q-prediction
    - Markovian TODO management
    - Policy exploration when stuck
    - Smart context management
    - Online prompt learning

    Usage:
        # Generic example - works with ANY agents
        swarm = MultiAgentsOrchestrator(
            actors=[
                ActorConfig("agent1", MyAgent1, ["agent1_architect.md"], ["agent1_auditor.md"]),
                ActorConfig("agent2", MyAgent2, ["agent2_architect.md"], ["agent2_auditor.md"]),
            ],
            config=swarm_config,
            global_architect="plan_validation.md",
            global_auditor="output_validation.md"
        )

        result = await swarm.run(goal="Complete the task", **kwargs)
    """
    
    def __init__(
        self,
        actors: List[ActorConfig],
        metadata_provider,  # ✅ A-TEAM: Generic protocol
        config: JottyConfig = None,
        global_architect: Optional[str] = None,
        global_auditor: Optional[str] = None,
        annotations_path: Optional[str] = None,
        context_providers: Optional[Dict[str, Any]] = None,
        enable_data_registry: bool = True,  # 🔑 NEW: Enable agentic data discovery
        custom_mappings: Optional[Dict[str, List[str]]] = None,  # 🆕 A-TEAM: User-defined parameter mappings!
        # LangGraph integration
        use_langgraph: bool = False,  # Enable LangGraph orchestration
        langgraph_mode: str = "dynamic",  # "dynamic" or "static"
        agent_order: Optional[List[str]] = None,  # Required for static mode
        # Task Queue integration
        task_queue = None,  # Optional: TaskQueue instance for task management
    ):
        self.config = config or JottyConfig()
        self.metadata_provider = metadata_provider  # ✅ A-TEAM: Store protocol
        self.context_providers = context_providers or {}
        
        # Task Queue integration (optional)
        self.task_queue = task_queue
        if task_queue:
            logger.info(f"📋 Task Queue enabled: {type(task_queue).__name__}")
        
        # LangGraph integration
        self.use_langgraph = use_langgraph
        self.langgraph_mode = langgraph_mode
        self.agent_order = agent_order
        self.langgraph_orchestrator = None
        
        if use_langgraph:
            try:
                from .langgraph_orchestrator import LangGraphOrchestrator, GraphMode
                
                mode = GraphMode.DYNAMIC if langgraph_mode == "dynamic" else GraphMode.STATIC
                self.langgraph_orchestrator = LangGraphOrchestrator(
                    conductor=self,
                    mode=mode,
                    agent_order=agent_order
                )
                logger.info(f"✅ LangGraph orchestration enabled (mode: {langgraph_mode})")
            except ImportError as e:
                logger.warning(f"⚠️  LangGraph not available: {e}. Install with: pip install langgraph langchain-core")
                self.use_langgraph = False
                self.langgraph_orchestrator = None

        # 🔧 A-TEAM: Initialize MetadataToolRegistry (LLM-driven tool discovery!)
        # Only initialize if metadata_provider is provided (not None)
        if metadata_provider is not None:
            # Check if it's a Mock object (has 'mock_calls' attribute)
            is_mock = hasattr(metadata_provider, 'mock_calls')

            if not is_mock:
                from ..metadata.metadata_tool_registry import MetadataToolRegistry
                self.metadata_tool_registry = MetadataToolRegistry(metadata_provider)
                logger.info(f"🔧 MetadataToolRegistry initialized - {len(self.metadata_tool_registry.tools)} tools discovered!")
            else:
                self.metadata_tool_registry = None
                logger.debug("ℹ️  MetadataToolRegistry disabled (Mock metadata_provider for testing)")
        else:
            self.metadata_tool_registry = None
            logger.debug("ℹ️  MetadataToolRegistry disabled (no metadata_provider)")
        
        # 🆕 A-TEAM: Build parameter mappings (user + config + defaults)
        self.param_mappings = self._build_param_mappings(custom_mappings)
        
        # 🆕 A-TEAM: IOManager for typed output management
        self.io_manager = IOManager()
        logger.info("📦 IOManager enabled - typed output management!")
        
        # 🔑 NEW: Smart Data Transformer with FORMAT TOOLS (ReAct + json.loads, csv, etc.)
        self.data_transformer = SmartDataTransformer()
        logger.info("🔄 SmartDataTransformer enabled - ReAct agent with format tools!")
        
        # 🔥 CRITICAL: SmartAgentSlack - Intelligent agent communication with embedded helpers
        # User insight: "Agent Slack should HAVE transformer, chunker, compressor. It's a SMART slack"
        # 🆕 A-TEAM: Now with cooperation tracking!
        self.agent_slack = SmartAgentSlack(config={}, enable_cooperation=True)
        logger.info("💬 SmartAgentSlack initialized - intelligent agent communication!")
        logger.info("   ✅ Embedded: Transformer, Chunker, Compressor")
        logger.info("   ✅ Auto-detects format needs from signatures")
        logger.info("   ✅ Manages context length automatically")
        logger.info("   🤝 Cooperation tracking enabled (for credit assignment)")
        
        # 🔑 NEW: Agentic Parameter Resolver for semantic matching
        from ..data.parameter_resolver import AgenticParameterResolver
        self.param_resolver = AgenticParameterResolver(llm=None)  # Uses dspy.settings.lm
        logger.info("🧠 AgenticParameterResolver enabled - LLM-based semantic parameter matching!")
        
        # 🔑 NEW: Data Registry for agentic data discovery
        if enable_data_registry:
            self.data_registry = DataRegistry()
            self.data_registry_tool = DataRegistryTool(self.data_registry)
            logger.info("📚 Data Registry enabled - agents can discover and retrieve data")
            
            # 🎯 NEW: Agentic Discovery Orchestrator (LLM-based registration)
            from ..data.agentic_discovery import RegistrationOrchestrator
            
            # A-Team: Pass model and config for token counting
            model_for_tokens = getattr(config, 'token_model_name', None) or getattr(config, 'model', None)
            
            self.registration_orchestrator = RegistrationOrchestrator(
                data_registry=self.data_registry,
                lm=None,  # Will use dspy.settings.lm
                model=model_for_tokens,
                config=config,
                enable_validation=True
            )
            logger.info("🎯 RegistrationOrchestrator initialized - LLM-based tagging & discovery!")
        else:
            self.data_registry = None
            self.data_registry_tool = None
            self.registration_orchestrator = None
        
        # 🤝 NEW: FeedbackChannel for agent coordination
        self.feedback_channel = FeedbackChannel()
        logger.info("📧 FeedbackChannel enabled - agents can consult each other!")

        # =====================================================================
        # 🔥 A-TEAM: Wire Axon (SmartAgentSlack) into the running swarm
        # =====================================================================
        # Before: agent_slack existed but was not used by orchestration.
        # Now: we register each actor so messages can be delivered, logged, and
        # bridged into FeedbackChannel for injection on the next execution.
        def _make_slack_callback(target_actor_name: str):
            def _callback(message):
                try:
                    # Bridge Slack message → FeedbackChannel message for context injection
                    fb = FeedbackMessage(
                        source_actor=message.from_agent,
                        target_actor=target_actor_name,
                        feedback_type=FeedbackType.RESPONSE,
                        content=str(message.data),
                        context={
                            "format": getattr(message, "format", "unknown"),
                            "size_bytes": getattr(message, "size_bytes", None),
                            "metadata": getattr(message, "metadata", {}) or {},
                            "timestamp": getattr(message, "timestamp", None),
                        },
                        requires_response=False,
                        priority=2,
                    )
                    self.feedback_channel.send(fb)
                except Exception as e:
                    logger.warning(f"⚠️ Slack callback failed for {target_actor_name}: {e}")
            return _callback
        
        # 🎯 NEW: MetaDataFetcher + SharedContext (NO MCP SERVER NEEDED!)
        # Auto-discovers @jotty_method methods and creates dspy.Tool objects directly
        from ..metadata.metadata_fetcher import MetaDataFetcher
        from ..persistence.shared_context import SharedContext
        
        self.shared_context = SharedContext()
        logger.info("🗂️  SharedContext initialized - agents share data here!")

        # MetaDataFetcher with direct dspy.Tool (no MCP server)
        # Only initialize if metadata_provider is provided
        if metadata_provider is not None:
            is_mock = hasattr(metadata_provider, 'mock_calls')
            if not is_mock:
                self.metadata_fetcher = MetaDataFetcher(metadata_provider=metadata_provider)
                logger.info("🔍 MetaDataFetcher initialized - auto-discovered @jotty_method tools!")
            else:
                self.metadata_fetcher = None
                logger.debug("ℹ️  MetaDataFetcher disabled (Mock metadata_provider for testing)")
        else:
            self.metadata_fetcher = None
            logger.debug("ℹ️  MetaDataFetcher disabled (no metadata_provider)")
        
        # 🔑 Auto-introspect actor signatures
        self.actor_signatures = {}
        self.dependency_graph_dict = {}  # Legacy dict-based graph (for signature introspection)
        
        # 🆕 LangGraph: Initialize DynamicDependencyGraph if using dynamic mode
        # Note: We'll initialize this lazily in run() method (can't use await in __init__)
        self._should_init_dependency_graph = use_langgraph and langgraph_mode == "dynamic"
        self.dependency_graph = None  # LangGraph DynamicDependencyGraph (initialized lazily)
        
        logger.info("🔍 Introspecting actor signatures...")
        
        # Wrap actors with Jotty wrapper if they have tools or validation prompts
        self.actors = {}
        for actor_config in actors:
            if self._should_wrap_actor(actor_config):
                wrapped_actor = self._wrap_actor_with_jotty(actor_config)
                # Create new config with wrapped actor
                # 🚨 CRITICAL FIX: Copy ALL fields from original config!
                new_config = ActorConfig(
                    name=actor_config.name,
                    agent=wrapped_actor,
                    architect_prompts=[],  # Already wrapped
                    auditor_prompts=[], # Already wrapped
                    architect_tools=[],
                    auditor_tools=[],
                    enabled=actor_config.enabled,
                    # 🔑 CRITICAL: Copy parameter_mappings!
                    parameter_mappings=actor_config.parameter_mappings,
                    # Copy other important fields
                    outputs=actor_config.outputs,
                    capabilities=actor_config.capabilities,
                    dependencies=actor_config.dependencies,
                    metadata=actor_config.metadata,
                    enable_architect=actor_config.enable_architect,
                    enable_auditor=actor_config.enable_auditor,
                    validation_mode=actor_config.validation_mode,
                    is_critical=actor_config.is_critical,
                    max_retries=actor_config.max_retries,
                    retry_strategy=actor_config.retry_strategy,
                    is_executor=getattr(actor_config, "is_executor", False),
                )
                self.actors[actor_config.name] = new_config
                # 🔑 Introspect the ORIGINAL actor (before wrapping)
                self._introspect_actor_signature(actor_config)
            else:
                self.actors[actor_config.name] = actor_config
                # 🔑 Introspect this actor too
                self._introspect_actor_signature(actor_config)

            # ✅ Register agent with Axon (SmartAgentSlack) so it can receive messages.
            # We pass signature when available; otherwise Axon uses safe defaults.
            try:
                actor_obj = (new_config.agent if self._should_wrap_actor(actor_config) else actor_config.agent)
                signature_obj = getattr(actor_obj, "signature", None)
                self.agent_slack.register_agent(
                    agent_name=actor_config.name,
                    signature=signature_obj if hasattr(signature_obj, "input_fields") else None,
                    callback=_make_slack_callback(actor_config.name),
                    max_context=getattr(self.config, "max_context_tokens", 16000),
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not register {actor_config.name} with SmartAgentSlack: {e}")
        
        # 🆕 LangGraph: Will initialize DynamicDependencyGraph lazily in run() method
        # (Can't use await in __init__, so we do it when needed)
        
        # Global validation prompts
        self.global_architect_path = global_architect
        self.global_auditor_path = global_auditor
        
        # Load annotations for Auditor enrichment
        self.annotations = self._load_annotations(annotations_path)
        
        # Initialize core components
        self.todo = MarkovianTODO()
        # 🧠 A-TEAM FIX: Use FULL Q-learning implementation from q_learning.py, not the simplified local one!
        self.q_predictor = NaturalLanguageQTable(self.config)
        self.context_guard = SmartContextGuard(
            max_tokens=self.config.max_context_tokens
        )
        self.policy_explorer = PolicyExplorer(self.config)
        
        # =====================================================================
        # BRAIN-INSPIRED COMPONENTS (v2.1 - A-Team Approved)
        # =====================================================================
        
        # 🧠 BRAIN-INSPIRED MEMORY: Use BrainInspiredMemoryManager (Neuralink-level neuroscience)
        self.brain = None
        if BRAIN_MEMORY_MANAGER_AVAILABLE:
            # Get consolidation interval from config
            consolidation_interval = getattr(self.config, 'consolidation_interval', 10)
            
            self.brain = BrainInspiredMemoryManager(
                sleep_interval=consolidation_interval,
                max_hippocampus_size=getattr(self.config, 'hippocampus_max_size', 100),
                max_neocortex_size=getattr(self.config, 'neocortex_max_size', 200),
                replay_threshold=getattr(self.config, 'brain_memory_threshold', 0.7),
                novelty_weight=getattr(self.config, 'brain_novelty_weight', 0.4),
                reward_weight=getattr(self.config, 'brain_reward_salience_weight', 0.3),
                frequency_weight=getattr(self.config, 'brain_frequency_weight', 0.3)
            )
            logger.info(f"🧠 BrainInspiredMemoryManager initialized (Neuralink-level neuroscience)")
            logger.info(f"   - Sharp-Wave Ripple: Consolidation every {consolidation_interval} episodes")
            logger.info(f"   - Hippocampus → Neocortex: Experience → Patterns")
            logger.info(f"   - Synaptic Pruning: Forget bad, strengthen good")
        else:
            logger.warning("⚠️  BrainInspiredMemoryManager not available - memory disabled")
        
        # Episode counter for brain consolidation tracking
        self.episode_counter = 0
        
        # LEGACY: Keep BrainStateMachine for advanced features
        if BRAIN_MODES_AVAILABLE:
            brain_config = BrainModeConfig(
                enabled=True,
                sleep_interval=3,  # 🔧 A-TEAM: Consolidate every 3 episodes (prevent memory buildup!)
                sharp_wave_ripple=True,
                hippocampal_filtering=True,
                reward_salience_weight=self.config.brain_reward_salience_weight,  # 🔧 STANFORD FIX
                novelty_weight=self.config.brain_novelty_weight,  # 🔧 STANFORD FIX
                goal_relevance_weight=self.config.brain_goal_relevance_weight,  # 🔧 STANFORD FIX
                memory_threshold=self.config.brain_memory_threshold,  # 🔧 STANFORD FIX
                prune_threshold=self.config.brain_prune_threshold,  # 🔧 STANFORD FIX
                strengthen_threshold=self.config.brain_strengthen_threshold,  # 🔧 STANFORD FIX
                max_prune_percentage=self.config.brain_max_prune_percentage  # 🔧 STANFORD FIX
            )
            self.brain_state = BrainStateMachine(brain_config)
            self.agent_abstractor = AgentAbstractor(brain_config)
        else:
            self.brain_state = None
            self.agent_abstractor = None
        
        # Predictive MARL: Predict other agents, learn from divergence
        if PREDICTIVE_MARL_AVAILABLE:
            self.trajectory_predictor = LLMTrajectoryPredictor(
                self.config,
                horizon=5  # Predict 5 steps ahead
            )
            self.divergence_memory = DivergenceMemory(self.config)
            self.cooperative_credit = CooperativeCreditAssigner(self.config)
            logger.info("🔮 Predictive MARL components initialized")
        else:
            self.trajectory_predictor = None
            self.divergence_memory = None
            self.cooperative_credit = None
        
        # =====================================================================
        # 🧠 Q-LEARNING WITH NEUROCHUNK TIERED MEMORY (v2.4 - A-Team Approved)
        # =====================================================================
        if Q_LEARNING_AVAILABLE:
            self.q_learner = NaturalLanguageQTable(self.config)
            logger.info("🧠 Q-Learning with NeuroChunk Memory initialized")
            logger.info("   - Tier 1: Working Memory (always in context)")
            logger.info("   - Tier 2: Semantic Clusters (retrieval-based)")
            logger.info("   - Tier 3: Long-term Archive (causal impact pruning)")
        else:
            self.q_learner = None
            logger.warning("⚠️  Q-Learning not available")
        
        # TD(λ) Learning (Temporal Difference with Eligibility Traces)
        # Only initialize if RL is enabled
        if self.config.enable_rl and TD_LAMBDA_AVAILABLE:
            from .learning import AdaptiveLearningRate
            adaptive_lr = AdaptiveLearningRate(self.config)
            self.td_learner = TDLambdaLearner(self.config, adaptive_lr)
            # Log based on verbosity setting
            if getattr(self.config, 'rl_verbosity', 'normal') == 'verbose':
                logger.info("📊 TD(λ) Learning initialized (eligibility traces)")
            else:
                logger.debug("📊 TD(λ) Learning initialized (eligibility traces)")
        else:
            self.td_learner = None
            if self.config.enable_rl and not TD_LAMBDA_AVAILABLE:
                # Only warn if RL is enabled but TD(λ) is not available
                logger.warning("⚠️  TD(λ) Learning not available (install required dependencies)")
            else:
                logger.debug("ℹ️  TD(λ) Learning disabled (enable_rl=False)")
        
        # =====================================================================
        # 🔥 TIMEOUT & CIRCUIT BREAKER (v2.5 - A-Team Production Resilience)
        # =====================================================================
        from ..utils.timeouts import (
            CircuitBreaker, CircuitBreakerConfig,
            DeadLetterQueue, AdaptiveTimeout
        )
        
        if self.config.enable_circuit_breakers:
            # LLM Circuit Breaker
            self.llm_circuit_breaker = CircuitBreaker(
                CircuitBreakerConfig(
                    name="llm_calls",
                    failure_threshold=self.config.llm_circuit_failure_threshold,
                    timeout=self.config.llm_circuit_timeout,
                    success_threshold=self.config.llm_circuit_success_threshold
                )
            )
            # Tool Circuit Breaker
            self.tool_circuit_breaker = CircuitBreaker(
                CircuitBreakerConfig(
                    name="tool_calls",
                    failure_threshold=self.config.tool_circuit_failure_threshold,
                    timeout=self.config.tool_circuit_timeout,
                    success_threshold=self.config.tool_circuit_success_threshold
                )
            )
            logger.info("🔌 Circuit Breakers initialized")
            logger.info(f"   - LLM: {self.config.llm_circuit_failure_threshold} failures → open")
            logger.info(f"   - Tool: {self.config.tool_circuit_failure_threshold} failures → open")
        else:
            self.llm_circuit_breaker = None
            self.tool_circuit_breaker = None
            logger.info("⚠️  Circuit Breakers disabled")
        
        # Adaptive Timeout
        if self.config.enable_adaptive_timeouts:
            self.adaptive_timeout = AdaptiveTimeout(
                initial=self.config.initial_timeout,
                percentile=self.config.timeout_percentile,
                min_timeout=self.config.min_timeout,
                max_timeout=self.config.max_timeout
            )
            logger.info(f"⏱️  Adaptive Timeout initialized ({self.config.timeout_percentile}th percentile)")
        else:
            self.adaptive_timeout = None
            logger.info("⚠️  Adaptive Timeout disabled")
        
        # Dead Letter Queue
        if self.config.enable_dead_letter_queue:
            self.dead_letter_queue = DeadLetterQueue(
                max_size=self.config.dlq_max_size
            )
            logger.info(f"📮 Dead Letter Queue initialized (max_size={self.config.dlq_max_size})")
        else:
            self.dead_letter_queue = None
            logger.info("⚠️  Dead Letter Queue disabled")
        # =====================================================================
        
        self.swarm_learner = SwarmLearner(self.config)
        
        # Shared memory (across all actors)
        self.shared_memory = HierarchicalMemory(
            config=self.config,
            agent_name="SwarmShared"
        )
        
        # Per-actor local memories
        self.local_memories: Dict[str, HierarchicalMemory] = {}
        for name in self.actors:
            self.local_memories[name] = HierarchicalMemory(
                config=self.config,
                agent_name=name
            )
        
        # State
        self.episode_count = 0
        self.total_episodes = 0
        self.trajectory: List[Dict] = []
        
        # =====================================================================
        # PERSISTENCE MANAGER (v9.1 - Complete Integration)
        # =====================================================================
        self.persistence_manager: Optional['Vault'] = None
        # Will be initialized in run() when output directory is known

        # =====================================================================
        # 🆕 REFACTORING: Parameter Resolver Component (Fix #1)
        # =====================================================================
        # Extract parameter resolution logic to dedicated component
        self.parameter_resolver = ParameterResolver(
            io_manager=self.io_manager,
            param_resolver=self.param_resolver,
            metadata_fetcher=self.metadata_fetcher,
            actors=self.actors,
            actor_signatures=self.actor_signatures,
            param_mappings=self.param_mappings,
            data_registry=self.data_registry,
            registration_orchestrator=self.registration_orchestrator,
            data_transformer=self.data_transformer,
            shared_context=self.shared_context,
            config=self.config
        )
        logger.info("✅ ParameterResolver component initialized")

        # =====================================================================
        # 🆕 REFACTORING: Tool Manager Component (Fix #1)
        # =====================================================================
        # Extract tool management logic to dedicated component
        self.tool_manager = ToolManager(
            metadata_tool_registry=self.metadata_tool_registry,
            data_registry_tool=self.data_registry_tool,
            metadata_fetcher=self.metadata_fetcher,
            config=self.config
        )
        logger.info("✅ ToolManager component initialized")

        # =====================================================================
        # 🆕 REFACTORING: State Manager Component (Fix #1)
        # =====================================================================
        # Extract state management and output registration logic to dedicated component
        self.state_manager = StateManager(
            io_manager=self.io_manager,
            data_registry=self.data_registry,
            metadata_provider=self.metadata_provider,
            context_guard=self.context_guard,
            shared_context=self.shared_context,
            todo=self.todo,
            trajectory=self.trajectory,
            config=self.config,
            actors=self.actors,
            actor_signatures=self.actor_signatures,
            metadata_fetcher=self.metadata_fetcher,
            param_resolver=self.param_resolver
        )
        logger.info("✅ StateManager component initialized")

        logger.info(f"🚀 MultiAgentsOrchestrator initialized with {len(actors)} actors")
        logger.info("🎯 All components initialized: ParameterResolver, ToolManager, StateManager")
    
    def _should_wrap_actor(self, actor_config: ActorConfig) -> bool:
        """
        Determine if an actor needs to be wrapped with JOTTY.
        
        Wrapping is needed if:
        - Actor has validation prompts (architect_prompts or auditor_prompts)
        - Actor has tools (architect_tools or auditor_tools)
        """
        has_validation = (
            actor_config.architect_prompts or 
            actor_config.auditor_prompts
        )
        has_tools = (
            actor_config.architect_tools or 
            actor_config.auditor_tools
        )
        return has_validation or has_tools
    
    def _get_auto_discovered_dspy_tools(self) -> List[Any]:
        """
        Get ALL auto-discovered metadata tools as dspy.Tool objects.
        
        🔥 A-TEAM CONSENSUS SOLUTION (Post-Debate):
        - Individual DSPy tools (one per metadata method) ✅
        - Smart parameter resolution (4-level fallback) ✅
        - Caching via SharedScratchpad ✅
        - Enhanced descriptions for LLM reasoning ✅
        - FULLY GENERIC - works for ANY tool! ✅
        
        Returns:
            List of dspy.Tool objects for all @jotty_method decorated methods
        """
        import dspy
        import json
        
        if not hasattr(self, 'metadata_tool_registry'):
            logger.warning("⚠️  No metadata_tool_registry found, returning empty tool list")
            return []
        
        tools = []
        tool_names = self.metadata_tool_registry.list_tools()
        
        logger.info(f"🔧 Creating {len(tool_names)} individual DSPy tools with smart param resolution...")
        
        for tool_name in tool_names:
            tool_info = self.metadata_tool_registry.get_tool_info(tool_name)
            
            # Create closure to capture tool_name and tool_info
            def make_smart_tool_func(tname, tinfo):
                def smart_tool_func(*args, **kwargs):
                    """
                    🔥 A-TEAM SMART WRAPPER with 4-level parameter resolution:
                    1. Explicit override (user provides param)
                    2. IOManager auto-resolution (exact name match from actor outputs)
                    3. SharedContext auto-resolution (exact name match from global data)
                    4. Type-based resolution (generic! works for ANY type)
                    """
                    try:
                        # Get parameter specs
                        signature = tinfo.get('signature', {})
                        params_spec = signature.get('parameters', {})
                        
                        # Build final kwargs with smart resolution
                        final_kwargs = dict(kwargs)
                        
                        for param_name, param_info in params_spec.items():
                            if param_name in final_kwargs:
                                # Level 1: Explicit override (highest priority)
                                logger.debug(f"✅ {tname}({param_name}): Using explicit value")
                                continue
                            
                            # Level 2: Exact match in IOManager (actor outputs)
                            resolved_val = self._resolve_param_from_iomanager(param_name)
                            if resolved_val is not None:
                                final_kwargs[param_name] = resolved_val
                                logger.info(f"✅ {tname}({param_name}): Auto-resolved from IOManager")
                                continue
                            
                            # Level 3: Exact match in SharedContext (global data)
                            if hasattr(self, 'shared_context') and self.shared_context.has(param_name):
                                resolved_val = self.shared_context.get(param_name)
                                final_kwargs[param_name] = resolved_val
                                logger.info(f"✅ {tname}({param_name}): Auto-resolved from SharedContext")
                                continue
                            
                            # Level 4: Type-based resolution (GENERIC!)
                            param_type = param_info.get('annotation', '')
                            if param_type:
                                resolved_val = self._resolve_param_by_type(param_name, param_type)
                                if resolved_val is not None:
                                    final_kwargs[param_name] = resolved_val
                                    logger.info(f"✅ {tname}({param_name}): Auto-resolved by type ({param_type})")
                                    continue
                            
                            # Not resolved - let tool fail with clear error
                            if param_info.get('required', True):
                                logger.warning(f"⚠️  {tname}({param_name}): Required parameter not resolved")
                        
                        # Check cache first (via SharedScratchpad)
                        result = self._call_tool_with_cache(tname, **final_kwargs)
                        
                        # Ensure result is string (DSPy tools return strings)
                        if not isinstance(result, str):
                            return json.dumps(result, indent=2)
                        return result
                        
                    except Exception as e:
                        error_msg = self._build_helpful_error_message(tname, tinfo, e)
                        logger.error(f"❌ Tool {tname} error: {e}")
                        return json.dumps({"error": error_msg})
                
                return smart_tool_func
            
            # Build enhanced tool description for LLM reasoning
            tool_desc = self._build_enhanced_tool_description(tool_name, tool_info)
            
            # Create dspy.Tool with smart wrapper
            tool = dspy.Tool(
                func=make_smart_tool_func(tool_name, tool_info),
                name=tool_name,
                desc=tool_desc
            )
            
            # 🔥 A-TEAM CRITICAL FIX: Attach Val agent flags to dspy.Tool!
            tool._jotty_for_architect = tool_info.get('for_architect', False)
            tool._jotty_for_auditor = tool_info.get('for_auditor', False)
            
            tools.append(tool)
            logger.debug(f"  ✅ {tool_name} (architect={tool._jotty_for_architect}, auditor={tool._jotty_for_auditor})")
        
        logger.info(f"✅ Auto-discovered {len(tools)} tools for Val agents")
        for tool in tools:
            logger.info(f"   - {tool.name}: {tool.desc[:80]}...")
        
        return tools
    
    def _resolve_param_from_iomanager(self, param_name: str) -> Any:
        """
        Resolve parameter from IOManager (previous actor outputs).
        
        🔥 A-TEAM: Level 2 resolution - searches actor outputs for param
        
        Args:
            param_name: Name of parameter to resolve (e.g., 'tables')
        
        Returns:
            Resolved value or None if not found
        """
        if not hasattr(self, 'io_manager') or not self.io_manager:
            return None
        
        # Try exact name match in all actor outputs
        all_outputs = self.io_manager.get_all_outputs()
        for actor_name, output in all_outputs.items():
            if hasattr(output, 'output_fields') and isinstance(output.output_fields, dict):
                if param_name in output.output_fields:
                    value = output.output_fields[param_name]
                    logger.debug(f"   📦 Found '{param_name}' in {actor_name} output")
                    return value
        
        return None
    
    def _resolve_param_by_type(self, param_name: str, param_type: Any) -> Any:
        """
        Resolve parameter by type matching in IOManager.
        
        🔥 A-TEAM: Level 4 resolution - GENERIC type-based matching!
        Works for ANY type, not just hardcoded names.
        
        Args:
            param_name: Name of parameter (for logging)
            param_type: Type annotation (can be type object or string)
        
        Returns:
            Resolved value or None if not found
        """
        if not hasattr(self, 'io_manager') or not self.io_manager:
            return None
        
        # 🔥 A-TEAM FIX: Convert type annotation to string for comparison
        # Handle both type objects and string annotations
        type_str = str(param_type) if not isinstance(param_type, str) else param_type
        
        # Parse type (simplified - handles common cases)
        target_type = None
        if 'List' in type_str or 'list' in type_str:
            target_type = list
        elif 'Dict' in type_str or 'dict' in type_str:
            target_type = dict
        elif type_str in ['str', 'string'] or 'str' in type_str:
            target_type = str
        elif type_str in ['int', 'integer'] or 'int' in type_str:
            target_type = int
        elif type_str in ['float'] or 'float' in type_str:
            target_type = float
        elif type_str in ['bool', 'boolean'] or 'bool' in type_str:
            target_type = bool
        
        if not target_type:
            return None
        
        # Search all actor outputs for matching type
        all_outputs = self.io_manager.get_all_outputs()
        for actor_name, output in all_outputs.items():
            if hasattr(output, 'output_fields') and isinstance(output.output_fields, dict):
                for field_name, field_value in output.output_fields.items():
                    if isinstance(field_value, target_type):
                        logger.debug(f"   🎯 Type match for '{param_name}': found {field_name} ({type(field_value).__name__}) in {actor_name}")
                        return field_value
        
        return None
    
    def _call_tool_with_cache(self, tool_name: str, **kwargs) -> Any:
        """
        Call tool with caching via SharedScratchpad.
        
        🔥 A-TEAM: Prevents duplicate tool calls across Val agents!
        
        Args:
            tool_name: Name of tool to call
            **kwargs: Parameters for tool
        
        Returns:
            Tool result (cached if available)
        """
        import json
        
        # Create cache key
        cache_key = f"tool_call:{tool_name}:{json.dumps(kwargs, sort_keys=True)}"
        
        # Check cache
        if hasattr(self, 'shared_scratchpad') and self.shared_scratchpad:
            if cache_key in self.shared_scratchpad:
                logger.debug(f"💾 Cache HIT: {tool_name}({list(kwargs.keys())})")
                return self.shared_scratchpad[cache_key]
        
        # Call actual tool
        logger.debug(f"📞 Calling {tool_name}({list(kwargs.keys())})")
        result = self.metadata_tool_registry.call_tool(tool_name, **kwargs)
        
        # Store in cache
        if hasattr(self, 'shared_scratchpad') and self.shared_scratchpad:
            self.shared_scratchpad[cache_key] = result
        
        return result
    
    def _build_helpful_error_message(self, tool_name: str, tool_info: Dict, error: Exception) -> str:
        """
        Build helpful error message for Val agents when tool call fails.
        
        🔥 A-TEAM: Shows available data and how to fix the issue!
        
        Args:
            tool_name: Name of tool that failed
            tool_info: Tool metadata
            error: Exception that occurred
        
        Returns:
            Helpful error message string
        """
        import json
        
        # Extract missing parameters if it's a TypeError
        error_str = str(error)
        
        # Build helpful message
        msg_parts = [f"❌ {tool_name}() failed: {error_str}"]
        
        # If missing parameters, show available data
        if 'missing' in error_str.lower() or 'required' in error_str.lower():
            msg_parts.append("\n📦 AVAILABLE DATA (IOManager):")
            
            if hasattr(self, 'io_manager') and self.io_manager:
                all_outputs = self.io_manager.get_all_outputs()
                for actor_name, output in all_outputs.items():
                    if hasattr(output, 'output_fields'):
                        fields = list(output.output_fields.keys()) if isinstance(output.output_fields, dict) else []
                        msg_parts.append(f"  • {actor_name}: {fields}")
            else:
                msg_parts.append("  (No IOManager available)")
            
            msg_parts.append("\n💡 TIP: You can provide parameters explicitly in your tool call.")
            msg_parts.append(f"   Example: {tool_name}(param_name=value)")
        
        return "\n".join(msg_parts)
    
    def _build_enhanced_tool_description(self, tool_name: str, tool_info: Dict) -> str:
        """
        Build enhanced tool description for LLM reasoning.
        
        🔥 A-TEAM: Includes parameters, when to use, auto-resolution hints!
        
        Args:
            tool_name: Name of tool
            tool_info: Tool metadata
        
        Returns:
            Enhanced description string
        """
        desc_parts = [tool_info['desc']]
        
        # Add "when to use"
        desc_parts.append(f"\n🎯 WHEN TO USE:\n{tool_info['when']}")
        
        # Add parameters
        signature = tool_info.get('signature', {})
        params = signature.get('parameters', {})
        
        if params:
            desc_parts.append("\n📋 PARAMETERS:")
            for param_name, param_info in params.items():
                ptype = param_info.get('annotation', 'Any')
                required = 'REQUIRED' if param_info.get('required', True) else 'OPTIONAL'
                pdesc = param_info.get('desc', '')
                default = param_info.get('default')
                
                if default is not None:
                    desc_parts.append(f"  • {param_name} ({ptype}) [{required}]: {pdesc} (default: {default})")
                else:
                    desc_parts.append(f"  • {param_name} ({ptype}) [{required}]: {pdesc}")
        else:
            desc_parts.append("\n📋 PARAMETERS: None (simple getter)")
        
        # Add returns
        returns = tool_info.get('returns', 'str')
        desc_parts.append(f"\n↩️  RETURNS: {returns}")
        
        # Add auto-resolution hint
        if params:
            desc_parts.append("""
🤖 AUTO-RESOLUTION:
Parameters you don't provide will be auto-resolved from:
1. Previous actor outputs (IOManager) - exact name match
2. Shared global context - exact name match
3. Type-based matching - finds any matching type

You can override any parameter by providing it explicitly.""")
        
        return "\n".join(desc_parts)
    
    def _get_architect_tools(self, all_tools: List[Any]) -> List[Any]:
        """
        Filter tools marked for Architect by metadata manager.
        
        🔥 A-TEAM CRITICAL FIX: NO HARDCODING!
        
        This method checks the `for_architect` flag set by @jotty_method decorator.
        The METADATA MANAGER (user-provided, domain-specific) decides which tools
        are for Architect, NOT JOTTY core (which is generic).
        
        This keeps JOTTY domain-agnostic and reusable for ANY use case!
        """
        filtered = []
        for tool in all_tools:
            # 🔥 A-TEAM FIX: Tools are dictionaries from MetadataToolRegistry
            if isinstance(tool, dict) and tool.get('for_architect', False):
                filtered.append(tool)
            # Legacy support for function objects
            elif hasattr(tool, 'func') and hasattr(tool.func, '_jotty_for_architect'):
                if tool.func._jotty_for_architect:
                    filtered.append(tool)
            elif hasattr(tool, '_jotty_for_architect'):
                if tool._jotty_for_architect:
                    filtered.append(tool)
        
        logger.info(f"🔍 Architect tools: {len(filtered)}/{len(all_tools)} tools (marked by metadata manager)")
        for tool in filtered:
            tool_name = tool.get('name') if isinstance(tool, dict) else (tool.name if hasattr(tool, 'name') else tool.__name__)
            logger.debug(f"   ✅ {tool_name}")
        
        return filtered
    
    def _get_auditor_tools(self, all_tools: List[Any]) -> List[Any]:
        """
        Filter tools marked for Auditor by metadata manager.
        
        🔥 A-TEAM CRITICAL FIX: NO HARDCODING!
        
        This method checks the `for_auditor` flag set by @jotty_method decorator.
        The METADATA MANAGER (user-provided, domain-specific) decides which tools
        are for Auditor, NOT JOTTY core (which is generic).
        
        This keeps JOTTY domain-agnostic and reusable for ANY use case!
        """
        filtered = []
        for tool in all_tools:
            # 🔥 A-TEAM FIX: Tools are dictionaries from MetadataToolRegistry
            if isinstance(tool, dict) and tool.get('for_auditor', False):
                filtered.append(tool)
            # Legacy support for function objects
            elif hasattr(tool, 'func') and hasattr(tool.func, '_jotty_for_auditor'):
                if tool.func._jotty_for_auditor:
                    filtered.append(tool)
            elif hasattr(tool, '_jotty_for_auditor'):
                if tool._jotty_for_auditor:
                    filtered.append(tool)
        
        logger.info(f"✅ Auditor tools: {len(filtered)}/{len(all_tools)} tools (marked by metadata manager)")
        for tool in filtered:
            tool_name = tool.get('name') if isinstance(tool, dict) else (tool.name if hasattr(tool, 'name') else tool.__name__)
            logger.debug(f"   ✅ {tool_name}")
        
        return filtered
    
    def _wrap_actor_with_jotty(self, actor_config: ActorConfig):
        """
        Wrap an actor with Jotty wrapper for validation and tool support.
        
        This enables per-actor validation and tools in Conductor.
        
        🔥 A-TEAM PHASE 2 FIX: Separate tool sets for Architect vs Auditor!
        - Architect gets exploration tools (4-5 tools)
        - Auditor gets verification tools (6-7 tools)
        - NO overlap of inappropriate tools!
        """
        from .jotty_core import JottyCore
        
        # Get all available tools
        all_tools = self._get_auto_discovered_dspy_tools()
        
        # 🔥 A-TEAM: Separate tool sets for Architect vs Auditor
        architect_tools = actor_config.architect_tools
        if architect_tools is None or len(architect_tools) == 0:
            architect_tools = self._get_architect_tools(all_tools)
            logger.info(f"✅ Auto-filtered {len(architect_tools)} exploration tools for Architect")
        
        auditor_tools = actor_config.auditor_tools
        if auditor_tools is None or len(auditor_tools) == 0:
            auditor_tools = self._get_auditor_tools(all_tools)
            logger.info(f"✅ Auto-filtered {len(auditor_tools)} verification tools for Auditor")
        
        logger.info(f"🔧 Wrapping actor '{actor_config.name}' with JOTTY")
        logger.info(f"  📝 Architect prompts: {len(actor_config.architect_prompts)}")
        logger.info(f"  📝 Auditor prompts: {len(actor_config.auditor_prompts)}")
        logger.info(f"  🔍 Architect tools: {len(architect_tools)} (exploration)")
        logger.info(f"  ✅ Auditor tools: {len(auditor_tools)} (verification)")
        
        return JottyCore(
            actor=actor_config.agent,
            architect_prompts=actor_config.architect_prompts or [],
            auditor_prompts=actor_config.auditor_prompts or [],
            architect_tools=architect_tools or [],
            auditor_tools=auditor_tools or [],
            config=self.config,
            actor_config=actor_config,  # 🔑 A-TEAM: Pass actor_config for validation control
            shared_context=self.shared_context  # 🔥 A-TEAM: Pass SharedContext for metadata access
        )
    
    def _load_annotations(self, path: Optional[str]) -> Dict[str, Any]:
        """Load annotations for validation enrichment."""
        if not path:
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load annotations: {e}")
            return {}
    
    def run_sync(
        self,
        goal: str,
        max_iterations: int = 100,
        target_reward: float = 1.0,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> SwarmResult:
        """
        Synchronous wrapper for run().
        
        Handles asyncio automatically - works in both sync and async contexts.
        
        Returns:
            SwarmResult with typed outputs
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - can't use asyncio.run()
            logger.warning("⚠️  run_sync() called from async context! Use await run() instead.")
            # Create a task and return it (caller must await)
            return asyncio.create_task(self.run(goal, max_iterations, target_reward, output_dir, **kwargs))
        except RuntimeError:
            # No running loop - create one
            return asyncio.run(self.run(goal, max_iterations, target_reward, output_dir, **kwargs))
    
    async def enqueue_goal(
        self,
        goal: str,
        priority: int = 5,
        agent_type: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Enqueue a goal as a task in the task queue
        
        Args:
            goal: Goal description
            priority: Task priority (1-5, higher = more important)
            agent_type: Agent type to use (claude, cursor, opencode)
            **kwargs: Additional task metadata
        
        Returns:
            task_id if task_queue is enabled, None otherwise
        """
        if not self.task_queue:
            logger.warning("⚠️  Task queue not enabled. Call Conductor with task_queue parameter.")
            return None
        
        from ..queue.task import Task
        
        # Create task
        task_id = await self.task_queue.create_task(
            title=goal[:100],  # Truncate if too long
            description=goal,
            priority=priority,
            category=kwargs.get('category', ''),
            context_files=kwargs.get('context_files'),
            status='pending',
            agent_type=agent_type,
            **{k: v for k, v in kwargs.items() if k not in ['category', 'context_files']}
        )
        
        if task_id:
            logger.info(f"📥 Enqueued goal as task: {task_id} (priority={priority})")
        
        return task_id
    
    async def process_queue(
        self,
        max_tasks: Optional[int] = None,
        max_concurrent: int = 3
    ):
        """
        Process tasks from queue using Conductor
        
        Args:
            max_tasks: Maximum number of tasks to process (None = unlimited)
            max_concurrent: Maximum concurrent task executions
        """
        if not self.task_queue:
            logger.warning("⚠️  Task queue not enabled. Call Conductor with task_queue parameter.")
            return
        
        from ..queue.queue_manager import TaskQueueManager
        
        manager = TaskQueueManager(
            conductor=self,
            task_queue=self.task_queue,
            max_concurrent=max_concurrent
        )
        
        if max_tasks:
            # Process limited number of tasks
            processed = 0
            while processed < max_tasks:
                task = await self.task_queue.dequeue()
                if not task:
                    await asyncio.sleep(1)
                    continue
                
                await manager._process_task(task)
                processed += 1
        else:
            # Process indefinitely
            await manager.start()
    
    async def run(
        self, 
        goal: str, 
        max_iterations: int = 100,
        target_reward: float = 1.0,
        output_dir: Optional[str] = None,
        mode: Optional[str] = None,  # Override: "dynamic" or "static"
        agent_order: Optional[List[str]] = None,  # Override: for static mode
        **kwargs
    ) -> SwarmResult:
        """
        Run the swarm until goal achieved or max iterations.
        
        Explores and retries until all Auditors pass (full reward).
        
        Args:
            goal: High-level goal to achieve
            max_iterations: Max iterations
            target_reward: Target reward to achieve
            output_dir: Directory for persistence
            mode: Override mode ("dynamic" or "static")
            agent_order: Override agent order (for static mode)
            **kwargs: Additional context
        
        Returns:
            SwarmResult with typed outputs and metadata
        """
        # Use LangGraph if enabled
        if self.use_langgraph and self.langgraph_orchestrator:
            # Initialize dependency graph if needed (lazy initialization)
            if (mode or self.langgraph_mode) == "dynamic" and not self.dependency_graph:
                from .dynamic_dependency_graph import DynamicDependencyGraph
                self.dependency_graph = DynamicDependencyGraph()
                # Add agents to dependency graph
                actors_dict = self.actors
                actors_list = list(actors_dict.values()) if isinstance(actors_dict, dict) else actors_dict
                for actor_config in actors_list:
                    await self.dependency_graph.add_task(actor_config.name)
                    # Add dependencies if specified
                    if hasattr(actor_config, 'dependencies') and actor_config.dependencies:
                        for dep in actor_config.dependencies:
                            await self.dependency_graph.add_dependency(dep, actor_config.name)
                logger.info("📊 DynamicDependencyGraph initialized")
            
            # Override mode/order if provided
            if mode or agent_order:
                try:
                    from .langgraph_orchestrator import LangGraphOrchestrator, GraphMode
                    actual_mode = GraphMode.DYNAMIC if (mode or self.langgraph_mode) == "dynamic" else GraphMode.STATIC
                    actual_order = agent_order or self.agent_order
                    
                    orchestrator = LangGraphOrchestrator(
                        conductor=self,
                        mode=actual_mode,
                        agent_order=actual_order
                    )
                except ImportError:
                    raise ImportError("langgraph is required for LangGraph orchestration. Install with: pip install langgraph")
            else:
                orchestrator = self.langgraph_orchestrator
            
            # Execute via LangGraph
            result = await orchestrator.run(goal, context=kwargs, max_iterations=max_iterations)
            
            # Convert to SwarmResult format
            # Aggregate output from all agent results
            agent_results = result.get("results", {})
            aggregated_output = result.get("aggregated_output", "")
            
            if not aggregated_output and agent_results:
                # Build aggregated output from results
                output_parts = []
                for name, res in agent_results.items():
                    if hasattr(res, 'output'):
                        output_parts.append(f"## {name}\n{res.output}")
                    elif isinstance(res, dict):
                        output_parts.append(f"## {name}\n{res.get('output', '')}")
                    else:
                        output_parts.append(f"## {name}\n{str(res)}")
                aggregated_output = "\n\n".join(output_parts)
            
            actual_mode_value = actual_mode.value if (mode or agent_order) else self.langgraph_mode
            
            # Convert agent_results to ActorOutput format for SwarmResult
            from ..data.io_manager import ActorOutput
            actor_outputs = {}
            for name, res in agent_results.items():
                if hasattr(res, 'output'):
                    output_fields = {"output": res.output}
                elif isinstance(res, dict):
                    output_fields = res
                else:
                    output_fields = {"output": str(res)}
                
                actor_outputs[name] = ActorOutput(
                    actor_name=name,
                    output_fields=output_fields,
                    success=getattr(res, 'success', True) if not isinstance(res, dict) else res.get('success', True)
                )
            
            return SwarmResult(
                success=result.get("success", True),
                final_output=aggregated_output or "",
                actor_outputs=actor_outputs,
                trajectory=[],
                metadata={
                    "langgraph_mode": actual_mode_value,
                    "completed_agents": result.get("completed_agents", []),
                    "stream_events": result.get("stream_events", []),
                    "agent_results": agent_results,
                },
                error=result.get("error")
            )
        
        # Legacy execution (existing code)
        self.episode_count += 1
        start_time = time.time()

        # ⏱️  Enable profiling if configured
        if getattr(self.config, 'enable_profiling', False):
            from ..utils.profiler import enable_profiling, reset_profiling, set_output_dir
            enable_profiling()
            reset_profiling()
            logger.info("⏱️  Profiling enabled")

        # ✅ Ensure we always have an output directory when persistence/logging is enabled.
        # This keeps RUN_DIR non-null and standardizes artifacts under ./outputs/run_YYYYMMDD_HHMMSS/
        if output_dir is None:
            try:
                base = getattr(self.config, "output_base_dir", "./outputs")
                create_run = getattr(self.config, "create_run_folder", True)
                if create_run:
                    ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
                    output_dir = str(Path(base) / ts)
                else:
                    output_dir = str(Path(base))
            except Exception:
                output_dir = "./outputs"

        # 📁 Setup file logging if enabled
        if getattr(self.config, 'enable_beautified_logs', False) or getattr(self.config, 'enable_debug_logs', False):
            try:
                from ..utils.file_logger import setup_file_logging
                setup_file_logging(
                    output_dir=output_dir,
                    enable_beautified=getattr(self.config, 'enable_beautified_logs', True),
                    enable_debug=getattr(self.config, 'enable_debug_logs', True),
                    log_level=getattr(self.config, 'log_level', 'INFO')
                )
            except Exception as e:
                logger.warning(f"⚠️  Failed to setup file logging: {e}")

        # ⏱️  Initialize profiling report with output directory
        if getattr(self.config, 'enable_profiling', False):
            try:
                from ..utils.profiler import set_output_dir
                set_output_dir(output_dir)
                logger.info(f"⏱️  Profiling report will be saved to: {output_dir}/profiling/")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize profiling report: {e}")

        # 🔥 A-TEAM CRITICAL FIX: Store goal + current date/time + kwargs in SharedContext for ALL actors
        if hasattr(self, 'shared_context') and self.shared_context:
            # Store goal/query
            self.shared_context.set('goal', goal)
            self.shared_context.set('query', goal)  # Also map to 'query' for first actor
            logger.info(f"✅ Stored 'goal' and 'query' in SharedContext: {goal[:100]}...")

            # 🔥 NEW: Store all kwargs in SharedContext for parameter resolution
            # This makes run() kwargs available to all actors for parameter resolution
            for key, value in kwargs.items():
                if not key.startswith('_'):  # Skip internal parameters
                    self.shared_context.set(key, value)
                    logger.info(f"✅ Stored '{key}' in SharedContext for parameter resolution")

            # 🔥 A-TEAM: Store current date/time for ALL agents (prevents date hallucinations!)
            try:
                from datetime import datetime
                from zoneinfo import ZoneInfo
                current_dt = datetime.now(ZoneInfo("Asia/Kolkata"))
                current_date = current_dt.strftime("%Y-%m-%d")
                current_datetime = current_dt.strftime("%Y-%m-%d %H:%M:%S")

                self.shared_context.set('current_date', current_date)
                self.shared_context.set('current_datetime', current_datetime)
                logger.info(f"✅ Stored current_date={current_date}, current_datetime={current_datetime}")
            except Exception:
                # Fallback to basic datetime if zoneinfo not available
                from datetime import datetime
                current_dt = datetime.now()
                current_date = current_dt.strftime("%Y-%m-%d")
                current_datetime = current_dt.strftime("%Y-%m-%d %H:%M:%S")

                self.shared_context.set('current_date', current_date)
                self.shared_context.set('current_datetime', current_datetime)
                logger.info(f"✅ Stored current_date={current_date}, current_datetime={current_datetime} (UTC)")
                logger.warning("⚠️  Could not import zoneinfo, using UTC")
        
        # 🎯 PHASE 1: Proactive Metadata Fetch (if fetcher available)
        if self.metadata_fetcher and hasattr(self.metadata_fetcher, 'fetch'):
            logger.info("=" * 80)
            logger.info("🎯 PHASE 1: Direct Metadata Fetch (NO ReAct guessing!)")
            logger.info("=" * 80)
            
            try:
                # 🔥 A-TEAM CRITICAL FIX (User Insight): Just call ALL metadata methods directly!
                # NO ReAct agent, NO guessing, NO missing data!
                # User question: "Why is react agent fetching when it's already in metadata manager?"
                # Answer: YOU'RE RIGHT! We should just call them directly!
                fetched_data = self._fetch_all_metadata_directly()
                
                # 🔥 A-TEAM FIX: Store metadata in SEPARATE namespace to avoid confusion with actor outputs!
                # Metadata = Reference data (for context/prompts)
                # Actor outputs = Execution data (for parameter flow)
                self.shared_context.set('metadata', fetched_data)
                
                logger.info(f"✅ Fetched and stored {len(fetched_data)} metadata items in SharedContext['metadata']")
                logger.info(f"   Keys: {list(fetched_data.keys())}")
                
                # 🔥 A-TEAM CRITICAL FIX: ENRICH BUSINESS TERMS WITH PARSED FILTERS!
                # This bridges metadata (context) with parameter flow (actor inputs)
                self._enrich_business_terms_with_filters(fetched_data)
            except Exception as e:
                logger.warning(f"⚠️  Metadata fetch failed: {e}")
                logger.warning("   Continuing without proactive metadata fetch")
                import traceback
                traceback.print_exc()
            
            logger.info("=" * 80)
        
        # Initialize persistence manager (NO HARDCODING - takes output_dir dynamically)
        if output_dir and not self.persistence_manager:
            self.persistence_manager = Vault(
                base_output_dir=output_dir,
                auto_save_interval=getattr(self.config, 'save_interval', 1)  # ✅ FIX: Correct attribute name
            )
            logger.info(f"💾 Persistence enabled: {output_dir}/jotty_state/")
        
        # Set root goal
        self.todo.root_task = goal
        logger.info("🔍 Setting root task...")

        # Initialize TODO from goal
        logger.info("🔍 Starting TODO initialization...")
        if getattr(self.config, 'enable_profiling', False):
            from ..utils.profiler import timed_block
            with timed_block("TodoInitialization", component="Initialization", name="Initialize TODO from goal"):
                await self._initialize_todo_from_goal(goal, kwargs)
        else:
            await self._initialize_todo_from_goal(goal, kwargs)
        logger.info("✅ TODO initialization complete")

        # Build initial context
        logger.info("🔍 Starting context guard setup...")
        if getattr(self.config, 'enable_profiling', False):
            from ..utils.profiler import timed_block
            with timed_block("ContextGuardSetup", component="Initialization", name="Setup Context Guard"):
                logger.info("🔍 Clearing context guard...")
                self.context_guard.clear()
                logger.info("🔍 Registering ROOT_GOAL...")
                self.context_guard.register_critical("ROOT_GOAL", goal)
                logger.info("🔍 Getting TODO state summary...")
                todo_summary = self.todo.get_state_summary()
                logger.info(f"🔍 Registering TODO_STATE (summary has {len(str(todo_summary))} chars)...")
                self.context_guard.register_critical("TODO_STATE", todo_summary)
        else:
            logger.info("🔍 Clearing context guard...")
            self.context_guard.clear()
            logger.info("🔍 Registering ROOT_GOAL...")
            self.context_guard.register_critical("ROOT_GOAL", goal)
            logger.info("🔍 Getting TODO state summary...")
            todo_summary = self.todo.get_state_summary()
            logger.info(f"🔍 Registering TODO_STATE (summary has {len(str(todo_summary))} chars)...")
            self.context_guard.register_critical("TODO_STATE", todo_summary)
        logger.info("✅ Context guard setup complete")
        logger.info("🔍 Entering main loop...")
        
        # Main loop - iterate until full reward or max iterations
        iteration = 0
        cumulative_reward = 0.0
        all_auditors_passed = False
        
        while iteration < max_iterations and not all_auditors_passed:
            iteration += 1
            logger.info(f"🔄 Starting iteration {iteration}")
            task_start_time = time.time()  # Track task timing (NO HARDCODING)

            # 🧩 Dependency-await support: unblock tasks whose deps are now satisfied
            logger.info(f"🔍 Checking for unblocked tasks...")
            try:
                unblocked = self.todo.unblock_ready_tasks()
                if unblocked:
                    logger.info(f"🟢 Unblocked {unblocked} task(s) (dependencies satisfied)")
                else:
                    logger.info(f"   No tasks to unblock")
            except Exception as e:
                logger.debug(f"Unblock check failed: {e}")

            # Get next task
            logger.info(f"🔍 Getting next task...")
            if getattr(self.config, 'enable_profiling', False):
                from ..utils.profiler import timed_block
                with timed_block(f"GetNextTask_iter{iteration}", component="Orchestration", name=f"Get next task (iteration {iteration})"):
                    task = self.todo.get_next_task()
            else:
                task = self.todo.get_next_task()
            logger.info(f"✅ Got task: {task.task_id if task else 'None'}")

            if not task:
                # No tasks available - try exploration
                if self.policy_explorer.should_explore(self.todo):
                    new_tasks = self.policy_explorer.explore(
                        self.todo,
                        self._get_available_actions(),
                        goal
                    )
                    for new_task in new_tasks:
                        self.todo.subtasks[new_task.task_id] = new_task  # Fixed: .id → .task_id
                    continue
                else:
                    logger.warning("No tasks available and exploration exhausted")
                    break

            # Predict Q-value for this task (ONLY if RL is enabled)
            # 🔥 CRITICAL FIX: Skip expensive state building when RL is disabled
            # This was causing 96-193 second overhead per iteration!
            if self.config.enable_rl:
                if getattr(self.config, 'enable_profiling', False):
                    from ..utils.profiler import timed_block
                    with timed_block(f"GetCurrentState_iter{iteration}", component="Orchestration", name=f"Build state (iteration {iteration})"):
                        state = self._get_current_state()
                else:
                    state = self._get_current_state()

                action = {'actor': task.actor, 'task': task.description}

                if getattr(self.config, 'enable_profiling', False):
                    from ..utils.profiler import timed_block
                    with timed_block(f"QPrediction_iter{iteration}", component="Orchestration", name=f"Q-value prediction (iteration {iteration})"):
                        q_value, confidence, alternative = self.q_predictor.predict_q_value(
                            state, action, goal
                        )
                else:
                    q_value, confidence, alternative = self.q_predictor.predict_q_value(
                        state, action, goal
                    )
            else:
                # RL disabled - skip expensive state building and Q-prediction
                state = {}
                action = {'actor': task.actor, 'task': task.description}
                q_value, confidence, alternative = None, 1.0, None
                logger.info(f"⏭️  Skipped state building and Q-prediction (enable_rl=False)")

            logger.info(f"✅ RL/Q-prediction phase complete")

            # If Q-value is low and we have alternative, consider it
            # ✅ A-TEAM FIX: Check q_value is not None before comparison
            if q_value is not None and q_value < 0.3 and alternative and confidence > 0.6:
                logger.info(f"🔄 Low Q-value ({q_value:.2f}), trying alternative: {alternative}")
                # Could add alternative as new task here

            # Execute task with appropriate actor
            actor_config = self.actors.get(task.actor)
            if not actor_config:
                self.todo.fail_task(task.task_id, f"Unknown actor: {task.actor}")
                continue

            logger.info(f"🎯 Preparing to execute actor '{actor_config.name}'...")

            # =================================================================
            # PREDICTIVE MARL: Predict trajectory BEFORE execution (A-Team Fix)
            # =================================================================
            self._last_prediction = None
            if self.trajectory_predictor:
                logger.info(f"🔮 Running trajectory prediction...")
                try:
                    other_actors = [a for a in self.actors.keys() if a != task.actor]
                    self._last_prediction = self.trajectory_predictor.predict(
                        current_state=state,
                        acting_agent=task.actor,
                        proposed_action={'task': task.description},
                        other_agents=other_actors,
                        goal=goal
                    )
                    logger.info(f"✅ Trajectory prediction complete: reward={self._last_prediction.predicted_reward:.2f}")
                except Exception as e:
                    logger.warning(f"⚠️  Trajectory prediction failed: {e}")
            else:
                logger.info(f"⏭️  Trajectory predictor disabled")
            
            # Build context for actor (with future-aware retrieval)
            logger.info(f"🔧 Building context for actor '{actor_config.name}'...")
            if getattr(self.config, 'enable_profiling', False):
                from ..utils.profiler import timed_block
                with timed_block(f"BuildContext_{actor_config.name}_iter{iteration}", component="ContextBuilding", name=f"Build context for {actor_config.name}"):
                    context, actor_context_dict = await self._build_actor_context(task, actor_config)
            else:
                context, actor_context_dict = await self._build_actor_context(task, actor_config)
            logger.info(f"✅ Context built for '{actor_config.name}'")
            
            # Execute actor
            try:
                # ⏱️  Profile actor execution
                if getattr(self.config, 'enable_profiling', False):
                    from ..utils.profiler import timed_block
                    with timed_block(f"{actor_config.name}", component="Agent", name=actor_config.name, task=task.description[:100] if task.description else ""):
                        result = await self._execute_actor(
                            actor_config,
                            task,
                            context,
                            kwargs,
                            actor_context_dict  # 🔥 Pass actor_context for parameter resolution
                        )
                else:
                    result = await self._execute_actor(
                        actor_config,
                        task,
                        context,
                        kwargs,
                        actor_context_dict  # 🔥 Pass actor_context for parameter resolution
                    )
                
                # Auditor check
                auditor_passed, auditor_reward, auditor_feedback = await self._run_auditor(
                    actor_config,
                    result,
                    task
                )
                
                if auditor_passed:
                    # ================================================================
                    # COOPERATIVE REWARD (A-Team Fix)
                    # 30% own success + 40% help others + 30% predictability
                    # ================================================================
                    base_reward = auditor_reward * self.config.base_reward_weight  # 🔧 STANFORD FIX
                    
                    # Cooperation bonus: check if this helps subsequent tasks
                    coop_bonus = 0.0
                    upcoming_tasks = [t for t in self.todo.subtasks.values() 
                                      if task.task_id in t.depends_on and t.status == TaskStatus.PENDING]  # Fixed: task.id → task.task_id
                    if upcoming_tasks:
                        coop_bonus = self.config.cooperation_bonus  # 🔧 STANFORD FIX
                    
                    # Predictability bonus: was this predicted correctly?
                    pred_bonus = 0.0
                    if self.trajectory_predictor and hasattr(self, '_last_prediction'):
                        if self._last_prediction and task.actor in str(self._last_prediction.steps):
                            pred_bonus = self.config.predictability_bonus  # 🔧 STANFORD FIX
                    
                    cooperative_reward = base_reward + coop_bonus + pred_bonus
                    
                    self.todo.complete_task(task.task_id, cooperative_reward)
                    cumulative_reward += cooperative_reward
                    
                    # ✅ INCREMENTAL SAVE: Save after first successful task
                    if iteration == 1 and self.persistence_manager:
                        logger.info("💾 Saving initial state after first task completion")
                        self.persistence_manager.save_markovian_todo(self.todo)
                    
                    # ✅ INCREMENTAL SAVE: Save after critical actors complete
                    if task.actor in ['loader', 'diffuser'] and self.persistence_manager:
                        logger.info(f"💾 Saving state after {task.actor} completion")
                        self.persistence_manager.save_markovian_todo(self.todo)
                        if hasattr(self, 'shared_memory'):
                            self.persistence_manager.save_memory(self.shared_memory, name="shared_memory")
                    
                    # Record intermediary values (NO HARDCODING - dynamic tracking)
                    task_duration = time.time() - task_start_time
                    self.todo.record_intermediary_values(task.task_id, {  # Fixed: task.id → task.task_id
                        'duration_seconds': task_duration,
                        'reward_obtained': cooperative_reward,
                        'auditor_passed': True,
                        'iteration': iteration,
                        'timestamp': time.time(),
                        'q_value': q_value,
                        'confidence': confidence
                    })
                    
                    # Record success for learning
                    self.q_predictor.record_outcome(state, action, cooperative_reward)
                    
                    # Extract domain/task_type from actor and task
                    domain = self._infer_domain_from_actor(task.actor)
                    task_type = self._infer_task_type_from_task(task.description)
                    
                    # Store SUCCESS with summary (low info content = less detail)
                    self.shared_memory.store(
                        content=f"✅ {task.description}: {auditor_feedback[:200]}",
                        level=MemoryLevel.SEMANTIC,  # Summary level
                        context={'actor': task.actor, 'task': task.task_id},  # Fixed: task.id → task.task_id
                        goal=goal,
                        initial_value=0.4,  # Lower weight (common event)
                        domain=domain,
                        task_type=task_type
                    )
                else:
                    self.todo.fail_task(task.task_id, auditor_feedback)  # Fixed: task.id → task.task_id
                    
                    # Record intermediary values for failure (NO HARDCODING)
                    task_duration = time.time() - task_start_time
                    self.todo.record_intermediary_values(task.task_id, {  # Fixed: task.id → task.task_id
                        'duration_seconds': task_duration,
                        'reward_obtained': 0.0,
                        'auditor_passed': False,
                        'iteration': iteration,
                        'timestamp': time.time(),
                        'q_value': q_value,
                        'confidence': confidence,
                        'failure_reason': auditor_feedback[:200]
                    })
                    
                    # Record failure
                    self.q_predictor.record_outcome(state, action, 0.0)
                    
                    # Extract domain/task_type from actor and task
                    domain = self._infer_domain_from_actor(task.actor)
                    task_type = self._infer_task_type_from_task(task.description)
                    
                    # Store FAILURE with full detail (HIGH info = store more!)
                    # Shannon: -log P(failure) = high info when failures are rare
                    self.shared_memory.store(
                        content=f"""❌ FAILURE ANALYSIS (High Information Event)
Task: {task.description}
Actor: {task.actor}
Reason: {auditor_feedback}
State: {json.dumps(state, default=str)[:500]}
Attempt #{task.attempts + 1}""",
                        level=MemoryLevel.CAUSAL,  # Causal - learn why!
                        context={'actor': task.actor, 'task': task.task_id, 'failure': True},  # Fixed: task.id → task.task_id
                        goal=goal,
                        initial_value=0.9  # HIGH weight (rare event = valuable)
                    )
                
                # Track in trajectory
                # 🔧 CRITICAL FIX: Include FULL ReAct trajectory for Auditor visibility
                task_end_time = time.time()  # 🔧 FIX: Set task end time
                react_trajectory = {}
                if hasattr(result, 'output'):
                    inner = result.output
                    if hasattr(inner, 'trajectory') and isinstance(inner.trajectory, dict):
                        react_trajectory = inner.trajectory
                    elif hasattr(inner, '__dict__') and 'trajectory' in vars(inner):
                        react_trajectory = vars(inner).get('trajectory', {})
                
                step_result = {
                    'iteration': iteration,
                    'task': task.task_id,  # Fixed: task.id → task.task_id
                    'actor': task.actor,
                    'passed': auditor_passed,
                    'reward': auditor_reward if auditor_passed else 0.0,
                    'feedback': auditor_feedback,
                    'success': auditor_passed,
                    'actor_output': result,  # ✅ Full actor output
                    'react_trajectory': react_trajectory,  # ✅ NEW: Full ReAct trajectory with thoughts/observations
                    'execution_time': task_end_time - task_start_time
                }
                self.trajectory.append(step_result)
                
                # ================================================================
                # DIVERGENCE LEARNING: Compare prediction with reality (A-Team Fix)
                # ================================================================
                # 🧠 A-TEAM FIX: Wire divergence → Q-table update → Lesson extraction → Prompt update
                if self.trajectory_predictor and self._last_prediction and self.divergence_memory:
                    try:
                        # Build actual trajectory from what happened
                        actual = ActualTrajectory(
                            steps=[step_result],
                            actual_reward=auditor_reward if auditor_passed else 0.0
                        )
                        
                        # Compute divergence (this is the "loss")
                        divergence = self.trajectory_predictor.compute_divergence(
                            self._last_prediction, actual
                        )
                        
                        # Store by information content (Shannon)
                        self.divergence_memory.store(divergence)
                        
                        # Update predictor from divergence (this is the "weight update")
                        self.trajectory_predictor.update_from_divergence(divergence)
                        
                        # ================================================================
                        # 🧠 A-TEAM FIX: Use divergence as TD error for Q-table update!
                        # ================================================================
                        # Information asymmetry = what we learned from execution vs planning
                        info_asymmetry = divergence.information_content
                        
                        # Divergence-weighted reward: penalize for wrong predictions
                        divergence_penalty = 1.0 - min(1.0, divergence.total_divergence())
                        q_reward = (auditor_reward if auditor_passed else 0.0) * divergence_penalty
                        
                        # Update Q-table with divergence-adjusted reward
                        self.q_predictor.add_experience(
                            state=state,
                            action={'actor': actor_config.name, 'task': task.task_id},
                            reward=q_reward,
                            next_state=self._get_current_state(),
                            done=False
                        )
                        
                        # ================================================================
                        # 🧠 A-TEAM FIX: Extract lessons from divergence for prompt updates
                        # ================================================================
                        if hasattr(self.q_predictor, 'Q') and self.q_predictor.Q:
                            key = list(self.q_predictor.Q.keys())[-1]  # Most recent
                            lessons = self.q_predictor.Q[key].get('learned_lessons', [])
                            if lessons:
                                logger.info(f"📚 Learned: {lessons[-1][:80]}...")
                        
                        logger.debug(f"📊 Divergence: {divergence.total_divergence():.2f}, info={info_asymmetry:.2f}, Q-reward={q_reward:.2f}")
                    except Exception as e:
                        logger.debug(f"Divergence learning failed: {e}")
                
                # Update context with new TODO state
                self.context_guard.clear()
                self.context_guard.register_critical("ROOT_GOAL", goal)
                self.context_guard.register_critical("TODO_STATE", self.todo.get_state_summary())
                
                # Check if all tasks completed successfully
                all_completed = all(
                    t.status == TaskStatus.COMPLETED 
                    for t in self.todo.subtasks.values()  # Fixed: items → subtasks
                )
                if all_completed:
                    all_auditors_passed = True
                    logger.info("🎉 All tasks completed successfully!")
                
                # Auto-save state periodically (NO HARDCODED INTERVAL - from config)
                if self.persistence_manager and self.persistence_manager.should_auto_save(iteration):
                    self.persistence_manager.save_markovian_todo(self.todo)
                    logger.info(f"💾 Auto-saved state at iteration {iteration}")
                
            except Exception as e:
                logger.error(f"Actor execution failed: {e}")
                import traceback as tb  # Local import to avoid scope issues
                logger.error(f"Full traceback: {tb.format_exc()}")
                
                # 🔥 A-TEAM FIX: Store failure in memory BEFORE deciding if fatal!
                # This ensures we learn from ALL failures, not just Auditor failures
                failure_context = {
                    'actor': task.actor,
                    'task': task.task_id,
                    'error_type': type(e).__name__,
                    'error_message': str(e)[:500],
                    'iteration': iteration
                }
                
                self.shared_memory.store(
                    content=f"""❌ EXECUTION FAILURE (High Information Event)
Task: {task.description}
Actor: {task.actor}
Error: {type(e).__name__}: {str(e)[:500]}
Iteration: {iteration}

This failure contains valuable learning information!
Strategy: Check if recoverable, adapt retry approach.
""",
                    level=MemoryLevel.EPISODIC,  # Full detail
                    context=failure_context,
                    goal=goal,
                    initial_value=0.95  # VERY HIGH weight (execution failures are rare and valuable)
                )
                logger.info(f"💾 Stored execution failure in memory (weight=0.95) for learning")
                
                # 🔥 A-TEAM FIX: Smart TypeError classification
                # Not all TypeErrors are fatal! Some are recoverable.
                ALWAYS_FATAL_ERRORS = (AttributeError, NameError, ModuleNotFoundError, ImportError)
                
                if isinstance(e, TypeError):
                    # Check if it's a recoverable type error
                    error_msg = str(e)
                    RECOVERABLE_TYPE_ERROR_PATTERNS = [
                        "cannot be used with isinstance",  # Union, generics - Python limitation
                        "got an unexpected keyword argument",  # Parameter mismatch - can filter
                    ]
                    
                    is_recoverable = any(pattern in error_msg for pattern in RECOVERABLE_TYPE_ERROR_PATTERNS)
                    
                    if is_recoverable:
                        logger.warning(f"⚠️  Recoverable TypeError detected: {error_msg[:200]}")
                        logger.info(f"🧠 LEARNING: This error has been stored in memory for future adaptation")
                        logger.info(f"   📦 Marking task as failed, but not stopping execution")
                        self.todo.fail_task(task.task_id, str(e))
                        # Don't re-raise! Continue to next task
                    else:
                        # Truly fatal TypeError (e.g., calling non-callable)
                        logger.critical(f"🔴 FATAL TypeError detected: {error_msg[:200]}")
                        logger.critical("This error indicates a code bug, not a transient issue")
                        if self.persistence_manager:
                            logger.info("💾 Saving state after fatal error (for debugging)")
                            try:
                                self.persistence_manager.save_markovian_todo(self.todo)
                            except Exception as save_error:
                                logger.error(f"Failed to save state after error: {save_error}")
                        self.todo.fail_task(task.task_id, str(e))
                        raise  # Re-raise to stop execution
                
                elif isinstance(e, ALWAYS_FATAL_ERRORS):
                    logger.critical(f"🔴 FATAL ERROR detected ({type(e).__name__}), not retrying")
                    logger.critical("This error indicates a code bug, not a transient issue")
                    # Save state for debugging
                    if self.persistence_manager:
                        logger.info("💾 Saving state after fatal error (for debugging)")
                        try:
                            self.persistence_manager.save_markovian_todo(self.todo)
                        except Exception as save_error:
                            logger.error(f"Failed to save state after error: {save_error}")
                    self.todo.fail_task(task.task_id, str(e))
                    raise  # Re-raise to stop execution
                
                else:
                    # Other exceptions - potentially recoverable
                    logger.warning(f"⚠️  Exception during execution: {type(e).__name__}")
                    logger.info(f"🧠 LEARNING: Stored in memory, marking task as failed, continuing")
                    
                    # 📮 DLQ: Add to Dead Letter Queue for potential retry
                    if self.dead_letter_queue:
                        self.dead_letter_queue.add(
                            operation_name=f"{actor_config.name}.execute",
                            args=(actor_config, task, context, kwargs, actor_context_dict),
                            kwargs={},
                            error=e
                        )
                        logger.info(f"📮 Added failed operation to DLQ (queue size: {len(self.dead_letter_queue.queue)})")
                
                # ✅ SAVE STATE BEFORE FAILURE: Preserve state for debugging
                if self.persistence_manager:
                    logger.info("💾 Saving state after error (for debugging)")
                    try:
                        self.persistence_manager.save_markovian_todo(self.todo)
                    except Exception as save_error:
                        logger.error(f"Failed to save state after error: {save_error}")
                
                    self.todo.fail_task(task.task_id, str(e))  # Mark failed, but don't raise
        
        # Episode complete - update learner
        self.swarm_learner.record_episode(
            self.trajectory,
            outcome=all_auditors_passed,
            insights=[t['feedback'] for t in self.trajectory if t.get('feedback')]
        )
        
        # Update prompts if threshold reached
        if self.swarm_learner.should_update_prompts():
            for actor_name, actor_config in self.actors.items():
                for prompt_path in actor_config.architect_prompts + actor_config.auditor_prompts:
                    if Path(prompt_path).exists():
                        current = Path(prompt_path).read_text()
                        updated, changes = self.swarm_learner.update_prompt(
                            prompt_path, current
                        )
                        if changes:
                            Path(prompt_path).write_text(updated)
                            logger.info(f"📝 Updated prompt: {prompt_path}")
        
        # =====================================================================
        # FINAL COMPREHENSIVE STATE SAVE (NO HARDCODING)
        # =====================================================================
        if self.persistence_manager:
            self.persistence_manager.save_all(self)
        
        # =====================================================================
        # DEAD LETTER QUEUE: Retry failed operations if system recovered
        # =====================================================================
        if self.dead_letter_queue and len(self.dead_letter_queue.queue) > 0:
            logger.info(f"📮 DLQ: {len(self.dead_letter_queue.queue)} failed operations in queue")
            
            # Only retry if we had overall success (system recovered)
            if all_auditors_passed:
                logger.info("📮 DLQ: System recovered successfully, retrying failed operations...")
                
                def retry_operation(operation_name: str, *args, **kwargs):
                    """Retry a failed operation."""
                    logger.info(f"🔄 DLQ: Retrying {operation_name}...")
                    # For now, we'll just log - actual retry would need async handling
                    # TODO: Implement full async retry mechanism
                    logger.warning(f"⚠️  DLQ: Async retry not yet implemented for {operation_name}")
                
                stats = self.dead_letter_queue.retry_all(retry_operation)
                logger.info(f"📮 DLQ Retry Stats: {stats}")
            else:
                logger.info("📮 DLQ: System did not fully recover, keeping failed operations for later")
        
        # =====================================================================
        # BRAIN CONSOLIDATION: Feed experience and trigger consolidation
        # =====================================================================
        if hasattr(self, 'brain') and self.brain and BRAIN_MEMORY_MANAGER_AVAILABLE:
            try:
                # Create episode reward
                episode_reward = 1.0 if all_auditors_passed else 0.0
                
                # Store each actor's experience in Hippocampus
                for actor_config in self.actors.values():  # ← FIX: self.actors is a DICT, not a list!
                    actor_name = actor_config.name
                    # Check if this actor executed
                    if hasattr(self, 'io_manager') and self.io_manager and actor_name in self.io_manager.outputs:
                        actor_output = self.io_manager.outputs[actor_name]
                        actor_success = actor_output.success if hasattr(actor_output, 'success') else True
                        
                        # Create experience dictionary
                        experience = {
                            "goal": goal[:200],
                            "actor": actor_name,
                            "success": actor_success,
                            "iteration": self.episode_counter,
                            "output_fields": list(actor_output.output_fields.keys()) if hasattr(actor_output, 'output_fields') else []
                        }
                        
                        self.brain.store_experience(
                            experience=experience,
                            reward=1.0 if actor_success else 0.0
                        )
                        logger.debug(f"🧠 Stored experience for {actor_name} (reward={1.0 if actor_success else 0.0:.2f})")
                
                # Increment episode counter
                self.episode_counter += 1
                self.brain.episodes_since_sleep += 1
                
                # Trigger consolidation (Sharp-Wave Ripple) if it's time
                if self.brain.should_consolidate():
                    logger.info("🧠 Triggering Sharp-Wave Ripple consolidation...")
                    logger.info(f"   Hippocampus size: {len(self.brain.hippocampus)} experiences")
                    logger.info(f"   Episodes since last sleep: {self.brain.episodes_since_sleep}")
                    
                    self.brain.trigger_consolidation()
                    
                    logger.info(f"🧠 Consolidation complete!")
                    logger.info(f"   Neocortex size: {len(self.brain.neocortex)} semantic patterns")
                    logger.info(f"   Total consolidations: {self.brain.total_consolidations}")
                    
                    # Log memory stats
                    stats = self.brain.get_statistics()
                    logger.info(f"🧠 Memory Stats: {stats}")
                    
                    # Reset counter
                    self.brain.episodes_since_sleep = 0
                
            except Exception as e:
                logger.error(f"Brain consolidation error: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # ✅ A-TEAM: Return typed SwarmResult!
        # Get final output from last actor result (GENERIC - no domain knowledge)
        final_output = None
        if self.io_manager.outputs:
            last_actor = list(self.io_manager.outputs.values())[-1]
            if last_actor.output_fields:
                # Generic field names - try common patterns first, then use ALL fields
                # Priority: final_output (standard), output (common), result (generic)
                final_output = (
                    last_actor.output_fields.get('final_output') or
                    last_actor.output_fields.get('output') or
                    last_actor.output_fields.get('result') or
                    last_actor.output_fields.get('sql_query') or  # Common for SQL agents
                    last_actor.output_fields.get('answer') or     # Common for QA agents
                    last_actor.output_fields.get('response')      # Common for chat agents
                )
                
                # If still None, use the FULL output_fields dict as final_output
                if final_output is None and last_actor.output_fields:
                    final_output = last_actor.output_fields
                
                # 🔍 A-TEAM DEBUG: Log what we found (generic logging)
                logger.info(f"🔍 [FINAL OUTPUT] Last actor: {list(self.io_manager.outputs.keys())[-1]}")
                logger.info(f"🔍 [FINAL OUTPUT] Available fields: {list(last_actor.output_fields.keys())}")
                logger.info(f"🔍 [FINAL OUTPUT] Extracted final_output: {final_output is not None} (type={type(final_output).__name__ if final_output is not None else 'None'})")
                if final_output:
                    logger.info(f"🔍 [FINAL OUTPUT] Preview: {str(final_output)[:200]}...")
        
        # =====================================================================
        # 🧠 NEUROCHUNK MEMORY MANAGEMENT: Tier memories at end of episode
        # =====================================================================
        if hasattr(self, 'q_learner') and self.q_learner:
            try:
                episode_reward = cumulative_reward
                logger.info("🧠 Tiering Q-Learning memories (NeuroChunk)...")
                
                # Promote/demote memories between tiers based on retention scores
                self.q_learner._promote_demote_memories(episode_reward=episode_reward)
                
                # Prune Tier 3 using causal impact scoring (every episode)
                # Sample 10% for efficiency
                self.q_learner.prune_tier3_by_causal_impact(sample_rate=0.1)
                
                # Log memory statistics
                summary = self.q_learner.get_q_table_summary()
                logger.info(f"🧠 Q-Table Stats:")
                logger.info(f"   Total entries: {summary['size']}")
                logger.info(f"   Tier 1 (Working): {summary.get('tier1_size', 0)} memories")
                logger.info(f"   Tier 2 (Clusters): {summary.get('tier2_clusters', 0)} clusters")
                logger.info(f"   Tier 3 (Archive): {summary.get('tier3_size', 0)} memories")
                logger.info(f"   Tier 1 threshold: {summary.get('tier1_threshold', 0.8):.3f}")
                logger.info(f"   Avg Q-value: {summary.get('avg_value', 0.0):.3f}")
                
            except Exception as e:
                logger.warning(f"⚠️  NeuroChunk memory tiering failed: {e}")
        # =====================================================================
        
        # ✅ Final success should not be True if no actor outputs exist (nothing executed).
        actor_outputs = self.io_manager.get_all_outputs()
        success = bool(all_auditors_passed) and bool(actor_outputs)

        # ⏱️  Print profiling summary and save reports if enabled
        if getattr(self.config, 'enable_profiling', False):
            end_time = time.time()
            from ..utils.profiler import print_profile_summary, save_profiling_reports, set_overall_timing

            # Set overall timing for the profiling report
            set_overall_timing(start_time, end_time)

            # Print console summary
            print_profile_summary()

            # Save detailed profiling reports to files
            saved_files = save_profiling_reports()
            if saved_files:
                logger.info(f"\n📊 Profiling reports available at:")
                logger.info(f"   - Gantt Chart: {saved_files.get('gantt', 'N/A')}")
                logger.info(f"   - Timeline Report: {saved_files.get('timeline', 'N/A')}")
                logger.info(f"   - Markdown Report: {saved_files.get('markdown', 'N/A')}")
                logger.info(f"   - JSON Data: {saved_files.get('json', 'N/A')}")

        return SwarmResult(
            success=success,
            final_output=final_output,
            actor_outputs=actor_outputs,
            trajectory=self.trajectory,
            metadata={
                'iterations': iteration,
                'cumulative_reward': cumulative_reward,
                'todo_state': self.todo.get_state_summary(),
                'execution_time': time.time() - start_time,
                'exploration_count': self.policy_explorer.exploration_count,
                'run_dir': output_dir,
            }
        )
    
    async def _initialize_todo_from_goal(self, goal: str, kwargs: Dict):
        """Initialize TODO items from goal."""
        # Add default tasks for each actor
        for i, (name, config) in enumerate(self.actors.items()):
            if config.enabled:
                self.todo.add_task(
                    task_id=f"{name}_main",
                    description=f"Execute {name} pipeline",
                    actor=name,
                    depends_on=[f"{prev}_main" for prev in list(self.actors.keys())[:i]],  # Fixed: dependencies → depends_on
                    priority=1.0 - (i * 0.1)
                )
    
    def _get_current_state(self) -> Dict[str, Any]:
        """
        Get RICH current state for Q-prediction.
        
        🔥 A-TEAM CRITICAL: State must capture semantic context!
        
        Includes:
        1. Query semantics (what user asked)
        2. Metadata context (tables, columns, partitions)
        3. Error patterns (what failed)
        4. Tool usage (what worked)
        5. Actor outputs (what was produced)
        """
        state = {
            # === 1. TASK PROGRESS ===
            'todo': {
                'completed': len(self.todo.completed),
                'pending': len([t for t in self.todo.subtasks.values() if t.status == TaskStatus.PENDING]),
                'failed': len(self.todo.failed_tasks)
            },
            'trajectory_length': len(self.trajectory),
            'recent_outcomes': [t.get('passed', False) for t in self.trajectory[-5:]]
        }
        
        # === 2. QUERY CONTEXT (CRITICAL!) ===
        # Try multiple sources for query
        query = None
        
        # Source 1: SharedContext
        if hasattr(self, 'shared_context') and self.shared_context:
            query = self.shared_context.get('query') or self.shared_context.get('goal')
        
        # Source 2: Context guard buffers (if available)
        if not query and hasattr(self, 'context_guard') and self.context_guard:
            # SmartContextGuard stores content in buffers
            for priority_buffer in self.context_guard.buffers.values():
                for key, content, _ in priority_buffer:
                    if key == 'ROOT_GOAL':
                        query = content
                        break
                if query:
                    break
        
        # Source 3: TODO root task
        if not query and hasattr(self, 'todo') and self.todo:
            query = self.todo.root_task
        
        if query:
            state['query'] = str(query)[:200]
        
        # === 3. METADATA CONTEXT ===
        if hasattr(self, 'shared_context') and self.shared_context:
            # Get table info
            tables = self.shared_context.get('table_names') or self.shared_context.get('relevant_tables')
            if tables:
                state['tables'] = tables if isinstance(tables, list) else [str(tables)]
            
            # Get filter info
            filters = self.shared_context.get('filters') or self.shared_context.get('filter_conditions')
            if filters:
                state['filters'] = filters
            
            # Get resolved terms
            resolved = self.shared_context.get('resolved_terms')
            if resolved:
                if isinstance(resolved, dict):
                    state['resolved_terms'] = list(resolved.keys())[:5]
        
        # === 4. ACTOR OUTPUT CONTEXT ===
        if hasattr(self, 'io_manager') and self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            output_summary = {}
            for actor_name, output in all_outputs.items():
                if hasattr(output, 'output_fields') and output.output_fields:
                    output_summary[actor_name] = list(output.output_fields.keys())
            if output_summary:
                state['actor_outputs'] = output_summary
        
        # === 5. ERROR PATTERNS (CRITICAL FOR LEARNING!) ===
        if self.trajectory:
            errors = []
            columns_tried = []
            working_column = None
            
            for step in self.trajectory:
                # Check for errors in trajectory
                if step.get('error'):
                    err = step['error']
                    if 'COLUMN_NOT_FOUND' in str(err):
                        # Extract column name from error
                        import re
                        match = re.search(r"Column '(\w+)' cannot be resolved", str(err))
                        if match:
                            col = match.group(1)
                            columns_tried.append(col)
                            errors.append({'type': 'COLUMN_NOT_FOUND', 'column': col})
                
                # Check for success
                if step.get('passed') and step.get('tool_calls'):
                    for tc in step.get('tool_calls', []):
                        if isinstance(tc, dict) and tc.get('success'):
                            # Extract working column if SQL-related
                            if 'query' in str(tc):
                                # Try to find date column that worked
                                query = str(tc.get('query', ''))
                                for possible_col in ['dl_last_updated', 'dt', 'date', 'created_at']:
                                    if possible_col in query.lower():
                                        working_column = possible_col
                                        break
            
            if errors:
                state['errors'] = errors[-5:]  # Last 5 errors
            if columns_tried:
                state['columns_tried'] = list(dict.fromkeys(columns_tried))  # Unique
            if working_column:
                state['working_column'] = working_column
                state['error_resolution'] = f"use {working_column} instead of {','.join(columns_tried[:3])}"
        
        # === 6. TOOL USAGE PATTERNS ===
        successful_tools = []
        failed_tools = []
        tool_calls = []
        
        for step in self.trajectory:
            if step.get('tool_calls'):
                for tc in step.get('tool_calls', []):
                    tool_name = tc.get('tool') if isinstance(tc, dict) else str(tc)
                    tool_calls.append(tool_name)
                    
                    if isinstance(tc, dict):
                        if tc.get('success'):
                            successful_tools.append(tool_name)
                        else:
                            failed_tools.append(tool_name)
        
        if tool_calls:
            state['tool_calls'] = tool_calls[-10:]
        if successful_tools:
            state['successful_tools'] = list(dict.fromkeys(successful_tools))
        if failed_tools:
            state['failed_tools'] = list(dict.fromkeys(failed_tools))
        
        # === 7. CURRENT ACTOR ===
        if self.trajectory and self.trajectory[-1].get('actor'):
            state['current_actor'] = self.trajectory[-1]['actor']
        
        # === 8. VALIDATION CONTEXT ===
        for step in self.trajectory[-3:]:  # Last 3 steps
            if step.get('architect_confidence'):
                state['architect_confidence'] = step['architect_confidence']
            if step.get('auditor_result'):
                state['auditor_result'] = step['auditor_result']
            if step.get('validation_passed') is not None:
                state['validation_passed'] = step['validation_passed']
        
        # === 9. EXECUTION STATS ===
        state['attempts'] = len(self.trajectory)
        state['success'] = any(t.get('passed', False) for t in self.trajectory)
        
        return state
    
    def _get_available_actions(self) -> List[Dict[str, Any]]:
        """Get available actions for exploration."""
        actions = []
        for name, config in self.actors.items():
            actions.append({
                'actor': name,
                'action': 'execute',
                'enabled': config.enabled
            })
        return actions
    
    async def _build_actor_context(
        self,
        task: TodoItem,
        actor_config: ActorConfig
    ) -> Tuple[str, Dict]:
        """
        Build context for actor execution.

        A-Team Enhancement: FUTURE-AWARE retrieval.
        Considers not just current task but upcoming TODO items.

        🚀 OPTIMIZED: Parallel memory retrieval for 3-5x speedup!
        """
        import asyncio

        # Use context guard to build within limits
        guard = SmartContextGuard(self.config.max_context_tokens)

        # CRITICAL: Task and goal
        guard.register_critical("CURRENT_TASK", task.description)
        guard.register_critical("ROOT_GOAL", self.todo.root_task)

        # HIGH: Recent trajectory
        recent = self.trajectory[-5:] if self.trajectory else []
        guard.register("RECENT_TRAJECTORY", json.dumps(recent, default=str), guard.HIGH)

        # ================================================================
        # 🚀 PARALLEL FUTURE-AWARE RETRIEVAL (Performance Optimized!)
        # Get memories for CURRENT + UPCOMING tasks IN PARALLEL
        # Previously: 0s → 6.6s → 11.7s (sequential)
        # Now: 0s → 2.5s → 3.5s (parallel, 3-5x faster!)
        # ================================================================

        # Prepare upcoming tasks list
        upcoming = [t for t in self.todo.subtasks.values()
                    if t.status == TaskStatus.PENDING and t.task_id != task.task_id][:3]

        # 🚀 FAST PATH: Skip retrieval if no memories exist yet (early iterations)
        shared_has_memories = len(self.shared_memory.memories[list(self.shared_memory.memories.keys())[0]]) > 0 if self.shared_memory.memories else False
        local_has_memories = len(self.local_memories[actor_config.name].memories[list(self.local_memories[actor_config.name].memories.keys())[0]]) > 0 if self.local_memories[actor_config.name].memories else False

        if not shared_has_memories and not local_has_memories:
            logger.info(f"⚡ Fast path: No memories exist yet, skipping retrieval (0ms)")
            guard.register("SHARED_MEMORIES", "[]", guard.MEDIUM)
            guard.register("LOCAL_MEMORIES", "[]", guard.MEDIUM)
        else:
            # 🧠 CONFIGURABLE RETRIEVAL MODE (Synthesis DEFAULT!)
            retrieval_mode = getattr(self.config, 'retrieval_mode', 'synthesize')

            if retrieval_mode == "synthesize":
                # ==========================================
                # 🧠 SYNTHESIS MODE (DEFAULT - Brain-Inspired!)
                # ==========================================
                logger.info(f"🧠 Using synthesis retrieval (neuroscience-aligned)")

                # Run synthesis in parallel for shared + local memories
                retrieval_tasks = []

                # Synthesize shared memories (includes current + future tasks)
                synthesis_query = f"{task.description}\n\nUpcoming tasks: {', '.join([t.description for t in upcoming])}"
                retrieval_tasks.append(
                    self.shared_memory.retrieve_and_synthesize_async(
                        query=synthesis_query,
                        goal=self.todo.root_task
                    )
                )

                # Synthesize local actor memories
                retrieval_tasks.append(
                    self.local_memories[actor_config.name].retrieve_and_synthesize_async(
                        query=task.description,
                        goal=self.todo.root_task
                    )
                )

                # Execute both syntheses in parallel
                logger.debug(f"🧠 Running 2 memory syntheses in parallel...")
                results = await asyncio.gather(*retrieval_tasks)

                shared_wisdom = results[0]
                local_wisdom = results[1]

                logger.info(f"✅ Synthesized shared wisdom ({len(shared_wisdom)} chars) + local wisdom ({len(local_wisdom)} chars)")

                # Register synthesized wisdom (coherent text, not discrete memories!)
                guard.register("SHARED_WISDOM", shared_wisdom, guard.MEDIUM)
                guard.register("LOCAL_WISDOM", local_wisdom, guard.MEDIUM)

            else:
                # ==========================================
                # 📋 DISCRETE MODE (Legacy - Faster but Less Intelligent)
                # ==========================================
                logger.info(f"📋 Using discrete retrieval (legacy mode)")

                # 🚀 Run all memory retrievals in PARALLEL using asyncio.gather()
                retrieval_tasks = []

                # 1. Current task memories (60% of budget)
                retrieval_tasks.append(
                    self.shared_memory.retrieve_async(
                        query=task.description,
                        goal=self.todo.root_task,
                        budget_tokens=1000
                    )
                )

                # 2. Local actor memory
                retrieval_tasks.append(
                    self.local_memories[actor_config.name].retrieve_async(
                        query=task.description,
                        goal=self.todo.root_task,
                        budget_tokens=1000
                    )
                )

                # 3. Future tasks memories (batched as individual async tasks)
                for upcoming_task in upcoming:
                    retrieval_tasks.append(
                        self.shared_memory.retrieve_async(
                            query=upcoming_task.description,
                            goal=self.todo.root_task,
                            budget_tokens=500
                        )
                    )

                # Execute all retrievals in parallel
                logger.debug(f"🚀 Running {len(retrieval_tasks)} memory retrievals in parallel...")
                results = await asyncio.gather(*retrieval_tasks)

                # Unpack results
                current_memories = results[0]
                local_mem = results[1]
                future_memories_list = results[2:] if len(results) > 2 else []

                # Flatten future memories
                future_memories = []
                for fm in future_memories_list:
                    future_memories.extend(fm)

                # Combine and deduplicate
                seen_keys = set()
                all_memories = []
                for mem in current_memories + future_memories:
                    if hasattr(mem, 'key') and mem.key not in seen_keys:
                        seen_keys.add(mem.key)
                        all_memories.append(mem)

                logger.debug(f"✅ Retrieved {len(all_memories)} shared + {len(local_mem)} local memories in parallel")

                guard.register("SHARED_MEMORIES", str(all_memories), guard.MEDIUM)
                guard.register("LOCAL_MEMORIES", str(local_mem), guard.MEDIUM)
        
        # ✅ A-TEAM FIX: Call metadata_provider protocol for domain-specific context!
        actor_context_from_provider = None
        if self.metadata_provider and hasattr(self.metadata_provider, 'get_context_for_actor'):
            try:
                # Get previous outputs for context-aware provision
                previous_outputs = {}
                if hasattr(self, 'shared_context') and 'actor_outputs' in self.shared_context:
                    for dep_name in actor_config.dependencies:
                        if dep_name in self.shared_context['actor_outputs']:
                            previous_outputs[dep_name] = self.shared_context['actor_outputs'][dep_name]
                
                # Call the GENERIC protocol (no SQL hardcoding!)
                actor_context_from_provider = self.metadata_provider.get_context_for_actor(
                    actor_name=actor_config.name,
                    query=task.description,
                    previous_outputs=previous_outputs,
                    actor_config=actor_config,
                    **self.context_providers
                )
                
                if actor_context_from_provider and isinstance(actor_context_from_provider, dict):
                    # Register each metadata field with the guard
                    for key, value in actor_context_from_provider.items():
                        if value and isinstance(value, str) and len(value) > 10:
                            # Use MEDIUM priority for metadata (important but not critical)
                            guard.register(f"METADATA_{key.upper()}", value, guard.MEDIUM)
                    logger.debug(f"✅ Registered {len(actor_context_from_provider)} metadata fields from provider for {actor_config.name}")
            except Exception as e:
                logger.error(f"❌ MetadataProvider call failed for {actor_config.name}: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # LOW: Annotations
        if self.annotations:
            guard.register("ANNOTATIONS", json.dumps(self.annotations)[:1000], guard.LOW)
        
        # 🔥 A-TEAM CRITICAL FIX: Return both context string AND actor_context dict
        # The dict will be merged into resolved_kwargs in _execute_actor
        return guard.build_context(), actor_context_from_provider
    
    def _build_param_mappings(self, custom_mappings: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """
        Build parameter mappings: defaults + config + user-provided.
        
        🆕 A-TEAM: Makes JOTTY truly generic by allowing users to define
        domain-specific parameter name mappings!
        
        Priority: user_mappings > config_mappings > defaults
        """
        # Minimal generic defaults (domain-agnostic)
        default_mappings = {
            # Core generic patterns
            'content': ['content', 'text', 'body'],
            'data': ['data', 'output_data', 'results'],
            'file': ['file', 'filepath', 'path', 'file_path'],
            'url': ['url', 'uri', 'link', 'href'],
        }
        
        # Start with defaults
        mappings = default_mappings.copy()
        
        # Merge config mappings (if provided)
        if hasattr(self.config, 'custom_param_mappings') and self.config.custom_param_mappings:
            for key, aliases in self.config.custom_param_mappings.items():
                if key in mappings:
                    # Extend existing (remove duplicates)
                    mappings[key] = list(set(mappings[key] + aliases))
                else:
                    # Add new
                    mappings[key] = aliases
        
        # Merge user mappings (highest priority)
        if custom_mappings:
            for key, aliases in custom_mappings.items():
                if key in mappings:
                    # Extend existing (remove duplicates)
                    mappings[key] = list(set(mappings[key] + aliases))
                else:
                    # Add new
                    mappings[key] = aliases
        
        logger.info(f"📋 Parameter mappings: {len(mappings)} keys, {sum(len(v) for v in mappings.values())} total aliases")
        if mappings:
            logger.debug(f"   Mappings: {list(mappings.keys())}")
        
        return mappings
    
    
    async def _attempt_parameter_recovery(
        self,
        actor_config: ActorConfig,
        missing_params: List[str],
        context: Dict[str, Any],
        shared_context: Dict[str, Any],
        max_attempts: int = 2
    ) -> Dict[str, Any]:
        """
        🛠️ INTELLIGENT PARAMETER RECOVERY
        
        PRIORITY ORDER (A-TEAM FINAL):
        1. Check IOManager (actor outputs) ← WHERE AGENT DATA LIVES!
        2. Check SharedContext['metadata'] (only if not found in IOManager)
        3. Invoke MetaDataFetcher (on-demand fetch)
        4. (Future) Re-invoke failed dependencies
        """
        recovered = {}
        logger.info(f"🔍 Recovery Strategy for {actor_config.name}: analyzing {len(missing_params)} missing parameters")
        
        for param in missing_params:
            logger.info(f"   🔍 Attempting recovery for '{param}'...")
            
            # 🔥 STRATEGY 1: Check IOManager (ACTOR OUTPUTS) - THIS IS THE PRIMARY DATA FLOW!
            if hasattr(self, 'io_manager') and self.io_manager:
                all_outputs = self.io_manager.get_all_outputs()
                for actor_name, actor_output in all_outputs.items():
                    if hasattr(actor_output, 'output_fields') and actor_output.output_fields:
                        # Exact match
                        if param in actor_output.output_fields:
                            recovered[param] = actor_output.output_fields[param]
                            logger.info(f"   ✅ Found '{param}' in IOManager['{actor_name}']")
                            break
                        
                        # Check aliases (e.g., 'tables' matches 'relevant_tables')
                        aliases = {
                            'tables': ['relevant_tables', 'selected_tables', 'table_list'],
                            'columns': ['selected_columns', 'column_list'],
                            'resolved_terms': ['terms', 'business_terms'],
                            'filters': ['filter_conditions', 'where_conditions'],
                        }
                        
                        if param in aliases:
                            for alias in aliases[param]:
                                if alias in actor_output.output_fields:
                                    recovered[param] = actor_output.output_fields[alias]
                                    logger.info(f"   ✅ Found '{param}' as '{alias}' in IOManager['{actor_name}']")
                                    break
                        
                        if param in recovered:
                            break
            
            if param in recovered:
                continue
            
            # Strategy 2: Check SharedContext['metadata'] (ONLY if not in IOManager!)
            if self.shared_context and 'metadata' in self.shared_context.data:
                metadata = self.shared_context.get('metadata')
                if metadata and isinstance(metadata, dict):
                    # Direct match
                    if param in metadata:
                        recovered[param] = metadata[param]
                        logger.info(f"   ✅ Found '{param}' in SharedContext['metadata']")
                        continue
                    
                    # 🔥 A-TEAM: ALIAS MATCHING (before agentic search)
                    aliases = {
                        'tables': ['relevant_tables', 'selected_tables', 'table_list', 'available_tables'],
                        'table_names': ['available_tables', 'all_tables', 'tables'],
                        'columns': ['selected_columns', 'column_list'],
                        'resolved_terms': ['terms', 'business_terms'],
                        'filters': ['filter_conditions', 'where_conditions'],
                    }
                    
                    if param in aliases:
                        for alias in aliases[param]:
                            if alias in metadata:
                                recovered[param] = metadata[alias]
                                logger.info(f"   ✅ Found '{param}' via alias '{alias}' in SharedContext['metadata']")
                                break
                    
                    if param in recovered:
                        continue
                    
                    # 🔥 A-TEAM: AGENTIC SEMANTIC SEARCH (NO FUZZY MATCHING!)
                    # Use AgenticParameterResolver for intelligent, LLM-based matching
                    logger.info(f"   🔍 Using agentic search for '{param}' in metadata...")
                    
                    if hasattr(self, 'param_resolver') and self.param_resolver:
                        # Get parameter info from signature
                        param_info = {}
                        param_type_str = 'str'
                        if actor_config.name in self.actor_signatures:
                            sig = self.actor_signatures[actor_config.name]
                            if isinstance(sig, dict) and param in sig:
                                param_info = sig[param]
                                param_type = param_info.get('annotation')
                                if param_type and hasattr(param_type, '__name__'):
                                    param_type_str = param_type.__name__
                        
                        # Prepare available data with rich descriptions
                        available_data = {}
                        for key, value in metadata.items():
                            available_data[key] = {
                                'value': value,
                                'type': type(value).__name__,
                                'description': f"Metadata key '{key}' containing {type(value).__name__}",
                                'tags': ['metadata', 'available'],
                                'source': 'SharedContext[metadata]'
                            }
                        
                        try:
                            # Use agentic resolver with LLM intelligence  
                            matched_key, confidence, reasoning = self.param_resolver.resolve_parameter(
                                actor_name=actor_config.name,
                                parameter_name=param,
                                parameter_type=param_type_str,
                                parameter_purpose=f"Required parameter for {actor_config.name}",
                                available_data=available_data
                            )
                            
                            if matched_key and matched_key in metadata:
                                if confidence > 0.7:  # High confidence threshold
                                    recovered[param] = metadata[matched_key]
                                    logger.info(f"   ✅ Agentic match: '{param}' → '{matched_key}' (confidence: {confidence:.2f})")
                                    logger.info(f"   📝 Reasoning: {reasoning[:150]}...")
                                else:
                                    logger.warning(f"   ⚠️ Low confidence match: '{param}' → '{matched_key}' (conf: {confidence:.2f})")
                                    logger.info(f"   📝 Reasoning: {reasoning[:100]}...")
                            else:
                                logger.warning(f"   ❌ No agentic match found for '{param}'")
                        
                        except Exception as e:
                            logger.error(f"   ❌ Agentic resolver failed: {e}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                    else:
                        logger.warning(f"   ⚠️ AgenticParameterResolver not available - parameter '{param}' cannot be resolved from metadata")
            
            if param in recovered:
                continue
            
            # Strategy 3: Invoke MetaDataFetcher for on-demand fetch
            if self.metadata_fetcher and hasattr(self.metadata_fetcher, 'tools'):
                logger.info(f"   🔍 Asking MetaDataFetcher to fetch '{param}'...")
                try:
                    # Check if any tool matches this parameter
                    for tool in self.metadata_fetcher.tools:
                        tool_name = tool.name.lower() if hasattr(tool, 'name') else str(tool)
                        if param.lower() in tool_name or tool_name in param.lower():
                            logger.info(f"   📞 Calling tool '{tool.name}' for '{param}'")
                            try:
                                result = tool.func() if hasattr(tool, 'func') else None
                                if result:
                                    recovered[param] = result
                                    logger.info(f"   ✅ Fetcher retrieved '{param}'")
                                    break
                            except Exception as e:
                                logger.warning(f"   ⚠️  Tool call failed: {e}")
                except Exception as e:
                    logger.warning(f"   ⚠️  MetaDataFetcher recovery failed: {e}")
            
            if param not in recovered:
                logger.warning(f"   ❌ Could not recover '{param}'")
        
        still_missing = [p for p in missing_params if p not in recovered]
        
        logger.info(f"🏁 Recovery complete: {len(recovered)}/{len(missing_params)} recovered")
        if still_missing:
            logger.warning(f"   Still missing: {still_missing}")
        
        return {
            'recovered': recovered,
            'still_missing': still_missing
        }
    
    def _find_parameter_producer(self, parameter_name: str, requesting_actor: str) -> Optional[str]:
        """
        Find which actor produces a given parameter using dependency graph.
        
        🔥 A-TEAM: Uses signature introspection + IOManager to route requests intelligently.
        """
        logger.info(f"🔍 [DEP GRAPH] Finding producer for '{parameter_name}' (requested by {requesting_actor})")
        
        # Check actor signatures for outputs
        for actor_name, sig in self.actor_signatures.items():
            if actor_name == requesting_actor:
                continue  # Don't ask yourself
            
            # Check if signature has this in outputs
            if isinstance(sig, dict) and 'outputs' in sig:
                if parameter_name in sig['outputs']:
                    logger.info(f"✅ [DEP GRAPH] {actor_name} produces '{parameter_name}' (from signature)")
                    return actor_name
        
        # Check IOManager for actual outputs
        if hasattr(self, 'io_manager') and self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            for actor_name in all_outputs:
                if actor_name == requesting_actor:
                    continue
                output_fields = self.io_manager.get_output_fields(actor_name)
                if parameter_name in output_fields:
                    logger.info(f"✅ [DEP GRAPH] {actor_name} produces '{parameter_name}' (from IOManager)")
                    return actor_name
        
        logger.warning(f"⚠️ [DEP GRAPH] No producer found for '{parameter_name}'")
        return None
    
    async def _route_to_producer(
        self,
        producer_actor: str,
        parameters_needed: List[str],
        requesting_actor: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route request to producer actor and retrieve data.
        
        🔥 A-TEAM: Intelligent routing using dependency graph.
        If producer already executed, retrieve from IOManager.
        If not, execute producer first.
        """
        logger.info(f"📢 [ROUTING] {requesting_actor} → {producer_actor}")
        logger.info(f"   Requesting: {parameters_needed}")
        
        # Check if producer already executed
        if hasattr(self, 'io_manager') and self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            if producer_actor in all_outputs:
                logger.info(f"✅ [ROUTING] {producer_actor} already executed")
                output_fields = self.io_manager.get_output_fields(producer_actor)
                
                recovered = {}
                for param in parameters_needed:
                    if param in output_fields:
                        recovered[param] = output_fields[param]
                        logger.info(f"   ✅ Retrieved '{param}': {type(output_fields[param]).__name__}")
                
                if recovered:
                    return recovered
        
        # Producer hasn't executed - need to execute it first
        logger.info(f"⚠️ [ROUTING] {producer_actor} not executed yet, executing now...")
        
        if producer_actor not in self.actors:
            logger.error(f"❌ [ROUTING] Producer actor '{producer_actor}' not found in swarm")
            return {}
        
        producer_config = self.actors[producer_actor]
        
        # Build kwargs for producer
        try:
            producer_kwargs = {}
            
            # Get producer's signature
            if producer_actor in self.actor_signatures:
                sig = self.actor_signatures[producer_actor]
                if isinstance(sig, dict):
                    for param_name, param_info in sig.items():
                        # Resolve producer's parameters
                        value = self._resolve_parameter(
                            param_name,
                            param_info,
                            context.get('kwargs', {}),
                            self.shared_context.data if hasattr(self, 'shared_context') else {}
                        )
                        if value is not None:
                            producer_kwargs[param_name] = value
            
            # Execute producer
            logger.info(f"🔄 [ROUTING] Executing {producer_actor} to generate data...")
            
            # Get or create JOTTY-wrapped actor
            actor = producer_config.actor
            if hasattr(actor, 'arun'):
                result = await actor.arun(context.get('goal', ''), **producer_kwargs)
            else:
                # Not wrapped, execute directly
                result = actor(**producer_kwargs)
                # Wrap in EpisodeResult
                from ..foundation.data_structures import EpisodeResult
                result = EpisodeResult(
                    output=result,
                    success=True,
                    trajectory=[],
                    tagged_outputs=[],
                    episode=0
                )
            
            if result.success and result.output:
                # Register output in IOManager
                if hasattr(self, 'io_manager'):
                    self.io_manager.register_output(
                        actor_name=producer_actor,
                        output=result.output,
                        actor=actor,
                        success=True
                    )
                
                # Extract requested parameters
                recovered = {}
                for param in parameters_needed:
                    if hasattr(result.output, param):
                        recovered[param] = getattr(result.output, param)
                    elif hasattr(result.output, '_store') and param in result.output._store:
                        recovered[param] = result.output._store[param]
                
                logger.info(f"✅ [ROUTING] Producer executed, recovered {len(recovered)} params")
                return recovered
            else:
                logger.error(f"❌ [ROUTING] Producer {producer_actor} failed")
                return {}
        
        except Exception as e:
            logger.error(f"❌ [ROUTING] Failed to execute producer {producer_actor}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
    
    async def _intelligent_recovery_with_routing(
        self,
        actor_config: ActorConfig,
        missing_params: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Intelligent parameter recovery using dependency graph routing.
        
        🔥 A-TEAM: Full intelligence stack:
        1. Check IOManager (already executed actors)
        2. Check SharedContext metadata
        3. Use dependency graph to find producers
        4. Route requests to producer actors
        5. Re-execute producers if needed
        6. Fallback to MetaDataFetcher
        7. Fallback to smart defaults
        """
        recovered = {}
        logger.info(f"🛠️  [INTELLIGENT RECOVERY] for {actor_config.name}")
        logger.info(f"   Missing: {missing_params}")
        
        for param in missing_params:
            # Strategy 1: IOManager (already executed)
            if hasattr(self, 'io_manager') and self.io_manager:
                all_outputs = self.io_manager.get_all_outputs()
                for actor_name in all_outputs:
                    output_fields = self.io_manager.get_output_fields(actor_name)
                    if param in output_fields:
                        recovered[param] = output_fields[param]
                        logger.info(f"   ✅ Found '{param}' in IOManager[{actor_name}]")
                        break
            
            if param in recovered:
                continue
            
            # Strategy 2: SharedContext metadata
            if hasattr(self, 'shared_context') and self.shared_context:
                if self.shared_context.has('metadata'):
                    metadata = self.shared_context.get('metadata')
                    if isinstance(metadata, dict) and param in metadata:
                        recovered[param] = metadata[param]
                        logger.info(f"   ✅ Found '{param}' in SharedContext[metadata]")
                        continue
            
            # Strategy 3: 🔥 DEPENDENCY GRAPH ROUTING
            producer = self._find_parameter_producer(param, actor_config.name)
            if producer:
                logger.info(f"   🎯 Dependency graph: '{param}' from {producer}")
                
                # Route request to producer
                producer_data = await self._route_to_producer(
                    producer_actor=producer,
                    parameters_needed=[param],
                    requesting_actor=actor_config.name,
                    context=context
                )
                
                if param in producer_data:
                    recovered[param] = producer_data[param]
                    logger.info(f"   ✅ Routed to {producer}, retrieved '{param}'")
                    continue
            
            # Strategy 4: MetaDataFetcher (fallback)
            if hasattr(self, 'metadata_fetcher') and self.metadata_fetcher:
                logger.info(f"   🔍 Asking MetaDataFetcher for '{param}'...")
                # ... (existing MetaDataFetcher logic if needed)
        
        logger.info(f"🏁 [INTELLIGENT RECOVERY] Complete: {len(recovered)}/{len(missing_params)} recovered")
        if len(recovered) < len(missing_params):
            still_missing = [p for p in missing_params if p not in recovered]
            logger.warning(f"   Still missing: {still_missing}")
        
        return recovered
    
    def _introspect_actor_signature(self, actor_config: ActorConfig):
        """
        Introspect actor's signature for auto-resolution.
        🔥 A-TEAM FIX: Use DSPy Signature object, NOT forward() method directly!
        """
        actor = actor_config.agent

        # Strategy 1a: DSPy ChainOfThought (has predict.signature)
        if hasattr(actor, 'predict') and hasattr(actor.predict, 'signature'):
            try:
                signature = actor.predict.signature
                if hasattr(signature, 'input_fields'):
                    params = {}
                    for field_name in signature.input_fields:
                        # 🔥 A-TEAM CRITICAL FIX: Extract REAL type from DSPy field!
                        field = signature.input_fields[field_name]
                        field_type = Any  # default

                        # DSPy fields have a 'annotation' or '_type' attribute
                        if hasattr(field, 'annotation'):
                            field_type = field.annotation
                        elif hasattr(field, '_type'):
                            field_type = field._type
                        elif hasattr(field, '__annotations__'):
                            # Check class annotations
                            for cls in type(field).__mro__:
                                if hasattr(cls, '__annotations__') and field_name in cls.__annotations__:
                                    field_type = cls.__annotations__[field_name]
                                    break

                        # If still Any, try to infer from field's json_schema_extra or desc
                        if field_type is Any:
                            logger.debug(f"   ⚠️  Could not extract type for '{field_name}', defaulting to Any")

                        params[field_name] = {
                            'annotation': field_type,
                            'default': inspect.Parameter.empty,
                            'required': True
                        }
                    self.actor_signatures[actor_config.name] = params
                    self.dependency_graph_dict[actor_config.name] = []
                    logger.info(f"  📋 {actor_config.name}: {len(params)} params (DSPy ChainOfThought), deps=[]")
                    return
            except Exception as e:
                logger.warning(f"⚠️  Failed to extract DSPy ChainOfThought signature for {actor_config.name}: {e}")

        # Strategy 1b: DSPy module with signature attribute (BEST)
        if hasattr(actor, 'signature') and hasattr(actor.signature, 'input_fields'):
            try:
                params = {}
                for field_name in actor.signature.input_fields:
                    # 🔥 A-TEAM CRITICAL FIX: Extract REAL type from DSPy field!
                    field = actor.signature.input_fields[field_name]
                    field_type = Any  # default

                    # DSPy fields have a 'annotation' or '_type' attribute
                    if hasattr(field, 'annotation'):
                        field_type = field.annotation
                    elif hasattr(field, '_type'):
                        field_type = field._type
                    elif hasattr(field, '__annotations__'):
                        # Check class annotations
                        for cls in type(field).__mro__:
                            if hasattr(cls, '__annotations__') and field_name in cls.__annotations__:
                                field_type = cls.__annotations__[field_name]
                                break

                    # If still Any, try to infer from field's json_schema_extra or desc
                    if field_type is Any:
                        logger.debug(f"   ⚠️  Could not extract type for '{field_name}', defaulting to Any")

                    params[field_name] = {
                        'annotation': field_type,
                        'default': inspect.Parameter.empty,
                        'required': True
                    }
                self.actor_signatures[actor_config.name] = params
                self.dependency_graph_dict[actor_config.name] = []
                logger.info(f"  📋 {actor_config.name}: {len(params)} params (DSPy signature), deps=[]")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to extract DSPy signature for {actor_config.name}: {e}")
        
        # Strategy 2: Inspect forward method (WITHOUT calling it)
        forward_method = getattr(actor, 'forward', None)
        if forward_method:
            try:
                sig = inspect.signature(forward_method)
                params = {}
                dependencies = []
                
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    params[param_name] = {
                        'annotation': param.annotation,
                        'default': param.default,
                        'required': param.default == inspect.Parameter.empty
                    }
                
                self.actor_signatures[actor_config.name] = params
                self.dependency_graph_dict[actor_config.name] = dependencies
                
                logger.info(f"  📋 {actor_config.name}: {len(params)} params, deps={dependencies}")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to introspect forward method for {actor_config.name}: {e}")
        
        # Strategy 3: Fallback - inspect __call__
        if hasattr(actor, '__call__'):
            try:
                sig = inspect.signature(actor.__call__)
                params = {}
                for param_name, param in sig.parameters.items():
                    if param_name in ('self', 'args', 'kwargs'):
                        continue
                    params[param_name] = {
                        'annotation': param.annotation,
                        'default': param.default,
                        'required': param.default == inspect.Parameter.empty
                    }
                self.actor_signatures[actor_config.name] = params
                self.dependency_graph_dict[actor_config.name] = []
                logger.info(f"  📋 {actor_config.name}: {len(params)} params (__call__), deps=[]")
                return
            except Exception as e:
                logger.warning(f"⚠️  Failed to introspect __call__ for {actor_config.name}: {e}")
        
        # Fallback: No signature
        logger.warning(f"⚠️  Actor {actor_config.name} has no inspectable signature")
        self.actor_signatures[actor_config.name] = {}
        self.dependency_graph_dict[actor_config.name] = []
    
    def resolve_input(self, input_spec: str, resolution_context: Dict) -> Any:
        """
        🆕 DECLARATIVE INPUT RESOLUTION (A-Team FINAL Solution)
        
        Resolve input from specification string with natural syntax:
        
        Examples:
            "input.query" → From swarm.run() kwargs
            "BusinessTermResolver.required_tables" → From previous actor output
            "context.current_date" → From context_providers
            "metadata.get_all_validations()" → Call metadata_manager method
            "metadata.get_partition_info_for_tables(BusinessTermResolver.required_tables)" → Method with arg
        
        Args:
            input_spec: Specification string describing where to get the value
            resolution_context: Dict containing:
                - 'inputs': kwargs from swarm.run()
                - 'actor_outputs': Dict of previous actor outputs
                - 'context_providers': Dict of context providers (metadata_manager, etc.)
        
        Returns:
            Resolved value or None
        """
        input_spec = input_spec.strip()
        
        # Pattern 1: input.{field} - From swarm.run() kwargs
        if input_spec.startswith("input."):
            field = input_spec.split(".", 1)[1]
            value = resolution_context.get('inputs', {}).get(field)
            if value is not None:
                logger.debug(f"   ✅ Resolved from input.{field}")
            return value
        
        # Pattern 2: context.{field} - From context_providers
        elif input_spec.startswith("context."):
            field = input_spec.split(".", 1)[1]
            value = resolution_context.get('context_providers', {}).get(field)
            if value is not None:
                logger.debug(f"   ✅ Resolved from context.{field}")
            return value
        
        # Pattern 3: metadata.{method}(...) or metadata.{attr} - From metadata_manager
        elif input_spec.startswith("metadata."):
            method_or_attr = input_spec.split(".", 1)[1]
            context_providers = resolution_context.get('context_providers', {})
            # ✅ A-TEAM FIX: Look for 'metadata' (new) or 'metadata_manager' (legacy)
            metadata_manager = context_providers.get('metadata') or context_providers.get('metadata_manager')
            
            logger.info(f"   🔍 Trying to resolve metadata.{method_or_attr}")
            logger.info(f"   🔍 context_providers keys: {list(context_providers.keys())}")
            logger.info(f"   🔍 metadata_manager found: {metadata_manager is not None}")
            
            if not metadata_manager:
                logger.warning(f"⚠️  metadata_manager not found in context_providers for '{input_spec}'")
                logger.warning(f"⚠️  Available providers: {list(context_providers.keys())}")
                return None
            
            # Check if it's a method call (has parentheses)
            if "(" in method_or_attr:
                method_name = method_or_attr.split("(")[0]
                args_str = method_or_attr.split("(")[1].rstrip(")")
                
                if not hasattr(metadata_manager, method_name):
                    logger.warning(f"⚠️  Method '{method_name}' not found on metadata_manager")
                    return None
                
                method = getattr(metadata_manager, method_name)
                
                # Parse and resolve arguments
                try:
                    if args_str.strip():
                        args = []
                        for arg_spec in args_str.split(","):
                            arg_spec = arg_spec.strip()
                            # Recursively resolve the argument
                            arg_value = self.resolve_input(arg_spec, resolution_context)
                            args.append(arg_value)
                        
                        logger.info(f"   🔧 Calling {method_name}({args})")
                        result = method(*args)
                        logger.info(f"   ✅ Result: {str(result)[:200]}")
                        return result
                    else:
                        logger.info(f"   🔧 Calling {method_name}()")
                        result = method()
                        logger.info(f"   ✅ Result: {str(result)[:200]}")
                        return result
                except Exception as e:
                    logger.error(f"   ❌ Error calling metadata.{method_name}: {e}")
                    import traceback
                    logger.error(f"   ❌ Traceback: {traceback.format_exc()}")
                    return None
            else:
                # Attribute access
                if hasattr(metadata_manager, method_or_attr):
                    value = getattr(metadata_manager, method_or_attr)
                    logger.debug(f"   ✅ Resolved from metadata.{method_or_attr}")
                    return value
                else:
                    logger.warning(f"⚠️  Attribute '{method_or_attr}' not found on metadata_manager")
                    return None
        
        # Pattern 4: {Actor}.{field} - From previous actor output
        elif "." in input_spec:
            actor_name, field = input_spec.split(".", 1)
            actor_outputs = resolution_context.get('actor_outputs', {})
            actor_output = actor_outputs.get(actor_name)
            
            if not actor_output:
                logger.debug(f"   ⚠️  Actor '{actor_name}' output not found")
                return None
            
            # Extract field from output using existing extraction logic
            value = self._extract_from_output(actor_output, field)
            if value is not None:
                logger.debug(f"   ✅ Resolved from {actor_name}.{field}")
            return value
        
        # Pattern 5: Direct value (fallback)
        logger.warning(f"⚠️  Unknown input specification format: '{input_spec}'")
        return None
    
    def _resolve_parameter(self, param_name: str, param_info: Dict, kwargs: Dict, shared_context: Dict) -> Any:
        """
        Resolve parameter from multiple sources.
        
#         🔥 A-TEAM FIX: CORRECT priority order!
        1. Direct kwargs (explicit overrides)
        2. SharedContext (for GLOBAL parameters like current_date)
        3. IOManager (previous actor outputs) ← THE MAIN DATA FLOW
        4. Metadata tools
        5. Defaults
        
#         ❌ NEVER resolve domain data from SharedContext['metadata'] - agents should call tools!
        """
        
        # Priority 1: Direct kwargs (explicit overrides)
        if param_name in kwargs:
            logger.debug(f"✅ Resolved '{param_name}' from kwargs")
            return kwargs[param_name]

        # Priority 2: SharedContext - Check for parameters stored from run() kwargs or global parameters
        # This includes: run() kwargs (like 'code', 'request'), global params (current_date), goal/query
        if hasattr(self, 'shared_context') and self.shared_context:
            if self.shared_context.has(param_name):
                value = self.shared_context.get(param_name)
                logger.info(f"✅ Resolved '{param_name}' from SharedContext")
                return value
        
        # Priority 3: IOManager - previous actor outputs (MAIN DATA FLOW!)
        if hasattr(self, 'io_manager') and self.io_manager:
            all_outputs = self.io_manager.get_all_outputs()
            for actor_name, actor_output in all_outputs.items():
                if hasattr(actor_output, 'output_fields') and actor_output.output_fields:
                    # Exact match
                    if param_name in actor_output.output_fields:
                        value = actor_output.output_fields[param_name]
                        logger.info(f"✅ Resolved '{param_name}' from IOManager['{actor_name}']")
                        
                        # 🔥 A-TEAM FIX: AGENTIC TYPE TRANSFORMATION (ReAct agent with sandbox!)
                        expected_type = param_info.get('annotation', Any)
                        if expected_type != Any and value is not None:
                            actual_type = type(value)
                            # Get origin for typing generics (List[str] → list, Dict[str, Any] → dict)
                            import typing
                            expected_origin = typing.get_origin(expected_type) or expected_type
                            
                            # 🔥 A-TEAM FIX: Skip transformation for Union types IMMEDIATELY!
                            # Union types ALWAYS fail isinstance() checks and waste tokens
                            expected_str = str(expected_type)
                            if 'Union' in expected_str or expected_origin is typing.Union:
                                logger.debug(f"⏩ Skipping transformation for Union type: {expected_str}")
                                logger.debug(f"   Using raw value: {actual_type.__name__} (DSPy will coerce)")
                                continue  # Skip to next parameter
                            
                            if actual_type != expected_origin:
                                actual_name = actual_type.__name__
                                expected_name = getattr(expected_origin, '__name__', str(expected_type))
                                logger.info(f"🔄 Type mismatch detected: {actual_name} → {expected_name}")
                                
                                # 🔥 A-TEAM LEARNING FIX: Check memory for past transformation failures!
                                should_skip_transformation = False
                                if hasattr(self, 'shared_memory') and self.shared_memory:
                                    try:
                                        past_failures = self.shared_memory.retrieve(
                                            query=f"SmartDataTransformer {param_name} {actor_name} transformation failure",
                                            top_k=3
                                        )
                                        
                                        if past_failures:
                                            logger.info(f"🧠 LEARNING: Found {len(past_failures)} past transformation failures for {param_name}")
                                            
                                            # Check if Union isinstance error happened before
                                            for failure in past_failures:
                                                failure_content = str(failure.content) if hasattr(failure, 'content') else str(failure)
                                                if "Union cannot be used with isinstance" in failure_content or "Union" in failure_content:
                                                    logger.info(f"🧠 LEARNING: Skipping SmartDataTransformer for Union type (learned from past failure)")
                                                    logger.info(f"   📦 Using raw value: {type(value).__name__}")
                                                    should_skip_transformation = True
                                                    break
                                    except Exception as mem_e:
                                        logger.debug(f"Memory query failed: {mem_e}")
                                
                                if should_skip_transformation:
                                    # Skip transformation, use raw value
                                    logger.info(f"   📦 DSPy ReAct will handle type conversion")
                                else:
                                    logger.info(f"   🤖 Invoking SmartDataTransformer (ReAct agent with sandbox)")
                                    try:
                                        # Call transformer synchronously (it handles async internally)
                                        transformed = self.data_transformer.transform(
                                            source=value,
                                            target_type=expected_origin,  # Use origin type (list, dict, not List[str])
                                            context=f"Parameter '{param_name}' from {actor_name}. Agent needs {expected_name} to proceed.",
                                            param_name=param_name
                                        )
                                        if transformed is not None:
                                            value = transformed
                                            logger.info(f"✅ Agentic transformation successful: {type(value).__name__}")
                                        else:
                                            logger.warning(f"⚠️  Transformer returned None - using original value")
                                    except Exception as e:
                                        # 🔥 A-TEAM FIX: Don't fail the entire actor execution on transformation failure!
                                        # Just log the error and continue with the original value
                                        # DSPy ReAct signature system will handle type coercion
                                        logger.warning(f"⚠️  Agentic transformation failed: {e}")
                                        logger.info(f"   📦 Continuing with original value, DSPy will handle type conversion")
                        
                        return value
        
                    # 🔥 A-TEAM FIX: GENERIC PARAMETER ALIASING
                    # Common semantic aliases (NO domain-specific logic!)
                    aliases = {
                        'tables': ['relevant_tables', 'selected_tables', 'table_list', 'available_tables'],
                        'table_names': ['available_tables', 'all_tables', 'tables'],
                        'columns': ['selected_columns', 'column_list'],
                        'resolved_terms': ['terms', 'business_terms'],
                        'filters': ['filter_conditions', 'where_conditions'],
                    }
                    
                    # Check if param_name has known aliases
                    if param_name in aliases:
                        for alias in aliases[param_name]:
                            if alias in actor_output.output_fields:
                                value = actor_output.output_fields[alias]
                                logger.info(f"✅ Resolved '{param_name}' from IOManager['{actor_name}']['{alias}'] (alias)")
                                
                                # 🔥 CRITICAL FIX: Return immediately! Don't wait for type transformation!
                                # Type transformation is optional - if types match, we still need to return!
                                return value
                                
                                # NOTE: Code below (type transformation) is now unreachable but kept for reference
                                # TODO: Remove this dead code in future cleanup
                                # 🔥 A-TEAM FIX: AGENTIC TYPE TRANSFORMATION (ReAct agent with sandbox!)
                                expected_type = param_info.get('annotation', Any)
                                if expected_type != Any and value is not None:
                                    actual_type = type(value)
                                    # Get origin for typing generics
                                    import typing
                                    expected_origin = typing.get_origin(expected_type) or expected_type
                                    
                                    # 🔥 A-TEAM FIX: Skip transformation for Union types IMMEDIATELY!
                                    expected_str = str(expected_type)
                                    if 'Union' in expected_str or expected_origin is typing.Union:
                                        logger.debug(f"⏩ Skipping transformation for Union type: {expected_str}")
                                        logger.debug(f"   Using raw value: {actual_type.__name__} (DSPy will coerce)")
                                    elif actual_type != expected_origin:
                                        actual_name = actual_type.__name__
                                        expected_name = getattr(expected_origin, '__name__', str(expected_type))
                                        logger.info(f"🔄 Type mismatch detected: {actual_name} → {expected_name}")
                                        logger.info(f"   🤖 Invoking SmartDataTransformer (ReAct agent)")
                                        try:
                                            # Call transformer synchronously
                                            transformed = self.data_transformer.transform(
                                                source=value,
                                                target_type=expected_origin,
                                                context=f"Parameter '{param_name}' from {actor_name}. Agent needs {expected_name}.",
                                                param_name=param_name
                                            )
                                            if transformed is not None:
                                                value = transformed
                                                logger.info(f"✅ Agentic transformation successful: {type(value).__name__}")
                                            else:
                                                logger.error(f"❌ Transformer returned None!")
                                        except Exception as e:
                                            logger.error(f"❌ Agentic transformation failed: {e}")
                                            raise RuntimeError(f"SmartDataTransformer failed: {e}") from e
                                
                                return value
        
        # Priority 3: Previous actor outputs from shared_context['actor_outputs'] (legacy path)
        actor_outputs = shared_context.get('actor_outputs', {})
        for actor_name, output in actor_outputs.items():
            value = self._extract_from_output(output, param_name)
            if value is not None:
                logger.info(f"✅ Resolved '{param_name}' from actor '{actor_name}' output")
                return value
        
        # Priority 4: SharedContext - ONLY for 'goal', 'query', 'conversation_history' (NOT metadata!)
        if hasattr(self, 'shared_context') and self.shared_context:
            # Whitelist of allowed SharedContext keys for parameters
            allowed_keys = {'goal', 'query', 'conversation_history', 'session_id'}
            
            if param_name in allowed_keys:
                # Try exact match
                if self.shared_context.has(param_name):
                    value = self.shared_context.get(param_name)
                    logger.info(f"✅ Resolved '{param_name}' from SharedContext")
                    return value
                
                # Try semantic match ONLY within allowed keys
                for key in allowed_keys:
                    if self.shared_context.has(key) and (param_name.lower() in key.lower() or key.lower() in param_name.lower()):
                        value = self.shared_context.get(key)
                        logger.info(f"✅ Resolved '{param_name}' from SharedContext['{key}'] (semantic match)")
                        return value  # 🔧 A-TEAM FIX: Return immediately when found!
        
        # Priority 5: Context providers (direct match)
        if param_name in self.context_providers:
            logger.info(f"✅ Resolved '{param_name}' from context_providers")
            return self.context_providers[param_name]
        
        # Priority 6: Shared context dict (non-metadata keys only)
        if param_name in shared_context and param_name != 'metadata':
            logger.info(f"✅ Resolved '{param_name}' from shared_context dict")
            return shared_context[param_name]
        
        # ✅ Priority 6.5: SharedContext['metadata'] - USER FEEDBACK: This was missing!
        # Check metadata directly instead of waiting for recovery
        metadata = shared_context.get('metadata', {})
        if param_name in metadata:
            logger.info(f"✅ Resolved '{param_name}' from SharedContext['metadata']")
            return metadata[param_name]
        
        # Check metadata with aliases (GENERIC - not domain specific!)
        aliases = {
            'tables': ['relevant_tables', 'selected_tables', 'table_list', 'available_tables', 'get_all_tables'],
            'table_names': ['available_tables', 'tables', 'relevant_tables', 'selected_tables', 'table_list', 'get_all_tables'],
            'columns': ['column_list', 'selected_columns', 'relevant_columns'],
            'resolved_terms': ['business_terms', 'terms', 'term_mapping', 'get_business_terms'],
            'tables_metadata': ['get_all_table_metadata', 'table_metadata', 'schema_info'],
            'columns_metadata': ['column_metadata', 'columns_info'],
            'business_terms': ['get_business_terms', 'business_context', 'get_business_context'],
        }
        param_aliases = aliases.get(param_name, [])
        for alias in param_aliases:
            if alias in metadata:
                logger.info(f"✅ Resolved '{param_name}' via alias '{alias}' from SharedContext['metadata']")
                return metadata[alias]
        
        # Priority 7: Default value from signature
        if param_info['default'] != inspect.Parameter.empty:
            logger.debug(f"✅ Using default value for '{param_name}'")
            return param_info['default']
        
        # No resolution found
        logger.debug(f"❌ Cannot resolve parameter '{param_name}'")
        return None
    
    def _extract_from_metadata_manager(self, metadata_manager, param_name: str) -> Any:
        """
        Extract parameter from metadata_manager (user-provided metadata).
        
        NO HARDCODING - just simple attribute mapping!
        """
        # Generic attribute mapping (NO domain-specific logic!)
        metadata_mappings = {
            'validation_criterias': 'validations',
            'business_context': 'business_context',
            'term_glossary': 'term_glossary',
            'filter_conditions': 'filter_conditions',
            'joining_conditions': 'joining_conditions',
            'table_metadata': 'table_metadata',
            'column_metadata': 'column_metadata',
            'widgets_context': 'widgets_context',
        }
        
        # Try direct mapping first
        attr_name = metadata_mappings.get(param_name, param_name)
        
        if hasattr(metadata_manager, attr_name):
            value = getattr(metadata_manager, attr_name)
            if value:
                logger.debug(f"   Extracted from metadata_manager.{attr_name}")
                return value
        
        return None
    
    def _semantic_extract(self, output: Any, param_name: str) -> Any:
        """
        Semantic extraction using LLM understanding.
        
        Replaces fuzzy matching with LLM-based semantic understanding.
        Handles synonyms, typos, variations.
        
        A-Team Decision: This is the NEW way to resolve parameters.
        """
        try:
            # Build available fields
            available = {}
            if hasattr(output, '__dict__'):
                available = {k: v for k, v in vars(output).items() if not k.startswith('_')}
            elif isinstance(output, dict):
                available = output
            else:
                return None  # Can't extract from non-object/non-dict
            
            if not available:
                return None
            
            # Quick check: exact match (skip LLM)
            if param_name in available:
                return available[param_name]
            
            # Ask LLM which field matches param_name
            match_result = self._llm_match_field(param_name, available)
            
            if match_result and match_result.confidence > 0.5:
                matched_field = match_result.field_name
                if matched_field in available:
                    logger.info(f"✅ Semantic match: '{param_name}' → '{matched_field}' (confidence={match_result.confidence:.2f})")
                    logger.info(f"   Reasoning: {match_result.reasoning}")
                    return available[matched_field]
            
            return None
        
        except Exception as e:
            logger.error(f"❌ Semantic extraction error: {e}")
            return None
    
    def _llm_match_field(self, param_name: str, available_fields: Dict) -> Optional[Any]:
        """Use LLM to match parameter name to available fields."""
        try:
            import dspy
            from dataclasses import dataclass
            
            class FieldMatchSignature(dspy.Signature):
                """Match parameter name to available field."""
                
                parameter_needed = dspy.InputField(desc="Parameter name being requested")
                available_fields = dspy.InputField(desc="Available fields with previews")
                
                best_match = dspy.OutputField(desc="""
                    Best matching field name from available fields.
                    
                    Consider:
                    - Semantic similarity (e.g., 'schema' matches 'structure')
                    - Typos (e.g., 'meta' matches 'metadata')
                    - Variations (e.g., 'user_data' matches 'userData')
                    
                    Return the exact field name or 'none' if no good match.
                """)
                
                confidence = dspy.OutputField(desc="Confidence in match (0.0-1.0)")
                reasoning = dspy.OutputField(desc="Brief explanation")
            
            # Format available fields for LLM
            fields_str = "\n".join([
                f"  - {name}: {type(value).__name__} = {str(value)[:100]}..."
                for name, value in available_fields.items()
            ])
            
            matcher = dspy.ChainOfThought(FieldMatchSignature)
            result = matcher(
                parameter_needed=param_name,
                available_fields=fields_str
            )
            
            @dataclass
            class MatchResult:
                field_name: str
                confidence: float
                reasoning: str
            
            # Parse confidence
            try:
                confidence = float(result.confidence)
                confidence = max(0.0, min(1.0, confidence))  # Clamp
            except (ValueError, TypeError, AttributeError):
                # Default to 0.5 if confidence parsing fails
                confidence = 0.5
            
            return MatchResult(
                field_name=result.best_match,
                confidence=confidence,
                reasoning=result.reasoning
            )
        
        except Exception as e:
            logger.warning(f"⚠️ LLM field matching unavailable: {e}")
            return None
    
    def _extract_from_output(self, output, param_name: str) -> Any:
        """Extract parameter from previous actor's output.
        
        Handles:
        1. EpisodeResult (from wrapped JOTTY actors) → extract from .output or tagged_outputs
        2. DSPy Prediction objects
        3. Dict-like outputs
        4. Parameter name variations (generic patterns)
        5. DataFrames, files, web content (extensible)
        
        🔑 A-TEAM FIX: Unwrap nested EpisodeResults WITHOUT recursion to prevent stack overflow!
        🆕 A-TEAM: Now uses SEMANTIC extraction instead of fuzzy matching!
        """
        # 🔍 A-TEAM DEBUG: Log extraction attempt (DEBUG level to reduce noise)
        logger.debug(f"🔍 [EXTRACT] Attempting to extract '{param_name}' from output type: {type(output)}")
        
        # 🔑 CRITICAL FIX: Unwrap nested EpisodeResults iteratively (NO RECURSION!)
        unwrapped = output
        max_unwrap = 10  # Safety limit
        unwrap_count = 0
        
        # ✅ A-TEAM FIX: Check if it's ACTUALLY an EpisodeResult, not just has 'output' attribute
        # DSPy Predictions may have 'output' field in signature, but they're not EpisodeResults!
        from ..foundation.data_structures import EpisodeResult
        
        logger.debug(f"🔍 [EXTRACT] Is EpisodeResult: {isinstance(output, EpisodeResult)}")
        
        while isinstance(unwrapped, EpisodeResult) and unwrap_count < max_unwrap:
            logger.debug(f"🔍 [EXTRACT] Unwrapping EpisodeResult #{unwrap_count+1}")
            # This is an EpisodeResult
            if unwrapped.output is not None:
                logger.debug(f"🔍 [EXTRACT] EpisodeResult.output is NOT None, unwrapping...")
                unwrapped = unwrapped.output
                unwrap_count += 1
            else:
                logger.warning(f"⚠️  EpisodeResult.output is None, no trajectory output for {param_name}")
                # EpisodeResult.output is None, try tagged_outputs
                if hasattr(unwrapped, 'tagged_outputs') and unwrapped.tagged_outputs:
                    logger.info(f"🔍 Checking {len(unwrapped.tagged_outputs)} tagged outputs for {param_name}")
                    for tagged in unwrapped.tagged_outputs:
                        if hasattr(tagged, 'content'):
                            # Recursively unwrap tagged content (but with depth limit now)
                            value = self._extract_from_output(tagged.content, param_name)
                            if value is not None:
                                logger.info(f"✅ Found {param_name} in tagged output")
                                return value
                
                # Try trajectory if available
                if hasattr(unwrapped, 'trajectory'):
                    for step in unwrapped.trajectory:
                        if step.get('step') == 'actor' and 'output' in step:
                            step_output = step['output']
                            if step_output is not None:
                                logger.info(f"🔄 Found output in trajectory, extracting '{param_name}'")
                                # Don't recurse, just set unwrapped and let the loop continue
                                unwrapped = step_output
                                unwrap_count += 1
                                break
                    else:
                        # No output in trajectory either
                        logger.warning(f"⚠️  EpisodeResult.output is None, no trajectory output for {param_name}")
                        return None
                else:
                    logger.warning(f"⚠️  EpisodeResult.output is None, cannot extract {param_name}")
                    return None
        
        if unwrap_count >= max_unwrap:
            logger.error(f"❌ Max unwrap depth ({max_unwrap}) reached for '{param_name}' - possible circular reference!")
            return None
        
        # Now unwrapped is the actual data (Prediction, dict, etc.)
        output = unwrapped
        
        # Direct attribute access
        if hasattr(output, param_name):
            return getattr(output, param_name)
        
        # Dict access
        if isinstance(output, dict) and param_name in output:
            return output[param_name]
        
        # 🆕 A-TEAM: Use user-configurable mappings (NOT hardcoded!)
        # Try pattern matching with user-defined mappings
        if param_name in self.param_mappings:
            for attr in self.param_mappings[param_name]:
                if hasattr(output, attr):
                    value = getattr(output, attr)
                    if value is not None:  # Changed: accept 0, False, empty string
                        return value
                if isinstance(output, dict) and attr in output:
                    value = output[attr]
                    if value is not None:
                        return value
        
        # 🆕 A-TEAM: Semantic extraction (LLM-based, pure agentic)
        if self.registration_orchestrator:
            try:
                value = self._semantic_extract(output, param_name)
                if value is not None:
                    return value
                else:
                    # A-TEAM FIX: This is normal - trying to extract input params from output
                    # Should be DEBUG, not ERROR
                    logger.debug(f"🔍 Semantic extraction: '{param_name}' not found in {type(output).__name__}")
                    logger.debug(f"   (This is normal - tried to extract from previous actor output)")
                    return None
            except Exception as e:
                logger.error(f"❌ Semantic extraction failed for '{param_name}': {e}")
                logger.error(f"   Fix the semantic extraction logic or actor output structure!")
                raise RuntimeError(f"Parameter extraction failed for '{param_name}' - no fallbacks allowed!") from e
        
        # No registration orchestrator - this is a configuration error
        logger.error(f"❌ RegistrationOrchestrator not available - cannot extract '{param_name}'")
        raise RuntimeError(f"RegistrationOrchestrator must be enabled for semantic extraction!")
        
        return None
    
    async def _execute_actor(
        self,
        actor_config: ActorConfig,
        task: TodoItem,
        context: str,
        kwargs: Dict,
        actor_context_dict: Optional[Dict] = None  # 🔥 A-TEAM: Actor-specific context from metadata_provider
    ) -> Any:
        """
        Execute actor with parameter resolution.
        
        🆕 DECLARATIVE MODE (A-Team FINAL): If actor_config.inputs is provided,
        use declarative resolution. Otherwise, fall back to signature introspection.
        
        🤝 COORDINATION: Check for pending feedback from other agents before execution.
        """
        actor = actor_config.agent
        
        # 🤝 NEW: Check for feedback from other agents
        if self.feedback_channel and self.feedback_channel.has_feedback(actor_config.name):
            messages = self.feedback_channel.get_for_actor(actor_config.name, clear=True)
            logger.info(f"📧 Actor '{actor_config.name}' received {len(messages)} feedback message(s)")
            
            # Format messages for injection
            feedback_context = self.feedback_channel.format_messages_for_agent(actor_config.name, messages)
            
            # Inject feedback into context for actor to see
            if 'feedback' not in kwargs:
                kwargs['feedback'] = feedback_context
            else:
                kwargs['feedback'] += "\n\n" + feedback_context
            
            logger.debug(f"📧 Feedback injected for '{actor_config.name}':\n{feedback_context[:200]}...")
        
        # Get shared context
        shared_context = kwargs.get('_shared_context', {'actor_outputs': {}})
        
        # 🔍 DEBUG: Log current state
        logger.info(f"🔍 DEBUG: Executing actor '{actor_config.name}'")
        logger.info(f"🔍 DEBUG: Available actor_outputs: {list(shared_context.get('actor_outputs', {}).keys())}")
        logger.info(f"🔍 DEBUG: _shared_context in kwargs: {'_shared_context' in kwargs}")
        
        # 🔄 AUTO-RESOLVE parameters with optional mappings override (A-Team FINAL - User Corrected)
        logger.info(f"✅ Resolving parameters for '{actor_config.name}'")

        # ⏱️  Profile parameter resolution
        if getattr(self.config, 'enable_profiling', False):
            from ..utils.profiler import timed_block
            _profiling_enabled = True
            _param_resolution_timer = timed_block(f"ParameterResolution_{actor_config.name}", component="ParameterResolution", name=f"Resolve parameters for {actor_config.name}")
            _param_resolution_timer.__enter__()
        else:
            _profiling_enabled = False
        signature = self.actor_signatures.get(actor_config.name, {})
        
        if not signature:
            # Fallback: basic kwargs
            logger.warning(f"⚠️  No signature for {actor_config.name}, using basic kwargs")
            # 🔥 A-TEAM CRITICAL FIX: Map 'goal' to 'query' for first actor!
            resolved_kwargs = {}
            for k, v in kwargs.items():
                if k in ['query', 'conversation_history', 'session_id', 'goal']:
                    resolved_kwargs[k] = v
            # Auto-map goal → query if query is missing
            if 'query' not in resolved_kwargs and 'goal' in kwargs:
                resolved_kwargs['query'] = kwargs['goal']
                logger.info(f"🔄 Auto-mapped 'goal' → 'query' for {actor_config.name}")
        else:
            # Build resolution context for mappings
            resolution_context = {
                'inputs': kwargs,  # swarm.run() kwargs
                'actor_outputs': shared_context.get('actor_outputs', {}),
                'context_providers': self.context_providers,
            }
            
            # Build resolved kwargs
            resolved_kwargs = {}
            missing_required = []
            
            for param_name, param_info in signature.items():
                # 🆕 Check if there's an explicit mapping for this parameter
                if actor_config.parameter_mappings and param_name in actor_config.parameter_mappings:
                    # Use explicit mapping (for special cases)
                    mapping_spec = actor_config.parameter_mappings[param_name]
                    logger.info(f"   🔑 Using EXPLICIT mapping: {param_name} = {mapping_spec}")
                    try:
                        value = self.resolve_input(mapping_spec, resolution_context)
                        if value is not None:
                            logger.info(f"   ✅ Resolved '{param_name}' = {str(value)[:100]}")
                        else:
                            logger.warning(f"   ⚠️  Mapping for '{param_name}' returned None!")
                    except Exception as e:
                        logger.error(f"   ❌ Error resolving '{param_name}': {e}")
                        value = None
                else:
                    # Use auto-resolution (default)
                    value = self._resolve_parameter(param_name, param_info, kwargs, shared_context)
                
                if value is not None:
                    # 🔄 A-TEAM: SMART TYPE TRANSFORMATION
                    # Check if value type matches expected type from signature
                    expected_type = param_info.get('annotation', type(None))
                    if expected_type != inspect.Parameter.empty and expected_type != type(None):
                        actual_type = type(value)
                        if actual_type != expected_type and expected_type in (dict, list, str, int, float, bool):
                            logger.debug(f"🔄 Type mismatch for '{param_name}': {actual_type.__name__} → {expected_type.__name__}")
                            try:
                                value = self.data_transformer.transform(
                                    source=value,
                                    target_type=expected_type,
                                    context=f"Parameter for {actor_config.name}",
                                    param_name=param_name
                                )
                                logger.info(f"✅ Transformed '{param_name}' to {expected_type.__name__}")
                            except Exception as e:
                                logger.warning(f"⚠️  Transformation failed for '{param_name}': {e}")
                    
                    resolved_kwargs[param_name] = value
                elif param_info['required']:
                    missing_required.append(param_name)
            
            if missing_required:
                logger.warning(f"⚠️  {actor_config.name} missing: {missing_required}")
                
                # 🔥 A-TEAM INTELLIGENT RECOVERY: Attempt to recover missing parameters
                enable_recovery = getattr(self.config, 'enable_auto_recovery', True)  # Default True
                if enable_recovery:
                    logger.info(f"🛠️  Attempting intelligent recovery for {len(missing_required)} missing parameters...")
                    recovery_result = await self._attempt_parameter_recovery(
                        actor_config=actor_config,
                        missing_params=missing_required,
                        context=kwargs,
                        shared_context=shared_context
                    )
                    
                    if recovery_result['recovered']:
                        logger.info(f"✅ Recovered {len(recovery_result['recovered'])} parameters!")
                        resolved_kwargs.update(recovery_result['recovered'])
                        # Update missing list
                        missing_required = [p for p in missing_required if p not in recovery_result['recovered']]
                    
                    if missing_required:
                        logger.error(f"❌ Still missing after recovery: {missing_required}")
                        # Return early with failure if critical parameters still missing
                        allow_partial = getattr(self.config, 'allow_partial_execution', False)
                        if not allow_partial:
                            logger.error(f"❌ Cannot execute {actor_config.name} - missing required parameters: {missing_required}")
                            # Return a minimal EpisodeResult indicating failure
                            return EpisodeResult(
                                output=None,
                                success=False,
                                trajectory=[],
                                tagged_outputs=[],
                                episode=0,
                                execution_time=0.0,
                                architect_results=[],
                                auditor_results=[],
                                agent_contributions={},
                                alerts=[f"Missing required parameters: {missing_required}"]
                            )
        
        # 🔥 A-TEAM CRITICAL FIX: Merge actor_context_dict into resolved_kwargs!
        # This is the context from metadata_provider.get_context_for_actor()
        if actor_context_dict and isinstance(actor_context_dict, dict):
            logger.info(f"🔧 Merging {len(actor_context_dict)} context items from metadata_provider into resolved_kwargs")
            for key, value in actor_context_dict.items():
                if key not in resolved_kwargs:  # Don't override existing
                    resolved_kwargs[key] = value
                    logger.debug(f"   ✅ Added '{key}' from actor_context")
                else:
                    logger.debug(f"   ⏭️  Skipped '{key}' (already in resolved_kwargs)")
        
        # 🔧 A-TEAM FIX: DON'T inject _metadata_tools as kwarg
        # The metadata tools are already available via metadata_manager
        # Injecting them as kwarg causes TypeError if forward() doesn't accept **kwargs
        if hasattr(self, 'metadata_tool_registry') and self.metadata_tool_registry:
            logger.debug(f"🔧 Metadata tools available ({len(self.metadata_tool_registry.tools)} tools) for '{actor_config.name}'")

        # ⏱️  End parameter resolution profiling
        if _profiling_enabled:
            _param_resolution_timer.__exit__(None, None, None)

        # Execute
        # 🔥 A-TEAM CRITICAL FIX: Call DSPy modules correctly!
        # DSPy modules should be called via __call__, NOT .forward()
        # The __call__ method sets up tracking and calls forward() internally
        if asyncio.iscoroutinefunction(getattr(actor, 'run', None)):
            result = await actor.run(**resolved_kwargs)
        elif asyncio.iscoroutinefunction(getattr(actor, 'arun', None)):
            result = await actor.arun(**resolved_kwargs)
        elif hasattr(actor, '__call__'):
            # ✅ CORRECT: Call actor directly (works for DSPy modules and regular callables)
            if asyncio.iscoroutinefunction(actor):
                result = await actor(**resolved_kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: actor(**resolved_kwargs))
        else:
            raise ValueError(f"Actor {actor_config.name} has no callable method")
        
        # 🔑 Store output in shared context
        shared_context['actor_outputs'][actor_config.name] = result
        kwargs['_shared_context'] = shared_context
        
        # 🔥 A-TEAM: RETRY LOGIC FOR EMPTY EPISODES
        # If actor failed or returned None, attempt intelligent recovery and retry
        if (hasattr(result, 'success') and not result.success) or \
           (hasattr(result, 'output') and result.output is None):
            logger.warning(f"⚠️ [{actor_config.name}] Failed or returned None - attempting intelligent recovery")
            
            # Extract missing parameters from error or signature
            missing_params = []
            
            # Check if result has error message with missing params
            if hasattr(result, 'error') and result.error:
                import re
                matches = re.findall(r"missing.*?'(\w+)'|(\w+).*required", str(result.error), re.IGNORECASE)
                for match in matches:
                    param = match[0] or match[1]
                    if param and param not in missing_params:
                        missing_params.append(param)
            
            # Check signature for required params not in resolved_kwargs
            if actor_config.name in self.actor_signatures:
                sig = self.actor_signatures[actor_config.name]
                if isinstance(sig, dict):
                    for param_name, param_info in sig.items():
                        if param_info.get('required') and param_name not in resolved_kwargs:
                            if param_name not in missing_params:
                                missing_params.append(param_name)
            
            if missing_params:
                logger.info(f"🔄 [RETRY] Attempting recovery for {len(missing_params)} missing params: {missing_params}")
                
                # Use intelligent recovery with routing
                recovered = await self._intelligent_recovery_with_routing(
                    actor_config,
                    missing_params,
                    {'goal': kwargs.get('goal', kwargs.get('query', '')), 'kwargs': kwargs}
                )
                
                if recovered:
                    logger.info(f"✅ [RETRY] Recovered {len(recovered)} parameters")
                    
                    # 🔥 A-TEAM: BUILD RETRY CONTEXT + REASON for agent agency!
                    retry_context = f"""
🔄 RETRY ATTEMPT - You are being re-executed with additional context and data.

📋 REASON FOR RETRY:
- Previous execution failed or returned None
- Missing parameters were identified: {', '.join(missing_params)}

✅ WHAT WE DID TO FIX IT:
- Recovered {len(recovered)} parameter(s) from dependency graph and data sources
- Parameters now available: {', '.join(recovered.keys())}

📊 RECOVERED DATA:
"""
                    for param, value in recovered.items():
                        value_preview = str(value)[:100] if value else "None"
                        retry_context += f"- {param}: {type(value).__name__} = {value_preview}\n"
                    
                    retry_context += f"""
🎯 WHAT YOU SHOULD DO NOW:
- Use the newly provided parameters: {', '.join(recovered.keys())}
- Re-analyze the query with complete context
- Generate output based on ALL available data
- Previous attempt lacked: {', '.join(missing_params)}

💡 ADDITIONAL GUIDANCE:
- All required data is now available
- Focus on producing valid, complete output
- If you still encounter issues, clearly state what's missing
"""
                    
                    # Inject retry context into actor's context
                    if 'retry_context' not in resolved_kwargs:
                        resolved_kwargs['retry_context'] = retry_context
                    
                    # Update resolved_kwargs with recovered data
                    resolved_kwargs.update(recovered)
                    
                    # Log the retry context for debugging
                    logger.info(f"📝 [RETRY CONTEXT]:\n{retry_context}")
                    
                    # RETRY actor execution WITH CONTEXT
                    logger.info(f"🔄 [RETRY] Re-executing {actor_config.name} with context + recovered data...")
                    
                    try:
                        if asyncio.iscoroutinefunction(getattr(actor, 'run', None)):
                            result = await actor.run(**resolved_kwargs)
                        elif asyncio.iscoroutinefunction(getattr(actor, 'arun', None)):
                            result = await actor.arun(**resolved_kwargs)
                        elif hasattr(actor, '__call__'):
                            if asyncio.iscoroutinefunction(actor):
                                result = await actor(**resolved_kwargs)
                            else:
                                loop = asyncio.get_event_loop()
                                result = await loop.run_in_executor(None, lambda: actor(**resolved_kwargs))
                        
                        # Update shared context with retry result
                        shared_context['actor_outputs'][actor_config.name] = result
                        kwargs['_shared_context'] = shared_context
                        
                        if hasattr(result, 'success') and result.success:
                            logger.info(f"✅ [RETRY] Retry successful for {actor_config.name}")
                        else:
                            logger.warning(f"⚠️ [RETRY] Retry failed for {actor_config.name}")
                    
                    except Exception as e:
                        logger.error(f"❌ [RETRY] Retry execution failed: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                else:
                    logger.warning(f"⚠️ [RETRY] Could not recover any parameters")
            else:
                logger.warning(f"⚠️ [RETRY] No missing parameters identified for recovery")
        
        # ✅ A-TEAM: Register output in IOManager (typed outputs!)
        # 🔥 CRITICAL FIX: Extract the ACTUAL output (DSPy Prediction) from EpisodeResult wrapper!
        try:
            # 🔍 A-TEAM DEBUG: Log EVERYTHING about the result
            logger.info(f"🔍 [IOManager PREP] result type: {type(result)}")
            logger.info(f"🔍 [IOManager PREP] result has 'output': {hasattr(result, 'output')}")
            if hasattr(result, 'output'):
                logger.info(f"🔍 [IOManager PREP] result.output type: {type(result.output)}")
                logger.info(f"🔍 [IOManager PREP] result.output is None: {result.output is None}")
                if result.output and hasattr(result.output, '_store'):
                    logger.info(f"🔍 [IOManager PREP] result.output has _store with keys: {list(result.output._store.keys())}")
            if hasattr(result, 'success'):
                logger.info(f"🔍 [IOManager PREP] result.success: {result.success}")
            
            actual_output = result.output if hasattr(result, 'output') else result
            logger.info(f"🔍 [IOManager PREP] actual_output type: {type(actual_output)}")
            logger.info(f"🔍 [IOManager PREP] actual_output is None: {actual_output is None}")
            
            # 🆕 A-TEAM: Extract tagged attempts from trajectory (if available)
            tagged_attempts = []
            if hasattr(result, 'trajectory') and isinstance(result.trajectory, list):
                # Get the last trajectory entry (actor execution)
                for traj_entry in reversed(result.trajectory):
                    if isinstance(traj_entry, dict) and traj_entry.get('step') == 'actor':
                        tagged_attempts = traj_entry.get('tagged_attempts', [])
                        if tagged_attempts:
                            logger.info(f"🏷️  Retrieved {len(tagged_attempts)} tagged attempts from trajectory")
                            break
            
            self.io_manager.register_output(
                actor_name=actor_config.name,
                output=actual_output,  # ← FIXED! Pass the actual DSPy Prediction, not EpisodeResult!
                actor=actor,  # ✅ Pass actor for signature extraction!
                success=result.success if hasattr(result, 'success') else True,
                tagged_attempts=tagged_attempts  # 🆕 Pass tagged attempts!
            )
            logger.info(f"📦 Registered '{actor_config.name}' output in IOManager with {len(tagged_attempts)} tagged attempts")
        except Exception as e:
            logger.warning(f"⚠️  IOManager registration failed for '{actor_config.name}': {e}")
            import traceback as tb
            logger.warning(f"⚠️  Full traceback: {tb.format_exc()}")
        
        # =================================================================
        # 🎯 Q-LEARNING UPDATE: Natural Language Q-Table
        # =================================================================
        if hasattr(self, 'q_learner') and self.q_learner:
            try:
                # Generate natural language state description
                completed_actors = list(self.io_manager.outputs.keys()) if hasattr(self, 'io_manager') and self.io_manager else []
                state = {
                    "goal": kwargs.get('goal', '')[:100],
                    "actor": actor_config.name,
                    "completed": completed_actors,
                    "attempts": len(tagged_attempts)
                }
                
                action = {
                    "actor": actor_config.name,
                    "task": f"Execute {actor_config.name}"
                }
                
                next_state = state.copy()
                next_state["completed"] = completed_actors + [actor_config.name]
                
                # Compute reward (1.0 if Auditor passed, 0.0 otherwise)
                auditor_success = all(r.is_valid for r in result.auditor_results) if hasattr(result, 'auditor_results') and result.auditor_results else True
                reward = 1.0 if auditor_success else 0.0
                
                # Check if terminal (all actors done)
                is_terminal = len(next_state["completed"]) == len(self.actors)
                
                # Add experience (this updates Q-table AND stores in buffer)
                self.q_learner.add_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=is_terminal
                )
                
                logger.debug(f"🎯 Q-Learning updated: {actor_config.name} reward={reward:.2f}, terminal={is_terminal}")
                
                # Get learned context for injection into actor prompts
                learned_context = self.q_learner.get_learned_context(state, action)
                if learned_context:
                    logger.debug(f"📚 Learned context ({len(learned_context)} chars):")
                    logger.debug(learned_context[:200] + "..." if len(learned_context) > 200 else learned_context)
                
            except Exception as e:
                logger.warning(f"⚠️  Q-Learning update failed: {e}")
        
        # =================================================================
        # 📊 TD(λ) LEARNING UPDATE: Temporal Difference with Eligibility Traces
        # =================================================================
        # 🔥 FIX: TD(λ) updates happen at episode end via end_episode(), not per-actor.
        # The TDLambdaLearner doesn't have an update() method - it uses:
        # - record_access() during execution  
        # - end_episode() at terminal state
        # This per-actor update was incorrect - TD(λ) is called in run() at episode end.
        
        # 🔍 DEBUG: Log after storage
        logger.info(f"🔍 DEBUG: Stored output for '{actor_config.name}'")
        logger.info(f"🔍 DEBUG: Updated actor_outputs: {list(shared_context['actor_outputs'].keys())}")
        logger.info(f"🔍 DEBUG: Output type: {type(result)}")
        if hasattr(result, '__dict__'):
            logger.info(f"🔍 DEBUG: Output fields: {list(vars(result).keys())[:10]}")
        
        # 🔑 NEW: Register output in Data Registry (AGENTIC DISCOVERY)
        if self.data_registry and self.registration_orchestrator:
            # 🎯 A-TEAM FIX: Use await (not asyncio.run) since we're in async function
            logger.info("🎯 REGISTERING OUTPUT - START")
            logger.info(f"  Actor: {actor_config.name}")
            logger.info(f"  Output type: {type(result).__name__}")
            logger.info(f"  Registry: {self.data_registry is not None}")
            logger.info(f"  Orchestrator: {self.registration_orchestrator is not None}")
            
            registration_context = {
                'task': self.todo.root_task if hasattr(self.todo, 'root_task') else None,
                'goal': kwargs.get('goal', ''),
                'iteration': len(self.trajectory),
                'actor_config': actor_config
            }
            
            try:
                # ✅ A-TEAM: Use await (we're already in async function!)
                artifact_ids = await self.registration_orchestrator.register_output(
                    actor_name=actor_config.name,
                    output=result,
                    context=registration_context
                )
                logger.info("🎯 REGISTERING OUTPUT - COMPLETE")
                logger.info(f"  Artifacts registered: {len(artifact_ids)}")
                logger.info(f"  Artifact IDs: {artifact_ids}")
            except Exception as e:
                logger.error(f"❌ Agentic registration failed: {e}")
                logger.error(f"   Actor: {actor_config.name}")
                logger.error(f"   Output type: {type(result)}")
                raise RuntimeError(
                    f"Agentic registration failed for {actor_config.name}. "
                    f"This is a critical error - fix the registration logic or actor output!"
                ) from e
        
        return result
    
    def _detect_output_type(self, output: Any) -> str:
        """Auto-detect output type."""
        if hasattr(output, 'to_dict'):  # DataFrame
            return 'dataframe'
        elif isinstance(output, str):
            if len(output) > 100:
                if '<html' in output[:100].lower():
                    return 'html'
                elif '#' in output[:100]:
                    return 'markdown'
            return 'text'
        elif isinstance(output, bytes):
            return 'binary'
        elif isinstance(output, dict):
            return 'json'
        elif hasattr(output, 'output'):  # EpisodeResult
            return 'episode_result'
        elif hasattr(output, '__dict__'):
            return 'prediction'
        return 'unknown'
    
    def _extract_schema(self, output: Any) -> Dict[str, str]:
        """Extract schema from output."""
        schema = {}
        
        # Handle EpisodeResult
        if hasattr(output, 'output') and hasattr(output, 'success'):
            if output.output is not None:
                return self._extract_schema(output.output)
            return {}
        
        if hasattr(output, '__dict__'):
            for field_name, field_value in vars(output).items():
                if not field_name.startswith('_'):
                    schema[field_name] = type(field_value).__name__
        
        elif isinstance(output, dict):
            for key, value in output.items():
                schema[key] = type(value).__name__
        
        elif hasattr(output, 'columns'):  # DataFrame
            schema = {col: 'column' for col in output.columns}
        
        return schema
    
    def _generate_preview(self, output: Any) -> str:
        """Generate preview of output."""
        try:
            if isinstance(output, str):
                return output[:200]
            elif hasattr(output, '__str__'):
                return str(output)[:200]
            elif hasattr(output, 'head'):  # DataFrame
                return str(output.head(3))[:200]
            return f"<{type(output).__name__}>"
        except (AttributeError, TypeError, ValueError, Exception) as e:
            # Preview generation failed, return safe fallback
            logger.debug(f"Preview generation failed: {e}")
            return "<preview unavailable>"
    
    def _generate_tags(self, actor_name: str, output: Any, output_type: str) -> List[str]:
        """Generate semantic tags for output."""
        tags = [output_type, actor_name.lower()]
        
        # Handle EpisodeResult
        if hasattr(output, 'output') and hasattr(output, 'success'):
            if output.output is not None:
                return self._generate_tags(actor_name, output.output, output_type)
            return tags
        
        # Add field names as tags
        if hasattr(output, '__dict__'):
            field_names = [f for f in vars(output).keys() if not f.startswith('_')]
            tags.extend(field_names[:5])  # Top 5 fields
        
        elif isinstance(output, dict):
            tags.extend(list(output.keys())[:5])
        
        return tags
    
    def _register_output_in_registry(self, actor_name: str, output: Any):
        """Register output in Data Registry."""
        try:
            # Detect type
            output_type = self._detect_output_type(output)
            
            # Extract schema
            schema = self._extract_schema(output)
            
            # Generate tags
            tags = self._generate_tags(actor_name, output, output_type)
            
            # Generate preview
            preview = self._generate_preview(output)
            
            # Calculate size
            try:
                size = len(str(output))
            except (TypeError, AttributeError):
                # Size calculation failed, use 0
                size = 0
            
            # Create artifact
            artifact = DataArtifact(
                id=f"{actor_name}_{int(time.time() * 1000)}",
                name=actor_name,
                source_actor=actor_name,
                data=output,
                data_type=output_type,
                schema=schema,
                tags=tags,
                description=f"Output from {actor_name}",
                timestamp=time.time(),
                depends_on=[],
                size=size,
                preview=preview
            )
            
            # Register
            self.data_registry.register(artifact)
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to register output in registry: {e}")
    
    def _register_output_in_registry_fallback(self, actor_name: str, output: Any):
        """
        DEPRECATED: Fallback registration removed - use semantic registration only!
        
        This method is kept for backwards compatibility but will raise an error.
        All registration MUST go through the agentic RegistrationOrchestrator.
        """
        raise RuntimeError(
            f"❌ Fallback registration called for {actor_name}! "
            f"This is not allowed - use RegistrationOrchestrator for semantic registration."
        )
    
    def _should_inject_registry_tool(self, actor_name: str) -> bool:
        """Check if actor signature requests data_registry."""
        signature = self.actor_signatures.get(actor_name, {})
        return 'data_registry' in signature
    
    async def _run_auditor(
        self,
        actor_config: ActorConfig,
        result: Any,
        task: TodoItem
    ) -> Tuple[bool, float, str]:
        """
        Run Auditor for actor result.
        
        Incorporates:
        - Auditor prompts
        - Annotations
        - Learned patterns
        
        Returns:
            (passed, reward, feedback)
        """
        # For now, simple check - in full impl would use Auditor agents
        # TODO: Integrate full Auditor with prompts and annotations
        
        # Check if result indicates success
        if isinstance(result, dict):
            if result.get('success', True):
                return True, 1.0, "Auditor passed"
            else:
                return False, 0.0, result.get('error', 'Auditor failed')
        
        # Default: assume success
        return True, 0.8, "Result received"
    
    def get_actor_outputs(self) -> Dict[str, Any]:
        """
        Extract all actor outputs from trajectory.
        
        Returns:
            Dict mapping actor_name -> latest output
        """
        outputs = {}
        for step in self.trajectory:
            actor = step.get('actor')
            if actor and 'actor_output' in step:
                outputs[actor] = step['actor_output']
        return outputs
    
    def get_output_from_actor(self, actor_name: str, field: Optional[str] = None) -> Any:
        """
        Get specific output from an actor.
        
        Args:
            actor_name: Name of the actor
            field: Optional field to extract from output dict
        
        Returns:
            Actor output or specific field value
        """
        # Search from most recent to oldest
        for step in reversed(self.trajectory):
            if step.get('actor') == actor_name and 'actor_output' in step:
                output = step['actor_output']
                if field and isinstance(output, dict):
                    return output.get(field)
                return output
        return None
    
    def _fetch_all_metadata_directly(self) -> Dict[str, Any]:
        """
        🔥 A-TEAM CRITICAL FIX (User Insight): Fetch ALL metadata directly!
        
        USER QUESTION: "Why is react agent fetching when it's already in metadata manager?"
        ANSWER: YOU'RE RIGHT! We should just call ALL methods directly!
        
        NO ReAct agent overhead, NO guessing, NO missing data!
        
        This method calls ALL @jotty_method decorated methods from metadata_provider
        and returns a complete dictionary with ALL metadata.
        
        Returns:
            Dict with ALL metadata, using semantic keys (e.g., 'business_terms', 'table_names', etc.)
        """
        logger.info("🔍 Fetching ALL metadata directly (no ReAct agent, no guessing)...")
        metadata = {}
        start_time = time.time()
        
        # 🔥 Call ALL metadata methods directly!
        # 🔥 USER INSIGHT: No hardcoding! Store with original method names only.
        # AgenticResolver will do semantic matching with ONE LLM call.
        # This is generic and works for ANY naming convention!
        
        try:
            if hasattr(self.metadata_provider, 'get_all_business_contexts'):
                logger.debug("   📞 Calling get_all_business_contexts()...")
                result = self.metadata_provider.get_all_business_contexts()
                metadata['get_all_business_contexts'] = result
                logger.info(f"   ✅ get_all_business_contexts: {len(str(result))} chars")
        except Exception as e:
            logger.warning(f"   ⚠️  get_all_business_contexts() failed: {e}")
        
        try:
            if hasattr(self.metadata_provider, 'get_all_table_metadata'):
                logger.debug("   📞 Calling get_all_table_metadata()...")
                result = self.metadata_provider.get_all_table_metadata()
                metadata['get_all_table_metadata'] = result
                logger.info(f"   ✅ get_all_table_metadata: {len(str(result))} chars")
        except Exception as e:
            logger.warning(f"   ⚠️  get_all_table_metadata() failed: {e}")
        
        try:
            if hasattr(self.metadata_provider, 'get_all_filter_definitions'):
                logger.debug("   📞 Calling get_all_filter_definitions()...")
                result = self.metadata_provider.get_all_filter_definitions()
                metadata['get_all_filter_definitions'] = result
                logger.info(f"   ✅ get_all_filter_definitions: {len(str(result))} chars")
        except Exception as e:
            logger.warning(f"   ⚠️  get_all_filter_definitions() failed: {e}")
        
        try:
            if hasattr(self.metadata_provider, 'get_all_column_metadata'):
                logger.debug("   📞 Calling get_all_column_metadata()...")
                result = self.metadata_provider.get_all_column_metadata()
                metadata['get_all_column_metadata'] = result
                logger.info(f"   ✅ get_all_column_metadata: {len(str(result))} chars")
        except Exception as e:
            logger.warning(f"   ⚠️  get_all_column_metadata() failed: {e}")
        
        try:
            if hasattr(self.metadata_provider, 'get_all_term_definitions'):
                logger.debug("   📞 Calling get_all_term_definitions()...")
                result = self.metadata_provider.get_all_term_definitions()
                metadata['get_all_term_definitions'] = result
                logger.info(f"   ✅ get_all_term_definitions: {len(str(result))} chars")
        except Exception as e:
            logger.warning(f"   ⚠️  get_all_term_definitions() failed: {e}")
        
        try:
            if hasattr(self.metadata_provider, 'get_all_validations'):
                logger.debug("   📞 Calling get_all_validations()...")
                result = self.metadata_provider.get_all_validations()
                metadata['get_all_validations'] = result
                logger.info(f"   ✅ get_all_validations: {len(str(result))} chars")
        except Exception as e:
            logger.warning(f"   ⚠️  get_all_validations() failed: {e}")
        
        # 🔥 Add any other discovered methods from metadata_tool_registry
        if hasattr(self, 'metadata_tool_registry'):
            for tool_name in self.metadata_tool_registry.tools.keys():
                if tool_name not in ['get_all_business_contexts', 'get_all_table_metadata', 
                                     'get_all_filter_definitions', 'get_all_column_metadata',
                                     'get_all_term_definitions', 'get_all_validations']:
                    try:
                        if hasattr(self.metadata_provider, tool_name):
                            # ✅ FIX: Skip methods requiring positional args (prefetch phase has no args).
                            try:
                                sig = inspect.signature(getattr(self.metadata_provider, tool_name))
                                required_positional = [
                                    p.name for p in sig.parameters.values()
                                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
                                ]
                                if required_positional:
                                    logger.debug(f"   ⏭️  Skipping {tool_name}() prefetch (requires args: {required_positional})")
                                    continue
                            except Exception as e:
                                logger.debug(f"   ⏭️  Skipping {tool_name}() prefetch (signature inspect failed: {e})")
                                continue

                            logger.debug(f"   📞 Calling {tool_name}()...")
                            result = getattr(self.metadata_provider, tool_name)()
                            metadata[tool_name] = result
                            logger.info(f"   ✅ {tool_name}: {len(str(result))} chars")
                    except Exception as e:
                        logger.warning(f"   ⚠️  {tool_name}() failed: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Fetched {len(metadata)} metadata items in {elapsed:.2f}s (direct calls, no LLM!)")
        
        return metadata
    
    def _enrich_business_terms_with_filters(self, fetched_data: Dict[str, Any]):
        """
        🔥 A-TEAM CRITICAL FIX: Enrich business_terms with parsed filter conditions!
        
        This bridges metadata (SharedContext) with parameter flow (actor inputs).
        
        GENERIC - No hardcoding! Works for ANY domain where:
        - There are "business_terms" (categories/concepts)
        - There are "filter_conditions_parsed" (structured constraints)
        
        Args:
            fetched_data: Metadata returned by _fetch_all_metadata_directly()
        """
        # Check if we have both business terms and parsed filters
        business_terms = fetched_data.get('business_terms', {})
        parsed_filters = fetched_data.get('filter_conditions_parsed', {})
        
        if not business_terms or not parsed_filters:
            logger.debug("No enrichment needed (missing business_terms or parsed_filters)")
            return
        
        logger.info(f"🔄 Enriching {len(business_terms)} business terms with {len(parsed_filters)} filter specs")
        
        # Enrich each business term with matching filter conditions
        enriched_count = 0
        for term_name, term_data in business_terms.items():
            # Try to find matching filter (generic matching!)
            # Try exact match first
            if term_name in parsed_filters:
                filter_spec = parsed_filters[term_name]
                self._merge_filter_into_term(term_data, filter_spec, term_name)
                enriched_count += 1
                continue
            
            # Try case-insensitive match
            term_lower = term_name.lower()
            for filter_key, filter_spec in parsed_filters.items():
                if filter_key.lower() == term_lower:
                    self._merge_filter_into_term(term_data, filter_spec, term_name)
                    enriched_count += 1
                    break
            
            # Try substring match (e.g., "P2P" matches "P2P transactions")
            if not isinstance(term_data, dict) or 'fields' not in term_data:
                for filter_key, filter_spec in parsed_filters.items():
                    if term_lower in filter_key.lower() or filter_key.lower() in term_lower:
                        self._merge_filter_into_term(term_data, filter_spec, term_name)
                        enriched_count += 1
                        break
        
        logger.info(f"✅ Enriched {enriched_count}/{len(business_terms)} business terms with filter conditions")
        
        # Update in SharedContext
        fetched_data['business_terms'] = business_terms
        self.shared_context.set('metadata', fetched_data)
    
    def _merge_filter_into_term(self, term_data: Any, filter_spec: Dict[str, Any], term_name: str):
        """
        Merge filter specification into business term data.
        
        GENERIC - works with any field names!
        """
        if not isinstance(term_data, dict):
            # Can't merge into non-dict, convert to dict first
            term_data_new = {'original': term_data}
            term_data = term_data_new
        
        # Merge filter fields
        for key, value in filter_spec.items():
            if key not in term_data:
                term_data[key] = value
                logger.debug(f"  ✅ {term_name}: Added '{key}' from filter spec")
            else:
                logger.debug(f"  ⚠️  {term_name}: '{key}' already exists, skipping")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_conductor(
    actors: List[Tuple[str, Any, List[str], List[str]]],
    config_path: str = "user_configs/config/jotty_config.yml",
    annotations_path: str = "user_configs/human_input_kb/annotations/annotations.json",
    global_architect: str = None,
    global_auditor: str = None
) -> 'MultiAgentsOrchestrator':
    """
    Factory to create MultiAgentsOrchestrator from simple configuration.

    Args:
        actors: List of (name, actor_instance, architect_prompts, auditor_prompts)
        config_path: Path to JOTTY config YAML
        annotations_path: Path to annotations JSON
        global_architect: Optional global Architect prompt path
        global_auditor: Optional global Auditor prompt path

    Returns:
        Configured MultiAgentsOrchestrator instance
    """
    import yaml
    
    # Load config
    config = JottyConfig()
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            
            # ✅ COMPLETE CONFIG MAPPING (NO HARDCODING)
            
            # Persistence config
            persistence_cfg = cfg.get('persistence', {})
            config.save_interval = persistence_cfg.get('save_interval', 1)
            config.auto_save = persistence_cfg.get('auto_save', True)
            config.auto_load = persistence_cfg.get('auto_load', True)
            
            # Context config
            context_cfg = cfg.get('context', {})
            config.max_context_tokens = context_cfg.get('max_tokens', 28000)
            
            # Learning config
            learning_cfg = cfg.get('learning', {})
            config.gamma = learning_cfg.get('gamma', 0.95)
            
            # (More mappings can be added as needed)
    
    # Create actor configs
    actor_configs = []
    for name, actor, architects, auditors in actors:
        actor_configs.append(ActorConfig(
            name=name,
            actor=actor,
            architect_prompts=architects,
            auditor_prompts=auditors
        ))
    
    return MultiAgentsOrchestrator(
        actors=actor_configs,
        config=config,
        global_architect=global_architect,
        global_auditor=global_auditor,
        annotations_path=annotations_path
    )

# =============================================================================
# BACKWARD COMPATIBILITY - DEPRECATED ALIASES
# =============================================================================
# REFACTORING PHASE 1.3: Deprecation aliases for renamed classes
# These will be removed in a future version.

import warnings

class Conductor(MultiAgentsOrchestrator):
    """
    Deprecated: Use MultiAgentsOrchestrator instead.

    This is a backward compatibility alias that will be removed in a future version.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Conductor is deprecated and will be removed in a future version. "
            "Use MultiAgentsOrchestrator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
