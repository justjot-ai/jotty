"""
SingleAgentOrchestrator v6.0 - Single-Agent Episode Orchestrator
===================================================================

Complete JOTTY system with all A-Team enhancements:
- Dr. Manning: Adaptive learning, intermediate rewards, TD(λ) correct
- Dr. Chen: Inter-agent communication, multi-round validation, reasoning credit
- Dr. Agarwal: LLM-based RAG, dynamic budget, size-aware storage
- Aristotle: Causal learning, goal hierarchy, conditional wisdom
- Shannon: Deduplication, compression, information-theoretic selection
- Alex: JSON/SQLite persistence, health monitoring

Main orchestration flow:
1. Episode initialization
2. Architect (multi-round if needed) - plans execution
3. Actor execution with injections
4. Auditor (multi-round if needed) - validates output
5. Learning update (TD(λ), credit assignment, offline)
6. State persistence
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import dspy

logger = logging.getLogger(__name__)

from ..foundation.data_structures import (
    JottyConfig, EpisodeResult, TaggedOutput, ValidationResult,
    StoredEpisode, OutputTag, SharedScratchpad, MemoryLevel,
    GoalHierarchy, AlertType
)
from ..agents.inspector import InspectorAgent, MultiRoundValidator
from ..memory.cortex import HierarchicalMemory
from ..metadata.tool_interceptor import ToolInterceptor
from .retry_mechanism import (
    RetryMechanism, RetryConfig, RetryResult,
    build_architect_feedback, build_auditor_feedback,
    update_confidence, create_retry_trajectory_entry
)
from ..learning.learning import (
    TDLambdaLearner, AdaptiveLearningRate, IntermediateRewardCalculator,
    ReasoningCreditAssigner, AdaptiveExploration, LearningHealthMonitor,
    DynamicBudgetManager
)
from ..learning.offline_learning import OfflineLearner


# =============================================================================
# PERSISTENCE MANAGER (Alex Enhancement)
# =============================================================================

class PersistenceManager:
    """
    JSON/SQLite-based persistence (no pickle!).
    
    Features:
    - Human-readable JSON for config/state
    - SQLite for large memory storage
    - Automatic backups
    - Migration support
    """
    
    def __init__(self, base_path: str, config: JottyConfig):
        self.base_path = Path(base_path).expanduser()
        self.config = config
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Paths
        self.state_file = self.base_path / "state.json"
        self.memories_dir = self.base_path / "memories"
        self.backups_dir = self.base_path / "backups"
        self.offline_file = self.base_path / "offline.json"
        
        self.memories_dir.mkdir(exist_ok=True)
        self.backups_dir.mkdir(exist_ok=True)
    
    def save_state(self, state: Dict[str, Any]):
        """Save global state to JSON."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load global state from JSON."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_memory(self, agent_name: str, memory: HierarchicalMemory):
        """Save agent memory to JSON."""
        memory_file = self.memories_dir / f"{agent_name}.json"
        data = memory.to_dict()
        with open(memory_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_memory(self, agent_name: str) -> Optional[Dict]:
        """Load agent memory from JSON."""
        memory_file = self.memories_dir / f"{agent_name}.json"
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_offline(self, offline_learner: OfflineLearner):
        """Save offline learner state."""
        data = offline_learner.to_dict()
        with open(self.offline_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_offline(self) -> Optional[Dict]:
        """Load offline learner state."""
        if self.offline_file.exists():
            with open(self.offline_file, 'r') as f:
                return json.load(f)
        return None
    
    def create_backup(self, episode: int):
        """Create backup of current state."""
        if not self.config.enable_backups:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backups_dir / f"backup_{episode}_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        
        # Copy files
        import shutil
        if self.state_file.exists():
            shutil.copy(self.state_file, backup_dir / "state.json")
        
        for mem_file in self.memories_dir.glob("*.json"):
            shutil.copy(mem_file, backup_dir / mem_file.name)
        
        # Cleanup old backups
        backups = sorted(self.backups_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        while len(backups) > self.config.max_backups:
            shutil.rmtree(backups[0])
            backups.pop(0)


# =============================================================================
# MAIN JOTTY CORE CLASS
# =============================================================================

class SingleAgentOrchestrator:
    """
    SingleAgentOrchestrator v6.0 - Single-Agent Episode Manager

    Manages complete validation workflow for one agent:
    - Architect (planning) → Agent Execution → Auditor (validation)
    - Learning loop (TD-lambda, Q-learning, credit assignment)
    - Retry mechanisms with confidence-based override

    Complete implementation with all A-Team enhancements.

    Usage:
        # Configure DSPy
        dspy.configure(lm=dspy.LM("openai/gpt-4o", api_key="..."))

        # Create config
        config = JottyConfig(
            base_path="~/.jotty/my_project",
            enable_causal_learning=True,
            enable_multi_round=True,
            ...
        )

        # Create agent
        agent = dspy.ReAct(MySignature, tools=[...])

        # Create SingleAgentOrchestrator
        orchestrator = SingleAgentOrchestrator(
            agent=agent,
            architect_prompts=["prompts/planning.md", "prompts/context.md"],
            auditor_prompts=["prompts/validator.md"],
            architect_tools=[tool1, tool2],
            auditor_tools=[tool3],
            config=config
        )

        # Run
        result = await orchestrator.arun(question="...", context="...")
    """
    
    def __init__(self,
                 agent: dspy.Module = None,
                 architect_prompts: List[str] = None,
                 auditor_prompts: List[str] = None,
                 architect_tools: List[Any] = None,
                 auditor_tools: List[Any] = None,
                 config: JottyConfig = None,
                 agent_config: 'ActorConfig' = None,
                 shared_context: Optional[Dict[str, Any]] = None,

                 # 🆕 Phase 8: Gold Standard Learning (optional)
                 enable_gold_standard_learning: bool = False,
                 gold_standards: Optional[List[Dict[str, Any]]] = None,
                 validation_cases: Optional[List[Dict[str, Any]]] = None,
                 domain: Optional[str] = None,
                 domain_validator: Optional[Callable[[Any], bool]] = None,
                 max_training_iterations: int = 5,
                 min_validation_score: float = 1.0,

                 # Backward compatibility parameters
                 actor: dspy.Module = None,
                 actor_config: 'ActorConfig' = None):
        """
        Initialize SingleAgentOrchestrator.

        Parameters:
            agent: The DSPy agent to validate (new parameter name)
            architect_prompts: Paths to Architect agent markdown prompts (planning)
            auditor_prompts: Paths to Auditor agent markdown prompts (validation)
            architect_tools: Tools for Architect agents
            auditor_tools: Tools for Auditor agents
            config: JOTTY configuration (defaults if None)
            agent_config: ActorConfig with enable_architect/enable_auditor flags (optional)
            shared_context: SharedContext for accessing metadata (optional)

            # Phase 8: Gold Standard Learning
            enable_gold_standard_learning: Enable expert training with gold standards
            gold_standards: List of {input, expected_output} training examples
            validation_cases: List of validation test cases
            domain: Domain name for the expert (e.g., "mermaid", "sql")
            domain_validator: Custom validation function (input) -> bool
            max_training_iterations: Max optimization iterations
            min_validation_score: Minimum score to pass validation

            actor: DEPRECATED - Use 'agent' instead
            actor_config: DEPRECATED - Use 'agent_config' instead
        """
        # 🔄 Phase 7: Backward compatibility for 'actor' parameter
        if agent is None and actor is not None:
            import warnings
            warnings.warn(
                "'actor' parameter is deprecated. Use 'agent' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            agent = actor

        # 🔄 Phase 7: Backward compatibility for 'actor_config' parameter
        if agent_config is None and actor_config is not None:
            import warnings
            warnings.warn(
                "'actor_config' parameter is deprecated. Use 'agent_config' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            agent_config = actor_config

        self.agent = agent
        self.config = config or JottyConfig()
        self.agent_config = agent_config  # 🔑 A-TEAM: Store agent_config for validation control
        self.shared_context = shared_context  # 🔥 A-TEAM: Store shared_context for metadata access
        
        # 🆕 A-TEAM: Initialize trajectory parser (GENERIC - no domain logic)
        from ..utils.trajectory_parser import TrajectoryParser
        self.trajectory_parser = TrajectoryParser(lm=None)  # LLM tagging TODO
        logger.info("🏷️  TrajectoryParser initialized in SingleAgentOrchestrator")
        
        # 🔧 A-TEAM: Tool interceptor for tracking all tool calls
        agent_name = agent_config.name if agent_config else self.agent.__class__.__name__
        self.tool_interceptor = ToolInterceptor(agent_name)
        logger.info(f"🔧 [JOTTY] Tool interceptor initialized for '{agent_name}'")
        
        # Shared scratchpad for agent communication
        self.scratchpad = SharedScratchpad()
        
        # Create Architect agents (planning & context assessment)
        self.architect_agents = [
            InspectorAgent(
                md_path=Path(prompt),
                is_architect=True,
                tools=architect_tools,
                config=self.config,
                scratchpad=self.scratchpad
            )
            for prompt in architect_prompts
        ]
        
        # Create Auditor agents (output validation)
        # ✅ A-TEAM DECISION: Two-level validation architecture
        # - Actor-level (here): Role-specific quality checks
        # - Swarm-level (conductor.py): Orchestration & coordination checks
        # Both serve different purposes and should be kept!
        self.auditor_agents = [
            InspectorAgent(
                md_path=Path(prompt),
                is_architect=False,
                tools=auditor_tools,
                config=self.config,
                scratchpad=self.scratchpad
            )
            for prompt in auditor_prompts
        ]
        
        # Multi-round validators
        self.architect_validator = MultiRoundValidator(self.architect_agents, self.config)
        self.auditor_validator = MultiRoundValidator(self.auditor_agents, self.config)
        
        # Learning components
        self.adaptive_lr = AdaptiveLearningRate(self.config)
        self.td_learner = TDLambdaLearner(self.config, self.adaptive_lr)
        self.intermediate_rewards = IntermediateRewardCalculator(self.config)
        self.credit_assigner = ReasoningCreditAssigner(self.config)
        self.exploration = AdaptiveExploration(self.config)
        self.health_monitor = LearningHealthMonitor(self.config)
        self.budget_manager = DynamicBudgetManager(self.config)
        
        # Offline learning
        self.offline_learner = OfflineLearner(self.config)
        self.offline_learner.td_learner = self.td_learner

        # 🆕 Phase 8: Gold Standard Learning (optional)
        self.enable_gold_standard_learning = enable_gold_standard_learning
        self.gold_standards = gold_standards or []
        self.validation_cases = validation_cases or []
        self.domain = domain
        self.domain_validator = domain_validator
        self.max_training_iterations = max_training_iterations
        self.min_validation_score = min_validation_score
        self.optimization_pipeline = None

        if enable_gold_standard_learning:
            # 🎓 Phase 8: Gold Standard Learning Configuration
            # For now, we store the configuration. Training will be implemented
            # via a separate .train() method when OptimizationPipeline is refactored
            # for single-agent use cases (currently designed for multi-agent pipelines)

            logger.info(f"🎓 [PHASE 8] Gold standard learning enabled for domain: {domain}")
            logger.info(f"🎓 [PHASE 8] Loaded {len(self.gold_standards)} gold standard examples")
            if domain_validator:
                logger.info(f"🎓 [PHASE 8] Domain validator configured")

            # Note: OptimizationPipeline integration will be added in future when
            # it supports single-agent gold standard training patterns

        # Goal hierarchy (shared)
        self.goal_hierarchy = GoalHierarchy()
        
        # Persistence
        self.persistence = PersistenceManager(self.config.base_path, self.config)
        
        # State
        self.episode_count = 0
        self.trajectory: List[Dict] = []
        self.context: Dict[str, Any] = {}
        self.learned_instructions: Dict[str, List[str]] = {
            'architect': [],
            'auditor': [],
            'actor': []
        }
        
        # Load state if exists
        if self.config.auto_load:
            self._load_state()
    
    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================
    
    async def arun(self, **kwargs) -> EpisodeResult:
        """
        Run a complete episode.
        
        Parameters:
            **kwargs: Arguments passed to actor
        
        Returns:
            EpisodeResult with output and metadata
        """
        start_time = time.time()
        self.episode_count += 1
        
        # Reset episode state
        self.trajectory = []
        self.scratchpad.clear()
        self.intermediate_rewards.reset()
        
        # Extract goal
        goal = self._extract_goal(kwargs)
        
        # Start TD learning
        self.td_learner.start_episode(goal)
        self.exploration.record_goal_visit(goal)
        
        # Update goal hierarchy
        if self.config.enable_goal_hierarchy:
            domain = kwargs.get('domain', 'general')  # 🔧 A-TEAM: Generic default, not 'sql'!
            entities = kwargs.get('entities', [])
            self.goal_hierarchy.add_goal(goal, domain, 'query', entities)
        
        # Inject learned instructions
        kwargs = self._inject_learned_instructions(kwargs)
        
        # Track accessed memories for offline learning
        memories_accessed: Dict[str, List[str]] = {}
        
        # =====================================================================
        # ARCHITECT PHASE
        # =====================================================================
        
        # 🔑 A-TEAM FIX: Skip Architect if disabled for this actor
        if self.agent_config and self.agent_config.enable_architect:
            # 🔍 A-TEAM DEBUG: Log what Architect receives (FULL VALUES - NO SLICING!)
            logger.info(f"🔍 [ARCHITECT INPUT] kwargs keys: {list(kwargs.keys())}")
            for key in ['query', 'business_terms', 'table_names', 'tables', 'resolved_terms']:
                if key in kwargs:
                    value_preview = str(kwargs[key]) if kwargs[key] else "None"  # NO SLICING!
                    logger.info(f"🔍 [ARCHITECT INPUT] {key}: {value_preview}")
                else:
                    logger.info(f"🔍 [ARCHITECT INPUT] {key}: NOT IN KWARGS")
            
            # ✅ FIX: Pass kwargs directly so Architect can see actual query, conversation_history, etc.
            # 🔥 A-TEAM CRITICAL FIX: MINIMAL CONTEXT - Force Tool Usage!
            # Architect should CALL TOOLS to get data, not receive it pre-loaded.
            # 
            # What we pass:
            # ✅ actor_kwargs_summary: What actor is ABOUT to receive (for validation)
            # ✅ goal: What actor is supposed to do
            # ❌ NO schemas, business terms, or data - Val agent must call tools!
            #
            # Mental Model:
            # "Actor is about to be called with these parameters. Are they sufficient?"
            # "Call your tools to verify the parameters are valid."
            
            # Build minimal summary of actor kwargs (parameter names only, not full data)
            actor_param_summary = {}
            for key, value in kwargs.items():
                # For Architect, just show WHAT parameters exist, not their full content
                if value is None:
                    actor_param_summary[key] = "None"
                elif isinstance(value, (list, dict)):
                    actor_param_summary[key] = f"{type(value).__name__} with {len(value)} items"
                elif isinstance(value, str) and len(value) > 100:
                    actor_param_summary[key] = f"str (length: {len(value)})"
                else:
                    actor_param_summary[key] = str(value)
            
            architect_inputs = {
                'goal': goal,
                'current_thought': "Starting validation",
                'proposed_action': 'execute',
                'actor_parameters': json.dumps(actor_param_summary, indent=2),  # Summary, not full data
                'query': kwargs.get('query', ''),  # Keep query for context
                'session_id': kwargs.get('session_id', ''),  # Keep session context
                # ❌ NO: **kwargs - don't pass all data!
                # ❌ NO: schemas, business_terms - agent must call tools!
            }
            
            logger.info(f"✅ [ARCHITECT CONTEXT] Minimal context - {len(actor_param_summary)} parameter summaries (agent must call tools for data)")
            
            architect_results, _ = await self.architect_validator.validate(
                goal=goal,
                inputs=architect_inputs,
                trajectory=[],  # 🔥 FIX: Don't pass past failures to Architect - only validate current inputs!
                is_architect=True
            )
            
            # 🔥 A-TEAM PHASE 2 FIX: Architect is exploration-only, ALWAYS proceed!
            proceed = True  # Architect doesn't block!
            
            # 🔍 A-TEAM DEBUG: Log Architect exploration
            logger.info(f"🔍 [ARCHITECT PLANNING] Architect planning completed for '{self.agent_config.name if self.agent_config else 'UNKNOWN'}'")
            logger.info(f"🔍 [ARCHITECT PLANNING] Always proceed: {proceed} (Architect is advisor, not gatekeeper)")
            logger.info(f"🔍 [ARCHITECT PLANNING] num architect_results: {len(architect_results)}")
            for i, result in enumerate(architect_results):
                logger.info(f"🔍 [ARCHITECT PLANNING] Result {i+1}: confidence={result.confidence}")
                if hasattr(result, 'exploration_summary') and result.exploration_summary:
                    logger.info(f"🔍 [ARCHITECT PLANNING] Summary: {result.exploration_summary[:200]}...")
                if hasattr(result, 'recommendations') and result.recommendations:
                    logger.info(f"🔍 [ARCHITECT PLANNING] Recommendations: {result.recommendations[:200]}...")
            
            # Record Architect exploration in trajectory
            self.trajectory.append({
                'step': 'architect_exploration',
                'always_proceed': True,  # Architect doesn't block!
                'agents': [r.agent_name for r in architect_results],
                'confidences': [r.confidence for r in architect_results],
                'exploration_summaries': [getattr(r, 'exploration_summary', '') for r in architect_results]
            })
            
            # Intermediate reward for Architect exploration
            avg_confidence = sum(r.confidence for r in architect_results) / len(architect_results)
            self.intermediate_rewards.reward_architect_proceed(avg_confidence)
            
            # 🔥 A-TEAM: Collect exploration insights to brief actor
            exploration_briefings = []
            recommendations_list = []
            data_quality_notes_list = []
            for result in architect_results:
                if hasattr(result, 'exploration_summary') and result.exploration_summary:
                    exploration_briefings.append(result.exploration_summary)
                if hasattr(result, 'recommendations') and result.recommendations:
                    recommendations_list.append(result.recommendations)
                if hasattr(result, 'data_quality_notes') and result.data_quality_notes:
                    data_quality_notes_list.append(result.data_quality_notes)
            
            # Build briefing for actor
            injected_context = []
            injected_instructions = []
            if exploration_briefings:
                briefing = "📊 Architect Exploration Findings:\n" + "\n".join(f"- {b}" for b in exploration_briefings)
                injected_context.append(briefing)
            if recommendations_list:
                recommendations = "💡 Architect Recommendations:\n" + "\n".join(f"- {r}" for r in recommendations_list)
                injected_instructions.append(recommendations)
            if data_quality_notes_list:
                quality_notes = "⚠️  Data Quality Notes:\n" + "\n".join(f"- {n}" for n in data_quality_notes_list)
                injected_context.append(quality_notes)
        else:
            # Architect disabled or no actor_config - proceed directly
            if self.agent_config:
                logger.info(f"⏩ Architect skipped for actor '{self.agent_config.name}'")
            architect_results = []
            proceed = True
            injected_context = []
            injected_instructions = []
        
        # =====================================================================
        # RETRY MECHANISM WITH MOVING AVERAGE CONFIDENCE
        # =====================================================================
        
        # 🔄 A-TEAM: Implement retry mechanism as per user's plan
        # Val blocks → Actor retries with feedback → Actor's confidence increases via moving average
        retry_count = 0
        max_retries = getattr(self.agent_config, 'max_retries', 3) if self.agent_config else 3
        
        # Initialize actor confidence (tracked across retries)
        if not hasattr(self, '_actor_confidence'):
            self._actor_confidence = 0.7  # Initial confidence
        
        # 🧮 A-TEAM: CONFIDENCE GAP MATHEMATICS
        # Calculate Architect's confidence in its blocking decision
        architect_confidence = 0.5  # Default: unsure
        if architect_results:
            architect_confidence = sum(
                getattr(r, 'confidence', 0.5) for r in architect_results
            ) / len(architect_results)
        
        # Calculate override score using Formula v2 (Aggressive)
        # Override Score = 0.6 * (Actor_Confidence - Architect_Confidence) + 0.4 * (1 - Architect_Confidence)
        confidence_gap = self._actor_confidence - architect_confidence
        override_score = (
            0.6 * confidence_gap +               # Gap weight
            0.4 * (1 - architect_confidence)        # Uncertainty weight
        )
        
        OVERRIDE_THRESHOLD = 0.3  # From A-Team consensus
        
        logger.info(f"📊 [OVERRIDE CALC] Architect confidence: {architect_confidence:.3f}")
        logger.info(f"📊 [OVERRIDE CALC] Actor confidence: {self._actor_confidence:.3f}")
        logger.info(f"📊 [OVERRIDE CALC] Confidence gap: {confidence_gap:.3f}")
        logger.info(f"📊 [OVERRIDE CALC] Override score: {override_score:.3f}")
        logger.info(f"📊 [OVERRIDE CALC] Override threshold: {OVERRIDE_THRESHOLD}")
        
        # 🔥 A-TEAM CRITICAL FIX: Use confidence gap to decide immediate override!
        if not proceed and override_score > OVERRIDE_THRESHOLD:
            logger.info(f"✅ [OVERRIDE] Override score ({override_score:.3f}) > threshold ({OVERRIDE_THRESHOLD})")
            logger.info(f"✅ [OVERRIDE] Architect has LOW confidence ({architect_confidence:.3f}), actor has confidence ({self._actor_confidence:.3f})")
            logger.info(f"✅ [OVERRIDE] Proceeding immediately - Architect is too unsure to block!")
            proceed = True  # Immediate override!
        
        # 🔥 RETRY LOOP: Send feedback to actor until Val passes or max retries
        # ✅ REFACTORED: Using RetryMechanism to eliminate duplication
        if not proceed:
            retry_mechanism = RetryMechanism(
                config=RetryConfig(
                    max_retries=max_retries,
                    confidence_divisor=4.0,
                    feedback_field_name='_architect_feedback',
                    phase_name='ARCHITECT'
                ),
                initial_confidence=self._actor_confidence
            )

            retry_result = await retry_mechanism.retry_until_valid(
                should_proceed_fn=lambda results: all(
                    getattr(r, 'should_proceed', False) for r in results
                ),
                validate_fn=self.architect_validator.validate,
                build_feedback_fn=build_architect_feedback,
                kwargs=kwargs,
                validate_inputs=architect_inputs,
                goal=goal,
                is_architect=True
            )

            # Update state from retry result
            proceed = retry_result.success
            retry_count = retry_result.retry_count
            self._actor_confidence = retry_result.final_confidence

            # Add retry trajectory entries to main trajectory
            self.trajectory.extend(retry_result.trajectory_entries)
        
        # 🔥 A-TEAM CRITICAL FIX: Actor should ALWAYS execute, even if Architect blocks!
        # Architect provides guidance, but doesn't completely block execution
        # This ensures graceful degradation instead of complete failure
        if not proceed and retry_count >= max_retries:
            logger.warning(f"⚠️ [ARCHITECT RETRY] Max retries ({max_retries}) exhausted. Architect still blocking.")
            logger.warning(f"⚠️ [ARCHITECT OVERRIDE] Actor will execute anyway - Architect concerns noted in metadata")
            # Mark that we're proceeding despite Architect concerns
            proceed = True  # Override blocking for resilience
        
        # =====================================================================
        # ACTOR PHASE
        # =====================================================================
        
        actor_output = None
        actor_error = None
        architect_override = (retry_count >= max_retries)  # Track if we overrode Architect
        
        # 🎯 USER DESIGN: Actor ALWAYS executes (with Architect guidance)
        # Architect is advisory, not blocking!
        
        # ✅ FIX: Store injections in context for logging ONLY - DON'T add to kwargs!
        # DSPy modules don't accept these parameters
        if injected_context:
            self.context['_injected_context'] = "\n".join(injected_context)
        if injected_instructions:
            self.context['_injected_instructions'] = "\n".join(injected_instructions)
        
        try:
            actor_output = await self._run_actor_with_timeout(kwargs)
            
            # 🔍 A-TEAM DEBUG: Check actor output (ALWAYS log, not just debug mode)
            logger.info(f"[🔍 ACTOR OUTPUT] Actor '{self.agent_config.name if self.agent_config else 'UNKNOWN'}' output type: {type(actor_output)}")
            logger.info(f"[🔍 ACTOR OUTPUT] Actor output is None: {actor_output is None}")
            if hasattr(actor_output, '__dict__'):
                logger.info(f"[🔍 ACTOR OUTPUT] Actor output attrs: {list(vars(actor_output).keys())}")
            if hasattr(actor_output, '_store'):
                logger.info(f"[🔍 ACTOR OUTPUT] Actor output has _store (DSPy Prediction)")
                if isinstance(actor_output._store, dict):
                    logger.info(f"[🔍 ACTOR OUTPUT] _store keys: {list(actor_output._store.keys())}")
                    
                    # 🔬 DIAGNOSTIC: Log RAW _store values to check for parsing errors!
                    for key in list(actor_output._store.keys())[:10]:  # First 10 keys
                        value = actor_output._store.get(key)
                        if isinstance(value, (dict, list)):
                            logger.info(f"[🔬 DIAGNOSTIC] _store['{key}'] = {value}")  # Show full structure
                        else:
                            value_str = str(value)[:200] if value is not None else "None"
                            logger.info(f"[🔬 DIAGNOSTIC] _store['{key}'] (first 200 chars) = {value_str}")
            
            # 🆕 A-TEAM CRITICAL: Parse trajectory and tag attempts (GENERIC!)
            # This is the ONLY place where tagging happens - centralized in JottyCore
            tagged_attempts = []
            if hasattr(actor_output, '_store'):
                try:
                    tagged_attempts = self.trajectory_parser.parse_trajectory(
                        result=actor_output,
                        tool_name_filter=None,  # Parse all tool calls
                        expected_outcome=kwargs.get('query') or kwargs.get('goal') or None
                    )
                    logger.info(f"🏷️  Tagged {len(tagged_attempts)} attempts:")
                    for attempt in tagged_attempts:
                        logger.info(f"   #{attempt.attempt_number}: tag='{attempt.tag}', tool='{attempt.tool_name}'")
                except Exception as e:
                    logger.warning(f"⚠️  Trajectory parsing failed: {e}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # 🔧 CRITICAL FIX: Store FULL output, not truncated string
            # Extract ReAct trajectory if available (for DSPy ReAct agents)
            react_trajectory = {}
            if hasattr(actor_output, 'trajectory') and isinstance(actor_output.trajectory, dict):
                react_trajectory = actor_output.trajectory
            elif hasattr(actor_output, '__dict__') and 'trajectory' in vars(actor_output):
                react_trajectory = vars(actor_output).get('trajectory', {})
            
            self.trajectory.append({
                'step': 'actor',
                'output': actor_output,  # ✅ Full output, not truncated
                'react_trajectory': react_trajectory,  # ✅ ReAct trajectory if available
                'tagged_attempts': tagged_attempts,  # 🆕 Tagged attempts (answer/error/exploratory)
                'error': None,
                'architect_override': architect_override  # Track if we overrode Architect
            })
            
            # Intermediate reward for successful actor
            self.intermediate_rewards.reward_tool_success('actor', True)
            
        except Exception as e:
            actor_error = str(e)
            self.trajectory.append({
                'step': 'actor',
                'output': None,
                'error': actor_error
            })
            self.intermediate_rewards.reward_tool_success('actor', False)
        
        # =====================================================================
        # AUDITOR PHASE
        # =====================================================================
        
        auditor_results = []
        tagged_outputs = []
        valid = False
        
        # ✅ A-TEAM DECISION: Two-Level Validation Architecture
        # This is ACTOR-LEVEL Auditor (role-specific quality checks)
        # Swarm-level Auditor (orchestration checks) happens in conductor.py
        # Both serve different purposes!
        
        if self.agent_config and self.agent_config.enable_auditor and proceed and actor_output is not None:
            logger.info(f"🔍 Running ACTOR-level Auditor for {self.agent_config.name}")
            
            # 🔥 JOTTY v1.0: Use is_executor flag instead of hardcoded name checks!
            # Executors (query runners, API callers, etc.) return execution metadata
            actor_name = self.agent_config.name
            is_executor = getattr(self.agent_config, 'is_executor', False)
            
            if is_executor:
                execution_metadata = self._extract_execution_metadata(actor_output)
                if self.config.enable_debug_logging:
                    logger.info(f"🔍 DEBUG: Execution metadata for executor agent: {execution_metadata}")
            else:
                # Non-executor agents don't have execution metadata
                execution_metadata = {}
                logger.info(f"✅ [CONTEXT] Skipping execution metadata extraction for agent: {actor_name}")
            
            # 🔧 A-TEAM FIX: Send FULL structured data, NO TRUNCATION!
            # Extract key fields from DSPy Prediction for validation
            if hasattr(actor_output, '_store') and isinstance(actor_output._store, dict):
                # DSPy Prediction - extract all relevant fields
                action_result_dict = {
                    'type': 'dspy_prediction',
                    'fields': list(actor_output._store.keys()),
                }
                
                # Only add execution metadata if this is a SQL actor
                if execution_metadata:
                    action_result_dict.update({
                        'execution_status': execution_metadata.get('status', 'unknown'),
                        'execution_success': execution_metadata.get('success', None),
                        'row_count': execution_metadata.get('row_count', None),
                        'error': execution_metadata.get('error', None),
                    })
                # Add commonly needed fields if present
                # 🔧 A-TEAM FIX: Use ALL fields, not hardcoded SQL-specific list!
                action_result_dict.update(actor_output._store)
                action_result_str = json.dumps(action_result_dict, indent=2, default=str)
            elif hasattr(actor_output, 'trajectory') and isinstance(actor_output.trajectory, dict):
                # ReAct agent - provide trajectory info
                num_steps = len(actor_output.trajectory) // 3  # thought-action-observation triplets
                action_result_dict = {
                    'type': 'react_prediction',
                    'num_steps': num_steps,
                    'trajectory_available': True,
                    'note': 'Full trajectory in trajectory field of validation inputs'
                }
                # Try to get final observation
                final_obs_key = f'observation_{num_steps - 1}'
                if final_obs_key in actor_output.trajectory:
                    action_result_dict['final_observation'] = actor_output.trajectory[final_obs_key]
                action_result_str = json.dumps(action_result_dict, indent=2, default=str)
            else:
                # Generic output - send full string representation
                action_result_str = str(actor_output)  # NO TRUNCATION!
            
            # 🔥 A-TEAM FIX: Build Auditor inputs based on actor type
            # Different actors need different context!
            auditor_inputs = {
                'goal': goal,
                'action_taken': 'execute',
                'action_result': action_result_str,
                
                # 🎯 A-TEAM: Pass ONLY the actor's extracted output
                # Auditor will call tools to verify it!
                'actor_reasoning': getattr(actor_output, 'reasoning', '') if hasattr(actor_output, 'reasoning') else '',
                'actor_output': actor_output._store if hasattr(actor_output, '_store') else (actor_output.__dict__ if hasattr(actor_output, '__dict__') else {}),
                
                # For TaggedOutput: Pass all attempts and latest correct
                'all_attempts': getattr(actor_output, 'all_attempts', []),
                'latest_correct': None  # Will be set below if TaggedOutput
            }
            
            # 🔥 A-TEAM CRITICAL FIX: Only pass execution metadata for SQL actors!
            # JOTTY v1.0: Use is_executor flag instead of hardcoded name checks!
            actor_name = self.agent_config.name if self.agent_config else "Unknown"
            is_executor = getattr(self.agent_config, 'is_executor', False) if self.agent_config else False
            if is_executor and execution_metadata:
                # This is an executor agent - add execution metadata
                auditor_inputs.update({
                    'execution_status': execution_metadata.get('status'),
                    'execution_success': execution_metadata.get('success'),
                    'row_count': execution_metadata.get('row_count'),
                    'has_data': execution_metadata.get('has_data'),
                    'error_message': execution_metadata.get('error'),
                })
                logger.info(f"✅ [AUDITOR CONTEXT] Added execution metadata for executor: {actor_name}")
            else:
                logger.info(f"✅ [AUDITOR CONTEXT] No execution metadata for agent: {actor_name}")
            
            # 🔧 A-TEAM: Inject intercepted tool calls (GENERIC!)
            # If actor used tools, add them to Auditor context
            if hasattr(self, '_last_tool_calls') and self._last_tool_calls:
                logger.info(f"🔧 [TOOL INJECT] Adding {len(self._last_tool_calls)} intercepted tool calls to Auditor context")
                auditor_inputs['intercepted_tool_calls'] = self._last_tool_calls
                
                # 🔥 GENERIC: Convert tool calls to TaggedAttempts if actor didn't do it
                if not auditor_inputs['all_attempts']:  # Only if actor didn't create them
                    try:
                        tagged_attempts = self.tool_interceptor.to_tagged_attempts()
                        if tagged_attempts:
                            auditor_inputs['all_attempts'] = tagged_attempts
                            logger.info(f"🔧 [TOOL INJECT] Converted {len(tagged_attempts)} tool calls to TaggedAttempts")
                            
                            # Set latest_correct for convenience
                            for att in reversed(tagged_attempts):  # Most recent first
                                if att.tag == 'correct':
                                    auditor_inputs['latest_correct'] = att.output
                                    logger.info(f"🔧 [TOOL INJECT] Set latest_correct from intercepted calls")
                                    break
                    except Exception as e:
                        logger.warning(f"⚠️ [TOOL INJECT] Failed to convert to TaggedAttempts: {e}")
            
            # 🎯 Extract latest correct attempt for TaggedOutput
            try:
                from jotty.data_structures import TaggedOutput
                if isinstance(actor_output, TaggedOutput):
                    best = actor_output.get_best_attempt()
                    if best:
                        auditor_inputs['latest_correct'] = best.output
                        logger.info(f"📊 [AUDITOR INPUT] Tagged output: {len(actor_output.all_attempts)} attempts, best tag='{best.tag}'")
            except ImportError:
                pass  # TaggedOutput not available
            
            auditor_results, valid = await self.auditor_validator.validate(
                goal=goal,
                inputs=auditor_inputs,
                trajectory=self.trajectory,
                is_architect=False
            )
            
            # ✅ Extract tagged outputs - THIS SECTION IS DEPRECATED
            # TaggedOutput is now created by the actor itself (e.g., SQLGenerator)
            # We no longer create it here in Auditor
            # The actor output is already a TaggedOutput if the actor uses ReAct with tagging
            
            # ✅ USER CRITICAL FIX: Save FULL feedback to trajectory for future agents!
            self.trajectory.append({
                'step': 'auditor_actor',
                'valid': valid,
                'agents': [r.agent_name for r in auditor_results],
                'decisions': [r.is_valid for r in auditor_results],
                'tags': [r.output_tag.value if r.output_tag else None for r in auditor_results],
                # 🆕 CRITICAL: Save feedback so future agents can see it!
                'feedback': [
                    {
                        'agent': r.agent_name,
                        'reasoning': getattr(r, 'reasoning', ''),
                        'why_invalid': getattr(r, 'why_invalid', ''),
                        'suggested_fixes': getattr(r, 'suggested_fixes', []),
                        'confidence': getattr(r, 'confidence', 0.0),
                        'validation_status': getattr(r, 'validation_status', 'unknown')
                    }
                    for r in auditor_results
                ]
            })
            
            # 🆕 USER FIX: Log Auditor reasoning for debugging!
            logger.info(f"✅ ACTOR-level Auditor: valid={valid}")
            for result in auditor_results:
                logger.info(f"   📋 Auditor Agent: {result.agent_name}")
                logger.info(f"   ✓ Valid: {result.is_valid}")
                if hasattr(result, 'reasoning') and result.reasoning:
                    logger.info(f"   💭 Reasoning: {result.reasoning}")
                if hasattr(result, 'why_invalid') and result.why_invalid:
                    logger.info(f"   ⚠️ Why Invalid: {result.why_invalid}")
                if hasattr(result, 'suggested_fixes') and result.suggested_fixes:
                    logger.info(f"   🔧 Suggested Fixes: {result.suggested_fixes}")
                if hasattr(result, 'validation_status') and result.validation_status:
                    logger.info(f"   📊 Status: {result.validation_status}")
                if hasattr(result, 'confidence') and result.confidence:
                    logger.info(f"   🎯 Confidence: {result.confidence}")
            
            # Log feedback saved to trajectory for future agents
            logger.info(f"💾 Feedback saved to trajectory for future agents to learn from")
            
            # 🔥 USER FIX: Simple 3-retry strategy
            # If valid=True: PASS (no advisory override)
            # If valid=False: RETRY up to max_auditor_retries (default 3)
            # After 3 retries with feedback, actor should learn and improve
            
            # 🔄 A-TEAM: Retry mechanism for Auditor failures (3 attempts with feedback)
            auditor_retry_count = 0
            max_auditor_retries = getattr(self.config, 'max_validation_retries', 3)  # ✅ 3 retries as per user request
            
            # Track actor confidence for Auditor retries too
            if not hasattr(self, '_auditor_actor_confidence'):
                self._auditor_actor_confidence = 0.7
            
            while not valid and auditor_retry_count < max_auditor_retries:
                auditor_retry_count += 1
                
                # Build detailed feedback from Auditor reasoning
                # ✅ REFACTORED: Using helper function to eliminate duplication
                auditor_feedback = build_auditor_feedback(auditor_results)
                
                retry_type = "BLOCKED (Auditor failed)"
                logger.info(f"🔄 [AUDITOR RETRY] Type: {retry_type}")
                logger.info(f"🔄 [AUDITOR RETRY] Auditor blocked. Retry {auditor_retry_count}/{max_auditor_retries}")
                logger.info(f"🔄 [AUDITOR RETRY] Actor confidence (moving avg): {self._auditor_actor_confidence:.3f}")
                logger.info(f"🔄 [AUDITOR RETRY] Feedback (FULL):\n{auditor_feedback}")
                
                # 🔥 A-TEAM FIX: Inject feedback into query context, not as a parameter
                # ✅ CLEAN FIX: Use separate feedback field, keep query clean!
                # This preserves semantic clarity and allows LLM to process feedback separately
                
                # Keep original query clean
                if 'query' in kwargs:
                    original_query = kwargs['query']
                else:
                    original_query = None
                
                # Add feedback as SEPARATE field (generic, works for any actor)
                kwargs['auditor_feedback'] = (
                    f"=== VALIDATION FEEDBACK (Attempt {auditor_retry_count}/{max_auditor_retries}) ===\n"
                    f"{auditor_feedback}\n"
                    f"=== END FEEDBACK ===\n\n"
                    f"Please address the feedback above and revise your output accordingly.\n"
                    f"Focus on the specific issues mentioned without changing unrelated aspects."
                )
                logger.info(f"🔄 [AUDITOR RETRY] Injected feedback into SEPARATE 'auditor_feedback' field")
                logger.info(f"🔄 [AUDITOR RETRY] Original query kept clean (FULL): {original_query if original_query else 'N/A'}")
                
                # Store retry metadata for tracking (but don't pass as params)
                self._auditor_retry_metadata = {
                    'feedback': auditor_feedback,
                    'retry_count': auditor_retry_count,
                    'actor_confidence': self._auditor_actor_confidence
                }
                
                try:
                    actor_output = await self._run_actor_with_timeout(kwargs)
                    logger.info(f"🔄 [AUDITOR RETRY] Actor re-executed with feedback")
                    
                    # Re-extract execution metadata ONLY for executor agents
                    if is_executor:
                        execution_metadata = self._extract_execution_metadata(actor_output)
                    else:
                        execution_metadata = {}
                    
                    if hasattr(actor_output, '_store') and isinstance(actor_output._store, dict):
                        action_result_dict = {
                            'type': 'dspy_prediction',
                            'fields': list(actor_output._store.keys()),
                        }
                        
                        # Only add execution metadata for SQL actors
                        if execution_metadata:
                            action_result_dict.update({
                                'execution_status': execution_metadata.get('status', 'unknown'),
                                'execution_success': execution_metadata.get('success', None),
                                'row_count': execution_metadata.get('row_count', None),
                                'error': execution_metadata.get('error', None),
                            })
                        # 🔥 A-TEAM FIX: Use ALL fields (GENERIC!), not hardcoded list!
                        action_result_dict.update(actor_output._store)
                    else:
                        action_result_dict = {'type': 'unknown', 'data': str(actor_output)[:500]}
                    
                    # 🔧 A-TEAM FIX: Use correct MultiRoundValidator signature!
                    # MultiRoundValidator.validate() expects: goal, inputs, trajectory, is_architect
                    auditor_inputs_dict = {
                        'action_result': json.dumps(action_result_dict, indent=2, default=str),
                        'execution_status': execution_metadata.get('status'),
                        'execution_metadata': execution_metadata
                    }
                    
                    # 🔥 A-TEAM CRITICAL FIX: Inject available metadata into Auditor context
                    # Auditor NEEDS schemas to validate - make them directly available
                    # This is GENERIC - any metadata in SharedContext['metadata'] is passed
                    if hasattr(self, 'shared_context') and self.shared_context:
                        available_schemas = {}
                        business_terms = {}
                        
                        # Access SharedContext properly - it has .get(key) NOT .get(key, default)
                        metadata_dict = self.shared_context.get('metadata')
                        if not metadata_dict:
                            metadata_dict = {}
                        
                        if metadata_dict:
                            for key, value in metadata_dict.items():
                                # Inject table schemas
                                if key.endswith('_schema') or key == 'table_schema':
                                    available_schemas[key] = value
                                # Inject business terms
                                elif key == 'business_terms' and value:
                                    business_terms = value
                            
                            if available_schemas:
                                auditor_inputs_dict['available_schemas'] = json.dumps(available_schemas, indent=2, default=str)
                                logger.info(f"✅ [AUDITOR CONTEXT] Injected {len(available_schemas)} table schemas")
                            
                            if business_terms:
                                auditor_inputs_dict['business_terms_reference'] = json.dumps(business_terms, indent=2, default=str)
                                logger.info(f"✅ [AUDITOR CONTEXT] Injected business terms")
                    
                    auditor_results, valid = await self.auditor_validator.validate(
                        goal=goal,
                        inputs=auditor_inputs_dict,
                        trajectory=self.trajectory,
                        is_architect=False  # This is Auditor, not Architect
                    )
                    
                    # ✅ MOVING AVERAGE CONFIDENCE UPDATE
                    # ✅ REFACTORED: Using helper function to eliminate duplication
                    self._auditor_actor_confidence = update_confidence(
                        self._auditor_actor_confidence,
                        auditor_results,
                        confidence_divisor=4.0
                    )
                    
                    logger.info(f"🔄 [AUDITOR RETRY] Auditor retry result: valid={valid}")
                    
                    # Record retry in trajectory
                    # ✅ REFACTORED: Using helper function to eliminate duplication
                    self.trajectory.append(create_retry_trajectory_entry(
                        phase_name='auditor',
                        retry_count=auditor_retry_count,
                        confidence=self._auditor_actor_confidence,
                        success=valid,
                        feedback=auditor_feedback,
                        validation_results=auditor_results
                    ))
                    
                except Exception as e:
                    logger.error(f"🔄 [AUDITOR RETRY] Actor retry failed: {e}")
                    break
            
            # =====================================================================
            # 🎯 A-TEAM: CONFIDENCE-BASED OVERRIDE MECHANISM (Dec 29, 2025)
            # =====================================================================
            # If exhausted retries and still invalid, check if actor can override
            if not valid and auditor_retry_count >= max_auditor_retries:
                logger.warning(f"⚠️ [AUDITOR RETRY] Max retries ({max_auditor_retries}) exhausted. Auditor still blocking.")
                
                if self.config.enable_confidence_override:
                    # Calculate confidence differential
                    actor_conf_final = self._auditor_actor_confidence
                    validator_conf_avg = sum(
                        getattr(r, 'confidence', 0.5) for r in auditor_results
                    ) / len(auditor_results) if auditor_results else 0.5
                    
                    confidence_gap = actor_conf_final - validator_conf_avg
                    
                    logger.info(f"")
                    logger.info(f"{'='*80}")
                    logger.info(f"🎯 CONFIDENCE-BASED OVERRIDE DECISION")
                    logger.info(f"{'='*80}")
                    logger.info(f"   Actor Confidence (Final):     {actor_conf_final:.3f}")
                    logger.info(f"   Validator Confidence (Avg):   {validator_conf_avg:.3f}")
                    logger.info(f"   Confidence Gap:               {confidence_gap:+.3f}")
                    logger.info(f"   Override Threshold:           {self.config.confidence_override_threshold:.3f}")
                    logger.info(f"   Min Actor Confidence:         {self.config.min_confidence_for_override:.3f}")
                    logger.info(f"   Max Validator to Override:    {self.config.max_validator_confidence_to_override:.3f}")
                    
                    override_decision = False
                    override_reason = ""
                    
                    # Decision logic (as per A-Team consensus)
                    if confidence_gap >= self.config.confidence_override_threshold:
                        if actor_conf_final >= self.config.min_confidence_for_override:
                            if validator_conf_avg < self.config.max_validator_confidence_to_override:
                                override_decision = True
                                override_reason = (
                                    f"Actor confidence ({actor_conf_final:.2f}) significantly exceeds "
                                    f"validator ({validator_conf_avg:.2f}) with gap {confidence_gap:+.2f} "
                                    f">= threshold {self.config.confidence_override_threshold:.2f}"
                                )
                            else:
                                override_reason = (
                                    f"Validator highly confident ({validator_conf_avg:.2f}) in blocking, "
                                    f"exceeds max override threshold {self.config.max_validator_confidence_to_override:.2f}"
                                )
                        else:
                            override_reason = (
                                f"Actor confidence ({actor_conf_final:.2f}) below minimum required "
                                f"{self.config.min_confidence_for_override:.2f} for override"
                            )
                    elif actor_conf_final < 0.6 and validator_conf_avg < 0.6:
                        override_decision = True
                        override_reason = (
                            f"Both uncertain (actor:{actor_conf_final:.2f}, validator:{validator_conf_avg:.2f}) - "
                            "pass through for downstream decision"
                        )
                    else:
                        override_reason = (
                            f"Confidence gap ({confidence_gap:+.2f}) below threshold "
                            f"({self.config.confidence_override_threshold:.2f})"
                        )
                    
                    logger.info(f"")
                    logger.info(f"   🎲 Decision: {'✅ OVERRIDE ALLOWED' if override_decision else '❌ RESPECT VALIDATOR'}")
                    logger.info(f"   📝 Reason: {override_reason}")
                    logger.info(f"{'='*80}")
                    logger.info(f"")
                    
                    # Apply override decision
                    if override_decision:
                        valid = True  # Override validator's decision
                        logger.info(f"✅ [OVERRIDE] Actor's output will be accepted (marked as 'UNVALIDATED')")
                        
                        # Store override metadata in result
                        self._override_metadata = {
                            'override_applied': True,
                            'override_reason': override_reason,
                            'actor_confidence': actor_conf_final,
                            'validator_confidence': validator_conf_avg,
                            'confidence_gap': confidence_gap,
                            'validation_status': 'UNVALIDATED'
                        }
                    else:
                        logger.warning(f"❌ [BLOCKED] Actor's output blocked by validator")
                        
                        # Store non-override metadata
                        self._override_metadata = {
                            'override_applied': False,
                            'override_reason': override_reason,
                            'actor_confidence': actor_conf_final,
                            'validator_confidence': validator_conf_avg,
                            'confidence_gap': confidence_gap,
                            'validation_status': 'BLOCKED'
                        }
                else:
                    logger.info(f"⚠️  [OVERRIDE] Confidence-based override disabled in config")
                    self._override_metadata = {
                        'override_applied': False,
                        'override_reason': 'Confidence override disabled',
                        'validation_status': 'BLOCKED'
                    }
        
        elif self.agent_config and not self.agent_config.enable_auditor:
            # Auditor disabled - assume valid if actor succeeded
            logger.info(f"⏩ ACTOR-level Auditor skipped for '{self.agent_config.name}' (disabled)")
            valid = (actor_output is not None and actor_error is None)
            self._override_metadata = None
        elif proceed and actor_output is not None:
            # Auditor enabled but conditions not met
            valid = False
            self._override_metadata = None
        
        # =====================================================================
        # LEARNING PHASE
        # =====================================================================
        
        # Determine success
        success = proceed and valid and actor_error is None
        
        # Compute final reward
        if success:
            final_reward = 1.0
            if valid and len(tagged_outputs) > 0:
                final_reward += self.config.approval_reward_bonus
        elif not proceed:
            final_reward = -0.5  # Blocked
        else:
            final_reward = -0.8  # Failed after execution
        
        # Credit assignment
        actor_succeeded = actor_output is not None and actor_error is None
        contributions = self.credit_assigner.analyze_contributions(
            success=success,
            architect_results=architect_results,
            auditor_results=auditor_results,
            actor_succeeded=actor_succeeded,
            trajectory=self.trajectory
        )
        
        # Per-agent TD updates
        td_updates = []
        for agent in self.architect_agents + self.auditor_agents:
            contribution = contributions.get(agent.agent_name)
            if contribution:
                # Modulate reward by contribution
                modulated_reward = final_reward * (0.5 + 0.5 * contribution.compute_final_contribution())
            else:
                modulated_reward = final_reward * 0.5
            
            # Get all memories
            all_mems = {}
            for level in MemoryLevel:
                all_mems.update(agent.memory.memories[level])
            
            # TD update
            updates = self.td_learner.end_episode(
                modulated_reward, 
                all_mems,
                self.goal_hierarchy if self.config.enable_goal_hierarchy else None
            )
            td_updates.extend(updates)
            
            # Track accessed memories
            memories_accessed[agent.agent_name] = [u[0] for u in updates]
            
            # Record for exploration
            for _, old_v, new_v in updates:
                self.exploration.record_value_change(new_v - old_v)
        
        # Health monitoring
        alerts = self.health_monitor.record_episode(
            success=success,
            goal=goal,
            architect_decisions=[r.should_proceed for r in architect_results],
            auditor_decisions=[r.is_valid for r in auditor_results if r.is_valid is not None],
            value_updates=td_updates
        )
        
        # Extract learned instructions from failures
        if not success:
            self._extract_failure_lessons(goal, architect_results, auditor_results, actor_error)
        
        # Store episode for offline learning
        stored_episode = StoredEpisode(
            episode_id=self.episode_count,
            goal=goal,
            goal_id=self.goal_hierarchy.add_goal(goal) if self.config.enable_goal_hierarchy else "",
            trajectory=self.trajectory,
            kwargs={k: str(v) for k, v in kwargs.items()},  # 🔧 A-TEAM FIX: NO TRUNCATION!
            success=success,
            final_reward=final_reward,
            architect_results=architect_results,
            auditor_results=auditor_results,
            actor_output=str(actor_output) if actor_output else None,  # 🔧 A-TEAM FIX: NO TRUNCATION!
            actor_error=actor_error,
            agent_contributions=contributions,
            memories_accessed=memories_accessed,
            execution_time=time.time() - start_time
        )
        
        avg_td_error = sum(abs(old - new) for _, old, new in td_updates) / max(1, len(td_updates))
        self.offline_learner.store_episode(stored_episode, avg_td_error)
        
        # Periodic offline update
        if self.episode_count % self.config.offline_update_interval == 0:
            agent_memories = {a.agent_name: a.memory for a in self.architect_agents + self.auditor_agents}
            await self.offline_learner.batch_update(agent_memories, self.episode_count)
        
        # Periodic consolidation
        if self.episode_count % self.config.consolidation_threshold == 0:
            episodes = self.offline_learner.buffer.get_recent(100)
            for agent in self.architect_agents + self.auditor_agents:
                await agent.memory.consolidate(episodes)
        
        # Periodic backup
        if self.config.enable_backups and self.episode_count % self.config.backup_interval == 0:
            self.persistence.create_backup(self.episode_count)
        
        # Auto-save
        if self.config.auto_save and self.episode_count % self.config.save_interval == 0:
            self._save_state()
        
        # =====================================================================
        # BUILD RESULT
        # =====================================================================
        
        # Get causal insights if any
        causal_insights = []
        if self.config.enable_causal_learning and not success:
            for agent in self.architect_agents + self.auditor_agents:
                causal = agent.memory.retrieve_causal(goal, kwargs)
                for link in causal:
                    causal_insights.append(f"Because {link.cause}, therefore {link.effect}")
        
        # 🔍 A-TEAM DEBUG: Log what's being stored in EpisodeResult
        logger.info(f"[🔍 EPISODE RESULT] Creating EpisodeResult for '{self.agent_config.name if self.agent_config else 'UNKNOWN'}'")
        logger.info(f"[🔍 EPISODE RESULT]   actor_output type: {type(actor_output)}")
        logger.info(f"[🔍 EPISODE RESULT]   actor_output is None: {actor_output is None}")
        logger.info(f"[🔍 EPISODE RESULT]   success: {success}")
        if actor_output and hasattr(actor_output, '_store'):
            logger.info(f"[🔍 EPISODE RESULT]   actor_output is DSPy Prediction with _store")
        
        return EpisodeResult(
            output=actor_output,
            success=success,
            trajectory=self.trajectory,
            tagged_outputs=tagged_outputs,
            episode=self.episode_count,
            execution_time=time.time() - start_time,
            architect_results=architect_results,
            auditor_results=auditor_results,
            agent_contributions=contributions,
            memories_updated=len(td_updates),
            alerts=alerts,
            causal_insights=causal_insights,
            validation_rounds=max(
                len(r.previous_rounds) + 1 for r in architect_results + auditor_results
            ) if architect_results or auditor_results else 1,
            override_metadata=getattr(self, '_override_metadata', None),
            refinement_improvements=[]
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _extract_goal(self, kwargs: Dict) -> str:
        """Extract goal from kwargs."""
        for key in ['question', 'query', 'goal', 'task', 'input']:
            if key in kwargs:
                return str(kwargs[key])
        return str(list(kwargs.values())[0]) if kwargs else "unknown"
    
    def _inject_learned_instructions(self, kwargs: Dict) -> Dict:
        """Inject learned instructions into kwargs.
        
#         ✅ FIX: Store in context for logging ONLY - DON'T add to kwargs!
        DSPy modules don't accept this parameter.
        """
        instructions = []
        
        for source in ['actor', 'architect', 'auditor']:
            for inst in self.learned_instructions[source]:  # Last 10
                instructions.append(f"LEARNED: {inst}")
        
        if instructions:
            # Store in context, NOT in kwargs!
            self.context['_learned_instructions'] = "\n".join(instructions)
        
        return kwargs  # Return unchanged!
    
    def _extract_execution_metadata(self, actor_output) -> dict:
        """
        🆕 A-TEAM FIX V3: Extract execution results from actor output.
        
        Supports BOTH:
        1. TaggedOutput (from SQLGenerator) - has all_attempts
        2. DSPy Prediction with trajectory (from pure ReAct agents)
        
        Returns dict with keys: status, success, row_count, error, has_data
        """
        metadata = {
            'status': 'unknown',
            'success': None,
            'row_count': None,
            'error': None,
            'has_data': False
        }
        
        # 🔥 STRATEGY 1: TaggedOutput (SQLGenerator pattern)
        if hasattr(actor_output, 'all_attempts') and hasattr(actor_output, 'final_result'):
            attempts = actor_output.all_attempts
            if attempts:
                # Check if any attempt is an 'answer' (success)
                successful_attempts = [a for a in attempts if a.is_answer()]
                if successful_attempts:
                    metadata['status'] = 'success'
                    metadata['success'] = True
                    # Check last successful observation for data
                    last_success = successful_attempts[-1]
                    obs_str = str(last_success.observation).lower()
                    metadata['has_data'] = bool(last_success.observation and 'result:' in obs_str)
                else:
                    # All attempts failed
                    metadata['status'] = 'failed'
                    metadata['success'] = False
                    last_attempt = attempts[-1]
                    if last_attempt.observation:
                        metadata['error'] = str(last_attempt.observation)[:200]
            else:
                # No attempts tracked
                metadata['status'] = 'unknown'
                # But check if final_result has data
                if actor_output.final_result:
                    result_str = str(actor_output.final_result).lower()
                    if 'result:' in result_str or 'rows' in result_str:
                        metadata['has_data'] = True
                        metadata['status'] = 'success'
                        metadata['success'] = True
            
            return metadata
        
        # 🔥 STRATEGY 2: DSPy Prediction with trajectory (pure ReAct)
        if hasattr(actor_output, 'trajectory') and isinstance(actor_output.trajectory, dict):
            # Find all observations
            observations = [k for k in actor_output.trajectory.keys() if k.startswith('observation_')]
            
            # Check observation_0 FIRST (tool result), not the last one!
            target_observation = None
            if 'observation_0' in actor_output.trajectory:
                target_observation = actor_output.trajectory['observation_0']
            elif observations:
                # Fallback: use last observation
                last_obs_key = sorted(observations, key=lambda x: int(x.split('_')[1]))[-1]
                target_observation = actor_output.trajectory.get(last_obs_key, '')
            
            if target_observation:
                
                # Parse observation for execution info
                obs_str = str(target_observation)
                
                # 🆕 A-TEAM CRITICAL FIX: Parse dict string first!
                # The tool returns {"result": "...", "execution_time": ...}
                # DSPy stores it as str(dict) in trajectory
                result_text = obs_str
                try:
                    import ast
                    obs_dict = ast.literal_eval(obs_str)
                    if isinstance(obs_dict, dict) and 'result' in obs_dict:
                        result_text = obs_dict['result']  # Extract actual result!
                        logger.info(f"✅ DEBUG: Successfully parsed dict, extracted 'result' field")
                        if self.config.enable_debug_logging:
                            logger.info(f"🔍 DEBUG: Result text (first 300 chars): {result_text}")
                except (ValueError, SyntaxError, TypeError) as e:
                    # Not a dict string, use as-is
                    if self.config.enable_debug_logging:
                        logger.info(f"🔍 DEBUG: Could not parse as dict ({type(e).__name__}), using raw string")
                    result_text = obs_str
                
                # Now check patterns in result_text
                if 'Query executed successfully' in result_text or 'rows returned' in result_text:
                    metadata['status'] = 'success'
                    metadata['success'] = True
                    logger.info(f"✅ DEBUG: Detected successful execution")
                    
                    # Try to extract row count
                    import re
                    row_match = re.search(r'(\d+)\s+rows', result_text)
                    if row_match:
                        metadata['row_count'] = int(row_match.group(1))
                        metadata['has_data'] = metadata['row_count'] > 0
                        logger.info(f"✅ DEBUG: Extracted row count: {metadata['row_count']}")
                
                # Check for error indicators
                elif 'Error' in result_text or 'Failed' in result_text or 'Exception' in result_text:
                    metadata['status'] = 'error'
                    metadata['success'] = False
                    metadata['error'] = result_text  # ALL chars of error
                    logger.info(f"❌ DEBUG: Detected error in execution")
                
                # Check for no data
                elif 'No rows' in result_text or 'Empty result' in result_text:
                    metadata['status'] = 'success'
                    metadata['success'] = True
                    metadata['row_count'] = 0
                    metadata['has_data'] = False
                    logger.info(f"⚠️ DEBUG: Detected empty result (0 rows)")
        
        if self.config.enable_debug_logging:
            logger.info(f"🔍 DEBUG: Final metadata: {metadata}")
        return metadata
    
    async def _run_actor_with_timeout(self, kwargs: Dict) -> Any:
        """Run actor with timeout."""
        try:
            # 🔑 A-TEAM FIX: Use actor_timeout (900s) instead of async_timeout (60s)!
            timeout = getattr(self.config, 'actor_timeout', self.config.async_timeout)
            return await asyncio.wait_for(
                self._run_actor(kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            timeout = getattr(self.config, 'actor_timeout', self.config.async_timeout)
            raise TimeoutError(f"Actor execution timed out after {timeout}s")
    
    async def _run_actor(self, kwargs: Dict) -> Any:
        """Run the actor (handles both sync and async)."""
        # 🔍 A-TEAM: EXTREME DEBUG - Trace actor execution
        logger.info(f"[🔍 ACTOR EXEC] Actor: {self.actor.__class__.__name__ if hasattr(self.actor, '__class__') else type(self.actor)}")
        logger.info(f"[🔍 ACTOR EXEC] Is coroutine function: {asyncio.iscoroutinefunction(self.actor)}")
        logger.info(f"[🔍 ACTOR EXEC] Input kwargs keys (before filtering): {list(kwargs.keys())}")
        
        # 🔧 A-TEAM: Tool interception strategy
        # For actors that set tools dynamically in forward() (like SQLGenerator),
        # we CAN'T wrap here because tools don't exist yet.
        # Instead, actors should track attempts internally and return TaggedOutput.
        # 
        # JOTTY's tool interceptor is available for actors that DON'T track internally.
        # Check if actor has a 'generator' attribute (DSPy ReAct pattern)
        
        original_tools = None
        wrapped_tools_applied = False
        
        # DISABLED: Wrapping conflicts with dynamic tool binding in forward()
        # if hasattr(self.actor, 'generator') and hasattr(self.actor.generator, 'tools'):
        #     # For ReAct agents with nested `generator` module
        #     if isinstance(self.actor.generator.tools, dict) and self.actor.generator.tools:
        #         logger.info(f"🔧 [TOOL WRAP] Detected DSPy ReAct tools: {list(self.actor.generator.tools.keys())}")
        #         original_tools = self.actor.generator.tools.copy()
        #         wrapped_tools = self.tool_interceptor.wrap_tools(original_tools)
        #         self.actor.generator.tools = wrapped_tools
        #         wrapped_tools_applied = True
        #         logger.info(f"🔧 [TOOL WRAP] Applied wrapping")
        
        logger.info("🔧 [TOOL WRAP] Skipping (actor tracks attempts internally via TaggedOutput)")
        
        # 🔥 A-TEAM RESILIENT SYSTEM: Auto-detect actor signature and filter dynamically!
        # This makes the system FOOLPROOF - it adapts to ANY actor signature!
        
        # Step 1: Get actor's expected parameters from its signature
        expected_params = set()
        try:
            if hasattr(self.actor, 'forward'):
                import inspect
                sig = inspect.signature(self.actor.forward)
                expected_params = set(sig.parameters.keys())
                logger.info(f"[🔍 SIGNATURE] Actor expects parameters: {expected_params}")
            elif hasattr(self.actor, '__call__'):
                import inspect
                sig = inspect.signature(self.actor.__call__)
                expected_params = set(sig.parameters.keys())
                logger.info(f"[🔍 SIGNATURE] Actor expects parameters: {expected_params}")
        except Exception as e:
            logger.warning(f"⚠️ Could not introspect actor signature: {e}")
            # Fallback to static filtering if signature introspection fails
            expected_params = None
        
        # Step 2: Filter kwargs to ONLY include expected params
        if expected_params:
            # RESILIENT APPROACH: Only pass params the actor expects
            actor_kwargs = {k: v for k, v in kwargs.items() if k in expected_params}
            
            # Log what we filtered out
            filtered_out = set(kwargs.keys()) - expected_params
            if filtered_out:
                logger.info(f"[🔍 FILTER] Filtered out {len(filtered_out)} params not in signature: {filtered_out}")
        else:
            # Fallback: Use static internal params list
            internal_params = {
                # Architect retry params
                '_architect_feedback',
                '_retry_count',
                '_actor_confidence',
                '_instruction',
                # Auditor retry params
                '_auditor_feedback',
                '_auditor_retry_count',
                '_auditor_actor_confidence',
                # Any other internal JOTTY params
                '_jotty_internal',
                '_learned_instructions',
                '_injected_context',
                '_injected_instructions'
            }
            actor_kwargs = {k: v for k, v in kwargs.items() if k not in internal_params}
            logger.warning(f"[🔍 FILTER] Using static filter (fallback) - removed {len(set(kwargs.keys()) - set(actor_kwargs.keys()))} params")
        
        logger.info(f"[🔍 ACTOR EXEC] Input kwargs keys (after filtering): {list(actor_kwargs.keys())}")
        
        try:
            if asyncio.iscoroutinefunction(self.actor):
                result = await self.actor(**actor_kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self.actor(**actor_kwargs))
            
            # 🔍 A-TEAM: EXTREME DEBUG - Trace actor output
            logger.info(f"[🔍 ACTOR OUTPUT] Result type: {type(result)}")
            logger.info(f"[🔍 ACTOR OUTPUT] Result is None: {result is None}")
            
            if result is not None:
                # Show what fields the result has
                if hasattr(result, '_store'):
                    logger.info(f"[🔍 ACTOR OUTPUT] DSPy Prediction with _store keys: {list(result._store.keys())}")
                elif isinstance(result, dict):
                    logger.info(f"[🔍 ACTOR OUTPUT] Dict with keys: {list(result.keys())}")
                else:
                    logger.info(f"[🔍 ACTOR OUTPUT] Has attributes (FULL): {dir(result)}")
            else:
                logger.error(f"[🔍 ACTOR OUTPUT] ❌ ACTOR RETURNED NONE!")
                
            return result
        except Exception as e:
            logger.error(f"[🔍 ACTOR EXEC] ❌ Exception during actor execution: {e}")
            import traceback
            logger.error(f"[🔍 ACTOR EXEC] Traceback: {traceback.format_exc()}")
            raise
        finally:
            # 🔧 A-TEAM: Cleanup (tool wrapping currently disabled)
            # Actors track attempts internally via TaggedOutput
            if wrapped_tools_applied:
                logger.info(f"🔧 [TOOL WRAP] Cleanup (if needed)")
    
    def _extract_failure_lessons(self,
                                  goal: str,
                                  architect_results: List[ValidationResult],
                                  auditor_results: List[ValidationResult],
                                  actor_error: Optional[str]):
        """Extract lessons from failures."""
        lessons = []
        
        # From Architect rejections
        for result in architect_results:
            if not result.should_proceed and result.confidence > 0.7:
                lessons.append(f"Blocked by {result.agent_name}: {result.reasoning}")  # 🔧 A-TEAM FIX: NO TRUNCATION!
        
        # From Auditor failures
        for result in auditor_results:
            if not result.is_valid and result.confidence > 0.7:
                lessons.append(f"Invalid per {result.agent_name}: {result.reasoning}")  # 🔧 A-TEAM FIX: NO TRUNCATION!
        
        # From actor error
        if actor_error:
            lessons.append(f"Actor error: {actor_error}")  # 🔧 A-TEAM FIX: NO TRUNCATION!
        
        # Store lessons (keep reasonable limit on NUMBER, not length)
        for lesson in lessons:  # 🔧 A-TEAM FIX: Increased from 3 to 10
            if lesson not in self.learned_instructions['actor']:
                self.learned_instructions['actor'].append(lesson)
        
        # Keep bounded
        for key in self.learned_instructions:
            self.learned_instructions[key] = self.learned_instructions[key]
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def _save_state(self):
        """Save complete state."""
        # Global state
        state = {
            'episode_count': self.episode_count,
            'learned_instructions': self.learned_instructions,
            'goal_hierarchy': {
                'nodes': {k: vars(v) for k, v in self.goal_hierarchy.nodes.items()},
                'root_id': self.goal_hierarchy.root_id
            },
            'health_metrics': self.health_monitor.get_health_summary(),
            'last_saved': datetime.now().isoformat()
        }
        self.persistence.save_state(state)
        
        # Agent memories
        for agent in self.architect_agents + self.auditor_agents:
            self.persistence.save_memory(agent.agent_name, agent.memory)
        
        # Offline learner
        self.persistence.save_offline(self.offline_learner)
    
    def _load_state(self):
        """Load state if exists."""
        state = self.persistence.load_state()
        
        if state:
            self.episode_count = state.get('episode_count', 0)
            loaded_instructions = state.get('learned_instructions', {})
            
            # 🔧 JOTTY FIX: Migrate old 'preval'/'postval' keys to 'architect'/'auditor'
            # This ensures backward compatibility with old state files
            self.learned_instructions = {
                'architect': loaded_instructions.get('architect', loaded_instructions.get('preval', [])),
                'auditor': loaded_instructions.get('auditor', loaded_instructions.get('postval', [])),
                'actor': loaded_instructions.get('actor', [])
            }
            
            # Load goal hierarchy
            gh_data = state.get('goal_hierarchy', {})
            self.goal_hierarchy.root_id = gh_data.get('root_id', 'root')
            from ..foundation.data_structures import GoalNode
            for node_id, node_data in gh_data.get('nodes', {}).items():
                self.goal_hierarchy.nodes[node_id] = GoalNode(**node_data)
        
        # Load agent memories
        for agent in self.architect_agents + self.auditor_agents:
            mem_data = self.persistence.load_memory(agent.agent_name)
            if mem_data:
                agent.memory = HierarchicalMemory.from_dict(mem_data, self.config)
        
        # Load offline learner
        offline_data = self.persistence.load_offline()
        if offline_data:
            self.offline_learner = OfflineLearner.from_dict(offline_data, self.config)
            self.offline_learner.td_learner = self.td_learner
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'episode_count': self.episode_count,
            'health': self.health_monitor.get_health_summary(),
            'exploration': {
                'epsilon': self.exploration.epsilon,
                'goals_visited': dict(self.exploration.goal_visit_counts)
            },
            'learning_rate': self.adaptive_lr.alpha,
            'agents': {
                agent.agent_name: agent.get_statistics()
                for agent in self.architect_agents + self.auditor_agents
            },
            'offline': self.offline_learner.get_statistics(),
            'learned_instructions': {
                k: len(v) for k, v in self.learned_instructions.items()
            }
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_jotty(actor: dspy.Module,
                 architect_prompts: List[str],
                 auditor_prompts: List[str],
                 architect_tools: List[Any],
                 auditor_tools: List[Any],
                 **config_kwargs) -> 'JottyCore':
    """
    Convenience function to create JottyCore with config kwargs.
    
    Example:
        jotty = create_jotty(
            actor=my_actor,
            architect_prompts=["security.md"],
            auditor_prompts=["validator.md"],
            architect_tools=[check_table],
            auditor_tools=[verify_result],
            base_path="~/.jotty/my_project",
            enable_causal_learning=True
        )
    """
    config = JottyConfig(**config_kwargs)
    return JottyCore(
        actor=actor,
        architect_prompts=architect_prompts,
        auditor_prompts=auditor_prompts,
        architect_tools=architect_tools,
        auditor_tools=auditor_tools,
        config=config
    )


# Backward compatibility alias
create_reval = create_jotty

# =============================================================================
# PHASE 7: BACKWARD COMPATIBILITY ALIAS
# =============================================================================

# Deprecated alias: JottyCore → SingleAgentOrchestrator
JottyCore = SingleAgentOrchestrator
