"""
Base Swarm Infrastructure
=========================

Foundation for all Jotty swarms with:
- Self-improving feedback loop (Expert → Reviewer → Planner → Actor)
- Shared resources (memory, context, bus, learner)
- Gold standard evaluation and agent improvement
- Execution tracking and learning

This is the CORE infrastructure that all domain swarms inherit from.

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│                         SELF-IMPROVING LOOP                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Expert  │───►│ Reviewer │───►│ Planner  │───►│  Actor   │          │
│  │  Agent   │    │  Agent   │    │  Agent   │    │  Agent   │          │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘          │
│       │               │               │               │                 │
│       ▼               ▼               ▼               ▼                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    SHARED RESOURCES                             │    │
│  │  Memory (5-level) | Context | Message Bus | TD-Lambda Learner  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      GOLD STANDARD DB                           │    │
│  │  Expected outputs, evaluation criteria, improvement history     │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
import json
import hashlib
import dspy
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Callable, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class AgentRole(Enum):
    """Roles in the self-improving loop."""
    EXPERT = "expert"          # Evaluates against gold standard
    REVIEWER = "reviewer"      # Reviews agent performance
    PLANNER = "planner"        # Plans task execution
    ACTOR = "actor"            # Executes tasks
    ORCHESTRATOR = "orchestrator"  # Coordinates agents


class EvaluationResult(Enum):
    """Evaluation outcomes."""
    EXCELLENT = "excellent"    # Exceeds gold standard
    GOOD = "good"              # Meets gold standard
    ACCEPTABLE = "acceptable"  # Minor issues
    NEEDS_IMPROVEMENT = "needs_improvement"
    FAILED = "failed"


class ImprovementType(Enum):
    """Types of improvements suggested."""
    PROMPT_REFINEMENT = "prompt_refinement"
    PARAMETER_TUNING = "parameter_tuning"
    WORKFLOW_CHANGE = "workflow_change"
    AGENT_REPLACEMENT = "agent_replacement"
    TRAINING_DATA = "training_data"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GoldStandard:
    """Gold standard for evaluation."""
    id: str
    domain: str
    task_type: str
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    evaluation_criteria: Dict[str, float]  # criterion -> weight
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1


@dataclass
class Evaluation:
    """Result of expert evaluation."""
    gold_standard_id: str
    actual_output: Dict[str, Any]
    scores: Dict[str, float]  # criterion -> score (0-1)
    overall_score: float
    result: EvaluationResult
    feedback: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementSuggestion:
    """Suggestion from reviewer for improvement."""
    agent_role: AgentRole
    improvement_type: ImprovementType
    description: str
    priority: int  # 1-5, 5 being highest
    expected_impact: float  # 0-1
    implementation_details: Dict[str, Any]
    based_on_evaluations: List[str]  # evaluation IDs


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    role: AgentRole
    name: str
    model: str = "sonnet"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class ExecutionTrace:
    """Trace of agent execution for learning."""
    agent_name: str
    agent_role: AgentRole
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SwarmConfig:
    """Base configuration for all swarms."""
    name: str = "BaseSwarm"
    domain: str = "general"
    version: str = "1.0.0"
    enable_self_improvement: bool = True
    enable_learning: bool = True
    parallel_execution: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    gold_standard_path: Optional[str] = None
    improvement_threshold: float = 0.7  # Below this triggers improvement
    output_dir: str = field(default_factory=lambda: str(Path.home() / "jotty" / "swarm_outputs"))


@dataclass
class SwarmResult:
    """Base result from any swarm."""
    success: bool
    swarm_name: str
    domain: str
    output: Dict[str, Any]
    execution_time: float
    agent_traces: List[ExecutionTrace] = field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    improvements: List[ImprovementSuggestion] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DSPy SIGNATURES FOR SELF-IMPROVING LOOP
# =============================================================================

class ExpertEvaluationSignature(dspy.Signature):
    """Evaluate output against gold standard.

    You are an EXPERT EVALUATOR. Compare actual output against expected output
    using the provided evaluation criteria. Be STRICT but FAIR.

    Score each criterion from 0.0 to 1.0:
    - 1.0: Perfect match or exceeds expectations
    - 0.8: Minor deviations, still excellent
    - 0.6: Acceptable with some issues
    - 0.4: Significant gaps
    - 0.2: Major problems
    - 0.0: Complete failure

    Provide specific, actionable feedback.
    """
    gold_standard: str = dspy.InputField(desc="JSON of expected output and criteria")
    actual_output: str = dspy.InputField(desc="JSON of actual output to evaluate")
    context: str = dspy.InputField(desc="Additional context about the task")

    scores: str = dspy.OutputField(desc="JSON dict of criterion -> score (0.0-1.0)")
    overall_score: float = dspy.OutputField(desc="Weighted overall score 0.0-1.0")
    result: str = dspy.OutputField(desc="EXCELLENT, GOOD, ACCEPTABLE, NEEDS_IMPROVEMENT, or FAILED")
    feedback: str = dspy.OutputField(desc="Specific feedback items separated by |")


class ReviewerAnalysisSignature(dspy.Signature):
    """Analyze evaluations and suggest improvements.

    You are a SENIOR REVIEWER analyzing patterns in agent performance.
    Based on multiple evaluations, identify systematic issues and suggest
    concrete improvements.

    Focus on:
    1. Recurring failure patterns
    2. Prompt engineering improvements
    3. Workflow optimizations
    4. Parameter tuning opportunities
    """
    evaluations: str = dspy.InputField(desc="JSON list of recent evaluations")
    agent_configs: str = dspy.InputField(desc="JSON of current agent configurations")
    improvement_history: str = dspy.InputField(desc="JSON of past improvements and their outcomes")

    analysis: str = dspy.OutputField(desc="Analysis of patterns and issues")
    suggestions: str = dspy.OutputField(desc="JSON list of ImprovementSuggestion objects")
    priority_actions: str = dspy.OutputField(desc="Top 3 priority actions separated by |")


class PlannerOptimizationSignature(dspy.Signature):
    """Optimize task planning based on feedback.

    You are a PLANNING EXPERT. Given improvement suggestions and past performance,
    optimize how tasks should be broken down and sequenced.

    Consider:
    1. Task granularity
    2. Dependency ordering
    3. Parallel execution opportunities
    4. Error recovery strategies
    """
    task_description: str = dspy.InputField(desc="Description of the task to plan")
    improvement_suggestions: str = dspy.InputField(desc="JSON of relevant improvements")
    past_plans: str = dspy.InputField(desc="JSON of similar past plans and their success rates")

    optimized_plan: str = dspy.OutputField(desc="JSON of optimized task breakdown")
    rationale: str = dspy.OutputField(desc="Explanation of planning decisions")
    risk_mitigations: str = dspy.OutputField(desc="Identified risks and mitigations separated by |")


class ActorExecutionSignature(dspy.Signature):
    """Execute a task with self-awareness of improvement areas.

    You are an EXPERT ACTOR. Execute the task while being mindful of
    past feedback and improvement suggestions.

    Apply learnings from:
    1. Previous evaluation feedback
    2. Reviewer suggestions
    3. Quality standards
    """
    task: str = dspy.InputField(desc="Task to execute")
    context: str = dspy.InputField(desc="Execution context and requirements")
    learnings: str = dspy.InputField(desc="Relevant learnings from past executions")

    output: str = dspy.OutputField(desc="Task output")
    confidence: float = dspy.OutputField(desc="Confidence in output 0.0-1.0")
    applied_learnings: str = dspy.OutputField(desc="Which learnings were applied, separated by |")


# =============================================================================
# GOLD STANDARD DATABASE
# =============================================================================

class GoldStandardDB:
    """
    Database for storing and retrieving gold standards.

    Supports:
    - JSON file storage
    - In-memory caching
    - Version tracking
    - Domain filtering
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else Path.home() / "jotty" / "gold_standards"
        self.path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, GoldStandard] = {}
        self._load_cache()

    def _load_cache(self):
        """Load all gold standards into cache."""
        for file in self.path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    gs = GoldStandard(
                        id=data['id'],
                        domain=data['domain'],
                        task_type=data['task_type'],
                        input_data=data['input_data'],
                        expected_output=data['expected_output'],
                        evaluation_criteria=data['evaluation_criteria'],
                        created_at=datetime.fromisoformat(data.get('created_at', datetime.now().isoformat())),
                        version=data.get('version', 1)
                    )
                    self._cache[gs.id] = gs
            except Exception as e:
                logger.warning(f"Failed to load gold standard from {file}: {e}")

    def add(self, gold_standard: GoldStandard) -> str:
        """Add a gold standard to the database."""
        # Generate ID if not provided
        if not gold_standard.id:
            content = json.dumps({
                'domain': gold_standard.domain,
                'task_type': gold_standard.task_type,
                'input_data': gold_standard.input_data
            }, sort_keys=True)
            gold_standard.id = hashlib.md5(content.encode()).hexdigest()[:12]

        # Save to file
        file_path = self.path / f"{gold_standard.id}.json"
        with open(file_path, 'w') as f:
            json.dump({
                'id': gold_standard.id,
                'domain': gold_standard.domain,
                'task_type': gold_standard.task_type,
                'input_data': gold_standard.input_data,
                'expected_output': gold_standard.expected_output,
                'evaluation_criteria': gold_standard.evaluation_criteria,
                'created_at': gold_standard.created_at.isoformat(),
                'version': gold_standard.version
            }, f, indent=2)

        self._cache[gold_standard.id] = gold_standard
        return gold_standard.id

    def get(self, id: str) -> Optional[GoldStandard]:
        """Get a gold standard by ID."""
        return self._cache.get(id)

    def find_by_domain(self, domain: str) -> List[GoldStandard]:
        """Find all gold standards for a domain."""
        return [gs for gs in self._cache.values() if gs.domain == domain]

    def find_similar(self, task_type: str, input_data: Dict[str, Any]) -> Optional[GoldStandard]:
        """Find the most similar gold standard for evaluation."""
        candidates = [gs for gs in self._cache.values() if gs.task_type == task_type]
        if not candidates:
            return None
        # Simple similarity: prefer exact task_type match
        return candidates[0]

    def list_all(self) -> List[GoldStandard]:
        """List all gold standards."""
        return list(self._cache.values())


# =============================================================================
# IMPROVEMENT HISTORY
# =============================================================================

class ImprovementHistory:
    """
    Tracks improvement suggestions and their outcomes.

    Used for:
    - Learning what improvements work
    - Avoiding repeated failed improvements
    - Measuring improvement velocity
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else Path.home() / "jotty" / "improvements"
        self.path.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self._load_history()

    def _load_history(self):
        """Load improvement history."""
        history_file = self.path / "history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.history = json.load(f)

    def _save_history(self):
        """Save improvement history."""
        history_file = self.path / "history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def record_suggestion(self, suggestion: ImprovementSuggestion) -> str:
        """Record an improvement suggestion."""
        entry = {
            'id': hashlib.md5(f"{suggestion.agent_role.value}:{suggestion.description}:{datetime.now()}".encode()).hexdigest()[:12],
            'suggestion': asdict(suggestion),
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'applied_at': None,
            'outcome': None,
            'impact_measured': None
        }
        self.history.append(entry)
        self._save_history()
        return entry['id']

    def mark_applied(self, suggestion_id: str):
        """Mark a suggestion as applied."""
        for entry in self.history:
            if entry['id'] == suggestion_id:
                entry['status'] = 'applied'
                entry['applied_at'] = datetime.now().isoformat()
                break
        self._save_history()

    def record_outcome(self, suggestion_id: str, success: bool, impact: float, notes: str = ""):
        """Record the outcome of an applied improvement."""
        for entry in self.history:
            if entry['id'] == suggestion_id:
                entry['status'] = 'completed'
                entry['outcome'] = 'success' if success else 'failure'
                entry['impact_measured'] = impact
                entry['notes'] = notes
                break
        self._save_history()

    def get_successful_improvements(self, agent_role: Optional[AgentRole] = None) -> List[Dict]:
        """Get successful improvements, optionally filtered by role."""
        successful = [e for e in self.history if e.get('outcome') == 'success']
        if agent_role:
            successful = [e for e in successful if e['suggestion']['agent_role'] == agent_role.value]
        return successful

    def get_pending_suggestions(self) -> List[Dict]:
        """Get pending suggestions."""
        return [e for e in self.history if e['status'] == 'pending']


class EvaluationHistory:
    """Persistent evaluation tracking across sessions.
    Follows same pattern as ImprovementHistory."""

    def __init__(self, path=None):
        self.path = Path(path) if path else Path.home() / "jotty" / "evaluations"
        self.path.mkdir(parents=True, exist_ok=True)
        self.evaluations: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        history_file = self.path / "evaluations.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.evaluations = json.load(f)

    def _save(self):
        history_file = self.path / "evaluations.json"
        with open(history_file, 'w') as f:
            json.dump(self.evaluations[-200:], f, indent=2, default=str)

    def record(self, evaluation) -> None:
        entry = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': evaluation.overall_score if hasattr(evaluation, 'overall_score') else 0,
            'status': evaluation.status if hasattr(evaluation, 'status') else 'unknown',
            'scores': evaluation.dimension_scores if hasattr(evaluation, 'dimension_scores') else {},
            'feedback': evaluation.feedback if hasattr(evaluation, 'feedback') else '',
        }
        self.evaluations.append(entry)
        self._save()

    def get_recent(self, n=10) -> List[Dict]:
        return self.evaluations[-n:]

    def get_average_score(self, n=10) -> float:
        recent = self.get_recent(n)
        if not recent:
            return 0.0
        return sum(e.get('overall_score', 0) for e in recent) / len(recent)

    def get_failures(self, n=20) -> List[Dict]:
        """Get recent failures for failure recovery analysis."""
        return [e for e in self.evaluations[-n:] if e.get('overall_score', 1.0) < 0.5]


# =============================================================================
# SELF-IMPROVING AGENTS
# =============================================================================

class ExpertAgent:
    """
    Expert Agent - Evaluates outputs against gold standards.

    The Expert is the quality gatekeeper. It:
    1. Compares actual output to expected output
    2. Scores each evaluation criterion
    3. Provides specific feedback
    4. Determines if improvement is needed
    """

    def __init__(self, config: AgentConfig, gold_db: GoldStandardDB):
        self.config = config
        self.gold_db = gold_db
        self._evaluator = dspy.ChainOfThought(ExpertEvaluationSignature)

    async def evaluate(
        self,
        gold_standard_id: str,
        actual_output: Dict[str, Any],
        context: str = ""
    ) -> Evaluation:
        """Evaluate output against gold standard."""
        gold_standard = self.gold_db.get(gold_standard_id)
        if not gold_standard:
            raise ValueError(f"Gold standard not found: {gold_standard_id}")

        try:
            result = self._evaluator(
                gold_standard=json.dumps({
                    'expected_output': gold_standard.expected_output,
                    'criteria': gold_standard.evaluation_criteria
                }),
                actual_output=json.dumps(actual_output),
                context=context
            )

            # Parse scores
            try:
                scores = json.loads(result.scores)
            except:
                scores = {}

            # Parse result enum
            result_enum = EvaluationResult.NEEDS_IMPROVEMENT
            result_str = str(result.result).upper().replace(' ', '_')
            for er in EvaluationResult:
                if er.value.upper() == result_str:
                    result_enum = er
                    break

            # Parse feedback
            feedback = [f.strip() for f in str(result.feedback).split('|') if f.strip()]

            return Evaluation(
                gold_standard_id=gold_standard_id,
                actual_output=actual_output,
                scores=scores,
                overall_score=float(result.overall_score) if result.overall_score else 0.0,
                result=result_enum,
                feedback=feedback
            )

        except Exception as e:
            logger.error(f"Expert evaluation failed: {e}")
            return Evaluation(
                gold_standard_id=gold_standard_id,
                actual_output=actual_output,
                scores={},
                overall_score=0.0,
                result=EvaluationResult.FAILED,
                feedback=[f"Evaluation error: {str(e)}"]
            )


class ReviewerAgent:
    """
    Reviewer Agent - Analyzes patterns and suggests improvements.

    The Reviewer is the strategist. It:
    1. Analyzes multiple evaluations for patterns
    2. Identifies systematic issues
    3. Suggests concrete improvements
    4. Prioritizes based on impact
    """

    def __init__(self, config: AgentConfig, history: ImprovementHistory):
        self.config = config
        self.history = history
        self._analyzer = dspy.ChainOfThought(ReviewerAnalysisSignature)

    async def analyze_and_suggest(
        self,
        evaluations: List[Evaluation],
        agent_configs: Dict[AgentRole, AgentConfig]
    ) -> List[ImprovementSuggestion]:
        """Analyze evaluations and suggest improvements."""
        if not evaluations:
            return []

        try:
            # Get improvement history
            past_improvements = self.history.get_successful_improvements()

            result = self._analyzer(
                evaluations=json.dumps([asdict(e) for e in evaluations], default=str),
                agent_configs=json.dumps({k.value: asdict(v) for k, v in agent_configs.items()}, default=str),
                improvement_history=json.dumps(past_improvements, default=str)
            )

            # Parse suggestions
            try:
                suggestions_data = json.loads(result.suggestions)
            except:
                suggestions_data = []

            suggestions = []
            for s in suggestions_data:
                try:
                    suggestions.append(ImprovementSuggestion(
                        agent_role=AgentRole(s.get('agent_role', 'actor')),
                        improvement_type=ImprovementType(s.get('improvement_type', 'prompt_refinement')),
                        description=s.get('description', ''),
                        priority=int(s.get('priority', 3)),
                        expected_impact=float(s.get('expected_impact', 0.5)),
                        implementation_details=s.get('implementation_details', {}),
                        based_on_evaluations=[e.gold_standard_id for e in evaluations]
                    ))
                except Exception as e:
                    logger.debug(f"Failed to parse suggestion: {e}")

            return suggestions

        except Exception as e:
            logger.error(f"Reviewer analysis failed: {e}")
            return []


class PlannerAgent:
    """
    Planner Agent - Plans task execution with optimization.

    The Planner is the architect. It:
    1. Breaks down tasks into subtasks
    2. Optimizes based on past performance
    3. Identifies parallelization opportunities
    4. Plans error recovery
    """

    def __init__(self, config: AgentConfig, history: ImprovementHistory):
        self.config = config
        self.history = history
        self._optimizer = dspy.ChainOfThought(PlannerOptimizationSignature)

    async def create_plan(
        self,
        task_description: str,
        relevant_improvements: List[ImprovementSuggestion] = None,
        past_plans: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an optimized execution plan."""
        try:
            result = self._optimizer(
                task_description=task_description,
                improvement_suggestions=json.dumps(
                    [asdict(s) for s in (relevant_improvements or [])],
                    default=str
                ),
                past_plans=json.dumps(past_plans or [], default=str)
            )

            # Parse plan
            try:
                plan = json.loads(result.optimized_plan)
            except:
                plan = {'tasks': [{'name': 'execute', 'description': task_description}]}

            # Parse risk mitigations
            risks = [r.strip() for r in str(result.risk_mitigations).split('|') if r.strip()]

            return {
                'plan': plan,
                'rationale': str(result.rationale),
                'risks': risks
            }

        except Exception as e:
            logger.error(f"Planner failed: {e}")
            return {
                'plan': {'tasks': [{'name': 'execute', 'description': task_description}]},
                'rationale': 'Fallback plan due to error',
                'risks': [str(e)]
            }


class ActorAgent:
    """
    Actor Agent - Executes tasks with learned improvements.

    The Actor is the executor. It:
    1. Executes individual tasks
    2. Applies learnings from past feedback
    3. Adapts based on improvement suggestions
    4. Reports confidence levels
    """

    def __init__(self, config: AgentConfig, history: ImprovementHistory):
        self.config = config
        self.history = history
        self._executor = dspy.ChainOfThought(ActorExecutionSignature)

    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        learnings: List[str] = None
    ) -> Tuple[Dict[str, Any], float, List[str]]:
        """Execute a task and return output, confidence, and applied learnings."""
        try:
            # Get relevant successful improvements
            successful = self.history.get_successful_improvements(AgentRole.ACTOR)
            all_learnings = (learnings or []) + [
                s['suggestion']['description'] for s in successful[:5]
            ]

            result = self._executor(
                task=task,
                context=json.dumps(context, default=str),
                learnings="\n".join(all_learnings) if all_learnings else "No prior learnings"
            )

            # Parse output
            try:
                output = json.loads(result.output)
            except:
                output = {'result': str(result.output)}

            confidence = float(result.confidence) if result.confidence else 0.5
            applied = [l.strip() for l in str(result.applied_learnings).split('|') if l.strip()]

            return output, confidence, applied

        except Exception as e:
            logger.error(f"Actor execution failed: {e}")
            return {'error': str(e)}, 0.0, []


# =============================================================================
# BASE SWARM CLASS
# =============================================================================

class BaseSwarm(ABC):
    """
    Base class for all Jotty swarms.

    Provides:
    - Self-improving feedback loop
    - Shared resource management
    - Execution tracking
    - Gold standard evaluation

    Subclasses implement domain-specific logic.
    """

    def __init__(self, config: SwarmConfig):
        self.config = config
        self._initialized = False

        # Shared resources (lazy init)
        self._memory = None
        self._context = None
        self._bus = None
        self._learner = None

        # Self-improvement components
        self._gold_db = None
        self._improvement_history = None
        self._expert = None
        self._reviewer = None
        self._planner = None
        self._actor = None

        # Agent0 curriculum integration (SwarmIntelligence)
        self._swarm_intelligence = None
        self._training_mode = False

        # Execution tracking
        self._traces: List[ExecutionTrace] = []
        self._evaluation_history = EvaluationHistory()

        # Learning lifecycle (populated by _pre_execute_learning)
        self._learned_context: Optional[Dict[str, Any]] = None

    def _init_shared_resources(self):
        """Initialize shared swarm resources."""
        if self._initialized:
            return

        # Auto-configure DSPy if needed
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            try:
                from ..integration.direct_claude_cli_lm import DirectClaudeCLI
                lm = DirectClaudeCLI(model="sonnet")
                dspy.configure(lm=lm)
                logger.info("🔧 Auto-configured DSPy with DirectClaudeCLI")
            except Exception as e:
                logger.warning(f"Could not configure DSPy LM: {e}")

        # Initialize shared resources
        try:
            from .research_swarm import BaseAgent
            from ..agents.dag_agents import SwarmResources
            from ..foundation.data_structures import JottyConfig

            jotty_config = JottyConfig()
            resources = SwarmResources.get_instance(jotty_config)

            self._memory = resources.memory
            self._context = resources.context
            self._bus = resources.bus
            self._learner = resources.learner

            logger.info("✅ Shared swarm resources initialized")
        except Exception as e:
            logger.warning(f"SwarmResources not available: {e}")

        # Initialize self-improvement components
        if self.config.enable_self_improvement:
            self._init_self_improvement()

        self._initialized = True

    def _init_self_improvement(self):
        """Initialize self-improvement loop components."""
        # Gold standard database
        gold_path = self.config.gold_standard_path or str(
            Path.home() / "jotty" / "gold_standards" / self.config.domain
        )
        self._gold_db = GoldStandardDB(gold_path)

        # Improvement history
        history_path = str(Path.home() / "jotty" / "improvements" / self.config.domain)
        self._improvement_history = ImprovementHistory(history_path)

        # Create agent configs
        expert_config = AgentConfig(
            role=AgentRole.EXPERT,
            name=f"{self.config.name}_expert",
            system_prompt="You are an expert evaluator for the {domain} domain."
        )
        reviewer_config = AgentConfig(
            role=AgentRole.REVIEWER,
            name=f"{self.config.name}_reviewer",
            system_prompt="You are a senior reviewer analyzing agent performance."
        )
        planner_config = AgentConfig(
            role=AgentRole.PLANNER,
            name=f"{self.config.name}_planner",
            system_prompt="You are a planning expert optimizing task execution."
        )
        actor_config = AgentConfig(
            role=AgentRole.ACTOR,
            name=f"{self.config.name}_actor",
            system_prompt="You are an expert executor applying learnings."
        )

        # Initialize agents
        self._expert = ExpertAgent(expert_config, self._gold_db)
        self._reviewer = ReviewerAgent(reviewer_config, self._improvement_history)
        self._planner = PlannerAgent(planner_config, self._improvement_history)
        self._actor = ActorAgent(actor_config, self._improvement_history)

        logger.info("✅ Self-improvement loop initialized")

    def _get_intelligence_save_path(self) -> str:
        """Get persistent file path for SwarmIntelligence state."""
        domain = self.config.domain or 'default'
        name = self.config.name or 'base_swarm'
        safe_name = name.replace(' ', '_').replace('/', '_')
        save_dir = Path.home() / "jotty" / "intelligence"
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir / f"{safe_name}_{domain}.json")

    def connect_swarm_intelligence(self, swarm_intelligence=None, enable_training: bool = False):
        """
        Connect to SwarmIntelligence for Agent0 curriculum integration.

        When connected, the swarm will:
        - Auto-load previous learning state from disk
        - Send executor feedback after each task execution
        - Auto-save state after each feedback event
        - Optionally use curriculum-generated training tasks
        - Benefit from tool-aware weakness detection

        Args:
            swarm_intelligence: SwarmIntelligence instance (creates new if None)
            enable_training: Enable curriculum-based training mode
        """
        if swarm_intelligence is None:
            try:
                from ..orchestration.v2.swarm_intelligence import SwarmIntelligence
                swarm_intelligence = SwarmIntelligence()
            except ImportError:
                logger.warning("SwarmIntelligence not available")
                return

        self._swarm_intelligence = swarm_intelligence
        self._training_mode = enable_training

        # Auto-load previous learning state from disk
        save_path = self._get_intelligence_save_path()
        if Path(save_path).exists():
            loaded = self._swarm_intelligence.load(save_path)
            if loaded:
                stats = self._swarm_intelligence.curriculum_generator.get_curriculum_stats()
                logger.info(
                    f"📂 Loaded previous learning: {stats['feedback_count']} feedback events, "
                    f"{len(stats['tool_success_rates'])} tools tracked"
                )

        if enable_training:
            self._swarm_intelligence.enable_training_mode(True, memory_system=self._memory)

        # Register swarm as an agent
        swarm_name = self.config.name or 'base_swarm'
        self._swarm_intelligence.register_agent(swarm_name)

        logger.info(f"✅ SwarmIntelligence connected (training={enable_training})")

    def _send_executor_feedback(
        self,
        task_type: str,
        success: bool,
        tools_used: List[str] = None,
        execution_time: float = 0.0,
        error_type: str = None
    ):
        """
        Send executor feedback to SwarmIntelligence for curriculum adaptation.

        Agent0 closed-loop: Executor feedback → Curriculum adaptation.
        """
        if not self._swarm_intelligence:
            return

        try:
            import time
            swarm_name = self.config.name or 'base_swarm'

            # Record task result for profile building
            self._swarm_intelligence.record_task_result(
                agent_name=swarm_name,
                task_type=task_type,
                success=success,
                execution_time=execution_time
            )

            # Send executor feedback for curriculum
            self._swarm_intelligence.receive_executor_feedback(
                task_id=f"{swarm_name}_{task_type}_{int(time.time())}",
                success=success,
                tools_used=tools_used or [],
                execution_time=execution_time,
                error_type=error_type,
                task_type=task_type
            )

            logger.debug(f"Agent0 feedback sent: {task_type} success={success}")

            # Auto-save learning state to disk after each feedback
            try:
                save_path = self._get_intelligence_save_path()
                self._swarm_intelligence.save(save_path)
                logger.debug(f"Agent0 state saved to {save_path}")
            except Exception as save_err:
                logger.debug(f"Failed to save Agent0 state: {save_err}")

        except Exception as e:
            logger.debug(f"Failed to send Agent0 feedback: {e}")

    def get_training_task(self, tool_aware: bool = True):
        """
        Get a curriculum-generated training task targeting swarm weaknesses.

        Agent0: Returns task designed to improve weak areas.

        Args:
            tool_aware: Use tool-aware task generation

        Returns:
            SyntheticTask or None if training mode disabled
        """
        if not self._swarm_intelligence or not self._training_mode:
            return None

        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.get_training_task(
            target_agent=swarm_name,
            tool_aware=tool_aware
        )

    # =========================================================================
    # LEARNING LIFECYCLE (Agent0 + MorphAgent + Expert Knowledge Stitched)
    # =========================================================================

    async def _pre_execute_learning(self) -> Dict[str, Any]:
        """
        Pre-execution learning hook. Called at start of execute().

        Auto-connects SwarmIntelligence, loads saved state, runs warmup
        on first run, computes MorphAgent scores, analyzes tool performance,
        and stitches Agent0 + MorphAgent findings into a learned context dict.

        Returns:
            Dict with learning context (has_learning, tool_performance,
            agent_scores, weak_tools, recommendations, etc.)
        """
        learned_context = {
            'has_learning': False,
            'tool_performance': {},
            'agent_scores': {},
            'weak_tools': [],
            'strong_tools': [],
            'recommendations': [],
            'warmup_completed': False,
        }

        try:
            # 1. Auto-connect SwarmIntelligence if not connected
            if not self._swarm_intelligence:
                self.connect_swarm_intelligence()

            si = self._swarm_intelligence
            if not si:
                self._learned_context = learned_context
                return learned_context

            # 2. Auto-warmup if first run (no feedback history yet)
            stats = si.curriculum_generator.get_curriculum_stats()
            save_path = self._get_intelligence_save_path()
            if stats['feedback_count'] == 0 and not Path(save_path).exists():
                warmup_result = await self._run_auto_warmup()
                learned_context['warmup_completed'] = True
                logger.info("Auto-warmup complete — seeded initial learning data")

            # 3. Compute MorphAgent scores for all registered agents
            if si.agent_profiles:
                morph_scores = si.morph_scorer.compute_all_scores(si.agent_profiles)
                for agent_name, scores in morph_scores.items():
                    learned_context['agent_scores'][agent_name] = {
                        'rcs': scores.rcs,
                        'rds': scores.rds,
                        'tras': scores.tras,
                        'consistency': scores.rcs_components.get('consistency', 0.5),
                        'focus': scores.rcs_components.get('focus', 0.5),
                        'specialization': scores.rcs_components.get('specialization_clarity', 0.5),
                    }

            # 4. Analyze tool success rates via ToolManager
            tool_analysis = self._manage_tools()
            learned_context['tool_performance'] = stats.get('tool_success_rates', {})
            learned_context['weak_tools'] = tool_analysis.get('weak_tools', [])
            learned_context['strong_tools'] = tool_analysis.get('strong_tools', [])

            # 5. STITCH: Combine weak tool + inconsistent agent = PRIORITY
            recommendations = []
            for weak in learned_context['weak_tools']:
                tool_name = weak['tool']
                rate = weak['success_rate']
                for agent_name, agent_data in learned_context['agent_scores'].items():
                    consistency = agent_data.get('consistency', 0.5)
                    if consistency < 0.5:
                        recommendations.insert(0, {
                            'priority': 'HIGH',
                            'type': 'tool_and_agent',
                            'tool': tool_name,
                            'tool_rate': rate,
                            'agent': agent_name,
                            'consistency': consistency,
                            'action': f"PRIORITY: Replace {tool_name} ({rate:.0%} success) AND "
                                      f"stabilize {agent_name} (consistency={consistency:.2f})"
                        })
                    else:
                        recommendations.append({
                            'priority': 'MEDIUM',
                            'type': 'tool_only',
                            'tool': tool_name,
                            'tool_rate': rate,
                            'action': f"Replace {tool_name} ({rate:.0%} success) — agent {agent_name} is stable"
                        })

            # Add agent-only warnings (consistent tool but inconsistent agent)
            for agent_name, agent_data in learned_context['agent_scores'].items():
                consistency = agent_data.get('consistency', 0.5)
                if consistency < 0.5 and not any(r.get('agent') == agent_name for r in recommendations):
                    recommendations.append({
                        'priority': 'LOW',
                        'type': 'agent_only',
                        'agent': agent_name,
                        'consistency': consistency,
                        'action': f"Warn {agent_name}: outputs inconsistent (consistency={consistency:.2f})"
                    })

            learned_context['recommendations'] = recommendations

            # 6. Retrieve expert domain knowledge from HierarchicalMemory
            expert_knowledge = self._retrieve_expert_knowledge()
            learned_context['expert_knowledge'] = expert_knowledge

            # 7. Analyze prior failures for recovery
            prior_failures = self._analyze_prior_failures()
            learned_context['prior_failures'] = prior_failures

            # 8. Analyze morph score trends (improving vs declining)
            score_trends = {}
            if si and si.morph_score_history and len(si.morph_score_history) >= 2:
                latest = si.morph_score_history[-1].get('scores', {})
                # Compare with 3 runs ago (or earliest available)
                compare_idx = max(0, len(si.morph_score_history) - 4)
                earlier = si.morph_score_history[compare_idx].get('scores', {})
                for agent_name_key in latest:
                    curr_rcs = latest[agent_name_key].get('rcs', 0)
                    prev_rcs = earlier.get(agent_name_key, {}).get('rcs', 0)
                    if prev_rcs > 0:
                        delta = curr_rcs - prev_rcs
                        if abs(delta) > 0.02:  # Only report meaningful changes
                            score_trends[agent_name_key] = {
                                'current': curr_rcs,
                                'previous': prev_rcs,
                                'delta': delta,
                                'direction': 'improving' if delta > 0 else 'declining'
                            }
            learned_context['score_trends'] = score_trends

            learned_context['has_learning'] = bool(
                learned_context['tool_performance'] or
                learned_context['agent_scores'] or
                learned_context['warmup_completed'] or
                learned_context['expert_knowledge'] or
                learned_context['prior_failures'] or
                learned_context['score_trends']
            )

            self._learned_context = learned_context
            if learned_context['has_learning']:
                expert_count = len(learned_context.get('expert_knowledge', []))
                logger.info(
                    f"Pre-execution learning: {len(learned_context['tool_performance'])} tools tracked, "
                    f"{len(learned_context['agent_scores'])} agents scored, "
                    f"{len(recommendations)} recommendations, "
                    f"{expert_count} expert patterns loaded"
                )

        except Exception as e:
            logger.debug(f"Pre-execution learning skipped: {e}")
            self._learned_context = learned_context

        return learned_context

    async def _run_auto_warmup(self, num_episodes: int = 3) -> Dict:
        """
        Lightweight bootstrap on first ever run.

        Generates curriculum tasks and seeds tool_success_rates + agent profiles
        with conservative initial data (60% success baseline). This is NOT real
        execution — just curriculum bootstrapping to avoid cold-start.

        Args:
            num_episodes: Number of seed episodes to generate

        Returns:
            Dict with warmup statistics
        """
        si = self._swarm_intelligence
        if not si:
            return {'seeded': 0}

        swarm_name = self.config.name or 'base_swarm'
        si.register_agent(swarm_name)

        seeded = 0
        import time as _time

        for i in range(num_episodes):
            # Generate a curriculum task
            task = si.curriculum_generator.generate_training_task(
                profiles=si.agent_profiles,
                target_agent=swarm_name
            )

            # Simulate conservative results (60% success rate)
            simulated_success = (i % 3) != 2  # 2 out of 3 succeed
            simulated_tools = [f"tool_{task.task_type}"]
            simulated_time = 5.0 + (i * 2.0)

            # Send as executor feedback to seed the rates
            si.receive_executor_feedback(
                task_id=f"warmup_{swarm_name}_{i}_{int(_time.time())}",
                success=simulated_success,
                tools_used=simulated_tools,
                execution_time=simulated_time,
                task_type=task.task_type
            )

            # Record task result for agent profile seeding
            si.record_task_result(
                agent_name=swarm_name,
                task_type=task.task_type,
                success=simulated_success,
                execution_time=simulated_time
            )

            seeded += 1

        # Save seeded state
        try:
            save_path = self._get_intelligence_save_path()
            si.save(save_path)
        except Exception as e:
            logger.debug(f"Failed to save warmup state: {e}")

        logger.info(f"Auto-warmup: seeded {seeded} episodes for {swarm_name}")
        return {'seeded': seeded, 'swarm': swarm_name}

    def _build_learned_context_string(self, agent_name: str = None) -> str:
        """
        Convert self._learned_context into injectable prompt text.

        Produces a compact string suitable for appending to DSPy agent inputs,
        so agents are aware of prior tool performance, agent consistency, and
        priority actions.

        Args:
            agent_name: Optional specific agent to tailor context for

        Returns:
            String like:
            '## Prior Learning
            Tool Performance: arxiv_fetch 100% RELIABLE, content_generate 45% WEAK
            Agent Notes: ContentPolisher has inconsistent outputs (consistency=0.3)
            Action: Validate content_generate output carefully before using.'
        """
        if not self._learned_context or not self._learned_context.get('has_learning'):
            return ""

        ctx = self._learned_context
        lines = ["## Prior Learning"]

        # Tool performance summary
        tool_parts = []
        for tool_info in ctx.get('strong_tools', []):
            tool_parts.append(f"{tool_info['tool']} {tool_info['success_rate']:.0%} RELIABLE")
        for tool_info in ctx.get('weak_tools', []):
            tool_parts.append(f"{tool_info['tool']} {tool_info['success_rate']:.0%} WEAK")

        if tool_parts:
            lines.append(f"Tool Performance: {', '.join(tool_parts)}")

        # Agent-specific context (both positive reinforcement and warnings)
        agent_notes = []
        scores = ctx.get('agent_scores', {})
        if agent_name and agent_name in scores:
            agent_data = scores[agent_name]
            rcs = agent_data.get('rcs', 0)
            consistency = agent_data.get('consistency', 0.5)
            focus = agent_data.get('focus', 0.5)
            # Tiered competence feedback — always push for higher
            if rcs >= 0.85:
                agent_notes.append(
                    f"Competence {rcs:.2f} — excellent, maintain this standard"
                )
            elif rcs >= 0.6:
                agent_notes.append(
                    f"Competence {rcs:.2f} — good but target >0.85, push harder on quality"
                )
            elif rcs >= 0.4:
                agent_notes.append(
                    f"Competence {rcs:.2f} — needs improvement, aim for >0.6"
                )
            elif rcs > 0:
                agent_notes.append(
                    f"Competence {rcs:.2f} — critical, significant quality issues"
                )
            # Focus feedback
            if focus >= 0.85:
                agent_notes.append("Focus is excellent — stay specialized")
            elif focus >= 0.6:
                agent_notes.append(f"Focus {focus:.2f} — good but tighten specialization")
            elif focus > 0 and focus < 0.4:
                agent_notes.append(f"Focus {focus:.2f} — too scattered, narrow your scope")
            # Consistency warnings
            if consistency < 0.5:
                agent_notes.append(
                    f"Consistency {consistency:.2f} — outputs vary too much, "
                    f"be extra careful with accuracy"
                )
        elif scores:
            # Summary for orchestrator or unmatched agents
            high_performers = []
            needs_improvement = []
            low_performers = []
            for name, agent_data in scores.items():
                rcs = agent_data.get('rcs', 0)
                consistency = agent_data.get('consistency', 0.5)
                if rcs >= 0.85:
                    high_performers.append(name)
                elif rcs < 0.5 and rcs > 0:
                    needs_improvement.append(f"{name}({rcs:.2f})")
                if consistency < 0.5:
                    low_performers.append(
                        f"{name} inconsistent ({consistency:.2f})"
                    )
            if high_performers:
                agent_notes.append(f"Strong agents: {', '.join(high_performers)}")
            if needs_improvement:
                agent_notes.append(f"Need improvement: {', '.join(needs_improvement)}")
            if low_performers:
                agent_notes.extend(low_performers)

        # Specialization label from AgentProfile
        si = self._swarm_intelligence
        if si and agent_name and agent_name in getattr(si, 'agent_profiles', {}):
            from ..orchestration.v2.swarm_intelligence import AgentSpecialization
            profile = si.agent_profiles[agent_name]
            spec = profile.specialization
            if spec != AgentSpecialization.GENERALIST:
                agent_notes.append(f"Specialization: {spec.value} — leverage this strength")

            # Per-agent time budget from profile
            if profile.avg_execution_time > 0 and profile.total_tasks >= 2:
                avg_t = profile.avg_execution_time
                agent_notes.append(f"Avg execution: {avg_t:.0f}s over {profile.total_tasks} tasks")

        if agent_notes:
            lines.append(f"Agent Notes: {'; '.join(agent_notes)}")

        # Execution patterns from collective memory (what works, typical timings)
        if si and si.collective_memory:
            recent = si.collective_memory[-20:]
            successes = [m for m in recent if m.get('success')]
            if successes:
                # Build timing expectations per task type
                from collections import defaultdict
                task_times = defaultdict(list)
                for m in successes:
                    tt = m.get('task_type', '')
                    if tt and m.get('execution_time', 0) > 0:
                        task_times[tt].append(m['execution_time'])
                if task_times:
                    timing_parts = []
                    for tt, times in task_times.items():
                        avg = sum(times) / len(times)
                        timing_parts.append(f"{tt}~{avg:.0f}s")
                    if len(timing_parts) <= 6:
                        lines.append(f"Typical timing: {', '.join(timing_parts)}")
                # Success streak info
                total_recent = len(recent)
                success_rate = len(successes) / total_recent if total_recent else 0
                if success_rate >= 0.9 and total_recent >= 5:
                    lines.append(
                        f"Track record: {len(successes)}/{total_recent} recent successes — "
                        f"maintain this standard"
                    )

        # Priority recommendations (HIGH first)
        recommendations = ctx.get('recommendations', [])
        high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
        if high_priority:
            actions = [r['action'] for r in high_priority[:3]]
            lines.append(f"Action: {'; '.join(actions)}")
        elif recommendations:
            lines.append(f"Action: {recommendations[0]['action']}")

        # Evaluation quality bar from persistent history
        if hasattr(self, '_evaluation_history'):
            avg_score = self._evaluation_history.get_average_score(5)
            eval_count = len(self._evaluation_history.evaluations)
            if eval_count >= 2 and avg_score > 0:
                if avg_score >= 0.9:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"excellent standard, don't regress"
                    )
                elif avg_score >= 0.7:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"good but push for higher"
                    )
                else:
                    lines.append(
                        f"Quality bar: avg {avg_score:.0%} over {eval_count} evaluations — "
                        f"needs significant improvement"
                    )

        # Improvement suggestions from prior cycles (what to improve + what worked)
        if hasattr(self, '_improvement_history') and self._improvement_history:
            pending = self._improvement_history.get_pending_suggestions()
            successful = self._improvement_history.get_successful_improvements()
            if pending or successful:
                imp_lines = []
                # Show successful improvements so agents know what works
                for s in successful[-3:]:
                    suggestion = s.get('suggestion', {})
                    desc = suggestion.get('description', '')
                    if desc:
                        imp_lines.append(f"- Applied successfully: {desc[:120]}")
                # Show pending improvements as directives
                for p in pending[-3:]:
                    suggestion = p.get('suggestion', {})
                    desc = suggestion.get('description', '')
                    priority = suggestion.get('priority', 'MEDIUM')
                    target_agent = suggestion.get('agent_role', '')
                    # Only show agent-specific improvements to that agent
                    if agent_name and target_agent and target_agent != agent_name:
                        continue
                    if desc:
                        imp_lines.append(f"- [{priority}] TODO: {desc[:120]}")
                if imp_lines:
                    lines.append("## Improvement Directives")
                    lines.extend(imp_lines)

        # Expert domain knowledge from HierarchicalMemory
        expert_knowledge = ctx.get('expert_knowledge', [])
        if expert_knowledge:
            expert_lines = []
            for imp in expert_knowledge[:5]:  # Top 5 patterns
                pattern = imp.get('learned_pattern', '')
                if pattern:
                    # Truncate long patterns for prompt efficiency
                    if len(pattern) > 150:
                        pattern = pattern[:147] + "..."
                    expert_lines.append(f"- {pattern}")

            if expert_lines:
                lines.append("## Expert Knowledge")
                lines.extend(expert_lines)

        # Failure recovery from prior runs
        prior_failures = ctx.get('prior_failures', [])
        if prior_failures:
            failure_lines = ["## Prior Failures (Avoid Repeating)"]
            for f in prior_failures[:3]:
                if f.get('source') == 'evaluation':
                    feedback = f.get('feedback', '')
                    if feedback:
                        failure_lines.append(f"- Previous run scored {f.get('score', 0):.0%}: {feedback[:100]}")
                elif f.get('source') == 'collective_memory':
                    failure_lines.append(
                        f"- Agent {f.get('agent', '?')} failed task {f.get('task_type', '?')}"
                    )
                elif f.get('source') == 'memory':
                    failure_lines.append(f"- {f.get('pattern', 'unknown failure')}")
            if len(failure_lines) > 1:
                lines.extend(failure_lines)

        # Morph score trends — show improvement/decline direction
        score_trends = ctx.get('score_trends', {})
        if score_trends:
            trend_lines = []
            for trend_agent, trend_data in score_trends.items():
                # Show trend for this specific agent or all agents for orchestrator
                if agent_name and trend_agent != agent_name:
                    continue
                delta = trend_data['delta']
                direction = trend_data['direction']
                current = trend_data['current']
                if direction == 'improving':
                    trend_lines.append(
                        f"{trend_agent}: {current:.2f} (+{delta:.2f}) improving — keep pushing"
                    )
                else:
                    trend_lines.append(
                        f"{trend_agent}: {current:.2f} ({delta:.2f}) declining — investigate and fix"
                    )
            if not agent_name:
                # Orchestrator sees all trends
                for trend_agent, trend_data in score_trends.items():
                    delta = trend_data['delta']
                    current = trend_data['current']
                    direction = trend_data['direction']
                    if direction == 'improving':
                        trend_lines.append(
                            f"{trend_agent}: {current:.2f} (+{delta:.2f}) improving"
                        )
                    else:
                        trend_lines.append(
                            f"{trend_agent}: {current:.2f} ({delta:.2f}) DECLINING"
                        )
            if trend_lines:
                lines.append("Score trends: " + "; ".join(trend_lines))

        return "\n".join(lines) if len(lines) > 1 else ""

    @classmethod
    def test_learning_pathways(cls) -> Dict[str, Dict[str, Any]]:
        """Diagnostic: inject synthetic data to verify all 5 learning pathways produce prompt text."""
        import tempfile
        results = {}

        # Create concrete subclass to bypass ABC restriction
        class _TestSwarm(cls):
            async def execute(self, *args, **kwargs):
                pass

        # Create minimal swarm instance for testing (no disk I/O beyond tempdir)
        config = SwarmConfig(name='pathway_tester', enable_self_improvement=True)
        instance = _TestSwarm.__new__(_TestSwarm)
        instance.config = config
        instance._swarm_intelligence = None
        instance._memory = None
        instance._learned_context = None
        tmp = tempfile.mkdtemp()
        instance._evaluation_history = EvaluationHistory(path=tmp + '/eval')
        instance._improvement_history = ImprovementHistory(path=tmp + '/imp')

        # === Pathway 1: weak_tools ===
        instance._learned_context = {
            'has_learning': True,
            'tool_performance': {'bad_tool': 0.3},
            'agent_scores': {},
            'weak_tools': [{'tool': 'bad_tool', 'success_rate': 0.3, 'total': 5}],
            'strong_tools': [{'tool': 'good_tool', 'success_rate': 0.95, 'total': 10}],
            'recommendations': [],
            'warmup_completed': True,
            'expert_knowledge': [],
            'prior_failures': [],
            'score_trends': {},
        }
        text = instance._build_learned_context_string()
        results['weak_tools'] = {
            'triggered': 'WEAK' in text,
            'prompt_snippet': text[:200] if text else '(empty)',
        }

        # === Pathway 2: expert_knowledge ===
        instance._learned_context['expert_knowledge'] = [
            {'learned_pattern': 'Always validate API responses before processing'},
            {'learned_pattern': 'Use batch processing for >100 items'},
        ]
        text = instance._build_learned_context_string()
        results['expert_knowledge'] = {
            'triggered': 'Expert Knowledge' in text,
            'prompt_snippet': text[text.find('Expert'):text.find('Expert') + 150] if 'Expert' in text else '(empty)',
        }

        # === Pathway 3: prior_failures ===
        instance._learned_context['prior_failures'] = [
            {'source': 'evaluation', 'score': 0.3, 'feedback': 'Missing key concepts', 'timestamp': datetime.now().isoformat()},
            {'source': 'collective_memory', 'agent': 'ConceptExtractor', 'task_type': 'expert', 'timestamp': datetime.now().isoformat()},
        ]
        text = instance._build_learned_context_string()
        results['prior_failures'] = {
            'triggered': 'Prior Failures' in text,
            'prompt_snippet': text[text.find('Prior Failures'):text.find('Prior Failures') + 200] if 'Prior Failures' in text else '(empty)',
        }

        # === Pathway 4: improvement_directives ===
        # ImprovementHistory uses self.history (list of dicts)
        # get_pending_suggestions() checks status == 'pending'
        # get_successful_improvements() checks outcome == 'success'
        instance._improvement_history.history = [
            {
                'id': 'test_pending_1',
                'suggestion': {'description': 'Improve concept extraction depth', 'priority': 5, 'agent_role': ''},
                'status': 'pending',
                'outcome': None,
            },
            {
                'id': 'test_success_1',
                'suggestion': {'description': 'Use more examples in explanations', 'priority': 3, 'agent_role': ''},
                'status': 'completed',
                'outcome': 'success',
            },
        ]
        text = instance._build_learned_context_string()
        results['improvement_directives'] = {
            'triggered': 'Improvement Directives' in text,
            'prompt_snippet': text[text.find('Improvement'):text.find('Improvement') + 200] if 'Improvement' in text else '(empty)',
        }

        # === Pathway 5: recommendations ===
        instance._learned_context['recommendations'] = [
            {'priority': 'HIGH', 'type': 'tool_and_agent', 'tool': 'bad_tool', 'tool_rate': 0.3,
             'agent': 'SlowAgent', 'consistency': 0.3,
             'action': 'PRIORITY: Replace bad_tool (30% success) AND stabilize SlowAgent (consistency=0.30)'}
        ]
        text = instance._build_learned_context_string()
        results['recommendations'] = {
            'triggered': 'Action:' in text and 'PRIORITY' in text,
            'prompt_snippet': text[text.find('Action:'):text.find('Action:') + 150] if 'Action:' in text else '(empty)',
        }

        # Summary
        all_passed = all(r['triggered'] for r in results.values())
        results['_summary'] = {
            'total': 5,
            'passed': sum(1 for r in results.values() if isinstance(r, dict) and r.get('triggered')),
            'all_passed': all_passed,
        }

        return results

    def _manage_tools(self) -> Dict:
        """
        Analyze tool performance and log warnings for weak tools.

        Delegates to SwarmIntelligence.tool_manager.analyze_tools() using
        the curriculum generator's tracked tool success rates.

        Returns:
            Dict with weak_tools, strong_tools, suggested_removals, replacements
        """
        if not self._swarm_intelligence:
            return {'weak_tools': [], 'strong_tools': [], 'suggested_removals': [], 'replacements': {}}

        si = self._swarm_intelligence
        swarm_name = self.config.name or 'base_swarm'
        tool_rates = si.curriculum_generator._tool_success_rates

        analysis = si.tool_manager.analyze_tools(tool_rates, swarm_name)

        # Log warnings for weak tools
        for weak in analysis.get('weak_tools', []):
            logger.warning(
                f"Weak tool detected: {weak['tool']} "
                f"({weak['success_rate']:.0%} success over {weak['total']} uses)"
            )

        # After logging warnings, ACTUALLY update tool assignments
        replacements = analysis.get('replacements', {})
        removals = analysis.get('suggested_removals', [])

        if replacements or removals:
            add_tools = []
            remove_tools = []

            for weak_tool, replacement_list in replacements.items():
                if replacement_list:
                    best = replacement_list[0]  # First replacement
                    add_tools.append(best['name'])
                    remove_tools.append(weak_tool)
                    logger.info(
                        f"Tool swap: {weak_tool} -> {best['name']} "
                        f"(reason: {best.get('reason', 'low success rate')})"
                    )

            for tool_name in removals:
                if tool_name not in remove_tools:
                    remove_tools.append(tool_name)

            if add_tools or remove_tools:
                si.tool_manager.update_assignments(
                    swarm_name,
                    add=add_tools,
                    remove=remove_tools
                )
                logger.info(f"Tool assignments updated: +{add_tools} -{remove_tools}")

        return analysis

    def _get_active_tools(self, default_tools: List[str] = None) -> List[str]:
        """
        Get dynamic tool list: defaults + additions - deactivated.

        Delegates to SwarmIntelligence.tool_manager.get_active_tools().

        Args:
            default_tools: Default tool names for this swarm

        Returns:
            List of active tool names
        """
        if not self._swarm_intelligence:
            return default_tools or []

        swarm_name = self.config.name or 'base_swarm'
        return self._swarm_intelligence.tool_manager.get_active_tools(
            swarm_name, default_tools or []
        )

    def _retrieve_expert_knowledge(self) -> List[Dict[str, Any]]:
        """
        Retrieve expert-learned domain patterns from HierarchicalMemory.

        Queries HierarchicalMemory for improvements stored by BaseExpert agents
        (via memory_integration.py). Returns patterns relevant to this swarm's
        domain so they can be injected into DSPy agent prompts.

        Returns:
            List of expert improvement dicts with 'learned_pattern', 'domain',
            'source', etc. Empty list if memory unavailable.
        """
        if not self._memory:
            # Try initializing shared resources to get memory
            if not self._initialized:
                self._init_shared_resources()
            if not self._memory:
                return []

        domain = getattr(self.config, 'domain', None) or self.config.name or 'general'
        swarm_name = self.config.name or 'base_swarm'

        try:
            from ..foundation.data_structures import MemoryLevel

            # Query for expert improvements at all levels (PROCEDURAL, META, SEMANTIC)
            memory_entries = self._memory.retrieve(
                query=f"expert agent improvements domain {domain} patterns best practices",
                goal=f"expert_{domain}_improvements",
                budget_tokens=5000,
                levels=[MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC]
            )

            if not memory_entries:
                # Also try broader search without domain filter
                memory_entries = self._memory.retrieve(
                    query=f"expert improvements learned patterns {swarm_name}",
                    goal=f"expert_{domain}_improvements",
                    budget_tokens=3000,
                    levels=[MemoryLevel.SEMANTIC, MemoryLevel.META]
                )

            improvements = []
            for entry in memory_entries[:10]:  # Cap at 10 patterns
                try:
                    improvement_data = json.loads(entry.content)
                    if isinstance(improvement_data, dict):
                        improvements.append(improvement_data)
                    elif isinstance(improvement_data, list):
                        improvements.extend(improvement_data[:5])
                except (json.JSONDecodeError, TypeError):
                    # Raw text pattern from consolidation
                    if entry.content and len(entry.content) > 10:
                        improvements.append({
                            'learned_pattern': entry.content,
                            'domain': domain,
                            'source': 'expert_memory',
                            'memory_level': entry.level.value if hasattr(entry, 'level') else 'unknown',
                        })

            if improvements:
                logger.info(f"Retrieved {len(improvements)} expert patterns from memory for domain '{domain}'")

            return improvements

        except Exception as e:
            logger.debug(f"Expert knowledge retrieval skipped: {e}")
            return []

    def _analyze_prior_failures(self) -> List[Dict[str, Any]]:
        """Analyze prior execution failures from collective_memory and evaluation history.
        Returns list of failure patterns with avoidance suggestions."""
        failures = []

        # Source 1: Evaluation history failures
        if hasattr(self, '_evaluation_history'):
            eval_failures = self._evaluation_history.get_failures(20)
            for f in eval_failures[-5:]:  # Last 5 failures
                failures.append({
                    'source': 'evaluation',
                    'score': f.get('overall_score', 0),
                    'feedback': f.get('feedback', ''),
                    'timestamp': f.get('timestamp', ''),
                })

        # Source 2: Collective memory from SwarmIntelligence
        si = self._swarm_intelligence
        if si and si.collective_memory:
            failed_tasks = [
                m for m in si.collective_memory[-50:]
                if not m.get('success', True)
            ]
            for m in failed_tasks[-5:]:
                failures.append({
                    'source': 'collective_memory',
                    'agent': m.get('agent', 'unknown'),
                    'task_type': m.get('task_type', 'unknown'),
                    'timestamp': m.get('timestamp', ''),
                })

        # Source 3: Execution traces stored in memory
        if self._memory:
            try:
                from ..foundation.data_structures import MemoryLevel
                failure_entries = self._memory.retrieve(
                    query=f"failed execution error {self.config.name or 'swarm'}",
                    goal="failure_analysis",
                    budget_tokens=2000,
                    levels=[MemoryLevel.META]
                )
                for entry in failure_entries[:3]:
                    try:
                        data = json.loads(entry.content)
                        if isinstance(data, dict) and not data.get('success', True):
                            failures.append({
                                'source': 'memory',
                                'pattern': data.get('learned_pattern', entry.content[:100]),
                            })
                    except (json.JSONDecodeError, TypeError):
                        pass
            except Exception:
                pass

        return failures

    def _store_execution_as_improvement(
        self,
        success: bool,
        execution_time: float,
        tools_used: List[str],
        task_type: str
    ):
        """
        Store execution outcome as an expert improvement in HierarchicalMemory.

        This bridges swarm execution results back into the expert memory system,
        so future expert training and swarm executions can learn from outcomes.

        Args:
            success: Whether execution succeeded
            execution_time: Total execution time in seconds
            tools_used: List of tool names used
            task_type: Type of task executed
        """
        if not self._memory:
            return

        domain = getattr(self.config, 'domain', None) or self.config.name or 'general'
        swarm_name = self.config.name or 'base_swarm'

        try:
            from ..foundation.data_structures import MemoryLevel

            # Build improvement from execution outcome
            if success:
                pattern = (
                    f"Successful {task_type} execution by {swarm_name}: "
                    f"tools [{', '.join(tools_used)}] completed in {execution_time:.1f}s"
                )
                level = MemoryLevel.PROCEDURAL
            else:
                pattern = (
                    f"Failed {task_type} execution by {swarm_name}: "
                    f"tools [{', '.join(tools_used)}] failed after {execution_time:.1f}s — "
                    f"consider alternative approach or tool substitution"
                )
                level = MemoryLevel.META  # Failures are learning wisdom

            improvement = {
                'timestamp': datetime.now().isoformat(),
                'task': task_type,
                'learned_pattern': pattern,
                'improvement_type': 'execution_outcome',
                'source': f'swarm_{swarm_name}',
                'success': success,
                'execution_time': execution_time,
                'tools_used': tools_used,
            }

            context = {
                'expert_name': swarm_name,
                'domain': domain,
                'task': task_type,
                'improvement_type': 'execution_outcome',
                'source': 'swarm_lifecycle',
            }

            self._memory.store(
                content=json.dumps(improvement, ensure_ascii=False),
                level=level,
                context=context,
                goal=f"expert_{domain}_improvements",
                initial_value=0.8 if success else 1.0,  # Failures are more valuable for learning
            )

            logger.debug(f"Stored execution outcome to expert memory: {task_type} {'success' if success else 'failure'}")

        except Exception as e:
            logger.debug(f"Failed to store execution improvement: {e}")

    async def _post_execute_learning(
        self,
        success: bool,
        execution_time: float,
        tools_used: List[str],
        task_type: str,
        output_data: Dict[str, Any] = None,
        input_data: Dict[str, Any] = None
    ):
        """
        Post-execution learning hook. Called at end of execute().

        Sends executor feedback, recomputes MorphAgent scores, re-analyzes
        tools, evaluates output, runs improvement cycle, and saves all state.

        Args:
            success: Whether execution succeeded
            execution_time: Total execution time in seconds
            tools_used: List of tool names used during execution
            task_type: Type of task that was executed
            output_data: Optional dict of output metrics for evaluation
            input_data: Optional dict of input params for evaluation matching
        """
        try:
            # 1. Send executor feedback (tools, success, timing)
            self._send_executor_feedback(
                task_type=task_type,
                success=success,
                tools_used=tools_used,
                execution_time=execution_time,
                error_type=None if success else 'execution_failure'
            )

            si = self._swarm_intelligence
            if not si:
                return

            # 2. Recompute MorphAgent scores with new data
            if si.agent_profiles:
                morph_scores = si.morph_scorer.compute_all_scores(si.agent_profiles)
                si.morph_score_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'scores': {
                        name: {'rcs': s.rcs, 'rds': s.rds, 'tras': s.tras}
                        for name, s in morph_scores.items()
                    }
                })
                # Bound history
                if len(si.morph_score_history) > 50:
                    si.morph_score_history = si.morph_score_history[-50:]

            # 3. Re-analyze tools and update assignments
            self._manage_tools()

            # 4. Evaluate output against gold standard (centralized for all swarms)
            evaluation = None
            if success and output_data and self.config.enable_self_improvement:
                try:
                    evaluation = await self._evaluate_output(
                        output=output_data,
                        task_type=task_type,
                        input_data=input_data or {}
                    )
                    if evaluation:
                        logger.info(
                            f"Evaluation: {evaluation.result.value} "
                            f"(score: {evaluation.overall_score:.2f})"
                        )
                except Exception as eval_err:
                    logger.debug(f"Evaluation skipped: {eval_err}")

            # 5. Run improvement cycle if evaluation below threshold
            if evaluation and evaluation.overall_score < self.config.improvement_threshold:
                try:
                    suggestions = await self._run_improvement_cycle()
                    if suggestions:
                        logger.info(f"Generated {len(suggestions)} improvement suggestions")
                except Exception as imp_err:
                    logger.debug(f"Improvement cycle skipped: {imp_err}")

            # 6. Save state to disk
            try:
                save_path = self._get_intelligence_save_path()
                si.save(save_path)
                logger.debug(f"Post-execution learning state saved to {save_path}")
            except Exception as save_err:
                logger.debug(f"Failed to save post-execution state: {save_err}")

            # 7. Store execution outcome as expert improvement in HierarchicalMemory
            self._store_execution_as_improvement(
                success=success,
                execution_time=execution_time,
                tools_used=tools_used,
                task_type=task_type
            )

        except Exception as e:
            logger.debug(f"Post-execution learning skipped: {e}")

    async def _evaluate_output(
        self,
        output: Dict[str, Any],
        task_type: str,
        input_data: Dict[str, Any]
    ) -> Optional[Evaluation]:
        """Evaluate output against gold standard if available."""
        if not self.config.enable_self_improvement or not self._expert:
            return None

        # Find matching gold standard
        gold_standard = self._gold_db.find_similar(task_type, input_data)
        if not gold_standard:
            logger.debug(f"No gold standard found for task_type: {task_type}")
            return None

        evaluation = await self._expert.evaluate(
            gold_standard_id=gold_standard.id,
            actual_output=output,
            context=json.dumps({'task_type': task_type, 'input': input_data})
        )

        self._evaluation_history.record(evaluation)
        return evaluation

    async def _run_improvement_cycle(self) -> List[ImprovementSuggestion]:
        """Run the self-improvement cycle."""
        if not self.config.enable_self_improvement or not self._reviewer:
            return []

        # Check if improvement is needed (persistent across sessions)
        recent_evals = self._evaluation_history.get_recent(10)
        avg_score = self._evaluation_history.get_average_score(10)
        if not recent_evals or avg_score >= self.config.improvement_threshold:
            logger.info(f"Performance good ({avg_score:.2f}), skipping improvement cycle")
            return []

        # Get suggestions from reviewer
        agent_configs = {
            AgentRole.EXPERT: self._expert.config if self._expert else None,
            AgentRole.REVIEWER: self._reviewer.config if self._reviewer else None,
            AgentRole.PLANNER: self._planner.config if self._planner else None,
            AgentRole.ACTOR: self._actor.config if self._actor else None,
        }
        agent_configs = {k: v for k, v in agent_configs.items() if v}

        suggestions = await self._reviewer.analyze_and_suggest(
            recent_evals,
            agent_configs
        )

        # Record suggestions
        for suggestion in suggestions:
            self._improvement_history.record_suggestion(suggestion)

        return suggestions

    def _record_trace(
        self,
        agent_name: str,
        agent_role: AgentRole,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        success: bool,
        error: Optional[str] = None,
        tools_used: List[str] = None
    ):
        """Record execution trace for learning and Agent0 feedback."""
        trace = ExecutionTrace(
            agent_name=agent_name,
            agent_role=agent_role,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            success=success,
            error=error
        )
        self._traces.append(trace)

        # Agent0: Automatically send feedback when trace is recorded
        task_type = agent_role.value if agent_role else 'unknown'
        self._send_executor_feedback(
            task_type=task_type,
            success=success,
            tools_used=tools_used or [],
            execution_time=execution_time,
            error_type=type(error).__name__ if error and not isinstance(error, str) else None
        )

        # MorphAgent: Update agent profile for per-agent tracking
        swarm_name = self.config.name or 'base_swarm'
        if self._swarm_intelligence and hasattr(self._swarm_intelligence, 'agent_profiles'):
            self._swarm_intelligence.register_agent(agent_name)
            # Record task result under individual agent name (not swarm name)
            # so per-agent profiles accumulate real task_success data
            if agent_name != swarm_name:
                task_type_label = agent_role.value if agent_role else 'unknown'
                self._swarm_intelligence.record_task_result(
                    agent_name=agent_name,
                    task_type=task_type_label,
                    success=success,
                    execution_time=execution_time
                )

        # Store in memory for learning
        if self._memory and self.config.enable_learning:
            try:
                from ..foundation.data_structures import MemoryLevel
                self._memory.store(
                    content=json.dumps(asdict(trace), default=str),
                    level=MemoryLevel.EPISODIC,
                    context={'swarm': self.config.name, 'agent': agent_name},
                    goal=f"Execution trace: {agent_name}"
                )
            except Exception as e:
                logger.debug(f"Failed to store trace in memory: {e}")

    def _agent_context(self, agent_name: str) -> str:
        """Build per-agent learned context string.
        Convenience wrapper for subclasses to use in _init_agents()."""
        if not self._learned_context:
            return ""
        return self._build_learned_context_string(agent_name=agent_name)

    def _trace_phase(
        self,
        agent_name: str,
        agent_role: AgentRole,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        success: bool,
        phase_start: datetime,
        tools_used: List[str] = None
    ):
        """Record a phase trace with automatic timing.
        Convenience wrapper for subclasses to call after each execution phase."""
        elapsed = (datetime.now() - phase_start).total_seconds()
        self._record_trace(
            agent_name=agent_name,
            agent_role=agent_role,
            input_data=input_data,
            output_data=output_data,
            execution_time=elapsed,
            success=success,
            tools_used=tools_used or []
        )

    @abstractmethod
    async def execute(self, *args, **kwargs) -> SwarmResult:
        """Execute the swarm's main task. Implemented by subclasses."""
        pass

    def add_gold_standard(
        self,
        task_type: str,
        input_data: Dict[str, Any],
        expected_output: Dict[str, Any],
        evaluation_criteria: Dict[str, float]
    ) -> str:
        """Add a gold standard for evaluation."""
        if not self._gold_db:
            self._init_shared_resources()

        gold_standard = GoldStandard(
            id="",  # Will be generated
            domain=self.config.domain,
            task_type=task_type,
            input_data=input_data,
            expected_output=expected_output,
            evaluation_criteria=evaluation_criteria
        )

        return self._gold_db.add(gold_standard)

    def get_improvement_suggestions(self) -> List[Dict]:
        """Get pending improvement suggestions."""
        if not self._improvement_history:
            return []
        return self._improvement_history.get_pending_suggestions()

    def apply_improvement(self, suggestion_id: str):
        """Mark an improvement as applied."""
        if self._improvement_history:
            self._improvement_history.mark_applied(suggestion_id)

    def record_improvement_outcome(
        self,
        suggestion_id: str,
        success: bool,
        impact: float,
        notes: str = ""
    ):
        """Record the outcome of an applied improvement."""
        if self._improvement_history:
            self._improvement_history.record_outcome(suggestion_id, success, impact, notes)


# =============================================================================
# SWARM REGISTRY
# =============================================================================

class SwarmRegistry:
    """Registry for all available swarms."""

    _swarms: Dict[str, Type[BaseSwarm]] = {}

    @classmethod
    def register(cls, name: str, swarm_class: Type[BaseSwarm]):
        """Register a swarm class."""
        cls._swarms[name] = swarm_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseSwarm]]:
        """Get a swarm class by name."""
        return cls._swarms.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered swarms."""
        return list(cls._swarms.keys())

    @classmethod
    def create(cls, name: str, config: SwarmConfig = None) -> Optional[BaseSwarm]:
        """Create a swarm instance by name."""
        swarm_class = cls.get(name)
        if not swarm_class:
            return None

        if config is None:
            config = SwarmConfig(name=name, domain=name)

        return swarm_class(config)


# =============================================================================
# CONVENIENCE DECORATOR
# =============================================================================

def register_swarm(name: str):
    """Decorator to register a swarm class."""
    def decorator(cls):
        SwarmRegistry.register(name, cls)
        return cls
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'AgentRole',
    'EvaluationResult',
    'ImprovementType',

    # Data classes
    'GoldStandard',
    'Evaluation',
    'ImprovementSuggestion',
    'AgentConfig',
    'ExecutionTrace',
    'SwarmConfig',
    'SwarmResult',

    # DSPy Signatures
    'ExpertEvaluationSignature',
    'ReviewerAnalysisSignature',
    'PlannerOptimizationSignature',
    'ActorExecutionSignature',

    # Core classes
    'GoldStandardDB',
    'ImprovementHistory',
    'ExpertAgent',
    'ReviewerAgent',
    'PlannerAgent',
    'ActorAgent',
    'BaseSwarm',
    'SwarmRegistry',
    'register_swarm',
]
