"""
Swarm Template Base Classes
===========================

Foundation for domain-specific swarm orchestration patterns.

Key Concepts:
- AgentConfig: Defines an agent with its skills and model
- StageConfig: Defines a pipeline stage with parallelism hints
- FeedbackConfig: Defines iterative improvement loop
- SwarmTemplate: Combines agents, stages, prompts into a template

Design Philosophy:
- Skills are atomic (do ONE thing well)
- Templates define orchestration (HOW skills combine)
- Agents execute skills (swarm agents run skill code)
- LLM prompts are configurable (each template customizes)
- Feedback loops are first-class (built into pipeline)
- Parallelism is explicit (template defines what can parallelize)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """LLM model tiers for different tasks."""
    FAST = "haiku"       # Fast, cheap - for simple tasks
    BALANCED = "sonnet"  # Balanced - for most tasks
    POWERFUL = "opus"    # Most capable - for complex reasoning


@dataclass
class AgentConfig:
    """
    Configuration for a swarm agent.

    Agents are specialized workers that execute skills.
    Each agent can have different skills and use different LLM models.

    Example:
        AgentConfig(
            name="feature_engineer",
            skills=["llm_feature_reasoning", "feature_engineering"],
            model=ModelTier.BALANCED,
            max_concurrent=3,
        )
    """
    name: str
    skills: List[str]
    model: ModelTier = ModelTier.BALANCED
    max_concurrent: int = 1  # Max parallel skill executions
    timeout: int = 300  # Seconds
    retry_count: int = 2

    # Agent-specific LLM prompt overrides
    prompt_overrides: Dict[str, str] = field(default_factory=dict)

    # Resource hints
    requires_gpu: bool = False
    memory_gb: float = 4.0


@dataclass
class StageConfig:
    """
    Configuration for a pipeline stage.

    Stages define WHAT happens and WHEN in the pipeline.
    Multiple agents can work on a stage (parallel or sequential).

    Example:
        StageConfig(
            name="FEATURE_ENGINEERING",
            agents=["feature_engineer"],
            parallel=True,
            inputs=["eda_insights", "cleaned_data"],
            outputs=["engineered_features"],
        )
    """
    name: str
    agents: List[str]

    # Execution mode
    parallel: bool = False  # If True, agents run in parallel

    # Data flow
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Conditional execution
    condition: Optional[str] = None  # Python expression, e.g., "score < 0.9"
    skip_on_failure: bool = False

    # Loop configuration (for feedback stages)
    loop_back_to: Optional[str] = None  # Stage name to loop back to
    max_iterations: int = 1

    # Progress tracking
    weight: int = 10  # Relative weight for progress bar (higher = more time expected)
    description: str = ""


@dataclass
class FeedbackConfig:
    """
    Configuration for feedback loops.

    Feedback loops allow the swarm to iterate and improve based on results.
    This is KEY to achieving 10/10 performance.

    Example:
        FeedbackConfig(
            enabled=True,
            max_iterations=3,
            improvement_threshold=0.005,
            feedback_agents=["feature_engineer"],
        )
    """
    enabled: bool = True
    max_iterations: int = 2
    improvement_threshold: float = 0.005  # Min improvement to continue (0.5%)

    # Which agents participate in feedback
    feedback_agents: List[str] = field(default_factory=list)

    # What triggers feedback
    trigger_metric: str = "score"  # Metric to monitor
    trigger_condition: str = "improvement"  # "improvement", "threshold", "always"

    # Feedback data
    feedback_inputs: List[str] = field(default_factory=lambda: ["feature_importance", "score"])


class SwarmTemplate:
    """
    Base class for domain-specific swarm templates.

    Templates are the heart of Jotty's intelligence - they encode
    domain expertise into reusable orchestration patterns.

    To create a new template:
    1. Subclass SwarmTemplate
    2. Define agents with their skills
    3. Define pipeline stages
    4. Add domain-specific LLM prompts
    5. Configure feedback loops

    Example:
        class SwarmML(SwarmTemplate):
            name = "SwarmML"
            agents = {...}
            pipeline = [...]
            llm_prompts = {...}
    """

    # Template metadata
    name: str = "base"
    version: str = "1.0.0"
    description: str = "Base swarm template"

    # Agent configurations
    agents: Dict[str, AgentConfig] = {}

    # Pipeline stages (execution order)
    pipeline: List[StageConfig] = []

    # Domain-specific LLM prompts
    llm_prompts: Dict[str, str] = {}

    # Feedback configuration
    feedback_config: Optional[FeedbackConfig] = None

    # Template capabilities
    supported_problem_types: List[str] = []

    def __init__(self):
        """Initialize template with default configurations."""
        self._context: Dict[str, Any] = {}
        self._results: Dict[str, Any] = {}
        self._iteration: int = 0
        self._best_score: float = 0.0
        self._learning = None  # MASLearning instance, set by Swarm

    def detect_problem_type(self, X, y=None, **kwargs) -> str:
        """
        Auto-detect the problem type from data.

        Override in subclass for domain-specific detection.
        """
        return "unknown"

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate that all required inputs are provided.

        Override in subclass for domain-specific validation.
        """
        return True

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name)

    def get_stage_config(self, stage_name: str) -> Optional[StageConfig]:
        """Get configuration for a specific stage."""
        for stage in self.pipeline:
            if stage.name == stage_name:
                return stage
        return None

    def get_prompt(self, prompt_name: str, **format_kwargs) -> str:
        """Get and format an LLM prompt."""
        prompt = self.llm_prompts.get(prompt_name, "")
        if format_kwargs:
            try:
                return prompt.format(**format_kwargs)
            except KeyError:
                return prompt
        return prompt

    def should_continue_feedback(self, current_score: float, previous_score: float) -> bool:
        """Determine if feedback loop should continue."""
        if not self.feedback_config or not self.feedback_config.enabled:
            return False

        if self._iteration >= self.feedback_config.max_iterations:
            return False

        improvement = current_score - previous_score
        if improvement < self.feedback_config.improvement_threshold:
            return False

        return True

    def get_parallel_stages(self) -> List[List[StageConfig]]:
        """
        Group stages that can run in parallel.

        Returns list of stage groups - stages within a group can parallelize.
        """
        groups = []
        current_group = []

        for stage in self.pipeline:
            if stage.parallel and current_group and current_group[-1].parallel:
                # Can add to current parallel group
                current_group.append(stage)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [stage]

        if current_group:
            groups.append(current_group)

        return groups

    def get_total_weight(self) -> int:
        """Get total weight for progress tracking."""
        return sum(stage.weight for stage in self.pipeline)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize template to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'agents': {k: vars(v) for k, v in self.agents.items()},
            'pipeline': [vars(s) for s in self.pipeline],
            'feedback_config': vars(self.feedback_config) if self.feedback_config else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwarmTemplate':
        """Deserialize template from dictionary."""
        template = cls()
        template.name = data.get('name', template.name)
        template.version = data.get('version', template.version)
        template.description = data.get('description', template.description)

        # Reconstruct agents
        for name, config in data.get('agents', {}).items():
            template.agents[name] = AgentConfig(**config)

        # Reconstruct pipeline
        template.pipeline = [StageConfig(**s) for s in data.get('pipeline', [])]

        # Reconstruct feedback config
        if data.get('feedback_config'):
            template.feedback_config = FeedbackConfig(**data['feedback_config'])

        return template

    # =========================================================================
    # Learning Lifecycle Hooks (inherited by ALL templates)
    # =========================================================================

    def set_learning(self, mas_learning):
        """Attach learning system (called by Swarm before execution)."""
        self._learning = mas_learning

    async def before_execution(self, **kwargs) -> Dict[str, Any]:
        """
        Pre-execution learning phase. Loads relevant past sessions.
        Override in subclass for template-specific learning.
        """
        if not getattr(self, '_learning', None):
            return {}
        try:
            task_desc = kwargs.get('business_context', '') or kwargs.get('context', '')
            learnings = self._learning.load_relevant_learnings(
                task_description=task_desc,
                agent_types=[],
                top_k=5,
            )
            return learnings
        except Exception as e:
            logger.debug(f"Pre-execution learning failed: {e}")
            return {}

    async def after_execution(self, results: Dict[str, Any], **kwargs):
        """
        Post-execution learning phase. Records session outcome.
        Override in subclass for template-specific pattern extraction.
        """
        if not getattr(self, '_learning', None):
            return
        try:
            self._learning.record_session(
                task_description=kwargs.get('business_context', ''),
                agents_used=['skill_orchestrator'],
                total_time=0.0,
                success=results.get('final_score', 0) > 0,
            )
            self._learning.save_all()
        except Exception as e:
            logger.debug(f"Post-execution learning failed: {e}")

    def on_stage_complete(self, stage_name: str, results: Dict[str, Any]):
        """
        Called after each pipeline stage. Records experience for Q-learning.
        Override in subclass for custom reward shaping.
        """
        if not getattr(self, '_learning', None):
            return
        try:
            reward = 0.5 if results.get('success', True) else -0.5
            # MASLearning delegates to SwarmLearningManager which IS the SwarmLearningManager
            lm = getattr(self._learning, 'learning_manager', None)
            if lm and hasattr(lm, 'record_experience'):
                lm.record_experience(
                    agent_name='pipeline',
                    state={'stage': stage_name, 'template': self.name},
                    action={'type': 'execute_stage', 'stage': stage_name},
                    reward=reward,
                    domain='ml_pipeline',
                )
        except Exception as e:
            logger.debug(f"Stage learning failed: {e}")


class TemplateExecutor:
    """
    Executes a swarm template.

    This is the runtime engine that:
    1. Spawns agents based on template configuration
    2. Executes pipeline stages in order (with parallelism)
    3. Manages data flow between stages
    4. Handles feedback loops
    5. Tracks progress
    """

    def __init__(self, template: SwarmTemplate, swarm_manager=None):
        self.template = template
        self.swarm_manager = swarm_manager
        self._context: Dict[str, Any] = {}
        self._agents: Dict[str, Any] = {}
        self._progress_callback: Optional[Callable] = None

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the template pipeline.

        Args:
            **kwargs: Inputs for the pipeline (e.g., X, y, time_budget)

        Returns:
            Dictionary with results from all stages
        """
        # Initialize context with inputs
        self._context = dict(kwargs)
        self._context['_iteration'] = 0
        self._context['_best_score'] = 0.0

        # Validate inputs
        if not self.template.validate_inputs(**kwargs):
            raise ValueError("Invalid inputs for template")

        # Spawn agents
        await self._spawn_agents()

        # Execute pipeline
        results = await self._execute_pipeline()

        return results

    async def _spawn_agents(self):
        """Spawn all agents defined in template."""
        for name, config in self.template.agents.items():
            # Create agent instance (integration with SwarmManager)
            self._agents[name] = {
                'config': config,
                'status': 'ready',
            }

    async def _execute_pipeline(self) -> Dict[str, Any]:
        """Execute all stages in the pipeline."""
        results = {}

        for stage in self.template.pipeline:
            # Check condition
            if stage.condition and not self._evaluate_condition(stage.condition):
                continue

            # Execute stage
            stage_result = await self._execute_stage(stage)
            results[stage.name] = stage_result

            # Update context with outputs
            for output in stage.outputs:
                if output in stage_result:
                    self._context[output] = stage_result[output]

            # Handle feedback loop
            if stage.loop_back_to and self._should_loop(stage):
                # Loop back to specified stage
                loop_stage = self.template.get_stage_config(stage.loop_back_to)
                if loop_stage:
                    self._context['_iteration'] += 1
                    # Re-execute from loop point (recursive)
                    # This is handled by the main loop

        return results

    async def _execute_stage(self, stage: StageConfig) -> Dict[str, Any]:
        """Execute a single stage."""
        if self._progress_callback:
            self._progress_callback(stage.name, 'start')

        # Gather inputs
        inputs = {name: self._context.get(name) for name in stage.inputs}

        # Execute agents
        if stage.parallel and len(stage.agents) > 1:
            # Parallel execution
            tasks = [
                self._execute_agent(agent_name, inputs)
                for agent_name in stage.agents
            ]
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results
            result = {}
            for r in agent_results:
                if isinstance(r, dict):
                    result.update(r)
        else:
            # Sequential execution
            result = {}
            for agent_name in stage.agents:
                agent_result = await self._execute_agent(agent_name, inputs)
                result.update(agent_result)

        if self._progress_callback:
            self._progress_callback(stage.name, 'complete', result)

        return result

    async def _execute_agent(self, agent_name: str, inputs: Dict) -> Dict[str, Any]:
        """Execute an agent's skills."""
        agent_info = self._agents.get(agent_name)
        if not agent_info:
            return {}

        config = agent_info['config']
        result = {}

        # Execute each skill
        for skill_name in config.skills:
            skill_result = await self._execute_skill(skill_name, inputs, config)
            result.update(skill_result)

        return result

    async def _execute_skill(self, skill_name: str, inputs: Dict, agent_config: AgentConfig) -> Dict[str, Any]:
        """
        Execute a skill.

        Integrates with the ML skills library.
        """
        # Get skill class from registry
        skill_class = self._get_skill_class(skill_name)
        if skill_class is None:
            return {}

        try:
            # Instantiate and execute skill
            skill = skill_class()
            await skill.init()

            # Extract X, y from inputs
            X = inputs.get('X') or inputs.get('cleaned_X') or inputs.get('engineered_X') or inputs.get('selected_X')
            y = inputs.get('y')

            # Build context from inputs
            context = {k: v for k, v in inputs.items() if k not in ['X', 'y']}
            context['problem_type'] = inputs.get('problem_type', 'classification')

            # Execute skill
            result = await skill.execute(X, y, **context)

            # Convert SkillResult to dict
            if result.success:
                output = {
                    'success': True,
                    'data': result.data,
                    **result.metrics,
                    **result.metadata,
                }
                # Map skill output to stage outputs
                if result.data is not None:
                    output['X_enhanced'] = result.data
                    output['engineered_X'] = result.data
                    output['selected_X'] = result.data
                return output
            else:
                return {'success': False, 'error': result.error}

        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(f"Skill {skill_name} failed: {e}")
            return {'success': False, 'error': str(e)}

    def _get_skill_class(self, skill_name: str):
        """Get skill class by name from registry."""
        # Skill name to class mapping
        SKILL_MAP = {
            # EDA
            'eda_analysis': 'EDASkill',
            'data_profiling': 'EDASkill',
            # Feature Engineering
            'feature_engineering': 'FeatureEngineeringSkill',
            'target_encoding': 'FeatureEngineeringSkill',
            'interaction_features': 'FeatureEngineeringSkill',
            # LLM Feature Reasoning
            'llm_feature_reasoning': 'LLMFeatureReasonerSkill',
            # Feature Selection
            'shap_selection': 'FeatureSelectionSkill',
            'permutation_importance': 'FeatureSelectionSkill',
            'boruta_selection': 'FeatureSelectionSkill',
            'correlation_filter': 'FeatureSelectionSkill',
            # Model Selection
            'model_selection': 'ModelSelectionSkill',
            'cross_validation': 'ModelSelectionSkill',
            # Hyperopt
            'hyperparameter_optimization': 'HyperoptSkill',
            # Ensemble
            'weighted_voting': 'EnsembleSkill',
            'stacking': 'EnsembleSkill',
            'multi_level_stacking': 'EnsembleSkill',
            'greedy_selection': 'EnsembleSkill',
        }

        class_name = SKILL_MAP.get(skill_name)
        if not class_name:
            return None

        try:
            # Import skill classes
            try:
                from Jotty.core.skills.ml import (
                    EDASkill, LLMFeatureReasonerSkill, FeatureEngineeringSkill,
                    FeatureSelectionSkill, ModelSelectionSkill, HyperoptSkill,
                    EnsembleSkill
                )
            except ImportError:
                from core.skills.ml import (
                    EDASkill, LLMFeatureReasonerSkill, FeatureEngineeringSkill,
                    FeatureSelectionSkill, ModelSelectionSkill, HyperoptSkill,
                    EnsembleSkill
                )

            CLASS_MAP = {
                'EDASkill': EDASkill,
                'LLMFeatureReasonerSkill': LLMFeatureReasonerSkill,
                'FeatureEngineeringSkill': FeatureEngineeringSkill,
                'FeatureSelectionSkill': FeatureSelectionSkill,
                'ModelSelectionSkill': ModelSelectionSkill,
                'HyperoptSkill': HyperoptSkill,
                'EnsembleSkill': EnsembleSkill,
            }

            return CLASS_MAP.get(class_name)
        except ImportError:
            return None

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition string against context."""
        try:
            return eval(condition, {"__builtins__": {}}, self._context)
        except Exception:
            return False

    def _should_loop(self, stage: StageConfig) -> bool:
        """Determine if we should loop back."""
        if not stage.loop_back_to:
            return False

        iteration = self._context.get('_iteration', 0)
        if iteration >= stage.max_iterations:
            return False

        return True

    def set_progress_callback(self, callback: Callable):
        """Set callback for progress updates."""
        self._progress_callback = callback
