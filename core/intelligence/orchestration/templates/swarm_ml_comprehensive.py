from typing import Any
"""
SwarmML Comprehensive - Learning-Enhanced Machine Learning Template
====================================================================

The ultimate AutoML swarm template combining:
- World-class ML pipeline (from SwarmML)
- Cross-session learning (TransferableLearningStore)
- Online learning during execution (LearningManager)
- Adaptive exploration/exploitation (AdaptiveController)
- Pattern discovery and reuse (PatternLearner)
- Self-improving feedback loops
- **MLflow Integration** - Full experiment tracking
- **PDF Report Generation** - Comprehensive analysis reports

This template represents the full power of Jotty's learning infrastructure
applied to machine learning problems.

Key Innovations:
1. LEARNING FROM HISTORY - Uses past sessions to inform current execution
2. ADAPTIVE STRATEGIES - Adjusts exploration based on task similarity
3. PATTERN REUSE - Successful feature patterns are remembered and reused
4. ONLINE LEARNING - Q-values updated in real-time during execution
5. DUAL SCORING - Separate scoring for learning vs execution relevance
6. MLFLOW TRACKING - All experiments tracked with metrics, params, artifacts
7. PDF REPORTS - Comprehensive reports with visualizations and insights

Usage:
    from jotty import Swarm

    result = await Swarm.solve(
        template="ml_comprehensive",
        X=X, y=y,
        time_budget=300,
        context="Predict customer churn",
        use_learning=True,    # Enable cross-session learning
        use_mlflow=True,      # Enable MLflow tracking
        generate_report=True, # Generate PDF report
    )

Performance:
    - Improves over time as it learns from more sessions
    - Adapts to different problem types
    - Reuses successful patterns across similar tasks
"""

import time
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd

from .base import (
    SwarmTemplate, AgentConfig, StageConfig, FeedbackConfig, ModelTier
)
from .swarm_ml import SwarmML

logger = logging.getLogger(__name__)


# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

@dataclass
class MLflowConfig:
    """Configuration for MLflow integration."""
    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "jotty_ml_comprehensive"
    run_name_prefix: str = "swarm_ml"
    log_models: bool = True
    log_artifacts: bool = True
    log_feature_importance: bool = True
    log_shap_values: bool = True
    registered_model_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# PDF REPORT CONFIGURATION
# =============================================================================

@dataclass
class ReportConfig:
    """Configuration for PDF report generation."""
    enabled: bool = True
    output_dir: str = "ml_reports"
    include_executive_summary: bool = True
    include_data_profile: bool = True
    include_feature_importance: bool = True
    include_model_benchmarking: bool = True
    include_shap_analysis: bool = True
    include_confusion_matrix: bool = True
    include_roc_curves: bool = True
    include_precision_recall: bool = True
    include_learning_curves: bool = True
    include_baseline_comparison: bool = True
    include_recommendations: bool = True
    max_features_in_report: int = 20
    figure_dpi: int = 150


# =============================================================================
# TELEGRAM NOTIFICATION CONFIGURATION
# =============================================================================

@dataclass
class TelegramConfig:
    """Configuration for Telegram notifications."""
    enabled: bool = True
    bot_token: str = ""  # From env: TELEGRAM_BOT_TOKEN
    chat_id: str = ""    # From env: TELEGRAM_CHAT_ID
    send_report_pdf: bool = True
    send_summary_message: bool = True
    include_metrics_in_message: bool = True
    include_feature_importance: bool = True
    max_features_in_message: int = 5


# =============================================================================
# LEARNING-ENHANCED CONFIGURATIONS
# =============================================================================

@dataclass
class LearningConfig:
    """Configuration for learning components."""
    # Cross-session learning
    enable_transfer_learning: bool = True
    max_history_sessions: int = 100
    learning_relevance_threshold: float = 0.5
    execution_relevance_threshold: float = 0.6

    # Online learning
    enable_online_learning: bool = True
    initial_learning_rate: float = 0.1
    discount_factor: float = 0.95  # For TD-Lambda
    lambda_trace: float = 0.8

    # Adaptive exploration
    enable_adaptive_exploration: bool = True
    initial_epsilon: float = 0.3
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.95

    # Pattern learning
    enable_pattern_learning: bool = True
    pattern_confidence_threshold: float = 0.7
    min_pattern_uses: int = 3

    # Memory management
    max_experiences: int = 10000
    prioritized_replay: bool = True
    priority_alpha: float = 0.6


@dataclass
class LearningState:
    """Runtime state for learning during execution."""
    session_id: str = ""
    task_description: str = ""
    similar_sessions: List[Dict] = field(default_factory=list)
    learned_patterns: List[Dict] = field(default_factory=list)
    current_epsilon: float = 0.3
    q_values: Dict[str, float] = field(default_factory=dict)
    experience_buffer: List[Dict] = field(default_factory=list)
    iteration: int = 0
    best_score: float = 0.0
    score_history: List[float] = field(default_factory=list)



from ._mlflow_mixin import MLflowMixin
from ._report_mixin import ReportMixin
from ._world_class_report_mixin import WorldClassReportMixin
from ._telegram_mixin import TelegramMixin

class SwarmMLComprehensive(MLflowMixin, ReportMixin, WorldClassReportMixin, TelegramMixin, SwarmML):
    """
    Comprehensive Machine Learning swarm template with full learning integration.

    Extends SwarmML with:
    - TransferableLearningStore for cross-session learning
    - LearningManager for online learning
    - Adaptive exploration strategies
    - Pattern discovery and reuse

    Learning Lifecycle:
    1. BEFORE EXECUTION:
       - Load similar sessions from history
       - Retrieve successful patterns
       - Set initial exploration rate based on task familiarity

    2. DURING EXECUTION:
       - Update Q-values based on stage outcomes
       - Record experiences for replay
       - Adapt epsilon based on progress

    3. AFTER EXECUTION:
       - Store session in history
       - Extract and save successful patterns
       - Update transfer learning store
    """

    name = "SwarmMLComprehensive"
    version = "3.0.0"
    description = "Learning-enhanced AutoML with cross-session learning and pattern reuse"

    # ================================================================
    # EXTENDED AGENT CONFIGURATIONS
    # ================================================================
    agents = {
        **SwarmML.agents,  # Inherit all ML agents

        # Add learning-focused agent
        "learning_analyst": AgentConfig(
            name="learning_analyst",
            skills=[
                "pattern_extraction",
                "session_analysis",
                "strategy_recommendation",
            ],
            model=ModelTier.BALANCED,
            max_concurrent=1,
            timeout=60,
        ),

        # Add meta-learning agent
        "meta_learner": AgentConfig(
            name="meta_learner",
            skills=[
                "task_similarity_scoring",
                "strategy_transfer",
                "adaptive_control",
            ],
            model=ModelTier.FAST,
            max_concurrent=1,
            timeout=30,
        ),
    }

    # ================================================================
    # EXTENDED PIPELINE WITH LEARNING STAGES
    # ================================================================
    pipeline = [
        # Stage 0: Learning Initialization (NEW)
        StageConfig(
            name="LEARNING_INIT",
            agents=["meta_learner"],
            parallel=False,
            inputs=["X", "y", "business_context"],
            outputs=["similar_sessions", "learned_patterns", "initial_strategy"],
            weight=3,
            description="Load historical learning, find similar sessions, retrieve patterns",
        ),

        # Stage 1: Data Understanding (with learning context)
        StageConfig(
            name="DATA_UNDERSTANDING",
            agents=["data_analyst"],
            parallel=False,
            inputs=["X", "y", "learned_patterns"],
            outputs=["eda_insights", "data_profile", "cleaned_X"],
            weight=5,
            description="Analyze data with learned pattern hints",
        ),

        # Stage 2: Feature Engineering (with pattern reuse)
        StageConfig(
            name="FEATURE_ENGINEERING",
            agents=["feature_engineer"],
            parallel=True,
            inputs=["cleaned_X", "y", "eda_insights", "business_context", "learned_patterns"],
            outputs=["engineered_X", "feature_suggestions"],
            weight=15,
            description="LLM feature generation with successful pattern hints",
        ),

        # Stage 3: Feature Selection
        StageConfig(
            name="FEATURE_SELECTION",
            agents=["feature_selector"],
            parallel=False,
            inputs=["engineered_X", "y"],
            outputs=["selected_X", "feature_scores", "shap_importance"],
            weight=10,
            description="SHAP + multi-model importance selection",
        ),

        # Stage 4: Model Selection (with strategy from history)
        StageConfig(
            name="MODEL_SELECTION",
            agents=["model_architect"],
            parallel=True,
            inputs=["selected_X", "y", "problem_type", "initial_strategy"],
            outputs=["model_scores", "best_model", "oof_predictions"],
            weight=25,
            description="Evaluate models with historically successful configurations",
        ),

        # Stage 5: Hyperparameter Optimization
        StageConfig(
            name="HYPERPARAMETER_OPTIMIZATION",
            agents=["model_architect"],
            parallel=True,
            inputs=["selected_X", "y", "model_scores", "initial_strategy"],
            outputs=["optimized_model", "best_params", "tuning_history"],
            weight=30,
            description="Warm-start tuning from historical best params",
        ),

        # Stage 6: Ensemble
        StageConfig(
            name="ENSEMBLE",
            agents=["ensemble_expert"],
            parallel=False,
            inputs=["selected_X", "y", "optimized_model", "model_scores", "oof_predictions"],
            outputs=["final_model", "ensemble_score"],
            weight=20,
            description="Multi-level stacking ensemble",
        ),

        # Stage 7: Evaluation & Explanation
        StageConfig(
            name="EVALUATION",
            agents=["explainer"],
            parallel=False,
            inputs=["final_model", "selected_X", "y"],
            outputs=["final_score", "feature_importance", "shap_values"],
            weight=5,
            description="Final metrics and SHAP explanations",
        ),

        # Stage 8: Learning Update (NEW)
        StageConfig(
            name="LEARNING_UPDATE",
            agents=["learning_analyst"],
            parallel=False,
            inputs=["final_score", "feature_importance", "feature_suggestions",
                    "best_params", "tuning_history", "business_context"],
            outputs=["extracted_patterns", "session_record"],
            weight=3,
            description="Extract patterns, update learning store, record session",
        ),

        # Stage 9: Feedback Loop (Adaptive)
        StageConfig(
            name="FEEDBACK_LOOP",
            agents=["feature_engineer"],
            parallel=False,
            inputs=["X", "y", "feature_importance", "final_score", "business_context",
                    "learned_patterns", "extracted_patterns"],
            outputs=["improved_features"],
            condition="score < target_score and iteration < max_iterations",
            loop_back_to="FEATURE_ENGINEERING",
            max_iterations=3,  # More iterations for learning
            weight=10,
            description="Adaptive feedback with pattern-informed feature generation",
        ),
    ]

    # ================================================================
    # LEARNING CONFIGURATION
    # ================================================================
    learning_config = LearningConfig()

    # ================================================================
    # EXTENDED FEEDBACK CONFIGURATION
    # ================================================================
    feedback_config = FeedbackConfig(
        enabled=True,
        max_iterations=3,  # More iterations for learning template
        improvement_threshold=0.003,  # Lower threshold - be more persistent
        feedback_agents=["feature_engineer", "meta_learner"],
        trigger_metric="score",
        trigger_condition="improvement",
        feedback_inputs=["feature_importance", "score", "shap_values",
                        "learned_patterns", "q_values"],
    )

    # ================================================================
    # EXTENDED LLM PROMPTS (Learning-Enhanced)
    # ================================================================
    llm_prompts = {
        **SwarmML.llm_prompts,  # Inherit all ML prompts

        # New: Pattern-informed feature generation
        "pattern_informed": """You are a **Kaggle Grandmaster** using LEARNED PATTERNS from successful past sessions.

## SUCCESSFUL PATTERNS FROM SIMILAR TASKS
{learned_patterns}

These patterns WORKED on similar problems. Use them as STRONG hints.

## Current Task
Problem: {problem_type} to predict {target}
Features: {features}
Context: {context}

## Instructions
1. PRIORITIZE features similar to the successful patterns
2. ADAPT the patterns to current column names
3. Create VARIATIONS of successful patterns
4. Add 2-3 NEW features based on your analysis

Return ONLY executable Python code:
X['feature_name'] = <transformation>

Code only:""",

        # New: Strategy transfer prompt
        "strategy_transfer": """You are a **Meta-Learning Expert** analyzing task similarity.

## Current Task
Problem: {problem_type}
Features: {features}
Context: {context}

## Similar Historical Sessions
{similar_sessions}

## Analysis Task
1. Identify the MOST similar historical session
2. Extract the STRATEGY that worked (model choice, hyperparams, features)
3. Recommend how to ADAPT that strategy to current task

Output a JSON strategy:
{{
    "recommended_models": ["model1", "model2"],
    "hyperparameter_hints": {{"param": "value"}},
    "feature_patterns_to_try": ["pattern1", "pattern2"],
    "confidence": 0.8
}}

JSON only:""",

        # New: Learning summary prompt
        "learning_summary": """You are a **Learning Analyst** extracting reusable patterns from this ML session.

## Session Results
Final Score: {final_score}
Top Features: {top_features}
Best Model: {best_model}
Best Params: {best_params}
Problem Type: {problem_type}
Context: {context}

## Analysis Task
1. Identify the KEY PATTERNS that led to success
2. Extract GENERALIZABLE feature engineering code
3. Note model/hyperparameter combinations that worked

Output patterns as JSON:
{{
    "feature_patterns": [
        {{"name": "pattern_name", "code": "X['feat'] = ...", "reason": "why it worked"}}
    ],
    "model_strategy": {{"model": "...", "key_params": {{}}}},
    "task_type": "classification/regression",
    "success_indicators": ["indicator1", "indicator2"]
}}

JSON only:""",
    }

    def __init__(self) -> None:
        """Initialize comprehensive ML template with learning components."""
        super().__init__()
        self._learning_state = LearningState()
        self._transfer_store = None
        self._learning_coordinator = None
        self._initialized_learning = False

    async def init_learning(self) -> Any:
        """Initialize learning components."""
        if self._initialized_learning:
            return

        try:
            from Jotty.core.intelligence.learning.transfer_learning import TransferableLearningStore
            from Jotty.core.intelligence.learning.learning_coordinator import LearningManager

            self._transfer_store = TransferableLearningStore()
            store_path = os.path.join('Jotty', 'outputs', 'transfer_store.json')
            if os.path.exists(store_path):
                self._transfer_store.load(store_path)

            class _LCConfig:
                output_base_dir = os.path.join('Jotty', 'outputs')
                enable_rl = False

            self._learning_coordinator = LearningManager(
                config=_LCConfig(),
                base_dir=os.path.join('Jotty', 'outputs'),
            )
            self._learning_coordinator.initialize(auto_load=True)

            self._initialized_learning = True
            logger.info("Learning components initialized")

        except Exception as e:
            logger.warning(f"Learning components not available: {e}")
            self._initialized_learning = False

    async def before_execution(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Pre-execution learning phase.

        Loads similar sessions, retrieves patterns, sets initial strategy.
        """
        await self.init_learning()

        task_desc = kwargs.get('business_context', '') or self._build_task_description(**kwargs)
        self._learning_state.task_description = task_desc

        result = {
            'similar_sessions': [],
            'learned_patterns': [],
            'initial_strategy': {},
        }

        if not self._transfer_store:
            return result

        try:
            # Get sessions relevant for EXECUTION (find best strategy to copy)
            exec_sessions = self._transfer_store.get_relevant_sessions(
                task_desc,
                top_k=5
            )

            # Get sessions relevant for LEARNING (find lessons to apply)
            learning_sessions = self._transfer_store.get_learning_sessions(
                task_desc,
                top_k=10
            )

            result['similar_sessions'] = exec_sessions

            # Extract patterns from successful sessions
            patterns = self._extract_patterns_from_sessions(exec_sessions)
            result['learned_patterns'] = patterns
            self._learning_state.learned_patterns = patterns

            # Build initial strategy from best session
            if exec_sessions:
                best_session = exec_sessions[0]
                result['initial_strategy'] = {
                    'recommended_models': best_session.get('best_models', []),
                    'hyperparameter_hints': best_session.get('best_params', {}),
                    'feature_patterns': patterns[:5],
                }

            # Set adaptive epsilon based on task familiarity
            if exec_sessions:
                max_relevance = max(s.get('relevance_score', 0) for s in exec_sessions)
                # More familiar = less exploration needed
                self._learning_state.current_epsilon = max(
                    self.learning_config.min_epsilon,
                    self.learning_config.initial_epsilon * (1 - max_relevance)
                )
            else:
                # Unknown task = more exploration
                self._learning_state.current_epsilon = self.learning_config.initial_epsilon

            logger.info(f"Pre-execution: Found {len(exec_sessions)} similar sessions, "
                       f"{len(patterns)} patterns, epsilon={self._learning_state.current_epsilon:.3f}")

        except Exception as e:
            logger.warning(f"Pre-execution learning failed: {e}")

        return result

    async def after_execution(self, results: Dict[str, Any], **kwargs: Any) -> Any:
        """
        Post-execution learning phase.

        Stores session, extracts patterns, updates learning store.
        """
        if not self._transfer_store:
            return

        try:
            # Build session record
            session_record = {
                'task_description': self._learning_state.task_description,
                'problem_type': kwargs.get('problem_type', 'classification'),
                'final_score': results.get('final_score', 0),
                'feature_importance': results.get('feature_importance', {}),
                'feature_suggestions': results.get('feature_suggestions', []),
                'best_model': str(results.get('best_model', '')),
                'best_params': results.get('best_params', {}),
                'n_features': kwargs.get('X', pd.DataFrame()).shape[1] if kwargs.get('X') is not None else 0,
                'iterations': self._learning_state.iteration,
                'score_history': self._learning_state.score_history,
                'timestamp': time.time(),
            }

            # Store session using actual API
            self._transfer_store.record_session(
                task_description=self._learning_state.task_description,
                agents_used=['ml_pipeline'],
                total_time=time.time() - session_record.get('timestamp', time.time()),
                success=results.get('final_score', 0) > 0,
            )

            # Extract and store patterns as experiences
            if results.get('final_score', 0) > 0.7:  # Only learn from good sessions
                patterns = self._extract_patterns_from_results(results, kwargs)
                for pattern in patterns:
                    self._transfer_store.record_experience(
                        query=self._learning_state.task_description,
                        action=pattern.get('code', ''),
                        reward=results.get('final_score', 0),
                        success=True,
                        agent='ml_pipeline',
                    )

            # Persist learning
            self._save_learning()
            logger.info(f"Post-execution: Stored session with score {results.get('final_score', 0):.4f}")

        except Exception as e:
            logger.warning(f"Post-execution learning failed: {e}")

    def on_stage_complete(self, stage_name: str, results: Dict[str, Any]) -> None:
        """
        ML-specific stage completion with custom reward shaping.

        Updates Q-values, records experience, adapts epsilon.
        """
        # Base learning (Q-value update via MASLearning)
        super().on_stage_complete(stage_name, results)

        try:
            # Calculate reward from stage results
            reward = self._calculate_stage_reward(stage_name, results)

            # Record experience
            experience = {
                'stage': stage_name,
                'state': self._get_current_state(),
                'action': stage_name,
                'reward': reward,
                'results': results,
            }
            self._learning_state.experience_buffer.append(experience)

            # Update Q-value for this stage
            state_key = f"{stage_name}_{self._learning_state.iteration}"
            old_q = self._learning_state.q_values.get(state_key, 0)
            new_q = old_q + self.learning_config.initial_learning_rate * (reward - old_q)
            self._learning_state.q_values[state_key] = new_q

            # Track score if this is evaluation stage
            if stage_name == "EVALUATION":
                score = results.get('final_score', 0)
                self._learning_state.score_history.append(score)
                if score > self._learning_state.best_score:
                    self._learning_state.best_score = score

            # Adaptive epsilon decay
            if self._learning_state.score_history:
                # Decay epsilon if we're making progress
                self._learning_state.current_epsilon = max(
                    self.learning_config.min_epsilon,
                    self._learning_state.current_epsilon * self.learning_config.epsilon_decay
                )

        except Exception as e:
            logger.debug(f"Stage learning update failed: {e}")

    def _extract_patterns_from_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """Extract reusable patterns from historical sessions."""
        patterns = []
        seen_codes = set()

        for session in sessions:
            for suggestion in session.get('feature_suggestions', []):
                code = suggestion.get('code', '')
                if code and code not in seen_codes:
                    patterns.append({
                        'code': code,
                        'perspective': suggestion.get('perspective', 'unknown'),
                        'success_rate': session.get('final_score', 0.5),
                        'source': 'history',
                    })
                    seen_codes.add(code)

        # Sort by success rate
        patterns.sort(key=lambda p: p['success_rate'], reverse=True)
        return patterns[:20]  # Top 20 patterns

    def _extract_patterns_from_results(self, results: Dict, kwargs: Dict) -> List[Dict]:
        """Extract patterns from current session results."""
        patterns = []

        feature_importance = results.get('feature_importance', {})
        suggestions = results.get('feature_suggestions', [])

        # Find which suggestions led to important features
        important_features = {k for k, v in feature_importance.items() if v > 0.01}

        for suggestion in suggestions:
            code = suggestion.get('code', '')
            # Check if this code created an important feature
            for feat in important_features:
                if feat in code:
                    patterns.append({
                        'code': code,
                        'perspective': suggestion.get('perspective', 'unknown'),
                        'importance': feature_importance.get(feat, 0),
                        'problem_type': kwargs.get('problem_type', 'classification'),
                        'source': 'current_session',
                    })
                    break

        return patterns

    def _calculate_stage_reward(self, stage_name: str, results: Dict) -> float:
        """Calculate reward for a stage based on its results."""
        if not results.get('success', True):
            return -0.5

        reward = 0.1  # Base reward for completion

        if stage_name == "DATA_UNDERSTANDING":
            # Reward for finding insights
            n_recommendations = len(results.get('eda_insights', {}).get('recommendations', []))
            reward += min(0.3, n_recommendations * 0.03)

        elif stage_name == "FEATURE_ENGINEERING":
            # Reward for creating features
            n_features = results.get('n_features_created', 0)
            reward += min(0.4, n_features * 0.02)

        elif stage_name == "FEATURE_SELECTION":
            # Reward for good selection
            n_selected = len(results.get('selected_features', []))
            reward += 0.2 if 10 <= n_selected <= 50 else 0.1

        elif stage_name == "MODEL_SELECTION":
            # Reward based on best score
            best_score = results.get('best_score', 0)
            reward += best_score * 0.5

        elif stage_name == "EVALUATION":
            # Major reward for final score
            final_score = results.get('final_score', 0)
            reward = final_score  # Direct mapping

        return reward

    def _get_current_state(self) -> Dict:
        """Get current state for learning."""
        return {
            'iteration': self._learning_state.iteration,
            'best_score': self._learning_state.best_score,
            'epsilon': self._learning_state.current_epsilon,
            'n_patterns': len(self._learning_state.learned_patterns),
            'score_history': self._learning_state.score_history[-5:] if self._learning_state.score_history else [],
        }

    def _build_task_description(self, **kwargs: Any) -> str:
        """Build task description from inputs."""
        parts = []

        X = kwargs.get('X')
        if X is not None and hasattr(X, 'columns'):
            parts.append(f"Features: {list(X.columns)[:10]}")
            parts.append(f"Shape: {X.shape}")

        y = kwargs.get('y')
        if y is not None:
            parts.append(f"Target: {y.name if hasattr(y, 'name') else 'target'}")
            if hasattr(y, 'nunique'):
                parts.append(f"Unique targets: {y.nunique()}")

        problem_type = kwargs.get('problem_type', self.detect_problem_type(X, y))
        parts.append(f"Problem: {problem_type}")

        return " | ".join(parts)

    def _save_learning(self) -> Any:
        """Persist all learning state."""
        try:
            if self._transfer_store:
                store_path = os.path.join('Jotty', 'outputs', 'transfer_store.json')
                os.makedirs(os.path.dirname(store_path), exist_ok=True)
                self._transfer_store.save(store_path)
            if self._learning_coordinator:
                self._learning_coordinator.save_all(
                    episode_count=self._learning_state.iteration,
                    avg_reward=self._learning_state.best_score,
                    domains=['ml_pipeline'],
                )
        except Exception as e:
            logger.debug(f"Save learning failed: {e}")

    def _guarded_section(self, section_name: str, report: Any, fn_name: str, *args: Any, **kwargs: Any) -> None:
        """Execute report section with skip guard and outcome recording."""
        manager = getattr(self, '_swarm_manager', None)

        if manager and manager.should_skip_report_section(section_name):
            logger.info(f"  - SKIPPING {section_name} (learned from past failures)")
            return

        try:
            fn = getattr(report, fn_name)
            fn(*args, **kwargs)
            if manager:
                manager.record_report_section_outcome(section_name, success=True)
        except Exception as e:
            logger.debug(f"{section_name} failed: {e}")
            if manager:
                manager.record_report_section_outcome(section_name, success=False, error=str(e))

    def format_learned_patterns_prompt(self, patterns: List[Dict]) -> str:
        """Format learned patterns for LLM prompt."""
        if not patterns:
            return "No learned patterns available."

        lines = []
        for i, pattern in enumerate(patterns[:10], 1):
            code = pattern.get('code', '')
            success = pattern.get('success_rate', pattern.get('importance', 0))
            perspective = pattern.get('perspective', 'unknown')
            lines.append(f"{i}. [{perspective}] (success={success:.2f})")
            lines.append(f"   {code}")

        return "\n".join(lines)

    def should_explore(self) -> bool:
        """Determine if we should explore (try new things) vs exploit (use known good)."""
        import random
        return random.random() < self._learning_state.current_epsilon

    def get_exploration_status(self) -> Dict[str, Any]:
        """Get current exploration/exploitation status."""
        return {
            'epsilon': self._learning_state.current_epsilon,
            'mode': 'explore' if self.should_explore() else 'exploit',
            'iteration': self._learning_state.iteration,
            'best_score': self._learning_state.best_score,
            'n_patterns': len(self._learning_state.learned_patterns),
        }

    # =========================================================================
    # MLFLOW INTEGRATION
    # =========================================================================

