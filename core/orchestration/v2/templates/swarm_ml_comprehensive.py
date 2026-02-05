"""
SwarmML Comprehensive - Learning-Enhanced Machine Learning Template
====================================================================

The ultimate AutoML swarm template combining:
- World-class ML pipeline (from SwarmML)
- Cross-session learning (TransferableLearningStore)
- Online learning during execution (LearningCoordinator)
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
import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
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


class SwarmMLComprehensive(SwarmML):
    """
    Comprehensive Machine Learning swarm template with full learning integration.

    Extends SwarmML with:
    - TransferableLearningStore for cross-session learning
    - LearningCoordinator for online learning
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

    def __init__(self):
        """Initialize comprehensive ML template with learning components."""
        super().__init__()
        self._learning_state = LearningState()
        self._transfer_store = None
        self._learning_coordinator = None
        self._initialized_learning = False

    async def init_learning(self):
        """Initialize learning components."""
        if self._initialized_learning:
            return

        try:
            # Import learning components
            from ...learning.transfer_learning import TransferableLearningStore
            from ...learning.learning_coordinator import LearningCoordinator

            # Initialize transfer learning store
            self._transfer_store = TransferableLearningStore()

            # Initialize learning coordinator
            self._learning_coordinator = LearningCoordinator(
                learning_rate=self.learning_config.initial_learning_rate,
                gamma=self.learning_config.discount_factor,
                lambda_trace=self.learning_config.lambda_trace,
            )

            self._initialized_learning = True
            logger.info("Learning components initialized successfully")

        except ImportError as e:
            logger.warning(f"Learning components not available: {e}")
            self._initialized_learning = False

    async def before_execution(self, **kwargs) -> Dict[str, Any]:
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

    async def after_execution(self, results: Dict[str, Any], **kwargs):
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

            # Store session
            self._transfer_store.store_session(session_record)

            # Extract and store patterns
            if results.get('final_score', 0) > 0.7:  # Only learn from good sessions
                patterns = self._extract_patterns_from_results(results, kwargs)
                for pattern in patterns:
                    self._transfer_store.store_pattern(pattern)

            logger.info(f"Post-execution: Stored session with score {results.get('final_score', 0):.4f}")

        except Exception as e:
            logger.warning(f"Post-execution learning failed: {e}")

    def on_stage_complete(self, stage_name: str, results: Dict[str, Any]):
        """
        Called after each stage completes.

        Updates Q-values, records experience, adapts epsilon.
        """
        if not self._learning_coordinator:
            return

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

    def _build_task_description(self, **kwargs) -> str:
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

    def init_mlflow(self, config: MLflowConfig = None):
        """
        Initialize MLflow tracking for this swarm execution.

        Args:
            config: MLflow configuration (uses default if None)
        """
        self._mlflow_config = config or MLflowConfig()
        self._mlflow_run = None
        self._mlflow_available = False

        if not self._mlflow_config.enabled:
            return

        try:
            import mlflow
            self._mlflow = mlflow

            # Set tracking URI
            mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)

            # Set or create experiment
            mlflow.set_experiment(self._mlflow_config.experiment_name)

            # Start a new run
            run_name = f"{self._mlflow_config.run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._mlflow_run = mlflow.start_run(run_name=run_name)

            # Log default tags
            tags = {
                "swarm_template": self.name,
                "version": self.version,
                **self._mlflow_config.tags
            }
            mlflow.set_tags(tags)

            self._mlflow_available = True
            logger.info(f"MLflow initialized: experiment={self._mlflow_config.experiment_name}, run={run_name}")

        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            # MLflow params must be strings or numbers
            flat_params = self._flatten_dict(params)
            # Truncate long values
            for k, v in flat_params.items():
                if isinstance(v, str) and len(v) > 250:
                    flat_params[k] = v[:250] + "..."
            self._mlflow.log_params(flat_params)
        except Exception as e:
            logger.debug(f"Failed to log params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self._mlflow.log_metric(name, value, step=step)
        except Exception as e:
            logger.debug(f"Failed to log metrics: {e}")

    def log_model(self, model, artifact_path: str = "model", input_example=None):
        """Log trained model to MLflow."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        if not self._mlflow_config.log_models:
            return

        try:
            # Try sklearn first (most common)
            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    input_example=input_example,
                    registered_model_name=self._mlflow_config.registered_model_name
                )
                return
            except Exception:
                pass

            # Try xgboost
            try:
                import mlflow.xgboost
                mlflow.xgboost.log_model(model, artifact_path)
                return
            except Exception:
                pass

            # Try lightgbm
            try:
                import mlflow.lightgbm
                mlflow.lightgbm.log_model(model, artifact_path)
                return
            except Exception:
                pass

            # Fallback to pickle
            import pickle
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                pickle.dump(model, f)
                self._mlflow.log_artifact(f.name, artifact_path)

        except Exception as e:
            logger.debug(f"Failed to log model: {e}")

    def log_feature_importance(self, importance: Dict[str, float], top_n: int = 30):
        """Log feature importance as artifact and metrics."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        if not self._mlflow_config.log_feature_importance:
            return

        try:
            # Sort by importance
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

            # Log top features as metrics
            for rank, (feat, imp) in enumerate(sorted_imp[:10], 1):
                safe_name = feat.replace(" ", "_")[:50]
                self._mlflow.log_metric(f"feat_imp_{rank}_{safe_name}", imp)

            # Create and log importance plot
            if self._mlflow_config.log_artifacts:
                self._log_importance_plot(sorted_imp)

        except Exception as e:
            logger.debug(f"Failed to log feature importance: {e}")

    def log_shap_values(self, shap_values, feature_names: List[str], X_sample=None):
        """Log SHAP values and summary plot."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        if not self._mlflow_config.log_shap_values:
            return

        try:
            import shap
            import matplotlib.pyplot as plt
            import tempfile

            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                self._mlflow.log_artifact(f.name, "shap")
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to log SHAP values: {e}")

    def log_confusion_matrix(self, y_true, y_pred, labels=None):
        """Log confusion matrix as artifact."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            import tempfile

            cm = confusion_matrix(y_true, y_pred, labels=labels)

            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap='Blues', ax=plt.gca())
            plt.title('Confusion Matrix')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                self._mlflow.log_artifact(f.name, "metrics")
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to log confusion matrix: {e}")

    def log_roc_curve(self, y_true, y_prob, pos_label=1):
        """Log ROC curve as artifact."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt
            import tempfile

            fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                self._mlflow.log_artifact(f.name, "metrics")
            plt.close()

            # Log AUC as metric
            self._mlflow.log_metric("roc_auc", roc_auc)

        except Exception as e:
            logger.debug(f"Failed to log ROC curve: {e}")

    def end_mlflow_run(self, status: str = "FINISHED"):
        """End the current MLflow run."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            self._mlflow.end_run(status=status)
            logger.info(f"MLflow run ended with status: {status}")
        except Exception as e:
            logger.debug(f"Failed to end MLflow run: {e}")

    def _log_importance_plot(self, sorted_importance: List[Tuple[str, float]]):
        """Create and log feature importance plot."""
        try:
            import matplotlib.pyplot as plt
            import tempfile

            features = [x[0] for x in sorted_importance]
            values = [x[1] for x in sorted_importance]

            plt.figure(figsize=(10, max(6, len(features) * 0.3)))
            plt.barh(range(len(features)), values[::-1], color='steelblue')
            plt.yticks(range(len(features)), features[::-1])
            plt.xlabel('Importance')
            plt.title('Feature Importance (Top Features)')
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                self._mlflow.log_artifact(f.name, "feature_importance")
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create importance plot: {e}")

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # =========================================================================
    # PDF REPORT GENERATION
    # =========================================================================

    def init_report(self, config: ReportConfig = None):
        """
        Initialize PDF report generation.

        Args:
            config: Report configuration (uses default if None)
        """
        self._report_config = config or ReportConfig()
        self._report_data = {
            'sections': [],
            'figures': [],
            'tables': [],
            'metrics': {},
            'timestamp': datetime.now(),
        }
        self._report_available = False

        if not self._report_config.enabled:
            return

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            self._reportlab = {
                'colors': colors,
                'letter': letter,
                'A4': A4,
                'SimpleDocTemplate': SimpleDocTemplate,
                'Paragraph': Paragraph,
                'Spacer': Spacer,
                'Table': Table,
                'TableStyle': TableStyle,
                'Image': Image,
                'PageBreak': PageBreak,
                'getSampleStyleSheet': getSampleStyleSheet,
                'ParagraphStyle': ParagraphStyle,
                'inch': inch,
            }

            self._report_available = True

            # Create output directory
            os.makedirs(self._report_config.output_dir, exist_ok=True)

            logger.info(f"Report generation initialized: output_dir={self._report_config.output_dir}")

        except ImportError:
            logger.warning("reportlab not installed. Install with: pip install reportlab")
        except Exception as e:
            logger.warning(f"Report initialization failed: {e}")

    def add_executive_summary(self, results: Dict[str, Any], context: str = ""):
        """Add executive summary section to report."""
        if not self._report_available:
            return

        summary = {
            'type': 'executive_summary',
            'title': 'Executive Summary',
            'content': {
                'context': context,
                'final_score': results.get('final_score', 0),
                'best_model': str(results.get('best_model', 'N/A')),
                'n_features_used': results.get('n_features', 0),
                'iterations': self._learning_state.iteration,
                'improvement': self._calculate_improvement(),
                'key_findings': self._extract_key_findings(results),
            }
        }
        self._report_data['sections'].append(summary)

    def add_data_profile(self, eda_insights: Dict[str, Any]):
        """Add data profiling section to report."""
        if not self._report_available or not self._report_config.include_data_profile:
            return

        profile = {
            'type': 'data_profile',
            'title': 'Data Profile & EDA',
            'content': {
                'shape': eda_insights.get('shape', {}),
                'dtypes': eda_insights.get('dtypes', {}),
                'missing': eda_insights.get('missing_values', {}),
                'statistics': eda_insights.get('statistics', {}),
                'correlations': eda_insights.get('correlations', {}),
                'recommendations': eda_insights.get('recommendations', []),
            }
        }
        self._report_data['sections'].append(profile)

    def add_feature_importance(self, importance: Dict[str, float], top_n: int = None):
        """Add feature importance section to report."""
        if not self._report_available or not self._report_config.include_feature_importance:
            return

        top_n = top_n or self._report_config.max_features_in_report
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

        section = {
            'type': 'feature_importance',
            'title': 'Feature Importance Analysis',
            'content': {
                'importance_ranking': sorted_imp,
                'total_features': len(importance),
                'top_n_shown': top_n,
            }
        }
        self._report_data['sections'].append(section)

        # Create importance figure
        self._create_importance_figure(sorted_imp)

    def add_model_benchmarking(self, model_scores: Dict[str, Dict[str, float]]):
        """Add model benchmarking comparison to report."""
        if not self._report_available or not self._report_config.include_model_benchmarking:
            return

        section = {
            'type': 'model_benchmarking',
            'title': 'Model Benchmarking',
            'content': {
                'models': model_scores,
                'best_model': max(model_scores.items(), key=lambda x: x[1].get('cv_score', 0))[0] if model_scores else None,
            }
        }
        self._report_data['sections'].append(section)

    def add_confusion_matrix_report(self, y_true, y_pred, labels=None):
        """Add confusion matrix to report."""
        if not self._report_available or not self._report_config.include_confusion_matrix:
            return

        try:
            from sklearn.metrics import confusion_matrix, classification_report

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            report = classification_report(y_true, y_pred, labels=labels, output_dict=True)

            section = {
                'type': 'confusion_matrix',
                'title': 'Confusion Matrix & Classification Report',
                'content': {
                    'matrix': cm.tolist(),
                    'labels': labels.tolist() if hasattr(labels, 'tolist') else labels,
                    'classification_report': report,
                }
            }
            self._report_data['sections'].append(section)

            # Create confusion matrix figure
            self._create_confusion_matrix_figure(cm, labels)

        except Exception as e:
            logger.debug(f"Failed to add confusion matrix: {e}")

    def add_roc_analysis(self, y_true, y_prob, pos_label=1):
        """Add ROC curve analysis to report."""
        if not self._report_available or not self._report_config.include_roc_curves:
            return

        try:
            from sklearn.metrics import roc_curve, auc, roc_auc_score

            fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)

            section = {
                'type': 'roc_analysis',
                'title': 'ROC Curve Analysis',
                'content': {
                    'auc': roc_auc,
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'optimal_threshold': self._find_optimal_threshold(fpr, tpr, thresholds),
                }
            }
            self._report_data['sections'].append(section)

            # Create ROC figure
            self._create_roc_figure(fpr, tpr, roc_auc)

        except Exception as e:
            logger.debug(f"Failed to add ROC analysis: {e}")

    def add_precision_recall_analysis(self, y_true, y_prob, pos_label=1):
        """Add precision-recall curve analysis to report."""
        if not self._report_available or not self._report_config.include_precision_recall:
            return

        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score

            precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
            avg_precision = average_precision_score(y_true, y_prob, pos_label=pos_label)

            section = {
                'type': 'precision_recall',
                'title': 'Precision-Recall Analysis',
                'content': {
                    'average_precision': avg_precision,
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                }
            }
            self._report_data['sections'].append(section)

            # Create PR curve figure
            self._create_pr_figure(precision, recall, avg_precision)

        except Exception as e:
            logger.debug(f"Failed to add precision-recall analysis: {e}")

    def add_shap_analysis(self, shap_values, feature_names: List[str], X_sample=None):
        """Add SHAP analysis to report."""
        if not self._report_available or not self._report_config.include_shap_analysis:
            return

        try:
            import shap

            # Calculate mean absolute SHAP values per feature
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values

            mean_shap = np.abs(values).mean(axis=0)
            shap_importance = dict(zip(feature_names, mean_shap.tolist()))

            section = {
                'type': 'shap_analysis',
                'title': 'SHAP Feature Analysis',
                'content': {
                    'shap_importance': shap_importance,
                    'top_features': sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:20],
                }
            }
            self._report_data['sections'].append(section)

            # Create SHAP figures
            self._create_shap_figures(shap_values, feature_names, X_sample)

        except Exception as e:
            logger.debug(f"Failed to add SHAP analysis: {e}")

    def add_baseline_comparison(self, baseline_score: float, final_score: float,
                               baseline_model: str = "DummyClassifier"):
        """Add baseline vs final model comparison."""
        if not self._report_available or not self._report_config.include_baseline_comparison:
            return

        improvement = final_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

        section = {
            'type': 'baseline_comparison',
            'title': 'Baseline Comparison',
            'content': {
                'baseline_model': baseline_model,
                'baseline_score': baseline_score,
                'final_score': final_score,
                'improvement': improvement,
                'improvement_percent': improvement_pct,
            }
        }
        self._report_data['sections'].append(section)

    def add_recommendations(self, recommendations: List[str]):
        """Add recommendations section to report."""
        if not self._report_available or not self._report_config.include_recommendations:
            return

        section = {
            'type': 'recommendations',
            'title': 'Recommendations & Next Steps',
            'content': {
                'recommendations': recommendations,
            }
        }
        self._report_data['sections'].append(section)

    def generate_report(self, filename: str = None) -> Optional[str]:
        """
        Generate the final PDF report.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to generated PDF file, or None if generation failed
        """
        if not self._report_available:
            logger.warning("Report generation not available")
            return None

        try:
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"ml_report_{timestamp}.pdf"

            filepath = os.path.join(self._report_config.output_dir, filename)

            # Build PDF document
            doc = self._reportlab['SimpleDocTemplate'](
                filepath,
                pagesize=self._reportlab['letter'],
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            # Build story (content)
            story = self._build_report_story()

            # Generate PDF
            doc.build(story)

            logger.info(f"Report generated: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None

    def _build_report_story(self) -> List:
        """Build the report content (story) for reportlab."""
        styles = self._reportlab['getSampleStyleSheet']()
        story = []

        # Title page
        story.append(self._reportlab['Paragraph'](
            "Machine Learning Analysis Report",
            styles['Title']
        ))
        story.append(self._reportlab['Spacer'](1, 12))
        story.append(self._reportlab['Paragraph'](
            f"Generated by Jotty SwarmML Comprehensive v{self.version}",
            styles['Normal']
        ))
        story.append(self._reportlab['Paragraph'](
            f"Date: {self._report_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        story.append(self._reportlab['Spacer'](1, 24))

        # Process each section
        for section in self._report_data['sections']:
            story.extend(self._render_section(section, styles))
            story.append(self._reportlab['Spacer'](1, 12))

        # Add figures
        for fig_path in self._report_data['figures']:
            if os.path.exists(fig_path):
                try:
                    img = self._reportlab['Image'](fig_path, width=6*self._reportlab['inch'],
                                                   height=4*self._reportlab['inch'])
                    story.append(img)
                    story.append(self._reportlab['Spacer'](1, 12))
                except Exception:
                    pass

        return story

    def _render_section(self, section: Dict, styles) -> List:
        """Render a report section to reportlab elements."""
        elements = []

        # Section title
        elements.append(self._reportlab['Paragraph'](
            section['title'],
            styles['Heading1']
        ))
        elements.append(self._reportlab['Spacer'](1, 6))

        content = section['content']
        section_type = section['type']

        if section_type == 'executive_summary':
            elements.extend(self._render_executive_summary(content, styles))
        elif section_type == 'data_profile':
            elements.extend(self._render_data_profile(content, styles))
        elif section_type == 'feature_importance':
            elements.extend(self._render_feature_importance(content, styles))
        elif section_type == 'model_benchmarking':
            elements.extend(self._render_model_benchmarking(content, styles))
        elif section_type == 'confusion_matrix':
            elements.extend(self._render_confusion_matrix(content, styles))
        elif section_type == 'roc_analysis':
            elements.extend(self._render_roc_analysis(content, styles))
        elif section_type == 'precision_recall':
            elements.extend(self._render_precision_recall(content, styles))
        elif section_type == 'baseline_comparison':
            elements.extend(self._render_baseline_comparison(content, styles))
        elif section_type == 'recommendations':
            elements.extend(self._render_recommendations(content, styles))

        return elements

    def _render_executive_summary(self, content: Dict, styles) -> List:
        """Render executive summary section."""
        elements = []

        if content.get('context'):
            elements.append(self._reportlab['Paragraph'](
                f"<b>Context:</b> {content['context']}",
                styles['Normal']
            ))

        elements.append(self._reportlab['Paragraph'](
            f"<b>Final Score:</b> {content.get('final_score', 0):.4f}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Best Model:</b> {content.get('best_model', 'N/A')}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Features Used:</b> {content.get('n_features_used', 0)}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Iterations:</b> {content.get('iterations', 0)}",
            styles['Normal']
        ))

        if content.get('key_findings'):
            elements.append(self._reportlab['Spacer'](1, 6))
            elements.append(self._reportlab['Paragraph']("<b>Key Findings:</b>", styles['Normal']))
            for finding in content['key_findings']:
                elements.append(self._reportlab['Paragraph'](f"• {finding}", styles['Normal']))

        return elements

    def _render_data_profile(self, content: Dict, styles) -> List:
        """Render data profile section."""
        elements = []

        shape = content.get('shape', {})
        elements.append(self._reportlab['Paragraph'](
            f"<b>Dataset Shape:</b> {shape.get('rows', 0)} rows × {shape.get('columns', 0)} columns",
            styles['Normal']
        ))

        if content.get('missing'):
            missing = content['missing']
            if any(v > 0 for v in missing.values()):
                elements.append(self._reportlab['Paragraph']("<b>Missing Values:</b>", styles['Normal']))
                for col, count in list(missing.items())[:10]:
                    if count > 0:
                        elements.append(self._reportlab['Paragraph'](f"  • {col}: {count}", styles['Normal']))

        if content.get('recommendations'):
            elements.append(self._reportlab['Spacer'](1, 6))
            elements.append(self._reportlab['Paragraph']("<b>EDA Recommendations:</b>", styles['Normal']))
            for rec in content['recommendations'][:5]:
                elements.append(self._reportlab['Paragraph'](f"• {rec}", styles['Normal']))

        return elements

    def _render_feature_importance(self, content: Dict, styles) -> List:
        """Render feature importance section."""
        elements = []

        ranking = content.get('importance_ranking', [])
        elements.append(self._reportlab['Paragraph'](
            f"<b>Total Features:</b> {content.get('total_features', 0)}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Top {content.get('top_n_shown', 20)} Features:</b>",
            styles['Normal']
        ))

        # Create table
        table_data = [['Rank', 'Feature', 'Importance']]
        for rank, (feat, imp) in enumerate(ranking[:15], 1):
            table_data.append([str(rank), feat[:30], f"{imp:.4f}"])

        table = self._reportlab['Table'](table_data, colWidths=[40, 250, 80])
        table.setStyle(self._reportlab['TableStyle']([
            ('BACKGROUND', (0, 0), (-1, 0), self._reportlab['colors'].grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), self._reportlab['colors'].whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, self._reportlab['colors'].black),
        ]))
        elements.append(table)

        return elements

    def _render_model_benchmarking(self, content: Dict, styles) -> List:
        """Render model benchmarking section."""
        elements = []

        models = content.get('models', {})
        if not models:
            return elements

        elements.append(self._reportlab['Paragraph'](
            f"<b>Best Model:</b> {content.get('best_model', 'N/A')}",
            styles['Normal']
        ))

        # Create comparison table
        table_data = [['Model', 'CV Score', 'Std', 'Train Time']]
        for model_name, scores in models.items():
            table_data.append([
                model_name[:25],
                f"{scores.get('cv_score', 0):.4f}",
                f"{scores.get('cv_std', 0):.4f}",
                f"{scores.get('train_time', 0):.2f}s"
            ])

        table = self._reportlab['Table'](table_data, colWidths=[150, 80, 80, 80])
        table.setStyle(self._reportlab['TableStyle']([
            ('BACKGROUND', (0, 0), (-1, 0), self._reportlab['colors'].grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), self._reportlab['colors'].whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, self._reportlab['colors'].black),
        ]))
        elements.append(table)

        return elements

    def _render_confusion_matrix(self, content: Dict, styles) -> List:
        """Render confusion matrix section."""
        elements = []

        report = content.get('classification_report', {})
        if report:
            elements.append(self._reportlab['Paragraph']("<b>Classification Metrics:</b>", styles['Normal']))

            for cls, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    elements.append(self._reportlab['Paragraph'](
                        f"  Class {cls}: Precision={metrics['precision']:.3f}, "
                        f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}",
                        styles['Normal']
                    ))

        return elements

    def _render_roc_analysis(self, content: Dict, styles) -> List:
        """Render ROC analysis section."""
        elements = []

        elements.append(self._reportlab['Paragraph'](
            f"<b>AUC Score:</b> {content.get('auc', 0):.4f}",
            styles['Normal']
        ))

        if content.get('optimal_threshold'):
            elements.append(self._reportlab['Paragraph'](
                f"<b>Optimal Threshold:</b> {content['optimal_threshold']:.4f}",
                styles['Normal']
            ))

        return elements

    def _render_precision_recall(self, content: Dict, styles) -> List:
        """Render precision-recall analysis section."""
        elements = []

        elements.append(self._reportlab['Paragraph'](
            f"<b>Average Precision:</b> {content.get('average_precision', 0):.4f}",
            styles['Normal']
        ))

        return elements

    def _render_baseline_comparison(self, content: Dict, styles) -> List:
        """Render baseline comparison section."""
        elements = []

        elements.append(self._reportlab['Paragraph'](
            f"<b>Baseline Model:</b> {content.get('baseline_model', 'N/A')}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Baseline Score:</b> {content.get('baseline_score', 0):.4f}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Final Score:</b> {content.get('final_score', 0):.4f}",
            styles['Normal']
        ))
        elements.append(self._reportlab['Paragraph'](
            f"<b>Improvement:</b> {content.get('improvement', 0):.4f} "
            f"({content.get('improvement_percent', 0):.1f}%)",
            styles['Normal']
        ))

        return elements

    def _render_recommendations(self, content: Dict, styles) -> List:
        """Render recommendations section."""
        elements = []

        for rec in content.get('recommendations', []):
            elements.append(self._reportlab['Paragraph'](f"• {rec}", styles['Normal']))

        return elements

    def _create_importance_figure(self, sorted_importance: List[Tuple[str, float]]):
        """Create and save feature importance figure."""
        try:
            import matplotlib.pyplot as plt
            import tempfile

            features = [x[0] for x in sorted_importance[:20]]
            values = [x[1] for x in sorted_importance[:20]]

            plt.figure(figsize=(10, 8))
            plt.barh(range(len(features)), values[::-1], color='steelblue')
            plt.yticks(range(len(features)), features[::-1], fontsize=8)
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importance')
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create importance figure: {e}")

    def _create_confusion_matrix_figure(self, cm, labels):
        """Create and save confusion matrix figure."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            import tempfile

            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap='Blues', ax=plt.gca())
            plt.title('Confusion Matrix')
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create confusion matrix figure: {e}")

    def _create_roc_figure(self, fpr, tpr, roc_auc):
        """Create and save ROC curve figure."""
        try:
            import matplotlib.pyplot as plt
            import tempfile

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create ROC figure: {e}")

    def _create_pr_figure(self, precision, recall, avg_precision):
        """Create and save precision-recall curve figure."""
        try:
            import matplotlib.pyplot as plt
            import tempfile

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'PR curve (AP = {avg_precision:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create PR figure: {e}")

    def _create_shap_figures(self, shap_values, feature_names, X_sample):
        """Create and save SHAP analysis figures."""
        try:
            import shap
            import matplotlib.pyplot as plt
            import tempfile

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                            plot_type="bar", show=False)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png',
                                            dir=self._report_config.output_dir) as f:
                plt.savefig(f.name, dpi=self._report_config.figure_dpi, bbox_inches='tight')
                self._report_data['figures'].append(f.name)
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create SHAP figures: {e}")

    def _calculate_improvement(self) -> float:
        """Calculate score improvement over iterations."""
        history = self._learning_state.score_history
        if len(history) < 2:
            return 0.0
        return history[-1] - history[0]

    def _extract_key_findings(self, results: Dict) -> List[str]:
        """Extract key findings from results for executive summary."""
        findings = []

        # Best score finding
        score = results.get('final_score', 0)
        if score > 0.9:
            findings.append(f"Achieved excellent performance with {score:.1%} accuracy")
        elif score > 0.8:
            findings.append(f"Achieved good performance with {score:.1%} accuracy")
        else:
            findings.append(f"Model achieved {score:.1%} accuracy")

        # Feature importance finding
        importance = results.get('feature_importance', {})
        if importance:
            top_feat = max(importance.items(), key=lambda x: x[1])
            findings.append(f"Most important feature: {top_feat[0]} ({top_feat[1]:.2%} importance)")

        # Improvement finding
        improvement = self._calculate_improvement()
        if improvement > 0.01:
            findings.append(f"Improved {improvement:.1%} through iterative refinement")

        # Learning finding
        n_patterns = len(self._learning_state.learned_patterns)
        if n_patterns > 0:
            findings.append(f"Leveraged {n_patterns} learned patterns from historical sessions")

        return findings

    def _find_optimal_threshold(self, fpr, tpr, thresholds) -> float:
        """Find optimal classification threshold using Youden's J statistic."""
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    # =========================================================================
    # PROFESSIONAL PDF REPORT (Pandoc + LaTeX)
    # =========================================================================

    def generate_professional_report(
        self,
        results: Dict[str, Any],
        y_true=None,
        y_pred=None,
        y_prob=None,
        X_sample=None,
        shap_values=None,
        title: str = "ML Analysis Report",
        context: str = "",
        filename: str = None
    ) -> Optional[str]:
        """
        Generate a professional-grade PDF report using Pandoc + LaTeX.

        This creates publication-quality reports with:
        - Elegant typography
        - Professional visualizations
        - Proper tables and charts
        - Table of contents

        Args:
            results: Dict with keys like 'metrics', 'model_scores', 'feature_importance', etc.
            y_true: True labels for classification metrics
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            X_sample: Sample data for SHAP analysis
            shap_values: Pre-computed SHAP values
            title: Report title
            context: Business context description
            filename: Output filename

        Returns:
            Path to generated PDF or None if failed
        """
        try:
            from .ml_report_generator import ProfessionalMLReport

            # Use configured output dir or default
            output_dir = getattr(self, '_report_config', ReportConfig()).output_dir

            report = ProfessionalMLReport(output_dir=output_dir)

            # Set metadata
            report.set_metadata(
                title=title,
                subtitle=context[:100] if context else "Automated ML Analysis",
                author="Jotty SwarmMLComprehensive",
                dataset=results.get('dataset', 'Unknown'),
                problem_type=results.get('problem_type', 'Classification')
            )

            # Extract components from results
            metrics = results.get('metrics', {})
            if not metrics and 'final_score' in results:
                metrics = {'accuracy': results['final_score']}

            best_model = results.get('best_model', 'Unknown')
            feature_importance = results.get('feature_importance', {})
            model_scores = results.get('model_scores', {})
            n_features = results.get('n_features', len(feature_importance))

            # Add sections
            report.add_executive_summary(
                metrics=metrics,
                best_model=best_model,
                n_features=n_features,
                context=context
            )

            # Data profile if available
            if 'data_profile' in results:
                dp = results['data_profile']
                report.add_data_profile(
                    shape=dp.get('shape', (0, n_features)),
                    dtypes=dp.get('dtypes', {}),
                    missing=dp.get('missing', {}),
                    recommendations=dp.get('recommendations', [])
                )

            # Feature importance
            if feature_importance:
                report.add_feature_importance(feature_importance)

            # Model benchmarking
            if model_scores:
                report.add_model_benchmarking(model_scores)

            # Classification metrics
            if y_true is not None and y_pred is not None:
                labels = results.get('labels', ['Class 0', 'Class 1'])
                report.add_confusion_matrix(y_true, y_pred, labels)

                if y_prob is not None:
                    report.add_roc_analysis(y_true, y_prob)
                    report.add_precision_recall(y_true, y_prob)

            # SHAP analysis
            if shap_values is not None and X_sample is not None:
                feature_names = list(feature_importance.keys()) if feature_importance else []
                report.add_shap_analysis(shap_values, feature_names, X_sample)

            # Baseline comparison
            if 'baseline_score' in results:
                report.add_baseline_comparison(
                    results['baseline_score'],
                    metrics.get('accuracy', results.get('final_score', 0))
                )

            # Recommendations
            recommendations = results.get('recommendations', [
                f"Best model {best_model} achieved strong performance",
                "Consider hyperparameter tuning for further improvement",
                "Monitor model performance over time for drift",
                "Regular retraining recommended as data patterns evolve"
            ])
            report.add_recommendations(recommendations)

            # Generate PDF
            pdf_path = report.generate(filename)

            if pdf_path:
                logger.info(f"Professional report generated: {pdf_path}")

            return pdf_path

        except ImportError as e:
            logger.warning(f"Professional report generator not available: {e}")
            # Fallback to basic report
            return self.generate_report(filename)
        except Exception as e:
            logger.error(f"Failed to generate professional report: {e}")
            return None

    def generate_world_class_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        results: Dict[str, Any],
        y_pred=None,
        y_prob=None,
        shap_values=None,
        title: str = "Comprehensive ML Analysis",
        context: str = "",
        filename: str = None,
        include_all: bool = True,
        theme: str = "professional",
        # New parameters
        generate_html: bool = False,
        llm_narrative: bool = False,
        sensitive_features: Dict[str, Any] = None,
        X_reference=None,
        pipeline_steps: List[Dict] = None,
        study_or_trials=None,
        validation_datasets: Dict = None,
    ) -> Optional[str]:
        """
        Generate the world's most comprehensive ML report.

        Themes:
        - 'professional': Modern blue, sans-serif, bold headers (default)
        - 'goldman': Goldman Sachs style, navy/serif, uppercase headers, institutional

        Includes ALL advanced analysis:
        - Data Quality Analysis (outliers, missing patterns, distributions)
        - Correlation & Multicollinearity (VIF analysis)
        - Learning Curves & Bias-Variance Analysis
        - Calibration Analysis
        - Lift & Gain Charts (KS statistic)
        - Cross-Validation Detailed Analysis
        - Error Analysis (misclassification patterns)
        - SHAP Deep Dive (summary, dependence, waterfall)
        - Threshold Optimization (cost-benefit)
        - Full Reproducibility Section
        - Data Drift Monitoring (if X_reference provided)
        - Fairness & Bias Audit (if sensitive_features provided)
        - Hyperparameter Search Visualization (if study_or_trials provided)
        - Multi-Dataset Validation (if validation_datasets provided)
        - Confidence-Calibrated Predictions
        - Pipeline DAG Visualization (if pipeline_steps provided)
        - Deep Learning Analysis (if model is neural network)
        - LLM-Generated Narrative Insights (if llm_narrative=True)
        - Interactive HTML Report (if generate_html=True)

        Args:
            X: Feature DataFrame
            y: Target Series
            model: Trained model
            results: Results dictionary
            y_pred: Predictions
            y_prob: Probabilities
            shap_values: SHAP values
            title: Report title
            context: Business context
            filename: Output filename
            include_all: Include all sections (True) or only essential (False)
            theme: Report theme ('professional' or 'goldman')
            generate_html: Also generate interactive HTML report
            llm_narrative: Enable LLM-generated narrative insights
            sensitive_features: Dict mapping feature_name -> column data for fairness audit
            X_reference: Reference dataset for drift analysis
            pipeline_steps: List of pipeline step dicts for DAG visualization
            study_or_trials: Optuna study or List[Dict] for hyperparameter viz
            validation_datasets: Dict[name -> (X, y)] for cross-dataset validation

        Returns:
            Path to generated PDF
        """
        try:
            from .ml_report_generator import ProfessionalMLReport

            # Use configured output dir
            output_dir = getattr(self, '_report_config', ReportConfig()).output_dir
            report = ProfessionalMLReport(output_dir=output_dir, theme=theme,
                                          llm_narrative=llm_narrative, html_enabled=generate_html)

            logger.info(f"  Theme: {report.theme['name']}")

            # Set metadata
            report.set_metadata(
                title=title,
                subtitle=context[:100] if context else "World-Class ML Analysis",
                author="Jotty SwarmMLComprehensive",
                dataset=results.get('dataset', 'Custom Dataset'),
                problem_type=results.get('problem_type', 'Classification')
            )

            # Extract components
            metrics = results.get('metrics', {})
            if not metrics and 'final_score' in results:
                metrics = {'accuracy': results['final_score']}

            best_model = str(results.get('best_model', type(model).__name__))
            feature_importance = results.get('feature_importance', {})
            model_scores = results.get('model_scores', {})
            feature_names = list(X.columns)

            problem_type = results.get('problem_type', 'Classification')
            is_regression = problem_type.lower() == 'regression'

            logger.info("Generating world-class comprehensive report...")

            # ==== SECTION 1: EXECUTIVE SUMMARY ====
            report.add_executive_summary(
                metrics=metrics,
                best_model=best_model,
                n_features=len(feature_names),
                context=context
            )

            # ==== SECTION 1.5: EXECUTIVE DASHBOARD (NEW) ====
            if include_all and metrics:
                logger.info("  - Adding executive dashboard...")
                try:
                    report.add_executive_dashboard(
                        metrics=metrics,
                        model_name=best_model,
                        dataset_name=results.get('dataset', '')
                    )
                except Exception as e:
                    logger.debug(f"Executive dashboard failed: {e}")

            # ==== SECTION 2: DATA QUALITY ANALYSIS ====
            if include_all:
                logger.info("  - Adding data quality analysis...")
                report.add_data_quality_analysis(X, y)

            # ==== SECTION 2.5: CLASS DISTRIBUTION (NEW) ====
            if include_all and not is_regression:
                logger.info("  - Adding class distribution analysis...")
                try:
                    labels = results.get('labels', None)
                    report.add_class_distribution(y, y_pred, labels)
                except Exception as e:
                    logger.debug(f"Class distribution failed: {e}")

            # ==== SECTION 3: CORRELATION ANALYSIS ====
            if include_all and len(X.select_dtypes(include=[np.number]).columns) >= 2:
                logger.info("  - Adding correlation analysis...")
                report.add_correlation_analysis(X)

            # ==== SECTION 4: DATA PROFILE ====
            if 'data_profile' in results:
                dp = results['data_profile']
                report.add_data_profile(
                    shape=dp.get('shape', X.shape),
                    dtypes=dp.get('dtypes', dict(X.dtypes.value_counts())),
                    missing=dp.get('missing', dict(X.isnull().sum())),
                    recommendations=dp.get('recommendations', [])
                )

            # ==== SECTION 4.5: PIPELINE DAG VISUALIZATION (NEW) ====
            if include_all and pipeline_steps:
                logger.info("  - Adding pipeline visualization...")
                try:
                    report.add_pipeline_visualization(pipeline_steps)
                except Exception as e:
                    logger.debug(f"Pipeline visualization failed: {e}")

            # ==== SECTION 5: FEATURE IMPORTANCE ====
            if feature_importance:
                report.add_feature_importance(feature_importance)

            # ==== SECTION 5.5: PERMUTATION IMPORTANCE (NEW) ====
            if include_all and model is not None:
                logger.info("  - Adding permutation importance...")
                try:
                    report.add_permutation_importance(model, X, y)
                except Exception as e:
                    logger.debug(f"Permutation importance failed: {e}")

            # ==== SECTION 5.7: PARTIAL DEPENDENCE PLOTS (NEW) ====
            if include_all and model is not None:
                logger.info("  - Adding partial dependence plots...")
                try:
                    report.add_partial_dependence(model, X, feature_names)
                except Exception as e:
                    logger.debug(f"Partial dependence failed: {e}")

            # ==== SECTION 6: MODEL BENCHMARKING ====
            if model_scores:
                report.add_model_benchmarking(model_scores)

            # ==== SECTION 6.5: MULTI-DATASET VALIDATION (NEW) ====
            if include_all and validation_datasets:
                logger.info("  - Adding cross-dataset validation...")
                try:
                    report.add_cross_dataset_validation(validation_datasets, model)
                except Exception as e:
                    logger.debug(f"Cross-dataset validation failed: {e}")

            # ==== SECTION 7: LEARNING CURVES ====
            if include_all and model is not None:
                logger.info("  - Adding learning curves...")
                try:
                    report.add_learning_curves(model, X, y)
                except Exception as e:
                    logger.debug(f"Learning curves failed: {e}")

            # ==== SECTION 8: CV DETAILED ANALYSIS ====
            if include_all and model is not None:
                logger.info("  - Adding CV analysis...")
                try:
                    report.add_cv_detailed_analysis(model, X, y)
                except Exception as e:
                    logger.debug(f"CV analysis failed: {e}")

            # ==== SECTION 8.5: STATISTICAL SIGNIFICANCE (NEW) ====
            if include_all and y_pred is not None:
                logger.info("  - Adding statistical significance tests...")
                try:
                    report.add_statistical_tests(y, y_pred, y_prob)
                except Exception as e:
                    logger.debug(f"Statistical tests failed: {e}")

            # ==== SECTION 9/9R: CLASSIFICATION METRICS or REGRESSION ANALYSIS ====
            if is_regression and y_pred is not None:
                # ==== SECTION 9R: REGRESSION ANALYSIS (NEW) ====
                logger.info("  - Adding regression analysis...")
                try:
                    report.add_regression_analysis(y, y_pred)
                except Exception as e:
                    logger.debug(f"Regression analysis failed: {e}")
            elif y_pred is not None:
                labels = results.get('labels', ['Negative', 'Positive'])
                report.add_confusion_matrix(y, y_pred, labels)

                if y_prob is not None:
                    report.add_roc_analysis(y, y_prob)
                    report.add_precision_recall(y, y_prob)

                    # ==== SECTION 10: CALIBRATION ====
                    if include_all:
                        logger.info("  - Adding calibration analysis...")
                        report.add_calibration_analysis(y, y_prob)

                    # ==== SECTION 10.5: CONFIDENCE-CALIBRATED PREDICTIONS (NEW) ====
                    if include_all:
                        logger.info("  - Adding prediction confidence analysis...")
                        try:
                            report.add_prediction_confidence_analysis(X, y, y_pred, y_prob)
                        except Exception as e:
                            logger.debug(f"Prediction confidence analysis failed: {e}")

                    # ==== SECTION 11: LIFT & GAIN ====
                    if include_all:
                        logger.info("  - Adding lift/gain analysis...")
                        report.add_lift_gain_analysis(y, y_prob)

                    # ==== SECTION 12: THRESHOLD OPTIMIZATION ====
                    if include_all:
                        logger.info("  - Adding threshold optimization...")
                        report.add_threshold_optimization(y, y_prob)

                    # ==== SECTION 12.5: SCORE DISTRIBUTION (NEW) ====
                    if include_all:
                        logger.info("  - Adding score distribution...")
                        try:
                            report.add_score_distribution(y, y_prob, labels)
                        except Exception as e:
                            logger.debug(f"Score distribution failed: {e}")

                # ==== SECTION 13: ERROR ANALYSIS ====
                if include_all:
                    logger.info("  - Adding error analysis...")
                    report.add_error_analysis(X, y, y_pred, y_prob)

            # ==== SECTION 13.5: DATA DRIFT MONITORING (NEW) ====
            if include_all and X_reference is not None:
                logger.info("  - Adding drift analysis...")
                try:
                    report.add_drift_analysis(X_reference, X,
                                              feature_importance=feature_importance)
                except Exception as e:
                    logger.debug(f"Drift analysis failed: {e}")

            # ==== SECTION 14: SHAP DEEP DIVE ====
            if shap_values is not None:
                logger.info("  - Adding SHAP deep analysis...")
                X_array = X.values if hasattr(X, 'values') else X
                report.add_shap_deep_analysis(shap_values, feature_names, X_array, model)

                # ==== SECTION 14.5: FEATURE INTERACTIONS (NEW) ====
                if include_all:
                    logger.info("  - Adding feature interactions...")
                    try:
                        report.add_feature_interactions(
                            shap_values, feature_names, X_array, model
                        )
                    except Exception as e:
                        logger.debug(f"Feature interactions failed: {e}")

            # ==== SECTION 14.6: INTERPRETABILITY ANALYSIS (NEW) ====
            if include_all and model is not None and y_pred is not None:
                logger.info("  - Adding interpretability analysis...")
                try:
                    X_array = X.values if hasattr(X, 'values') else X
                    report.add_interpretability_analysis(model, X_array, y_pred, feature_names, top_n=5)
                except Exception as e:
                    logger.debug(f"Interpretability analysis failed: {e}")

            # ==== SECTION 14.7: DEEP LEARNING ANALYSIS (NEW) ====
            if include_all and model is not None:
                logger.info("  - Adding deep learning analysis...")
                try:
                    X_array = X.values if hasattr(X, 'values') else X
                    report.add_deep_learning_analysis(
                        model, X_array,
                        training_history=results.get('training_history')
                    )
                except Exception as e:
                    logger.debug(f"Deep learning analysis failed: {e}")

            # ==== SECTION 15: BASELINE COMPARISON ====
            if 'baseline_score' in results:
                report.add_baseline_comparison(
                    results['baseline_score'],
                    metrics.get('accuracy', results.get('final_score', 0))
                )

            # ==== SECTION 16: RECOMMENDATIONS ====
            recommendations = self._generate_smart_recommendations(results, model, feature_importance)
            report.add_recommendations(recommendations)

            # ==== SECTION 17: REPRODUCIBILITY ====
            if include_all:
                logger.info("  - Adding reproducibility section...")
                report.add_reproducibility_section(
                    model,
                    params=results.get('best_params', {}),
                    random_state=42
                )

            # ==== SECTION 17.2: DEPLOYMENT READINESS (NEW) ====
            if include_all and model is not None:
                logger.info("  - Adding deployment readiness...")
                try:
                    X_array = X.values if hasattr(X, 'values') else X
                    report.add_deployment_readiness(model, X_array)
                except Exception as e:
                    logger.debug(f"Deployment readiness failed: {e}")

            # ==== SECTION 17.5: HYPERPARAMETER SEARCH VIZ (NEW) ====
            if include_all and study_or_trials is not None:
                logger.info("  - Adding hyperparameter visualization...")
                try:
                    report.add_hyperparameter_visualization(study_or_trials)
                except Exception as e:
                    logger.debug(f"Hyperparameter visualization failed: {e}")

            # ==== SECTION 18: MODEL CARD (NEW) ====
            if include_all:
                logger.info("  - Adding model card...")
                try:
                    report.add_model_card(
                        model=model,
                        results=results,
                        intended_use=results.get('intended_use', ''),
                        limitations=results.get('limitations', ''),
                        ethical=results.get('ethical_considerations', '')
                    )
                except Exception as e:
                    logger.debug(f"Model card failed: {e}")

            # ==== SECTION 18.5: FAIRNESS & BIAS AUDIT (NEW) ====
            if include_all and sensitive_features is not None:
                logger.info("  - Adding fairness audit...")
                try:
                    report.add_fairness_audit(X, y, y_pred, y_prob, sensitive_features)
                except Exception as e:
                    logger.debug(f"Fairness audit failed: {e}")

            # ==== INSIGHT PRIORITIZATION (LAST — reads all section data) ====
            if include_all:
                logger.info("  - Adding insight prioritization...")
                try:
                    report.add_insight_prioritization()
                except Exception as e:
                    logger.debug(f"Insight prioritization failed: {e}")

            # Generate PDF
            logger.info("  - Generating PDF...")
            pdf_path = report.generate(filename)

            if pdf_path:
                logger.info(f"World-class report generated: {pdf_path}")
                # Get file size
                import os
                size_kb = os.path.getsize(pdf_path) / 1024
                logger.info(f"  - Size: {size_kb:.1f} KB")

            # Generate HTML report if requested
            if generate_html:
                logger.info("  - Generating interactive HTML report...")
                html_filename = filename.replace('.pdf', '.html') if filename else None
                html_path = report.generate_html(html_filename)
                if html_path:
                    logger.info(f"  - HTML report: {html_path}")

            return pdf_path

        except Exception as e:
            logger.error(f"Failed to generate world-class report: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_smart_recommendations(self, results: Dict, model, feature_importance: Dict) -> List[str]:
        """Generate intelligent recommendations based on results."""
        recommendations = []

        # Model performance
        score = results.get('final_score', results.get('metrics', {}).get('accuracy', 0))
        if score >= 0.95:
            recommendations.append(f"Excellent performance ({score:.1%}) achieved - monitor for overfitting")
        elif score >= 0.85:
            recommendations.append(f"Good performance ({score:.1%}) - consider ensemble methods for improvement")
        elif score >= 0.75:
            recommendations.append(f"Moderate performance ({score:.1%}) - feature engineering may help")
        else:
            recommendations.append(f"Performance needs improvement ({score:.1%}) - consider more complex models or better features")

        # Model type recommendation
        model_name = type(model).__name__ if model else 'Unknown'
        if 'Logistic' in model_name:
            recommendations.append("Logistic Regression provides good interpretability - ideal for regulated industries")
        elif 'Forest' in model_name or 'Gradient' in model_name:
            recommendations.append("Tree-based model captures non-linear patterns well")

        # Feature importance insights
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            top_names = [f[0] for f in top_features]
            recommendations.append(f"Top predictive features: {', '.join(top_names)}")

            # Concentration check
            total_imp = sum(feature_importance.values())
            top3_imp = sum(f[1] for f in top_features)
            if top3_imp / total_imp > 0.6:
                recommendations.append("High feature importance concentration - model relies heavily on few features")

        # AUC-based recommendation
        auc = results.get('metrics', {}).get('auc_roc', results.get('auc_roc', 0))
        if auc > 0:
            if auc >= 0.9:
                recommendations.append(f"Excellent discrimination (AUC={auc:.3f}) - suitable for production")
            elif auc >= 0.8:
                recommendations.append(f"Good discrimination (AUC={auc:.3f}) - threshold tuning recommended")
            else:
                recommendations.append(f"Moderate discrimination (AUC={auc:.3f}) - consider feature engineering")

        # General recommendations
        recommendations.extend([
            "Monitor model performance over time for concept drift",
            "Validate on held-out data before production deployment",
            "Document model decisions for regulatory compliance",
        ])

        return recommendations

    # =========================================================================
    # TELEGRAM NOTIFICATION
    # =========================================================================

    def init_telegram(self, config: TelegramConfig = None):
        """
        Initialize Telegram notifications.

        Args:
            config: Telegram configuration (uses env vars if None)
        """
        self._telegram_config = config or TelegramConfig()
        self._telegram_available = False

        if not self._telegram_config.enabled:
            return

        # Get credentials from env if not provided
        if not self._telegram_config.bot_token:
            self._telegram_config.bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
        if not self._telegram_config.chat_id:
            self._telegram_config.chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')

        if not self._telegram_config.bot_token or not self._telegram_config.chat_id:
            logger.warning("Telegram credentials not found. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.")
            return

        self._telegram_available = True
        logger.info("Telegram notifications initialized")

    def send_telegram_report(self, report_path: str, results: Dict[str, Any] = None,
                            caption: str = None) -> bool:
        """
        Send the PDF report to Telegram.

        Args:
            report_path: Path to the PDF report
            results: Optional results dict for summary message
            caption: Optional custom caption

        Returns:
            True if sent successfully
        """
        if not self._telegram_available:
            logger.warning("Telegram not available")
            return False

        try:
            import requests

            bot_token = self._telegram_config.bot_token
            chat_id = self._telegram_config.chat_id

            # Build caption
            if caption is None:
                caption = self._build_telegram_caption(results)

            # Send document
            if self._telegram_config.send_report_pdf and report_path and os.path.exists(report_path):
                url = f"https://api.telegram.org/bot{bot_token}/sendDocument"

                with open(report_path, 'rb') as f:
                    files = {'document': f}
                    data = {
                        'chat_id': chat_id,
                        'caption': caption[:1024],  # Telegram caption limit
                        'parse_mode': 'HTML'
                    }
                    response = requests.post(url, files=files, data=data, timeout=60)

                if response.status_code == 200:
                    logger.info(f"Report sent to Telegram: {report_path}")
                    return True
                else:
                    logger.error(f"Telegram send failed: {response.text}")
                    return False

            # Send summary message only
            elif self._telegram_config.send_summary_message:
                return self.send_telegram_message(caption)

            return False

        except ImportError:
            logger.warning("requests library not installed for Telegram")
            return False
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_telegram_message(self, message: str) -> bool:
        """
        Send a text message to Telegram.

        Args:
            message: Message text (supports HTML)

        Returns:
            True if sent successfully
        """
        if not self._telegram_available:
            return False

        try:
            import requests

            bot_token = self._telegram_config.bot_token
            chat_id = self._telegram_config.chat_id

            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message[:4096],  # Telegram message limit
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data, timeout=30)

            if response.status_code == 200:
                logger.info("Message sent to Telegram")
                return True
            else:
                logger.error(f"Telegram message failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Telegram message failed: {e}")
            return False

    def _build_telegram_caption(self, results: Dict[str, Any] = None) -> str:
        """Build caption for Telegram message."""
        lines = [
            "<b>🤖 ML Analysis Report</b>",
            f"<i>Generated by {self.name} v{self.version}</i>",
            "",
        ]

        if results:
            # Add metrics
            if self._telegram_config.include_metrics_in_message:
                score = results.get('final_score', 0)
                model = results.get('best_model', 'N/A')

                lines.append(f"📊 <b>Results:</b>")
                lines.append(f"  • Score: <code>{score:.4f}</code>")
                lines.append(f"  • Model: <code>{model}</code>")

                # Add other metrics if available
                for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
                    if key in results:
                        lines.append(f"  • {key.title()}: <code>{results[key]:.4f}</code>")

                lines.append("")

            # Add top features
            if self._telegram_config.include_feature_importance:
                importance = results.get('feature_importance', {})
                if importance:
                    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    top_n = self._telegram_config.max_features_in_message

                    lines.append(f"🎯 <b>Top {top_n} Features:</b>")
                    for feat, imp in sorted_imp[:top_n]:
                        lines.append(f"  • {feat[:20]}: <code>{imp:.3f}</code>")

        lines.append("")
        lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

MLComprehensive = SwarmMLComprehensive
MLLearning = SwarmMLComprehensive
SwarmMLLearning = SwarmMLComprehensive
