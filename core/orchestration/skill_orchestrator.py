"""
SkillOrchestrator - Auto-chain Skills for Any ML Task
======================================================

Automatically discovers, sequences, and executes ML skills
to solve ANY machine learning problem.

Usage:
    orchestrator = SkillOrchestrator()
    result = await orchestrator.solve(X, y)

The orchestrator:
1. Auto-detects problem type (classification/regression)
2. Discovers relevant skills from registry
3. Chains skills in optimal order
4. Executes pipeline and returns best model

Features:
- Progress tracking with visual progress bar
- LLM-powered feature reasoning from multiple perspectives

Components:
- skill_types.py: ProblemType, SkillCategory, SkillResult, PipelineResult, ProgressTracker, SkillAdapter
- core/skills/ml/eda.py: EDASkill (imported as EDAAnalyzer)
- core/skills/ml/llm_reasoner.py: LLMFeatureReasonerSkill (imported as LLMFeatureReasoner)
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd

# Import from extracted modules
from .skill_types import (
    ProblemType, SkillCategory, SkillResult, PipelineResult,
    ProgressTracker, SkillAdapter
)

# Import ML skills from proper location (core/skills/ml/)
try:
    from Jotty.core.skills.ml.eda import EDASkill as EDAAnalyzer
    from Jotty.core.skills.ml.llm_reasoner import LLMFeatureReasonerSkill as LLMFeatureReasoner
except ImportError:
    # Fallback: use inline implementations if skills not available
    EDAAnalyzer = None
    LLMFeatureReasoner = None

logger = logging.getLogger(__name__)


# ============================================================
# SKILL ORCHESTRATOR
# ============================================================

from ._feature_engineering_mixin import FeatureEngineeringMixin
from ._feature_selection_mixin import FeatureSelectionMixin
from ._model_pipeline_mixin import ModelPipelineMixin


class SkillOrchestrator(FeatureEngineeringMixin, FeatureSelectionMixin, ModelPipelineMixin):
    """
    Orchestrates ML skills to solve any machine learning problem.

    Automatically:
    - Detects problem type
    - Discovers relevant skills
    - Chains skills in optimal order
    - Executes pipeline
    - Returns best model
    """

    # Optimal skill execution order
    PIPELINE_ORDER = [
        SkillCategory.DATA_PROFILING,
        SkillCategory.DATA_CLEANING,
        SkillCategory.FEATURE_ENGINEERING,
        SkillCategory.FEATURE_SELECTION,
        SkillCategory.MODEL_SELECTION,
        SkillCategory.HYPERPARAMETER_OPTIMIZATION,
        SkillCategory.ENSEMBLE,
        SkillCategory.EVALUATION,
        SkillCategory.EXPLANATION,
    ]

    def __init__(self, use_llm_features: bool = True, show_progress: bool = True) -> None:
        self._skills_registry = None
        self._tools_registry = None
        self._skill_adapters: Dict[str, SkillAdapter] = {}
        self._initialized = False
        self._use_llm_features = use_llm_features and LLMFeatureReasoner is not None
        self._show_progress = show_progress
        self._llm_reasoner = LLMFeatureReasoner() if self._use_llm_features else None
        self._progress = None

    @staticmethod
    def _drop_target_leaks(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Drop features with near-perfect correlation to the target.

        Detects columns that are trivial transformations of y (e.g. the
        ``alive`` column in seaborn's Titanic dataset which is just
        ``survived`` spelled out).  Threshold: |correlation| > 0.95.
        """
        try:
            corr = X.corrwith(y).abs()
            leaky = corr[corr > 0.95].index.tolist()
            if leaky:
                logger.info(f"  Dropped {len(leaky)} target-leaking columns: {leaky}")
                X = X.drop(columns=leaky)
        except Exception:
            pass
        return X

    @staticmethod
    def _ensure_numeric(X: pd.DataFrame) -> pd.DataFrame:
        """Encode all remaining categorical/object columns to numeric.

        Called once after feature engineering so every downstream stage
        (feature selection, model selection, hyperopt, ensemble) receives
        a fully numeric DataFrame.  Also handles inf/NaN cleanup.
        """
        from sklearn.preprocessing import LabelEncoder

        X_out = X.copy()
        for col in X_out.select_dtypes(include=['object', 'category']).columns:
            try:
                X_out[col] = LabelEncoder().fit_transform(X_out[col].astype(str))
            except Exception:
                X_out = X_out.drop(columns=[col])

        # Replace inf with NaN, then fill NaN with 0
        X_out = X_out.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X_out

    @staticmethod
    def _encode_categoricals_and_scale(X: pd.DataFrame) -> np.ndarray:
        """Encode remaining categorical columns and apply StandardScaler.

        LLM feature reasoning can leave string columns (e.g. Sex, Embarked)
        that survive through data cleaning and feature engineering. This helper
        ensures they are label-encoded (or dropped) before scaling so that
        downstream model fitting never receives non-numeric data.

        Returns:
            np.ndarray of scaled numeric features.
        """
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        X_numeric = X.copy()
        for col in X_numeric.select_dtypes(include=['object', 'category']).columns:
            try:
                X_numeric[col] = LabelEncoder().fit_transform(X_numeric[col].astype(str))
            except Exception:
                X_numeric = X_numeric.drop(columns=[col])

        return StandardScaler().fit_transform(X_numeric)

    async def init(self) -> Any:
        """Initialize registries and discover skills."""
        if self._initialized:
            return

        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
            from Jotty.core.registry.tools_registry import get_tools_registry

            self._skills_registry = get_skills_registry()
            self._skills_registry.init()
            self._tools_registry = get_tools_registry()

            # Build skill adapters
            self._build_skill_adapters()
            self._initialized = True

            logger.info(f"SkillOrchestrator initialized with {len(self._skill_adapters)} skill adapters")

        except Exception as e:
            logger.warning(f"SkillOrchestrator init failed: {e}")
            self._initialized = True  # Continue with built-in implementations

    def _build_skill_adapters(self) -> Any:
        """Build adapters for all ML-related skills."""
        ml_skill_keywords = [
            'data', 'feature', 'model', 'automl', 'hyperopt', 'ensemble',
            'metric', 'shap', 'pycaret', 'sklearn', 'xgboost', 'lightgbm',
            'clustering', 'dimensionality', 'statistical', 'time-series'
        ]

        for skill_name, skill_def in self._skills_registry.loaded_skills.items():
            # Check if skill is ML-related
            name_lower = skill_name.lower()
            if any(kw in name_lower for kw in ml_skill_keywords):
                adapter = SkillAdapter(skill_name, skill_def, self._tools_registry)
                self._skill_adapters[skill_name] = adapter
                logger.debug(f"Created adapter for {skill_name} ({adapter.category})")

    def _detect_problem_type(self, y: pd.Series) -> ProblemType:
        """Auto-detect problem type from target variable."""
        if y is None:
            return ProblemType.CLUSTERING

        unique_ratio = y.nunique() / len(y)
        n_unique = y.nunique()

        # Check dtype
        if y.dtype == 'object' or y.dtype == 'bool':
            return ProblemType.CLASSIFICATION

        # Check if categorical (few unique values)
        if n_unique <= 20 and unique_ratio < 0.05:
            return ProblemType.CLASSIFICATION

        return ProblemType.REGRESSION

    def _get_skills_by_category(self, category: SkillCategory) -> List[SkillAdapter]:
        """Get all skills for a category."""
        return [
            adapter for adapter in self._skill_adapters.values()
            if adapter.category == category
        ]

    async def solve(self,
                    X: pd.DataFrame,
                    y: pd.Series = None,
                    problem_type: str = "auto",
                    target_metric: str = "auto",
                    time_budget: int = 300,
                    business_context: str = "",
                    on_stage_complete: Callable = None) -> PipelineResult:
        """
        Solve any ML problem by orchestrating skills.

        Args:
            X: Feature dataframe
            y: Target series (None for clustering)
            problem_type: "classification", "regression", "clustering", or "auto"
            target_metric: Metric to optimize ("auto" to infer)
            time_budget: Max seconds for optimization
            business_context: Optional business context for LLM feature reasoning

        Returns:
            PipelineResult with best model, score, and insights
        """
        await self.init()

        # Initialize progress tracker
        total_stages = len(self.PIPELINE_ORDER) + (1 if self._use_llm_features else 0)
        self._progress = ProgressTracker(total_stages) if self._show_progress else None

        logger.info("=" * 60)
        logger.info("SKILL ORCHESTRATOR - AUTONOMOUS ML PIPELINE")
        logger.info("=" * 60)

        # Step 1: Detect problem type
        if problem_type == "auto":
            prob_type = self._detect_problem_type(y)
        else:
            prob_type = ProblemType(problem_type)

        # Step 2: Select target metric
        if target_metric == "auto":
            if prob_type == ProblemType.CLASSIFICATION:
                target_metric = "accuracy"
            elif prob_type == ProblemType.REGRESSION:
                target_metric = "r2"
            else:
                target_metric = "silhouette"

        logger.info(f"Problem: {prob_type.value} | Metric: {target_metric} | Features: {X.shape[1]} | Samples: {len(X)}")
        logger.info("=" * 60)

        # Step 3: Execute pipeline
        context = {
            'X': X.copy(),
            'y': y.copy() if y is not None else None,
            'X_original': X.copy(),
            'problem_type': prob_type,
            'target_metric': target_metric,
            'time_budget': time_budget,
            'business_context': business_context,
        }

        skill_results = []

        # Step 3a: LLM Feature Reasoning (if enabled)
        if self._use_llm_features and self._llm_reasoner:
            if self._progress:
                self._progress.start_stage("LLM_FEATURE_REASONING")

            try:
                suggestions = await self._llm_reasoner.reason_features(
                    context['X'], context['y'], prob_type.value, business_context
                )
                if suggestions:
                    context['X'] = self._llm_reasoner.apply_suggestions(context['X'], suggestions)
                    context['llm_features'] = len(suggestions)

                if self._progress:
                    self._progress.complete_stage("LLM_FEATURE_REASONING",
                        {'suggestions': len(suggestions), 'applied': context.get('llm_features', 0)})
            except Exception as e:
                logger.debug(f"LLM reasoning skipped: {e}")
                if self._progress:
                    self._progress.complete_stage("LLM_FEATURE_REASONING", {'skipped': True})

        # Step 4: Execute pipeline stages
        for category in self.PIPELINE_ORDER:
            if self._progress:
                self._progress.start_stage(category.value.upper())
            result = await self._execute_stage(category, context)
            skill_results.append(result)

            if result.success:
                # Update context with result
                if result.data is not None:
                    if category in [SkillCategory.DATA_CLEANING,
                                    SkillCategory.FEATURE_ENGINEERING,
                                    SkillCategory.FEATURE_SELECTION]:
                        context['X'] = result.data

                # After feature engineering, ensure all columns are numeric
                # LLM feature reasoning and various skills can leave string
                # columns that break downstream StandardScaler / corr calls
                if category == SkillCategory.FEATURE_ENGINEERING:
                    context['X'] = self._ensure_numeric(context['X'])
                    context['X'] = self._drop_target_leaks(context['X'], context['y'])

                if result.data is not None:
                    if category in [SkillCategory.MODEL_SELECTION,
                                     SkillCategory.HYPERPARAMETER_OPTIMIZATION,
                                     SkillCategory.ENSEMBLE]:
                        context['model'] = result.data
                        context['score'] = result.metrics.get('score', 0)

                if self._progress:
                    self._progress.complete_stage(category.value, result.metrics)

                if on_stage_complete:
                    try:
                        on_stage_complete(category.value, {
                            'success': result.success,
                            'metrics': result.metrics,
                            'error': result.error,
                        })
                    except Exception:
                        pass
            else:
                if self._progress:
                    self._progress.complete_stage(category.value, {'skipped': result.error or 'failed'})

                if on_stage_complete:
                    try:
                        on_stage_complete(category.value, {
                            'success': False,
                            'metrics': result.metrics,
                            'error': result.error,
                        })
                    except Exception:
                        pass

        # Compile initial result
        best_score = context.get('score', 0)
        best_model = context.get('model')

        # ================================================================
        # LLM FEEDBACK LOOP (10/10 Feature) - Iterate if time allows
        # ================================================================
        feature_importance = context.get('feature_importance', {})

        # Lower threshold to 60s to allow feedback loop in most cases
        if (self._use_llm_features and self._llm_reasoner and
            feature_importance and time_budget >= 60 and len(feature_importance) > 5):

            logger.info("Running LLM Feedback Loop...")
            try:
                import lightgbm as lgb
                from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
                from sklearn.preprocessing import StandardScaler, LabelEncoder

                # Generate improved features based on what worked
                feedback_suggestions = await self._llm_reasoner.reason_with_feedback(
                    context['X_original'], context['y'], prob_type.value,
                    business_context, feature_importance, iteration=1
                )

                if feedback_suggestions:
                    # Apply new features to original X
                    X_improved = self._llm_reasoner.apply_suggestions(
                        context['X_original'], feedback_suggestions, drop_text_cols=True
                    )

                    # Quick evaluation: does it improve score?
                    if prob_type == ProblemType.CLASSIFICATION:
                        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        test_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                        scoring = 'accuracy'
                    else:
                        cv = KFold(n_splits=3, shuffle=True, random_state=42)
                        test_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                        scoring = 'r2'

                    # Prepare improved features
                    X_improved = X_improved.fillna(0).replace([np.inf, -np.inf], 0)

                    # Encode any remaining categoricals
                    for col in X_improved.select_dtypes(include=['object', 'category']).columns:
                        le = LabelEncoder()
                        X_improved[col] = le.fit_transform(X_improved[col].astype(str))

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_improved)

                    feedback_score = cross_val_score(test_model, X_scaled, y, cv=cv, scoring=scoring).mean()

                    if feedback_score > best_score:
                        logger.info(f"Feedback Loop IMPROVED: {best_score:.4f} -> {feedback_score:.4f}")
                        best_score = feedback_score
                        test_model.fit(X_scaled, y)
                        best_model = test_model
                        context['X'] = X_improved
                    else:
                        logger.info(f"Feedback Loop: no improvement ({feedback_score:.4f} vs {best_score:.4f})")

            except Exception as e:
                logger.debug(f"LLM feedback loop failed: {e}")

        result = PipelineResult(
            problem_type=prob_type,
            best_score=best_score,
            best_model=best_model,
            feature_count=context['X'].shape[1],
            skill_results=skill_results,
            predictions=context.get('predictions'),
            feature_importance=context.get('feature_importance'),
            processed_X=context['X'],
        )

        # Final progress summary
        if self._progress:
            self._progress.finish(best_score)
        else:
            logger.info("=" * 60)
            logger.info(f"COMPLETE | Score: {best_score:.4f} | Features: {X.shape[1]} -> {result.feature_count}")
            logger.info("=" * 60)

        return result

    async def _execute_stage(self, category: SkillCategory, context: Dict) -> SkillResult:
        """Execute a pipeline stage using available skills or built-in."""
        # Get skills for this category
        skills = self._get_skills_by_category(category)

        if skills:
            # Try each skill until one succeeds
            for skill in skills:
                result = await skill.execute(context)
                if result.success:
                    return result

        # Fallback to built-in implementation
        return await self._builtin_stage(category, context)

    async def _builtin_stage(self, category: SkillCategory, context: Dict) -> SkillResult:
        """Built-in implementations for each stage."""
        X = context.get('X')
        y = context.get('y')
        problem_type = context.get('problem_type')

        try:
            if category == SkillCategory.DATA_PROFILING:
                return await self._builtin_profiling(X, y)

            elif category == SkillCategory.DATA_CLEANING:
                return await self._builtin_cleaning(X, y)

            elif category == SkillCategory.FEATURE_ENGINEERING:
                return await self._builtin_feature_engineering(X, y, problem_type)

            elif category == SkillCategory.FEATURE_SELECTION:
                return await self._builtin_feature_selection(X, y, problem_type)

            elif category == SkillCategory.MODEL_SELECTION:
                return await self._builtin_model_selection(X, y, problem_type)

            elif category == SkillCategory.HYPERPARAMETER_OPTIMIZATION:
                return await self._builtin_hyperopt(X, y, problem_type, context)

            elif category == SkillCategory.ENSEMBLE:
                return await self._builtin_ensemble(X, y, problem_type, context)

            elif category == SkillCategory.EVALUATION:
                return await self._builtin_evaluation(context)

            elif category == SkillCategory.EXPLANATION:
                return await self._builtin_explanation(context)

            else:
                return SkillResult(
                    skill_name=f"builtin_{category.value}",
                    category=category,
                    success=False,
                    error="Not implemented"
                )

        except Exception as e:
            return SkillResult(
                skill_name=f"builtin_{category.value}",
                category=category,
                success=False,
                error=str(e)
            )

    async def _builtin_profiling(self, X: pd.DataFrame, y: pd.Series) -> SkillResult:
        """Profile the dataset."""
        profile = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'missing_total': int(X.isnull().sum().sum()),
            'missing_pct': float(X.isnull().sum().sum() / X.size * 100),
            'numeric_cols': len(X.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(X.select_dtypes(include=['object']).columns),
        }

        if y is not None:
            profile['target_unique'] = int(y.nunique())
            profile['target_missing'] = int(y.isnull().sum())

        return SkillResult(
            skill_name="builtin_profiler",
            category=SkillCategory.DATA_PROFILING,
            success=True,
            metrics=profile,
            metadata=profile
        )

    async def _builtin_cleaning(self, X: pd.DataFrame, y: pd.Series) -> SkillResult:
        """Clean the dataset."""
        X_clean = X.copy()

        # Handle missing values
        for col in X_clean.columns:
            if X_clean[col].dtype in ['int64', 'float64']:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mode().iloc[0] if len(X_clean[col].mode()) > 0 else 'MISSING')

        return SkillResult(
            skill_name="builtin_cleaner",
            category=SkillCategory.DATA_CLEANING,
            success=True,
            data=X_clean,
            metrics={'cleaned_missing': int(X.isnull().sum().sum())}
        )

    async def _builtin_evaluation(self, context: Dict) -> SkillResult:
        """Evaluate the model."""
        score = context.get('score', 0)
        model = context.get('model')

        # Handle model type safely (model could be None, DataFrame, or actual model)
        if model is None:
            model_type = 'None'
        elif isinstance(model, pd.DataFrame):
            model_type = 'DataFrame'
        else:
            model_type = type(model).__name__

        return SkillResult(
            skill_name="builtin_evaluator",
            category=SkillCategory.EVALUATION,
            success=True,
            metrics={'final_score': score},
            metadata={'model_type': model_type}
        )

    async def _builtin_explanation(self, context: Dict) -> SkillResult:
        """Generate model explanations."""
        model = context.get('model')
        X = context.get('X')

        feature_importance = {}

        # Check model safely (avoid DataFrame truthiness issue)
        if model is not None and not isinstance(model, pd.DataFrame) and hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(X.columns, importance))
        elif model is not None and not isinstance(model, pd.DataFrame) and hasattr(model, 'estimators_'):
            # Voting ensemble - aggregate from sub-models
            importances = []
            for name, est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
            if importances:
                avg_importance = np.mean(importances, axis=0)
                feature_importance = dict(zip(X.columns, avg_importance))

        context['feature_importance'] = feature_importance

        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])

        return SkillResult(
            skill_name="builtin_explainer",
            category=SkillCategory.EXPLANATION,
            success=True,
            metrics={'top_features': len(sorted_importance)},
            metadata={'feature_importance': sorted_importance}
        )


# Singleton accessor
_orchestrator_instance = None


def get_skill_orchestrator() -> SkillOrchestrator:
    """Get or create the skill orchestrator singleton."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SkillOrchestrator()
    return _orchestrator_instance


def reset_skill_orchestrator() -> None:
    """Reset the singleton skill orchestrator (for testing)."""
    global _orchestrator_instance
    _orchestrator_instance = None
