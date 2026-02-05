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
import asyncio
import sys
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

# Import from extracted modules
from .skill_types import (
    ProblemType, SkillCategory, SkillResult, PipelineResult,
    ProgressTracker, SkillAdapter
)

# Import ML skills from proper location (core/skills/ml/)
try:
    from ...skills.ml.eda import EDASkill as EDAAnalyzer
    from ...skills.ml.llm_reasoner import LLMFeatureReasonerSkill as LLMFeatureReasoner
except ImportError:
    # Fallback: use inline implementations if skills not available
    EDAAnalyzer = None
    LLMFeatureReasoner = None

logger = logging.getLogger(__name__)


# ============================================================
# SKILL ORCHESTRATOR
# ============================================================

class SkillOrchestrator:
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

    def __init__(self, use_llm_features: bool = True, show_progress: bool = True):
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

    async def init(self):
        """Initialize registries and discover skills."""
        if self._initialized:
            return

        try:
            from ...registry.skills_registry import get_skills_registry
            from ...registry.tools_registry import get_tools_registry

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

    def _build_skill_adapters(self):
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

        print("\n" + "=" * 60)
        print("🚀 SKILL ORCHESTRATOR - AUTONOMOUS ML PIPELINE")
        print("=" * 60)

        # Step 1: Detect problem type
        if problem_type == "auto":
            prob_type = self._detect_problem_type(y)
        else:
            prob_type = ProblemType(problem_type)

        print(f"📋 Problem: {prob_type.value} | 📊 Metric: ", end="")

        # Step 2: Select target metric
        if target_metric == "auto":
            if prob_type == ProblemType.CLASSIFICATION:
                target_metric = "accuracy"
            elif prob_type == ProblemType.REGRESSION:
                target_metric = "r2"
            else:
                target_metric = "silhouette"

        print(f"{target_metric} | 📦 Features: {X.shape[1]} | 📝 Samples: {len(X)}")
        print("=" * 60 + "\n")

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

            print("\n   🔄 Running LLM Feedback Loop...")
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
                        print(f"   ✅ Feedback Loop IMPROVED: {best_score:.4f} → {feedback_score:.4f}")
                        best_score = feedback_score
                        test_model.fit(X_scaled, y)
                        best_model = test_model
                        context['X'] = X_improved
                    else:
                        print(f"   ℹ️ Feedback Loop: no improvement ({feedback_score:.4f} vs {best_score:.4f})")

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
            print(f"\n{'='*60}")
            print(f"🏆 COMPLETE | Score: {best_score:.4f} | Features: {X.shape[1]} → {result.feature_count}")
            print(f"{'='*60}\n")

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

    async def _builtin_feature_engineering(self, X: pd.DataFrame, y: pd.Series,
                                           problem_type: ProblemType) -> SkillResult:
        """
        Advanced Kaggle-style feature engineering.

        Techniques from top Kaggle solutions:
        1. Groupby aggregations (MOST POWERFUL)
        2. Target encoding
        3. Frequency encoding
        4. Binning/Discretization
        5. Polynomial features
        6. Log transforms
        7. NaN pattern encoding
        8. Categorical combinations
        9. Interaction features
        10. Statistical aggregations
        """
        from sklearn.preprocessing import LabelEncoder

        X_eng = X.copy()
        original_cols = list(X_eng.columns)
        n_original = len(original_cols)

        # ================================================================
        # PRE-PROCESSING: Convert any Categorical dtype to numeric codes
        # This handles columns created by pd.cut/pd.qcut with labels
        # ================================================================
        for col in X_eng.columns:
            if X_eng[col].dtype.name == 'category':
                X_eng[col] = X_eng[col].cat.codes

        # Identify column types
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_eng.select_dtypes(include=['object']).columns.tolist()

        # Store original categorical columns before encoding
        cat_cols_original = cat_cols.copy()

        # ================================================================
        # 1. TARGET ENCODING - DISABLED (causes leakage outside CV)
        # ================================================================
        # NOTE: Target encoding must be done INSIDE cross-validation folds
        # to avoid leakage. Doing it here on full data before CV causes
        # the model to see target information from validation rows.
        # TODO: Implement proper target encoding with CV-aware pipeline
        # if y is not None and len(cat_cols) > 0:
        #     for col in cat_cols[:5]:
        #         target_mean = X_eng.groupby(col).apply(...)
        #         X_eng[f'{col}_target_enc'] = ...

        # ================================================================
        # 2. FREQUENCY ENCODING (count-based)
        # ================================================================
        for col in cat_cols[:5]:
            freq = X_eng[col].value_counts(normalize=True)
            X_eng[f'{col}_freq'] = X_eng[col].map(freq).fillna(0)

        # ================================================================
        # 3. LABEL ENCODE categoricals (after extracting target/freq)
        # ================================================================
        for col in cat_cols:
            le = LabelEncoder()
            X_eng[col] = le.fit_transform(X_eng[col].astype(str))

        # Update numeric cols after encoding
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()

        # ================================================================
        # 4. GROUPBY AGGREGATIONS (THE MOST POWERFUL - from NVIDIA blog)
        # ================================================================
        # Use encoded categoricals as groupby keys
        groupby_cols = [c for c in cat_cols_original if c in X_eng.columns][:3]
        agg_cols = [c for c in original_cols if c in numeric_cols][:5]

        for grp_col in groupby_cols:
            for agg_col in agg_cols:
                if grp_col != agg_col:
                    try:
                        # Mean, std, count per group
                        grp_mean = X_eng.groupby(grp_col)[agg_col].transform('mean')
                        grp_std = X_eng.groupby(grp_col)[agg_col].transform('std').fillna(0)
                        grp_count = X_eng.groupby(grp_col)[agg_col].transform('count')

                        X_eng[f'{grp_col}_{agg_col}_grp_mean'] = grp_mean
                        X_eng[f'{grp_col}_{agg_col}_grp_std'] = grp_std
                        X_eng[f'{grp_col}_{agg_col}_grp_cnt'] = grp_count

                        # Deviation from group mean
                        X_eng[f'{grp_col}_{agg_col}_dev'] = X_eng[agg_col] - grp_mean
                    except:
                        pass

        # ================================================================
        # 5. BINNING / DISCRETIZATION
        # ================================================================
        for col in numeric_cols[:5]:
            try:
                # Quantile-based binning (5 bins)
                X_eng[f'{col}_qbin'] = pd.qcut(X_eng[col], q=5, labels=False, duplicates='drop')
            except:
                pass

            # Round-based binning
            if X_eng[col].std() > 0:
                X_eng[f'{col}_round1'] = X_eng[col].round(1)
                X_eng[f'{col}_round0'] = X_eng[col].round(0)

        # ================================================================
        # 6. POLYNOMIAL FEATURES (squared, cubed)
        # ================================================================
        for col in numeric_cols[:5]:
            X_eng[f'{col}_sq'] = X_eng[col] ** 2
            X_eng[f'{col}_sqrt'] = np.sqrt(np.abs(X_eng[col]))

        # ================================================================
        # 7. LOG TRANSFORMS (for skewed data)
        # ================================================================
        for col in numeric_cols[:5]:
            if (X_eng[col] > 0).all():
                X_eng[f'{col}_log'] = np.log1p(X_eng[col])
            elif (X_eng[col] >= 0).all():
                X_eng[f'{col}_log'] = np.log1p(X_eng[col])

        # ================================================================
        # 8. NaN PATTERN ENCODING
        # ================================================================
        nan_cols = X.columns[X.isnull().any()].tolist()
        if len(nan_cols) > 0:
            # Binary NaN indicator per column
            for col in nan_cols[:10]:
                X_eng[f'{col}_isna'] = X[col].isnull().astype(int)

            # Combined NaN pattern (binary encoding)
            X_eng['_nan_pattern'] = 0
            for i, col in enumerate(nan_cols[:10]):
                X_eng['_nan_pattern'] += X[col].isnull().astype(int) * (2 ** i)

            # Total NaN count per row
            X_eng['_nan_count'] = X[nan_cols].isnull().sum(axis=1)

        # ================================================================
        # 9. CATEGORICAL COMBINATIONS
        # ================================================================
        encoded_cats = [c for c in cat_cols_original if c in X_eng.columns][:3]
        for i, col1 in enumerate(encoded_cats):
            for col2 in encoded_cats[i+1:]:
                try:
                    max_val = X_eng[col2].max() + 1
                    X_eng[f'{col1}_{col2}_comb'] = (X_eng[col1] + 1) + (X_eng[col2] + 1) / max_val
                except:
                    pass

        # ================================================================
        # 10. INTERACTION FEATURES (multiply & divide)
        # ================================================================
        interact_cols = [c for c in original_cols if c in numeric_cols][:6]

        for i, col1 in enumerate(interact_cols):
            for col2 in interact_cols[i+1:]:
                X_eng[f'{col1}_x_{col2}'] = X_eng[col1] * X_eng[col2]
                denom = X_eng[col2].abs() + 0.001
                X_eng[f'{col1}_div_{col2}'] = X_eng[col1] / denom

        # ================================================================
        # 11. ROW-LEVEL AGGREGATIONS
        # ================================================================
        if len(numeric_cols) >= 3:
            orig_numeric = [c for c in original_cols if c in numeric_cols]
            if len(orig_numeric) >= 3:
                X_eng['_row_sum'] = X_eng[orig_numeric].sum(axis=1)
                X_eng['_row_mean'] = X_eng[orig_numeric].mean(axis=1)
                X_eng['_row_std'] = X_eng[orig_numeric].std(axis=1)
                X_eng['_row_max'] = X_eng[orig_numeric].max(axis=1)
                X_eng['_row_min'] = X_eng[orig_numeric].min(axis=1)
                X_eng['_row_range'] = X_eng['_row_max'] - X_eng['_row_min']
                X_eng['_row_skew'] = X_eng[orig_numeric].skew(axis=1)

        # ================================================================
        # 12. QUANTILE FEATURES
        # ================================================================
        for col in numeric_cols[:3]:
            try:
                q25 = X_eng[col].quantile(0.25)
                q75 = X_eng[col].quantile(0.75)
                X_eng[f'{col}_below_q25'] = (X_eng[col] < q25).astype(int)
                X_eng[f'{col}_above_q75'] = (X_eng[col] > q75).astype(int)
            except:
                pass

        # ================================================================
        # 13. TARGET ENCODING WITH CV (NO LEAKAGE - World Class)
        # ================================================================
        # Proper target encoding: use leave-one-out or K-fold to prevent leakage
        if y is not None and len(cat_cols_original) > 0:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for col in cat_cols_original[:5]:
                if col in X_eng.columns:
                    try:
                        target_enc = np.zeros(len(X_eng))
                        col_encoded = X_eng[col]  # Already label encoded

                        # CV-based target encoding
                        for train_idx, val_idx in kf.split(X_eng):
                            # Calculate target mean on training fold only
                            train_means = pd.Series(y.iloc[train_idx].values).groupby(
                                col_encoded.iloc[train_idx].values
                            ).mean()

                            # Apply to validation fold
                            target_enc[val_idx] = col_encoded.iloc[val_idx].map(train_means).fillna(y.mean()).values

                        X_eng[f'{col}_target_enc_cv'] = target_enc
                    except Exception as e:
                        logger.debug(f"Target encoding failed for {col}: {e}")

        # ================================================================
        # 14. CV-VALIDATED INTERACTIONS (Only keep if improves score)
        # ================================================================
        if y is not None and len(numeric_cols) >= 2:
            from sklearn.model_selection import cross_val_score
            import lightgbm as lgb

            # Quick baseline score with current features
            try:
                X_temp = X_eng.fillna(0).replace([np.inf, -np.inf], 0)
                baseline_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1) \
                    if problem_type == ProblemType.CLASSIFICATION else \
                    lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)

                baseline_score = cross_val_score(baseline_model, X_temp, y, cv=3, scoring='accuracy' if problem_type == ProblemType.CLASSIFICATION else 'r2').mean()

                # Try top interaction candidates
                orig_numeric = [c for c in original_cols if c in numeric_cols][:4]
                validated_interactions = 0

                for i, col1 in enumerate(orig_numeric):
                    for col2 in orig_numeric[i+1:]:
                        # Create candidate interaction
                        interaction_name = f'{col1}_x_{col2}_validated'
                        X_test = X_temp.copy()
                        X_test[interaction_name] = X_test[col1] * X_test[col2]

                        # Test if it improves score
                        test_score = cross_val_score(baseline_model, X_test, y, cv=3, scoring='accuracy' if problem_type == ProblemType.CLASSIFICATION else 'r2').mean()

                        if test_score > baseline_score + 0.001:  # Must improve by 0.1%
                            X_eng[interaction_name] = X_eng[col1] * X_eng[col2]
                            baseline_score = test_score
                            validated_interactions += 1

                if validated_interactions > 0:
                    logger.debug(f"Added {validated_interactions} CV-validated interactions")
            except Exception as e:
                logger.debug(f"CV-validated interactions failed: {e}")

        # ================================================================
        # 15. EARLY FEATURE PRUNING (Remove obviously useless features)
        # ================================================================
        # Remove features with near-zero variance or perfect correlation
        try:
            # Remove constant features
            constant_cols = [col for col in X_eng.columns if X_eng[col].nunique() <= 1]
            if constant_cols:
                X_eng = X_eng.drop(columns=constant_cols)

            # Remove features with >99% same value
            near_constant = []
            for col in X_eng.columns:
                top_freq = X_eng[col].value_counts(normalize=True).iloc[0] if len(X_eng[col].value_counts()) > 0 else 0
                if top_freq > 0.99:
                    near_constant.append(col)
            if near_constant:
                X_eng = X_eng.drop(columns=near_constant)

        except Exception as e:
            logger.debug(f"Early pruning failed: {e}")

        # Clean up
        X_eng = X_eng.fillna(0)
        X_eng = X_eng.replace([np.inf, -np.inf], 0)

        # Defragment DataFrame to improve performance (fixes fragmentation warning)
        X_eng = X_eng.copy()

        # ================================================================
        # FINAL: ENCODE ALL REMAINING CATEGORICAL COLUMNS
        # ================================================================
        # LLM may have created new categorical columns - encode them all
        remaining_cats = X_eng.select_dtypes(include=['object', 'category']).columns.tolist()
        if remaining_cats:
            from sklearn.preprocessing import LabelEncoder
            for col in remaining_cats:
                try:
                    le = LabelEncoder()
                    X_eng[col] = le.fit_transform(X_eng[col].astype(str))
                except:
                    # If encoding fails, drop the column
                    X_eng = X_eng.drop(columns=[col])

        return SkillResult(
            skill_name="builtin_kaggle_fe",
            category=SkillCategory.FEATURE_ENGINEERING,
            success=True,
            data=X_eng,
            metrics={
                'original_features': n_original,
                'engineered_features': len(X_eng.columns),
                'new_features': len(X_eng.columns) - n_original,
                'techniques_used': [
                    'frequency_encoding', 'groupby_aggs', 'binning', 'polynomial',
                    'log_transform', 'nan_patterns', 'cat_combinations', 'interactions',
                    'row_stats', 'quantiles', 'target_encoding_cv', 'validated_interactions',
                    'early_pruning'
                ]
            }
        )

    async def _builtin_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                         problem_type: ProblemType) -> SkillResult:
        """
        World-class feature selection using multiple advanced techniques.

        Techniques used:
        1. CORRELATION FILTER - Remove highly correlated features (redundancy)
        2. NULL IMPORTANCE - Compare real vs shuffled target importance
        3. PERMUTATION IMPORTANCE - Measure actual impact on CV score
        4. MULTI-MODEL VOTING - Ensemble importance from LightGBM, XGBoost, RF
        5. STABILITY SELECTION - Features consistently important across seeds
        6. BORUTA-LIKE - Compare against shadow (shuffled) features
        """
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.inspection import permutation_importance
        from collections import defaultdict

        if y is None:
            return SkillResult(
                skill_name="builtin_feature_selector",
                category=SkillCategory.FEATURE_SELECTION,
                success=True,
                data=X,
                metrics={'selected': X.shape[1]}
            )

        n_features = X.shape[1]
        feature_scores = defaultdict(float)  # Accumulate scores across methods
        method_results = {}

        # ================================================================
        # 1. CORRELATION FILTER - Remove redundant features
        # ================================================================
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        corr_threshold = 0.98  # Higher threshold = keep more features

        to_drop_corr = set()
        for col in upper_tri.columns:
            correlated = upper_tri.index[upper_tri[col] > corr_threshold].tolist()
            if correlated:
                # Keep the one with higher variance
                for corr_col in correlated:
                    if X[col].var() >= X[corr_col].var():
                        to_drop_corr.add(corr_col)
                    else:
                        to_drop_corr.add(col)
                    high_corr_pairs.append((col, corr_col))

        X_filtered = X.drop(columns=list(to_drop_corr), errors='ignore')
        method_results['correlation_removed'] = len(to_drop_corr)

        # ================================================================
        # 2. MULTI-MODEL IMPORTANCE VOTING
        # ================================================================
        if problem_type == ProblemType.CLASSIFICATION:
            models = {
                'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
                'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
                'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            }
        else:
            models = {
                'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            }

        model_importances = {}
        for name, model in models.items():
            try:
                model.fit(X_filtered, y)
                imp = pd.Series(model.feature_importances_, index=X_filtered.columns)
                imp_normalized = imp / (imp.sum() + 1e-10)  # Normalize to sum=1
                model_importances[name] = imp_normalized

                # Add to feature scores (weight by model type)
                for feat, score in imp_normalized.items():
                    feature_scores[feat] += score
            except Exception as e:
                logger.debug(f"Model {name} importance failed: {e}")

        method_results['multi_model'] = list(model_importances.keys())

        # ================================================================
        # 3. NULL IMPORTANCE TEST - Identify truly predictive features
        # ================================================================
        null_importance_scores = defaultdict(list)
        n_null_runs = 5

        lgb_model = models['lgb']
        for i in range(n_null_runs):
            # Shuffle target
            y_shuffled = y.sample(frac=1, random_state=42 + i).reset_index(drop=True)
            try:
                lgb_model.fit(X_filtered, y_shuffled)
                null_imp = pd.Series(lgb_model.feature_importances_, index=X_filtered.columns)
                for feat in X_filtered.columns:
                    null_importance_scores[feat].append(null_imp[feat])
            except:
                pass

        # Features where real importance > 95th percentile of null importance
        null_passed = set()
        if model_importances.get('lgb') is not None:
            real_imp = model_importances['lgb']
            for feat in X_filtered.columns:
                null_vals = null_importance_scores.get(feat, [0])
                null_95 = np.percentile(null_vals, 95) if null_vals else 0
                real_val = real_imp.get(feat, 0)
                if real_val > null_95 * 1.1:  # 10% margin
                    null_passed.add(feat)
                    feature_scores[feat] += 0.5  # Bonus for passing null test

        method_results['null_test_passed'] = len(null_passed)

        # ================================================================
        # 4. STABILITY SELECTION - Consistent importance across seeds
        # ================================================================
        stability_counts = defaultdict(int)
        n_stability_runs = 5
        top_pct = 0.3  # Top 30% features

        for seed in range(n_stability_runs):
            try:
                lgb_temp = lgb.LGBMClassifier(n_estimators=50, random_state=seed, verbose=-1) \
                    if problem_type == ProblemType.CLASSIFICATION else \
                    lgb.LGBMRegressor(n_estimators=50, random_state=seed, verbose=-1)

                # Bootstrap sample
                n_samples = len(X_filtered)
                idx = np.random.RandomState(seed).choice(n_samples, size=n_samples, replace=True)
                lgb_temp.fit(X_filtered.iloc[idx], y.iloc[idx])

                imp = pd.Series(lgb_temp.feature_importances_, index=X_filtered.columns)
                top_k = int(len(imp) * top_pct)
                top_features = imp.nlargest(top_k).index.tolist()

                for feat in top_features:
                    stability_counts[feat] += 1
            except:
                pass

        # Features selected in majority of runs
        stable_features = {f for f, c in stability_counts.items() if c >= n_stability_runs * 0.6}
        for feat in stable_features:
            feature_scores[feat] += 0.3  # Bonus for stability

        method_results['stable_features'] = len(stable_features)

        # ================================================================
        # 5. BORUTA-LIKE SHADOW FEATURES TEST
        # ================================================================
        # Create shadow features (shuffled copies)
        X_shadow = X_filtered.copy()
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col].values)
        X_shadow.columns = [f'shadow_{c}' for c in X_shadow.columns]

        X_with_shadow = pd.concat([X_filtered, X_shadow], axis=1)

        try:
            lgb_shadow = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) \
                if problem_type == ProblemType.CLASSIFICATION else \
                lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_shadow.fit(X_with_shadow, y)

            imp_all = pd.Series(lgb_shadow.feature_importances_, index=X_with_shadow.columns)
            shadow_max = imp_all[[c for c in imp_all.index if c.startswith('shadow_')]].max()

            # Features beating the best shadow feature
            boruta_passed = set()
            for feat in X_filtered.columns:
                if imp_all.get(feat, 0) > shadow_max:
                    boruta_passed.add(feat)
                    feature_scores[feat] += 0.4  # Bonus for beating shadows

            method_results['boruta_passed'] = len(boruta_passed)
        except Exception as e:
            logger.debug(f"Boruta test failed: {e}")
            method_results['boruta_passed'] = 0

        # ================================================================
        # 6. PERMUTATION IMPORTANCE (on a quick model)
        # ================================================================
        try:
            if problem_type == ProblemType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'r2'

            lgb_perm = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1) \
                if problem_type == ProblemType.CLASSIFICATION else \
                lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            lgb_perm.fit(X_filtered, y)

            perm_imp = permutation_importance(lgb_perm, X_filtered, y,
                                              n_repeats=5, random_state=42, n_jobs=-1)
            perm_scores = pd.Series(perm_imp.importances_mean, index=X_filtered.columns)
            perm_scores_normalized = perm_scores / (perm_scores.sum() + 1e-10)

            for feat, score in perm_scores_normalized.items():
                if score > 0:  # Only positive permutation importance
                    feature_scores[feat] += score * 0.5

            method_results['permutation_done'] = True
        except Exception as e:
            logger.debug(f"Permutation importance failed: {e}")
            method_results['permutation_done'] = False

        # ================================================================
        # 7. SHAP-BASED IMPORTANCE (World-Class 10/10 Feature)
        # ================================================================
        try:
            import shap

            # Use TreeExplainer for fast SHAP on tree models
            lgb_shap = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) \
                if problem_type == ProblemType.CLASSIFICATION else \
                lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_shap.fit(X_filtered, y)

            # Sample for speed if dataset is large
            n_shap_samples = min(500, len(X_filtered))
            X_sample = X_filtered.sample(n=n_shap_samples, random_state=42) if len(X_filtered) > 500 else X_filtered

            explainer = shap.TreeExplainer(lgb_shap)
            shap_values = explainer.shap_values(X_sample)

            # Handle multiclass case
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)

            # Mean absolute SHAP value per feature
            shap_importance = np.abs(shap_values).mean(axis=0)
            shap_scores = pd.Series(shap_importance, index=X_filtered.columns)
            shap_scores_normalized = shap_scores / (shap_scores.sum() + 1e-10)

            # SHAP scores are highly reliable - weight them more
            for feat, score in shap_scores_normalized.items():
                feature_scores[feat] += score * 1.5  # Higher weight for SHAP

            method_results['shap_done'] = True

            # Store SHAP importance for LLM feedback loop
            method_results['shap_importance'] = shap_scores_normalized.to_dict()

        except ImportError:
            logger.debug("SHAP not installed - skipping SHAP importance")
            method_results['shap_done'] = False
        except Exception as e:
            logger.debug(f"SHAP importance failed: {e}")
            method_results['shap_done'] = False

        # ================================================================
        # FINAL SELECTION - Quantile/Decile-based selection
        # ================================================================
        final_scores = pd.Series(feature_scores).sort_values(ascending=False)

        # Calculate deciles for adaptive selection
        if len(final_scores) > 10:
            # Assign decile ranks (1=top 10%, 10=bottom 10%)
            final_scores_df = pd.DataFrame({
                'feature': final_scores.index,
                'score': final_scores.values
            })
            final_scores_df['decile'] = pd.qcut(
                final_scores_df['score'].rank(method='first'),
                q=10, labels=range(10, 0, -1)  # 10=top, 1=bottom
            ).astype(int)

            # SELECTION STRATEGY (World-class - keep quality features):
            # - Decile 10-8 (top 30%): Always keep - best features
            # - Decile 7 (top 40%): Keep if score > median of decile
            # - Decile 6 (top 50%): Keep if passed multiple tests (score > 1.0)
            # - Decile 5 (top 60%): Keep if strong multi-test (score > 1.5)
            # - Below: Only keep if exceptional (score > 2.0)

            selected = []

            # Top 3 deciles (top 30%) - always keep
            top_deciles = final_scores_df[final_scores_df['decile'] >= 8]['feature'].tolist()
            selected.extend(top_deciles)

            # Decile 7 (top 40%) - keep above median
            decile_7 = final_scores_df[final_scores_df['decile'] == 7]
            if len(decile_7) > 0:
                d7_median = decile_7['score'].median()
                d7_selected = decile_7[decile_7['score'] >= d7_median]['feature'].tolist()
                selected.extend(d7_selected)

            # Decile 6 (top 50%) - keep if multi-test passed (score > 1.0)
            decile_6 = final_scores_df[final_scores_df['decile'] == 6]
            d6_selected = decile_6[decile_6['score'] >= 1.0]['feature'].tolist()
            selected.extend(d6_selected)

            # Decile 5 (top 60%) - keep if strong score (score > 1.5)
            decile_5 = final_scores_df[final_scores_df['decile'] == 5]
            d5_selected = decile_5[decile_5['score'] >= 1.5]['feature'].tolist()
            selected.extend(d5_selected)

            # Deciles 4 and below - only exceptional features (score > 2.0)
            lower_deciles = final_scores_df[final_scores_df['decile'] <= 4]
            exceptional = lower_deciles[lower_deciles['score'] >= 2.0]['feature'].tolist()
            selected.extend(exceptional)

            # Remove duplicates while preserving order
            selected = list(dict.fromkeys(selected))

            method_results['decile_distribution'] = final_scores_df.groupby('decile').size().to_dict()
        else:
            selected = final_scores.index.tolist()

        # Ensure minimum features (at least 15 or 15% of filtered)
        min_features = max(15, int(len(X_filtered.columns) * 0.15))
        if len(selected) < min_features:
            # Add more from top scores
            for feat in final_scores.index:
                if feat not in selected:
                    selected.append(feat)
                if len(selected) >= min_features:
                    break

        # ================================================================
        # CV VALIDATION - Ensure selection doesn't hurt performance
        # ================================================================
        try:
            if problem_type == ProblemType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                val_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                val_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'r2'

            # Score with selected features
            score_selected = cross_val_score(val_model, X_filtered[selected], y, cv=cv, scoring=scoring).mean()

            # Score with all features (after correlation filter)
            score_all = cross_val_score(val_model, X_filtered, y, cv=cv, scoring=scoring).mean()

            # If selected is significantly worse (>1%), add more features
            if score_selected < score_all - 0.01:
                # Add features until performance matches
                remaining = [f for f in final_scores.index if f not in selected]
                for feat in remaining:
                    selected.append(feat)
                    new_score = cross_val_score(val_model, X_filtered[selected], y, cv=cv, scoring=scoring).mean()
                    if new_score >= score_all - 0.005:  # Within 0.5% of full set
                        break
                    if len(selected) >= len(X_filtered.columns) * 0.6:  # Cap at 60%
                        break

            method_results['cv_selected_score'] = score_selected
            method_results['cv_all_score'] = score_all
        except Exception as e:
            logger.debug(f"CV validation failed: {e}")

        X_selected = X_filtered[selected]

        return SkillResult(
            skill_name="builtin_feature_selector",
            category=SkillCategory.FEATURE_SELECTION,
            success=True,
            data=X_selected,
            metrics={
                'original': n_features,
                'after_corr_filter': len(X_filtered.columns),
                'selected': len(selected),
                'removed': n_features - len(selected),
                'null_passed': method_results.get('null_test_passed', 0),
                'boruta_passed': method_results.get('boruta_passed', 0),
                'stable': method_results.get('stable_features', 0),
            },
            metadata={
                'selected_features': selected,
                'feature_scores': final_scores.head(20).to_dict(),
                'decile_dist': method_results.get('decile_distribution', {}),
                'methods_used': ['correlation', 'multi_model', 'null_importance',
                                'stability', 'boruta', 'permutation']
            }
        )

    async def _builtin_model_selection(self, X: pd.DataFrame, y: pd.Series,
                                       problem_type: ProblemType) -> SkillResult:
        """
        World-class model selection with:
        1. Extended model zoo (7+ algorithms)
        2. Data-adaptive configurations
        3. Cross-validation with OOF predictions for stacking
        4. Model diversity tracking for ensemble
        """
        from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.svm import SVC, SVR
        import lightgbm as lgb
        import xgboost as xgb

        n_samples, n_features = X.shape

        X_scaled = self._encode_categoricals_and_scale(X)

        # Adaptive configuration based on data size
        n_estimators = 100 if n_samples < 5000 else 200
        max_depth_tree = None if n_samples > 1000 else 10

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'

            models = {
                # Gradient Boosting Family (usually best)
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=n_estimators, random_state=42, verbose=-1,
                    learning_rate=0.1, num_leaves=31
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=n_estimators, random_state=42,
                    eval_metric='logloss', verbosity=0, learning_rate=0.1
                ),
                'histgb': HistGradientBoostingClassifier(
                    max_iter=n_estimators, random_state=42, learning_rate=0.1
                ),

                # Tree Ensembles (good diversity)
                'random_forest': RandomForestClassifier(
                    n_estimators=n_estimators, random_state=42,
                    max_depth=max_depth_tree, n_jobs=-1
                ),
                'extra_trees': ExtraTreesClassifier(
                    n_estimators=n_estimators, random_state=42,
                    max_depth=max_depth_tree, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=min(n_estimators, 100), random_state=42,
                    learning_rate=0.1
                ),

                # Linear Model (different hypothesis space)
                'logistic': LogisticRegression(
                    max_iter=1000, random_state=42, C=1.0
                ),
            }

            # Add SVM for small datasets (slow on large)
            if n_samples < 5000:
                models['svm'] = SVC(probability=True, random_state=42, C=1.0)

        else:  # Regression
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'

            models = {
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=n_estimators, random_state=42, verbose=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=n_estimators, random_state=42, verbosity=0
                ),
                'histgb': HistGradientBoostingRegressor(
                    max_iter=n_estimators, random_state=42
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
                'extra_trees': ExtraTreesRegressor(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=min(n_estimators, 100), random_state=42
                ),
                'ridge': Ridge(alpha=1.0),
            }

            if n_samples < 5000:
                models['svr'] = SVR(C=1.0)

        # Evaluate all models and collect OOF predictions
        best_model = None
        best_name = None
        best_score = -np.inf
        all_scores = {}
        all_std = {}
        oof_predictions = {}  # For stacking

        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
                mean_score = scores.mean()
                std_score = scores.std()
                all_scores[name] = mean_score
                all_std[name] = std_score

                # Collect OOF predictions for top models (for stacking later)
                if mean_score > best_score * 0.95:  # Within 5% of best
                    try:
                        if problem_type == ProblemType.CLASSIFICATION:
                            oof = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')
                            oof_predictions[name] = oof[:, 1] if oof.ndim > 1 else oof
                        else:
                            oof_predictions[name] = cross_val_predict(model, X_scaled, y, cv=cv)
                    except:
                        pass

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name

            except Exception as e:
                logger.debug(f"Model {name} failed: {e}")

        # Rank models for reporting
        sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        return SkillResult(
            skill_name="builtin_model_selector",
            category=SkillCategory.MODEL_SELECTION,
            success=True,
            data=best_model,
            metrics={'score': best_score, 'model': best_name, **{k: all_scores[k] for k in list(all_scores)[:4]}},
            metadata={
                'best_model': best_name,
                'all_scores': all_scores,
                'all_std': all_std,
                'model_ranking': [m[0] for m in sorted_models],
                'oof_predictions': oof_predictions,
                'X_scaled': X_scaled,
            }
        )

    async def _builtin_hyperopt(self, X: pd.DataFrame, y: pd.Series,
                                problem_type: ProblemType, context: Dict) -> SkillResult:
        """
        World-class hyperparameter optimization:
        1. Multi-model tuning (LightGBM, XGBoost, RandomForest)
        2. Optuna pruning (MedianPruner - stop bad trials early)
        3. Expanded parameter space with regularization
        4. Efficient TPE sampler with warm-starting
        """
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import lightgbm as lgb
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_scaled = self._encode_categoricals_and_scale(X)

        # Get top models from model selection stage
        model_metadata = {}
        for result in context.get('skill_results', []):
            if hasattr(result, 'metadata') and result.metadata:
                model_metadata.update(result.metadata)

        model_ranking = model_metadata.get('model_ranking', ['lightgbm', 'xgboost', 'random_forest'])
        top_models = model_ranking[:3]  # Tune top 3 models

        # Progress tracking
        n_trials_per_model = min(20, context.get('time_budget', 60) // 4)
        total_trials = n_trials_per_model * len(top_models)
        best_score_so_far = [0.0]
        trial_count = [0]
        current_model = ['']

        def progress_callback(study, trial):
            trial_count[0] += 1
            if trial.value and trial.value > best_score_so_far[0]:
                best_score_so_far[0] = trial.value
            pct = (trial_count[0] / total_trials) * 100
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = '▓' * filled + '░' * (bar_len - filled)
            model_short = current_model[0][:8]
            sys.stdout.write(f'\r      Hyperopt [{bar}] {trial_count[0]:2d}/{total_trials} | {model_short} | best={best_score_so_far[0]:.4f}')
            sys.stdout.flush()

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'

        # Store best results per model
        model_results = {}

        # ================================================================
        # TUNE EACH TOP MODEL
        # ================================================================
        for model_name in top_models:
            current_model[0] = model_name

            # Create study with pruner for efficiency
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

            if model_name in ['lightgbm', 'lgb']:
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                        'random_state': 42,
                        'verbose': -1
                    }
                    model = lgb.LGBMClassifier(**params) if problem_type == ProblemType.CLASSIFICATION else lgb.LGBMRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            elif model_name in ['xgboost', 'xgb']:
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'random_state': 42,
                        'verbosity': 0,
                        'eval_metric': 'logloss' if problem_type == ProblemType.CLASSIFICATION else 'rmse'
                    }
                    model = xgb.XGBClassifier(**params) if problem_type == ProblemType.CLASSIFICATION else xgb.XGBRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            elif model_name in ['random_forest', 'rf']:
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    model = RandomForestClassifier(**params) if problem_type == ProblemType.CLASSIFICATION else RandomForestRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            else:
                # Default: LightGBM
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'random_state': 42,
                        'verbose': -1
                    }
                    model = lgb.LGBMClassifier(**params) if problem_type == ProblemType.CLASSIFICATION else lgb.LGBMRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            try:
                study.optimize(objective, n_trials=n_trials_per_model, callbacks=[progress_callback], show_progress_bar=False)
                model_results[model_name] = {
                    'score': study.best_value,
                    'params': study.best_params,
                    'study': study
                }
            except Exception as e:
                logger.debug(f"Hyperopt failed for {model_name}: {e}")

        print()  # New line after progress

        # ================================================================
        # SELECT BEST MODEL AND BUILD
        # ================================================================
        best_model_name = None
        best_score = 0
        best_params = {}

        for model_name, result in model_results.items():
            if result['score'] > best_score:
                best_score = result['score']
                best_model_name = model_name
                best_params = result['params']

        # Build optimized model
        if best_model_name in ['lightgbm', 'lgb']:
            final_params = {**best_params, 'random_state': 42, 'verbose': -1}
            optimized_model = lgb.LGBMClassifier(**final_params) if problem_type == ProblemType.CLASSIFICATION else lgb.LGBMRegressor(**final_params)
        elif best_model_name in ['xgboost', 'xgb']:
            final_params = {**best_params, 'random_state': 42, 'verbosity': 0}
            optimized_model = xgb.XGBClassifier(**final_params) if problem_type == ProblemType.CLASSIFICATION else xgb.XGBRegressor(**final_params)
        elif best_model_name in ['random_forest', 'rf']:
            final_params = {**best_params, 'random_state': 42, 'n_jobs': -1}
            optimized_model = RandomForestClassifier(**final_params) if problem_type == ProblemType.CLASSIFICATION else RandomForestRegressor(**final_params)
        else:
            # Fallback
            optimized_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) if problem_type == ProblemType.CLASSIFICATION else lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

        return SkillResult(
            skill_name="builtin_hyperopt",
            category=SkillCategory.HYPERPARAMETER_OPTIMIZATION,
            success=True,
            data=optimized_model,
            metrics={
                'score': best_score,
                'n_trials': total_trials,
                'best_model': best_model_name,
                'models_tuned': len(model_results)
            },
            metadata={
                'best_params': best_params,
                'all_model_scores': {k: v['score'] for k, v in model_results.items()},
                'best_model_name': best_model_name
            }
        )

    async def _builtin_ensemble(self, X: pd.DataFrame, y: pd.Series,
                                problem_type: ProblemType, context: Dict) -> SkillResult:
        """
        World-class ensemble with multiple strategies:
        1. Weighted Voting - weight by CV score
        2. Stacking - meta-learner on OOF predictions
        3. Greedy Selection - iteratively add models that improve score
        4. Blending - average top diverse models
        """
        from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        import lightgbm as lgb
        import xgboost as xgb

        X_scaled = self._encode_categoricals_and_scale(X)

        # Get optimized model and scores from previous stages
        optimized = context.get('model')
        best_single_score = context.get('score', 0)

        # Get model scores from model selection stage
        model_metadata = {}
        for result in context.get('skill_results', []):
            if hasattr(result, 'metadata') and result.metadata:
                model_metadata.update(result.metadata)

        all_scores = model_metadata.get('all_scores', {})
        oof_predictions = model_metadata.get('oof_predictions', {})

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'

            # Base models with different characteristics for diversity
            base_models = {
                'lgb': lgb.LGBMClassifier(n_estimators=150, random_state=42, verbose=-1),
                'xgb': xgb.XGBClassifier(n_estimators=150, random_state=42, eval_metric='logloss', verbosity=0),
                'rf': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)

        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'

            base_models = {
                'lgb': lgb.LGBMRegressor(n_estimators=150, random_state=42, verbose=-1),
                'xgb': xgb.XGBRegressor(n_estimators=150, random_state=42, verbosity=0),
                'rf': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            }
            meta_learner = Ridge(alpha=1.0)

        ensemble_results = {}

        # ================================================================
        # STRATEGY 1: Weighted Voting (weight by CV score)
        # ================================================================
        try:
            # Calculate weights from scores
            if all_scores:
                weights = []
                estimators = []
                for name, model in base_models.items():
                    score = all_scores.get(name, 0.5)
                    weights.append(max(score, 0.1))  # Min weight 0.1
                    estimators.append((name, model))

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                if problem_type == ProblemType.CLASSIFICATION:
                    weighted_ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
                else:
                    weighted_ensemble = VotingRegressor(estimators=estimators, weights=weights)

                weighted_scores = cross_val_score(weighted_ensemble, X_scaled, y, cv=cv, scoring=scoring)
                ensemble_results['weighted_voting'] = weighted_scores.mean()
        except Exception as e:
            logger.debug(f"Weighted voting failed: {e}")

        # ================================================================
        # STRATEGY 2: Stacking with Meta-Learner
        # ================================================================
        try:
            estimators_list = [(name, model) for name, model in base_models.items()]

            if problem_type == ProblemType.CLASSIFICATION:
                stacking = StackingClassifier(
                    estimators=estimators_list,
                    final_estimator=meta_learner,
                    cv=3,  # Faster inner CV
                    passthrough=False,
                    n_jobs=-1
                )
            else:
                stacking = StackingRegressor(
                    estimators=estimators_list,
                    final_estimator=meta_learner,
                    cv=3,
                    passthrough=False,
                    n_jobs=-1
                )

            stacking_scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring=scoring)
            ensemble_results['stacking'] = stacking_scores.mean()
        except Exception as e:
            logger.debug(f"Stacking failed: {e}")

        # ================================================================
        # STRATEGY 3: Greedy Ensemble Selection
        # ================================================================
        try:
            # Start with best single model, greedily add models that improve
            sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            greedy_estimators = []

            if sorted_models:
                # Start with best model
                best_name = sorted_models[0][0]
                if best_name in base_models:
                    greedy_estimators.append((best_name, base_models[best_name]))

                current_best = all_scores.get(best_name, 0)

                # Try adding each remaining model
                for name, score in sorted_models[1:4]:  # Top 4 models
                    if name in base_models:
                        test_estimators = greedy_estimators + [(name, base_models[name])]

                        if problem_type == ProblemType.CLASSIFICATION:
                            test_ensemble = VotingClassifier(estimators=test_estimators, voting='soft')
                        else:
                            test_ensemble = VotingRegressor(estimators=test_estimators)

                        test_scores = cross_val_score(test_ensemble, X_scaled, y, cv=cv, scoring=scoring)
                        test_score = test_scores.mean()

                        # Add if improves
                        if test_score > current_best:
                            greedy_estimators.append((name, base_models[name]))
                            current_best = test_score

                ensemble_results['greedy'] = current_best
                ensemble_results['greedy_models'] = [e[0] for e in greedy_estimators]
        except Exception as e:
            logger.debug(f"Greedy selection failed: {e}")

        # ================================================================
        # STRATEGY 4: Simple Average (Baseline)
        # ================================================================
        try:
            simple_estimators = [(name, model) for name, model in list(base_models.items())[:3]]

            if problem_type == ProblemType.CLASSIFICATION:
                simple_ensemble = VotingClassifier(estimators=simple_estimators, voting='soft')
            else:
                simple_ensemble = VotingRegressor(estimators=simple_estimators)

            simple_scores = cross_val_score(simple_ensemble, X_scaled, y, cv=cv, scoring=scoring)
            ensemble_results['simple_avg'] = simple_scores.mean()
        except Exception as e:
            logger.debug(f"Simple average failed: {e}")

        # ================================================================
        # STRATEGY 5: MULTI-LEVEL STACKING (10/10 Kaggle Winner Strategy)
        # ================================================================
        # 2-Layer stacking: Layer 1 = diverse base models, Layer 2 = meta-learner
        # This is how top Kaggle solutions achieve SOTA scores
        multi_level_stacking = None
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            from sklearn.linear_model import RidgeClassifier
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            from sklearn.svm import SVC, SVR

            # Layer 1: Diverse base models with different learning paradigms
            if problem_type == ProblemType.CLASSIFICATION:
                layer1_models = {
                    # Boosting family
                    'lgb': lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
                    'xgb': xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0),
                    'histgb': HistGradientBoostingClassifier(max_iter=200, random_state=42),
                    # Bagging family
                    'rf': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                    'et': ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                }

                # Layer 2: Meta-learner (simpler model to combine Layer 1 outputs)
                layer2_meta = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.05,
                    random_state=42, verbose=-1
                )
            else:
                layer1_models = {
                    'lgb': lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
                    'xgb': xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
                    'histgb': HistGradientBoostingRegressor(max_iter=200, random_state=42),
                    'rf': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                    'et': ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                }
                layer2_meta = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.05,
                    random_state=42, verbose=-1
                )

            # Build 2-layer stacking
            layer1_estimators = [(name, model) for name, model in layer1_models.items()]

            if problem_type == ProblemType.CLASSIFICATION:
                multi_level_stacking = StackingClassifier(
                    estimators=layer1_estimators,
                    final_estimator=layer2_meta,
                    cv=5,  # More CV for better OOF predictions
                    passthrough=True,  # Include original features in layer 2
                    n_jobs=-1,
                    stack_method='predict_proba'
                )
            else:
                multi_level_stacking = StackingRegressor(
                    estimators=layer1_estimators,
                    final_estimator=layer2_meta,
                    cv=5,
                    passthrough=True,
                    n_jobs=-1
                )

            multi_level_scores = cross_val_score(multi_level_stacking, X_scaled, y, cv=cv, scoring=scoring)
            ensemble_results['multi_level_stacking'] = multi_level_scores.mean()

        except Exception as e:
            logger.debug(f"Multi-level stacking failed: {e}")

        # ================================================================
        # SELECT BEST STRATEGY
        # ================================================================
        best_ensemble_strategy = None
        best_ensemble_score = 0

        for strategy, score in ensemble_results.items():
            if isinstance(score, (int, float)) and score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble_strategy = strategy

        # Compare ensemble vs single model
        if best_ensemble_score > best_single_score:
            # Build the winning ensemble
            if best_ensemble_strategy == 'multi_level_stacking' and multi_level_stacking is not None:
                final_model = multi_level_stacking
                n_estimators = 5  # 5 layer-1 models + meta-learner
            elif best_ensemble_strategy == 'stacking':
                final_model = stacking
                n_estimators = len(base_models)
            elif best_ensemble_strategy == 'weighted_voting':
                final_model = weighted_ensemble
                n_estimators = len(base_models)
            elif best_ensemble_strategy == 'greedy' and greedy_estimators:
                if problem_type == ProblemType.CLASSIFICATION:
                    final_model = VotingClassifier(estimators=greedy_estimators, voting='soft')
                else:
                    final_model = VotingRegressor(estimators=greedy_estimators)
                n_estimators = len(greedy_estimators)
            else:
                final_model = simple_ensemble
                n_estimators = 3

            final_model.fit(X_scaled, y)
            final_score = best_ensemble_score
            used_ensemble = True
        else:
            # Single model is better - need to fit the optimized model
            final_model = optimized
            final_model.fit(X_scaled, y)  # FIT THE MODEL!
            final_score = best_single_score
            used_ensemble = False
            best_ensemble_strategy = 'single_model'
            n_estimators = 1

        return SkillResult(
            skill_name="builtin_ensemble",
            category=SkillCategory.ENSEMBLE,
            success=True,
            data=final_model,
            metrics={
                'score': final_score,
                'n_estimators': n_estimators,
                'ensemble_score': best_ensemble_score,
                'single_score': best_single_score,
                'strategy': best_ensemble_strategy,
            },
            metadata={
                'all_ensemble_scores': ensemble_results,
                'decision': best_ensemble_strategy,
                'used_ensemble': used_ensemble,
            }
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
