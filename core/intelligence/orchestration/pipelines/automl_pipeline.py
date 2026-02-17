"""
AutoMLWorkflow - Skill-Based AutoML Pipeline
============================================

Orchestrates AutoML skills to solve any machine learning problem.
Uses skills from the registry instead of hard-coded mixins.

Usage:
    workflow = AutoMLWorkflow()
    result = await workflow.solve(X, y)

The workflow:
1. Auto-detects problem type (classification/regression)
2. Discovers relevant skills from registry
3. Chains skills in optimal order
4. Executes pipeline and returns best model

Architecture:
- Standalone workflow class (no BaseWorkflow yet)
- Calls skills from skills/automl/ via registry
- No hard-coded business logic (uses skill orchestration)
- Configurable pipeline order (not hard-coded)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from Jotty.core.capabilities.registry.skills_registry import get_skills_registry

logger = logging.getLogger(__name__)


# ============================================================
# TYPES (shared with old SkillOrchestrator for compatibility)
# ============================================================


class ProblemType(str, Enum):
    """ML problem types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class SkillCategory(str, Enum):
    """ML skill categories."""

    DATA_PROFILING = "data_profiling"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ENSEMBLE = "ensemble"
    EVALUATION = "evaluation"
    EXPLANATION = "explanation"


# ============================================================
# AUTOML WORKFLOW
# ============================================================


class AutoMLWorkflow:
    """
    Orchestrates AutoML skills to solve any machine learning problem.

    Key differences from old SkillOrchestrator:
    - Uses skills from registry (not mixins)
    - Lives in execution layer (not intelligence/orchestration)
    - Configurable pipeline (not hard-coded)
    - Properly separated: WHAT (skills) vs HOW (workflow orchestration)
    """

    # Default pipeline order (can be overridden via constructor)
    DEFAULT_PIPELINE_ORDER = [
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

    def __init__(
        self,
        pipeline_order: Optional[List[SkillCategory]] = None,
        use_llm_features: bool = True,
        show_progress: bool = True,
    ) -> None:
        """
        Initialize AutoML workflow.

        Args:
            pipeline_order: Custom pipeline order (None = use default)
            use_llm_features: Enable LLM-powered feature reasoning
            show_progress: Show progress bars
        """
        self.pipeline_order = pipeline_order or self.DEFAULT_PIPELINE_ORDER
        self._skills_registry = None
        self._initialized = False
        self._use_llm_features = use_llm_features
        self._show_progress = show_progress
        self._available_skills: Dict[SkillCategory, List[Any]] = {}

    async def init(self) -> None:
        """Initialize skills registry and discover available skills."""
        if self._initialized:
            return

        try:
            self._skills_registry = get_skills_registry()
            self._skills_registry.init()  # type: ignore[attr-defined]

            # Discover skills for each category
            self._discover_skills()
            self._initialized = True

            logger.info(
                f"AutoMLWorkflow initialized with skills for {len(self._available_skills)} categories"
            )

        except Exception as e:
            logger.warning(f"AutoMLWorkflow init failed: {e}")
            self._initialized = True  # Continue with fallback implementations

    def _discover_skills(self) -> None:
        """Discover AutoML skills from registry."""
        automl_keywords = [
            "automl",
            "eda",
            "feature",
            "model",
            "hyperopt",
            "ensemble",
            "selection",
            "engineering",
        ]

        for skill_name, skill_def in self._skills_registry.loaded_skills.items():  # type: ignore[attr-defined]
            name_lower = skill_name.lower()

            # Check if it's an AutoML skill
            if any(kw in name_lower for kw in automl_keywords):
                # Try to categorize the skill
                category = self._categorize_skill(skill_name)
                if category:
                    if category not in self._available_skills:
                        self._available_skills[category] = []
                    self._available_skills[category].append((skill_name, skill_def))
                    logger.debug(f"Discovered {skill_name} for {category.value}")

    def _categorize_skill(self, skill_name: str) -> Optional[SkillCategory]:
        """Categorize a skill based on its name."""
        name_lower = skill_name.lower()

        if "eda" in name_lower or "profil" in name_lower:
            return SkillCategory.DATA_PROFILING
        elif "clean" in name_lower:
            return SkillCategory.DATA_CLEANING
        elif "feature_engineering" in name_lower or "engineer" in name_lower:
            return SkillCategory.FEATURE_ENGINEERING
        elif "feature_selection" in name_lower or "selection" in name_lower:
            return SkillCategory.FEATURE_SELECTION
        elif "model_selection" in name_lower:
            return SkillCategory.MODEL_SELECTION
        elif "hyperopt" in name_lower or "hyperparam" in name_lower:
            return SkillCategory.HYPERPARAMETER_OPTIMIZATION
        elif "ensemble" in name_lower:
            return SkillCategory.ENSEMBLE
        elif "evaluat" in name_lower or "metric" in name_lower:
            return SkillCategory.EVALUATION
        elif "explain" in name_lower or "shap" in name_lower:
            return SkillCategory.EXPLANATION

        return None

    def _detect_problem_type(self, y: pd.Series) -> ProblemType:
        """Auto-detect problem type from target variable."""
        if y is None:
            return ProblemType.CLUSTERING

        unique_ratio = y.nunique() / len(y)
        n_unique = y.nunique()

        # Check dtype
        if y.dtype == "object" or y.dtype == "bool":
            return ProblemType.CLASSIFICATION

        # Check if categorical (few unique values)
        if n_unique <= 20 and unique_ratio < 0.05:
            return ProblemType.CLASSIFICATION

        return ProblemType.REGRESSION

    async def solve(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        problem_type: str = "auto",
        target_metric: str = "auto",
        time_budget: int = 300,
        business_context: str = "",
        on_stage_complete: Callable | None = None,
    ) -> Dict[str, Any]:
        """
        Solve any ML problem by orchestrating skills.

        Args:
            X: Feature dataframe
            y: Target series (None for clustering)
            problem_type: "classification", "regression", "clustering", or "auto"
            target_metric: Metric to optimize ("auto" to infer)
            time_budget: Max seconds for optimization
            business_context: Optional business context for LLM feature reasoning
            on_stage_complete: Callback for stage completion

        Returns:
            Dict with best model, score, and insights
        """
        await self.init()

        logger.info("=" * 60)
        logger.info("AUTOML WORKFLOW - SKILL-BASED PIPELINE")
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

        logger.info(
            f"Problem: {prob_type.value} | Metric: {target_metric} | "
            f"Features: {X.shape[1]} | Samples: {len(X)}"
        )
        logger.info("=" * 60)

        # Step 3: Build execution context
        context = {
            "X": X.copy(),
            "y": y.copy() if y is not None else None,
            "X_original": X.copy(),
            "problem_type": prob_type,
            "target_metric": target_metric,
            "time_budget": time_budget,
            "business_context": business_context,
        }

        stage_results = []

        # Step 4: Execute pipeline stages using skills
        for category in self.pipeline_order:
            logger.info(f"Executing stage: {category.value}")

            result = await self._execute_stage(category, context)
            stage_results.append(result)

            if result.get("success"):
                # Update context based on stage outputs
                self._update_context(context, category, result)

                if on_stage_complete:
                    try:
                        on_stage_complete(category.value, result)
                    except Exception as e:
                        logger.debug(f"Stage callback failed: {e}")

        # Step 5: Compile final result
        final_result = {
            "success": True,
            "problem_type": prob_type.value,
            "best_score": context.get("score", 0),
            "best_model": context.get("model"),
            "feature_count": context["X"].shape[1],
            "stage_results": stage_results,
            "feature_importance": context.get("feature_importance", {}),
            "processed_X": context["X"],
        }

        logger.info("=" * 60)
        logger.info(
            f"COMPLETE | Score: {final_result['best_score']:.4f} | "
            f"Features: {X.shape[1]} -> {final_result['feature_count']}"
        )
        logger.info("=" * 60)

        return final_result

    async def _execute_stage(self, category: SkillCategory, context: Dict) -> Dict[str, Any]:
        """Execute a pipeline stage using available skills."""
        # Get skills for this category
        skills = self._available_skills.get(category, [])

        if skills:
            # Try each skill until one succeeds
            for skill_name, skill_def in skills:
                try:
                    # Call skill via registry
                    result = await self._call_skill(skill_name, skill_def, context)
                    if result.get("success"):
                        return result
                except Exception as e:
                    logger.debug(f"Skill {skill_name} failed: {e}")
                    continue

        # Fallback to built-in implementation
        logger.debug(f"No skills available for {category.value}, using fallback")
        return await self._builtin_stage(category, context)

    async def _call_skill(self, skill_name: str, skill_def: Any, context: Dict) -> Dict[str, Any]:
        """Call a skill from the registry."""
        # This is a simplified version - in production, we'd use proper skill calling
        # For now, assume skills have an execute() method
        try:
            if hasattr(skill_def, "execute"):
                result = await skill_def.execute(context)
                return {"success": True, "data": result}
            else:
                return {"success": False, "error": "Skill has no execute method"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _update_context(self, context: Dict, category: SkillCategory, result: Dict) -> None:
        """Update context based on stage results."""
        if not result.get("success"):
            return

        data = result.get("data")
        if data is None:
            return

        # Update X for data transformation stages
        if category in [
            SkillCategory.DATA_CLEANING,
            SkillCategory.FEATURE_ENGINEERING,
            SkillCategory.FEATURE_SELECTION,
        ]:
            if isinstance(data, pd.DataFrame):
                context["X"] = data

        # Update model for model stages
        elif category in [
            SkillCategory.MODEL_SELECTION,
            SkillCategory.HYPERPARAMETER_OPTIMIZATION,
            SkillCategory.ENSEMBLE,
        ]:
            context["model"] = data
            if "score" in result:
                context["score"] = result["score"]

        # Update feature importance
        if "feature_importance" in result:
            context["feature_importance"] = result["feature_importance"]

    async def _builtin_stage(self, category: SkillCategory, context: Dict) -> Dict[str, Any]:
        """Fallback built-in implementations for each stage."""
        X = context.get("X")
        y = context.get("y")

        try:
            if category == SkillCategory.DATA_PROFILING:
                return self._builtin_profiling(X, y)
            elif category == SkillCategory.DATA_CLEANING:
                return self._builtin_cleaning(X, y)
            else:
                # For other stages, return minimal success
                return {
                    "success": True,
                    "stage": category.value,
                    "message": "Skipped (no skill available)",
                }
        except Exception as e:
            return {"success": False, "stage": category.value, "error": str(e)}

    def _builtin_profiling(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Built-in data profiling."""
        profile = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "missing_total": int(X.isnull().sum().sum()),
            "missing_pct": float(X.isnull().sum().sum() / X.size * 100),
            "numeric_cols": len(X.select_dtypes(include=[np.number]).columns),
            "categorical_cols": len(X.select_dtypes(include=["object"]).columns),
        }

        if y is not None:
            profile["target_unique"] = int(y.nunique())
            profile["target_missing"] = int(y.isnull().sum())

        return {
            "success": True,
            "stage": "data_profiling",
            "metrics": profile,
        }

    def _builtin_cleaning(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Built-in data cleaning."""
        X_clean = X.copy()

        # Handle missing values
        for col in X_clean.columns:
            if X_clean[col].dtype in ["int64", "float64"]:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                X_clean[col] = X_clean[col].fillna(
                    X_clean[col].mode().iloc[0] if len(X_clean[col].mode()) > 0 else "MISSING"
                )

        return {
            "success": True,
            "stage": "data_cleaning",
            "data": X_clean,
            "metrics": {"cleaned_missing": int(X.isnull().sum().sum())},
        }


# ============================================================
# SINGLETON ACCESSOR
# ============================================================

_workflow_instance = None


def get_automl_workflow() -> AutoMLWorkflow:
    """Get or create the AutoML workflow singleton."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = AutoMLWorkflow()
    return _workflow_instance


def reset_automl_workflow() -> None:
    """Reset the singleton workflow (for testing)."""
    global _workflow_instance
    _workflow_instance = None
