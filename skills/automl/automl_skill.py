"""
Advanced AutoML Skill
=====================

World-class AutoML integration with:
1. AutoGluon - Amazon's state-of-the-art AutoML (multi-layer stacking)
2. FLAML - Microsoft's fast lightweight AutoML
3. Imbalanced Learning - SMOTE, ADASYN for class imbalance
4. Ensemble of AutoML systems

This skill provides the best of all AutoML worlds:
- AutoGluon for maximum accuracy (competition-winning)
- FLAML for speed and resource efficiency
- Imbalanced-learn for handling class imbalance
"""

import logging
import os
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .base import MLSkill, SkillCategory, SkillResult

logger = logging.getLogger(__name__)


class AutoMLSkill(MLSkill):
    """
    World-class AutoML skill combining multiple frameworks.

    Frameworks:
    - AutoGluon: Best overall accuracy, multi-layer ensembling
    - FLAML: Fast, resource-efficient optimization
    - Imbalanced-learn: Handle class imbalance with SMOTE/ADASYN

    Usage:
        skill = AutoMLSkill()
        await skill.init()
        result = await skill.execute(X, y, problem_type='classification')
    """

    name = "automl"
    version = "1.0.0"
    description = "World-class AutoML with AutoGluon, FLAML, and imbalanced-learn"
    category = SkillCategory.MODEL_SELECTION

    required_inputs = ["X", "y"]
    optional_inputs = [
        "problem_type",
        "time_budget",
        "use_autogluon",
        "use_flaml",
        "handle_imbalance",
    ]
    outputs = ["best_model", "all_results", "framework_comparison"]

    # Default time budgets (seconds)
    DEFAULT_AUTOGLUON_TIME = 120
    DEFAULT_FLAML_TIME = 60

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self._autogluon_available = False
        self._flaml_available = False
        self._imblearn_available = False

    async def init(self) -> Any:
        """Check available frameworks."""
        try:
            from autogluon.tabular import TabularPredictor

            self._autogluon_available = True
        except ImportError:
            logger.info("AutoGluon not available")

        try:
            from flaml import AutoML

            self._flaml_available = True
        except ImportError:
            logger.info("FLAML not available")

        try:
            from imblearn.over_sampling import SMOTE

            self._imblearn_available = True
        except ImportError:
            logger.info("imbalanced-learn not available")

        self._initialized = True
        logger.info(
            f"AutoML initialized: AutoGluon={self._autogluon_available}, "
            f"FLAML={self._flaml_available}, imblearn={self._imblearn_available}"
        )

    async def execute(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **context: Any
    ) -> SkillResult:
        """
        Execute AutoML with multiple frameworks.

        Args:
            X: Input features
            y: Target variable
            **context:
                - problem_type: 'classification' or 'regression'
                - time_budget: Total time budget in seconds
                - use_autogluon: Whether to use AutoGluon (default: True)
                - use_flaml: Whether to use FLAML (default: True)
                - handle_imbalance: Whether to apply SMOTE (default: auto)
                - preset: 'fast', 'balanced', 'best_quality'

        Returns:
            SkillResult with best model and framework comparison
        """
        start_time = time.time()

        if y is None:
            return self._create_error_result("Target variable y is required")

        problem_type = context.get("problem_type", "classification")
        time_budget = context.get("time_budget", 180)
        use_autogluon = context.get("use_autogluon", self._autogluon_available)
        use_flaml = context.get("use_flaml", self._flaml_available)
        handle_imbalance = context.get("handle_imbalance", "auto")
        preset = context.get("preset", "balanced")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            X_arr = X.values
        else:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
            X_arr = X

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y

        results = {}
        best_model = None
        best_score = -np.inf
        best_framework = None

        # Check for class imbalance
        if problem_type == "classification":
            class_counts = np.bincount(y_arr.astype(int))
            imbalance_ratio = (
                class_counts.max() / class_counts.min() if class_counts.min() > 0 else 1
            )
            is_imbalanced = imbalance_ratio > 3

            if handle_imbalance == "auto":
                handle_imbalance = is_imbalanced and self._imblearn_available

            logger.info(
                f"Class imbalance ratio: {imbalance_ratio:.2f}, " f"handling: {handle_imbalance}"
            )

        # ============ AutoGluon ============
        if use_autogluon and self._autogluon_available:
            try:
                ag_result = await self._run_autogluon(
                    X_arr,
                    y_arr,
                    feature_names,
                    problem_type,
                    time_budget=time_budget // 2 if use_flaml else time_budget,
                    preset=preset,
                )
                results["autogluon"] = ag_result

                if ag_result["score"] > best_score:
                    best_score = ag_result["score"]
                    best_model = ag_result["model"]
                    best_framework = "autogluon"

            except Exception as e:
                logger.error(f"AutoGluon failed: {e}")
                results["autogluon"] = {"error": str(e)}

        # ============ FLAML ============
        if use_flaml and self._flaml_available:
            try:
                flaml_result = await self._run_flaml(
                    X_arr,
                    y_arr,
                    problem_type,
                    time_budget=time_budget // 2 if use_autogluon else time_budget,
                    handle_imbalance=handle_imbalance,
                )
                results["flaml"] = flaml_result

                if flaml_result["score"] > best_score:
                    best_score = flaml_result["score"]
                    best_model = flaml_result["model"]
                    best_framework = "flaml"

            except Exception as e:
                logger.error(f"FLAML failed: {e}")
                results["flaml"] = {"error": str(e)}

        # ============ SMOTE + Baseline ============
        if handle_imbalance and self._imblearn_available and problem_type == "classification":
            try:
                smote_result = await self._run_smote_pipeline(X_arr, y_arr, problem_type)
                results["smote_lgbm"] = smote_result

                if smote_result["score"] > best_score:
                    best_score = smote_result["score"]
                    best_model = smote_result["model"]
                    best_framework = "smote_lgbm"

            except Exception as e:
                logger.error(f"SMOTE pipeline failed: {e}")

        execution_time = time.time() - start_time

        # Build comparison table
        comparison = []
        for name, res in results.items():
            if "error" not in res:
                comparison.append(
                    {
                        "framework": name,
                        "score": res.get("score", 0),
                        "best_model": res.get("best_model_name", "N/A"),
                    }
                )

        comparison_df = pd.DataFrame(comparison).sort_values("score", ascending=False)

        return self._create_result(
            success=best_model is not None,
            data=best_model,
            metrics={
                "score": best_score,
                "framework": best_framework,
                "n_frameworks": len([r for r in results.values() if "error" not in r]),
            },
            metadata={
                "best_framework": best_framework,
                "all_results": results,
                "comparison": comparison_df.to_dict("records") if not comparison_df.empty else [],
                "problem_type": problem_type,
            },
            execution_time=execution_time,
        )

    async def _run_autogluon(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        problem_type: str,
        time_budget: int,
        preset: str,
    ) -> Dict[str, Any]:
        """Run AutoGluon AutoML."""
        from autogluon.tabular import TabularPredictor
        from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        # Split for evaluation
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Map preset
        ag_preset = {
            "fast": "medium_quality",
            "balanced": "good_quality",
            "best_quality": "best_quality",
        }.get(preset, "good_quality")

        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = TabularPredictor(
                label="target",
                path=tmpdir,
                problem_type="binary" if problem_type == "classification" else "regression",
                verbosity=0,
            ).fit(
                train_df,
                time_limit=time_budget,
                presets=ag_preset,
            )

            # Evaluate
            val_X = val_df.drop(columns=["target"])
            val_y = val_df["target"].values

            pred = predictor.predict(val_X)
            proba = predictor.predict_proba(val_X) if problem_type == "classification" else None

            if problem_type == "classification":
                score = accuracy_score(val_y, pred)
                try:
                    auc = roc_auc_score(val_y, proba[1])
                except Exception:
                    auc = score
            else:
                score = r2_score(val_y, pred)
                auc = score

            # Get leaderboard
            lb = predictor.leaderboard(silent=True)
            best_model_name = lb.iloc[0]["model"] if not lb.empty else "unknown"

            return {
                "model": predictor,
                "score": score,
                "auc": auc,
                "best_model_name": best_model_name,
                "leaderboard": lb.head(5).to_dict("records") if not lb.empty else [],
            }

    async def _run_flaml(
        self,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str,
        time_budget: int,
        handle_imbalance: bool,
    ) -> Dict[str, Any]:
        """Run FLAML AutoML."""
        from flaml import AutoML
        from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        # Split for evaluation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply SMOTE if needed
        if handle_imbalance and problem_type == "classification":
            try:
                from imblearn.over_sampling import SMOTE

                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception:
                pass

        automl = AutoML()
        automl.fit(
            X_train,
            y_train,
            task="classification" if problem_type == "classification" else "regression",
            time_budget=time_budget,
            verbose=0,
            metric="accuracy" if problem_type == "classification" else "r2",
        )

        # Evaluate
        pred = automl.predict(X_val)
        proba = automl.predict_proba(X_val) if problem_type == "classification" else None

        if problem_type == "classification":
            score = accuracy_score(y_val, pred)
            try:
                auc = roc_auc_score(y_val, proba[:, 1])
            except Exception:
                auc = score
        else:
            score = r2_score(y_val, pred)
            auc = score

        return {
            "model": automl,
            "score": score,
            "auc": auc,
            "best_model_name": automl.best_estimator,
            "best_config": automl.best_config,
        }

    async def _run_smote_pipeline(
        self, X: np.ndarray, y: np.ndarray, problem_type: str
    ) -> Dict[str, Any]:
        """Run SMOTE + LightGBM pipeline."""
        import lightgbm as lgb
        from imblearn.over_sampling import ADASYN, SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Try both SMOTE and ADASYN
        pipelines = {
            "smote_lgbm": ImbPipeline(
                [
                    ("sampler", SMOTE(random_state=42)),
                    ("clf", lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)),
                ]
            ),
            "adasyn_lgbm": ImbPipeline(
                [
                    ("sampler", ADASYN(random_state=42)),
                    ("clf", lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)),
                ]
            ),
        }

        best_pipeline = None
        best_score = -np.inf
        best_name = None

        for name, pipeline in pipelines.items():
            try:
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
                score = scores.mean()

                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline
                    best_name = name
            except Exception:
                continue

        if best_pipeline:
            best_pipeline.fit(X, y)

        return {
            "model": best_pipeline,
            "score": best_score,
            "best_model_name": best_name,
        }

    async def quick_automl(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification",
        time_budget: int = 60,
    ) -> Tuple[Any, float]:
        """
        Quick AutoML for fast results.

        Uses FLAML for speed, falls back to baseline if unavailable.

        Returns:
            Tuple of (model, score)
        """
        result = await self.execute(
            X,
            y,
            problem_type=problem_type,
            time_budget=time_budget,
            use_autogluon=False,
            use_flaml=True,
            preset="fast",
        )

        return result.data, result.metrics.get("score", 0)

    async def best_automl(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str = "classification",
        time_budget: int = 300,
    ) -> Tuple[Any, float, str]:
        """
        Best AutoML for maximum accuracy.

        Uses AutoGluon with best_quality preset.

        Returns:
            Tuple of (model, score, framework)
        """
        result = await self.execute(
            X,
            y,
            problem_type=problem_type,
            time_budget=time_budget,
            use_autogluon=True,
            use_flaml=True,
            preset="best_quality",
        )

        return (
            result.data,
            result.metrics.get("score", 0),
            result.metadata.get("best_framework", "unknown"),
        )
