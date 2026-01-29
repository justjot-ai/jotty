"""
Model Selection Skill
=====================

World-class model selection with:
1. Extended model zoo (7+ algorithms)
2. Data-adaptive configurations
3. Cross-validation with OOF predictions for stacking
4. Model diversity tracking for ensemble
"""

import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

from .base import MLSkill, SkillResult, SkillCategory

logger = logging.getLogger(__name__)


class ModelSelectionSkill(MLSkill):
    """
    World-class model selection skill.

    Evaluates multiple algorithms and selects the best for the problem.
    """

    name = "model_selection"
    version = "2.0.0"
    description = "Multi-model evaluation with 7+ algorithms"
    category = SkillCategory.MODEL_SELECTION

    required_inputs = ["X", "y"]
    optional_inputs = ["problem_type"]
    outputs = ["best_model", "all_scores", "oof_predictions"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    async def execute(self,
                      X: pd.DataFrame,
                      y: Optional[pd.Series] = None,
                      **context) -> SkillResult:
        """
        Execute model selection.

        Args:
            X: Input features
            y: Target variable (required)
            **context: problem_type, etc.

        Returns:
            SkillResult with best model and scores
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

        start_time = time.time()

        if y is None:
            return self._create_error_result("Target variable y is required")

        if not self.validate_inputs(X, y):
            return self._create_error_result("Invalid inputs")

        problem_type = context.get('problem_type', 'classification')
        n_samples, n_features = X.shape

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Adaptive configuration based on data size
        n_estimators = 100 if n_samples < 5000 else 200
        max_depth_tree = None if n_samples > 1000 else 10

        # Build model zoo
        if problem_type == 'classification':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'
            models = self._get_classification_models(n_estimators, max_depth_tree, n_samples)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'
            models = self._get_regression_models(n_estimators, max_depth_tree, n_samples)

        # Evaluate all models
        best_model = None
        best_name = None
        best_score = -np.inf
        all_scores = {}
        all_std = {}
        oof_predictions = {}

        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
                mean_score = scores.mean()
                std_score = scores.std()
                all_scores[name] = mean_score
                all_std[name] = std_score

                # Collect OOF predictions for top models
                if mean_score > best_score * 0.95:
                    try:
                        if problem_type == 'classification':
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

        # Rank models
        sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        execution_time = time.time() - start_time

        return self._create_result(
            success=True,
            data=best_model,
            metrics={
                'score': best_score,
                'model': best_name,
                **{k: all_scores[k] for k in list(all_scores)[:4]}
            },
            metadata={
                'best_model': best_name,
                'all_scores': all_scores,
                'all_std': all_std,
                'model_ranking': [m[0] for m in sorted_models],
                'oof_predictions': oof_predictions,
                'scaler': scaler,
                'X_scaled': X_scaled,
            },
            execution_time=execution_time,
        )

    def _get_classification_models(self, n_estimators: int, max_depth: int,
                                    n_samples: int) -> Dict:
        """Get classification models including CatBoost."""
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        models = {
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
            'random_forest': RandomForestClassifier(
                n_estimators=n_estimators, random_state=42,
                max_depth=max_depth, n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=n_estimators, random_state=42,
                max_depth=max_depth, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=min(n_estimators, 100), random_state=42,
                learning_rate=0.1
            ),
            'logistic': LogisticRegression(
                max_iter=1000, random_state=42, C=1.0
            ),
        }

        # Add CatBoost (handles categoricals natively - no leakage!)
        try:
            from catboost import CatBoostClassifier
            models['catboost'] = CatBoostClassifier(
                iterations=n_estimators, random_state=42,
                verbose=False, learning_rate=0.1
            )
        except ImportError:
            pass

        if n_samples < 5000:
            models['svm'] = SVC(probability=True, random_state=42, C=1.0)

        return models

    def _get_regression_models(self, n_estimators: int, max_depth: int,
                                n_samples: int) -> Dict:
        """Get regression models including CatBoost."""
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.svm import SVR

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

        # Add CatBoost
        try:
            from catboost import CatBoostRegressor
            models['catboost'] = CatBoostRegressor(
                iterations=n_estimators, random_state=42,
                verbose=False, learning_rate=0.1
            )
        except ImportError:
            pass

        if n_samples < 5000:
            models['svr'] = SVR(C=1.0)

        return models
