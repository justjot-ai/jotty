"""
Ensemble Skill
==============

World-class ensemble with multiple strategies:
1. Weighted Voting - weight by CV score
2. Stacking - meta-learner on OOF predictions
3. Greedy Selection - iteratively add models that improve score
4. Blending - average top diverse models
5. Multi-Level Stacking - 2-layer Kaggle winner strategy
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from .base import MLSkill, SkillResult, SkillCategory

logger = logging.getLogger(__name__)


class EnsembleSkill(MLSkill):
    """
    World-class ensemble skill.

    Implements multiple ensemble strategies and selects the best.
    """

    name = "ensemble"
    version = "2.0.0"
    description = "Multi-strategy ensemble with multi-level stacking"
    category = SkillCategory.ENSEMBLE

    required_inputs = ["X", "y"]
    optional_inputs = ["problem_type", "optimized_model", "best_single_score", "all_scores"]
    outputs = ["final_model", "ensemble_score"]

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)

    async def execute(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **context: Any) -> SkillResult:
        """
        Execute ensemble building.

        Args:
            X: Input features
            y: Target variable (required)
            **context: problem_type, optimized_model, best_single_score, all_scores, etc.

        Returns:
            SkillResult with ensemble model
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        import lightgbm as lgb
        import xgboost as xgb

        start_time = time.time()

        if y is None:
            return self._create_error_result("Target variable y is required")

        if not self.validate_inputs(X, y):
            return self._create_error_result("Invalid inputs")

        problem_type = context.get('problem_type', 'classification')
        optimized_model = context.get('optimized_model')
        best_single_score = context.get('best_single_score', 0)
        all_scores = context.get('all_scores', {})

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if problem_type == 'classification':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'
            base_models = self._get_classification_base_models()
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'
            base_models = self._get_regression_base_models()
            meta_learner = Ridge(alpha=1.0)

        ensemble_results = {}
        stacking = None
        weighted_ensemble = None
        greedy_estimators = []

        # Strategy 1: Weighted Voting
        try:
            weighted_ensemble, weighted_score = self._weighted_voting(
                base_models, all_scores, X_scaled, y, cv, scoring, problem_type
            )
            ensemble_results['weighted_voting'] = weighted_score
        except Exception as e:
            logger.debug(f"Weighted voting failed: {e}")

        # Strategy 2: Stacking
        try:
            stacking, stacking_score = self._stacking(
                base_models, meta_learner, X_scaled, y, cv, scoring, problem_type
            )
            ensemble_results['stacking'] = stacking_score
        except Exception as e:
            logger.debug(f"Stacking failed: {e}")

        # Strategy 3: Greedy Ensemble
        try:
            greedy_estimators, greedy_score = self._greedy_selection(
                base_models, all_scores, X_scaled, y, cv, scoring, problem_type
            )
            ensemble_results['greedy'] = greedy_score
            ensemble_results['greedy_models'] = [e[0] for e in greedy_estimators]
        except Exception as e:
            logger.debug(f"Greedy selection failed: {e}")

        # Strategy 4: Simple Average
        try:
            simple_ensemble, simple_score = self._simple_average(
                base_models, X_scaled, y, cv, scoring, problem_type
            )
            ensemble_results['simple_avg'] = simple_score
        except Exception as e:
            logger.debug(f"Simple average failed: {e}")

        # Strategy 5: Multi-Level Stacking (10/10 Kaggle Winner)
        multi_level_stacking = None
        try:
            multi_level_stacking, ml_score = self._multi_level_stacking(
                X_scaled, y, cv, scoring, problem_type
            )
            ensemble_results['multi_level_stacking'] = ml_score
        except Exception as e:
            logger.debug(f"Multi-level stacking failed: {e}")

        # Select best strategy
        best_ensemble_strategy = None
        best_ensemble_score = 0

        for strategy, score in ensemble_results.items():
            if isinstance(score, (int, float)) and score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble_strategy = strategy

        # Compare ensemble vs single model
        if best_ensemble_score > best_single_score:
            final_model, n_estimators = self._build_winning_ensemble(
                best_ensemble_strategy, multi_level_stacking, stacking,
                weighted_ensemble, greedy_estimators, base_models, simple_ensemble,
                problem_type
            )
            final_model.fit(X_scaled, y)
            final_score = best_ensemble_score
            used_ensemble = True
        else:
            final_model = optimized_model
            if final_model is not None:
                final_model.fit(X_scaled, y)
            final_score = best_single_score
            used_ensemble = False
            best_ensemble_strategy = 'single_model'
            n_estimators = 1

        execution_time = time.time() - start_time

        return self._create_result(
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
            },
            execution_time=execution_time,
        )

    def _get_classification_base_models(self) -> Dict:
        """Get classification base models."""
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier

        return {
            'lgb': lgb.LGBMClassifier(n_estimators=150, random_state=42, verbose=-1),
            'xgb': xgb.XGBClassifier(n_estimators=150, random_state=42, eval_metric='logloss', verbosity=0),
            'rf': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

    def _get_regression_base_models(self) -> Dict:
        """Get regression base models."""
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import GradientBoostingRegressor

        return {
            'lgb': lgb.LGBMRegressor(n_estimators=150, random_state=42, verbose=-1),
            'xgb': xgb.XGBRegressor(n_estimators=150, random_state=42, verbosity=0),
            'rf': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

    def _weighted_voting(self, base_models: Any, all_scores: Any, X_scaled: Any, y: Any, cv: Any, scoring: Any, problem_type: Any) -> Tuple:
        """Weighted voting ensemble."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        if all_scores:
            weights = []
            estimators = []
            for name, model in base_models.items():
                score = all_scores.get(name, 0.5)
                weights.append(max(score, 0.1))
                estimators.append((name, model))

            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            if problem_type == 'classification':
                ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
            else:
                ensemble = VotingRegressor(estimators=estimators, weights=weights)

            scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring=scoring)
            return ensemble, scores.mean()

        return None, 0

    def _stacking(self, base_models: Any, meta_learner: Any, X_scaled: Any, y: Any, cv: Any, scoring: Any, problem_type: Any) -> Tuple:
        """Stacking ensemble."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import StackingClassifier, StackingRegressor

        estimators_list = [(name, model) for name, model in base_models.items()]

        if problem_type == 'classification':
            stacking = StackingClassifier(
                estimators=estimators_list,
                final_estimator=meta_learner,
                cv=3,
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

        scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring=scoring)
        return stacking, scores.mean()

    def _greedy_selection(self, base_models: Any, all_scores: Any, X_scaled: Any, y: Any, cv: Any, scoring: Any, problem_type: Any) -> Tuple:
        """Greedy ensemble selection."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        greedy_estimators = []

        if sorted_models:
            best_name = sorted_models[0][0]
            if best_name in base_models:
                greedy_estimators.append((best_name, base_models[best_name]))

            current_best = all_scores.get(best_name, 0)

            for name, score in sorted_models[1:4]:
                if name in base_models:
                    test_estimators = greedy_estimators + [(name, base_models[name])]

                    if problem_type == 'classification':
                        test_ensemble = VotingClassifier(estimators=test_estimators, voting='soft')
                    else:
                        test_ensemble = VotingRegressor(estimators=test_estimators)

                    test_scores = cross_val_score(test_ensemble, X_scaled, y, cv=cv, scoring=scoring)
                    test_score = test_scores.mean()

                    if test_score > current_best:
                        greedy_estimators.append((name, base_models[name]))
                        current_best = test_score

            return greedy_estimators, current_best

        return [], 0

    def _simple_average(self, base_models: Any, X_scaled: Any, y: Any, cv: Any, scoring: Any, problem_type: Any) -> Tuple:
        """Simple average ensemble."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        simple_estimators = [(name, model) for name, model in list(base_models.items())[:3]]

        if problem_type == 'classification':
            ensemble = VotingClassifier(estimators=simple_estimators, voting='soft')
        else:
            ensemble = VotingRegressor(estimators=simple_estimators)

        scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring=scoring)
        return ensemble, scores.mean()

    def _multi_level_stacking(self, X_scaled: Any, y: Any, cv: Any, scoring: Any, problem_type: Any) -> Tuple:
        """Multi-level stacking (10/10 Kaggle Winner Strategy)."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import lightgbm as lgb
        import xgboost as xgb

        if problem_type == 'classification':
            layer1_models = {
                'lgb': lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
                'xgb': xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', verbosity=0),
                'histgb': HistGradientBoostingClassifier(max_iter=200, random_state=42),
                'rf': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                'et': ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            }
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

        layer1_estimators = [(name, model) for name, model in layer1_models.items()]

        if problem_type == 'classification':
            stacking = StackingClassifier(
                estimators=layer1_estimators,
                final_estimator=layer2_meta,
                cv=5,
                passthrough=True,
                n_jobs=-1,
                stack_method='predict_proba'
            )
        else:
            stacking = StackingRegressor(
                estimators=layer1_estimators,
                final_estimator=layer2_meta,
                cv=5,
                passthrough=True,
                n_jobs=-1
            )

        scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring=scoring)
        return stacking, scores.mean()

    def _build_winning_ensemble(self, strategy: Any, multi_level_stacking: Any, stacking: Any, weighted_ensemble: Any, greedy_estimators: Any, base_models: Any, simple_ensemble: Any, problem_type: Any) -> Any:
        """Build the winning ensemble model."""
        from sklearn.ensemble import VotingClassifier, VotingRegressor

        if strategy == 'multi_level_stacking' and multi_level_stacking is not None:
            return multi_level_stacking, 5

        elif strategy == 'stacking' and stacking is not None:
            return stacking, len(base_models)

        elif strategy == 'weighted_voting' and weighted_ensemble is not None:
            return weighted_ensemble, len(base_models)

        elif strategy == 'greedy' and greedy_estimators:
            if problem_type == 'classification':
                return VotingClassifier(estimators=greedy_estimators, voting='soft'), len(greedy_estimators)
            else:
                return VotingRegressor(estimators=greedy_estimators), len(greedy_estimators)

        else:
            return simple_ensemble, 3

    @staticmethod
    def extract_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract feature importance from any ensemble model type.

        Handles:
        - VotingClassifier/VotingRegressor - averages base estimator importances
        - StackingClassifier/StackingRegressor - combines base + final estimator
        - Direct models with feature_importances_

        Args:
            model: The fitted ensemble model
            feature_names: List of feature names

        Returns:
            Dict mapping feature name to importance score
        """
        import numpy as np
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from sklearn.ensemble import StackingClassifier, StackingRegressor

        feature_importance = {}

        try:
            # Case 1: Direct feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                if len(imp) == len(feature_names):
                    for name, score in zip(feature_names, imp):
                        feature_importance[name] = float(score)
                    return feature_importance

            # Case 2: VotingClassifier/VotingRegressor
            if isinstance(model, (VotingClassifier, VotingRegressor)):
                importances = []
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        imp = estimator.feature_importances_
                        if len(imp) == len(feature_names):
                            importances.append(imp)

                if importances:
                    avg_imp = np.mean(importances, axis=0)
                    for name, score in zip(feature_names, avg_imp):
                        feature_importance[name] = float(score)
                    return feature_importance

            # Case 3: StackingClassifier/StackingRegressor
            if isinstance(model, (StackingClassifier, StackingRegressor)):
                importances = []

                # Get importances from base estimators
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        imp = estimator.feature_importances_
                        # Stacking may have passthrough, check length
                        if len(imp) >= len(feature_names):
                            importances.append(imp[:len(feature_names)])

                # Get importances from final estimator (if it has feature_importances_)
                if hasattr(model.final_estimator_, 'feature_importances_'):
                    final_imp = model.final_estimator_.feature_importances_

                    # Final estimator sees: [base_predictions...] + [original_features if passthrough]
                    # We care about the original feature contributions
                    n_estimators = len(model.estimators_)

                    if model.passthrough and len(final_imp) > n_estimators:
                        # Extract importance for original features
                        original_imp = final_imp[n_estimators:n_estimators + len(feature_names)]
                        if len(original_imp) == len(feature_names):
                            # Weight final estimator's view more (it sees the meta-level)
                            importances.append(original_imp * 1.5)

                if importances:
                    # Average and normalize
                    avg_imp = np.mean(importances, axis=0)
                    avg_imp = avg_imp / (avg_imp.sum() + 1e-10)
                    for name, score in zip(feature_names, avg_imp):
                        feature_importance[name] = float(score)
                    return feature_importance

            # Case 4: Try to access estimators_ directly
            if hasattr(model, 'estimators_'):
                importances = []
                estimators = model.estimators_

                # Handle different formats
                if isinstance(estimators, dict):
                    estimators = list(estimators.values())
                elif hasattr(estimators, 'items'):
                    estimators = list(dict(estimators).values())

                for est in estimators:
                    if hasattr(est, 'feature_importances_'):
                        imp = est.feature_importances_
                        if len(imp) == len(feature_names):
                            importances.append(imp)

                if importances:
                    avg_imp = np.mean(importances, axis=0)
                    for name, score in zip(feature_names, avg_imp):
                        feature_importance[name] = float(score)
                    return feature_importance

            # Case 5: Fallback - train a quick model to get importance
            logger.debug("No direct feature importance available, using fallback")

        except Exception as e:
            logger.debug(f"Feature importance extraction failed: {e}")

        # Return empty dict if nothing worked
        return feature_importance
