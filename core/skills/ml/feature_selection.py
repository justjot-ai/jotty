"""
Feature Selection Skill
=======================

World-class feature selection using multiple advanced techniques.

Techniques:
1. CORRELATION FILTER - Remove highly correlated features
2. NULL IMPORTANCE - Compare real vs shuffled target importance
3. PERMUTATION IMPORTANCE - Measure actual impact on CV score
4. MULTI-MODEL VOTING - Ensemble importance from LightGBM, XGBoost, RF
5. STABILITY SELECTION - Features consistently important across seeds
6. BORUTA-LIKE - Compare against shadow (shuffled) features
7. SHAP - TreeExplainer-based importance
8. SUCCESSIVE HALVING - Progressive elimination of weak features
9. HYPERBAND - Multi-fidelity feature selection with brackets
10. RECURSIVE FEATURE ELIMINATION - Backward elimination with CV
11. DIVERSE RF TREES - Multiple RF configurations for robustness
12. PCA IMPORTANCE - Variance-based feature scoring
13. BOHB - Bayesian Optimized Hyperband for feature subset search (NEW)
14. PASHA - Progressive Adaptive Successive Halving with parallel workers (NEW)
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
import logging

from .base import MLSkill, SkillResult, SkillCategory

logger = logging.getLogger(__name__)


class FeatureSelectionSkill(MLSkill):
    """
    World-class feature selection skill.

    Uses multiple techniques and voting to select best features.
    """

    name = "feature_selection"
    version = "2.0.0"
    description = "Multi-method feature selection with SHAP"
    category = SkillCategory.FEATURE_SELECTION

    required_inputs = ["X", "y"]
    optional_inputs = ["problem_type"]
    outputs = ["X_selected", "selected_features", "feature_scores"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._method_results = {}

    async def execute(self,
                      X: pd.DataFrame,
                      y: Optional[pd.Series] = None,
                      **context) -> SkillResult:
        """
        Execute feature selection.

        Args:
            X: Input features
            y: Target variable (required)
            **context: problem_type, etc.

        Returns:
            SkillResult with selected features
        """
        start_time = time.time()

        if y is None:
            return self._create_result(
                success=True, data=X,
                metrics={'selected': X.shape[1]}
            )

        if not self.validate_inputs(X, y):
            return self._create_error_result("Invalid inputs")

        problem_type = context.get('problem_type', 'classification')
        n_features = X.shape[1]
        feature_scores = defaultdict(float)
        self._method_results = {}

        try:
            # 1. Correlation filter
            X_filtered, to_drop_corr = self._correlation_filter(X)
            self._method_results['correlation_removed'] = len(to_drop_corr)

            # 2. Multi-model importance voting
            feature_scores, model_importances = self._multi_model_importance(
                X_filtered, y, problem_type, feature_scores
            )

            # 3. Null importance test
            feature_scores = self._null_importance_test(
                X_filtered, y, problem_type, feature_scores, model_importances
            )

            # 4. Stability selection
            feature_scores = self._stability_selection(
                X_filtered, y, problem_type, feature_scores
            )

            # 5. Boruta-like shadow test
            feature_scores = self._boruta_test(
                X_filtered, y, problem_type, feature_scores
            )

            # 6. Permutation importance
            feature_scores = self._permutation_importance(
                X_filtered, y, problem_type, feature_scores
            )

            # 7. SHAP importance
            feature_scores, shap_importance = self._shap_importance(
                X_filtered, y, problem_type, feature_scores
            )

            # 8. Successive Halving (progressive elimination)
            feature_scores = self._successive_halving(
                X_filtered, y, problem_type, feature_scores
            )

            # 9. Hyperband (multi-fidelity selection)
            feature_scores = self._hyperband_selection(
                X_filtered, y, problem_type, feature_scores
            )

            # 10. Recursive Feature Elimination with CV
            feature_scores = self._rfecv_selection(
                X_filtered, y, problem_type, feature_scores
            )

            # 11. Diverse RF Trees for feature importance (different configurations)
            feature_scores = self._diverse_rf_importance(
                X_filtered, y, problem_type, feature_scores
            )

            # 12. PCA-based feature scoring (variance explained)
            feature_scores = self._pca_importance(
                X_filtered, feature_scores
            )

            # 13. BOHB - Bayesian Optimized Hyperband for feature subsets
            feature_scores = self._bohb_selection(
                X_filtered, y, problem_type, feature_scores
            )

            # 14. PASHA - Progressive Adaptive Successive Halving (parallel)
            feature_scores = self._pasha_selection(
                X_filtered, y, problem_type, feature_scores
            )

            # Final selection using decile-based method
            selected = self._decile_selection(X_filtered, feature_scores)

            # CV validation
            selected = self._cv_validation(
                X_filtered, y, selected, feature_scores, problem_type
            )

            X_selected = X_filtered[selected]

        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return self._create_error_result(str(e))

        execution_time = time.time() - start_time
        final_scores = pd.Series(feature_scores).sort_values(ascending=False)

        return self._create_result(
            success=True,
            data=X_selected,
            metrics={
                'original': n_features,
                'after_corr_filter': len(X_filtered.columns),
                'selected': len(selected),
                'removed': n_features - len(selected),
            },
            metadata={
                'selected_features': selected,
                'feature_scores': final_scores.head(20).to_dict(),
                'methods_used': list(self._method_results.keys()),
                'shap_importance': shap_importance if 'shap_importance' in dir() else {},
            },
            execution_time=execution_time,
        )

    def _correlation_filter(self, X: pd.DataFrame, threshold: float = 0.98):
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper_tri.columns:
            correlated = upper_tri.index[upper_tri[col] > threshold].tolist()
            if correlated:
                for corr_col in correlated:
                    if X[col].var() >= X[corr_col].var():
                        to_drop.add(corr_col)
                    else:
                        to_drop.add(col)

        X_filtered = X.drop(columns=list(to_drop), errors='ignore')
        return X_filtered, to_drop

    def _multi_model_importance(self, X: pd.DataFrame, y: pd.Series,
                                 problem_type: str, feature_scores: Dict):
        """Multi-model importance voting."""
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        if problem_type == 'classification':
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
                model.fit(X, y)
                imp = pd.Series(model.feature_importances_, index=X.columns)
                imp_normalized = imp / (imp.sum() + 1e-10)
                model_importances[name] = imp_normalized

                for feat, score in imp_normalized.items():
                    feature_scores[feat] += score
            except Exception as e:
                logger.debug(f"Model {name} importance failed: {e}")

        self._method_results['multi_model'] = list(model_importances.keys())
        return feature_scores, model_importances

    def _null_importance_test(self, X: pd.DataFrame, y: pd.Series,
                               problem_type: str, feature_scores: Dict,
                               model_importances: Dict):
        """Null importance test - identify truly predictive features."""
        import lightgbm as lgb

        n_null_runs = 5
        null_importance_scores = defaultdict(list)

        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) \
            if problem_type == 'classification' else \
            lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

        for i in range(n_null_runs):
            y_shuffled = y.sample(frac=1, random_state=42 + i).reset_index(drop=True)
            try:
                lgb_model.fit(X, y_shuffled)
                null_imp = pd.Series(lgb_model.feature_importances_, index=X.columns)
                for feat in X.columns:
                    null_importance_scores[feat].append(null_imp[feat])
            except:
                pass

        null_passed = set()
        if model_importances.get('lgb') is not None:
            real_imp = model_importances['lgb']
            for feat in X.columns:
                null_vals = null_importance_scores.get(feat, [0])
                null_95 = np.percentile(null_vals, 95) if null_vals else 0
                real_val = real_imp.get(feat, 0)
                if real_val > null_95 * 1.1:
                    null_passed.add(feat)
                    feature_scores[feat] += 0.5

        self._method_results['null_test_passed'] = len(null_passed)
        return feature_scores

    def _stability_selection(self, X: pd.DataFrame, y: pd.Series,
                              problem_type: str, feature_scores: Dict):
        """Stability selection - consistent importance across seeds."""
        import lightgbm as lgb

        stability_counts = defaultdict(int)
        n_runs = 5
        top_pct = 0.3

        for seed in range(n_runs):
            try:
                lgb_temp = lgb.LGBMClassifier(n_estimators=50, random_state=seed, verbose=-1) \
                    if problem_type == 'classification' else \
                    lgb.LGBMRegressor(n_estimators=50, random_state=seed, verbose=-1)

                n_samples = len(X)
                idx = np.random.RandomState(seed).choice(n_samples, size=n_samples, replace=True)
                lgb_temp.fit(X.iloc[idx], y.iloc[idx])

                imp = pd.Series(lgb_temp.feature_importances_, index=X.columns)
                top_k = int(len(imp) * top_pct)
                top_features = imp.nlargest(top_k).index.tolist()

                for feat in top_features:
                    stability_counts[feat] += 1
            except:
                pass

        stable_features = {f for f, c in stability_counts.items() if c >= n_runs * 0.6}
        for feat in stable_features:
            feature_scores[feat] += 0.3

        self._method_results['stable_features'] = len(stable_features)
        return feature_scores

    def _boruta_test(self, X: pd.DataFrame, y: pd.Series,
                      problem_type: str, feature_scores: Dict):
        """Boruta-like shadow features test."""
        import lightgbm as lgb

        X_shadow = X.copy()
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col].values)
        X_shadow.columns = [f'shadow_{c}' for c in X_shadow.columns]

        X_with_shadow = pd.concat([X, X_shadow], axis=1)

        try:
            lgb_shadow = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) \
                if problem_type == 'classification' else \
                lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_shadow.fit(X_with_shadow, y)

            imp_all = pd.Series(lgb_shadow.feature_importances_, index=X_with_shadow.columns)
            shadow_max = imp_all[[c for c in imp_all.index if c.startswith('shadow_')]].max()

            boruta_passed = set()
            for feat in X.columns:
                if imp_all.get(feat, 0) > shadow_max:
                    boruta_passed.add(feat)
                    feature_scores[feat] += 0.4

            self._method_results['boruta_passed'] = len(boruta_passed)
        except Exception as e:
            logger.debug(f"Boruta test failed: {e}")
            self._method_results['boruta_passed'] = 0

        return feature_scores

    def _permutation_importance(self, X: pd.DataFrame, y: pd.Series,
                                 problem_type: str, feature_scores: Dict):
        """Permutation importance."""
        import lightgbm as lgb
        from sklearn.inspection import permutation_importance

        try:
            lgb_perm = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1) \
                if problem_type == 'classification' else \
                lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            lgb_perm.fit(X, y)

            perm_imp = permutation_importance(lgb_perm, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            perm_scores = pd.Series(perm_imp.importances_mean, index=X.columns)
            perm_scores_normalized = perm_scores / (perm_scores.sum() + 1e-10)

            for feat, score in perm_scores_normalized.items():
                if score > 0:
                    feature_scores[feat] += score * 0.5

            self._method_results['permutation_done'] = True
        except Exception as e:
            logger.debug(f"Permutation importance failed: {e}")
            self._method_results['permutation_done'] = False

        return feature_scores

    def _shap_importance(self, X: pd.DataFrame, y: pd.Series,
                          problem_type: str, feature_scores: Dict):
        """SHAP-based importance (World-Class)."""
        import lightgbm as lgb
        shap_importance = {}

        try:
            import shap

            lgb_shap = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) \
                if problem_type == 'classification' else \
                lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_shap.fit(X, y)

            n_shap_samples = min(500, len(X))
            X_sample = X.sample(n=n_shap_samples, random_state=42) if len(X) > 500 else X

            explainer = shap.TreeExplainer(lgb_shap)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)

            shap_imp = np.abs(shap_values).mean(axis=0)
            shap_scores = pd.Series(shap_imp, index=X.columns)
            shap_scores_normalized = shap_scores / (shap_scores.sum() + 1e-10)

            # SHAP is highly reliable - weight more
            for feat, score in shap_scores_normalized.items():
                feature_scores[feat] += score * 1.5

            shap_importance = shap_scores_normalized.to_dict()
            self._method_results['shap_done'] = True

        except ImportError:
            logger.debug("SHAP not installed")
            self._method_results['shap_done'] = False
        except Exception as e:
            logger.debug(f"SHAP importance failed: {e}")
            self._method_results['shap_done'] = False

        return feature_scores, shap_importance

    def _successive_halving(self, X: pd.DataFrame, y: pd.Series,
                             problem_type: str, feature_scores: Dict):
        """
        Successive Halving for feature selection.

        Progressively eliminates weakest features in rounds:
        Round 1: Train on all features with small budget, keep top 50%
        Round 2: Train on remaining with more budget, keep top 50%
        ...until convergence

        Features that survive multiple rounds get higher scores.
        """
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

        try:
            if problem_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'r2'

            current_features = list(X.columns)
            n_features = len(current_features)

            # Successive halving rounds
            round_num = 0
            budget_multiplier = 1

            while len(current_features) > max(10, n_features * 0.1):
                round_num += 1
                n_estimators = min(20 * budget_multiplier, 100)

                # Train model with current budget
                model = lgb.LGBMClassifier(n_estimators=n_estimators, random_state=42, verbose=-1) \
                    if problem_type == 'classification' else \
                    lgb.LGBMRegressor(n_estimators=n_estimators, random_state=42, verbose=-1)

                model.fit(X[current_features], y)
                imp = pd.Series(model.feature_importances_, index=current_features)

                # Keep top 50% (or at least 10)
                n_keep = max(10, len(current_features) // 2)
                survivors = imp.nlargest(n_keep).index.tolist()

                # Features that survive get bonus (more rounds survived = higher bonus)
                for feat in survivors:
                    feature_scores[feat] += 0.1 * round_num

                current_features = survivors
                budget_multiplier *= 2

                # Stop after 4 rounds max
                if round_num >= 4:
                    break

            self._method_results['successive_halving_rounds'] = round_num
            self._method_results['successive_halving_survivors'] = len(current_features)

        except Exception as e:
            logger.debug(f"Successive halving failed: {e}")
            self._method_results['successive_halving_done'] = False

        return feature_scores

    def _hyperband_selection(self, X: pd.DataFrame, y: pd.Series,
                              problem_type: str, feature_scores: Dict):
        """
        Hyperband-style feature selection.

        Runs multiple brackets with different starting budgets:
        - Bracket 1: Many features, small budget, aggressive elimination
        - Bracket 2: Fewer features, medium budget, moderate elimination
        - Bracket 3: Few features, large budget, gentle elimination

        Combines results across brackets for robust selection.
        """
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

        try:
            if problem_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'r2'

            n_features = X.shape[1]
            bracket_survivors = {}

            # Define brackets (starting features ratio, elimination rate)
            brackets = [
                (1.0, 0.5),   # Start with all, eliminate 50% per round
                (0.7, 0.6),   # Start with 70%, eliminate 40% per round
                (0.5, 0.7),   # Start with 50%, eliminate 30% per round
            ]

            for bracket_idx, (start_ratio, keep_ratio) in enumerate(brackets):
                # Get initial feature importance to select starting features
                model = lgb.LGBMClassifier(n_estimators=50, random_state=42+bracket_idx, verbose=-1) \
                    if problem_type == 'classification' else \
                    lgb.LGBMRegressor(n_estimators=50, random_state=42+bracket_idx, verbose=-1)
                model.fit(X, y)
                imp = pd.Series(model.feature_importances_, index=X.columns)

                # Select starting features for this bracket
                n_start = max(10, int(n_features * start_ratio))
                current_features = imp.nlargest(n_start).index.tolist()

                # Run successive halving within bracket
                round_num = 0
                while len(current_features) > max(5, n_start * 0.2):
                    round_num += 1
                    budget = 30 + round_num * 20  # Increasing budget

                    model = lgb.LGBMClassifier(n_estimators=budget, random_state=42, verbose=-1) \
                        if problem_type == 'classification' else \
                        lgb.LGBMRegressor(n_estimators=budget, random_state=42, verbose=-1)
                    model.fit(X[current_features], y)
                    imp = pd.Series(model.feature_importances_, index=current_features)

                    n_keep = max(5, int(len(current_features) * keep_ratio))
                    current_features = imp.nlargest(n_keep).index.tolist()

                    if round_num >= 3:
                        break

                bracket_survivors[f'bracket_{bracket_idx}'] = current_features

                # Score features by bracket survival
                for feat in current_features:
                    feature_scores[feat] += 0.2 * (bracket_idx + 1)

            # Features surviving multiple brackets get extra bonus
            all_survivors = set()
            for survivors in bracket_survivors.values():
                all_survivors.update(survivors)

            for feat in all_survivors:
                n_brackets_survived = sum(1 for s in bracket_survivors.values() if feat in s)
                if n_brackets_survived >= 2:
                    feature_scores[feat] += 0.3 * n_brackets_survived

            self._method_results['hyperband_done'] = True
            self._method_results['hyperband_multi_bracket'] = len([f for f in all_survivors
                if sum(1 for s in bracket_survivors.values() if f in s) >= 2])

        except Exception as e:
            logger.debug(f"Hyperband selection failed: {e}")
            self._method_results['hyperband_done'] = False

        return feature_scores

    def _rfecv_selection(self, X: pd.DataFrame, y: pd.Series,
                          problem_type: str, feature_scores: Dict):
        """
        Recursive Feature Elimination with Cross-Validation.

        Backward elimination: remove weakest features iteratively
        while monitoring CV score. Stop when score starts dropping.
        """
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

        try:
            if problem_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'r2'

            current_features = list(X.columns)
            best_score = cross_val_score(model, X, y, cv=cv, scoring=scoring).mean()
            best_features = current_features.copy()

            # Iteratively remove weakest features
            n_to_remove = max(1, len(current_features) // 10)  # Remove 10% at a time
            patience = 2
            no_improve_count = 0

            while len(current_features) > max(10, len(X.columns) * 0.1):
                # Train and get importance
                model.fit(X[current_features], y)
                imp = pd.Series(model.feature_importances_, index=current_features)

                # Remove weakest features
                weakest = imp.nsmallest(n_to_remove).index.tolist()
                new_features = [f for f in current_features if f not in weakest]

                # Check score
                new_score = cross_val_score(model, X[new_features], y, cv=cv, scoring=scoring).mean()

                if new_score >= best_score - 0.002:  # Allow 0.2% degradation
                    if new_score > best_score:
                        best_score = new_score
                        best_features = new_features.copy()
                        no_improve_count = 0
                    current_features = new_features
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        break
                    current_features = new_features

            # Score features that survived RFECV
            for feat in best_features:
                feature_scores[feat] += 0.5

            self._method_results['rfecv_done'] = True
            self._method_results['rfecv_selected'] = len(best_features)
            self._method_results['rfecv_best_score'] = best_score

        except Exception as e:
            logger.debug(f"RFECV selection failed: {e}")
            self._method_results['rfecv_done'] = False

        return feature_scores

    def _diverse_rf_importance(self, X: pd.DataFrame, y: pd.Series,
                                problem_type: str, feature_scores: Dict):
        """
        Use diverse Random Forest configurations for feature importance.

        Different RF configs capture different feature interactions:
        - Shallow trees: linear-ish relationships
        - Deep trees: complex interactions
        - Different max_features: different feature subsets
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        try:
            # Different RF configurations
            configs = [
                {'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100},
                {'max_depth': 10, 'max_features': 'log2', 'n_estimators': 100},
                {'max_depth': None, 'max_features': 0.5, 'n_estimators': 100},
                {'max_depth': 8, 'max_features': None, 'n_estimators': 100},  # All features
            ]

            config_importances = []

            for i, config in enumerate(configs):
                if problem_type == 'classification':
                    # Alternate between RF and ExtraTrees
                    if i % 2 == 0:
                        model = RandomForestClassifier(**config, random_state=42+i, n_jobs=-1)
                    else:
                        model = ExtraTreesClassifier(**config, random_state=42+i, n_jobs=-1)
                else:
                    if i % 2 == 0:
                        model = RandomForestRegressor(**config, random_state=42+i, n_jobs=-1)
                    else:
                        model = ExtraTreesRegressor(**config, random_state=42+i, n_jobs=-1)

                model.fit(X, y)
                imp = pd.Series(model.feature_importances_, index=X.columns)
                imp_normalized = imp / (imp.sum() + 1e-10)
                config_importances.append(imp_normalized)

            # Average importance across configurations
            avg_importance = pd.concat(config_importances, axis=1).mean(axis=1)

            # Features consistently important across configs get bonus
            for feat in X.columns:
                consistency = sum(1 for imp in config_importances
                                  if imp[feat] > imp.median())
                if consistency >= 3:  # Important in 3+ configs
                    feature_scores[feat] += 0.4
                elif consistency >= 2:
                    feature_scores[feat] += 0.2

            self._method_results['diverse_rf_done'] = True
            self._method_results['diverse_rf_configs'] = len(configs)

        except Exception as e:
            logger.debug(f"Diverse RF importance failed: {e}")
            self._method_results['diverse_rf_done'] = False

        return feature_scores

    def _pca_importance(self, X: pd.DataFrame, feature_scores: Dict):
        """
        PCA-based feature scoring.

        Features that contribute most to top principal components
        are likely important for prediction.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        try:
            # Standardize for PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit PCA keeping components that explain 95% variance
            pca = PCA(n_components=0.95, random_state=42)
            pca.fit(X_scaled)

            # Get absolute loadings for each feature
            # loadings = components (n_components x n_features)
            loadings = np.abs(pca.components_)

            # Weight by variance explained
            weighted_loadings = loadings * pca.explained_variance_ratio_[:, np.newaxis]

            # Sum across components to get feature importance
            pca_importance = weighted_loadings.sum(axis=0)
            pca_scores = pd.Series(pca_importance, index=X.columns)
            pca_scores_normalized = pca_scores / (pca_scores.sum() + 1e-10)

            # Add to feature scores (PCA captures variance, not necessarily prediction)
            for feat, score in pca_scores_normalized.items():
                feature_scores[feat] += score * 0.3  # Lower weight than prediction-based

            self._method_results['pca_done'] = True
            self._method_results['pca_n_components'] = pca.n_components_
            self._method_results['pca_variance_explained'] = pca.explained_variance_ratio_.sum()

        except Exception as e:
            logger.debug(f"PCA importance failed: {e}")
            self._method_results['pca_done'] = False

        return feature_scores

    def _decile_selection(self, X: pd.DataFrame, feature_scores: Dict) -> List[str]:
        """Decile-based feature selection."""
        final_scores = pd.Series(feature_scores).sort_values(ascending=False)

        if len(final_scores) > 10:
            final_scores_df = pd.DataFrame({
                'feature': final_scores.index,
                'score': final_scores.values
            })
            final_scores_df['decile'] = pd.qcut(
                final_scores_df['score'].rank(method='first'),
                q=10, labels=range(10, 0, -1)
            ).astype(int)

            selected = []

            # Top 3 deciles (top 30%) - always keep
            top_deciles = final_scores_df[final_scores_df['decile'] >= 8]['feature'].tolist()
            selected.extend(top_deciles)

            # Decile 7 - keep above median
            decile_7 = final_scores_df[final_scores_df['decile'] == 7]
            if len(decile_7) > 0:
                d7_median = decile_7['score'].median()
                d7_selected = decile_7[decile_7['score'] >= d7_median]['feature'].tolist()
                selected.extend(d7_selected)

            # Decile 6 - keep if score > 1.0
            decile_6 = final_scores_df[final_scores_df['decile'] == 6]
            d6_selected = decile_6[decile_6['score'] >= 1.0]['feature'].tolist()
            selected.extend(d6_selected)

            # Decile 5 - keep if score > 1.5
            decile_5 = final_scores_df[final_scores_df['decile'] == 5]
            d5_selected = decile_5[decile_5['score'] >= 1.5]['feature'].tolist()
            selected.extend(d5_selected)

            # Lower deciles - only exceptional (score > 2.0)
            lower_deciles = final_scores_df[final_scores_df['decile'] <= 4]
            exceptional = lower_deciles[lower_deciles['score'] >= 2.0]['feature'].tolist()
            selected.extend(exceptional)

            selected = list(dict.fromkeys(selected))
            self._method_results['decile_distribution'] = final_scores_df.groupby('decile').size().to_dict()
        else:
            selected = final_scores.index.tolist()

        # Ensure minimum features
        min_features = max(15, int(len(X.columns) * 0.15))
        if len(selected) < min_features:
            for feat in final_scores.index:
                if feat not in selected:
                    selected.append(feat)
                if len(selected) >= min_features:
                    break

        return selected

    def _cv_validation(self, X: pd.DataFrame, y: pd.Series,
                        selected: List[str], feature_scores: Dict,
                        problem_type: str) -> List[str]:
        """CV validation to ensure selection doesn't hurt performance."""
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

        try:
            if problem_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                val_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                val_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'r2'

            score_selected = cross_val_score(val_model, X[selected], y, cv=cv, scoring=scoring).mean()
            score_all = cross_val_score(val_model, X, y, cv=cv, scoring=scoring).mean()

            # If selected is worse, add more features
            if score_selected < score_all - 0.01:
                final_scores = pd.Series(feature_scores).sort_values(ascending=False)
                remaining = [f for f in final_scores.index if f not in selected]
                for feat in remaining:
                    selected.append(feat)
                    new_score = cross_val_score(val_model, X[selected], y, cv=cv, scoring=scoring).mean()
                    if new_score >= score_all - 0.005:
                        break
                    if len(selected) >= len(X.columns) * 0.6:
                        break

            self._method_results['cv_selected_score'] = score_selected
            self._method_results['cv_all_score'] = score_all
        except Exception as e:
            logger.debug(f"CV validation failed: {e}")

        return selected

    def _bohb_selection(self, X: pd.DataFrame, y: pd.Series,
                        problem_type: str, feature_scores: Dict):
        """
        BOHB - Bayesian Optimized Hyperband for feature subset search.

        Combines:
        - TPE (Tree Parzen Estimator) for smart feature subset sampling
        - Hyperband scheduling for efficient budget allocation

        Treats feature selection as hyperparameter optimization where
        each feature is a binary hyperparameter (include/exclude).
        """
        try:
            import ConfigSpace as CS
            import ConfigSpace.hyperparameters as CSH
            from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
            import lightgbm as lgb

            if problem_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'accuracy'
                base_model = lgb.LGBMClassifier
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'r2'
                base_model = lgb.LGBMRegressor

            features = list(X.columns)
            n_features = len(features)

            # For efficiency, work with top features from current scores
            sorted_features = sorted(features, key=lambda f: feature_scores.get(f, 0), reverse=True)
            candidate_features = sorted_features[:min(30, n_features)]  # Top 30 for BOHB

            # Build ConfigSpace - each feature is a binary HP
            cs = CS.ConfigurationSpace(seed=42)
            for feat in candidate_features:
                cs.add_hyperparameter(CSH.CategoricalHyperparameter(
                    name=feat, choices=[0, 1], default_value=1
                ))

            # TPE-like sampling with Hyperband brackets
            def evaluate_config(config, budget):
                """Evaluate a feature subset configuration."""
                selected_feats = [f for f in candidate_features if config.get(f, 0) == 1]
                if len(selected_feats) < 3:
                    return -1.0  # Too few features

                n_estimators = int(budget * 20)  # budget 1-5 -> 20-100 trees
                model = base_model(n_estimators=n_estimators, random_state=42, verbose=-1)

                try:
                    score = cross_val_score(model, X[selected_feats], y, cv=cv, scoring=scoring).mean()
                    return score
                except:
                    return -1.0

            # BOHB brackets
            brackets = [
                {'n_configs': 27, 'min_budget': 1, 'max_budget': 3},  # Aggressive
                {'n_configs': 9, 'min_budget': 2, 'max_budget': 4},   # Moderate
                {'n_configs': 6, 'min_budget': 3, 'max_budget': 5},   # Conservative
            ]

            best_configs = []

            for bracket in brackets:
                n_configs = bracket['n_configs']
                min_budget = bracket['min_budget']
                max_budget = bracket['max_budget']

                # Sample configurations (TPE-inspired: exploit good regions)
                configs = []
                for _ in range(n_configs):
                    if best_configs and np.random.random() < 0.3:
                        # Exploit: mutate best config
                        base_config = best_configs[-1].copy()
                        mutate_feat = np.random.choice(candidate_features)
                        base_config[mutate_feat] = 1 - base_config.get(mutate_feat, 0)
                        configs.append(base_config)
                    else:
                        # Explore: random sample
                        config = cs.sample_configuration()
                        configs.append(dict(config))

                # Successive halving within bracket
                budget = min_budget
                while budget <= max_budget and len(configs) > 1:
                    # Evaluate all configs at current budget
                    scores = [(c, evaluate_config(c, budget)) for c in configs]
                    scores.sort(key=lambda x: x[1], reverse=True)

                    # Keep top half
                    n_keep = max(1, len(configs) // 2)
                    configs = [c for c, s in scores[:n_keep]]

                    if scores[0][1] > 0:
                        best_configs.append(scores[0][0])

                    budget += 1

            # Score features by how often they appear in best configs
            feature_inclusion = defaultdict(int)
            for config in best_configs:
                for feat in candidate_features:
                    if config.get(feat, 0) == 1:
                        feature_inclusion[feat] += 1

            # Normalize and add to scores
            if best_configs:
                max_inclusion = max(feature_inclusion.values()) if feature_inclusion else 1
                for feat, count in feature_inclusion.items():
                    # BOHB-selected features get bonus
                    feature_scores[feat] += 0.5 * (count / max_inclusion)

            self._method_results['bohb_done'] = True
            self._method_results['bohb_configs_evaluated'] = sum(b['n_configs'] for b in brackets)
            self._method_results['bohb_best_configs'] = len(best_configs)

        except ImportError as e:
            logger.debug(f"BOHB requires ConfigSpace: {e}")
            self._method_results['bohb_done'] = False
        except Exception as e:
            logger.debug(f"BOHB selection failed: {e}")
            self._method_results['bohb_done'] = False

        return feature_scores

    def _pasha_selection(self, X: pd.DataFrame, y: pd.Series,
                         problem_type: str, feature_scores: Dict):
        """
        PASHA - Progressive Adaptive Successive Halving Algorithm.

        Based on https://github.com/ondrejbohdal/pasha

        Key innovations:
        - Progressive: Start with small budgets, increase adaptively
        - Adaptive: Allocate more budget to promising feature subsets
        - Parallel-friendly: Can run evaluations concurrently

        For feature selection:
        - Configurations = feature subsets (random or seeded)
        - Budget = number of estimators / CV folds
        - Rung = evaluation level (higher rung = more budget)
        """
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
            import lightgbm as lgb
            from concurrent.futures import ThreadPoolExecutor
            import threading

            if problem_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'accuracy'
                base_model = lgb.LGBMClassifier
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'r2'
                base_model = lgb.LGBMRegressor

            features = list(X.columns)
            n_features = len(features)

            # PASHA parameters
            eta = 3  # Reduction factor (keep top 1/eta at each rung)
            min_budget = 1  # Minimum budget (rung 0)
            max_budget = 5  # Maximum budget (final rung)
            n_workers = 4   # Parallel workers

            # Generate initial configurations (feature subsets)
            # Use feature scores to bias towards good features
            sorted_features = sorted(features, key=lambda f: feature_scores.get(f, 0), reverse=True)

            def generate_config(seed):
                """Generate a feature subset configuration."""
                np.random.seed(seed)
                # Bias towards top features
                probs = np.array([0.7 if f in sorted_features[:n_features//2] else 0.3
                                  for f in features])
                mask = np.random.random(n_features) < probs
                # Ensure at least 3 features
                if mask.sum() < 3:
                    top_3_idx = [features.index(f) for f in sorted_features[:3]]
                    mask[top_3_idx] = True
                return {f: int(m) for f, m in zip(features, mask)}

            # Initial population
            n_configs = 27  # Start with 27 configs (like Hyperband)
            configs = [generate_config(i) for i in range(n_configs)]

            # PASHA rungs
            rungs = []
            rung = 0
            budget = min_budget

            results_lock = threading.Lock()
            rung_results = {}

            def evaluate_at_rung(config_idx, config, budget):
                """Evaluate config at given budget rung."""
                selected_feats = [f for f, v in config.items() if v == 1]
                if len(selected_feats) < 3:
                    return config_idx, -1.0

                n_estimators = int(budget * 25)  # Scale budget
                model = base_model(n_estimators=n_estimators, random_state=42, verbose=-1)

                try:
                    # Progressive CV: more folds at higher rungs
                    n_folds = min(3 + budget - 1, 5)
                    if problem_type == 'classification':
                        cv_temp = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                    else:
                        cv_temp = KFold(n_splits=n_folds, shuffle=True, random_state=42)

                    score = cross_val_score(model, X[selected_feats], y, cv=cv_temp, scoring=scoring).mean()
                    return config_idx, score
                except:
                    return config_idx, -1.0

            # PASHA main loop
            while budget <= max_budget and len(configs) > 1:
                # Parallel evaluation at current rung
                rung_scores = {}

                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = [
                        executor.submit(evaluate_at_rung, i, c, budget)
                        for i, c in enumerate(configs)
                    ]
                    for future in futures:
                        idx, score = future.result()
                        rung_scores[idx] = score

                # Sort by score
                sorted_configs = sorted(
                    [(i, configs[i], rung_scores[i]) for i in range(len(configs))],
                    key=lambda x: x[2],
                    reverse=True
                )

                # Adaptive promotion: keep top 1/eta
                n_promote = max(1, len(configs) // eta)
                configs = [c for _, c, s in sorted_configs[:n_promote] if s > 0]

                # Record rung results
                rungs.append({
                    'budget': budget,
                    'configs': len(sorted_configs),
                    'promoted': len(configs),
                    'best_score': sorted_configs[0][2] if sorted_configs else 0
                })

                # Progressive budget increase
                budget += 1
                rung += 1

            # Score features by survival across rungs
            if configs:
                # Final surviving configs
                for config in configs:
                    for feat, included in config.items():
                        if included:
                            feature_scores[feat] += 0.4  # PASHA survivor bonus

                # Extra bonus for features in ALL surviving configs
                if len(configs) > 1:
                    common_features = set(features)
                    for config in configs:
                        config_features = {f for f, v in config.items() if v == 1}
                        common_features &= config_features

                    for feat in common_features:
                        feature_scores[feat] += 0.3  # Consensus bonus

            self._method_results['pasha_done'] = True
            self._method_results['pasha_rungs'] = len(rungs)
            self._method_results['pasha_final_configs'] = len(configs)
            if rungs:
                self._method_results['pasha_best_score'] = rungs[-1].get('best_score', 0)

        except Exception as e:
            logger.debug(f"PASHA selection failed: {e}")
            self._method_results['pasha_done'] = False

        return feature_scores
