"""SkillOrchestrator mixin — builtin feature selection stage."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .skill_types import ProblemType, SkillResult

logger = logging.getLogger(__name__)


class FeatureSelectionMixin:
    """Mixin providing _builtin_feature_selection for SkillOrchestrator."""

    async def _builtin_feature_selection(
        self, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType
    ) -> SkillResult:
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
        from collections import defaultdict

        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        if y is None:
            return SkillResult(
                skill_name="builtin_feature_selector",
                category=SkillCategory.FEATURE_SELECTION,
                success=True,
                data=X,
                metrics={"selected": X.shape[1]},
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

        X_filtered = X.drop(columns=list(to_drop_corr), errors="ignore")
        method_results["correlation_removed"] = len(to_drop_corr)

        # ================================================================
        # 2. MULTI-MODEL IMPORTANCE VOTING
        # ================================================================
        if problem_type == ProblemType.CLASSIFICATION:
            models = {
                "lgb": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
                "xgb": xgb.XGBClassifier(
                    n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0
                ),
                "rf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            }
        else:
            models = {
                "lgb": lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                "xgb": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                "rf": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
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

        method_results["multi_model"] = list(model_importances.keys())

        # ================================================================
        # 3. NULL IMPORTANCE TEST - Identify truly predictive features
        # ================================================================
        null_importance_scores = defaultdict(list)
        n_null_runs = 5

        lgb_model = models["lgb"]
        for i in range(n_null_runs):
            # Shuffle target
            y_shuffled = y.sample(frac=1, random_state=42 + i).reset_index(drop=True)
            try:
                lgb_model.fit(X_filtered, y_shuffled)
                null_imp = pd.Series(lgb_model.feature_importances_, index=X_filtered.columns)
                for feat in X_filtered.columns:
                    null_importance_scores[feat].append(null_imp[feat])
            except Exception:
                pass

        # Features where real importance > 95th percentile of null importance
        null_passed = set()
        if model_importances.get("lgb") is not None:
            real_imp = model_importances["lgb"]
            for feat in X_filtered.columns:
                null_vals = null_importance_scores.get(feat, [0])
                null_95 = np.percentile(null_vals, 95) if null_vals else 0
                real_val = real_imp.get(feat, 0)
                if real_val > null_95 * 1.1:  # 10% margin
                    null_passed.add(feat)
                    feature_scores[feat] += 0.5  # Bonus for passing null test

        method_results["null_test_passed"] = len(null_passed)

        # ================================================================
        # 4. STABILITY SELECTION - Consistent importance across seeds
        # ================================================================
        stability_counts = defaultdict(int)
        n_stability_runs = 5
        top_pct = 0.3  # Top 30% features

        for seed in range(n_stability_runs):
            try:
                lgb_temp = (
                    lgb.LGBMClassifier(n_estimators=50, random_state=seed, verbose=-1)
                    if problem_type == ProblemType.CLASSIFICATION
                    else lgb.LGBMRegressor(n_estimators=50, random_state=seed, verbose=-1)
                )

                # Bootstrap sample
                n_samples = len(X_filtered)
                idx = np.random.RandomState(seed).choice(n_samples, size=n_samples, replace=True)
                lgb_temp.fit(X_filtered.iloc[idx], y.iloc[idx])

                imp = pd.Series(lgb_temp.feature_importances_, index=X_filtered.columns)
                top_k = int(len(imp) * top_pct)
                top_features = imp.nlargest(top_k).index.tolist()

                for feat in top_features:
                    stability_counts[feat] += 1
            except Exception:
                pass

        # Features selected in majority of runs
        stable_features = {f for f, c in stability_counts.items() if c >= n_stability_runs * 0.6}
        for feat in stable_features:
            feature_scores[feat] += 0.3  # Bonus for stability

        method_results["stable_features"] = len(stable_features)

        # ================================================================
        # 5. BORUTA-LIKE SHADOW FEATURES TEST
        # ================================================================
        # Create shadow features (shuffled copies)
        X_shadow = X_filtered.copy()
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col].values)
        X_shadow.columns = [f"shadow_{c}" for c in X_shadow.columns]

        X_with_shadow = pd.concat([X_filtered, X_shadow], axis=1)

        try:
            lgb_shadow = (
                lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                if problem_type == ProblemType.CLASSIFICATION
                else lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            )
            lgb_shadow.fit(X_with_shadow, y)

            imp_all = pd.Series(lgb_shadow.feature_importances_, index=X_with_shadow.columns)
            shadow_max = imp_all[[c for c in imp_all.index if c.startswith("shadow_")]].max()

            # Features beating the best shadow feature
            boruta_passed = set()
            for feat in X_filtered.columns:
                if imp_all.get(feat, 0) > shadow_max:
                    boruta_passed.add(feat)
                    feature_scores[feat] += 0.4  # Bonus for beating shadows

            method_results["boruta_passed"] = len(boruta_passed)
        except Exception as e:
            logger.debug(f"Boruta test failed: {e}")
            method_results["boruta_passed"] = 0

        # ================================================================
        # 6. PERMUTATION IMPORTANCE (on a quick model)
        # ================================================================
        try:
            if problem_type == ProblemType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = "accuracy"
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = "r2"

            lgb_perm = (
                lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
                if problem_type == ProblemType.CLASSIFICATION
                else lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            )
            lgb_perm.fit(X_filtered, y)

            perm_imp = permutation_importance(
                lgb_perm, X_filtered, y, n_repeats=5, random_state=42, n_jobs=-1
            )
            perm_scores = pd.Series(perm_imp.importances_mean, index=X_filtered.columns)
            perm_scores_normalized = perm_scores / (perm_scores.sum() + 1e-10)

            for feat, score in perm_scores_normalized.items():
                if score > 0:  # Only positive permutation importance
                    feature_scores[feat] += score * 0.5

            method_results["permutation_done"] = True
        except Exception as e:
            logger.debug(f"Permutation importance failed: {e}")
            method_results["permutation_done"] = False

        # ================================================================
        # 7. SHAP-BASED IMPORTANCE (World-Class 10/10 Feature)
        # ================================================================
        try:
            import shap

            # Use TreeExplainer for fast SHAP on tree models
            lgb_shap = (
                lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                if problem_type == ProblemType.CLASSIFICATION
                else lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            )
            lgb_shap.fit(X_filtered, y)

            # Sample for speed if dataset is large
            n_shap_samples = min(500, len(X_filtered))
            X_sample = (
                X_filtered.sample(n=n_shap_samples, random_state=42)
                if len(X_filtered) > 500
                else X_filtered
            )

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

            method_results["shap_done"] = True

            # Store SHAP importance for LLM feedback loop
            method_results["shap_importance"] = shap_scores_normalized.to_dict()

        except ImportError:
            logger.debug("SHAP not installed - skipping SHAP importance")
            method_results["shap_done"] = False
        except Exception as e:
            logger.debug(f"SHAP importance failed: {e}")
            method_results["shap_done"] = False

        # ================================================================
        # FINAL SELECTION - Quantile/Decile-based selection
        # ================================================================
        final_scores = pd.Series(feature_scores).sort_values(ascending=False)

        # Calculate deciles for adaptive selection
        if len(final_scores) > 10:
            # Assign decile ranks (1=top 10%, 10=bottom 10%)
            final_scores_df = pd.DataFrame(
                {"feature": final_scores.index, "score": final_scores.values}
            )
            final_scores_df["decile"] = pd.qcut(
                final_scores_df["score"].rank(method="first"),
                q=10,
                labels=range(10, 0, -1),  # 10=top, 1=bottom
            ).astype(int)

            # SELECTION STRATEGY (World-class - keep quality features):
            # - Decile 10-8 (top 30%): Always keep - best features
            # - Decile 7 (top 40%): Keep if score > median of decile
            # - Decile 6 (top 50%): Keep if passed multiple tests (score > 1.0)
            # - Decile 5 (top 60%): Keep if strong multi-test (score > 1.5)
            # - Below: Only keep if exceptional (score > 2.0)

            selected = []

            # Top 3 deciles (top 30%) - always keep
            top_deciles = final_scores_df[final_scores_df["decile"] >= 8]["feature"].tolist()
            selected.extend(top_deciles)

            # Decile 7 (top 40%) - keep above median
            decile_7 = final_scores_df[final_scores_df["decile"] == 7]
            if len(decile_7) > 0:
                d7_median = decile_7["score"].median()
                d7_selected = decile_7[decile_7["score"] >= d7_median]["feature"].tolist()
                selected.extend(d7_selected)

            # Decile 6 (top 50%) - keep if multi-test passed (score > 1.0)
            decile_6 = final_scores_df[final_scores_df["decile"] == 6]
            d6_selected = decile_6[decile_6["score"] >= 1.0]["feature"].tolist()
            selected.extend(d6_selected)

            # Decile 5 (top 60%) - keep if strong score (score > 1.5)
            decile_5 = final_scores_df[final_scores_df["decile"] == 5]
            d5_selected = decile_5[decile_5["score"] >= 1.5]["feature"].tolist()
            selected.extend(d5_selected)

            # Deciles 4 and below - only exceptional features (score > 2.0)
            lower_deciles = final_scores_df[final_scores_df["decile"] <= 4]
            exceptional = lower_deciles[lower_deciles["score"] >= 2.0]["feature"].tolist()
            selected.extend(exceptional)

            # Remove duplicates while preserving order
            selected = list(dict.fromkeys(selected))

            method_results["decile_distribution"] = (
                final_scores_df.groupby("decile").size().to_dict()
            )
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
                scoring = "accuracy"
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                val_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
                scoring = "r2"

            # Score with selected features
            score_selected = cross_val_score(
                val_model, X_filtered[selected], y, cv=cv, scoring=scoring
            ).mean()

            # Score with all features (after correlation filter)
            score_all = cross_val_score(val_model, X_filtered, y, cv=cv, scoring=scoring).mean()

            # If selected is significantly worse (>1%), add more features
            if score_selected < score_all - 0.01:
                # Add features until performance matches
                remaining = [f for f in final_scores.index if f not in selected]
                for feat in remaining:
                    selected.append(feat)
                    new_score = cross_val_score(
                        val_model, X_filtered[selected], y, cv=cv, scoring=scoring
                    ).mean()
                    if new_score >= score_all - 0.005:  # Within 0.5% of full set
                        break
                    if len(selected) >= len(X_filtered.columns) * 0.6:  # Cap at 60%
                        break

            method_results["cv_selected_score"] = score_selected
            method_results["cv_all_score"] = score_all
        except Exception as e:
            logger.debug(f"CV validation failed: {e}")

        X_selected = X_filtered[selected]

        return SkillResult(
            skill_name="builtin_feature_selector",
            category=SkillCategory.FEATURE_SELECTION,
            success=True,
            data=X_selected,
            metrics={
                "original": n_features,
                "after_corr_filter": len(X_filtered.columns),
                "selected": len(selected),
                "removed": n_features - len(selected),
                "null_passed": method_results.get("null_test_passed", 0),
                "boruta_passed": method_results.get("boruta_passed", 0),
                "stable": method_results.get("stable_features", 0),
            },
            metadata={
                "selected_features": selected,
                "feature_scores": final_scores.head(20).to_dict(),
                "decile_dist": method_results.get("decile_distribution", {}),
                "methods_used": [
                    "correlation",
                    "multi_model",
                    "null_importance",
                    "stability",
                    "boruta",
                    "permutation",
                ],
            },
        )
