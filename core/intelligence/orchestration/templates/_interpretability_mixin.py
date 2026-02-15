"""
Interpretability mixin for ProfessionalMLReport.

Provides feature interaction detection, SHAP-based and perturbation-based
local interpretability analysis, multiple counterfactual generation strategies,
and LIME-style visualization helpers. Extracted from ml_report_generator.py
to keep the main class focused and maintainable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InterpretabilityMixin:
    """Mixin that adds interpretability and feature-interaction analysis
    capabilities to ProfessionalMLReport.

    This class is not intended to be instantiated on its own. It should be
    mixed into ProfessionalMLReport (or a subclass) which provides:
        - self._content          (list)
        - self._store_section_data()
        - self._maybe_add_narrative()
        - self.theme             (dict)
        - self.figures_dir       (Path)
    """

    _TREE_MODEL_TYPES = (
        "RandomForest",
        "GradientBoosting",
        "XGB",
        "LGBM",
        "LightGBM",
        "ExtraTrees",
        "DecisionTree",
        "CatBoost",
        "HistGradientBoosting",
    )

    @staticmethod
    def _is_tree_model(model: Any) -> bool:
        """Check if a model is tree-based via isinstance or string matching.

        Tries isinstance checks for sklearn, xgboost, lightgbm, catboost.
        Falls back to class name string matching if imports fail.
        """
        try:
            from sklearn.ensemble import (
                ExtraTreesClassifier,
                ExtraTreesRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
                RandomForestClassifier,
                RandomForestRegressor,
            )
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

            sklearn_trees = (
                RandomForestClassifier,
                RandomForestRegressor,
                GradientBoostingClassifier,
                GradientBoostingRegressor,
                ExtraTreesClassifier,
                ExtraTreesRegressor,
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
                DecisionTreeClassifier,
                DecisionTreeRegressor,
            )
            if isinstance(model, sklearn_trees):
                return True
        except ImportError:
            pass

        try:
            import xgboost

            if isinstance(model, (xgboost.XGBClassifier, xgboost.XGBRegressor)):
                return True
        except (ImportError, AttributeError):
            pass

        try:
            import lightgbm

            if isinstance(model, (lightgbm.LGBMClassifier, lightgbm.LGBMRegressor)):
                return True
        except (ImportError, AttributeError):
            pass

        try:
            import catboost

            if isinstance(model, (catboost.CatBoostClassifier, catboost.CatBoostRegressor)):
                return True
        except (ImportError, AttributeError):
            pass

        # Fallback: string matching on class name
        model_name = type(model).__name__
        return any(t.lower() in model_name.lower() for t in InterpretabilityMixin._TREE_MODEL_TYPES)

    def add_feature_interactions(
        self,
        shap_values: Any,
        feature_names: List[str],
        X_sample: Any = None,
        model: Any = None,
        top_n: int = 3,
    ) -> Any:
        """
        Add feature interaction detection:
        - SHAP interaction values for tree models
        - Pairwise SHAP correlation fallback
        - 2D scatter colored by interaction value
        """
        try:
            import shap

            # Extract SHAP values
            if hasattr(shap_values, "values"):
                values = shap_values.values
            else:
                values = shap_values

            if len(values.shape) == 3:
                values = values[:, :, 1]  # Binary classification

            # Try SHAP interaction values for tree models
            interaction_data = None
            try:
                if model is not None and hasattr(model, "predict_proba"):
                    explainer = shap.TreeExplainer(model)
                    interaction_values = explainer.shap_interaction_values(X_sample)
                    if isinstance(interaction_values, list):
                        interaction_values = interaction_values[1]  # Binary
                    interaction_data = interaction_values
            except Exception as e:
                self._record_internal_warning(
                    "SHAPInteraction", "Failed to compute SHAP interaction values for tree model", e
                )
                pass

            # Fallback: pairwise SHAP correlation
            n_features = min(len(feature_names), values.shape[1])
            shap_corr = np.corrcoef(values.T[:n_features])

            # Find top interactions
            interactions = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interactions.append(
                        {
                            "feature_1": feature_names[i],
                            "feature_2": feature_names[j],
                            "correlation": abs(shap_corr[i, j]),
                            "idx_1": i,
                            "idx_2": j,
                        }
                    )

            interactions.sort(key=lambda x: x["correlation"], reverse=True)
            top_interactions = interactions[:top_n]

            # Create visualization
            fig_path = self._create_interaction_chart(
                values, feature_names, X_sample, top_interactions
            )

            # Build table
            table_md = "| Feature 1 | Feature 2 | SHAP Correlation | Strength |\n|-----------|-----------|-----------------|----------|\n"
            for inter in top_interactions:
                strength = (
                    "Strong"
                    if inter["correlation"] > 0.5
                    else ("Moderate" if inter["correlation"] > 0.3 else "Weak")
                )
                table_md += f"| {inter['feature_1'][:20]} | {inter['feature_2'][:20]} | {inter['correlation']:.4f} | {strength} |\n"

            content = f"""
# Feature Interaction Analysis

Feature interactions reveal how pairs of features jointly influence model predictions.
High SHAP correlation between features suggests they interact in the model.

## Top {top_n} Feature Interactions

{table_md}

## Interaction Visualizations

![Feature Interactions]({fig_path})

## Interpretation

- High SHAP correlation between two features means they jointly influence predictions
- Colored scatter shows how the interaction affects SHAP values
- Non-linear patterns suggest complex interactions the model has learned

---
"""
            self._content.append(content)
            self._store_section_data(
                "feature_interactions",
                "Feature Interactions",
                {
                    "n_interactions": len(top_interactions),
                    "top_pairs": [
                        {"f1": i["feature_1"], "f2": i["feature_2"], "corr": i["correlation"]}
                        for i in top_interactions
                    ],
                },
            )

        except Exception as e:
            self._record_section_failure("Feature Interactions", e)

    def _create_interaction_chart(
        self,
        shap_values: Any,
        feature_names: List[str],
        X_sample: Any,
        top_interactions: List[Dict],
    ) -> str:
        """Create feature interaction scatter plots."""
        try:
            n = len(top_interactions)

            with self._chart_context(
                "feature_interactions", figsize=(5 * n, 4.5), nrows=1, ncols=n
            ) as (fig, axes):
                if n == 1:
                    axes = [axes]

                X_arr = (
                    X_sample
                    if isinstance(X_sample, np.ndarray)
                    else X_sample.values if hasattr(X_sample, "values") else np.array(X_sample)
                )

                for i, inter in enumerate(top_interactions):
                    ax = axes[i]
                    idx1 = inter["idx_1"]
                    idx2 = inter["idx_2"]

                    feat1_vals = X_arr[:, idx1]
                    feat2_vals = X_arr[:, idx2]
                    shap_vals = shap_values[:, idx1]

                    scatter = ax.scatter(
                        feat1_vals, feat2_vals, c=shap_vals, cmap="coolwarm", alpha=0.6, s=20
                    )
                    fig.colorbar(scatter, ax=ax, label="SHAP Value", shrink=0.8)

                    ax.set_xlabel(inter["feature_1"][:15], fontsize=10)
                    ax.set_ylabel(inter["feature_2"][:15], fontsize=10)
                    ax.set_title(f"r={inter['correlation']:.3f}", fontsize=11, fontweight="medium")

                fig.suptitle(
                    "Feature Interaction Plots",
                    fontsize=14,
                    fontweight="bold",
                    color=self.theme["primary"],
                    y=1.05,
                )

            return "figures/feature_interactions.png"
        except Exception as e:
            logger.debug(f"Failed to create interaction chart: {e}")
            return ""

    # =========================================================================
    # INTERPRETABILITY ANALYSIS (LIME / PERTURBATION)
    # =========================================================================

    def add_interpretability_analysis(
        self, model: Any, X_sample: Any, y_pred: Any, feature_names: List[str], top_n: int = 5
    ) -> Any:
        """
        Add local interpretability analysis using LIME or perturbation-based explanations.

        Includes:
        - Most/least confident prediction explanations
        - Feature contribution analysis per instance
        - Counterfactual generation (greedy perturbation until prediction flips)

        Args:
            model: Trained model with predict/predict_proba
            X_sample: Feature array or DataFrame
            y_pred: Predictions array
            feature_names: List of feature names
            top_n: Number of most/least confident samples to explain
        """
        try:
            X_arr = (
                X_sample
                if isinstance(X_sample, np.ndarray)
                else (X_sample.values if hasattr(X_sample, "values") else np.array(X_sample))
            )
            # Ensure float64 to avoid numpy ufunc errors with object dtypes
            if X_arr.dtype == object:
                X_arr = X_arr.astype(np.float64)
            y_pred_arr = np.asarray(y_pred)

            # Get prediction confidences
            has_proba = hasattr(model, "predict_proba")
            if has_proba:
                probas = model.predict_proba(X_arr)
                confidences = np.max(probas, axis=1)
            else:
                confidences = np.ones(len(y_pred_arr))

            sorted_idx = np.argsort(confidences)
            most_confident = sorted_idx[-top_n:][::-1]
            least_confident = sorted_idx[:top_n]

            # Explanation cascade: SHAP -> LIME -> perturbation (last resort)
            method = "perturbation"
            explanations = []

            # Try SHAP first (global feature importance)
            shap_global = self._shap_explain(model, X_arr, feature_names)
            if shap_global is not None:
                method = "shap"
                # Use SHAP for per-sample explanations via KernelExplainer
                try:
                    import shap

                    is_tree = self._is_tree_model(model)

                    if is_tree:
                        explainer = shap.TreeExplainer(model)
                    else:
                        background = shap.sample(X_arr, min(100, len(X_arr)), random_state=42)
                        predict_fn = model.predict_proba if has_proba else model.predict
                        explainer = shap.KernelExplainer(predict_fn, background)

                    for idx in list(most_confident) + list(least_confident):
                        sv = (
                            explainer.shap_values(X_arr[idx : idx + 1], nsamples=200)
                            if not is_tree
                            else explainer.shap_values(X_arr[idx : idx + 1])
                        )
                        if isinstance(sv, list):
                            sv = sv[1]
                        contributions = {
                            feature_names[i]: float(sv[0][i]) for i in range(len(feature_names))
                        }
                        explanations.append(
                            {
                                "sample_idx": int(idx),
                                "confidence": float(confidences[idx]),
                                "predicted": int(y_pred_arr[idx]),
                                "contributions": contributions,
                            }
                        )
                except Exception as e:
                    # SHAP per-sample failed, fall back to perturbation for per-sample
                    self._record_internal_warning(
                        "SHAPPerSample",
                        "Per-sample SHAP explanation failed, falling back to perturbation",
                        e,
                    )
                    method = "shap_global_perturbation_local"
                    for idx in list(most_confident) + list(least_confident):
                        contributions = self._perturbation_explain(
                            model, X_arr, idx, feature_names, has_proba
                        )
                        explanations.append(
                            {
                                "sample_idx": int(idx),
                                "confidence": float(confidences[idx]),
                                "predicted": int(y_pred_arr[idx]),
                                "contributions": contributions,
                            }
                        )

            # Try LIME if SHAP not available
            if not explanations:
                try:
                    from lime.lime_tabular import (
                        LimeTabularExplainer,  # type: ignore[import-not-found]
                    )

                    explainer = LimeTabularExplainer(
                        X_arr,
                        feature_names=feature_names,
                        mode="classification",
                        discretize_continuous=True,
                        random_state=42,
                    )
                    method = "lime"

                    for idx in list(most_confident) + list(least_confident):
                        exp = explainer.explain_instance(
                            X_arr[idx],
                            model.predict_proba if has_proba else model.predict,
                            num_features=min(10, len(feature_names)),
                        )
                        contributions = {f: w for f, w in exp.as_list()}
                        explanations.append(
                            {
                                "sample_idx": int(idx),
                                "confidence": float(confidences[idx]),
                                "predicted": int(y_pred_arr[idx]),
                                "contributions": contributions,
                            }
                        )
                except ImportError as e:
                    self._record_internal_warning(
                        "LIMEImport", "LIME library import failed, skipping LIME explanations", e
                    )
                    pass

            # Perturbation-based last resort
            if not explanations:
                method = "perturbation"
                for idx in list(most_confident) + list(least_confident):
                    contributions = self._perturbation_explain(
                        model, X_arr, idx, feature_names, has_proba
                    )
                    explanations.append(
                        {
                            "sample_idx": int(idx),
                            "confidence": float(confidences[idx]),
                            "predicted": int(y_pred_arr[idx]),
                            "contributions": contributions,
                        }
                    )

            # SHAP interaction values (tree models only)
            interaction_data = self._shap_interaction_values(model, X_arr, feature_names)

            # Counterfactual generation (multiple per sample)
            all_counterfactuals = []
            for idx in most_confident[:3]:
                cf_list = self._generate_counterfactual(model, X_arr, idx, feature_names, has_proba)
                if cf_list:
                    all_counterfactuals.extend(cf_list)

            # Sensitivity analysis: effect by perturbation magnitude for top features
            sensitivity_data = {}
            if explanations:
                sample_idx = most_confident[0]
                for mag in [0.5, 1.0, 2.0]:
                    contrib = self._perturbation_explain(
                        model, X_arr, sample_idx, feature_names, has_proba, magnitudes=[mag]
                    )
                    sensitivity_data[mag] = contrib

            # Create visualization
            fig_path = self._create_lime_chart(explanations[:top_n], feature_names)

            # Build content
            content = f"""
# Interpretability Analysis

Local explanations showing why the model makes specific predictions.

**Method:** {'SHAP (SHapley Additive exPlanations)' if method.startswith('shap') else ('LIME (Local Interpretable Model-agnostic Explanations)' if method == 'lime' else 'Bidirectional perturbation-based feature contribution')}

## Most Confident Predictions

| Sample | Predicted | Confidence | Top Contributing Features |
|--------|-----------|------------|--------------------------|
"""
            for exp in explanations[:top_n]:
                top_contribs = sorted(
                    exp["contributions"].items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
                contribs_str = ", ".join([f"{f[:15]}={w:+.3f}" for f, w in top_contribs])
                content += f"| #{exp['sample_idx']} | {exp['predicted']} | {exp['confidence']:.4f} | {contribs_str} |\n"

            content += """
## Least Confident Predictions

| Sample | Predicted | Confidence | Top Contributing Features |
|--------|-----------|------------|--------------------------|
"""
            for exp in explanations[top_n:]:
                top_contribs = sorted(
                    exp["contributions"].items(), key=lambda x: abs(x[1]), reverse=True
                )[:3]
                contribs_str = ", ".join([f"{f[:15]}={w:+.3f}" for f, w in top_contribs])
                content += f"| #{exp['sample_idx']} | {exp['predicted']} | {exp['confidence']:.4f} | {contribs_str} |\n"

            # Sensitivity Analysis subsection
            if sensitivity_data:
                content += """
## Sensitivity Analysis

Effect of perturbation magnitude on top-5 features (bidirectional, normalized):

| Feature | 0.5x std | 1.0x std | 2.0x std |
|---------|----------|----------|----------|
"""
                # Get top 5 features from 1.0x magnitude
                if 1.0 in sensitivity_data:
                    top5 = sorted(
                        sensitivity_data[1.0].items(), key=lambda x: abs(x[1]), reverse=True
                    )[:5]
                    for feat, _ in top5:
                        vals = []
                        for mag in [0.5, 1.0, 2.0]:
                            v = sensitivity_data.get(mag, {}).get(feat, 0.0)
                            vals.append(f"{v:.4f}")
                        content += f"| {feat[:25]} | {vals[0]} | {vals[1]} | {vals[2]} |\n"

            if fig_path:
                content += f"""
## Feature Contributions (Most Confident Samples)

![Interpretability Analysis]({fig_path})

"""

            if all_counterfactuals:
                # Show the best counterfactual (fewest changes) from multiple
                best_cf = min(all_counterfactuals, key=lambda x: x["n_changes"])
                content += f"""## Counterfactual Analysis

What minimal changes would flip the prediction? (Generated {len(all_counterfactuals)} counterfactual(s), showing best)

**Best counterfactual** for sample #{best_cf['sample_idx']}: {best_cf['n_changes']} feature change(s) needed

| Sample | Original Pred | New Pred | # Changes | Changes Required |
|--------|--------------|----------|-----------|-----------------|
"""
                for cf in all_counterfactuals:
                    changes = ", ".join(
                        [
                            f"{c['feature'][:15]}: {c['from']:.2f}\u2192{c['to']:.2f}"
                            for c in cf["changes"][:3]
                        ]
                    )
                    content += f"| #{cf['sample_idx']} | {cf['original_pred']} | {cf['new_pred']} | {cf['n_changes']} | {changes} |\n"

            # Feature Interactions subsection (SHAP interaction values)
            if interaction_data:
                content += """## Feature Interactions (SHAP)

Top feature interaction pairs detected via SHAP interaction values:

| Feature 1 | Feature 2 | Interaction Strength |
|-----------|-----------|---------------------|
"""
                for inter in interaction_data:
                    content += f"| {inter['feature_1'][:20]} | {inter['feature_2'][:20]} | {inter['interaction_value']:.4f} |\n"
                content += "\n"

            narrative = self._maybe_add_narrative(
                "Interpretability Analysis",
                f"Method: {method}, Explained: {len(explanations)}, Counterfactuals: {len(all_counterfactuals)}",
                section_type="interpretability",
            )

            content += f"""
{narrative}

---
"""
            self._content.append(content)
            self._store_section_data(
                "interpretability",
                "Interpretability Analysis",
                {
                    "method": method,
                    "n_explained": len(explanations),
                    "n_counterfactuals": len(all_counterfactuals),
                    "sensitivity_data": (
                        {
                            str(k): {fk: float(fv) for fk, fv in v.items()}
                            for k, v in sensitivity_data.items()
                        }
                        if sensitivity_data
                        else {}
                    ),
                    "interaction_data": interaction_data if interaction_data else [],
                },
                [{"type": "importance_bar"}],
            )

        except Exception as e:
            self._record_section_failure("Interpretability Analysis", e)

    def _shap_explain(self, model: Any, X_arr: Any, feature_names: Any) -> Any:
        """Compute SHAP-based feature contributions using TreeExplainer or KernelExplainer.

        Args:
            model: Trained model with predict/predict_proba
            X_arr: Feature array (n_samples, n_features)
            feature_names: List of feature names

        Returns:
            Dict mapping feature_name -> mean absolute SHAP value, or None if SHAP fails
        """
        try:
            import shap

            # Ensure float64 to avoid numpy ufunc errors with object dtypes
            if hasattr(X_arr, "dtype") and X_arr.dtype == object:
                X_arr = X_arr.astype(np.float64)

            # Try TreeExplainer for tree-based models
            is_tree = self._is_tree_model(model)

            if is_tree:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_arr)
                    # Handle binary classification (list of arrays)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Positive class
                    mean_abs = np.mean(np.abs(shap_values), axis=0)
                    return {feature_names[i]: float(mean_abs[i]) for i in range(len(feature_names))}
                except Exception as e:
                    self._record_internal_warning(
                        "SHAPTreeExplainer",
                        "TreeExplainer failed, falling back to KernelExplainer",
                        e,
                    )
                    pass  # Fall through to KernelExplainer

            # KernelExplainer fallback for non-tree models
            try:
                background_size = min(100, len(X_arr))
                background = (
                    shap.sample(X_arr, background_size, random_state=42)
                    if len(X_arr) > background_size
                    else X_arr
                )

                predict_fn = (
                    model.predict_proba if hasattr(model, "predict_proba") else model.predict
                )
                explainer = shap.KernelExplainer(predict_fn, background)

                sample_size = min(50, len(X_arr))
                shap_values = explainer.shap_values(X_arr[:sample_size], nsamples=200)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class

                mean_abs = np.mean(np.abs(shap_values), axis=0)
                return {feature_names[i]: float(mean_abs[i]) for i in range(len(feature_names))}
            except Exception as e:
                self._record_internal_warning(
                    "SHAPKernelExplainer", "KernelExplainer failed to compute SHAP values", e
                )
                pass

        except ImportError as e:
            self._record_internal_warning("SHAPExplain", "SHAP library import failed", e)
            pass
        except Exception as e:
            self._record_internal_warning(
                "SHAPExplain", "Unexpected error during SHAP explanation", e
            )
            pass

        return None

    def _shap_interaction_values(
        self, model: Any, X_arr: Any, feature_names: Any, top_k: Any = 5
    ) -> Any:
        """Compute SHAP interaction values for tree-based models.

        Uses TreeExplainer.shap_interaction_values to extract top-k feature interaction pairs.

        Args:
            model: Trained tree-based model
            X_arr: Feature array
            feature_names: List of feature names
            top_k: Number of top interaction pairs to return

        Returns:
            List[Dict] with feature_1, feature_2, interaction_value, or empty list
        """
        try:
            import shap

            is_tree = self._is_tree_model(model)

            if not is_tree:
                return []

            explainer = shap.TreeExplainer(model)
            sample_size = min(100, len(X_arr))
            interaction_values = explainer.shap_interaction_values(X_arr[:sample_size])

            # Handle binary classification
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[1]

            # Average absolute interaction values across samples
            mean_interactions = np.mean(np.abs(interaction_values), axis=0)

            # Extract top-k off-diagonal pairs
            n_features = min(len(feature_names), mean_interactions.shape[0])
            pairs = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    pairs.append(
                        {
                            "feature_1": feature_names[i],
                            "feature_2": feature_names[j],
                            "interaction_value": float(mean_interactions[i, j]),
                        }
                    )

            pairs.sort(key=lambda x: x["interaction_value"], reverse=True)
            return pairs[:top_k]

        except Exception as e:
            self._record_internal_warning(
                "SHAPInteractionValues", "Failed to compute SHAP interaction values", e
            )
            return []

    def _perturbation_explain(
        self,
        model: Any,
        X_arr: Any,
        idx: Any,
        feature_names: Any,
        has_proba: Any,
        magnitudes: Any = None,
    ) -> Any:
        """Compute feature contributions via bidirectional multi-magnitude perturbation.

        Args:
            model: Trained model with predict/predict_proba
            X_arr: Full feature array (used for computing stats)
            idx: Sample index to explain
            feature_names: List of feature names
            has_proba: Whether model supports predict_proba
            magnitudes: List of perturbation magnitudes as fractions of std (default [0.5, 1.0, 2.0])

        Returns:
            Dict mapping feature_name -> average absolute effect (normalized)
        """
        if magnitudes is None:
            magnitudes = [0.5, 1.0, 2.0]

        contributions = {}
        original = X_arr[idx].copy()

        if has_proba:
            base_pred = model.predict_proba(original.reshape(1, -1))[0]
            base_magnitude = float(np.max(base_pred))
        else:
            base_pred = model.predict(original.reshape(1, -1))[0]
            base_magnitude = float(abs(base_pred)) if abs(base_pred) > 0 else 1.0

        col_means = np.mean(X_arr, axis=0)
        col_stds = np.std(X_arr, axis=0)

        # Build all perturbation samples in batch for efficiency
        batch_rows = []
        batch_meta = []  # (feature_index, perturbation_index)

        for i, feat in enumerate(feature_names):
            std_val = col_stds[i] if col_stds[i] > 0 else 1.0

            for mag in magnitudes:
                for direction in [1.0, -1.0]:
                    perturbed = original.copy()
                    delta = direction * mag * std_val
                    perturbed[i] += delta

                    # Reject out-of-distribution perturbations (>3 std from mean)
                    if abs(perturbed[i] - col_means[i]) > 3.0 * std_val:
                        continue

                    batch_rows.append(perturbed)
                    batch_meta.append(i)

        if not batch_rows:
            return {feat: 0.0 for feat in feature_names}

        batch_array = np.array(batch_rows)

        # Single batched prediction call
        if has_proba:
            batch_preds = model.predict_proba(batch_array)
            diffs = np.max(np.abs(batch_preds - base_pred), axis=1)
        else:
            batch_preds = model.predict(batch_array)
            diffs = np.abs(batch_preds - base_pred)

        # Aggregate effects per feature
        feature_effects = {i: [] for i in range(len(feature_names))}
        for j, feat_idx in enumerate(batch_meta):
            feature_effects[feat_idx].append(float(diffs[j]))

        for i, feat in enumerate(feature_names):
            effects = feature_effects[i]
            if effects:
                avg_effect = float(np.mean(effects))
                contributions[feat] = (
                    avg_effect / base_magnitude if base_magnitude > 0 else avg_effect
                )
            else:
                contributions[feat] = 0.0

        return contributions

    def _find_prototype_counterfactual(
        self, model: Any, X_arr: Any, idx: Any, feature_names: Any
    ) -> Any:
        """Find nearest training instance with opposite prediction (L2 in standardized space).

        Always valid since it's a real observed data point.

        Args:
            model: Trained model with predict
            X_arr: Full feature array
            idx: Sample index
            feature_names: List of feature names

        Returns:
            Dict with counterfactual info, or None if no opposite-class instance found
        """
        try:
            original = X_arr[idx].copy()
            original_pred = int(model.predict(original.reshape(1, -1))[0])

            # Standardize for distance computation
            col_means = np.mean(X_arr, axis=0)
            col_stds = np.std(X_arr, axis=0)
            col_stds[col_stds == 0] = 1.0

            original_std = (original - col_means) / col_stds

            # Get predictions for all training instances
            all_preds = model.predict(X_arr)

            # Find instances with opposite prediction
            opposite_mask = all_preds != original_pred
            if not opposite_mask.any():
                return None

            opposite_indices = np.where(opposite_mask)[0]
            X_opposite_std = (X_arr[opposite_indices] - col_means) / col_stds

            # L2 distance in standardized space
            distances = np.sqrt(
                np.asarray((X_opposite_std - original_std) ** 2).sum(axis=1).astype(np.float64)
            )
            nearest_idx = opposite_indices[np.argmin(distances)]
            nearest = X_arr[nearest_idx]
            new_pred = int(all_preds[nearest_idx])

            # Record changes (features that differ significantly)
            changes = []
            for i, feat in enumerate(feature_names):
                diff = abs(nearest[i] - original[i])
                if diff > 1e-10:
                    changes.append(
                        {
                            "feature": feat,
                            "from": float(original[i]),
                            "to": float(nearest[i]),
                        }
                    )

            return {
                "sample_idx": int(idx),
                "original_pred": original_pred,
                "new_pred": new_pred,
                "changes": changes,
                "n_changes": len(changes),
                "method": "prototype",
            }
        except Exception as e:
            self._record_internal_warning(
                "PrototypeCounterfactual", "Failed to find prototype counterfactual", e
            )
            return None

    def _growing_spheres_counterfactual(
        self,
        model: Any,
        X_arr: Any,
        idx: Any,
        feature_names: Any,
        n_directions: Any = 50,
        max_radius_steps: Any = 20,
    ) -> Any:
        """Find counterfactual by expanding random perturbation radius until prediction flips.

        Args:
            model: Trained model with predict
            X_arr: Full feature array
            idx: Sample index
            feature_names: List of feature names
            n_directions: Random directions per radius step
            max_radius_steps: Maximum expansion steps

        Returns:
            Dict with counterfactual info, or None
        """
        try:
            original = X_arr[idx].copy()
            original_pred = int(model.predict(original.reshape(1, -1))[0])

            col_mins = np.min(X_arr, axis=0)
            col_maxs = np.max(X_arr, axis=0)
            col_stds = np.std(X_arr, axis=0)
            col_stds[col_stds == 0] = 1.0

            rng = np.random.RandomState(42)
            n_features = len(feature_names)

            best_cf = None
            best_radius = float("inf")

            for step in range(1, max_radius_steps + 1):
                radius = step * 0.2  # Start small, expand in 0.2 std increments

                for _ in range(n_directions):
                    direction = rng.randn(n_features)
                    direction = direction / (np.linalg.norm(direction) + 1e-10)

                    perturbed = original + direction * radius * col_stds
                    perturbed = np.clip(perturbed, col_mins, col_maxs)

                    new_pred = int(model.predict(perturbed.reshape(1, -1))[0])
                    if new_pred != original_pred:
                        actual_radius = np.sqrt(
                            np.float64(np.sum(((perturbed - original) / col_stds) ** 2))
                        )
                        if actual_radius < best_radius:
                            best_radius = actual_radius
                            changes = []
                            for i, feat in enumerate(feature_names):
                                diff = abs(perturbed[i] - original[i])
                                if diff > 1e-10:
                                    changes.append(
                                        {
                                            "feature": feat,
                                            "from": float(original[i]),
                                            "to": float(perturbed[i]),
                                        }
                                    )
                            best_cf = {
                                "sample_idx": int(idx),
                                "original_pred": original_pred,
                                "new_pred": new_pred,
                                "changes": changes,
                                "n_changes": len(changes),
                                "method": "growing_spheres",
                            }

                if best_cf is not None:
                    break  # Found at this radius, no need to expand

            return best_cf
        except Exception as e:
            self._record_internal_warning(
                "GrowingSpheresCounterfactual",
                "Failed to generate growing spheres counterfactual",
                e,
            )
            return None

    def _sparse_counterfactual(
        self, model: Any, X_arr: Any, idx: Any, feature_names: Any, actionability: Any = None
    ) -> Any:
        """Generate counterfactual with explicit L0 sparsity constraint.

        Tries single-feature changes first, then pairs, then triples.
        Uses top-10 features by importance only.

        Args:
            model: Trained model with predict
            X_arr: Full feature array
            idx: Sample index
            feature_names: List of feature names
            actionability: Optional dict {feature: 'increase_only'|'decrease_only'|'fixed'|'free'}

        Returns:
            Dict with counterfactual info, or None
        """
        try:
            from itertools import combinations

            original = X_arr[idx].copy()
            original_pred = int(model.predict(original.reshape(1, -1))[0])

            col_means = np.mean(X_arr, axis=0)
            col_stds = np.std(X_arr, axis=0)
            col_stds[col_stds == 0] = 1.0

            # Get feature importance for ordering, use top-10
            contributions = self._perturbation_explain(model, X_arr, idx, feature_names, True)
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = [f for f, _ in sorted_features[:10]]

            percentiles = [25, 50, 75]
            feature_percentiles = {}
            for feat in top_features:
                feat_idx = feature_names.index(feat)
                feature_percentiles[feat] = np.percentile(X_arr[:, feat_idx], percentiles).tolist()
                feature_percentiles[feat].append(float(col_means[feat_idx]))

            actionability = actionability or {}

            # Try single features, then pairs, then triples
            for n_changes in range(1, min(4, len(top_features) + 1)):
                for combo in combinations(top_features, n_changes):
                    # Generate target value combinations
                    target_sets = [feature_percentiles[f] for f in combo]

                    # Iterate over all target value combinations
                    from itertools import product as iter_product

                    for targets in iter_product(*target_sets):
                        perturbed = original.copy()
                        changes = []
                        valid = True

                        for feat, target_val in zip(combo, targets):
                            feat_idx = feature_names.index(feat)
                            old_val = original[feat_idx]

                            # Check actionability constraints
                            constraint = actionability.get(feat, "free")
                            if constraint == "fixed":
                                valid = False
                                break
                            elif constraint == "increase_only" and target_val < old_val:
                                valid = False
                                break
                            elif constraint == "decrease_only" and target_val > old_val:
                                valid = False
                                break

                            perturbed[feat_idx] = target_val
                            if abs(target_val - old_val) > 1e-10:
                                changes.append(
                                    {
                                        "feature": feat,
                                        "from": float(old_val),
                                        "to": float(target_val),
                                    }
                                )

                        if not valid or not changes:
                            continue

                        new_pred = int(model.predict(perturbed.reshape(1, -1))[0])
                        if new_pred != original_pred:
                            return {
                                "sample_idx": int(idx),
                                "original_pred": original_pred,
                                "new_pred": new_pred,
                                "changes": changes,
                                "n_changes": len(changes),
                                "method": "sparse",
                            }

            return None
        except Exception as e:
            self._record_internal_warning(
                "SparseCounterfactual", "Failed to generate sparse counterfactual", e
            )
            return None

    def _greedy_mean_counterfactual(
        self,
        model: Any,
        X_arr: Any,
        idx: Any,
        feature_names: Any,
        has_proba: Any,
        n_counterfactuals: Any = 3,
        step_fraction: Any = 0.1,
    ) -> Any:
        """Generate counterfactuals by incremental perturbation toward feature means (legacy).

        Args:
            model: Trained model with predict
            X_arr: Full feature array
            idx: Sample index
            feature_names: List of feature names
            has_proba: Whether model supports predict_proba
            n_counterfactuals: Number of diverse counterfactuals to attempt
            step_fraction: Fraction of distance to mean per step

        Returns:
            List of counterfactual result dicts, or None
        """
        try:
            original = X_arr[idx].copy()
            original_pred = int(model.predict(original.reshape(1, -1))[0])

            col_mins = np.min(X_arr, axis=0)
            col_maxs = np.max(X_arr, axis=0)
            col_means = np.mean(X_arr, axis=0)

            contributions = self._perturbation_explain(model, X_arr, idx, feature_names, has_proba)
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

            results = []
            rng = np.random.RandomState(42)

            for cf_idx in range(n_counterfactuals):
                perturbed = original.copy()
                changes = []

                if cf_idx == 0:
                    feature_order = [f for f, _ in sorted_features]
                else:
                    feature_order = [f for f, _ in sorted_features]
                    rng.shuffle(feature_order)

                found = False
                for feat in feature_order:
                    feat_idx = feature_names.index(feat)
                    old_val = perturbed[feat_idx]
                    target_val = col_means[feat_idx]
                    distance = target_val - old_val

                    if abs(distance) < 1e-10:
                        continue

                    n_steps = max(1, int(1.0 / step_fraction))
                    for step in range(1, n_steps + 1):
                        new_val = old_val + distance * step * step_fraction
                        new_val = float(np.clip(new_val, col_mins[feat_idx], col_maxs[feat_idx]))
                        perturbed[feat_idx] = new_val

                    changes.append(
                        {
                            "feature": feat,
                            "from": float(old_val),
                            "to": float(perturbed[feat_idx]),
                        }
                    )

                    new_pred = int(model.predict(perturbed.reshape(1, -1))[0])
                    if new_pred != original_pred:
                        results.append(
                            {
                                "sample_idx": int(idx),
                                "original_pred": original_pred,
                                "new_pred": new_pred,
                                "changes": changes,
                                "n_changes": len(changes),
                                "method": "greedy_mean",
                            }
                        )
                        found = True
                        break

                if not found and changes:
                    pass

            if results:
                results.sort(key=lambda x: x["n_changes"])
                return results
            return None
        except Exception as e:
            self._record_internal_warning(
                "GreedyMeanCounterfactual", "Failed to generate greedy mean counterfactual", e
            )
            return None

    def _generate_counterfactual(
        self,
        model: Any,
        X_arr: Any,
        idx: Any,
        feature_names: Any,
        has_proba: Any,
        n_counterfactuals: Any = 3,
        step_fraction: Any = 0.1,
        actionability: Any = None,
    ) -> Any:
        """Orchestrate counterfactual generation using multiple strategies.

        Priority: prototype -> sparse -> growing_spheres -> greedy_mean.
        Collects up to n_counterfactuals diverse results.

        Args:
            model: Trained model with predict
            X_arr: Full feature array
            idx: Sample index
            feature_names: List of feature names
            has_proba: Whether model supports predict_proba
            n_counterfactuals: Max number of counterfactuals to return
            step_fraction: Step fraction for greedy mean fallback
            actionability: Optional dict {feature: constraint} for sparse method

        Returns:
            List of counterfactual result dicts, or None if none found
        """
        try:
            results = []

            # 1. Prototype (nearest opposite-class instance)
            proto = self._find_prototype_counterfactual(model, X_arr, idx, feature_names)
            if proto:
                results.append(proto)

            # 2. Sparse (L0-minimal changes)
            if len(results) < n_counterfactuals:
                sparse = self._sparse_counterfactual(
                    model, X_arr, idx, feature_names, actionability=actionability
                )
                if sparse:
                    results.append(sparse)

            # 3. Growing spheres (random direction expansion)
            if len(results) < n_counterfactuals:
                gs = self._growing_spheres_counterfactual(model, X_arr, idx, feature_names)
                if gs:
                    results.append(gs)

            # 4. Greedy mean (legacy fallback)
            if len(results) < n_counterfactuals:
                greedy = self._greedy_mean_counterfactual(
                    model,
                    X_arr,
                    idx,
                    feature_names,
                    has_proba,
                    n_counterfactuals=max(1, n_counterfactuals - len(results)),
                    step_fraction=step_fraction,
                )
                if greedy:
                    results.extend(greedy)

            # Deduplicate and limit
            if results:
                results.sort(key=lambda x: x["n_changes"])
                return results[:n_counterfactuals]
            return None
        except Exception as e:
            self._record_internal_warning(
                "CounterfactualGeneration", "Failed to orchestrate counterfactual generation", e
            )
            return None

    def _create_lime_chart(self, explanations: Any, feature_names: Any) -> Any:
        """Create horizontal bar chart of feature contributions."""
        try:
            n_exp = min(len(explanations), 4)
            if n_exp == 0:
                return ""

            with self._chart_context(
                "interpretability_analysis", figsize=(5 * n_exp, 5), nrows=1, ncols=n_exp
            ) as (fig, axes):
                if n_exp == 1:
                    axes = [axes]

                for i, exp in enumerate(explanations[:n_exp]):
                    ax = axes[i]
                    sorted_contribs = sorted(exp["contributions"].items(), key=lambda x: x[1])[-10:]
                    names = [c[0][:20] for c in sorted_contribs]
                    vals = [c[1] for c in sorted_contribs]
                    colors = [
                        self.theme["success"] if v > 0 else self.theme["danger"] for v in vals
                    ]

                    ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
                    ax.set_yticks(range(len(names)))
                    ax.set_yticklabels(names, fontsize=8)
                    ax.set_title(
                        f"Sample #{exp['sample_idx']}\n(conf={exp['confidence']:.3f})",
                        fontsize=10,
                        fontweight="bold",
                        color=self.theme["primary"],
                    )
                    ax.axvline(x=0, color="gray", linewidth=0.5)

                fig.suptitle(
                    "Feature Contributions (Local Explanations)",
                    fontsize=14,
                    fontweight="bold",
                    color=self.theme["primary"],
                    y=1.02,
                )

            return "figures/interpretability_analysis.png"
        except Exception as e:
            logger.debug(f"Failed to create interpretability chart: {e}")
            return ""
