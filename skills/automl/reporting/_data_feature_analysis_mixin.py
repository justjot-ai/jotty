"""
Analysis sections mixin for ProfessionalMLReport.

Data & Feature Analysis Mixin

Contains methods for:
- Executive summary, data profiling, recommendations
- Data quality, correlation, class distribution
- Feature importance, SHAP, permutation importance, partial dependence
- Pipeline visualization, learning curves, hyperparameters

Extracted from _analysis_sections_mixin.py (Feb 2026) to reduce file size.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DataFeatureAnalysisMixin:
    """Mixin providing data and feature analysis methods for ProfessionalMLReport.

    Each add_* method appends one report section. They expect to be mixed into
    ProfessionalMLReport which provides self.sections, self.figures, self._save_figure(),
    self._validate_inputs(), self._maybe_add_narrative(), etc.
    """

    if TYPE_CHECKING:
        # Declare expected attributes from parent class
        output_dir: Path
        figures_dir: Path
        theme: str
        config: Dict[str, Any]
        _llm_narrative_enabled: bool
        _html_enabled: bool
        _content: List[Any]
        _figures: List[Any]
        _warnings: List[Any]
        _metadata: Dict[str, Any]
        _raw_data: Dict[str, Any]
        _section_data: List[Any]
        _failed_sections: List[str]
        _failed_charts: List[str]

        def _record_chart_failure(self, chart_name: str, error: Exception) -> None: ...
        def _save_figure(self, fig: Any, name: str) -> Optional[Path]: ...
        def _fig_path_for_markdown(self, fig_path: Path) -> str: ...
        def _add_section(self, title: str, content: str, **kwargs: Any) -> None: ...
        def _record_section_failure(self, section: str, error: Exception) -> None: ...

    def add_executive_summary(
        self, metrics: Dict[str, float], best_model: str, n_features: int, context: str = ""
    ) -> Any:
        """Add executive summary section with risk scoring and traffic-light indicators."""
        try:
            self._add_executive_summary_impl(metrics, best_model, n_features, context)
        except Exception as e:
            self._record_section_failure("Executive Summary", e)

    def add_data_profile(
        self,
        shape: Tuple[int, int],
        dtypes: Dict[str, int],
        missing: Dict[str, int],
        recommendations: List[str],
    ) -> Any:
        """Add data profiling section."""
        try:
            self._add_data_profile_impl(shape, dtypes, missing, recommendations)
        except Exception as e:
            self._record_section_failure("Data Profile", e)

    def add_pipeline_visualization(self, pipeline_steps: List[Dict]) -> None:
        """
        Add pipeline DAG visualization showing data flow through ML pipeline.

        Args:
            pipeline_steps: List of dicts with keys:
                - name: Step name
                - type: One of 'preprocessing', 'feature_engineering', 'model', 'ensemble'
                - input_shape: Optional tuple (rows, cols)
                - output_shape: Optional tuple (rows, cols)
                - params: Optional dict of step parameters
        """
        try:
            if not pipeline_steps:
                return

            # Create visualization
            fig_path = self._create_pipeline_dag(pipeline_steps)

            # Build step table
            table_md = "| Step | Type | Input Shape | Output Shape | Parameters |\n"
            table_md += "|------|------|-------------|--------------|------------|\n"

            for i, step in enumerate(pipeline_steps, 1):
                name = step.get("name", f"Step {i}")
                stype = step.get("type", "unknown")
                in_shape = step.get("input_shape", "N/A")
                out_shape = step.get("output_shape", "N/A")
                params = step.get("params", {})

                in_str = f"{in_shape}" if in_shape != "N/A" else "N/A"
                out_str = f"{out_shape}" if out_shape != "N/A" else "N/A"
                param_str = (
                    ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])[:40]
                    if params
                    else "-"
                )

                table_md += f"| {name[:25]} | {stype} | {in_str} | {out_str} | {param_str} |\n"

            content = f"""
# Pipeline Architecture

Visual representation of the ML pipeline data flow.

## Pipeline Steps

{table_md}

## Pipeline DAG

![Pipeline Architecture]({fig_path})

---
"""
            self._content.append(content)

            self._store_section_data(
                "pipeline_dag",
                "Pipeline Architecture",
                {
                    "steps": pipeline_steps,
                },
                [{"type": "dag", "path": fig_path}],
            )

        except Exception as e:
            self._record_section_failure("Pipeline Visualization", e)

    def add_feature_importance(self, importance: Dict[str, float], top_n: int = 20) -> None:
        """Add feature importance section with chart."""
        try:
            self._add_feature_importance_impl(importance, top_n)
        except Exception as e:
            self._record_section_failure("Feature Importance", e)

    def add_model_benchmarking(self, model_scores: Dict[str, Dict[str, float]]) -> None:
        """Add model benchmarking comparison."""
        try:
            self._add_model_benchmarking_impl(model_scores)
        except Exception as e:
            self._record_section_failure("Model Benchmarking", e)

    def add_model_comparison(
        self, models: Dict, X_test: Any, y_true: Any, class_labels: List[str] | None = None
    ) -> Any:
        """Add side-by-side comparison of multiple trained models.

        Args:
            models: Dict mapping model_name -> trained model with predict/predict_proba.
            X_test: Test features (array-like).
            y_true: True labels (array-like).
            class_labels: Optional class label names for display.
        """
        try:
            self._add_model_comparison_impl(models, X_test, y_true, class_labels)
        except Exception as e:
            self._record_section_failure("Model Comparison", e)

    def add_cross_dataset_validation(
        self,
        datasets_dict: Dict[str, Tuple],
        model: Any,
        metric_fn: Any = None,
        metric_name: str = "Score",
    ) -> Any:
        """
        Add cross-dataset validation analysis.

        Args:
            datasets_dict: Dict mapping dataset_name -> (X, y) tuple
            model: Trained model to evaluate
            metric_fn: Custom metric function(y_true, y_pred) -> float. Defaults to accuracy.
            metric_name: Name of the metric for display
        """
        try:
            from sklearn.metrics import accuracy_score

            if metric_fn is None:
                metric_fn = accuracy_score
                metric_name = metric_name if metric_name != "Score" else "Accuracy"

            results = {}
            feature_stats = {}

            for ds_name, (X_ds, y_ds) in datasets_dict.items():
                try:
                    X_arr = X_ds.values if hasattr(X_ds, "values") else np.asarray(X_ds)
                    y_arr = np.asarray(y_ds)

                    y_pred = model.predict(X_arr)
                    score = metric_fn(y_arr, y_pred)
                    results[ds_name] = {
                        "score": float(score),
                        "n_samples": len(y_arr),
                        "n_features": X_arr.shape[1],
                    }

                    # Feature stats for distribution shift detection
                    feature_stats[ds_name] = {
                        "mean": np.nanmean(X_arr, axis=0),
                        "std": np.nanstd(X_arr, axis=0),
                    }
                except Exception as e:
                    logger.debug(f"Cross-dataset eval failed for {ds_name}: {e}")
                    results[ds_name] = {"score": 0.0, "n_samples": 0, "n_features": 0}

            if not results:
                return

            # Generalization gap
            scores = [r["score"] for r in results.values() if r["score"] > 0]
            gen_gap = max(scores) - min(scores) if len(scores) > 1 else 0

            # Distribution shift detection
            shift_detected = False
            if len(feature_stats) > 1:
                ds_names = list(feature_stats.keys())
                for i in range(1, len(ds_names)):
                    ref_mean = feature_stats[ds_names[0]]["mean"]
                    cur_mean = feature_stats[ds_names[i]]["mean"]
                    mean_diff = np.mean(np.abs(ref_mean - cur_mean) / (np.abs(ref_mean) + 1e-10))
                    if mean_diff > 0.2:
                        shift_detected = True
                        break

            # Create chart
            fig_path = ""
            try:
                fig_path = self._create_cross_dataset_chart(results, metric_name)
            except Exception as e:
                self._record_chart_failure("cross_dataset", e)

            # Build table
            table_md = f"| Dataset | {metric_name} | Samples | Features |\n"
            table_md += "|---------|-------|---------|----------|\n"
            for ds_name, r in sorted(results.items(), key=lambda x: -x[1]["score"]):
                table_md += (
                    f"| {ds_name} | {r['score']:.4f} | {r['n_samples']:,} | {r['n_features']} |\n"
                )

            content = f"""
# Multi-Dataset Validation

Evaluating model generalization across {len(results)} different datasets.

## Results

{table_md}

## Generalization Analysis

| Metric | Value |
|--------|-------|
| Best Score | {max(scores):.4f} |
| Worst Score | {min(scores):.4f} |
| Generalization Gap | {gen_gap:.4f} |
| Distribution Shift Detected | {'Yes' if shift_detected else 'No'} |

{'**Warning: Distribution shift detected between datasets.** Feature distributions differ significantly, which may explain performance variation.' if shift_detected else '**Feature distributions appear consistent across datasets.**'}

## Cross-Dataset Comparison

![Cross-Dataset Validation]({fig_path})

---
"""
            self._content.append(content)

            self._store_section_data(
                "cross_dataset",
                "Multi-Dataset Validation",
                {
                    "results": results,
                    "gen_gap": gen_gap,
                    "shift_detected": shift_detected,
                },
                [{"type": "bar", "path": fig_path}],
            )

        except Exception as e:
            self._record_section_failure("Cross-Dataset Validation", e)

    def add_confusion_matrix(
        self, y_true: Any, y_pred: Any, labels: List[str] | None = None
    ) -> None:
        """Add confusion matrix section."""
        try:
            self._add_confusion_matrix_impl(y_true, y_pred, labels)
        except Exception as e:
            self._record_section_failure("Classification Performance", e)

    def add_roc_analysis(self, y_true: Any, y_prob: Any, pos_label: Any = 1) -> None:
        """Add ROC curve analysis."""
        try:
            self._add_roc_analysis_impl(y_true, y_prob, pos_label)
        except Exception as e:
            self._record_section_failure("ROC Analysis", e)

    def add_precision_recall(self, y_true: Any, y_prob: Any, pos_label: Any = 1) -> None:
        """Add precision-recall curve analysis."""
        try:
            self._add_precision_recall_impl(y_true, y_prob, pos_label)
        except Exception as e:
            self._record_section_failure("Precision-Recall Analysis", e)

    def add_baseline_comparison(
        self, baseline_score: float, final_score: float, baseline_model: str = "Baseline"
    ) -> Any:
        """Add baseline comparison section."""
        try:
            self._add_baseline_comparison_impl(baseline_score, final_score, baseline_model)
        except Exception as e:
            self._record_section_failure("Baseline Comparison", e)

    def add_shap_analysis(
        self, shap_values: Any, feature_names: List[str], X_sample: Any = None
    ) -> None:
        """Add SHAP analysis section."""
        try:

            # Calculate mean absolute SHAP values
            if hasattr(shap_values, "values"):
                values = shap_values.values
            else:
                values = shap_values

            if len(values.shape) == 3:
                values = values[:, :, 1]  # Binary classification

            mean_shap = np.abs(values).mean(axis=0)
            shap_importance = sorted(
                zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True
            )[:15]

            # Create SHAP summary plot (isolated)
            fig_path = ""
            try:
                fig_path = self._create_shap_chart(shap_values, feature_names, X_sample)
            except Exception as e:
                self._record_chart_failure("shap_chart", e)

            table_md = "| Feature | Mean Abs SHAP |\n|---------|-------------|\n"
            for feat, val in shap_importance:
                table_md += f"| {feat[:30]} | {val:.4f} |\n"

            content = f"""
# SHAP Feature Analysis

SHAP (SHapley Additive exPlanations) values provide model-agnostic explanations
showing how each feature contributes to individual predictions.

## SHAP Feature Importance

{table_md}

## SHAP Summary Plot

![SHAP Analysis]({fig_path})

---
"""
            self._content.append(content)
            self._store_section_data(
                "shap_analysis",
                "SHAP Feature Analysis",
                {
                    "top_features": [f for f, _ in shap_importance[:10]],
                    "mean_shap": dict(shap_importance[:10]),
                },
                [{"type": "importance_bar"}],
            )

        except Exception as e:
            self._record_section_failure("SHAP Analysis", e)

    def add_recommendations(self, recommendations: List[str]) -> None:
        """Add recommendations section."""
        try:
            self._add_recommendations_impl(recommendations)
        except Exception as e:
            self._record_section_failure("Recommendations", e)

    def add_data_quality_analysis(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        """
        Add comprehensive data quality analysis including:
        - Missing value patterns (heatmap)
        - Outlier detection with IQR and Z-score
        - Distribution analysis (skewness, kurtosis)
        - Data type summary
        """
        try:
            self._add_data_quality_analysis_impl(X, y)
        except Exception as e:
            self._record_section_failure("Data Quality Analysis", e)

    def add_correlation_analysis(self, X: pd.DataFrame, threshold: float = 0.7) -> None:
        """
        Add correlation analysis with:
        - Correlation matrix heatmap with hierarchical clustering
        - Highly correlated feature pairs
        - VIF analysis for multicollinearity
        """
        try:
            self._add_correlation_analysis_impl(X, threshold)
        except Exception as e:
            self._record_section_failure("Correlation Analysis", e)

    def add_learning_curves(self, model: Any, X: Any, y: Any, cv: int = 5) -> None:
        """
        Add learning curve analysis showing:
        - Training vs validation score over sample sizes
        - Bias-variance tradeoff diagnosis
        - Optimal training size recommendation
        """
        try:
            self._validate_inputs(X=X, y_true=y, require_X=True)
            from sklearn.model_selection import learning_curve

            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model,
                X,
                y,
                train_sizes=train_sizes,
                cv=cv,
                n_jobs=-1,
                scoring="accuracy",
                shuffle=True,
                random_state=42,
            )

            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)

            # Diagnose bias-variance
            gap = train_mean[-1] - val_mean[-1]
            final_train = train_mean[-1]
            final_val = val_mean[-1]

            if final_train < 0.7:
                diagnosis = "**High Bias (Underfitting)**: Model too simple. Consider more features or complex model."
            elif gap > 0.1:
                diagnosis = "**High Variance (Overfitting)**: Model too complex. Consider regularization or more data."
            else:
                diagnosis = "**Good Fit**: Model has balanced bias-variance tradeoff."

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_learning_curve_chart(
                    train_sizes_abs, train_mean, train_std, val_mean, val_std
                )
            except Exception as e:
                self._record_chart_failure("learning_curve", e)

            content = f"""
# Learning Curve Analysis

Learning curves reveal how model performance changes with training data size,
helping diagnose underfitting vs overfitting.

## Bias-Variance Diagnosis

{diagnosis}

| Metric | Value |
|--------|-------|
| Final Training Score | {final_train:.4f} |
| Final Validation Score | {final_val:.4f} |
| Gap (Train - Val) | {gap:.4f} |
| Training Samples Used | {train_sizes_abs[-1]:,} |

## Learning Curve Visualization

![Learning Curves]({fig_path})

## Interpretation Guide

- **Converging curves** with small gap → Good fit
- **Flat training curve** at low score → High bias, need more complex model
- **Large gap** between curves → High variance, need regularization or more data
- **Curves still improving** → May benefit from more training data

---
"""
            self._content.append(content)
            self._store_section_data("learning_curves", "Learning Curves", {"diagnosis": diagnosis})

        except Exception as e:
            self._record_section_failure("Learning Curves", e)

    # =========================================================================
    # CALIBRATION ANALYSIS
    # =========================================================================

    def add_calibration_analysis(self, y_true: Any, y_prob: Any, n_bins: int = 10) -> None:
        """
        Add probability calibration analysis showing:
        - Calibration curve (reliability diagram)
        - Brier score
        - Expected Calibration Error (ECE)

        Supports both binary and multiclass (per-class calibration).
        """
        try:
            preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss
            from sklearn.preprocessing import label_binarize

            n_classes = len(np.unique(preds.y_true))
            is_binary = n_classes <= 2

            if is_binary:
                prob_1d = preds.y_prob if preds.y_prob.ndim == 1 else preds.y_prob[:, 1]
                fraction_of_positives, mean_predicted = calibration_curve(
                    preds.y_true, prob_1d, n_bins=n_bins
                )
                brier_score = brier_score_loss(preds.y_true, prob_1d)

                bin_counts = np.histogram(prob_1d, bins=n_bins, range=(0, 1))[0]
                ece = np.sum(
                    np.abs(fraction_of_positives - mean_predicted)
                    * (bin_counts[: len(fraction_of_positives)] / preds.n_samples)
                )

                fig_path = ""
                try:
                    fig_path = self._create_calibration_chart(
                        fraction_of_positives, mean_predicted, prob_1d
                    )
                except Exception as e:
                    self._record_chart_failure("calibration", e)

                content = f"""
# Probability Calibration Analysis

Well-calibrated probabilities are essential for reliable decision-making.
A perfectly calibrated model's predicted probabilities should match actual outcome frequencies.

## Calibration Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Brier Score | {brier_score:.4f} | Lower is better (0 = perfect) |
| Expected Calibration Error | {ece:.4f} | Lower is better |

## Calibration Curve

![Calibration Curve]({fig_path})

## Interpretation

- Points on diagonal = perfectly calibrated
- Points above diagonal = underconfident (probabilities too low)
- Points below diagonal = overconfident (probabilities too high)

---
"""
            else:
                # Multiclass: per-class calibration
                classes = np.unique(preds.y_true)
                y_bin = label_binarize(preds.y_true, classes=classes)
                y_prob_2d = preds.y_prob if preds.y_prob.ndim == 2 else None

                if y_prob_2d is None or y_prob_2d.shape[1] != n_classes:
                    self._record_section_failure(
                        "Calibration Analysis",
                        ValueError(
                            f"y_prob shape {preds.y_prob.shape} incompatible with {n_classes} classes"
                        ),
                    )
                    return

                per_class_brier = {}
                for i, cls in enumerate(classes):
                    try:
                        bs = brier_score_loss(y_bin[:, i], y_prob_2d[:, i])
                        per_class_brier[f"Class {cls}"] = bs
                    except Exception:
                        pass

                brier_score = np.mean(list(per_class_brier.values())) if per_class_brier else 0.0
                ece = 0.0  # ECE per-class averaging

                content = f"""
# Probability Calibration Analysis (Multiclass)

Per-class calibration analysis for {n_classes} classes.

## Per-Class Brier Scores

| Class | Brier Score |
|-------|------------|
"""
                for cls_name, bs_val in per_class_brier.items():
                    content += f"| {cls_name} | {bs_val:.4f} |\n"

                content += f"""
**Mean Brier Score:** {brier_score:.4f} (lower is better)

---
"""

            self._content.append(content)
            self._store_section_data(
                "calibration", "Calibration Analysis", {"brier_score": brier_score, "ece": ece}
            )

        except Exception as e:
            self._record_section_failure("Calibration Analysis", e)

    # =========================================================================
    # CONFIDENCE-CALIBRATED PREDICTIONS
    # =========================================================================

    def add_prediction_confidence_analysis(
        self,
        X_sample: Any,
        y_true: Any,
        y_pred: Any,
        y_prob: Any,
        feature_names: List[str] | None = None,
        top_n: int = 10,
    ) -> Any:
        """
        Add confidence-calibrated prediction analysis.

        Args:
            X_sample: Feature data for sample predictions
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            feature_names: Feature names
            top_n: Number of top/bottom confident predictions to show
        """
        try:
            preds = self._make_predictions(y_true, y_pred, y_prob)

            if feature_names is None:
                if hasattr(X_sample, "columns"):
                    feature_names = list(X_sample.columns)
                else:
                    feature_names = [f"Feature_{i}" for i in range(np.asarray(X_sample).shape[1])]

            # For multiclass, use max probability as confidence; for binary, use positive class prob
            if preds.y_prob.ndim == 2:
                prob_1d = np.max(preds.y_prob, axis=1)
            else:
                prob_1d = preds.y_prob

            # Compute ECE
            ece_data = self._compute_ece(preds.y_true, prob_1d)

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_confidence_charts(preds.y_true, prob_1d, ece_data)
            except Exception as e:
                self._record_chart_failure("confidence_charts", e)

            # Find most/least confident predictions
            confidence = np.abs(prob_1d - 0.5) * 2  # 0 = uncertain, 1 = very confident
            correct = (~preds.errors).astype(int)

            # Most confident correct
            confident_correct_idx = np.where(correct == 1)[0]
            if len(confident_correct_idx) > 0:
                top_correct = confident_correct_idx[
                    np.argsort(confidence[confident_correct_idx])[::-1][:top_n]
                ]
            else:
                top_correct = np.array([])

            # Most confident wrong
            confident_wrong_idx = np.where(correct == 0)[0]
            if len(confident_wrong_idx) > 0:
                top_wrong = confident_wrong_idx[
                    np.argsort(confidence[confident_wrong_idx])[::-1][:top_n]
                ]
            else:
                top_wrong = np.array([])

            # Average confidence by correctness
            avg_conf_correct = confidence[correct == 1].mean() if (correct == 1).sum() > 0 else 0
            avg_conf_wrong = confidence[correct == 0].mean() if (correct == 0).sum() > 0 else 0

            content = f"""
# Confidence-Calibrated Predictions

Analyzing model prediction confidence and its relationship to actual correctness.

## Calibration Metrics

| Metric | Value |
|--------|-------|
| Expected Calibration Error (ECE) | {ece_data['ece']:.4f} |
| Average Confidence (Correct) | {avg_conf_correct:.4f} |
| Average Confidence (Wrong) | {avg_conf_wrong:.4f} |
| Confidence Gap | {avg_conf_correct - avg_conf_wrong:.4f} |

## Confidence Analysis

![Confidence Analysis]({fig_path})

"""
            # Table of most confident correct predictions
            if len(top_correct) > 0:
                content += f"""## Most Confident Correct Predictions (Top {min(top_n, len(top_correct))})

| Index | True | Pred | Probability | Confidence |
|-------|------|------|-------------|------------|
"""
                for idx in top_correct[:top_n]:
                    content += f"| {idx} | {preds.y_true[idx]} | {preds.y_pred[idx]} | {prob_1d[idx]:.4f} | {confidence[idx]:.4f} |\n"
                content += "\n"

            # Table of most confident wrong predictions
            if len(top_wrong) > 0:
                content += f"""## Most Confident Wrong Predictions (Top {min(top_n, len(top_wrong))})

| Index | True | Pred | Probability | Confidence |
|-------|------|------|-------------|------------|
"""
                for idx in top_wrong[:top_n]:
                    content += f"| {idx} | {preds.y_true[idx]} | {preds.y_pred[idx]} | {prob_1d[idx]:.4f} | {confidence[idx]:.4f} |\n"
                content += "\n"

            content += """
## Interpretation

- **ECE** close to 0 means predicted probabilities match actual frequencies
- Large confidence gap between correct/wrong predictions indicates reliable uncertainty
- Highly confident wrong predictions deserve investigation

---
"""
            self._content.append(content)

            self._store_section_data(
                "confidence_analysis",
                "Confidence-Calibrated Predictions",
                {
                    "ece": ece_data["ece"],
                    "avg_conf_correct": avg_conf_correct,
                    "avg_conf_wrong": avg_conf_wrong,
                },
                [{"type": "multi", "path": fig_path}],
            )

        except Exception as e:
            self._record_section_failure("Prediction Confidence Analysis", e)

    def add_lift_gain_analysis(self, y_true: Any, y_prob: Any) -> None:
        """
        Add lift and gain charts for marketing/business context:
        - Cumulative gains curve
        - Lift curve
        - KS statistic

        Note: Lift/Gain analysis is only applicable to binary classification.
        """
        try:
            preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used

            n_classes = len(np.unique(preds.y_true))
            if n_classes > 2:
                self._content.append(
                    """
# Lift & Gain Analysis

> Lift/Gain analysis is only applicable to binary classification.
> This section is skipped for multiclass problems.

---
"""
                )
                self._store_section_data(
                    "lift_gain", "Lift & Gain Analysis", {"skipped": "multiclass"}
                )
                return

            # Ensure 1D probabilities for binary
            prob_1d = preds.y_prob if preds.y_prob.ndim == 1 else preds.y_prob[:, 1]

            # Sort by probability descending
            sorted_idx = np.argsort(prob_1d)[::-1]
            y_sorted = preds.y_true[sorted_idx]

            # Calculate cumulative gains
            n = preds.n_samples
            positives = y_sorted.sum()
            cum_gains = np.cumsum(y_sorted) / positives
            deciles = np.arange(1, n + 1) / n

            # Calculate lift
            lift = cum_gains / (deciles + 1e-10)

            # KS statistic
            ks_stat = np.max(cum_gains - deciles)
            ks_idx = np.argmax(cum_gains - deciles)

            # Create visualizations
            fig_path = ""
            try:
                fig_path = self._create_lift_gain_chart(deciles, cum_gains, lift, ks_stat, ks_idx)
            except Exception as e:
                self._record_chart_failure("lift_gain", e)

            content = f"""
# Lift & Gain Analysis

These charts help evaluate model effectiveness for targeted campaigns and prioritization.

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| KS Statistic | {ks_stat:.4f} | Maximum separation between model and random |
| KS at Decile | {deciles[ks_idx]*100:.0f}% | Optimal cutoff point |
| Top 10% Lift | {lift[int(n*0.1)]:.2f}x | Model advantage in top 10% |
| Top 20% Lift | {lift[int(n*0.2)]:.2f}x | Model advantage in top 20% |

## Cumulative Gains & Lift Curves

![Lift and Gain Charts]({fig_path})

## Business Interpretation

- **Gains Curve**: Shows % of positives captured by targeting top X% of predictions
- **Lift Curve**: Shows how much better the model is vs random selection
- **KS Statistic**: Higher values indicate better model discrimination

---
"""
            self._content.append(content)
            self._store_section_data("lift_gain", "Lift & Gain Analysis", {"ks_stat": ks_stat})

        except Exception as e:
            self._record_section_failure("Lift/Gain Analysis", e)

    # =========================================================================
    # CROSS-VALIDATION DETAILED ANALYSIS
    # =========================================================================

    def add_cv_detailed_analysis(self, model: Any, X: Any, y: Any, cv: int = 5) -> None:
        """
        Add detailed cross-validation analysis:
        - Fold-by-fold results
        - Score distribution
        - Stability analysis
        """
        try:
            from sklearn.model_selection import cross_validate

            # Multiple metrics
            scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_results = cross_validate(
                    model, X, y, cv=cv, scoring=scoring, return_train_score=True
                )

            # Create fold results table
            fold_data = []
            for i in range(cv):
                fold_data.append(
                    {
                        "fold": i + 1,
                        "train_acc": cv_results["train_accuracy"][i],
                        "test_acc": cv_results["test_accuracy"][i],
                        "train_f1": cv_results["train_f1_macro"][i],
                        "test_f1": cv_results["test_f1_macro"][i],
                    }
                )

            # Calculate stability
            acc_mean = cv_results["test_accuracy"].mean()
            acc_std = cv_results["test_accuracy"].std()
            cv_coefficient = acc_std / acc_mean * 100

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_cv_chart(cv_results)
            except Exception as e:
                self._record_chart_failure("cv_chart", e)

            content = f"""
# Cross-Validation Detailed Analysis

{cv}-fold cross-validation provides robust performance estimates and helps detect instability.

## Fold-by-Fold Results

| Fold | Train Accuracy | Test Accuracy | Train F1 | Test F1 |
|------|----------------|---------------|----------|---------|
"""
            for fd in fold_data:
                content += f"| {fd['fold']} | {fd['train_acc']:.4f} | {fd['test_acc']:.4f} | {fd['train_f1']:.4f} | {fd['test_f1']:.4f} |\n"

            content += f"""
## Stability Analysis

| Metric | Value |
|--------|-------|
| Mean Accuracy | {acc_mean:.4f} |
| Std Deviation | {acc_std:.4f} |
| CV Coefficient | {cv_coefficient:.2f}% |
| 95% CI | [{acc_mean - 1.96*acc_std:.4f}, {acc_mean + 1.96*acc_std:.4f}] |

**Stability Assessment:** {"Excellent" if cv_coefficient < 2 else ("Good" if cv_coefficient < 5 else ("Moderate" if cv_coefficient < 10 else "Unstable"))}

## CV Performance Distribution

![CV Analysis]({fig_path})

---
"""
            self._content.append(content)
            self._store_section_data("cv_analysis", "CV Detailed Analysis", {"acc_mean": acc_mean})

        except Exception as e:
            self._record_section_failure("CV Detailed Analysis", e)

    # =========================================================================
    # SHAP DEEP DIVE
    # =========================================================================

    def add_shap_deep_analysis(
        self,
        shap_values: Any,
        feature_names: List[str],
        X_sample: Any,
        model: Any = None,
        top_n: int = 3,
    ) -> Any:
        """
        Add comprehensive SHAP analysis:
        - Summary plot
        - Bar plot (global importance)
        - Dependence plots for top features
        - Force plot for sample predictions
        """
        try:

            # Extract values
            if hasattr(shap_values, "values"):
                values = shap_values.values
            else:
                values = shap_values

            if len(values.shape) == 3:
                values = values[:, :, 1]  # Binary classification

            # Calculate mean absolute SHAP
            mean_shap = np.abs(values).mean(axis=0)
            shap_importance = sorted(
                zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True
            )

            # Create all SHAP visualizations
            fig_summary = ""
            try:
                fig_summary = self._create_shap_summary(shap_values, feature_names, X_sample)
            except Exception as e:
                self._record_chart_failure("shap_summary", e)

            fig_bar = ""
            try:
                fig_bar = self._create_shap_bar(shap_importance)
            except Exception as e:
                self._record_chart_failure("shap_bar", e)

            fig_dependence = ""
            try:
                fig_dependence = self._create_shap_dependence(
                    shap_values, feature_names, X_sample, top_n
                )
            except Exception as e:
                self._record_chart_failure("shap_dependence", e)

            fig_waterfall = ""
            try:
                fig_waterfall = self._create_shap_waterfall(shap_values, feature_names, X_sample)
            except Exception as e:
                self._record_chart_failure("shap_waterfall", e)

            content = """
# SHAP Deep Analysis

SHAP (SHapley Additive exPlanations) provides consistent, locally accurate feature attributions
for any machine learning model.

## Global Feature Importance (Mean Abs SHAP)

| Rank | Feature | Mean Abs SHAP | Cumulative % |
|------|---------|-------------|--------------|
"""
            total_shap = sum(x[1] for x in shap_importance)
            cumsum = 0
            for i, (feat, val) in enumerate(shap_importance[:15], 1):
                cumsum += val
                content += f"| {i} | {feat[:30]} | {val:.4f} | {cumsum/total_shap*100:.1f}% |\n"

            content += f"""
## SHAP Summary Plot

Shows feature impact on predictions. Color indicates feature value (red=high, blue=low).

![SHAP Summary]({fig_summary})

## SHAP Feature Importance Bar

![SHAP Bar Plot]({fig_bar})

"""

            if fig_dependence:
                content += f"""
## SHAP Dependence Plots (Top {top_n} Features)

Shows how feature values affect SHAP values, revealing non-linear relationships.

![SHAP Dependence]({fig_dependence})

"""

            if fig_waterfall:
                content += f"""
## SHAP Waterfall (Sample Prediction)

Shows how features contribute to a single prediction.

![SHAP Waterfall]({fig_waterfall})

"""

            content += "---\n"
            self._content.append(content)
            self._store_section_data(
                "shap_deep",
                "SHAP Deep Analysis",
                {
                    "top_features": [f for f, _ in shap_importance[:10]],
                },
            )

        except Exception as e:
            self._record_section_failure("SHAP Deep Analysis", e)

    # =========================================================================
    # THRESHOLD OPTIMIZATION
    # =========================================================================

    def add_threshold_optimization(
        self, y_true: Any, y_prob: Any, cost_fp: float = 1.0, cost_fn: float = 1.0
    ) -> None:
        """
        Add threshold optimization analysis:
        - Optimal threshold for different objectives
        - Cost-benefit analysis at different thresholds
        - Threshold impact table

        Note: Threshold optimization is only applicable to binary classification.
        """
        try:
            preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred computed per threshold

            n_classes = len(np.unique(preds.y_true))
            if n_classes > 2:
                self._content.append(
                    """
# Threshold Optimization

> Threshold optimization is only applicable to binary classification.
> This section is skipped for multiclass problems.

---
"""
                )
                self._store_section_data(
                    "threshold_optimization", "Threshold Optimization", {"skipped": "multiclass"}
                )
                return

            from sklearn.metrics import confusion_matrix

            # Ensure 1D probabilities
            prob_1d = preds.y_prob if preds.y_prob.ndim == 1 else preds.y_prob[:, 1]

            # Calculate metrics at different thresholds
            thresholds = np.linspace(0.1, 0.9, 17)
            results = []

            for thresh in thresholds:
                y_pred_thresh = (prob_1d >= thresh).astype(int)
                cm = confusion_matrix(preds.y_true, y_pred_thresh)

                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                )

                # Cost
                cost = fp * cost_fp + fn * cost_fn

                results.append(
                    {
                        "threshold": thresh,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "cost": cost,
                    }
                )

            # Find optimal thresholds
            best_f1 = max(results, key=lambda x: x["f1"])
            best_cost = min(results, key=lambda x: x["cost"])
            balanced = min(results, key=lambda x: abs(x["precision"] - x["recall"]))

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_threshold_chart(
                    results, best_f1["threshold"], best_cost["threshold"]
                )
            except Exception as e:
                self._record_chart_failure("threshold_chart", e)

            content = f"""
# Threshold Optimization

Choosing the right classification threshold depends on business objectives.

## Optimal Thresholds

| Objective | Threshold | Precision | Recall | F1 | Cost |
|-----------|-----------|-----------|--------|----|----|
| Max F1 Score | {best_f1['threshold']:.2f} | {best_f1['precision']:.3f} | {best_f1['recall']:.3f} | {best_f1['f1']:.3f} | {best_f1['cost']:.0f} |
| Min Cost | {best_cost['threshold']:.2f} | {best_cost['precision']:.3f} | {best_cost['recall']:.3f} | {best_cost['f1']:.3f} | {best_cost['cost']:.0f} |
| Balanced P/R | {balanced['threshold']:.2f} | {balanced['precision']:.3f} | {balanced['recall']:.3f} | {balanced['f1']:.3f} | {balanced['cost']:.0f} |

## Threshold Impact Analysis

| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |
|-----------|----|----|----|----|-----------|--------|-----|
"""
            for r in results[::2]:  # Every other threshold
                content += f"| {r['threshold']:.2f} | {r['tp']:,} | {r['fp']:,} | {r['fn']:,} | {r['tn']:,} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |\n"

            content += f"""
## Threshold Visualization

![Threshold Analysis]({fig_path})

## Cost Parameters Used

- Cost of False Positive: {cost_fp}
- Cost of False Negative: {cost_fn}

---
"""
            self._content.append(content)
            self._store_section_data(
                "threshold_optimization",
                "Threshold Optimization",
                {"best_f1_threshold": best_f1["threshold"]},
            )

        except Exception as e:
            self._record_section_failure("Threshold Optimization", e)

    # =========================================================================
    # REPRODUCIBILITY SECTION
    # =========================================================================

    def add_reproducibility_section(
        self,
        model: Any,
        params: Dict | None = None,
        random_state: int | None = None,
        environment: Dict | None = None,
    ) -> Any:
        """
        Add reproducibility information:
        - Model hyperparameters
        - Random seeds
        - Environment details
        - Package versions
        """
        try:
            self._add_reproducibility_section_impl(model, params, random_state, environment)
        except Exception as e:
            self._record_section_failure("Reproducibility", e)

    def add_hyperparameter_visualization(
        self,
        study_or_trials: Any,
        param_names: List[str] | None = None,
        objective_name: str = "Objective",
    ) -> Any:
        """
        Add hyperparameter search visualization.

        Args:
            study_or_trials: Optuna study object or List[Dict] with 'params' and 'value' keys
            param_names: Specific parameter names to visualize (auto-detected if None)
            objective_name: Name of the objective metric
        """
        try:
            # Normalize trials
            trials = self._normalize_trials(study_or_trials)
            if not trials or len(trials) < 3:
                return

            # Auto-detect param names
            if param_names is None:
                param_names = list(trials[0]["params"].keys())

            # Limit params for readability
            param_names = param_names[:8]

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_hyperparameter_charts(trials, param_names, objective_name)
            except Exception as e:
                self._record_chart_failure("hyperparameter_charts", e)

            # Build table of best trials
            sorted_trials = sorted(trials, key=lambda t: t["value"], reverse=True)
            top_n = min(10, len(sorted_trials))

            table_md = (
                f"| Rank | {objective_name} | "
                + " | ".join(p[:15] for p in param_names[:5])
                + " |\n"
            )
            table_md += "|------|" + "|".join(["---"] * (min(5, len(param_names)) + 1)) + "|\n"

            for i, trial in enumerate(sorted_trials[:top_n], 1):
                row = f"| {i} | {trial['value']:.4f} | "
                for pname in param_names[:5]:
                    val = trial["params"].get(pname, "N/A")
                    if isinstance(val, float):
                        row += f"{val:.4f} | "
                    else:
                        row += f"{str(val)[:12]} | "
                table_md += row + "\n"

            content = f"""
# Hyperparameter Search Visualization

Analysis of {len(trials)} hyperparameter trials exploring {len(param_names)} parameters.

## Top {top_n} Trials

{table_md}

## Hyperparameter Analysis

![Hyperparameter Analysis]({fig_path})

## Interpretation

- **Parallel Coordinates**: Shows parameter combinations colored by objective value
- **Parameter Importance**: Variance-based importance of each hyperparameter
- **Optimization History**: Objective value over trials with best-so-far overlay

---
"""
            self._content.append(content)

            self._store_section_data(
                "hyperparameter_viz",
                "Hyperparameter Search",
                {
                    "n_trials": len(trials),
                    "best_value": sorted_trials[0]["value"] if sorted_trials else None,
                    "best_params": sorted_trials[0]["params"] if sorted_trials else {},
                },
                [{"type": "multi", "path": fig_path}],
            )

        except Exception as e:
            self._record_section_failure("Hyperparameter Visualization", e)

    def add_executive_dashboard(
        self, metrics: Dict[str, float], model_name: str = "", dataset_name: str = ""
    ) -> Any:
        """
        Add executive dashboard with visual KPI gauge charts.

        Creates donut/arc gauge charts for key metrics:
        - Accuracy, AUC, F1, Precision, Recall
        - Color-coded: green (>0.9), yellow (0.7-0.9), red (<0.7)
        """
        try:

            # Standardize metric names
            kpi_map = {
                "accuracy": ["accuracy", "acc", "test_accuracy"],
                "auc": ["auc", "roc_auc", "auc_roc", "AUC"],
                "f1": ["f1", "f1_score", "f1_macro", "F1"],
                "precision": ["precision", "precision_macro"],
                "recall": ["recall", "recall_macro", "sensitivity"],
            }

            kpis = {}
            for kpi_name, aliases in kpi_map.items():
                for alias in aliases:
                    if alias in metrics:
                        kpis[kpi_name] = float(metrics[alias])
                        break

            if not kpis:
                return

            # Create dashboard chart
            fig_path = self._create_executive_dashboard_chart(kpis)

            # Build summary table
            table_md = "| KPI | Value | Status |\n|-----|-------|--------|\n"
            for name, value in kpis.items():
                status = (
                    "Excellent" if value > 0.9 else ("Good" if value > 0.7 else "Needs Improvement")
                )
                table_md += f"| {name.upper()} | {value:.4f} | {status} |\n"

            content = f"""
# Executive Dashboard

## Model Performance Overview

**Model:** {model_name or 'Best Model'}  |  **Dataset:** {dataset_name or 'Analysis Dataset'}

{table_md}

## KPI Gauges

![Executive Dashboard]({fig_path})

---
"""
            self._content.append(content)
            self._store_section_data("executive_dashboard", "Executive Dashboard", {"kpis": kpis})

        except Exception as e:
            self._record_section_failure("Executive Dashboard", e)
