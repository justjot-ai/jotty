"""
Analysis sections mixin for ProfessionalMLReport.

Model Performance & Specialized Analysis Mixin

Contains methods for:
- Insight prioritization, executive dashboard
- Model comparison, benchmarking, CV analysis
- Performance metrics (confusion matrix, ROC, PR, calibration)
- Advanced analysis (threshold optimization, confidence, lift/gain)
- Statistical tests, regression, deep learning, model card

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


class ModelPerformanceMixin:
    """Mixin providing model performance analysis methods for ProfessionalMLReport.

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

    def add_insight_prioritization(self) -> None:
        """
        Scan all previously stored section data for concerning patterns and
        generate a prioritized findings table.

        Severity levels:
        - CRITICAL: drift PSI > 0.25, fairness DI < 0.8
        - HIGH: ECE > 0.1, accuracy < 0.7, AUC not significant
        - MEDIUM: feature concentration (top-1 > 50%), class imbalance > 3:1
        """
        try:
            findings = []
            section_lookup = {s["type"]: s["data"] for s in self._section_data}

            # CRITICAL checks
            drift_data = section_lookup.get("drift_analysis", {})
            drift_results = drift_data.get("drift_results", [])
            for dr in drift_results:
                if isinstance(dr, dict) and dr.get("psi", 0) > 0.25:
                    findings.append(
                        {
                            "severity": "CRITICAL",
                            "source": "Drift Analysis",
                            "description": f"Feature '{dr.get('feature', '?')}' has PSI={dr.get('psi', 0):.3f} (>0.25)",
                            "action": "Retrain model with recent data or investigate distribution shift",
                        }
                    )

            fairness_data = section_lookup.get("fairness_audit", {})
            for feat_name, groups in fairness_data.get("metrics", {}).items():
                if isinstance(groups, dict):
                    for group_name, m in groups.items():
                        if isinstance(m, dict) and m.get("disparate_impact", 1.0) < 0.8:
                            findings.append(
                                {
                                    "severity": "CRITICAL",
                                    "source": "Fairness Audit",
                                    "description": f"Group '{group_name}' in '{feat_name}' has DI={m['disparate_impact']:.3f} (<0.8)",
                                    "action": "Apply bias mitigation (reweighting, threshold adjustment)",
                                }
                            )

            # HIGH checks
            confidence_data = section_lookup.get("confidence_analysis", {})
            if confidence_data.get("ece", 0) > 0.1:
                findings.append(
                    {
                        "severity": "HIGH",
                        "source": "Confidence Analysis",
                        "description": f"Expected Calibration Error = {confidence_data['ece']:.3f} (>0.1)",
                        "action": "Apply calibration (Platt scaling or isotonic regression)",
                    }
                )

            exec_data = section_lookup.get("executive_summary", {})
            acc = exec_data.get("accuracy", exec_data.get("acc", None))
            if acc is not None and acc < 0.7:
                findings.append(
                    {
                        "severity": "HIGH",
                        "source": "Executive Summary",
                        "description": f"Accuracy = {acc:.3f} (<0.70 threshold)",
                        "action": "Consider more powerful models, feature engineering, or more data",
                    }
                )

            stat_data = section_lookup.get("statistical_tests", {})
            if stat_data.get("significant") is False:
                findings.append(
                    {
                        "severity": "HIGH",
                        "source": "Statistical Tests",
                        "description": "AUC confidence interval includes 0.5 — not statistically significant",
                        "action": "Collect more data or improve feature quality",
                    }
                )

            # MEDIUM checks
            importance_data = section_lookup.get("feature_importance", {})
            importance = importance_data.get("importance", {})
            if importance:
                sorted_vals = sorted(importance.values(), reverse=True)
                total = sum(sorted_vals)
                if total > 0 and sorted_vals[0] / total > 0.5:
                    findings.append(
                        {
                            "severity": "MEDIUM",
                            "source": "Feature Importance",
                            "description": f"Top feature accounts for {sorted_vals[0]/total*100:.0f}% of total importance",
                            "action": "Investigate feature reliability and add complementary features",
                        }
                    )

            class_data = section_lookup.get("class_distribution", {})
            class_counts = class_data.get("counts", {})
            if class_counts and isinstance(class_counts, dict):
                count_vals = list(class_counts.values())
                if len(count_vals) >= 2 and min(count_vals) > 0:
                    ratio = max(count_vals) / min(count_vals)
                    if ratio > 3:
                        findings.append(
                            {
                                "severity": "MEDIUM",
                                "source": "Class Distribution",
                                "description": f"Class imbalance ratio = {ratio:.1f}:1 (>3:1)",
                                "action": "Apply SMOTE, class weights, or threshold tuning",
                            }
                        )

            # Model benchmarking: CV-test gap (overfitting)
            bench_data = section_lookup.get("model_benchmarking", {})
            model_scores = bench_data.get("model_scores", {})
            for model_name, scores in model_scores.items():
                if isinstance(scores, dict):
                    cv = scores.get("cv_score", 0)
                    test = scores.get("test_score", 0)
                    if cv > 0 and test > 0 and (cv - test) > 0.1:
                        findings.append(
                            {
                                "severity": "HIGH",
                                "source": "Model Benchmarking",
                                "description": f"Model '{model_name}' CV-test gap = {cv - test:.3f} (>0.1, overfitting)",
                                "action": "Regularize model, reduce complexity, or gather more training data",
                            }
                        )

            # Deployment readiness: latency check
            deploy_data = section_lookup.get("deployment_readiness", {})
            checklist = deploy_data.get("checklist", {})
            if checklist.get("latency_ok") is False:
                findings.append(
                    {
                        "severity": "HIGH",
                        "source": "Deployment Readiness",
                        "description": "Model inference latency exceeds acceptable threshold",
                        "action": "Optimize model (pruning, quantization) or use faster hardware",
                    }
                )

            # Error analysis: dominant error cluster
            error_data = section_lookup.get("error_analysis", {})
            error_clusters = error_data.get("clusters", [])
            for cluster in error_clusters:
                if isinstance(cluster, dict) and cluster.get("percentage", 0) > 30:
                    findings.append(
                        {
                            "severity": "HIGH",
                            "source": "Error Analysis",
                            "description": f"Error cluster '{cluster.get('name', '?')}' contains {cluster.get('percentage', 0):.0f}% of all errors",
                            "action": "Investigate and address this systematic failure mode",
                        }
                    )

            # Deployment readiness: model size check
            if checklist.get("size_ok") is False:
                findings.append(
                    {
                        "severity": "MEDIUM",
                        "source": "Deployment Readiness",
                        "description": "Model size exceeds deployment size limit",
                        "action": "Compress model (distillation, pruning) or increase size budget",
                    }
                )

            # Regression: low R²
            reg_data = section_lookup.get("regression", {})
            r2_val = reg_data.get("r2")
            if r2_val is not None and r2_val < 0.5:
                findings.append(
                    {
                        "severity": "MEDIUM",
                        "source": "Regression Analysis",
                        "description": f"R² = {r2_val:.3f} (<0.5, poor explanatory power)",
                        "action": "Add features, try non-linear models, or investigate data quality",
                    }
                )

            # Regression: heteroscedasticity
            if reg_data.get("is_heteroscedastic") is True:
                findings.append(
                    {
                        "severity": "MEDIUM",
                        "source": "Regression Analysis",
                        "description": "Heteroscedasticity detected in residuals",
                        "action": "Use weighted regression or variance-stabilizing transformations",
                    }
                )

            # Correlation: near-perfect collinearity
            corr_data = section_lookup.get("correlation", {})
            high_pairs = corr_data.get("high_corr_pairs", [])
            for pair in high_pairs:
                if isinstance(pair, dict) and abs(pair.get("corr", 0)) > 0.95:
                    findings.append(
                        {
                            "severity": "MEDIUM",
                            "source": "Correlation Analysis",
                            "description": f"Features '{pair.get('f1', '?')}' and '{pair.get('f2', '?')}' have |r|={abs(pair['corr']):.3f} (>0.95)",
                            "action": "Remove one feature or use PCA to reduce collinearity",
                        }
                    )

            # Sort by severity
            severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
            findings.sort(key=lambda f: severity_order.get(f["severity"], 3))

            n_critical = sum(1 for f in findings if f["severity"] == "CRITICAL")
            n_high = sum(1 for f in findings if f["severity"] == "HIGH")
            n_medium = sum(1 for f in findings if f["severity"] == "MEDIUM")

            # Build content
            content = f"""
# Insight Prioritization

Automated scan of all analysis sections for actionable findings.

**Summary:** {n_critical} Critical, {n_high} High, {n_medium} Medium findings

## Prioritized Findings

| # | Severity | Source | Finding | Recommended Action |
|---|----------|--------|---------|-------------------|
"""
            if findings:
                for i, f in enumerate(findings, 1):
                    content += f"| {i} | **{f['severity']}** | {f['source']} | {f['description']} | {f['action']} |\n"
            else:
                content += "| - | - | - | No concerning patterns detected | Continue monitoring |\n"

            content += """
## Severity Definitions

- **CRITICAL**: Immediate action required — model may be unreliable or biased
- **HIGH**: Should be addressed before production deployment
- **MEDIUM**: Recommended improvements for model robustness

---
"""
            # Insert after Executive Summary (position 1 in content list)
            if len(self._content) > 1:
                self._content.insert(1, content)
            else:
                self._content.append(content)

            self._store_section_data(
                "insight_prioritization",
                "Insight Prioritization",
                {
                    "n_critical": n_critical,
                    "n_high": n_high,
                    "n_medium": n_medium,
                    "findings": findings,
                },
            )

        except Exception as e:
            self._record_section_failure("Insight Prioritization", e)

    # =========================================================================
    # CLASS DISTRIBUTION ANALYSIS (Phase 2)
    # =========================================================================

    def add_class_distribution(
        self, y_true: Any, y_pred: Any = None, labels: List[str] | None = None
    ) -> None:
        """
        Add class distribution analysis with:
        - Class balance bar chart
        - MCC, balanced accuracy, Cohen's kappa
        - Resampling suggestions if imbalanced (ratio > 3:1)
        """
        try:
            from sklearn.metrics import (
                balanced_accuracy_score,
                cohen_kappa_score,
                matthews_corrcoef,
            )

            y_true_arr = np.asarray(y_true)
            unique_classes, class_counts = np.unique(y_true_arr, return_counts=True)

            if labels is None:
                labels = [f"Class {c}" for c in unique_classes]

            # Calculate class ratios
            max_count = class_counts.max()
            min_count = class_counts.min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

            # Calculate metrics if predictions available
            metrics_md = ""
            if y_pred is not None:
                preds = self._make_predictions(y_true, y_pred)
                mcc = matthews_corrcoef(preds.y_true, preds.y_pred)
                balanced_acc = balanced_accuracy_score(preds.y_true, preds.y_pred)
                kappa = cohen_kappa_score(preds.y_true, preds.y_pred)

                metrics_md = f"""
## Class-Aware Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Matthews Correlation Coefficient | {mcc:.4f} | -1 to 1, 0 = random |
| Balanced Accuracy | {balanced_acc:.4f} | Accuracy adjusted for class imbalance |
| Cohen's Kappa | {kappa:.4f} | Agreement beyond chance |

"""

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_class_distribution_chart(labels, class_counts)
            except Exception as e:
                self._record_chart_failure("class_distribution", e)

            # Build class table
            class_table = (
                "| Class | Count | Percentage | Ratio |\n|-------|-------|------------|-------|\n"
            )
            total = class_counts.sum()
            for label, count in zip(labels, class_counts):
                pct = count / total * 100
                ratio = count / min_count
                class_table += f"| {label} | {count:,} | {pct:.1f}% | {ratio:.1f}:1 |\n"

            # Resampling suggestions
            resampling_md = ""
            if imbalance_ratio > 3:
                resampling_md = f"""
## Resampling Recommendations

**Warning: Class imbalance detected (ratio {imbalance_ratio:.1f}:1)**

The following resampling strategies are recommended:

- **SMOTE** (Synthetic Minority Over-sampling): Generate synthetic minority samples
- **Random Undersampling**: Reduce majority class to match minority
- **ADASYN**: Adaptive synthetic sampling focusing on harder-to-learn samples
- **Class Weights**: Use `class_weight='balanced'` in the model
- **Stratified K-Fold**: Ensure each fold preserves class distribution

"""

            content = f"""
# Class Distribution Analysis

Understanding the class distribution is critical for evaluating model performance
and choosing appropriate evaluation metrics.

## Class Balance

{class_table}

**Imbalance Ratio:** {imbalance_ratio:.1f}:1 {'(Imbalanced)' if imbalance_ratio > 3 else '(Balanced)'}

## Class Distribution Visualization

![Class Distribution]({fig_path})

{metrics_md}{resampling_md}
---
"""
            self._content.append(content)
            self._store_section_data(
                "class_distribution", "Class Distribution", {"imbalance_ratio": imbalance_ratio}
            )

        except Exception as e:
            self._record_section_failure("Class Distribution", e)

    # =========================================================================
    # PERMUTATION FEATURE IMPORTANCE (Phase 3)
    # =========================================================================

    def add_permutation_importance(self, model: Any, X: Any, y: Any, n_repeats: int = 10) -> None:
        """
        Add permutation feature importance analysis with:
        - sklearn.inspection.permutation_importance with error bars
        - Side-by-side comparison with native importance
        - Result stored in self._raw_data for PDP phase
        """
        try:
            from sklearn.inspection import permutation_importance as perm_imp

            # Calculate permutation importance
            result = perm_imp(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)

            # Store for PDP phase
            self._raw_data["permutation_importance"] = result

            feature_names = (
                list(X.columns)
                if hasattr(X, "columns")
                else [f"Feature_{i}" for i in range(X.shape[1])]
            )

            # Sort by importance
            sorted_idx = result.importances_mean.argsort()[::-1]
            top_n = min(20, len(sorted_idx))
            top_idx = sorted_idx[:top_n]

            # Create chart
            fig_path = ""
            try:
                fig_path = self._create_permutation_importance_chart(result, feature_names, top_idx)
            except Exception as e:
                self._record_chart_failure("permutation_importance", e)

            # Build table
            table_md = "| Rank | Feature | Importance (Mean) | Std Dev |\n|------|---------|-------------------|--------|\n"
            for rank, idx in enumerate(top_idx, 1):
                table_md += f"| {rank} | {feature_names[idx][:30]} | {result.importances_mean[idx]:.4f} | ±{result.importances_std[idx]:.4f} |\n"

            content = f"""
# Permutation Feature Importance

Permutation importance measures the decrease in model performance when a feature's
values are randomly shuffled, breaking the relationship with the target.

**Method:** {n_repeats} random permutations per feature

## Top {top_n} Features by Permutation Importance

{table_md}

## Permutation Importance Visualization

![Permutation Importance]({fig_path})

## Interpretation

- Features with high permutation importance are critical for model predictions
- Error bars show variability across permutations (wider = less stable)
- Negative importance suggests the feature may add noise

---
"""
            self._content.append(content)
            self._store_section_data(
                "permutation_importance",
                "Permutation Importance",
                {
                    "top_features": [feature_names[idx] for idx in top_idx],
                    "importance_values": {
                        feature_names[idx]: float(result.importances_mean[idx]) for idx in top_idx
                    },
                },
                [{"type": "importance_bar"}],
            )

        except Exception as e:
            self._record_section_failure("Permutation Importance", e)

    # =========================================================================
    # PARTIAL DEPENDENCE PLOTS (Phase 4)
    # =========================================================================

    def add_partial_dependence(
        self, model: Any, X: Any, feature_names: List[str] | None = None, top_n: int = 3
    ) -> Any:
        """
        Add Partial Dependence Plots (PDP) with ICE lines:
        - sklearn.inspection.partial_dependence
        - ICE (Individual Conditional Expectation) background lines
        - Top N features from permutation importance or native importance
        """
        try:

            if feature_names is None:
                feature_names = (
                    list(X.columns)
                    if hasattr(X, "columns")
                    else [f"Feature_{i}" for i in range(X.shape[1])]
                )

            # Determine top features
            top_features = []
            if "permutation_importance" in self._raw_data:
                perm_result = self._raw_data["permutation_importance"]
                sorted_idx = perm_result.importances_mean.argsort()[::-1][:top_n]
                top_features = [feature_names[i] for i in sorted_idx]
            elif hasattr(model, "feature_importances_"):
                sorted_idx = np.argsort(model.feature_importances_)[::-1][:top_n]
                top_features = [feature_names[i] for i in sorted_idx]
            else:
                top_features = feature_names[:top_n]

            # Create PDP chart
            fig_path = ""
            try:
                fig_path = self._create_pdp_chart(model, X, top_features, feature_names)
            except Exception as e:
                self._record_chart_failure("pdp_chart", e)

            features_list = "\n".join([f"- **{f}**" for f in top_features])

            content = f"""
# Partial Dependence Plots

Partial Dependence Plots show the marginal effect of a feature on the predicted outcome,
averaging over the values of all other features.

## Analyzed Features (Top {top_n})

{features_list}

## PDP with ICE Lines

![Partial Dependence Plots]({fig_path})

## Interpretation

- **Bold line**: Average partial dependence (PDP)
- **Thin lines**: Individual Conditional Expectation (ICE) for sample instances
- **Flat PDP**: Feature has little effect on predictions
- **Steep PDP**: Feature has strong effect on predictions
- **Non-linear PDP**: Complex relationship between feature and target

---
"""
            self._content.append(content)
            self._store_section_data(
                "partial_dependence", "Partial Dependence", {"top_features": top_features}
            )

        except Exception as e:
            self._record_section_failure("Partial Dependence", e)

    # =========================================================================
    # STATISTICAL SIGNIFICANCE TESTING (Phase 5)
    # =========================================================================

    def add_statistical_tests(self, y_true: Any, y_pred: Any, y_prob: Any = None) -> None:
        """
        Add statistical significance testing:
        - Bootstrap CI for AUC (1000 iterations)
        - Histogram of bootstrap AUC distribution with CI bands
        """
        try:
            from sklearn.metrics import accuracy_score

            preds = self._make_predictions(y_true, y_pred, y_prob)

            # Bootstrap accuracy
            n_boot = 1000
            boot_accuracies = []
            n = preds.n_samples

            for _ in range(n_boot):
                idx = np.random.choice(n, n, replace=True)
                boot_accuracies.append(accuracy_score(preds.y_true[idx], preds.y_pred[idx]))

            acc_mean = np.mean(boot_accuracies)
            acc_ci_lower = np.percentile(boot_accuracies, 2.5)
            acc_ci_upper = np.percentile(boot_accuracies, 97.5)

            # Bootstrap AUC if probabilities available
            auc_data = None
            if preds.has_probabilities:
                auc_data = self._bootstrap_auc_ci(preds.y_true, preds.y_prob, n_boot)

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_bootstrap_auc_chart(boot_accuracies, auc_data)
            except Exception as e:
                self._record_chart_failure("bootstrap_auc", e)

            content = f"""
# Statistical Significance Testing

Bootstrap resampling provides robust confidence intervals for model performance metrics.

## Bootstrap Analysis ({n_boot:,} iterations)

### Accuracy
| Statistic | Value |
|-----------|-------|
| Mean Accuracy | {acc_mean:.4f} |
| 95% CI Lower | {acc_ci_lower:.4f} |
| 95% CI Upper | {acc_ci_upper:.4f} |
| CI Width | {acc_ci_upper - acc_ci_lower:.4f} |

"""
            if auc_data:
                content += f"""#### AUC-ROC
| Statistic | Value |
|-----------|-------|
| Mean AUC | {auc_data['mean']:.4f} |
| 95% CI Lower | {auc_data['ci_lower']:.4f} |
| 95% CI Upper | {auc_data['ci_upper']:.4f} |
| CI Width | {auc_data['ci_upper'] - auc_data['ci_lower']:.4f} |
| Standard Error | {auc_data['std']:.4f} |

"""

            content += f"""
## Bootstrap Distribution

![Bootstrap Analysis]({fig_path})

## Interpretation

- Narrow CI = stable, reliable performance estimate
- Wide CI = high variability, need more data
- CI not including 0.5 (AUC) confirms model is better than random

---
"""
            auc_ci = f"{auc_data['ci_lower']:.4f}-{auc_data['ci_upper']:.4f}" if auc_data else None
            significant = auc_data["ci_lower"] > 0.5 if auc_data else None
            self._content.append(content)
            self._store_section_data(
                "statistical_tests",
                "Statistical Significance",
                {
                    "acc_mean": float(acc_mean),
                    "n_bootstraps": n_boot,
                    "auc_ci": auc_ci,
                    "significant": significant,
                },
            )

        except Exception as e:
            self._record_section_failure("Statistical Tests", e)

    def add_score_distribution(
        self, y_true: Any, y_prob: Any, labels: List[str] | None = None
    ) -> None:
        """
        Add predicted probability distribution by actual class:
        - KDE/histogram of predicted probabilities split by actual class
        - Overlap region shading
        - KL divergence
        - Optimal threshold annotation
        """
        try:
            preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used
            from scipy import stats as scipy_stats
            from sklearn.metrics import roc_curve

            unique_classes = np.unique(preds.y_true)
            n_classes = len(unique_classes)
            is_binary = n_classes <= 2
            if labels is None:
                labels = [f"Class {c}" for c in unique_classes]

            # Get 1D probabilities per class
            if is_binary:
                prob_1d = preds.y_prob if preds.y_prob.ndim == 1 else preds.y_prob[:, 1]
            else:
                # For multiclass, use max probability (confidence) per sample
                prob_1d = np.max(preds.y_prob, axis=1) if preds.y_prob.ndim == 2 else preds.y_prob

            # Separate probabilities by class
            class_probs = {}
            for cls, label in zip(unique_classes, labels):
                class_probs[label] = prob_1d[preds.y_true == cls]

            # Calculate KL divergence between class distributions
            kl_div = None
            if is_binary:
                p0 = class_probs[labels[0]]
                p1 = class_probs[labels[1]]

                # Use histogram-based KL divergence
                bins = np.linspace(0, 1, 51)
                hist0, _ = np.histogram(p0, bins=bins, density=True)
                hist1, _ = np.histogram(p1, bins=bins, density=True)

                # Add small epsilon to avoid log(0)
                hist0 = hist0 + 1e-10
                hist1 = hist1 + 1e-10
                hist0 = hist0 / hist0.sum()
                hist1 = hist1 / hist1.sum()

                kl_div = float(scipy_stats.entropy(hist0, hist1))

            # Find optimal threshold (Youden's J)
            if is_binary:
                fpr, tpr, thresholds = roc_curve(preds.y_true, prob_1d)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = (
                    thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
                )
            else:
                optimal_threshold = 0.5

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_score_distribution_chart(class_probs, optimal_threshold)
            except Exception as e:
                self._record_chart_failure("score_distribution", e)

            content = """
# Score Distribution by Class

Analyzing how predicted probabilities are distributed across actual classes
reveals model discrimination capability.

## Distribution Statistics

"""
            for label, probs in class_probs.items():
                content += f"""**{label}:** Mean = {probs.mean():.4f}, Std = {probs.std():.4f}, Median = {np.median(probs):.4f}

"""

            if kl_div is not None:
                content += f"**KL Divergence:** {kl_div:.4f} (higher = better separation)\n\n"

            content += f"""**Optimal Threshold (Youden's J):** {optimal_threshold:.4f}

## Score Distribution Visualization

![Score Distribution]({fig_path})

## Interpretation

- Well-separated distributions indicate strong model discrimination
- Overlapping distributions suggest classification uncertainty
- The optimal threshold maximizes the gap between TPR and FPR

---
"""
            self._content.append(content)
            self._store_section_data(
                "score_distribution",
                "Score Distribution",
                {
                    "n_classes": len(unique_classes),
                    "optimal_threshold": float(optimal_threshold),
                },
            )

        except Exception as e:
            self._record_section_failure("Score Distribution", e)

    # =========================================================================
    # DEEP LEARNING ANALYSIS
    # =========================================================================

    def add_deep_learning_analysis(
        self,
        model: Any,
        X_sample: Any = None,
        layer_names: List[str] | None = None,
        training_history: Dict | None = None,
    ) -> Any:
        """
        Add deep learning-specific analysis (conditional — only if model is a neural network).

        Args:
            model: Trained model (PyTorch, Keras, or TensorFlow)
            X_sample: Sample input data for gradient analysis
            layer_names: Specific layers to analyze
            training_history: Dict with 'loss', 'val_loss', optionally 'accuracy', 'val_accuracy'
        """
        try:
            if not self._is_neural_network(model):
                return

            content = """
# Deep Learning Analysis

Neural network-specific analysis including training dynamics and model architecture.

"""
            fig_paths = []

            # Training curves
            if training_history:
                fig_path = self._create_training_curves(training_history)
                if fig_path:
                    fig_paths.append(fig_path)

                    # Extract training stats
                    loss_hist = training_history.get("loss", [])
                    val_loss_hist = training_history.get("val_loss", [])

                    n_epochs = len(loss_hist)
                    best_epoch = int(np.argmin(val_loss_hist)) + 1 if val_loss_hist else n_epochs
                    final_loss = loss_hist[-1] if loss_hist else 0
                    final_val_loss = val_loss_hist[-1] if val_loss_hist else 0
                    best_val_loss = min(val_loss_hist) if val_loss_hist else 0

                    content += f"""## Training Summary

| Metric | Value |
|--------|-------|
| Total Epochs | {n_epochs} |
| Best Epoch | {best_epoch} |
| Final Training Loss | {final_loss:.6f} |
| Final Validation Loss | {final_val_loss:.6f} |
| Best Validation Loss | {best_val_loss:.6f} |
| Overfit Gap | {final_loss - final_val_loss:.6f} |

## Training Curves

![Training Curves]({fig_path})

"""

            # Architecture summary
            arch_info = self._get_nn_architecture_info(model)
            if arch_info:
                content += f"""## Architecture Summary

| Property | Value |
|----------|-------|
| Framework | {arch_info.get('framework', 'Unknown')} |
| Total Parameters | {arch_info.get('total_params', 'N/A'):,} |
| Trainable Parameters | {arch_info.get('trainable_params', 'N/A'):,} |
| Number of Layers | {arch_info.get('n_layers', 'N/A')} |

"""

            # Gradient analysis (simple gradient*input if possible)
            if X_sample is not None:
                grad_info = self._compute_gradient_attribution(model, X_sample)
                if grad_info:
                    content += """## Gradient Attribution

Feature importance via input gradient analysis (gradient × input).

"""

            content += """
## Interpretation

- **Converging loss curves** with small train-val gap indicate good generalization
- **Diverging curves** after a point suggest overfitting (consider early stopping)
- **Flat loss curves** suggest learning rate may be too low or model capacity insufficient

---
"""
            self._content.append(content)

            self._store_section_data(
                "deep_learning",
                "Deep Learning Analysis",
                {
                    "is_nn": True,
                    "training_history": training_history,
                },
                [{"type": "line", "path": p} for p in fig_paths],
            )

        except Exception as e:
            self._record_section_failure("Deep Learning Analysis", e)

    def add_model_card(
        self,
        model: Any,
        results: Dict[str, Any],
        intended_use: str = "",
        limitations: str = "",
        ethical: str = "",
    ) -> Any:
        """
        Add Model Card section following Google Model Card standard:
        - Model details (type, version, framework)
        - Intended use and users
        - Limitations and out-of-scope uses
        - Ethical considerations
        - Auto-generated limitations from analysis results
        """
        try:
            self._add_model_card_impl(model, results, intended_use, limitations, ethical)
        except Exception as e:
            self._record_section_failure("Model Card", e)

    def add_regression_analysis(self, y_true: Any, y_pred: Any) -> None:
        """
        Add regression analysis with:
        - 2x2 subplot: Predicted vs Actual, Residuals, Q-Q plot, Residual histogram
        - R², MAE, RMSE, MAPE metrics
        - Breusch-Pagan heteroscedasticity test
        """
        try:
            from sklearn.metrics import (
                mean_absolute_error,
                mean_absolute_percentage_error,
                mean_squared_error,
                r2_score,
            )

            preds = self._make_predictions(y_true, y_pred)
            y_true_arr = preds.y_true.astype(float)
            y_pred_arr = preds.y_pred.astype(float)
            residuals = y_true_arr - y_pred_arr

            # Calculate metrics
            r2 = r2_score(y_true_arr, y_pred_arr)
            mae = mean_absolute_error(y_true_arr, y_pred_arr)
            rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
            try:
                mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100
            except Exception as e:
                if hasattr(self, "_warnings"):
                    self._record_internal_warning(
                        "MAPEComputation", "sklearn MAPE failed, using manual", e
                    )
                mape = np.mean(np.abs((y_true_arr - y_pred_arr) / (y_true_arr + 1e-10))) * 100

            # Adjusted R²
            n = len(y_true_arr)
            p = 1  # approximation
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

            # Heteroscedasticity test
            hetero_result = self._detect_heteroscedasticity(y_pred_arr, residuals)

            # Create visualization
            fig_path = self._create_regression_charts(y_true_arr, y_pred_arr, residuals)

            content = f"""
# Regression Analysis

Comprehensive evaluation of regression model performance including residual diagnostics.

## Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² Score | {r2:.4f} | {'Excellent' if r2 > 0.9 else ('Good' if r2 > 0.7 else ('Fair' if r2 > 0.5 else 'Poor'))} fit |
| Adjusted R² | {adj_r2:.4f} | Adjusted for number of predictors |
| MAE | {mae:.4f} | Average absolute error |
| RMSE | {rmse:.4f} | Root mean squared error |
| MAPE | {mape:.2f}% | Mean absolute percentage error |

## Residual Diagnostics

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| Breusch-Pagan | {hetero_result.get('statistic', 'N/A')} | {hetero_result.get('p_value', 'N/A')} | {hetero_result.get('result', 'N/A')} |

{'**Warning: Heteroscedasticity detected.** Consider weighted regression or variance-stabilizing transformations.' if hetero_result.get('is_heteroscedastic', False) else '**Homoscedasticity assumption holds.** Residual variance appears constant.'}

## Diagnostic Plots

![Regression Diagnostics]({fig_path})

## Interpretation Guide

- **Predicted vs Actual**: Points near diagonal indicate accurate predictions
- **Residual Plot**: Should show random scatter around zero (no patterns)
- **Q-Q Plot**: Points on diagonal indicate normally distributed residuals
- **Residual Histogram**: Should approximate a bell curve centered at zero

---
"""
            self._content.append(content)
            self._store_section_data(
                "regression",
                "Regression Analysis",
                {
                    "r2": r2,
                    "rmse": rmse,
                    "is_heteroscedastic": hetero_result.get("is_heteroscedastic", False),
                },
            )

        except Exception as e:
            self._record_section_failure("Regression Analysis", e)
