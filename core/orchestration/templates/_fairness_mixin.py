"""
Fairness and bias audit mixin for ML report generation.

Provides methods to evaluate model fairness across sensitive demographic features,
compute intersectional fairness metrics, find fair decision thresholds, and
generate fairness radar charts. Designed to be mixed into ProfessionalMLReport.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .ml_report_generator import ReportContext

logger = logging.getLogger(__name__)


class FairnessMixin:
    """
    Mixin class providing fairness and bias audit capabilities for ProfessionalMLReport.

    This mixin adds methods for evaluating model fairness across sensitive demographic
    features using multiple fairness criteria including disparate impact, equalized odds,
    demographic parity, and calibration parity. It also supports intersectional analysis,
    fair threshold optimization, and radar chart visualization.

    Should be used as a mixin with ProfessionalMLReport which provides:
        - self._content (list)
        - self._maybe_add_narrative(...)
        - self._store_section_data(...)
        - self.theme (dict)
        - self.figures_dir (Path)
    """

    def add_fairness_audit(self, X, y_true, y_pred, y_prob,
                           sensitive_features: Dict[str, Any],
                           labels: List[str] = None):
        """
        Add fairness and bias audit for model predictions across sensitive groups.

        Args:
            X: Feature DataFrame
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            sensitive_features: Dict mapping feature_name -> column data or column name in X
            labels: Class labels
        """
        try:
            from .ml_report_generator import PredictionResult
            preds = PredictionResult.from_predictions(y_true, y_pred, y_prob)
            y_true_arr = preds.y_true
            y_pred_arr = preds.y_pred
            y_prob_arr = preds.y_prob

            content = f"""
# Fairness & Bias Audit

Evaluating model fairness across sensitive demographic features using multiple
fairness criteria and the 80% rule (disparate impact ratio > 0.8).

"""

            all_metrics = {}

            for feat_name, feat_data in sensitive_features.items():
                # Resolve feature data
                if isinstance(feat_data, str) and hasattr(X, 'columns') and feat_data in X.columns:
                    groups = np.asarray(X[feat_data])
                elif hasattr(feat_data, '__len__'):
                    groups = np.asarray(feat_data)
                else:
                    continue

                # Compute fairness metrics per group
                metrics = self._compute_fairness_metrics(y_true_arr, y_pred_arr, y_prob_arr, groups)

                if not metrics:
                    continue

                all_metrics[feat_name] = metrics

                # Build table for this feature
                content += f"""## {feat_name}

| Group | Pos Rate | TPR | FPR | PPV | Disparate Impact |
|-------|----------|-----|-----|-----|-----------------|
"""
                # Calculate reference group (largest group), excluding _aggregate
                display_groups = {k: v for k, v in metrics.items() if k != '_aggregate'}
                group_names = list(display_groups.keys())
                ref_group = max(group_names, key=lambda g: display_groups[g]['n'])
                ref_pos_rate = display_groups[ref_group]['positive_rate']

                for group, m in sorted(display_groups.items(), key=lambda x: -x[1]['n']):
                    di = m['positive_rate'] / ref_pos_rate if ref_pos_rate > 0 else 0
                    content += (f"| {group} (n={m['n']}) | {m['positive_rate']:.3f} | "
                               f"{m['tpr']:.3f} | {m['fpr']:.3f} | {m['ppv']:.3f} | {di:.3f} |\n")

                # Pass/Fail based on 80% rule
                all_di = []
                for group, m in display_groups.items():
                    di = m['positive_rate'] / ref_pos_rate if ref_pos_rate > 0 else 0
                    all_di.append(di)

                min_di = min(all_di) if all_di else 0
                passes_80 = min_di >= 0.8

                content += f"""
**Disparate Impact Assessment:** {'PASS' if passes_80 else 'FAIL'} (min ratio: {min_di:.3f}, threshold: 0.80)

"""

            # Aggregate fairness summary
            aggregate_data = {}
            for feat_name, metrics in all_metrics.items():
                if '_aggregate' in metrics:
                    aggregate_data[feat_name] = metrics['_aggregate']

            if aggregate_data:
                content += """## Aggregate Fairness Summary

| Feature | Equalized Odds Diff | Demographic Parity Diff | Calibration Parity Diff |
|---------|--------------------|-----------------------|-----------------------|
"""
                for feat_name, agg in aggregate_data.items():
                    eo = agg['equalized_odds_diff']
                    dp = agg['demographic_parity_diff']
                    cp = agg['calibration_parity_diff']
                    content += f"| {feat_name} | {eo:.4f} | {dp:.4f} | {cp:.4f} |\n"

            # Intersectional analysis
            intersectional_metrics = {}
            if len(sensitive_features) >= 2:
                intersectional_metrics = self._compute_intersectional_fairness(
                    y_true_arr, y_pred_arr, y_prob_arr, sensitive_features, X)

                if intersectional_metrics:
                    # Filter out _aggregate key for display
                    display_metrics = {k: v for k, v in intersectional_metrics.items() if k != '_aggregate'}
                    content += """
## Intersectional Analysis

Fairness metrics computed on intersected groups:

| Group | N | Pos Rate | TPR | FPR | Accuracy |
|-------|---|----------|-----|-----|----------|
"""
                    for group, m in sorted(display_metrics.items(), key=lambda x: -x[1]['n']):
                        content += (f"| {group[:40]} | {m['n']} | {m['positive_rate']:.3f} | "
                                   f"{m['tpr']:.3f} | {m['fpr']:.3f} | {m['accuracy']:.3f} |\n")

            # Fair thresholds
            fair_thresholds = {}
            if all_metrics and y_prob_arr is not None:
                first_feat_name = list(sensitive_features.keys())[0]
                first_feat_data = sensitive_features[first_feat_name]
                if isinstance(first_feat_data, str) and hasattr(X, 'columns') and first_feat_data in X.columns:
                    groups_arr = np.asarray(X[first_feat_data])
                elif hasattr(first_feat_data, '__len__'):
                    groups_arr = np.asarray(first_feat_data)
                else:
                    groups_arr = None

                if groups_arr is not None:
                    fair_thresholds = self._find_fair_thresholds(
                        y_true_arr, y_prob_arr, groups_arr,
                        fairness_criterion='equalized_odds')
                    if fair_thresholds:
                        content += f"""
## Fair Decision Thresholds ({first_feat_name})

Group-specific thresholds optimized for equalized odds:

| Group | Optimal Threshold |
|-------|------------------|
"""
                        for group, thresh in sorted(fair_thresholds.items()):
                            content += f"| {group} | {thresh:.2f} |\n"

            # Create radar chart for first feature
            fig_path = ""
            if all_metrics:
                first_feat = list(all_metrics.keys())[0]
                # Filter out _aggregate for chart
                chart_metrics = {k: v for k, v in all_metrics[first_feat].items() if k != '_aggregate'}
                fig_path = self._create_fairness_radar_chart(chart_metrics, first_feat)

                if fig_path:
                    content += f"""
## Fairness Radar Chart ({first_feat})

![Fairness Radar]({fig_path})

"""

            # Specific mitigation recommendations based on which metrics fail
            content += """## Mitigation Recommendations

"""
            if aggregate_data:
                for feat_name, agg in aggregate_data.items():
                    if agg['equalized_odds_diff'] > 0.1:
                        content += f"- **{feat_name}:** High equalized odds difference ({agg['equalized_odds_diff']:.3f}). Consider in-processing fairness constraints or post-processing threshold adjustments.\n"
                    if agg['demographic_parity_diff'] > 0.1:
                        content += f"- **{feat_name}:** Significant demographic parity gap ({agg['demographic_parity_diff']:.3f}). Consider re-sampling or re-weighting training data.\n"
                    if agg['calibration_parity_diff'] > 0.1:
                        content += f"- **{feat_name}:** Calibration disparity detected ({agg['calibration_parity_diff']:.3f}). Consider group-wise calibration (e.g., Platt scaling per group).\n"

            content += """
- **Pre-processing:** Re-sample or re-weight training data to achieve demographic parity
- **In-processing:** Add fairness constraints to the model objective function
- **Post-processing:** Adjust decision thresholds per group to equalize outcome rates
- **Monitoring:** Continuously track fairness metrics in production

"""
            narrative = self._maybe_add_narrative('Fairness Audit',
                f'Metrics: {aggregate_data}, Thresholds: {fair_thresholds}',
                section_type='fairness_audit')
            content += f"""{narrative}

---
"""
            self._content.append(content)

            # Store comprehensive section data
            stored_metrics = {}
            for k, v in all_metrics.items():
                stored_metrics[k] = {gk: {mk: mv for mk, mv in gv.items()}
                                     for gk, gv in v.items()}

            self._store_section_data('fairness_audit', 'Fairness & Bias Audit', {
                'metrics': stored_metrics,
                'aggregate': aggregate_data,
                'intersectional': {k: v for k, v in intersectional_metrics.items() if k != '_aggregate'} if intersectional_metrics else {},
                'fair_thresholds': fair_thresholds,
            }, [{'type': 'radar', 'path': fig_path}, {'type': 'box'}] if all_metrics else [])

        except Exception as e:
            self._record_section_failure('Fairness Audit', e)

    def _compute_fairness_metrics(self, y_true, y_pred, y_prob, groups) -> Dict:
        """Compute comprehensive fairness metrics per group.

        Includes: demographic parity, equalized odds, predictive parity,
        TNR, FNR, FDR, accuracy, calibration, and aggregate measures.
        """
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return {}

        config = getattr(self, 'config', {})
        min_group_size = config.get('min_group_size_fairness', 5)

        metrics = {}
        for group in unique_groups:
            mask = groups == group
            n = mask.sum()
            if n < min_group_size:
                continue

            yt = y_true[mask]
            yp = y_pred[mask]

            positive_rate = yp.mean()
            accuracy = float(np.mean(yt == yp))

            # TPR (sensitivity / recall)
            pos_mask = yt == 1
            tpr = float(yp[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0

            # FPR
            neg_mask = yt == 0
            fpr = float(yp[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0

            # TNR (specificity)
            tnr = float(1.0 - fpr)

            # FNR (miss rate)
            fnr = float(1.0 - tpr)

            # PPV (precision)
            pred_pos_mask = yp == 1
            ppv = float(yt[pred_pos_mask].mean()) if pred_pos_mask.sum() > 0 else 0.0

            # FDR (false discovery rate)
            fdr = float(1.0 - ppv)

            # Calibration error (|mean predicted prob - mean true label|)
            calibration = 0.0
            if y_prob is not None:
                yprob_group = y_prob[mask]
                calibration = float(abs(np.mean(yprob_group) - np.mean(yt)))

            metrics[str(group)] = {
                'n': int(n),
                'positive_rate': float(positive_rate),
                'tpr': tpr,
                'fpr': fpr,
                'tnr': tnr,
                'fnr': fnr,
                'ppv': ppv,
                'fdr': fdr,
                'accuracy': accuracy,
                'calibration': calibration,
            }

        # Compute aggregate fairness measures across groups
        if len(metrics) >= 2:
            group_tprs = [m['tpr'] for m in metrics.values()]
            group_fprs = [m['fpr'] for m in metrics.values()]
            group_pos_rates = [m['positive_rate'] for m in metrics.values()]
            group_calibrations = [m['calibration'] for m in metrics.values()]

            metrics['_aggregate'] = {
                'equalized_odds_diff': float(max(max(group_tprs) - min(group_tprs), max(group_fprs) - min(group_fprs))),
                'demographic_parity_diff': float(max(group_pos_rates) - min(group_pos_rates)),
                'calibration_parity_diff': float(max(group_calibrations) - min(group_calibrations)),
            }

        return metrics

    def _compute_intersectional_fairness(self, y_true, y_pred, y_prob,
                                          sensitive_features: Dict[str, Any], X) -> Dict:
        """Compute fairness metrics on intersected groups (e.g., gender=F x race=Black).

        Only invoked when 2+ sensitive features are provided.
        """
        feature_arrays = {}
        for feat_name, feat_data in sensitive_features.items():
            if isinstance(feat_data, str) and hasattr(X, 'columns') and feat_data in X.columns:
                feature_arrays[feat_name] = np.asarray(X[feat_data])
            elif hasattr(feat_data, '__len__'):
                feature_arrays[feat_name] = np.asarray(feat_data)

        if len(feature_arrays) < 2:
            return {}

        # Create intersection groups from first 2 features
        feat_names = list(feature_arrays.keys())[:2]
        arr1 = feature_arrays[feat_names[0]]
        arr2 = feature_arrays[feat_names[1]]

        intersection_groups = np.array([
            f"{feat_names[0]}={a} x {feat_names[1]}={b}"
            for a, b in zip(arr1, arr2)
        ])

        return self._compute_fairness_metrics(y_true, y_pred, y_prob, intersection_groups)

    def _find_fair_thresholds(self, y_true, y_prob, groups,
                               fairness_criterion='equalized_odds',
                               min_performance=None) -> Dict:
        """Find per-group thresholds to satisfy fairness criterion using scipy optimization.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            groups: Group membership array
            fairness_criterion: 'equalized_odds', 'demographic_parity', or 'predictive_parity'
            min_performance: Minimum accuracy constraint (falls back to 0.5 if violated)

        Returns:
            Dict mapping group -> optimal_threshold
        """
        from scipy.optimize import minimize_scalar

        unique_groups = np.unique(groups)
        if len(unique_groups) < 2 or y_prob is None:
            return {}

        def _compute_metrics_at_threshold(yt, yp, threshold) -> Dict:
            preds = (yp >= threshold).astype(int)
            pos_mask = yt == 1
            neg_mask = yt == 0
            tpr = float(preds[pos_mask].mean()) if pos_mask.sum() > 0 else 0.0
            fpr = float(preds[neg_mask].mean()) if neg_mask.sum() > 0 else 0.0
            positive_rate = float(preds.mean())
            pred_pos = preds == 1
            ppv = float(yt[pred_pos].mean()) if pred_pos.sum() > 0 else 0.0
            accuracy = float(np.mean(yt == preds))
            return {'tpr': tpr, 'fpr': fpr, 'positive_rate': positive_rate,
                    'ppv': ppv, 'accuracy': accuracy}

        # Get target values from median across groups at threshold 0.5
        group_metrics_05 = {}
        for group in unique_groups:
            mask = groups == group
            yt = y_true[mask]
            yp = y_prob[mask]
            if (yt == 1).sum() > 0:
                group_metrics_05[str(group)] = _compute_metrics_at_threshold(yt, yp, 0.5)

        if not group_metrics_05:
            return {}

        # Determine which metrics to equalize based on criterion
        criterion_metrics = {
            'equalized_odds': ['tpr', 'fpr'],
            'demographic_parity': ['positive_rate'],
            'predictive_parity': ['ppv'],
        }
        metrics_to_equalize = criterion_metrics.get(fairness_criterion, ['tpr', 'fpr'])
        targets = {}
        for metric in metrics_to_equalize:
            vals = [m[metric] for m in group_metrics_05.values()]
            targets[metric] = float(np.median(vals))

        # Optimize per group using scipy bounded minimization
        thresholds = {}
        for group in unique_groups:
            mask = groups == group
            yt = y_true[mask]
            yp = y_prob[mask]
            pos_mask = yt == 1

            if pos_mask.sum() == 0:
                thresholds[str(group)] = 0.5
                continue

            def objective(threshold):
                m = _compute_metrics_at_threshold(yt, yp, threshold)
                return sum((m[metric] - targets[metric]) ** 2 for metric in metrics_to_equalize)

            result = minimize_scalar(objective, bounds=(0.05, 0.95), method='bounded')
            optimal_threshold = float(result.x)

            # Check minimum performance constraint
            if min_performance is not None:
                m = _compute_metrics_at_threshold(yt, yp, optimal_threshold)
                if m['accuracy'] < min_performance:
                    optimal_threshold = 0.5

            thresholds[str(group)] = round(optimal_threshold, 4)

        return thresholds

    def _create_fairness_radar_chart(self, metrics_per_group: Dict, feature_name: str) -> str:
        """Create radar/spider chart showing fairness metrics per group."""
        try:
            chart_name = f'fairness_radar_{feature_name[:20].replace(" ", "_")}'

            metric_names = ['Pos Rate', 'TPR', 'FPR', 'PPV', 'TNR', 'Accuracy']
            metric_keys = ['positive_rate', 'tpr', 'fpr', 'ppv', 'tnr', 'accuracy']
            n_metrics = len(metric_names)
            angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Close the polygon

            with self._chart_context(chart_name, figsize=(8, 8),
                                     subplot_kw=dict(polar=True)) as (fig, ax):
                colors = self.theme['chart_palette']

                for i, (group, m) in enumerate(metrics_per_group.items()):
                    values = [m.get(k, 0.0) for k in metric_keys]
                    values += values[:1]  # Close

                    color = colors[i % len(colors)]
                    ax.plot(angles, values, 'o-', linewidth=2, label=f'{group} (n={m["n"]})',
                           color=color)
                    ax.fill(angles, values, alpha=0.15, color=color)

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metric_names, fontsize=10)
                ax.set_ylim(0, 1)
                ax.set_title(f'Fairness Metrics: {feature_name}', fontsize=14,
                            fontweight='bold', color=self.theme['primary'], pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

            return f'figures/{chart_name}.png'
        except Exception as e:
            logger.debug(f"Failed to create fairness radar chart: {e}")
            return ""
