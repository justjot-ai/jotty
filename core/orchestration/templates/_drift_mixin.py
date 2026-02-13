"""
Drift analysis mixin for ProfessionalMLReport.

Provides data drift monitoring capabilities including PSI, KS test,
Jensen-Shannon divergence, Chi-squared test, Cramer's V, Mahalanobis distance,
Maximum Mean Discrepancy (MMD), classifier-based drift detection, and ensemble
drift scoring.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .ml_report_generator import ReportContext

logger = logging.getLogger(__name__)


class DriftMixin:
    """Mixin class providing drift analysis methods for ProfessionalMLReport.

    This mixin is designed to be used with ProfessionalMLReport and relies on
    attributes and methods defined in that class, such as self.theme,
    self.figures_dir, self._content, self._maybe_add_narrative, and
    self._store_section_data.
    """

    def _compute_categorical_drift(self, ref_col, cur_col, feature_name: str) -> Dict:
        """Compute drift for categorical features using Chi-squared test and Cramer's V."""
        try:
            from scipy.stats import chi2_contingency

            # Get all unique categories
            all_cats = np.union1d(np.unique(ref_col), np.unique(cur_col))

            # Build contingency table
            ref_counts = np.array([np.sum(ref_col == c) for c in all_cats])
            cur_counts = np.array([np.sum(cur_col == c) for c in all_cats])
            contingency = np.array([ref_counts, cur_counts])

            # Remove zero columns
            nonzero = contingency.sum(axis=0) > 0
            contingency = contingency[:, nonzero]

            if contingency.shape[1] < 2:
                return {'chi2_stat': 0.0, 'chi2_pval': 1.0, 'cramers_v': 0.0, 'status': 'OK'}

            chi2, pval, dof, _ = chi2_contingency(contingency)

            # Cramer's V for effect size
            n = contingency.sum()
            k = min(contingency.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n * k))) if n * k > 0 else 0.0

            if cramers_v > 0.3:
                status = "ALERT"
            elif cramers_v > 0.1:
                status = "WARNING"
            else:
                status = "OK"

            return {
                'feature': feature_name,
                'chi2_stat': float(chi2),
                'chi2_pval': float(pval),
                'cramers_v': cramers_v,
                'status': status,
                'dtype': 'categorical',
            }
        except Exception as e:
            self._record_internal_warning('CategoricalDrift', 'Failed to compute categorical drift metrics', e)
            return {'chi2_stat': 0.0, 'chi2_pval': 1.0, 'cramers_v': 0.0, 'status': 'OK'}

    def _compute_mahalanobis_drift(self, X_ref_numeric, X_cur_numeric) -> Dict:
        """Compute multivariate Mahalanobis distance between reference and current datasets."""
        try:
            from scipy.stats import chi2

            mean_ref = np.mean(X_ref_numeric, axis=0)
            mean_cur = np.mean(X_cur_numeric, axis=0)

            # Pooled covariance with pseudo-inverse for collinearity handling
            cov_ref = np.cov(X_ref_numeric, rowvar=False)
            cov_cur = np.cov(X_cur_numeric, rowvar=False)
            pooled_cov = (cov_ref + cov_cur) / 2.0

            cov_inv = np.linalg.pinv(pooled_cov)

            diff = mean_ref - mean_cur
            mahal_dist = float(np.sqrt(diff @ cov_inv @ diff))

            # Chi-squared p-value approximation
            p_features = X_ref_numeric.shape[1]
            pval = float(1.0 - chi2.cdf(mahal_dist ** 2, df=p_features))

            return {
                'mahalanobis_distance': mahal_dist,
                'p_value': pval,
                'n_features': p_features,
            }
        except Exception as e:
            self._record_internal_warning('MahalanobisDrift', 'Failed to compute Mahalanobis distance drift', e)
            return {'mahalanobis_distance': 0.0, 'p_value': 1.0, 'n_features': 0}

    def _compute_mmd(self, X_ref, X_cur, n_permutations=None) -> Dict:
        """Compute Maximum Mean Discrepancy with RBF kernel (median heuristic for bandwidth).

        Args:
            X_ref: Reference data array
            X_cur: Current data array
            n_permutations: Number of permutations for p-value estimation

        Returns:
            Dict with mmd_stat, p_value, significant
        """
        try:
            config = getattr(self, 'config', {})
            if n_permutations is None:
                n_permutations = config.get('mmd_permutations', 500)
            # Subsample for tractability
            max_samples = 500
            rng = np.random.RandomState(42)
            if len(X_ref) > max_samples:
                idx = rng.choice(len(X_ref), max_samples, replace=False)
                X_ref = X_ref[idx]
            if len(X_cur) > max_samples:
                idx = rng.choice(len(X_cur), max_samples, replace=False)
                X_cur = X_cur[idx]

            # RBF kernel with median heuristic for bandwidth
            from scipy.spatial.distance import cdist

            combined = np.vstack([X_ref, X_cur])
            dists = cdist(combined, combined, 'sqeuclidean')
            median_dist = float(np.median(dists[dists > 0]))
            sigma2 = median_dist if median_dist > 0 else 1.0

            def _rbf_kernel(X, Y):
                d = cdist(X, Y, 'sqeuclidean')
                return np.exp(-d / (2.0 * sigma2))

            n_ref = len(X_ref)
            n_cur = len(X_cur)

            K_rr = _rbf_kernel(X_ref, X_ref)
            K_cc = _rbf_kernel(X_cur, X_cur)
            K_rc = _rbf_kernel(X_ref, X_cur)

            mmd_stat = float(
                np.mean(K_rr) - 2 * np.mean(K_rc) + np.mean(K_cc))

            # Permutation test for p-value
            combined_data = np.vstack([X_ref, X_cur])
            n_total = len(combined_data)
            null_stats = []

            for _ in range(n_permutations):
                perm = rng.permutation(n_total)
                perm_ref = combined_data[perm[:n_ref]]
                perm_cur = combined_data[perm[n_ref:]]

                K_rr_p = _rbf_kernel(perm_ref, perm_ref)
                K_cc_p = _rbf_kernel(perm_cur, perm_cur)
                K_rc_p = _rbf_kernel(perm_ref, perm_cur)

                null_stat = float(np.mean(K_rr_p) - 2 * np.mean(K_rc_p) + np.mean(K_cc_p))
                null_stats.append(null_stat)

            p_value = float(np.mean(np.array(null_stats) >= mmd_stat))

            return {
                'mmd_stat': mmd_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'bandwidth': float(np.sqrt(sigma2)),
            }
        except Exception as e:
            self._record_internal_warning('MMDComputation', 'Failed to compute Maximum Mean Discrepancy', e)
            return {'mmd_stat': 0.0, 'p_value': 1.0, 'significant': False, 'bandwidth': 0.0}

    def _classifier_drift_detection(self, X_ref, X_cur) -> Dict:
        """Detect drift by training a classifier to distinguish reference vs current.

        If the classifier can easily tell them apart (AUC > 0.6), drift is present.

        Returns:
            Dict with auc, drift_detected, feature_importances
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            # Label: ref=0, cur=1
            n_ref = len(X_ref)
            n_cur = len(X_cur)
            X_combined = np.vstack([X_ref, X_cur])
            y_combined = np.concatenate([np.zeros(n_ref), np.ones(n_cur)])

            # 5-fold cross-val AUC
            import warnings as _warnings
            clf = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
            with _warnings.catch_warnings():
                _warnings.simplefilter('ignore')
                scores = cross_val_score(clf, X_combined, y_combined,
                                          cv=min(5, min(n_ref, n_cur)), scoring='roc_auc')
            mean_auc = float(np.mean(scores))

            # Fit full model for feature importances
            clf.fit(X_combined, y_combined)
            importances = np.abs(clf.coef_[0])
            top_indices = np.argsort(importances)[::-1][:10]

            feature_importances = {
                f'feature_{i}': float(importances[i])
                for i in top_indices
            }

            config = getattr(self, 'config', {})
            auc_threshold = config.get('classifier_drift_auc_threshold', 0.6)
            return {
                'auc': mean_auc,
                'drift_detected': mean_auc > auc_threshold,
                'feature_importances': feature_importances,
            }
        except Exception as e:
            self._record_internal_warning('ClassifierDrift', 'Failed to perform classifier-based drift detection', e)
            return {'auc': 0.5, 'drift_detected': False, 'feature_importances': {}}

    def _compute_ensemble_drift_score(self, mahal_result, mmd_result,
                                        classifier_result) -> Dict:
        """Combine Mahalanobis, MMD, and classifier drift scores into ensemble.

        Weights: Mahalanobis=0.3, MMD=0.35, Classifier=0.35.
        Maps to composite 0-1 score with severity interpretation.

        Returns:
            Dict with composite_score, severity, component_scores, disclaimer
        """
        # Normalize each to 0-1
        # Mahalanobis: use 1 - p_value as score
        mahal_score = 1.0 - mahal_result.get('p_value', 1.0)

        # MMD: use 1 - p_value as score
        mmd_score = 1.0 - mmd_result.get('p_value', 1.0)

        # Classifier: (AUC - 0.5) * 2, clipped to [0, 1]
        clf_auc = classifier_result.get('auc', 0.5)
        clf_score = float(np.clip((clf_auc - 0.5) * 2, 0, 1))

        config = getattr(self, 'config', {})
        weights = config.get('ensemble_drift_weights', (0.3, 0.35, 0.35))
        composite = weights[0] * mahal_score + weights[1] * mmd_score + weights[2] * clf_score
        composite = float(np.clip(composite, 0, 1))

        if composite < 0.2:
            severity = 'NONE'
        elif composite < 0.4:
            severity = 'MILD'
        elif composite < 0.7:
            severity = 'MODERATE'
        else:
            severity = 'SEVERE'

        return {
            'composite_score': composite,
            'severity': severity,
            'mahal_score': mahal_score,
            'mmd_score': mmd_score,
            'classifier_score': clf_score,
            'disclaimer': ('Mahalanobis distance assumes multivariate normality; '
                          'results may be unreliable for non-Gaussian features.'),
        }

    def add_drift_analysis(self, X_reference, X_current, feature_names: List[str] = None,
                           psi_warn: float = 0.1, psi_alert: float = 0.25,
                           feature_importance: Dict[str, float] = None,
                           y_reference=None, y_current=None):
        """
        Add data drift monitoring analysis between reference and current datasets.

        Uses PSI, KS test, and Jensen-Shannon divergence for numeric features,
        Chi-squared test and Cramer's V for categorical features, and
        Mahalanobis distance for multivariate drift detection.

        Args:
            X_reference: Reference dataset (training data)
            X_current: Current dataset (production/new data)
            feature_names: Feature names (auto-detected from DataFrame)
            psi_warn: PSI threshold for warning (default 0.1)
            psi_alert: PSI threshold for alert (default 0.25)
            feature_importance: Dict mapping feature_name -> importance for severity scoring
            y_reference: Reference target for concept drift detection
            y_current: Current target for concept drift detection
        """
        try:
            from scipy.stats import ks_2samp
            from scipy.spatial.distance import jensenshannon

            # Preserve DataFrame info before converting
            ref_df = X_reference if hasattr(X_reference, 'columns') else None
            cur_df = X_current if hasattr(X_current, 'columns') else None

            X_ref = np.asarray(X_reference)
            X_cur = np.asarray(X_current)

            if feature_names is None:
                if ref_df is not None:
                    feature_names = list(ref_df.columns)
                else:
                    feature_names = [f'Feature_{i}' for i in range(X_ref.shape[1])]

            n_features = min(X_ref.shape[1], X_cur.shape[1], len(feature_names))
            drift_results = []
            categorical_results = []
            numeric_indices = []

            for i in range(n_features):
                ref_col = X_ref[:, i]
                cur_col = X_cur[:, i]

                # Auto-detect categorical: object dtype or <= 20 unique values
                is_categorical = False
                if ref_df is not None and ref_df[feature_names[i]].dtype == 'object':
                    is_categorical = True
                elif len(np.unique(ref_col[~pd.isna(ref_col)])) <= 20:
                    is_categorical = True

                if is_categorical:
                    # Route to categorical drift
                    cat_result = self._compute_categorical_drift(ref_col, cur_col, feature_names[i])
                    cat_result['feature'] = feature_names[i]
                    categorical_results.append(cat_result)
                    continue

                # Numeric drift
                ref_col = ref_col.astype(float)
                cur_col = cur_col.astype(float)

                ref_col = ref_col[~np.isnan(ref_col)]
                cur_col = cur_col[~np.isnan(cur_col)]

                if len(ref_col) < 10 or len(cur_col) < 10:
                    continue

                numeric_indices.append(i)

                psi = self._calculate_psi(ref_col, cur_col)
                ks_stat, ks_pval = ks_2samp(ref_col, cur_col)
                js_div = self._calculate_js_divergence(ref_col, cur_col)

                if psi >= psi_alert:
                    status = "ALERT"
                elif psi >= psi_warn:
                    status = "WARNING"
                else:
                    status = "OK"

                result = {
                    'feature': feature_names[i],
                    'psi': psi,
                    'ks_stat': ks_stat,
                    'ks_pval': ks_pval,
                    'js_div': js_div,
                    'status': status,
                    'dtype': 'numeric',
                }

                # Drift severity score (drift * importance) if importance provided
                if feature_importance and feature_names[i] in feature_importance:
                    imp = feature_importance[feature_names[i]]
                    result['importance'] = float(imp)
                    result['severity_score'] = float(psi * imp)

                drift_results.append(result)

            if not drift_results and not categorical_results:
                return

            # Create visualization
            fig_path = self._create_drift_heatmap(drift_results, psi_warn, psi_alert) if drift_results else ""

            # Summary stats
            n_alert = sum(1 for r in drift_results if r['status'] == 'ALERT')
            n_warn = sum(1 for r in drift_results if r['status'] == 'WARNING')
            n_ok = sum(1 for r in drift_results if r['status'] == 'OK')

            # Build numeric table
            table_md = "| Feature | PSI | KS Stat | KS p-value | JS Divergence | Status |\n"
            table_md += "|---------|-----|---------|------------|---------------|--------|\n"
            for r in sorted(drift_results, key=lambda x: x['psi'], reverse=True)[:20]:
                status_icon = "ALERT" if r['status'] == 'ALERT' else ("WARN" if r['status'] == 'WARNING' else "OK")
                table_md += (f"| {r['feature'][:25]} | {r['psi']:.4f} | {r['ks_stat']:.4f} | "
                            f"{r['ks_pval']:.4f} | {r['js_div']:.4f} | {status_icon} |\n")

            content = f"""
# Data Drift Monitoring

Monitoring for distribution shift between reference (training) and current (production) data.

## Drift Summary

| Status | Count |
|--------|-------|
| Alert (PSI >= {psi_alert}) | {n_alert} |
| Warning (PSI >= {psi_warn}) | {n_warn} |
| OK | {n_ok} |

**Total Numeric Features Analyzed:** {len(drift_results)}

## Feature-Level Drift Analysis (Numeric)

{table_md}

"""
            # Categorical drift section
            if categorical_results:
                n_cat_alert = sum(1 for r in categorical_results if r.get('status') == 'ALERT')
                content += f"""## Categorical Feature Drift

| Feature | Chi-squared | p-value | Cramer's V | Status |
|---------|------------|---------|-----------|--------|
"""
                for r in sorted(categorical_results, key=lambda x: x.get('cramers_v', 0), reverse=True):
                    content += (f"| {r['feature'][:25]} | {r['chi2_stat']:.2f} | "
                               f"{r['chi2_pval']:.4f} | {r['cramers_v']:.4f} | {r['status']} |\n")
                content += f"\n**Categorical features with significant drift:** {n_cat_alert}\n\n"

            # Multivariate drift (Mahalanobis, MMD, Classifier)
            mahal_result = {}
            mmd_result = {}
            classifier_result = {}
            ensemble_result = {}

            if len(numeric_indices) >= 2:
                X_ref_num = X_ref[:, numeric_indices].astype(float)
                X_cur_num = X_cur[:, numeric_indices].astype(float)
                # Remove rows with NaN
                ref_mask = ~np.any(np.isnan(X_ref_num), axis=1)
                cur_mask = ~np.any(np.isnan(X_cur_num), axis=1)
                if ref_mask.sum() > 10 and cur_mask.sum() > 10:
                    X_ref_clean = X_ref_num[ref_mask]
                    X_cur_clean = X_cur_num[cur_mask]

                    mahal_result = self._compute_mahalanobis_drift(
                        X_ref_clean, X_cur_clean)

                    content += f"""## Multivariate Drift (Mahalanobis Distance)

| Metric | Value |
|--------|-------|
| Mahalanobis Distance | {mahal_result['mahalanobis_distance']:.4f} |
| p-value (chi-squared) | {mahal_result['p_value']:.4f} |
| Features Included | {mahal_result['n_features']} |
| Significant | {'YES' if mahal_result['p_value'] < 0.05 else 'NO'} |

"""

                    # MMD (Maximum Mean Discrepancy)
                    mmd_result = self._compute_mmd(X_ref_clean, X_cur_clean)
                    content += f"""## Maximum Mean Discrepancy (MMD)

| Metric | Value |
|--------|-------|
| MMD Statistic | {mmd_result['mmd_stat']:.6f} |
| p-value (permutation) | {mmd_result['p_value']:.4f} |
| Significant | {'YES' if mmd_result['significant'] else 'NO'} |
| RBF Bandwidth | {mmd_result['bandwidth']:.4f} |

"""

                    # Classifier-based drift detection
                    classifier_result = self._classifier_drift_detection(
                        X_ref_clean, X_cur_clean)
                    content += f"""## Classifier-Based Drift Detection

| Metric | Value |
|--------|-------|
| Discriminator AUC | {classifier_result['auc']:.4f} |
| Drift Detected | {'YES' if classifier_result['drift_detected'] else 'NO'} |

"""
                    if classifier_result.get('feature_importances'):
                        content += "**Top drift-driving features:** "
                        top_feats = sorted(classifier_result['feature_importances'].items(),
                                          key=lambda x: x[1], reverse=True)[:5]
                        content += ', '.join([f"{f}({v:.3f})" for f, v in top_feats])
                        content += "\n\n"

                    # Ensemble drift score
                    ensemble_result = self._compute_ensemble_drift_score(
                        mahal_result, mmd_result, classifier_result)
                    content += f"""## Ensemble Drift Score

| Component | Score | Weight |
|-----------|-------|--------|
| Mahalanobis | {ensemble_result['mahal_score']:.4f} | 0.30 |
| MMD | {ensemble_result['mmd_score']:.4f} | 0.35 |
| Classifier | {ensemble_result['classifier_score']:.4f} | 0.35 |
| **Composite** | **{ensemble_result['composite_score']:.4f}** | |

**Severity:** {ensemble_result['severity']}

> {ensemble_result['disclaimer']}

"""

            # Drift severity ranking (if importance provided)
            severity_results = [r for r in drift_results if 'severity_score' in r]
            if severity_results:
                content += """## Drift Severity Ranking (Drift x Importance)

| Feature | PSI | Importance | Severity Score |
|---------|-----|-----------|---------------|
"""
                for r in sorted(severity_results, key=lambda x: x['severity_score'], reverse=True)[:10]:
                    content += (f"| {r['feature'][:25]} | {r['psi']:.4f} | "
                               f"{r['importance']:.4f} | {r['severity_score']:.4f} |\n")

            # Concept drift (target distribution shift)
            concept_drift = {}
            if y_reference is not None and y_current is not None:
                y_ref = np.asarray(y_reference)
                y_cur = np.asarray(y_current)
                if len(y_ref) >= 10 and len(y_cur) >= 10:
                    try:
                        y_ref_f = y_ref.astype(float)
                        y_cur_f = y_cur.astype(float)
                        target_psi = self._calculate_psi(y_ref_f, y_cur_f)
                        target_ks, target_kp = ks_2samp(y_ref_f, y_cur_f)
                        concept_drift = {'psi': target_psi, 'ks_stat': target_ks, 'ks_pval': target_kp}

                        content += f"""## Concept Drift (Target Distribution)

| Metric | Value |
|--------|-------|
| Target PSI | {target_psi:.4f} |
| Target KS Stat | {target_ks:.4f} |
| Target KS p-value | {target_kp:.4f} |
| Status | {'ALERT' if target_psi >= psi_alert else ('WARNING' if target_psi >= psi_warn else 'OK')} |

"""
                    except Exception as e:
                        self._record_internal_warning('ConceptDrift', 'Failed to compute concept drift for target distribution', e)

            if fig_path:
                content += f"""## Drift Heatmap

![Data Drift Heatmap]({fig_path})

"""

            content += f"""## Interpretation

- **PSI** (Population Stability Index): < {psi_warn} = stable, {psi_warn}-{psi_alert} = moderate drift, > {psi_alert} = significant drift
- **KS Test**: p-value < 0.05 suggests statistically significant distribution change
- **JS Divergence**: Symmetric measure of distribution difference (0 = identical)
- **Cramer's V**: Effect size for categorical drift (> 0.1 moderate, > 0.3 large)
- **Mahalanobis**: Multivariate distance accounting for feature correlations
- **MMD**: Maximum Mean Discrepancy — distribution-free test using kernel embeddings
- **Classifier**: Discriminator AUC — if a classifier can distinguish ref/cur, drift is present

"""
            narrative = self._maybe_add_narrative('Drift Analysis',
                f'Alerts: {n_alert}, Warnings: {n_warn}, Categorical: {len(categorical_results)}',
                section_type='drift_analysis')
            content += f"""{narrative}

---
"""
            self._content.append(content)

            self._store_section_data('drift_analysis', 'Data Drift Monitoring', {
                'drift_results': drift_results,
                'categorical_results': categorical_results,
                'mahalanobis': mahal_result,
                'mmd_result': mmd_result,
                'classifier_result': classifier_result,
                'ensemble_result': ensemble_result,
                'concept_drift': concept_drift,
                'n_alert': n_alert, 'n_warn': n_warn, 'n_ok': n_ok,
            }, [{'type': 'heatmap', 'path': fig_path}, {'type': 'violin'}] if drift_results else [])

        except Exception as e:
            self._record_section_failure('Drift Analysis', e)

    def _calculate_psi(self, reference, current, n_bins: int = None) -> float:
        """Calculate Population Stability Index between two distributions.

        Uses quantile-based binning by default (configurable via self.config).
        """
        config = getattr(self, 'config', {})
        if n_bins is None:
            n_bins = config.get('psi_bins', 10)
        binning_method = config.get('psi_binning_method', 'quantile')
        eps = 1e-4

        if binning_method == 'quantile':
            # Quantile-based binning from reference distribution
            quantiles = np.linspace(0, 100, n_bins + 1)
            breakpoints = np.percentile(reference, quantiles)
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf
            # Ensure unique breakpoints
            breakpoints = np.unique(breakpoints)
            # Fallback to equal-width if too few unique breakpoints (tied values)
            if len(breakpoints) < 4:
                logger.warning(
                    f"PSI: quantile binning produced only {len(breakpoints)} unique "
                    f"breakpoints (< 4), falling back to equal-width binning"
                )
                breakpoints = np.linspace(np.min(reference), np.max(reference), n_bins + 1)
                breakpoints[0] = -np.inf
                breakpoints[-1] = np.inf
        else:
            # Linear binning (legacy)
            breakpoints = np.linspace(np.min(reference), np.max(reference), n_bins + 1)
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = ref_counts / len(reference) + eps
        cur_pct = cur_counts / len(current) + eps

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def _calculate_js_divergence(self, reference, current, n_bins: int = None) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        try:
            from scipy.spatial.distance import jensenshannon

            config = getattr(self, 'config', {})
            if n_bins is None:
                n_bins = config.get('js_divergence_bins', 50)

            all_data = np.concatenate([reference, current])
            bins = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)

            ref_hist = np.histogram(reference, bins=bins, density=True)[0] + 1e-10
            cur_hist = np.histogram(current, bins=bins, density=True)[0] + 1e-10

            ref_hist = ref_hist / ref_hist.sum()
            cur_hist = cur_hist / cur_hist.sum()

            return float(jensenshannon(ref_hist, cur_hist))
        except Exception as e:
            self._record_internal_warning('JensenShannonDivergence', 'Failed to calculate Jensen-Shannon divergence', e)
            return 0.0

    def _create_drift_heatmap(self, drift_results: List[Dict],
                               psi_warn: float, psi_alert: float) -> str:
        """Create drift heatmap visualization."""
        try:
            from matplotlib.colors import LinearSegmentedColormap

            n = min(len(drift_results), 25)
            sorted_results = sorted(drift_results, key=lambda x: x['psi'], reverse=True)[:n]

            features = [r['feature'][:20] for r in sorted_results]
            metrics = ['PSI', 'KS Stat', 'JS Div']

            data = np.array([
                [r['psi'] for r in sorted_results],
                [r['ks_stat'] for r in sorted_results],
                [r['js_div'] for r in sorted_results],
            ]).T

            with self._chart_context('drift_heatmap', figsize=(14, max(6, n * 0.35)),
                                     nrows=1, ncols=2,
                                     gridspec_kw={'width_ratios': [3, 1]}) as (fig, (ax1, ax2)):
                # Heatmap
                colors_cmap = LinearSegmentedColormap.from_list('drift',
                    [self.theme['success'], self.theme['warning'], self.theme['danger']])
                im = ax1.imshow(data, cmap=colors_cmap, aspect='auto')

                ax1.set_xticks(range(len(metrics)))
                ax1.set_xticklabels(metrics, fontsize=10)
                ax1.set_yticks(range(len(features)))
                ax1.set_yticklabels(features, fontsize=9)

                # Annotate
                for i in range(len(features)):
                    for j in range(len(metrics)):
                        ax1.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                                fontsize=8, color='white' if data[i, j] > 0.5 else 'black')

                ax1.set_title('Drift Metrics Heatmap', fontsize=14, fontweight='bold',
                             color=self.theme['primary'])
                fig.colorbar(im, ax=ax1, shrink=0.8, label='Metric Value')

                # PSI bar chart
                psi_values = [r['psi'] for r in sorted_results]
                colors = [self.theme['danger'] if v >= psi_alert
                         else (self.theme['warning'] if v >= psi_warn
                               else self.theme['success']) for v in psi_values]

                ax2.barh(range(len(features)), psi_values, color=colors, alpha=0.8)
                ax2.axvline(x=psi_warn, color=self.theme['warning'], linestyle='--',
                           alpha=0.7, label=f'Warn ({psi_warn})')
                ax2.axvline(x=psi_alert, color=self.theme['danger'], linestyle='--',
                           alpha=0.7, label=f'Alert ({psi_alert})')
                ax2.set_yticks([])
                ax2.set_xlabel('PSI', fontsize=10)
                ax2.set_title('PSI Values', fontsize=12, fontweight='bold', color=self.theme['primary'])
                ax2.legend(fontsize=8, loc='lower right')
                ax2.invert_yaxis()

            return 'figures/drift_heatmap.png'
        except Exception as e:
            logger.debug(f"Failed to create drift heatmap: {e}")
            return ""
