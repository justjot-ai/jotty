"""
Error analysis mixin for ProfessionalMLReport.

Provides error analysis capabilities including misclassification breakdown,
error clustering, error decomposition, rejection thresholds, error pattern
detection, and error analysis chart generation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .ml_report_generator import ReportContext

logger = logging.getLogger(__name__)


class ErrorAnalysisMixin:
    """Mixin class providing error analysis methods for ProfessionalMLReport.

    This mixin encapsulates all error-analysis-related functionality used by
    ProfessionalMLReport, including misclassification summaries, adaptive error
    clustering, FP/FN decomposition, rejection threshold computation, error
    pattern analysis by feature, and error analysis chart creation.

    Intended to be used via multiple inheritance alongside the base report class.
    """

    def add_error_analysis(self, X: pd.DataFrame, y_true, y_pred, y_prob=None, top_n: int = 10):
        """
        Add error analysis including:
        - Misclassification breakdown
        - Hardest samples to classify
        - Error patterns by feature
        """
        try:
            from sklearn.metrics import confusion_matrix
            from .ml_report_generator import PredictionResult

            preds = PredictionResult.from_predictions(y_true, y_pred, y_prob)

            # Find misclassified samples
            errors = preds.errors
            n_errors = preds.n_errors
            error_rate = n_errors / preds.n_samples * 100

            # Confusion matrix
            cm = confusion_matrix(preds.y_true, preds.y_pred)

            # Find hardest samples (most confidently wrong)
            hardest_samples = []
            if preds.has_probabilities:
                error_idx = np.where(errors)[0]
                # Get 1D confidence: max prob for multiclass, raw prob for binary
                if preds.y_prob.ndim == 2:
                    all_confidence = np.max(preds.y_prob, axis=1)
                else:
                    all_confidence = preds.y_prob
                error_conf = all_confidence[error_idx]
                # For wrong predictions, high confidence = more confident error
                if hasattr(error_conf, '__len__') and len(error_conf) > 0:
                    sorted_idx = np.argsort(error_conf)[::-1][:top_n]

                    for i in sorted_idx:
                        orig_idx = error_idx[i]
                        hardest_samples.append({
                            'index': int(orig_idx),
                            'true': int(preds.y_true[orig_idx]),
                            'pred': int(preds.y_pred[orig_idx]),
                            'prob': float(all_confidence[orig_idx]),
                            'confidence': float(error_conf[i])
                        })

            # Error distribution by feature (for top features)
            error_by_feature = self._analyze_error_patterns(X, errors)

            # Create visualization
            fig_path = self._create_error_analysis_chart(X, errors, preds.y_true, preds.y_pred, error_by_feature)

            content = f"""
# Error Analysis

Understanding where the model fails helps improve performance and set realistic expectations.

## Misclassification Summary

| Metric | Value |
|--------|-------|
| Total Errors | {n_errors:,} |
| Error Rate | {error_rate:.2f}% |
| Accuracy | {100 - error_rate:.2f}% |

## Confusion Matrix Breakdown

"""
            # Add confusion matrix details
            n_classes = len(cm)
            labels = [f'Class {i}' for i in range(n_classes)]
            for i in range(n_classes):
                for j in range(n_classes):
                    if i != j and cm[i, j] > 0:
                        content += f"- {labels[i]} misclassified as {labels[j]}: {cm[i, j]:,} ({cm[i, j]/cm[i].sum()*100:.1f}%)\n"

            if hardest_samples:
                content += f"""
## Hardest to Classify Samples (Most Confident Errors)

| Sample | True | Predicted | Probability | Confidence |
|--------|------|-----------|-------------|------------|
"""
                for s in hardest_samples[:top_n]:
                    content += f"| {s['index']} | {s['true']} | {s['pred']} | {s['prob']:.3f} | {s['confidence']:.3f} |\n"

            # Error clustering
            error_clusters = self._cluster_errors(X, errors)

            # Error decomposition (FP/FN)
            error_decomposition = self._compute_error_decomposition(y_true, y_pred, cm)

            # Rejection analysis
            rejection_result = {}
            if y_prob is not None:
                rejection_result = self._compute_rejection_thresholds(y_true, y_pred, y_prob)

            narrative = self._maybe_add_narrative('Error Analysis', f'Error rate: {error_rate:.2f}%, Total errors: {n_errors}', section_type='error_analysis')

            content += f"""
## Error Distribution Analysis

![Error Analysis]({fig_path})

"""

            # Error decomposition subsection
            if error_decomposition:
                content += """## Error Decomposition

| Error Type | Count | Percentage |
|-----------|-------|-----------|
"""
                for err_type, info in error_decomposition.items():
                    content += f"| {err_type} | {info['count']:,} | {info['percentage']:.1f}% |\n"

            # Error clusters subsection
            if error_clusters:
                sil_score = error_clusters[0].get('silhouette_score', 'N/A')
                sil_display = f" (silhouette={sil_score:.3f})" if isinstance(sil_score, float) else ""
                content += f"""
## Error Clusters

Adaptive clustering of misclassified samples reveals {len(error_clusters)} distinct error patterns{sil_display}:

| Cluster | Count | % of Errors | Top Distinguishing Features (effect size) |
|---------|-------|-------------|------------------------------------------|
"""
                for cluster in error_clusters:
                    top_feats = ', '.join([f"{f['feature'][:12]}(d={f.get('effect_size', f.get('centroid', 0)):.2f})"
                                          for f in cluster['top_features'][:3]])
                    content += f"| {cluster['cluster_id']} | {cluster['count']} | {cluster['percentage']:.1f}% | {top_feats} |\n"

            # Rejection analysis subsection
            if rejection_result:
                content += f"""
## Rejection Analysis

Selective prediction: reject uncertain samples to improve accuracy.

| Target Accuracy | Confidence Threshold | Rejection Rate | Achieved Accuracy |
|----------------|---------------------|---------------|-------------------|
| 95% | {rejection_result.get('threshold_95', 'N/A')} | {rejection_result.get('rejection_rate_95', 'N/A')} | {rejection_result.get('achieved_accuracy_95', 'N/A')} |
| 99% | {rejection_result.get('threshold_99', 'N/A')} | {rejection_result.get('rejection_rate_99', 'N/A')} | {rejection_result.get('achieved_accuracy_99', 'N/A')} |

"""

            content += f"""
{narrative}

---
"""
            self._content.append(content)
            self._store_section_data('error_analysis', 'Error Analysis', {
                'error_rate': error_rate,
                'n_errors': n_errors,
                'error_clusters': error_clusters,
                'error_decomposition': error_decomposition,
                'rejection': rejection_result,
            }, [{'type': 'scatter'}])

        except Exception as e:
            self._record_section_failure('Error Analysis', e)

    def _cluster_errors(self, X: pd.DataFrame, errors) -> List[Dict]:
        """Cluster misclassified samples using adaptive method selection.

        Uses PCA for dimensionality reduction, silhouette analysis for optimal k,
        DBSCAN as alternative, and t-test for discriminative features.
        """
        try:
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            from sklearn.neighbors import NearestNeighbors
            from scipy import stats as scipy_stats

            error_X = X.loc[errors].select_dtypes(include=[np.number])
            if len(error_X) < 10 or error_X.shape[1] < 2:
                return []

            feature_names = list(error_X.columns)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(error_X.fillna(0))

            # PCA when >20 features: keep components explaining 95% variance
            pca_used = False
            if X_scaled.shape[1] > 20:
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=0.95, random_state=42)
                    X_scaled = pca.fit_transform(X_scaled)
                    pca_used = True
                except Exception as e:
                    self._record_internal_warning('PCAReduction', 'PCA dimensionality reduction failed', e)
                    pass

            # Silhouette analysis for optimal k (KMeans)
            max_k = min(8, len(error_X) // 10)
            if max_k < 2:
                return []

            best_k = 2
            best_silhouette = -1.0
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_k = km.fit_predict(X_scaled)
                if len(np.unique(labels_k)) < 2:
                    continue
                sil = silhouette_score(X_scaled, labels_k)
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_k = k

            # KMeans with optimal k
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            kmeans_silhouette = best_silhouette

            # Try DBSCAN as alternative
            dbscan_labels = None
            dbscan_silhouette = -1.0
            try:
                nn = NearestNeighbors(n_neighbors=min(5, len(X_scaled) - 1))
                nn.fit(X_scaled)
                distances, _ = nn.kneighbors(X_scaled)
                eps = float(np.percentile(distances[:, -1], 90))
                if eps > 0:
                    db = DBSCAN(eps=eps, min_samples=max(3, len(X_scaled) // 20))
                    db_labels = db.fit_predict(X_scaled)
                    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                    if n_db_clusters >= 2:
                        non_noise = db_labels != -1
                        if non_noise.sum() >= 10:
                            dbscan_silhouette = silhouette_score(
                                X_scaled[non_noise], db_labels[non_noise])
                            dbscan_labels = db_labels
            except Exception as e:
                self._record_internal_warning('DBSCANClustering', 'DBSCAN clustering failed', e)
                pass

            # Choose best method
            if dbscan_labels is not None and dbscan_silhouette > kmeans_silhouette:
                labels = dbscan_labels
                cluster_ids = sorted(set(labels) - {-1})
                chosen_silhouette = dbscan_silhouette
            else:
                labels = kmeans_labels
                cluster_ids = list(range(best_k))
                chosen_silhouette = kmeans_silhouette

            # Build cluster results with discriminative features via t-test
            # Use original (non-PCA) scaled data for feature interpretation
            X_original_scaled = scaler.fit_transform(error_X.fillna(0))

            clusters = []
            for c in cluster_ids:
                mask = labels == c
                count = int(mask.sum())
                if count < 2:
                    continue

                # T-test: cluster vs rest for each feature
                cluster_data = X_original_scaled[mask]
                rest_data = X_original_scaled[~mask]

                disc_features = []
                for i, feat in enumerate(feature_names):
                    if rest_data.shape[0] < 2:
                        continue
                    try:
                        t_stat, p_val = scipy_stats.ttest_ind(
                            cluster_data[:, i], rest_data[:, i], equal_var=False)
                        # Cohen's d effect size
                        pooled_std = np.sqrt(
                            (np.var(cluster_data[:, i]) + np.var(rest_data[:, i])) / 2)
                        effect_size = float(
                            (np.mean(cluster_data[:, i]) - np.mean(rest_data[:, i])) /
                            pooled_std) if pooled_std > 0 else 0.0
                        disc_features.append({
                            'feature': feat,
                            't_stat': float(t_stat),
                            'p_value': float(p_val),
                            'effect_size': effect_size,
                            'centroid': float(np.mean(cluster_data[:, i])),
                        })
                    except Exception as e:
                        self._record_internal_warning('FeatureTTest', 'Feature t-test analysis failed', e)
                        continue

                disc_features.sort(key=lambda x: abs(x['effect_size']), reverse=True)

                clusters.append({
                    'cluster_id': c,
                    'count': count,
                    'percentage': float(count / len(error_X) * 100),
                    'silhouette_score': float(chosen_silhouette),
                    'top_features': disc_features[:5],
                })

            return clusters
        except Exception as e:
            self._record_internal_warning('ErrorClustering', 'Error clustering failed', e)
            return []

    def _compute_error_decomposition(self, y_true, y_pred, cm) -> Dict:
        """Compute FP/FN breakdown for binary; per-class error rates for multiclass."""
        decomposition = {}
        n_classes = len(cm)
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        total_errors = int((y_true_arr != y_pred_arr).sum())

        if total_errors == 0:
            return {}

        if n_classes == 2:
            fp = int(cm[0, 1])
            fn = int(cm[1, 0])
            decomposition['False Positives (Type I)'] = {
                'count': fp, 'percentage': float(fp / total_errors * 100) if total_errors > 0 else 0.0
            }
            decomposition['False Negatives (Type II)'] = {
                'count': fn, 'percentage': float(fn / total_errors * 100) if total_errors > 0 else 0.0
            }
        else:
            for i in range(n_classes):
                class_errors = int(cm[i].sum() - cm[i, i])
                if class_errors > 0:
                    decomposition[f'Class {i} errors'] = {
                        'count': class_errors,
                        'percentage': float(class_errors / total_errors * 100),
                    }

        return decomposition

    def _compute_rejection_thresholds(self, y_true, y_pred, y_prob,
                                       targets=(0.95, 0.99)) -> Dict:
        """Find confidence thresholds where accuracy reaches target levels.

        Returns dict with threshold, rejection rate, and achieved accuracy for each target.
        """
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        y_prob_arr = np.asarray(y_prob)

        # For multiclass, use max probability as confidence; for binary, distance from 0.5
        if y_prob_arr.ndim == 2:
            confidence = np.max(y_prob_arr, axis=1)
        else:
            confidence = np.abs(y_prob_arr - 0.5) * 2  # Distance from 0.5 scaled to [0, 1]
        result = {}

        for target in targets:
            target_key = str(int(target * 100))
            best_thresh = None
            best_acc = 0.0
            best_rej = 1.0

            for thresh in np.arange(0.0, 1.0, 0.05):
                accepted = confidence >= thresh
                if accepted.sum() < 5:
                    continue

                acc = float(np.mean(y_true_arr[accepted] == y_pred_arr[accepted]))
                rej_rate = float(1.0 - accepted.mean())

                if acc >= target and rej_rate < best_rej:
                    best_thresh = float(thresh)
                    best_acc = acc
                    best_rej = rej_rate

            if best_thresh is not None:
                result[f'threshold_{target_key}'] = f'{best_thresh:.2f}'
                result[f'rejection_rate_{target_key}'] = f'{best_rej:.1%}'
                result[f'achieved_accuracy_{target_key}'] = f'{best_acc:.4f}'
            else:
                result[f'threshold_{target_key}'] = 'N/A'
                result[f'rejection_rate_{target_key}'] = 'N/A'
                result[f'achieved_accuracy_{target_key}'] = 'N/A'

        return result

    def _analyze_error_patterns(self, X: pd.DataFrame, errors) -> Dict:
        """Analyze error patterns by feature."""
        patterns = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:10]

        for col in numeric_cols:
            correct_mean = X.loc[~errors, col].mean()
            error_mean = X.loc[errors, col].mean()
            diff = abs(error_mean - correct_mean) / (correct_mean + 1e-10) * 100

            patterns[col] = {
                'correct_mean': correct_mean,
                'error_mean': error_mean,
                'diff_pct': diff
            }

        return patterns

    def _create_error_analysis_chart(self, X, errors, y_true, y_pred, patterns) -> str:
        """Create error analysis visualization."""
        try:
            with self._chart_context('error_analysis', figsize=(12, 5), nrows=1, ncols=2) as (fig, axes):
                # Error rate by predicted class
                ax1 = axes[0]
                unique_pred = np.unique(y_pred)
                error_rates = []
                for cls in unique_pred:
                    mask = y_pred == cls
                    if mask.sum() > 0:
                        error_rates.append((errors[mask].sum() / mask.sum()) * 100)
                    else:
                        error_rates.append(0)

                ax1.bar(unique_pred, error_rates, color=self.theme['danger'], alpha=0.8)
                ax1.set_xlabel('Predicted Class', fontsize=11)
                ax1.set_ylabel('Error Rate (%)', fontsize=11)
                ax1.set_title('Error Rate by Predicted Class', fontsize=14, fontweight='bold', color=self.theme['primary'])

                # Feature difference in errors
                ax2 = axes[1]
                if patterns:
                    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['diff_pct'], reverse=True)[:10]
                    features = [x[0][:15] for x in sorted_patterns]
                    diffs = [x[1]['diff_pct'] for x in sorted_patterns]

                    ax2.barh(range(len(features)), diffs, color=self.theme['warning'], alpha=0.8)
                    ax2.set_yticks(range(len(features)))
                    ax2.set_yticklabels(features, fontsize=9)
                    ax2.set_xlabel('% Difference (Error vs Correct)', fontsize=11)
                    ax2.set_title('Features with Different Error Patterns', fontsize=14, fontweight='bold', color=self.theme['primary'])

            return 'figures/error_analysis.png'
        except Exception as e:
            logger.debug(f"Failed to create error analysis chart: {e}")
            return ""
