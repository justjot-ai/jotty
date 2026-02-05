"""
Visualization mixin for ProfessionalMLReport.

Provides all _create_* chart/visualization methods used by the report generator.
These methods rely on self.theme, self.figures_dir, self._record_chart_failure(),
self._save_figure(), and self._fig_path_for_markdown() from the main class.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .ml_report_generator import ReportContext

logger = logging.getLogger(__name__)


class VisualizationMixin:
    """Mixin class providing visualization/chart creation methods for ProfessionalMLReport.

    This mixin is designed to be used with ProfessionalMLReport and relies on
    attributes and methods defined in that class, such as self.theme,
    self.figures_dir, self._record_chart_failure(), self.output_dir, etc.
    """

    from contextlib import contextmanager as _contextmanager

    @_contextmanager
    def _chart_context(self, chart_name, figsize=(10, 6), nrows=1, ncols=1,
                       subplot_kw=None, gridspec_kw=None):
        """Context manager that creates a matplotlib figure, yields it, then saves and closes.

        On normal exit the figure is tight-layouted, saved to
        ``self.figures_dir/<chart_name>.png``, and closed.  On exception the
        figure is still closed (preventing leaked handles) and the exception
        propagates so the caller's ``except`` block can record the failure.

        Yields:
            (fig, axes) — axes is a single Axes when *nrows* == *ncols* == 1,
            otherwise an ndarray of Axes.
        """
        import matplotlib.pyplot as plt
        kw = {}
        if subplot_kw:
            kw['subplot_kw'] = subplot_kw
        if gridspec_kw:
            kw['gridspec_kw'] = gridspec_kw
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
        try:
            yield fig, axes
            plt.tight_layout()
            path = self.figures_dir / f'{chart_name}.png'
            plt.savefig(path, dpi=300, bbox_inches='tight',
                        facecolor=self.theme['chart_bg'])
        finally:
            plt.close(fig)

    def _create_pipeline_dag(self, pipeline_steps: List[Dict]) -> str:
        """Create pipeline flowchart using matplotlib FancyBboxPatch + FancyArrowPatch."""
        try:
            type_colors = {
                'preprocessing': self.theme['accent'],
                'feature_engineering': self.theme['success'],
                'model': self.theme['warning'],
                'ensemble': self.theme['danger'],
                'unknown': self.theme['muted'],
            }

            n_steps = len(pipeline_steps)
            fig_width = max(10, n_steps * 2.5)
            with self._chart_context('pipeline_dag', figsize=(fig_width, 4)) as (fig, ax):
                from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

                ax.set_xlim(-0.5, n_steps * 2.5 + 0.5)
                ax.set_ylim(-1, 3)
                ax.axis('off')

                box_width = 1.8
                box_height = 1.5
                y_center = 1.0

                for i, step in enumerate(pipeline_steps):
                    x = i * 2.5 + 0.5
                    name = step.get('name', f'Step {i + 1}')
                    stype = step.get('type', 'unknown')
                    color = type_colors.get(stype, type_colors['unknown'])

                    # Draw box
                    box = FancyBboxPatch(
                        (x, y_center - box_height / 2), box_width, box_height,
                        boxstyle="round,pad=0.1",
                        facecolor=color, edgecolor='white', alpha=0.85, linewidth=2
                    )
                    ax.add_patch(box)

                    # Step name
                    ax.text(x + box_width / 2, y_center + 0.15, name[:18],
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color='white')

                    # Step type
                    ax.text(x + box_width / 2, y_center - 0.25, stype[:18],
                           ha='center', va='center', fontsize=7, color='white', alpha=0.9)

                    # Shape info
                    in_shape = step.get('input_shape', None)
                    out_shape = step.get('output_shape', None)
                    if out_shape:
                        ax.text(x + box_width / 2, y_center - box_height / 2 - 0.2,
                               f'{out_shape}', ha='center', va='top', fontsize=7,
                               color=self.theme['muted'])

                    # Arrow to next step
                    if i < n_steps - 1:
                        arrow = FancyArrowPatch(
                            (x + box_width, y_center),
                            (x + 2.5 + 0.5, y_center),
                            arrowstyle='->', mutation_scale=15,
                            color=self.theme['text'], linewidth=1.5
                        )
                        ax.add_patch(arrow)

                # Legend
                legend_elements = []
                for stype, color in type_colors.items():
                    if stype != 'unknown':
                        legend_elements.append(Patch(facecolor=color, alpha=0.85, label=stype.replace('_', ' ').title()))

                ax.legend(handles=legend_elements, loc='upper center',
                         bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=9)

                ax.set_title('ML Pipeline Architecture', fontsize=14, fontweight='bold',
                            color=self.theme['primary'], pad=30)

            return 'figures/pipeline_dag.png'
        except Exception as e:
            self._record_chart_failure('pipeline_dag', e)
            return ""

    def _create_cross_dataset_chart(self, results: Dict, metric_name: str) -> str:
        """Create grouped bar chart of metrics per dataset."""
        try:
            import matplotlib.pyplot as plt

            ds_names = list(results.keys())
            scores = [results[ds]['score'] for ds in ds_names]
            n_samples = [results[ds]['n_samples'] for ds in ds_names]

            with self._chart_context('cross_dataset_validation', figsize=(12, 5), nrows=1, ncols=2) as (fig, (ax1, ax2)):
                # Score comparison
                colors = [self.theme['chart_palette'][i % len(self.theme['chart_palette'])]
                         for i in range(len(ds_names))]
                bars = ax1.bar(ds_names, scores, color=colors, alpha=0.85, edgecolor='white')

                for bar, score in zip(bars, scores):
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{score:.4f}', ha='center', va='bottom', fontsize=9, fontweight='medium')

                ax1.set_ylabel(metric_name, fontsize=11)
                ax1.set_title(f'{metric_name} by Dataset', fontsize=14, fontweight='bold',
                             color=self.theme['primary'])
                ax1.set_ylim(0, max(scores) * 1.15 if scores else 1)
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

                # Sample size comparison
                ax2.bar(ds_names, n_samples, color=self.theme['accent'], alpha=0.7)
                ax2.set_ylabel('Number of Samples', fontsize=11)
                ax2.set_title('Dataset Sizes', fontsize=14, fontweight='bold',
                             color=self.theme['primary'])
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

            return 'figures/cross_dataset_validation.png'
        except Exception as e:
            self._record_chart_failure('cross_dataset', e)
            return ""

    def _create_missing_pattern_chart(self, df: pd.DataFrame) -> str:
        """Create missing value pattern heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            missing_cols = df.columns[df.isnull().any()].tolist()
            if not missing_cols:
                return ""

            # Limit to top 20 columns with most missing
            missing_counts = df[missing_cols].isnull().sum().sort_values(ascending=False)
            top_cols = missing_counts.head(20).index.tolist()

            with self._chart_context('missing_pattern', figsize=(12, 6)) as (fig, ax):
                # Sample if too many rows
                sample_df = df[top_cols].head(200) if len(df) > 200 else df[top_cols]

                sns.heatmap(sample_df.isnull(), cmap='RdYlBu_r', cbar_kws={'label': 'Missing'},
                           yticklabels=False, ax=ax)
                ax.set_title('Missing Value Pattern', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax.set_xlabel('Features', fontsize=11)
                ax.set_ylabel('Samples', fontsize=11)
                plt.xticks(rotation=45, ha='right', fontsize=8)

            return 'figures/missing_pattern.png'
        except Exception as e:
            self._record_chart_failure('missing_pattern', e)
            return ""

    def _create_distribution_overview(self, df: pd.DataFrame) -> str:
        """Create distribution overview with histograms."""
        try:
            # Select top features to display
            n_features = min(12, len(df.columns))
            cols = df.columns[:n_features]

            n_rows = (n_features + 3) // 4
            with self._chart_context('distributions', figsize=(14, 3 * n_rows), nrows=n_rows, ncols=4) as (fig, axes):
                axes = axes.flatten() if n_features > 1 else [axes]

                for i, col in enumerate(cols):
                    ax = axes[i]
                    data = df[col].dropna()
                    ax.hist(data, bins=30, color=self.theme['accent'], alpha=0.7, edgecolor='white')
                    ax.set_title(col[:20], fontsize=10, fontweight='medium')
                    ax.tick_params(labelsize=8)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                # Hide empty subplots
                for i in range(n_features, len(axes)):
                    axes[i].set_visible(False)

                fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold', color=self.theme['primary'], y=1.02)

            return 'figures/distributions.png'
        except Exception as e:
            self._record_chart_failure('distribution_overview', e)
            return ""

    def _create_outlier_boxplot(self, df: pd.DataFrame, outlier_summary: Dict) -> str:
        """Create boxplot for features with outliers."""
        try:
            import matplotlib.pyplot as plt

            # Select features with most outliers
            outlier_cols = sorted(outlier_summary.keys(),
                                 key=lambda x: outlier_summary[x]['count'],
                                 reverse=True)[:10]

            if not outlier_cols:
                return ""

            with self._chart_context('outlier_boxplot', figsize=(12, 6)) as (fig, ax):
                # Normalize data for comparison
                plot_data = df[outlier_cols].copy()
                plot_data = (plot_data - plot_data.mean()) / plot_data.std()

                bp = ax.boxplot([plot_data[col].dropna() for col in outlier_cols],
                               labels=[c[:15] for c in outlier_cols],
                               patch_artist=True)

                colors_list = plt.cm.Blues(np.linspace(0.4, 0.8, len(outlier_cols)))
                for patch, color in zip(bp['boxes'], colors_list):
                    patch.set_facecolor(color)

                ax.set_title('Outlier Distribution (Standardized)', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax.set_ylabel('Standardized Value', fontsize=11)
                plt.xticks(rotation=45, ha='right', fontsize=9)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            return 'figures/outlier_boxplot.png'
        except Exception as e:
            self._record_chart_failure('outlier_boxplot', e)
            return ""

    def _create_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> str:
        """Create clustered correlation heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.cluster import hierarchy
            from scipy.spatial.distance import squareform

            # Limit size
            if len(corr_matrix) > 30:
                # Select most variable features
                cols = corr_matrix.columns[:30]
                corr_matrix = corr_matrix.loc[cols, cols]

            with self._chart_context('correlation_matrix', figsize=(12, 10)) as (fig, ax):
                # Cluster the correlation matrix
                try:
                    linkage = hierarchy.linkage(squareform(1 - np.abs(corr_matrix)), method='average')
                    order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage, squareform(1 - np.abs(corr_matrix))))
                    corr_matrix = corr_matrix.iloc[order, order]
                except Exception as e:
                    self._record_internal_warning('CorrelationClustering', 'correlation matrix clustering failed', e)
                    pass

                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                sns.heatmap(corr_matrix, mask=mask, cmap=self.theme['chart_diverging_cmap'], center=0,
                           annot=len(corr_matrix) <= 15, fmt='.2f', ax=ax,
                           square=True, linewidths=0.5,
                           cbar_kws={'shrink': 0.8, 'label': 'Correlation'})

                ax.set_title('Feature Correlation Matrix (Clustered)', fontsize=14, fontweight='bold', color=self.theme['primary'])
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(fontsize=8)

            return 'figures/correlation_matrix.png'
        except Exception as e:
            self._record_chart_failure('correlation_heatmap', e)
            return ""

    def _create_vif_chart(self, vif_data: Dict) -> str:
        """Create VIF bar chart."""
        try:
            if not vif_data:
                return ""

            sorted_vif = sorted(vif_data.items(), key=lambda x: x[1], reverse=True)[:20]
            features = [x[0][:20] for x in sorted_vif]
            values = [x[1] for x in sorted_vif]

            with self._chart_context('vif_analysis', figsize=(10, 6)) as (fig, ax):
                colors = ['#e53e3e' if v > 10 else ('#d69e2e' if v > 5 else '#38a169') for v in values]
                bars = ax.barh(range(len(features)), values, color=colors)

                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features, fontsize=9)
                ax.set_xlabel('Variance Inflation Factor', fontsize=11)
                ax.set_title('VIF Analysis', fontsize=14, fontweight='bold', color=self.theme['primary'])

                # Add threshold lines
                ax.axvline(x=5, color=self.theme['warning'], linestyle='--', alpha=0.7, label='Moderate (5)')
                ax.axvline(x=10, color=self.theme['danger'], linestyle='--', alpha=0.7, label='Severe (10)')
                ax.legend(loc='lower right')

            return 'figures/vif_analysis.png'
        except Exception as e:
            self._record_chart_failure('vif_chart', e)
            return ""

    def _create_learning_curve_chart(self, sizes, train_mean, train_std, val_mean, val_std) -> str:
        """Create learning curve visualization."""
        try:
            with self._chart_context('learning_curves', figsize=(10, 6)) as (fig, ax):
                # Training curve
                ax.plot(sizes, train_mean, 'o-', color=self.theme['accent'], linewidth=2, label='Training Score')
                ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                               alpha=0.2, color=self.theme['accent'])

                # Validation curve
                ax.plot(sizes, val_mean, 's-', color=self.theme['success'], linewidth=2, label='Validation Score')
                ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                               alpha=0.2, color=self.theme['success'])

                ax.set_xlabel('Training Set Size', fontsize=11)
                ax.set_ylabel('Score', fontsize=11)
                ax.set_title('Learning Curves', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])

            return 'figures/learning_curves.png'
        except Exception as e:
            self._record_chart_failure('learning_curve', e)
            return ""

    def _create_calibration_chart(self, fraction_pos, mean_pred, y_prob) -> str:
        """Create calibration curve visualization."""
        try:
            with self._chart_context('calibration', figsize=(12, 5), nrows=1, ncols=2) as (fig, (ax1, ax2)):
                # Calibration curve
                ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
                ax1.plot(mean_pred, fraction_pos, 's-', color=self.theme['accent'], linewidth=2,
                        markersize=8, label='Model')
                ax1.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.2, color=self.theme['warning'])

                ax1.set_xlabel('Mean Predicted Probability', fontsize=11)
                ax1.set_ylabel('Fraction of Positives', fontsize=11)
                ax1.set_title('Calibration Curve', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax1.legend(loc='lower right')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1])

                # Prediction histogram
                ax2.hist(y_prob, bins=30, color=self.theme['accent'], alpha=0.7, edgecolor='white')
                ax2.set_xlabel('Predicted Probability', fontsize=11)
                ax2.set_ylabel('Count', fontsize=11)
                ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold', color=self.theme['primary'])

            return 'figures/calibration.png'
        except Exception as e:
            self._record_chart_failure('calibration', e)
            return ""

    def _create_confidence_charts(self, y_true, y_prob, ece_data: Dict) -> str:
        """Create confidence vs accuracy curve and per-bin reliability bar chart."""
        try:
            with self._chart_context('confidence_analysis', figsize=(12, 5), nrows=1, ncols=2) as (fig, (ax1, ax2)):
                bins = ece_data['bins']
                confidences = [b['confidence'] for b in bins if b['count'] > 0]
                accuracies = [b['accuracy'] for b in bins if b['count'] > 0]
                counts = [b['count'] for b in bins if b['count'] > 0]
                gaps = [b['gap'] for b in bins if b['count'] > 0]

                # 1. Reliability diagram
                ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                ax1.plot(confidences, accuracies, 'o-', color=self.theme['accent'],
                        linewidth=2, markersize=8, label='Model')

                # Shade gap areas
                for conf, acc in zip(confidences, accuracies):
                    ax1.plot([conf, conf], [conf, acc], color=self.theme['danger'],
                            alpha=0.5, linewidth=1.5)

                ax1.set_xlabel('Mean Predicted Confidence', fontsize=11)
                ax1.set_ylabel('Fraction of Positives', fontsize=11)
                ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold',
                             color=self.theme['primary'])
                ax1.legend(loc='lower right')
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1])
                ax1.grid(True, alpha=0.3)

                # 2. Per-bin bar chart with gap overlay
                bin_centers = np.arange(len(confidences))
                width = 0.35

                ax2.bar(bin_centers - width / 2, accuracies, width, label='Accuracy',
                       color=self.theme['success'], alpha=0.8)
                ax2.bar(bin_centers + width / 2, confidences, width, label='Confidence',
                       color=self.theme['accent'], alpha=0.8)

                ax2.set_xlabel('Bin', fontsize=11)
                ax2.set_ylabel('Value', fontsize=11)
                ax2.set_title(f'Per-Bin Calibration (ECE={ece_data["ece"]:.4f})', fontsize=14,
                             fontweight='bold', color=self.theme['primary'])
                ax2.legend()
                ax2.set_ylim([0, 1.1])

                # Add count labels
                for i, count in enumerate(counts):
                    ax2.text(i, max(accuracies[i], confidences[i]) + 0.02,
                            f'n={count}', ha='center', fontsize=7)

            return 'figures/confidence_analysis.png'
        except Exception as e:
            self._record_chart_failure('confidence_charts', e)
            return ""

    def _create_lift_gain_chart(self, deciles, gains, lift, ks_stat, ks_idx) -> str:
        """Create lift and gain visualization."""
        try:
            with self._chart_context('lift_gain', figsize=(12, 5), nrows=1, ncols=2) as (fig, (ax1, ax2)):
                # Cumulative gains
                ax1.plot(deciles * 100, gains * 100, color=self.theme['accent'], linewidth=2, label='Model')
                ax1.plot(deciles * 100, deciles * 100, 'k--', label='Random')
                ax1.fill_between(deciles * 100, gains * 100, deciles * 100, alpha=0.2, color=self.theme['accent'])

                # KS line
                ax1.axvline(x=deciles[ks_idx] * 100, color=self.theme['danger'], linestyle=':', alpha=0.7)
                ax1.annotate(f'KS = {ks_stat:.3f}', xy=(deciles[ks_idx] * 100, (gains[ks_idx] + deciles[ks_idx]) / 2 * 100),
                            fontsize=10, color=self.theme['danger'])

                ax1.set_xlabel('% of Population', fontsize=11)
                ax1.set_ylabel('% of Positives Captured', fontsize=11)
                ax1.set_title('Cumulative Gains Curve', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax1.legend(loc='lower right')
                ax1.grid(True, alpha=0.3)

                # Lift curve
                sample_idx = np.linspace(0, len(lift) - 1, 100).astype(int)
                ax2.plot(deciles[sample_idx] * 100, lift[sample_idx], color=self.theme['success'], linewidth=2)
                ax2.axhline(y=1, color='k', linestyle='--', label='Random (Lift=1)')
                ax2.fill_between(deciles[sample_idx] * 100, lift[sample_idx], 1, alpha=0.2, color=self.theme['success'])

                ax2.set_xlabel('% of Population', fontsize=11)
                ax2.set_ylabel('Lift', fontsize=11)
                ax2.set_title('Lift Curve', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)

            return 'figures/lift_gain.png'
        except Exception as e:
            self._record_chart_failure('lift_gain', e)
            return ""

    def _create_cv_chart(self, cv_results: Dict) -> str:
        """Create cross-validation visualization."""
        try:
            with self._chart_context('cv_analysis', figsize=(12, 5), nrows=1, ncols=2) as (fig, (ax1, ax2)):
                n_folds = len(cv_results['test_accuracy'])
                folds = range(1, n_folds + 1)

                # Fold comparison
                width = 0.35
                ax1.bar([x - width/2 for x in folds], cv_results['train_accuracy'], width,
                       label='Train', color=self.theme['accent'], alpha=0.8)
                ax1.bar([x + width/2 for x in folds], cv_results['test_accuracy'], width,
                       label='Test', color=self.theme['success'], alpha=0.8)

                ax1.set_xlabel('Fold', fontsize=11)
                ax1.set_ylabel('Accuracy', fontsize=11)
                ax1.set_title('Accuracy by Fold', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax1.set_xticks(folds)
                ax1.legend()
                ax1.set_ylim([0, 1.05])

                # Box plot of metrics
                metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
                data = [cv_results[f'test_{m}'] for m in metrics]

                bp = ax2.boxplot(data, labels=['Accuracy', 'Precision', 'Recall', 'F1'],
                                patch_artist=True)

                colors_list = [self.theme['accent'], self.theme['success'], self.theme['warning'], self.theme['info']]
                for patch, color in zip(bp['boxes'], colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax2.set_ylabel('Score', fontsize=11)
                ax2.set_title('Metric Distribution', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax2.set_ylim([0, 1.05])

            return 'figures/cv_analysis.png'
        except Exception as e:
            self._record_chart_failure('cv_chart', e)
            return ""

    def _create_shap_summary(self, shap_values, feature_names, X_sample) -> str:
        """Create SHAP summary plot."""
        try:
            import matplotlib.pyplot as plt
            import shap

            # Extract values for binary classification (use positive class)
            plot_values = shap_values
            if hasattr(shap_values, 'values'):
                vals = shap_values.values
                if len(vals.shape) == 3:
                    plot_values = vals[:, :, 1]
                else:
                    plot_values = vals

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(plot_values, X_sample, feature_names=feature_names,
                            show=False, max_display=20)
            plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', color=self.theme['primary'])
            plt.tight_layout()

            path = self.figures_dir / 'shap_summary.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/shap_summary.png'
        except Exception as e:
            self._record_chart_failure('shap_summary', e)
            return ""

    def _create_shap_bar(self, shap_importance: List[Tuple]) -> str:
        """Create SHAP bar plot."""
        try:
            import matplotlib.pyplot as plt

            top_n = min(20, len(shap_importance))
            features = [x[0][:25] for x in shap_importance[:top_n]]
            values = [x[1] for x in shap_importance[:top_n]]

            with self._chart_context('shap_bar', figsize=(10, 8)) as (fig, ax):
                colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(features)))[::-1]
                ax.barh(range(len(features)), values[::-1], color=colors)

                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features[::-1], fontsize=9)
                ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
                ax.set_title('SHAP Feature Importance', fontsize=14, fontweight='bold', color=self.theme['primary'])

                # Add value labels
                for i, val in enumerate(values[::-1]):
                    ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

            return 'figures/shap_bar.png'
        except Exception as e:
            self._record_chart_failure('shap_bar', e)
            return ""

    def _create_shap_dependence(self, shap_values, feature_names, X_sample, top_n: int) -> str:
        """Create SHAP dependence plots."""
        try:
            import shap

            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values

            if len(values.shape) == 3:
                values = values[:, :, 1]

            mean_shap = np.abs(values).mean(axis=0)
            top_idx = np.argsort(mean_shap)[::-1][:top_n]

            with self._chart_context('shap_dependence', figsize=(5 * top_n, 4), nrows=1, ncols=top_n) as (fig, axes):
                if top_n == 1:
                    axes = [axes]

                for i, idx in enumerate(top_idx):
                    ax = axes[i]
                    feat_name = feature_names[idx]
                    feat_values = X_sample[:, idx] if isinstance(X_sample, np.ndarray) else X_sample.iloc[:, idx]
                    shap_vals = values[:, idx]

                    scatter = ax.scatter(feat_values, shap_vals, c=feat_values, cmap='coolwarm', alpha=0.6, s=20)
                    ax.set_xlabel(feat_name[:20], fontsize=10)
                    ax.set_ylabel('SHAP Value', fontsize=10)
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                    ax.set_title(f'{feat_name[:20]}', fontsize=11, fontweight='medium')

                fig.suptitle('SHAP Dependence Plots', fontsize=14, fontweight='bold', color=self.theme['primary'], y=1.02)

            return 'figures/shap_dependence.png'
        except Exception as e:
            self._record_chart_failure('shap_dependence', e)
            return ""

    def _create_shap_waterfall(self, shap_values, feature_names, X_sample) -> str:
        """Create SHAP waterfall for a sample."""
        try:
            import matplotlib.pyplot as plt
            import shap

            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(shap_values[0], max_display=15, show=False)

            plt.tight_layout()
            path = self.figures_dir / 'shap_waterfall.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/shap_waterfall.png'
        except Exception as e:
            self._record_chart_failure('shap_waterfall', e)
            return ""

    def _create_threshold_chart(self, results: List[Dict], best_f1_thresh: float, best_cost_thresh: float) -> str:
        """Create threshold analysis visualization."""
        try:
            with self._chart_context('threshold_optimization', figsize=(12, 5), nrows=1, ncols=2) as (fig, (ax1, ax2)):
                thresholds = [r['threshold'] for r in results]
                precision = [r['precision'] for r in results]
                recall = [r['recall'] for r in results]
                f1 = [r['f1'] for r in results]
                cost = [r['cost'] for r in results]

                # Precision/Recall/F1 vs threshold
                ax1.plot(thresholds, precision, 'o-', color=self.theme['accent'], linewidth=2, label='Precision')
                ax1.plot(thresholds, recall, 's-', color=self.theme['success'], linewidth=2, label='Recall')
                ax1.plot(thresholds, f1, '^-', color=self.theme['warning'], linewidth=2, label='F1 Score')

                ax1.axvline(x=best_f1_thresh, color=self.theme['danger'], linestyle='--', alpha=0.7, label=f'Best F1 ({best_f1_thresh:.2f})')

                ax1.set_xlabel('Threshold', fontsize=11)
                ax1.set_ylabel('Score', fontsize=11)
                ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim([0, 1])
                ax1.set_ylim([0, 1.05])

                # Cost vs threshold
                ax2.plot(thresholds, cost, 'o-', color=self.theme['danger'], linewidth=2)
                ax2.fill_between(thresholds, cost, alpha=0.2, color=self.theme['danger'])
                ax2.axvline(x=best_cost_thresh, color=self.theme['success'], linestyle='--', alpha=0.7, label=f'Min Cost ({best_cost_thresh:.2f})')

                ax2.set_xlabel('Threshold', fontsize=11)
                ax2.set_ylabel('Total Cost', fontsize=11)
                ax2.set_title('Cost vs Threshold', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)

            return 'figures/threshold_optimization.png'
        except Exception as e:
            self._record_chart_failure('threshold_chart', e)
            return ""

    def _create_hyperparameter_charts(self, trials: List[Dict], param_names: List[str],
                                       objective_name: str) -> str:
        """Create 1x3 subplot: parallel coordinates, importance bar, optimization history."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection

            with self._chart_context('hyperparameter_analysis', figsize=(18, 6), nrows=1, ncols=3) as (fig, axes):
                values = [t['value'] for t in trials]
                norm_values = np.array(values)
                if norm_values.max() != norm_values.min():
                    norm_values = (norm_values - norm_values.min()) / (norm_values.max() - norm_values.min())
                else:
                    norm_values = np.ones_like(norm_values) * 0.5

                # 1. Parallel Coordinates
                ax1 = axes[0]
                n_params = len(param_names)

                # Normalize parameter values to [0, 1]
                param_data = {}
                for pname in param_names:
                    raw = [t['params'].get(pname, 0) for t in trials]
                    # Handle non-numeric
                    try:
                        raw_arr = np.array(raw, dtype=float)
                        if raw_arr.max() != raw_arr.min():
                            param_data[pname] = (raw_arr - raw_arr.min()) / (raw_arr.max() - raw_arr.min())
                        else:
                            param_data[pname] = np.ones_like(raw_arr) * 0.5
                    except (ValueError, TypeError):
                        # Categorical - encode
                        unique_vals = sorted(set(str(v) for v in raw))
                        val_map = {v: i / max(1, len(unique_vals) - 1) for i, v in enumerate(unique_vals)}
                        param_data[pname] = np.array([val_map.get(str(v), 0.5) for v in raw])

                cmap = plt.cm.RdYlGn
                for i in range(len(trials)):
                    coords = [param_data[pname][i] for pname in param_names]
                    color = cmap(norm_values[i])
                    ax1.plot(range(n_params), coords, alpha=0.4, color=color, linewidth=1)

                ax1.set_xticks(range(n_params))
                ax1.set_xticklabels([p[:12] for p in param_names], rotation=45, ha='right', fontsize=8)
                ax1.set_ylabel('Normalized Value', fontsize=10)
                ax1.set_title('Parallel Coordinates', fontsize=12, fontweight='bold',
                             color=self.theme['primary'])

                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(values), max(values)))
                fig.colorbar(sm, ax=ax1, label=objective_name, shrink=0.8)

                # 2. Parameter Importance (variance-based)
                ax2 = axes[1]
                importances = {}
                for pname in param_names:
                    param_vals = param_data.get(pname, np.zeros(len(trials)))
                    # Correlation with objective as proxy for importance
                    if np.std(param_vals) > 0:
                        corr = abs(np.corrcoef(param_vals, values)[0, 1])
                        importances[pname] = corr if not np.isnan(corr) else 0
                    else:
                        importances[pname] = 0

                sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                imp_names = [x[0][:15] for x in sorted_imp]
                imp_vals = [x[1] for x in sorted_imp]

                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp_names)))[::-1]
                ax2.barh(range(len(imp_names)), imp_vals[::-1], color=colors)
                ax2.set_yticks(range(len(imp_names)))
                ax2.set_yticklabels(imp_names[::-1], fontsize=9)
                ax2.set_xlabel('Importance (|correlation|)', fontsize=10)
                ax2.set_title('Parameter Importance', fontsize=12, fontweight='bold',
                             color=self.theme['primary'])

                # 3. Optimization History
                ax3 = axes[2]
                ax3.scatter(range(len(values)), values, alpha=0.5, s=15,
                           color=self.theme['accent'], label='Trial')

                # Best-so-far overlay
                best_so_far = np.maximum.accumulate(values)
                ax3.plot(range(len(values)), best_so_far, color=self.theme['danger'],
                        linewidth=2, label='Best So Far')

                ax3.set_xlabel('Trial Number', fontsize=10)
                ax3.set_ylabel(objective_name, fontsize=10)
                ax3.set_title('Optimization History', fontsize=12, fontweight='bold',
                             color=self.theme['primary'])
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)

            return 'figures/hyperparameter_analysis.png'
        except Exception as e:
            self._record_chart_failure('hyperparameter_charts', e)
            return ""

    def _create_gauge_chart(self, ax, value: float, title: str):
        """Create a single donut/arc gauge chart on given axes."""
        # Color based on value
        if value > 0.9:
            color = self.theme['success']
        elif value > 0.7:
            color = self.theme['warning']
        else:
            color = self.theme['danger']

        # Create donut arc
        theta1 = 180  # Start angle (left)
        theta2 = 180 - (value * 180)  # End angle based on value

        # Background arc (full semicircle)
        bg_theta = np.linspace(0, np.pi, 100)
        bg_x = np.cos(bg_theta)
        bg_y = np.sin(bg_theta)
        ax.fill_between(bg_x, bg_y * 0.6, bg_y, alpha=0.15, color='gray')

        # Value arc
        val_theta = np.linspace(0, np.pi * value, 100)
        val_x = np.cos(val_theta)
        val_y = np.sin(val_theta)
        ax.fill_between(val_x, val_y * 0.6, val_y, alpha=0.8, color=color)

        # Center text
        ax.text(0, 0.3, f'{value:.1%}', ha='center', va='center',
                fontsize=18, fontweight='bold', color=color)
        ax.text(0, -0.1, title.upper(), ha='center', va='center',
                fontsize=9, fontweight='medium', color=self.theme['text'])

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

    def _create_executive_dashboard_chart(self, kpis: Dict[str, float]) -> str:
        """Create the executive dashboard with gauge charts for all KPIs."""
        try:
            n_kpis = len(kpis)
            with self._chart_context('executive_dashboard', figsize=(4 * n_kpis, 3.5), nrows=1, ncols=n_kpis) as (fig, axes):
                if n_kpis == 1:
                    axes = [axes]

                for ax, (name, value) in zip(axes, kpis.items()):
                    self._create_gauge_chart(ax, value, name)

                fig.suptitle('Key Performance Indicators', fontsize=14, fontweight='bold',
                            color=self.theme['primary'], y=1.05)

            return 'figures/executive_dashboard.png'
        except Exception as e:
            self._record_chart_failure('executive_dashboard', e)
            return ""

    def _create_class_distribution_chart(self, labels: List[str], counts) -> str:
        """Create class distribution bar chart."""
        try:
            with self._chart_context('class_distribution', figsize=(12, 5), nrows=1, ncols=2) as (fig, (ax1, ax2)):
                # Bar chart
                colors = [self.theme['chart_palette'][i % len(self.theme['chart_palette'])]
                         for i in range(len(labels))]
                bars = ax1.bar(labels, counts, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

                ax1.set_xlabel('Class', fontsize=11)
                ax1.set_ylabel('Count', fontsize=11)
                ax1.set_title('Class Distribution', fontsize=14, fontweight='bold', color=self.theme['primary'])

                # Add value labels
                for bar, count in zip(bars, counts):
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                            f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='medium')

                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)

                # Pie chart
                explode = [0.05] * len(labels)
                ax2.pie(counts, labels=labels, autopct='%1.1f%%', explode=explode,
                       colors=colors, startangle=90, textprops={'fontsize': 10})
                ax2.set_title('Class Proportions', fontsize=14, fontweight='bold', color=self.theme['primary'])

            return 'figures/class_distribution.png'
        except Exception as e:
            self._record_chart_failure('class_distribution', e)
            return ""

    def _create_permutation_importance_chart(self, result, feature_names: List[str],
                                              top_idx) -> str:
        """Create permutation importance chart with error bars."""
        try:
            n = len(top_idx)
            with self._chart_context('permutation_importance', figsize=(10, max(6, n * 0.4))) as (fig, ax):
                names = [feature_names[i][:25] for i in top_idx][::-1]
                means = result.importances_mean[top_idx][::-1]
                stds = result.importances_std[top_idx][::-1]

                colors = [self.theme['success'] if m > 0 else self.theme['danger'] for m in means]

                ax.barh(range(n), means, xerr=stds, color=colors, alpha=0.8,
                       capsize=3, edgecolor='white', linewidth=0.5)

                ax.set_yticks(range(n))
                ax.set_yticklabels(names, fontsize=9)
                ax.set_xlabel('Mean Importance Decrease', fontsize=11)
                ax.set_title('Permutation Feature Importance', fontsize=14, fontweight='bold',
                            color=self.theme['primary'])
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

            return 'figures/permutation_importance.png'
        except Exception as e:
            self._record_chart_failure('permutation_importance', e)
            return ""

    def _create_pdp_chart(self, model, X, top_features: List[str],
                           all_feature_names: List[str]) -> str:
        """Create PDP charts with ICE background lines."""
        try:
            from sklearn.inspection import partial_dependence

            n = len(top_features)
            with self._chart_context('pdp_ice', figsize=(5 * n, 4.5), nrows=1, ncols=n) as (fig, axes):
                if n == 1:
                    axes = [axes]

                X_array = X.values if hasattr(X, 'values') else X

                for i, feat_name in enumerate(top_features):
                    ax = axes[i]
                    feat_idx = all_feature_names.index(feat_name) if feat_name in all_feature_names else i

                    # Compute PDP
                    pdp_result = partial_dependence(
                        model, X, features=[feat_idx], kind='both',
                        grid_resolution=50
                    )

                    # ICE lines (individual)
                    if 'individual' in pdp_result:
                        ice_values = pdp_result['individual'][0]
                        grid_values = pdp_result['grid_values'][0]

                        # Plot ICE lines (subsample for clarity)
                        n_ice = min(50, ice_values.shape[0])
                        ice_sample_idx = np.random.choice(ice_values.shape[0], n_ice, replace=False)
                        for idx in ice_sample_idx:
                            ax.plot(grid_values, ice_values[idx], color=self.theme['accent'],
                                   alpha=0.1, linewidth=0.5)

                    # PDP line (average)
                    avg_values = pdp_result['average'][0]
                    grid_values = pdp_result['grid_values'][0]
                    ax.plot(grid_values, avg_values, color=self.theme['primary'],
                           linewidth=2.5, label='PDP (Average)')

                    ax.set_xlabel(feat_name[:20], fontsize=10)
                    ax.set_ylabel('Partial Dependence', fontsize=10)
                    ax.set_title(f'{feat_name[:20]}', fontsize=11, fontweight='medium')
                    ax.legend(fontsize=8, loc='best')

                fig.suptitle('Partial Dependence Plots with ICE', fontsize=14, fontweight='bold',
                            color=self.theme['primary'], y=1.05)

            return 'figures/pdp_ice.png'
        except Exception as e:
            self._record_chart_failure('pdp_chart', e)
            return ""

    def _create_bootstrap_auc_chart(self, boot_accuracies: List[float],
                                     auc_data: Dict = None) -> str:
        """Create bootstrap distribution histogram with CI bands."""
        try:
            n_plots = 2 if auc_data else 1
            with self._chart_context('bootstrap_analysis', figsize=(6 * n_plots, 5), nrows=1, ncols=n_plots) as (fig, axes):
                if n_plots == 1:
                    axes = [axes]

                # Accuracy distribution
                ax1 = axes[0]
                ax1.hist(boot_accuracies, bins=40, color=self.theme['accent'], alpha=0.7,
                        edgecolor='white', density=True)
                acc_ci_lower = np.percentile(boot_accuracies, 2.5)
                acc_ci_upper = np.percentile(boot_accuracies, 97.5)
                ax1.axvline(np.mean(boot_accuracies), color=self.theme['primary'],
                           linestyle='-', linewidth=2, label=f'Mean: {np.mean(boot_accuracies):.4f}')
                ax1.axvline(acc_ci_lower, color=self.theme['danger'], linestyle='--',
                           linewidth=1.5, label=f'95% CI: [{acc_ci_lower:.4f}, {acc_ci_upper:.4f}]')
                ax1.axvline(acc_ci_upper, color=self.theme['danger'], linestyle='--', linewidth=1.5)
                ax1.axvspan(acc_ci_lower, acc_ci_upper, alpha=0.15, color=self.theme['danger'])
                ax1.set_xlabel('Accuracy', fontsize=11)
                ax1.set_ylabel('Density', fontsize=11)
                ax1.set_title('Bootstrap Accuracy Distribution', fontsize=14, fontweight='bold',
                             color=self.theme['primary'])
                ax1.legend(fontsize=9)

                # AUC distribution
                if auc_data and n_plots > 1:
                    ax2 = axes[1]
                    ax2.hist(auc_data['values'], bins=40, color=self.theme['success'], alpha=0.7,
                            edgecolor='white', density=True)
                    ax2.axvline(auc_data['mean'], color=self.theme['primary'], linestyle='-',
                               linewidth=2, label=f'Mean: {auc_data["mean"]:.4f}')
                    ax2.axvline(auc_data['ci_lower'], color=self.theme['danger'], linestyle='--',
                               linewidth=1.5, label=f'95% CI: [{auc_data["ci_lower"]:.4f}, {auc_data["ci_upper"]:.4f}]')
                    ax2.axvline(auc_data['ci_upper'], color=self.theme['danger'], linestyle='--', linewidth=1.5)
                    ax2.axvspan(auc_data['ci_lower'], auc_data['ci_upper'], alpha=0.15, color=self.theme['danger'])
                    ax2.set_xlabel('AUC-ROC', fontsize=11)
                    ax2.set_ylabel('Density', fontsize=11)
                    ax2.set_title('Bootstrap AUC Distribution', fontsize=14, fontweight='bold',
                                 color=self.theme['primary'])
                    ax2.legend(fontsize=9)

            return 'figures/bootstrap_analysis.png'
        except Exception as e:
            self._record_chart_failure('bootstrap_auc', e)
            return ""

    def _create_score_distribution_chart(self, class_probs: Dict[str, np.ndarray],
                                          optimal_threshold: float) -> str:
        """Create score distribution chart with overlap shading."""
        try:
            from scipy.stats import gaussian_kde

            with self._chart_context('score_distribution', figsize=(12, 5), nrows=1, ncols=2) as (fig, (ax1, ax2)):
                colors = [self.theme['chart_palette'][i % len(self.theme['chart_palette'])]
                         for i in range(len(class_probs))]

                # Histogram
                for (label, probs), color in zip(class_probs.items(), colors):
                    ax1.hist(probs, bins=40, alpha=0.5, color=color, label=label,
                            density=True, edgecolor='white')

                ax1.axvline(optimal_threshold, color=self.theme['danger'], linestyle='--',
                           linewidth=2, label=f'Threshold: {optimal_threshold:.3f}')
                ax1.set_xlabel('Predicted Probability', fontsize=11)
                ax1.set_ylabel('Density', fontsize=11)
                ax1.set_title('Score Histogram by Class', fontsize=14, fontweight='bold',
                             color=self.theme['primary'])
                ax1.legend(fontsize=9)

                # KDE with overlap shading
                x_range = np.linspace(0, 1, 200)
                kde_curves = {}

                for (label, probs), color in zip(class_probs.items(), colors):
                    if len(probs) > 2:
                        try:
                            kde = gaussian_kde(probs, bw_method=0.1)
                            kde_values = kde(x_range)
                            ax2.plot(x_range, kde_values, color=color, linewidth=2, label=label)
                            ax2.fill_between(x_range, kde_values, alpha=0.2, color=color)
                            kde_curves[label] = kde_values
                        except Exception as e:
                            self._record_internal_warning('KDEComputation', 'KDE computation failed', e)
                            pass

                # Shade overlap region
                if len(kde_curves) == 2:
                    curves = list(kde_curves.values())
                    overlap = np.minimum(curves[0], curves[1])
                    ax2.fill_between(x_range, overlap, alpha=0.4, color=self.theme['warning'],
                                   label='Overlap', hatch='//')

                ax2.axvline(optimal_threshold, color=self.theme['danger'], linestyle='--',
                           linewidth=2, label=f'Threshold: {optimal_threshold:.3f}')
                ax2.set_xlabel('Predicted Probability', fontsize=11)
                ax2.set_ylabel('Density', fontsize=11)
                ax2.set_title('KDE with Overlap Region', fontsize=14, fontweight='bold',
                             color=self.theme['primary'])
                ax2.legend(fontsize=9)

            return 'figures/score_distribution.png'
        except Exception as e:
            self._record_chart_failure('score_distribution', e)
            return ""

    def _create_training_curves(self, history: Dict) -> str:
        """Create training curves with loss/val_loss and early stopping annotation."""
        try:
            has_accuracy = 'accuracy' in history
            n_cols = 2 if has_accuracy else 1
            fig_width = 12 if has_accuracy else 8
            with self._chart_context('training_curves', figsize=(fig_width, 5), nrows=1, ncols=n_cols) as (fig, axes):
                if not isinstance(axes, np.ndarray):
                    axes = [axes]

                # Loss curves
                ax1 = axes[0]
                loss = history.get('loss', [])
                val_loss = history.get('val_loss', [])
                epochs = range(1, len(loss) + 1)

                ax1.plot(epochs, loss, color=self.theme['accent'], linewidth=2, label='Training Loss')
                if val_loss:
                    ax1.plot(epochs, val_loss, color=self.theme['danger'], linewidth=2,
                            label='Validation Loss')

                    # Best epoch annotation
                    best_epoch = int(np.argmin(val_loss)) + 1
                    best_val = min(val_loss)
                    ax1.axvline(x=best_epoch, color=self.theme['success'], linestyle='--',
                               alpha=0.7, label=f'Best Epoch ({best_epoch})')
                    ax1.scatter([best_epoch], [best_val], color=self.theme['success'],
                               s=100, zorder=5, marker='*')

                ax1.set_xlabel('Epoch', fontsize=11)
                ax1.set_ylabel('Loss', fontsize=11)
                ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold',
                             color=self.theme['primary'])
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)

                # Accuracy curves (if available)
                if has_accuracy and len(axes) > 1:
                    ax2 = axes[1]
                    acc = history.get('accuracy', [])
                    val_acc = history.get('val_accuracy', [])

                    ax2.plot(range(1, len(acc) + 1), acc, color=self.theme['accent'],
                            linewidth=2, label='Training Accuracy')
                    if val_acc:
                        ax2.plot(range(1, len(val_acc) + 1), val_acc, color=self.theme['danger'],
                                linewidth=2, label='Validation Accuracy')

                    ax2.set_xlabel('Epoch', fontsize=11)
                    ax2.set_ylabel('Accuracy', fontsize=11)
                    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold',
                                 color=self.theme['primary'])
                    ax2.legend(fontsize=9)
                    ax2.grid(True, alpha=0.3)

            return 'figures/training_curves.png'
        except Exception as e:
            self._record_chart_failure('training_curves', e)
            return ""

    def _create_regression_charts(self, y_true, y_pred, residuals) -> str:
        """Create 2x2 regression diagnostic plots."""
        try:
            from scipy import stats as scipy_stats

            with self._chart_context('regression_diagnostics', figsize=(12, 10), nrows=2, ncols=2) as (fig, axes):
                # 1. Predicted vs Actual
                ax1 = axes[0, 0]
                ax1.scatter(y_true, y_pred, alpha=0.5, s=20, color=self.theme['accent'])
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Perfect')
                ax1.set_xlabel('Actual', fontsize=11)
                ax1.set_ylabel('Predicted', fontsize=11)
                ax1.set_title('Predicted vs Actual', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax1.legend()

                # 2. Residual Plot
                ax2 = axes[0, 1]
                ax2.scatter(y_pred, residuals, alpha=0.5, s=20, color=self.theme['success'])
                ax2.axhline(0, color='k', linestyle='--', linewidth=1.5)
                ax2.set_xlabel('Predicted', fontsize=11)
                ax2.set_ylabel('Residuals', fontsize=11)
                ax2.set_title('Residual Plot', fontsize=14, fontweight='bold', color=self.theme['primary'])

                # 3. Q-Q Plot
                ax3 = axes[1, 0]
                sorted_residuals = np.sort(residuals)
                n = len(sorted_residuals)
                theoretical_quantiles = scipy_stats.norm.ppf(np.linspace(0.01, 0.99, n))
                standardized_residuals = (sorted_residuals - sorted_residuals.mean()) / sorted_residuals.std()

                ax3.scatter(theoretical_quantiles, standardized_residuals[:len(theoretical_quantiles)],
                           alpha=0.5, s=20, color=self.theme['warning'])
                lim = max(abs(theoretical_quantiles.min()), abs(theoretical_quantiles.max()))
                ax3.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1.5)
                ax3.set_xlabel('Theoretical Quantiles', fontsize=11)
                ax3.set_ylabel('Standardized Residuals', fontsize=11)
                ax3.set_title('Q-Q Plot', fontsize=14, fontweight='bold', color=self.theme['primary'])

                # 4. Residual Histogram
                ax4 = axes[1, 1]
                ax4.hist(residuals, bins=40, color=self.theme['accent'], alpha=0.7,
                        edgecolor='white', density=True)

                # Overlay normal distribution
                x_range = np.linspace(residuals.min(), residuals.max(), 100)
                normal_pdf = scipy_stats.norm.pdf(x_range, residuals.mean(), residuals.std())
                ax4.plot(x_range, normal_pdf, color=self.theme['danger'], linewidth=2, label='Normal')

                ax4.set_xlabel('Residual Value', fontsize=11)
                ax4.set_ylabel('Density', fontsize=11)
                ax4.set_title('Residual Distribution', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax4.legend()

            return 'figures/regression_diagnostics.png'
        except Exception as e:
            self._record_chart_failure('regression_charts', e)
            return ""

    def _create_plotly_chart(self, chart_type: str, section_data: Dict, chart_id: str) -> str:
        """Map chart types to Plotly.js graph objects. Returns JS code string or empty."""
        try:
            data = section_data.get('data', {})
            title = section_data.get('title', '')
            t = self.theme

            if chart_type == 'bar' and 'results' in data:
                # Cross-dataset or benchmarking bar chart
                results = data['results']
                names = list(results.keys())
                values = [results[n].get('score', 0) for n in names]

                return f"""
Plotly.newPlot('{chart_id}', [{{
    x: {names},
    y: {values},
    type: 'bar',
    marker: {{color: '{t["accent"]}'}},
}}], {{
    title: '{title}',
    yaxis: {{title: 'Score'}},
    paper_bgcolor: '{t["chart_bg"]}',
    plot_bgcolor: 'white',
}});"""

            elif chart_type == 'heatmap' and 'drift_results' in data:
                # Drift heatmap
                results = data['drift_results']
                features = [r['feature'][:20] for r in results[:20]]
                psi = [r['psi'] for r in results[:20]]
                ks = [r['ks_stat'] for r in results[:20]]

                return f"""
Plotly.newPlot('{chart_id}', [{{
    z: [{psi}, {ks}],
    x: {features},
    y: ['PSI', 'KS Stat'],
    type: 'heatmap',
    colorscale: 'RdYlGn_r',
}}], {{
    title: '{title}',
    paper_bgcolor: '{t["chart_bg"]}',
}});"""

            elif chart_type == 'radar' and 'metrics' in data:
                # Fairness radar
                metrics = data['metrics']
                if metrics:
                    first_feat = list(metrics.keys())[0]
                    groups = {k: v for k, v in metrics[first_feat].items() if k != '_aggregate'}
                    traces = []
                    categories = ['Pos Rate', 'TPR', 'FPR', 'PPV', 'TNR', 'Accuracy']
                    cat_keys = ['positive_rate', 'tpr', 'fpr', 'ppv', 'tnr', 'accuracy']

                    for group_name, m in groups.items():
                        vals = [m.get(k, 0.0) for k in cat_keys]
                        traces.append(f"""{{
    type: 'scatterpolar',
    r: {vals + [vals[0]]},
    theta: {categories + [categories[0]]},
    fill: 'toself',
    name: '{group_name}',
}}""")

                    return f"""
Plotly.newPlot('{chart_id}', [{','.join(traces)}], {{
    title: '{title}',
    polar: {{radialaxis: {{visible: true, range: [0, 1]}}}},
    paper_bgcolor: '{t["chart_bg"]}',
}});"""

            elif chart_type == 'line' and 'training_history' in data:
                # Training curves
                history = data.get('training_history', {})
                if history:
                    loss = history.get('loss', [])
                    val_loss = history.get('val_loss', [])
                    epochs = list(range(1, len(loss) + 1))

                    traces = [f"""{{
    x: {epochs},
    y: {loss},
    mode: 'lines',
    name: 'Training Loss',
    line: {{color: '{t["accent"]}'}},
}}"""]
                    if val_loss:
                        traces.append(f"""{{
    x: {list(range(1, len(val_loss) + 1))},
    y: {val_loss},
    mode: 'lines',
    name: 'Validation Loss',
    line: {{color: '{t["danger"]}'}},
}}""")

                    return f"""
Plotly.newPlot('{chart_id}', [{','.join(traces)}], {{
    title: '{title}',
    xaxis: {{title: 'Epoch'}},
    yaxis: {{title: 'Loss'}},
    paper_bgcolor: '{t["chart_bg"]}',
    plot_bgcolor: 'white',
}});"""

            elif chart_type == 'confusion_heatmap' and 'cm' in data:
                # Confusion matrix heatmap
                cm = data['cm']
                labels = data.get('labels', [f'Class {i}' for i in range(len(cm))])
                n = len(cm)
                # Build annotation text
                annotations = []
                for i in range(n):
                    for j in range(n):
                        annotations.append(f"""{{
    x: {j}, y: {i}, text: '{cm[i][j]}',
    font: {{color: '{'white' if cm[i][j] > max(max(row) for row in cm) / 2 else 'black'}'}},
    showarrow: false,
}}""")
                return f"""
Plotly.newPlot('{chart_id}', [{{
    z: {cm},
    x: {labels},
    y: {labels},
    type: 'heatmap',
    colorscale: 'Blues',
    showscale: true,
}}], {{
    title: '{title}',
    xaxis: {{title: 'Predicted'}},
    yaxis: {{title: 'Actual', autorange: 'reversed'}},
    annotations: [{','.join(annotations)}],
    paper_bgcolor: '{t["chart_bg"]}',
}});"""

            elif chart_type == 'importance_bar':
                # Feature importance horizontal bar chart
                importance = data.get('mean_shap') or data.get('importance') or data.get('importance_values')
                if importance and isinstance(importance, dict):
                    sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
                    names = [item[0] for item in sorted_items][::-1]
                    values = [round(item[1], 4) for item in sorted_items][::-1]
                    return f"""
Plotly.newPlot('{chart_id}', [{{
    y: {names},
    x: {values},
    type: 'bar',
    orientation: 'h',
    marker: {{color: '{t["accent"]}'}},
}}], {{
    title: '{title}',
    xaxis: {{title: 'Importance'}},
    margin: {{l: 200}},
    paper_bgcolor: '{t["chart_bg"]}',
    plot_bgcolor: 'white',
}});"""

            elif chart_type == 'scatter':
                # Scatter plot: x_values, y_values, optional color_values
                x_values = data.get('x_values', [])
                y_values = data.get('y_values', [])
                color_values = data.get('color_values')
                x_label = data.get('x_label', 'X')
                y_label = data.get('y_label', 'Y')

                if not x_values and 'error_rate' in data:
                    # Auto-generate from error analysis data
                    pass

                if x_values and y_values:
                    marker_config = f"color: '{t['accent']}'"
                    if color_values:
                        marker_config = f"color: {color_values}, colorscale: 'RdYlGn', showscale: true"

                    return f"""
Plotly.newPlot('{chart_id}', [{{
    x: {x_values},
    y: {y_values},
    mode: 'markers',
    type: 'scatter',
    marker: {{{marker_config}, size: 6, opacity: 0.7}},
}}], {{
    title: '{title}',
    xaxis: {{title: '{x_label}'}},
    yaxis: {{title: '{y_label}'}},
    paper_bgcolor: '{t["chart_bg"]}',
    plot_bgcolor: 'white',
}});"""

            elif chart_type == 'box':
                # Box plot: groups dict {group_name: [values]}
                groups = data.get('groups', {})
                if not groups and 'metrics' in data:
                    # Auto-generate from fairness metrics (score distribution per group)
                    pass

                if groups:
                    traces = []
                    palette = t['chart_palette']
                    for i, (group_name, values) in enumerate(groups.items()):
                        color = palette[i % len(palette)]
                        traces.append(f"""{{
    y: {values},
    type: 'box',
    name: '{group_name}',
    marker: {{color: '{color}'}},
    boxpoints: 'outliers',
}}""")

                    return f"""
Plotly.newPlot('{chart_id}', [{','.join(traces)}], {{
    title: '{title}',
    yaxis: {{title: 'Value'}},
    paper_bgcolor: '{t["chart_bg"]}',
    plot_bgcolor: 'white',
}});"""

            elif chart_type == 'violin':
                # Violin plot: groups dict {group_name: [values]}
                groups = data.get('groups', {})
                if not groups and 'drift_results' in data:
                    # Could auto-generate from drift data
                    pass

                if groups:
                    traces = []
                    palette = t['chart_palette']
                    for i, (group_name, values) in enumerate(groups.items()):
                        color = palette[i % len(palette)]
                        traces.append(f"""{{
    type: 'violin',
    y: {values},
    name: '{group_name}',
    box: {{visible: true}},
    meanline: {{visible: true}},
    line: {{color: '{color}'}},
    fillcolor: '{color}',
    opacity: 0.6,
    points: 'outliers',
}}""")

                    return f"""
Plotly.newPlot('{chart_id}', [{','.join(traces)}], {{
    title: '{title}',
    yaxis: {{title: 'Distribution'}},
    paper_bgcolor: '{t["chart_bg"]}',
    plot_bgcolor: 'white',
    violinmode: 'group',
}});"""

        except Exception as e:
            self._record_chart_failure('plotly_chart', e)

        return ""

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 data URI for self-contained HTML."""
        try:
            import base64
            full_path = self.output_dir / image_path
            if full_path.exists():
                with open(full_path, 'rb') as f:
                    data = base64.b64encode(f.read()).decode('utf-8')
                return f'data:image/png;base64,{data}'
        except Exception as e:
            self._record_internal_warning('ImageBase64', 'base64 conversion failed', e)
            pass
        return image_path

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to LaTeX RGB format."""
        h = hex_color.lstrip('#')
        return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"

    def _create_importance_chart(self, sorted_importance: List[Tuple[str, float]]) -> str:
        """Create feature importance bar chart."""
        try:
            import matplotlib.pyplot as plt

            features = [x[0][:25] for x in sorted_importance]
            values = [x[1] for x in sorted_importance]
            n = len(features)

            with self._chart_context('feature_importance', figsize=(10, max(6, n * 0.35))) as (fig, ax):
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, n))[::-1]
                bars = ax.barh(range(n), values[::-1], color=colors)

                ax.set_yticks(range(n))
                ax.set_yticklabels(features[::-1], fontsize=9)
                ax.set_xlabel('Importance', fontsize=11)
                ax.set_title('Feature Importance', fontsize=14, fontweight='bold', color=self.theme['primary'])

                # Add value labels
                for bar, val in zip(bars, values[::-1]):
                    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{val:.3f}', va='center', fontsize=8)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            return 'figures/feature_importance.png'

        except Exception as e:
            self._record_chart_failure('importance_chart', e)
            return ""

    def _create_benchmark_chart(self, sorted_models: List[Tuple[str, Dict]]) -> str:
        """Create model benchmarking chart."""
        try:
            import matplotlib.pyplot as plt

            models = [x[0] for x in sorted_models]
            scores = [x[1].get('test_score', x[1].get('cv_score', 0)) for x in sorted_models]

            with self._chart_context('model_benchmark', figsize=(10, 6)) as (fig, ax):
                colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
                bars = ax.barh(range(len(models)), scores, color=colors)

                ax.set_yticks(range(len(models)))
                ax.set_yticklabels(models, fontsize=10)
                ax.set_xlabel('Test Score', fontsize=11)
                ax.set_title('Model Comparison', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax.set_xlim(0, 1)

                for bar, score in zip(bars, scores):
                    ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{score:.4f}', va='center', fontsize=9)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

            return 'figures/model_benchmark.png'

        except Exception as e:
            self._record_chart_failure('benchmark_chart', e)
            return ""

    def _create_confusion_matrix_chart(self, cm, labels: List[str] = None) -> str:
        """Create confusion matrix heatmap."""
        try:
            import seaborn as sns

            with self._chart_context('confusion_matrix', figsize=(8, 6)) as (fig, ax):
                sns.heatmap(cm, annot=True, fmt='d', cmap=self.theme['chart_cmap'],
                           xticklabels=labels or ['0', '1'],
                           yticklabels=labels or ['0', '1'],
                           ax=ax, annot_kws={'size': 14})

                ax.set_xlabel('Predicted', fontsize=11)
                ax.set_ylabel('Actual', fontsize=11)
                ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color=self.theme['primary'])

            return 'figures/confusion_matrix.png'

        except Exception as e:
            self._record_chart_failure('confusion_matrix', e)
            return ""

    def _create_roc_chart(self, fpr, tpr, roc_auc: float) -> str:
        """Create ROC curve chart."""
        try:
            with self._chart_context('roc_curve', figsize=(8, 6)) as (fig, ax):
                ax.plot(fpr, tpr, color=self.theme['accent'], lw=2,
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')

                ax.fill_between(fpr, tpr, alpha=0.3, color=self.theme['accent'])

                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=11)
                ax.set_ylabel('True Positive Rate', fontsize=11)
                ax.set_title('ROC Curve', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            return 'figures/roc_curve.png'

        except Exception as e:
            self._record_chart_failure('roc_chart', e)
            return ""

    def _create_pr_chart(self, precision, recall, avg_precision: float) -> str:
        """Create precision-recall curve chart."""
        try:
            with self._chart_context('pr_curve', figsize=(8, 6)) as (fig, ax):
                ax.plot(recall, precision, color=self.theme['success'], lw=2,
                       label=f'PR curve (AP = {avg_precision:.3f})')

                ax.fill_between(recall, precision, alpha=0.3, color=self.theme['success'])

                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall', fontsize=11)
                ax.set_ylabel('Precision', fontsize=11)
                ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', color=self.theme['primary'])
                ax.legend(loc='lower left')
                ax.grid(True, alpha=0.3)

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            return 'figures/pr_curve.png'

        except Exception as e:
            self._record_chart_failure('pr_chart', e)
            return ""

    def _create_shap_chart(self, shap_values, feature_names, X_sample) -> str:
        """Create SHAP summary plot."""
        try:
            import matplotlib.pyplot as plt
            import shap

            # Extract values for binary classification (use positive class)
            plot_values = shap_values
            if hasattr(shap_values, 'values'):
                vals = shap_values.values
                if len(vals.shape) == 3:
                    plot_values = vals[:, :, 1]
                else:
                    plot_values = vals

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(plot_values, X_sample, feature_names=feature_names,
                            show=False, max_display=15)
            plt.title('SHAP Feature Impact', fontsize=14, fontweight='bold', color=self.theme['primary'])
            plt.tight_layout()

            path = self.figures_dir / 'shap_summary.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()

            return 'figures/shap_summary.png'

        except Exception as e:
            self._record_chart_failure('shap_chart', e)
            return ""

    def _create_overlaid_roc_chart(self, roc_curves: Dict) -> str:
        """Create overlaid ROC curves for model comparison.

        Args:
            roc_curves: Dict mapping model_name -> {'fpr': array, 'tpr': array}.

        Returns:
            Relative path to the saved chart image.
        """
        try:
            palette = self.theme['chart_palette']

            with self._chart_context('model_comparison_roc', figsize=(8, 8)) as (fig, ax):
                for i, (model_name, curve) in enumerate(roc_curves.items()):
                    color = palette[i % len(palette)]
                    ax.plot(curve['fpr'], curve['tpr'],
                            color=color, linewidth=2, label=model_name)

                ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1, label='Random')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate', fontsize=12)
                ax.set_ylabel('True Positive Rate', fontsize=12)
                ax.set_title('Model Comparison — ROC Curves', fontsize=14,
                             fontweight='bold', color=self.theme['primary'])
                ax.legend(loc='lower right', fontsize=10)
                ax.grid(True, alpha=self.theme['chart_grid_alpha'])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            return 'figures/model_comparison_roc.png'

        except Exception as e:
            self._record_chart_failure('model_comparison_roc', e)
            return ""

    def _create_metrics_radar_chart(self, comparison: Dict) -> str:
        """Create radar chart comparing metrics across models.

        Args:
            comparison: Dict mapping model_name -> {metric_name: value}.

        Returns:
            Relative path to the saved chart image.
        """
        try:
            palette = self.theme['chart_palette']
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            # Filter to metrics that have values for at least one model
            available_metrics = [m for m in metrics
                                 if any(v.get(m) is not None for v in comparison.values())]
            if len(available_metrics) < 3:
                return ""

            n_metrics = len(available_metrics)
            angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            angles += angles[:1]  # close the polygon

            with self._chart_context('model_comparison_radar', figsize=(8, 8),
                                     subplot_kw={'projection': 'polar'}) as (fig, ax):
                for i, (model_name, scores) in enumerate(comparison.items()):
                    color = palette[i % len(palette)]
                    values = [scores.get(m, 0) or 0 for m in available_metrics]
                    values += values[:1]
                    ax.plot(angles, values, color=color, linewidth=2, label=model_name)
                    ax.fill(angles, values, color=color, alpha=0.1)

                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(available_metrics, fontsize=10)
                ax.set_ylim(0, 1.05)
                ax.set_title('Model Comparison — Metrics Radar', fontsize=14,
                             fontweight='bold', color=self.theme['primary'],
                             pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

            return 'figures/model_comparison_radar.png'

        except Exception as e:
            self._record_chart_failure('model_comparison_radar', e)
            return ""
