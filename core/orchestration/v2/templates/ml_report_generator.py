"""
Professional ML Report Generator - World Class Edition
=======================================================

Generates the world's most comprehensive ML analysis reports using
LaTeX/Pandoc for publication-quality typesetting.

Features:
- Executive Summary with actionable insights
- Comprehensive Data Quality Analysis (outliers, missing patterns, distributions)
- Advanced Feature Analysis (correlations, VIF, interactions)
- Model Benchmarking with statistical significance
- Learning Curves & Bias-Variance Analysis
- Calibration Analysis & Probability Plots
- ROC/PR Curves with confidence bands
- Lift/Gain Charts & KS Statistics
- SHAP Deep Dive (summary, dependence, interactions, force plots)
- Error Analysis (misclassification patterns, hardest samples)
- Threshold Optimization & Cost-Benefit Analysis
- Cross-Validation Fold-by-Fold Results
- Reproducibility Section (environment, seeds, hyperparameters)
- Professional visualizations with consistent styling
- Table of contents with hyperlinks
"""

import os
import subprocess
import tempfile
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# =============================================================================
# THEME DEFINITIONS
# =============================================================================

THEMES = {
    'professional': {
        'name': 'Professional',
        'description': 'Clean modern blue theme with sans-serif typography',
        # Colors
        'primary': '#1a365d',       # Deep navy
        'secondary': '#2c5282',     # Medium blue
        'accent': '#3182ce',        # Bright blue
        'success': '#38a169',       # Green
        'warning': '#d69e2e',       # Gold
        'danger': '#e53e3e',        # Red
        'info': '#00b5d8',          # Cyan
        'text': '#1a202c',          # Dark gray
        'muted': '#718096',         # Light gray
        'background': '#f7fafc',    # Off-white
        'table_header': '#2c5282',  # Navy
        'table_alt': '#f0f5fa',     # Light blue-gray
        # Chart palette (matplotlib colors for bar charts, lines, etc.)
        'chart_palette': ['#3182ce', '#38a169', '#d69e2e', '#e53e3e',
                          '#00b5d8', '#805ad5', '#ed8936', '#667eea'],
        'chart_cmap': 'Blues',
        'chart_diverging_cmap': 'RdBu_r',
        'chart_bg': 'white',
        'chart_grid_alpha': 0.3,
        # Typography (LaTeX)
        'font_family': 'sans-serif',
        'heading_style': 'bold',
        'heading_transform': '',  # No transform
        # LaTeX header/footer
        'header_brand': 'Jotty ML Comprehensive Report',
        'footer_text': 'Professional ML Analysis',
    },
    'goldman': {
        'name': 'Goldman Sachs',
        'description': 'Institutional-grade navy theme with serif typography, inspired by GS reports',
        # Colors - Goldman Sachs palette
        'primary': '#10294a',       # GS Navy
        'secondary': '#1a4075',     # GS Medium
        'accent': '#7399c6',        # GS Light Blue
        'success': '#2e7d32',       # Deep Green
        'warning': '#f9a825',       # Amber
        'danger': '#c62828',        # Deep Red
        'info': '#6b7c93',          # Steel Gray
        'text': '#1a1a1a',          # Near Black
        'muted': '#6b7c93',         # Steel Gray
        'background': '#f5f7fa',    # Cool Gray
        'table_header': '#10294a',  # GS Navy
        'table_alt': '#f5f7fa',     # Cool Gray
        # Chart palette
        'chart_palette': ['#10294a', '#1a4075', '#7399c6', '#2e7d32',
                          '#f9a825', '#c62828', '#6b7c93', '#4a6fa5'],
        'chart_cmap': 'bone_r',
        'chart_diverging_cmap': 'RdYlGn',
        'chart_bg': '#fafbfc',
        'chart_grid_alpha': 0.2,
        # Typography (LaTeX) - serif like GS
        'font_family': 'serif',
        'heading_style': 'normal',  # GS uses light weight headings
        'heading_transform': 'uppercase',
        # LaTeX header/footer
        'header_brand': 'Jotty Research',
        'footer_text': 'Confidential',
    },
}

# Module-level reference (set by ProfessionalMLReport instance)
COLORS = THEMES['professional']


class ProfessionalMLReport:
    """
    World-Class ML Report Generator using Markdown + Pandoc + LaTeX.

    Creates the most comprehensive ML analysis reports with:
    - Multiple themes: 'professional' (modern blue) and 'goldman' (GS institutional)
    - Publication-quality typography
    - Professional visualizations with consistent theme-matched styling
    - Statistical rigor with confidence intervals
    - Deep model explainability (SHAP, PDP, ICE)
    - Business-focused insights and recommendations
    - Full reproducibility information

    Usage:
        report = ProfessionalMLReport(theme='goldman')  # Goldman Sachs style
        report = ProfessionalMLReport(theme='professional')  # Default blue
    """

    def __init__(self, output_dir: str = "professional_reports", theme: str = "professional",
                 llm_narrative: bool = False, html_enabled: bool = False):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self._content = []
        self._figures = []
        self._metadata = {}
        self._raw_data = {}

        # Structured data for HTML generation
        self._section_data = []
        self._llm_narrative_enabled = llm_narrative
        self._html_enabled = html_enabled
        self._llm = None  # Lazy-loaded UnifiedLLM

        # Set theme
        self.theme_name = theme
        self.theme = THEMES.get(theme, THEMES['professional'])

        # Update module-level COLORS for backward compat
        global COLORS
        COLORS = self.theme

        # Configure matplotlib style for this theme
        self._setup_plot_style()

    def _fig_path_for_markdown(self, filename: str) -> str:
        """Get the figure path for use in markdown (relative to output_dir)."""
        return f"figures/{filename}"

    def _save_figure(self, filename: str) -> str:
        """Save figure and return path for markdown."""
        import matplotlib.pyplot as plt
        full_path = self.figures_dir / filename
        plt.savefig(full_path, dpi=300, bbox_inches='tight',
                    facecolor=self.theme['chart_bg'])
        plt.close()
        return self._fig_path_for_markdown(filename)

    def _setup_plot_style(self):
        """Configure matplotlib for theme-matched professional visualizations."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from matplotlib import cycler

            t = self.theme

            if self.theme_name == 'goldman':
                # Goldman: clean, minimal, serif fonts, subtle grid
                try:
                    plt.style.use('seaborn-v0_8-whitegrid')
                except:
                    pass
                mpl.rcParams.update({
                    'font.family': 'serif',
                    'font.serif': ['DejaVu Serif', 'Georgia', 'Times New Roman'],
                    'font.size': 10,
                    'axes.titlesize': 13,
                    'axes.titleweight': 'normal',
                    'axes.labelsize': 10,
                    'axes.labelweight': 'normal',
                    'axes.spines.top': False,
                    'axes.spines.right': False,
                    'axes.spines.left': True,
                    'axes.spines.bottom': True,
                    'axes.edgecolor': '#d0d7de',
                    'axes.facecolor': t['chart_bg'],
                    'axes.prop_cycle': cycler('color', t['chart_palette']),
                    'figure.facecolor': t['chart_bg'],
                    'figure.dpi': 300,
                    'savefig.dpi': 300,
                    'savefig.bbox': 'tight',
                    'savefig.facecolor': t['chart_bg'],
                    'grid.alpha': t['chart_grid_alpha'],
                    'grid.color': '#d0d7de',
                    'grid.linewidth': 0.5,
                    'legend.framealpha': 0.95,
                    'legend.edgecolor': '#d0d7de',
                    'legend.fontsize': 9,
                    'xtick.color': '#6b7c93',
                    'ytick.color': '#6b7c93',
                    'text.color': t['text'],
                })
            else:
                # Professional: modern, bold, sans-serif
                try:
                    plt.style.use('seaborn-v0_8-whitegrid')
                except:
                    pass
                mpl.rcParams.update({
                    'font.family': 'sans-serif',
                    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                    'font.size': 10,
                    'axes.titlesize': 14,
                    'axes.titleweight': 'bold',
                    'axes.labelsize': 11,
                    'axes.labelweight': 'medium',
                    'axes.spines.top': False,
                    'axes.spines.right': False,
                    'axes.prop_cycle': cycler('color', t['chart_palette']),
                    'axes.facecolor': 'white',
                    'figure.facecolor': 'white',
                    'figure.dpi': 300,
                    'savefig.dpi': 300,
                    'savefig.bbox': 'tight',
                    'savefig.facecolor': 'white',
                    'grid.alpha': 0.3,
                    'legend.framealpha': 0.9,
                    'legend.edgecolor': '#cccccc',
                })
        except Exception as e:
            logger.debug(f"Could not setup plot style: {e}")

    def _store_section_data(self, section_type: str, title: str, data: Dict,
                            chart_configs: List[Dict] = None):
        """Store structured section data for HTML generation."""
        self._section_data.append({
            'type': section_type,
            'title': title,
            'data': data,
            'chart_configs': chart_configs or [],
        })

    def _init_llm(self) -> bool:
        """Lazy-initialize LLM for narrative generation. Returns True if available."""
        if self._llm is not None:
            return True
        try:
            from Jotty.core.llm.unified import UnifiedLLM
            self._llm = UnifiedLLM()
            return True
        except Exception as e:
            logger.debug(f"LLM initialization failed (expected if no API key): {e}")
            return False

    def _maybe_add_narrative(self, section_name: str, data_context: str) -> str:
        """Generate LLM narrative insight if enabled. Returns markdown blockquote or empty string."""
        if not self._llm_narrative_enabled:
            return ""

        if not self._init_llm():
            return ""

        try:
            prompt = (
                "You are a senior data scientist. Given the following ML analysis data, "
                "write a 2-3 sentence insight in plain English. Focus on actionable takeaways. "
                f"Section: {section_name}. Data: {data_context[:2000]}"
            )

            response = self._llm.generate(
                prompt=prompt,
                timeout=30,
                max_tokens=300,
                fallback=True
            )

            if response.success and response.text:
                return f"\n> **Insight:** {response.text.strip()}\n\n"
        except Exception as e:
            logger.debug(f"LLM narrative generation failed: {e}")

        return ""

    def set_metadata(self, title: str, subtitle: str = "", author: str = "Jotty ML",
                    dataset: str = "", problem_type: str = "Classification"):
        """Set report metadata."""
        self._metadata = {
            'title': title,
            'subtitle': subtitle,
            'author': author,
            'date': datetime.now().strftime('%B %d, %Y'),
            'dataset': dataset,
            'problem_type': problem_type,
        }

    def add_executive_summary(self, metrics: Dict[str, float], best_model: str,
                             n_features: int, context: str = ""):
        """Add executive summary section."""

        # Create metrics table
        metrics_md = "| Metric | Value |\n|--------|-------|\n"
        for name, value in metrics.items():
            if isinstance(value, float):
                if value < 1:
                    metrics_md += f"| {name.replace('_', ' ').title()} | {value:.4f} |\n"
                else:
                    metrics_md += f"| {name.replace('_', ' ').title()} | {value:.2f} |\n"
            else:
                metrics_md += f"| {name.replace('_', ' ').title()} | {value} |\n"

        summary = f"""
# Executive Summary

{context if context else "This report presents the results of an automated machine learning analysis."}

## Key Results

**Best Model:** {best_model}

**Performance Metrics:**

{metrics_md}

**Dataset:** {n_features} features analyzed

{self._maybe_add_narrative('Executive Summary', f'Best model: {best_model}, Metrics: {metrics}, Features: {n_features}')}

---
"""
        self._content.append(summary)
        self._store_section_data('executive_summary', 'Executive Summary', {'metrics': metrics, 'best_model': best_model})

    def add_data_profile(self, shape: Tuple[int, int], dtypes: Dict[str, int],
                        missing: Dict[str, int], recommendations: List[str]):
        """Add data profiling section."""

        dtype_md = "| Data Type | Count |\n|-----------|-------|\n"
        for dtype, count in dtypes.items():
            dtype_md += f"| {dtype} | {count} |\n"

        missing_md = ""
        if missing and any(v > 0 for v in missing.values()):
            missing_md = "\n**Missing Values:**\n\n| Feature | Missing |\n|---------|--------|\n"
            for feat, count in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]:
                if count > 0:
                    missing_md += f"| {feat} | {count} |\n"

        recs_md = "\n".join([f"- {rec}" for rec in recommendations]) if recommendations else ""

        content = f"""
# Data Profile

## Dataset Overview

- **Total Samples:** {shape[0]:,}
- **Total Features:** {shape[1]}

## Data Types

{dtype_md}
{missing_md}

## EDA Recommendations

{recs_md}

---
"""
        self._content.append(content)
        self._store_section_data('data_profile', 'Data Profile', {'shape': shape, 'dtypes': dtypes})

    # =========================================================================
    # PIPELINE DAG VISUALIZATION
    # =========================================================================

    def add_pipeline_visualization(self, pipeline_steps: List[Dict]):
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
                name = step.get('name', f'Step {i}')
                stype = step.get('type', 'unknown')
                in_shape = step.get('input_shape', 'N/A')
                out_shape = step.get('output_shape', 'N/A')
                params = step.get('params', {})

                in_str = f"{in_shape}" if in_shape != 'N/A' else 'N/A'
                out_str = f"{out_shape}" if out_shape != 'N/A' else 'N/A'
                param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])[:40] if params else '-'

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

            self._store_section_data('pipeline_dag', 'Pipeline Architecture', {
                'steps': pipeline_steps,
            }, [{'type': 'dag', 'path': fig_path}])

        except Exception as e:
            logger.warning(f"Pipeline visualization failed: {e}")

    def _create_pipeline_dag(self, pipeline_steps: List[Dict]) -> str:
        """Create pipeline flowchart using matplotlib FancyBboxPatch + FancyArrowPatch."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

            type_colors = {
                'preprocessing': self.theme['accent'],
                'feature_engineering': self.theme['success'],
                'model': self.theme['warning'],
                'ensemble': self.theme['danger'],
                'unknown': self.theme['muted'],
            }

            n_steps = len(pipeline_steps)
            fig_width = max(10, n_steps * 2.5)
            fig, ax = plt.subplots(figsize=(fig_width, 4))
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
            from matplotlib.patches import Patch
            for stype, color in type_colors.items():
                if stype != 'unknown':
                    legend_elements.append(Patch(facecolor=color, alpha=0.85, label=stype.replace('_', ' ').title()))

            ax.legend(handles=legend_elements, loc='upper center',
                     bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=9)

            ax.set_title('ML Pipeline Architecture', fontsize=14, fontweight='bold',
                        color=self.theme['primary'], pad=30)

            plt.tight_layout()
            path = self.figures_dir / 'pipeline_dag.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/pipeline_dag.png'
        except Exception as e:
            logger.debug(f"Failed to create pipeline DAG: {e}")
            return ""

    def add_feature_importance(self, importance: Dict[str, float], top_n: int = 20):
        """Add feature importance section with chart."""

        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Create table
        table_md = "| Rank | Feature | Importance |\n|------|---------|------------|\n"
        for i, (feat, imp) in enumerate(sorted_imp, 1):
            table_md += f"| {i} | {feat[:40]} | {imp:.4f} |\n"

        # Create bar chart
        fig_path = self._create_importance_chart(sorted_imp)

        content = f"""
# Feature Importance Analysis

Feature importance measures how much each feature contributes to the model's predictions.
Higher values indicate more influential features.

## Top {top_n} Features

{table_md}

## Feature Importance Visualization

![Feature Importance]({fig_path})

{self._maybe_add_narrative('Feature Importance', f'Top features: {sorted_imp[:5]}')}

---
"""
        self._content.append(content)
        self._store_section_data('feature_importance', 'Feature Importance',
                                {'importance': dict(sorted_imp[:10])},
                                [{'type': 'importance_bar'}])

    def add_model_benchmarking(self, model_scores: Dict[str, Dict[str, float]]):
        """Add model benchmarking comparison."""

        # Create comparison table
        table_md = "| Model | CV Score | Std Dev | Test Score | Time (s) |\n"
        table_md += "|-------|----------|---------|------------|----------|\n"

        sorted_models = sorted(model_scores.items(),
                              key=lambda x: x[1].get('test_score', x[1].get('cv_score', 0)),
                              reverse=True)

        for model, scores in sorted_models:
            cv = scores.get('cv_score', 0)
            std = scores.get('cv_std', 0)
            test = scores.get('test_score', 0)
            time_s = scores.get('train_time', 0)
            table_md += f"| {model} | {cv:.4f} | ±{std:.4f} | {test:.4f} | {time_s:.2f} |\n"

        # Create benchmark chart
        fig_path = self._create_benchmark_chart(sorted_models)

        content = f"""
# Model Benchmarking

Multiple machine learning algorithms were evaluated using 5-fold cross-validation.
The table below shows the performance of each model.

## Model Comparison

{table_md}

## Performance Visualization

![Model Benchmarking]({fig_path})

{self._maybe_add_narrative('Model Benchmarking', f'Model comparison results: {list(model_scores.keys())}')}

---
"""
        self._content.append(content)
        self._store_section_data('model_benchmarking', 'Model Benchmarking',
                                {'model_scores': model_scores},
                                [{'type': 'bar'}])

    # =========================================================================
    # MULTI-DATASET VALIDATION
    # =========================================================================

    def add_cross_dataset_validation(self, datasets_dict: Dict[str, Tuple],
                                      model, metric_fn=None, metric_name: str = "Score"):
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
                    X_arr = X_ds.values if hasattr(X_ds, 'values') else np.asarray(X_ds)
                    y_arr = np.asarray(y_ds)

                    y_pred = model.predict(X_arr)
                    score = metric_fn(y_arr, y_pred)
                    results[ds_name] = {
                        'score': float(score),
                        'n_samples': len(y_arr),
                        'n_features': X_arr.shape[1],
                    }

                    # Feature stats for distribution shift detection
                    feature_stats[ds_name] = {
                        'mean': np.nanmean(X_arr, axis=0),
                        'std': np.nanstd(X_arr, axis=0),
                    }
                except Exception as e:
                    logger.debug(f"Cross-dataset eval failed for {ds_name}: {e}")
                    results[ds_name] = {'score': 0.0, 'n_samples': 0, 'n_features': 0}

            if not results:
                return

            # Generalization gap
            scores = [r['score'] for r in results.values() if r['score'] > 0]
            gen_gap = max(scores) - min(scores) if len(scores) > 1 else 0

            # Distribution shift detection
            shift_detected = False
            if len(feature_stats) > 1:
                ds_names = list(feature_stats.keys())
                for i in range(1, len(ds_names)):
                    ref_mean = feature_stats[ds_names[0]]['mean']
                    cur_mean = feature_stats[ds_names[i]]['mean']
                    mean_diff = np.mean(np.abs(ref_mean - cur_mean) / (np.abs(ref_mean) + 1e-10))
                    if mean_diff > 0.2:
                        shift_detected = True
                        break

            # Create chart
            fig_path = self._create_cross_dataset_chart(results, metric_name)

            # Build table
            table_md = f"| Dataset | {metric_name} | Samples | Features |\n"
            table_md += "|---------|-------|---------|----------|\n"
            for ds_name, r in sorted(results.items(), key=lambda x: -x[1]['score']):
                table_md += f"| {ds_name} | {r['score']:.4f} | {r['n_samples']:,} | {r['n_features']} |\n"

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

            self._store_section_data('cross_dataset', 'Multi-Dataset Validation', {
                'results': results,
                'gen_gap': gen_gap,
                'shift_detected': shift_detected,
            }, [{'type': 'bar', 'path': fig_path}])

        except Exception as e:
            logger.warning(f"Cross-dataset validation failed: {e}")

    def _create_cross_dataset_chart(self, results: Dict, metric_name: str) -> str:
        """Create grouped bar chart of metrics per dataset."""
        try:
            import matplotlib.pyplot as plt

            ds_names = list(results.keys())
            scores = [results[ds]['score'] for ds in ds_names]
            n_samples = [results[ds]['n_samples'] for ds in ds_names]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()
            path = self.figures_dir / 'cross_dataset_validation.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/cross_dataset_validation.png'
        except Exception as e:
            logger.debug(f"Failed to create cross-dataset chart: {e}")
            return ""

    def add_confusion_matrix(self, y_true, y_pred, labels: List[str] = None):
        """Add confusion matrix section."""
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

        # Create classification report table
        table_md = "| Class | Precision | Recall | F1-Score | Support |\n"
        table_md += "|-------|-----------|--------|----------|--------|\n"

        for cls in (labels or [str(i) for i in range(len(cm))]):
            if cls in report:
                p = report[cls]['precision']
                r = report[cls]['recall']
                f1 = report[cls]['f1-score']
                sup = int(report[cls]['support'])
                table_md += f"| {cls} | {p:.3f} | {r:.3f} | {f1:.3f} | {sup} |\n"

        table_md += f"| **Accuracy** | | | **{report['accuracy']:.3f}** | |\n"

        # Create confusion matrix figure
        fig_path = self._create_confusion_matrix_chart(cm, labels)

        content = f"""
# Classification Performance

## Classification Report

{table_md}

## Confusion Matrix

![Confusion Matrix]({fig_path})

---
"""
        self._content.append(content)
        self._store_section_data('confusion_matrix', 'Classification Performance', {
            'cm': cm.tolist(),
            'report': {k: v for k, v in report.items() if isinstance(v, dict)},
            'labels': labels or [str(i) for i in range(len(cm))],
        }, [{'type': 'confusion_heatmap'}])

    def add_roc_analysis(self, y_true, y_prob, pos_label=1):
        """Add ROC curve analysis."""
        from sklearn.metrics import roc_curve, auc, roc_auc_score

        fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        # Create ROC curve figure
        fig_path = self._create_roc_chart(fpr, tpr, roc_auc)

        content = f"""
# ROC Curve Analysis

The Receiver Operating Characteristic (ROC) curve shows the trade-off between
true positive rate and false positive rate at various classification thresholds.

## Key Metrics

- **AUC-ROC:** {roc_auc:.4f}
- **Optimal Threshold:** {optimal_threshold:.4f}

## ROC Curve

![ROC Curve]({fig_path})

---
"""
        self._content.append(content)
        self._store_section_data('roc_analysis', 'ROC Curve Analysis', {'auc': roc_auc})

    def add_precision_recall(self, y_true, y_prob, pos_label=1):
        """Add precision-recall curve analysis."""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
        avg_precision = average_precision_score(y_true, y_prob, pos_label=pos_label)

        # Create PR curve figure
        fig_path = self._create_pr_chart(precision, recall, avg_precision)

        content = f"""
# Precision-Recall Analysis

The Precision-Recall curve is especially useful for imbalanced datasets,
showing the trade-off between precision and recall.

## Key Metrics

- **Average Precision:** {avg_precision:.4f}

## Precision-Recall Curve

![Precision-Recall Curve]({fig_path})

---
"""
        self._content.append(content)
        self._store_section_data('precision_recall', 'Precision-Recall Analysis', {'avg_precision': avg_precision})

    def add_baseline_comparison(self, baseline_score: float, final_score: float,
                               baseline_model: str = "Baseline"):
        """Add baseline comparison section."""

        improvement = final_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

        content = f"""
# Baseline Comparison

## Performance Improvement

| Model | Score | Improvement |
|-------|-------|-------------|
| {baseline_model} | {baseline_score:.4f} | - |
| **Best Model** | **{final_score:.4f}** | **+{improvement:.4f} ({improvement_pct:+.1f}%)** |

The final model achieves a **{improvement_pct:.1f}%** improvement over the baseline.

---
"""
        self._content.append(content)
        self._store_section_data('baseline_comparison', 'Baseline Comparison', {'improvement_pct': improvement_pct})

    def add_shap_analysis(self, shap_values, feature_names: List[str], X_sample=None):
        """Add SHAP analysis section."""
        try:
            import shap

            # Calculate mean absolute SHAP values
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values

            if len(values.shape) == 3:
                values = values[:, :, 1]  # Binary classification

            mean_shap = np.abs(values).mean(axis=0)
            shap_importance = sorted(zip(feature_names, mean_shap),
                                    key=lambda x: x[1], reverse=True)[:15]

            # Create SHAP summary plot
            fig_path = self._create_shap_chart(shap_values, feature_names, X_sample)

            table_md = "| Feature | Mean |SHAP| |\n|---------|-------------|\n"
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
            self._store_section_data('shap_analysis', 'SHAP Feature Analysis', {
                'top_features': [f for f, _ in shap_importance[:10]],
                'mean_shap': dict(shap_importance[:10]),
            }, [{'type': 'importance_bar'}])

        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")

    def add_recommendations(self, recommendations: List[str]):
        """Add recommendations section."""

        recs_md = "\n".join([f"{i}. {rec}" for i, rec in enumerate(recommendations, 1)])

        content = f"""
# Recommendations & Next Steps

{recs_md}

---

*Report generated by Jotty SwarmMLComprehensive on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        self._content.append(content)
        self._store_section_data('recommendations', 'Recommendations', {'recommendations': recommendations})

    # =========================================================================
    # ADVANCED DATA QUALITY ANALYSIS
    # =========================================================================

    def add_data_quality_analysis(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Add comprehensive data quality analysis including:
        - Missing value patterns (heatmap)
        - Outlier detection with IQR and Z-score
        - Distribution analysis (skewness, kurtosis)
        - Data type summary
        """
        self._raw_data['X'] = X
        self._raw_data['y'] = y

        # Calculate statistics
        n_samples, n_features = X.shape
        missing_counts = X.isnull().sum()
        missing_pct = (missing_counts / n_samples * 100).round(2)
        n_missing_cols = (missing_counts > 0).sum()

        # Numeric column stats
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        n_numeric = len(numeric_cols)
        n_categorical = n_features - n_numeric

        # Outlier detection
        outlier_summary = self._detect_outliers(X[numeric_cols]) if n_numeric > 0 else {}

        # Distribution stats
        dist_stats = self._calculate_distribution_stats(X[numeric_cols]) if n_numeric > 0 else {}

        # Create visualizations
        fig_missing = self._create_missing_pattern_chart(X) if n_missing_cols > 0 else ""
        fig_dist = self._create_distribution_overview(X[numeric_cols]) if n_numeric > 0 else ""
        fig_outlier = self._create_outlier_boxplot(X[numeric_cols], outlier_summary) if outlier_summary else ""

        # Build content
        content = f"""
# Data Quality Analysis

A comprehensive analysis of data quality, identifying potential issues before modeling.

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Samples | {n_samples:,} |
| Total Features | {n_features} |
| Numeric Features | {n_numeric} |
| Categorical Features | {n_categorical} |
| Features with Missing | {n_missing_cols} |
| Total Missing Values | {missing_counts.sum():,} ({missing_counts.sum() / (n_samples * n_features) * 100:.2f}%) |

"""

        # Missing values detail
        if n_missing_cols > 0:
            content += """
## Missing Value Analysis

| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
"""
            for col in missing_counts[missing_counts > 0].sort_values(ascending=False).head(15).index:
                content += f"| {col[:30]} | {missing_counts[col]:,} | {missing_pct[col]:.1f}% |\n"

            if fig_missing:
                content += f"""
## Missing Value Pattern

![Missing Value Pattern]({fig_missing})

"""

        # Distribution analysis
        if dist_stats:
            content += """
## Distribution Analysis

| Feature | Skewness | Kurtosis | Assessment |
|---------|----------|----------|------------|
"""
            for col, stats in list(dist_stats.items())[:15]:
                skew = stats['skewness']
                kurt = stats['kurtosis']
                assessment = self._assess_distribution(skew, kurt)
                content += f"| {col[:25]} | {skew:.2f} | {kurt:.2f} | {assessment} |\n"

            if fig_dist:
                content += f"""
## Feature Distributions

![Feature Distributions]({fig_dist})

"""

        # Outlier analysis
        if outlier_summary:
            total_outliers = sum(v['count'] for v in outlier_summary.values())
            content += f"""
## Outlier Analysis

**Method:** Interquartile Range (IQR) with 1.5x multiplier

**Total Outliers Detected:** {total_outliers:,} across {len([v for v in outlier_summary.values() if v['count'] > 0])} features

| Feature | Outliers | % of Data | Min | Max |
|---------|----------|-----------|-----|-----|
"""
            for col, stats in sorted(outlier_summary.items(), key=lambda x: x[1]['count'], reverse=True)[:15]:
                if stats['count'] > 0:
                    content += f"| {col[:25]} | {stats['count']:,} | {stats['pct']:.1f}% | {stats['min']:.2f} | {stats['max']:.2f} |\n"

            if fig_outlier:
                content += f"""
## Outlier Distribution

![Outlier Boxplot]({fig_outlier})

"""

        content += "\n---\n"
        self._content.append(content)
        self._store_section_data('data_quality', 'Data Quality Analysis', {'n_samples': n_samples, 'n_features': n_features})

    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method."""
        outliers = {}
        for col in df.columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask = (data < lower) | (data > upper)
            outliers[col] = {
                'count': outlier_mask.sum(),
                'pct': outlier_mask.mean() * 100,
                'min': data.min(),
                'max': data.max(),
                'lower_bound': lower,
                'upper_bound': upper,
            }
        return outliers

    def _calculate_distribution_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate distribution statistics."""
        from scipy import stats as scipy_stats
        dist_stats = {}
        for col in df.columns:
            data = df[col].dropna()
            if len(data) < 3:
                continue
            dist_stats[col] = {
                'skewness': scipy_stats.skew(data),
                'kurtosis': scipy_stats.kurtosis(data),
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
            }
        return dist_stats

    def _assess_distribution(self, skew: float, kurt: float) -> str:
        """Assess distribution based on skewness and kurtosis."""
        assessments = []
        if abs(skew) < 0.5:
            assessments.append("Symmetric")
        elif skew > 0:
            assessments.append("Right-skewed")
        else:
            assessments.append("Left-skewed")

        if abs(kurt) > 3:
            assessments.append("Heavy-tailed")

        return ", ".join(assessments) if assessments else "Normal"

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

            fig, ax = plt.subplots(figsize=(12, 6))

            # Sample if too many rows
            sample_df = df[top_cols].head(200) if len(df) > 200 else df[top_cols]

            sns.heatmap(sample_df.isnull(), cmap='RdYlBu_r', cbar_kws={'label': 'Missing'},
                       yticklabels=False, ax=ax)
            ax.set_title('Missing Value Pattern', fontsize=14, fontweight='bold', color=self.theme['primary'])
            ax.set_xlabel('Features', fontsize=11)
            ax.set_ylabel('Samples', fontsize=11)
            plt.xticks(rotation=45, ha='right', fontsize=8)

            plt.tight_layout()
            path = self.figures_dir / 'missing_pattern.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/missing_pattern.png'
        except Exception as e:
            logger.debug(f"Failed to create missing pattern chart: {e}")
            return ""

    def _create_distribution_overview(self, df: pd.DataFrame) -> str:
        """Create distribution overview with histograms."""
        try:
            import matplotlib.pyplot as plt

            # Select top features to display
            n_features = min(12, len(df.columns))
            cols = df.columns[:n_features]

            n_rows = (n_features + 3) // 4
            fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3 * n_rows))
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
            plt.tight_layout()

            path = self.figures_dir / 'distributions.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/distributions.png'
        except Exception as e:
            logger.debug(f"Failed to create distribution chart: {e}")
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

            fig, ax = plt.subplots(figsize=(12, 6))

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

            plt.tight_layout()
            path = self.figures_dir / 'outlier_boxplot.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/outlier_boxplot.png'
        except Exception as e:
            logger.debug(f"Failed to create outlier boxplot: {e}")
            return ""

    # =========================================================================
    # CORRELATION & MULTICOLLINEARITY ANALYSIS
    # =========================================================================

    def add_correlation_analysis(self, X: pd.DataFrame, threshold: float = 0.7):
        """
        Add correlation analysis with:
        - Correlation matrix heatmap with hierarchical clustering
        - Highly correlated feature pairs
        - VIF analysis for multicollinearity
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return

        corr_matrix = X[numeric_cols].corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        # Calculate VIF
        vif_data = self._calculate_vif(X[numeric_cols])

        # Create visualizations
        fig_corr = self._create_correlation_heatmap(corr_matrix)
        fig_vif = self._create_vif_chart(vif_data) if vif_data else ""

        content = f"""
# Correlation & Multicollinearity Analysis

Understanding feature relationships is critical for model interpretation and feature selection.

## Correlation Matrix

![Correlation Matrix]({fig_corr})

## Highly Correlated Feature Pairs (|r| >= {threshold})

"""
        if high_corr_pairs:
            content += "| Feature 1 | Feature 2 | Correlation |\n|-----------|-----------|-------------|\n"
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:15]:
                content += f"| {pair['feature_1'][:20]} | {pair['feature_2'][:20]} | {pair['correlation']:.3f} |\n"
        else:
            content += "*No highly correlated pairs found above threshold.*\n"

        if vif_data:
            content += f"""
## Variance Inflation Factor (VIF)

VIF measures multicollinearity. VIF > 5 indicates moderate, VIF > 10 indicates severe multicollinearity.

| Feature | VIF | Assessment |
|---------|-----|------------|
"""
            for feat, vif in sorted(vif_data.items(), key=lambda x: x[1], reverse=True)[:15]:
                assessment = "Critical" if vif > 10 else ("High" if vif > 5 else "OK")
                content += f"| {feat[:25]} | {vif:.2f} | {assessment} |\n"

            if fig_vif:
                content += f"""
## VIF Visualization

![VIF Analysis]({fig_vif})

"""

        content += "\n---\n"
        self._content.append(content)
        vif_severe_count = sum(1 for v in vif_data.values() if v > 10) if vif_data else 0
        self._store_section_data('correlation', 'Correlation Analysis', {
            'n_high_corr': len(high_corr_pairs),
            'high_corr_pairs': [{'f1': p['feature_1'], 'f2': p['feature_2'], 'corr': p['correlation']}
                                for p in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]],
            'vif_severe_count': vif_severe_count,
        })

    def _calculate_vif(self, df: pd.DataFrame) -> Dict:
        """Calculate Variance Inflation Factor for each feature."""
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # Handle missing values
            clean_df = df.dropna()
            if len(clean_df) < 10 or len(df.columns) < 2:
                return {}

            # Limit features for performance
            cols = clean_df.columns[:30].tolist()
            clean_df = clean_df[cols]

            vif_data = {}
            for i, col in enumerate(clean_df.columns):
                try:
                    vif = variance_inflation_factor(clean_df.values, i)
                    if not np.isinf(vif) and not np.isnan(vif):
                        vif_data[col] = vif
                except:
                    pass

            return vif_data
        except ImportError:
            return {}
        except Exception as e:
            logger.debug(f"VIF calculation failed: {e}")
            return {}

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

            fig, ax = plt.subplots(figsize=(12, 10))

            # Cluster the correlation matrix
            try:
                linkage = hierarchy.linkage(squareform(1 - np.abs(corr_matrix)), method='average')
                order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage, squareform(1 - np.abs(corr_matrix))))
                corr_matrix = corr_matrix.iloc[order, order]
            except:
                pass

            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(corr_matrix, mask=mask, cmap=self.theme['chart_diverging_cmap'], center=0,
                       annot=len(corr_matrix) <= 15, fmt='.2f', ax=ax,
                       square=True, linewidths=0.5,
                       cbar_kws={'shrink': 0.8, 'label': 'Correlation'})

            ax.set_title('Feature Correlation Matrix (Clustered)', fontsize=14, fontweight='bold', color=self.theme['primary'])
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)

            plt.tight_layout()
            path = self.figures_dir / 'correlation_matrix.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/correlation_matrix.png'
        except Exception as e:
            logger.debug(f"Failed to create correlation heatmap: {e}")
            return ""

    def _create_vif_chart(self, vif_data: Dict) -> str:
        """Create VIF bar chart."""
        try:
            import matplotlib.pyplot as plt

            if not vif_data:
                return ""

            sorted_vif = sorted(vif_data.items(), key=lambda x: x[1], reverse=True)[:20]
            features = [x[0][:20] for x in sorted_vif]
            values = [x[1] for x in sorted_vif]

            fig, ax = plt.subplots(figsize=(10, 6))

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

            plt.tight_layout()
            path = self.figures_dir / 'vif_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/vif_analysis.png'
        except Exception as e:
            logger.debug(f"Failed to create VIF chart: {e}")
            return ""

    # =========================================================================
    # LEARNING CURVES & BIAS-VARIANCE ANALYSIS
    # =========================================================================

    def add_learning_curves(self, model, X, y, cv: int = 5):
        """
        Add learning curve analysis showing:
        - Training vs validation score over sample sizes
        - Bias-variance tradeoff diagnosis
        - Optimal training size recommendation
        """
        try:
            from sklearn.model_selection import learning_curve

            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1,
                scoring='accuracy', shuffle=True, random_state=42
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
            fig_path = self._create_learning_curve_chart(
                train_sizes_abs, train_mean, train_std, val_mean, val_std
            )

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
            self._store_section_data('learning_curves', 'Learning Curves', {'diagnosis': diagnosis})

        except Exception as e:
            logger.warning(f"Learning curve analysis failed: {e}")

    def _create_learning_curve_chart(self, sizes, train_mean, train_std, val_mean, val_std) -> str:
        """Create learning curve visualization."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

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

            plt.tight_layout()
            path = self.figures_dir / 'learning_curves.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/learning_curves.png'
        except Exception as e:
            logger.debug(f"Failed to create learning curve chart: {e}")
            return ""

    # =========================================================================
    # CALIBRATION ANALYSIS
    # =========================================================================

    def add_calibration_analysis(self, y_true, y_prob, n_bins: int = 10):
        """
        Add probability calibration analysis showing:
        - Calibration curve (reliability diagram)
        - Brier score
        - Expected Calibration Error (ECE)
        """
        try:
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss

            fraction_of_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins)

            # Calculate metrics
            brier_score = brier_score_loss(y_true, y_prob)

            # Expected Calibration Error
            bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
            ece = np.sum(np.abs(fraction_of_positives - mean_predicted) * (bin_counts[:len(fraction_of_positives)] / len(y_true)))

            # Create visualization
            fig_path = self._create_calibration_chart(fraction_of_positives, mean_predicted, y_prob)

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
            self._content.append(content)
            self._store_section_data('calibration', 'Calibration Analysis', {'brier_score': brier_score, 'ece': ece})

        except Exception as e:
            logger.warning(f"Calibration analysis failed: {e}")

    def _create_calibration_chart(self, fraction_pos, mean_pred, y_prob) -> str:
        """Create calibration curve visualization."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()
            path = self.figures_dir / 'calibration.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/calibration.png'
        except Exception as e:
            logger.debug(f"Failed to create calibration chart: {e}")
            return ""

    # =========================================================================
    # CONFIDENCE-CALIBRATED PREDICTIONS
    # =========================================================================

    def add_prediction_confidence_analysis(self, X_sample, y_true, y_pred, y_prob,
                                            feature_names: List[str] = None, top_n: int = 10):
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
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            y_prob_arr = np.asarray(y_prob)

            if feature_names is None:
                if hasattr(X_sample, 'columns'):
                    feature_names = list(X_sample.columns)
                else:
                    feature_names = [f'Feature_{i}' for i in range(np.asarray(X_sample).shape[1])]

            X_arr = np.asarray(X_sample)

            # Compute ECE
            ece_data = self._compute_ece(y_true_arr, y_prob_arr)

            # Create visualization
            fig_path = self._create_confidence_charts(y_true_arr, y_prob_arr, ece_data)

            # Find most/least confident predictions
            confidence = np.abs(y_prob_arr - 0.5) * 2  # 0 = uncertain, 1 = very confident
            correct = (y_true_arr == y_pred_arr).astype(int)

            # Most confident correct
            confident_correct_idx = np.where(correct == 1)[0]
            if len(confident_correct_idx) > 0:
                top_correct = confident_correct_idx[np.argsort(confidence[confident_correct_idx])[::-1][:top_n]]
            else:
                top_correct = np.array([])

            # Most confident wrong
            confident_wrong_idx = np.where(correct == 0)[0]
            if len(confident_wrong_idx) > 0:
                top_wrong = confident_wrong_idx[np.argsort(confidence[confident_wrong_idx])[::-1][:top_n]]
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
                    content += f"| {idx} | {y_true_arr[idx]} | {y_pred_arr[idx]} | {y_prob_arr[idx]:.4f} | {confidence[idx]:.4f} |\n"
                content += "\n"

            # Table of most confident wrong predictions
            if len(top_wrong) > 0:
                content += f"""## Most Confident Wrong Predictions (Top {min(top_n, len(top_wrong))})

| Index | True | Pred | Probability | Confidence |
|-------|------|------|-------------|------------|
"""
                for idx in top_wrong[:top_n]:
                    content += f"| {idx} | {y_true_arr[idx]} | {y_pred_arr[idx]} | {y_prob_arr[idx]:.4f} | {confidence[idx]:.4f} |\n"
                content += "\n"

            content += """
## Interpretation

- **ECE** close to 0 means predicted probabilities match actual frequencies
- Large confidence gap between correct/wrong predictions indicates reliable uncertainty
- Highly confident wrong predictions deserve investigation

---
"""
            self._content.append(content)

            self._store_section_data('confidence_analysis', 'Confidence-Calibrated Predictions', {
                'ece': ece_data['ece'],
                'avg_conf_correct': avg_conf_correct,
                'avg_conf_wrong': avg_conf_wrong,
            }, [{'type': 'multi', 'path': fig_path}])

        except Exception as e:
            logger.warning(f"Prediction confidence analysis failed: {e}")

    def _compute_ece(self, y_true, y_prob, n_bins: int = 10) -> Dict:
        """Compute Expected Calibration Error with per-bin breakdown."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_data = []
        ece = 0.0

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

            n_in_bin = mask.sum()
            if n_in_bin == 0:
                bin_data.append({'confidence': (bin_edges[i] + bin_edges[i + 1]) / 2,
                                'accuracy': 0, 'count': 0, 'gap': 0})
                continue

            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            gap = abs(avg_accuracy - avg_confidence)
            ece += gap * (n_in_bin / len(y_true))

            bin_data.append({
                'confidence': avg_confidence,
                'accuracy': avg_accuracy,
                'count': int(n_in_bin),
                'gap': gap,
            })

        return {'ece': ece, 'bins': bin_data}

    def _create_confidence_charts(self, y_true, y_prob, ece_data: Dict) -> str:
        """Create confidence vs accuracy curve and per-bin reliability bar chart."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()
            path = self.figures_dir / 'confidence_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/confidence_analysis.png'
        except Exception as e:
            logger.debug(f"Failed to create confidence charts: {e}")
            return ""

    # =========================================================================
    # LIFT & GAIN ANALYSIS
    # =========================================================================

    def add_lift_gain_analysis(self, y_true, y_prob):
        """
        Add lift and gain charts for marketing/business context:
        - Cumulative gains curve
        - Lift curve
        - KS statistic
        """
        try:
            # Sort by probability descending
            sorted_idx = np.argsort(y_prob)[::-1]
            y_sorted = np.array(y_true)[sorted_idx]

            # Calculate cumulative gains
            n = len(y_true)
            positives = y_sorted.sum()
            cum_gains = np.cumsum(y_sorted) / positives
            deciles = np.arange(1, n + 1) / n

            # Calculate lift
            random_gains = deciles
            lift = cum_gains / (deciles + 1e-10)

            # KS statistic
            ks_stat = np.max(cum_gains - deciles)
            ks_idx = np.argmax(cum_gains - deciles)

            # Create visualizations
            fig_path = self._create_lift_gain_chart(deciles, cum_gains, lift, ks_stat, ks_idx)

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
            self._store_section_data('lift_gain', 'Lift & Gain Analysis', {'ks_stat': ks_stat})

        except Exception as e:
            logger.warning(f"Lift/gain analysis failed: {e}")

    def _create_lift_gain_chart(self, deciles, gains, lift, ks_stat, ks_idx) -> str:
        """Create lift and gain visualization."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()
            path = self.figures_dir / 'lift_gain.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/lift_gain.png'
        except Exception as e:
            logger.debug(f"Failed to create lift/gain chart: {e}")
            return ""

    # =========================================================================
    # CROSS-VALIDATION DETAILED ANALYSIS
    # =========================================================================

    def add_cv_detailed_analysis(self, model, X, y, cv: int = 5):
        """
        Add detailed cross-validation analysis:
        - Fold-by-fold results
        - Score distribution
        - Stability analysis
        """
        try:
            from sklearn.model_selection import cross_val_score, cross_validate

            # Multiple metrics
            scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)

            # Create fold results table
            fold_data = []
            for i in range(cv):
                fold_data.append({
                    'fold': i + 1,
                    'train_acc': cv_results['train_accuracy'][i],
                    'test_acc': cv_results['test_accuracy'][i],
                    'train_f1': cv_results['train_f1_macro'][i],
                    'test_f1': cv_results['test_f1_macro'][i],
                })

            # Calculate stability
            acc_mean = cv_results['test_accuracy'].mean()
            acc_std = cv_results['test_accuracy'].std()
            cv_coefficient = acc_std / acc_mean * 100

            # Create visualization
            fig_path = self._create_cv_chart(cv_results)

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
            self._store_section_data('cv_analysis', 'CV Detailed Analysis', {'acc_mean': acc_mean})

        except Exception as e:
            logger.warning(f"CV detailed analysis failed: {e}")

    def _create_cv_chart(self, cv_results: Dict) -> str:
        """Create cross-validation visualization."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()
            path = self.figures_dir / 'cv_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/cv_analysis.png'
        except Exception as e:
            logger.debug(f"Failed to create CV chart: {e}")
            return ""

    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================

    def add_error_analysis(self, X: pd.DataFrame, y_true, y_pred, y_prob=None, top_n: int = 10):
        """
        Add error analysis including:
        - Misclassification breakdown
        - Hardest samples to classify
        - Error patterns by feature
        """
        try:
            from sklearn.metrics import confusion_matrix

            # Find misclassified samples
            errors = y_true != y_pred
            n_errors = errors.sum()
            error_rate = n_errors / len(y_true) * 100

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Find hardest samples (most confidently wrong)
            hardest_samples = []
            if y_prob is not None:
                error_idx = np.where(errors)[0]
                error_probs = y_prob[error_idx]
                # For wrong predictions, high probability = more confident error
                if hasattr(error_probs, '__len__') and len(error_probs) > 0:
                    confidence = np.abs(error_probs - 0.5) * 2  # Distance from 0.5
                    sorted_idx = np.argsort(confidence)[::-1][:top_n]

                    for i in sorted_idx:
                        orig_idx = error_idx[i]
                        hardest_samples.append({
                            'index': int(orig_idx),
                            'true': int(y_true.iloc[orig_idx] if hasattr(y_true, 'iloc') else y_true[orig_idx]),
                            'pred': int(y_pred[orig_idx]),
                            'prob': float(y_prob[orig_idx]),
                            'confidence': float(confidence[i])
                        })

            # Error distribution by feature (for top features)
            error_by_feature = self._analyze_error_patterns(X, errors)

            # Create visualization
            fig_path = self._create_error_analysis_chart(X, errors, y_true, y_pred, error_by_feature)

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

            narrative = self._maybe_add_narrative('Error Analysis', f'Error rate: {error_rate:.2f}%, Total errors: {n_errors}')

            content += f"""
## Error Distribution Analysis

![Error Analysis]({fig_path})

{narrative}

---
"""
            self._content.append(content)
            self._store_section_data('error_analysis', 'Error Analysis', {'error_rate': error_rate, 'n_errors': n_errors})

        except Exception as e:
            logger.warning(f"Error analysis failed: {e}")

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
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()
            path = self.figures_dir / 'error_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/error_analysis.png'
        except Exception as e:
            logger.debug(f"Failed to create error analysis chart: {e}")
            return ""

    # =========================================================================
    # SHAP DEEP DIVE
    # =========================================================================

    def add_shap_deep_analysis(self, shap_values, feature_names: List[str], X_sample,
                               model=None, top_n: int = 3):
        """
        Add comprehensive SHAP analysis:
        - Summary plot
        - Bar plot (global importance)
        - Dependence plots for top features
        - Force plot for sample predictions
        """
        try:
            import shap

            # Extract values
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values

            if len(values.shape) == 3:
                values = values[:, :, 1]  # Binary classification

            # Calculate mean absolute SHAP
            mean_shap = np.abs(values).mean(axis=0)
            shap_importance = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)

            # Create all SHAP visualizations
            fig_summary = self._create_shap_summary(shap_values, feature_names, X_sample)
            fig_bar = self._create_shap_bar(shap_importance)
            fig_dependence = self._create_shap_dependence(shap_values, feature_names, X_sample, top_n)
            fig_waterfall = self._create_shap_waterfall(shap_values, feature_names, X_sample)

            content = f"""
# SHAP Deep Analysis

SHAP (SHapley Additive exPlanations) provides consistent, locally accurate feature attributions
for any machine learning model.

## Global Feature Importance (Mean |SHAP|)

| Rank | Feature | Mean |SHAP| | Cumulative % |
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
            self._store_section_data('shap_deep', 'SHAP Deep Analysis', {
                'top_features': [f for f, _ in shap_importance[:10]],
            })

        except Exception as e:
            logger.warning(f"SHAP deep analysis failed: {e}")

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
            logger.debug(f"Failed to create SHAP summary: {e}")
            return ""

    def _create_shap_bar(self, shap_importance: List[Tuple]) -> str:
        """Create SHAP bar plot."""
        try:
            import matplotlib.pyplot as plt

            top_n = min(20, len(shap_importance))
            features = [x[0][:25] for x in shap_importance[:top_n]]
            values = [x[1] for x in shap_importance[:top_n]]

            fig, ax = plt.subplots(figsize=(10, 8))

            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(features)))[::-1]
            ax.barh(range(len(features)), values[::-1], color=colors)

            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features[::-1], fontsize=9)
            ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
            ax.set_title('SHAP Feature Importance', fontsize=14, fontweight='bold', color=self.theme['primary'])

            # Add value labels
            for i, val in enumerate(values[::-1]):
                ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

            plt.tight_layout()
            path = self.figures_dir / 'shap_bar.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/shap_bar.png'
        except Exception as e:
            logger.debug(f"Failed to create SHAP bar: {e}")
            return ""

    def _create_shap_dependence(self, shap_values, feature_names, X_sample, top_n: int) -> str:
        """Create SHAP dependence plots."""
        try:
            import matplotlib.pyplot as plt
            import shap

            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values

            if len(values.shape) == 3:
                values = values[:, :, 1]

            mean_shap = np.abs(values).mean(axis=0)
            top_idx = np.argsort(mean_shap)[::-1][:top_n]

            fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
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

            plt.suptitle('SHAP Dependence Plots', fontsize=14, fontweight='bold', color=self.theme['primary'], y=1.02)
            plt.tight_layout()

            path = self.figures_dir / 'shap_dependence.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/shap_dependence.png'
        except Exception as e:
            logger.debug(f"Failed to create SHAP dependence: {e}")
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
            logger.debug(f"Failed to create SHAP waterfall: {e}")
            return ""

    # =========================================================================
    # THRESHOLD OPTIMIZATION
    # =========================================================================

    def add_threshold_optimization(self, y_true, y_prob, cost_fp: float = 1.0, cost_fn: float = 1.0):
        """
        Add threshold optimization analysis:
        - Optimal threshold for different objectives
        - Cost-benefit analysis at different thresholds
        - Threshold impact table
        """
        try:
            from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

            # Calculate metrics at different thresholds
            thresholds = np.linspace(0.1, 0.9, 17)
            results = []

            for thresh in thresholds:
                y_pred = (y_prob >= thresh).astype(int)
                cm = confusion_matrix(y_true, y_pred)

                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                # Cost
                cost = fp * cost_fp + fn * cost_fn

                results.append({
                    'threshold': thresh,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'cost': cost
                })

            # Find optimal thresholds
            best_f1 = max(results, key=lambda x: x['f1'])
            best_cost = min(results, key=lambda x: x['cost'])
            balanced = min(results, key=lambda x: abs(x['precision'] - x['recall']))

            # Create visualization
            fig_path = self._create_threshold_chart(results, best_f1['threshold'], best_cost['threshold'])

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
            self._store_section_data('threshold_optimization', 'Threshold Optimization', {'best_f1_threshold': best_f1['threshold']})

        except Exception as e:
            logger.warning(f"Threshold optimization failed: {e}")

    def _create_threshold_chart(self, results: List[Dict], best_f1_thresh: float, best_cost_thresh: float) -> str:
        """Create threshold analysis visualization."""
        try:
            import matplotlib.pyplot as plt

            thresholds = [r['threshold'] for r in results]
            precision = [r['precision'] for r in results]
            recall = [r['recall'] for r in results]
            f1 = [r['f1'] for r in results]
            cost = [r['cost'] for r in results]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()
            path = self.figures_dir / 'threshold_optimization.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/threshold_optimization.png'
        except Exception as e:
            logger.debug(f"Failed to create threshold chart: {e}")
            return ""

    # =========================================================================
    # REPRODUCIBILITY SECTION
    # =========================================================================

    def add_reproducibility_section(self, model, params: Dict = None, random_state: int = None,
                                    environment: Dict = None):
        """
        Add reproducibility information:
        - Model hyperparameters
        - Random seeds
        - Environment details
        - Package versions
        """
        import sys
        import platform

        # Get environment info
        env_info = environment or {}
        env_info.update({
            'Python Version': platform.python_version(),
            'Platform': platform.platform(),
            'Processor': platform.processor(),
        })

        # Get package versions
        packages = {}
        for pkg in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'shap']:
            try:
                mod = __import__(pkg.replace('-', '_'))
                packages[pkg] = getattr(mod, '__version__', 'N/A')
            except:
                pass

        # Get model params
        if params is None and hasattr(model, 'get_params'):
            params = model.get_params()

        content = f"""
# Reproducibility

Full information for reproducing this analysis.

## Model Configuration

**Model Type:** {type(model).__name__ if model else 'N/A'}

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
"""
        if params:
            for k, v in list(params.items())[:20]:
                v_str = str(v)[:50] if len(str(v)) > 50 else str(v)
                content += f"| {k} | `{v_str}` |\n"

        if random_state is not None:
            content += f"""
## Random Seeds

| Component | Seed |
|-----------|------|
| Main Random State | {random_state} |
| NumPy | {random_state} |
| Train/Test Split | {random_state} |
"""

        content += f"""
## Environment

| Component | Version |
|-----------|---------|
"""
        for k, v in env_info.items():
            content += f"| {k} | {v} |\n"

        content += f"""
## Package Versions

| Package | Version |
|---------|---------|
"""
        for pkg, ver in packages.items():
            content += f"| {pkg} | {ver} |\n"

        content += f"""
## Generation Timestamp

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
"""
        self._content.append(content)
        self._store_section_data('reproducibility', 'Reproducibility', {
            'random_state': random_state,
            'python_version': platform.python_version(),
            'sklearn_version': packages.get('scikit-learn', 'N/A'),
            'timestamp': datetime.now().isoformat(),
        })

    # =========================================================================
    # DEPLOYMENT READINESS ASSESSMENT
    # =========================================================================

    def add_deployment_readiness(self, model, X_sample,
                                  batch_sizes: List[int] = None):
        """
        Add deployment readiness assessment:
        - Prediction latency at various batch sizes
        - Model serialization size
        - Memory footprint estimation
        - Deployment checklist
        """
        try:
            import time as _time
            import sys
            import pickle

            X_arr = X_sample if isinstance(X_sample, np.ndarray) else (
                X_sample.values if hasattr(X_sample, 'values') else np.array(X_sample))

            if batch_sizes is None:
                batch_sizes = [1, 10, 100, 1000]
            batch_sizes = [b for b in batch_sizes if b <= len(X_arr)]
            if not batch_sizes:
                batch_sizes = [min(len(X_arr), 1)]

            # Measure latency
            latency_results = []
            n_runs = 10

            for batch_size in batch_sizes:
                X_batch = X_arr[:batch_size]
                times = []
                for _ in range(n_runs):
                    start = _time.perf_counter()
                    model.predict(X_batch)
                    elapsed = (_time.perf_counter() - start) * 1000  # ms
                    times.append(elapsed)

                times_arr = np.array(times)
                latency_results.append({
                    'batch_size': batch_size,
                    'mean_ms': float(np.mean(times_arr)),
                    'p95_ms': float(np.percentile(times_arr, 95)),
                    'throughput': float(batch_size / (np.mean(times_arr) / 1000)) if np.mean(times_arr) > 0 else 0,
                })

            # Model serialization size
            model_size_mb = None
            try:
                serialized = pickle.dumps(model)
                model_size_mb = len(serialized) / (1024 * 1024)
            except Exception:
                pass

            # Memory footprint
            memory_bytes = sys.getsizeof(model)
            memory_mb = memory_bytes / (1024 * 1024)

            # Deployment checklist
            checklist = {
                'serializable': model_size_mb is not None,
                'has_predict': hasattr(model, 'predict'),
                'has_predict_proba': hasattr(model, 'predict_proba'),
                'deterministic': True,  # Assumed for sklearn
                'latency_ok': latency_results[0]['mean_ms'] < 10 if latency_results else False,
                'size_ok': model_size_mb < 100 if model_size_mb else False,
            }

            # Test determinism
            if len(X_arr) > 0:
                pred1 = model.predict(X_arr[:1])
                pred2 = model.predict(X_arr[:1])
                checklist['deterministic'] = np.array_equal(pred1, pred2)

            passed = sum(1 for v in checklist.values() if v)
            total = len(checklist)

            # Create latency chart
            fig_path = self._create_latency_chart(latency_results)

            # Build content
            content = f"""
# Deployment Readiness Assessment

Evaluating model readiness for production deployment.

## Prediction Latency

| Batch Size | Mean Latency (ms) | P95 Latency (ms) | Throughput (samples/sec) |
|-----------|-------------------|-------------------|-------------------------|
"""
            for lr in latency_results:
                content += f"| {lr['batch_size']:,} | {lr['mean_ms']:.2f} | {lr['p95_ms']:.2f} | {lr['throughput']:,.0f} |\n"

            content += f"""
## Model Size

| Property | Value |
|----------|-------|
| Serialized Size | {f'{model_size_mb:.2f} MB' if model_size_mb else 'Not serializable'} |
| Memory Footprint | {memory_mb:.4f} MB |
| Model Type | {type(model).__name__} |

## Deployment Checklist

| Check | Status |
|-------|--------|
| Serializable (pickle) | {'PASS' if checklist['serializable'] else 'FAIL'} |
| Has predict() | {'PASS' if checklist['has_predict'] else 'FAIL'} |
| Has predict_proba() | {'PASS' if checklist['has_predict_proba'] else 'FAIL'} |
| Deterministic | {'PASS' if checklist['deterministic'] else 'FAIL'} |
| Latency < 10ms (single) | {'PASS' if checklist['latency_ok'] else 'FAIL'} |
| Size < 100MB | {'PASS' if checklist['size_ok'] else 'FAIL'} |

**Overall: {passed}/{total} checks passed**

"""
            if fig_path:
                content += f"""## Latency Profile

![Deployment Latency]({fig_path})

"""

            content += "---\n"
            self._content.append(content)
            self._store_section_data('deployment_readiness', 'Deployment Readiness', {
                'latency_results': latency_results,
                'model_size_mb': model_size_mb,
                'checklist': checklist,
            }, [{'type': 'line'}])

        except Exception as e:
            logger.warning(f"Deployment readiness assessment failed: {e}")

    def _create_latency_chart(self, latency_results):
        """Create 2-panel latency and throughput chart."""
        try:
            import matplotlib.pyplot as plt

            if not latency_results:
                return ""

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            batch_sizes = [r['batch_size'] for r in latency_results]
            mean_latencies = [r['mean_ms'] for r in latency_results]
            p95_latencies = [r['p95_ms'] for r in latency_results]
            throughputs = [r['throughput'] for r in latency_results]

            # Latency panel
            ax1.plot(batch_sizes, mean_latencies, 'o-', color=self.theme['accent'],
                    linewidth=2, markersize=8, label='Mean')
            ax1.plot(batch_sizes, p95_latencies, 's--', color=self.theme['warning'],
                    linewidth=2, markersize=6, label='P95')
            ax1.axhline(y=10, color=self.theme['danger'], linestyle=':', alpha=0.7, label='10ms threshold')
            ax1.set_xlabel('Batch Size', fontsize=11)
            ax1.set_ylabel('Latency (ms)', fontsize=11)
            ax1.set_title('Prediction Latency', fontsize=14, fontweight='bold',
                         color=self.theme['primary'])
            ax1.legend(fontsize=9)
            if len(batch_sizes) > 1:
                ax1.set_xscale('log')

            # Throughput panel
            ax2.bar(range(len(batch_sizes)), throughputs, color=self.theme['accent'],
                   alpha=0.85, edgecolor='white')
            ax2.set_xticks(range(len(batch_sizes)))
            ax2.set_xticklabels([str(b) for b in batch_sizes])
            ax2.set_xlabel('Batch Size', fontsize=11)
            ax2.set_ylabel('Throughput (samples/sec)', fontsize=11)
            ax2.set_title('Prediction Throughput', fontsize=14, fontweight='bold',
                         color=self.theme['primary'])

            for i, tp in enumerate(throughputs):
                ax2.text(i, tp + max(throughputs) * 0.02, f'{tp:,.0f}',
                        ha='center', va='bottom', fontsize=9, fontweight='medium')

            plt.tight_layout()
            path = self.figures_dir / 'deployment_latency.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/deployment_latency.png'
        except Exception as e:
            logger.debug(f"Failed to create latency chart: {e}")
            return ""

    # =========================================================================
    # HYPERPARAMETER SEARCH VISUALIZATION
    # =========================================================================

    def add_hyperparameter_visualization(self, study_or_trials, param_names: List[str] = None,
                                          objective_name: str = "Objective"):
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
                param_names = list(trials[0]['params'].keys())

            # Limit params for readability
            param_names = param_names[:8]

            # Create visualization
            fig_path = self._create_hyperparameter_charts(trials, param_names, objective_name)

            # Build table of best trials
            sorted_trials = sorted(trials, key=lambda t: t['value'], reverse=True)
            top_n = min(10, len(sorted_trials))

            table_md = f"| Rank | {objective_name} | " + " | ".join(p[:15] for p in param_names[:5]) + " |\n"
            table_md += "|------|" + "|".join(["---"] * (min(5, len(param_names)) + 1)) + "|\n"

            for i, trial in enumerate(sorted_trials[:top_n], 1):
                row = f"| {i} | {trial['value']:.4f} | "
                for pname in param_names[:5]:
                    val = trial['params'].get(pname, 'N/A')
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

            self._store_section_data('hyperparameter_viz', 'Hyperparameter Search', {
                'n_trials': len(trials),
                'best_value': sorted_trials[0]['value'] if sorted_trials else None,
                'best_params': sorted_trials[0]['params'] if sorted_trials else {},
            }, [{'type': 'multi', 'path': fig_path}])

        except Exception as e:
            logger.warning(f"Hyperparameter visualization failed: {e}")

    def _normalize_trials(self, study_or_trials) -> List[Dict]:
        """Normalize Optuna study or list of dicts into uniform trial format."""
        trials = []

        # Check if it's an Optuna study
        if hasattr(study_or_trials, 'trials'):
            for trial in study_or_trials.trials:
                if hasattr(trial, 'value') and trial.value is not None:
                    trials.append({
                        'params': dict(trial.params),
                        'value': float(trial.value),
                    })
        elif isinstance(study_or_trials, list):
            for item in study_or_trials:
                if isinstance(item, dict) and 'params' in item and 'value' in item:
                    trials.append({
                        'params': dict(item['params']),
                        'value': float(item['value']),
                    })

        return trials

    def _create_hyperparameter_charts(self, trials: List[Dict], param_names: List[str],
                                       objective_name: str) -> str:
        """Create 1x3 subplot: parallel coordinates, importance bar, optimization history."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

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
            plt.colorbar(sm, ax=ax1, label=objective_name, shrink=0.8)

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

            plt.tight_layout()
            path = self.figures_dir / 'hyperparameter_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/hyperparameter_analysis.png'
        except Exception as e:
            logger.debug(f"Failed to create hyperparameter charts: {e}")
            return ""

    # =========================================================================
    # EXECUTIVE DASHBOARD (Phase 1)
    # =========================================================================

    def add_executive_dashboard(self, metrics: Dict[str, float], model_name: str = "",
                                dataset_name: str = ""):
        """
        Add executive dashboard with visual KPI gauge charts.

        Creates donut/arc gauge charts for key metrics:
        - Accuracy, AUC, F1, Precision, Recall
        - Color-coded: green (>0.9), yellow (0.7-0.9), red (<0.7)
        """
        try:
            import matplotlib.pyplot as plt

            # Standardize metric names
            kpi_map = {
                'accuracy': ['accuracy', 'acc', 'test_accuracy'],
                'auc': ['auc', 'roc_auc', 'auc_roc', 'AUC'],
                'f1': ['f1', 'f1_score', 'f1_macro', 'F1'],
                'precision': ['precision', 'precision_macro'],
                'recall': ['recall', 'recall_macro', 'sensitivity'],
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
                status = "Excellent" if value > 0.9 else ("Good" if value > 0.7 else "Needs Improvement")
                color_indicator = "🟢" if value > 0.9 else ("🟡" if value > 0.7 else "🔴")
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
            self._store_section_data('executive_dashboard', 'Executive Dashboard', {'kpis': kpis})

        except Exception as e:
            logger.warning(f"Executive dashboard failed: {e}")

    def add_insight_prioritization(self):
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
            section_lookup = {s['type']: s['data'] for s in self._section_data}

            # CRITICAL checks
            drift_data = section_lookup.get('drift_analysis', {})
            drift_results = drift_data.get('drift_results', [])
            for dr in drift_results:
                if isinstance(dr, dict) and dr.get('psi', 0) > 0.25:
                    findings.append({
                        'severity': 'CRITICAL',
                        'source': 'Drift Analysis',
                        'description': f"Feature '{dr.get('feature', '?')}' has PSI={dr.get('psi', 0):.3f} (>0.25)",
                        'action': 'Retrain model with recent data or investigate distribution shift',
                    })

            fairness_data = section_lookup.get('fairness_audit', {})
            for feat_name, groups in fairness_data.get('metrics', {}).items():
                if isinstance(groups, dict):
                    for group_name, m in groups.items():
                        if isinstance(m, dict) and m.get('disparate_impact', 1.0) < 0.8:
                            findings.append({
                                'severity': 'CRITICAL',
                                'source': 'Fairness Audit',
                                'description': f"Group '{group_name}' in '{feat_name}' has DI={m['disparate_impact']:.3f} (<0.8)",
                                'action': 'Apply bias mitigation (reweighting, threshold adjustment)',
                            })

            # HIGH checks
            confidence_data = section_lookup.get('confidence_analysis', {})
            if confidence_data.get('ece', 0) > 0.1:
                findings.append({
                    'severity': 'HIGH',
                    'source': 'Confidence Analysis',
                    'description': f"Expected Calibration Error = {confidence_data['ece']:.3f} (>0.1)",
                    'action': 'Apply calibration (Platt scaling or isotonic regression)',
                })

            exec_data = section_lookup.get('executive_summary', {})
            acc = exec_data.get('accuracy', exec_data.get('acc', None))
            if acc is not None and acc < 0.7:
                findings.append({
                    'severity': 'HIGH',
                    'source': 'Executive Summary',
                    'description': f"Accuracy = {acc:.3f} (<0.70 threshold)",
                    'action': 'Consider more powerful models, feature engineering, or more data',
                })

            stat_data = section_lookup.get('statistical_tests', {})
            if stat_data.get('significant') is False:
                findings.append({
                    'severity': 'HIGH',
                    'source': 'Statistical Tests',
                    'description': 'AUC confidence interval includes 0.5 — not statistically significant',
                    'action': 'Collect more data or improve feature quality',
                })

            # MEDIUM checks
            importance_data = section_lookup.get('feature_importance', {})
            importance = importance_data.get('importance', {})
            if importance:
                sorted_vals = sorted(importance.values(), reverse=True)
                total = sum(sorted_vals)
                if total > 0 and sorted_vals[0] / total > 0.5:
                    findings.append({
                        'severity': 'MEDIUM',
                        'source': 'Feature Importance',
                        'description': f"Top feature accounts for {sorted_vals[0]/total*100:.0f}% of total importance",
                        'action': 'Investigate feature reliability and add complementary features',
                    })

            class_data = section_lookup.get('class_distribution', {})
            class_counts = class_data.get('counts', {})
            if class_counts and isinstance(class_counts, dict):
                count_vals = list(class_counts.values())
                if len(count_vals) >= 2 and min(count_vals) > 0:
                    ratio = max(count_vals) / min(count_vals)
                    if ratio > 3:
                        findings.append({
                            'severity': 'MEDIUM',
                            'source': 'Class Distribution',
                            'description': f"Class imbalance ratio = {ratio:.1f}:1 (>3:1)",
                            'action': 'Apply SMOTE, class weights, or threshold tuning',
                        })

            # Sort by severity
            severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}
            findings.sort(key=lambda f: severity_order.get(f['severity'], 3))

            n_critical = sum(1 for f in findings if f['severity'] == 'CRITICAL')
            n_high = sum(1 for f in findings if f['severity'] == 'HIGH')
            n_medium = sum(1 for f in findings if f['severity'] == 'MEDIUM')

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

            self._store_section_data('insight_prioritization', 'Insight Prioritization', {
                'n_critical': n_critical,
                'n_high': n_high,
                'n_medium': n_medium,
                'findings': findings,
            })

        except Exception as e:
            logger.warning(f"Insight prioritization failed: {e}")

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
            import matplotlib.pyplot as plt

            n_kpis = len(kpis)
            fig, axes = plt.subplots(1, n_kpis, figsize=(4 * n_kpis, 3.5))
            if n_kpis == 1:
                axes = [axes]

            for ax, (name, value) in zip(axes, kpis.items()):
                self._create_gauge_chart(ax, value, name)

            fig.suptitle('Key Performance Indicators', fontsize=14, fontweight='bold',
                        color=self.theme['primary'], y=1.05)
            plt.tight_layout()

            path = self.figures_dir / 'executive_dashboard.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/executive_dashboard.png'
        except Exception as e:
            logger.debug(f"Failed to create executive dashboard chart: {e}")
            return ""

    # =========================================================================
    # CLASS DISTRIBUTION ANALYSIS (Phase 2)
    # =========================================================================

    def add_class_distribution(self, y_true, y_pred=None, labels: List[str] = None):
        """
        Add class distribution analysis with:
        - Class balance bar chart
        - MCC, balanced accuracy, Cohen's kappa
        - Resampling suggestions if imbalanced (ratio > 3:1)
        """
        try:
            from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score,
                                         cohen_kappa_score)

            y_true_arr = np.asarray(y_true)
            unique_classes, class_counts = np.unique(y_true_arr, return_counts=True)
            n_classes = len(unique_classes)

            if labels is None:
                labels = [f'Class {c}' for c in unique_classes]

            # Calculate class ratios
            max_count = class_counts.max()
            min_count = class_counts.min()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            # Calculate metrics if predictions available
            metrics_md = ""
            if y_pred is not None:
                y_pred_arr = np.asarray(y_pred)
                mcc = matthews_corrcoef(y_true_arr, y_pred_arr)
                balanced_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
                kappa = cohen_kappa_score(y_true_arr, y_pred_arr)

                metrics_md = f"""
## Class-Aware Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Matthews Correlation Coefficient | {mcc:.4f} | -1 to 1, 0 = random |
| Balanced Accuracy | {balanced_acc:.4f} | Accuracy adjusted for class imbalance |
| Cohen's Kappa | {kappa:.4f} | Agreement beyond chance |

"""

            # Create visualization
            fig_path = self._create_class_distribution_chart(labels, class_counts)

            # Build class table
            class_table = "| Class | Count | Percentage | Ratio |\n|-------|-------|------------|-------|\n"
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
            self._store_section_data('class_distribution', 'Class Distribution', {'imbalance_ratio': imbalance_ratio})

        except Exception as e:
            logger.warning(f"Class distribution analysis failed: {e}")

    def _create_class_distribution_chart(self, labels: List[str], counts) -> str:
        """Create class distribution bar chart."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

            plt.tight_layout()
            path = self.figures_dir / 'class_distribution.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/class_distribution.png'
        except Exception as e:
            logger.debug(f"Failed to create class distribution chart: {e}")
            return ""

    # =========================================================================
    # PERMUTATION FEATURE IMPORTANCE (Phase 3)
    # =========================================================================

    def add_permutation_importance(self, model, X, y, n_repeats: int = 10):
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
            self._raw_data['permutation_importance'] = result

            feature_names = list(X.columns) if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]

            # Sort by importance
            sorted_idx = result.importances_mean.argsort()[::-1]
            top_n = min(20, len(sorted_idx))
            top_idx = sorted_idx[:top_n]

            # Create chart
            fig_path = self._create_permutation_importance_chart(
                result, feature_names, top_idx
            )

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
            self._store_section_data('permutation_importance', 'Permutation Importance', {
                'top_features': [feature_names[idx] for idx in top_idx],
                'importance_values': {feature_names[idx]: float(result.importances_mean[idx]) for idx in top_idx},
            }, [{'type': 'importance_bar'}])

        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")

    def _create_permutation_importance_chart(self, result, feature_names: List[str],
                                              top_idx) -> str:
        """Create permutation importance chart with error bars."""
        try:
            import matplotlib.pyplot as plt

            n = len(top_idx)
            fig, ax = plt.subplots(figsize=(10, max(6, n * 0.4)))

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

            plt.tight_layout()
            path = self.figures_dir / 'permutation_importance.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/permutation_importance.png'
        except Exception as e:
            logger.debug(f"Failed to create permutation importance chart: {e}")
            return ""

    # =========================================================================
    # PARTIAL DEPENDENCE PLOTS (Phase 4)
    # =========================================================================

    def add_partial_dependence(self, model, X, feature_names: List[str] = None,
                                top_n: int = 3):
        """
        Add Partial Dependence Plots (PDP) with ICE lines:
        - sklearn.inspection.partial_dependence
        - ICE (Individual Conditional Expectation) background lines
        - Top N features from permutation importance or native importance
        """
        try:
            from sklearn.inspection import partial_dependence

            if feature_names is None:
                feature_names = list(X.columns) if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]

            # Determine top features
            top_features = []
            if 'permutation_importance' in self._raw_data:
                perm_result = self._raw_data['permutation_importance']
                sorted_idx = perm_result.importances_mean.argsort()[::-1][:top_n]
                top_features = [feature_names[i] for i in sorted_idx]
            elif hasattr(model, 'feature_importances_'):
                sorted_idx = np.argsort(model.feature_importances_)[::-1][:top_n]
                top_features = [feature_names[i] for i in sorted_idx]
            else:
                top_features = feature_names[:top_n]

            # Create PDP chart
            fig_path = self._create_pdp_chart(model, X, top_features, feature_names)

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
            self._store_section_data('partial_dependence', 'Partial Dependence', {'top_features': top_features})

        except Exception as e:
            logger.warning(f"Partial dependence analysis failed: {e}")

    def _create_pdp_chart(self, model, X, top_features: List[str],
                           all_feature_names: List[str]) -> str:
        """Create PDP charts with ICE background lines."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.inspection import partial_dependence

            n = len(top_features)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
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
            plt.tight_layout()

            path = self.figures_dir / 'pdp_ice.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/pdp_ice.png'
        except Exception as e:
            logger.debug(f"Failed to create PDP chart: {e}")
            return ""

    # =========================================================================
    # STATISTICAL SIGNIFICANCE TESTING (Phase 5)
    # =========================================================================

    def add_statistical_tests(self, y_true, y_pred, y_prob=None):
        """
        Add statistical significance testing:
        - Bootstrap CI for AUC (1000 iterations)
        - Histogram of bootstrap AUC distribution with CI bands
        """
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)

            # Bootstrap accuracy
            n_boot = 1000
            boot_accuracies = []
            n = len(y_true_arr)

            for _ in range(n_boot):
                idx = np.random.choice(n, n, replace=True)
                boot_accuracies.append(accuracy_score(y_true_arr[idx], y_pred_arr[idx]))

            acc_mean = np.mean(boot_accuracies)
            acc_ci_lower = np.percentile(boot_accuracies, 2.5)
            acc_ci_upper = np.percentile(boot_accuracies, 97.5)

            # Bootstrap AUC if probabilities available
            auc_data = None
            if y_prob is not None:
                y_prob_arr = np.asarray(y_prob)
                auc_data = self._bootstrap_auc_ci(y_true_arr, y_prob_arr, n_boot)

            # Create visualization
            fig_path = self._create_bootstrap_auc_chart(
                boot_accuracies, auc_data
            )

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
            significant = auc_data['ci_lower'] > 0.5 if auc_data else None
            self._content.append(content)
            self._store_section_data('statistical_tests', 'Statistical Significance', {
                'acc_mean': float(acc_mean),
                'n_bootstraps': n_boot,
                'auc_ci': auc_ci,
                'significant': significant,
            })

        except Exception as e:
            logger.warning(f"Statistical testing failed: {e}")

    def _bootstrap_auc_ci(self, y_true, y_prob, n_boot: int = 1000) -> Dict:
        """Compute bootstrap confidence interval for AUC."""
        from sklearn.metrics import roc_auc_score

        boot_aucs = []
        n = len(y_true)

        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            try:
                if len(np.unique(y_true[idx])) > 1:
                    boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
            except:
                pass

        if not boot_aucs:
            return None

        return {
            'values': boot_aucs,
            'mean': np.mean(boot_aucs),
            'std': np.std(boot_aucs),
            'ci_lower': np.percentile(boot_aucs, 2.5),
            'ci_upper': np.percentile(boot_aucs, 97.5),
        }

    def _create_bootstrap_auc_chart(self, boot_accuracies: List[float],
                                     auc_data: Dict = None) -> str:
        """Create bootstrap distribution histogram with CI bands."""
        try:
            import matplotlib.pyplot as plt

            n_plots = 2 if auc_data else 1
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
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

            plt.tight_layout()
            path = self.figures_dir / 'bootstrap_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/bootstrap_analysis.png'
        except Exception as e:
            logger.debug(f"Failed to create bootstrap chart: {e}")
            return ""

    # =========================================================================
    # SCORE DISTRIBUTION BY CLASS (Phase 6)
    # =========================================================================

    def add_score_distribution(self, y_true, y_prob, labels: List[str] = None):
        """
        Add predicted probability distribution by actual class:
        - KDE/histogram of predicted probabilities split by actual class
        - Overlap region shading
        - KL divergence
        - Optimal threshold annotation
        """
        try:
            from scipy import stats as scipy_stats
            from sklearn.metrics import roc_curve

            y_true_arr = np.asarray(y_true)
            y_prob_arr = np.asarray(y_prob)

            unique_classes = np.unique(y_true_arr)
            if labels is None:
                labels = [f'Class {c}' for c in unique_classes]

            # Separate probabilities by class
            class_probs = {}
            for cls, label in zip(unique_classes, labels):
                class_probs[label] = y_prob_arr[y_true_arr == cls]

            # Calculate KL divergence between class distributions
            kl_div = None
            if len(unique_classes) == 2:
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
            if len(unique_classes) == 2:
                fpr, tpr, thresholds = roc_curve(y_true_arr, y_prob_arr)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            else:
                optimal_threshold = 0.5

            # Create visualization
            fig_path = self._create_score_distribution_chart(
                class_probs, optimal_threshold
            )

            content = f"""
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
            self._store_section_data('score_distribution', 'Score Distribution', {
                'n_classes': len(unique_classes),
                'optimal_threshold': float(optimal_threshold),
            })

        except Exception as e:
            logger.warning(f"Score distribution analysis failed: {e}")

    def _create_score_distribution_chart(self, class_probs: Dict[str, np.ndarray],
                                          optimal_threshold: float) -> str:
        """Create score distribution chart with overlap shading."""
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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
                    except:
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

            plt.tight_layout()
            path = self.figures_dir / 'score_distribution.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/score_distribution.png'
        except Exception as e:
            logger.debug(f"Failed to create score distribution chart: {e}")
            return ""

    # =========================================================================
    # FEATURE INTERACTION DETECTION (Phase 7)
    # =========================================================================

    def add_feature_interactions(self, shap_values, feature_names: List[str],
                                 X_sample=None, model=None, top_n: int = 3):
        """
        Add feature interaction detection:
        - SHAP interaction values for tree models
        - Pairwise SHAP correlation fallback
        - 2D scatter colored by interaction value
        """
        try:
            import shap

            # Extract SHAP values
            if hasattr(shap_values, 'values'):
                values = shap_values.values
            else:
                values = shap_values

            if len(values.shape) == 3:
                values = values[:, :, 1]  # Binary classification

            # Try SHAP interaction values for tree models
            interaction_data = None
            try:
                if model is not None and hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                    interaction_values = explainer.shap_interaction_values(X_sample)
                    if isinstance(interaction_values, list):
                        interaction_values = interaction_values[1]  # Binary
                    interaction_data = interaction_values
            except:
                pass

            # Fallback: pairwise SHAP correlation
            n_features = min(len(feature_names), values.shape[1])
            shap_corr = np.corrcoef(values.T[:n_features])

            # Find top interactions
            interactions = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interactions.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'correlation': abs(shap_corr[i, j]),
                        'idx_1': i,
                        'idx_2': j,
                    })

            interactions.sort(key=lambda x: x['correlation'], reverse=True)
            top_interactions = interactions[:top_n]

            # Create visualization
            fig_path = self._create_interaction_chart(
                values, feature_names, X_sample, top_interactions
            )

            # Build table
            table_md = "| Feature 1 | Feature 2 | SHAP Correlation | Strength |\n|-----------|-----------|-----------------|----------|\n"
            for inter in top_interactions:
                strength = "Strong" if inter['correlation'] > 0.5 else ("Moderate" if inter['correlation'] > 0.3 else "Weak")
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
            self._store_section_data('feature_interactions', 'Feature Interactions', {
                'n_interactions': len(top_interactions),
                'top_pairs': [{'f1': i['feature_1'], 'f2': i['feature_2'], 'corr': i['correlation']}
                              for i in top_interactions],
            })

        except Exception as e:
            logger.warning(f"Feature interaction analysis failed: {e}")

    def _create_interaction_chart(self, shap_values, feature_names: List[str],
                                   X_sample, top_interactions: List[Dict]) -> str:
        """Create feature interaction scatter plots."""
        try:
            import matplotlib.pyplot as plt

            n = len(top_interactions)
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
            if n == 1:
                axes = [axes]

            X_arr = X_sample if isinstance(X_sample, np.ndarray) else X_sample.values if hasattr(X_sample, 'values') else np.array(X_sample)

            for i, inter in enumerate(top_interactions):
                ax = axes[i]
                idx1 = inter['idx_1']
                idx2 = inter['idx_2']

                feat1_vals = X_arr[:, idx1]
                feat2_vals = X_arr[:, idx2]
                shap_vals = shap_values[:, idx1]

                scatter = ax.scatter(feat1_vals, feat2_vals, c=shap_vals,
                                   cmap='coolwarm', alpha=0.6, s=20)
                plt.colorbar(scatter, ax=ax, label='SHAP Value', shrink=0.8)

                ax.set_xlabel(inter['feature_1'][:15], fontsize=10)
                ax.set_ylabel(inter['feature_2'][:15], fontsize=10)
                ax.set_title(f"r={inter['correlation']:.3f}", fontsize=11, fontweight='medium')

            fig.suptitle('Feature Interaction Plots', fontsize=14, fontweight='bold',
                        color=self.theme['primary'], y=1.05)
            plt.tight_layout()

            path = self.figures_dir / 'feature_interactions.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/feature_interactions.png'
        except Exception as e:
            logger.debug(f"Failed to create interaction chart: {e}")
            return ""

    # =========================================================================
    # INTERPRETABILITY ANALYSIS (LIME / PERTURBATION)
    # =========================================================================

    def add_interpretability_analysis(self, model, X_sample, y_pred,
                                       feature_names: List[str], top_n: int = 5):
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
            X_arr = X_sample if isinstance(X_sample, np.ndarray) else (
                X_sample.values if hasattr(X_sample, 'values') else np.array(X_sample))
            y_pred_arr = np.asarray(y_pred)

            # Get prediction confidences
            has_proba = hasattr(model, 'predict_proba')
            if has_proba:
                probas = model.predict_proba(X_arr)
                confidences = np.max(probas, axis=1)
            else:
                confidences = np.ones(len(y_pred_arr))

            sorted_idx = np.argsort(confidences)
            most_confident = sorted_idx[-top_n:][::-1]
            least_confident = sorted_idx[:top_n]

            # Try LIME, fall back to perturbation
            method = 'perturbation'
            explanations = []

            try:
                from lime.lime_tabular import LimeTabularExplainer
                explainer = LimeTabularExplainer(
                    X_arr, feature_names=feature_names, mode='classification',
                    discretize_continuous=True, random_state=42)
                method = 'lime'

                for idx in list(most_confident) + list(least_confident):
                    exp = explainer.explain_instance(X_arr[idx], model.predict_proba if has_proba else model.predict,
                                                     num_features=min(10, len(feature_names)))
                    contributions = {f: w for f, w in exp.as_list()}
                    explanations.append({
                        'sample_idx': int(idx),
                        'confidence': float(confidences[idx]),
                        'predicted': int(y_pred_arr[idx]),
                        'contributions': contributions,
                    })
            except ImportError:
                # Perturbation-based fallback
                for idx in list(most_confident) + list(least_confident):
                    contributions = self._perturbation_explain(
                        model, X_arr, idx, feature_names, has_proba)
                    explanations.append({
                        'sample_idx': int(idx),
                        'confidence': float(confidences[idx]),
                        'predicted': int(y_pred_arr[idx]),
                        'contributions': contributions,
                    })

            # Counterfactual generation
            counterfactuals = []
            for idx in most_confident[:3]:
                cf = self._generate_counterfactual(model, X_arr, idx, feature_names, has_proba)
                if cf:
                    counterfactuals.append(cf)

            # Create visualization
            fig_path = self._create_lime_chart(explanations[:top_n], feature_names)

            # Build content
            content = f"""
# Interpretability Analysis

Local explanations showing why the model makes specific predictions.

**Method:** {'LIME (Local Interpretable Model-agnostic Explanations)' if method == 'lime' else 'Perturbation-based feature contribution'}

## Most Confident Predictions

| Sample | Predicted | Confidence | Top Contributing Features |
|--------|-----------|------------|--------------------------|
"""
            for exp in explanations[:top_n]:
                top_contribs = sorted(exp['contributions'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                contribs_str = ', '.join([f"{f[:15]}={w:+.3f}" for f, w in top_contribs])
                content += f"| #{exp['sample_idx']} | {exp['predicted']} | {exp['confidence']:.4f} | {contribs_str} |\n"

            content += f"""
## Least Confident Predictions

| Sample | Predicted | Confidence | Top Contributing Features |
|--------|-----------|------------|--------------------------|
"""
            for exp in explanations[top_n:]:
                top_contribs = sorted(exp['contributions'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                contribs_str = ', '.join([f"{f[:15]}={w:+.3f}" for f, w in top_contribs])
                content += f"| #{exp['sample_idx']} | {exp['predicted']} | {exp['confidence']:.4f} | {contribs_str} |\n"

            if fig_path:
                content += f"""
## Feature Contributions (Most Confident Samples)

![Interpretability Analysis]({fig_path})

"""

            if counterfactuals:
                content += """## Counterfactual Analysis

What minimal changes would flip the prediction?

| Sample | Original Pred | Changes Required |
|--------|--------------|-----------------|
"""
                for cf in counterfactuals:
                    changes = ', '.join([f"{c['feature'][:15]}: {c['from']:.2f}→{c['to']:.2f}"
                                        for c in cf['changes'][:3]])
                    content += f"| #{cf['sample_idx']} | {cf['original_pred']} | {changes} |\n"

            content += "\n---\n"
            self._content.append(content)
            self._store_section_data('interpretability', 'Interpretability Analysis', {
                'method': method,
                'n_explained': len(explanations),
            }, [{'type': 'importance_bar'}])

        except Exception as e:
            logger.warning(f"Interpretability analysis failed: {e}")

    def _perturbation_explain(self, model, X_arr, idx, feature_names, has_proba):
        """Compute feature contributions via single-feature perturbation."""
        contributions = {}
        original = X_arr[idx].copy()

        if has_proba:
            base_pred = model.predict_proba(original.reshape(1, -1))[0]
        else:
            base_pred = model.predict(original.reshape(1, -1))[0]

        for i, feat in enumerate(feature_names):
            perturbed = original.copy()
            col_std = np.std(X_arr[:, i])
            perturbed[i] += col_std if col_std > 0 else 1.0
            if has_proba:
                new_pred = model.predict_proba(perturbed.reshape(1, -1))[0]
                diff = float(np.max(np.abs(new_pred - base_pred)))
            else:
                new_pred = model.predict(perturbed.reshape(1, -1))[0]
                diff = float(abs(new_pred - base_pred))
            contributions[feat] = diff

        return contributions

    def _generate_counterfactual(self, model, X_arr, idx, feature_names, has_proba):
        """Generate counterfactual by greedy perturbation of features."""
        try:
            original = X_arr[idx].copy()
            original_pred = int(model.predict(original.reshape(1, -1))[0])

            # Get feature importance via perturbation
            contributions = self._perturbation_explain(model, X_arr, idx, feature_names, has_proba)
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

            perturbed = original.copy()
            changes = []

            for feat, _ in sorted_features[:len(feature_names)]:
                feat_idx = feature_names.index(feat)
                col_mean = np.mean(X_arr[:, feat_idx])
                old_val = perturbed[feat_idx]
                perturbed[feat_idx] = col_mean

                new_pred = int(model.predict(perturbed.reshape(1, -1))[0])
                changes.append({'feature': feat, 'from': float(old_val), 'to': float(col_mean)})

                if new_pred != original_pred:
                    return {
                        'sample_idx': int(idx),
                        'original_pred': original_pred,
                        'new_pred': new_pred,
                        'changes': changes,
                    }

            return None
        except Exception:
            return None

    def _create_lime_chart(self, explanations, feature_names):
        """Create horizontal bar chart of feature contributions."""
        try:
            import matplotlib.pyplot as plt

            n_exp = min(len(explanations), 4)
            if n_exp == 0:
                return ""

            fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 5))
            if n_exp == 1:
                axes = [axes]

            for i, exp in enumerate(explanations[:n_exp]):
                ax = axes[i]
                sorted_contribs = sorted(exp['contributions'].items(), key=lambda x: x[1])[-10:]
                names = [c[0][:20] for c in sorted_contribs]
                vals = [c[1] for c in sorted_contribs]
                colors = [self.theme['success'] if v > 0 else self.theme['danger'] for v in vals]

                ax.barh(range(len(names)), vals, color=colors, alpha=0.85)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names, fontsize=8)
                ax.set_title(f"Sample #{exp['sample_idx']}\n(conf={exp['confidence']:.3f})",
                            fontsize=10, fontweight='bold', color=self.theme['primary'])
                ax.axvline(x=0, color='gray', linewidth=0.5)

            fig.suptitle('Feature Contributions (Local Explanations)', fontsize=14,
                        fontweight='bold', color=self.theme['primary'], y=1.02)
            plt.tight_layout()
            path = self.figures_dir / 'interpretability_analysis.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/interpretability_analysis.png'
        except Exception as e:
            logger.debug(f"Failed to create interpretability chart: {e}")
            return ""

    # =========================================================================
    # DEEP LEARNING ANALYSIS
    # =========================================================================

    def add_deep_learning_analysis(self, model, X_sample=None, layer_names: List[str] = None,
                                    training_history: Dict = None):
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
                    loss_hist = training_history.get('loss', [])
                    val_loss_hist = training_history.get('val_loss', [])

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

            self._store_section_data('deep_learning', 'Deep Learning Analysis', {
                'is_nn': True,
                'training_history': training_history,
            }, [{'type': 'line', 'path': p} for p in fig_paths])

        except Exception as e:
            logger.warning(f"Deep learning analysis failed: {e}")

    def _is_neural_network(self, model) -> bool:
        """Check if model is a neural network using string-based type check (no framework imports)."""
        model_type = str(type(model))
        nn_indicators = [
            'torch.nn.Module', 'torch.nn.modules',
            'keras.Model', 'keras.engine', 'keras.src',
            'tensorflow.keras', 'tensorflow.python.keras',
        ]
        return any(indicator in model_type for indicator in nn_indicators)

    def _create_training_curves(self, history: Dict) -> str:
        """Create training curves with loss/val_loss and early stopping annotation."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2 if 'accuracy' in history else 1,
                                     figsize=(12 if 'accuracy' in history else 8, 5))
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
            if 'accuracy' in history and len(axes) > 1:
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

            plt.tight_layout()
            path = self.figures_dir / 'training_curves.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/training_curves.png'
        except Exception as e:
            logger.debug(f"Failed to create training curves: {e}")
            return ""

    def _get_nn_architecture_info(self, model) -> Dict:
        """Extract neural network architecture information."""
        info = {}
        model_type = str(type(model))

        # PyTorch
        if 'torch' in model_type:
            info['framework'] = 'PyTorch'
            try:
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                info['total_params'] = total
                info['trainable_params'] = trainable
                info['n_layers'] = len(list(model.modules())) - 1
            except Exception:
                pass

        # Keras / TensorFlow
        elif 'keras' in model_type or 'tensorflow' in model_type:
            info['framework'] = 'Keras/TensorFlow'
            try:
                info['total_params'] = model.count_params()
                trainable_count = sum(
                    int(np.prod(w.shape)) for w in model.trainable_weights
                )
                info['trainable_params'] = trainable_count
                info['n_layers'] = len(model.layers)
            except Exception:
                pass

        return info

    def _compute_gradient_attribution(self, model, X_sample) -> Dict:
        """Attempt simple gradient×input attribution. Returns empty dict on failure."""
        try:
            model_type = str(type(model))

            if 'torch' in model_type:
                import torch
                X_tensor = torch.tensor(X_sample[:10], dtype=torch.float32, requires_grad=True)
                output = model(X_tensor)
                if output.dim() > 1:
                    output = output[:, -1]
                output.sum().backward()
                grads = X_tensor.grad.detach().numpy()
                attributions = np.abs(grads * X_sample[:10]).mean(axis=0)
                return {'attributions': attributions}

            elif 'keras' in model_type or 'tensorflow' in model_type:
                import tensorflow as tf
                X_tensor = tf.constant(X_sample[:10], dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(X_tensor)
                    output = model(X_tensor)
                grads = tape.gradient(output, X_tensor).numpy()
                attributions = np.abs(grads * X_sample[:10]).mean(axis=0)
                return {'attributions': attributions}

        except Exception as e:
            logger.debug(f"Gradient attribution failed: {e}")

        return {}

    # =========================================================================
    # DATA DRIFT MONITORING
    # =========================================================================

    def add_drift_analysis(self, X_reference, X_current, feature_names: List[str] = None,
                           psi_warn: float = 0.1, psi_alert: float = 0.25):
        """
        Add data drift monitoring analysis between reference and current datasets.

        Uses PSI, KS test, and Jensen-Shannon divergence to detect feature-level drift.

        Args:
            X_reference: Reference dataset (training data)
            X_current: Current dataset (production/new data)
            feature_names: Feature names (auto-detected from DataFrame)
            psi_warn: PSI threshold for warning (default 0.1)
            psi_alert: PSI threshold for alert (default 0.25)
        """
        try:
            from scipy.stats import ks_2samp
            from scipy.spatial.distance import jensenshannon

            X_ref = np.asarray(X_reference)
            X_cur = np.asarray(X_current)

            if feature_names is None:
                if hasattr(X_reference, 'columns'):
                    feature_names = list(X_reference.columns)
                else:
                    feature_names = [f'Feature_{i}' for i in range(X_ref.shape[1])]

            n_features = min(X_ref.shape[1], X_cur.shape[1], len(feature_names))
            drift_results = []

            for i in range(n_features):
                ref_col = X_ref[:, i].astype(float)
                cur_col = X_cur[:, i].astype(float)

                # Remove NaN
                ref_col = ref_col[~np.isnan(ref_col)]
                cur_col = cur_col[~np.isnan(cur_col)]

                if len(ref_col) < 10 or len(cur_col) < 10:
                    continue

                # PSI
                psi = self._calculate_psi(ref_col, cur_col)

                # KS test
                ks_stat, ks_pval = ks_2samp(ref_col, cur_col)

                # Jensen-Shannon divergence
                js_div = self._calculate_js_divergence(ref_col, cur_col)

                # Severity
                if psi >= psi_alert:
                    status = "ALERT"
                elif psi >= psi_warn:
                    status = "WARNING"
                else:
                    status = "OK"

                drift_results.append({
                    'feature': feature_names[i],
                    'psi': psi,
                    'ks_stat': ks_stat,
                    'ks_pval': ks_pval,
                    'js_div': js_div,
                    'status': status,
                })

            if not drift_results:
                return

            # Create visualization
            fig_path = self._create_drift_heatmap(drift_results, psi_warn, psi_alert)

            # Summary stats
            n_alert = sum(1 for r in drift_results if r['status'] == 'ALERT')
            n_warn = sum(1 for r in drift_results if r['status'] == 'WARNING')
            n_ok = sum(1 for r in drift_results if r['status'] == 'OK')

            # Build table
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

**Total Features Analyzed:** {len(drift_results)}

## Feature-Level Drift Analysis

{table_md}

## Drift Heatmap

![Data Drift Heatmap]({fig_path})

## Interpretation

- **PSI** (Population Stability Index): < {psi_warn} = stable, {psi_warn}-{psi_alert} = moderate drift, > {psi_alert} = significant drift
- **KS Test**: p-value < 0.05 suggests statistically significant distribution change
- **JS Divergence**: Symmetric measure of distribution difference (0 = identical)

---
"""
            self._content.append(content)

            self._store_section_data('drift_analysis', 'Data Drift Monitoring', {
                'drift_results': drift_results,
                'n_alert': n_alert, 'n_warn': n_warn, 'n_ok': n_ok,
            }, [{'type': 'heatmap', 'path': fig_path}])

        except Exception as e:
            logger.warning(f"Drift analysis failed: {e}")

    def _calculate_psi(self, reference, current, n_bins: int = 10) -> float:
        """Calculate Population Stability Index between two distributions."""
        eps = 1e-4

        # Create bins from reference distribution
        breakpoints = np.linspace(np.min(reference), np.max(reference), n_bins + 1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = ref_counts / len(reference) + eps
        cur_pct = cur_counts / len(current) + eps

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def _calculate_js_divergence(self, reference, current, n_bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence between two distributions."""
        try:
            from scipy.spatial.distance import jensenshannon

            all_data = np.concatenate([reference, current])
            bins = np.linspace(np.min(all_data), np.max(all_data), n_bins + 1)

            ref_hist = np.histogram(reference, bins=bins, density=True)[0] + 1e-10
            cur_hist = np.histogram(current, bins=bins, density=True)[0] + 1e-10

            ref_hist = ref_hist / ref_hist.sum()
            cur_hist = cur_hist / cur_hist.sum()

            return float(jensenshannon(ref_hist, cur_hist))
        except Exception:
            return 0.0

    def _create_drift_heatmap(self, drift_results: List[Dict],
                               psi_warn: float, psi_alert: float) -> str:
        """Create drift heatmap visualization."""
        try:
            import matplotlib.pyplot as plt
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

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, n * 0.35)),
                                            gridspec_kw={'width_ratios': [3, 1]})

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
            plt.colorbar(im, ax=ax1, shrink=0.8, label='Metric Value')

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

            plt.tight_layout()
            path = self.figures_dir / 'drift_heatmap.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/drift_heatmap.png'
        except Exception as e:
            logger.debug(f"Failed to create drift heatmap: {e}")
            return ""

    # =========================================================================
    # MODEL CARD (Phase 8)
    # =========================================================================

    def add_model_card(self, model, results: Dict[str, Any],
                       intended_use: str = "", limitations: str = "",
                       ethical: str = ""):
        """
        Add Model Card section following Google Model Card standard:
        - Model details (type, version, framework)
        - Intended use and users
        - Limitations and out-of-scope uses
        - Ethical considerations
        - Auto-generated limitations from analysis results
        """
        import platform

        model_type = type(model).__name__ if model else 'Unknown'
        model_params = {}
        if model and hasattr(model, 'get_params'):
            try:
                model_params = model.get_params()
            except:
                pass

        # Auto-generate limitations from analysis
        auto_limitations = []
        metrics = results.get('metrics', {})
        score = metrics.get('accuracy', results.get('final_score', 0))

        if score < 0.8:
            auto_limitations.append("Model accuracy below 80% — may not be suitable for high-stakes decisions")
        if score > 0.98:
            auto_limitations.append("Very high accuracy may indicate overfitting — validate on new data")

        # Check class imbalance
        if 'class_distribution' in results:
            dist = results['class_distribution']
            if isinstance(dist, dict):
                counts = list(dist.values())
                if len(counts) >= 2:
                    ratio = max(counts) / min(counts)
                    if ratio > 3:
                        auto_limitations.append(f"Training data is imbalanced (ratio {ratio:.1f}:1) — predictions for minority class may be less reliable")

        # Check feature count
        n_features = results.get('n_features', 0)
        n_samples = results.get('n_samples', 0)
        if n_features > 0 and n_samples > 0 and n_features > n_samples / 10:
            auto_limitations.append("High feature-to-sample ratio — risk of overfitting")

        auto_limitations.append("Model performance may degrade with data distribution shift")
        auto_limitations.append("Predictions are based on correlations, not causal relationships")

        # Build content
        limitations_combined = limitations
        if auto_limitations:
            auto_text = "\n".join([f"- {lim}" for lim in auto_limitations])
            limitations_combined = f"{limitations}\n\n**Auto-detected Limitations:**\n\n{auto_text}" if limitations else f"**Auto-detected Limitations:**\n\n{auto_text}"

        # Model parameters table
        params_md = ""
        if model_params:
            params_md = "| Parameter | Value |\n|-----------|-------|\n"
            for k, v in list(model_params.items())[:15]:
                v_str = str(v)[:40]
                params_md += f"| {k} | `{v_str}` |\n"

        content = f"""
# Model Card

Following the [Google Model Card](https://modelcards.withgoogle.com/) standard for
transparent model documentation.

## Model Details

| Property | Value |
|----------|-------|
| Model Type | {model_type} |
| Framework | scikit-learn |
| Python Version | {platform.python_version()} |
| Training Date | {datetime.now().strftime('%Y-%m-%d')} |
| Problem Type | {results.get('problem_type', 'Classification')} |

{params_md}

## Intended Use

{intended_use or "This model is intended for automated prediction tasks based on the provided training data. It should be used as a decision-support tool, not as the sole basis for critical decisions."}

## Limitations

{limitations_combined}

## Ethical Considerations

{ethical or "- Ensure training data is representative and does not encode historical biases" + chr(10) + "- Model predictions should be monitored for fairness across demographic groups" + chr(10) + "- Human oversight is recommended for high-impact decisions" + chr(10) + "- Regular auditing for bias and drift is advised"}

## Performance Summary

| Metric | Value |
|--------|-------|
"""
        for name, value in metrics.items():
            if isinstance(value, float):
                content += f"| {name.replace('_', ' ').title()} | {value:.4f} |\n"
            else:
                content += f"| {name.replace('_', ' ').title()} | {value} |\n"

        content += """
---
"""
        self._content.append(content)
        self._store_section_data('model_card', 'Model Card', {
            'model_type': model_type,
            'metric_count': len(metrics),
        })

    # =========================================================================
    # FAIRNESS & BIAS AUDIT
    # =========================================================================

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
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            y_prob_arr = np.asarray(y_prob) if y_prob is not None else None

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
                # Calculate reference group (largest group)
                group_names = list(metrics.keys())
                ref_group = max(group_names, key=lambda g: metrics[g]['n'])
                ref_pos_rate = metrics[ref_group]['positive_rate']

                for group, m in sorted(metrics.items(), key=lambda x: -x[1]['n']):
                    di = m['positive_rate'] / ref_pos_rate if ref_pos_rate > 0 else 0
                    content += (f"| {group} (n={m['n']}) | {m['positive_rate']:.3f} | "
                               f"{m['tpr']:.3f} | {m['fpr']:.3f} | {m['ppv']:.3f} | {di:.3f} |\n")

                # Pass/Fail based on 80% rule
                all_di = []
                for group, m in metrics.items():
                    di = m['positive_rate'] / ref_pos_rate if ref_pos_rate > 0 else 0
                    all_di.append(di)

                min_di = min(all_di) if all_di else 0
                passes_80 = min_di >= 0.8

                content += f"""
**Disparate Impact Assessment:** {'PASS' if passes_80 else 'FAIL'} (min ratio: {min_di:.3f}, threshold: 0.80)

"""

            # Create radar chart for first feature
            if all_metrics:
                first_feat = list(all_metrics.keys())[0]
                fig_path = self._create_fairness_radar_chart(all_metrics[first_feat], first_feat)

                if fig_path:
                    content += f"""## Fairness Radar Chart ({first_feat})

![Fairness Radar]({fig_path})

"""

            # Mitigation recommendations
            content += """## Mitigation Recommendations

- **Pre-processing:** Re-sample or re-weight training data to achieve demographic parity
- **In-processing:** Add fairness constraints to the model objective function
- **Post-processing:** Adjust decision thresholds per group to equalize outcome rates
- **Monitoring:** Continuously track fairness metrics in production

---
"""
            self._content.append(content)

            self._store_section_data('fairness_audit', 'Fairness & Bias Audit', {
                'metrics': {k: {gk: {mk: mv for mk, mv in gv.items()}
                               for gk, gv in v.items()} for k, v in all_metrics.items()},
            }, [{'type': 'radar', 'path': fig_path}] if all_metrics else [])

        except Exception as e:
            logger.warning(f"Fairness audit failed: {e}")

    def _compute_fairness_metrics(self, y_true, y_pred, y_prob, groups) -> Dict:
        """Compute fairness metrics per group: demographic parity, equalized odds, predictive parity."""
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return {}

        metrics = {}
        for group in unique_groups:
            mask = groups == group
            n = mask.sum()
            if n < 5:
                continue

            yt = y_true[mask]
            yp = y_pred[mask]

            positive_rate = yp.mean()

            # TPR (sensitivity)
            pos_mask = yt == 1
            tpr = yp[pos_mask].mean() if pos_mask.sum() > 0 else 0.0

            # FPR
            neg_mask = yt == 0
            fpr = yp[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

            # PPV (precision)
            pred_pos_mask = yp == 1
            ppv = yt[pred_pos_mask].mean() if pred_pos_mask.sum() > 0 else 0.0

            metrics[str(group)] = {
                'n': int(n),
                'positive_rate': float(positive_rate),
                'tpr': float(tpr),
                'fpr': float(fpr),
                'ppv': float(ppv),
            }

        return metrics

    def _create_fairness_radar_chart(self, metrics_per_group: Dict, feature_name: str) -> str:
        """Create radar/spider chart showing fairness metrics per group."""
        try:
            import matplotlib.pyplot as plt

            metric_names = ['Pos Rate', 'TPR', 'FPR', 'PPV']
            n_metrics = len(metric_names)
            angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Close the polygon

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            colors = self.theme['chart_palette']

            for i, (group, m) in enumerate(metrics_per_group.items()):
                values = [m['positive_rate'], m['tpr'], m['fpr'], m['ppv']]
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

            plt.tight_layout()
            path = self.figures_dir / f'fairness_radar_{feature_name[:20].replace(" ", "_")}.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return f'figures/{path.name}'
        except Exception as e:
            logger.debug(f"Failed to create fairness radar chart: {e}")
            return ""

    # =========================================================================
    # REGRESSION ANALYSIS (Phase 9)
    # =========================================================================

    def add_regression_analysis(self, y_true, y_pred):
        """
        Add regression analysis with:
        - 2x2 subplot: Predicted vs Actual, Residuals, Q-Q plot, Residual histogram
        - R², MAE, RMSE, MAPE metrics
        - Breusch-Pagan heteroscedasticity test
        """
        try:
            from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                         r2_score, mean_absolute_percentage_error)

            y_true_arr = np.asarray(y_true, dtype=float)
            y_pred_arr = np.asarray(y_pred, dtype=float)
            residuals = y_true_arr - y_pred_arr

            # Calculate metrics
            r2 = r2_score(y_true_arr, y_pred_arr)
            mae = mean_absolute_error(y_true_arr, y_pred_arr)
            rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
            try:
                mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100
            except:
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
            self._store_section_data('regression', 'Regression Analysis', {'r2': r2, 'rmse': rmse})

        except Exception as e:
            logger.warning(f"Regression analysis failed: {e}")

    def _detect_heteroscedasticity(self, y_pred, residuals) -> Dict:
        """Perform Breusch-Pagan test for heteroscedasticity."""
        try:
            from scipy import stats as scipy_stats

            # Simple Breusch-Pagan approximation
            # Regress squared residuals on predictions
            res_sq = residuals ** 2
            n = len(residuals)

            # Correlation between |residuals| and predictions
            corr, p_value = scipy_stats.pearsonr(np.abs(residuals), y_pred)

            # Approximate chi-squared test
            bp_stat = n * corr ** 2
            bp_p_value = 1 - scipy_stats.chi2.cdf(bp_stat, df=1)

            is_heteroscedastic = bp_p_value < 0.05

            return {
                'statistic': f'{bp_stat:.4f}',
                'p_value': f'{bp_p_value:.4f}',
                'result': 'Heteroscedastic' if is_heteroscedastic else 'Homoscedastic',
                'is_heteroscedastic': is_heteroscedastic,
            }
        except Exception:
            return {
                'statistic': 'N/A',
                'p_value': 'N/A',
                'result': 'Test unavailable',
                'is_heteroscedastic': False,
            }

    def _create_regression_charts(self, y_true, y_pred, residuals) -> str:
        """Create 2x2 regression diagnostic plots."""
        try:
            import matplotlib.pyplot as plt
            from scipy import stats as scipy_stats

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

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

            plt.tight_layout()
            path = self.figures_dir / 'regression_diagnostics.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()
            return 'figures/regression_diagnostics.png'
        except Exception as e:
            logger.debug(f"Failed to create regression charts: {e}")
            return ""

    def generate(self, filename: str = None) -> Optional[str]:
        """Generate the PDF report using Pandoc + LaTeX."""

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ml_report_{timestamp}.pdf"

        pdf_path = self.output_dir / filename

        # Build markdown content
        markdown = self._build_markdown()

        # Save markdown temporarily
        md_path = self.output_dir / f"{filename.replace('.pdf', '')}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        # Try pandoc conversion
        success = self._convert_with_pandoc(md_path, pdf_path)

        if success and pdf_path.exists():
            logger.info(f"Report generated: {pdf_path}")
            return str(pdf_path)
        else:
            # Fallback to basic reportlab
            logger.warning("Pandoc failed, using fallback PDF generator")
            return self._fallback_pdf_generation(markdown, pdf_path)

    # =========================================================================
    # INTERACTIVE HTML REPORT GENERATION
    # =========================================================================

    def generate_html(self, filename: str = None) -> Optional[str]:
        """
        Generate an interactive HTML report with Plotly charts and collapsible sections.

        Args:
            filename: Output HTML filename. Auto-generated if None.

        Returns:
            Path to generated HTML file, or None on failure.
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"ml_report_{timestamp}.html"

            html_path = self.output_dir / filename
            html_content = self._build_html_document()

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report generated: {html_path}")
            return str(html_path)

        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            return None

    def generate_all(self, filename_base: str = None) -> Dict[str, Optional[str]]:
        """
        Generate both PDF and HTML reports.

        Args:
            filename_base: Base filename (without extension). Auto-generated if None.

        Returns:
            Dict with 'pdf' and 'html' keys mapping to file paths.
        """
        if not filename_base:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_base = f"ml_report_{timestamp}"

        pdf_path = self.generate(f"{filename_base}.pdf")
        html_path = self.generate_html(f"{filename_base}.html")

        return {'pdf': pdf_path, 'html': html_path}

    def _get_html_template(self) -> str:
        """Get base HTML template with inline CSS, sidebar nav, collapsible sections, print styles."""
        t = self.theme
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{title}}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root {{
    --primary: {t['primary']};
    --secondary: {t['secondary']};
    --accent: {t['accent']};
    --success: {t['success']};
    --warning: {t['warning']};
    --danger: {t['danger']};
    --text: {t['text']};
    --muted: {t['muted']};
    --bg: {t['background']};
    --table-header: {t['table_header']};
    --table-alt: {t['table_alt']};
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: {'"Georgia", "Times New Roman", serif' if self.theme_name == 'goldman' else '"Segoe UI", "Helvetica Neue", Arial, sans-serif'};
    color: var(--text);
    background: var(--bg);
    line-height: 1.6;
}}
.sidebar {{
    position: fixed;
    left: 0;
    top: 0;
    width: 260px;
    height: 100vh;
    background: var(--primary);
    color: white;
    overflow-y: auto;
    padding: 20px 0;
    z-index: 100;
}}
.sidebar h2 {{
    padding: 0 20px;
    margin-bottom: 15px;
    font-size: 16px;
    font-weight: 600;
    opacity: 0.9;
}}
.sidebar a {{
    display: block;
    padding: 8px 20px;
    color: rgba(255,255,255,0.8);
    text-decoration: none;
    font-size: 13px;
    transition: all 0.2s;
    border-left: 3px solid transparent;
}}
.sidebar a:hover, .sidebar a.active {{
    background: rgba(255,255,255,0.1);
    color: white;
    border-left-color: var(--accent);
}}
.main-content {{
    margin-left: 260px;
    padding: 40px;
    max-width: 1200px;
}}
.section {{
    background: white;
    border-radius: 8px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid #eee;
}}
.section-header {{
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.section-header h2 {{
    color: var(--primary);
    font-size: 22px;
    margin: 0;
}}
.section-header .toggle {{
    font-size: 18px;
    color: var(--muted);
    transition: transform 0.3s;
}}
.section-header .toggle.collapsed {{
    transform: rotate(-90deg);
}}
.section-body {{ padding-top: 15px; }}
.section-body.hidden {{ display: none; }}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 14px;
}}
th {{
    background: var(--table-header);
    color: white;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
}}
td {{
    padding: 8px 12px;
    border-bottom: 1px solid #eee;
}}
tr:nth-child(even) {{ background: var(--table-alt); }}
tr:hover {{ background: rgba(0,0,0,0.03); }}
.chart-container {{ margin: 20px 0; }}
.chart-container img {{ max-width: 100%; height: auto; border-radius: 4px; }}
.plotly-chart {{ width: 100%; min-height: 400px; }}
.metric-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin: 15px 0;
}}
.metric-card {{
    background: white;
    border: 1px solid #eee;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
    border-top: 3px solid var(--accent);
}}
.metric-card .value {{
    font-size: 28px;
    font-weight: 700;
    color: var(--primary);
}}
.metric-card .label {{
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}}
blockquote {{
    border-left: 4px solid var(--accent);
    padding: 10px 15px;
    margin: 15px 0;
    background: rgba(49,130,206,0.05);
    border-radius: 0 4px 4px 0;
}}
.status-ok {{ color: var(--success); font-weight: 600; }}
.status-warn {{ color: var(--warning); font-weight: 600; }}
.status-alert {{ color: var(--danger); font-weight: 600; }}
.report-header {{
    text-align: center;
    padding: 40px 0;
    border-bottom: 2px solid var(--primary);
    margin-bottom: 30px;
}}
.report-header h1 {{
    font-size: 32px;
    color: var(--primary);
    margin-bottom: 8px;
}}
.report-header .subtitle {{
    font-size: 16px;
    color: var(--muted);
}}
@media print {{
    .sidebar {{ display: none; }}
    .main-content {{ margin-left: 0; padding: 20px; }}
    .section {{ break-inside: avoid; }}
    .section-body.hidden {{ display: block !important; }}
}}
</style>
</head>
<body>
{{sidebar}}
<div class="main-content">
{{header}}
{{content}}
<footer style="text-align:center;padding:30px;color:var(--muted);font-size:12px;">
    Generated by Jotty ML Report Generator | {{date}}
</footer>
</div>
<script>
document.querySelectorAll('.section-header').forEach(header => {{
    header.addEventListener('click', () => {{
        const body = header.nextElementSibling;
        const toggle = header.querySelector('.toggle');
        body.classList.toggle('hidden');
        toggle.classList.toggle('collapsed');
    }});
}});
// Sidebar active tracking
const sections = document.querySelectorAll('.section');
const navLinks = document.querySelectorAll('.sidebar a');
window.addEventListener('scroll', () => {{
    let current = '';
    sections.forEach(section => {{
        const top = section.offsetTop - 100;
        if (window.scrollY >= top) current = section.id;
    }});
    navLinks.forEach(link => {{
        link.classList.toggle('active', link.getAttribute('href') === '#' + current);
    }});
}});
</script>
</body>
</html>"""

    def _build_html_document(self) -> str:
        """Build complete HTML document from section data and markdown content."""
        import re

        title = self._metadata.get('title', 'ML Analysis Report')
        subtitle = self._metadata.get('subtitle', '')
        date = self._metadata.get('date', datetime.now().strftime('%B %d, %Y'))

        template = self._get_html_template()

        # Build sidebar navigation
        nav_items = []
        sections_html = []

        # Convert markdown content to HTML sections
        for i, content_block in enumerate(self._content):
            section_title = self._extract_section_title(content_block)
            section_id = f"section-{i}"

            nav_items.append(f'<a href="#{section_id}">{section_title}</a>')

            section_html = self._render_markdown_to_html(content_block)
            sections_html.append(f"""
<div class="section" id="{section_id}">
    <div class="section-header">
        <h2>{section_title}</h2>
        <span class="toggle">&#9660;</span>
    </div>
    <div class="section-body">
        {section_html}
    </div>
</div>""")

        # Also render any stored section data with Plotly charts
        plotly_scripts = self._generate_plotly_scripts()

        sidebar_html = f"""
<div class="sidebar">
    <h2>{self.theme['header_brand']}</h2>
    {''.join(nav_items)}
</div>"""

        header_html = f"""
<div class="report-header">
    <h1>{title}</h1>
    <div class="subtitle">{subtitle}</div>
    <div class="subtitle">{date}</div>
</div>"""

        content_html = '\n'.join(sections_html) + plotly_scripts

        html = template.replace('{title}', title)
        html = html.replace('{sidebar}', sidebar_html)
        html = html.replace('{header}', header_html)
        html = html.replace('{content}', content_html)
        html = html.replace('{date}', date)

        return html

    def _extract_section_title(self, content: str) -> str:
        """Extract H1 title from markdown content block."""
        import re
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1).strip() if match else 'Section'

    def _render_markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown content to HTML. Handles tables, images, bold, lists."""
        import re
        html = markdown_content.strip()

        # Remove H1 (already in section header)
        html = re.sub(r'^#\s+.+$', '', html, count=1, flags=re.MULTILINE)

        # H2, H3
        html = re.sub(r'^###\s+(.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        html = re.sub(r'^##\s+(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Italic
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # Images - convert to embedded or chart container
        def replace_image(match):
            alt = match.group(1)
            src = match.group(2)
            full_path = self.output_dir / src
            return f'<div class="chart-container"><img src="{src}" alt="{alt}" loading="lazy"><p style="text-align:center;color:var(--muted);font-size:12px;">{alt}</p></div>'

        html = re.sub(r'!\[(.+?)\]\((.+?)\)', replace_image, html)

        # Tables
        html = self._convert_markdown_tables(html)

        # Blockquotes
        lines = html.split('\n')
        new_lines = []
        in_blockquote = False
        for line in lines:
            if line.strip().startswith('>'):
                if not in_blockquote:
                    new_lines.append('<blockquote>')
                    in_blockquote = True
                new_lines.append(line.strip().lstrip('> '))
            else:
                if in_blockquote:
                    new_lines.append('</blockquote>')
                    in_blockquote = False
                new_lines.append(line)
        if in_blockquote:
            new_lines.append('</blockquote>')
        html = '\n'.join(new_lines)

        # Unordered lists
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.*</li>\n?)+', lambda m: f'<ul>{m.group(0)}</ul>', html)

        # Ordered lists
        html = re.sub(r'^\d+\.\s+(.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Horizontal rules
        html = re.sub(r'^---+$', '<hr>', html, flags=re.MULTILINE)

        # Paragraphs (wrap remaining text)
        html = re.sub(r'\n\n+', '</p><p>', html)

        # Clean up empty paragraphs
        html = re.sub(r'<p>\s*</p>', '', html)

        return f'<div>{html}</div>'

    def _convert_markdown_tables(self, html: str) -> str:
        """Convert markdown tables to HTML tables."""
        import re

        def table_replacer(match):
            table_text = match.group(0)
            rows = [r.strip() for r in table_text.strip().split('\n') if r.strip()]

            if len(rows) < 2:
                return table_text

            # Parse header
            header_cells = [c.strip() for c in rows[0].split('|') if c.strip()]

            # Skip separator row
            data_rows = rows[2:] if len(rows) > 2 else []

            table_html = '<table><thead><tr>'
            for cell in header_cells:
                table_html += f'<th>{cell}</th>'
            table_html += '</tr></thead><tbody>'

            for row in data_rows:
                cells = [c.strip() for c in row.split('|') if c.strip()]
                table_html += '<tr>'
                for cell in cells:
                    # Color-code status cells
                    css_class = ''
                    if cell in ('OK', 'Excellent', 'Good', 'PASS', 'Homoscedastic'):
                        css_class = ' class="status-ok"'
                    elif cell in ('WARNING', 'WARN', 'Moderate', 'Needs Improvement'):
                        css_class = ' class="status-warn"'
                    elif cell in ('ALERT', 'FAIL', 'Critical', 'Unstable', 'Heteroscedastic'):
                        css_class = ' class="status-alert"'
                    table_html += f'<td{css_class}>{cell}</td>'
                table_html += '</tr>'

            table_html += '</tbody></table>'
            return table_html

        # Match markdown table blocks (lines with |)
        pattern = r'(?:^\|.+\|$\n?){2,}'
        html = re.sub(pattern, table_replacer, html, flags=re.MULTILINE)
        return html

    def _generate_plotly_scripts(self) -> str:
        """Generate Plotly chart scripts from stored section data."""
        scripts = []

        for i, section in enumerate(self._section_data):
            chart_configs = section.get('chart_configs', [])
            for j, config in enumerate(chart_configs):
                chart_type = config.get('type', '')
                chart_id = f"plotly-chart-{i}-{j}"

                plotly_script = self._create_plotly_chart(chart_type, section, chart_id)
                if plotly_script:
                    scripts.append(f"""
<div class="section" id="plotly-section-{i}-{j}">
    <div class="section-header">
        <h2>{section.get('title', 'Chart')} (Interactive)</h2>
        <span class="toggle">&#9660;</span>
    </div>
    <div class="section-body">
        <div id="{chart_id}" class="plotly-chart"></div>
        <script>{plotly_script}</script>
    </div>
</div>""")

        return '\n'.join(scripts)

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
                    groups = metrics[first_feat]
                    traces = []
                    categories = ['Pos Rate', 'TPR', 'FPR', 'PPV']

                    for group_name, m in groups.items():
                        vals = [m['positive_rate'], m['tpr'], m['fpr'], m['ppv']]
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

        except Exception as e:
            logger.debug(f"Plotly chart generation failed: {e}")

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
        except Exception:
            pass
        return image_path

    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to LaTeX RGB format."""
        h = hex_color.lstrip('#')
        return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"

    def _build_markdown(self) -> str:
        """Build the full markdown document with theme-aware LaTeX formatting."""

        title = self._metadata.get('title', 'ML Analysis Report')
        subtitle = self._metadata.get('subtitle', '')
        author = self._metadata.get('author', 'Jotty SwarmMLComprehensive')
        date = self._metadata.get('date', datetime.now().strftime('%B %d, %Y'))
        t = self.theme

        # Convert theme colors to RGB
        primary_rgb = self._hex_to_rgb(t['primary'])
        secondary_rgb = self._hex_to_rgb(t['secondary'])
        accent_rgb = self._hex_to_rgb(t['accent'])
        success_rgb = self._hex_to_rgb(t['success'])
        muted_rgb = self._hex_to_rgb(t['muted'])
        table_alt_rgb = self._hex_to_rgb(t['table_alt'])

        # Theme-specific LaTeX settings
        if self.theme_name == 'goldman':
            # Goldman: serif, uppercase headers, minimal lines, cool gray bg
            section_format = "\\\\titleformat{{\\\\section}}{{\\\\Large\\\\color{{Primary}}\\\\scshape}}{{\\\\thesection}}{{1em}}{{}}"
            subsection_format = "\\\\titleformat{{\\\\subsection}}{{\\\\large\\\\color{{Secondary}}}}{{\\\\thesubsection}}{{1em}}{{}}"
            font_pkg = "\\\\usepackage{{charter}}"
            extra_packages = """  - \\\\usepackage{{titlesec}}
  - {section_format}
  - {subsection_format}""".format(section_format=section_format, subsection_format=subsection_format)
        else:
            # Professional: sans-serif, bold headers
            font_pkg = ""
            extra_packages = ""

        header_brand = t['header_brand']
        footer_text = t['footer_text']

        title_page = f"""---
title: "{title}"
subtitle: "{subtitle}"
author: "{author}"
date: "{date}"
geometry: "margin=1in"
fontsize: 11pt
documentclass: article
colorlinks: true
linkcolor: Primary
urlcolor: Primary
toccolor: Primary
toc-depth: 3
numbersections: true
header-includes:
  - \\usepackage{{booktabs}}
  - \\usepackage{{longtable}}
  - \\usepackage{{array}}
  - \\usepackage{{multirow}}
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
  - \\usepackage{{colortbl}}
  - \\usepackage{{graphicx}}
  - \\usepackage{{xcolor}}
  - \\definecolor{{Primary}}{{RGB}}{{{primary_rgb}}}
  - \\definecolor{{Secondary}}{{RGB}}{{{secondary_rgb}}}
  - \\definecolor{{Accent}}{{RGB}}{{{accent_rgb}}}
  - \\definecolor{{Success}}{{RGB}}{{{success_rgb}}}
  - \\definecolor{{Muted}}{{RGB}}{{{muted_rgb}}}
  - \\definecolor{{TableAlt}}{{RGB}}{{{table_alt_rgb}}}
  - \\usepackage{{fancyhdr}}
  - \\pagestyle{{fancy}}
  - \\fancyhf{{}}
  - \\fancyhead[L]{{\\small\\textit{{{title[:35]}}}}}
  - \\fancyhead[R]{{\\small\\thepage}}
  - \\fancyfoot[L]{{\\small\\textcolor{{Muted}}{{{header_brand}}}}}
  - \\fancyfoot[R]{{\\small\\textcolor{{Muted}}{{{footer_text}}}}}
  - \\renewcommand{{\\headrulewidth}}{{0.4pt}}
  - \\renewcommand{{\\footrulewidth}}{{0.2pt}}
  - \\renewcommand{{\\arraystretch}}{{1.3}}
---

\\newpage

"""

        # Table of contents
        toc = """\\tableofcontents
\\newpage

"""

        # Combine all content
        body = "\n".join(self._content)

        return title_page + toc + body

    def _convert_with_pandoc(self, md_path: Path, pdf_path: Path) -> bool:
        """Convert markdown to PDF using pandoc."""

        if not shutil.which('pandoc'):
            logger.warning("Pandoc not found")
            return False

        # Try pdflatex first (most compatible), then xelatex
        engines = ['pdflatex', 'xelatex']

        for engine in engines:
            # Build command with proper resource path
            cmd = [
                'pandoc',
                str(md_path.name),  # Just filename, we'll cd to the directory
                '-o', str(pdf_path.name),
                f'--pdf-engine={engine}',
                '--toc-depth=3',
                '--highlight-style=tango',
                '-V', 'geometry:margin=1in',
            ]

            try:
                # Run from the output directory so relative paths work
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minutes timeout
                    cwd=str(self.output_dir)  # Run from output directory
                )

                if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                    logger.info(f"PDF generated with {engine}")
                    return True
                else:
                    # Log error for debugging
                    if result.stderr:
                        logger.debug(f"Pandoc {engine} stderr: {result.stderr[:500]}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Pandoc with {engine} timed out")
            except Exception as e:
                logger.warning(f"Pandoc with {engine} failed: {e}")

        return False

    def _fallback_pdf_generation(self, markdown: str, pdf_path: Path) -> Optional[str]:
        """Enhanced fallback PDF generation using reportlab with TOC, tables, images, and styling."""
        try:
            import re
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import (
                BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
                PageBreak, Table, TableStyle, Image as RLImage,
                NextPageTemplate,
            )
            from reportlab.lib.units import inch
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
            from reportlab.platypus.tableofcontents import TableOfContents

            t = self.theme
            primary_color = rl_colors.HexColor(t['primary'])
            accent_color = rl_colors.HexColor(t['accent'])
            text_color = rl_colors.HexColor(t['text'])
            muted_color = rl_colors.HexColor(t['muted'])
            table_header_color = rl_colors.HexColor(t['table_header'])
            table_alt_color = rl_colors.HexColor(t['table_alt'])

            # Custom styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                'ReportTitle', parent=styles['Title'],
                fontSize=24, textColor=primary_color,
                spaceAfter=20, alignment=TA_CENTER,
            ))
            styles.add(ParagraphStyle(
                'ReportH1', parent=styles['Heading1'],
                fontSize=18, textColor=primary_color,
                spaceBefore=24, spaceAfter=10,
            ))
            styles.add(ParagraphStyle(
                'ReportH2', parent=styles['Heading2'],
                fontSize=14, textColor=accent_color,
                spaceBefore=16, spaceAfter=8,
            ))
            styles.add(ParagraphStyle(
                'ReportH3', parent=styles['Heading3'],
                fontSize=12, textColor=text_color,
                spaceBefore=12, spaceAfter=6,
            ))
            styles.add(ParagraphStyle(
                'ReportBody', parent=styles['Normal'],
                fontSize=10, textColor=text_color,
                spaceBefore=3, spaceAfter=6,
                leading=14,
            ))
            styles.add(ParagraphStyle(
                'ReportBullet', parent=styles['Normal'],
                fontSize=10, textColor=text_color,
                leftIndent=20, bulletIndent=10,
                spaceBefore=2, spaceAfter=2,
            ))
            styles.add(ParagraphStyle(
                'FooterStyle', parent=styles['Normal'],
                fontSize=8, textColor=muted_color,
                alignment=TA_CENTER,
            ))

            # Track headings for TOC
            heading_entries = []

            def _on_page(canvas, doc_obj):
                """Add page number and brand to footer."""
                canvas.saveState()
                page_num = canvas.getPageNumber()
                footer_text = f"{t.get('footer_text', 'ML Report')}  |  Page {page_num}"
                canvas.setFont('Helvetica', 8)
                canvas.setFillColor(muted_color)
                canvas.drawCentredString(letter[0] / 2, 30, footer_text)
                canvas.restoreState()

            # Build document
            frame = Frame(72, 60, letter[0] - 144, letter[1] - 132, id='main')
            page_template = PageTemplate('main', frames=[frame], onPage=_on_page)
            doc = BaseDocTemplate(str(pdf_path), pagesize=letter, pageTemplates=[page_template])

            story = []

            # Title page
            story.append(Spacer(1, 2 * inch))
            story.append(Paragraph(
                self._metadata.get('title', 'ML Analysis Report'), styles['ReportTitle']))
            story.append(Spacer(1, 0.3 * inch))
            subtitle = self._metadata.get('subtitle', '')
            if subtitle:
                story.append(Paragraph(subtitle, ParagraphStyle(
                    'SubTitle', parent=styles['Normal'],
                    fontSize=13, textColor=accent_color, alignment=TA_CENTER)))
            story.append(Spacer(1, 0.5 * inch))
            date_str = self._metadata.get('date', datetime.now().strftime('%Y-%m-%d'))
            story.append(Paragraph(f"Generated: {date_str}", ParagraphStyle(
                'DateLine', parent=styles['Normal'],
                fontSize=10, textColor=muted_color, alignment=TA_CENTER)))
            story.append(PageBreak())

            # Table of Contents
            toc = TableOfContents()
            toc.levelStyles = [
                ParagraphStyle('TOC1', fontSize=12, leftIndent=20, spaceBefore=5,
                               textColor=primary_color),
                ParagraphStyle('TOC2', fontSize=10, leftIndent=40, spaceBefore=3,
                               textColor=accent_color),
            ]
            story.append(Paragraph("Table of Contents", styles['ReportH1']))
            story.append(toc)
            story.append(PageBreak())

            def _render_markdown_inline(text):
                """Convert basic markdown inline formatting to reportlab XML."""
                text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
                text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
                text = re.sub(r'`([^`]+?)`', r'<font face="Courier">\1</font>', text)
                text = re.sub(r'<', '&lt;', text)
                # Undo our escaping of our own tags
                text = text.replace('&lt;b>', '<b>').replace('&lt;/b>', '</b>')
                text = text.replace('&lt;i>', '<i>').replace('&lt;/i>', '</i>')
                text = text.replace('&lt;font', '<font').replace('&lt;/font>', '</font>')
                return text

            def _parse_md_table(lines):
                """Parse markdown pipe table into list of rows."""
                rows = []
                for line in lines:
                    line = line.strip()
                    if line.startswith('|') and not re.match(r'^\|[\s\-|]+\|$', line):
                        cells = [c.strip() for c in line.strip('|').split('|')]
                        rows.append(cells)
                return rows

            # Process markdown content
            lines = markdown.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                # Headings
                if stripped.startswith('# ') and not stripped.startswith('## '):
                    heading_text = stripped[2:].strip()
                    story.append(Paragraph(heading_text, styles['ReportH1']))
                    heading_entries.append((0, heading_text, len(story)))
                    i += 1
                    continue
                elif stripped.startswith('## '):
                    heading_text = stripped[3:].strip()
                    story.append(Paragraph(heading_text, styles['ReportH2']))
                    heading_entries.append((1, heading_text, len(story)))
                    i += 1
                    continue
                elif stripped.startswith('### '):
                    heading_text = stripped[4:].strip()
                    story.append(Paragraph(heading_text, styles['ReportH3']))
                    i += 1
                    continue

                # Horizontal rule / page break
                elif stripped == '---':
                    story.append(Spacer(1, 12))
                    i += 1
                    continue

                # Images
                elif re.match(r'^!\[.*?\]\((.+?)\)$', stripped):
                    img_match = re.match(r'^!\[.*?\]\((.+?)\)$', stripped)
                    img_path = img_match.group(1)
                    full_img_path = self.output_dir / img_path
                    if full_img_path.exists():
                        try:
                            img = RLImage(str(full_img_path), width=6 * inch, height=4 * inch)
                            img.hAlign = 'CENTER'
                            story.append(img)
                            story.append(Spacer(1, 8))
                        except Exception:
                            pass
                    i += 1
                    continue

                # Tables (collect consecutive pipe lines)
                elif stripped.startswith('|'):
                    table_lines = []
                    while i < len(lines) and lines[i].strip().startswith('|'):
                        table_lines.append(lines[i])
                        i += 1
                    rows = _parse_md_table(table_lines)
                    if rows:
                        try:
                            # Convert to Paragraph cells for wrapping
                            table_data = []
                            for r_idx, row in enumerate(rows):
                                table_data.append([
                                    Paragraph(_render_markdown_inline(cell),
                                             styles['ReportBody']) for cell in row
                                ])

                            col_count = max(len(r) for r in table_data)
                            avail_width = letter[0] - 144
                            col_widths = [avail_width / col_count] * col_count

                            tbl = Table(table_data, colWidths=col_widths)
                            tbl_style = [
                                ('BACKGROUND', (0, 0), (-1, 0), table_header_color),
                                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
                                ('FONTSIZE', (0, 0), (-1, -1), 9),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                                ('TOPPADDING', (0, 0), (-1, -1), 4),
                                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#d0d0d0')),
                                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ]
                            # Alternating row colors
                            for row_idx in range(1, len(table_data)):
                                if row_idx % 2 == 0:
                                    tbl_style.append(
                                        ('BACKGROUND', (0, row_idx), (-1, row_idx), table_alt_color))

                            tbl.setStyle(TableStyle(tbl_style))
                            story.append(tbl)
                            story.append(Spacer(1, 8))
                        except Exception:
                            pass
                    continue

                # Bullet lists
                elif stripped.startswith('- ') or stripped.startswith('* '):
                    bullet_text = _render_markdown_inline(stripped[2:])
                    story.append(Paragraph(
                        f"\u2022 {bullet_text}", styles['ReportBullet']))
                    i += 1
                    continue

                # Numbered lists
                elif re.match(r'^\d+\.\s', stripped):
                    num_match = re.match(r'^(\d+)\.\s(.+)', stripped)
                    if num_match:
                        num = num_match.group(1)
                        text = _render_markdown_inline(num_match.group(2))
                        story.append(Paragraph(
                            f"{num}. {text}", styles['ReportBullet']))
                    i += 1
                    continue

                # Regular paragraph
                elif stripped:
                    rendered = _render_markdown_inline(stripped)
                    try:
                        story.append(Paragraph(rendered, styles['ReportBody']))
                    except Exception:
                        pass
                    i += 1
                    continue

                else:
                    i += 1

            # Build with TOC notification
            class _TOCBuilder:
                """Handles TOC heading registration during multiBuild."""
                def __init__(self, toc_obj):
                    self._toc = toc_obj

                def afterFlowable(self, flowable):
                    if isinstance(flowable, Paragraph):
                        style_name = flowable.style.name
                        if style_name == 'ReportH1':
                            self._toc.addEntry(0, flowable.getPlainText(), 0)
                        elif style_name == 'ReportH2':
                            self._toc.addEntry(1, flowable.getPlainText(), 0)

            try:
                # Try multiBuild for TOC
                builder = _TOCBuilder(toc)
                original_afterFlowable = doc.afterFlowable if hasattr(doc, 'afterFlowable') else None
                doc.afterFlowable = builder.afterFlowable
                doc.multiBuild(story)
            except Exception:
                # Simple build fallback
                try:
                    doc2 = BaseDocTemplate(str(pdf_path), pagesize=letter,
                                           pageTemplates=[page_template])
                    doc2.build(story)
                except Exception:
                    pass

            return str(pdf_path)

        except Exception as e:
            logger.error(f"Enhanced fallback PDF generation failed: {e}")
            # Minimal fallback — just dump text
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                import re

                doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
                styles = getSampleStyleSheet()
                story = [Paragraph(self._metadata.get('title', 'ML Report'), styles['Title']),
                         Spacer(1, 12)]

                text = re.sub(r'[#*_`\[\]!]', '', markdown)
                text = re.sub(r'\n{3,}', '\n\n', text)
                for para in text.split('\n\n'):
                    if para.strip():
                        try:
                            story.append(Paragraph(para.strip(), styles['Normal']))
                            story.append(Spacer(1, 6))
                        except Exception:
                            pass
                doc.build(story)
                return str(pdf_path)
            except Exception as e2:
                logger.error(f"Minimal fallback PDF also failed: {e2}")
                return None

    def _create_importance_chart(self, sorted_importance: List[Tuple[str, float]]) -> str:
        """Create feature importance bar chart."""
        try:
            import matplotlib.pyplot as plt

            features = [x[0][:25] for x in sorted_importance]
            values = [x[1] for x in sorted_importance]

            fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.35)))

            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]
            bars = ax.barh(range(len(features)), values[::-1], color=colors)

            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features[::-1], fontsize=9)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title('Feature Importance', fontsize=14, fontweight='bold', color=self.theme['primary'])

            # Add value labels
            for bar, val in zip(bars, values[::-1]):
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=8)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            path = self.figures_dir / 'feature_importance.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()

            return 'figures/feature_importance.png'

        except Exception as e:
            logger.warning(f"Failed to create importance chart: {e}")
            return ""

    def _create_benchmark_chart(self, sorted_models: List[Tuple[str, Dict]]) -> str:
        """Create model benchmarking chart."""
        try:
            import matplotlib.pyplot as plt

            models = [x[0] for x in sorted_models]
            scores = [x[1].get('test_score', x[1].get('cv_score', 0)) for x in sorted_models]

            fig, ax = plt.subplots(figsize=(10, 6))

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

            plt.tight_layout()

            path = self.figures_dir / 'model_benchmark.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()

            return 'figures/model_benchmark.png'

        except Exception as e:
            logger.warning(f"Failed to create benchmark chart: {e}")
            return ""

    def _create_confusion_matrix_chart(self, cm, labels: List[str] = None) -> str:
        """Create confusion matrix heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 6))

            sns.heatmap(cm, annot=True, fmt='d', cmap=self.theme['chart_cmap'],
                       xticklabels=labels or ['0', '1'],
                       yticklabels=labels or ['0', '1'],
                       ax=ax, annot_kws={'size': 14})

            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color=self.theme['primary'])

            plt.tight_layout()

            path = self.figures_dir / 'confusion_matrix.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()

            return 'figures/confusion_matrix.png'

        except Exception as e:
            logger.warning(f"Failed to create confusion matrix: {e}")
            return ""

    def _create_roc_chart(self, fpr, tpr, roc_auc: float) -> str:
        """Create ROC curve chart."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))

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

            plt.tight_layout()

            path = self.figures_dir / 'roc_curve.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()

            return 'figures/roc_curve.png'

        except Exception as e:
            logger.warning(f"Failed to create ROC chart: {e}")
            return ""

    def _create_pr_chart(self, precision, recall, avg_precision: float) -> str:
        """Create precision-recall curve chart."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))

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

            plt.tight_layout()

            path = self.figures_dir / 'pr_curve.png'
            plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.theme['chart_bg'])
            plt.close()

            return 'figures/pr_curve.png'

        except Exception as e:
            logger.warning(f"Failed to create PR chart: {e}")
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
            logger.warning(f"Failed to create SHAP chart: {e}")
            return ""
