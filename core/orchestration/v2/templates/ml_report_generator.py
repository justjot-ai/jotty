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
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Protocol, Tuple, Union
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mixin imports for decomposed functionality
from ._interpretability_mixin import InterpretabilityMixin
from ._fairness_mixin import FairnessMixin
from ._drift_mixin import DriftMixin
from ._error_analysis_mixin import ErrorAnalysisMixin
from ._deployment_mixin import DeploymentMixin
from ._visualization_mixin import VisualizationMixin


# =============================================================================
# MIXIN PROTOCOL — documents the interface mixins expect from the host class
# =============================================================================

class ReportContext(Protocol):
    """Interface contract that mixins expect from ProfessionalMLReport.

    This Protocol exists solely for documentation and IDE support.
    It is never instantiated at runtime. Mixins access these attributes
    and methods via ``self`` when mixed into ProfessionalMLReport.
    """
    theme: Dict[str, Any]
    theme_name: str
    figures_dir: Path
    output_dir: Path
    config: Dict[str, Any]
    _content: List[str]
    _section_data: List[Dict]
    _failed_sections: List[Dict]
    _failed_charts: List[Dict]
    _warnings: List[Dict]
    _llm_narrative_enabled: bool

    def _record_section_failure(self, section_name: str, error: Exception) -> None: ...
    def _record_chart_failure(self, chart_name: str, error: Exception) -> None: ...
    def _record_internal_warning(self, component: str, message: str, error: Exception = None) -> None: ...
    def _store_section_data(self, section_type: str, title: str, data: Dict, chart_configs: List[Dict] = None) -> None: ...
    def _maybe_add_narrative(self, section_name: str, data_context: str, section_type: str = 'general') -> str: ...
    def _validate_inputs(self, **kwargs) -> None: ...


# =============================================================================
# PREDICTION DATA CONTAINER
# =============================================================================

@dataclass
class PredictionResult:
    """Consolidated container for y_true / y_pred / y_prob triplets.

    Handles numpy conversion and length validation in one place, eliminating
    repeated ``np.asarray`` calls scattered across 13+ methods.

    Public ``add_*`` method signatures remain unchanged — each method creates
    a ``PredictionResult`` internally at the start of its implementation.
    """
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray] = None

    def __post_init__(self):
        self.y_true = np.asarray(self.y_true)
        self.y_pred = np.asarray(self.y_pred)
        if self.y_prob is not None:
            self.y_prob = np.asarray(self.y_prob)
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"y_true length ({len(self.y_true)}) != y_pred length ({len(self.y_pred)})"
            )
        if self.y_prob is not None and len(self.y_prob) != len(self.y_true):
            raise ValueError(
                f"y_prob length ({len(self.y_prob)}) != y_true length ({len(self.y_true)})"
            )

    @classmethod
    def from_predictions(cls, y_true, y_pred, y_prob=None):
        """Factory method accepting any array-like inputs."""
        return cls(y_true, y_pred, y_prob)

    @property
    def n_samples(self) -> int:
        return len(self.y_true)

    @property
    def has_probabilities(self) -> bool:
        return self.y_prob is not None

    @property
    def errors(self) -> np.ndarray:
        return self.y_true != self.y_pred

    @property
    def n_errors(self) -> int:
        return int(self.errors.sum())


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

# DEPRECATED: Module-level COLORS is kept for backward compatibility only.
# All internal code uses self.theme instead. Do not rely on this global;
# it is not updated when a different theme is selected at instance creation.
COLORS = THEMES['professional']


class ProfessionalMLReport(VisualizationMixin, InterpretabilityMixin, DriftMixin,
                           FairnessMixin, ErrorAnalysisMixin, DeploymentMixin):
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

    _REPORT_VERSION = '4.0.0'

    _SECTION_PROMPTS = {
        'executive_summary': (
            "You are a senior data scientist writing an executive summary. "
            "First analyze the key performance numbers, then provide 2-3 sentences of strategic insight. "
            "Focus on whether the model is production-ready and what the main risk factors are."
        ),
        'error_analysis': (
            "You are a senior data scientist analyzing model errors. "
            "First examine the error patterns and rates, then provide 2-3 sentences on "
            "the most actionable improvements. Focus on systematic failure modes."
        ),
        'fairness_audit': (
            "You are a senior data scientist evaluating model fairness. "
            "First analyze the fairness metrics across groups, then provide 2-3 sentences "
            "on whether the model exhibits bias and what specific mitigation steps to take."
        ),
        'drift_analysis': (
            "You are a senior data scientist monitoring data drift. "
            "First assess which features show significant distribution shift, then provide "
            "2-3 sentences on the operational implications and recommended actions."
        ),
        'interpretability': (
            "You are a senior data scientist explaining model behavior. "
            "First summarize the key feature contributions, then provide 2-3 sentences "
            "on whether the model's reasoning aligns with domain knowledge."
        ),
        'deployment_readiness': (
            "You are a senior ML engineer assessing deployment readiness. "
            "First review the latency, size, and checklist results, then provide "
            "2-3 sentences on production readiness and any blockers to address."
        ),
        'feature_importance': (
            "You are a senior data scientist analyzing feature importance. "
            "First identify the dominant features, then provide 2-3 sentences on "
            "potential feature engineering opportunities or redundant features to remove."
        ),
        'general': (
            "You are a senior data scientist. Given the following ML analysis data, "
            "write a 2-3 sentence insight in plain English. Focus on actionable takeaways."
        ),
    }

    _DEFAULT_CONFIG = {
        'psi_bins': 10,
        'psi_binning_method': 'quantile',
        'js_divergence_bins': 50,
        'mmd_permutations': 500,
        'classifier_drift_auc_threshold': 0.6,
        'ensemble_drift_weights': (0.3, 0.35, 0.35),
        'min_group_size_fairness': 5,
    }

    def __init__(self, output_dir: str = "professional_reports", theme: str = "professional",
                 llm_narrative: bool = False, html_enabled: bool = False,
                 config: Dict = None):
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
        self._failed_sections = []
        self._failed_charts = []
        self._warnings = []
        self._llm_narrative_enabled = llm_narrative
        self._html_enabled = html_enabled
        self._llm = None  # Lazy-loaded UnifiedLLM

        # Configurable parameters (merge user config over defaults)
        self.config = dict(self._DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        # Set theme
        self.theme_name = theme
        self.theme = THEMES.get(theme, THEMES['professional'])

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
        """Configure matplotlib for theme-matched professional visualizations.

        Warning — Thread Safety:
            This method mutates the process-global ``mpl.rcParams`` dict.
            Concurrent ProfessionalMLReport instances with different themes
            will overwrite each other's settings. If thread-safety is required,
            use ``matplotlib.rc_context()`` around individual chart renders or
            generate reports sequentially.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from matplotlib import cycler

            # Save original rcParams for later restoration
            self._saved_rcparams = dict(mpl.rcParams)

            t = self.theme

            if self.theme_name == 'goldman':
                # Goldman: clean, minimal, serif fonts, subtle grid
                try:
                    plt.style.use('seaborn-v0_8-whitegrid')
                except Exception as e:
                    if hasattr(self, '_warnings'):
                        self._record_internal_warning('PlotStyleGoldman', 'seaborn style not available', e)
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
                except Exception as e:
                    if hasattr(self, '_warnings'):
                        self._record_internal_warning('PlotStyleProfessional', 'seaborn style not available', e)
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

    def restore_plot_style(self):
        """Restore matplotlib rcParams to their state before report initialization."""
        try:
            import matplotlib as mpl
            if hasattr(self, '_saved_rcparams'):
                mpl.rcParams.update(self._saved_rcparams)
        except Exception as e:
            if hasattr(self, '_warnings'):
                self._record_internal_warning('RestorePlotStyle', 'failed to restore rcParams', e)

    def _record_section_failure(self, section_name: str, error: Exception):
        """Record a section generation failure for health tracking.

        Logs a warning, appends to _failed_sections, and adds a visible
        placeholder to _content so failures are not silently swallowed.
        """
        self._failed_sections.append({
            'section': section_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
        })
        logger.warning(f"{section_name} failed: {error}")
        self._content.append(
            f"\n> **{section_name}:** Section generation failed — {type(error).__name__}: {error}\n\n---\n"
        )

    def _validate_inputs(self, **kwargs):
        """Validate common ML inputs before processing.

        Checks for None/empty data, length mismatches, probability ranges,
        and feature name types. Raises ValueError with clear messages.

        Keyword Args:
            X: Feature matrix (DataFrame or array)
            y_true: True labels array
            y_pred: Predicted labels array
            y_prob: Predicted probabilities array
            feature_names: List of feature name strings
            require_X: Whether X is mandatory (default False)
        """
        X = kwargs.get('X')
        y_true = kwargs.get('y_true')
        y_pred = kwargs.get('y_pred')
        y_prob = kwargs.get('y_prob')
        feature_names = kwargs.get('feature_names')
        require_X = kwargs.get('require_X', False)

        # Validate X
        if require_X:
            if X is None:
                raise ValueError("X (feature matrix) is required but was None")
            x_len = len(X) if hasattr(X, '__len__') else 0
            if x_len == 0:
                raise ValueError("X (feature matrix) is empty")

        # Validate y_true / y_pred length match
        if y_true is not None and y_pred is not None:
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            if len(y_true_arr) != len(y_pred_arr):
                raise ValueError(
                    f"y_true length ({len(y_true_arr)}) != y_pred length ({len(y_pred_arr)})"
                )

        # Validate X rows match y_true length
        if X is not None and y_true is not None:
            x_len = len(X) if hasattr(X, '__len__') else 0
            y_len = len(np.asarray(y_true))
            if x_len > 0 and x_len != y_len:
                raise ValueError(
                    f"X rows ({x_len}) != y_true length ({y_len})"
                )

        # Validate y_prob in [0, 1]
        if y_prob is not None:
            y_prob_arr = np.asarray(y_prob)
            if y_prob_arr.size > 0:
                if np.any(y_prob_arr < 0) or np.any(y_prob_arr > 1):
                    raise ValueError(
                        f"y_prob values must be in [0, 1], got range "
                        f"[{y_prob_arr.min():.4f}, {y_prob_arr.max():.4f}]"
                    )

        # Validate feature_names are strings
        if feature_names is not None:
            if not all(isinstance(f, str) for f in feature_names):
                bad_types = {type(f).__name__ for f in feature_names if not isinstance(f, str)}
                raise ValueError(
                    f"feature_names must all be strings, found types: {bad_types}"
                )

    @staticmethod
    def _make_predictions(y_true, y_pred, y_prob=None) -> 'PredictionResult':
        """Create a validated PredictionResult from raw arrays.

        Convenience wrapper used at the start of internal ``_impl`` methods
        to consolidate numpy conversion and length validation.
        """
        return PredictionResult.from_predictions(y_true, y_pred, y_prob)

    def _record_chart_failure(self, chart_name: str, error: Exception):
        """Record a chart generation failure for health tracking.

        Logs a warning and appends to _failed_charts so chart failures
        are visible in the report health summary.
        """
        self._failed_charts.append({
            'chart': chart_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
        })
        logger.warning(f"Chart '{chart_name}' failed: {error}")

    def _record_internal_warning(self, component: str, message: str, error: Exception = None):
        """Record a non-fatal internal warning for health tracking.

        Used to replace silent ``except: pass`` blocks so that suppressed
        errors are still visible in the report health summary without
        changing return values or control flow.
        """
        if hasattr(self, '_warnings'):
            self._warnings.append({
                'component': component,
                'message': message,
                'error_type': type(error).__name__ if error else None,
                'error_message': str(error) if error else None,
            })
        logger.debug(
            f"Internal warning [{component}]: {message}"
            + (f" ({error})" if error else "")
        )

    def _capture_environment(self) -> Dict:
        """Capture environment info for report reproducibility.

        Collects Python version, platform details, library versions,
        ISO timestamp, and report generator version.

        Returns:
            Dict with python_version, platform, machine, libraries,
            timestamp, and report_version.
        """
        import sys
        import platform
        import importlib

        env = {
            'python_version': sys.version,
            'platform': platform.system(),
            'machine': platform.machine(),
            'timestamp': datetime.now().isoformat(),
            'report_version': self._REPORT_VERSION,
        }

        libraries = {}
        for lib_name in ['sklearn', 'numpy', 'pandas', 'matplotlib', 'shap']:
            try:
                mod = importlib.import_module(lib_name)
                libraries[lib_name] = getattr(mod, '__version__', 'unknown')
            except Exception as e:
                self._record_internal_warning(
                    'EnvironmentCapture',
                    f'Could not get version for {lib_name}',
                    e
                )
                libraries[lib_name] = 'not installed'
        env['libraries'] = libraries

        return env

    def get_report_health(self) -> Dict:
        """Return health summary of the report generation.

        Returns:
            Dict with total_sections, succeeded, failed, failed_sections,
            failed_charts, total_charts_failed, warnings, total_warnings,
            healthy
        """
        total = len(self._section_data) + len(self._failed_sections)
        warnings_list = list(self._warnings) if hasattr(self, '_warnings') else []
        return {
            'total_sections': total,
            'succeeded': len(self._section_data),
            'failed': len(self._failed_sections),
            'failed_sections': list(self._failed_sections),
            'failed_charts': list(self._failed_charts),
            'total_charts_failed': len(self._failed_charts),
            'warnings': warnings_list,
            'total_warnings': len(warnings_list),
            'healthy': len(self._failed_sections) == 0,
        }

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

    def _validate_narrative(self, text: str, section_type: str = 'general') -> Dict:
        """Validate LLM-generated narrative for quality, accuracy, and format.

        Checks:
        1. Length: 15-500 words (penalize outside range)
        2. Numbers present: must contain at least one numeric value
        3. Topic relevance: section-specific keywords
        4. Impossible values: probability > 1.0, negative counts
        5. Fabricated metrics: unknown metric names in "the X score" patterns
        6. Format enforcement: strip markdown headers, collapse triple newlines

        Args:
            text: Raw narrative text
            section_type: Section type for topic relevance check

        Returns:
            Dict with valid (bool), score (float 0-1), issues (list), cleaned_text (str)
        """
        import re

        issues = []
        score = 1.0
        cleaned = text.strip()

        # 1. Length check (15-500 words)
        word_count = len(cleaned.split())
        if word_count < 15:
            score -= 0.4
            issues.append(f'Too short ({word_count} words, min 15)')
        elif word_count > 500:
            score -= 0.2
            issues.append(f'Too long ({word_count} words, max 500)')
            cleaned = ' '.join(cleaned.split()[:500])

        # 2. Numbers present
        has_numbers = bool(re.search(r'\d+\.?\d*', cleaned))
        if not has_numbers:
            score -= 0.2
            issues.append('No numeric values found')

        # 3. Topic relevance (section-specific keywords)
        topic_keywords = {
            'fairness_audit': ['bias', 'group', 'disparit', 'fair', 'parity', 'equit'],
            'drift_analysis': ['drift', 'shift', 'distribut', 'monitor', 'change'],
            'error_analysis': ['error', 'misclass', 'wrong', 'fail', 'incorrect'],
            'interpretability': ['feature', 'explain', 'contribut', 'import', 'shap'],
            'deployment_readiness': ['latency', 'deploy', 'production', 'throughput', 'size'],
            'executive_summary': ['model', 'performance', 'accuracy', 'result'],
            'feature_importance': ['feature', 'import', 'contribut', 'impact'],
        }
        keywords = topic_keywords.get(section_type, [])
        if keywords:
            text_lower = cleaned.lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches == 0:
                score -= 0.15
                issues.append(f'No topic-relevant keywords for {section_type}')

        # 4. Impossible values
        prob_matches = re.findall(r'(?:probability|prob|p-value|confidence)\s*(?:of|=|:)?\s*(\d+\.?\d*)', cleaned.lower())
        for m in prob_matches:
            val = float(m)
            if val > 1.0:
                score -= 0.3
                issues.append(f'Impossible probability value: {val}')

        count_matches = re.findall(r'(\-\d+)\s*(?:samples|instances|observations|errors|count)', cleaned.lower())
        for m in count_matches:
            score -= 0.3
            issues.append(f'Negative count: {m}')

        # 5. Fabricated metrics
        known_metrics = {
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'roc', 'rmse', 'mae',
            'mse', 'r2', 'mape', 'psi', 'ks', 'silhouette', 'mmd', 'mahalanobis',
            'gini', 'log loss', 'brier', 'cohen', 'kappa', 'specificity',
            'sensitivity', 'tpr', 'fpr', 'ppv', 'npv', 'fnr', 'fdr',
        }
        fabricated_pattern = re.findall(r'the\s+(\w+(?:\s+\w+)?)\s+score', cleaned.lower())
        for metric in fabricated_pattern:
            metric_clean = metric.strip().lower()
            if not any(km in metric_clean for km in known_metrics):
                score -= 0.15
                issues.append(f'Potentially fabricated metric: "{metric}"')

        # 6. Format enforcement
        cleaned = re.sub(r'^#+\s+.*$', '', cleaned, flags=re.MULTILINE).strip()
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        score = max(0.0, min(1.0, score))

        return {
            'valid': score >= 0.5,
            'score': score,
            'issues': issues,
            'cleaned_text': cleaned,
        }

    def _maybe_add_narrative(self, section_name: str, data_context: str,
                              section_type: str = 'general') -> str:
        """Generate LLM narrative insight if enabled using section-specific prompts.

        Includes structured validation and fact-checking of generated narratives.

        Args:
            section_name: Display name for the section
            data_context: Data to pass to the LLM for analysis
            section_type: Key into _SECTION_PROMPTS for section-specific prompt

        Returns:
            Markdown blockquote string or empty string
        """
        if not self._llm_narrative_enabled:
            return ""

        if not self._init_llm():
            return ""

        # Use section-specific prompt
        base_prompt = self._SECTION_PROMPTS.get(section_type, self._SECTION_PROMPTS['general'])

        # Chain-of-thought prompt with extended context
        prompt = (
            f"{base_prompt}\n\n"
            f"First analyze the numbers, then provide your insight.\n\n"
            f"Section: {section_name}.\nData: {data_context[:4000]}"
        )

        # Refusal patterns to validate against
        refusal_patterns = ['i cannot', 'i can\'t', 'as an ai', 'i\'m unable', 'i don\'t have']

        # Try with full prompt, retry with simpler prompt on validation failure
        for attempt in range(2):
            try:
                if attempt == 1:
                    # Simpler fallback prompt
                    prompt = (
                        f"Summarize the key finding from this ML analysis in 2 sentences. "
                        f"Section: {section_name}. Data: {data_context[:2000]}"
                    )

                response = self._llm.generate(
                    prompt=prompt,
                    timeout=30,
                    max_tokens=400,
                    fallback=True
                )

                if response.success and response.text:
                    text = response.text.strip()
                    # Basic refusal check
                    if len(text) < 20:
                        continue
                    if any(pat in text.lower() for pat in refusal_patterns):
                        continue

                    # Structured validation
                    validation = self._validate_narrative(text, section_type)

                    if validation['valid']:
                        return f"\n> **Insight:** {validation['cleaned_text']}\n\n"

                    # On first attempt failure, retry with simpler prompt
                    if attempt == 0:
                        continue

                    # On second attempt, accept if score >= 0.3
                    if validation['score'] >= 0.3:
                        return f"\n> **Insight:** {validation['cleaned_text']}\n\n"

            except Exception as e:
                logger.debug(f"LLM narrative generation attempt {attempt + 1} failed: {e}")

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
        self._metadata['environment'] = self._capture_environment()

    def _compute_risk_score(self, metrics: Dict[str, float]) -> Dict:
        """Compute overall risk score with per-metric traffic-light assessment.

        Thresholds:
            AUC:      GREEN >= 0.95, AMBER >= 0.80, RED < 0.80
            Accuracy: GREEN >= 0.95, AMBER >= 0.85, RED < 0.85
            F1:       GREEN >= 0.90, AMBER >= 0.75, RED < 0.75

        Returns:
            Dict with 'overall_risk' (float 0-1), 'metric_assessments' (list), 'traffic_light' (str)
        """
        thresholds = {
            'auc': {'green': 0.95, 'amber': 0.80, 'red': 0.70},
            'roc_auc': {'green': 0.95, 'amber': 0.80, 'red': 0.70},
            'accuracy': {'green': 0.95, 'amber': 0.85, 'red': 0.70},
            'f1': {'green': 0.90, 'amber': 0.75, 'red': 0.60},
            'f1_score': {'green': 0.90, 'amber': 0.75, 'red': 0.60},
            'precision': {'green': 0.90, 'amber': 0.75, 'red': 0.60},
            'recall': {'green': 0.90, 'amber': 0.75, 'red': 0.60},
        }

        assessments = []
        risk_scores = []

        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            name_lower = metric_name.lower().replace(' ', '_')
            thresh = thresholds.get(name_lower)

            if thresh:
                if value >= thresh['green']:
                    light = 'GREEN'
                    score = 0.0
                elif value >= thresh['amber']:
                    light = 'AMBER'
                    score = 0.5
                else:
                    light = 'RED'
                    score = 1.0
            else:
                # Unknown metric: assess based on generic 0-1 scale
                if value >= 0.9:
                    light = 'GREEN'
                    score = 0.0
                elif value >= 0.7:
                    light = 'AMBER'
                    score = 0.5
                else:
                    light = 'RED'
                    score = 1.0

            assessments.append({
                'metric': metric_name,
                'value': value,
                'traffic_light': light,
            })
            risk_scores.append(score)

        overall_risk = float(np.mean(risk_scores)) if risk_scores else 0.5

        if overall_risk <= 0.2:
            traffic_light = 'GREEN'
        elif overall_risk <= 0.6:
            traffic_light = 'AMBER'
        else:
            traffic_light = 'RED'

        return {
            'overall_risk': overall_risk,
            'metric_assessments': assessments,
            'traffic_light': traffic_light,
        }

    def add_executive_summary(self, metrics: Dict[str, float], best_model: str,
                             n_features: int, context: str = ""):
        """Add executive summary section with risk scoring and traffic-light indicators."""
        try:
            self._add_executive_summary_impl(metrics, best_model, n_features, context)
        except Exception as e:
            self._record_section_failure('Executive Summary', e)

    def _add_executive_summary_impl(self, metrics, best_model, n_features, context):
        # Compute risk score
        risk_data = self._compute_risk_score(metrics)

        # Traffic light indicator mapping
        light_icons = {'GREEN': 'Good', 'AMBER': 'Needs Improvement', 'RED': 'Critical'}

        # Create metrics table with traffic lights
        metrics_md = "| Metric | Value | Assessment |\n|--------|-------|------------|\n"
        assessment_map = {a['metric']: a['traffic_light'] for a in risk_data['metric_assessments']}
        for name, value in metrics.items():
            light = assessment_map.get(name, '')
            if isinstance(value, float):
                if value < 1:
                    metrics_md += f"| {name.replace('_', ' ').title()} | {value:.4f} | {light} |\n"
                else:
                    metrics_md += f"| {name.replace('_', ' ').title()} | {value:.2f} | {light} |\n"
            else:
                metrics_md += f"| {name.replace('_', ' ').title()} | {value} | {light} |\n"

        # Generate key findings
        key_findings = []
        for assessment in risk_data['metric_assessments']:
            if assessment['traffic_light'] == 'RED':
                key_findings.append(f"- **{assessment['metric']}** ({assessment['value']:.4f}) is below acceptable threshold")
            elif assessment['traffic_light'] == 'GREEN':
                key_findings.append(f"- **{assessment['metric']}** ({assessment['value']:.4f}) meets production standards")

        if not key_findings:
            key_findings = [f"- Model {best_model} achieves moderate performance across all metrics"]

        # Limit to 5 findings
        findings_md = '\n'.join(key_findings[:5])

        # Overall assessment paragraph
        overall_light = risk_data['traffic_light']
        if overall_light == 'GREEN':
            assessment_text = f"The model demonstrates strong performance across all evaluated metrics. {best_model} appears ready for production deployment with standard monitoring."
        elif overall_light == 'AMBER':
            assessment_text = f"The model shows moderate performance with some areas needing improvement. Consider additional tuning or feature engineering before production deployment."
        else:
            assessment_text = f"The model has significant performance gaps that should be addressed before deployment. Review the detailed analysis sections for specific improvement recommendations."

        summary = f"""
# Executive Summary

{context if context else "This report presents the results of an automated machine learning analysis."}

## Overall Assessment

**Risk Level:** {overall_light} ({light_icons.get(overall_light, '')})

{assessment_text}

## Key Results

**Best Model:** {best_model}

**Performance Metrics:**

{metrics_md}

**Dataset:** {n_features} features analyzed

## Key Findings

{findings_md}

{self._maybe_add_narrative('Executive Summary', f'Best model: {best_model}, Metrics: {metrics}, Features: {n_features}, Risk: {overall_light}', section_type='executive_summary')}

---
"""
        self._content.append(summary)
        self._store_section_data('executive_summary', 'Executive Summary', {
            'metrics': metrics,
            'best_model': best_model,
            'risk_score': risk_data,
        })

    def add_data_profile(self, shape: Tuple[int, int], dtypes: Dict[str, int],
                        missing: Dict[str, int], recommendations: List[str]):
        """Add data profiling section."""
        try:
            self._add_data_profile_impl(shape, dtypes, missing, recommendations)
        except Exception as e:
            self._record_section_failure('Data Profile', e)

    def _add_data_profile_impl(self, shape, dtypes, missing, recommendations):
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
            self._record_section_failure('Pipeline Visualization', e)


    def add_feature_importance(self, importance: Dict[str, float], top_n: int = 20):
        """Add feature importance section with chart."""
        try:
            self._add_feature_importance_impl(importance, top_n)
        except Exception as e:
            self._record_section_failure('Feature Importance', e)

    def _add_feature_importance_impl(self, importance, top_n):
        self._validate_inputs(feature_names=list(importance.keys()))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Create table
        table_md = "| Rank | Feature | Importance |\n|------|---------|------------|\n"
        for i, (feat, imp) in enumerate(sorted_imp, 1):
            table_md += f"| {i} | {feat[:40]} | {imp:.4f} |\n"

        # Create bar chart (isolated so chart failure doesn't lose table)
        fig_path = ""
        try:
            fig_path = self._create_importance_chart(sorted_imp)
        except Exception as e:
            self._record_chart_failure('importance_chart', e)

        content = f"""
# Feature Importance Analysis

Feature importance measures how much each feature contributes to the model's predictions.
Higher values indicate more influential features.

## Top {top_n} Features

{table_md}

## Feature Importance Visualization

![Feature Importance]({fig_path})

{self._maybe_add_narrative('Feature Importance', f'Top features: {sorted_imp[:5]}', section_type='feature_importance')}

---
"""
        self._content.append(content)
        self._store_section_data('feature_importance', 'Feature Importance',
                                {'importance': dict(sorted_imp[:10])},
                                [{'type': 'importance_bar'}])

    def add_model_benchmarking(self, model_scores: Dict[str, Dict[str, float]]):
        """Add model benchmarking comparison."""
        try:
            self._add_model_benchmarking_impl(model_scores)
        except Exception as e:
            self._record_section_failure('Model Benchmarking', e)

    def _add_model_benchmarking_impl(self, model_scores):
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

        # Create benchmark chart (isolated)
        fig_path = ""
        try:
            fig_path = self._create_benchmark_chart(sorted_models)
        except Exception as e:
            self._record_chart_failure('benchmark_chart', e)

        content = f"""
# Model Benchmarking

Multiple machine learning algorithms were evaluated using 5-fold cross-validation.
The table below shows the performance of each model.

## Model Comparison

{table_md}

## Performance Visualization

![Model Benchmarking]({fig_path})

{self._maybe_add_narrative('Model Benchmarking', f'Model comparison results: {list(model_scores.keys())}', section_type='general')}

---
"""
        self._content.append(content)
        self._store_section_data('model_benchmarking', 'Model Benchmarking',
                                {'model_scores': model_scores},
                                [{'type': 'bar'}])

    # =========================================================================
    # MODEL COMPARISON (Phase 3 — Round 4)
    # =========================================================================

    def add_model_comparison(self, models: Dict, X_test, y_true,
                             class_labels: List[str] = None):
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
            self._record_section_failure('Model Comparison', e)

    def _add_model_comparison_impl(self, models, X_test, y_true, class_labels=None):
        """Implementation for side-by-side model comparison."""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, roc_curve)

        y_true_arr = np.asarray(y_true)
        X_arr = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)

        comparison = {}
        roc_curves = {}
        metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

        for model_name, model in models.items():
            try:
                y_pred = model.predict(X_arr)
                y_prob = None
                auc_val = None

                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X_arr)
                        y_prob = proba[:, 1] if proba.shape[1] == 2 else proba
                    except Exception:
                        pass

                acc = accuracy_score(y_true_arr, y_pred)
                prec = precision_score(y_true_arr, y_pred, average='binary', zero_division=0)
                rec = recall_score(y_true_arr, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_true_arr, y_pred, average='binary', zero_division=0)

                if y_prob is not None and len(np.unique(y_true_arr)) == 2:
                    try:
                        prob_1d = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
                        auc_val = roc_auc_score(y_true_arr, prob_1d)
                        fpr, tpr, _ = roc_curve(y_true_arr, prob_1d)
                        roc_curves[model_name] = {'fpr': fpr, 'tpr': tpr}
                    except Exception:
                        auc_val = None

                comparison[model_name] = {
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                    'AUC': auc_val,
                }

            except Exception as e:
                self._record_internal_warning(
                    'ModelComparison', f'Failed to evaluate model {model_name}', e)

        if not comparison:
            return

        # Find best model per metric
        best_per_metric = {}
        for metric in metrics_list:
            valid = {m: v[metric] for m, v in comparison.items()
                     if v.get(metric) is not None}
            if valid:
                best_model = max(valid, key=valid.get)
                best_per_metric[metric] = {
                    'model': best_model, 'value': valid[best_model]
                }

        # Build comparison table
        table_md = "| Model |"
        for m in metrics_list:
            table_md += f" {m} |"
        table_md += "\n|-------|"
        for _ in metrics_list:
            table_md += "--------|"
        table_md += "\n"

        for model_name, scores in comparison.items():
            table_md += f"| {model_name} |"
            for metric in metrics_list:
                val = scores.get(metric)
                if val is None:
                    table_md += " N/A |"
                else:
                    is_best = (metric in best_per_metric and
                               best_per_metric[metric]['model'] == model_name)
                    fmt = f"{val:.4f}"
                    table_md += f" **{fmt}** |" if is_best else f" {fmt} |"
            table_md += "\n"

        # Best per metric summary table
        best_md = "| Metric | Best Model | Score |\n|--------|-----------|-------|\n"
        for metric in metrics_list:
            if metric in best_per_metric:
                bp = best_per_metric[metric]
                best_md += f"| {metric} | {bp['model']} | {bp['value']:.4f} |\n"

        # Create charts
        roc_fig_path = ""
        if roc_curves:
            try:
                roc_fig_path = self._create_overlaid_roc_chart(roc_curves)
            except Exception as e:
                self._record_chart_failure('model_comparison_roc', e)

        radar_fig_path = ""
        try:
            radar_fig_path = self._create_metrics_radar_chart(comparison)
        except Exception as e:
            self._record_chart_failure('model_comparison_radar', e)

        content = f"""
# Model Comparison

Side-by-side evaluation of {len(models)} trained models on the test set.

## Performance Comparison

{table_md}

## Best Model Per Metric

{best_md}
"""
        if roc_fig_path:
            content += f"""
## Overlaid ROC Curves

![Model Comparison ROC]({roc_fig_path})

"""
        if radar_fig_path:
            content += f"""
## Metrics Radar Chart

![Model Comparison Radar]({radar_fig_path})

"""
        content += f"""
{self._maybe_add_narrative('Model Comparison', f'Compared models: {list(comparison.keys())}', section_type='general')}

---
"""
        self._content.append(content)
        self._store_section_data('model_comparison', 'Model Comparison', {
            'n_models': len(comparison),
            'comparison': comparison,
            'best_per_metric': best_per_metric,
        })

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
            fig_path = ""
            try:
                fig_path = self._create_cross_dataset_chart(results, metric_name)
            except Exception as e:
                self._record_chart_failure('cross_dataset', e)

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
            self._record_section_failure('Cross-Dataset Validation', e)


    def add_confusion_matrix(self, y_true, y_pred, labels: List[str] = None):
        """Add confusion matrix section."""
        try:
            self._add_confusion_matrix_impl(y_true, y_pred, labels)
        except Exception as e:
            self._record_section_failure('Classification Performance', e)

    def _add_confusion_matrix_impl(self, y_true, y_pred, labels):
        preds = self._make_predictions(y_true, y_pred)
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(preds.y_true, preds.y_pred)
        report = classification_report(preds.y_true, preds.y_pred, target_names=labels, output_dict=True)

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

        # Create confusion matrix figure (isolated)
        fig_path = ""
        try:
            fig_path = self._create_confusion_matrix_chart(cm, labels)
        except Exception as e:
            self._record_chart_failure('confusion_matrix', e)

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
        try:
            self._add_roc_analysis_impl(y_true, y_prob, pos_label)
        except Exception as e:
            self._record_section_failure('ROC Analysis', e)

    def _add_roc_analysis_impl(self, y_true, y_prob, pos_label):
        preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used, pass y_true
        from sklearn.metrics import roc_curve, auc, roc_auc_score

        fpr, tpr, thresholds = roc_curve(preds.y_true, preds.y_prob, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        # Create ROC curve figure (isolated)
        fig_path = ""
        try:
            fig_path = self._create_roc_chart(fpr, tpr, roc_auc)
        except Exception as e:
            self._record_chart_failure('roc_chart', e)

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
        try:
            self._add_precision_recall_impl(y_true, y_prob, pos_label)
        except Exception as e:
            self._record_section_failure('Precision-Recall Analysis', e)

    def _add_precision_recall_impl(self, y_true, y_prob, pos_label):
        preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, _ = precision_recall_curve(preds.y_true, preds.y_prob, pos_label=pos_label)
        avg_precision = average_precision_score(preds.y_true, preds.y_prob, pos_label=pos_label)

        # Create PR curve figure (isolated)
        fig_path = ""
        try:
            fig_path = self._create_pr_chart(precision, recall, avg_precision)
        except Exception as e:
            self._record_chart_failure('pr_chart', e)

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
        try:
            self._add_baseline_comparison_impl(baseline_score, final_score, baseline_model)
        except Exception as e:
            self._record_section_failure('Baseline Comparison', e)

    def _add_baseline_comparison_impl(self, baseline_score, final_score, baseline_model):
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

            # Create SHAP summary plot (isolated)
            fig_path = ""
            try:
                fig_path = self._create_shap_chart(shap_values, feature_names, X_sample)
            except Exception as e:
                self._record_chart_failure('shap_chart', e)

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
            self._store_section_data('shap_analysis', 'SHAP Feature Analysis', {
                'top_features': [f for f, _ in shap_importance[:10]],
                'mean_shap': dict(shap_importance[:10]),
            }, [{'type': 'importance_bar'}])

        except Exception as e:
            self._record_section_failure('SHAP Analysis', e)

    def add_recommendations(self, recommendations: List[str]):
        """Add recommendations section."""
        try:
            self._add_recommendations_impl(recommendations)
        except Exception as e:
            self._record_section_failure('Recommendations', e)

    def _add_recommendations_impl(self, recommendations):
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
        try:
            self._add_data_quality_analysis_impl(X, y)
        except Exception as e:
            self._record_section_failure('Data Quality Analysis', e)

    def _add_data_quality_analysis_impl(self, X, y):
        self._validate_inputs(X=X, require_X=True)
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

        # Create visualizations (isolated so chart failure doesn't lose tables)
        fig_missing = ""
        if n_missing_cols > 0:
            try:
                fig_missing = self._create_missing_pattern_chart(X)
            except Exception as e:
                self._record_chart_failure('missing_pattern', e)
        fig_dist = ""
        if n_numeric > 0:
            try:
                fig_dist = self._create_distribution_overview(X[numeric_cols])
            except Exception as e:
                self._record_chart_failure('distribution_overview', e)
        fig_outlier = ""
        if outlier_summary:
            try:
                fig_outlier = self._create_outlier_boxplot(X[numeric_cols], outlier_summary)
            except Exception as e:
                self._record_chart_failure('outlier_boxplot', e)

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
        try:
            self._add_correlation_analysis_impl(X, threshold)
        except Exception as e:
            self._record_section_failure('Correlation Analysis', e)

    def _add_correlation_analysis_impl(self, X, threshold):
        self._validate_inputs(X=X, require_X=True)
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

        # Create visualizations (isolated)
        fig_corr = ""
        try:
            fig_corr = self._create_correlation_heatmap(corr_matrix)
        except Exception as e:
            self._record_chart_failure('correlation_heatmap', e)
        fig_vif = ""
        if vif_data:
            try:
                fig_vif = self._create_vif_chart(vif_data)
            except Exception as e:
                self._record_chart_failure('vif_chart', e)

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
                except Exception as e:
                    if hasattr(self, '_warnings'):
                        self._record_internal_warning('VIFComputation', f'VIF failed for {col}', e)

            return vif_data
        except ImportError:
            return {}
        except Exception as e:
            logger.debug(f"VIF calculation failed: {e}")
            return {}


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
            self._validate_inputs(X=X, y_true=y, require_X=True)
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
            fig_path = ""
            try:
                fig_path = self._create_learning_curve_chart(
                    train_sizes_abs, train_mean, train_std, val_mean, val_std
                )
            except Exception as e:
                self._record_chart_failure('learning_curve', e)

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
            self._record_section_failure('Learning Curves', e)

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
            preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss

            fraction_of_positives, mean_predicted = calibration_curve(preds.y_true, preds.y_prob, n_bins=n_bins)

            # Calculate metrics
            brier_score = brier_score_loss(preds.y_true, preds.y_prob)

            # Expected Calibration Error
            bin_counts = np.histogram(preds.y_prob, bins=n_bins, range=(0, 1))[0]
            ece = np.sum(np.abs(fraction_of_positives - mean_predicted) * (bin_counts[:len(fraction_of_positives)] / preds.n_samples))

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_calibration_chart(fraction_of_positives, mean_predicted, y_prob)
            except Exception as e:
                self._record_chart_failure('calibration', e)

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
            self._record_section_failure('Calibration Analysis', e)

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
            preds = self._make_predictions(y_true, y_pred, y_prob)

            if feature_names is None:
                if hasattr(X_sample, 'columns'):
                    feature_names = list(X_sample.columns)
                else:
                    feature_names = [f'Feature_{i}' for i in range(np.asarray(X_sample).shape[1])]

            X_arr = np.asarray(X_sample)

            # Compute ECE
            ece_data = self._compute_ece(preds.y_true, preds.y_prob)

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_confidence_charts(preds.y_true, preds.y_prob, ece_data)
            except Exception as e:
                self._record_chart_failure('confidence_charts', e)

            # Find most/least confident predictions
            confidence = np.abs(preds.y_prob - 0.5) * 2  # 0 = uncertain, 1 = very confident
            correct = (~preds.errors).astype(int)

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
            self._record_section_failure('Prediction Confidence Analysis', e)

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
            preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used
            # Sort by probability descending
            sorted_idx = np.argsort(preds.y_prob)[::-1]
            y_sorted = preds.y_true[sorted_idx]

            # Calculate cumulative gains
            n = preds.n_samples
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
            fig_path = ""
            try:
                fig_path = self._create_lift_gain_chart(deciles, cum_gains, lift, ks_stat, ks_idx)
            except Exception as e:
                self._record_chart_failure('lift_gain', e)

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
            self._record_section_failure('Lift/Gain Analysis', e)

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
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
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
            fig_path = ""
            try:
                fig_path = self._create_cv_chart(cv_results)
            except Exception as e:
                self._record_chart_failure('cv_chart', e)

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
            self._record_section_failure('CV Detailed Analysis', e)

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
            fig_summary = ""
            try:
                fig_summary = self._create_shap_summary(shap_values, feature_names, X_sample)
            except Exception as e:
                self._record_chart_failure('shap_summary', e)

            fig_bar = ""
            try:
                fig_bar = self._create_shap_bar(shap_importance)
            except Exception as e:
                self._record_chart_failure('shap_bar', e)

            fig_dependence = ""
            try:
                fig_dependence = self._create_shap_dependence(shap_values, feature_names, X_sample, top_n)
            except Exception as e:
                self._record_chart_failure('shap_dependence', e)

            fig_waterfall = ""
            try:
                fig_waterfall = self._create_shap_waterfall(shap_values, feature_names, X_sample)
            except Exception as e:
                self._record_chart_failure('shap_waterfall', e)

            content = f"""
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
            self._store_section_data('shap_deep', 'SHAP Deep Analysis', {
                'top_features': [f for f, _ in shap_importance[:10]],
            })

        except Exception as e:
            self._record_section_failure('SHAP Deep Analysis', e)

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
            preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred computed per threshold
            from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

            # Calculate metrics at different thresholds
            thresholds = np.linspace(0.1, 0.9, 17)
            results = []

            for thresh in thresholds:
                y_pred_thresh = (preds.y_prob >= thresh).astype(int)
                cm = confusion_matrix(preds.y_true, y_pred_thresh)

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
            fig_path = ""
            try:
                fig_path = self._create_threshold_chart(results, best_f1['threshold'], best_cost['threshold'])
            except Exception as e:
                self._record_chart_failure('threshold_chart', e)

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
            self._record_section_failure('Threshold Optimization', e)

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
        try:
            self._add_reproducibility_section_impl(model, params, random_state, environment)
        except Exception as e:
            self._record_section_failure('Reproducibility', e)

    def _add_reproducibility_section_impl(self, model, params, random_state, environment):
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
            except Exception as e:
                if hasattr(self, '_warnings'):
                    self._record_internal_warning('PackageVersion', f'Could not get version for {pkg}', e)

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
            fig_path = ""
            try:
                fig_path = self._create_hyperparameter_charts(trials, param_names, objective_name)
            except Exception as e:
                self._record_chart_failure('hyperparameter_charts', e)

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
            self._record_section_failure('Hyperparameter Visualization', e)

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
            self._record_section_failure('Executive Dashboard', e)

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

            # Model benchmarking: CV-test gap (overfitting)
            bench_data = section_lookup.get('model_benchmarking', {})
            model_scores = bench_data.get('model_scores', {})
            for model_name, scores in model_scores.items():
                if isinstance(scores, dict):
                    cv = scores.get('cv_score', 0)
                    test = scores.get('test_score', 0)
                    if cv > 0 and test > 0 and (cv - test) > 0.1:
                        findings.append({
                            'severity': 'HIGH',
                            'source': 'Model Benchmarking',
                            'description': f"Model '{model_name}' CV-test gap = {cv - test:.3f} (>0.1, overfitting)",
                            'action': 'Regularize model, reduce complexity, or gather more training data',
                        })

            # Deployment readiness: latency check
            deploy_data = section_lookup.get('deployment_readiness', {})
            checklist = deploy_data.get('checklist', {})
            if checklist.get('latency_ok') is False:
                findings.append({
                    'severity': 'HIGH',
                    'source': 'Deployment Readiness',
                    'description': 'Model inference latency exceeds acceptable threshold',
                    'action': 'Optimize model (pruning, quantization) or use faster hardware',
                })

            # Error analysis: dominant error cluster
            error_data = section_lookup.get('error_analysis', {})
            error_clusters = error_data.get('clusters', [])
            for cluster in error_clusters:
                if isinstance(cluster, dict) and cluster.get('percentage', 0) > 30:
                    findings.append({
                        'severity': 'HIGH',
                        'source': 'Error Analysis',
                        'description': f"Error cluster '{cluster.get('name', '?')}' contains {cluster.get('percentage', 0):.0f}% of all errors",
                        'action': 'Investigate and address this systematic failure mode',
                    })

            # Deployment readiness: model size check
            if checklist.get('size_ok') is False:
                findings.append({
                    'severity': 'MEDIUM',
                    'source': 'Deployment Readiness',
                    'description': 'Model size exceeds deployment size limit',
                    'action': 'Compress model (distillation, pruning) or increase size budget',
                })

            # Regression: low R²
            reg_data = section_lookup.get('regression', {})
            r2_val = reg_data.get('r2')
            if r2_val is not None and r2_val < 0.5:
                findings.append({
                    'severity': 'MEDIUM',
                    'source': 'Regression Analysis',
                    'description': f"R² = {r2_val:.3f} (<0.5, poor explanatory power)",
                    'action': 'Add features, try non-linear models, or investigate data quality',
                })

            # Regression: heteroscedasticity
            if reg_data.get('is_heteroscedastic') is True:
                findings.append({
                    'severity': 'MEDIUM',
                    'source': 'Regression Analysis',
                    'description': 'Heteroscedasticity detected in residuals',
                    'action': 'Use weighted regression or variance-stabilizing transformations',
                })

            # Correlation: near-perfect collinearity
            corr_data = section_lookup.get('correlation', {})
            high_pairs = corr_data.get('high_corr_pairs', [])
            for pair in high_pairs:
                if isinstance(pair, dict) and abs(pair.get('corr', 0)) > 0.95:
                    findings.append({
                        'severity': 'MEDIUM',
                        'source': 'Correlation Analysis',
                        'description': f"Features '{pair.get('f1', '?')}' and '{pair.get('f2', '?')}' have |r|={abs(pair['corr']):.3f} (>0.95)",
                        'action': 'Remove one feature or use PCA to reduce collinearity',
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
            self._record_section_failure('Insight Prioritization', e)

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
                self._record_chart_failure('class_distribution', e)

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
            self._record_section_failure('Class Distribution', e)

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
            fig_path = ""
            try:
                fig_path = self._create_permutation_importance_chart(
                    result, feature_names, top_idx
                )
            except Exception as e:
                self._record_chart_failure('permutation_importance', e)

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
            self._record_section_failure('Permutation Importance', e)

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
            fig_path = ""
            try:
                fig_path = self._create_pdp_chart(model, X, top_features, feature_names)
            except Exception as e:
                self._record_chart_failure('pdp_chart', e)

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
            self._record_section_failure('Partial Dependence', e)

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
                fig_path = self._create_bootstrap_auc_chart(
                    boot_accuracies, auc_data
                )
            except Exception as e:
                self._record_chart_failure('bootstrap_auc', e)

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
            self._record_section_failure('Statistical Tests', e)

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
            except Exception as e:
                if hasattr(self, '_warnings'):
                    self._record_internal_warning('BootstrapAUC', 'single bootstrap iteration failed', e)

        if not boot_aucs:
            return None

        return {
            'values': boot_aucs,
            'mean': np.mean(boot_aucs),
            'std': np.std(boot_aucs),
            'ci_lower': np.percentile(boot_aucs, 2.5),
            'ci_upper': np.percentile(boot_aucs, 97.5),
        }

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
            preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used
            from scipy import stats as scipy_stats
            from sklearn.metrics import roc_curve

            unique_classes = np.unique(preds.y_true)
            if labels is None:
                labels = [f'Class {c}' for c in unique_classes]

            # Separate probabilities by class
            class_probs = {}
            for cls, label in zip(unique_classes, labels):
                class_probs[label] = preds.y_prob[preds.y_true == cls]

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
                fpr, tpr, thresholds = roc_curve(preds.y_true, preds.y_prob)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            else:
                optimal_threshold = 0.5

            # Create visualization
            fig_path = ""
            try:
                fig_path = self._create_score_distribution_chart(
                    class_probs, optimal_threshold
                )
            except Exception as e:
                self._record_chart_failure('score_distribution', e)

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
            self._record_section_failure('Score Distribution', e)

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
            self._record_section_failure('Deep Learning Analysis', e)

    def _is_neural_network(self, model) -> bool:
        """Check if model is a neural network using string-based type check (no framework imports)."""
        model_type = str(type(model))
        nn_indicators = [
            'torch.nn.Module', 'torch.nn.modules',
            'keras.Model', 'keras.engine', 'keras.src',
            'tensorflow.keras', 'tensorflow.python.keras',
        ]
        return any(indicator in model_type for indicator in nn_indicators)

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
            except Exception as e:
                if hasattr(self, '_warnings'):
                    self._record_internal_warning('PyTorchModelInfo', 'failed to extract PyTorch model info', e)

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
            except Exception as e:
                if hasattr(self, '_warnings'):
                    self._record_internal_warning('KerasModelInfo', 'failed to extract Keras model info', e)

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
        try:
            self._add_model_card_impl(model, results, intended_use, limitations, ethical)
        except Exception as e:
            self._record_section_failure('Model Card', e)

    def _add_model_card_impl(self, model, results, intended_use, limitations, ethical):
        import platform

        model_type = type(model).__name__ if model else 'Unknown'
        model_params = {}
        if model and hasattr(model, 'get_params'):
            try:
                model_params = model.get_params()
            except Exception as e:
                if hasattr(self, '_warnings'):
                    self._record_internal_warning('ModelParams', 'failed to get model params', e)

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
                if hasattr(self, '_warnings'):
                    self._record_internal_warning('MAPEComputation', 'sklearn MAPE failed, using manual', e)
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
            self._store_section_data('regression', 'Regression Analysis', {
                'r2': r2, 'rmse': rmse,
                'is_heteroscedastic': hetero_result.get('is_heteroscedastic', False),
            })

        except Exception as e:
            self._record_section_failure('Regression Analysis', e)

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
        except Exception as e:
            if hasattr(self, '_warnings'):
                self._record_internal_warning('Heteroscedasticity', 'Breusch-Pagan test failed', e)
            return {
                'statistic': 'N/A',
                'p_value': 'N/A',
                'result': 'Test unavailable',
                'is_heteroscedastic': False,
            }

    def generate(self, filename: str = None) -> Optional[str]:
        """Generate the PDF report using Pandoc + LaTeX."""
        try:
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
        finally:
            self.restore_plot_style()

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
            self._record_section_failure('HTML Report Generation', e)
            return None
        finally:
            self.restore_plot_style()

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

        # Append Environment & Reproducibility section
        env = self._metadata.get('environment', {})
        if env:
            libs = env.get('libraries', {})
            env_md = f"""
# Environment & Reproducibility

## Report Generator

| Property | Value |
|----------|-------|
| Report Version | {env.get('report_version', 'N/A')} |
| Generated At | {env.get('timestamp', 'N/A')} |

## System

| Property | Value |
|----------|-------|
| Python Version | {env.get('python_version', 'N/A').split(chr(10))[0]} |
| Platform | {env.get('platform', 'N/A')} |
| Machine | {env.get('machine', 'N/A')} |

## Library Versions

| Library | Version |
|---------|---------|
"""
            for lib_name, lib_ver in libs.items():
                env_md += f"| {lib_name} | {lib_ver} |\n"

            env_md += "\n---\n"
            body += env_md

            self._store_section_data('environment', 'Environment & Reproducibility', {
                'report_version': env.get('report_version'),
                'python_version': env.get('python_version', '').split('\n')[0],
                'platform': env.get('platform'),
                'libraries': libs,
            })

        # Append report health summary if there were failures or warnings
        health = self.get_report_health()
        has_issues = self._failed_sections or self._failed_charts or health['total_warnings'] > 0
        if has_issues:
            health_md = f"""
# Report Health Summary

| Metric | Value |
|--------|-------|
| Total Sections Attempted | {health['total_sections']} |
| Succeeded | {health['succeeded']} |
| Failed Sections | {health['failed']} |
| Failed Charts | {health['total_charts_failed']} |
| Warnings | {health['total_warnings']} |

"""
            if health['failed_sections']:
                health_md += """## Failed Sections

| Section | Error Type | Error Message |
|---------|-----------|---------------|
"""
                for fs in health['failed_sections']:
                    health_md += f"| {fs['section']} | {fs['error_type']} | {fs['error_message'][:80]} |\n"
                health_md += "\n"

            if health['failed_charts']:
                health_md += """## Failed Charts

| Chart | Error Type | Error Message |
|-------|-----------|---------------|
"""
                for fc in health['failed_charts']:
                    health_md += f"| {fc['chart']} | {fc['error_type']} | {fc['error_message'][:80]} |\n"
                health_md += "\n"

            if health['warnings']:
                health_md += """## Warnings

| Component | Message | Error Type |
|-----------|---------|-----------|
"""
                for w in health['warnings']:
                    err_type = w.get('error_type') or ''
                    health_md += f"| {w['component']} | {w['message'][:80]} | {err_type} |\n"
                health_md += "\n"

            health_md += "---\n"
            body += health_md

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






