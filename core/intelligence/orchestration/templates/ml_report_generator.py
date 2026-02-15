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
from ._rendering_mixin import RenderingMixin
from ._analysis_sections_mixin import AnalysisSectionsMixin


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
    def _validate_inputs(self, **kwargs: Any) -> None: ...


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

    def __post_init__(self) -> None:
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
    def from_predictions(cls, y_true: Any, y_pred: Any, y_prob: Any = None) -> Any:
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


class ProfessionalMLReport(VisualizationMixin, InterpretabilityMixin, DriftMixin, RenderingMixin, AnalysisSectionsMixin,
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

    def __init__(self, output_dir: str = 'professional_reports', theme: str = 'professional', llm_narrative: bool = False, html_enabled: bool = False, config: Dict = None) -> None:
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

    def _setup_plot_style(self) -> Any:
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

    def restore_plot_style(self) -> None:
        """Restore matplotlib rcParams to their state before report initialization."""
        try:
            import matplotlib as mpl
            if hasattr(self, '_saved_rcparams'):
                mpl.rcParams.update(self._saved_rcparams)
        except Exception as e:
            if hasattr(self, '_warnings'):
                self._record_internal_warning('RestorePlotStyle', 'failed to restore rcParams', e)

    def _record_section_failure(self, section_name: str, error: Exception) -> Any:
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

    def _validate_inputs(self, **kwargs: Any) -> Any:
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
    def _make_predictions(y_true: Any, y_pred: Any, y_prob: Any = None) -> 'PredictionResult':
        """Create a validated PredictionResult from raw arrays.

        Convenience wrapper used at the start of internal ``_impl`` methods
        to consolidate numpy conversion and length validation.
        """
        return PredictionResult.from_predictions(y_true, y_pred, y_prob)

    def _record_chart_failure(self, chart_name: str, error: Exception) -> Any:
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

    def _record_internal_warning(self, component: str, message: str, error: Exception = None) -> Any:
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

    def _store_section_data(self, section_type: str, title: str, data: Dict, chart_configs: List[Dict] = None) -> Any:
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
            from Jotty.core.infrastructure.integration.llm.unified import UnifiedLLM
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

    def set_metadata(self, title: str, subtitle: str = '', author: str = 'Jotty ML', dataset: str = '', problem_type: str = 'Classification') -> Any:
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

    def _add_executive_summary_impl(self, metrics: Any, best_model: Any, n_features: Any, context: Any) -> Any:
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

    def _add_data_profile_impl(self, shape: Any, dtypes: Any, missing: Any, recommendations: Any) -> Any:
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

    def _add_feature_importance_impl(self, importance: Any, top_n: Any) -> Any:
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

    def _add_model_benchmarking_impl(self, model_scores: Any) -> Any:
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

    def _add_model_comparison_impl(self, models: Any, X_test: Any, y_true: Any, class_labels: Any = None) -> Any:
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

                n_classes = len(np.unique(y_true_arr))
                avg_method = 'binary' if n_classes <= 2 else 'weighted'

                acc = accuracy_score(y_true_arr, y_pred)
                prec = precision_score(y_true_arr, y_pred, average=avg_method, zero_division=0)
                rec = recall_score(y_true_arr, y_pred, average=avg_method, zero_division=0)
                f1 = f1_score(y_true_arr, y_pred, average=avg_method, zero_division=0)

                if y_prob is not None:
                    try:
                        if n_classes <= 2:
                            prob_1d = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
                            auc_val = roc_auc_score(y_true_arr, prob_1d)
                            fpr, tpr, _ = roc_curve(y_true_arr, prob_1d)
                            roc_curves[model_name] = {'fpr': fpr, 'tpr': tpr}
                        else:
                            auc_val = roc_auc_score(y_true_arr, y_prob, multi_class='ovr', average='weighted')
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

    def _add_confusion_matrix_impl(self, y_true: Any, y_pred: Any, labels: Any) -> Any:
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

    def _add_roc_analysis_impl(self, y_true: Any, y_prob: Any, pos_label: Any) -> Any:
        preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used, pass y_true
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        from sklearn.preprocessing import label_binarize

        n_classes = len(np.unique(preds.y_true))
        is_binary = n_classes <= 2

        if is_binary:
            # Binary: single ROC curve
            prob_1d = preds.y_prob if preds.y_prob.ndim == 1 else preds.y_prob[:, 1]
            fpr, tpr, thresholds = roc_curve(preds.y_true, prob_1d, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)

            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

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
        else:
            # Multiclass: One-vs-Rest ROC curves
            classes = np.unique(preds.y_true)
            y_bin = label_binarize(preds.y_true, classes=classes)
            y_prob_2d = preds.y_prob if preds.y_prob.ndim == 2 else None

            if y_prob_2d is None or y_prob_2d.shape[1] != n_classes:
                self._record_section_failure('ROC Analysis',
                    ValueError(f"y_prob shape {preds.y_prob.shape} incompatible with {n_classes} classes"))
                return

            roc_curves = {}
            per_class_auc = {}
            for i, cls in enumerate(classes):
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob_2d[:, i])
                auc_i = auc(fpr_i, tpr_i)
                roc_curves[f"Class {cls}"] = {'fpr': fpr_i, 'tpr': tpr_i}
                per_class_auc[f"Class {cls}"] = auc_i

            roc_auc = np.mean(list(per_class_auc.values()))

            fig_path = ""
            try:
                fig_path = self._create_overlaid_roc_chart(roc_curves)
            except Exception as e:
                self._record_chart_failure('roc_chart_multiclass', e)

            content = f"""
# ROC Curve Analysis (One-vs-Rest)

Multiclass ROC analysis using One-vs-Rest strategy for {n_classes} classes.

## Per-Class AUC

| Class | AUC |
|-------|-----|
"""
            for cls_name, auc_val in per_class_auc.items():
                content += f"| {cls_name} | {auc_val:.4f} |\n"

            content += f"""
**Macro-Average AUC:** {roc_auc:.4f}

## ROC Curves

![ROC Curves]({fig_path})

---
"""
        self._content.append(content)
        self._store_section_data('roc_analysis', 'ROC Curve Analysis', {'auc': roc_auc})

    def _add_precision_recall_impl(self, y_true: Any, y_prob: Any, pos_label: Any) -> Any:
        preds = self._make_predictions(y_true, y_true, y_prob)  # y_pred not used
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.preprocessing import label_binarize

        n_classes = len(np.unique(preds.y_true))
        is_binary = n_classes <= 2

        if is_binary:
            prob_1d = preds.y_prob if preds.y_prob.ndim == 1 else preds.y_prob[:, 1]
            precision, recall, _ = precision_recall_curve(preds.y_true, prob_1d, pos_label=pos_label)
            avg_precision = average_precision_score(preds.y_true, prob_1d, pos_label=pos_label)

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
        else:
            # Multiclass: One-vs-Rest PR curves
            classes = np.unique(preds.y_true)
            y_bin = label_binarize(preds.y_true, classes=classes)
            y_prob_2d = preds.y_prob if preds.y_prob.ndim == 2 else None

            if y_prob_2d is None or y_prob_2d.shape[1] != n_classes:
                self._record_section_failure('Precision-Recall Analysis',
                    ValueError(f"y_prob shape {preds.y_prob.shape} incompatible with {n_classes} classes"))
                return

            per_class_ap = {}
            for i, cls in enumerate(classes):
                ap_i = average_precision_score(y_bin[:, i], y_prob_2d[:, i])
                per_class_ap[f"Class {cls}"] = ap_i

            avg_precision = np.mean(list(per_class_ap.values()))

            content = f"""
# Precision-Recall Analysis (One-vs-Rest)

Multiclass precision-recall analysis for {n_classes} classes.

## Per-Class Average Precision

| Class | Average Precision |
|-------|------------------|
"""
            for cls_name, ap_val in per_class_ap.items():
                content += f"| {cls_name} | {ap_val:.4f} |\n"

            content += f"""
**Macro-Average Precision:** {avg_precision:.4f}

---
"""
        self._content.append(content)
        self._store_section_data('precision_recall', 'Precision-Recall Analysis', {'avg_precision': avg_precision})

    def _add_baseline_comparison_impl(self, baseline_score: Any, final_score: Any, baseline_model: Any) -> Any:
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

    def _add_recommendations_impl(self, recommendations: Any) -> Any:
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

    def _add_data_quality_analysis_impl(self, X: Any, y: Any) -> Any:
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

    def _add_correlation_analysis_impl(self, X: Any, threshold: Any) -> Any:
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

    def _compute_ece(self, y_true: Any, y_prob: Any, n_bins: int = 10) -> Dict:
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

    def _add_reproducibility_section_impl(self, model: Any, params: Any, random_state: Any, environment: Any) -> Any:
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

    def _normalize_trials(self, study_or_trials: Any) -> List[Dict]:
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

    def _bootstrap_auc_ci(self, y_true: Any, y_prob: Any, n_boot: int = 1000) -> Dict:
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

    def _is_neural_network(self, model: Any) -> bool:
        """Check if model is a neural network using string-based type check (no framework imports)."""
        model_type = str(type(model))
        nn_indicators = [
            'torch.nn.Module', 'torch.nn.modules',
            'keras.Model', 'keras.engine', 'keras.src',
            'tensorflow.keras', 'tensorflow.python.keras',
        ]
        return any(indicator in model_type for indicator in nn_indicators)

    def _get_nn_architecture_info(self, model: Any) -> Dict:
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

    def _compute_gradient_attribution(self, model: Any, X_sample: Any) -> Dict:
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

    def _add_model_card_impl(self, model: Any, results: Any, intended_use: Any, limitations: Any, ethical: Any) -> Any:
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

    def _detect_heteroscedasticity(self, y_pred: Any, residuals: Any) -> Dict:
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

    def _extract_section_title(self, content: str) -> str:
        """Extract H1 title from markdown content block."""
        import re
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        return match.group(1).strip() if match else 'Section'

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


