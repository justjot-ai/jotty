"""
Skill Types - Data structures for ML skill orchestration
=========================================================

Contains:
- ProblemType: Types of ML problems (orchestration-specific)
- SkillCategory, SkillResult: Re-exported from core/skills/ml/base.py
- PipelineResult: Result from full pipeline
- ProgressTracker: Visual progress tracking
- SkillAdapter: Wraps skills with standardized interface

Note: SkillCategory and SkillResult are imported from core/skills/ml/base.py
to follow DRY principles - single source of truth for ML skill types.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# IMPORT FROM CANONICAL SOURCE (DRY)
# =============================================================================

try:
    from Jotty.core.skills.ml.base import SkillCategory, SkillResult
except ImportError:
    # Fallback if skills module not available
    class SkillCategory(Enum):
        """Categories of ML skills in the pipeline."""
        DATA_PROFILING = "data_profiling"
        DATA_CLEANING = "data_cleaning"
        FEATURE_ENGINEERING = "feature_engineering"
        FEATURE_SELECTION = "feature_selection"
        MODEL_SELECTION = "model_selection"
        HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
        ENSEMBLE = "ensemble"
        EVALUATION = "evaluation"
        EXPLANATION = "explanation"

    @dataclass
    class SkillResult:
        """Result from a skill execution."""
        skill_name: str
        category: SkillCategory
        success: bool
        data: Any = None
        metrics: Dict[str, float] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
        error: Optional[str] = None


# =============================================================================
# ORCHESTRATION-SPECIFIC TYPES
# =============================================================================

class ProblemType(Enum):
    """Types of machine learning problems."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"


@dataclass
class PipelineResult:
    """Result from full pipeline execution."""
    problem_type: ProblemType
    best_score: float
    best_model: Any
    feature_count: int
    skill_results: List[SkillResult] = field(default_factory=list)
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    processed_X: Optional[Any] = None


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

class ProgressTracker:
    """Visual progress tracking for ML pipeline stages."""

    STAGE_WEIGHTS = {
        'DATA_PROFILING': 2,
        'DATA_CLEANING': 3,
        'LLM_FEATURE_REASONING': 10,
        'FEATURE_ENGINEERING': 8,
        'FEATURE_SELECTION': 5,
        'MODEL_SELECTION': 25,
        'HYPERPARAMETER_OPTIMIZATION': 30,
        'ENSEMBLE': 15,
        'EVALUATION': 1,
        'EXPLANATION': 1,
    }

    def __init__(self, total_stages: int = 9):
        self.total_stages = total_stages
        self.current_stage = 0
        self.current_stage_name = ""
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.completed_weight = 0
        self.total_weight = sum(self.STAGE_WEIGHTS.values())

    def start_stage(self, stage_name: str) -> None:
        """Start a new stage."""
        self.current_stage += 1
        self.current_stage_name = stage_name
        self.stage_start_time = time.time()
        self._print_progress()

    def complete_stage(self, stage_name: str, metrics: Dict = None) -> None:
        """Complete current stage."""
        weight = self.STAGE_WEIGHTS.get(stage_name.upper(), 5)
        self.completed_weight += weight
        elapsed = time.time() - self.stage_start_time
        self._print_completion(stage_name, elapsed, metrics)

    def _print_progress(self):
        """Log progress bar."""
        pct = (self.completed_weight / self.total_weight) * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        elapsed = time.time() - self.start_time

        logger.info(f'[{bar}] {pct:5.1f}% | Stage {self.current_stage}/{self.total_stages}: {self.current_stage_name:<25} | {elapsed:.0f}s')

    def _print_completion(self, stage_name: str, elapsed: float, metrics: Dict = None):
        """Print stage completion."""
        pct = (self.completed_weight / self.total_weight) * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)

        metric_str = ""
        if metrics:
            key_metrics = {k: v for k, v in list(metrics.items())[:3]}
            metric_str = " | " + ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in key_metrics.items()
            )

        logger.info(f'[{bar}] {pct:5.1f}% | Done: {stage_name:<25} ({elapsed:.1f}s){metric_str}')

    def finish(self, final_score: float) -> None:
        """Log final summary."""
        total_time = time.time() - self.start_time
        logger.info(f'{"="*60}')
        logger.info(f'COMPLETE | Score: {final_score:.4f} | Total time: {total_time:.1f}s')
        logger.info(f'{"="*60}')


# =============================================================================
# SKILL ADAPTER
# =============================================================================

class SkillAdapter:
    """
    Adapter that wraps skills with a standardized ML interface.
    Converts skill tools to fit/transform/predict pattern.
    """

    def __init__(self, skill_name: str, skill_def, tools_registry: Any):
        self.skill_name = skill_name
        self.skill_def = skill_def
        self.tools_registry = tools_registry
        self.category = self._infer_category()

    def _infer_category(self) -> SkillCategory:
        """Infer skill category from name/description."""
        name = self.skill_name.lower()
        # Handle both dict and SkillDefinition object
        if hasattr(self.skill_def, 'description'):
            desc = str(self.skill_def.description).lower()
        elif isinstance(self.skill_def, dict):
            desc = str(self.skill_def.get('description', '')).lower()
        else:
            desc = ''

        if 'profil' in name or 'profil' in desc:
            return SkillCategory.DATA_PROFILING
        elif 'clean' in name or 'valid' in name:
            return SkillCategory.DATA_CLEANING
        elif 'feature' in name and 'select' in name:
            return SkillCategory.FEATURE_SELECTION
        elif 'feature' in name or 'engineer' in name:
            return SkillCategory.FEATURE_ENGINEERING
        elif 'hyper' in name or 'optim' in name or 'optuna' in name:
            return SkillCategory.HYPERPARAMETER_OPTIMIZATION
        elif 'ensemble' in name or 'stack' in name or 'blend' in name:
            return SkillCategory.ENSEMBLE
        elif 'automl' in name or 'auto-ml' in name or 'model' in name:
            return SkillCategory.MODEL_SELECTION
        elif 'metric' in name or 'eval' in name:
            return SkillCategory.EVALUATION
        elif 'shap' in name or 'explain' in name or 'interpret' in name:
            return SkillCategory.EXPLANATION
        else:
            return SkillCategory.FEATURE_ENGINEERING

    async def execute(self, context: Dict) -> SkillResult:
        """Execute the skill with given context."""
        try:
            # Get tool functions from skill (handle both dict and SkillDefinition)
            if hasattr(self.skill_def, 'tools'):
                tools = self.skill_def.tools or {}
            elif isinstance(self.skill_def, dict):
                tools = self.skill_def.get('tools', {})
            else:
                tools = {}

            if not tools:
                return SkillResult(
                    skill_name=self.skill_name,
                    category=self.category,
                    success=False,
                    error="No tools defined"
                )

            # Execute based on category
            result = await self._execute_by_category(context)
            return result

        except Exception as e:
            logger.warning(f"Skill {self.skill_name} failed: {e}")
            return SkillResult(
                skill_name=self.skill_name,
                category=self.category,
                success=False,
                error=str(e)
            )

    async def _execute_by_category(self, context: Dict) -> SkillResult:
        """Execute skill based on its category."""
        # Stub: no real skill execution yet — fall through to builtins
        return SkillResult(
            skill_name=self.skill_name,
            category=self.category,
            success=False,
            error="No real implementation — using builtin fallback"
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ProblemType',
    'SkillCategory',
    'SkillResult',
    'PipelineResult',
    'ProgressTracker',
    'SkillAdapter',
]
