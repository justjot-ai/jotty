"""
Jotty Swarm - Unified API
=========================

The main entry point for Jotty's intelligent swarm system.

This is the WORLD'S BEST agentic system for solving complex problems.

Features:
- Template-based orchestration (SwarmML, SwarmNLP, SwarmCV, etc.)
- Auto-detection of problem type
- Multi-agent collaboration
- LLM-powered reasoning with chain-of-thought
- Feedback loops for iterative improvement
- Progress tracking with visual feedback

Usage:
    from jotty import Swarm

    # Solve ML problem
    result = await Swarm.solve(template="ml", X=X, y=y)

    # Auto-detect and solve
    result = await Swarm.auto_solve(X, y)

    # Custom configuration
    result = await Swarm.solve(
        template="ml",
        X=X, y=y,
        time_budget=300,
        context="Predict customer churn",
        feedback_iterations=2,
    )
"""

from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np
import asyncio
import time
import logging

from .templates import TemplateRegistry, SwarmTemplate, SwarmML
from .templates.base import TemplateExecutor
from .swarm_manager import SwarmManager

logger = logging.getLogger(__name__)


class SwarmResult:
    """
    Result from a Swarm execution.

    Contains all outputs from the template execution.
    """

    def __init__(self,
                 success: bool,
                 score: float = 0.0,
                 model: Any = None,
                 data: Any = None,
                 feature_count: int = 0,
                 execution_time: float = 0.0,
                 template_name: str = "",
                 stage_results: Dict[str, Any] = None,
                 metadata: Dict[str, Any] = None):
        self.success = success
        self.score = score
        self.model = model
        self.data = data
        self.feature_count = feature_count
        self.execution_time = execution_time
        self.template_name = template_name
        self.stage_results = stage_results or {}
        self.metadata = metadata or {}

    def __repr__(self):
        return f"SwarmResult(success={self.success}, score={self.score:.4f}, template={self.template_name})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'score': self.score,
            'feature_count': self.feature_count,
            'execution_time': self.execution_time,
            'template_name': self.template_name,
            'metadata': self.metadata,
        }


class Swarm:
    """
    Main entry point for Jotty Swarm System.

    This is the WORLD'S BEST agentic system for solving complex problems.

    Designed to be:
    - Simple: One line to solve any problem
    - Powerful: Best-in-class algorithms and techniques
    - Intelligent: LLM-powered reasoning and feedback loops
    - Extensible: Custom templates and skills

    Example:
        # Solve ML problem
        result = await Swarm.solve(template="ml", X=X, y=y)

        # Auto-detect
        result = await Swarm.auto_solve(X, y)

        # With configuration
        result = await Swarm.solve(
            template="ml",
            X=X, y=y,
            time_budget=300,
            context="Predict customer churn"
        )
    """

    _instance: Optional['Swarm'] = None
    _swarm_manager: Optional[SwarmManager] = None

    @classmethod
    async def solve(cls,
                    template: Union[str, SwarmTemplate] = "ml",
                    X: pd.DataFrame = None,
                    y: pd.Series = None,
                    time_budget: int = 300,
                    context: str = "",
                    feedback_iterations: int = 2,
                    show_progress: bool = True,
                    **kwargs) -> SwarmResult:
        """
        Solve a problem using a swarm template.

        This is the main entry point for Jotty.

        Args:
            template: Template name ("ml", "nlp", "cv") or SwarmTemplate instance
            X: Input features (DataFrame or path)
            y: Target variable (optional for some templates)
            time_budget: Maximum seconds for execution
            context: Business context for LLM reasoning
            feedback_iterations: Number of feedback loop iterations
            show_progress: Show progress bar
            **kwargs: Additional template-specific parameters

        Returns:
            SwarmResult with model, score, and metadata
        """
        start_time = time.time()

        # Get template
        if isinstance(template, str):
            template_instance = TemplateRegistry.get(template)
        else:
            template_instance = template

        # Log start
        print(f"\n{'='*60}")
        print(f"🚀 JOTTY SWARM - {template_instance.name}")
        print(f"{'='*60}")

        # Detect problem type
        problem_type = template_instance.detect_problem_type(X, y)
        print(f"📋 Problem: {problem_type}")

        # For ML templates, use the existing SkillOrchestrator
        # (which we'll eventually migrate to the new template system)
        if template_instance.name == "SwarmML":
            return await cls._solve_ml(
                X, y, time_budget, context, feedback_iterations, show_progress, **kwargs
            )

        # For other templates, use the template executor
        executor = TemplateExecutor(template_instance)

        # Build context
        execution_context = {
            'X': X,
            'y': y,
            'time_budget': time_budget,
            'business_context': context,
            'problem_type': problem_type,
            'feedback_iterations': feedback_iterations,
            **kwargs
        }

        # Set progress callback
        if show_progress:
            executor.set_progress_callback(cls._progress_callback)

        # Execute
        try:
            results = await executor.execute(**execution_context)

            execution_time = time.time() - start_time

            # Extract final results
            score = results.get('final_score', 0)
            model = results.get('final_model')
            data = results.get('selected_X', X)

            print(f"\n{'='*60}")
            print(f"🏆 COMPLETE | Score: {score:.4f} | Time: {execution_time:.1f}s")
            print(f"{'='*60}\n")

            return SwarmResult(
                success=True,
                score=score,
                model=model,
                data=data,
                feature_count=data.shape[1] if hasattr(data, 'shape') else 0,
                execution_time=execution_time,
                template_name=template_instance.name,
                stage_results=results,
            )

        except Exception as e:
            logger.error(f"Swarm execution failed: {e}")
            return SwarmResult(
                success=False,
                template_name=template_instance.name,
                metadata={'error': str(e)},
            )

    @classmethod
    async def _solve_ml(cls,
                        X: pd.DataFrame,
                        y: pd.Series,
                        time_budget: int,
                        context: str,
                        feedback_iterations: int,
                        show_progress: bool,
                        **kwargs) -> SwarmResult:
        """
        Solve ML problem using the existing SkillOrchestrator.

        This provides backward compatibility while we migrate to the new template system.
        """
        from .skill_orchestrator import get_skill_orchestrator

        start_time = time.time()

        orchestrator = get_skill_orchestrator()
        result = await orchestrator.solve(
            X=X,
            y=y,
            time_budget=time_budget,
            business_context=context,
        )

        execution_time = time.time() - start_time

        return SwarmResult(
            success=True,
            score=result.best_score,
            model=result.best_model,
            data=None,  # Could extract from result
            feature_count=result.feature_count,
            execution_time=execution_time,
            template_name="SwarmML",
            stage_results={r.skill_name: r.metrics for r in result.skill_results},
            metadata={
                'problem_type': result.problem_type.value,
                'feature_importance': result.feature_importance,
            },
        )

    @classmethod
    async def auto_solve(cls, X, y=None, **kwargs) -> SwarmResult:
        """
        Auto-detect problem type and solve using best template.

        This is the simplest API - just pass your data and Jotty figures out the rest.

        Args:
            X: Input data (DataFrame, array, or path)
            y: Target variable (optional)
            **kwargs: Additional parameters

        Returns:
            SwarmResult
        """
        # Auto-detect template
        template = TemplateRegistry.auto_detect(X, y)
        logger.info(f"Auto-detected template: {template.name}")

        return await cls.solve(template=template, X=X, y=y, **kwargs)

    @classmethod
    def _progress_callback(cls, stage_name: str, status: str, result: Dict = None):
        """Progress callback for visual feedback."""
        if status == 'start':
            print(f"  ⏳ {stage_name}...")
        elif status == 'complete':
            metrics = result or {}
            metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in list(metrics.items())[:3])
            print(f"  ✅ {stage_name} | {metric_str}")

    @classmethod
    def list_templates(cls) -> List[Dict]:
        """List all available templates."""
        return TemplateRegistry.list_templates()

    @classmethod
    def get_template(cls, name: str) -> SwarmTemplate:
        """Get a template by name."""
        return TemplateRegistry.get(name)


# Convenience functions for module-level access
async def solve(template: str = "ml", **kwargs) -> SwarmResult:
    """Solve using specified template."""
    return await Swarm.solve(template=template, **kwargs)


async def auto_solve(X, y=None, **kwargs) -> SwarmResult:
    """Auto-detect and solve."""
    return await Swarm.auto_solve(X=X, y=y, **kwargs)


# Make Swarm available at package level
__all__ = ['Swarm', 'SwarmResult', 'solve', 'auto_solve']
