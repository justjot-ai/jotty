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

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .swarm_manager import Orchestrator
from .templates import SwarmML, SwarmTemplate, TemplateRegistry
from .templates.base import TemplateExecutor

logger = logging.getLogger(__name__)


class TemplateSwarmResult:
    """
    Result from a Swarm execution.

    Contains all outputs from the template execution.
    """

    def __init__(
        self,
        success: bool,
        score: float = 0.0,
        model: Any = None,
        data: Any = None,
        feature_count: int = 0,
        execution_time: float = 0.0,
        template_name: str = "",
        stage_results: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        self.success = success
        self.score = score
        self.model = model
        self.data = data
        self.feature_count = feature_count
        self.execution_time = execution_time
        self.template_name = template_name
        self.stage_results = stage_results or {}
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"TemplateSwarmResult(success={self.success}, score={self.score:.4f}, template={self.template_name})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "score": self.score,
            "feature_count": self.feature_count,
            "execution_time": self.execution_time,
            "template_name": self.template_name,
            "metadata": self.metadata,
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

    _instance: Optional["Swarm"] = None
    _swarm_manager: Optional[Orchestrator] = None
    _mas_learning = None

    @classmethod
    def _get_swarm_manager(cls) -> Any:
        """Get or create Orchestrator for learning integration."""
        if cls._swarm_manager is None:
            try:
                cls._swarm_manager = Orchestrator(config=None)
            except Exception as e:
                logger.debug(f"Orchestrator init failed, learning disabled: {e}")
        return cls._swarm_manager

    @classmethod
    def _get_learning(cls) -> Any:
        """Get learning system, with or without full Orchestrator."""
        manager = cls._get_swarm_manager()
        if manager:
            return manager.get_ml_learning()
        # Fallback: create lightweight MASLearning directly
        if cls._mas_learning is None:
            try:
                from .mas_learning import MASLearning

                cls._mas_learning = MASLearning(config=None, workspace_path=None)
            except Exception:
                pass
        return cls._mas_learning

    @classmethod
    async def solve(
        cls,
        template: Union[str, SwarmTemplate] = "ml",
        data: pd.DataFrame = None,
        target: str = None,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        time_budget: int = 300,
        context: str = "",
        feedback_iterations: int = 2,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> TemplateSwarmResult:
        """
        Solve a problem using a swarm template.

        This is the main entry point for Jotty.

        Args:
            data: Full DataFrame (alternative to X/y). Swarm splits internally.
            target: Target column name (used with data param)
            template: Template name ("ml", "nlp", "cv") or SwarmTemplate instance
            X: Input features (DataFrame) — used if data/target not provided
            y: Target variable (Series) — used if data/target not provided
            time_budget: Maximum seconds for execution
            context: Business context for LLM reasoning
            feedback_iterations: Number of feedback loop iterations
            show_progress: Show progress bar
            **kwargs: Additional template-specific parameters

        Returns:
            TemplateSwarmResult with model, score, and metadata
        """
        start_time = time.time()

        # If data + target provided, split into X/y automatically
        if data is not None and target is not None:
            if target not in data.columns:
                raise ValueError(
                    f"Target column '{target}' not found in data. "
                    f"Available: {list(data.columns)}"
                )
            y = data[target]
            X = data.drop(columns=[target])
        elif data is not None and target is None:
            raise ValueError("When passing 'data', you must also specify 'target' column name")

        # Get template
        if isinstance(template, str):
            template_instance = TemplateRegistry.get(template)
        else:
            template_instance = template

        # Log start
        logger.info(f"{'='*60}")
        logger.info(f"JOTTY SWARM - {template_instance.name}")
        logger.info(f"{'='*60}")

        # Detect problem type
        problem_type = template_instance.detect_problem_type(X, y)
        logger.info(f"Problem type: {problem_type}")

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
            "X": X,
            "y": y,
            "time_budget": time_budget,
            "business_context": context,
            "problem_type": problem_type,
            "feedback_iterations": feedback_iterations,
            **kwargs,
        }

        # Set progress callback
        if show_progress:
            executor.set_progress_callback(cls._progress_callback)

        # Execute
        try:
            results = await executor.execute(**execution_context)

            execution_time = time.time() - start_time

            # Extract final results
            score = results.get("final_score", 0)
            model = results.get("final_model")
            data = results.get("selected_X", X)

            logger.info(f"{'='*60}")
            logger.info(f"COMPLETE | Score: {score:.4f} | Time: {execution_time:.1f}s")
            logger.info(f"{'='*60}")

            return TemplateSwarmResult(
                success=True,
                score=score,
                model=model,
                data=data,
                feature_count=data.shape[1] if hasattr(data, "shape") else 0,
                execution_time=execution_time,
                template_name=template_instance.name,
                stage_results=results,
            )

        except Exception as e:
            logger.error(f"Swarm execution failed: {e}")
            return TemplateSwarmResult(
                success=False,
                template_name=template_instance.name,
                metadata={"error": str(e)},
            )

    @classmethod
    async def _solve_ml(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        time_budget: int,
        context: str,
        feedback_iterations: int,
        show_progress: bool,
        **kwargs: Any,
    ) -> TemplateSwarmResult:
        """
        Solve ML problem using the existing SkillOrchestrator.

        This provides backward compatibility while we migrate to the new template system.
        Integrates learning lifecycle: before_execution -> pipeline -> after_execution.
        """
        from .skill_orchestrator import get_skill_orchestrator

        start_time = time.time()

        orchestrator = get_skill_orchestrator()

        # --- Initialize learning from Orchestrator ---
        template_instance = kwargs.get("_template_instance", None)
        if template_instance is None:
            template_instance = TemplateRegistry.get("ml")

        manager = cls._get_swarm_manager()
        learning = cls._get_learning()
        if learning:
            try:
                template_instance.set_learning(learning)
                await template_instance.before_execution(business_context=context, X=X, y=y)
            except Exception as e:
                logger.debug(f"Learning pre-execution failed: {e}")

        # --- Execute pipeline with stage callback ---
        result = await orchestrator.solve(
            X=X,
            y=y,
            time_budget=time_budget,
            business_context=context,
            on_stage_complete=template_instance.on_stage_complete,
        )

        execution_time = time.time() - start_time

        # --- Post-execution learning ---
        try:
            await template_instance.after_execution(
                results={
                    "final_score": result.best_score,
                    "best_model": type(result.best_model).__name__ if result.best_model else "",
                    "feature_importance": result.feature_importance or {},
                    "problem_type": result.problem_type.value,
                },
                business_context=context,
            )
        except Exception as e:
            logger.debug(f"Learning post-execution failed: {e}")

        swarm_result = TemplateSwarmResult(
            success=True,
            score=result.best_score,
            model=result.best_model,
            data=result.processed_X,
            feature_count=result.feature_count,
            execution_time=execution_time,
            template_name="SwarmML",
            stage_results={r.skill_name: r.metrics for r in result.skill_results},
            metadata={
                "problem_type": result.problem_type.value,
                "feature_importance": result.feature_importance,
                "y": y,
            },
        )

        # Auto-generate report if requested
        if kwargs.get("generate_report", False) and result.best_model is not None:
            swarm_result = cls._generate_report(swarm_result, context, _manager=manager, **kwargs)

        return swarm_result

    @classmethod
    def _generate_report(
        cls, swarm_result: "TemplateSwarmResult", context: str, _manager: Any = None, **kwargs: Any
    ) -> "TemplateSwarmResult":
        """Generate world-class PDF report and optionally send to Telegram."""
        try:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
            from sklearn.preprocessing import StandardScaler

            from .skill_orchestrator import SkillOrchestrator
            from .templates.swarm_ml_comprehensive import SwarmMLComprehensive

            model = swarm_result.model
            X_processed = swarm_result.data
            y = swarm_result.metadata["y"]

            # Model was trained on scaled data — scale before predicting
            X_scaled = StandardScaler().fit_transform(X_processed)

            y_pred = model.predict(X_scaled)

            # Detect binary vs multiclass for correct metric averaging
            n_classes = len(set(y))
            is_binary = n_classes <= 2
            avg = "binary" if is_binary else "weighted"

            y_prob = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_scaled)
                y_prob = proba[:, 1] if is_binary else proba

            metrics = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, average=avg, zero_division=0)),
                "recall": float(recall_score(y, y_pred, average=avg, zero_division=0)),
                "f1": float(f1_score(y, y_pred, average=avg, zero_division=0)),
            }
            if y_prob is not None and is_binary:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
                except Exception:
                    pass

            results_dict = {
                "final_score": swarm_result.score,
                "best_model": type(model).__name__,
                "metrics": metrics,
                "feature_importance": swarm_result.metadata.get("feature_importance", {}),
                "problem_type": swarm_result.metadata.get("problem_type", "Classification"),
                "dataset": kwargs.get("dataset_name", "Dataset"),
            }

            swarm_comp = SwarmMLComprehensive()
            if _manager:
                learning = _manager.get_ml_learning()
                if learning:
                    swarm_comp.set_learning(learning)
                swarm_comp._swarm_manager = _manager

            pdf_path = swarm_comp.generate_world_class_report(
                X=X_processed,
                y=y,
                model=model,
                results=results_dict,
                y_pred=y_pred,
                y_prob=y_prob,
                title=kwargs.get("report_title", "ML Analysis Report"),
                context=context,
                filename=kwargs.get("report_filename", "swarm_report.pdf"),
                include_all=True,
                theme=kwargs.get("report_theme", "professional"),
                generate_html=kwargs.get("generate_html", True),
            )

            swarm_result.metadata["report_path"] = pdf_path
            swarm_result.metadata["metrics"] = metrics

        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

        return swarm_result

    @classmethod
    async def auto_solve(cls, X: Any, y: Any = None, **kwargs: Any) -> TemplateSwarmResult:
        """
        Auto-detect problem type and solve using best template.

        This is the simplest API - just pass your data and Jotty figures out the rest.

        Args:
            X: Input data (DataFrame, array, or path)
            y: Target variable (optional)
            **kwargs: Additional parameters

        Returns:
            TemplateSwarmResult
        """
        # Auto-detect template
        template = TemplateRegistry.auto_detect(X, y)
        logger.info(f"Auto-detected template: {template.name}")

        return await cls.solve(template=template, X=X, y=y, **kwargs)

    @classmethod
    def _progress_callback(cls, stage_name: str, status: str, result: Dict = None) -> Any:
        """Progress callback for visual feedback."""
        if status == "start":
            logger.info(f"  Stage starting: {stage_name}...")
        elif status == "complete":
            metrics = result or {}
            metric_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in list(metrics.items())[:3]
            )
            logger.info(f"  Stage complete: {stage_name} | {metric_str}")

    @classmethod
    def list_templates(cls) -> List[Dict]:
        """List all available templates."""
        return TemplateRegistry.list_templates()

    @classmethod
    def get_template(cls, name: str) -> SwarmTemplate:
        """Get a template by name."""
        return TemplateRegistry.get(name)


# Convenience functions for module-level access
async def solve(template: str = "ml", **kwargs: Any) -> TemplateSwarmResult:
    """Solve using specified template."""
    return await Swarm.solve(template=template, **kwargs)


async def auto_solve(X: Any, y: Any = None, **kwargs: Any) -> TemplateSwarmResult:
    """Auto-detect and solve."""
    return await Swarm.auto_solve(X=X, y=y, **kwargs)


# Make Swarm available at package level
__all__ = ["Swarm", "TemplateSwarmResult", "solve", "auto_solve"]
