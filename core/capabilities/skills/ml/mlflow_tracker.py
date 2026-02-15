"""
MLflow Tracking Skill
=====================

Integrates MLflow for experiment tracking, model logging, and model registry.

Features:
1. Automatic experiment tracking
2. Model versioning and registry
3. Feature importance logging
4. Artifact storage (plots, data profiles)
5. Model comparison and retrieval

Usage:
    tracker = MLflowTrackerSkill()
    await tracker.init()

    # Start tracking
    await tracker.start_run("titanic_experiment")

    # Log parameters, metrics, model
    await tracker.log_params({"n_estimators": 100})
    await tracker.log_metrics({"accuracy": 0.85})
    await tracker.log_model(model, "best_model")

    # End and get run info
    result = await tracker.end_run()
"""

import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import os

from .base import MLSkill, SkillResult, SkillCategory

logger = logging.getLogger(__name__)


class MLflowTrackerSkill(MLSkill):
    """
    MLflow tracking skill for experiment management.

    Provides seamless MLflow integration for:
    - Experiment tracking
    - Model logging and versioning
    - Metrics and parameters logging
    - Artifact storage
    - Model registry
    """

    name = "mlflow_tracker"
    version = "1.0.0"
    description = "MLflow experiment tracking and model registry"
    category = SkillCategory.EVALUATION

    required_inputs = []
    optional_inputs = ["experiment_name", "tracking_uri"]
    outputs = ["run_id", "experiment_id", "artifact_uri"]

    # Default MLflow settings
    DEFAULT_TRACKING_URI = "mlruns"
    DEFAULT_EXPERIMENT = "jotty_ml"

    def __init__(self, config: Dict[str, Any] = None) -> None:
        super().__init__(config)
        self._mlflow = None
        self._active_run = None
        self._tracking_uri = None
        self._experiment_name = None

    async def init(self, tracking_uri: str = None, experiment_name: str = None) -> Any:
        """
        Initialize MLflow tracking.

        Args:
            tracking_uri: MLflow tracking server URI or local path
            experiment_name: Default experiment name
        """
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
            self._mlflow = None
            self._initialized = True
            return

        # Set tracking URI
        self._tracking_uri = tracking_uri or self.config.get(
            'tracking_uri',
            os.environ.get('MLFLOW_TRACKING_URI', self.DEFAULT_TRACKING_URI)
        )
        self._mlflow.set_tracking_uri(self._tracking_uri)

        # Set default experiment
        self._experiment_name = experiment_name or self.config.get(
            'experiment_name',
            os.environ.get('MLFLOW_EXPERIMENT_NAME', self.DEFAULT_EXPERIMENT)
        )
        self._mlflow.set_experiment(self._experiment_name)

        self._initialized = True
        logger.info(f"MLflow initialized: {self._tracking_uri}")

    async def execute(self, X: pd.DataFrame = None, y: Optional[pd.Series] = None, **context: Any) -> SkillResult:
        """
        Execute tracking operation based on context.

        Args:
            X: Optional features for logging data profile
            y: Optional target for logging distribution
            **context: Operation parameters

        Returns:
            SkillResult with tracking info
        """
        start_time = time.time()

        if self._mlflow is None:
            return self._create_error_result("MLflow not available")

        operation = context.get('operation', 'status')

        if operation == 'start_run':
            return await self._start_run_operation(context)
        elif operation == 'log_params':
            return await self._log_params_operation(context)
        elif operation == 'log_metrics':
            return await self._log_metrics_operation(context)
        elif operation == 'log_model':
            return await self._log_model_operation(context)
        elif operation == 'log_artifact':
            return await self._log_artifact_operation(context)
        elif operation == 'end_run':
            return await self._end_run_operation()
        elif operation == 'load_model':
            return await self._load_model_operation(context)
        elif operation == 'list_runs':
            return await self._list_runs_operation(context)
        else:
            return await self._get_status()

    async def start_run(self,
                        run_name: str = None,
                        experiment_name: str = None,
                        tags: Dict[str, str] = None,
                        nested: bool = False) -> Optional[str]:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for the run
            experiment_name: Override default experiment
            tags: Run tags
            nested: Allow nested runs

        Returns:
            Run ID if successful
        """
        if self._mlflow is None:
            logger.warning("MLflow not available - tracking disabled")
            return None

        if experiment_name:
            self._mlflow.set_experiment(experiment_name)

        self._active_run = self._mlflow.start_run(
            run_name=run_name,
            tags=tags,
            nested=nested
        )

        run_id = self._active_run.info.run_id
        logger.info(f"Started MLflow run: {run_id}")
        return run_id

    async def log_params(self, params: Dict[str, Any]) -> bool:
        """
        Log parameters to the active run.

        Args:
            params: Dictionary of parameters

        Returns:
            True if successful
        """
        if self._mlflow is None or self._active_run is None:
            return False

        # MLflow params must be strings, truncate if too long
        clean_params = {}
        for k, v in params.items():
            str_v = str(v)
            if len(str_v) > 500:
                str_v = str_v[:497] + "..."
            clean_params[k] = str_v

        self._mlflow.log_params(clean_params)
        return True

    async def log_metrics(self,
                          metrics: Dict[str, float],
                          step: int = None) -> bool:
        """
        Log metrics to the active run.

        Args:
            metrics: Dictionary of metrics (must be numeric)
            step: Optional step number for time-series metrics

        Returns:
            True if successful
        """
        if self._mlflow is None or self._active_run is None:
            return False

        # Filter to numeric values only
        numeric_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.number)) and not np.isnan(v)
        }

        if step is not None:
            for k, v in numeric_metrics.items():
                self._mlflow.log_metric(k, v, step=step)
        else:
            self._mlflow.log_metrics(numeric_metrics)

        return True

    async def log_model(self,
                        model: Any,
                        model_name: str = "model",
                        registered_name: str = None,
                        signature: Any = None,
                        input_example: Any = None) -> Optional[str]:
        """
        Log a model to MLflow.

        Args:
            model: Trained model object
            model_name: Artifact path name
            registered_name: Name for model registry (optional)
            signature: Model signature (auto-inferred if not provided)
            input_example: Example input for signature inference

        Returns:
            Model URI if successful
        """
        if self._mlflow is None or self._active_run is None:
            return None

        # Determine model type and log appropriately
        model_type = type(model).__name__
        model_module = type(model).__module__

        try:
            if 'sklearn' in model_module or 'lightgbm' in model_module or 'xgboost' in model_module:
                # Use sklearn flavor for most models
                from mlflow import sklearn as mlflow_sklearn

                model_info = mlflow_sklearn.log_model(
                    model,
                    artifact_path=model_name,  # Use artifact_path explicitly
                    registered_model_name=registered_name,
                    signature=signature,
                    input_example=input_example
                )

            elif 'catboost' in model_module:
                # CatBoost has its own flavor
                try:
                    from mlflow import catboost as mlflow_catboost
                    model_info = mlflow_catboost.log_model(
                        model,
                        artifact_path=model_name,
                        registered_model_name=registered_name
                    )
                except ImportError:
                    # Fallback to sklearn flavor
                    from mlflow import sklearn as mlflow_sklearn
                    model_info = mlflow_sklearn.log_model(
                        model,
                        artifact_path=model_name,
                        registered_model_name=registered_name
                    )

            elif 'tensorflow' in model_module or 'keras' in model_module:
                from mlflow import tensorflow as mlflow_tf
                model_info = mlflow_tf.log_model(
                    model,
                    artifact_path=model_name,
                    registered_model_name=registered_name
                )

            elif 'torch' in model_module:
                from mlflow import pytorch as mlflow_pytorch
                model_info = mlflow_pytorch.log_model(
                    model,
                    artifact_path=model_name,
                    registered_model_name=registered_name
                )

            else:
                # Generic pickle-based logging
                from mlflow import sklearn as mlflow_sklearn
                model_info = mlflow_sklearn.log_model(
                    model,
                    artifact_path=model_name,
                    registered_model_name=registered_name
                )

            logger.info(f"Logged model: {model_name} ({model_type})")
            return model_info.model_uri

        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            return None

    async def log_feature_importance(self,
                                     importance: Dict[str, float],
                                     top_n: int = 20) -> bool:
        """
        Log feature importance as metrics and artifact.

        Args:
            importance: Feature name -> importance mapping
            top_n: Number of top features to log as metrics

        Returns:
            True if successful
        """
        if self._mlflow is None or self._active_run is None:
            return False

        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: -abs(x[1]))

        # Log top N as metrics
        top_metrics = {
            f"feature_importance_{i+1}_{name}": float(value)
            for i, (name, value) in enumerate(sorted_importance[:top_n])
        }
        await self.log_metrics(top_metrics)

        # Log full importance as artifact
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dict(sorted_importance), f, indent=2)
            temp_path = f.name

        self._mlflow.log_artifact(temp_path, "feature_importance")
        os.unlink(temp_path)

        return True

    async def log_artifact(self,
                           local_path: str,
                           artifact_path: str = None) -> bool:
        """
        Log a file or directory as an artifact.

        Args:
            local_path: Path to local file/directory
            artifact_path: Destination path in artifact store

        Returns:
            True if successful
        """
        if self._mlflow is None or self._active_run is None:
            return False

        if os.path.isdir(local_path):
            self._mlflow.log_artifacts(local_path, artifact_path)
        else:
            self._mlflow.log_artifact(local_path, artifact_path)

        return True

    async def log_dataframe(self,
                            df: pd.DataFrame,
                            name: str = "data",
                            format: str = "parquet") -> bool:
        """
        Log a DataFrame as an artifact.

        Args:
            df: DataFrame to log
            name: Name for the artifact
            format: File format (parquet, csv, json)

        Returns:
            True if successful
        """
        if self._mlflow is None or self._active_run is None:
            return False

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            if format == "parquet":
                path = os.path.join(tmpdir, f"{name}.parquet")
                df.to_parquet(path)
            elif format == "csv":
                path = os.path.join(tmpdir, f"{name}.csv")
                df.to_csv(path, index=False)
            else:
                path = os.path.join(tmpdir, f"{name}.json")
                df.to_json(path, orient='records')

            self._mlflow.log_artifact(path)

        return True

    async def set_tag(self, key: str, value: str) -> bool:
        """Set a tag on the active run."""
        if self._mlflow is None or self._active_run is None:
            return False
        self._mlflow.set_tag(key, value)
        return True

    async def set_tags(self, tags: Dict[str, str]) -> bool:
        """Set multiple tags on the active run."""
        if self._mlflow is None or self._active_run is None:
            return False
        self._mlflow.set_tags(tags)
        return True

    async def end_run(self, status: str = "FINISHED") -> Optional[Dict[str, Any]]:
        """
        End the active MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)

        Returns:
            Run info dictionary
        """
        if self._mlflow is None or self._active_run is None:
            return None

        run_id = self._active_run.info.run_id
        self._mlflow.end_run(status=status)

        # Get run info
        run = self._mlflow.get_run(run_id)

        result = {
            'run_id': run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'artifact_uri': run.info.artifact_uri,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'metrics': run.data.metrics,
            'params': run.data.params,
        }

        self._active_run = None
        logger.info(f"Ended MLflow run: {run_id}")
        return result

    async def load_model(self,
                         model_uri: str = None,
                         run_id: str = None,
                         model_name: str = "model") -> Optional[Any]:
        """
        Load a model from MLflow.

        Args:
            model_uri: Full model URI (e.g., "runs:/run_id/model")
            run_id: Run ID to load from (uses model_name as artifact path)
            model_name: Model artifact name (default: "model")

        Returns:
            Loaded model object
        """
        if self._mlflow is None:
            return None

        try:
            if model_uri:
                uri = model_uri
            elif run_id:
                uri = f"runs:/{run_id}/{model_name}"
            else:
                return None

            # Try sklearn flavor first (most common)
            try:
                from mlflow import sklearn as mlflow_sklearn
                return mlflow_sklearn.load_model(uri)
            except Exception:
                pass

            # Try pyfunc as fallback
            return self._mlflow.pyfunc.load_model(uri)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    async def list_runs(self,
                        experiment_name: str = None,
                        max_results: int = 100,
                        filter_string: str = None,
                        order_by: List[str] = None) -> List[Dict[str, Any]]:
        """
        List runs in an experiment.

        Args:
            experiment_name: Experiment name (uses default if not specified)
            max_results: Maximum number of runs to return
            filter_string: MLflow search filter
            order_by: List of columns to order by

        Returns:
            List of run info dictionaries
        """
        if self._mlflow is None:
            return []

        exp_name = experiment_name or self._experiment_name
        experiment = self._mlflow.get_experiment_by_name(exp_name)

        if not experiment:
            return []

        runs = self._mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            filter_string=filter_string,
            order_by=order_by or ["metrics.accuracy DESC"]
        )

        return runs.to_dict('records') if not runs.empty else []

    async def get_best_run(self,
                           experiment_name: str = None,
                           metric: str = "accuracy",
                           ascending: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the best run from an experiment.

        Args:
            experiment_name: Experiment name
            metric: Metric to optimize
            ascending: Sort ascending (True) or descending (False)

        Returns:
            Best run info
        """
        order = "ASC" if ascending else "DESC"
        runs = await self.list_runs(
            experiment_name=experiment_name,
            max_results=1,
            order_by=[f"metrics.{metric} {order}"]
        )
        return runs[0] if runs else None

    async def compare_runs(self,
                           run_ids: List[str],
                           metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare

        Returns:
            DataFrame with run comparisons
        """
        if self._mlflow is None:
            return pd.DataFrame()

        comparison = []
        for run_id in run_ids:
            try:
                run = self._mlflow.get_run(run_id)
                row = {'run_id': run_id, **run.data.metrics, **run.data.params}
                comparison.append(row)
            except Exception:
                continue

        df = pd.DataFrame(comparison)

        if metrics and not df.empty:
            cols = ['run_id'] + [m for m in metrics if m in df.columns]
            df = df[cols]

        return df

    # Internal operation handlers
    async def _start_run_operation(self, context: Dict) -> SkillResult:
        run_id = await self.start_run(
            run_name=context.get('run_name'),
            experiment_name=context.get('experiment_name'),
            tags=context.get('tags')
        )
        return self._create_result(
            success=run_id is not None,
            data=run_id,
            metadata={'run_id': run_id}
        )

    async def _log_params_operation(self, context: Dict) -> SkillResult:
        success = await self.log_params(context.get('params', {}))
        return self._create_result(success=success)

    async def _log_metrics_operation(self, context: Dict) -> SkillResult:
        success = await self.log_metrics(
            context.get('metrics', {}),
            step=context.get('step')
        )
        return self._create_result(success=success)

    async def _log_model_operation(self, context: Dict) -> SkillResult:
        model_uri = await self.log_model(
            model=context.get('model'),
            model_name=context.get('model_name', 'model'),
            registered_name=context.get('registered_name')
        )
        return self._create_result(
            success=model_uri is not None,
            data=model_uri
        )

    async def _log_artifact_operation(self, context: Dict) -> SkillResult:
        success = await self.log_artifact(
            local_path=context.get('local_path'),
            artifact_path=context.get('artifact_path')
        )
        return self._create_result(success=success)

    async def _end_run_operation(self) -> SkillResult:
        run_info = await self.end_run()
        return self._create_result(
            success=run_info is not None,
            data=run_info,
            metrics=run_info.get('metrics', {}) if run_info else {}
        )

    async def _load_model_operation(self, context: Dict) -> SkillResult:
        model = await self.load_model(
            model_uri=context.get('model_uri'),
            run_id=context.get('run_id'),
            model_name=context.get('model_name', 'model')
        )
        return self._create_result(
            success=model is not None,
            data=model
        )

    async def _list_runs_operation(self, context: Dict) -> SkillResult:
        runs = await self.list_runs(
            experiment_name=context.get('experiment_name'),
            max_results=context.get('max_results', 100)
        )
        return self._create_result(
            success=True,
            data=runs,
            metrics={'n_runs': len(runs)}
        )

    async def _get_status(self) -> SkillResult:
        return self._create_result(
            success=True,
            data={
                'mlflow_available': self._mlflow is not None,
                'tracking_uri': self._tracking_uri,
                'experiment_name': self._experiment_name,
                'active_run': self._active_run.info.run_id if self._active_run else None
            }
        )
