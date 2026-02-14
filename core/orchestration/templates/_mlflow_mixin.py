"""MLflow Tracking Mixin - Experiment tracking and model logging."""
from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from Jotty.core.orchestration.templates.swarm_ml_comprehensive import MLflowConfig

logger = logging.getLogger(__name__)


class MLflowMixin:
    def init_mlflow(self, config: "MLflowConfig" = None) -> None:
        """
        Initialize MLflow tracking for this swarm execution.

        Args:
            config: MLflow configuration (uses default if None)
        """
        from Jotty.core.orchestration.templates.swarm_ml_comprehensive import MLflowConfig
        self._mlflow_config = config or MLflowConfig()
        self._mlflow_run = None
        self._mlflow_available = False

        if not self._mlflow_config.enabled:
            return

        try:
            import mlflow
            self._mlflow = mlflow

            # Set tracking URI
            mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)

            # Set or create experiment
            mlflow.set_experiment(self._mlflow_config.experiment_name)

            # Start a new run
            run_name = f"{self._mlflow_config.run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._mlflow_run = mlflow.start_run(run_name=run_name)

            # Log default tags
            tags = {
                "swarm_template": self.name,
                "version": self.version,
                **self._mlflow_config.tags
            }
            mlflow.set_tags(tags)

            self._mlflow_available = True
            logger.info(f"MLflow initialized: experiment={self._mlflow_config.experiment_name}, run={run_name}")

        except ImportError:
            logger.warning("MLflow not installed. Install with: pip install mlflow")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            # MLflow params must be strings or numbers
            flat_params = self._flatten_dict(params)
            # Truncate long values
            for k, v in flat_params.items():
                if isinstance(v, str) and len(v) > 250:
                    flat_params[k] = v[:250] + "..."
            self._mlflow.log_params(flat_params)
        except Exception as e:
            logger.debug(f"Failed to log params: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log metrics to MLflow."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self._mlflow.log_metric(name, value, step=step)
        except Exception as e:
            logger.debug(f"Failed to log metrics: {e}")

    def log_model(self, model: Any, artifact_path: str = 'model', input_example: Any = None) -> None:
        """Log trained model to MLflow."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        if not self._mlflow_config.log_models:
            return

        try:
            # Try sklearn first (most common)
            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    input_example=input_example,
                    registered_model_name=self._mlflow_config.registered_model_name
                )
                return
            except Exception:
                pass

            # Try xgboost
            try:
                import mlflow.xgboost
                mlflow.xgboost.log_model(model, artifact_path)
                return
            except Exception:
                pass

            # Try lightgbm
            try:
                import mlflow.lightgbm
                mlflow.lightgbm.log_model(model, artifact_path)
                return
            except Exception:
                pass

            # Fallback to pickle
            import pickle
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
                pickle.dump(model, f)
                self._mlflow.log_artifact(f.name, artifact_path)

        except Exception as e:
            logger.debug(f"Failed to log model: {e}")

    def log_feature_importance(self, importance: Dict[str, float], top_n: int = 30) -> None:
        """Log feature importance as artifact and metrics."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        if not self._mlflow_config.log_feature_importance:
            return

        try:
            # Sort by importance
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

            # Log top features as metrics
            for rank, (feat, imp) in enumerate(sorted_imp[:10], 1):
                safe_name = feat.replace(" ", "_")[:50]
                self._mlflow.log_metric(f"feat_imp_{rank}_{safe_name}", imp)

            # Create and log importance plot
            if self._mlflow_config.log_artifacts:
                self._log_importance_plot(sorted_imp)

        except Exception as e:
            logger.debug(f"Failed to log feature importance: {e}")

    def log_shap_values(self, shap_values: Any, feature_names: List[str], X_sample: Any = None) -> None:
        """Log SHAP values and summary plot."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        if not self._mlflow_config.log_shap_values:
            return

        try:
            import shap
            import matplotlib.pyplot as plt
            import tempfile

            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                self._mlflow.log_artifact(f.name, "shap")
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to log SHAP values: {e}")

    def log_confusion_matrix(self, y_true: Any, y_pred: Any, labels: Any = None) -> None:
        """Log confusion matrix as artifact."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt
            import tempfile

            cm = confusion_matrix(y_true, y_pred, labels=labels)

            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap='Blues', ax=plt.gca())
            plt.title('Confusion Matrix')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                self._mlflow.log_artifact(f.name, "metrics")
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to log confusion matrix: {e}")

    def log_roc_curve(self, y_true: Any, y_prob: Any, pos_label: Any = 1) -> None:
        """Log ROC curve as artifact."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            from sklearn.metrics import roc_curve, auc
            import matplotlib.pyplot as plt
            import tempfile

            fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                self._mlflow.log_artifact(f.name, "metrics")
            plt.close()

            # Log AUC as metric
            self._mlflow.log_metric("roc_auc", roc_auc)

        except Exception as e:
            logger.debug(f"Failed to log ROC curve: {e}")

    def end_mlflow_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        if not self._mlflow_available or not self._mlflow_run:
            return

        try:
            self._mlflow.end_run(status=status)
            logger.info(f"MLflow run ended with status: {status}")
        except Exception as e:
            logger.debug(f"Failed to end MLflow run: {e}")

    def _log_importance_plot(self, sorted_importance: List[Tuple[str, float]]) -> Any:
        """Create and log feature importance plot."""
        try:
            import matplotlib.pyplot as plt
            import tempfile

            features = [x[0] for x in sorted_importance]
            values = [x[1] for x in sorted_importance]

            plt.figure(figsize=(10, max(6, len(features) * 0.3)))
            plt.barh(range(len(features)), values[::-1], color='steelblue')
            plt.yticks(range(len(features)), features[::-1])
            plt.xlabel('Importance')
            plt.title('Feature Importance (Top Features)')
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
                plt.savefig(f.name, dpi=150, bbox_inches='tight')
                self._mlflow.log_artifact(f.name, "feature_importance")
            plt.close()

        except Exception as e:
            logger.debug(f"Failed to create importance plot: {e}")

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # =========================================================================
    # PDF REPORT GENERATION
    # =========================================================================

