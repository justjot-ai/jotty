"""
Model Metrics Skill for Jotty
=============================

Comprehensive model evaluation metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

# Status emitter for progress updates
status = SkillStatus("model-metrics")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def metrics_classify_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.

    Args:
        params: Dict with keys:
            - y_true: True labels
            - y_pred: Predicted labels
            - y_proba: Optional predicted probabilities
            - average: 'binary', 'macro', 'micro', 'weighted' (default 'weighted')
            - labels: Optional list of label names

    Returns:
        Dict with all classification metrics
    """
    status.set_callback(params.pop("_status_callback", None))

    logger.info("[Metrics] Calculating classification metrics...")

    y_true = np.array(params.get("y_true"))
    y_pred = np.array(params.get("y_pred"))
    y_proba = params.get("y_proba")
    average = params.get("average", "weighted")
    labels = params.get("labels")

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["per_class"] = report

    # AUC metrics if probabilities provided
    if y_proba is not None:
        y_proba = np.array(y_proba)
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_proba.ndim == 2:
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba_pos))
                metrics["average_precision"] = float(average_precision_score(y_true, y_proba_pos))
                metrics["log_loss"] = float(log_loss(y_true, y_proba_pos))
            else:
                # Multi-class
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception as e:
            logger.warning(f"Could not compute AUC metrics: {e}")

    # Additional derived metrics
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    if cm.shape == (2, 2):
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        metrics["balanced_accuracy"] = float((metrics["recall"] + metrics["specificity"]) / 2)
        metrics["mcc"] = float(
            (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-10)
        )

    logger.info(f"[Metrics] Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

    return {
        "success": True,
        "metrics": metrics,
        "summary": {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc": metrics.get("roc_auc", None),
        },
    }


@async_tool_wrapper()
async def metrics_regress_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive regression metrics.

    Args:
        params: Dict with keys:
            - y_true: True values
            - y_pred: Predicted values

    Returns:
        Dict with all regression metrics
    """
    status.set_callback(params.pop("_status_callback", None))

    logger.info("[Metrics] Calculating regression metrics...")

    y_true = np.array(params.get("y_true"))
    y_pred = np.array(params.get("y_pred"))

    metrics = {}

    # Basic metrics
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))
    metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))

    # MAPE (handle zeros)
    try:
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mask = y_true != 0
        if mask.any():
            metrics["mape"] = float(
                np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            )
        else:
            metrics["mape"] = None

    # Additional metrics
    residuals = y_true - y_pred
    metrics["residual_mean"] = float(np.mean(residuals))
    metrics["residual_std"] = float(np.std(residuals))
    metrics["max_error"] = float(np.max(np.abs(residuals)))

    # Correlation
    metrics["correlation"] = float(np.corrcoef(y_true, y_pred)[0, 1])

    logger.info(f"[Metrics] RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

    return {
        "success": True,
        "metrics": metrics,
        "summary": {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        },
    }


@async_tool_wrapper()
async def metrics_crossval_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform cross-validation and return metrics.

    Args:
        params: Dict with keys:
            - model: Model to evaluate
            - data: DataFrame with features
            - target: Target column name
            - cv: Number of folds (default 5)
            - scoring: Scoring metric (default depends on task)
            - task: 'classification' or 'regression'

    Returns:
        Dict with cross-validation scores
    """
    status.set_callback(params.pop("_status_callback", None))

    logger.info("[Metrics] Running cross-validation...")

    model = params.get("model")
    data = params.get("data")
    target = params.get("target")
    cv = params.get("cv", 5)
    scoring = params.get("scoring")
    task = params.get("task", "classification")

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(X.median())

    # Default scoring
    if scoring is None:
        scoring = "accuracy" if task == "classification" else "r2"

    # CV strategy
    if task == "classification":
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Multiple scoring metrics
    if task == "classification":
        scorings = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    else:
        scorings = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

    results = {}
    for sc in scorings:
        try:
            scores = cross_val_score(model, X, y, cv=cv_split, scoring=sc)
            results[sc] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores.tolist(),
            }
        except Exception as e:
            results[sc] = {"error": str(e)}

    # Primary score
    primary_scores = cross_val_score(model, X, y, cv=cv_split, scoring=scoring)

    logger.info(
        f"[Metrics] CV {scoring}: {np.mean(primary_scores):.4f} +/- {np.std(primary_scores):.4f}"
    )

    return {
        "success": True,
        "primary_score": float(np.mean(primary_scores)),
        "primary_std": float(np.std(primary_scores)),
        "all_metrics": results,
        "cv_folds": cv,
        "scoring": scoring,
    }


@async_tool_wrapper()
async def metrics_calibration_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze and improve probability calibration.

    Args:
        params: Dict with keys:
            - y_true: True labels
            - y_proba: Predicted probabilities
            - n_bins: Number of bins for calibration curve (default 10)
            - method: Calibration method if recalibrating ('isotonic', 'sigmoid')

    Returns:
        Dict with calibration metrics and curve
    """
    status.set_callback(params.pop("_status_callback", None))

    from sklearn.calibration import calibration_curve

    logger.info("[Metrics] Analyzing probability calibration...")

    y_true = np.array(params.get("y_true"))
    y_proba = np.array(params.get("y_proba"))
    n_bins = params.get("n_bins", 10)

    # Handle 2D probability array
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if bin_mask.sum() > 0:
            bin_acc = y_true[bin_mask].mean()
            bin_conf = y_proba[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(y_proba)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    # Maximum Calibration Error
    mce = float(np.max(np.abs(prob_true - prob_pred)))

    # Brier score
    brier = float(np.mean((y_proba - y_true) ** 2))

    # Reliability metrics
    over_confident = float(np.mean(y_proba > y_true.mean()))
    under_confident = 1 - over_confident

    calibration_curve_data = {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }

    logger.info(f"[Metrics] ECE: {ece:.4f}, Brier: {brier:.4f}")

    return {
        "success": True,
        "ece": float(ece),
        "mce": mce,
        "brier_score": brier,
        "calibration_curve": calibration_curve_data,
        "is_well_calibrated": ece < 0.1,
        "over_confident_ratio": over_confident,
    }


@async_tool_wrapper()
async def metrics_threshold_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find optimal classification threshold.

    Args:
        params: Dict with keys:
            - y_true: True labels
            - y_proba: Predicted probabilities
            - metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            - min_precision: Minimum precision constraint
            - min_recall: Minimum recall constraint

    Returns:
        Dict with optimal threshold and metrics at that threshold
    """
    status.set_callback(params.pop("_status_callback", None))

    logger.info("[Metrics] Finding optimal threshold...")

    y_true = np.array(params.get("y_true"))
    y_proba = np.array(params.get("y_proba"))
    metric = params.get("metric", "f1")
    min_precision = params.get("min_precision")
    min_recall = params.get("min_recall")

    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]

    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0

    results = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        # Apply constraints
        if min_precision and prec < min_precision:
            continue
        if min_recall and rec < min_recall:
            continue

        score = {"f1": f1, "precision": prec, "recall": rec, "accuracy": acc}.get(metric, f1)

        results.append(
            {
                "threshold": float(thresh),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "accuracy": float(acc),
            }
        )

        if score > best_score:
            best_score = score
            best_threshold = thresh

    # Get metrics at optimal threshold
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    optimal_metrics = {
        "threshold": float(best_threshold),
        "precision": float(precision_score(y_true, y_pred_opt, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_opt, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_opt, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred_opt)),
    }

    logger.info(f"[Metrics] Optimal threshold: {best_threshold:.2f}")

    return {
        "success": True,
        "optimal_threshold": float(best_threshold),
        "optimal_metrics": optimal_metrics,
        "all_thresholds": results[::5],  # Every 5th for brevity
        "optimized_metric": metric,
    }
