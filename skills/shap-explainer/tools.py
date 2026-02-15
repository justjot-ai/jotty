"""
SHAP Explainer Skill for Jotty
==============================

Model explainability and interpretation using SHAP values.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

# Status emitter for progress updates
status = SkillStatus("shap-explainer")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def shap_explain_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate SHAP explanations for a model.

    Args:
        params: Dict with keys:
            - model: Trained model object
            - data: DataFrame or array of features
            - feature_names: Optional list of feature names
            - explainer_type: 'tree', 'linear', 'kernel', or 'auto' (default)
            - max_samples: Max samples for background (default 100)

    Returns:
        Dict with SHAP values and feature importance
    """
    status.set_callback(params.pop("_status_callback", None))

    import shap

    logger.info("[SHAP] Generating explanations...")

    model = params.get("model")
    data = params.get("data")
    feature_names = params.get("feature_names")
    explainer_type = params.get("explainer_type", "auto")
    max_samples = params.get("max_samples", 100)

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if feature_names is None:
            feature_names = data.columns.tolist()
        X = data.values
    else:
        X = np.array(data)

    # Select appropriate explainer
    if explainer_type == "tree" or (
        explainer_type == "auto" and hasattr(model, "feature_importances_")
    ):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    elif explainer_type == "linear":
        explainer = shap.LinearExplainer(model, X[:max_samples])
        shap_values = explainer.shap_values(X)
    else:  # kernel or auto fallback
        background = shap.sample(X, min(max_samples, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

    # Handle multi-class output
    if isinstance(shap_values, list):
        # For binary classification, use class 1
        main_shap = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        main_shap = shap_values

    # Calculate mean absolute SHAP for feature importance
    importance = np.abs(main_shap).mean(axis=0)
    if feature_names:
        importance_dict = dict(zip(feature_names, importance.tolist()))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_importance = [(f"feature_{i}", v) for i, v in enumerate(importance)]

    logger.info(f"[SHAP] Explained {len(X)} samples")

    return {
        "success": True,
        "shap_values": main_shap.tolist() if isinstance(main_shap, np.ndarray) else main_shap,
        "expected_value": (
            float(explainer.expected_value[0])
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else float(explainer.expected_value)
        ),
        "feature_importance": dict(sorted_importance),
        "top_features": [f[0] for f in sorted_importance[:10]],
    }


@async_tool_wrapper()
async def shap_importance_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get feature importance based on SHAP values.

    Args:
        params: Dict with keys:
            - model: Trained model object
            - data: DataFrame of features
            - target: Optional target column to exclude
            - top_k: Number of top features to return (default 20)

    Returns:
        Dict with feature importance ranking
    """
    status.set_callback(params.pop("_status_callback", None))

    import shap

    logger.info("[SHAP] Calculating feature importance...")

    model = params.get("model")
    data = params.get("data")
    target = params.get("target")
    top_k = params.get("top_k", 20)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    if target and target in df.columns:
        df = df.drop(columns=[target])

    feature_names = df.columns.tolist()
    X = df.values

    # Use TreeExplainer if possible
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

    # Handle multi-class
    if isinstance(shap_values, list):
        main_shap = (
            shap_values[1]
            if len(shap_values) == 2
            else np.mean([np.abs(sv) for sv in shap_values], axis=0)
        )
    else:
        main_shap = shap_values

    # Mean absolute SHAP
    importance = np.abs(main_shap).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    )

    top_features = importance_df.head(top_k)

    logger.info(f"[SHAP] Top feature: {top_features.iloc[0]['feature']}")

    return {
        "success": True,
        "feature_importance": importance_df.to_dict("records"),
        "top_features": top_features["feature"].tolist(),
        "importance_values": top_features["importance"].tolist(),
    }


@async_tool_wrapper()
async def shap_local_explain_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Explain individual predictions using SHAP.

    Args:
        params: Dict with keys:
            - model: Trained model object
            - data: DataFrame with features
            - instance_idx: Index of instance to explain (or list of indices)
            - feature_names: Optional feature names

    Returns:
        Dict with local explanation for the instance(s)
    """
    status.set_callback(params.pop("_status_callback", None))

    import shap

    logger.info("[SHAP] Generating local explanation...")

    model = params.get("model")
    data = params.get("data")
    instance_idx = params.get("instance_idx", 0)
    feature_names = params.get("feature_names")

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if feature_names is None:
            feature_names = data.columns.tolist()
        X = data.values
    else:
        X = np.array(data)

    # Get SHAP values
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)

    # Handle single or multiple instances
    if isinstance(instance_idx, int):
        instance_idx = [instance_idx]

    instances = X[instance_idx]
    shap_values = explainer.shap_values(instances)

    if isinstance(shap_values, list):
        main_shap = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        main_shap = shap_values

    explanations = []
    for i, idx in enumerate(instance_idx):
        instance_shap = main_shap[i] if len(instance_idx) > 1 else main_shap[0]
        instance_values = instances[i] if len(instance_idx) > 1 else instances[0]

        # Create feature contributions
        contributions = []
        for j, (shap_val, feat_val) in enumerate(zip(instance_shap, instance_values)):
            feat_name = feature_names[j] if feature_names else f"feature_{j}"
            contributions.append(
                {
                    "feature": feat_name,
                    "value": float(feat_val),
                    "shap_value": float(shap_val),
                    "impact": "positive" if shap_val > 0 else "negative",
                }
            )

        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        explanations.append(
            {
                "instance_idx": idx,
                "contributions": contributions[:15],  # Top 15 features
                "base_value": (
                    float(explainer.expected_value[0])
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else float(explainer.expected_value)
                ),
            }
        )

    logger.info(f"[SHAP] Explained {len(explanations)} instance(s)")

    return {
        "success": True,
        "explanations": explanations,
    }


@async_tool_wrapper()
async def shap_interaction_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute SHAP interaction values to understand feature interactions.

    Args:
        params: Dict with keys:
            - model: Trained tree-based model
            - data: DataFrame with features
            - max_samples: Max samples to compute (default 500)

    Returns:
        Dict with interaction values
    """
    status.set_callback(params.pop("_status_callback", None))

    import shap

    logger.info("[SHAP] Computing interaction values...")

    model = params.get("model")
    data = params.get("data")
    max_samples = params.get("max_samples", 500)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    feature_names = df.columns.tolist()
    X = df.values[:max_samples]

    try:
        explainer = shap.TreeExplainer(model)
        interaction_values = explainer.shap_interaction_values(X)
    except Exception as e:
        return {"success": False, "error": f"Interaction values require tree-based model: {str(e)}"}

    if isinstance(interaction_values, list):
        main_interactions = (
            interaction_values[1] if len(interaction_values) == 2 else interaction_values[0]
        )
    else:
        main_interactions = interaction_values

    # Mean absolute interaction values
    mean_interactions = np.abs(main_interactions).mean(axis=0)

    # Find top interactions (off-diagonal)
    top_interactions = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            top_interactions.append(
                {
                    "feature1": feature_names[i],
                    "feature2": feature_names[j],
                    "interaction_strength": float(mean_interactions[i, j]),
                }
            )

    top_interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)

    logger.info(f"[SHAP] Found {len(top_interactions)} interactions")

    return {
        "success": True,
        "top_interactions": top_interactions[:20],
        "interaction_matrix": mean_interactions.tolist(),
        "feature_names": feature_names,
    }
