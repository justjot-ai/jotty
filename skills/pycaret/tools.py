"""
PyCaret AutoML Skill for Jotty
==============================

Full AutoML pipeline using PyCaret for classification and regression.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


async def pycaret_classify_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    AutoML classification using PyCaret.

    Args:
        params: Dict with keys:
            - data: DataFrame or path to CSV
            - target: Target column name
            - features: Optional list of feature columns
            - exclude_models: Optional list of models to exclude
            - n_select: Number of top models to return (default 5)

    Returns:
        Dict with best models, scores, and comparison results
    """
    from pycaret.classification import setup, compare_models, pull, get_config

    logger.info("[PyCaret] Starting AutoML classification...")

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target')
    features = params.get('features')
    exclude_models = params.get('exclude_models', [])
    n_select = params.get('n_select', 5)

    df = data.copy()
    if features:
        df = df[features + [target]]

    # Setup PyCaret
    setup(
        data=df,
        target=target,
        session_id=42,
        verbose=False,
        html=False,
    )

    # Compare models
    best_models = compare_models(
        exclude=exclude_models if exclude_models else None,
        n_select=n_select,
        sort='Accuracy'
    )

    # Get comparison dataframe
    comparison_df = pull()

    # Extract results
    if isinstance(best_models, list):
        model_names = [type(m).__name__ for m in best_models]
    else:
        model_names = [type(best_models).__name__]
        best_models = [best_models]

    logger.info(f"[PyCaret] Best models: {model_names}")

    return {
        'success': True,
        'best_models': best_models,
        'best_model_name': model_names[0],
        'model_names': model_names,
        'comparison': comparison_df.to_dict(),
        'best_score': float(comparison_df.iloc[0]['Accuracy']),
    }


async def pycaret_regress_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    AutoML regression using PyCaret.

    Args:
        params: Dict with keys:
            - data: DataFrame or path to CSV
            - target: Target column name
            - features: Optional list of feature columns
            - exclude_models: Optional list of models to exclude
            - n_select: Number of top models to return (default 5)

    Returns:
        Dict with best models, scores, and comparison results
    """
    from pycaret.regression import setup, compare_models, pull

    logger.info("[PyCaret] Starting AutoML regression...")

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target')
    features = params.get('features')
    exclude_models = params.get('exclude_models', [])
    n_select = params.get('n_select', 5)

    df = data.copy()
    if features:
        df = df[features + [target]]

    # Setup PyCaret
    setup(
        data=df,
        target=target,
        session_id=42,
        verbose=False,
        html=False,
    )

    # Compare models
    best_models = compare_models(
        exclude=exclude_models if exclude_models else None,
        n_select=n_select,
        sort='R2'
    )

    comparison_df = pull()

    if isinstance(best_models, list):
        model_names = [type(m).__name__ for m in best_models]
    else:
        model_names = [type(best_models).__name__]
        best_models = [best_models]

    logger.info(f"[PyCaret] Best regression models: {model_names}")

    return {
        'success': True,
        'best_models': best_models,
        'best_model_name': model_names[0],
        'model_names': model_names,
        'comparison': comparison_df.to_dict(),
        'best_score': float(comparison_df.iloc[0]['R2']),
    }


async def pycaret_tune_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tune a model using PyCaret's hyperparameter optimization.

    Args:
        params: Dict with keys:
            - model: Model object or model name
            - data: DataFrame
            - target: Target column
            - task: 'classification' or 'regression'
            - n_iter: Number of iterations (default 50)
            - optimize: Metric to optimize

    Returns:
        Dict with tuned model and best parameters
    """
    logger.info("[PyCaret] Tuning model...")

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target')
    task = params.get('task', 'classification')
    n_iter = params.get('n_iter', 50)
    model = params.get('model')
    optimize = params.get('optimize', 'Accuracy' if task == 'classification' else 'R2')

    if task == 'classification':
        from pycaret.classification import setup, create_model, tune_model, pull
    else:
        from pycaret.regression import setup, create_model, tune_model, pull

    setup(
        data=data,
        target=target,
        session_id=42,
        verbose=False,
        html=False,
    )

    # Create or use provided model
    if isinstance(model, str):
        base_model = create_model(model, verbose=False)
    else:
        base_model = model

    # Tune the model
    tuned_model = tune_model(
        base_model,
        n_iter=n_iter,
        optimize=optimize,
        verbose=False
    )

    results = pull()

    logger.info(f"[PyCaret] Model tuned: {type(tuned_model).__name__}")

    return {
        'success': True,
        'tuned_model': tuned_model,
        'model_name': type(tuned_model).__name__,
        'best_params': tuned_model.get_params() if hasattr(tuned_model, 'get_params') else {},
        'tuning_results': results.to_dict() if hasattr(results, 'to_dict') else {},
    }


async def pycaret_ensemble_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create ensemble models using PyCaret.

    Args:
        params: Dict with keys:
            - models: List of model objects
            - data: DataFrame
            - target: Target column
            - task: 'classification' or 'regression'
            - method: 'Bagging', 'Boosting', 'Blending', or 'Stacking'

    Returns:
        Dict with ensemble model and performance
    """
    logger.info("[PyCaret] Creating ensemble...")

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target')
    task = params.get('task', 'classification')
    method = params.get('method', 'Blending')
    models = params.get('models', [])

    if task == 'classification':
        from pycaret.classification import (
            setup, blend_models, stack_models, ensemble_model, pull
        )
    else:
        from pycaret.regression import (
            setup, blend_models, stack_models, ensemble_model, pull
        )

    setup(
        data=data,
        target=target,
        session_id=42,
        verbose=False,
        html=False,
    )

    if method == 'Blending':
        ensemble = blend_models(models) if models else blend_models()
    elif method == 'Stacking':
        ensemble = stack_models(models) if models else stack_models()
    elif method == 'Bagging':
        base = models[0] if models else None
        ensemble = ensemble_model(base, method='Bagging') if base else ensemble_model()
    elif method == 'Boosting':
        base = models[0] if models else None
        ensemble = ensemble_model(base, method='Boosting') if base else ensemble_model()
    else:
        ensemble = blend_models(models) if models else blend_models()

    results = pull()

    logger.info(f"[PyCaret] Ensemble created: {method}")

    return {
        'success': True,
        'ensemble_model': ensemble,
        'method': method,
        'results': results.to_dict() if hasattr(results, 'to_dict') else {},
    }


async def pycaret_predict_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate predictions using a trained PyCaret model.

    Args:
        params: Dict with keys:
            - model: Trained model object
            - data: DataFrame to predict on
            - task: 'classification' or 'regression'

    Returns:
        Dict with predictions
    """
    logger.info("[PyCaret] Generating predictions...")

    model = params.get('model')
    data = params.get('data')
    task = params.get('task', 'classification')

    if isinstance(data, str):
        data = pd.read_csv(data)

    if task == 'classification':
        from pycaret.classification import predict_model
    else:
        from pycaret.regression import predict_model

    predictions_df = predict_model(model, data=data)

    if task == 'classification':
        pred_col = 'prediction_label' if 'prediction_label' in predictions_df.columns else 'Label'
        predictions = predictions_df[pred_col].tolist()
    else:
        pred_col = 'prediction_label' if 'prediction_label' in predictions_df.columns else 'Label'
        predictions = predictions_df[pred_col].tolist()

    logger.info(f"[PyCaret] Generated {len(predictions)} predictions")

    return {
        'success': True,
        'predictions': predictions,
        'predictions_df': predictions_df,
    }
