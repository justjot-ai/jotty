"""
Auto-sklearn Skill for Jotty
============================

AutoML using auto-sklearn with automated model selection and ensembling.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("auto-sklearn")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def autosklearn_classify_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    AutoML classification using auto-sklearn.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - target: Target column name
            - time_limit: Total time budget in seconds (default 300)
            - per_run_time_limit: Per model time limit (default 60)
            - ensemble_size: Size of ensemble (default 50)
            - memory_limit: Memory limit in MB (default 3072)

    Returns:
        Dict with best models, ensemble info, and performance
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        import autosklearn.classification
    except ImportError:
        return {
            'success': False,
            'error': 'auto-sklearn not installed. Install with: pip install auto-sklearn'
        }

    logger.info("[AutoSklearn] Starting AutoML classification...")

    data = params.get('data')
    target = params.get('target')
    time_limit = params.get('time_limit', 300)
    per_run_time_limit = params.get('per_run_time_limit', 60)
    ensemble_size = params.get('ensemble_size', 50)
    memory_limit = params.get('memory_limit', 3072)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(X.median())

    # Create AutoSklearn classifier
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time_limit,
        per_run_time_limit=per_run_time_limit,
        ensemble_size=ensemble_size,
        memory_limit=memory_limit,
        n_jobs=-1,
        seed=42,
    )

    # Fit
    automl.fit(X, y)

    # Get results
    score = automl.score(X, y)

    # Get ensemble composition
    ensemble_info = []
    for model_id, weight in automl.get_models_with_weights():
        ensemble_info.append({
            'model_id': model_id,
            'weight': float(weight),
        })

    # Get leaderboard
    leaderboard = automl.leaderboard()

    logger.info(f"[AutoSklearn] Best score: {score:.4f}")

    return {
        'success': True,
        'model': automl,
        'score': float(score),
        'ensemble_info': ensemble_info[:10],
        'leaderboard': leaderboard.head(10).to_dict() if hasattr(leaderboard, 'to_dict') else {},
        'n_models_evaluated': len(automl.cv_results_['mean_test_score']) if hasattr(automl, 'cv_results_') else 0,
    }


@async_tool_wrapper()
async def autosklearn_regress_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    AutoML regression using auto-sklearn.

    Args:
        params: Dict with keys:
            - data: DataFrame with features
            - target: Target column name
            - time_limit: Total time budget in seconds (default 300)
            - per_run_time_limit: Per model time limit (default 60)
            - ensemble_size: Size of ensemble (default 50)

    Returns:
        Dict with best models and performance
    """
    status.set_callback(params.pop('_status_callback', None))

    try:
        import autosklearn.regression
    except ImportError:
        return {
            'success': False,
            'error': 'auto-sklearn not installed. Install with: pip install auto-sklearn'
        }

    logger.info("[AutoSklearn] Starting AutoML regression...")

    data = params.get('data')
    target = params.get('target')
    time_limit = params.get('time_limit', 300)
    per_run_time_limit = params.get('per_run_time_limit', 60)
    ensemble_size = params.get('ensemble_size', 50)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(X.median())

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=time_limit,
        per_run_time_limit=per_run_time_limit,
        ensemble_size=ensemble_size,
        n_jobs=-1,
        seed=42,
    )

    automl.fit(X, y)
    score = automl.score(X, y)  # R2 score

    ensemble_info = []
    for model_id, weight in automl.get_models_with_weights():
        ensemble_info.append({
            'model_id': model_id,
            'weight': float(weight),
        })

    logger.info(f"[AutoSklearn] Best R2: {score:.4f}")

    return {
        'success': True,
        'model': automl,
        'r2_score': float(score),
        'ensemble_info': ensemble_info[:10],
    }


@async_tool_wrapper()
async def autosklearn_ensemble_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed ensemble information from trained auto-sklearn model.

    Args:
        params: Dict with keys:
            - model: Trained auto-sklearn model

    Returns:
        Dict with ensemble composition and statistics
    """
    status.set_callback(params.pop('_status_callback', None))

    logger.info("[AutoSklearn] Extracting ensemble info...")

    model = params.get('model')

    if model is None:
        return {'success': False, 'error': 'No model provided'}

    try:
        ensemble_info = []
        total_weight = 0

        for model_id, weight in model.get_models_with_weights():
            ensemble_info.append({
                'model_id': model_id,
                'weight': float(weight),
            })
            total_weight += weight

        # Get CV results if available
        cv_results = {}
        if hasattr(model, 'cv_results_'):
            cv_results = {
                'n_models': len(model.cv_results_['mean_test_score']),
                'best_score': float(max(model.cv_results_['mean_test_score'])),
                'mean_score': float(np.mean(model.cv_results_['mean_test_score'])),
            }

        logger.info(f"[AutoSklearn] Ensemble has {len(ensemble_info)} models")

        return {
            'success': True,
            'ensemble_models': ensemble_info,
            'total_weight': float(total_weight),
            'n_ensemble_members': len(ensemble_info),
            'cv_results': cv_results,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@async_tool_wrapper()
async def autosklearn_predict_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate predictions using trained auto-sklearn model.

    Args:
        params: Dict with keys:
            - model: Trained auto-sklearn model
            - data: DataFrame to predict on

    Returns:
        Dict with predictions
    """
    status.set_callback(params.pop('_status_callback', None))

    logger.info("[AutoSklearn] Generating predictions...")

    model = params.get('model')
    data = params.get('data')

    if model is None:
        return {'success': False, 'error': 'No model provided'}

    if isinstance(data, str):
        data = pd.read_csv(data)

    X = data.copy()
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(X.median())

    predictions = model.predict(X)

    # Get probabilities if classification
    proba = None
    try:
        proba = model.predict_proba(X)
        proba = proba.tolist()
    except Exception:
        pass

    logger.info(f"[AutoSklearn] Generated {len(predictions)} predictions")

    return {
        'success': True,
        'predictions': predictions.tolist(),
        'probabilities': proba,
    }
