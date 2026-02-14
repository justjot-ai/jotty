"""
Ensemble Builder Skill for Jotty
================================

Advanced model ensembling techniques.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("ensemble-builder")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def ensemble_stack_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a stacking ensemble with meta-learner.

    Args:
        params: Dict with keys:
            - models: List of (name, model) tuples or dict
            - data: DataFrame with features
            - target: Target column name
            - meta_learner: Meta-learner model (default: LogisticRegression)
            - task: 'classification' or 'regression'
            - cv_folds: Number of CV folds for OOF predictions (default 5)

    Returns:
        Dict with stacking ensemble and performance
    """
    status.set_callback(params.pop('_status_callback', None))

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import StackingClassifier, StackingRegressor

    logger.info("[Ensemble] Creating stacking ensemble...")

    models = params.get('models', {})
    data = params.get('data')
    target = params.get('target')
    meta_learner = params.get('meta_learner')
    task = params.get('task', 'classification')
    cv_folds = params.get('cv_folds', 5)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(X.median())

    # Prepare models
    if isinstance(models, dict):
        estimators = [(name, model) for name, model in models.items()]
    else:
        estimators = models

    if not estimators:
        return {'success': False, 'error': 'No models provided'}

    # Set default meta-learner
    if meta_learner is None:
        meta_learner = LogisticRegression() if task == 'classification' else Ridge()

    # Create stacking ensemble
    if task == 'classification':
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv_folds,
            passthrough=False
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv_folds,
            passthrough=False
        )
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Fit and evaluate
    ensemble.fit(X, y)

    # Cross-validation score
    cv_preds = cross_val_predict(ensemble, X, y, cv=cv)

    if task == 'classification':
        score = accuracy_score(y, cv_preds)
        f1 = f1_score(y, cv_preds, average='weighted')
        metrics = {'accuracy': score, 'f1': f1}
    else:
        score = r2_score(y, cv_preds)
        rmse = np.sqrt(mean_squared_error(y, cv_preds))
        metrics = {'r2': score, 'rmse': rmse}

    logger.info(f"[Ensemble] Stacking score: {score:.4f}")

    return {
        'success': True,
        'ensemble': ensemble,
        'score': score,
        'metrics': metrics,
        'meta_learner': type(meta_learner).__name__,
        'base_models': [name for name, _ in estimators],
    }


@async_tool_wrapper()
async def ensemble_blend_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a blending ensemble with holdout validation.

    Args:
        params: Dict with keys:
            - models: List of (name, model) tuples or dict
            - data: DataFrame with features
            - target: Target column name
            - meta_learner: Meta-learner model
            - task: 'classification' or 'regression'
            - holdout_ratio: Ratio for holdout set (default 0.2)

    Returns:
        Dict with blending ensemble and performance
    """
    status.set_callback(params.pop('_status_callback', None))

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split

    logger.info("[Ensemble] Creating blending ensemble...")

    models = params.get('models', {})
    data = params.get('data')
    target = params.get('target')
    meta_learner = params.get('meta_learner')
    task = params.get('task', 'classification')
    holdout_ratio = params.get('holdout_ratio', 0.2)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(X.median())

    if isinstance(models, dict):
        estimators = [(name, model) for name, model in models.items()]
    else:
        estimators = models

    if not estimators:
        return {'success': False, 'error': 'No models provided'}

    if meta_learner is None:
        meta_learner = LogisticRegression() if task == 'classification' else Ridge()

    # Split for blending
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=holdout_ratio, random_state=42,
        stratify=y if task == 'classification' else None
    )

    # Train base models and get holdout predictions
    blend_train = np.zeros((len(X_hold), len(estimators)))
    blend_test = np.zeros((len(X), len(estimators)))
    trained_models = {}

    for i, (name, model) in enumerate(estimators):
        model.fit(X_train, y_train)
        trained_models[name] = model

        if task == 'classification' and hasattr(model, 'predict_proba'):
            blend_train[:, i] = model.predict_proba(X_hold)[:, 1]
            blend_test[:, i] = model.predict_proba(X)[:, 1]
        else:
            blend_train[:, i] = model.predict(X_hold)
            blend_test[:, i] = model.predict(X)

    # Train meta-learner on holdout predictions
    meta_learner.fit(blend_train, y_hold)

    # Final predictions
    final_preds = meta_learner.predict(blend_test)

    if task == 'classification':
        score = accuracy_score(y, final_preds)
        f1 = f1_score(y, final_preds, average='weighted')
        metrics = {'accuracy': score, 'f1': f1}
    else:
        score = r2_score(y, final_preds)
        rmse = np.sqrt(mean_squared_error(y, final_preds))
        metrics = {'r2': score, 'rmse': rmse}

    logger.info(f"[Ensemble] Blending score: {score:.4f}")

    return {
        'success': True,
        'base_models': trained_models,
        'meta_learner': meta_learner,
        'score': score,
        'metrics': metrics,
        'model_names': [name for name, _ in estimators],
    }


@async_tool_wrapper()
async def ensemble_vote_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a voting ensemble.

    Args:
        params: Dict with keys:
            - models: List of (name, model) tuples or dict
            - data: DataFrame with features
            - target: Target column name
            - voting: 'hard' or 'soft' (default 'soft')
            - weights: Optional weights for each model

    Returns:
        Dict with voting ensemble and performance
    """
    status.set_callback(params.pop('_status_callback', None))

    from sklearn.ensemble import VotingClassifier, VotingRegressor

    logger.info("[Ensemble] Creating voting ensemble...")

    models = params.get('models', {})
    data = params.get('data')
    target = params.get('target')
    voting = params.get('voting', 'soft')
    weights = params.get('weights')
    task = params.get('task', 'classification')

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    X = X.fillna(X.median())

    if isinstance(models, dict):
        estimators = [(name, model) for name, model in models.items()]
    else:
        estimators = models

    if not estimators:
        return {'success': False, 'error': 'No models provided'}

    # Create voting ensemble
    if task == 'classification':
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        ensemble = VotingRegressor(
            estimators=estimators,
            weights=weights
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ensemble.fit(X, y)

    cv_preds = cross_val_predict(ensemble, X, y, cv=cv)

    if task == 'classification':
        score = accuracy_score(y, cv_preds)
        f1 = f1_score(y, cv_preds, average='weighted')
        metrics = {'accuracy': score, 'f1': f1}
    else:
        score = r2_score(y, cv_preds)
        rmse = np.sqrt(mean_squared_error(y, cv_preds))
        metrics = {'r2': score, 'rmse': rmse}

    logger.info(f"[Ensemble] Voting score: {score:.4f}")

    return {
        'success': True,
        'ensemble': ensemble,
        'score': score,
        'metrics': metrics,
        'voting': voting,
        'model_names': [name for name, _ in estimators],
    }


@async_tool_wrapper()
async def ensemble_weighted_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a weighted average ensemble with automatic weight optimization.

    Args:
        params: Dict with keys:
            - predictions: Dict of {model_name: predictions}
            - y_true: True labels
            - task: 'classification' or 'regression'
            - optimize_weights: Whether to optimize weights (default True)

    Returns:
        Dict with optimal weights and ensemble predictions
    """
    status.set_callback(params.pop('_status_callback', None))

    from scipy.optimize import minimize

    logger.info("[Ensemble] Creating weighted ensemble...")

    predictions = params.get('predictions', {})
    y_true = params.get('y_true')
    task = params.get('task', 'classification')
    optimize_weights = params.get('optimize_weights', True)

    if not predictions:
        return {'success': False, 'error': 'No predictions provided'}

    model_names = list(predictions.keys())
    pred_matrix = np.array([predictions[name] for name in model_names]).T  # (n_samples, n_models)
    y = np.array(y_true)

    n_models = len(model_names)

    if optimize_weights:
        def objective(weights):
            weighted_pred = np.average(pred_matrix, axis=1, weights=weights)
            if task == 'classification':
                weighted_pred = (weighted_pred > 0.5).astype(int)
                return -accuracy_score(y, weighted_pred)
            else:
                return mean_squared_error(y, weighted_pred)

        # Constraints: weights sum to 1, all positive
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        initial = np.ones(n_models) / n_models

        result = minimize(objective, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        weights = result.x
    else:
        weights = np.ones(n_models) / n_models

    # Final predictions
    ensemble_pred = np.average(pred_matrix, axis=1, weights=weights)
    if task == 'classification':
        final_pred = (ensemble_pred > 0.5).astype(int)
        score = accuracy_score(y, final_pred)
        f1 = f1_score(y, final_pred, average='weighted')
        metrics = {'accuracy': score, 'f1': f1}
    else:
        final_pred = ensemble_pred
        score = r2_score(y, final_pred)
        rmse = np.sqrt(mean_squared_error(y, final_pred))
        metrics = {'r2': score, 'rmse': rmse}

    weight_dict = dict(zip(model_names, weights.tolist()))

    logger.info(f"[Ensemble] Weighted ensemble score: {score:.4f}")

    return {
        'success': True,
        'weights': weight_dict,
        'predictions': final_pred.tolist(),
        'score': score,
        'metrics': metrics,
    }


@async_tool_wrapper()
async def ensemble_diversity_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze diversity of ensemble models.

    Args:
        params: Dict with keys:
            - predictions: Dict of {model_name: predictions}
            - y_true: True labels

    Returns:
        Dict with diversity metrics
    """
    status.set_callback(params.pop('_status_callback', None))

    logger.info("[Ensemble] Analyzing model diversity...")

    predictions = params.get('predictions', {})
    y_true = params.get('y_true')

    if not predictions:
        return {'success': False, 'error': 'No predictions provided'}

    model_names = list(predictions.keys())
    pred_matrix = np.array([predictions[name] for name in model_names])
    y = np.array(y_true)

    # Pairwise disagreement rate
    n_models = len(model_names)
    disagreement = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            disagree_rate = np.mean(pred_matrix[i] != pred_matrix[j])
            disagreement[i, j] = disagree_rate
            disagreement[j, i] = disagree_rate

    # Average pairwise disagreement
    avg_disagreement = disagreement.sum() / (n_models * (n_models - 1))

    # Individual model accuracies
    accuracies = {name: accuracy_score(y, predictions[name]) for name in model_names}

    # Correlation between predictions
    correlation = np.corrcoef(pred_matrix)

    logger.info(f"[Ensemble] Average disagreement: {avg_disagreement:.4f}")

    return {
        'success': True,
        'disagreement_matrix': disagreement.tolist(),
        'avg_disagreement': float(avg_disagreement),
        'correlation_matrix': correlation.tolist(),
        'model_accuracies': accuracies,
        'model_names': model_names,
        'diversity_score': float(avg_disagreement),  # Higher = more diverse
    }
