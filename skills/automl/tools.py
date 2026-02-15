"""
AutoML Skill Tools for Jotty
============================

Provides automatic machine learning capabilities:
- Model selection from 10+ algorithms
- Cross-validation and evaluation
- Ensemble creation
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("automl")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def automl_classify_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatic classifier selection and training.

    Args:
        params: Dict with keys:
            - data: DataFrame or path to CSV
            - target: Target column name
            - features: Optional list of feature columns
            - cv_folds: Number of CV folds (default: 5)

    Returns:
        Dict with best_model, best_score, all_scores, feature_importance
    """
    status.set_callback(params.pop('_status_callback', None))

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, AdaBoostClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier

    logger.info("[AutoML] Starting automatic model selection...")

    # Load data
    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target', 'target')
    features = params.get('features')
    cv_folds = params.get('cv_folds', 5)

    # Prepare features
    if features:
        X = data[features].copy()
    else:
        X = data.drop(columns=[target]).copy()

    y = data[target]

    # Encode categoricals
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Fill NaN
    X = X.fillna(X.median())

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }

    # Try XGBoost/LightGBM if available
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, verbose=-1
        )
    except ImportError:
        pass

    # Evaluate all models
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}
    trained_models = {}

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            results[name] = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'scores': scores.tolist()
            }
            model.fit(X_scaled, y)
            trained_models[name] = model
            logger.info(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['mean'])
    best_score = results[best_name]['mean']

    # Feature importance from best tree model
    feature_importance = {}
    if hasattr(trained_models[best_name], 'feature_importances_'):
        importance = trained_models[best_name].feature_importances_
        for i, col in enumerate(X.columns):
            feature_importance[col] = float(importance[i])

    logger.info(f"[AutoML] Best: {best_name} ({best_score:.4f})")

    return {
        'success': True,
        'best_model': best_name,
        'best_score': best_score,
        'all_scores': results,
        'feature_importance': feature_importance,
        'models': trained_models,
        'feature_columns': list(X.columns),
    }


@async_tool_wrapper()
async def automl_ensemble_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create ensemble from trained models.

    Args:
        params: Dict with keys:
            - models: Dict of trained models (from automl_classify)
            - data: DataFrame or path
            - target: Target column
            - top_k: Number of top models to ensemble (default: 5)
            - method: 'voting' or 'stacking' (default: 'voting')
    """
    status.set_callback(params.pop('_status_callback', None))

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    logger.info("[AutoML] Creating ensemble...")

    models = params.get('models', {})
    scores = params.get('all_scores', {})
    data = params.get('data')
    target = params.get('target', 'target')
    top_k = params.get('top_k', 5)
    method = params.get('method', 'voting')

    if isinstance(data, str):
        data = pd.read_csv(data)

    # Prepare data
    X = data.drop(columns=[target]).copy()
    y = data[target]

    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select top models
    sorted_models = sorted(scores.items(), key=lambda x: x[1]['mean'], reverse=True)[:top_k]
    top_model_names = [name for name, _ in sorted_models]

    estimators = [
        (name, models[name]) for name in top_model_names
        if name in models and hasattr(models[name], 'predict_proba')
    ]

    if len(estimators) < 2:
        return {'success': False, 'error': 'Need at least 2 models for ensemble'}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if method == 'voting':
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='accuracy')
        score = float(ensemble_scores.mean())
        logger.info(f"[AutoML] Voting ensemble: {score:.4f}")
    else:
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        ensemble_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='accuracy')
        score = float(ensemble_scores.mean())
        logger.info(f"[AutoML] Stacking ensemble: {score:.4f}")

    ensemble.fit(X_scaled, y)

    return {
        'success': True,
        'ensemble_type': method,
        'ensemble_score': score,
        'models_used': top_model_names,
        'ensemble': ensemble,
    }


@async_tool_wrapper()
async def automl_evaluate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate model on test data.

    Args:
        params: Dict with keys:
            - model: Trained model
            - test_data: Test DataFrame or path
            - target: Target column (if available)
    """
    status.set_callback(params.pop('_status_callback', None))

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    model = params.get('model')
    test_data = params.get('test_data')
    target = params.get('target')
    feature_columns = params.get('feature_columns', [])

    if isinstance(test_data, str):
        test_data = pd.read_csv(test_data)

    # Prepare features
    if feature_columns:
        X_test = test_data[feature_columns].copy()
    elif target and target in test_data.columns:
        X_test = test_data.drop(columns=[target]).copy()
    else:
        X_test = test_data.copy()

    # Encode
    for col in X_test.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_test[col] = le.fit_transform(X_test[col].astype(str))
    X_test = X_test.fillna(X_test.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)

    # Predict
    predictions = model.predict(X_scaled)

    result = {
        'success': True,
        'predictions': predictions.tolist(),
    }

    # If target available, compute metrics
    if target and target in test_data.columns:
        y_test = test_data[target]
        result['accuracy'] = float(accuracy_score(y_test, predictions))
        result['confusion_matrix'] = confusion_matrix(y_test, predictions).tolist()
        logger.info(f"[AutoML] Test accuracy: {result['accuracy']:.4f}")

    return result
