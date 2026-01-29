"""
Hyperparameter Optimization Skill for Jotty
============================================

Uses Optuna for Bayesian hyperparameter optimization.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


async def hyperopt_optimize_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a model using Optuna.

    Args:
        params: Dict with keys:
            - data: DataFrame or path to CSV
            - target: Target column name
            - model_type: 'xgboost', 'lightgbm', 'random_forest', 'gradient_boosting'
            - n_trials: Number of optimization trials (default: 50)
            - cv_folds: Cross-validation folds (default: 5)

    Returns:
        Dict with best_params, best_score, study_results
    """
    import optuna
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("[Hyperopt] Starting Bayesian optimization...")

    # Load data
    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target', 'target')
    model_type = params.get('model_type', 'xgboost')
    n_trials = params.get('n_trials', 50)
    cv_folds = params.get('cv_folds', 5)

    # Prepare data
    X = data.drop(columns=[target]).copy()
    y = data[target]

    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Define objective based on model type
    if model_type == 'xgboost':
        import xgboost as xgb

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**params)
            return cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy').mean()

        model_class = xgb.XGBClassifier

    elif model_type == 'lightgbm':
        import lightgbm as lgb

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'verbose': -1
            }
            model = lgb.LGBMClassifier(**params)
            return cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy').mean()

        model_class = lgb.LGBMClassifier

    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
            return cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy').mean()

        model_class = RandomForestClassifier

    elif model_type == 'catboost':
        from catboost import CatBoostClassifier

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': 42,
                'verbose': False
            }
            model = CatBoostClassifier(**params)
            return cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy').mean()

        model_class = CatBoostClassifier

    else:  # gradient_boosting
        from sklearn.ensemble import GradientBoostingClassifier

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
            return cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy').mean()

        model_class = GradientBoostingClassifier

    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_score = study.best_value
    best_params = study.best_params

    logger.info(f"[Hyperopt] Best score: {best_score:.4f}")
    logger.info(f"[Hyperopt] Best params: {best_params}")

    # Train final model
    if model_type == 'xgboost':
        best_params['random_state'] = 42
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
    elif model_type == 'lightgbm':
        best_params['random_state'] = 42
        best_params['verbose'] = -1
    elif model_type == 'catboost':
        best_params['random_seed'] = 42
        best_params['verbose'] = False
    else:
        best_params['random_state'] = 42

    best_model = model_class(**best_params)
    best_model.fit(X_scaled, y)

    return {
        'success': True,
        'best_score': float(best_score),
        'best_params': best_params,
        'model': best_model,
        'model_type': model_type,
        'n_trials': n_trials,
        'feature_columns': list(X.columns),
    }


async def hyperopt_multi_model_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize multiple model types and return the best.

    Args:
        params: Dict with keys:
            - data: DataFrame or path
            - target: Target column
            - model_types: List of model types to try
            - n_trials_per_model: Trials per model (default: 30)

    Returns:
        Dict with results for each model and overall best
    """
    model_types = params.get('model_types', ['xgboost', 'lightgbm', 'random_forest'])
    n_trials = params.get('n_trials_per_model', 30)

    results = {}
    best_overall = None
    best_overall_score = 0

    for model_type in model_types:
        logger.info(f"[Hyperopt] Optimizing {model_type}...")
        try:
            result = await hyperopt_optimize_tool({
                'data': params.get('data'),
                'target': params.get('target'),
                'model_type': model_type,
                'n_trials': n_trials,
            })

            results[model_type] = result

            if result['success'] and result['best_score'] > best_overall_score:
                best_overall_score = result['best_score']
                best_overall = model_type

        except Exception as e:
            logger.warning(f"[Hyperopt] {model_type} failed: {e}")
            results[model_type] = {'success': False, 'error': str(e)}

    logger.info(f"[Hyperopt] Best overall: {best_overall} ({best_overall_score:.4f})")

    return {
        'success': True,
        'results': results,
        'best_model_type': best_overall,
        'best_score': best_overall_score,
        'best_model': results[best_overall]['model'] if best_overall else None,
    }
