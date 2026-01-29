"""
Hyperparameter Optimization Skill
=================================

World-class hyperparameter optimization:
1. Multi-model tuning (LightGBM, XGBoost, RandomForest)
2. Optuna pruning (MedianPruner - stop bad trials early)
3. Expanded parameter space with regularization
4. Efficient TPE sampler
"""

import sys
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

from .base import MLSkill, SkillResult, SkillCategory

logger = logging.getLogger(__name__)


class HyperoptSkill(MLSkill):
    """
    World-class hyperparameter optimization skill.

    Uses Optuna for efficient multi-model tuning.
    """

    name = "hyperopt"
    version = "2.0.0"
    description = "Optuna-based multi-model hyperparameter optimization"
    category = SkillCategory.HYPERPARAMETER_OPTIMIZATION

    required_inputs = ["X", "y"]
    optional_inputs = ["problem_type", "time_budget", "model_ranking"]
    outputs = ["optimized_model", "best_params"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    async def execute(self,
                      X: pd.DataFrame,
                      y: Optional[pd.Series] = None,
                      **context) -> SkillResult:
        """
        Execute hyperparameter optimization.

        Args:
            X: Input features
            y: Target variable (required)
            **context: problem_type, time_budget, model_ranking, etc.

        Returns:
            SkillResult with optimized model
        """
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import lightgbm as lgb
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        start_time = time.time()

        if y is None:
            return self._create_error_result("Target variable y is required")

        if not self.validate_inputs(X, y):
            return self._create_error_result("Invalid inputs")

        problem_type = context.get('problem_type', 'classification')
        time_budget = context.get('time_budget', 60)
        model_ranking = context.get('model_ranking', ['lightgbm', 'xgboost', 'random_forest'])
        tune_all = context.get('tune_all_models', True)  # NEW: tune all models by default

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Tune ALL models (not just top 3) for better ensemble diversity
        if tune_all:
            top_models = model_ranking[:6]  # Tune top 6 models
        else:
            top_models = model_ranking[:3]

        # Progress tracking - adaptive trials based on number of models
        n_trials_per_model = min(15, max(8, time_budget // (len(top_models) * 2)))
        total_trials = n_trials_per_model * len(top_models)
        best_score_so_far = [0.0]
        trial_count = [0]
        current_model = ['']

        def progress_callback(study, trial):
            trial_count[0] += 1
            if trial.value and trial.value > best_score_so_far[0]:
                best_score_so_far[0] = trial.value
            pct = (trial_count[0] / total_trials) * 100
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = '▓' * filled + '░' * (bar_len - filled)
            model_short = current_model[0][:8]
            sys.stdout.write(f'\r      Hyperopt [{bar}] {trial_count[0]:2d}/{total_trials} | {model_short} | best={best_score_so_far[0]:.4f}')
            sys.stdout.flush()

        if problem_type == 'classification':
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'

        model_results = {}

        # Tune each top model
        for model_name in top_models:
            current_model[0] = model_name

            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

            objective = self._get_objective(model_name, X_scaled, y, cv, scoring, problem_type)

            try:
                study.optimize(objective, n_trials=n_trials_per_model,
                              callbacks=[progress_callback], show_progress_bar=False)
                model_results[model_name] = {
                    'score': study.best_value,
                    'params': study.best_params,
                }
            except Exception as e:
                logger.debug(f"Hyperopt failed for {model_name}: {e}")

        print()  # New line after progress

        # Select best model
        best_model_name = None
        best_score = 0
        best_params = {}

        for model_name, result in model_results.items():
            if result['score'] > best_score:
                best_score = result['score']
                best_model_name = model_name
                best_params = result['params']

        # Build optimized model
        optimized_model = self._build_optimized_model(
            best_model_name, best_params, problem_type
        )

        execution_time = time.time() - start_time

        return self._create_result(
            success=True,
            data=optimized_model,
            metrics={
                'score': best_score,
                'n_trials': total_trials,
                'best_model': best_model_name,
                'models_tuned': len(model_results)
            },
            metadata={
                'best_params': best_params,
                'all_model_scores': {k: v['score'] for k, v in model_results.items()},
                'best_model_name': best_model_name
            },
            execution_time=execution_time,
        )

    def _get_objective(self, model_name: str, X_scaled, y, cv, scoring: str,
                        problem_type: str):
        """Get objective function for model."""
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import lightgbm as lgb
        import xgboost as xgb

        if model_name in ['lightgbm', 'lgb']:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 8, 128),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params) if problem_type == 'classification' else lgb.LGBMRegressor(**params)
                return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()
            return objective

        elif model_name in ['xgboost', 'xgb']:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': 42,
                    'verbosity': 0,
                    'eval_metric': 'logloss' if problem_type == 'classification' else 'rmse'
                }
                model = xgb.XGBClassifier(**params) if problem_type == 'classification' else xgb.XGBRegressor(**params)
                return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()
            return objective

        elif model_name in ['random_forest', 'rf']:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params) if problem_type == 'classification' else RandomForestRegressor(**params)
                return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()
            return objective

        elif model_name in ['catboost', 'cat']:
            # CatBoost - handles categoricals natively
            def objective(trial):
                try:
                    from catboost import CatBoostClassifier, CatBoostRegressor
                    params = {
                        'iterations': trial.suggest_int('iterations', 50, 400),
                        'depth': trial.suggest_int('depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                        'random_state': 42,
                        'verbose': False
                    }
                    model = CatBoostClassifier(**params) if problem_type == 'classification' else CatBoostRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()
                except ImportError:
                    return 0.0
            return objective

        elif model_name in ['histgb', 'hist_gradient_boosting']:
            # HistGradientBoosting - fast, handles NaN
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            def objective(trial):
                params = {
                    'max_iter': trial.suggest_int('max_iter', 50, 400),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
                    'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
                    'max_bins': trial.suggest_int('max_bins', 64, 255),
                    'random_state': 42
                }
                model = HistGradientBoostingClassifier(**params) if problem_type == 'classification' else HistGradientBoostingRegressor(**params)
                return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()
            return objective

        elif model_name in ['extra_trees', 'et']:
            # ExtraTrees - more randomized than RF
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = ExtraTreesClassifier(**params) if problem_type == 'classification' else ExtraTreesRegressor(**params)
                return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()
            return objective

        elif model_name in ['gradient_boosting', 'gb']:
            # sklearn GradientBoosting
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params) if problem_type == 'classification' else GradientBoostingRegressor(**params)
                return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()
            return objective

        else:
            # Default: LightGBM
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params) if problem_type == 'classification' else lgb.LGBMRegressor(**params)
                return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()
            return objective

    def _build_optimized_model(self, model_name: str, params: Dict, problem_type: str):
        """Build optimized model with best params."""
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        if model_name in ['lightgbm', 'lgb']:
            final_params = {**params, 'random_state': 42, 'verbose': -1}
            return lgb.LGBMClassifier(**final_params) if problem_type == 'classification' else lgb.LGBMRegressor(**final_params)

        elif model_name in ['xgboost', 'xgb']:
            final_params = {**params, 'random_state': 42, 'verbosity': 0}
            return xgb.XGBClassifier(**final_params) if problem_type == 'classification' else xgb.XGBRegressor(**final_params)

        elif model_name in ['random_forest', 'rf']:
            final_params = {**params, 'random_state': 42, 'n_jobs': -1}
            return RandomForestClassifier(**final_params) if problem_type == 'classification' else RandomForestRegressor(**final_params)

        elif model_name in ['catboost', 'cat']:
            try:
                from catboost import CatBoostClassifier, CatBoostRegressor
                final_params = {**params, 'random_state': 42, 'verbose': False}
                return CatBoostClassifier(**final_params) if problem_type == 'classification' else CatBoostRegressor(**final_params)
            except ImportError:
                return lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

        elif model_name in ['histgb', 'hist_gradient_boosting']:
            final_params = {**params, 'random_state': 42}
            return HistGradientBoostingClassifier(**final_params) if problem_type == 'classification' else HistGradientBoostingRegressor(**final_params)

        elif model_name in ['extra_trees', 'et']:
            final_params = {**params, 'random_state': 42, 'n_jobs': -1}
            return ExtraTreesClassifier(**final_params) if problem_type == 'classification' else ExtraTreesRegressor(**final_params)

        elif model_name in ['gradient_boosting', 'gb']:
            final_params = {**params, 'random_state': 42}
            return GradientBoostingClassifier(**final_params) if problem_type == 'classification' else GradientBoostingRegressor(**final_params)

        else:
            return lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) if problem_type == 'classification' else lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
