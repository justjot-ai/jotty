"""
AutoML Skill - Automated Machine Learning

Provides world-class AutoML capabilities:
- AutoGluon for maximum accuracy
- FLAML for speed and efficiency
- Hyperparameter optimization
- Ensemble methods
- Backtesting for trading strategies
- Feature engineering and selection

Consolidated from core ML infrastructure.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

logger = logging.getLogger(__name__)
status = SkillStatus("automl")


@async_tool_wrapper()
async def automl_train(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train ML model using AutoML (AutoGluon or FLAML).

    Params:
        data: DataFrame or path to CSV file
        target: Target column name
        problem_type: 'classification' or 'regression'
        time_budget: Time budget in seconds (default: 120)
        framework: 'autogluon', 'flaml', or 'both' (default: 'autogluon')
        eval_metric: Evaluation metric (optional, auto-detected)

    Returns:
        {
            "success": True,
            "best_model": "...",
            "score": 0.95,
            "framework": "autogluon",
            "leaderboard": [...]
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .automl_skill import AutoMLSkill

        data = params.get("data")
        if isinstance(data, str):
            data = pd.read_csv(data)

        target = params.get("target", "target")
        X = data.drop(columns=[target])
        y = data[target]

        problem_type = params.get("problem_type", "classification")
        time_budget = params.get("time_budget", 120)
        framework = params.get("framework", "autogluon")

        status.update(f"Training {problem_type} model with {framework}...")

        skill = AutoMLSkill()
        await skill.init()

        result = await skill.execute(
            X=X,
            y=y,
            problem_type=problem_type,
            time_budget=time_budget,
            use_autogluon=(framework in ["autogluon", "both"]),
            use_flaml=(framework in ["flaml", "both"]),
        )

        status.complete("Model trained successfully")

        return tool_response(
            best_model=result.metadata.get("best_model"),
            score=result.metadata.get("best_score"),
            framework=result.metadata.get("best_framework"),
            leaderboard=result.metadata.get("leaderboard", []),
        )

    except ImportError as e:
        return tool_error(f"Missing dependency: {e}. Install: pip install autogluon or flaml")
    except Exception as e:
        logger.error(f"AutoML training failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@async_tool_wrapper()
async def hyperparameter_optimize(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna/Hyperopt.

    Params:
        data: DataFrame or CSV path
        target: Target column
        model_type: Model type (rf, xgboost, lightgbm, etc.)
        n_trials: Number of optimization trials (default: 50)

    Returns:
        {
            "success": True,
            "best_params": {...},
            "best_score": 0.95,
            "optimization_history": [...]
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .hyperopt import HyperparameterOptimizer

        data = params.get("data")
        if isinstance(data, str):
            data = pd.read_csv(data)

        target = params.get("target")
        X = data.drop(columns=[target])
        y = data[target]

        model_type = params.get("model_type", "xgboost")
        n_trials = params.get("n_trials", 50)

        status.update(f"Optimizing {model_type} hyperparameters...")

        optimizer = HyperparameterOptimizer(model_type=model_type)
        best_params, best_score, history = optimizer.optimize(X, y, n_trials=n_trials)

        status.complete("Optimization complete")

        return tool_response(
            best_params=best_params, best_score=best_score, optimization_history=history
        )

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@async_tool_wrapper()
async def backtest_strategy(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backtest trading strategy with ML predictions.

    Params:
        data: DataFrame with price/feature data
        strategy_type: 'ml_classification', 'ml_regression', 'technical'
        target: Target column (returns, signals, etc.)
        initial_capital: Starting capital (default: 10000)

    Returns:
        {
            "success": True,
            "total_return": 0.45,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.15,
            "trades": 120,
            "win_rate": 0.62
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .backtest_engine import BacktestEngine

        data = params.get("data")
        if isinstance(data, str):
            data = pd.read_csv(data, parse_dates=["date"], index_col="date")

        strategy_type = params.get("strategy_type", "ml_classification")
        initial_capital = params.get("initial_capital", 10000)

        status.update(f"Backtesting {strategy_type} strategy...")

        engine = BacktestEngine(initial_capital=initial_capital)
        results = engine.run(data, strategy_type=strategy_type)

        status.complete("Backtest complete")

        return tool_response(
            total_return=results["total_return"],
            sharpe_ratio=results["sharpe_ratio"],
            max_drawdown=results["max_drawdown"],
            trades=results["num_trades"],
            win_rate=results["win_rate"],
            equity_curve=results.get("equity_curve", []),
        )

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@async_tool_wrapper()
async def feature_engineering(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automated feature engineering and selection.

    Params:
        data: DataFrame
        target: Target column
        max_features: Maximum features to select (default: 50)

    Returns:
        {
            "success": True,
            "engineered_features": [...],
            "feature_importance": {...},
            "new_data": DataFrame
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .feature_engineering import FeatureEngineer
        from .feature_selection import FeatureSelector

        data = params.get("data")
        if isinstance(data, str):
            data = pd.read_csv(data)

        target = params.get("target")
        max_features = params.get("max_features", 50)

        status.update("Engineering features...")

        # Feature engineering
        engineer = FeatureEngineer()
        data_engineered = engineer.create_features(data)

        # Feature selection
        X = data_engineered.drop(columns=[target])
        y = data_engineered[target]

        selector = FeatureSelector(max_features=max_features)
        selected_features, importance = selector.select(X, y)

        status.complete(f"Selected {len(selected_features)} features")

        return tool_response(
            engineered_features=list(selected_features),
            feature_importance=importance,
            new_data=data_engineered[list(selected_features) + [target]].to_dict(),
        )

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


__all__ = [
    "automl_train",
    "hyperparameter_optimize",
    "backtest_strategy",
    "feature_engineering",
]
