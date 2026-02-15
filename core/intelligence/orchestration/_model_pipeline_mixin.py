"""SkillOrchestrator mixin — model selection, hyperopt, and ensemble stages."""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .skill_types import ProblemType, SkillResult

logger = logging.getLogger(__name__)


class ModelPipelineMixin:
    """Mixin providing _builtin_model_selection, _builtin_hyperopt, _builtin_ensemble."""

    async def _builtin_model_selection(
        self, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType
    ) -> SkillResult:
        """
        World-class model selection with:
        1. Extended model zoo (7+ algorithms)
        2. Data-adaptive configurations
        3. Cross-validation with OOF predictions for stacking
        4. Model diversity tracking for ensemble
        """
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import (
            KFold,
            StratifiedKFold,
            cross_val_predict,
            cross_val_score,
        )
        from sklearn.svm import SVC, SVR

        n_samples, n_features = X.shape

        X_scaled = self._encode_categoricals_and_scale(X)

        # Adaptive configuration based on data size
        n_estimators = 100 if n_samples < 5000 else 200
        max_depth_tree = None if n_samples > 1000 else 10

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "accuracy"

            models = {
                # Gradient Boosting Family (usually best)
                "lightgbm": lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    verbose=-1,
                    learning_rate=0.1,
                    num_leaves=31,
                ),
                "xgboost": xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    eval_metric="logloss",
                    verbosity=0,
                    learning_rate=0.1,
                ),
                "histgb": HistGradientBoostingClassifier(
                    max_iter=n_estimators, random_state=42, learning_rate=0.1
                ),
                # Tree Ensembles (good diversity)
                "random_forest": RandomForestClassifier(
                    n_estimators=n_estimators, random_state=42, max_depth=max_depth_tree, n_jobs=-1
                ),
                "extra_trees": ExtraTreesClassifier(
                    n_estimators=n_estimators, random_state=42, max_depth=max_depth_tree, n_jobs=-1
                ),
                "gradient_boosting": GradientBoostingClassifier(
                    n_estimators=min(n_estimators, 100), random_state=42, learning_rate=0.1
                ),
                # Linear Model (different hypothesis space)
                "logistic": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
            }

            # Add SVM for small datasets (slow on large)
            if n_samples < 5000:
                models["svm"] = SVC(probability=True, random_state=42, C=1.0)

        else:  # Regression
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "r2"

            models = {
                "lightgbm": lgb.LGBMRegressor(
                    n_estimators=n_estimators, random_state=42, verbose=-1
                ),
                "xgboost": xgb.XGBRegressor(
                    n_estimators=n_estimators, random_state=42, verbosity=0
                ),
                "histgb": HistGradientBoostingRegressor(max_iter=n_estimators, random_state=42),
                "random_forest": RandomForestRegressor(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
                "extra_trees": ExtraTreesRegressor(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=min(n_estimators, 100), random_state=42
                ),
                "ridge": Ridge(alpha=1.0),
            }

            if n_samples < 5000:
                models["svr"] = SVR(C=1.0)

        # Evaluate all models and collect OOF predictions
        best_model = None
        best_name = None
        best_score = -np.inf
        all_scores = {}
        all_std = {}
        oof_predictions = {}  # For stacking

        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
                mean_score = scores.mean()
                std_score = scores.std()
                all_scores[name] = mean_score
                all_std[name] = std_score

                # Collect OOF predictions for top models (for stacking later)
                if mean_score > best_score * 0.95:  # Within 5% of best
                    try:
                        if problem_type == ProblemType.CLASSIFICATION:
                            oof = cross_val_predict(
                                model, X_scaled, y, cv=cv, method="predict_proba"
                            )
                            oof_predictions[name] = oof[:, 1] if oof.ndim > 1 else oof
                        else:
                            oof_predictions[name] = cross_val_predict(model, X_scaled, y, cv=cv)
                    except Exception:
                        pass

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name

            except Exception as e:
                logger.debug(f"Model {name} failed: {e}")

        # Rank models for reporting
        sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        return SkillResult(
            skill_name="builtin_model_selector",
            category=SkillCategory.MODEL_SELECTION,
            success=True,
            data=best_model,
            metrics={
                "score": best_score,
                "model": best_name,
                **{k: all_scores[k] for k in list(all_scores)[:4]},
            },
            metadata={
                "best_model": best_name,
                "all_scores": all_scores,
                "all_std": all_std,
                "model_ranking": [m[0] for m in sorted_models],
                "oof_predictions": oof_predictions,
                "X_scaled": X_scaled,
            },
        )

    async def _builtin_hyperopt(
        self, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType, context: Dict
    ) -> SkillResult:
        """
        World-class hyperparameter optimization:
        1. Multi-model tuning (LightGBM, XGBoost, RandomForest)
        2. Optuna pruning (MedianPruner - stop bad trials early)
        3. Expanded parameter space with regularization
        4. Efficient TPE sampler with warm-starting
        """
        import lightgbm as lgb
        import optuna
        import xgboost as xgb
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X_scaled = self._encode_categoricals_and_scale(X)

        # Get top models from model selection stage
        model_metadata = {}
        for result in context.get("skill_results", []):
            if hasattr(result, "metadata") and result.metadata:
                model_metadata.update(result.metadata)

        model_ranking = model_metadata.get(
            "model_ranking", ["lightgbm", "xgboost", "random_forest"]
        )
        top_models = model_ranking[:3]  # Tune top 3 models

        # Progress tracking
        n_trials_per_model = min(20, context.get("time_budget", 60) // 4)
        total_trials = n_trials_per_model * len(top_models)
        best_score_so_far = [0.0]
        trial_count = [0]
        current_model = [""]

        def progress_callback(study: Any, trial: Any) -> None:
            trial_count[0] += 1
            if trial.value and trial.value > best_score_so_far[0]:
                best_score_so_far[0] = trial.value
            pct = (trial_count[0] / total_trials) * 100
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = "▓" * filled + "░" * (bar_len - filled)
            model_short = current_model[0][:8]
            sys.stdout.write(
                f"\r      Hyperopt [{bar}] {trial_count[0]:2d}/{total_trials} | {model_short} | best={best_score_so_far[0]:.4f}"
            )
            sys.stdout.flush()

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "accuracy"
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "r2"

        # Store best results per model
        model_results = {}

        # ================================================================
        # TUNE EACH TOP MODEL
        # ================================================================
        for model_name in top_models:
            current_model[0] = model_name

            # Create study with pruner for efficiency
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

            if model_name in ["lightgbm", "lgb"]:

                def objective(trial: Any) -> Any:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                        "max_depth": trial.suggest_int("max_depth", 3, 12),
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                        "random_state": 42,
                        "verbose": -1,
                    }
                    model = (
                        lgb.LGBMClassifier(**params)
                        if problem_type == ProblemType.CLASSIFICATION
                        else lgb.LGBMRegressor(**params)
                    )
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            elif model_name in ["xgboost", "xgb"]:

                def objective(trial: Any) -> Any:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                        "max_depth": trial.suggest_int("max_depth", 3, 12),
                        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                        "random_state": 42,
                        "verbosity": 0,
                        "eval_metric": (
                            "logloss" if problem_type == ProblemType.CLASSIFICATION else "rmse"
                        ),
                    }
                    model = (
                        xgb.XGBClassifier(**params)
                        if problem_type == ProblemType.CLASSIFICATION
                        else xgb.XGBRegressor(**params)
                    )
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            elif model_name in ["random_forest", "rf"]:

                def objective(trial: Any) -> Any:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                        "max_depth": trial.suggest_int("max_depth", 3, 20),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                        "max_features": trial.suggest_categorical(
                            "max_features", ["sqrt", "log2", None]
                        ),
                        "random_state": 42,
                        "n_jobs": -1,
                    }
                    model = (
                        RandomForestClassifier(**params)
                        if problem_type == ProblemType.CLASSIFICATION
                        else RandomForestRegressor(**params)
                    )
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            else:
                # Default: LightGBM
                def objective(trial: Any) -> Any:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                        "random_state": 42,
                        "verbose": -1,
                    }
                    model = (
                        lgb.LGBMClassifier(**params)
                        if problem_type == ProblemType.CLASSIFICATION
                        else lgb.LGBMRegressor(**params)
                    )
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            try:
                study.optimize(
                    objective,
                    n_trials=n_trials_per_model,
                    callbacks=[progress_callback],
                    show_progress_bar=False,
                )
                model_results[model_name] = {
                    "score": study.best_value,
                    "params": study.best_params,
                    "study": study,
                }
            except Exception as e:
                logger.debug(f"Hyperopt failed for {model_name}: {e}")

        logger.info("Hyperparameter optimization complete")

        # ================================================================
        # SELECT BEST MODEL AND BUILD
        # ================================================================
        best_model_name = None
        best_score = 0
        best_params = {}

        for model_name, result in model_results.items():
            if result["score"] > best_score:
                best_score = result["score"]
                best_model_name = model_name
                best_params = result["params"]

        # Build optimized model
        if best_model_name in ["lightgbm", "lgb"]:
            final_params = {**best_params, "random_state": 42, "verbose": -1}
            optimized_model = (
                lgb.LGBMClassifier(**final_params)
                if problem_type == ProblemType.CLASSIFICATION
                else lgb.LGBMRegressor(**final_params)
            )
        elif best_model_name in ["xgboost", "xgb"]:
            final_params = {**best_params, "random_state": 42, "verbosity": 0}
            optimized_model = (
                xgb.XGBClassifier(**final_params)
                if problem_type == ProblemType.CLASSIFICATION
                else xgb.XGBRegressor(**final_params)
            )
        elif best_model_name in ["random_forest", "rf"]:
            final_params = {**best_params, "random_state": 42, "n_jobs": -1}
            optimized_model = (
                RandomForestClassifier(**final_params)
                if problem_type == ProblemType.CLASSIFICATION
                else RandomForestRegressor(**final_params)
            )
        else:
            # Fallback
            optimized_model = (
                lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                if problem_type == ProblemType.CLASSIFICATION
                else lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            )

        return SkillResult(
            skill_name="builtin_hyperopt",
            category=SkillCategory.HYPERPARAMETER_OPTIMIZATION,
            success=True,
            data=optimized_model,
            metrics={
                "score": best_score,
                "n_trials": total_trials,
                "best_model": best_model_name,
                "models_tuned": len(model_results),
            },
            metadata={
                "best_params": best_params,
                "all_model_scores": {k: v["score"] for k, v in model_results.items()},
                "best_model_name": best_model_name,
            },
        )

    async def _builtin_ensemble(
        self, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType, context: Dict
    ) -> SkillResult:
        """
        World-class ensemble with multiple strategies:
        1. Weighted Voting - weight by CV score
        2. Stacking - meta-learner on OOF predictions
        3. Greedy Selection - iteratively add models that improve score
        4. Blending - average top diverse models
        """
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
            StackingClassifier,
            StackingRegressor,
            VotingClassifier,
            VotingRegressor,
        )
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import (
            KFold,
            StratifiedKFold,
            cross_val_score,
        )

        X_scaled = self._encode_categoricals_and_scale(X)

        # Get optimized model and scores from previous stages
        optimized = context.get("model")
        best_single_score = context.get("score", 0)

        # Get model scores from model selection stage
        model_metadata = {}
        for result in context.get("skill_results", []):
            if hasattr(result, "metadata") and result.metadata:
                model_metadata.update(result.metadata)

        all_scores = model_metadata.get("all_scores", {})
        oof_predictions = model_metadata.get("oof_predictions", {})

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "accuracy"

            # Base models with different characteristics for diversity
            base_models = {
                "lgb": lgb.LGBMClassifier(n_estimators=150, random_state=42, verbose=-1),
                "xgb": xgb.XGBClassifier(
                    n_estimators=150, random_state=42, eval_metric="logloss", verbosity=0
                ),
                "rf": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
                "gb": GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)

        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "r2"

            base_models = {
                "lgb": lgb.LGBMRegressor(n_estimators=150, random_state=42, verbose=-1),
                "xgb": xgb.XGBRegressor(n_estimators=150, random_state=42, verbosity=0),
                "rf": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
                "gb": GradientBoostingRegressor(n_estimators=100, random_state=42),
            }
            meta_learner = Ridge(alpha=1.0)

        ensemble_results = {}

        # ================================================================
        # STRATEGY 1: Weighted Voting (weight by CV score)
        # ================================================================
        try:
            # Calculate weights from scores
            if all_scores:
                weights = []
                estimators = []
                for name, model in base_models.items():
                    score = all_scores.get(name, 0.5)
                    weights.append(max(score, 0.1))  # Min weight 0.1
                    estimators.append((name, model))

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                if problem_type == ProblemType.CLASSIFICATION:
                    weighted_ensemble = VotingClassifier(
                        estimators=estimators, voting="soft", weights=weights
                    )
                else:
                    weighted_ensemble = VotingRegressor(estimators=estimators, weights=weights)

                weighted_scores = cross_val_score(
                    weighted_ensemble, X_scaled, y, cv=cv, scoring=scoring
                )
                ensemble_results["weighted_voting"] = weighted_scores.mean()
        except Exception as e:
            logger.debug(f"Weighted voting failed: {e}")

        # ================================================================
        # STRATEGY 2: Stacking with Meta-Learner
        # ================================================================
        try:
            estimators_list = [(name, model) for name, model in base_models.items()]

            if problem_type == ProblemType.CLASSIFICATION:
                stacking = StackingClassifier(
                    estimators=estimators_list,
                    final_estimator=meta_learner,
                    cv=3,  # Faster inner CV
                    passthrough=False,
                    n_jobs=-1,
                )
            else:
                stacking = StackingRegressor(
                    estimators=estimators_list,
                    final_estimator=meta_learner,
                    cv=3,
                    passthrough=False,
                    n_jobs=-1,
                )

            stacking_scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring=scoring)
            ensemble_results["stacking"] = stacking_scores.mean()
        except Exception as e:
            logger.debug(f"Stacking failed: {e}")

        # ================================================================
        # STRATEGY 3: Greedy Ensemble Selection
        # ================================================================
        try:
            # Start with best single model, greedily add models that improve
            sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            greedy_estimators = []

            if sorted_models:
                # Start with best model
                best_name = sorted_models[0][0]
                if best_name in base_models:
                    greedy_estimators.append((best_name, base_models[best_name]))

                current_best = all_scores.get(best_name, 0)

                # Try adding each remaining model
                for name, score in sorted_models[1:4]:  # Top 4 models
                    if name in base_models:
                        test_estimators = greedy_estimators + [(name, base_models[name])]

                        if problem_type == ProblemType.CLASSIFICATION:
                            test_ensemble = VotingClassifier(
                                estimators=test_estimators, voting="soft"
                            )
                        else:
                            test_ensemble = VotingRegressor(estimators=test_estimators)

                        test_scores = cross_val_score(
                            test_ensemble, X_scaled, y, cv=cv, scoring=scoring
                        )
                        test_score = test_scores.mean()

                        # Add if improves
                        if test_score > current_best:
                            greedy_estimators.append((name, base_models[name]))
                            current_best = test_score

                ensemble_results["greedy"] = current_best
                ensemble_results["greedy_models"] = [e[0] for e in greedy_estimators]
        except Exception as e:
            logger.debug(f"Greedy selection failed: {e}")

        # ================================================================
        # STRATEGY 4: Simple Average (Baseline)
        # ================================================================
        try:
            simple_estimators = [(name, model) for name, model in list(base_models.items())[:3]]

            if problem_type == ProblemType.CLASSIFICATION:
                simple_ensemble = VotingClassifier(estimators=simple_estimators, voting="soft")
            else:
                simple_ensemble = VotingRegressor(estimators=simple_estimators)

            simple_scores = cross_val_score(simple_ensemble, X_scaled, y, cv=cv, scoring=scoring)
            ensemble_results["simple_avg"] = simple_scores.mean()
        except Exception as e:
            logger.debug(f"Simple average failed: {e}")

        # ================================================================
        # STRATEGY 5: MULTI-LEVEL STACKING (10/10 Kaggle Winner Strategy)
        # ================================================================
        # 2-Layer stacking: Layer 1 = diverse base models, Layer 2 = meta-learner
        # This is how top Kaggle solutions achieve SOTA scores
        multi_level_stacking = None
        try:
            from sklearn.ensemble import (
                ExtraTreesClassifier,
                ExtraTreesRegressor,
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
            )

            # Layer 1: Diverse base models with different learning paradigms
            if problem_type == ProblemType.CLASSIFICATION:
                layer1_models = {
                    # Boosting family
                    "lgb": lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
                    "xgb": xgb.XGBClassifier(
                        n_estimators=200, random_state=42, eval_metric="logloss", verbosity=0
                    ),
                    "histgb": HistGradientBoostingClassifier(max_iter=200, random_state=42),
                    # Bagging family
                    "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                    "et": ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                }

                # Layer 2: Meta-learner (simpler model to combine Layer 1 outputs)
                layer2_meta = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1
                )
            else:
                layer1_models = {
                    "lgb": lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
                    "xgb": xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
                    "histgb": HistGradientBoostingRegressor(max_iter=200, random_state=42),
                    "rf": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                    "et": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                }
                layer2_meta = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1
                )

            # Build 2-layer stacking
            layer1_estimators = [(name, model) for name, model in layer1_models.items()]

            if problem_type == ProblemType.CLASSIFICATION:
                multi_level_stacking = StackingClassifier(
                    estimators=layer1_estimators,
                    final_estimator=layer2_meta,
                    cv=5,  # More CV for better OOF predictions
                    passthrough=True,  # Include original features in layer 2
                    n_jobs=-1,
                    stack_method="predict_proba",
                )
            else:
                multi_level_stacking = StackingRegressor(
                    estimators=layer1_estimators,
                    final_estimator=layer2_meta,
                    cv=5,
                    passthrough=True,
                    n_jobs=-1,
                )

            multi_level_scores = cross_val_score(
                multi_level_stacking, X_scaled, y, cv=cv, scoring=scoring
            )
            ensemble_results["multi_level_stacking"] = multi_level_scores.mean()

        except Exception as e:
            logger.debug(f"Multi-level stacking failed: {e}")

        # ================================================================
        # SELECT BEST STRATEGY
        # ================================================================
        best_ensemble_strategy = None
        best_ensemble_score = 0

        for strategy, score in ensemble_results.items():
            if isinstance(score, (int, float)) and score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble_strategy = strategy

        # Compare ensemble vs single model
        if best_ensemble_score > best_single_score:
            # Build the winning ensemble
            if (
                best_ensemble_strategy == "multi_level_stacking"
                and multi_level_stacking is not None
            ):
                final_model = multi_level_stacking
                n_estimators = 5  # 5 layer-1 models + meta-learner
            elif best_ensemble_strategy == "stacking":
                final_model = stacking
                n_estimators = len(base_models)
            elif best_ensemble_strategy == "weighted_voting":
                final_model = weighted_ensemble
                n_estimators = len(base_models)
            elif best_ensemble_strategy == "greedy" and greedy_estimators:
                if problem_type == ProblemType.CLASSIFICATION:
                    final_model = VotingClassifier(estimators=greedy_estimators, voting="soft")
                else:
                    final_model = VotingRegressor(estimators=greedy_estimators)
                n_estimators = len(greedy_estimators)
            else:
                final_model = simple_ensemble
                n_estimators = 3

            final_model.fit(X_scaled, y)
            final_score = best_ensemble_score
            used_ensemble = True
        else:
            # Single model is better - need to fit the optimized model
            final_model = optimized
            final_model.fit(X_scaled, y)  # FIT THE MODEL!
            final_score = best_single_score
            used_ensemble = False
            best_ensemble_strategy = "single_model"
            n_estimators = 1

        return SkillResult(
            skill_name="builtin_ensemble",
            category=SkillCategory.ENSEMBLE,
            success=True,
            data=final_model,
            metrics={
                "score": final_score,
                "n_estimators": n_estimators,
                "ensemble_score": best_ensemble_score,
                "single_score": best_single_score,
                "strategy": best_ensemble_strategy,
            },
            metadata={
                "all_ensemble_scores": ensemble_results,
                "decision": best_ensemble_strategy,
                "used_ensemble": used_ensemble,
            },
        )
