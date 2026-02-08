"""
Jotty V2 Maximum Performance Mode
==================================

Takes learnings from autonomous swarm and combines best techniques.
Target: Push beyond 85% to top leaderboard ranks.
"""

import asyncio
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class MaxPerformanceAgent:
    """
    Combines all winning techniques discovered by autonomous swarm.
    """

    def __init__(self):
        self.best_score = 0
        self.models = {}

    async def run(self, data_path: str) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("JOTTY V2 MAXIMUM PERFORMANCE MODE")
        logger.info("=" * 70)

        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded: {df.shape}")

        # ====================================================================
        # PHASE 1: Research-informed Feature Engineering
        # ====================================================================
        logger.info("\n[Phase 1] Feature Engineering (research-informed)")

        df = self._engineer_features(df)
        logger.info(f"Features after engineering: {df.shape[1]}")

        # Prepare X, y
        target = 'Survived'
        drop_cols = [target, 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Surname']
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols].copy()
        y = df[target]

        logger.info(f"Training features: {len(feature_cols)}")

        # ====================================================================
        # PHASE 2: Multi-Model Optuna Optimization
        # ====================================================================
        logger.info("\n[Phase 2] Multi-Model Optuna Optimization")

        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        # Encode categoricals
        X_encoded = X.copy()
        label_encoders = {}
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Optimize multiple models
        best_models = await self._optimize_all_models(X_scaled, y)

        # ====================================================================
        # PHASE 3: Smart Ensemble with Weighted Blending
        # ====================================================================
        logger.info("\n[Phase 3] Smart Weighted Ensemble")

        final_score = await self._create_weighted_ensemble(X_scaled, y, best_models)

        # ====================================================================
        # PHASE 4: CatBoost Native (handles categoricals better)
        # ====================================================================
        logger.info("\n[Phase 4] CatBoost with Native Categoricals")

        catboost_score = await self._train_catboost_native(X, y)

        # ====================================================================
        # PHASE 5: Neural Network + Tree Blend
        # ====================================================================
        logger.info("\n[Phase 5] Neural Network Blend")

        nn_score = await self._train_neural_blend(X_scaled, y, best_models)

        # ====================================================================
        # FINAL: Best of All Approaches
        # ====================================================================
        all_scores = {
            'optuna_ensemble': final_score,
            'catboost_native': catboost_score,
            'neural_blend': nn_score,
        }

        best_approach = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_approach]

        logger.info("\n" + "=" * 70)
        logger.info("FINAL RESULTS")
        logger.info("=" * 70)
        for approach, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
            marker = " ← BEST" if approach == best_approach else ""
            logger.info(f"  {approach}: {score:.4f} ({score:.2%}){marker}")

        logger.info(f"\nFINAL SCORE: {best_score:.4f} ({best_score:.2%})")

        # Leaderboard comparison
        logger.info("\n" + "-" * 40)
        if best_score >= 0.87:
            logger.info(">> TOP 0.1% - GRANDMASTER LEVEL!")
        elif best_score >= 0.86:
            logger.info(">> TOP 0.5% - EXPERT LEVEL!")
        elif best_score >= 0.85:
            logger.info(">> TOP 1%!")
        elif best_score >= 0.84:
            logger.info(">> TOP 5%")

        return {'final_score': best_score, 'all_scores': all_scores}

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine all winning feature engineering techniques."""

        # Title extraction (proven +1%)
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
            'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
            'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
            'Jonkheer': 'Rare', 'Dona': 'Rare'
        }
        df['Title'] = df['Title'].replace(title_map)

        # Family (proven helpful)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['FamilySizeBin'] = pd.cut(df['FamilySize'], bins=[0, 1, 4, 20], labels=[0, 1, 2])

        # Age imputation using Title (better than median)
        age_by_title = df.groupby('Title')['Age'].median()
        for title in df['Title'].unique():
            mask = (df['Age'].isnull()) & (df['Title'] == title)
            if mask.any() and title in age_by_title.index:
                df.loc[mask, 'Age'] = age_by_title[title]
        df['Age'].fillna(df['Age'].median(), inplace=True)

        # Age bins
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
        df['IsChild'] = (df['Age'] < 12).astype(int)
        df['IsElderly'] = (df['Age'] > 60).astype(int)

        # Fare
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['FarePerPerson'] = df['Fare'] / df['FamilySize']
        df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')

        # Cabin
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df['CabinDeck'] = df['Cabin'].str[0].fillna('U')

        # Embarked
        df['Embarked'].fillna('S', inplace=True)

        # Interaction features (selective - only proven ones)
        df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
        df['FamilySize_Pclass'] = df['FamilySize'] * df['Pclass']

        # Ticket grouping (only group size, not prefix)
        ticket_counts = df['Ticket'].value_counts()
        df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
        df['IsSharedTicket'] = (df['TicketGroupSize'] > 1).astype(int)

        # Surname frequency (family survival groups)
        df['Surname'] = df['Name'].str.split(',').str[0]
        surname_counts = df['Surname'].value_counts()
        df['SurnameFreq'] = df['Surname'].map(surname_counts)

        return df

    async def _optimize_all_models(self, X: np.ndarray, y: pd.Series) -> Dict[str, Any]:
        """Optimize multiple models with Optuna."""
        import optuna
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_models = {}

        # XGBoost optimization
        logger.info("  Optimizing XGBoost...")

        def xgb_objective(trial):
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
            return cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
        xgb_best = xgb.XGBClassifier(**study.best_params, random_state=42,
                                      use_label_encoder=False, eval_metric='logloss')
        xgb_best.fit(X, y)
        xgb_score = study.best_value
        best_models['xgb'] = {'model': xgb_best, 'score': xgb_score}
        logger.info(f"    XGBoost: {xgb_score:.4f}")

        # LightGBM optimization
        logger.info("  Optimizing LightGBM...")

        def lgb_objective(trial):
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
            return cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(lgb_objective, n_trials=50, show_progress_bar=False)
        lgb_best = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
        lgb_best.fit(X, y)
        lgb_score = study.best_value
        best_models['lgb'] = {'model': lgb_best, 'score': lgb_score}
        logger.info(f"    LightGBM: {lgb_score:.4f}")

        # GradientBoosting optimization
        logger.info("  Optimizing GradientBoosting...")

        def gb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
            return cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(gb_objective, n_trials=30, show_progress_bar=False)
        gb_best = GradientBoostingClassifier(**study.best_params, random_state=42)
        gb_best.fit(X, y)
        gb_score = study.best_value
        best_models['gb'] = {'model': gb_best, 'score': gb_score}
        logger.info(f"    GradientBoosting: {gb_score:.4f}")

        # ExtraTrees optimization
        logger.info("  Optimizing ExtraTrees...")

        def et_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = ExtraTreesClassifier(**params)
            return cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(et_objective, n_trials=30, show_progress_bar=False)
        et_best = ExtraTreesClassifier(**study.best_params, random_state=42)
        et_best.fit(X, y)
        et_score = study.best_value
        best_models['et'] = {'model': et_best, 'score': et_score}
        logger.info(f"    ExtraTrees: {et_score:.4f}")

        return best_models

    async def _create_weighted_ensemble(
        self,
        X: np.ndarray,
        y: pd.Series,
        best_models: Dict[str, Any]
    ) -> float:
        """Create weighted ensemble with optimized weights."""
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.linear_model import LogisticRegression
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Get out-of-fold predictions
        n_models = len(best_models)
        oof_preds = np.zeros((len(X), n_models))
        model_names = list(best_models.keys())

        for i, name in enumerate(model_names):
            model = best_models[name]['model']
            for train_idx, val_idx in cv.split(X, y):
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X[train_idx], y.iloc[train_idx])
                if hasattr(model_clone, 'predict_proba'):
                    oof_preds[val_idx, i] = model_clone.predict_proba(X[val_idx])[:, 1]
                else:
                    oof_preds[val_idx, i] = model_clone.predict(X[val_idx])

        # Optimize weights
        def weight_objective(trial):
            weights = [trial.suggest_float(f'w_{i}', 0, 1) for i in range(n_models)]
            weights = np.array(weights) / sum(weights)  # Normalize

            blended = np.average(oof_preds, axis=1, weights=weights)
            preds = (blended > 0.5).astype(int)
            return (preds == y).mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(weight_objective, n_trials=100, show_progress_bar=False)

        # Get best weights
        best_weights = [study.best_params[f'w_{i}'] for i in range(n_models)]
        best_weights = np.array(best_weights) / sum(best_weights)

        logger.info("  Optimal weights:")
        for name, weight in zip(model_names, best_weights):
            logger.info(f"    {name}: {weight:.3f}")

        score = study.best_value
        logger.info(f"  Weighted ensemble: {score:.4f}")

        return score

    async def _train_catboost_native(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Train CatBoost with native categorical handling."""
        try:
            from catboost import CatBoostClassifier, Pool
            from sklearn.model_selection import StratifiedKFold

            X_cat = X.copy()
            cat_cols = X_cat.select_dtypes(include=['object', 'category']).columns.tolist()

            # Fill NaN
            for col in cat_cols:
                X_cat[col] = X_cat[col].fillna('Unknown').astype(str)
            for col in X_cat.select_dtypes(include=['number']).columns:
                X_cat[col] = X_cat[col].fillna(X_cat[col].median())

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []

            for train_idx, val_idx in cv.split(X_cat, y):
                X_train, X_val = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3,
                    cat_features=cat_cols,
                    random_state=42,
                    verbose=False,
                    early_stopping_rounds=50
                )

                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                preds = model.predict(X_val)
                score = (preds == y_val).mean()
                scores.append(score)

            final_score = np.mean(scores)
            logger.info(f"  CatBoost native: {final_score:.4f} (+/- {np.std(scores):.4f})")
            return final_score

        except Exception as e:
            logger.warning(f"  CatBoost failed: {e}")
            return 0.0

    async def _train_neural_blend(
        self,
        X: np.ndarray,
        y: pd.Series,
        best_models: Dict[str, Any]
    ) -> float:
        """Blend tree models with neural network meta-learner."""
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neural_network import MLPClassifier
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Get OOF predictions from tree models
        n_models = len(best_models)
        oof_preds = np.zeros((len(X), n_models))

        for i, (name, info) in enumerate(best_models.items()):
            model = info['model']
            for train_idx, val_idx in cv.split(X, y):
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X[train_idx], y.iloc[train_idx])
                if hasattr(model_clone, 'predict_proba'):
                    oof_preds[val_idx, i] = model_clone.predict_proba(X[val_idx])[:, 1]
                else:
                    oof_preds[val_idx, i] = model_clone.predict(X[val_idx])

        # Combine original features with OOF predictions
        X_meta = np.hstack([X, oof_preds])

        # Optimize neural network meta-learner
        def nn_objective(trial):
            hidden_size = trial.suggest_int('hidden_size', 50, 200)
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_layers = tuple([hidden_size] * n_layers)

            model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                learning_rate_init=trial.suggest_float('lr', 0.0001, 0.01),
                alpha=trial.suggest_float('alpha', 0.0001, 0.1),
                max_iter=500,
                random_state=42
            )
            return cross_val_score(model, X_meta, y, cv=cv, scoring='accuracy').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(nn_objective, n_trials=30, show_progress_bar=False)

        score = study.best_value
        logger.info(f"  Neural blend: {score:.4f}")
        return score


async def main():
    agent = MaxPerformanceAgent()
    data_path = Path(__file__).parent / "train.csv"

    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return

    results = await agent.run(str(data_path))
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
