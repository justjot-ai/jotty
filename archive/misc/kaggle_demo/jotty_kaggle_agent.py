"""
Jotty V2 Kaggle Competition Agent
==================================

Multi-agent system that outperforms typical data science workflows.

Agents:
1. DataProfiler - Analyzes data quality, distributions, correlations
2. FeatureEngineer - Creates advanced features based on domain knowledge
3. ModelExplorer - Tests multiple algorithms with hyperparameter tuning
4. Ensembler - Combines models for optimal predictions

Target: Titanic (benchmark ~80% accuracy, top ~84%)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add Jotty to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent's work."""
    agent: str
    success: bool
    output: Any
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0


class DataProfilerAgent:
    """
    Agent 1: Data Profiling & Quality Analysis

    Analyzes data quality, identifies patterns, and recommends preprocessing.
    """

    name = "DataProfiler"

    async def analyze(self, df: pd.DataFrame, target: str = None) -> AgentResult:
        start = time.time()
        logger.info(f"[{self.name}] Starting data profiling...")

        profile = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing': df.isnull().sum().to_dict(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
            'unique_counts': df.nunique().to_dict(),
        }

        # Identify column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        profile['numeric_columns'] = numeric_cols
        profile['categorical_columns'] = categorical_cols

        # Statistical summary for numeric
        if numeric_cols:
            profile['numeric_stats'] = df[numeric_cols].describe().to_dict()

        # Target analysis if provided
        if target and target in df.columns:
            profile['target_distribution'] = df[target].value_counts().to_dict()
            profile['target_balance'] = df[target].value_counts(normalize=True).to_dict()

        # Correlation analysis
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            profile['high_correlations'] = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        profile['high_correlations'].append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'corr': round(corr_matrix.iloc[i, j], 3)
                        })

        logger.info(f"[{self.name}] Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical columns")
        logger.info(f"[{self.name}] Missing data: {sum(1 for v in profile['missing'].values() if v > 0)} columns affected")

        return AgentResult(
            agent=self.name,
            success=True,
            output=profile,
            execution_time=time.time() - start
        )


class FeatureEngineerAgent:
    """
    Agent 2: Advanced Feature Engineering

    Creates domain-specific features to boost model performance.
    """

    name = "FeatureEngineer"

    async def engineer_features(
        self,
        df: pd.DataFrame,
        profile: Dict,
        target: str = None
    ) -> AgentResult:
        start = time.time()
        logger.info(f"[{self.name}] Starting feature engineering...")

        df = df.copy()
        new_features = []

        # ===== Titanic-specific Feature Engineering =====

        # 1. Title extraction from Name
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            # Consolidate rare titles
            title_mapping = {
                'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
                'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
                'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
                'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
                'Jonkheer': 'Rare', 'Dona': 'Rare'
            }
            df['Title'] = df['Title'].replace(title_mapping)
            new_features.append('Title')
            logger.info(f"[{self.name}] Created Title feature: {df['Title'].unique().tolist()}")

        # 2. Family size
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            df['FamilyType'] = pd.cut(df['FamilySize'],
                                       bins=[0, 1, 4, 20],
                                       labels=['Alone', 'Small', 'Large'])
            new_features.extend(['FamilySize', 'IsAlone', 'FamilyType'])
            logger.info(f"[{self.name}] Created family features")

        # 3. Age groups with intelligent binning
        if 'Age' in df.columns:
            # Fill missing ages based on Title
            if 'Title' in df.columns:
                age_by_title = df.groupby('Title')['Age'].median()
                for title in df['Title'].unique():
                    mask = (df['Age'].isnull()) & (df['Title'] == title)
                    if mask.any() and title in age_by_title.index:
                        df.loc[mask, 'Age'] = age_by_title[title]

            # Fill remaining with median
            df['Age'].fillna(df['Age'].median(), inplace=True)

            df['AgeGroup'] = pd.cut(df['Age'],
                                     bins=[0, 12, 18, 35, 60, 100],
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
            df['IsChild'] = (df['Age'] < 12).astype(int)
            new_features.extend(['AgeGroup', 'IsChild'])
            logger.info(f"[{self.name}] Created age features")

        # 4. Fare per person
        if 'Fare' in df.columns:
            df['Fare'].fillna(df['Fare'].median(), inplace=True)
            if 'FamilySize' in df.columns:
                df['FarePerPerson'] = df['Fare'] / df['FamilySize']
            df['FareGroup'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
            new_features.extend(['FarePerPerson', 'FareGroup'] if 'FamilySize' in df.columns else ['FareGroup'])
            logger.info(f"[{self.name}] Created fare features")

        # 5. Cabin features
        if 'Cabin' in df.columns:
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            df['CabinDeck'] = df['Cabin'].str[0].fillna('Unknown')
            new_features.extend(['HasCabin', 'CabinDeck'])
            logger.info(f"[{self.name}] Created cabin features")

        # 6. Embarked
        if 'Embarked' in df.columns:
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

        # 7. Interaction features
        if 'Sex' in df.columns and 'Pclass' in df.columns:
            df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
            new_features.append('Sex_Pclass')

        if 'Age' in df.columns and 'Pclass' in df.columns:
            df['Age_Pclass'] = df['Age'] * df['Pclass']
            new_features.append('Age_Pclass')

        logger.info(f"[{self.name}] Created {len(new_features)} new features: {new_features}")

        return AgentResult(
            agent=self.name,
            success=True,
            output={'df': df, 'new_features': new_features},
            execution_time=time.time() - start
        )


class ModelExplorerAgent:
    """
    Agent 3: Model Exploration & Hyperparameter Tuning

    Tests multiple algorithms and finds optimal hyperparameters.
    """

    name = "ModelExplorer"

    def __init__(self):
        self.models = {}
        self.cv_scores = {}

    async def explore_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5
    ) -> AgentResult:
        start = time.time()
        logger.info(f"[{self.name}] Starting model exploration...")

        # Prepare data
        X_processed = self._preprocess(X)

        # Define models to test
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                                    min_samples_split=4, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                            learning_rate=0.1, random_state=42),
            'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        }

        # Try XGBoost and LightGBM if available
        try:
            import xgboost as xgb
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                use_label_encoder=False, eval_metric='logloss'
            )
        except ImportError:
            pass

        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )
        except ImportError:
            pass

        # Cross-validation
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        results = {}
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_processed, y, cv=cv, scoring='accuracy')
                results[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
                logger.info(f"[{self.name}] {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

                # Train on full data for later use
                model.fit(X_processed, y)
                self.models[name] = model
                self.cv_scores[name] = scores.mean()
            except Exception as e:
                logger.warning(f"[{self.name}] {name} failed: {e}")

        # Sort by performance
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True))
        best_model = list(sorted_results.keys())[0]
        best_score = sorted_results[best_model]['mean']

        logger.info(f"[{self.name}] Best single model: {best_model} ({best_score:.4f})")

        return AgentResult(
            agent=self.name,
            success=True,
            output={
                'results': sorted_results,
                'best_model': best_model,
                'best_score': best_score,
                'trained_models': self.models
            },
            metrics={'best_cv_score': best_score},
            execution_time=time.time() - start
        )

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features for modeling."""
        X = X.copy()

        # Encode categoricals
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Scale numerics
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled


class EnsemblerAgent:
    """
    Agent 4: Model Ensemble & Stacking

    Combines multiple models for optimal predictions.
    """

    name = "Ensembler"

    async def create_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_results: Dict
    ) -> AgentResult:
        start = time.time()
        logger.info(f"[{self.name}] Creating ensemble...")

        # Prepare data
        X_processed = self._preprocess(X)

        trained_models = model_results.get('trained_models', {})
        cv_results = model_results.get('results', {})

        # Select top models for ensemble (top 5 by CV score)
        top_models = sorted(cv_results.items(), key=lambda x: x[1]['mean'], reverse=True)[:5]
        top_model_names = [name for name, _ in top_models]

        logger.info(f"[{self.name}] Top models for ensemble: {top_model_names}")

        # 1. Voting Ensemble (soft voting)
        voting_estimators = [(name, trained_models[name]) for name in top_model_names
                            if name in trained_models and hasattr(trained_models[name], 'predict_proba')]

        if len(voting_estimators) >= 2:
            voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
            voting_scores = cross_val_score(voting_clf, X_processed, y, cv=5, scoring='accuracy')
            voting_mean = voting_scores.mean()
            logger.info(f"[{self.name}] Voting Ensemble: {voting_mean:.4f} (+/- {voting_scores.std():.4f})")
        else:
            voting_mean = 0
            voting_clf = None

        # 2. Stacking Ensemble
        base_estimators = [(name, trained_models[name]) for name in top_model_names[:4]
                          if name in trained_models]

        if len(base_estimators) >= 2:
            stacking_clf = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5
            )
            stacking_scores = cross_val_score(stacking_clf, X_processed, y, cv=5, scoring='accuracy')
            stacking_mean = stacking_scores.mean()
            logger.info(f"[{self.name}] Stacking Ensemble: {stacking_mean:.4f} (+/- {stacking_scores.std():.4f})")
        else:
            stacking_mean = 0
            stacking_clf = None

        # 3. Weighted Average Ensemble (manual)
        # Get out-of-fold predictions from top models
        best_ensemble_score = max(voting_mean, stacking_mean)
        best_ensemble_type = 'Voting' if voting_mean >= stacking_mean else 'Stacking'
        best_ensemble = voting_clf if voting_mean >= stacking_mean else stacking_clf

        # Compare with best single model
        best_single = max(cv_results.values(), key=lambda x: x['mean'])['mean']

        improvement = (best_ensemble_score - best_single) * 100
        logger.info(f"[{self.name}] Ensemble improvement: {improvement:+.2f}% over best single model")

        # Train final ensemble
        if best_ensemble:
            best_ensemble.fit(X_processed, y)

        final_score = max(best_ensemble_score, best_single)

        return AgentResult(
            agent=self.name,
            success=True,
            output={
                'best_ensemble_type': best_ensemble_type,
                'best_ensemble_score': best_ensemble_score,
                'voting_score': voting_mean,
                'stacking_score': stacking_mean,
                'best_single_score': best_single,
                'final_model': best_ensemble if best_ensemble_score > best_single else None,
                'final_score': final_score
            },
            metrics={'final_score': final_score},
            execution_time=time.time() - start
        )

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features for modeling."""
        X = X.copy()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        scaler = StandardScaler()
        return scaler.fit_transform(X)


class JottyKaggleSwarm:
    """
    Jotty V2 Multi-Agent Swarm for Kaggle Competitions

    Orchestrates multiple specialized agents to achieve competitive performance.
    """

    def __init__(self):
        self.agents = {
            'profiler': DataProfilerAgent(),
            'engineer': FeatureEngineerAgent(),
            'explorer': ModelExplorerAgent(),
            'ensembler': EnsemblerAgent(),
        }
        self.results = {}

    async def run(self, data_path: str, target: str = 'Survived') -> Dict[str, Any]:
        """
        Run full Kaggle competition pipeline.

        Args:
            data_path: Path to training data CSV
            target: Target column name

        Returns:
            Final results with model performance
        """
        total_start = time.time()

        logger.info("=" * 60)
        logger.info("JOTTY V2 KAGGLE COMPETITION AGENT")
        logger.info("=" * 60)

        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # ===== Agent 1: Data Profiling =====
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 1: DATA PROFILING")
        logger.info("=" * 40)

        profile_result = await self.agents['profiler'].analyze(df, target)
        self.results['profile'] = profile_result

        # ===== Agent 2: Feature Engineering =====
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 2: FEATURE ENGINEERING")
        logger.info("=" * 40)

        engineer_result = await self.agents['engineer'].engineer_features(
            df, profile_result.output, target
        )
        self.results['engineering'] = engineer_result

        df_engineered = engineer_result.output['df']

        # Prepare features for modeling
        drop_cols = [target, 'PassengerId', 'Name', 'Ticket', 'Cabin']
        feature_cols = [c for c in df_engineered.columns if c not in drop_cols]

        X = df_engineered[feature_cols]
        y = df_engineered[target]

        logger.info(f"Features for modeling: {len(feature_cols)}")

        # ===== Agent 3: Model Exploration =====
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 3: MODEL EXPLORATION")
        logger.info("=" * 40)

        explore_result = await self.agents['explorer'].explore_models(X, y)
        self.results['exploration'] = explore_result

        # ===== Agent 4: Ensemble =====
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 4: ENSEMBLE CREATION")
        logger.info("=" * 40)

        ensemble_result = await self.agents['ensembler'].create_ensemble(
            X, y, explore_result.output
        )
        self.results['ensemble'] = ensemble_result

        # ===== Final Summary =====
        total_time = time.time() - total_start

        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)

        final_score = ensemble_result.output['final_score']
        best_single = explore_result.output['best_score']
        best_model = explore_result.output['best_model']

        logger.info(f"Best Single Model: {best_model} ({best_single:.4f})")
        logger.info(f"Best Ensemble: {ensemble_result.output['best_ensemble_type']} ({ensemble_result.output['best_ensemble_score']:.4f})")
        logger.info(f"FINAL CV SCORE: {final_score:.4f} ({final_score*100:.2f}%)")
        logger.info(f"Total execution time: {total_time:.2f}s")

        # Benchmark comparison
        logger.info("\n" + "-" * 40)
        logger.info("BENCHMARK COMPARISON (Titanic Leaderboard)")
        logger.info("-" * 40)
        logger.info(f"Top 1%:    ~84% accuracy")
        logger.info(f"Top 10%:   ~82% accuracy")
        logger.info(f"Top 25%:   ~80% accuracy")
        logger.info(f"Median:    ~77% accuracy")
        logger.info(f"OUR SCORE: {final_score*100:.2f}% accuracy")

        if final_score >= 0.84:
            logger.info(">> TOP 1% PERFORMANCE!")
        elif final_score >= 0.82:
            logger.info(">> TOP 10% PERFORMANCE!")
        elif final_score >= 0.80:
            logger.info(">> TOP 25% PERFORMANCE!")
        elif final_score >= 0.77:
            logger.info(">> ABOVE MEDIAN!")

        return {
            'final_score': final_score,
            'best_single_model': best_model,
            'best_single_score': best_single,
            'ensemble_type': ensemble_result.output['best_ensemble_type'],
            'ensemble_score': ensemble_result.output['best_ensemble_score'],
            'total_time': total_time,
            'features_created': len(engineer_result.output['new_features']),
            'models_tested': len(explore_result.output['results']),
        }


async def main():
    """Run Jotty Kaggle Agent."""
    swarm = JottyKaggleSwarm()

    # Run on Titanic data
    data_path = Path(__file__).parent / "train.csv"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    results = await swarm.run(str(data_path), target='Survived')

    return results


if __name__ == "__main__":
    results = asyncio.run(main())
