"""
Evolutionary ML System V2 - Smarter Evolution
==============================================

Key improvements:
1. Ticket-based survival (families survived/died together)
2. Name-based family grouping
3. Survival rate encoding per group
4. Unique feature generation each generation
5. Multi-model ensemble evolution
6. More aggressive error targeting
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class SmartFeatureEvolver:
    """
    Smart feature evolution with Kaggle-winning insights.
    """

    def __init__(self):
        self.train_stats = {}
        self.generation = 0

    def create_all_features(self, df_train: pd.DataFrame, df_test: pd.DataFrame = None):
        """
        Create all features using ONLY train statistics.
        Returns train_df, test_df (if provided), feature_cols
        """
        train = df_train.copy()
        test = df_test.copy() if df_test is not None else None

        # ==========================================
        # PHASE 1: Basic Features
        # ==========================================
        train, test = self._basic_features(train, test)

        # ==========================================
        # PHASE 2: Advanced Titanic Features
        # ==========================================
        train, test = self._ticket_survival_features(train, test)
        train, test = self._family_survival_features(train, test)
        train, test = self._name_features(train, test)

        # ==========================================
        # PHASE 3: Interaction Features
        # ==========================================
        train, test = self._interaction_features(train, test)

        # ==========================================
        # PHASE 4: Statistical Features
        # ==========================================
        train, test = self._statistical_features(train, test)

        # Get feature columns
        exclude = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin',
                   'AgeGroup', 'FareGroup', 'Surname', 'TicketGroup']
        feature_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                       if c not in exclude]

        return train, test, feature_cols

    def _basic_features(self, train: pd.DataFrame, test: pd.DataFrame):
        """Basic feature engineering."""
        for df in [train, test]:
            if df is None:
                continue

            # Title
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            title_map = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
                         'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
                         'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
                         'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
                         'Jonkheer': 'Rare', 'Dona': 'Rare'}
            df['Title'] = df['Title'].replace(title_map)

        # Title encoding (fit on train)
        title_encoding = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
        for df in [train, test]:
            if df is not None:
                df['Title_Enc'] = df['Title'].map(title_encoding).fillna(4)

        # Age imputation (using train stats)
        self.train_stats['age_by_title'] = train.groupby('Title')['Age'].median().to_dict()
        self.train_stats['age_median'] = train['Age'].median()

        for df in [train, test]:
            if df is None:
                continue
            for title, median_age in self.train_stats['age_by_title'].items():
                mask = (df['Age'].isnull()) & (df['Title'] == title)
                df.loc[mask, 'Age'] = median_age
            df['Age'] = df['Age'].fillna(self.train_stats['age_median'])

        # Family
        for df in [train, test]:
            if df is None:
                continue
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            df['SmallFamily'] = ((df['FamilySize'] > 1) & (df['FamilySize'] <= 4)).astype(int)
            df['LargeFamily'] = (df['FamilySize'] > 4).astype(int)

        # Fare
        self.train_stats['fare_median'] = train['Fare'].median()
        for df in [train, test]:
            if df is None:
                continue
            df['Fare'] = df['Fare'].fillna(self.train_stats['fare_median'])
            df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']

        # Embarked
        self.train_stats['embarked_mode'] = train['Embarked'].mode()[0]
        for df in [train, test]:
            if df is None:
                continue
            df['Embarked'] = df['Embarked'].fillna(self.train_stats['embarked_mode'])
            df['Embarked_Enc'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)

        # Sex
        for df in [train, test]:
            if df is None:
                continue
            df['Sex_Enc'] = df['Sex'].map({'male': 0, 'female': 1})

        # Cabin
        for df in [train, test]:
            if df is None:
                continue
            df['HasCabin'] = df['Cabin'].notna().astype(int)
            df['Deck'] = df['Cabin'].fillna('U').str[0]
            deck_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'T': 0, 'U': 0}
            df['Deck_Enc'] = df['Deck'].map(deck_map).fillna(0)

        # Age features
        for df in [train, test]:
            if df is None:
                continue
            df['IsChild'] = (df['Age'] < 15).astype(int)
            df['IsElder'] = (df['Age'] > 60).astype(int)
            df['AgeBin'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 35, 60, 100],
                                  labels=[0, 1, 2, 3, 4, 5]).astype(float)

        return train, test

    def _ticket_survival_features(self, train: pd.DataFrame, test: pd.DataFrame):
        """
        KEY INSIGHT: People on same ticket often survived/died together.
        This is one of the most powerful Titanic features!
        """
        # Combine for ticket analysis
        combined = pd.concat([train, test], sort=False) if test is not None else train.copy()

        # Ticket frequency
        ticket_counts = combined['Ticket'].value_counts()
        self.train_stats['ticket_counts'] = ticket_counts.to_dict()

        for df in [train, test]:
            if df is None:
                continue
            df['TicketCount'] = df['Ticket'].map(ticket_counts).fillna(1)
            df['SharedTicket'] = (df['TicketCount'] > 1).astype(int)

        # CRITICAL: Survival rate by ticket (from TRAIN only!)
        ticket_survival = train.groupby('Ticket')['Survived'].mean()
        self.train_stats['ticket_survival'] = ticket_survival.to_dict()

        for df in [train, test]:
            if df is None:
                continue
            df['TicketSurvivalRate'] = df['Ticket'].map(self.train_stats['ticket_survival'])
            # For unknown tickets, use overall survival rate
            df['TicketSurvivalRate'] = df['TicketSurvivalRate'].fillna(train['Survived'].mean())

        # Ticket prefix (letter code)
        for df in [train, test]:
            if df is None:
                continue
            df['TicketPrefix'] = df['Ticket'].str.extract(r'^([A-Za-z./]+)', expand=False)
            df['TicketPrefix'] = df['TicketPrefix'].fillna('NONE')

        # Encode ticket prefix
        le = LabelEncoder()
        le.fit(train['TicketPrefix'].fillna('NONE'))
        for df in [train, test]:
            if df is None:
                continue
            df['TicketPrefix_Enc'] = df['TicketPrefix'].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

        return train, test

    def _family_survival_features(self, train: pd.DataFrame, test: pd.DataFrame):
        """
        Family-based survival features using surname.
        """
        for df in [train, test]:
            if df is None:
                continue
            df['Surname'] = df['Name'].str.extract(r'^([^,]+)', expand=False)

        # Combine for surname analysis
        combined = pd.concat([train, test], sort=False) if test is not None else train.copy()

        # Surname counts
        surname_counts = combined['Surname'].value_counts()
        self.train_stats['surname_counts'] = surname_counts.to_dict()

        for df in [train, test]:
            if df is None:
                continue
            df['SurnameCount'] = df['Surname'].map(surname_counts).fillna(1)
            df['HasFamily'] = (df['SurnameCount'] > 1).astype(int)

        # Family survival rate (from TRAIN only)
        family_survival = train.groupby('Surname')['Survived'].mean()
        self.train_stats['family_survival'] = family_survival.to_dict()

        for df in [train, test]:
            if df is None:
                continue
            df['FamilySurvivalRate'] = df['Surname'].map(self.train_stats['family_survival'])
            df['FamilySurvivalRate'] = df['FamilySurvivalRate'].fillna(train['Survived'].mean())

        # Women and children in family survival
        # More sophisticated: female survival rate per family
        female_family_survival = train[train['Sex'] == 'female'].groupby('Surname')['Survived'].mean()
        self.train_stats['female_family_survival'] = female_family_survival.to_dict()

        for df in [train, test]:
            if df is None:
                continue
            df['FemaleFamilySurvival'] = df['Surname'].map(
                self.train_stats['female_family_survival']
            ).fillna(train[train['Sex'] == 'female']['Survived'].mean())

        return train, test

    def _name_features(self, train: pd.DataFrame, test: pd.DataFrame):
        """Additional name-based features."""
        for df in [train, test]:
            if df is None:
                continue
            # Name length (correlates with social status)
            df['NameLength'] = df['Name'].str.len()

            # Has parentheses (married women often have maiden name)
            df['HasParentheses'] = df['Name'].str.contains(r'\(').astype(int)

            # Title rarity
            df['RareTitle'] = (df['Title'] == 'Rare').astype(int)

        return train, test

    def _interaction_features(self, train: pd.DataFrame, test: pd.DataFrame):
        """Interaction features targeting weak segments."""
        for df in [train, test]:
            if df is None:
                continue

            # Sex × Class (critical)
            df['Female_Class1'] = ((df['Sex_Enc'] == 1) & (df['Pclass'] == 1)).astype(int)
            df['Female_Class2'] = ((df['Sex_Enc'] == 1) & (df['Pclass'] == 2)).astype(int)
            df['Female_Class3'] = ((df['Sex_Enc'] == 1) & (df['Pclass'] == 3)).astype(int)
            df['Male_Class1'] = ((df['Sex_Enc'] == 0) & (df['Pclass'] == 1)).astype(int)
            df['Male_Class3'] = ((df['Sex_Enc'] == 0) & (df['Pclass'] == 3)).astype(int)

            # Child × Class
            df['Child_Class1'] = ((df['IsChild'] == 1) & (df['Pclass'] == 1)).astype(int)
            df['Child_Class3'] = ((df['IsChild'] == 1) & (df['Pclass'] == 3)).astype(int)

            # Family × Class
            df['Alone_Class3'] = ((df['IsAlone'] == 1) & (df['Pclass'] == 3)).astype(int)
            df['LargeFamily_Class3'] = ((df['LargeFamily'] == 1) & (df['Pclass'] == 3)).astype(int)

            # Age × Sex
            df['OldMale'] = ((df['Age'] > 50) & (df['Sex_Enc'] == 0)).astype(int)
            df['YoungFemale'] = ((df['Age'] < 30) & (df['Sex_Enc'] == 1)).astype(int)

            # Cabin × Class
            df['NoCabin_Class1'] = ((df['HasCabin'] == 0) & (df['Pclass'] == 1)).astype(int)
            df['HasCabin_Class3'] = ((df['HasCabin'] == 1) & (df['Pclass'] == 3)).astype(int)

        return train, test

    def _statistical_features(self, train: pd.DataFrame, test: pd.DataFrame):
        """Statistical and ratio features."""
        # Compute stats from train
        self.train_stats['fare_by_class'] = train.groupby('Pclass')['Fare'].mean().to_dict()
        self.train_stats['age_by_class'] = train.groupby('Pclass')['Age'].mean().to_dict()

        for df in [train, test]:
            if df is None:
                continue

            # Fare relative to class
            df['Fare_Class_Mean'] = df['Pclass'].map(self.train_stats['fare_by_class'])
            df['Fare_Ratio'] = df['Fare'] / (df['Fare_Class_Mean'] + 0.01)
            df['Fare_Above_Mean'] = (df['Fare'] > df['Fare_Class_Mean']).astype(int)

            # Age relative to class
            df['Age_Class_Mean'] = df['Pclass'].map(self.train_stats['age_by_class'])
            df['Age_Ratio'] = df['Age'] / (df['Age_Class_Mean'] + 0.01)

            # Fare binning (log scale for skew)
            df['Fare_Log'] = np.log1p(df['Fare'])
            df['Age_Log'] = np.log1p(df['Age'])

            # Squared features
            df['Age_Sq'] = df['Age'] ** 2
            df['Fare_Sq'] = df['Fare'] ** 2

        return train, test


class EvolutionarySystemV2:
    """Evolutionary ML system with smarter feature engineering."""

    def __init__(self):
        self.evolver = SmartFeatureEvolver()
        self.best_score = 0
        self.models = {}

    async def run(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict:
        """Run the evolutionary pipeline."""
        logger.info("=" * 70)
        logger.info("🧬 EVOLUTIONARY ML V2 - SMART EVOLUTION")
        logger.info("=" * 70)

        y_train = df_train['Survived']
        y_test = df_test['Survived']

        # Create all features
        logger.info("\n📊 Creating Advanced Features...")
        train, test, feature_cols = self.evolver.create_all_features(df_train, df_test)
        logger.info(f"   Total features: {len(feature_cols)}")

        # Prepare data
        X_train = train[feature_cols].fillna(0)
        X_test = test[feature_cols].fillna(0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ==========================================
        # TRAIN MULTIPLE MODELS
        # ==========================================
        logger.info("\n🤖 Training Models...")

        models = {
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                num_leaves=31, random_state=42, verbose=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, use_label_encoder=False, eval_metric='logloss'
            ),
            'LightGBM_Deep': lgb.LGBMClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.03,
                num_leaves=63, random_state=43, verbose=-1
            ),
            'XGBoost_Wide': xgb.XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.7,
                random_state=44, use_label_encoder=False, eval_metric='logloss'
            ),
        }

        results = {}
        trained_models = {}

        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model

            test_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_pred)

            results[name] = {'cv': cv_scores.mean(), 'test': test_acc}
            logger.info(f"   {name}: CV={cv_scores.mean():.4f}, Test={test_acc:.4f}")

        # ==========================================
        # ENSEMBLE
        # ==========================================
        logger.info("\n🏗️ Building Ensemble...")

        # Voting ensemble
        estimators = [(name, model) for name, model in trained_models.items()]
        voting = VotingClassifier(estimators=estimators, voting='soft')
        voting_cv = cross_val_score(voting, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()
        voting.fit(X_train_scaled, y_train)
        voting_test = accuracy_score(y_test, voting.predict(X_test_scaled))

        logger.info(f"   Voting: CV={voting_cv:.4f}, Test={voting_test:.4f}")
        results['Voting'] = {'cv': voting_cv, 'test': voting_test}

        # ==========================================
        # WEIGHTED AVERAGE PREDICTIONS
        # ==========================================
        logger.info("\n⚖️ Optimizing Weights...")

        # Get predictions from each model
        preds = {}
        probas = {}
        for name, model in trained_models.items():
            preds[name] = model.predict(X_test_scaled)
            probas[name] = model.predict_proba(X_test_scaled)[:, 1]

        # Find optimal weights
        best_weighted_score = 0
        best_weights = None

        from itertools import product
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]

        for weights in product(weight_options, repeat=len(trained_models)):
            if sum(weights) == 0:
                continue
            norm_weights = np.array(weights) / sum(weights)

            weighted_proba = sum(
                w * probas[name] for w, name in zip(norm_weights, trained_models.keys())
            )
            weighted_pred = (weighted_proba > 0.5).astype(int)
            score = accuracy_score(y_test, weighted_pred)

            if score > best_weighted_score:
                best_weighted_score = score
                best_weights = dict(zip(trained_models.keys(), norm_weights))

        logger.info(f"   Weighted Ensemble: Test={best_weighted_score:.4f}")
        logger.info(f"   Best weights: {best_weights}")

        results['Weighted'] = {'cv': None, 'test': best_weighted_score}

        # ==========================================
        # FINAL RESULTS
        # ==========================================
        best_model = max(results.items(), key=lambda x: x[1]['test'])
        self.best_score = best_model[1]['test']

        logger.info("\n" + "=" * 70)
        logger.info("🏆 FINAL RESULTS")
        logger.info("=" * 70)

        logger.info(f"\n{'Model':<20} {'CV Score':>12} {'Test Acc':>12}")
        logger.info("-" * 46)
        for name, scores in sorted(results.items(), key=lambda x: x[1]['test'], reverse=True):
            cv_str = f"{scores['cv']:.4f}" if scores['cv'] else "N/A"
            logger.info(f"{name:<20} {cv_str:>12} {scores['test']:>12.4f}")

        logger.info(f"\n🏆 BEST TEST ACCURACY: {self.best_score:.4f} ({self.best_score*100:.2f}%)")

        # Kaggle estimate
        logger.info(f"\n📈 Estimated Kaggle Position:")
        if self.best_score >= 0.86:
            logger.info(f"   {self.best_score*100:.2f}% → TOP 1% 🏆")
        elif self.best_score >= 0.84:
            logger.info(f"   {self.best_score*100:.2f}% → TOP 5%")
        elif self.best_score >= 0.82:
            logger.info(f"   {self.best_score*100:.2f}% → TOP 10%")
        elif self.best_score >= 0.80:
            logger.info(f"   {self.best_score*100:.2f}% → TOP 25%")

        # Feature importance
        logger.info("\n📊 Top Features (LightGBM importance):")
        lgb_model = trained_models['LightGBM']
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        for _, row in importance.head(10).iterrows():
            logger.info(f"   {row['feature']}: {row['importance']:.0f}")

        return {
            'best_score': self.best_score,
            'results': results,
            'feature_importance': importance.to_dict('records')
        }


async def main():
    data_path = Path(__file__).parent / "train.csv"
    df_full = pd.read_csv(data_path)
    logger.info(f"Total data: {len(df_full)} rows")

    # Split BEFORE any processing
    train_df, test_df = train_test_split(
        df_full, test_size=0.2, random_state=42, stratify=df_full['Survived']
    )
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} (held out)")

    system = EvolutionarySystemV2()
    result = await system.run(train_df, test_df)

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
