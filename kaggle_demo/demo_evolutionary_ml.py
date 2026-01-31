"""
Evolutionary ML System
======================

A self-improving ML system that:
1. Makes predictions
2. Analyzes errors by dimension (where did we fail?)
3. Automatically engineers features to fix errors
4. Evolves through multiple generations
5. Learns what features help which segments

This is how grandmasters approach Kaggle.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyzes prediction errors by different dimensions."""

    def __init__(self):
        self.error_patterns = {}
        self.segment_accuracy = {}

    def analyze(self, df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze errors across multiple dimensions.
        Returns insights about where the model fails.
        """
        df = df.copy()
        df['y_true'] = y_true
        df['y_pred'] = y_pred
        df['correct'] = (y_true == y_pred).astype(int)
        df['error'] = 1 - df['correct']

        analysis = {
            'overall_accuracy': df['correct'].mean(),
            'dimensions': {},
            'worst_segments': [],
            'insights': []
        }

        # Analyze by key dimensions
        dimensions = self._get_dimensions(df)

        for dim_name, dim_col in dimensions.items():
            if dim_col not in df.columns:
                continue

            dim_analysis = self._analyze_dimension(df, dim_col, dim_name)
            analysis['dimensions'][dim_name] = dim_analysis

            # Track worst segments
            for segment, stats in dim_analysis['segments'].items():
                if stats['count'] >= 10 and stats['accuracy'] < analysis['overall_accuracy'] - 0.05:
                    analysis['worst_segments'].append({
                        'dimension': dim_name,
                        'segment': segment,
                        'accuracy': stats['accuracy'],
                        'count': stats['count'],
                        'gap': analysis['overall_accuracy'] - stats['accuracy']
                    })

        # Sort worst segments by gap
        analysis['worst_segments'] = sorted(
            analysis['worst_segments'],
            key=lambda x: x['gap'],
            reverse=True
        )[:10]

        # Generate insights
        analysis['insights'] = self._generate_insights(analysis)

        self.error_patterns = analysis
        return analysis

    def _get_dimensions(self, df: pd.DataFrame) -> Dict[str, str]:
        """Define dimensions to analyze."""
        dims = {
            'sex': 'Sex',
            'class': 'Pclass',
            'age_group': 'AgeGroup',
            'family_size': 'FamilySize',
            'fare_group': 'FareGroup',
            'embarked': 'Embarked',
            'title': 'Title',
            'is_alone': 'IsAlone',
            'cabin_known': 'HasCabin',
        }
        return {k: v for k, v in dims.items() if v in df.columns}

    def _analyze_dimension(self, df: pd.DataFrame, col: str, name: str) -> Dict:
        """Analyze accuracy by a single dimension."""
        segments = {}
        for value in df[col].unique():
            mask = df[col] == value
            subset = df[mask]
            if len(subset) > 0:
                segments[str(value)] = {
                    'accuracy': subset['correct'].mean(),
                    'count': len(subset),
                    'error_rate': subset['error'].mean(),
                    'survival_rate': subset['y_true'].mean()
                }

        return {
            'column': col,
            'segments': segments,
            'variance': np.var([s['accuracy'] for s in segments.values()])
        }

    def _generate_insights(self, analysis: Dict) -> List[str]:
        """Generate human-readable insights from error analysis."""
        insights = []

        for seg in analysis['worst_segments'][:5]:
            insights.append(
                f"Model struggles with {seg['dimension']}={seg['segment']}: "
                f"{seg['accuracy']*100:.1f}% accuracy (vs {analysis['overall_accuracy']*100:.1f}% overall)"
            )

        return insights


class FeatureEvolver:
    """Automatically evolves features based on error patterns."""

    def __init__(self):
        self.evolved_features = []
        self.feature_scores = {}
        self.generation = 0

    def evolve(self, df: pd.DataFrame, error_analysis: Dict, generation: int) -> pd.DataFrame:
        """
        Create new features based on error analysis.
        Each generation adds features targeting weak segments.
        """
        self.generation = generation
        df = df.copy()
        new_features = []

        logger.info(f"\n🧬 Generation {generation}: Evolving features...")

        # Strategy 1: Interaction features for worst segments
        new_features.extend(self._create_interaction_features(df, error_analysis))

        # Strategy 2: Ratio features
        new_features.extend(self._create_ratio_features(df))

        # Strategy 3: Statistical features (skew, etc.)
        new_features.extend(self._create_statistical_features(df))

        # Strategy 4: Segment-specific features
        new_features.extend(self._create_segment_features(df, error_analysis))

        # Strategy 5: Survival pattern features
        new_features.extend(self._create_survival_pattern_features(df))

        self.evolved_features.extend(new_features)
        logger.info(f"   Created {len(new_features)} new features")

        return df

    def _create_interaction_features(self, df: pd.DataFrame, analysis: Dict) -> List[str]:
        """Create interaction features for problematic segments."""
        new_features = []

        # Sex × Class interaction (classic Titanic feature)
        if 'Sex' in df.columns and 'Pclass' in df.columns:
            df['Sex_Class'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
            df['Sex_Class_Encoded'] = LabelEncoder().fit_transform(df['Sex_Class'])
            new_features.append('Sex_Class_Encoded')

            # Female in each class
            df['Female_Class1'] = ((df['Sex'] == 1) & (df['Pclass'] == 1)).astype(int)
            df['Female_Class2'] = ((df['Sex'] == 1) & (df['Pclass'] == 2)).astype(int)
            df['Female_Class3'] = ((df['Sex'] == 1) & (df['Pclass'] == 3)).astype(int)
            new_features.extend(['Female_Class1', 'Female_Class2', 'Female_Class3'])

        # Age × Class
        if 'Age' in df.columns and 'Pclass' in df.columns:
            df['Age_Class'] = df['Age'] * df['Pclass']
            df['Child_Class1'] = ((df['Age'] < 12) & (df['Pclass'] == 1)).astype(int)
            df['Child_Class3'] = ((df['Age'] < 12) & (df['Pclass'] == 3)).astype(int)
            new_features.extend(['Age_Class', 'Child_Class1', 'Child_Class3'])

        # Family × Class
        if 'FamilySize' in df.columns and 'Pclass' in df.columns:
            df['Family_Class'] = df['FamilySize'] * df['Pclass']
            df['BigFamily_Class3'] = ((df['FamilySize'] > 4) & (df['Pclass'] == 3)).astype(int)
            new_features.extend(['Family_Class', 'BigFamily_Class3'])

        return new_features

    def _create_ratio_features(self, df: pd.DataFrame) -> List[str]:
        """Create ratio-based features."""
        new_features = []

        if 'Fare' in df.columns and 'Pclass' in df.columns:
            # Fare relative to class average
            class_fare_mean = df.groupby('Pclass')['Fare'].transform('mean')
            df['Fare_Ratio_Class'] = df['Fare'] / (class_fare_mean + 0.01)
            df['Fare_Above_Class_Mean'] = (df['Fare'] > class_fare_mean).astype(int)
            new_features.extend(['Fare_Ratio_Class', 'Fare_Above_Class_Mean'])

        if 'Age' in df.columns:
            # Age relative to mean
            age_mean = df['Age'].mean()
            df['Age_Ratio_Mean'] = df['Age'] / (age_mean + 0.01)
            new_features.append('Age_Ratio_Mean')

        if 'FamilySize' in df.columns and 'Fare' in df.columns:
            # Cost per family member
            df['Fare_Per_Family'] = df['Fare'] / df['FamilySize']
            new_features.append('Fare_Per_Family')

        return new_features

    def _create_statistical_features(self, df: pd.DataFrame) -> List[str]:
        """Create statistical features like skew indicators."""
        new_features = []

        if 'Fare' in df.columns:
            fare_median = df['Fare'].median()
            fare_std = df['Fare'].std()

            # Skew-aware binning
            df['Fare_Skew_Bin'] = pd.cut(
                df['Fare'],
                bins=[0, fare_median*0.5, fare_median, fare_median*2, fare_median*5, float('inf')],
                labels=[0, 1, 2, 3, 4]
            ).astype(float).fillna(2)

            # Outlier indicator
            df['Fare_Outlier'] = (np.abs(df['Fare'] - fare_median) > 2 * fare_std).astype(int)
            new_features.extend(['Fare_Skew_Bin', 'Fare_Outlier'])

        if 'Age' in df.columns:
            # Age distribution features
            df['Age_Squared'] = df['Age'] ** 2
            df['Age_Log'] = np.log1p(df['Age'])
            new_features.extend(['Age_Squared', 'Age_Log'])

        return new_features

    def _create_segment_features(self, df: pd.DataFrame, analysis: Dict) -> List[str]:
        """Create features specifically targeting weak segments."""
        new_features = []

        worst_segments = analysis.get('worst_segments', [])

        for seg in worst_segments[:5]:
            dim = seg['dimension']
            segment_val = seg['segment']

            # Create binary indicator for this weak segment
            if dim == 'age_group' and 'AgeGroup' in df.columns:
                feature_name = f'Weak_Age_{segment_val}'
                df[feature_name] = (df['AgeGroup'].astype(str) == segment_val).astype(int)
                new_features.append(feature_name)

            elif dim == 'family_size' and 'FamilySize' in df.columns:
                feature_name = f'Weak_Family_{segment_val}'
                try:
                    df[feature_name] = (df['FamilySize'] == int(float(segment_val))).astype(int)
                    new_features.append(feature_name)
                except:
                    pass

        return new_features

    def _create_survival_pattern_features(self, df: pd.DataFrame) -> List[str]:
        """Create features based on known survival patterns."""
        new_features = []

        # Women and children first
        if 'Sex' in df.columns and 'Age' in df.columns:
            df['Woman_Or_Child'] = ((df['Sex'] == 1) | (df['Age'] < 15)).astype(int)
            new_features.append('Woman_Or_Child')

        # Deck from cabin (if extractable)
        if 'Cabin' in df.columns:
            df['Deck'] = df['Cabin'].fillna('U').str[0]
            deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 0}
            df['Deck_Num'] = df['Deck'].map(deck_map).fillna(0)
            new_features.append('Deck_Num')

        # Family survival rate proxy
        if 'FamilySize' in df.columns:
            df['Optimal_Family'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
            new_features.append('Optimal_Family')

        # Title-based survival
        if 'Title' in df.columns:
            # High survival titles
            high_survival = ['Mrs', 'Miss', 'Master', 1, 2, 3]  # Include encoded values
            df['High_Survival_Title'] = df['Title'].isin(high_survival).astype(int)
            new_features.append('High_Survival_Title')

        return new_features


class EvolutionaryMLSystem:
    """
    Main evolutionary ML system that iteratively improves.
    """

    def __init__(self):
        self.error_analyzer = ErrorAnalyzer()
        self.feature_evolver = FeatureEvolver()
        self.generation_history = []
        self.best_score = 0
        self.best_generation = 0

    def prepare_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare base features before evolution."""
        df = df.copy()

        # Title extraction
        if 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            title_map = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
                         'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
                         'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
                         'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
                         'Jonkheer': 'Rare', 'Dona': 'Rare'}
            df['Title'] = df['Title'].replace(title_map)
            title_encoding = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
            df['Title'] = df['Title'].map(title_encoding).fillna(4)

        # Age imputation and grouping
        if 'Age' in df.columns:
            if 'Title' in df.columns:
                age_by_title = df.groupby('Title')['Age'].median()
                for title in df['Title'].unique():
                    mask = (df['Age'].isnull()) & (df['Title'] == title)
                    if mask.any() and title in age_by_title.index:
                        df.loc[mask, 'Age'] = age_by_title[title]
            df['Age'] = df['Age'].fillna(df['Age'].median())
            df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                                     labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

        # Family features
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Fare
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            df['FareGroup'] = pd.qcut(df['Fare'], q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')

        # Cabin
        if 'Cabin' in df.columns:
            df['HasCabin'] = df['Cabin'].notna().astype(int)

        # Embarked
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
            df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)

        # Sex encoding
        if 'Sex' in df.columns:
            df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get all numeric feature columns."""
        exclude = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin',
                   'y_true', 'y_pred', 'correct', 'error', 'AgeGroup', 'FareGroup', 'Sex_Class', 'Deck']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    async def evolve(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                     max_generations: int = 5) -> Dict:
        """
        Run the evolutionary process for multiple generations.
        """
        logger.info("=" * 70)
        logger.info("🧬 EVOLUTIONARY ML SYSTEM")
        logger.info("=" * 70)

        # Prepare base features
        train = self.prepare_base_features(df_train)
        test = self.prepare_base_features(df_test)

        y_train = train['Survived']
        y_test = test['Survived']

        for gen in range(1, max_generations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 GENERATION {gen}")
            logger.info(f"{'='*60}")

            # Get current features
            feature_cols = self.get_feature_columns(train)
            logger.info(f"   Features: {len(feature_cols)}")

            X_train = train[feature_cols].fillna(0)
            X_test = test[feature_cols].fillna(0)

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, verbose=-1
            )

            # CV score
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            cv_mean = cv_scores.mean()

            # Train and evaluate on test
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)

            logger.info(f"\n📊 Generation {gen} Results:")
            logger.info(f"   CV Score:    {cv_mean:.4f}")
            logger.info(f"   Train Acc:   {train_acc:.4f}")
            logger.info(f"   Test Acc:    {test_acc:.4f} {'✨ NEW BEST!' if test_acc > self.best_score else ''}")

            # Track best
            if test_acc > self.best_score:
                self.best_score = test_acc
                self.best_generation = gen

            # Error Analysis
            logger.info(f"\n🔍 Error Analysis:")
            analysis = self.error_analyzer.analyze(train, y_train.values, y_pred_train)

            for insight in analysis['insights'][:3]:
                logger.info(f"   • {insight}")

            # Record generation
            self.generation_history.append({
                'generation': gen,
                'cv_score': cv_mean,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'n_features': len(feature_cols),
                'worst_segments': analysis['worst_segments'][:3]
            })

            # Evolve features for next generation
            if gen < max_generations:
                train = self.feature_evolver.evolve(train, analysis, gen)
                # Apply same features to test
                for feat in self.feature_evolver.evolved_features:
                    if feat in train.columns and feat not in test.columns:
                        # Recreate feature on test using same logic
                        test = self._apply_feature_to_test(test, train, feat)

        # Final Summary
        self._print_summary()

        return {
            'best_score': self.best_score,
            'best_generation': self.best_generation,
            'history': self.generation_history
        }

    def _apply_feature_to_test(self, test: pd.DataFrame, train: pd.DataFrame, feat: str) -> pd.DataFrame:
        """Apply evolved feature from train to test."""
        # For most features, we can recreate them using the same logic
        # This is a simplified version - in production you'd save the transformation
        if feat in train.columns:
            if feat not in test.columns:
                # Try to recreate based on column patterns
                if 'Sex_Class' in feat and 'Sex' in test.columns and 'Pclass' in test.columns:
                    test['Sex_Class'] = test['Sex'].astype(str) + '_' + test['Pclass'].astype(str)
                    test['Sex_Class_Encoded'] = LabelEncoder().fit_transform(test['Sex_Class'])
                elif feat.startswith('Female_Class'):
                    class_num = int(feat[-1])
                    test[feat] = ((test['Sex'] == 1) & (test['Pclass'] == class_num)).astype(int)
                elif feat == 'Age_Class':
                    test[feat] = test['Age'] * test['Pclass']
                elif feat == 'Family_Class':
                    test[feat] = test['FamilySize'] * test['Pclass']
                elif feat == 'Fare_Per_Family':
                    test[feat] = test['Fare'] / test['FamilySize']
                elif feat == 'Woman_Or_Child':
                    test[feat] = ((test['Sex'] == 1) | (test['Age'] < 15)).astype(int)
                elif feat == 'Optimal_Family':
                    test[feat] = ((test['FamilySize'] >= 2) & (test['FamilySize'] <= 4)).astype(int)
                elif feat == 'Deck_Num' and 'Cabin' in test.columns:
                    test['Deck'] = test['Cabin'].fillna('U').str[0]
                    deck_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 0}
                    test['Deck_Num'] = test['Deck'].map(deck_map).fillna(0)
                elif feat == 'Fare_Ratio_Class':
                    class_fare_mean = test.groupby('Pclass')['Fare'].transform('mean')
                    test[feat] = test['Fare'] / (class_fare_mean + 0.01)
                elif feat == 'Fare_Above_Class_Mean':
                    class_fare_mean = test.groupby('Pclass')['Fare'].transform('mean')
                    test[feat] = (test['Fare'] > class_fare_mean).astype(int)
                elif feat == 'Age_Ratio_Mean':
                    test[feat] = test['Age'] / (test['Age'].mean() + 0.01)
                elif feat == 'Age_Squared':
                    test[feat] = test['Age'] ** 2
                elif feat == 'Age_Log':
                    test[feat] = np.log1p(test['Age'])
                elif feat == 'Fare_Outlier':
                    fare_median = test['Fare'].median()
                    fare_std = test['Fare'].std()
                    test[feat] = (np.abs(test['Fare'] - fare_median) > 2 * fare_std).astype(int)
                elif 'Child_Class' in feat:
                    class_num = int(feat[-1])
                    test[feat] = ((test['Age'] < 12) & (test['Pclass'] == class_num)).astype(int)
                elif 'BigFamily_Class' in feat:
                    class_num = int(feat[-1])
                    test[feat] = ((test['FamilySize'] > 4) & (test['Pclass'] == class_num)).astype(int)
                else:
                    # Default: fill with 0
                    test[feat] = 0

        return test

    def _print_summary(self):
        """Print evolution summary."""
        logger.info("\n" + "=" * 70)
        logger.info("🏆 EVOLUTION SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\n{'Gen':<6} {'Features':<10} {'CV Score':<12} {'Test Acc':<12}")
        logger.info("-" * 42)

        for h in self.generation_history:
            marker = " ⭐" if h['generation'] == self.best_generation else ""
            logger.info(f"{h['generation']:<6} {h['n_features']:<10} {h['cv_score']:<12.4f} {h['test_acc']:<12.4f}{marker}")

        logger.info(f"\n🏆 Best Test Accuracy: {self.best_score:.4f} ({self.best_score*100:.2f}%)")
        logger.info(f"   Achieved in Generation: {self.best_generation}")

        improvement = self.generation_history[-1]['test_acc'] - self.generation_history[0]['test_acc']
        logger.info(f"   Total Improvement: {improvement*100:+.2f}%")

        # Kaggle estimate
        logger.info(f"\n📈 Estimated Kaggle Position:")
        if self.best_score >= 0.86:
            logger.info(f"   {self.best_score*100:.2f}% → TOP 1% 🏆")
        elif self.best_score >= 0.84:
            logger.info(f"   {self.best_score*100:.2f}% → TOP 5%")
        elif self.best_score >= 0.82:
            logger.info(f"   {self.best_score*100:.2f}% → TOP 10%")
        else:
            logger.info(f"   {self.best_score*100:.2f}% → TOP 25%")


async def main():
    logger.info("=" * 70)
    logger.info("🧬 EVOLUTIONARY ML SYSTEM - SELF-IMPROVING")
    logger.info("=" * 70)

    # Load data
    data_path = Path(__file__).parent / "train.csv"
    df_full = pd.read_csv(data_path)
    logger.info(f"\nTotal data: {len(df_full)} rows")

    # Split BEFORE any processing
    train_df, test_df = train_test_split(
        df_full, test_size=0.2, random_state=42, stratify=df_full['Survived']
    )
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} (held out)")

    # Run evolutionary system
    system = EvolutionaryMLSystem()
    result = await system.evolve(train_df, test_df, max_generations=5)

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
