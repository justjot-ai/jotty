"""
Proper ML Without Target Leakage
================================

Key insight: TicketSurvivalRate was directly encoding the target!

This version:
1. NO target-based features on train (they leak!)
2. Only uses features that would be available before seeing survival
3. Proper regularization to prevent overfitting
"""

import asyncio
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def create_clean_features(df_train: pd.DataFrame, df_test: pd.DataFrame = None):
    """
    Create features WITHOUT target leakage.
    Only uses information available BEFORE knowing survival.
    """
    train = df_train.copy()
    test = df_test.copy() if df_test is not None else None
    stats = {}

    # ==========================================
    # Title (from name - no leakage)
    # ==========================================
    for df in [train, test]:
        if df is None:
            continue
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
                     'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
                     'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
                     'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
                     'Jonkheer': 'Rare', 'Dona': 'Rare'}
        df['Title'] = df['Title'].replace(title_map)
        df['Title_Enc'] = df['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}).fillna(4)

    # ==========================================
    # Age (impute using title - no leakage)
    # ==========================================
    stats['age_by_title'] = train.groupby('Title')['Age'].median().to_dict()
    stats['age_median'] = train['Age'].median()

    for df in [train, test]:
        if df is None:
            continue
        for title, median_age in stats['age_by_title'].items():
            mask = (df['Age'].isnull()) & (df['Title'] == title)
            df.loc[mask, 'Age'] = median_age
        df['Age'] = df['Age'].fillna(stats['age_median'])

        # Age features
        df['IsChild'] = (df['Age'] < 15).astype(int)
        df['IsTeen'] = ((df['Age'] >= 15) & (df['Age'] < 20)).astype(int)
        df['IsElder'] = (df['Age'] > 60).astype(int)
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 35, 60, 100],
                              labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # ==========================================
    # Sex (no leakage)
    # ==========================================
    for df in [train, test]:
        if df is None:
            continue
        df['Sex_Enc'] = df['Sex'].map({'male': 0, 'female': 1})

    # ==========================================
    # Family (no leakage - from SibSp/Parch)
    # ==========================================
    for df in [train, test]:
        if df is None:
            continue
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['SmallFamily'] = ((df['FamilySize'] > 1) & (df['FamilySize'] <= 4)).astype(int)
        df['LargeFamily'] = (df['FamilySize'] > 4).astype(int)

    # ==========================================
    # Fare (no leakage)
    # ==========================================
    stats['fare_median'] = train['Fare'].median()
    stats['fare_by_class'] = train.groupby('Pclass')['Fare'].median().to_dict()

    for df in [train, test]:
        if df is None:
            continue
        df['Fare'] = df['Fare'].fillna(df['Pclass'].map(stats['fare_by_class']).fillna(stats['fare_median']))
        df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']
        df['Fare_Log'] = np.log1p(df['Fare'])

        # Fare relative to class (no leakage - uses median)
        df['Class_Fare_Median'] = df['Pclass'].map(stats['fare_by_class'])
        df['Fare_Ratio'] = df['Fare'] / (df['Class_Fare_Median'] + 0.01)

    # ==========================================
    # Embarked (no leakage)
    # ==========================================
    stats['embarked_mode'] = train['Embarked'].mode()[0]
    for df in [train, test]:
        if df is None:
            continue
        df['Embarked'] = df['Embarked'].fillna(stats['embarked_mode'])
        df['Embarked_Enc'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)

    # ==========================================
    # Cabin (no leakage)
    # ==========================================
    for df in [train, test]:
        if df is None:
            continue
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df['Deck'] = df['Cabin'].fillna('U').str[0]
        deck_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1, 'T': 0, 'U': 0}
        df['Deck_Enc'] = df['Deck'].map(deck_map).fillna(0)

    # ==========================================
    # Ticket (no leakage - just counts)
    # ==========================================
    # Combine for ticket counts (this is OK - just counting)
    combined = pd.concat([train, test], sort=False) if test is not None else train.copy()
    ticket_counts = combined['Ticket'].value_counts()

    for df in [train, test]:
        if df is None:
            continue
        df['TicketCount'] = df['Ticket'].map(ticket_counts).fillna(1)
        df['SharedTicket'] = (df['TicketCount'] > 1).astype(int)

        # Ticket prefix
        df['TicketPrefix'] = df['Ticket'].str.extract(r'^([A-Za-z./]+)', expand=False).fillna('NONE')
        prefix_map = {'PC': 1, 'CA': 2, 'STON': 3, 'NONE': 0}
        df['TicketPrefix_Enc'] = df['TicketPrefix'].apply(
            lambda x: prefix_map.get(x.split('/')[0].split('.')[0], 0)
        )

    # ==========================================
    # Surname (no leakage - just counts)
    # ==========================================
    for df in [train, test]:
        if df is None:
            continue
        df['Surname'] = df['Name'].str.extract(r'^([^,]+)', expand=False)

    combined = pd.concat([train, test], sort=False) if test is not None else train.copy()
    surname_counts = combined['Surname'].value_counts()

    for df in [train, test]:
        if df is None:
            continue
        df['SurnameCount'] = df['Surname'].map(surname_counts).fillna(1)
        df['NameLength'] = df['Name'].str.len()

    # ==========================================
    # Interactions (no leakage)
    # ==========================================
    for df in [train, test]:
        if df is None:
            continue
        # Key survival indicators
        df['Female_Child'] = ((df['Sex_Enc'] == 1) | (df['IsChild'] == 1)).astype(int)
        df['Female_Class1'] = ((df['Sex_Enc'] == 1) & (df['Pclass'] == 1)).astype(int)
        df['Female_Class3'] = ((df['Sex_Enc'] == 1) & (df['Pclass'] == 3)).astype(int)
        df['Male_Class3'] = ((df['Sex_Enc'] == 0) & (df['Pclass'] == 3)).astype(int)
        df['Male_Adult'] = ((df['Sex_Enc'] == 0) & (df['Age'] > 18)).astype(int)

        df['Alone_Male'] = ((df['IsAlone'] == 1) & (df['Sex_Enc'] == 0)).astype(int)
        df['Family_Female'] = ((df['FamilySize'] > 1) & (df['Sex_Enc'] == 1)).astype(int)

        df['Child_Class3'] = ((df['IsChild'] == 1) & (df['Pclass'] == 3)).astype(int)
        df['Master_Class3'] = ((df['Title'] == 'Master') & (df['Pclass'] == 3)).astype(int)

        # Age × Fare interaction
        df['Age_Fare'] = df['Age'] * df['Fare_Log']

    # Get feature columns
    exclude = ['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex',
               'Surname', 'Title', 'Deck', 'Embarked', 'TicketPrefix', 'Class_Fare_Median']
    feature_cols = [c for c in train.select_dtypes(include=[np.number]).columns
                   if c not in exclude]

    return train, test, feature_cols


async def main():
    logger.info("=" * 70)
    logger.info("🧪 PROPER ML - NO TARGET LEAKAGE")
    logger.info("=" * 70)

    data_path = Path(__file__).parent / "train.csv"
    df_full = pd.read_csv(data_path)
    logger.info(f"Total data: {len(df_full)} rows")

    # Split BEFORE any processing
    train_df, test_df = train_test_split(
        df_full, test_size=0.2, random_state=42, stratify=df_full['Survived']
    )
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} (held out)")

    # Create features WITHOUT leakage
    train, test, feature_cols = create_clean_features(train_df, test_df)
    logger.info(f"Features: {len(feature_cols)}")

    y_train = train['Survived']
    y_test = test['Survived']

    X_train = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ==========================================
    # TRAIN MODELS WITH REGULARIZATION
    # ==========================================
    logger.info("\n🤖 Training Models (with proper regularization)...")

    models = {
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=3,
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10,
            random_state=42
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

        results[name] = {'cv': cv_scores.mean(), 'cv_std': cv_scores.std(), 'test': test_acc}
        gap = cv_scores.mean() - test_acc
        logger.info(f"   {name}: CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}, Test={test_acc:.4f}, Gap={gap:+.4f}")

    # ==========================================
    # ENSEMBLE
    # ==========================================
    logger.info("\n🏗️ Building Ensemble...")

    estimators = [(name, model) for name, model in trained_models.items()]
    voting = VotingClassifier(estimators=estimators, voting='soft')
    voting_cv = cross_val_score(voting, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    voting.fit(X_train_scaled, y_train)
    voting_test = accuracy_score(y_test, voting.predict(X_test_scaled))

    gap = voting_cv.mean() - voting_test
    logger.info(f"   Voting: CV={voting_cv.mean():.4f}±{voting_cv.std():.4f}, Test={voting_test:.4f}, Gap={gap:+.4f}")

    results['Voting'] = {'cv': voting_cv.mean(), 'cv_std': voting_cv.std(), 'test': voting_test}

    # ==========================================
    # RESULTS
    # ==========================================
    best = max(results.items(), key=lambda x: x[1]['test'])
    best_score = best[1]['test']

    logger.info("\n" + "=" * 70)
    logger.info("🏆 FINAL RESULTS (No Leakage)")
    logger.info("=" * 70)

    logger.info(f"\n{'Model':<20} {'CV Score':>15} {'Test Acc':>12} {'Gap':>10}")
    logger.info("-" * 60)
    for name, scores in sorted(results.items(), key=lambda x: x[1]['test'], reverse=True):
        gap = scores['cv'] - scores['test']
        logger.info(f"{name:<20} {scores['cv']:.4f}±{scores['cv_std']:.4f}    {scores['test']:>10.4f} {gap:>+10.4f}")

    logger.info(f"\n🏆 BEST TEST ACCURACY: {best_score:.4f} ({best_score*100:.2f}%)")
    logger.info(f"   CV-Test Gap: {best[1]['cv'] - best_score:.4f} (healthy if < 0.02)")

    # Kaggle estimate
    logger.info(f"\n📈 Estimated Kaggle Position:")
    if best_score >= 0.84:
        logger.info(f"   {best_score*100:.2f}% → TOP 5% 🏆")
    elif best_score >= 0.82:
        logger.info(f"   {best_score*100:.2f}% → TOP 10%")
    elif best_score >= 0.80:
        logger.info(f"   {best_score*100:.2f}% → TOP 25%")
    else:
        logger.info(f"   {best_score*100:.2f}%")

    # Feature importance
    logger.info("\n📊 Top Features:")
    lgb_model = trained_models['LightGBM']
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.0f}")

    return {'best_score': best_score, 'results': results}


if __name__ == "__main__":
    result = asyncio.run(main())
