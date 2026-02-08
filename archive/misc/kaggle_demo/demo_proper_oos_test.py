"""
Proper Out-of-Sample Test
=========================

This test PROPERLY evaluates by:
1. Splitting data 80/20 BEFORE any processing
2. Feature engineering fitted ONLY on train
3. Models trained ONLY on train
4. Final evaluation on TRUE holdout

This simulates actual Kaggle submission.
"""

import asyncio
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def engineer_features_proper(df_train, df_test=None):
    """
    Feature engineering that ONLY uses train statistics.
    If df_test provided, transforms it using train-fitted values.
    """
    df = df_train.copy()

    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
        'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare'
    }
    df['Title'] = df['Title'].replace(title_map)

    # Store train statistics for later use on test
    train_stats = {}

    # Age by title (from train only)
    train_stats['age_by_title'] = df.groupby('Title')['Age'].median().to_dict()
    train_stats['age_median'] = df['Age'].median()

    for title, median_age in train_stats['age_by_title'].items():
        mask = (df['Age'].isnull()) & (df['Title'] == title)
        df.loc[mask, 'Age'] = median_age
    df['Age'].fillna(train_stats['age_median'], inplace=True)

    # Fare median (from train only)
    train_stats['fare_median'] = df['Fare'].median()
    df['Fare'].fillna(train_stats['fare_median'], inplace=True)

    # Embarked mode (from train only)
    train_stats['embarked_mode'] = df['Embarked'].mode()[0]
    df['Embarked'].fillna(train_stats['embarked_mode'], inplace=True)

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age bins
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                          labels=[0, 1, 2, 3, 4])
    df['IsChild'] = (df['Age'] < 12).astype(int)

    # Fare bins (using train quantiles)
    train_stats['fare_bins'] = [df['Fare'].min()-1,
                                 df['Fare'].quantile(0.25),
                                 df['Fare'].quantile(0.5),
                                 df['Fare'].quantile(0.75),
                                 df['Fare'].max()+1]
    df['FareBin'] = pd.cut(df['Fare'], bins=train_stats['fare_bins'], labels=[0,1,2,3])

    # Cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Sex encoding
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Embarked encoding
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Title encoding
    title_encoding = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
    df['Title'] = df['Title'].map(title_encoding).fillna(4)

    # FarePerPerson
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Process test set if provided
    if df_test is not None:
        df_t = df_test.copy()

        # Title
        df_t['Title'] = df_t['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df_t['Title'] = df_t['Title'].replace(title_map)

        # Age imputation using TRAIN statistics
        for title, median_age in train_stats['age_by_title'].items():
            mask = (df_t['Age'].isnull()) & (df_t['Title'] == title)
            df_t.loc[mask, 'Age'] = median_age
        df_t['Age'].fillna(train_stats['age_median'], inplace=True)

        # Fare using TRAIN median
        df_t['Fare'].fillna(train_stats['fare_median'], inplace=True)

        # Embarked using TRAIN mode
        df_t['Embarked'].fillna(train_stats['embarked_mode'], inplace=True)

        # Same transformations
        df_t['FamilySize'] = df_t['SibSp'] + df_t['Parch'] + 1
        df_t['IsAlone'] = (df_t['FamilySize'] == 1).astype(int)
        df_t['AgeBin'] = pd.cut(df_t['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4])
        df_t['IsChild'] = (df_t['Age'] < 12).astype(int)
        df_t['FareBin'] = pd.cut(df_t['Fare'], bins=train_stats['fare_bins'], labels=[0,1,2,3])
        df_t['HasCabin'] = df_t['Cabin'].notna().astype(int)
        df_t['Sex'] = df_t['Sex'].map({'male': 0, 'female': 1})
        df_t['Embarked'] = df_t['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        df_t['Title'] = df_t['Title'].map(title_encoding).fillna(4)
        df_t['FarePerPerson'] = df_t['Fare'] / df_t['FamilySize']

        return df, df_t, train_stats

    return df, train_stats


async def main():
    logger.info("=" * 70)
    logger.info("PROPER OUT-OF-SAMPLE TEST")
    logger.info("=" * 70)
    logger.info("This simulates actual Kaggle submission conditions")
    logger.info("=" * 70)

    # Load data
    data_path = Path(__file__).parent / "train.csv"
    df_full = pd.read_csv(data_path)
    logger.info(f"\nTotal data: {len(df_full)} rows")

    # CRITICAL: Split BEFORE any processing
    train_df, test_df = train_test_split(
        df_full, test_size=0.2, random_state=42, stratify=df_full['Survived']
    )
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)} (held out)")

    # Feature engineering using ONLY train statistics
    logger.info("\n📊 Feature Engineering (train stats only)...")
    train_fe, test_fe, stats = engineer_features_proper(train_df, test_df)

    # Prepare features
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                    'FamilySize', 'IsAlone', 'AgeBin', 'IsChild', 'FareBin',
                    'HasCabin', 'Title', 'FarePerPerson']

    X_train = train_fe[feature_cols].fillna(0).astype(float)
    y_train = train_fe['Survived']
    X_test = test_fe[feature_cols].fillna(0).astype(float)
    y_test = test_fe['Survived']

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform only, no fit!

    logger.info(f"Features: {len(feature_cols)}")

    # Train models
    logger.info("\n🤖 Training Models...")

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import lightgbm as lgb

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                      random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                        random_state=42, verbose=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    trained_models = {}

    for name, model in models.items():
        # CV score (on train only)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()

        # Train on full train set
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        # TRUE OUT-OF-SAMPLE score
        oos_pred = model.predict(X_test_scaled)
        oos_score = accuracy_score(y_test, oos_pred)

        results[name] = {'cv': cv_mean, 'oos': oos_score}
        logger.info(f"  {name}: CV={cv_mean:.4f}, OOS={oos_score:.4f}, Gap={cv_mean-oos_score:.4f}")

    # Hyperparameter tuning with Optuna (on train only)
    logger.info("\n⚡ Optuna Optimization (50 trials)...")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        model = xgb.XGBClassifier(**params)
        return cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    # Train optimized model
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'

    optimized_xgb = xgb.XGBClassifier(**best_params)
    optimized_xgb.fit(X_train_scaled, y_train)

    opt_cv = study.best_value
    opt_oos = accuracy_score(y_test, optimized_xgb.predict(X_test_scaled))

    logger.info(f"  Optimized XGB: CV={opt_cv:.4f}, OOS={opt_oos:.4f}, Gap={opt_cv-opt_oos:.4f}")

    trained_models['OptimizedXGB'] = optimized_xgb
    results['OptimizedXGB'] = {'cv': opt_cv, 'oos': opt_oos}

    # Ensemble
    logger.info("\n🏗️ Building Ensemble...")
    from sklearn.ensemble import VotingClassifier

    estimators = [(name, model) for name, model in trained_models.items()
                  if hasattr(model, 'predict_proba')]

    voting = VotingClassifier(estimators=estimators[:5], voting='soft')
    voting_cv = cross_val_score(voting, X_train_scaled, y_train, cv=cv, scoring='accuracy').mean()
    voting.fit(X_train_scaled, y_train)
    voting_oos = accuracy_score(y_test, voting.predict(X_test_scaled))

    logger.info(f"  Voting Ensemble: CV={voting_cv:.4f}, OOS={voting_oos:.4f}, Gap={voting_cv-voting_oos:.4f}")

    results['VotingEnsemble'] = {'cv': voting_cv, 'oos': voting_oos}

    # FINAL RESULTS
    logger.info("\n" + "=" * 70)
    logger.info("📊 FINAL COMPARISON: CV vs TRUE OUT-OF-SAMPLE")
    logger.info("=" * 70)

    logger.info(f"\n{'Model':<20} {'CV Score':>12} {'OOS Score':>12} {'Gap':>10}")
    logger.info("-" * 56)
    for name, scores in sorted(results.items(), key=lambda x: x[1]['oos'], reverse=True):
        gap = scores['cv'] - scores['oos']
        logger.info(f"{name:<20} {scores['cv']:>12.4f} {scores['oos']:>12.4f} {gap:>10.4f}")

    # Best OOS score
    best_model = max(results.items(), key=lambda x: x[1]['oos'])
    best_oos = best_model[1]['oos']

    logger.info("\n" + "=" * 70)
    logger.info(f"🏆 BEST TRUE OUT-OF-SAMPLE SCORE: {best_oos:.4f} ({best_oos*100:.2f}%)")
    logger.info(f"   Model: {best_model[0]}")
    logger.info("=" * 70)

    # Leaderboard estimate
    logger.info("\n📈 REALISTIC KAGGLE LEADERBOARD ESTIMATE:")
    if best_oos >= 0.84:
        logger.info(f"   {best_oos*100:.2f}% → TOP 5%")
    elif best_oos >= 0.82:
        logger.info(f"   {best_oos*100:.2f}% → TOP 10%")
    elif best_oos >= 0.80:
        logger.info(f"   {best_oos*100:.2f}% → TOP 25%")
    elif best_oos >= 0.78:
        logger.info(f"   {best_oos*100:.2f}% → TOP 50%")
    else:
        logger.info(f"   {best_oos*100:.2f}% → Below median")

    logger.info("\n💡 KEY INSIGHT:")
    avg_gap = np.mean([r['cv'] - r['oos'] for r in results.values()])
    logger.info(f"   Average CV→OOS gap: {avg_gap:.4f} ({avg_gap*100:.2f}%)")
    logger.info(f"   This is the 'overfitting tax' you pay on Kaggle submission")

    return best_oos


if __name__ == "__main__":
    result = asyncio.run(main())
