"""
Feature Engineering Skill for Jotty
====================================

Automatic feature engineering for tabular data.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from Jotty.core.utils.skill_status import SkillStatus

# Status emitter for progress updates
status = SkillStatus("feature-engineer")


logger = logging.getLogger(__name__)


async def feature_engineer_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatic feature engineering for tabular data.

    Args:
        params: Dict with keys:
            - data: DataFrame or path to CSV
            - target: Target column name (optional)
            - domain: Domain hint ('titanic', 'general', etc.)

    Returns:
        Dict with engineered DataFrame and list of new features
    """
    status.set_callback(params.pop('_status_callback', None))

    logger.info("[FeatureEngineer] Starting feature engineering...")

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    target = params.get('target')
    domain = params.get('domain', 'general')
    new_features = []

    # Domain-specific engineering
    if domain == 'titanic':
        df, new_features = _engineer_titanic(df, target)
    else:
        df, new_features = _engineer_general(df, target)

    logger.info(f"[FeatureEngineer] Created {len(new_features)} new features")

    return {
        'success': True,
        'data': df,
        'new_features': new_features,
        'original_shape': data.shape,
        'new_shape': df.shape,
    }


def _engineer_titanic(df: pd.DataFrame, target: str = None) -> tuple:
    """Titanic-specific feature engineering."""
    new_features = []

    # Title extraction
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
            'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare',
            'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
            'Jonkheer': 'Rare', 'Dona': 'Rare'
        }
        df['Title'] = df['Title'].replace(title_map)
        new_features.append('Title')

    # Family features
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['FamilySizeBin'] = pd.cut(
            df['FamilySize'], bins=[0, 1, 4, 20], labels=['Alone', 'Small', 'Large']
        )
        new_features.extend(['FamilySize', 'IsAlone', 'FamilySizeBin'])

    # Age imputation and features
    if 'Age' in df.columns:
        if 'Title' in df.columns:
            age_by_title = df.groupby('Title')['Age'].median()
            for title in df['Title'].unique():
                mask = (df['Age'].isnull()) & (df['Title'] == title)
                if mask.any() and title in age_by_title.index:
                    df.loc[mask, 'Age'] = age_by_title[title]
        df['Age'].fillna(df['Age'].median(), inplace=True)

        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                              labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
        df['IsChild'] = (df['Age'] < 12).astype(int)
        new_features.extend(['AgeBin', 'IsChild'])

    # Fare features
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        if 'FamilySize' in df.columns:
            df['FarePerPerson'] = df['Fare'] / df['FamilySize']
            new_features.append('FarePerPerson')
        df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Med', 'High', 'VHigh'],
                                duplicates='drop')
        new_features.append('FareBin')

    # Cabin features
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].notna().astype(int)
        df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
        new_features.extend(['HasCabin', 'CabinDeck'])

    # Embarked
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Interaction features
    if 'Sex' in df.columns and 'Pclass' in df.columns:
        df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
        new_features.append('Sex_Pclass')

    # Ticket group
    if 'Ticket' in df.columns:
        ticket_counts = df['Ticket'].value_counts()
        df['TicketGroupSize'] = df['Ticket'].map(ticket_counts)
        df['IsSharedTicket'] = (df['TicketGroupSize'] > 1).astype(int)
        new_features.extend(['TicketGroupSize', 'IsSharedTicket'])

    return df, new_features


def _engineer_general(df: pd.DataFrame, target: str = None) -> tuple:
    """General feature engineering for any dataset."""
    new_features = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target and target in numeric_cols:
        numeric_cols.remove(target)

    # Fill missing values
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Binning for numeric columns
    for col in numeric_cols[:5]:  # Top 5 numeric
        try:
            df[f'{col}_bin'] = pd.qcut(df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                        duplicates='drop')
            new_features.append(f'{col}_bin')
        except Exception:
            pass

    # Interaction between top 2 numeric
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[0], numeric_cols[1]
        df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 0.001)
        df[f'{col1}_{col2}_mult'] = df[col1] * df[col2]
        new_features.extend([f'{col1}_{col2}_ratio', f'{col1}_{col2}_mult'])

    return df, new_features


async def feature_select_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select best features using importance scores.

    Args:
        params: Dict with keys:
            - data: DataFrame
            - target: Target column
            - method: 'importance' or 'correlation'
            - top_k: Number of features to select

    Returns:
        Dict with selected features
    """
    status.set_callback(params.pop('_status_callback', None))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target')
    method = params.get('method', 'importance')
    top_k = params.get('top_k', 10)

    X = data.drop(columns=[target]).copy()
    y = data[target]

    # Encode categoricals
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(X.median())

    if method == 'importance':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)
        selected = importance.head(top_k).index.tolist()
    else:  # correlation
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected = correlations.head(top_k).index.tolist()

    logger.info(f"[FeatureSelect] Selected {len(selected)} features")

    return {
        'success': True,
        'selected_features': selected,
        'all_scores': importance.to_dict() if method == 'importance' else correlations.to_dict(),
    }
