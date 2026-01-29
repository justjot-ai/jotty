"""
Feature Engineering Skill
=========================

Kaggle-style feature engineering extracted from SkillOrchestrator.

Techniques used:
1. Groupby aggregations (MOST POWERFUL)
2. Target encoding (CV-based to prevent leakage)
3. Frequency encoding
4. Binning/Discretization
5. Polynomial features
6. Log transforms
7. NaN pattern encoding
8. Categorical combinations
9. Interaction features
10. Row-level statistics
"""

import time
import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

from .base import MLSkill, SkillResult, SkillCategory

logger = logging.getLogger(__name__)


class FeatureEngineeringSkill(MLSkill):
    """
    Kaggle-style feature engineering skill.

    Creates rich feature sets using proven techniques from top solutions.
    """

    name = "feature_engineering"
    version = "2.0.0"
    description = "Kaggle-style feature engineering with 15+ techniques"
    category = SkillCategory.FEATURE_ENGINEERING

    required_inputs = ["X"]
    optional_inputs = ["y", "problem_type"]
    outputs = ["X_engineered"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._techniques_used = []

    async def execute(self,
                      X: pd.DataFrame,
                      y: Optional[pd.Series] = None,
                      **context) -> SkillResult:
        """
        Execute feature engineering.

        Args:
            X: Input features
            y: Target variable (optional but recommended)
            **context: problem_type, etc.

        Returns:
            SkillResult with engineered features
        """
        start_time = time.time()

        if not self.validate_inputs(X, y):
            return self._create_error_result("Invalid inputs")

        problem_type = context.get('problem_type', 'classification')

        X_eng = X.copy()
        original_cols = list(X_eng.columns)
        n_original = len(original_cols)
        self._techniques_used = []

        try:
            # Pre-processing: Convert Categorical dtype to numeric
            X_eng = self._convert_categorical_to_numeric(X_eng)

            # Identify column types
            numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_eng.select_dtypes(include=['object']).columns.tolist()
            cat_cols_original = cat_cols.copy()

            # Apply all techniques
            X_eng = self._frequency_encoding(X_eng, cat_cols)
            X_eng = self._label_encode_categoricals(X_eng, cat_cols)

            # Update numeric cols after encoding
            numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()

            X_eng = self._groupby_aggregations(X_eng, cat_cols_original, original_cols, numeric_cols)
            X_eng = self._binning(X_eng, numeric_cols)
            X_eng = self._polynomial_features(X_eng, numeric_cols)
            X_eng = self._log_transforms(X_eng, numeric_cols)
            X_eng = self._nan_pattern_encoding(X_eng, X)
            X_eng = self._categorical_combinations(X_eng, cat_cols_original)
            X_eng = self._interaction_features(X_eng, original_cols, numeric_cols)
            X_eng = self._row_level_stats(X_eng, original_cols, numeric_cols)
            X_eng = self._quantile_features(X_eng, numeric_cols)

            # CV-based target encoding (no leakage)
            if y is not None:
                X_eng = self._target_encoding_cv(X_eng, y, cat_cols_original)
                X_eng = self._cv_validated_interactions(X_eng, y, original_cols, numeric_cols, problem_type)

            # Early feature pruning
            X_eng = self._early_pruning(X_eng)

            # Final cleanup
            X_eng = X_eng.fillna(0)
            X_eng = X_eng.replace([np.inf, -np.inf], 0)

            # Encode remaining categoricals
            X_eng = self._encode_remaining_categoricals(X_eng)

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return self._create_error_result(str(e))

        execution_time = time.time() - start_time

        return self._create_result(
            success=True,
            data=X_eng,
            metrics={
                'original_features': n_original,
                'engineered_features': len(X_eng.columns),
                'new_features': len(X_eng.columns) - n_original,
                'techniques_used': len(self._techniques_used),
            },
            metadata={
                'techniques': self._techniques_used,
                'original_columns': original_cols,
            },
            execution_time=execution_time,
        )

    def _convert_categorical_to_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert Categorical dtype to numeric codes."""
        for col in X.columns:
            if X[col].dtype.name == 'category':
                X[col] = X[col].cat.codes
        return X

    def _frequency_encoding(self, X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Frequency encoding for categorical columns."""
        for col in cat_cols[:5]:
            freq = X[col].value_counts(normalize=True)
            X[f'{col}_freq'] = X[col].map(freq).fillna(0)
        self._techniques_used.append('frequency_encoding')
        return X

    def _label_encode_categoricals(self, X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Label encode categorical columns."""
        from sklearn.preprocessing import LabelEncoder
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        self._techniques_used.append('label_encoding')
        return X

    def _groupby_aggregations(self, X: pd.DataFrame, cat_cols: List[str],
                               original_cols: List[str], numeric_cols: List[str]) -> pd.DataFrame:
        """Groupby aggregations - THE MOST POWERFUL technique."""
        groupby_cols = [c for c in cat_cols if c in X.columns][:3]
        agg_cols = [c for c in original_cols if c in numeric_cols][:5]

        for grp_col in groupby_cols:
            for agg_col in agg_cols:
                if grp_col != agg_col:
                    try:
                        grp_mean = X.groupby(grp_col)[agg_col].transform('mean')
                        grp_std = X.groupby(grp_col)[agg_col].transform('std').fillna(0)
                        grp_count = X.groupby(grp_col)[agg_col].transform('count')

                        X[f'{grp_col}_{agg_col}_grp_mean'] = grp_mean
                        X[f'{grp_col}_{agg_col}_grp_std'] = grp_std
                        X[f'{grp_col}_{agg_col}_grp_cnt'] = grp_count
                        X[f'{grp_col}_{agg_col}_dev'] = X[agg_col] - grp_mean
                    except:
                        pass

        self._techniques_used.append('groupby_aggregations')
        return X

    def _binning(self, X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Binning / Discretization."""
        for col in numeric_cols[:5]:
            try:
                X[f'{col}_qbin'] = pd.qcut(X[col], q=5, labels=False, duplicates='drop')
            except:
                pass

            if X[col].std() > 0:
                X[f'{col}_round1'] = X[col].round(1)
                X[f'{col}_round0'] = X[col].round(0)

        self._techniques_used.append('binning')
        return X

    def _polynomial_features(self, X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Polynomial features (squared, sqrt)."""
        for col in numeric_cols[:5]:
            X[f'{col}_sq'] = X[col] ** 2
            X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))

        self._techniques_used.append('polynomial')
        return X

    def _log_transforms(self, X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Log transforms for skewed data."""
        for col in numeric_cols[:5]:
            if (X[col] >= 0).all():
                X[f'{col}_log'] = np.log1p(X[col])

        self._techniques_used.append('log_transform')
        return X

    def _nan_pattern_encoding(self, X: pd.DataFrame, X_original: pd.DataFrame) -> pd.DataFrame:
        """NaN pattern encoding."""
        nan_cols = X_original.columns[X_original.isnull().any()].tolist()
        if len(nan_cols) > 0:
            for col in nan_cols[:10]:
                X[f'{col}_isna'] = X_original[col].isnull().astype(int)

            X['_nan_pattern'] = 0
            for i, col in enumerate(nan_cols[:10]):
                X['_nan_pattern'] += X_original[col].isnull().astype(int) * (2 ** i)

            X['_nan_count'] = X_original[nan_cols].isnull().sum(axis=1)
            self._techniques_used.append('nan_patterns')

        return X

    def _categorical_combinations(self, X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Categorical combinations."""
        encoded_cats = [c for c in cat_cols if c in X.columns][:3]
        for i, col1 in enumerate(encoded_cats):
            for col2 in encoded_cats[i+1:]:
                try:
                    max_val = X[col2].max() + 1
                    X[f'{col1}_{col2}_comb'] = (X[col1] + 1) + (X[col2] + 1) / max_val
                except:
                    pass

        self._techniques_used.append('cat_combinations')
        return X

    def _interaction_features(self, X: pd.DataFrame, original_cols: List[str],
                               numeric_cols: List[str]) -> pd.DataFrame:
        """Interaction features (multiply & divide)."""
        interact_cols = [c for c in original_cols if c in numeric_cols][:6]

        for i, col1 in enumerate(interact_cols):
            for col2 in interact_cols[i+1:]:
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                denom = X[col2].abs() + 0.001
                X[f'{col1}_div_{col2}'] = X[col1] / denom

        self._techniques_used.append('interactions')
        return X

    def _row_level_stats(self, X: pd.DataFrame, original_cols: List[str],
                          numeric_cols: List[str]) -> pd.DataFrame:
        """Row-level aggregations."""
        if len(numeric_cols) >= 3:
            orig_numeric = [c for c in original_cols if c in numeric_cols]
            if len(orig_numeric) >= 3:
                X['_row_sum'] = X[orig_numeric].sum(axis=1)
                X['_row_mean'] = X[orig_numeric].mean(axis=1)
                X['_row_std'] = X[orig_numeric].std(axis=1)
                X['_row_max'] = X[orig_numeric].max(axis=1)
                X['_row_min'] = X[orig_numeric].min(axis=1)
                X['_row_range'] = X['_row_max'] - X['_row_min']
                X['_row_skew'] = X[orig_numeric].skew(axis=1)
                self._techniques_used.append('row_stats')

        return X

    def _quantile_features(self, X: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Quantile features."""
        for col in numeric_cols[:3]:
            try:
                q25 = X[col].quantile(0.25)
                q75 = X[col].quantile(0.75)
                X[f'{col}_below_q25'] = (X[col] < q25).astype(int)
                X[f'{col}_above_q75'] = (X[col] > q75).astype(int)
            except:
                pass

        self._techniques_used.append('quantiles')
        return X

    def _target_encoding_cv(self, X: pd.DataFrame, y: pd.Series,
                             cat_cols: List[str]) -> pd.DataFrame:
        """CV-based target encoding (no leakage)."""
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for col in cat_cols[:5]:
            if col in X.columns:
                try:
                    target_enc = np.zeros(len(X))
                    col_encoded = X[col]

                    for train_idx, val_idx in kf.split(X):
                        train_means = pd.Series(y.iloc[train_idx].values).groupby(
                            col_encoded.iloc[train_idx].values
                        ).mean()
                        target_enc[val_idx] = col_encoded.iloc[val_idx].map(train_means).fillna(y.mean()).values

                    X[f'{col}_target_enc_cv'] = target_enc
                except Exception as e:
                    logger.debug(f"Target encoding failed for {col}: {e}")

        self._techniques_used.append('target_encoding_cv')
        return X

    def _cv_validated_interactions(self, X: pd.DataFrame, y: pd.Series,
                                    original_cols: List[str], numeric_cols: List[str],
                                    problem_type: str) -> pd.DataFrame:
        """CV-validated interactions (only keep if improves score)."""
        try:
            from sklearn.model_selection import cross_val_score
            import lightgbm as lgb

            X_temp = X.fillna(0).replace([np.inf, -np.inf], 0)

            if problem_type == 'classification':
                baseline_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'accuracy'
            else:
                baseline_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'r2'

            baseline_score = cross_val_score(baseline_model, X_temp, y, cv=3, scoring=scoring).mean()

            orig_numeric = [c for c in original_cols if c in numeric_cols][:4]
            validated_interactions = 0

            for i, col1 in enumerate(orig_numeric):
                for col2 in orig_numeric[i+1:]:
                    interaction_name = f'{col1}_x_{col2}_validated'
                    X_test = X_temp.copy()
                    X_test[interaction_name] = X_test[col1] * X_test[col2]

                    test_score = cross_val_score(baseline_model, X_test, y, cv=3, scoring=scoring).mean()

                    if test_score > baseline_score + 0.001:
                        X[interaction_name] = X[col1] * X[col2]
                        baseline_score = test_score
                        validated_interactions += 1

            if validated_interactions > 0:
                self._techniques_used.append('validated_interactions')

        except Exception as e:
            logger.debug(f"CV-validated interactions failed: {e}")

        return X

    def _early_pruning(self, X: pd.DataFrame) -> pd.DataFrame:
        """Early feature pruning - remove useless features."""
        try:
            # Remove constant features
            constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_cols:
                X = X.drop(columns=constant_cols)

            # Remove near-constant features
            near_constant = []
            for col in X.columns:
                top_freq = X[col].value_counts(normalize=True).iloc[0] if len(X[col].value_counts()) > 0 else 0
                if top_freq > 0.99:
                    near_constant.append(col)
            if near_constant:
                X = X.drop(columns=near_constant)

            self._techniques_used.append('early_pruning')
        except Exception as e:
            logger.debug(f"Early pruning failed: {e}")

        return X

    def _encode_remaining_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode any remaining categorical columns."""
        remaining_cats = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if remaining_cats:
            from sklearn.preprocessing import LabelEncoder
            for col in remaining_cats:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                except:
                    X = X.drop(columns=[col])
        return X
