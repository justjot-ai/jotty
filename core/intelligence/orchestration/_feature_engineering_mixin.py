"""SkillOrchestrator mixin — builtin feature engineering stage."""

import logging

import numpy as np
import pandas as pd

from .skill_types import ProblemType, SkillResult

logger = logging.getLogger(__name__)


class FeatureEngineeringMixin:
    """Mixin providing _builtin_feature_engineering for SkillOrchestrator."""

    async def _builtin_feature_engineering(
        self, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType
    ) -> SkillResult:
        """
        Advanced Kaggle-style feature engineering.

        Techniques from top Kaggle solutions:
        1. Groupby aggregations (MOST POWERFUL)
        2. Target encoding
        3. Frequency encoding
        4. Binning/Discretization
        5. Polynomial features
        6. Log transforms
        7. NaN pattern encoding
        8. Categorical combinations
        9. Interaction features
        10. Statistical aggregations
        """
        from sklearn.preprocessing import LabelEncoder

        X_eng = X.copy()
        original_cols = list(X_eng.columns)
        n_original = len(original_cols)

        # ================================================================
        # PRE-PROCESSING: Convert any Categorical dtype to numeric codes
        # This handles columns created by pd.cut/pd.qcut with labels
        # ================================================================
        for col in X_eng.columns:
            if X_eng[col].dtype.name == "category":
                X_eng[col] = X_eng[col].cat.codes

        # Identify column types
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_eng.select_dtypes(include=["object"]).columns.tolist()

        # Store original categorical columns before encoding
        cat_cols_original = cat_cols.copy()

        # ================================================================
        # 1. TARGET ENCODING - DISABLED (causes leakage outside CV)
        # ================================================================
        # NOTE: Target encoding must be done INSIDE cross-validation folds
        # to avoid leakage. Doing it here on full data before CV causes
        # the model to see target information from validation rows.
        # IMPLEMENT: Add proper target encoding with CV-aware pipeline
        # if y is not None and len(cat_cols) > 0:
        #     for col in cat_cols[:5]:
        #         target_mean = X_eng.groupby(col).apply(...)
        #         X_eng[f'{col}_target_enc'] = ...

        # ================================================================
        # 2. FREQUENCY ENCODING (count-based)
        # ================================================================
        for col in cat_cols[:5]:
            freq = X_eng[col].value_counts(normalize=True)
            X_eng[f"{col}_freq"] = X_eng[col].map(freq).fillna(0)

        # ================================================================
        # 3. LABEL ENCODE categoricals (after extracting target/freq)
        # ================================================================
        for col in cat_cols:
            le = LabelEncoder()
            X_eng[col] = le.fit_transform(X_eng[col].astype(str))

        # Update numeric cols after encoding
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()

        # ================================================================
        # 4. GROUPBY AGGREGATIONS (THE MOST POWERFUL - from NVIDIA blog)
        # ================================================================
        # Use encoded categoricals as groupby keys
        groupby_cols = [c for c in cat_cols_original if c in X_eng.columns][:3]
        agg_cols = [c for c in original_cols if c in numeric_cols][:5]

        for grp_col in groupby_cols:
            for agg_col in agg_cols:
                if grp_col != agg_col:
                    try:
                        # Mean, std, count per group
                        grp_mean = X_eng.groupby(grp_col)[agg_col].transform("mean")
                        grp_std = X_eng.groupby(grp_col)[agg_col].transform("std").fillna(0)
                        grp_count = X_eng.groupby(grp_col)[agg_col].transform("count")

                        X_eng[f"{grp_col}_{agg_col}_grp_mean"] = grp_mean
                        X_eng[f"{grp_col}_{agg_col}_grp_std"] = grp_std
                        X_eng[f"{grp_col}_{agg_col}_grp_cnt"] = grp_count

                        # Deviation from group mean
                        X_eng[f"{grp_col}_{agg_col}_dev"] = X_eng[agg_col] - grp_mean
                    except Exception:
                        pass

        # ================================================================
        # 5. BINNING / DISCRETIZATION
        # ================================================================
        for col in numeric_cols[:5]:
            try:
                # Quantile-based binning (5 bins)
                X_eng[f"{col}_qbin"] = pd.qcut(X_eng[col], q=5, labels=False, duplicates="drop")
            except Exception:
                pass

            # Round-based binning
            if X_eng[col].std() > 0:
                X_eng[f"{col}_round1"] = X_eng[col].round(1)
                X_eng[f"{col}_round0"] = X_eng[col].round(0)

        # ================================================================
        # 6. POLYNOMIAL FEATURES (squared, cubed)
        # ================================================================
        for col in numeric_cols[:5]:
            X_eng[f"{col}_sq"] = X_eng[col] ** 2
            X_eng[f"{col}_sqrt"] = np.sqrt(np.abs(X_eng[col]))

        # ================================================================
        # 7. LOG TRANSFORMS (for skewed data)
        # ================================================================
        for col in numeric_cols[:5]:
            if (X_eng[col] > 0).all():
                X_eng[f"{col}_log"] = np.log1p(X_eng[col])
            elif (X_eng[col] >= 0).all():
                X_eng[f"{col}_log"] = np.log1p(X_eng[col])

        # ================================================================
        # 8. NaN PATTERN ENCODING
        # ================================================================
        nan_cols = X.columns[X.isnull().any()].tolist()
        if len(nan_cols) > 0:
            # Binary NaN indicator per column
            for col in nan_cols[:10]:
                X_eng[f"{col}_isna"] = X[col].isnull().astype(int)

            # Combined NaN pattern (binary encoding)
            X_eng["_nan_pattern"] = 0
            for i, col in enumerate(nan_cols[:10]):
                X_eng["_nan_pattern"] += X[col].isnull().astype(int) * (2**i)

            # Total NaN count per row
            X_eng["_nan_count"] = X[nan_cols].isnull().sum(axis=1)

        # ================================================================
        # 9. CATEGORICAL COMBINATIONS
        # ================================================================
        encoded_cats = [c for c in cat_cols_original if c in X_eng.columns][:3]
        for i, col1 in enumerate(encoded_cats):
            for col2 in encoded_cats[i + 1 :]:
                try:
                    max_val = X_eng[col2].max() + 1
                    X_eng[f"{col1}_{col2}_comb"] = (X_eng[col1] + 1) + (X_eng[col2] + 1) / max_val
                except Exception:
                    pass

        # ================================================================
        # 10. INTERACTION FEATURES (multiply & divide)
        # ================================================================
        interact_cols = [c for c in original_cols if c in numeric_cols][:6]

        for i, col1 in enumerate(interact_cols):
            for col2 in interact_cols[i + 1 :]:
                X_eng[f"{col1}_x_{col2}"] = X_eng[col1] * X_eng[col2]
                denom = X_eng[col2].abs() + 0.001
                X_eng[f"{col1}_div_{col2}"] = X_eng[col1] / denom

        # ================================================================
        # 11. ROW-LEVEL AGGREGATIONS
        # ================================================================
        if len(numeric_cols) >= 3:
            orig_numeric = [c for c in original_cols if c in numeric_cols]
            if len(orig_numeric) >= 3:
                X_eng["_row_sum"] = X_eng[orig_numeric].sum(axis=1)
                X_eng["_row_mean"] = X_eng[orig_numeric].mean(axis=1)
                X_eng["_row_std"] = X_eng[orig_numeric].std(axis=1)
                X_eng["_row_max"] = X_eng[orig_numeric].max(axis=1)
                X_eng["_row_min"] = X_eng[orig_numeric].min(axis=1)
                X_eng["_row_range"] = X_eng["_row_max"] - X_eng["_row_min"]
                X_eng["_row_skew"] = X_eng[orig_numeric].skew(axis=1)

        # ================================================================
        # 12. QUANTILE FEATURES
        # ================================================================
        for col in numeric_cols[:3]:
            try:
                q25 = X_eng[col].quantile(0.25)
                q75 = X_eng[col].quantile(0.75)
                X_eng[f"{col}_below_q25"] = (X_eng[col] < q25).astype(int)
                X_eng[f"{col}_above_q75"] = (X_eng[col] > q75).astype(int)
            except Exception:
                pass

        # ================================================================
        # 13. TARGET ENCODING WITH CV (NO LEAKAGE - World Class)
        # ================================================================
        # Proper target encoding: use leave-one-out or K-fold to prevent leakage
        if y is not None and len(cat_cols_original) > 0:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for col in cat_cols_original[:5]:
                if col in X_eng.columns:
                    try:
                        target_enc = np.zeros(len(X_eng))
                        col_encoded = X_eng[col]  # Already label encoded

                        # CV-based target encoding
                        for train_idx, val_idx in kf.split(X_eng):
                            # Calculate target mean on training fold only
                            train_means = (
                                pd.Series(y.iloc[train_idx].values)
                                .groupby(col_encoded.iloc[train_idx].values)
                                .mean()
                            )

                            # Apply to validation fold
                            target_enc[val_idx] = (
                                col_encoded.iloc[val_idx].map(train_means).fillna(y.mean()).values
                            )

                        X_eng[f"{col}_target_enc_cv"] = target_enc
                    except Exception as e:
                        logger.debug(f"Target encoding failed for {col}: {e}")

        # ================================================================
        # 14. CV-VALIDATED INTERACTIONS (Only keep if improves score)
        # ================================================================
        if y is not None and len(numeric_cols) >= 2:
            import lightgbm as lgb
            from sklearn.model_selection import cross_val_score

            # Quick baseline score with current features
            try:
                X_temp = X_eng.fillna(0).replace([np.inf, -np.inf], 0)
                baseline_model = (
                    lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
                    if problem_type == ProblemType.CLASSIFICATION
                    else lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
                )

                baseline_score = cross_val_score(
                    baseline_model,
                    X_temp,
                    y,
                    cv=3,
                    scoring="accuracy" if problem_type == ProblemType.CLASSIFICATION else "r2",
                ).mean()

                # Try top interaction candidates
                orig_numeric = [c for c in original_cols if c in numeric_cols][:4]
                validated_interactions = 0

                for i, col1 in enumerate(orig_numeric):
                    for col2 in orig_numeric[i + 1 :]:
                        # Create candidate interaction
                        interaction_name = f"{col1}_x_{col2}_validated"
                        X_test = X_temp.copy()
                        X_test[interaction_name] = X_test[col1] * X_test[col2]

                        # Test if it improves score
                        test_score = cross_val_score(
                            baseline_model,
                            X_test,
                            y,
                            cv=3,
                            scoring=(
                                "accuracy" if problem_type == ProblemType.CLASSIFICATION else "r2"
                            ),
                        ).mean()

                        if test_score > baseline_score + 0.001:  # Must improve by 0.1%
                            X_eng[interaction_name] = X_eng[col1] * X_eng[col2]
                            baseline_score = test_score
                            validated_interactions += 1

                if validated_interactions > 0:
                    logger.debug(f"Added {validated_interactions} CV-validated interactions")
            except Exception as e:
                logger.debug(f"CV-validated interactions failed: {e}")

        # ================================================================
        # 15. EARLY FEATURE PRUNING (Remove obviously useless features)
        # ================================================================
        # Remove features with near-zero variance or perfect correlation
        try:
            # Remove constant features
            constant_cols = [col for col in X_eng.columns if X_eng[col].nunique() <= 1]
            if constant_cols:
                X_eng = X_eng.drop(columns=constant_cols)

            # Remove features with >99% same value
            near_constant = []
            for col in X_eng.columns:
                top_freq = (
                    X_eng[col].value_counts(normalize=True).iloc[0]
                    if len(X_eng[col].value_counts()) > 0
                    else 0
                )
                if top_freq > 0.99:
                    near_constant.append(col)
            if near_constant:
                X_eng = X_eng.drop(columns=near_constant)

        except Exception as e:
            logger.debug(f"Early pruning failed: {e}")

        # Clean up
        X_eng = X_eng.fillna(0)
        X_eng = X_eng.replace([np.inf, -np.inf], 0)

        # Defragment DataFrame to improve performance (fixes fragmentation warning)
        X_eng = X_eng.copy()

        # ================================================================
        # FINAL: ENCODE ALL REMAINING CATEGORICAL COLUMNS
        # ================================================================
        # LLM may have created new categorical columns - encode them all
        remaining_cats = X_eng.select_dtypes(include=["object", "category"]).columns.tolist()
        if remaining_cats:
            from sklearn.preprocessing import LabelEncoder

            for col in remaining_cats:
                try:
                    le = LabelEncoder()
                    X_eng[col] = le.fit_transform(X_eng[col].astype(str))
                except Exception:
                    # If encoding fails, drop the column
                    X_eng = X_eng.drop(columns=[col])

        return SkillResult(
            skill_name="builtin_kaggle_fe",
            category=SkillCategory.FEATURE_ENGINEERING,
            success=True,
            data=X_eng,
            metrics={
                "original_features": n_original,
                "engineered_features": len(X_eng.columns),
                "new_features": len(X_eng.columns) - n_original,
                "techniques_used": [
                    "frequency_encoding",
                    "groupby_aggs",
                    "binning",
                    "polynomial",
                    "log_transform",
                    "nan_patterns",
                    "cat_combinations",
                    "interactions",
                    "row_stats",
                    "quantiles",
                    "target_encoding_cv",
                    "validated_interactions",
                    "early_pruning",
                ],
            },
        )
