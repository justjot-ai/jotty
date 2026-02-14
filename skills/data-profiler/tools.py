"""
Data Profiler Skill for Jotty
==============================

Comprehensive EDA and data profiling.
"""

import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("data-profiler")


logger = logging.getLogger(__name__)


@async_tool_wrapper()
async def profile_data_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive data profile.

    Args:
        params: Dict with keys:
            - data: DataFrame or path to CSV
            - target: Optional target column for analysis

    Returns:
        Dict with shape, dtypes, missing, stats, correlations, recommendations
    """
    status.set_callback(params.pop('_status_callback', None))

    logger.info("[DataProfiler] Generating data profile...")

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target')
    df = data.copy()

    profile = {
        'shape': {'rows': df.shape[0], 'columns': df.shape[1]},
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }

    # Column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    profile['column_types'] = {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
    }

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    profile['missing'] = {
        col: {'count': int(missing[col]), 'percent': float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }
    profile['missing_summary'] = {
        'total_missing_cells': int(missing.sum()),
        'columns_with_missing': int((missing > 0).sum()),
        'total_missing_percent': round(missing.sum() / df.size * 100, 2),
    }

    # Numeric statistics
    if numeric_cols:
        stats = df[numeric_cols].describe().T
        stats['skew'] = df[numeric_cols].skew()
        stats['kurtosis'] = df[numeric_cols].kurtosis()
        profile['numeric_stats'] = stats.to_dict()

    # Categorical statistics
    if categorical_cols:
        cat_stats = {}
        for col in categorical_cols:
            cat_stats[col] = {
                'unique': int(df[col].nunique()),
                'top_values': df[col].value_counts().head(5).to_dict(),
                'missing': int(df[col].isnull().sum()),
            }
        profile['categorical_stats'] = cat_stats

    # Correlations (numeric only)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        profile['high_correlations'] = high_corr

    # Target analysis
    if target and target in df.columns:
        if target in numeric_cols:
            profile['target_analysis'] = {
                'type': 'numeric',
                'stats': df[target].describe().to_dict(),
            }
        else:
            profile['target_analysis'] = {
                'type': 'categorical',
                'distribution': df[target].value_counts().to_dict(),
                'balance_ratio': round(
                    df[target].value_counts().min() / df[target].value_counts().max(), 3
                ),
            }

    # Recommendations
    recommendations = []
    if profile['missing_summary']['total_missing_percent'] > 5:
        recommendations.append("High missing data - consider imputation strategies")
    if len(profile.get('high_correlations', [])) > 0:
        recommendations.append("Multicollinearity detected - consider feature selection")
    for col, stats in profile.get('numeric_stats', {}).items():
        if abs(stats.get('skew', 0)) > 2:
            recommendations.append(f"Column '{col}' is highly skewed - consider transformation")

    profile['recommendations'] = recommendations

    logger.info(f"[DataProfiler] Profile complete: {len(df.columns)} columns, {len(df)} rows")

    return {
        'success': True,
        'profile': profile,
    }


@async_tool_wrapper()
async def detect_outliers_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect outliers in numeric columns.

    Args:
        params: Dict with keys:
            - data: DataFrame or path
            - method: 'iqr', 'zscore', or 'isolation_forest'
            - threshold: Threshold for detection

    Returns:
        Dict with outliers per column and indices
    """
    status.set_callback(params.pop('_status_callback', None))

    from scipy import stats

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    method = params.get('method', 'iqr')
    threshold = params.get('threshold', 1.5 if method == 'iqr' else 3)

    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers = {}
    all_outlier_indices = set()

    for col in numeric_cols:
        col_data = df[col].dropna()

        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (df[col] < lower) | (df[col] > upper)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(col_data))
            mask = pd.Series(False, index=df.index)
            mask.loc[col_data.index] = z_scores > threshold

        else:  # isolation_forest
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            col_reshaped = col_data.values.reshape(-1, 1)
            preds = clf.fit_predict(col_reshaped)
            mask = pd.Series(False, index=df.index)
            mask.loc[col_data.index] = preds == -1

        outlier_idx = df[mask].index.tolist()
        if outlier_idx:
            outliers[col] = {
                'count': len(outlier_idx),
                'percent': round(len(outlier_idx) / len(df) * 100, 2),
                'indices': outlier_idx[:20],  # First 20
            }
            all_outlier_indices.update(outlier_idx)

    logger.info(f"[DataProfiler] Found {len(all_outlier_indices)} outlier rows")

    return {
        'success': True,
        'method': method,
        'outliers_by_column': outliers,
        'total_outlier_rows': len(all_outlier_indices),
        'outlier_indices': list(all_outlier_indices)[:100],
    }


@async_tool_wrapper()
async def analyze_correlations_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze correlations between features and optionally with target.

    Args:
        params: Dict with keys:
            - data: DataFrame or path
            - target: Optional target column
            - method: 'pearson', 'spearman', or 'kendall'

    Returns:
        Dict with correlation matrix and feature importance
    """
    status.set_callback(params.pop('_status_callback', None))

    data = params.get('data')
    if isinstance(data, str):
        data = pd.read_csv(data)

    target = params.get('target')
    method = params.get('method', 'pearson')

    df = data.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {'success': False, 'error': 'Need at least 2 numeric columns'}

    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr(method=method)

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': round(corr_val, 4),
                    'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                })

    result = {
        'success': True,
        'method': method,
        'correlation_matrix': corr_matrix.round(4).to_dict(),
        'high_correlation_pairs': sorted(high_corr_pairs,
                                          key=lambda x: abs(x['correlation']),
                                          reverse=True),
    }

    # Target correlations
    if target and target in numeric_cols:
        target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
        result['target_correlations'] = target_corr.round(4).to_dict()
        result['top_features'] = target_corr.abs().sort_values(ascending=False).head(10).index.tolist()

    return result
