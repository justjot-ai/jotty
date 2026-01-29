"""
Data Validator Skill for Jotty
==============================

Data validation and quality checking for ML pipelines.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


async def validate_schema_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate data against an expected schema.

    Args:
        params: Dict with keys:
            - data: DataFrame to validate
            - schema: Dict with column specifications
              e.g., {'col1': {'type': 'int', 'nullable': False}, ...}
            - strict: If True, fail on extra columns (default False)

    Returns:
        Dict with validation results
    """
    logger.info("[DataValidator] Validating schema...")

    data = params.get('data')
    schema = params.get('schema', {})
    strict = params.get('strict', False)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    errors = []
    warnings = []

    # Check required columns exist
    for col, spec in schema.items():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            continue

        # Check type
        expected_type = spec.get('type')
        if expected_type:
            type_mapping = {
                'int': [np.int64, np.int32, int],
                'float': [np.float64, np.float32, float],
                'str': [object, str],
                'bool': [bool, np.bool_],
                'datetime': ['datetime64[ns]']
            }
            actual_type = df[col].dtype
            expected_types = type_mapping.get(expected_type, [expected_type])
            if actual_type not in expected_types and str(actual_type) not in expected_types:
                errors.append(f"Column '{col}' expected type {expected_type}, got {actual_type}")

        # Check nullable
        nullable = spec.get('nullable', True)
        if not nullable and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} null values but is not nullable")

        # Check min/max
        if 'min' in spec and df[col].min() < spec['min']:
            errors.append(f"Column '{col}' has values below minimum {spec['min']}")
        if 'max' in spec and df[col].max() > spec['max']:
            errors.append(f"Column '{col}' has values above maximum {spec['max']}")

        # Check allowed values
        if 'values' in spec:
            invalid = set(df[col].dropna().unique()) - set(spec['values'])
            if invalid:
                errors.append(f"Column '{col}' has invalid values: {invalid}")

    # Check for extra columns in strict mode
    if strict:
        extra_cols = set(df.columns) - set(schema.keys())
        if extra_cols:
            errors.append(f"Unexpected columns found: {extra_cols}")

    is_valid = len(errors) == 0

    logger.info(f"[DataValidator] Schema validation: {'PASSED' if is_valid else 'FAILED'}")

    return {
        'success': True,
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'columns_checked': list(schema.keys()),
    }


async def validate_quality_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check data quality metrics.

    Args:
        params: Dict with keys:
            - data: DataFrame to validate
            - thresholds: Dict with quality thresholds
              e.g., {'missing_pct': 5, 'duplicate_pct': 1}

    Returns:
        Dict with quality scores and issues
    """
    logger.info("[DataValidator] Checking data quality...")

    data = params.get('data')
    thresholds = params.get('thresholds', {
        'missing_pct': 5,
        'duplicate_pct': 1,
        'cardinality_ratio': 0.95  # Max unique ratio for categoricals
    })

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    issues = []
    metrics = {}

    # Overall stats
    metrics['n_rows'] = len(df)
    metrics['n_columns'] = len(df.columns)

    # Missing values
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    metrics['missing_pct'] = round(missing_pct, 2)
    if missing_pct > thresholds.get('missing_pct', 5):
        issues.append(f"High missing data: {missing_pct:.2f}%")

    # Column-level missing
    col_missing = {}
    for col in df.columns:
        col_miss_pct = df[col].isnull().mean() * 100
        if col_miss_pct > 0:
            col_missing[col] = round(col_miss_pct, 2)
    metrics['columns_with_missing'] = col_missing

    # Duplicates
    n_duplicates = df.duplicated().sum()
    duplicate_pct = (n_duplicates / len(df)) * 100
    metrics['duplicate_rows'] = n_duplicates
    metrics['duplicate_pct'] = round(duplicate_pct, 2)
    if duplicate_pct > thresholds.get('duplicate_pct', 1):
        issues.append(f"High duplicate rate: {duplicate_pct:.2f}%")

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    metrics['constant_columns'] = constant_cols
    if constant_cols:
        issues.append(f"Constant columns found: {constant_cols}")

    # High cardinality categoricals
    high_cardinality = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        cardinality_ratio = df[col].nunique() / len(df)
        if cardinality_ratio > thresholds.get('cardinality_ratio', 0.95):
            high_cardinality.append(col)
    metrics['high_cardinality_cols'] = high_cardinality
    if high_cardinality:
        issues.append(f"High cardinality columns: {high_cardinality}")

    # Compute quality score (0-100)
    quality_score = 100
    quality_score -= min(missing_pct * 2, 30)  # Up to -30 for missing
    quality_score -= min(duplicate_pct * 5, 20)  # Up to -20 for duplicates
    quality_score -= len(constant_cols) * 5  # -5 per constant column
    quality_score -= len(high_cardinality) * 3  # -3 per high cardinality
    quality_score = max(0, quality_score)

    metrics['quality_score'] = round(quality_score, 1)

    logger.info(f"[DataValidator] Quality score: {quality_score:.1f}/100")

    return {
        'success': True,
        'quality_score': round(quality_score, 1),
        'metrics': metrics,
        'issues': issues,
        'is_quality': quality_score >= 70,
    }


async def validate_drift_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect distribution drift between reference and current data.

    Args:
        params: Dict with keys:
            - reference: Reference DataFrame
            - current: Current DataFrame to check
            - columns: Optional list of columns to check
            - threshold: P-value threshold for drift (default 0.05)

    Returns:
        Dict with drift detection results
    """
    from scipy import stats

    logger.info("[DataValidator] Detecting distribution drift...")

    reference = params.get('reference')
    current = params.get('current')
    columns = params.get('columns')
    threshold = params.get('threshold', 0.05)

    if isinstance(reference, str):
        reference = pd.read_csv(reference)
    if isinstance(current, str):
        current = pd.read_csv(current)

    ref_df = reference.copy()
    cur_df = current.copy()

    if columns is None:
        columns = list(set(ref_df.columns) & set(cur_df.columns))

    drift_results = {}
    drifted_columns = []

    for col in columns:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue

        ref_values = ref_df[col].dropna()
        cur_values = cur_df[col].dropna()

        if ref_df[col].dtype in [np.float64, np.int64, float, int]:
            # Numeric: use Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_values, cur_values)
            test_name = 'ks_test'
        else:
            # Categorical: use Chi-square test
            ref_counts = ref_values.value_counts()
            cur_counts = cur_values.value_counts()
            all_categories = set(ref_counts.index) | set(cur_counts.index)

            ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
            cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]

            # Normalize to same total
            ref_freq = np.array(ref_freq) / sum(ref_freq) * 1000
            cur_freq = np.array(cur_freq) / sum(cur_freq) * 1000

            try:
                statistic, p_value = stats.chisquare(cur_freq, ref_freq)
                test_name = 'chi2_test'
            except Exception:
                statistic, p_value = 0, 1
                test_name = 'chi2_test_failed'

        is_drifted = p_value < threshold
        drift_results[col] = {
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_drifted': is_drifted
        }

        if is_drifted:
            drifted_columns.append(col)

    drift_detected = len(drifted_columns) > 0

    logger.info(f"[DataValidator] Drift detected in {len(drifted_columns)} columns")

    return {
        'success': True,
        'drift_detected': drift_detected,
        'drifted_columns': drifted_columns,
        'drift_results': drift_results,
        'threshold': threshold,
        'columns_checked': columns,
    }


async def validate_constraints_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate business constraints on data.

    Args:
        params: Dict with keys:
            - data: DataFrame to validate
            - constraints: List of constraint dicts
              e.g., [
                {'type': 'positive', 'columns': ['price', 'quantity']},
                {'type': 'range', 'column': 'age', 'min': 0, 'max': 120},
                {'type': 'unique', 'columns': ['id']},
                {'type': 'relationship', 'condition': 'end_date >= start_date'},
              ]

    Returns:
        Dict with constraint validation results
    """
    logger.info("[DataValidator] Validating constraints...")

    data = params.get('data')
    constraints = params.get('constraints', [])

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    results = []
    all_passed = True

    for constraint in constraints:
        c_type = constraint.get('type')
        result = {'constraint': constraint, 'passed': True, 'violations': 0, 'message': ''}

        if c_type == 'positive':
            columns = constraint.get('columns', [])
            for col in columns:
                if col in df.columns:
                    violations = (df[col] < 0).sum()
                    if violations > 0:
                        result['passed'] = False
                        result['violations'] = int(violations)
                        result['message'] = f"Column '{col}' has {violations} negative values"
                        all_passed = False

        elif c_type == 'range':
            col = constraint.get('column')
            min_val = constraint.get('min')
            max_val = constraint.get('max')
            if col in df.columns:
                violations = 0
                if min_val is not None:
                    violations += (df[col] < min_val).sum()
                if max_val is not None:
                    violations += (df[col] > max_val).sum()
                if violations > 0:
                    result['passed'] = False
                    result['violations'] = int(violations)
                    result['message'] = f"Column '{col}' has {violations} out-of-range values"
                    all_passed = False

        elif c_type == 'unique':
            columns = constraint.get('columns', [])
            existing_cols = [c for c in columns if c in df.columns]
            if existing_cols:
                duplicates = df.duplicated(subset=existing_cols).sum()
                if duplicates > 0:
                    result['passed'] = False
                    result['violations'] = int(duplicates)
                    result['message'] = f"Columns {existing_cols} have {duplicates} duplicate combinations"
                    all_passed = False

        elif c_type == 'not_null':
            columns = constraint.get('columns', [])
            for col in columns:
                if col in df.columns:
                    nulls = df[col].isnull().sum()
                    if nulls > 0:
                        result['passed'] = False
                        result['violations'] = int(nulls)
                        result['message'] = f"Column '{col}' has {nulls} null values"
                        all_passed = False

        elif c_type == 'relationship':
            condition = constraint.get('condition')
            try:
                violations = (~df.eval(condition)).sum()
                if violations > 0:
                    result['passed'] = False
                    result['violations'] = int(violations)
                    result['message'] = f"Condition '{condition}' violated in {violations} rows"
                    all_passed = False
            except Exception as e:
                result['passed'] = False
                result['message'] = f"Could not evaluate condition: {str(e)}"
                all_passed = False

        results.append(result)

    passed_count = sum(1 for r in results if r['passed'])

    logger.info(f"[DataValidator] {passed_count}/{len(results)} constraints passed")

    return {
        'success': True,
        'all_passed': all_passed,
        'passed_count': passed_count,
        'total_constraints': len(results),
        'results': results,
    }


async def validate_completeness_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check data completeness and coverage.

    Args:
        params: Dict with keys:
            - data: DataFrame to validate
            - required_columns: List of required column names
            - min_rows: Minimum required rows
            - coverage_threshold: Min coverage percentage per column (default 95)

    Returns:
        Dict with completeness metrics
    """
    logger.info("[DataValidator] Checking completeness...")

    data = params.get('data')
    required_columns = params.get('required_columns', [])
    min_rows = params.get('min_rows', 1)
    coverage_threshold = params.get('coverage_threshold', 95)

    if isinstance(data, str):
        data = pd.read_csv(data)

    df = data.copy()
    issues = []

    # Check row count
    if len(df) < min_rows:
        issues.append(f"Insufficient rows: {len(df)} < {min_rows}")

    # Check required columns
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    # Calculate coverage per column
    coverage = {}
    low_coverage = []
    for col in df.columns:
        col_coverage = (1 - df[col].isnull().mean()) * 100
        coverage[col] = round(col_coverage, 2)
        if col_coverage < coverage_threshold:
            low_coverage.append(col)

    if low_coverage:
        issues.append(f"Low coverage columns: {low_coverage}")

    # Overall completeness score
    overall_coverage = (1 - df.isnull().sum().sum() / df.size) * 100
    completeness_score = min(100, overall_coverage * (len(df) >= min_rows))

    is_complete = len(issues) == 0

    logger.info(f"[DataValidator] Completeness: {completeness_score:.1f}%")

    return {
        'success': True,
        'is_complete': is_complete,
        'completeness_score': round(completeness_score, 1),
        'overall_coverage': round(overall_coverage, 2),
        'column_coverage': coverage,
        'low_coverage_columns': low_coverage,
        'issues': issues,
        'row_count': len(df),
    }
