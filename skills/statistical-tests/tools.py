"""
Statistical Tests Skill for Jotty
=================================

Statistical hypothesis testing and analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


async def ttest_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform t-test for mean comparison.

    Args:
        params: Dict with keys:
            - sample1: First sample (array or column name)
            - sample2: Second sample (for two-sample test) or population mean (for one-sample)
            - data: Optional DataFrame if using column names
            - test_type: 'one-sample', 'two-sample', 'paired' (default 'two-sample')
            - alternative: 'two-sided', 'less', 'greater' (default 'two-sided')
            - alpha: Significance level (default 0.05)

    Returns:
        Dict with t-statistic, p-value, and interpretation
    """
    logger.info("[Stats] Running t-test...")

    sample1 = params.get('sample1')
    sample2 = params.get('sample2')
    data = params.get('data')
    test_type = params.get('test_type', 'two-sample')
    alternative = params.get('alternative', 'two-sided')
    alpha = params.get('alpha', 0.05)

    # Get samples from dataframe if needed
    if data is not None:
        if isinstance(data, str):
            data = pd.read_csv(data)
        if isinstance(sample1, str):
            sample1 = data[sample1].dropna().values
        if isinstance(sample2, str) and test_type != 'one-sample':
            sample2 = data[sample2].dropna().values

    sample1 = np.array(sample1)

    if test_type == 'one-sample':
        popmean = float(sample2) if sample2 is not None else 0
        t_stat, p_value = stats.ttest_1samp(sample1, popmean, alternative=alternative)
        test_desc = f"One-sample t-test (pop mean = {popmean})"
    elif test_type == 'paired':
        sample2 = np.array(sample2)
        t_stat, p_value = stats.ttest_rel(sample1, sample2, alternative=alternative)
        test_desc = "Paired t-test"
    else:  # two-sample
        sample2 = np.array(sample2)
        t_stat, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)
        test_desc = "Independent two-sample t-test"

    # Effect size (Cohen's d)
    if test_type == 'two-sample':
        pooled_std = np.sqrt((sample1.std()**2 + sample2.std()**2) / 2)
        cohens_d = (sample1.mean() - sample2.mean()) / pooled_std
    elif test_type == 'paired':
        cohens_d = (sample1 - sample2).mean() / (sample1 - sample2).std()
    else:
        cohens_d = (sample1.mean() - float(sample2 if sample2 else 0)) / sample1.std()

    is_significant = p_value < alpha

    logger.info(f"[Stats] {test_desc}: t={t_stat:.4f}, p={p_value:.4f}")

    return {
        'success': True,
        'test_type': test_desc,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'alpha': alpha,
        'cohens_d': float(cohens_d),
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
        'sample1_mean': float(sample1.mean()),
        'sample1_std': float(sample1.std()),
        'interpretation': f"{'Reject' if is_significant else 'Fail to reject'} null hypothesis at alpha={alpha}",
    }


async def anova_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform ANOVA (Analysis of Variance).

    Args:
        params: Dict with keys:
            - groups: List of arrays (one per group) or dict of group_name: values
            - data: Optional DataFrame
            - group_col: Column containing group labels (if data provided)
            - value_col: Column containing values (if data provided)
            - alpha: Significance level (default 0.05)

    Returns:
        Dict with F-statistic, p-value, and post-hoc results
    """
    logger.info("[Stats] Running ANOVA...")

    groups = params.get('groups')
    data = params.get('data')
    group_col = params.get('group_col')
    value_col = params.get('value_col')
    alpha = params.get('alpha', 0.05)

    # Get groups from dataframe if provided
    if data is not None and group_col and value_col:
        if isinstance(data, str):
            data = pd.read_csv(data)
        groups = [data[data[group_col] == g][value_col].dropna().values
                  for g in data[group_col].unique()]
        group_names = data[group_col].unique().tolist()
    elif isinstance(groups, dict):
        group_names = list(groups.keys())
        groups = [np.array(v) for v in groups.values()]
    else:
        group_names = [f'Group_{i}' for i in range(len(groups))]
        groups = [np.array(g) for g in groups]

    # One-way ANOVA
    f_stat, p_value = stats.f_oneway(*groups)

    # Effect size (eta-squared)
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    is_significant = p_value < alpha

    # Post-hoc (Tukey HSD) if significant
    posthoc = None
    if is_significant and len(groups) > 2:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            labels = np.concatenate([[name] * len(g) for name, g in zip(group_names, groups)])
            tukey = pairwise_tukeyhsd(all_data, labels, alpha=alpha)
            posthoc = {
                'comparisons': [
                    {
                        'group1': str(tukey.groupsunique[int(i)]),
                        'group2': str(tukey.groupsunique[int(j)]),
                        'meandiff': float(tukey.meandiffs[idx]),
                        'p_adj': float(tukey.pvalues[idx]),
                        'reject': bool(tukey.reject[idx])
                    }
                    for idx, (i, j) in enumerate(zip(*np.triu_indices(len(tukey.groupsunique), 1)))
                ]
            }
        except Exception as e:
            logger.warning(f"Post-hoc test failed: {e}")

    # Group statistics
    group_stats = {
        name: {
            'mean': float(g.mean()),
            'std': float(g.std()),
            'n': len(g)
        }
        for name, g in zip(group_names, groups)
    }

    logger.info(f"[Stats] ANOVA: F={f_stat:.4f}, p={p_value:.4f}")

    return {
        'success': True,
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'alpha': alpha,
        'eta_squared': float(eta_squared),
        'effect_size': 'large' if eta_squared > 0.14 else 'medium' if eta_squared > 0.06 else 'small',
        'n_groups': len(groups),
        'group_stats': group_stats,
        'posthoc_tukey': posthoc,
    }


async def chisquare_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform Chi-square test for independence or goodness of fit.

    Args:
        params: Dict with keys:
            - observed: Observed frequencies (1D for goodness-of-fit, 2D for independence)
            - expected: Expected frequencies (optional for goodness-of-fit)
            - data: Optional DataFrame
            - col1, col2: Column names for contingency table (if data provided)
            - alpha: Significance level (default 0.05)

    Returns:
        Dict with chi-square statistic, p-value, and interpretation
    """
    logger.info("[Stats] Running Chi-square test...")

    observed = params.get('observed')
    expected = params.get('expected')
    data = params.get('data')
    col1 = params.get('col1')
    col2 = params.get('col2')
    alpha = params.get('alpha', 0.05)

    # Build contingency table from data if provided
    if data is not None and col1 and col2:
        if isinstance(data, str):
            data = pd.read_csv(data)
        contingency = pd.crosstab(data[col1], data[col2])
        observed = contingency.values
        test_type = 'independence'
    elif observed is not None:
        observed = np.array(observed)
        test_type = 'independence' if observed.ndim == 2 else 'goodness_of_fit'
    else:
        return {'success': False, 'error': 'No data provided'}

    if test_type == 'independence':
        chi2, p_value, dof, expected_freq = stats.chi2_contingency(observed)
        # Cramér's V for effect size
        n = observed.sum()
        min_dim = min(observed.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    else:
        if expected is None:
            expected = np.full(len(observed), observed.sum() / len(observed))
        chi2, p_value = stats.chisquare(observed, expected)
        dof = len(observed) - 1
        expected_freq = expected
        cramers_v = None

    is_significant = p_value < alpha

    logger.info(f"[Stats] Chi-square: χ²={chi2:.4f}, p={p_value:.4f}")

    return {
        'success': True,
        'test_type': test_type,
        'chi2_statistic': float(chi2),
        'p_value': float(p_value),
        'degrees_of_freedom': int(dof),
        'is_significant': is_significant,
        'alpha': alpha,
        'cramers_v': float(cramers_v) if cramers_v is not None else None,
        'observed': observed.tolist(),
        'expected': expected_freq.tolist() if isinstance(expected_freq, np.ndarray) else expected_freq,
    }


async def normality_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test for normality using multiple tests.

    Args:
        params: Dict with keys:
            - data: Array or DataFrame column
            - column: Column name if DataFrame
            - alpha: Significance level (default 0.05)

    Returns:
        Dict with test results and recommendation
    """
    logger.info("[Stats] Testing normality...")

    data = params.get('data')
    column = params.get('column')
    alpha = params.get('alpha', 0.05)

    if isinstance(data, str):
        data = pd.read_csv(data)

    if isinstance(data, pd.DataFrame):
        if column:
            values = data[column].dropna().values
        else:
            values = data.iloc[:, 0].dropna().values
    else:
        values = np.array(data)

    results = {}

    # Shapiro-Wilk (best for n < 5000)
    if len(values) <= 5000:
        stat, p = stats.shapiro(values)
        results['shapiro_wilk'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'is_normal': p > alpha
        }

    # D'Agostino-Pearson (n >= 20)
    if len(values) >= 20:
        stat, p = stats.normaltest(values)
        results['dagostino_pearson'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'is_normal': p > alpha
        }

    # Anderson-Darling
    ad_result = stats.anderson(values, dist='norm')
    # Use 5% significance level
    critical_idx = 2  # 5% critical value
    results['anderson_darling'] = {
        'statistic': float(ad_result.statistic),
        'critical_value_5pct': float(ad_result.critical_values[critical_idx]),
        'is_normal': ad_result.statistic < ad_result.critical_values[critical_idx]
    }

    # Skewness and Kurtosis
    skewness = float(stats.skew(values))
    kurtosis = float(stats.kurtosis(values))

    # Overall verdict
    normal_votes = sum(1 for r in results.values() if r.get('is_normal', False))
    is_normal = normal_votes > len(results) / 2

    logger.info(f"[Stats] Normality: {normal_votes}/{len(results)} tests passed")

    return {
        'success': True,
        'test_results': results,
        'is_normal': is_normal,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'n_samples': len(values),
        'recommendation': 'Data appears normally distributed' if is_normal else 'Consider non-parametric tests',
    }


async def correlation_test_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test correlation significance between variables.

    Args:
        params: Dict with keys:
            - x: First variable
            - y: Second variable
            - data: Optional DataFrame
            - method: 'pearson', 'spearman', 'kendall' (default 'pearson')
            - alpha: Significance level (default 0.05)

    Returns:
        Dict with correlation coefficient, p-value, and confidence interval
    """
    logger.info("[Stats] Testing correlation...")

    x = params.get('x')
    y = params.get('y')
    data = params.get('data')
    method = params.get('method', 'pearson')
    alpha = params.get('alpha', 0.05)

    if data is not None:
        if isinstance(data, str):
            data = pd.read_csv(data)
        if isinstance(x, str):
            x = data[x].values
        if isinstance(y, str):
            y = data[y].values

    x = np.array(x)
    y = np.array(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if method == 'pearson':
        corr, p_value = stats.pearsonr(x, y)
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(x, y)
    elif method == 'kendall':
        corr, p_value = stats.kendalltau(x, y)
    else:
        return {'success': False, 'error': f'Unknown method: {method}'}

    is_significant = p_value < alpha

    # Confidence interval for Pearson
    ci = None
    if method == 'pearson' and len(x) > 3:
        z = np.arctanh(corr)
        se = 1 / np.sqrt(len(x) - 3)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lower = np.tanh(z - z_crit * se)
        ci_upper = np.tanh(z + z_crit * se)
        ci = [float(ci_lower), float(ci_upper)]

    # Strength interpretation
    abs_corr = abs(corr)
    if abs_corr < 0.3:
        strength = 'weak'
    elif abs_corr < 0.7:
        strength = 'moderate'
    else:
        strength = 'strong'

    logger.info(f"[Stats] {method.capitalize()} correlation: r={corr:.4f}, p={p_value:.4f}")

    return {
        'success': True,
        'method': method,
        'correlation': float(corr),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'alpha': alpha,
        'confidence_interval': ci,
        'strength': strength,
        'direction': 'positive' if corr > 0 else 'negative',
        'n_samples': len(x),
    }


async def mannwhitney_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        params: Dict with keys:
            - sample1: First sample
            - sample2: Second sample
            - alternative: 'two-sided', 'less', 'greater' (default 'two-sided')
            - alpha: Significance level (default 0.05)

    Returns:
        Dict with U statistic, p-value, and effect size
    """
    logger.info("[Stats] Running Mann-Whitney U test...")

    sample1 = np.array(params.get('sample1'))
    sample2 = np.array(params.get('sample2'))
    alternative = params.get('alternative', 'two-sided')
    alpha = params.get('alpha', 0.05)

    u_stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)

    # Effect size (rank-biserial correlation)
    n1, n2 = len(sample1), len(sample2)
    r = 1 - (2 * u_stat) / (n1 * n2)

    is_significant = p_value < alpha

    logger.info(f"[Stats] Mann-Whitney U: U={u_stat:.4f}, p={p_value:.4f}")

    return {
        'success': True,
        'u_statistic': float(u_stat),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'alpha': alpha,
        'rank_biserial_r': float(r),
        'effect_size': 'large' if abs(r) > 0.5 else 'medium' if abs(r) > 0.3 else 'small',
        'sample1_median': float(np.median(sample1)),
        'sample2_median': float(np.median(sample2)),
    }
