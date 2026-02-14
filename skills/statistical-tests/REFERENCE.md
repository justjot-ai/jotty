# Statistical Tests Skill - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`ttest_tool`](#ttest_tool) | Perform t-test for mean comparison. |
| [`anova_tool`](#anova_tool) | Perform ANOVA (Analysis of Variance). |
| [`chisquare_tool`](#chisquare_tool) | Perform Chi-square test for independence or goodness of fit. |
| [`normality_tool`](#normality_tool) | Test for normality using multiple tests. |
| [`correlation_test_tool`](#correlation_test_tool) | Test correlation significance between variables. |
| [`mannwhitney_tool`](#mannwhitney_tool) | Mann-Whitney U test (non-parametric alternative to t-test). |

---

## `ttest_tool`

Perform t-test for mean comparison.

**Parameters:**

- **sample1**: First sample (array or column name)
- **sample2**: Second sample (for two-sample test) or population mean (for one-sample)
- **data**: Optional DataFrame if using column names
- **test_type**: 'one-sample', 'two-sample', 'paired' (default 'two-sample')
- **alternative**: 'two-sided', 'less', 'greater' (default 'two-sided')
- **alpha**: Significance level (default 0.05)

**Returns:** Dict with t-statistic, p-value, and interpretation

---

## `anova_tool`

Perform ANOVA (Analysis of Variance).

**Parameters:**

- **groups**: List of arrays (one per group) or dict of group_name: values
- **data**: Optional DataFrame
- **group_col**: Column containing group labels (if data provided)
- **value_col**: Column containing values (if data provided)
- **alpha**: Significance level (default 0.05)

**Returns:** Dict with F-statistic, p-value, and post-hoc results

---

## `chisquare_tool`

Perform Chi-square test for independence or goodness of fit.

**Parameters:**

- **observed**: Observed frequencies (1D for goodness-of-fit, 2D for independence)
- **expected**: Expected frequencies (optional for goodness-of-fit)
- **data**: Optional DataFrame col1, col2: Column names for contingency table (if data provided)
- **alpha**: Significance level (default 0.05)

**Returns:** Dict with chi-square statistic, p-value, and interpretation

---

## `normality_tool`

Test for normality using multiple tests.

**Parameters:**

- **data**: Array or DataFrame column
- **column**: Column name if DataFrame
- **alpha**: Significance level (default 0.05)

**Returns:** Dict with test results and recommendation

---

## `correlation_test_tool`

Test correlation significance between variables.

**Parameters:**

- **x**: First variable
- **y**: Second variable
- **data**: Optional DataFrame
- **method**: 'pearson', 'spearman', 'kendall' (default 'pearson')
- **alpha**: Significance level (default 0.05)

**Returns:** Dict with correlation coefficient, p-value, and confidence interval

---

## `mannwhitney_tool`

Mann-Whitney U test (non-parametric alternative to t-test).

**Parameters:**

- **sample1**: First sample
- **sample2**: Second sample
- **alternative**: 'two-sided', 'less', 'greater' (default 'two-sided')
- **alpha**: Significance level (default 0.05)

**Returns:** Dict with U statistic, p-value, and effect size
