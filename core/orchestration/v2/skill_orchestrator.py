"""
SkillOrchestrator - Auto-chain Skills for Any ML Task
======================================================

Automatically discovers, sequences, and executes ML skills
to solve ANY machine learning problem.

Usage:
    orchestrator = SkillOrchestrator()
    result = await orchestrator.solve(X, y)

The orchestrator:
1. Auto-detects problem type (classification/regression)
2. Discovers relevant skills from registry
3. Chains skills in optimal order
4. Executes pipeline and returns best model

Features:
- Progress tracking with visual progress bar
- LLM-powered feature reasoning from multiple perspectives
"""

import logging
import asyncio
import sys
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# PROGRESS TRACKER
# ============================================================
class ProgressTracker:
    """Visual progress tracking for ML pipeline stages."""

    STAGE_WEIGHTS = {
        'DATA_PROFILING': 2,
        'DATA_CLEANING': 3,
        'LLM_FEATURE_REASONING': 10,
        'FEATURE_ENGINEERING': 8,
        'FEATURE_SELECTION': 5,
        'MODEL_SELECTION': 25,
        'HYPERPARAMETER_OPTIMIZATION': 30,
        'ENSEMBLE': 15,
        'EVALUATION': 1,
        'EXPLANATION': 1,
    }

    def __init__(self, total_stages: int = 9):
        self.total_stages = total_stages
        self.current_stage = 0
        self.current_stage_name = ""
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.completed_weight = 0
        self.total_weight = sum(self.STAGE_WEIGHTS.values())

    def start_stage(self, stage_name: str):
        """Start a new stage."""
        self.current_stage += 1
        self.current_stage_name = stage_name
        self.stage_start_time = time.time()
        self._print_progress()

    def complete_stage(self, stage_name: str, metrics: Dict = None):
        """Complete current stage."""
        weight = self.STAGE_WEIGHTS.get(stage_name.upper(), 5)
        self.completed_weight += weight
        elapsed = time.time() - self.stage_start_time
        self._print_completion(stage_name, elapsed, metrics)

    def _print_progress(self):
        """Print progress bar."""
        pct = (self.completed_weight / self.total_weight) * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        elapsed = time.time() - self.start_time

        sys.stdout.write(f'\r[{bar}] {pct:5.1f}% | Stage {self.current_stage}/{self.total_stages}: {self.current_stage_name:<25} | {elapsed:.0f}s')
        sys.stdout.flush()

    def _print_completion(self, stage_name: str, elapsed: float, metrics: Dict = None):
        """Print stage completion."""
        pct = (self.completed_weight / self.total_weight) * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        total_elapsed = time.time() - self.start_time

        metric_str = ""
        if metrics:
            key_metrics = {k: v for k, v in list(metrics.items())[:3]}
            metric_str = " | " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in key_metrics.items())

        print(f'\r[{bar}] {pct:5.1f}% | ✅ {stage_name:<25} ({elapsed:.1f}s){metric_str}')

    def finish(self, final_score: float):
        """Print final summary."""
        total_time = time.time() - self.start_time
        print(f'\n{"="*60}')
        print(f'🏆 COMPLETE | Score: {final_score:.4f} | Total time: {total_time:.1f}s')
        print(f'{"="*60}\n')


# ============================================================
# EDA ANALYZER - Data-Driven Insights
# ============================================================
class EDAAnalyzer:
    """
    Automated EDA that generates actionable insights for feature engineering.

    Analyzes:
    1. Distributions (skewness, modality, outliers)
    2. Correlations (with target and between features)
    3. Missing patterns (randomness, correlation with target)
    4. Categorical patterns (cardinality, target rates)
    5. Interactions (non-linear relationships)
    """

    def analyze(self, X: pd.DataFrame, y: pd.Series, problem_type: str) -> Dict[str, Any]:
        """Run comprehensive EDA and return actionable insights."""
        insights = {
            'distributions': self._analyze_distributions(X),
            'correlations': self._analyze_correlations(X, y),
            'missing': self._analyze_missing(X, y),
            'categorical': self._analyze_categorical(X, y),
            'outliers': self._analyze_outliers(X),
            'interactions': self._analyze_interactions(X, y),
            'recommendations': []
        }

        # Generate actionable recommendations
        insights['recommendations'] = self._generate_recommendations(insights, X, y)

        return insights

    def _analyze_distributions(self, X: pd.DataFrame) -> Dict:
        """Analyze feature distributions."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        dist_info = {}
        for col in numeric_cols:
            data = X[col].dropna()
            if len(data) < 10:
                continue

            skewness = data.skew()
            kurtosis = data.kurtosis()
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1

            dist_info[col] = {
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'is_skewed': abs(skewness) > 1,
                'is_heavy_tailed': kurtosis > 3,
                'range': float(data.max() - data.min()),
                'cv': float(data.std() / (data.mean() + 0.001)),  # Coefficient of variation
                'n_unique': int(data.nunique()),
                'pct_unique': float(data.nunique() / len(data)),
            }

            # Detect potential bimodality (simple heuristic)
            median = data.median()
            below_median_mean = data[data < median].mean() if len(data[data < median]) > 0 else median
            above_median_mean = data[data >= median].mean() if len(data[data >= median]) > 0 else median
            gap = (above_median_mean - below_median_mean) / (data.std() + 0.001)
            dist_info[col]['potential_bimodal'] = gap > 1.5

        return dist_info

    def _analyze_correlations(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze correlations with target and between features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        corr_info = {
            'with_target': {},
            'feature_pairs': [],
            'high_correlation_pairs': []
        }

        if y is None or len(numeric_cols) == 0:
            return corr_info

        # Correlation with target
        for col in numeric_cols:
            try:
                corr = X[col].corr(y)
                if not np.isnan(corr):
                    corr_info['with_target'][col] = {
                        'correlation': float(corr),
                        'abs_correlation': float(abs(corr)),
                        'is_strong': abs(corr) > 0.3,
                        'is_weak': abs(corr) < 0.1,
                    }
            except:
                pass

        # Feature-feature correlations (find highly correlated pairs)
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr) and abs(corr) > 0.5:
                        corr_info['high_correlation_pairs'].append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': float(corr)
                        })

        return corr_info

    def _analyze_missing(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze missing value patterns."""
        missing_info = {
            'columns': {},
            'total_missing_pct': float(X.isnull().sum().sum() / X.size * 100),
            'predictive_missing': []
        }

        for col in X.columns:
            missing_pct = X[col].isnull().mean() * 100
            if missing_pct > 0:
                missing_info['columns'][col] = {
                    'missing_pct': float(missing_pct),
                    'is_high_missing': missing_pct > 30,
                }

                # Check if missingness correlates with target
                if y is not None:
                    missing_mask = X[col].isnull()
                    if missing_mask.any() and not missing_mask.all():
                        target_when_missing = y[missing_mask].mean()
                        target_when_present = y[~missing_mask].mean()
                        diff = abs(target_when_missing - target_when_present)

                        if diff > 0.1 * y.std():
                            missing_info['columns'][col]['predictive'] = True
                            missing_info['columns'][col]['target_diff'] = float(diff)
                            missing_info['predictive_missing'].append(col)

        return missing_info

    def _analyze_categorical(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze categorical features."""
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        cat_info = {}
        for col in cat_cols:
            n_unique = X[col].nunique()
            value_counts = X[col].value_counts()

            cat_info[col] = {
                'n_unique': int(n_unique),
                'is_high_cardinality': n_unique > 20,
                'is_binary': n_unique == 2,
                'top_values': value_counts.head(5).to_dict(),
                'rare_categories': int((value_counts < len(X) * 0.01).sum()),
            }

            # Target rate by category (for classification)
            if y is not None:
                try:
                    target_rates = X.groupby(col).apply(lambda g: y.iloc[g.index].mean())
                    cat_info[col]['target_rate_range'] = float(target_rates.max() - target_rates.min())
                    cat_info[col]['is_predictive'] = cat_info[col]['target_rate_range'] > 0.2
                except:
                    pass

        return cat_info

    def _analyze_outliers(self, X: pd.DataFrame) -> Dict:
        """Analyze outliers in numeric features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        outlier_info = {}
        for col in numeric_cols:
            data = X[col].dropna()
            if len(data) < 10:
                continue

            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            n_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_pct = n_outliers / len(data) * 100

            if outlier_pct > 1:  # More than 1% outliers
                outlier_info[col] = {
                    'n_outliers': int(n_outliers),
                    'outlier_pct': float(outlier_pct),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                }

        return outlier_info

    def _analyze_interactions(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze potential feature interactions."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()[:10]  # Limit for speed

        interactions = []

        if y is None or len(numeric_cols) < 2:
            return {'promising_interactions': interactions}

        # Check if product/ratio of features improves correlation
        base_corrs = {col: abs(X[col].corr(y)) for col in numeric_cols if not np.isnan(X[col].corr(y))}

        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i+1:6]:
                try:
                    # Product interaction
                    product = X[col1] * X[col2]
                    product_corr = abs(product.corr(y))

                    # Ratio interaction
                    ratio = X[col1] / (X[col2].abs() + 0.001)
                    ratio_corr = abs(ratio.corr(y))

                    max_base = max(base_corrs.get(col1, 0), base_corrs.get(col2, 0))

                    if product_corr > max_base + 0.05:
                        interactions.append({
                            'type': 'product',
                            'features': [col1, col2],
                            'correlation': float(product_corr),
                            'improvement': float(product_corr - max_base)
                        })

                    if ratio_corr > max_base + 0.05:
                        interactions.append({
                            'type': 'ratio',
                            'features': [col1, col2],
                            'correlation': float(ratio_corr),
                            'improvement': float(ratio_corr - max_base)
                        })
                except:
                    pass

        # Sort by improvement
        interactions.sort(key=lambda x: x['improvement'], reverse=True)

        return {'promising_interactions': interactions[:10]}

    def _generate_recommendations(self, insights: Dict, X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """Generate actionable feature engineering recommendations."""
        recommendations = []

        # Get correlation info for filtering weak features
        corr_with_target = insights['correlations'].get('with_target', {})

        # Distribution-based recommendations - ONLY for predictive features
        for col, dist in insights['distributions'].items():
            # Check if feature has meaningful correlation with target
            col_corr = corr_with_target.get(col, {}).get('abs_correlation', 0)
            is_predictive = col_corr > 0.1  # Only transform if |corr| > 0.1

            if dist['is_skewed'] and dist['skewness'] > 0 and is_predictive:
                recommendations.append({
                    'type': 'transform',
                    'feature': col,
                    'action': 'log_transform',
                    'reason': f"Skewed (skew={dist['skewness']:.2f}) AND predictive (corr={col_corr:.2f})",
                    'code': f"X['{col}_log'] = np.log1p(X['{col}'].clip(lower=0))"
                })
            elif dist['is_skewed'] and not is_predictive:
                # Skip - log message only, no feature added
                pass  # Don't add noise for weakly correlated features

            if dist['potential_bimodal'] and is_predictive:
                recommendations.append({
                    'type': 'transform',
                    'feature': col,
                    'action': 'binning',
                    'reason': f'Bimodal AND predictive (corr={col_corr:.2f})',
                    'code': f"X['{col}_bin'] = pd.qcut(X['{col}'], q=3, labels=[0,1,2], duplicates='drop')"
                })

        # Correlation-based recommendations
        for pair in insights['correlations'].get('high_correlation_pairs', []):
            recommendations.append({
                'type': 'interaction',
                'features': [pair['feature1'], pair['feature2']],
                'action': 'ratio',
                'reason': f"High correlation ({pair['correlation']:.2f}) - ratio may capture unique info",
                'code': f"X['{pair['feature1']}_div_{pair['feature2']}'] = X['{pair['feature1']}'] / (X['{pair['feature2']}'].abs() + 0.001)"
            })

        # Missing-based recommendations
        for col in insights['missing'].get('predictive_missing', []):
            recommendations.append({
                'type': 'missing',
                'feature': col,
                'action': 'missing_indicator',
                'reason': 'Missingness is predictive of target',
                'code': f"X['{col}_is_missing'] = X['{col}'].isnull().astype(int)"
            })

        # Categorical recommendations
        for col, cat in insights['categorical'].items():
            if cat.get('is_predictive', False):
                recommendations.append({
                    'type': 'categorical',
                    'feature': col,
                    'action': 'target_encode',
                    'reason': f"High target rate variance ({cat['target_rate_range']:.2f})",
                    'code': f"# Target encode {col}"
                })

            if cat.get('rare_categories', 0) > 3:
                recommendations.append({
                    'type': 'categorical',
                    'feature': col,
                    'action': 'group_rare',
                    'reason': f"{cat['rare_categories']} rare categories",
                    'code': f"# Group rare categories in {col}"
                })

        # Interaction recommendations
        for interaction in insights['interactions'].get('promising_interactions', [])[:5]:
            col1, col2 = interaction['features']
            if interaction['type'] == 'product':
                recommendations.append({
                    'type': 'interaction',
                    'features': [col1, col2],
                    'action': 'product',
                    'reason': f"Product improves correlation by {interaction['improvement']:.2f}",
                    'code': f"X['{col1}_x_{col2}'] = X['{col1}'] * X['{col2}']"
                })
            else:
                recommendations.append({
                    'type': 'interaction',
                    'features': [col1, col2],
                    'action': 'ratio',
                    'reason': f"Ratio improves correlation by {interaction['improvement']:.2f}",
                    'code': f"X['{col1}_div_{col2}'] = X['{col1}'] / (X['{col2}'].abs() + 0.001)"
                })

        # Outlier recommendations
        for col, outlier in insights['outliers'].items():
            if outlier['outlier_pct'] > 5:
                recommendations.append({
                    'type': 'outlier',
                    'feature': col,
                    'action': 'outlier_indicator',
                    'reason': f"{outlier['outlier_pct']:.1f}% outliers",
                    'code': f"X['{col}_is_outlier'] = ((X['{col}'] < {outlier['lower_bound']:.2f}) | (X['{col}'] > {outlier['upper_bound']:.2f})).astype(int)"
                })

        return recommendations

    def format_for_llm(self, insights: Dict) -> str:
        """Format EDA insights as text for LLM consumption."""
        lines = ["## EDA INSIGHTS FOR FEATURE ENGINEERING\n"]

        # Distribution insights
        skewed = [col for col, d in insights['distributions'].items() if d['is_skewed']]
        if skewed:
            lines.append(f"**Skewed features (need log transform):** {skewed[:5]}")

        bimodal = [col for col, d in insights['distributions'].items() if d.get('potential_bimodal')]
        if bimodal:
            lines.append(f"**Bimodal features (need binning):** {bimodal[:5]}")

        # Correlation insights
        strong_corr = [col for col, c in insights['correlations'].get('with_target', {}).items()
                      if c.get('is_strong')]
        if strong_corr:
            lines.append(f"**Strong predictors:** {strong_corr[:5]}")

        # High correlation pairs
        pairs = insights['correlations'].get('high_correlation_pairs', [])
        if pairs:
            pair_str = [f"{p['feature1']}-{p['feature2']}" for p in pairs[:3]]
            lines.append(f"**Correlated feature pairs:** {pair_str}")

        # Missing patterns
        predictive_missing = insights['missing'].get('predictive_missing', [])
        if predictive_missing:
            lines.append(f"**Predictive missing (create indicators):** {predictive_missing}")

        # Categorical insights
        predictive_cats = [col for col, c in insights['categorical'].items()
                         if c.get('is_predictive')]
        if predictive_cats:
            lines.append(f"**Predictive categoricals:** {predictive_cats}")

        # Promising interactions
        interactions = insights['interactions'].get('promising_interactions', [])
        if interactions:
            int_str = [f"{i['features'][0]}*{i['features'][1]}" if i['type']=='product'
                      else f"{i['features'][0]}/{i['features'][1]}" for i in interactions[:3]]
            lines.append(f"**Promising interactions:** {int_str}")

        # Top recommendations
        lines.append("\n**TOP RECOMMENDATIONS:**")
        for rec in insights['recommendations'][:10]:
            lines.append(f"- {rec['action']} on {rec.get('feature', rec.get('features'))}: {rec['reason']}")

        return "\n".join(lines)


# ============================================================
# LLM FEATURE REASONER
# ============================================================
class LLMFeatureReasoner:
    """
    Use LLM to reason about features from multiple business perspectives.

    Perspectives:
    1. Text Engineer - Extract features from text/string columns
    2. Domain Expert - Business/domain-specific features
    3. Data Science Head - Statistical patterns, interactions, transformations
    """

    PERSPECTIVE_PROMPTS = {
        'text_engineer': """You are a Text/String Feature Engineer for {problem_type} to predict {target}.

Features available: {features}
String columns with samples: {string_samples}

CRITICAL: Extract features from TEXT/STRING columns by analyzing the sample values above.
Look for patterns like:
- Titles or prefixes in text (extract with regex)
- First letter/prefix patterns
- Group sizes (count rows with same value)
- Length of text fields
- Categorical encoding

Analyze the sample values and generate extraction code.
Return ONLY executable Python code like:
X['new_feature'] = X['column'].str.extract(r'pattern', expand=False)
X['prefix'] = X['column'].str[0]
X['group_size'] = X.groupby('column')['column'].transform('count')

Code only, no explanations:""",

        'domain': """You are a Domain Expert analyzing data for {problem_type} to predict {target}.

Features available: {features}
Context: {context}

Generate 3-5 domain-relevant features. Think about:
- What combinations of features make business sense
- Family/group indicators
- Per-unit metrics (e.g., fare per person)
- Risk/priority segments
- Categorical groupings

Return ONLY executable Python code lines like:
X['family_size'] = X['SibSp'] + X['Parch'] + 1
X['is_alone'] = (X['family_size'] == 1).astype(int)
X['fare_per_person'] = X['Fare'] / (X['family_size'] + 0.001)

Code only, no explanations:""",

        'ds': """You are a Data Science Head analyzing data for {problem_type} to predict {target}.

Features available: {features}

Generate 5-8 statistical features for NUMERIC columns only. Think about:
- Log transforms for skewed distributions
- Polynomial features (squared)
- Ratio features between related numeric columns
- Outlier indicators (above/below percentiles)
- Binning numeric values

Return ONLY executable Python code for numeric columns like:
X['Age_log'] = np.log1p(X['Age'].clip(lower=0))
X['Fare_squared'] = X['Fare'] ** 2
X['Age_bin'] = pd.cut(X['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4])

Code only, no explanations:""",

        'group_analyst': """You are a Group/Aggregation Feature Engineer for {problem_type} to predict {target}.

Features available: {features}
String columns with samples: {string_samples}

CRITICAL: Find columns where multiple rows share the SAME VALUE (groups/clusters).
For each groupable column, create:
1. GROUP SIZE - count of rows with same value
2. GROUP AGGREGATIONS - mean/std of numeric columns per group

Look at sample values - if values repeat across rows, it's groupable.

Return ONLY executable Python code like:
X['column_group_size'] = X.groupby('column')['column'].transform('count')
X['column_numeric_mean'] = X.groupby('column')['numeric_col'].transform('mean')

Code only, no explanations:""",

        'distribution_analyst': """You are a Distribution/Outlier Analyst for {problem_type} to predict {target}.

Features available: {features}

Analyze numeric columns for:
1. SKEWED distributions - apply log1p transform
2. OUTLIERS - create binary outlier indicators (values beyond 1.5*IQR)
3. HEAVY TAILS - apply sqrt or clip extreme values
4. BIMODAL patterns - create binary split indicators

Return ONLY executable Python code like:
X['col_log'] = np.log1p(X['col'].clip(lower=0))
X['col_outlier'] = (X['col'] > X['col'].quantile(0.95)).astype(int)
X['col_sqrt'] = np.sqrt(X['col'].clip(lower=0))

Code only, no explanations:""",

        'interaction_analyst': """You are a Feature Interaction Analyst for {problem_type} to predict {target}.

Features available: {features}

Find MEANINGFUL interactions between features:
1. PRODUCTS - multiply related features (age * class, size * price)
2. RATIOS - divide related features (price / quantity, amount / count)
3. DIFFERENCES - subtract related features
4. LOGICAL combinations - AND/OR of binary features

Focus on features that LOGICALLY relate to each other.

Return ONLY executable Python code like:
X['feat1_x_feat2'] = X['feat1'] * X['feat2']
X['feat1_div_feat2'] = X['feat1'] / (X['feat2'] + 0.001)
X['feat1_minus_feat2'] = X['feat1'] - X['feat2']

Code only, no explanations:""",

        'missing_analyst': """You are a Missing Data Pattern Analyst for {problem_type} to predict {target}.

Features available: {features}

Analyze MISSING data patterns - missingness often carries information!
1. MISSING INDICATORS - binary flag for missing values
2. MISSING COUNT - how many features are missing per row
3. MISSING PATTERN - combination of missing fields

Return ONLY executable Python code like:
X['col_is_missing'] = X['col'].isnull().astype(int)
X['missing_count'] = X.isnull().sum(axis=1)

Code only, no explanations:""",

        'ratio_analyst': """You are a Ratio/Proportion Feature Engineer for {problem_type} to predict {target}.

Features available: {features}

Create MEANINGFUL ratio and proportion features:
1. PER-UNIT ratios - value divided by count/size (price per person, cost per item)
2. PERCENTAGE of total - part / whole
3. RELATIVE measures - value compared to group average
4. NORMALIZED values - value / max or value / sum

Focus on ratios that have REAL-WORLD meaning.

Return ONLY executable Python code like:
X['value_per_unit'] = X['value'] / (X['count'] + 0.001)
X['pct_of_total'] = X['part'] / (X['whole'] + 0.001)
X['relative_to_mean'] = X['value'] / (X['value'].mean() + 0.001)

Code only, no explanations:""",

        'binning_analyst': """You are a Binning/Discretization Expert for {problem_type} to predict {target}.

Features available: {features}

Create MEANINGFUL numeric bins from continuous variables:
1. AGE GROUPS - child=0, teenager=1, adult=2, middle-aged=3, senior=4
2. VALUE BRACKETS - low=0, medium=1, high=2 based on distribution
3. QUANTILE BINS - equal-frequency bins with numeric labels
4. CUSTOM BINS - domain-specific meaningful cutoffs with NUMERIC labels

IMPORTANT: Always use NUMERIC labels (0,1,2,3...) NOT string labels!
Bins capture NON-LINEAR relationships that linear models miss.

Return ONLY executable Python code like:
X['age_group'] = pd.cut(X['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4])
X['value_bin'] = pd.qcut(X['value'], q=4, labels=[0,1,2,3], duplicates='drop')
X['is_child'] = (X['Age'] < 12).astype(int)

Code only, no explanations:""",

        'rare_category_analyst': """You are a Rare Category Handler for {problem_type} to predict {target}.

Features available: {features}
String columns with samples: {string_samples}

Handle RARE/INFREQUENT categories that add noise:
1. RARE FLAGS - is this value rare (appears < 1% of data)?
2. GROUP RARE - combine rare categories into 'Other' or 'Rare'
3. FREQUENCY RANK - rank categories by frequency
4. IS_COMMON - binary flag for common vs rare values

Rare categories cause overfitting - group or flag them!

Return ONLY executable Python code like:
X['col_is_rare'] = X['col'].map(X['col'].value_counts(normalize=True) < 0.01).astype(int)
X['col_freq_rank'] = X['col'].map(X['col'].value_counts().rank(ascending=False))

Code only, no explanations:""",

        'clustering_analyst': """You are a Clustering/Segmentation Expert for {problem_type} to predict {target}.

Features available: {features}

Find HIDDEN SEGMENTS in the data using clustering:
1. CLUSTER ASSIGNMENT - which cluster does each row belong to?
2. DISTANCE TO CENTER - how far from cluster center?
3. SEGMENT FLAGS - binary flags for key segments
4. DENSITY - how crowded is the local neighborhood?

Use numeric features to find natural groupings.

Return ONLY executable Python code like:
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
numeric_cols = X.select_dtypes(include=[np.number]).columns[:5]
X_scaled = StandardScaler().fit_transform(X[numeric_cols].fillna(0))
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
X['cluster'] = kmeans.labels_
X['cluster_distance'] = kmeans.transform(X_scaled).min(axis=1)

Code only, no explanations:""",

        'frequency_analyst': """You are a Frequency/Count Feature Engineer for {problem_type} to predict {target}.

Features available: {features}
String columns with samples: {string_samples}

Create COUNT-BASED features:
1. FREQUENCY ENCODING - replace category with its count/frequency
2. COUNT IN GROUP - how many rows have this value?
3. RELATIVE FREQUENCY - frequency compared to most common
4. LOG FREQUENCY - log of count (for skewed distributions)

Frequency features are powerful for tree-based models!

Return ONLY executable Python code like:
X['col_count'] = X['col'].map(X['col'].value_counts())
X['col_freq'] = X['col'].map(X['col'].value_counts(normalize=True))
X['col_log_count'] = np.log1p(X['col'].map(X['col'].value_counts()))

Code only, no explanations:""",

        'eda_driven': """You are an Expert Data Scientist using EDA insights to create high-quality features.

{eda_insights}

Context: {context}
Features: {features}

CRITICAL RULES:
1. ONLY create features for PREDICTIVE columns (correlation > 0.1 with target)
2. DO NOT transform weakly correlated features - they add noise
3. Focus on the STRONG PREDICTORS identified in the EDA
4. Interactions should combine predictive features, not weak ones

Based on the EDA insights, generate 3-5 HIGH-QUALITY features that:
- Transform ONLY the strong predictors
- Create interactions between predictive features
- Skip anything weakly correlated with target

Return ONLY executable Python code:
X['feature_name'] = <transformation>

Code only, no explanations:"""
    }

    def __init__(self):
        self._llm = None
        self._llm_available = None

    def _init_llm(self):
        """Initialize LLM client using core.llm module."""
        if self._llm_available is None:
            try:
                from core.llm import generate_text
                # Test if it works
                self._llm = generate_text
                self._llm_available = True
                logger.info("LLM Feature Reasoner: Using Claude CLI")
            except Exception as e:
                logger.warning(f"LLM not available: {e}")
                self._llm_available = False
        return self._llm_available

    async def reason_features(self, X: pd.DataFrame, y: pd.Series,
                              problem_type: str, context: str = "") -> List[Dict]:
        """
        Generate feature ideas using EDA insights + LLM reasoning.

        Process:
        1. Run EDA to discover data patterns
        2. Apply rule-based features from EDA recommendations
        3. Feed EDA insights to LLM for intelligent feature generation

        Returns list of feature suggestions with code.
        """
        features = list(X.columns)
        target = y.name if y is not None and hasattr(y, 'name') else "target"

        suggestions = []

        # ================================================================
        # STEP 1: RUN EDA ANALYSIS
        # ================================================================
        print("\n   📊 Running EDA Analysis...")
        eda = EDAAnalyzer()
        eda_insights = eda.analyze(X, y, problem_type)
        eda_text = eda.format_for_llm(eda_insights)

        # Show EDA summary
        n_skewed = len([c for c, d in eda_insights['distributions'].items() if d['is_skewed']])
        n_interactions = len(eda_insights['interactions'].get('promising_interactions', []))
        n_predictive_cats = len([c for c, d in eda_insights['categorical'].items() if d.get('is_predictive')])
        print(f"   📈 EDA: {n_skewed} skewed features, {n_interactions} promising interactions, {n_predictive_cats} predictive categoricals")

        # ================================================================
        # STEP 2: APPLY EDA-BASED RULE FEATURES
        # ================================================================
        for rec in eda_insights['recommendations']:
            if rec.get('code') and not rec['code'].startswith('#'):
                suggestions.append({
                    'perspective': 'eda_rule',
                    'code': rec['code'],
                    'source': 'eda',
                    'reason': rec['reason']
                })

        if suggestions:
            print(f"   🔧 EDA Rules: {len(suggestions)} features from data patterns")

        # ================================================================
        # STEP 3: DETECT STRING COLUMNS FOR TEXT EXTRACTION
        # ================================================================
        string_cols = X.select_dtypes(include=['object']).columns.tolist()
        string_samples = {}
        for col in string_cols[:5]:
            samples = X[col].dropna().head(3).tolist()
            string_samples[col] = samples

        string_samples_str = "\n".join([
            f"  {col}: {samples}" for col, samples in string_samples.items()
        ]) if string_samples else "No string columns"

        # ================================================================
        # STEP 4: LLM FEATURE REASONING WITH EDA CONTEXT
        # ================================================================
        llm_available = self._init_llm()

        if llm_available and self._llm:
            # Skip eda_driven - it creates redundant features with other prompts
            prompts_to_use = {k: v for k, v in self.PERSPECTIVE_PROMPTS.items()
                             if k != 'eda_driven'}

            for perspective, prompt_template in prompts_to_use.items():
                try:
                    # Build prompt with EDA insights
                    full_prompt = prompt_template.format(
                        problem_type=problem_type,
                        features=features[:20],
                        target=target,
                        string_samples=string_samples_str,
                        context=context or "General prediction task",
                        eda_insights=eda_text,
                    )

                    # Add business context if not already in template
                    if context and '{context}' not in prompt_template:
                        full_prompt = f"Business Context:\n{context}\n\n{full_prompt}"

                    # Call LLM
                    response = self._llm(full_prompt, provider="claude-cli", timeout=45)

                    # Parse response
                    parsed = self._parse_llm_response(response, perspective)
                    suggestions.extend(parsed)

                    if parsed:
                        print(f"\n   💡 LLM ({perspective}): {len(parsed)} features")
                        for p in parsed[:3]:
                            print(f"      {p['code'][:70]}...")

                except Exception as e:
                    logger.debug(f"LLM reasoning failed for {perspective}: {e}")

        # ================================================================
        # STEP 5: ADD RULE-BASED FALLBACK
        # ================================================================
        rule_suggestions = self._rule_based_reasoning(X, y, problem_type)
        suggestions.extend(rule_suggestions)

        return suggestions

    def _parse_llm_response(self, response: str, perspective: str) -> List[Dict]:
        """Parse LLM response into feature suggestions."""
        suggestions = []
        # Simple parsing - look for code patterns
        lines = response.split('\n')
        for line in lines:
            if '=' in line and ('X[' in line or 'df[' in line):
                suggestions.append({
                    'perspective': perspective,
                    'code': line.strip(),
                    'source': 'llm'
                })
        return suggestions

    def _rule_based_reasoning(self, X: pd.DataFrame, y: pd.Series,
                              problem_type: str) -> List[Dict]:
        """Fallback rule-based feature generation."""
        suggestions = []
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()

        # Business perspective rules
        for col in numeric_cols[:5]:
            if any(kw in col.lower() for kw in ['price', 'cost', 'amount', 'value', 'revenue']):
                suggestions.append({
                    'perspective': 'business',
                    'code': f"X['{col}_log'] = np.log1p(X['{col}'].clip(lower=0))",
                    'source': 'rule',
                    'reason': 'Log transform for monetary values'
                })

        # Product perspective rules
        for col in numeric_cols[:5]:
            if any(kw in col.lower() for kw in ['count', 'visits', 'sessions', 'clicks', 'views']):
                suggestions.append({
                    'perspective': 'product',
                    'code': f"X['{col}_is_active'] = (X['{col}'] > 0).astype(int)",
                    'source': 'rule',
                    'reason': 'Binary activity indicator'
                })

        # DS perspective rules
        for i, col1 in enumerate(numeric_cols[:3]):
            for col2 in numeric_cols[i+1:4]:
                suggestions.append({
                    'perspective': 'ds',
                    'code': f"X['{col1}_per_{col2}'] = X['{col1}'] / (X['{col2}'] + 0.001)",
                    'source': 'rule',
                    'reason': 'Ratio feature'
                })

        return suggestions

    def apply_suggestions(self, X: pd.DataFrame, suggestions: List[Dict],
                          drop_text_cols: bool = True) -> pd.DataFrame:
        """
        Apply feature suggestions to dataframe.

        Args:
            X: Input dataframe
            suggestions: List of feature suggestions with code
            drop_text_cols: If True, drop original text columns after extraction
                           to prevent data leakage from unique identifiers (e.g., names)
        """
        X_new = X.copy()
        applied = 0

        # Remember original text columns (potential leakage sources)
        original_text_cols = X_new.select_dtypes(include=['object']).columns.tolist()

        # DEDUPLICATION: Extract feature names and skip duplicates
        created_features = set()
        dedup_suggestions = []
        for suggestion in suggestions:
            code = suggestion.get('code', '')
            # Extract feature name from X['feature_name'] = ...
            match = re.search(r"X\['([^']+)'\]\s*=", code)
            if match:
                feat_name = match.group(1)
                if feat_name not in created_features:
                    created_features.add(feat_name)
                    dedup_suggestions.append(suggestion)
            else:
                dedup_suggestions.append(suggestion)

        if len(dedup_suggestions) < len(suggestions):
            print(f"   🔄 Deduplicated: {len(suggestions)} → {len(dedup_suggestions)} suggestions")

        for suggestion in dedup_suggestions:
            try:
                code = suggestion.get('code', '')
                # Safe execution with limited scope
                local_vars = {'X': X_new, 'np': np, 'pd': pd}
                exec(code, {'__builtins__': {}}, local_vars)
                X_new = local_vars.get('X', X_new)
                applied += 1

                # SAFETY: Convert any Categorical columns to numeric codes immediately
                # This handles pd.cut/pd.qcut with labels that create Categorical dtype
                for col in X_new.columns:
                    if hasattr(X_new[col], 'cat') or X_new[col].dtype.name == 'category':
                        X_new[col] = X_new[col].cat.codes

            except Exception as e:
                logger.debug(f"Could not apply suggestion: {e}")

        # POST-PROCESSING: Apply smart transformations for common patterns
        X_new = self._post_process_features(X_new)

        # DROP only HIGH-CARDINALITY text columns to prevent data leakage
        # (unique names, long descriptions, etc. can leak target info)
        # KEEP low-cardinality categoricals (like Sex, Embarked) - they're useful features
        if drop_text_cols and original_text_cols:
            n_rows = len(X_new)
            # Only drop columns with many unique values (>50 or >10% of rows)
            HIGH_CARDINALITY_THRESHOLD = max(50, int(n_rows * 0.1))

            cols_to_drop = []
            cols_kept = []
            for col in original_text_cols:
                if col in X_new.columns:
                    n_unique = X_new[col].nunique()
                    if n_unique > HIGH_CARDINALITY_THRESHOLD:
                        cols_to_drop.append(col)
                    else:
                        cols_kept.append(col)

            if cols_to_drop:
                X_new = X_new.drop(columns=cols_to_drop)
                print(f"   🛡️ Dropped {len(cols_to_drop)} text columns to prevent leakage: {cols_to_drop}")
            if cols_kept:
                print(f"   ✅ Kept {len(cols_kept)} low-cardinality categoricals: {cols_kept}")

        logger.info(f"Applied {applied}/{len(dedup_suggestions)} LLM feature suggestions")
        return X_new

    def _post_process_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        GENERIC post-processing - no hardcoded domain knowledge.

        All transformations are data-driven:
        1. Auto-group rare categories (any high-cardinality categorical)
        2. Smart imputation using correlated features
        3. Relative features within groups (auto-detected)
        4. Optimal range detection for numeric features
        """
        X_new = X.copy()

        # ================================================================
        # 1. AUTO-GROUP RARE CATEGORIES (Generic - works on ANY categorical)
        # ================================================================
        cat_cols = X_new.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            value_counts = X_new[col].value_counts(normalize=True)
            n_unique = len(value_counts)

            # Only group if high cardinality (>10 unique) and has rare values
            if n_unique > 10:
                # Group categories appearing in <1% of data as 'Rare'
                rare_cats = value_counts[value_counts < 0.01].index.tolist()
                if len(rare_cats) > 2:
                    X_new[f'{col}_grouped'] = X_new[col].apply(
                        lambda x: 'Rare' if x in rare_cats else x
                    )
                    new_unique = X_new[f'{col}_grouped'].nunique()
                    print(f"   📝 Auto-grouped {col}: {n_unique} → {new_unique} categories")

        # ================================================================
        # 2. SMART IMPUTATION - Use correlated features (Generic)
        # ================================================================
        numeric_cols = X_new.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            missing_count = X_new[col].isnull().sum()
            if missing_count > 0 and missing_count < len(X_new) * 0.5:
                # Find best correlated categorical for group-based imputation
                best_group_col = None
                best_variance_reduction = 0

                for cat_col in cat_cols[:5]:  # Check top 5 categoricals
                    if cat_col in X_new.columns:
                        try:
                            group_medians = X_new.groupby(cat_col)[col].median()
                            if len(group_medians) > 1:
                                variance_reduction = group_medians.std() / (X_new[col].std() + 0.001)
                                if variance_reduction > best_variance_reduction:
                                    best_variance_reduction = variance_reduction
                                    best_group_col = cat_col
                        except:
                            pass

                # If found a good grouping column, use it for imputation
                if best_group_col and best_variance_reduction > 0.1:
                    group_medians = X_new.groupby(best_group_col)[col].transform('median')
                    X_new[col] = X_new[col].fillna(group_medians)
                    X_new[col] = X_new[col].fillna(X_new[col].median())
                    print(f"   🔧 Smart imputation: {col} by {best_group_col} ({missing_count} filled)")

        # ================================================================
        # 3. RELATIVE FEATURES WITHIN GROUPS (Generic - auto-detect groups)
        # ================================================================
        # Find categorical columns that create meaningful groups
        for cat_col in cat_cols[:3]:
            if cat_col in X_new.columns:
                n_groups = X_new[cat_col].nunique()
                if 2 <= n_groups <= 20:  # Reasonable number of groups
                    for num_col in numeric_cols[:3]:
                        if num_col in X_new.columns:
                            try:
                                group_median = X_new.groupby(cat_col)[num_col].transform('median')
                                if group_median.std() > 0:
                                    relative_col = f'{num_col}_rel_{cat_col}'
                                    X_new[relative_col] = X_new[num_col] / (group_median + 0.001)
                            except:
                                pass

        # ================================================================
        # 4. OPTIMAL RANGE DETECTION (Generic - find sweet spots in data)
        # ================================================================
        # For numeric columns, find if there's an "optimal range" correlated with target
        # This replaces the hardcoded "family_size 2-4" logic
        # Note: This is done in feature engineering, not here, to avoid target leakage

        return X_new


class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"


class SkillCategory(Enum):
    DATA_PROFILING = "data_profiling"
    DATA_CLEANING = "data_cleaning"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ENSEMBLE = "ensemble"
    EVALUATION = "evaluation"
    EXPLANATION = "explanation"


@dataclass
class SkillResult:
    """Result from a skill execution."""
    skill_name: str
    category: SkillCategory
    success: bool
    data: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result from full pipeline execution."""
    problem_type: ProblemType
    best_score: float
    best_model: Any
    feature_count: int
    skill_results: List[SkillResult] = field(default_factory=list)
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


class SkillAdapter:
    """
    Adapter that wraps skills with a standardized ML interface.
    Converts skill tools to fit/transform/predict pattern.
    """

    def __init__(self, skill_name: str, skill_def: Dict, tools_registry: Any):
        self.skill_name = skill_name
        self.skill_def = skill_def
        self.tools_registry = tools_registry
        self.category = self._infer_category()

    def _infer_category(self) -> SkillCategory:
        """Infer skill category from name/description."""
        name = self.skill_name.lower()
        desc = str(self.skill_def.get('description', '')).lower()

        if 'profil' in name or 'profil' in desc:
            return SkillCategory.DATA_PROFILING
        elif 'clean' in name or 'valid' in name:
            return SkillCategory.DATA_CLEANING
        elif 'feature' in name and 'select' in name:
            return SkillCategory.FEATURE_SELECTION
        elif 'feature' in name or 'engineer' in name:
            return SkillCategory.FEATURE_ENGINEERING
        elif 'hyper' in name or 'optim' in name or 'optuna' in name:
            return SkillCategory.HYPERPARAMETER_OPTIMIZATION
        elif 'ensemble' in name or 'stack' in name or 'blend' in name:
            return SkillCategory.ENSEMBLE
        elif 'automl' in name or 'auto-ml' in name or 'model' in name:
            return SkillCategory.MODEL_SELECTION
        elif 'metric' in name or 'eval' in name:
            return SkillCategory.EVALUATION
        elif 'shap' in name or 'explain' in name or 'interpret' in name:
            return SkillCategory.EXPLANATION
        else:
            return SkillCategory.FEATURE_ENGINEERING

    async def execute(self, context: Dict) -> SkillResult:
        """Execute the skill with given context."""
        try:
            # Get tool functions from skill
            tools = self.skill_def.get('tools', [])
            if not tools:
                return SkillResult(
                    skill_name=self.skill_name,
                    category=self.category,
                    success=False,
                    error="No tools defined"
                )

            # Execute based on category
            result = await self._execute_by_category(context)
            return result

        except Exception as e:
            logger.warning(f"Skill {self.skill_name} failed: {e}")
            return SkillResult(
                skill_name=self.skill_name,
                category=self.category,
                success=False,
                error=str(e)
            )

    async def _execute_by_category(self, context: Dict) -> SkillResult:
        """Execute skill based on its category."""
        # This is where we'd call actual skill tools
        # For now, use built-in implementations
        return SkillResult(
            skill_name=self.skill_name,
            category=self.category,
            success=True,
            data=context.get('X'),
            metadata={'executed': True}
        )


class SkillOrchestrator:
    """
    Orchestrates ML skills to solve any machine learning problem.

    Automatically:
    - Detects problem type
    - Discovers relevant skills
    - Chains skills in optimal order
    - Executes pipeline
    - Returns best model
    """

    # Optimal skill execution order
    PIPELINE_ORDER = [
        SkillCategory.DATA_PROFILING,
        SkillCategory.DATA_CLEANING,
        SkillCategory.FEATURE_ENGINEERING,
        SkillCategory.FEATURE_SELECTION,
        SkillCategory.MODEL_SELECTION,
        SkillCategory.HYPERPARAMETER_OPTIMIZATION,
        SkillCategory.ENSEMBLE,
        SkillCategory.EVALUATION,
        SkillCategory.EXPLANATION,
    ]

    def __init__(self, use_llm_features: bool = True, show_progress: bool = True):
        self._skills_registry = None
        self._tools_registry = None
        self._skill_adapters: Dict[str, SkillAdapter] = {}
        self._initialized = False
        self._use_llm_features = use_llm_features
        self._show_progress = show_progress
        self._llm_reasoner = LLMFeatureReasoner() if use_llm_features else None
        self._progress = None

    async def init(self):
        """Initialize registries and discover skills."""
        if self._initialized:
            return

        try:
            from ...registry.skills_registry import get_skills_registry
            from ...registry.tools_registry import get_tools_registry

            self._skills_registry = get_skills_registry()
            self._skills_registry.init()
            self._tools_registry = get_tools_registry()

            # Build skill adapters
            self._build_skill_adapters()
            self._initialized = True

            logger.info(f"SkillOrchestrator initialized with {len(self._skill_adapters)} skill adapters")

        except Exception as e:
            logger.warning(f"SkillOrchestrator init failed: {e}")
            self._initialized = True  # Continue with built-in implementations

    def _build_skill_adapters(self):
        """Build adapters for all ML-related skills."""
        ml_skill_keywords = [
            'data', 'feature', 'model', 'automl', 'hyperopt', 'ensemble',
            'metric', 'shap', 'pycaret', 'sklearn', 'xgboost', 'lightgbm',
            'clustering', 'dimensionality', 'statistical', 'time-series'
        ]

        for skill_name, skill_def in self._skills_registry.loaded_skills.items():
            # Check if skill is ML-related
            name_lower = skill_name.lower()
            if any(kw in name_lower for kw in ml_skill_keywords):
                adapter = SkillAdapter(skill_name, skill_def, self._tools_registry)
                self._skill_adapters[skill_name] = adapter
                logger.debug(f"Created adapter for {skill_name} ({adapter.category})")

    def _detect_problem_type(self, y: pd.Series) -> ProblemType:
        """Auto-detect problem type from target variable."""
        if y is None:
            return ProblemType.CLUSTERING

        unique_ratio = y.nunique() / len(y)
        n_unique = y.nunique()

        # Check dtype
        if y.dtype == 'object' or y.dtype == 'bool':
            return ProblemType.CLASSIFICATION

        # Check if categorical (few unique values)
        if n_unique <= 20 and unique_ratio < 0.05:
            return ProblemType.CLASSIFICATION

        return ProblemType.REGRESSION

    def _get_skills_by_category(self, category: SkillCategory) -> List[SkillAdapter]:
        """Get all skills for a category."""
        return [
            adapter for adapter in self._skill_adapters.values()
            if adapter.category == category
        ]

    async def solve(self,
                    X: pd.DataFrame,
                    y: pd.Series = None,
                    problem_type: str = "auto",
                    target_metric: str = "auto",
                    time_budget: int = 300,
                    business_context: str = "") -> PipelineResult:
        """
        Solve any ML problem by orchestrating skills.

        Args:
            X: Feature dataframe
            y: Target series (None for clustering)
            problem_type: "classification", "regression", "clustering", or "auto"
            target_metric: Metric to optimize ("auto" to infer)
            time_budget: Max seconds for optimization
            business_context: Optional business context for LLM feature reasoning

        Returns:
            PipelineResult with best model, score, and insights
        """
        await self.init()

        # Initialize progress tracker
        total_stages = len(self.PIPELINE_ORDER) + (1 if self._use_llm_features else 0)
        self._progress = ProgressTracker(total_stages) if self._show_progress else None

        print("\n" + "=" * 60)
        print("🚀 SKILL ORCHESTRATOR - AUTONOMOUS ML PIPELINE")
        print("=" * 60)

        # Step 1: Detect problem type
        if problem_type == "auto":
            prob_type = self._detect_problem_type(y)
        else:
            prob_type = ProblemType(problem_type)

        print(f"📋 Problem: {prob_type.value} | 📊 Metric: ", end="")

        # Step 2: Select target metric
        if target_metric == "auto":
            if prob_type == ProblemType.CLASSIFICATION:
                target_metric = "accuracy"
            elif prob_type == ProblemType.REGRESSION:
                target_metric = "r2"
            else:
                target_metric = "silhouette"

        print(f"{target_metric} | 📦 Features: {X.shape[1]} | 📝 Samples: {len(X)}")
        print("=" * 60 + "\n")

        # Step 3: Execute pipeline
        context = {
            'X': X.copy(),
            'y': y.copy() if y is not None else None,
            'X_original': X.copy(),
            'problem_type': prob_type,
            'target_metric': target_metric,
            'time_budget': time_budget,
            'business_context': business_context,
        }

        skill_results = []

        # Step 3a: LLM Feature Reasoning (if enabled)
        if self._use_llm_features and self._llm_reasoner:
            if self._progress:
                self._progress.start_stage("LLM_FEATURE_REASONING")

            try:
                suggestions = await self._llm_reasoner.reason_features(
                    context['X'], context['y'], prob_type.value, business_context
                )
                if suggestions:
                    context['X'] = self._llm_reasoner.apply_suggestions(context['X'], suggestions)
                    context['llm_features'] = len(suggestions)

                if self._progress:
                    self._progress.complete_stage("LLM_FEATURE_REASONING",
                        {'suggestions': len(suggestions), 'applied': context.get('llm_features', 0)})
            except Exception as e:
                logger.debug(f"LLM reasoning skipped: {e}")
                if self._progress:
                    self._progress.complete_stage("LLM_FEATURE_REASONING", {'skipped': True})

        # Step 4: Execute pipeline stages
        for category in self.PIPELINE_ORDER:
            if self._progress:
                self._progress.start_stage(category.value.upper())
            result = await self._execute_stage(category, context)
            skill_results.append(result)

            if result.success:
                # Update context with result
                if result.data is not None:
                    if category == SkillCategory.FEATURE_ENGINEERING:
                        context['X'] = result.data
                    elif category in [SkillCategory.MODEL_SELECTION,
                                     SkillCategory.HYPERPARAMETER_OPTIMIZATION,
                                     SkillCategory.ENSEMBLE]:
                        context['model'] = result.data
                        context['score'] = result.metrics.get('score', 0)

                if self._progress:
                    self._progress.complete_stage(category.value, result.metrics)
            else:
                if self._progress:
                    self._progress.complete_stage(category.value, {'skipped': result.error or 'failed'})

        # Compile final result
        best_score = context.get('score', 0)
        best_model = context.get('model')

        result = PipelineResult(
            problem_type=prob_type,
            best_score=best_score,
            best_model=best_model,
            feature_count=context['X'].shape[1],
            skill_results=skill_results,
            predictions=context.get('predictions'),
            feature_importance=context.get('feature_importance'),
        )

        # Final progress summary
        if self._progress:
            self._progress.finish(best_score)
        else:
            print(f"\n{'='*60}")
            print(f"🏆 COMPLETE | Score: {best_score:.4f} | Features: {X.shape[1]} → {result.feature_count}")
            print(f"{'='*60}\n")

        return result

    async def _execute_stage(self, category: SkillCategory, context: Dict) -> SkillResult:
        """Execute a pipeline stage using available skills or built-in."""
        # Get skills for this category
        skills = self._get_skills_by_category(category)

        if skills:
            # Try each skill until one succeeds
            for skill in skills:
                result = await skill.execute(context)
                if result.success:
                    return result

        # Fallback to built-in implementation
        return await self._builtin_stage(category, context)

    async def _builtin_stage(self, category: SkillCategory, context: Dict) -> SkillResult:
        """Built-in implementations for each stage."""
        X = context.get('X')
        y = context.get('y')
        problem_type = context.get('problem_type')

        try:
            if category == SkillCategory.DATA_PROFILING:
                return await self._builtin_profiling(X, y)

            elif category == SkillCategory.DATA_CLEANING:
                return await self._builtin_cleaning(X, y)

            elif category == SkillCategory.FEATURE_ENGINEERING:
                return await self._builtin_feature_engineering(X, y, problem_type)

            elif category == SkillCategory.FEATURE_SELECTION:
                return await self._builtin_feature_selection(X, y, problem_type)

            elif category == SkillCategory.MODEL_SELECTION:
                return await self._builtin_model_selection(X, y, problem_type)

            elif category == SkillCategory.HYPERPARAMETER_OPTIMIZATION:
                return await self._builtin_hyperopt(X, y, problem_type, context)

            elif category == SkillCategory.ENSEMBLE:
                return await self._builtin_ensemble(X, y, problem_type, context)

            elif category == SkillCategory.EVALUATION:
                return await self._builtin_evaluation(context)

            elif category == SkillCategory.EXPLANATION:
                return await self._builtin_explanation(context)

            else:
                return SkillResult(
                    skill_name=f"builtin_{category.value}",
                    category=category,
                    success=False,
                    error="Not implemented"
                )

        except Exception as e:
            return SkillResult(
                skill_name=f"builtin_{category.value}",
                category=category,
                success=False,
                error=str(e)
            )

    async def _builtin_profiling(self, X: pd.DataFrame, y: pd.Series) -> SkillResult:
        """Profile the dataset."""
        profile = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'missing_total': int(X.isnull().sum().sum()),
            'missing_pct': float(X.isnull().sum().sum() / X.size * 100),
            'numeric_cols': len(X.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(X.select_dtypes(include=['object']).columns),
        }

        if y is not None:
            profile['target_unique'] = int(y.nunique())
            profile['target_missing'] = int(y.isnull().sum())

        return SkillResult(
            skill_name="builtin_profiler",
            category=SkillCategory.DATA_PROFILING,
            success=True,
            metrics=profile,
            metadata=profile
        )

    async def _builtin_cleaning(self, X: pd.DataFrame, y: pd.Series) -> SkillResult:
        """Clean the dataset."""
        X_clean = X.copy()

        # Handle missing values
        for col in X_clean.columns:
            if X_clean[col].dtype in ['int64', 'float64']:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mode().iloc[0] if len(X_clean[col].mode()) > 0 else 'MISSING')

        return SkillResult(
            skill_name="builtin_cleaner",
            category=SkillCategory.DATA_CLEANING,
            success=True,
            data=X_clean,
            metrics={'cleaned_missing': int(X.isnull().sum().sum())}
        )

    async def _builtin_feature_engineering(self, X: pd.DataFrame, y: pd.Series,
                                           problem_type: ProblemType) -> SkillResult:
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
            if X_eng[col].dtype.name == 'category':
                X_eng[col] = X_eng[col].cat.codes

        # Identify column types
        numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_eng.select_dtypes(include=['object']).columns.tolist()

        # Store original categorical columns before encoding
        cat_cols_original = cat_cols.copy()

        # ================================================================
        # 1. TARGET ENCODING - DISABLED (causes leakage outside CV)
        # ================================================================
        # NOTE: Target encoding must be done INSIDE cross-validation folds
        # to avoid leakage. Doing it here on full data before CV causes
        # the model to see target information from validation rows.
        # TODO: Implement proper target encoding with CV-aware pipeline
        # if y is not None and len(cat_cols) > 0:
        #     for col in cat_cols[:5]:
        #         target_mean = X_eng.groupby(col).apply(...)
        #         X_eng[f'{col}_target_enc'] = ...

        # ================================================================
        # 2. FREQUENCY ENCODING (count-based)
        # ================================================================
        for col in cat_cols[:5]:
            freq = X_eng[col].value_counts(normalize=True)
            X_eng[f'{col}_freq'] = X_eng[col].map(freq).fillna(0)

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
                        grp_mean = X_eng.groupby(grp_col)[agg_col].transform('mean')
                        grp_std = X_eng.groupby(grp_col)[agg_col].transform('std').fillna(0)
                        grp_count = X_eng.groupby(grp_col)[agg_col].transform('count')

                        X_eng[f'{grp_col}_{agg_col}_grp_mean'] = grp_mean
                        X_eng[f'{grp_col}_{agg_col}_grp_std'] = grp_std
                        X_eng[f'{grp_col}_{agg_col}_grp_cnt'] = grp_count

                        # Deviation from group mean
                        X_eng[f'{grp_col}_{agg_col}_dev'] = X_eng[agg_col] - grp_mean
                    except:
                        pass

        # ================================================================
        # 5. BINNING / DISCRETIZATION
        # ================================================================
        for col in numeric_cols[:5]:
            try:
                # Quantile-based binning (5 bins)
                X_eng[f'{col}_qbin'] = pd.qcut(X_eng[col], q=5, labels=False, duplicates='drop')
            except:
                pass

            # Round-based binning
            if X_eng[col].std() > 0:
                X_eng[f'{col}_round1'] = X_eng[col].round(1)
                X_eng[f'{col}_round0'] = X_eng[col].round(0)

        # ================================================================
        # 6. POLYNOMIAL FEATURES (squared, cubed)
        # ================================================================
        for col in numeric_cols[:5]:
            X_eng[f'{col}_sq'] = X_eng[col] ** 2
            X_eng[f'{col}_sqrt'] = np.sqrt(np.abs(X_eng[col]))

        # ================================================================
        # 7. LOG TRANSFORMS (for skewed data)
        # ================================================================
        for col in numeric_cols[:5]:
            if (X_eng[col] > 0).all():
                X_eng[f'{col}_log'] = np.log1p(X_eng[col])
            elif (X_eng[col] >= 0).all():
                X_eng[f'{col}_log'] = np.log1p(X_eng[col])

        # ================================================================
        # 8. NaN PATTERN ENCODING
        # ================================================================
        nan_cols = X.columns[X.isnull().any()].tolist()
        if len(nan_cols) > 0:
            # Binary NaN indicator per column
            for col in nan_cols[:10]:
                X_eng[f'{col}_isna'] = X[col].isnull().astype(int)

            # Combined NaN pattern (binary encoding)
            X_eng['_nan_pattern'] = 0
            for i, col in enumerate(nan_cols[:10]):
                X_eng['_nan_pattern'] += X[col].isnull().astype(int) * (2 ** i)

            # Total NaN count per row
            X_eng['_nan_count'] = X[nan_cols].isnull().sum(axis=1)

        # ================================================================
        # 9. CATEGORICAL COMBINATIONS
        # ================================================================
        encoded_cats = [c for c in cat_cols_original if c in X_eng.columns][:3]
        for i, col1 in enumerate(encoded_cats):
            for col2 in encoded_cats[i+1:]:
                try:
                    max_val = X_eng[col2].max() + 1
                    X_eng[f'{col1}_{col2}_comb'] = (X_eng[col1] + 1) + (X_eng[col2] + 1) / max_val
                except:
                    pass

        # ================================================================
        # 10. INTERACTION FEATURES (multiply & divide)
        # ================================================================
        interact_cols = [c for c in original_cols if c in numeric_cols][:6]

        for i, col1 in enumerate(interact_cols):
            for col2 in interact_cols[i+1:]:
                X_eng[f'{col1}_x_{col2}'] = X_eng[col1] * X_eng[col2]
                denom = X_eng[col2].abs() + 0.001
                X_eng[f'{col1}_div_{col2}'] = X_eng[col1] / denom

        # ================================================================
        # 11. ROW-LEVEL AGGREGATIONS
        # ================================================================
        if len(numeric_cols) >= 3:
            orig_numeric = [c for c in original_cols if c in numeric_cols]
            if len(orig_numeric) >= 3:
                X_eng['_row_sum'] = X_eng[orig_numeric].sum(axis=1)
                X_eng['_row_mean'] = X_eng[orig_numeric].mean(axis=1)
                X_eng['_row_std'] = X_eng[orig_numeric].std(axis=1)
                X_eng['_row_max'] = X_eng[orig_numeric].max(axis=1)
                X_eng['_row_min'] = X_eng[orig_numeric].min(axis=1)
                X_eng['_row_range'] = X_eng['_row_max'] - X_eng['_row_min']
                X_eng['_row_skew'] = X_eng[orig_numeric].skew(axis=1)

        # ================================================================
        # 12. QUANTILE FEATURES
        # ================================================================
        for col in numeric_cols[:3]:
            try:
                q25 = X_eng[col].quantile(0.25)
                q75 = X_eng[col].quantile(0.75)
                X_eng[f'{col}_below_q25'] = (X_eng[col] < q25).astype(int)
                X_eng[f'{col}_above_q75'] = (X_eng[col] > q75).astype(int)
            except:
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
                            train_means = pd.Series(y.iloc[train_idx].values).groupby(
                                col_encoded.iloc[train_idx].values
                            ).mean()

                            # Apply to validation fold
                            target_enc[val_idx] = col_encoded.iloc[val_idx].map(train_means).fillna(y.mean()).values

                        X_eng[f'{col}_target_enc_cv'] = target_enc
                    except Exception as e:
                        logger.debug(f"Target encoding failed for {col}: {e}")

        # ================================================================
        # 14. CV-VALIDATED INTERACTIONS (Only keep if improves score)
        # ================================================================
        if y is not None and len(numeric_cols) >= 2:
            from sklearn.model_selection import cross_val_score
            import lightgbm as lgb

            # Quick baseline score with current features
            try:
                X_temp = X_eng.fillna(0).replace([np.inf, -np.inf], 0)
                baseline_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1) \
                    if problem_type == ProblemType.CLASSIFICATION else \
                    lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)

                baseline_score = cross_val_score(baseline_model, X_temp, y, cv=3, scoring='accuracy' if problem_type == ProblemType.CLASSIFICATION else 'r2').mean()

                # Try top interaction candidates
                orig_numeric = [c for c in original_cols if c in numeric_cols][:4]
                validated_interactions = 0

                for i, col1 in enumerate(orig_numeric):
                    for col2 in orig_numeric[i+1:]:
                        # Create candidate interaction
                        interaction_name = f'{col1}_x_{col2}_validated'
                        X_test = X_temp.copy()
                        X_test[interaction_name] = X_test[col1] * X_test[col2]

                        # Test if it improves score
                        test_score = cross_val_score(baseline_model, X_test, y, cv=3, scoring='accuracy' if problem_type == ProblemType.CLASSIFICATION else 'r2').mean()

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
                top_freq = X_eng[col].value_counts(normalize=True).iloc[0] if len(X_eng[col].value_counts()) > 0 else 0
                if top_freq > 0.99:
                    near_constant.append(col)
            if near_constant:
                X_eng = X_eng.drop(columns=near_constant)

        except Exception as e:
            logger.debug(f"Early pruning failed: {e}")

        # Clean up
        X_eng = X_eng.fillna(0)
        X_eng = X_eng.replace([np.inf, -np.inf], 0)

        # ================================================================
        # FINAL: ENCODE ALL REMAINING CATEGORICAL COLUMNS
        # ================================================================
        # LLM may have created new categorical columns - encode them all
        remaining_cats = X_eng.select_dtypes(include=['object', 'category']).columns.tolist()
        if remaining_cats:
            from sklearn.preprocessing import LabelEncoder
            for col in remaining_cats:
                try:
                    le = LabelEncoder()
                    X_eng[col] = le.fit_transform(X_eng[col].astype(str))
                except:
                    # If encoding fails, drop the column
                    X_eng = X_eng.drop(columns=[col])

        return SkillResult(
            skill_name="builtin_kaggle_fe",
            category=SkillCategory.FEATURE_ENGINEERING,
            success=True,
            data=X_eng,
            metrics={
                'original_features': n_original,
                'engineered_features': len(X_eng.columns),
                'new_features': len(X_eng.columns) - n_original,
                'techniques_used': [
                    'frequency_encoding', 'groupby_aggs', 'binning', 'polynomial',
                    'log_transform', 'nan_patterns', 'cat_combinations', 'interactions',
                    'row_stats', 'quantiles', 'target_encoding_cv', 'validated_interactions',
                    'early_pruning'
                ]
            }
        )

    async def _builtin_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                         problem_type: ProblemType) -> SkillResult:
        """
        World-class feature selection using multiple advanced techniques.

        Techniques used:
        1. CORRELATION FILTER - Remove highly correlated features (redundancy)
        2. NULL IMPORTANCE - Compare real vs shuffled target importance
        3. PERMUTATION IMPORTANCE - Measure actual impact on CV score
        4. MULTI-MODEL VOTING - Ensemble importance from LightGBM, XGBoost, RF
        5. STABILITY SELECTION - Features consistently important across seeds
        6. BORUTA-LIKE - Compare against shadow (shuffled) features
        """
        import lightgbm as lgb
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.inspection import permutation_importance
        from collections import defaultdict

        if y is None:
            return SkillResult(
                skill_name="builtin_feature_selector",
                category=SkillCategory.FEATURE_SELECTION,
                success=True,
                data=X,
                metrics={'selected': X.shape[1]}
            )

        n_features = X.shape[1]
        feature_scores = defaultdict(float)  # Accumulate scores across methods
        method_results = {}

        # ================================================================
        # 1. CORRELATION FILTER - Remove redundant features
        # ================================================================
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = []
        corr_threshold = 0.98  # Higher threshold = keep more features

        to_drop_corr = set()
        for col in upper_tri.columns:
            correlated = upper_tri.index[upper_tri[col] > corr_threshold].tolist()
            if correlated:
                # Keep the one with higher variance
                for corr_col in correlated:
                    if X[col].var() >= X[corr_col].var():
                        to_drop_corr.add(corr_col)
                    else:
                        to_drop_corr.add(col)
                    high_corr_pairs.append((col, corr_col))

        X_filtered = X.drop(columns=list(to_drop_corr), errors='ignore')
        method_results['correlation_removed'] = len(to_drop_corr)

        # ================================================================
        # 2. MULTI-MODEL IMPORTANCE VOTING
        # ================================================================
        if problem_type == ProblemType.CLASSIFICATION:
            models = {
                'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
                'xgb': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
                'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            }
        else:
            models = {
                'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            }

        model_importances = {}
        for name, model in models.items():
            try:
                model.fit(X_filtered, y)
                imp = pd.Series(model.feature_importances_, index=X_filtered.columns)
                imp_normalized = imp / (imp.sum() + 1e-10)  # Normalize to sum=1
                model_importances[name] = imp_normalized

                # Add to feature scores (weight by model type)
                for feat, score in imp_normalized.items():
                    feature_scores[feat] += score
            except Exception as e:
                logger.debug(f"Model {name} importance failed: {e}")

        method_results['multi_model'] = list(model_importances.keys())

        # ================================================================
        # 3. NULL IMPORTANCE TEST - Identify truly predictive features
        # ================================================================
        null_importance_scores = defaultdict(list)
        n_null_runs = 5

        lgb_model = models['lgb']
        for i in range(n_null_runs):
            # Shuffle target
            y_shuffled = y.sample(frac=1, random_state=42 + i).reset_index(drop=True)
            try:
                lgb_model.fit(X_filtered, y_shuffled)
                null_imp = pd.Series(lgb_model.feature_importances_, index=X_filtered.columns)
                for feat in X_filtered.columns:
                    null_importance_scores[feat].append(null_imp[feat])
            except:
                pass

        # Features where real importance > 95th percentile of null importance
        null_passed = set()
        if model_importances.get('lgb') is not None:
            real_imp = model_importances['lgb']
            for feat in X_filtered.columns:
                null_vals = null_importance_scores.get(feat, [0])
                null_95 = np.percentile(null_vals, 95) if null_vals else 0
                real_val = real_imp.get(feat, 0)
                if real_val > null_95 * 1.1:  # 10% margin
                    null_passed.add(feat)
                    feature_scores[feat] += 0.5  # Bonus for passing null test

        method_results['null_test_passed'] = len(null_passed)

        # ================================================================
        # 4. STABILITY SELECTION - Consistent importance across seeds
        # ================================================================
        stability_counts = defaultdict(int)
        n_stability_runs = 5
        top_pct = 0.3  # Top 30% features

        for seed in range(n_stability_runs):
            try:
                lgb_temp = lgb.LGBMClassifier(n_estimators=50, random_state=seed, verbose=-1) \
                    if problem_type == ProblemType.CLASSIFICATION else \
                    lgb.LGBMRegressor(n_estimators=50, random_state=seed, verbose=-1)

                # Bootstrap sample
                n_samples = len(X_filtered)
                idx = np.random.RandomState(seed).choice(n_samples, size=n_samples, replace=True)
                lgb_temp.fit(X_filtered.iloc[idx], y.iloc[idx])

                imp = pd.Series(lgb_temp.feature_importances_, index=X_filtered.columns)
                top_k = int(len(imp) * top_pct)
                top_features = imp.nlargest(top_k).index.tolist()

                for feat in top_features:
                    stability_counts[feat] += 1
            except:
                pass

        # Features selected in majority of runs
        stable_features = {f for f, c in stability_counts.items() if c >= n_stability_runs * 0.6}
        for feat in stable_features:
            feature_scores[feat] += 0.3  # Bonus for stability

        method_results['stable_features'] = len(stable_features)

        # ================================================================
        # 5. BORUTA-LIKE SHADOW FEATURES TEST
        # ================================================================
        # Create shadow features (shuffled copies)
        X_shadow = X_filtered.copy()
        for col in X_shadow.columns:
            X_shadow[col] = np.random.permutation(X_shadow[col].values)
        X_shadow.columns = [f'shadow_{c}' for c in X_shadow.columns]

        X_with_shadow = pd.concat([X_filtered, X_shadow], axis=1)

        try:
            lgb_shadow = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) \
                if problem_type == ProblemType.CLASSIFICATION else \
                lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_shadow.fit(X_with_shadow, y)

            imp_all = pd.Series(lgb_shadow.feature_importances_, index=X_with_shadow.columns)
            shadow_max = imp_all[[c for c in imp_all.index if c.startswith('shadow_')]].max()

            # Features beating the best shadow feature
            boruta_passed = set()
            for feat in X_filtered.columns:
                if imp_all.get(feat, 0) > shadow_max:
                    boruta_passed.add(feat)
                    feature_scores[feat] += 0.4  # Bonus for beating shadows

            method_results['boruta_passed'] = len(boruta_passed)
        except Exception as e:
            logger.debug(f"Boruta test failed: {e}")
            method_results['boruta_passed'] = 0

        # ================================================================
        # 6. PERMUTATION IMPORTANCE (on a quick model)
        # ================================================================
        try:
            if problem_type == ProblemType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = 'r2'

            lgb_perm = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1) \
                if problem_type == ProblemType.CLASSIFICATION else \
                lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            lgb_perm.fit(X_filtered, y)

            perm_imp = permutation_importance(lgb_perm, X_filtered, y,
                                              n_repeats=5, random_state=42, n_jobs=-1)
            perm_scores = pd.Series(perm_imp.importances_mean, index=X_filtered.columns)
            perm_scores_normalized = perm_scores / (perm_scores.sum() + 1e-10)

            for feat, score in perm_scores_normalized.items():
                if score > 0:  # Only positive permutation importance
                    feature_scores[feat] += score * 0.5

            method_results['permutation_done'] = True
        except Exception as e:
            logger.debug(f"Permutation importance failed: {e}")
            method_results['permutation_done'] = False

        # ================================================================
        # FINAL SELECTION - Quantile/Decile-based selection
        # ================================================================
        final_scores = pd.Series(feature_scores).sort_values(ascending=False)

        # Calculate deciles for adaptive selection
        if len(final_scores) > 10:
            # Assign decile ranks (1=top 10%, 10=bottom 10%)
            final_scores_df = pd.DataFrame({
                'feature': final_scores.index,
                'score': final_scores.values
            })
            final_scores_df['decile'] = pd.qcut(
                final_scores_df['score'].rank(method='first'),
                q=10, labels=range(10, 0, -1)  # 10=top, 1=bottom
            ).astype(int)

            # SELECTION STRATEGY (World-class - keep quality features):
            # - Decile 10-8 (top 30%): Always keep - best features
            # - Decile 7 (top 40%): Keep if score > median of decile
            # - Decile 6 (top 50%): Keep if passed multiple tests (score > 1.0)
            # - Decile 5 (top 60%): Keep if strong multi-test (score > 1.5)
            # - Below: Only keep if exceptional (score > 2.0)

            selected = []

            # Top 3 deciles (top 30%) - always keep
            top_deciles = final_scores_df[final_scores_df['decile'] >= 8]['feature'].tolist()
            selected.extend(top_deciles)

            # Decile 7 (top 40%) - keep above median
            decile_7 = final_scores_df[final_scores_df['decile'] == 7]
            if len(decile_7) > 0:
                d7_median = decile_7['score'].median()
                d7_selected = decile_7[decile_7['score'] >= d7_median]['feature'].tolist()
                selected.extend(d7_selected)

            # Decile 6 (top 50%) - keep if multi-test passed (score > 1.0)
            decile_6 = final_scores_df[final_scores_df['decile'] == 6]
            d6_selected = decile_6[decile_6['score'] >= 1.0]['feature'].tolist()
            selected.extend(d6_selected)

            # Decile 5 (top 60%) - keep if strong score (score > 1.5)
            decile_5 = final_scores_df[final_scores_df['decile'] == 5]
            d5_selected = decile_5[decile_5['score'] >= 1.5]['feature'].tolist()
            selected.extend(d5_selected)

            # Deciles 4 and below - only exceptional features (score > 2.0)
            lower_deciles = final_scores_df[final_scores_df['decile'] <= 4]
            exceptional = lower_deciles[lower_deciles['score'] >= 2.0]['feature'].tolist()
            selected.extend(exceptional)

            # Remove duplicates while preserving order
            selected = list(dict.fromkeys(selected))

            method_results['decile_distribution'] = final_scores_df.groupby('decile').size().to_dict()
        else:
            selected = final_scores.index.tolist()

        # Ensure minimum features (at least 15 or 15% of filtered)
        min_features = max(15, int(len(X_filtered.columns) * 0.15))
        if len(selected) < min_features:
            # Add more from top scores
            for feat in final_scores.index:
                if feat not in selected:
                    selected.append(feat)
                if len(selected) >= min_features:
                    break

        # ================================================================
        # CV VALIDATION - Ensure selection doesn't hurt performance
        # ================================================================
        try:
            if problem_type == ProblemType.CLASSIFICATION:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                val_model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'accuracy'
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                val_model = lgb.LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
                scoring = 'r2'

            # Score with selected features
            score_selected = cross_val_score(val_model, X_filtered[selected], y, cv=cv, scoring=scoring).mean()

            # Score with all features (after correlation filter)
            score_all = cross_val_score(val_model, X_filtered, y, cv=cv, scoring=scoring).mean()

            # If selected is significantly worse (>1%), add more features
            if score_selected < score_all - 0.01:
                # Add features until performance matches
                remaining = [f for f in final_scores.index if f not in selected]
                for feat in remaining:
                    selected.append(feat)
                    new_score = cross_val_score(val_model, X_filtered[selected], y, cv=cv, scoring=scoring).mean()
                    if new_score >= score_all - 0.005:  # Within 0.5% of full set
                        break
                    if len(selected) >= len(X_filtered.columns) * 0.6:  # Cap at 60%
                        break

            method_results['cv_selected_score'] = score_selected
            method_results['cv_all_score'] = score_all
        except Exception as e:
            logger.debug(f"CV validation failed: {e}")

        X_selected = X_filtered[selected]

        return SkillResult(
            skill_name="builtin_feature_selector",
            category=SkillCategory.FEATURE_SELECTION,
            success=True,
            data=X_selected,
            metrics={
                'original': n_features,
                'after_corr_filter': len(X_filtered.columns),
                'selected': len(selected),
                'removed': n_features - len(selected),
                'null_passed': method_results.get('null_test_passed', 0),
                'boruta_passed': method_results.get('boruta_passed', 0),
                'stable': method_results.get('stable_features', 0),
            },
            metadata={
                'selected_features': selected,
                'feature_scores': final_scores.head(20).to_dict(),
                'decile_dist': method_results.get('decile_distribution', {}),
                'methods_used': ['correlation', 'multi_model', 'null_importance',
                                'stability', 'boruta', 'permutation']
            }
        )

    async def _builtin_model_selection(self, X: pd.DataFrame, y: pd.Series,
                                       problem_type: ProblemType) -> SkillResult:
        """
        World-class model selection with:
        1. Extended model zoo (7+ algorithms)
        2. Data-adaptive configurations
        3. Cross-validation with OOF predictions for stacking
        4. Model diversity tracking for ensemble
        """
        from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.svm import SVC, SVR
        import lightgbm as lgb
        import xgboost as xgb

        n_samples, n_features = X.shape
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Adaptive configuration based on data size
        n_estimators = 100 if n_samples < 5000 else 200
        max_depth_tree = None if n_samples > 1000 else 10

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'

            models = {
                # Gradient Boosting Family (usually best)
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=n_estimators, random_state=42, verbose=-1,
                    learning_rate=0.1, num_leaves=31
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=n_estimators, random_state=42,
                    eval_metric='logloss', verbosity=0, learning_rate=0.1
                ),
                'histgb': HistGradientBoostingClassifier(
                    max_iter=n_estimators, random_state=42, learning_rate=0.1
                ),

                # Tree Ensembles (good diversity)
                'random_forest': RandomForestClassifier(
                    n_estimators=n_estimators, random_state=42,
                    max_depth=max_depth_tree, n_jobs=-1
                ),
                'extra_trees': ExtraTreesClassifier(
                    n_estimators=n_estimators, random_state=42,
                    max_depth=max_depth_tree, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=min(n_estimators, 100), random_state=42,
                    learning_rate=0.1
                ),

                # Linear Model (different hypothesis space)
                'logistic': LogisticRegression(
                    max_iter=1000, random_state=42, C=1.0
                ),
            }

            # Add SVM for small datasets (slow on large)
            if n_samples < 5000:
                models['svm'] = SVC(probability=True, random_state=42, C=1.0)

        else:  # Regression
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'

            models = {
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=n_estimators, random_state=42, verbose=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=n_estimators, random_state=42, verbosity=0
                ),
                'histgb': HistGradientBoostingRegressor(
                    max_iter=n_estimators, random_state=42
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
                'extra_trees': ExtraTreesRegressor(
                    n_estimators=n_estimators, random_state=42, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=min(n_estimators, 100), random_state=42
                ),
                'ridge': Ridge(alpha=1.0),
            }

            if n_samples < 5000:
                models['svr'] = SVR(C=1.0)

        # Evaluate all models and collect OOF predictions
        best_model = None
        best_name = None
        best_score = -np.inf
        all_scores = {}
        all_std = {}
        oof_predictions = {}  # For stacking

        for name, model in models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
                mean_score = scores.mean()
                std_score = scores.std()
                all_scores[name] = mean_score
                all_std[name] = std_score

                # Collect OOF predictions for top models (for stacking later)
                if mean_score > best_score * 0.95:  # Within 5% of best
                    try:
                        if problem_type == ProblemType.CLASSIFICATION:
                            oof = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')
                            oof_predictions[name] = oof[:, 1] if oof.ndim > 1 else oof
                        else:
                            oof_predictions[name] = cross_val_predict(model, X_scaled, y, cv=cv)
                    except:
                        pass

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name

            except Exception as e:
                logger.debug(f"Model {name} failed: {e}")

        # Rank models for reporting
        sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        return SkillResult(
            skill_name="builtin_model_selector",
            category=SkillCategory.MODEL_SELECTION,
            success=True,
            data=best_model,
            metrics={'score': best_score, 'model': best_name, **{k: all_scores[k] for k in list(all_scores)[:4]}},
            metadata={
                'best_model': best_name,
                'all_scores': all_scores,
                'all_std': all_std,
                'model_ranking': [m[0] for m in sorted_models],
                'oof_predictions': oof_predictions,
                'scaler': scaler,
                'X_scaled': X_scaled,
            }
        )

    async def _builtin_hyperopt(self, X: pd.DataFrame, y: pd.Series,
                                problem_type: ProblemType, context: Dict) -> SkillResult:
        """
        World-class hyperparameter optimization:
        1. Multi-model tuning (LightGBM, XGBoost, RandomForest)
        2. Optuna pruning (MedianPruner - stop bad trials early)
        3. Expanded parameter space with regularization
        4. Efficient TPE sampler with warm-starting
        """
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import lightgbm as lgb
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get top models from model selection stage
        model_metadata = {}
        for result in context.get('skill_results', []):
            if hasattr(result, 'metadata') and result.metadata:
                model_metadata.update(result.metadata)

        model_ranking = model_metadata.get('model_ranking', ['lightgbm', 'xgboost', 'random_forest'])
        top_models = model_ranking[:3]  # Tune top 3 models

        # Progress tracking
        n_trials_per_model = min(20, context.get('time_budget', 60) // 4)
        total_trials = n_trials_per_model * len(top_models)
        best_score_so_far = [0.0]
        trial_count = [0]
        current_model = ['']

        def progress_callback(study, trial):
            trial_count[0] += 1
            if trial.value and trial.value > best_score_so_far[0]:
                best_score_so_far[0] = trial.value
            pct = (trial_count[0] / total_trials) * 100
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = '▓' * filled + '░' * (bar_len - filled)
            model_short = current_model[0][:8]
            sys.stdout.write(f'\r      Hyperopt [{bar}] {trial_count[0]:2d}/{total_trials} | {model_short} | best={best_score_so_far[0]:.4f}')
            sys.stdout.flush()

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'

        # Store best results per model
        model_results = {}

        # ================================================================
        # TUNE EACH TOP MODEL
        # ================================================================
        for model_name in top_models:
            current_model[0] = model_name

            # Create study with pruner for efficiency
            sampler = TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

            if model_name in ['lightgbm', 'lgb']:
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                        'random_state': 42,
                        'verbose': -1
                    }
                    model = lgb.LGBMClassifier(**params) if problem_type == ProblemType.CLASSIFICATION else lgb.LGBMRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            elif model_name in ['xgboost', 'xgb']:
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'random_state': 42,
                        'verbosity': 0,
                        'eval_metric': 'logloss' if problem_type == ProblemType.CLASSIFICATION else 'rmse'
                    }
                    model = xgb.XGBClassifier(**params) if problem_type == ProblemType.CLASSIFICATION else xgb.XGBRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            elif model_name in ['random_forest', 'rf']:
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    model = RandomForestClassifier(**params) if problem_type == ProblemType.CLASSIFICATION else RandomForestRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            else:
                # Default: LightGBM
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'random_state': 42,
                        'verbose': -1
                    }
                    model = lgb.LGBMClassifier(**params) if problem_type == ProblemType.CLASSIFICATION else lgb.LGBMRegressor(**params)
                    return cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring).mean()

            try:
                study.optimize(objective, n_trials=n_trials_per_model, callbacks=[progress_callback], show_progress_bar=False)
                model_results[model_name] = {
                    'score': study.best_value,
                    'params': study.best_params,
                    'study': study
                }
            except Exception as e:
                logger.debug(f"Hyperopt failed for {model_name}: {e}")

        print()  # New line after progress

        # ================================================================
        # SELECT BEST MODEL AND BUILD
        # ================================================================
        best_model_name = None
        best_score = 0
        best_params = {}

        for model_name, result in model_results.items():
            if result['score'] > best_score:
                best_score = result['score']
                best_model_name = model_name
                best_params = result['params']

        # Build optimized model
        if best_model_name in ['lightgbm', 'lgb']:
            final_params = {**best_params, 'random_state': 42, 'verbose': -1}
            optimized_model = lgb.LGBMClassifier(**final_params) if problem_type == ProblemType.CLASSIFICATION else lgb.LGBMRegressor(**final_params)
        elif best_model_name in ['xgboost', 'xgb']:
            final_params = {**best_params, 'random_state': 42, 'verbosity': 0}
            optimized_model = xgb.XGBClassifier(**final_params) if problem_type == ProblemType.CLASSIFICATION else xgb.XGBRegressor(**final_params)
        elif best_model_name in ['random_forest', 'rf']:
            final_params = {**best_params, 'random_state': 42, 'n_jobs': -1}
            optimized_model = RandomForestClassifier(**final_params) if problem_type == ProblemType.CLASSIFICATION else RandomForestRegressor(**final_params)
        else:
            # Fallback
            optimized_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) if problem_type == ProblemType.CLASSIFICATION else lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

        return SkillResult(
            skill_name="builtin_hyperopt",
            category=SkillCategory.HYPERPARAMETER_OPTIMIZATION,
            success=True,
            data=optimized_model,
            metrics={
                'score': best_score,
                'n_trials': total_trials,
                'best_model': best_model_name,
                'models_tuned': len(model_results)
            },
            metadata={
                'best_params': best_params,
                'all_model_scores': {k: v['score'] for k, v in model_results.items()},
                'best_model_name': best_model_name
            }
        )

    async def _builtin_ensemble(self, X: pd.DataFrame, y: pd.Series,
                                problem_type: ProblemType, context: Dict) -> SkillResult:
        """
        World-class ensemble with multiple strategies:
        1. Weighted Voting - weight by CV score
        2. Stacking - meta-learner on OOF predictions
        3. Greedy Selection - iteratively add models that improve score
        4. Blending - average top diverse models
        """
        from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        import lightgbm as lgb
        import xgboost as xgb

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get optimized model and scores from previous stages
        optimized = context.get('model')
        best_single_score = context.get('score', 0)

        # Get model scores from model selection stage
        model_metadata = {}
        for result in context.get('skill_results', []):
            if hasattr(result, 'metadata') and result.metadata:
                model_metadata.update(result.metadata)

        all_scores = model_metadata.get('all_scores', {})
        oof_predictions = model_metadata.get('oof_predictions', {})

        if problem_type == ProblemType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'

            # Base models with different characteristics for diversity
            base_models = {
                'lgb': lgb.LGBMClassifier(n_estimators=150, random_state=42, verbose=-1),
                'xgb': xgb.XGBClassifier(n_estimators=150, random_state=42, eval_metric='logloss', verbosity=0),
                'rf': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
            meta_learner = LogisticRegression(max_iter=1000, random_state=42)

        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'

            base_models = {
                'lgb': lgb.LGBMRegressor(n_estimators=150, random_state=42, verbose=-1),
                'xgb': xgb.XGBRegressor(n_estimators=150, random_state=42, verbosity=0),
                'rf': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            }
            meta_learner = Ridge(alpha=1.0)

        ensemble_results = {}

        # ================================================================
        # STRATEGY 1: Weighted Voting (weight by CV score)
        # ================================================================
        try:
            # Calculate weights from scores
            if all_scores:
                weights = []
                estimators = []
                for name, model in base_models.items():
                    score = all_scores.get(name, 0.5)
                    weights.append(max(score, 0.1))  # Min weight 0.1
                    estimators.append((name, model))

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                if problem_type == ProblemType.CLASSIFICATION:
                    weighted_ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
                else:
                    weighted_ensemble = VotingRegressor(estimators=estimators, weights=weights)

                weighted_scores = cross_val_score(weighted_ensemble, X_scaled, y, cv=cv, scoring=scoring)
                ensemble_results['weighted_voting'] = weighted_scores.mean()
        except Exception as e:
            logger.debug(f"Weighted voting failed: {e}")

        # ================================================================
        # STRATEGY 2: Stacking with Meta-Learner
        # ================================================================
        try:
            estimators_list = [(name, model) for name, model in base_models.items()]

            if problem_type == ProblemType.CLASSIFICATION:
                stacking = StackingClassifier(
                    estimators=estimators_list,
                    final_estimator=meta_learner,
                    cv=3,  # Faster inner CV
                    passthrough=False,
                    n_jobs=-1
                )
            else:
                stacking = StackingRegressor(
                    estimators=estimators_list,
                    final_estimator=meta_learner,
                    cv=3,
                    passthrough=False,
                    n_jobs=-1
                )

            stacking_scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring=scoring)
            ensemble_results['stacking'] = stacking_scores.mean()
        except Exception as e:
            logger.debug(f"Stacking failed: {e}")

        # ================================================================
        # STRATEGY 3: Greedy Ensemble Selection
        # ================================================================
        try:
            # Start with best single model, greedily add models that improve
            sorted_models = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            greedy_estimators = []

            if sorted_models:
                # Start with best model
                best_name = sorted_models[0][0]
                if best_name in base_models:
                    greedy_estimators.append((best_name, base_models[best_name]))

                current_best = all_scores.get(best_name, 0)

                # Try adding each remaining model
                for name, score in sorted_models[1:4]:  # Top 4 models
                    if name in base_models:
                        test_estimators = greedy_estimators + [(name, base_models[name])]

                        if problem_type == ProblemType.CLASSIFICATION:
                            test_ensemble = VotingClassifier(estimators=test_estimators, voting='soft')
                        else:
                            test_ensemble = VotingRegressor(estimators=test_estimators)

                        test_scores = cross_val_score(test_ensemble, X_scaled, y, cv=cv, scoring=scoring)
                        test_score = test_scores.mean()

                        # Add if improves
                        if test_score > current_best:
                            greedy_estimators.append((name, base_models[name]))
                            current_best = test_score

                ensemble_results['greedy'] = current_best
                ensemble_results['greedy_models'] = [e[0] for e in greedy_estimators]
        except Exception as e:
            logger.debug(f"Greedy selection failed: {e}")

        # ================================================================
        # STRATEGY 4: Simple Average (Baseline)
        # ================================================================
        try:
            simple_estimators = [(name, model) for name, model in list(base_models.items())[:3]]

            if problem_type == ProblemType.CLASSIFICATION:
                simple_ensemble = VotingClassifier(estimators=simple_estimators, voting='soft')
            else:
                simple_ensemble = VotingRegressor(estimators=simple_estimators)

            simple_scores = cross_val_score(simple_ensemble, X_scaled, y, cv=cv, scoring=scoring)
            ensemble_results['simple_avg'] = simple_scores.mean()
        except Exception as e:
            logger.debug(f"Simple average failed: {e}")

        # ================================================================
        # SELECT BEST STRATEGY
        # ================================================================
        best_ensemble_strategy = None
        best_ensemble_score = 0

        for strategy, score in ensemble_results.items():
            if isinstance(score, (int, float)) and score > best_ensemble_score:
                best_ensemble_score = score
                best_ensemble_strategy = strategy

        # Compare ensemble vs single model
        if best_ensemble_score > best_single_score:
            # Build the winning ensemble
            if best_ensemble_strategy == 'stacking':
                final_model = stacking
            elif best_ensemble_strategy == 'weighted_voting':
                final_model = weighted_ensemble
            elif best_ensemble_strategy == 'greedy' and greedy_estimators:
                if problem_type == ProblemType.CLASSIFICATION:
                    final_model = VotingClassifier(estimators=greedy_estimators, voting='soft')
                else:
                    final_model = VotingRegressor(estimators=greedy_estimators)
            else:
                final_model = simple_ensemble

            final_model.fit(X_scaled, y)
            final_score = best_ensemble_score
            used_ensemble = True
            n_estimators = len(greedy_estimators) if best_ensemble_strategy == 'greedy' else len(base_models)
        else:
            # Single model is better
            final_model = optimized
            final_score = best_single_score
            used_ensemble = False
            best_ensemble_strategy = 'single_model'
            n_estimators = 1

        return SkillResult(
            skill_name="builtin_ensemble",
            category=SkillCategory.ENSEMBLE,
            success=True,
            data=final_model,
            metrics={
                'score': final_score,
                'n_estimators': n_estimators,
                'ensemble_score': best_ensemble_score,
                'single_score': best_single_score,
                'strategy': best_ensemble_strategy,
            },
            metadata={
                'all_ensemble_scores': ensemble_results,
                'decision': best_ensemble_strategy,
                'used_ensemble': used_ensemble,
            }
        )

    async def _builtin_evaluation(self, context: Dict) -> SkillResult:
        """Evaluate the model."""
        score = context.get('score', 0)
        model = context.get('model')

        return SkillResult(
            skill_name="builtin_evaluator",
            category=SkillCategory.EVALUATION,
            success=True,
            metrics={'final_score': score},
            metadata={'model_type': type(model).__name__ if model else 'None'}
        )

    async def _builtin_explanation(self, context: Dict) -> SkillResult:
        """Generate model explanations."""
        model = context.get('model')
        X = context.get('X')

        feature_importance = {}

        if model is not None and hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(X.columns, importance))
        elif model is not None and hasattr(model, 'estimators_'):
            # Voting ensemble - aggregate from sub-models
            importances = []
            for name, est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
            if importances:
                avg_importance = np.mean(importances, axis=0)
                feature_importance = dict(zip(X.columns, avg_importance))

        context['feature_importance'] = feature_importance

        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])

        return SkillResult(
            skill_name="builtin_explainer",
            category=SkillCategory.EXPLANATION,
            success=True,
            metrics={'top_features': len(sorted_importance)},
            metadata={'feature_importance': sorted_importance}
        )


# Singleton accessor
_orchestrator_instance = None


def get_skill_orchestrator() -> SkillOrchestrator:
    """Get or create the skill orchestrator singleton."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SkillOrchestrator()
    return _orchestrator_instance
