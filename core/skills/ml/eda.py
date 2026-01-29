"""
EDA Skill - Exploratory Data Analysis
=====================================

Automated EDA that generates actionable insights for feature engineering.

Analyzes:
1. Distributions (skewness, modality, outliers)
2. Correlations (with target and between features)
3. Missing patterns (randomness, correlation with target)
4. Categorical patterns (cardinality, target rates)
5. Interactions (non-linear relationships)

The insights are formatted for both human consumption and LLM prompts.
"""

import time
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .base import MLSkill, SkillResult, SkillCategory


class EDASkill(MLSkill):
    """
    Skill for exploratory data analysis.

    Produces actionable insights that inform feature engineering decisions.
    """

    name = "eda_analysis"
    version = "1.0.0"
    description = "Automated EDA with actionable insights"
    category = SkillCategory.DATA_PROFILING

    required_inputs = ["X"]
    optional_inputs = ["y", "problem_type"]
    outputs = ["insights", "recommendations"]

    async def execute(self,
                      X: pd.DataFrame,
                      y: Optional[pd.Series] = None,
                      **context) -> SkillResult:
        """
        Execute EDA analysis.

        Args:
            X: Input features
            y: Target variable (optional but recommended)
            **context: problem_type, etc.

        Returns:
            SkillResult with insights and recommendations
        """
        start_time = time.time()

        if not self.validate_inputs(X, y):
            return self._create_error_result("Invalid inputs")

        problem_type = context.get('problem_type', 'classification')

        try:
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

            # Format for LLM
            llm_text = self._format_for_llm(insights)

            execution_time = time.time() - start_time

            return self._create_result(
                success=True,
                data=insights,
                metrics={
                    'n_features': X.shape[1],
                    'n_samples': len(X),
                    'n_recommendations': len(insights['recommendations']),
                    'n_skewed': len([c for c, d in insights['distributions'].items() if d.get('is_skewed')]),
                    'n_high_corr_pairs': len(insights['correlations'].get('high_correlation_pairs', [])),
                },
                metadata={
                    'llm_formatted': llm_text,
                    'problem_type': problem_type,
                },
                execution_time=execution_time,
            )

        except Exception as e:
            return self._create_error_result(str(e))

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
                'cv': float(data.std() / (data.mean() + 0.001)),
                'n_unique': int(data.nunique()),
                'pct_unique': float(data.nunique() / len(data)),
            }

            # Detect potential bimodality
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

        # Feature-feature correlations
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

            # Target rate by category
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

            if outlier_pct > 1:
                outlier_info[col] = {
                    'n_outliers': int(n_outliers),
                    'outlier_pct': float(outlier_pct),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                }

        return outlier_info

    def _analyze_interactions(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze potential feature interactions."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()[:10]

        interactions = []

        if y is None or len(numeric_cols) < 2:
            return {'promising_interactions': interactions}

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

        interactions.sort(key=lambda x: x['improvement'], reverse=True)

        return {'promising_interactions': interactions[:10]}

    def _generate_recommendations(self, insights: Dict, X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """Generate actionable feature engineering recommendations."""
        recommendations = []

        corr_with_target = insights['correlations'].get('with_target', {})

        # Distribution-based recommendations
        for col, dist in insights['distributions'].items():
            col_corr = corr_with_target.get(col, {}).get('abs_correlation', 0)
            is_predictive = col_corr > 0.1

            if dist['is_skewed'] and dist['skewness'] > 0 and is_predictive:
                recommendations.append({
                    'type': 'transform',
                    'feature': col,
                    'action': 'log_transform',
                    'reason': f"Skewed (skew={dist['skewness']:.2f}) AND predictive (corr={col_corr:.2f})",
                    'code': f"X['{col}_log'] = np.log1p(X['{col}'].clip(lower=0))"
                })

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
                'reason': f"High correlation ({pair['correlation']:.2f})",
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

    def _format_for_llm(self, insights: Dict) -> str:
        """Format EDA insights as text for LLM consumption."""
        lines = ["## EDA INSIGHTS FOR FEATURE ENGINEERING\n"]

        # Distribution insights
        skewed = [col for col, d in insights['distributions'].items() if d.get('is_skewed')]
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
