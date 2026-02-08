"""
Intelligent Visualization Orchestrator - The Best of AI-Powered Financial Visualization

This module provides truly intelligent, adaptive visualization that:
1. Automatically selects optimal chart types based on data availability
2. Generates contextual narratives for each visualization
3. Detects anomalies and highlights key insights
4. Creates comparative analysis when peer data available
5. Forecasts trends with confidence intervals
6. Intelligently embeds charts in relevant report sections
7. Generates both static (PDF) and interactive (HTML) versions
"""
import asyncio
import logging
import json
import re
import inspect
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)


def safe_num(value, default: float = 0) -> float:
    """Safely convert value to number, handling None and invalid types."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class IntelligentVisualizationOrchestrator:
    """
    Orchestrates intelligent, adaptive financial visualization.
    
    This is the "best of AI" - it doesn't just generate charts,
    it understands the data, selects optimal visualizations,
    generates insights, detects patterns, and creates narratives.
    """
    
    def __init__(self):
        self.chart_priority_map = {
            'revenue_growth': {'priority': 10, 'requires': ['financial_statements.revenue']},
            'profitability': {'priority': 9, 'requires': ['key_ratios.roe', 'key_ratios.roa']},
            'valuation_metrics': {'priority': 8, 'requires': ['valuation_metrics.pe_ratio']},
            'financial_health': {'priority': 7, 'requires': ['key_ratios']},
            'radar_comparison': {'priority': 6, 'requires': ['key_ratios', 'valuation_metrics']},
            'trend_forecast': {'priority': 5, 'requires': ['financial_statements.revenue']},
            'anomaly_detection': {'priority': 4, 'requires': ['financial_statements']},
            'peer_comparison': {'priority': 3, 'requires': ['valuation_metrics', 'peer_data']},
        }
    
    async def analyze_data_completeness(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently analyze what data is available and score completeness.
        
        Returns:
            Dictionary with completeness scores, available metrics, missing data,
            and recommended chart types.
        """
        scores = {
            'financial_statements': 0,
            'valuation_metrics': 0,
            'key_ratios': 0,
            'growth_metrics': 0,
            'price_data': 0,
        }
        
        available_metrics = []
        missing_metrics = []
        
        # Score financial statements
        fs = extracted_data.get('financial_statements', {})
        if fs.get('revenue'):
            scores['financial_statements'] += 40
            available_metrics.append('revenue')
        else:
            missing_metrics.append('revenue')
        if fs.get('net_profit'):
            scores['financial_statements'] += 30
            available_metrics.append('net_profit')
        else:
            missing_metrics.append('net_profit')
        if fs.get('total_assets'):
            scores['financial_statements'] += 15
            available_metrics.append('total_assets')
        if fs.get('total_equity'):
            scores['financial_statements'] += 15
            available_metrics.append('total_equity')
        
        # Score valuation metrics
        vm = extracted_data.get('valuation_metrics', {})
        if vm.get('pe_ratio'):
            scores['valuation_metrics'] += 40
            available_metrics.append('pe_ratio')
        else:
            missing_metrics.append('pe_ratio')
        if vm.get('pb_ratio'):
            scores['valuation_metrics'] += 30
            available_metrics.append('pb_ratio')
        if vm.get('market_cap'):
            scores['valuation_metrics'] += 30
            available_metrics.append('market_cap')
        
        # Score key ratios
        kr = extracted_data.get('key_ratios', {})
        if kr.get('roe'):
            scores['key_ratios'] += 30
            available_metrics.append('roe')
        else:
            missing_metrics.append('roe')
        if kr.get('roa'):
            scores['key_ratios'] += 25
            available_metrics.append('roa')
        if kr.get('profit_margin'):
            scores['key_ratios'] += 25
            available_metrics.append('profit_margin')
        if kr.get('debt_to_equity'):
            scores['key_ratios'] += 20
            available_metrics.append('debt_to_equity')
        
        # Score growth metrics
        gm = extracted_data.get('growth_metrics', {})
        if gm.get('revenue_growth_yoy'):
            scores['growth_metrics'] += 50
            available_metrics.append('revenue_growth_yoy')
        if gm.get('profit_growth_yoy'):
            scores['growth_metrics'] += 50
            available_metrics.append('profit_growth_yoy')
        
        # Score price data
        pd = extracted_data.get('price_data', {})
        if pd.get('current_price'):
            scores['price_data'] += 50
            available_metrics.append('current_price')
        if pd.get('52w_high') and pd.get('52w_low'):
            scores['price_data'] += 50
        
        # Determine recommended chart types
        recommended_charts = []
        for chart_type, config in sorted(self.chart_priority_map.items(),
                                         key=lambda x: x[1]['priority'], reverse=True):
            requirements = config['requires']
            can_generate = True
            for req in requirements:
                # Handle both 'category.metric' and 'category' formats
                if '.' in req:
                    category, metric = req.split('.', 1)
                else:
                    category, metric = req, None
                if category not in extracted_data:
                    can_generate = False
                    break
                if metric and not extracted_data[category].get(metric):
                    can_generate = False
                    break
            
            if can_generate:
                recommended_charts.append(chart_type)
        
        return {
            'completeness_scores': scores,
            'available_metrics': available_metrics,
            'missing_metrics': missing_metrics,
            'recommended_charts': recommended_charts[:5],  # Top 5
            'overall_completeness': sum(scores.values()) / len(scores)
        }
    
    async def detect_anomalies(self, extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in financial data using statistical methods.
        
        Returns:
            List of detected anomalies with descriptions.
        """
        anomalies = []
        
        if not SCIPY_AVAILABLE or not NUMPY_AVAILABLE:
            return anomalies
        
        # Check revenue growth anomalies
        fs = extracted_data.get('financial_statements', {})
        revenue_data = fs.get('revenue', {})
        if isinstance(revenue_data, dict) and len(revenue_data) >= 3:
            years = sorted([int(y) for y in revenue_data.keys() if str(y).isdigit()])
            revenues = [revenue_data.get(str(y), 0) for y in years]
            
            if len(revenues) >= 3:
                growth_rates = []
                for i in range(1, len(revenues)):
                    if revenues[i-1] > 0:
                        growth = ((revenues[i] - revenues[i-1]) / revenues[i-1]) * 100
                        growth_rates.append(growth)
                
                if len(growth_rates) >= 2:
                    mean_growth = np.mean(growth_rates)
                    std_growth = np.std(growth_rates)
                    
                    # Detect outliers (beyond 2 standard deviations)
                    for i, gr in enumerate(growth_rates):
                        if abs(gr - mean_growth) > 2 * std_growth:
                            year = years[i+1]
                            anomalies.append({
                                'type': 'revenue_growth_anomaly',
                                'metric': f'Revenue Growth {year}',
                                'value': f'{gr:.1f}%',
                                'expected_range': f'{mean_growth - 2*std_growth:.1f}% to {mean_growth + 2*std_growth:.1f}%',
                                'severity': 'high' if abs(gr - mean_growth) > 3 * std_growth else 'medium',
                                'description': f'Unusual revenue growth of {gr:.1f}% in {year} compared to historical average of {mean_growth:.1f}%'
                            })
        
        # Check profitability anomalies
        kr = extracted_data.get('key_ratios', {})
        roe = kr.get('roe')
        roa = kr.get('roa')
        
        if roe and roa:
            # ROE should typically be higher than ROA
            if roe < roa:
                anomalies.append({
                    'type': 'profitability_anomaly',
                    'metric': 'ROE vs ROA',
                    'value': f'ROE: {roe:.1f}%, ROA: {roa:.1f}%',
                    'severity': 'medium',
                    'description': f'ROE ({roe:.1f}%) is lower than ROA ({roa:.1f}%), which is unusual and may indicate leverage issues'
                })
        
        # Check valuation anomalies
        vm = extracted_data.get('valuation_metrics', {})
        pe = vm.get('pe_ratio')
        
        if pe:
            # Very high or very low P/E ratios
            if pe > 50:
                anomalies.append({
                    'type': 'valuation_anomaly',
                    'metric': 'P/E Ratio',
                    'value': f'{pe:.1f}',
                    'severity': 'medium',
                    'description': f'Very high P/E ratio of {pe:.1f} suggests high growth expectations or overvaluation'
                })
            elif pe < 5 and pe > 0:
                anomalies.append({
                    'type': 'valuation_anomaly',
                    'metric': 'P/E Ratio',
                    'value': f'{pe:.1f}',
                    'severity': 'medium',
                    'description': f'Very low P/E ratio of {pe:.1f} may indicate undervaluation or fundamental concerns'
                })
        
        return anomalies
    
    async def generate_trend_forecast(
        self, 
        extracted_data: Dict[str, Any], 
        metric: str = 'revenue',
        periods: int = 2
    ) -> Dict[str, Any]:
        """
        Generate trend forecast with confidence intervals.
        
        Args:
            extracted_data: Extracted financial data
            metric: Metric to forecast ('revenue', 'profit')
            periods: Number of future periods to forecast
        
        Returns:
            Forecast with confidence intervals
        """
        if not NUMPY_AVAILABLE:
            return {'success': False, 'error': 'numpy required for forecasting'}
        
        fs = extracted_data.get('financial_statements', {})
        metric_data = fs.get(metric, {})
        
        if not isinstance(metric_data, dict) or len(metric_data) < 3:
            return {'success': False, 'error': 'Insufficient historical data'}
        
        years = sorted([int(y) for y in metric_data.keys() if str(y).isdigit()])
        values = [metric_data.get(str(y), 0) for y in years]
        
        if len(values) < 3:
            return {'success': False, 'error': 'Need at least 3 data points'}
        
        # Linear regression for trend
        x = np.array(years)
        y = np.array(values)
        
        # Fit linear trend
        coeffs = np.polyfit(x, y, 1)
        trend_line = np.poly1d(coeffs)
        
        # Forecast future periods
        future_years = list(range(max(years) + 1, max(years) + 1 + periods))
        forecast_values = [trend_line(year) for year in future_years]
        
        # Calculate confidence intervals (simplified)
        residuals = y - trend_line(x)
        std_error = np.std(residuals)
        
        forecast_with_intervals = []
        for fv in forecast_values:
            forecast_with_intervals.append({
                'value': float(fv),
                'lower_bound': float(max(0, fv - 1.96 * std_error)),  # 95% CI
                'upper_bound': float(fv + 1.96 * std_error),
                'confidence': 'high' if std_error < np.mean(values) * 0.1 else 'medium'
            })
        
        # Calculate growth rate
        if len(forecast_values) > 0 and values[-1] > 0:
            forecast_growth = ((forecast_values[0] - values[-1]) / values[-1]) * 100
        else:
            forecast_growth = 0
        
        return {
            'success': True,
            'historical_years': years,
            'historical_values': [float(v) for v in values],
            'future_years': future_years,
            'forecast': forecast_with_intervals,
            'forecast_growth_rate': float(forecast_growth),
            'trend_direction': 'increasing' if coeffs[0] > 0 else 'decreasing',
            'trend_strength': 'strong' if abs(coeffs[0]) > np.mean(values) * 0.05 else 'moderate'
        }
    
    async def generate_contextual_narrative(
        self,
        chart_type: str,
        chart_data: Dict[str, Any],
        extracted_data: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
        ticker: str,
        company_name: str
    ) -> str:
        """
        Generate intelligent, contextual narrative for a chart.
        
        This is the "best of AI" - it doesn't just describe the chart,
        it analyzes, interprets, and provides actionable insights.
        """
        try:
            try:
                from Jotty.core.registry.skills_registry import get_skills_registry
            except ImportError:
                from core.registry.skills_registry import get_skills_registry
            
            registry = get_skills_registry()
            registry.init()
            claude_skill = registry.get_skill('claude-cli-llm')
            
            if not claude_skill:
                return ""
            
            generate_tool = claude_skill.tools.get('generate_text_tool')
            if not generate_tool:
                return ""
            
            # Build context-rich prompt
            anomalies_text = ""
            if anomalies:
                anomalies_text = "\n**Detected Anomalies:**\n"
                for anomaly in anomalies[:3]:  # Top 3
                    anomalies_text += f"- {anomaly['description']}\n"
            
            narrative_prompt = f"""You are a senior financial analyst writing a comprehensive analysis narrative for a {chart_type} chart.

**Company:** {company_name} ({ticker})
**Chart Type:** {chart_type}
**Chart Data:** {json.dumps(chart_data, indent=2)[:1500]}
**Key Financial Data:** {json.dumps({k: v for k, v in extracted_data.items() if k in ['key_ratios', 'valuation_metrics']}, indent=2)[:1000]}
{anomalies_text}

**Your Task:**
Write a comprehensive, insightful narrative (3-4 paragraphs) that:

1. **Describes the Visual Story**: What does this chart reveal about the company's financial performance?
2. **Identifies Key Patterns**: What trends, cycles, or patterns are evident?
3. **Provides Context**: How does this performance compare to industry norms or historical patterns?
4. **Highlights Anomalies**: If anomalies were detected, explain their significance
5. **Offers Actionable Insights**: What should investors pay attention to? What are the implications?
6. **Assesses Investment Implications**: What does this mean for investment decisions?

**Writing Style:**
- Professional but accessible
- Data-driven with specific numbers
- Analytical and insightful
- Forward-looking where appropriate
- Focus on what matters for investors

**Output:** Write ONLY the narrative paragraphs, no markdown formatting, no headers, just the analysis text."""

            if inspect.iscoroutinefunction(generate_tool):
                result = await generate_tool({
                    'prompt': narrative_prompt,
                    'model': 'sonnet',
                    'timeout': 90
                })
            else:
                result = generate_tool({
                    'prompt': narrative_prompt,
                    'model': 'sonnet',
                    'timeout': 90
                })
            
            if result.get('success'):
                return result.get('text', '').strip()
        except Exception as e:
            logger.debug(f"Narrative generation failed: {e}")
        
        return ""
    
    async def intelligently_select_charts(
        self,
        extracted_data: Dict[str, Any],
        analysis_result: Dict[str, Any]
    ) -> List[str]:
        """
        Intelligently select the best chart types based on data availability,
        completeness scores, and priority.
        
        Returns:
            List of recommended chart types in priority order
        """
        recommended = analysis_result.get('recommended_charts', [])
        completeness = analysis_result.get('overall_completeness', 0)
        
        # Always include core charts if data available
        selected = []
        
        # High priority: Revenue growth if we have revenue data
        if 'revenue_growth' in recommended and completeness > 30:
            selected.append('revenue_growth')
        
        # High priority: Profitability if we have ratios
        if 'profitability' in recommended and completeness > 40:
            selected.append('profitability')
        
        # High priority: Valuation if we have valuation metrics
        if 'valuation_metrics' in recommended and completeness > 30:
            selected.append('valuation_metrics')
        
        # Medium priority: Financial health if we have good ratio data
        if 'financial_health' in recommended and completeness > 50:
            selected.append('financial_health')
        
        # Add trend forecast if we have enough historical data
        fs = extracted_data.get('financial_statements', {})
        revenue_data = fs.get('revenue', {})
        if isinstance(revenue_data, dict) and len(revenue_data) >= 3:
            selected.append('trend_forecast')
        
        # Add radar if we have multi-dimensional data
        if completeness > 60:
            selected.append('radar_comparison')
        
        return selected[:6]  # Limit to 6 charts max
    
    async def create_section_placements(
        self,
        charts: List[str],
        report_sections: List[str]
    ) -> Dict[str, List[str]]:
        """
        Intelligently determine where each chart should be placed in the report.
        
        Returns:
            Dictionary mapping section names to list of chart types for that section
        """
        placements = {
            'Financial Analysis': [],
            'Valuation Analysis': [],
            'Executive Summary': [],
            'Company Overview': [],
            'Conclusion': []
        }
        
        chart_to_section = {
            'revenue_growth': 'Financial Analysis',
            'profitability': 'Financial Analysis',
            'financial_health': 'Financial Analysis',
            'trend_forecast': 'Financial Analysis',
            'valuation_metrics': 'Valuation Analysis',
            'radar_comparison': 'Executive Summary',
            'anomaly_detection': 'Financial Analysis',
        }
        
        for chart_type in charts:
            section = chart_to_section.get(chart_type, 'Financial Analysis')
            if section in placements:
                placements[section].append(chart_type)
        
        return placements


# Import inspect for async function checking
import inspect
