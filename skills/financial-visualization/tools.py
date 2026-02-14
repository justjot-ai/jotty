"""
Financial Visualization Skill - Generate intelligent graphs and tables for financial reports.

Capabilities:
1. Extract structured financial data from research results
2. Generate financial charts (price trends, revenue growth, profitability, peer comparisons)
3. Generate formatted tables (financial statements, valuation metrics, ratios)
4. Embed visualizations in markdown reports
"""
import asyncio
import logging
import inspect
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import os
import io
import base64

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("financial-visualization")


logger = logging.getLogger(__name__)


def safe_num(value: Any, default: float = 0) -> float:
    """Safely convert value to number, handling None and invalid types."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_get_num(d: Dict, key: str, default: float = 0) -> float:
    """Safely get numeric value from dict, handling None values."""
    return safe_num(d.get(key), default)


# Try importing visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - charts will use text-based alternatives")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available - some features may be limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available - some features may be limited")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available - interactive charts disabled")

# Professional color schemes for financial reports
FINANCIAL_COLORS = {
    'primary': '#1f4788',      # Professional blue
    'secondary': '#2E86AB',    # Accent blue
    'success': '#4CAF50',      # Green for positive
    'warning': '#FF9800',      # Orange for caution
    'danger': '#F44336',       # Red for negative
    'neutral': '#607D8B',     # Gray for neutral
    'accent1': '#A23B72',      # Purple accent
    'accent2': '#F18F01',      # Orange accent
    'accent3': '#6A994E',      # Green accent
}

# Chart styling configuration
CHART_STYLE = {
    'figure_size': (12, 7),
    'dpi': 300,
    'font_family': 'DejaVu Sans',
    'font_size': 11,
    'title_size': 14,
    'grid_alpha': 0.3,
    'line_width': 2.5,
    'bar_width': 0.6,
}


@async_tool_wrapper()
async def extract_financial_data_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured financial data from research content using AI and pattern matching.
    
    Args:
        params:
            - research_content (str): Research text/content to extract from
            - data_types (list, optional): Types of data to extract
            - use_llm (bool, optional): Use LLM for extraction (default: True)
    
    Returns:
        Dictionary with extracted_data, confidence_scores, success status
    """
    status.set_callback(params.pop('_status_callback', None))

    research_content = params.get('research_content', '')
    data_types = params.get('data_types', [
        'financial_statements', 'valuation_metrics', 'key_ratios', 
        'price_data', 'growth_metrics'
    ])
    use_llm = params.get('use_llm', True)
    
    if not research_content:
        return {
            'success': False,
            'error': 'research_content is required'
        }
    
    extracted_data = {}
    confidence_scores = {}
    
    # Pattern-based extraction for common metrics
    patterns = {
        'revenue': [
            r'revenue[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)\s*(?:crore|million|billion|cr|mn|bn)?',
            r'sales[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)\s*(?:crore|million|billion|cr|mn|bn)?',
        ],
        'net_profit': [
            r'net\s+profit[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)\s*(?:crore|million|billion|cr|mn|bn)?',
            r'profit\s+after\s+tax[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)\s*(?:crore|million|billion|cr|mn|bn)?',
        ],
        'pe_ratio': [
            r'P/E\s+(?:ratio)?[:\s]+([\d,]+\.?\d*)',
            r'price[-\s]to[-\s]earnings[:\s]+([\d,]+\.?\d*)',
        ],
        'pb_ratio': [
            r'P/B\s+(?:ratio)?[:\s]+([\d,]+\.?\d*)',
            r'price[-\s]to[-\s]book[:\s]+([\d,]+\.?\d*)',
        ],
        'roe': [
            r'ROE[:\s]+([\d,]+\.?\d*)\s*%?',
            r'return\s+on\s+equity[:\s]+([\d,]+\.?\d*)\s*%?',
        ],
        'roa': [
            r'ROA[:\s]+([\d,]+\.?\d*)\s*%?',
            r'return\s+on\s+assets[:\s]+([\d,]+\.?\d*)\s*%?',
        ],
        'market_cap': [
            r'market\s+cap(?:italization)?[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)\s*(?:crore|million|billion|cr|mn|bn)?',
        ],
        'current_price': [
            r'current\s+price[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            r'stock\s+price[:\s]+(?:₹|Rs\.?|INR)?\s*([\d,]+\.?\d*)',
        ],
    }
    
    # Extract using patterns
    for metric, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, research_content, re.IGNORECASE)
            if matches:
                # Take the most recent/last mentioned value
                value_str = matches[-1].replace(',', '')
                try:
                    value = float(value_str)
                    extracted_data[metric] = value
                    confidence_scores[metric] = 0.7  # Pattern-based extraction confidence
                    break
                except ValueError:
                    continue
    
    # Use LLM for more complex extraction if requested
    if use_llm and ('financial_statements' in data_types or 'valuation_metrics' in data_types):
        try:
            try:
                from Jotty.core.registry.skills_registry import get_skills_registry
            except ImportError:
                from Jotty.core.registry.skills_registry import get_skills_registry

            registry = get_skills_registry()
            registry.init()
            claude_skill = registry.get_skill('claude-cli-llm')
            
            if claude_skill:
                generate_tool = claude_skill.tools.get('generate_text_tool')
                
                extraction_prompt = f"""Extract structured financial data from the following research content for a stock analysis report.

**RESEARCH CONTENT:**
{research_content[:8000]}  # Limit to avoid token limits

**EXTRACT THE FOLLOWING DATA (if available):**

1. **Financial Statements Summary:**
   - Revenue (last 3-5 years if available)
   - Net Profit (last 3-5 years if available)
   - Total Assets
   - Total Equity
   - Total Debt

2. **Valuation Metrics:**
   - P/E Ratio
   - P/B Ratio
   - EV/EBITDA
   - Market Capitalization
   - Current Stock Price

3. **Key Ratios:**
   - ROE (Return on Equity)
   - ROA (Return on Assets)
   - Debt-to-Equity Ratio
   - Current Ratio
   - Profit Margin

4. **Price Data:**
   - Current Price
   - 52-week High
   - 52-week Low
   - Price Range

5. **Growth Metrics:**
   - Revenue Growth Rate (YoY, 3-year CAGR)
   - Profit Growth Rate (YoY, 3-year CAGR)
   - EPS Growth Rate

**OUTPUT FORMAT:**
Return ONLY a valid JSON object with this structure:
{{
  "financial_statements": {{
    "revenue": {{"2023": 1000, "2022": 950, "2021": 900}},
    "net_profit": {{"2023": 150, "2022": 140, "2021": 130}},
    "total_assets": 2000,
    "total_equity": 1200,
    "total_debt": 800
  }},
  "valuation_metrics": {{
    "pe_ratio": 25.5,
    "pb_ratio": 4.2,
    "ev_ebitda": 18.3,
    "market_cap": 50000,
    "current_price": 1250.50
  }},
  "key_ratios": {{
    "roe": 12.5,
    "roa": 7.5,
    "debt_to_equity": 0.67,
    "current_ratio": 1.8,
    "profit_margin": 15.0
  }},
  "price_data": {{
    "current_price": 1250.50,
    "52w_high": 1400.00,
    "52w_low": 1100.00
  }},
  "growth_metrics": {{
    "revenue_growth_yoy": 5.3,
    "revenue_growth_3y_cagr": 4.8,
    "profit_growth_yoy": 7.1,
    "eps_growth_yoy": 6.5
  }}
}}

If data is not available, use null. Return ONLY the JSON, no other text."""

                if inspect.iscoroutinefunction(generate_tool):
                    llm_result = await generate_tool({
                        'prompt': extraction_prompt,
                        'model': 'sonnet',
                        'timeout': 120
                    })
                else:
                    llm_result = generate_tool({
                        'prompt': extraction_prompt,
                        'model': 'sonnet',
                        'timeout': 120
                    })
                
                if llm_result.get('success'):
                    llm_text = llm_result.get('text', '')
                    # Extract JSON from response
                    json_match = re.search(r'\{.*\}', llm_text, re.DOTALL)
                    if json_match:
                        try:
                            llm_data = json.loads(json_match.group())
                            # Merge LLM-extracted data (higher confidence)
                            for category, data in llm_data.items():
                                if isinstance(data, dict):
                                    if category not in extracted_data:
                                        extracted_data[category] = {}
                                    extracted_data[category].update(data)
                                    confidence_scores[f'{category}_llm'] = 0.9
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse LLM-extracted JSON")
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, continuing with pattern-based extraction")
    
    return {
        'success': True,
        'extracted_data': extracted_data,
        'confidence_scores': confidence_scores
    }


@async_tool_wrapper()
async def generate_intelligent_charts_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligently generate financial charts with AI-powered selection, analysis, and insights.

    This is the "best of AI" version - it:
    1. Analyzes data completeness
    2. Selects optimal chart types
    3. Detects anomalies
    4. Generates forecasts
    5. Creates contextual narratives
    6. Provides section placements

    Args:
        params:
            - ticker (str): Stock ticker
            - company_name (str): Company name
            - research_data (dict): Research results
            - chart_types (list, optional): Specific chart types (auto-selected if not provided)
            - enable_intelligence (bool, optional): Enable intelligent features (default: True)
            - output_dir (str, optional): Output directory
            - format (str): Chart format ('png', 'svg', 'pdf')

    Returns:
        Dictionary with charts, insights, narratives, anomalies, forecasts, section placements
    """
    status.set_callback(params.pop('_status_callback', None))
    # Import intelligent orchestrator - handle both relative and absolute imports
    try:
        # Try absolute import first (works when skill is loaded dynamically)
        import sys
        import os
        skill_dir = os.path.dirname(os.path.abspath(__file__))
        if skill_dir not in sys.path:
            sys.path.insert(0, skill_dir)
        from intelligent_orchestrator import IntelligentVisualizationOrchestrator
    except ImportError:
        # Fallback to relative import
        try:
            from .intelligent_orchestrator import IntelligentVisualizationOrchestrator
        except ImportError:
            logger.warning("IntelligentVisualizationOrchestrator not available, using standard charts")
            # Fallback to standard chart generation
            return await generate_financial_charts_tool(params)
    
    ticker = params.get('ticker', '')
    company_name = params.get('company_name', '')
    research_data = params.get('research_data', {})
    enable_intelligence = params.get('enable_intelligence', True)
    output_dir = params.get('output_dir', os.path.expanduser('~/jotty/charts'))
    chart_format = params.get('format', 'png')
    
    if not ticker or not company_name:
        return {
            'success': False,
            'error': 'ticker and company_name are required'
        }
    
    # Extract financial data first
    research_content = json.dumps(research_data, indent=2)[:10000]
    extract_result = await extract_financial_data_tool({
        'research_content': research_content,
        'use_llm': True
    })
    
    extracted_data = extract_result.get('extracted_data', {})
    
    orchestrator = IntelligentVisualizationOrchestrator()
    
    # Intelligent analysis
    analysis_result = await orchestrator.analyze_data_completeness(extracted_data)
    anomalies = await orchestrator.detect_anomalies(extracted_data) if enable_intelligence else []
    
    # Select chart types intelligently
    requested_charts = params.get('chart_types')
    if not requested_charts and enable_intelligence:
        chart_types = await orchestrator.intelligently_select_charts(extracted_data, analysis_result)
        logger.info(f"🧠 Intelligently selected charts: {chart_types}")
    else:
        chart_types = requested_charts or ['revenue_growth', 'profitability', 'valuation_metrics']
    
    # Generate forecasts if enabled
    forecasts = {}
    if enable_intelligence and 'trend_forecast' in chart_types:
        forecast_result = await orchestrator.generate_trend_forecast(extracted_data, 'revenue', 2)
        if forecast_result.get('success'):
            forecasts['revenue'] = forecast_result
    
    # Generate charts using existing function
    charts = []
    chart_descriptions = {}
    chart_insights = {}
    chart_narratives = {}
    
    for chart_type in chart_types:
        if chart_type == 'trend_forecast':
            # Handle forecast chart separately
            continue
        
        try:
            chart_path, description = await _generate_chart(
                chart_type=chart_type,
                ticker=ticker,
                company_name=company_name,
                extracted_data=extracted_data,
                output_dir=output_dir,
                format=chart_format,
                timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
            )
            
            if chart_path:
                charts.append(chart_path)
                chart_descriptions[chart_type] = description
                
                # Generate intelligent narrative
                if enable_intelligence:
                    chart_data = {chart_type: extracted_data}
                    narrative = await orchestrator.generate_contextual_narrative(
                        chart_type, chart_data, extracted_data, anomalies, ticker, company_name
                    )
                    if narrative:
                        chart_narratives[chart_type] = narrative
        except Exception as e:
            logger.error(f"Failed to generate {chart_type} chart: {e}", exc_info=True)
    
    # Determine section placements
    section_placements = {}
    if enable_intelligence:
        section_placements = await orchestrator.create_section_placements(
            chart_types, ['Financial Analysis', 'Valuation Analysis', 'Executive Summary']
        )
    
    return {
        'success': len(charts) > 0,
        'charts': charts,
        'chart_descriptions': chart_descriptions,
        'chart_narratives': chart_narratives,
        'chart_insights': chart_insights,
        'anomalies': anomalies,
        'forecasts': forecasts,
        'data_analysis': analysis_result,
        'section_placements': section_placements,
        'extracted_data': extracted_data,
        'intelligence_enabled': enable_intelligence
    }


@async_tool_wrapper()
async def generate_financial_charts_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate financial charts from extracted data.
    
    Args:
        params:
            - ticker (str): Stock ticker
            - company_name (str): Company name
            - research_data (dict): Research results
            - chart_types (list): Types of charts to generate
            - output_dir (str, optional): Output directory
            - format (str): Chart format ('png', 'svg', 'pdf')
    
    Returns:
        Dictionary with charts list, descriptions, success status
    """
    status.set_callback(params.pop('_status_callback', None))

    ticker = params.get('ticker', '')
    company_name = params.get('company_name', '')
    research_data = params.get('research_data', {})
    chart_types = params.get('chart_types', ['revenue_growth', 'profitability', 'valuation_metrics', 'financial_health'])
    output_dir = params.get('output_dir', os.path.expanduser('~/jotty/charts'))
    chart_format = params.get('format', 'png')
    
    if not ticker or not company_name:
        return {
            'success': False,
            'error': 'ticker and company_name are required'
        }
    
    if not MATPLOTLIB_AVAILABLE:
        return {
            'success': False,
            'error': 'matplotlib is not installed. Install with: pip install matplotlib'
        }
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # First extract financial data
    research_content = json.dumps(research_data, indent=2)[:10000]  # Limit size
    extract_result = await extract_financial_data_tool({
        'research_content': research_content,
        'use_llm': True
    })
    
    if not extract_result.get('success'):
        logger.warning("Data extraction failed, generating charts with limited data")
    
    extracted_data = extract_result.get('extracted_data', {})
    
    charts = []
    chart_descriptions = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate requested charts
    chart_insights_dict = {}
    for chart_type in chart_types:
        try:
            chart_path, description = await _generate_chart(
                chart_type=chart_type,
                ticker=ticker,
                company_name=company_name,
                extracted_data=extracted_data,
                output_dir=output_dir,
                format=chart_format,
                timestamp=timestamp
            )
            if chart_path:
                charts.append(chart_path)
                chart_descriptions[chart_type] = description
                # Extract insights if present
                if '| Key Insight:' in description:
                    parts = description.split('| Key Insight:')
                    chart_descriptions[chart_type] = parts[0].strip()
                    chart_insights_dict[chart_type] = parts[1].strip()
        except Exception as e:
            logger.error(f"Failed to generate {chart_type} chart: {e}", exc_info=True)
    
    return {
        'success': len(charts) > 0,
        'charts': charts,
        'chart_descriptions': chart_descriptions,
        'chart_insights': chart_insights_dict,
        'extracted_data': extracted_data
    }


async def _generate_chart_insights(
    chart_type: str,
    chart_data: Dict[str, Any],
    ticker: str,
    company_name: str
) -> str:
    """Generate AI-powered insights for a chart."""
    try:
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
        except ImportError:
            from Jotty.core.registry.skills_registry import get_skills_registry
        
        registry = get_skills_registry()
        registry.init()
        claude_skill = registry.get_skill('claude-cli-llm')
        
        if not claude_skill:
            return ""
        
        generate_tool = claude_skill.tools.get('generate_text_tool')
        if not generate_tool:
            return ""
        
        insights_prompt = f"""Analyze the following financial chart data for {company_name} ({ticker}) and provide 2-3 key insights in a single paragraph.

**Chart Type:** {chart_type}
**Data:** {json.dumps(chart_data, indent=2)[:1000]}

**Requirements:**
- Identify the most important trend or pattern
- Highlight any anomalies or notable points
- Provide actionable insight for investors
- Keep it concise (2-3 sentences maximum)
- Use professional financial language

**Output:** Just the insight paragraph, no markdown formatting."""

        if inspect.iscoroutinefunction(generate_tool):
            result = await generate_tool({
                'prompt': insights_prompt,
                'model': 'sonnet',
                'timeout': 60
            })
        else:
            result = generate_tool({
                'prompt': insights_prompt,
                'model': 'sonnet',
                'timeout': 60
            })
        
        if result.get('success'):
            return result.get('text', '').strip()
    except Exception as e:
        logger.debug(f"Chart insights generation failed: {e}")
    
    return ""


async def _generate_chart(
    chart_type: str,
    ticker: str,
    company_name: str,
    extracted_data: Dict[str, Any],
    output_dir: str,
    format: str,
    timestamp: str
) -> tuple:
    """Generate a specific chart type with professional styling."""
    
    fig, ax = plt.subplots(figsize=CHART_STYLE['figure_size'], dpi=CHART_STYLE['dpi'])
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Apply professional styling
    plt.rcParams['font.family'] = CHART_STYLE['font_family']
    plt.rcParams['font.size'] = CHART_STYLE['font_size']
    
    chart_insights = ""
    chart_data_for_insights = {}
    
    if chart_type == 'revenue_growth':
        # Enhanced revenue growth chart with trend line and growth annotations
        financial_data = extracted_data.get('financial_statements', {})
        revenue_data = financial_data.get('revenue', {})
        
        if revenue_data and isinstance(revenue_data, dict):
            years = sorted([int(y) for y in revenue_data.keys() if str(y).isdigit()])
            revenues = [revenue_data.get(str(y), 0) for y in years]
            
            chart_data_for_insights = {'years': years, 'revenues': revenues}
            
            # Main revenue line
            ax.plot(years, revenues, marker='o', linewidth=CHART_STYLE['line_width'], 
                   markersize=10, color=FINANCIAL_COLORS['primary'], 
                   markerfacecolor='white', markeredgewidth=2, label='Revenue')
            
            # Add trend line if we have enough data points
            if len(years) >= 3 and NUMPY_AVAILABLE:
                z = np.polyfit(years, revenues, 1)
                p = np.poly1d(z)
                ax.plot(years, p(years), "--", alpha=0.5, color=FINANCIAL_COLORS['neutral'], 
                       linewidth=1.5, label='Trend Line')
            
            # Calculate and display growth rates
            if len(revenues) >= 2:
                growth_rates = []
                for i in range(1, len(revenues)):
                    if revenues[i-1] > 0:
                        growth = ((revenues[i] - revenues[i-1]) / revenues[i-1]) * 100
                        growth_rates.append(growth)
                        # Annotate growth rate
                        ax.annotate(f'+{growth:.1f}%', 
                                  xy=(years[i], revenues[i]),
                                  xytext=(years[i], revenues[i] + max(revenues)*0.05),
                                  ha='center', fontsize=9, fontweight='bold',
                                  color=FINANCIAL_COLORS['success'] if growth > 0 else FINANCIAL_COLORS['danger'],
                                  arrowprops=dict(arrowstyle='->', lw=1, alpha=0.5))
            
            ax.set_xlabel('Year', fontsize=CHART_STYLE['font_size']+1, fontweight='bold', color='#333')
            ax.set_ylabel('Revenue (₹ Crore)', fontsize=CHART_STYLE['font_size']+1, fontweight='bold', color='#333')
            ax.set_title(f'{company_name} ({ticker}) - Revenue Growth Trend', 
                        fontsize=CHART_STYLE['title_size'], fontweight='bold', pad=20, color='#1a1a1a')
            ax.grid(True, alpha=CHART_STYLE['grid_alpha'], linestyle='--', linewidth=0.8)
            ax.set_xticks(years)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
            
            # Add value labels with better positioning
            for year, rev in zip(years, revenues):
                ax.text(year, rev, f'₹{rev:.0f}Cr', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold', color='#555',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
            
            # Add CAGR if available
            if len(revenues) >= 2:
                cagr = ((revenues[-1] / revenues[0]) ** (1 / (len(revenues) - 1)) - 1) * 100
                ax.text(0.02, 0.98, f'CAGR: {cagr:.1f}%', transform=ax.transAxes,
                       fontsize=10, fontweight='bold', color=FINANCIAL_COLORS['primary'],
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            ax.text(0.5, 0.5, 'Revenue data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#999')
        
        description = f"Revenue growth trend for {company_name} showing historical revenue performance with year-over-year growth rates."
    
    elif chart_type == 'profitability':
        # Enhanced profitability metrics with industry benchmarks
        ratios = extracted_data.get('key_ratios', {})
        
        metrics = []
        values = []
        colors = []
        benchmark_values = []  # Placeholder for industry benchmarks
        
        metric_configs = [
            ('roe', 'ROE (%)', FINANCIAL_COLORS['primary'], 15),  # Typical good ROE
            ('roa', 'ROA (%)', FINANCIAL_COLORS['secondary'], 8),  # Typical good ROA
            ('profit_margin', 'Profit Margin (%)', FINANCIAL_COLORS['accent1'], 10)
        ]
        
        for metric_key, metric_label, color, benchmark in metric_configs:
            if ratios.get(metric_key):
                metrics.append(metric_label)
                values.append(ratios[metric_key])
                colors.append(color)
                benchmark_values.append(benchmark)
                chart_data_for_insights[metric_key] = ratios[metric_key]
        
        if metrics:
            x_pos = np.arange(len(metrics))
            bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='white', 
                         linewidth=2, width=CHART_STYLE['bar_width'])
            
            # Add benchmark lines
            for i, (val, bench) in enumerate(zip(values, benchmark_values)):
                ax.axhline(y=bench, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(len(metrics)-0.5, bench, f'Benchmark: {bench}%', 
                       fontsize=8, color='gray', va='center')
            
            ax.set_ylabel('Percentage (%)', fontsize=CHART_STYLE['font_size']+1, fontweight='bold', color='#333')
            ax.set_title(f'{company_name} ({ticker}) - Profitability Analysis', 
                        fontsize=CHART_STYLE['title_size'], fontweight='bold', pad=20, color='#1a1a1a')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics, rotation=0, ha='center')
            ax.grid(True, alpha=CHART_STYLE['grid_alpha'], axis='y', linestyle='--')
            
            # Add value labels with performance indicators
            for i, (bar, val, bench) in enumerate(zip(bars, values, benchmark_values)):
                height = bar.get_height()
                performance = '✓' if val >= bench else '⚠'
                color_indicator = FINANCIAL_COLORS['success'] if val >= bench else FINANCIAL_COLORS['warning']
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                       f'{val:.1f}% {performance}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold', color=color_indicator,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        else:
            ax.text(0.5, 0.5, 'Profitability data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#999')
        
        description = f"Profitability metrics for {company_name} including ROE, ROA, and profit margins with industry benchmarks."
    
    elif chart_type == 'valuation_metrics':
        # Enhanced valuation metrics with context zones
        valuation = extracted_data.get('valuation_metrics', {})
        
        metrics = []
        values = []
        colors = []
        
        valuation_configs = [
            ('pe_ratio', 'P/E Ratio', FINANCIAL_COLORS['primary']),
            ('pb_ratio', 'P/B Ratio', FINANCIAL_COLORS['secondary']),
            ('ev_ebitda', 'EV/EBITDA', FINANCIAL_COLORS['accent2'])
        ]
        
        for metric_key, metric_label, color in valuation_configs:
            if valuation.get(metric_key):
                metrics.append(metric_label)
                values.append(valuation[metric_key])
                colors.append(color)
                chart_data_for_insights[metric_key] = valuation[metric_key]
        
        if metrics:
            y_pos = np.arange(len(metrics))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='white', 
                          linewidth=2, height=CHART_STYLE['bar_width'])
            
            ax.set_xlabel('Ratio Value', fontsize=CHART_STYLE['font_size']+1, fontweight='bold', color='#333')
            ax.set_title(f'{company_name} ({ticker}) - Valuation Analysis', 
                        fontsize=CHART_STYLE['title_size'], fontweight='bold', pad=20, color='#1a1a1a')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(metrics)
            ax.grid(True, alpha=CHART_STYLE['grid_alpha'], axis='x', linestyle='--')
            
            # Add value labels
            max_val = max(values) if values else 1
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.text(width + max_val*0.02, bar.get_y() + bar.get_height()/2.,
                       f'{val:.2f}', ha='left', va='center', fontsize=10, fontweight='bold',
                       color='#333', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
            # Add market cap if available
            if valuation.get('market_cap'):
                market_cap_text = f"Market Cap: ₹{valuation['market_cap']:.0f} Cr"
                ax.text(0.98, 0.02, market_cap_text, transform=ax.transAxes,
                       fontsize=9, fontweight='bold', color=FINANCIAL_COLORS['primary'],
                       ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        else:
            ax.text(0.5, 0.5, 'Valuation data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#999')
        
        description = f"Valuation metrics for {company_name} including P/E, P/B, and EV/EBITDA ratios with market capitalization."
    
    elif chart_type == 'radar_comparison':
        # Radar chart for multi-dimensional peer comparison (NEW!)
        ratios = extracted_data.get('key_ratios', {})
        valuation = extracted_data.get('valuation_metrics', {})
        
        if not ratios and not valuation:
            ax.text(0.5, 0.5, 'Insufficient data for radar chart', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#999')
            description = "Multi-dimensional comparison chart (data not available)"
        else:
            # Normalize metrics for radar chart
            categories = []
            values_normalized = []
            
            metric_mapping = {
                'roe': ('ROE', ratios.get('roe') or 0, 25),  # Normalize to 0-25%
                'roa': ('ROA', ratios.get('roa') or 0, 15),  # Normalize to 0-15%
                'profit_margin': ('Profit Margin', ratios.get('profit_margin') or 0, 30),
                'pe_ratio': ('P/E Ratio', valuation.get('pe_ratio') or 0, 50),  # Normalize
                'pb_ratio': ('P/B Ratio', valuation.get('pb_ratio') or 0, 10),
            }

            for key, (label, value, max_val) in metric_mapping.items():
                if value is not None and value > 0:
                    categories.append(label)
                    values_normalized.append(min(value / max_val * 100, 100))  # Cap at 100
            
            if categories:
                # Create radar chart - need to recreate figure with polar projection
                plt.close(fig)
                fig = plt.figure(figsize=CHART_STYLE['figure_size'], dpi=CHART_STYLE['dpi'])
                ax = plt.subplot(111, projection='polar')
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values_normalized += values_normalized[:1]  # Close the polygon
                angles += angles[:1]
                
                ax.plot(angles, values_normalized, 'o-', linewidth=CHART_STYLE['line_width'], 
                       color=FINANCIAL_COLORS['primary'], label=company_name, markersize=8)
                ax.fill(angles, values_normalized, alpha=0.25, color=FINANCIAL_COLORS['primary'])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=9)
                ax.set_ylim(0, 100)
                ax.set_title(f'{company_name} ({ticker}) - Multi-Dimensional Analysis', 
                            fontsize=CHART_STYLE['title_size'], fontweight='bold', pad=20)
                ax.grid(True, alpha=CHART_STYLE['grid_alpha'])
                
                chart_data_for_insights = dict(zip(categories, values_normalized[:-1]))
                description = f"Multi-dimensional radar chart showing {company_name}'s performance across key financial metrics."
            else:
                ax.text(0.5, 0.5, 'Insufficient data for radar chart', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#999')
                description = "Multi-dimensional comparison chart (data not available)"
    
    elif chart_type == 'financial_health':
        # Financial health scorecard (NEW!)
        ratios = extracted_data.get('key_ratios', {})
        financial_data = extracted_data.get('financial_statements', {})
        
        health_metrics = {
            'Liquidity': ratios.get('current_ratio') or 0,
            'Profitability': ratios.get('profit_margin') or 0,
            'Efficiency': ratios.get('roa') or 0,
            'Leverage': 100 - min((ratios.get('debt_to_equity') or 0) * 20, 100) if ratios.get('debt_to_equity') else 50,
            'Growth': extracted_data.get('growth_metrics', {}).get('revenue_growth_yoy') or 0
        }

        if any(v is not None and v > 0 for v in health_metrics.values()):
            categories = list(health_metrics.keys())
            values = [health_metrics[k] for k in categories]
            
            # Color code based on performance
            colors_list = [FINANCIAL_COLORS['success'] if v >= 70 else 
                          FINANCIAL_COLORS['warning'] if v >= 40 else 
                          FINANCIAL_COLORS['danger'] for v in values]
            
            bars = ax.barh(categories, values, color=colors_list, alpha=0.8, edgecolor='white', linewidth=2)
            ax.set_xlabel('Health Score (0-100)', fontsize=CHART_STYLE['font_size']+1, fontweight='bold')
            ax.set_title(f'{company_name} ({ticker}) - Financial Health Scorecard', 
                        fontsize=CHART_STYLE['title_size'], fontweight='bold', pad=20)
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=CHART_STYLE['grid_alpha'], axis='x')
            
            # Add score labels
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                       f'{val:.0f}', ha='left', va='center', fontsize=10, fontweight='bold')
            
            chart_data_for_insights = health_metrics
            description = f"Financial health scorecard for {company_name} across liquidity, profitability, efficiency, leverage, and growth metrics."
        else:
            ax.text(0.5, 0.5, 'Financial health data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#999')
            description = "Financial health scorecard (data not available)"
    
    else:
        # Default: Key metrics overview
        ax.text(0.5, 0.5, f'Chart type "{chart_type}" not yet implemented', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12, color='#999')
        description = f"Placeholder chart for {chart_type}"
    
    plt.tight_layout()
    
    # Generate AI insights for the chart
    if chart_data_for_insights:
        chart_insights = await _generate_chart_insights(chart_type, chart_data_for_insights, ticker, company_name)
        if chart_insights:
            description += f" | Key Insight: {chart_insights}"
    
    # Save chart
    chart_filename = f"{ticker}_{chart_type}_{timestamp}.{format}"
    chart_path = os.path.join(output_dir, chart_filename)
    fig.savefig(chart_path, dpi=CHART_STYLE['dpi'], bbox_inches='tight', format=format, 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return chart_path, description


@async_tool_wrapper()
async def generate_financial_tables_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate formatted financial tables from extracted data.
    
    Args:
        params:
            - ticker (str): Stock ticker
            - company_name (str): Company name
            - research_data (dict): Research results
            - table_types (list): Types of tables to generate
            - format (str): Table format ('markdown', 'html', 'latex')
    
    Returns:
        Dictionary with tables dict, descriptions, success status
    """
    status.set_callback(params.pop('_status_callback', None))

    ticker = params.get('ticker', '')
    company_name = params.get('company_name', '')
    research_data = params.get('research_data', {})
    table_types = params.get('table_types', ['financial_statements', 'valuation_metrics', 'key_ratios'])
    table_format = params.get('format', 'markdown')
    
    if not ticker or not company_name:
        return {
            'success': False,
            'error': 'ticker and company_name are required'
        }
    
    # Extract financial data
    research_content = json.dumps(research_data, indent=2)[:10000]
    extract_result = await extract_financial_data_tool({
        'research_content': research_content,
        'use_llm': True
    })
    
    extracted_data = extract_result.get('extracted_data', {})
    
    tables = {}
    table_descriptions = {}
    
    # Generate requested tables
    for table_type in table_types:
        try:
            table_content, description = await _generate_table(
                table_type=table_type,
                ticker=ticker,
                company_name=company_name,
                extracted_data=extracted_data,
                format=table_format
            )
            if table_content:
                tables[table_type] = table_content
                table_descriptions[table_type] = description
        except Exception as e:
            logger.error(f"Failed to generate {table_type} table: {e}")
    
    return {
        'success': len(tables) > 0,
        'tables': tables,
        'table_descriptions': table_descriptions,
        'extracted_data': extracted_data
    }


async def _generate_table(
    table_type: str,
    ticker: str,
    company_name: str,
    extracted_data: Dict[str, Any],
    format: str
) -> tuple:
    """Generate a specific table type."""

    # Normalize table type names (handle LLM variations)
    table_type_aliases = {
        'financials': 'financial_statements',
        'valuation': 'valuation_metrics',
        'ratios': 'key_ratios',
        'metrics': 'valuation_metrics',
        'statements': 'financial_statements',
    }
    table_type = table_type_aliases.get(table_type, table_type)

    if table_type == 'financial_statements':
        financial_data = extracted_data.get('financial_statements', {})
        revenue_data = financial_data.get('revenue', {})
        profit_data = financial_data.get('net_profit', {})
        
        if format == 'markdown':
            table_lines = [
                f"### Financial Statements Summary - {company_name} ({ticker})\n",
                "| Metric | " + " | ".join(sorted(revenue_data.keys() if revenue_data else ['N/A'])) + " |",
                "|--------|" + "|".join(["---"] * (len(revenue_data) + 1 if revenue_data else 2)) + "|"
            ]
            
            if revenue_data:
                years = sorted(revenue_data.keys())
                revenue_row = "| Revenue (₹ Cr) | " + " | ".join([f"{revenue_data.get(y, 'N/A')}" for y in years]) + " |"
                table_lines.append(revenue_row)
            
            if profit_data:
                years = sorted(profit_data.keys())
                profit_row = "| Net Profit (₹ Cr) | " + " | ".join([f"{profit_data.get(y, 'N/A')}" for y in years]) + " |"
                table_lines.append(profit_row)
            
            table_content = "\n".join(table_lines)
        else:
            table_content = f"Financial statements data for {company_name}"
        
        description = f"Summary of financial statements including revenue and profit trends."
    
    elif table_type == 'valuation_metrics':
        valuation = extracted_data.get('valuation_metrics', {})
        
        if format == 'markdown':
            table_lines = [
                f"### Valuation Metrics - {company_name} ({ticker})\n",
                "| Metric | Value |",
                "|--------|-------|",
            ]
            
            if valuation.get('pe_ratio'):
                table_lines.append(f"| P/E Ratio | {valuation['pe_ratio']:.2f} |")
            if valuation.get('pb_ratio'):
                table_lines.append(f"| P/B Ratio | {valuation['pb_ratio']:.2f} |")
            if valuation.get('ev_ebitda'):
                table_lines.append(f"| EV/EBITDA | {valuation['ev_ebitda']:.2f} |")
            if valuation.get('market_cap'):
                table_lines.append(f"| Market Cap (₹ Cr) | {valuation['market_cap']:.0f} |")
            if valuation.get('current_price'):
                table_lines.append(f"| Current Price (₹) | {valuation['current_price']:.2f} |")
            
            table_content = "\n".join(table_lines)
        else:
            table_content = f"Valuation metrics for {company_name}"
        
        description = f"Key valuation metrics including P/E, P/B, and EV/EBITDA ratios."
    
    elif table_type == 'key_ratios':
        ratios = extracted_data.get('key_ratios', {})
        
        if format == 'markdown':
            table_lines = [
                f"### Key Financial Ratios - {company_name} ({ticker})\n",
                "| Ratio | Value |",
                "|-------|-------|",
            ]
            
            if ratios.get('roe'):
                table_lines.append(f"| ROE (%) | {ratios['roe']:.2f} |")
            if ratios.get('roa'):
                table_lines.append(f"| ROA (%) | {ratios['roa']:.2f} |")
            if ratios.get('debt_to_equity'):
                table_lines.append(f"| Debt/Equity | {ratios['debt_to_equity']:.2f} |")
            if ratios.get('current_ratio'):
                table_lines.append(f"| Current Ratio | {ratios['current_ratio']:.2f} |")
            if ratios.get('profit_margin'):
                table_lines.append(f"| Profit Margin (%) | {ratios['profit_margin']:.2f} |")
            
            table_content = "\n".join(table_lines)
        else:
            table_content = f"Key financial ratios for {company_name}"
        
        description = f"Key financial ratios including ROE, ROA, debt-to-equity, and profitability metrics."
    
    elif table_type == 'peer_comparison':
        # Peer comparison table
        valuation = extracted_data.get('valuation_metrics', {})
        ratios = extracted_data.get('key_ratios', {})

        if format == 'markdown':
            table_lines = [
                f"### Peer Comparison - {company_name} ({ticker})\n",
                "| Metric | {ticker} | Sector Avg | Industry Avg |",
                "|--------|----------|------------|--------------|",
            ]

            # Add available metrics with placeholder peer data
            if valuation.get('pe_ratio'):
                table_lines.append(f"| P/E Ratio | {valuation['pe_ratio']:.2f} | - | - |")
            if valuation.get('pb_ratio'):
                table_lines.append(f"| P/B Ratio | {valuation['pb_ratio']:.2f} | - | - |")
            if ratios.get('roe'):
                table_lines.append(f"| ROE (%) | {ratios['roe']:.2f} | - | - |")
            if ratios.get('profit_margin'):
                table_lines.append(f"| Profit Margin (%) | {ratios['profit_margin']:.2f} | - | - |")

            if len(table_lines) <= 3:
                table_lines.append("| No data available | - | - | - |")

            table_content = "\n".join(table_lines)
        else:
            table_content = f"Peer comparison data for {company_name}"

        description = f"Comparison of {company_name} metrics against sector and industry averages."

    else:
        table_content = f"Table type '{table_type}' not yet implemented"
        description = f"Placeholder table for {table_type}"

    return table_content, description
