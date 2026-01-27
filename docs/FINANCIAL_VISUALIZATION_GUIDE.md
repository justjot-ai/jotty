# Financial Visualization Enhancement Guide

## Overview

The financial visualization skill adds intelligent graphs and tables to stock research reports, making them more professional and easier to understand.

## Features

### 1. **Data Extraction**
- **Pattern-based extraction**: Uses regex patterns to extract common financial metrics (revenue, profit, P/E, ROE, etc.)
- **AI-powered extraction**: Uses Claude LLM to extract structured data from unstructured research content
- **Confidence scoring**: Provides confidence scores for extracted data

### 2. **Chart Generation**
- **Revenue Growth Charts**: Historical revenue trends over time
- **Profitability Charts**: ROE, ROA, profit margins visualization
- **Valuation Metrics Charts**: P/E, P/B, EV/EBITDA comparisons
- **Price Trend Charts**: Stock price movements (when data available)
- **Peer Comparison Charts**: Side-by-side comparisons with industry peers

### 3. **Table Generation**
- **Financial Statements Tables**: P&L, Balance Sheet, Cash Flow summaries
- **Valuation Metrics Tables**: Key valuation ratios
- **Key Ratios Tables**: Financial health indicators
- **Peer Comparison Tables**: Competitive positioning

## Integration with Stock Research Skills

### Automatic Integration

The `stock-research-deep` skill automatically:
1. Extracts financial data from research results
2. Generates relevant charts and tables
3. Embeds them in the markdown report
4. Includes them in PDF conversion

### Manual Usage

You can also use the visualization skill independently:

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()
viz_skill = registry.get_skill('financial-visualization')

# Extract data
extract_tool = viz_skill.tools['extract_financial_data_tool']
data = await extract_tool({
    'research_content': research_text,
    'data_types': ['financial_statements', 'valuation_metrics']
})

# Generate charts
charts_tool = viz_skill.tools['generate_financial_charts_tool']
charts = await charts_tool({
    'ticker': 'COLPAL',
    'company_name': 'Colgate Palmolive India',
    'research_data': research_results,
    'chart_types': ['revenue_growth', 'profitability']
})

# Generate tables
tables_tool = viz_skill.tools['generate_financial_tables_tool']
tables = await tables_tool({
    'ticker': 'COLPAL',
    'company_name': 'Colgate Palmolive India',
    'research_data': research_results,
    'table_types': ['financial_statements', 'valuation_metrics']
})
```

## Chart Types

### Revenue Growth Chart
- **Type**: Line chart
- **Data**: Historical revenue (3-5 years)
- **Format**: Years on X-axis, Revenue (₹ Cr) on Y-axis
- **Features**: Value labels, grid, professional styling

### Profitability Chart
- **Type**: Bar chart
- **Data**: ROE, ROA, Profit Margin
- **Format**: Metrics on X-axis, Percentage on Y-axis
- **Features**: Color-coded bars, value labels

### Valuation Metrics Chart
- **Type**: Horizontal bar chart
- **Data**: P/E, P/B, EV/EBITDA
- **Format**: Metrics on Y-axis, Ratio values on X-axis
- **Features**: Clean, readable format

## Table Formats

### Markdown Tables
- Standard markdown table format
- Compatible with PDF conversion
- Easy to read in markdown viewers

### HTML Tables (Future)
- Rich formatting options
- Better for web display

### LaTeX Tables (Future)
- Professional academic formatting
- Perfect for PDF reports

## Dependencies

Required packages:
- `matplotlib>=3.7.0`: Chart generation
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical operations
- `beautifulsoup4>=4.12.0`: HTML parsing

Install with:
```bash
pip install matplotlib pandas numpy beautifulsoup4
```

Or use the skill's requirements.txt:
```bash
pip install -r ~/jotty/skills/financial-visualization/requirements.txt
```

## Output Locations

- **Charts**: `~/jotty/charts/` (default)
- **Tables**: Embedded in markdown report
- **Extracted Data**: Returned in tool response

## Best Practices

1. **Data Quality**: Better input data = better visualizations
   - Use comprehensive research results
   - Enable LLM extraction for complex data

2. **Chart Selection**: Choose relevant charts
   - Revenue growth for growth companies
   - Profitability for mature companies
   - Valuation metrics for all companies

3. **Table Integration**: Place tables in relevant sections
   - Financial statements in Financial Analysis section
   - Valuation metrics in Valuation Analysis section
   - Key ratios in Financial Analysis section

4. **Chart References**: Reference charts in text
   - "As shown in the revenue growth chart..."
   - "The profitability metrics indicate..."
   - Makes reports more cohesive

## Future Enhancements

1. **Interactive Charts**: Plotly integration for interactive HTML reports
2. **Peer Comparison**: Automatic peer identification and comparison
3. **Technical Charts**: Price charts with indicators (RSI, MACD, etc.)
4. **Custom Styling**: Brand colors, custom themes
5. **Export Options**: Multiple formats (PNG, SVG, PDF, HTML)

## Troubleshooting

### Charts Not Generating
- Check if matplotlib is installed: `pip install matplotlib`
- Verify data extraction succeeded
- Check logs for specific errors

### Tables Not Appearing
- Ensure markdown format is selected
- Check if data extraction found relevant metrics
- Verify table content is not empty

### Data Extraction Failing
- Try enabling LLM extraction (`use_llm=True`)
- Check research content quality
- Verify patterns match your data format

## Examples

See the `stock-research-deep` skill for a complete example of visualization integration in a comprehensive research report.
