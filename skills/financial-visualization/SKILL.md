---
name: visualizing-financials
description: "This skill extracts structured financial data from research results and generates: - **Financial Charts**: Price trends, revenue growth, profitability metrics, peer comparisons - **Data Tables**: Financial statements, valuation metrics, key ratios, peer comparisons - **Visual Analytics**: Trend analysis, comparative charts, performance dashboards. Use when the user wants to financial, finance, investment."
---

# Financial Visualization Skill

Generate intelligent graphs and tables for financial reports.

## Description

This skill extracts structured financial data from research results and generates:
- **Financial Charts**: Price trends, revenue growth, profitability metrics, peer comparisons
- **Data Tables**: Financial statements, valuation metrics, key ratios, peer comparisons
- **Visual Analytics**: Trend analysis, comparative charts, performance dashboards


## Type
derived

## Base Skills
- image-generator


## Capabilities
- visualize
- data-fetch

## Tools

### `generate_intelligent_charts_tool` ⭐ **BEST OF AI**

Intelligently generate financial charts with AI-powered selection, analysis, and insights.

**This is the "best of AI" version that:**
- 🧠 Analyzes data completeness and selects optimal chart types
- 🔍 Detects anomalies automatically
- 📈 Generates trend forecasts with confidence intervals
- 📝 Creates contextual narratives for each chart
- 📍 Intelligently places charts in relevant report sections
- 💡 Provides actionable insights

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `company_name` (str, required): Company name
- `research_data` (dict, required): Research results
- `chart_types` (list, optional): Specific chart types (auto-selected if not provided)
- `enable_intelligence` (bool, optional): Enable intelligent features (default: True)
- `output_dir` (str, optional): Output directory
- `format` (str): Chart format ('png', 'svg', 'pdf')

**Returns:**
- `success` (bool): Whether generation succeeded
- `charts` (list): Generated chart file paths
- `chart_descriptions` (dict): Descriptions for each chart
- `chart_narratives` (dict): **AI-generated comprehensive narratives** for each chart
- `anomalies` (list): **Detected anomalies** with descriptions and severity
- `forecasts` (dict): **Trend forecasts** with confidence intervals
- `data_analysis` (dict): **Data completeness analysis** and recommendations
- `section_placements` (dict): **Intelligent section placements** for charts
- `intelligence_enabled` (bool): Whether intelligence features were used

### `generate_financial_charts_tool`

Generate multiple financial charts from extracted data.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `company_name` (str, required): Company name
- `research_data` (dict, required): Research results containing financial data
- `chart_types` (list, optional): Types of charts to generate
  - `revenue_growth`: Revenue growth chart with trend lines and YoY growth rates
  - `profitability`: Profit margins, ROE, ROA with industry benchmarks
  - `valuation_metrics`: P/E, P/B, EV/EBITDA comparisons with market cap
  - `financial_health`: Multi-dimensional health scorecard
  - `radar_comparison`: Radar chart for multi-dimensional analysis
  - `price_trend`: Stock price trend over time (when data available)
- `output_dir` (str, optional): Directory to save charts (default: ~/jotty/charts)
- `format` (str, optional): Chart format - 'png', 'svg', 'pdf' (default: 'png')

**Returns:**
- `success` (bool): Whether generation succeeded
- `charts` (list): List of generated chart file paths
- `chart_descriptions` (dict): Descriptions for each chart
- `chart_insights` (dict): AI-generated insights for each chart
- `extracted_data` (dict): Structured financial data used for charts
- `error` (str, optional): Error message if failed

### `generate_financial_tables_tool`

Generate formatted financial tables from extracted data.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol
- `company_name` (str, required): Company name
- `research_data` (dict, required): Research results containing financial data
- `table_types` (list, optional): Types of tables to generate
  - `financial_statements`: P&L, Balance Sheet, Cash Flow summary
  - `valuation_metrics`: P/E, P/B, EV/EBITDA, etc.
  - `key_ratios`: ROE, ROA, Debt/Equity, Current Ratio, etc.
  - `peer_comparison`: Side-by-side comparison with peers
  - `growth_metrics`: Revenue, profit, EPS growth rates
- `format` (str, optional): Table format - 'markdown', 'html', 'latex' (default: 'markdown')

**Returns:**
- `success` (bool): Whether generation succeeded
- `tables` (dict): Dictionary of table_name -> table_content (markdown/html)
- `table_descriptions` (dict): Descriptions for each table
- `error` (str, optional): Error message if failed

### `extract_financial_data_tool`

Extract structured financial data from research results using AI.

**Parameters:**
- `research_content` (str, required): Research text/content to extract from
- `data_types` (list, optional): Types of data to extract
  - `financial_statements`: P&L, Balance Sheet, Cash Flow
  - `valuation_metrics`: P/E, P/B, EV/EBITDA, Market Cap
  - `key_ratios`: ROE, ROA, Debt/Equity, Current Ratio
  - `price_data`: Historical prices, 52-week high/low
  - `growth_metrics`: Revenue growth, profit growth, EPS growth
- `use_llm` (bool, optional): Use LLM for extraction (default: True)

**Returns:**
- `success` (bool): Whether extraction succeeded
- `extracted_data` (dict): Structured financial data
- `confidence_scores` (dict): Confidence scores for extracted data
- `error` (str, optional): Error message if failed

## Usage Examples

### Generate Charts for Stock Report

```python
result = await generate_financial_charts_tool({
    'ticker': 'COLPAL',
    'company_name': 'Colgate Palmolive India',
    'research_data': research_results,
    'chart_types': ['price_trend', 'revenue_growth', 'profitability', 'peer_comparison'],
    'format': 'png'
})
```

### Generate Financial Tables

```python
result = await generate_financial_tables_tool({
    'ticker': 'COLPAL',
    'company_name': 'Colgate Palmolive India',
    'research_data': research_results,
    'table_types': ['financial_statements', 'valuation_metrics', 'key_ratios'],
    'format': 'markdown'
})
```

### Extract Structured Data

```python
result = await extract_financial_data_tool({
    'research_content': research_text,
    'data_types': ['financial_statements', 'valuation_metrics', 'key_ratios']
})
```

## Advanced Features

### AI-Powered Insights
- Automatically generates key insights for each chart
- Identifies trends, anomalies, and actionable takeaways
- Provides context-aware analysis

### Professional Styling
- Publication-quality charts with professional color schemes
- Consistent branding and formatting
- High-resolution output (300 DPI)

### Enhanced Chart Types
- **Revenue Growth**: Trend lines, CAGR, YoY growth annotations
- **Profitability**: Industry benchmarks, performance indicators
- **Valuation**: Market cap context, ratio comparisons
- **Financial Health**: Multi-dimensional scorecard
- **Radar Charts**: Multi-metric comparison visualization

## Dependencies

- `matplotlib>=3.7.0`: Chart generation
- `plotly>=5.0.0`: Interactive charts (optional, for HTML reports)
- `pandas>=2.0.0`: Data manipulation
- `numpy>=1.24.0`: Numerical operations
- `beautifulsoup4>=4.12.0`: HTML parsing for data extraction
- `claude-cli-llm`: AI-powered data extraction and insights

## Integration

This skill integrates with:
- `stock-research-deep`: Add visualizations to comprehensive reports
- `stock-research-comprehensive`: Enhance reports with charts and tables
- `document-converter`: Charts embedded in PDF output

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Extract financial data
- [ ] Step 2: Select chart types
- [ ] Step 3: Generate charts
- [ ] Step 4: Generate data tables
- [ ] Step 5: Add insights
```

**Step 1: Extract financial data**
Parse research results to extract structured financial metrics.

**Step 2: Select chart types**
AI-powered selection of optimal chart types based on data completeness.

**Step 3: Generate charts**
Create financial charts: revenue growth, profitability, valuation, health scores.

**Step 4: Generate data tables**
Build formatted tables for financial statements, ratios, and peer comparisons.

**Step 5: Add insights**
Generate AI narratives, detect anomalies, and forecast trends for each chart.

## Triggers
- "financial visualization"
- "financial"
- "finance"
- "investment"
- "portfolio"
- "generate"

## Category
media-creation
