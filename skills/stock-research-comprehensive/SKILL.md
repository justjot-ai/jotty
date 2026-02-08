# Comprehensive Stock Research Skill

## Description
Performs comprehensive stock research covering fundamentals, technical analysis, and broker research reports. Generates a markdown report, converts to PDF, and sends to Telegram.


## Type
composite

## Base Skills
- web-search
- claude-cli-llm
- document-converter
- telegram-sender

## Execution
sequential

## Features
- **Fundamentals Research**: Financial metrics, company overview, business model
- **Technical Analysis**: Price trends, technical indicators, chart patterns
- **Broker Reports**: Analyst ratings, price targets, research reports
- **Markdown Generation**: Combines all research into structured markdown
- **PDF Conversion**: Converts markdown to PDF
- **Telegram Delivery**: Sends PDF report to Telegram

## Tools

### comprehensive_stock_research_tool
Performs comprehensive stock research and generates PDF report.

**Parameters:**
- `ticker` (str, required): Stock ticker symbol (e.g., "COLPAL" for Colgate Palmolive India)
- `company_name` (str, optional): Full company name (default: uses ticker)
- `country` (str, optional): Country/Exchange (e.g., 'India', 'NSE', 'BSE')
- `exchange` (str, optional): Exchange name (e.g., 'NSE', 'BSE', 'NYSE')
- `output_dir` (str, optional): Output directory for files (default: ~/jotty/reports)
- `title` (str, optional): Report title (default: auto-generated)
- `author` (str, optional): Report author (default: 'Jotty Stock Research')
- `page_size` (str, optional): PDF page size - 'a4', 'a5', 'a6', 'letter' (default: 'a4')
- `telegram_chat_id` (str, optional): Telegram chat ID (default: from env)
- `send_telegram` (bool, optional): Send to Telegram (default: True)
- `max_results_per_aspect` (int, optional): Max search results per aspect (default: 15)
- `target_pages` (int, optional): Target report length in pages (default: 10)

**Returns:**
- `success` (bool): Whether operation succeeded
- `ticker` (str): Stock ticker
- `company_name` (str): Company name
- `md_path` (str): Path to markdown file
- `pdf_path` (str): Path to PDF file
- `fundamentals_research` (dict): Fundamentals research results
- `technicals_research` (dict): Technical analysis research results
- `broker_research` (dict): Broker reports research results
- `telegram_sent` (bool): Whether Telegram send succeeded
- `error` (str, optional): Error message if failed

## Usage

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('stock-research-comprehensive')
tool = skill.tools['comprehensive_stock_research_tool']

result = await tool({
    'ticker': 'CL',
    'company_name': 'Colgate Palmolive',
    'send_telegram': True
})
```

## Workflow

1. **Comprehensive Parallel Research** (12 searches covering all aspects):
   - Fundamentals & Financial Metrics
   - Financial Statements & Annual Reports
   - Valuation Analysis
   - Business Model & Products
   - Industry Analysis & Sector Trends
   - Management & Corporate Governance
   - Technical Analysis
   - Broker Research & Analyst Reports
   - Recent News & Developments
   - Risks & Challenges
   - Growth Prospects & Opportunities
   - Dividend History & Policy

2. **AI Synthesis**: Uses Claude CLI to synthesize all research into comprehensive structured markdown (target: 10 pages)

3. **Generate Markdown**: Writes detailed formatted markdown file

4. **Convert to PDF**: Converts markdown to PDF

5. **Send to Telegram**: Uploads PDF to Telegram

## Requirements

- `web-search` skill
- `claude-cli-llm` skill
- `file-operations` skill
- `document-converter` skill
- `telegram-sender` skill
- Claude CLI installed and authenticated

## Example Output Structure

```markdown
# Colgate Palmolive India (COLPAL) - Comprehensive Research Report

## Executive Summary
[1-1.5 pages: Company overview, key highlights, investment thesis]

## Company Overview & Business Model
[1-1.5 pages: History, segments, products, competitive advantages]

## Industry Analysis & Market Position
[1 page: Industry trends, market size, competitive landscape]

## Financial Analysis
[2 pages: Revenue, profitability, balance sheet, cash flow, ratios]

## Valuation Analysis
[1-1.5 pages: Valuation metrics, peer comparison, intrinsic value]

## Technical Analysis
[1 page: Price trends, indicators, support/resistance, patterns]

## Management & Corporate Governance
[0.5-1 page: Management team, governance, ESG]

## Broker Research & Analyst Coverage
[1 page: Ratings, price targets, research highlights]

## Risks & Challenges
[1 page: Business risks, regulatory, competitive threats]

## Growth Prospects & Investment Outlook
[1 page: Growth drivers, expansion plans, opportunities]

## Dividend Analysis
[0.5 page: Dividend history, yield, sustainability]

## Conclusion & Investment Recommendation
[0.5-1 page: Key findings, thesis, recommendation]
```
