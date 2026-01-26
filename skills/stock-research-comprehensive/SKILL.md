# Comprehensive Stock Research Skill

## Description
Performs comprehensive stock research covering fundamentals, technical analysis, and broker research reports. Generates a markdown report, converts to PDF, and sends to Telegram.

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
- `ticker` (str, required): Stock ticker symbol (e.g., "CL" for Colgate Palmolive)
- `company_name` (str, optional): Full company name (default: uses ticker)
- `output_dir` (str, optional): Output directory for files (default: ~/jotty/reports)
- `title` (str, optional): Report title (default: auto-generated)
- `author` (str, optional): Report author (default: 'Jotty Stock Research')
- `page_size` (str, optional): PDF page size - 'a4', 'a5', 'a6', 'letter' (default: 'a4')
- `telegram_chat_id` (str, optional): Telegram chat ID (default: from env)
- `send_telegram` (bool, optional): Send to Telegram (default: True)
- `max_results_per_aspect` (int, optional): Max search results per aspect (default: 10)

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

1. **Parallel Research** (3 searches in parallel):
   - Fundamentals: "Colgate Palmolive fundamentals financial metrics"
   - Technicals: "CL stock technical analysis price trends"
   - Broker Reports: "Colgate Palmolive analyst reports ratings"

2. **Combine Research**: Uses Claude CLI to synthesize research into structured markdown

3. **Generate Markdown**: Writes formatted markdown file

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
# Colgate Palmolive (CL) - Comprehensive Research Report

## Executive Summary
[Generated summary]

## 1. Fundamentals Analysis
[Financial metrics, business overview, competitive position]

## 2. Technical Analysis
[Price trends, technical indicators, support/resistance levels]

## 3. Broker Research & Analyst Reports
[Analyst ratings, price targets, research highlights]

## Conclusion
[Summary and key takeaways]
```
