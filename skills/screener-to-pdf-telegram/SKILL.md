# Screener.in → Analysis → PDF → Telegram Pipeline Skill

## Description
Complete pipeline that fetches financial data from screener.in, analyzes it with Claude LLM, generates PDF, and sends to Telegram.


## Type
composite

## Base Skills
- screener-financials
- document-converter
- telegram-sender

## Execution
sequential

## Pipeline Flow
1. **Source**: Fetch financial data from screener.in for company symbol(s)
2. **Processor**: Synthesize and analyze data using Claude CLI LLM
3. **Processor**: Convert analysis to PDF format
4. **Sink**: Send PDF to Telegram

## Tools

### screener_analyze_pdf_telegram_tool
Complete workflow: Screener.in → Analysis → PDF → Telegram

**Parameters:**
- `symbols` (str or list, required): Company symbol(s) - e.g., "RELIANCE" or ["RELIANCE", "TCS"]
- `analysis_type` (str, optional): Analysis type - 'comprehensive', 'quick', 'ratios_only', default: 'comprehensive'
- `send_telegram` (bool, optional): Send to Telegram (default: True)
- `telegram_chat_id` (str, optional): Telegram chat ID (uses TELEGRAM_CHAT_ID env var if not provided)
- `output_dir` (str, optional): Output directory for PDF, default: './output'
- `title` (str, optional): Custom PDF title
- `use_proxy` (bool, optional): Use proxy for screener.in, default: True

**Note:** Uses the same Telegram credentials pattern as other Jotty skills. The `telegram-sender` skill automatically reads `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` from environment variables if not provided as parameters.

**Returns:**
- `success` (bool): Whether workflow succeeded
- `companies_analyzed` (list): List of companies processed
- `pdf_path` (str): Path to generated PDF
- `telegram_sent` (bool): Whether sent to Telegram
- `analysis` (str): Generated analysis text
- `error` (str, optional): Error message if failed
