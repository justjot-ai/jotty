# Screener.in → Analysis → PDF → Telegram Pipeline Skill

## Description
Complete pipeline that fetches financial data from screener.in, analyzes it with Claude LLM, generates PDF, and sends to Telegram.

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
- `telegram_chat_id` (str, optional): Telegram chat ID (overrides TELEGRAM_CHAT_ID env var)
- `telegram_token` (str, optional): Telegram bot token (overrides TELEGRAM_TOKEN env var)
- `output_dir` (str, optional): Output directory for PDF, default: './output'
- `title` (str, optional): Custom PDF title
- `use_proxy` (bool, optional): Use proxy for screener.in, default: True

**Note:** Telegram credentials are automatically read from environment variables (`TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`) if not provided. The skill will attempt to send to Telegram if credentials are available.

**Returns:**
- `success` (bool): Whether workflow succeeded
- `companies_analyzed` (list): List of companies processed
- `pdf_path` (str): Path to generated PDF
- `telegram_sent` (bool): Whether sent to Telegram
- `analysis` (str): Generated analysis text
- `error` (str, optional): Error message if failed
