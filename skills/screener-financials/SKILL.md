---
name: screening-financials
description: "Fetches financial data for Indian companies from screener.in. Includes free proxy rotation to avoid blocking and rate limits. Use when the user wants to financial, finance, investment."
---

# Screener.in Financials Skill

## Description
Fetches financial data for Indian companies from screener.in. Includes free proxy rotation to avoid blocking and rate limits.


## Type
derived

## Base Skills
- web-scraper


## Capabilities
- data-fetch

## Features
- Fetch company financials (P&L, Balance Sheet, Cash Flow)
- Free proxy rotation (automatic)
- Rate limiting protection
- User-agent rotation
- Retry logic with exponential backoff
- Multiple data formats (JSON, Markdown, CSV)

## Tools

### get_company_financials_tool
Fetches financial data for a company from screener.in.

**Parameters:**
- `company_name` (str, required): Company name or screener.in company code
- `data_type` (str, optional): Type of data - 'all', 'pl' (P&L), 'balance_sheet', 'cash_flow', 'ratios', default: 'all'
- `period` (str, optional): Period - 'annual', 'quarterly', default: 'annual'
- `format` (str, optional): Output format - 'json', 'markdown', 'csv', default: 'json'
- `use_proxy` (bool, optional): Use proxy rotation, default: True
- `max_retries` (int, optional): Maximum retry attempts, default: 3

**Returns:**
- `success` (bool): Whether fetch succeeded
- `company_name` (str): Company name
- `company_code` (str): Screener.in company code
- `data` (dict): Financial data (format depends on format parameter)
- `period` (str): Data period
- `error` (str, optional): Error message if failed

### search_company_tool
Search for a company on screener.in.

**Parameters:**
- `query` (str, required): Company name or search query
- `max_results` (int, optional): Maximum results, default: 10

**Returns:**
- `success` (bool): Whether search succeeded
- `results` (list): List of matching companies with codes
- `error` (str, optional): Error message if failed

### get_company_ratios_tool
Fetches key financial ratios for a company.

**Parameters:**
- `company_name` (str, required): Company name or code
- `period` (str, optional): Period - 'annual', 'quarterly', default: 'annual'

**Returns:**
- `success` (bool): Whether fetch succeeded
- `ratios` (dict): Key financial ratios
- `error` (str, optional): Error message if failed

## Triggers
- "screener financials"
- "financial"
- "finance"
- "investment"
- "portfolio"

## Category
financial-analysis
