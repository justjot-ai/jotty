# PMI Portfolio

Portfolio management via PlanMyInvesting API.

## Description

View portfolio holdings and P&L summary across all connected brokers.

## Type
derived

## Base Skills
- pmi-market-data

## Capabilities
- finance
- data-fetch

## Use When
User wants to check portfolio holdings or P&L summary

## Tools

### get_portfolio_tool
Get current portfolio holdings with live prices.

**Parameters:**
- `broker` (str, optional): Filter by broker name
- `include_closed` (bool, optional): Include closed positions

**Returns:**
- `holdings` (list), `total_value`, `total_pnl`, `count`

### get_pnl_summary_tool
Get P&L summary across all portfolios.

**Parameters:**
- `period` (str, optional): Period (today, week, month, year, all)

**Returns:**
- `realized_pnl`, `unrealized_pnl`, `total_pnl`, `day_pnl`


## Triggers
- "pmi portfolio"

## Category
financial-analysis
