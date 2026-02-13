# PMI Portfolio

Portfolio management via PlanMyInvesting API.

## Description

View portfolio holdings, P&L summary, available cash, and account limits across all connected brokers.

## Type
derived

## Base Skills
- pmi-market-data

## Capabilities
- finance
- data-fetch

## Use When
User wants to check portfolio holdings, P&L, available cash, or account limits

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

### get_available_cash_tool
Get available cash balance across brokers.

**Parameters:**
- `broker` (str, optional): Filter by broker

**Returns:**
- `cash`, `total`

### get_account_limits_tool
Get account limits (margin, exposure, collateral).

**Parameters:**
- `broker` (str, optional): Filter by broker

**Returns:**
- `margin_available`, `margin_used`, `collateral`, `exposure`
