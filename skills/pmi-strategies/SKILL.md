---
name: pmi-strategies
description: "List, execute, and monitor trading strategies. Generate trading signals from active strategies."
---

# PMI Strategies

Trading strategy management via PlanMyInvesting API.

## Description

List, execute, and monitor trading strategies. Generate trading signals from active strategies.

## Type
derived

## Base Skills
- pmi-market-data

## Capabilities
- finance
- analyze

## Use When
User wants to list strategies, run a trading strategy, check strategy performance, or generate trading signals

## Tools

### list_strategies_tool
List all available trading strategies.

**Parameters:**
- `active_only` (bool, optional): Only show active strategies

### run_strategy_tool
Execute a trading strategy.

**Parameters:**
- `strategy_id` (str, required): Strategy ID to run
- `dry_run` (bool, optional): Simulate without orders (default True)
- `symbols` (list[str], optional): Override symbol universe

### get_strategy_status_tool
Get strategy status and performance.

**Parameters:**
- `strategy_id` (str, required): Strategy ID

### generate_signals_tool
Generate trading signals from active strategies.

**Parameters:**
- `symbols` (list[str], optional): Filter for specific symbols
- `strategy_id` (str, optional): Specific strategy only

## Triggers
- "pmi strategies"

## Category
financial-analysis
