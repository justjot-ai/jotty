---
name: pmi-watchlist
description: "Create, manage, and monitor stock watchlists. Add/remove symbols and refresh live prices."
---

# PMI Watchlist

Watchlist management via PlanMyInvesting API.

## Description

Create, manage, and monitor stock watchlists. Add/remove symbols and refresh live prices.

## Type
derived

## Base Skills
- pmi-market-data

## Capabilities
- finance
- data-fetch

## Use When
User wants to create watchlists, add/remove symbols, or refresh watchlist prices

## Tools

### list_watchlists_tool
List all watchlists with their symbols.

### create_watchlist_tool
Create a new watchlist.

**Parameters:**
- `name` (str, required): Watchlist name
- `symbols` (list[str], optional): Initial symbols

### add_to_watchlist_tool
Add a symbol to a watchlist.

**Parameters:**
- `watchlist_id` (str, required): Watchlist ID
- `symbol` (str, required): Symbol to add

### remove_from_watchlist_tool
Remove a symbol from a watchlist.

**Parameters:**
- `watchlist_id` (str, required): Watchlist ID
- `symbol` (str, required): Symbol to remove

### refresh_watchlist_tool
Refresh live prices for all symbols in a watchlist.

**Parameters:**
- `watchlist_id` (str, required): Watchlist ID

## Triggers
- "pmi watchlist"

## Category
financial-analysis
