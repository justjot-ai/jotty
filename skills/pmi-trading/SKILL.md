---
name: pmi-trading
description: "Place market/limit/SL orders, smart orders with bracket logic, exit positions, cancel orders, and view order history across connected brokers. Use when the user wants to trade, trading, buy."
---

# PMI Trading

Order placement and position management via PlanMyInvesting API.

## Description

Place market/limit/SL orders, smart orders with bracket logic, exit positions, cancel orders, and view order history across connected brokers.

## Type
base

## Capabilities
- finance

## Use When
User wants to place orders, buy/sell stocks, exit positions, cancel orders, or check order history

## Tools

### place_order_tool
Place a trading order.

**Parameters:**
- `symbol` (str, required): Stock symbol
- `quantity` (int, required): Number of shares
- `order_type` (str, required): MARKET, LIMIT, SL, SL-M
- `transaction_type` (str, required): BUY or SELL
- `price` (float, optional): Price for LIMIT/SL orders
- `trigger_price` (float, optional): Trigger for SL orders
- `broker` (str, optional): Broker to use
- `product` (str, optional): CNC, MIS, NRML

### place_smart_order_tool
Place a smart order with automatic bracket logic.

**Parameters:**
- `symbol` (str, required): Stock symbol
- `quantity` (int, required): Number of shares
- `target_percent` (float, optional): Target profit %
- `stoploss_percent` (float, optional): Stop loss %

### exit_position_tool
Exit an open position.

**Parameters:**
- `symbol` (str, required): Symbol to exit
- `quantity` (int, optional): Partial exit quantity

### cancel_order_tool
Cancel a pending order.

**Parameters:**
- `order_id` (str, required): Order ID to cancel

### get_orders_tool
Get order history with filters.

**Parameters:**
- `status` (str, optional): Filter (open, completed, cancelled)
- `broker` (str, optional): Filter by broker
- `symbol` (str, optional): Filter by symbol

## Triggers
- "pmi trading"
- "trade"
- "trading"
- "buy"
- "sell"

## Category
financial-analysis
