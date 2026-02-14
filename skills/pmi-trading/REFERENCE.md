# PMI Trading - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`place_order_tool`](#place_order_tool) | Place a trading order. |
| [`place_smart_order_tool`](#place_smart_order_tool) | Place a smart order with automatic price optimization and bracket logic. |
| [`exit_position_tool`](#exit_position_tool) | Exit an open position (sell all holdings of a symbol). |
| [`cancel_order_tool`](#cancel_order_tool) | Cancel a pending order. |
| [`get_orders_tool`](#get_orders_tool) | Get order history with optional filters. |

---

## `place_order_tool`

Place a trading order.

**Parameters:**

- **symbol** (`str, required`): Stock symbol
- **quantity** (`int, required`): Number of shares
- **order_type** (`str, required`): MARKET, LIMIT, SL, SL-M
- **transaction_type** (`str, required`): BUY or SELL
- **price** (`float, optional`): Price for LIMIT/SL orders
- **trigger_price** (`float, optional`): Trigger price for SL orders
- **broker** (`str, optional`): Broker to use
- **product** (`str, optional`): CNC, MIS, NRML (default CNC)

**Returns:** Dictionary with: - order_id (str): Unique order identifier - status (str): Order status (placed, rejected, etc.) - symbol (str): Stock symbol - quantity (int): Number of shares ordered - transaction_type (str): BUY or SELL

---

## `place_smart_order_tool`

Place a smart order with automatic price optimization and bracket logic.

**Parameters:**

- **symbol** (`str, required`): Stock symbol
- **quantity** (`int, required`): Number of shares
- **transaction_type** (`str, optional`): BUY or SELL (default BUY)
- **target_percent** (`float, optional`): Target profit % (default 2.0)
- **stoploss_percent** (`float, optional`): Stop loss % (default 1.0)
- **broker** (`str, optional`): Broker to use

**Returns:** Dictionary with: - order_id (str): Unique order identifier - symbol (str): Stock symbol - entry_price (float): Entry price of the order - target (float): Target price - stoploss (float): Stop loss price

---

## `exit_position_tool`

Exit an open position (sell all holdings of a symbol).

**Parameters:**

- **symbol** (`str, required`): Symbol to exit
- **quantity** (`int, optional`): Partial exit quantity (default: all)
- **broker** (`str, optional`): Broker filter

**Returns:** Dictionary with: - order_id (str): Exit order identifier - symbol (str): Symbol exited - quantity_exited (int): Number of shares sold - exit_price (float): Price at which position was exited

---

## `cancel_order_tool`

Cancel a pending order.

**Parameters:**

- **order_id** (`str, required`): Order ID to cancel
- **broker** (`str, optional`): Broker filter

**Returns:** Dictionary with: - order_id (str): Cancelled order ID - cancelled (bool): Whether cancellation succeeded

---

## `get_orders_tool`

Get order history with optional filters.

**Parameters:**

- **status** (`str, optional`): Filter by status (open, completed, cancelled)
- **broker** (`str, optional`): Filter by broker
- **symbol** (`str, optional`): Filter by symbol
- **limit** (`int, optional`): Max results (default 50)

**Returns:** Dictionary with: - orders (list): Order objects with order_id, symbol, quantity, status, price - count (int): Number of orders returned
