# PMI Watchlist - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`list_watchlists_tool`](#list_watchlists_tool) | List all watchlists with their symbols. |
| [`create_watchlist_tool`](#create_watchlist_tool) | Create a new watchlist. |
| [`add_to_watchlist_tool`](#add_to_watchlist_tool) | Add a symbol to an existing watchlist. |
| [`remove_from_watchlist_tool`](#remove_from_watchlist_tool) | Remove a symbol from a watchlist. |
| [`refresh_watchlist_tool`](#refresh_watchlist_tool) | Refresh live prices for all symbols in a watchlist. |

---

## `list_watchlists_tool`

List all watchlists with their symbols.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - watchlists (list): Watchlist objects with id, name, symbols - count (int): Number of watchlists

---

## `create_watchlist_tool`

Create a new watchlist.

**Parameters:**

- **name** (`str, required`): Watchlist name
- **symbols** (`list, optional`): Initial symbols to add

**Returns:** Dictionary with: - watchlist_id (str): ID of created watchlist - name (str): Watchlist name - symbols (list): Symbols added to watchlist

---

## `add_to_watchlist_tool`

Add a symbol to an existing watchlist.

**Parameters:**

- **watchlist_id** (`str, required`): Watchlist ID
- **symbol** (`str, required`): Symbol to add

**Returns:** Dictionary with: - watchlist_id (str): Watchlist ID - symbol (str): Symbol that was added

---

## `remove_from_watchlist_tool`

Remove a symbol from a watchlist.

**Parameters:**

- **watchlist_id** (`str, required`): Watchlist ID
- **symbol** (`str, required`): Symbol to remove

**Returns:** Dictionary with: - watchlist_id (str): Watchlist ID - symbol (str): Symbol that was removed - removed (bool): Whether removal succeeded

---

## `refresh_watchlist_tool`

Refresh live prices for all symbols in a watchlist.

**Parameters:**

- **watchlist_id** (`str, required`): Watchlist ID to refresh

**Returns:** Dictionary with: - watchlist_id (str): Watchlist ID - symbols (list): Symbol objects with refreshed ltp, change, change_percent - count (int): Number of symbols
