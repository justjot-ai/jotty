# Database Tools - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`connect_database_tool`](#connect_database_tool) | Connect to a database and verify connectivity. |
| [`query_database_tool`](#query_database_tool) | Execute a SQL query and return results. |
| [`list_tables_tool`](#list_tables_tool) | List all tables in the database. |
| [`describe_table_tool`](#describe_table_tool) | Get detailed schema information for a table. |
| [`natural_language_query_tool`](#natural_language_query_tool) | Convert natural language to SQL and execute. |
| [`insert_data_tool`](#insert_data_tool) | Safely insert data into a table. |
| [`get_database_info_tool`](#get_database_info_tool) | Get database metadata and statistics. |

---

## `connect_database_tool`

Connect to a database and verify connectivity.

**Parameters:**

- **db_type** (`str, required`): Database type (postgresql, mysql, sqlite, mssql, oracle)
- **host** (`str, optional`): Database host (default: localhost)
- **port** (`int, optional`): Database port
- **database** (`str, required`): Database name or file path for SQLite
- **user** (`str, optional`): Username
- **password** (`str, optional`): Password
- **connection_string** (`str, optional`): Full connection URL (overrides other params)

**Returns:** Dictionary with success status and connection info

---

## `query_database_tool`

Execute a SQL query and return results.

**Parameters:**

- **sql** (`str, required`): SQL query to execute
- **params** (`dict, optional`): Query parameters for safe substitution
- **connection_string** (`str, optional`): Database connection URL db_type, host, port, database, user, password: Connection params
- **limit** (`int, optional`): Maximum rows to return (default: 100)

**Returns:** Dictionary with query results

---

## `list_tables_tool`

List all tables in the database.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with list of tables

---

## `describe_table_tool`

Get detailed schema information for a table.

**Parameters:**

- **table** (`str, required`): Table name Connection parameters

**Returns:** Dictionary with table schema

---

## `natural_language_query_tool`

Convert natural language to SQL and execute.

**Parameters:**

- **question** (`str, required`): Natural language question Connection parameters
- **execute** (`bool, optional`): Execute the generated SQL (default: True)

**Returns:** Dictionary with generated SQL and optional results

---

## `insert_data_tool`

Safely insert data into a table.

**Parameters:**

- **table** (`str, required`): Table name
- **data** (`dict or list, required`): Data to insert (single row dict or list of dicts) Connection parameters

**Returns:** Dictionary with insert status

---

## `get_database_info_tool`

Get database metadata and statistics.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with database info
