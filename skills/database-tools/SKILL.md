# Database Tools

## Description
Multi-database skill using SQLAlchemy with connection pooling, safe parameterized queries, and natural language query support.


## Type
base

## Supported Databases
- PostgreSQL
- MySQL
- SQLite
- SQL Server (MSSQL)
- Oracle

## Tools

### connect_database_tool
Connect to a database and verify connectivity.

**Parameters:**
- `db_type` (str, required): Database type (postgresql, mysql, sqlite, mssql, oracle)
- `host` (str, optional): Database host (default: localhost)
- `port` (int, optional): Database port
- `database` (str, required): Database name or file path for SQLite
- `user` (str, optional): Username
- `password` (str, optional): Password
- `connection_string` (str, optional): Full connection URL (overrides other params)

### query_database_tool
Execute a SQL query and return results.

**Parameters:**
- `sql` (str, required): SQL query to execute
- `params` (dict, optional): Query parameters for safe substitution
- `limit` (int, optional): Maximum rows to return (default: 100)
- Connection parameters (same as connect_database_tool)

### list_tables_tool
List all tables in the database.

**Parameters:**
- Connection parameters

### describe_table_tool
Get detailed schema information for a table.

**Parameters:**
- `table` (str, required): Table name
- Connection parameters

### natural_language_query_tool
Convert natural language to SQL and execute.

**Parameters:**
- `question` (str, required): Natural language question
- `execute` (bool, optional): Execute the generated SQL (default: True)
- Connection parameters

### insert_data_tool
Safely insert data into a table.

**Parameters:**
- `table` (str, required): Table name
- `data` (dict or list, required): Data to insert
- Connection parameters

### get_database_info_tool
Get database metadata and statistics.

**Parameters:**
- Connection parameters

## Usage Examples

```python
# Connect to PostgreSQL
result = connect_database_tool({
    'db_type': 'postgresql',
    'host': 'localhost',
    'database': 'mydb',
    'user': 'postgres',
    'password': 'secret'
})

# Or use connection string
result = connect_database_tool({
    'connection_string': 'postgresql://user:pass@localhost:5432/mydb'
})

# List tables
result = list_tables_tool({
    'db_type': 'postgresql',
    'database': 'mydb',
    'user': 'postgres',
    'password': 'secret'
})

# Query with parameters (safe from SQL injection)
result = query_database_tool({
    'sql': 'SELECT * FROM users WHERE status = :status',
    'params': {'status': 'active'},
    'db_type': 'postgresql',
    'database': 'mydb',
    'user': 'postgres',
    'password': 'secret'
})

# Natural language query
result = natural_language_query_tool({
    'question': 'Show me all users who signed up last month',
    'db_type': 'postgresql',
    'database': 'mydb',
    'user': 'postgres',
    'password': 'secret'
})

# SQLite (just file path)
result = query_database_tool({
    'sql': 'SELECT * FROM products',
    'db_type': 'sqlite',
    'database': '/path/to/database.db'
})

# Insert data
result = insert_data_tool({
    'table': 'users',
    'data': {'name': 'John', 'email': 'john@example.com'},
    'db_type': 'postgresql',
    'database': 'mydb',
    'user': 'postgres',
    'password': 'secret'
})
```

## Features
- **Connection Pooling**: Reuses connections for better performance
- **Safe Queries**: Parameterized queries prevent SQL injection
- **Natural Language**: Ask questions in plain English
- **Schema Introspection**: Explore database structure
- **Multi-Database**: Same API for all supported databases

## Required Packages
```bash
pip install sqlalchemy

# For specific databases:
pip install psycopg2-binary  # PostgreSQL
pip install pymysql          # MySQL
pip install pymssql          # SQL Server
pip install cx_Oracle        # Oracle
```
