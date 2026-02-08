"""
Database Tools Skill

Multi-database support using SQLAlchemy with connection pooling,
safe parameterized queries, and natural language query support.

Supports: PostgreSQL, MySQL, SQLite, SQL Server, Oracle
"""
import os
import logging
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Connection pool cache
_engines: Dict[str, Any] = {}


def _get_engine(connection_string: str, pool_size: int = 5):
    """
    Get or create a SQLAlchemy engine with connection pooling.

    Args:
        connection_string: Database URL
        pool_size: Connection pool size

    Returns:
        SQLAlchemy engine
    """
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool
    except ImportError:
        raise ImportError("sqlalchemy not installed. Install with: pip install sqlalchemy")

    if connection_string not in _engines:
        _engines[connection_string] = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            echo=False
        )

    return _engines[connection_string]


def _build_connection_string(
    db_type: str,
    host: str = "localhost",
    port: Optional[int] = None,
    database: str = "",
    user: str = "",
    password: str = "",
    **kwargs
) -> str:
    """
    Build a SQLAlchemy connection string.

    Args:
        db_type: Database type (postgresql, mysql, sqlite, mssql, oracle)
        host: Database host
        port: Database port (uses default if not specified)
        database: Database name
        user: Username
        password: Password
        **kwargs: Additional connection parameters

    Returns:
        SQLAlchemy connection URL
    """
    # Default ports
    default_ports = {
        "postgresql": 5432,
        "mysql": 3306,
        "mssql": 1433,
        "oracle": 1521,
    }

    # Driver mappings
    drivers = {
        "postgresql": "postgresql+psycopg2",
        "mysql": "mysql+pymysql",
        "sqlite": "sqlite",
        "mssql": "mssql+pymssql",
        "oracle": "oracle+oracledb",
    }

    driver = drivers.get(db_type)
    if not driver:
        raise ValueError(f"Unsupported database type: {db_type}. Supported: {list(drivers.keys())}")

    # SQLite is special - just file path
    if db_type == "sqlite":
        return f"sqlite:///{database}"

    port = port or default_ports.get(db_type)

    # URL encode password to handle special characters
    encoded_password = quote_plus(password) if password else ""

    # Build URL
    if user and encoded_password:
        url = f"{driver}://{user}:{encoded_password}@{host}:{port}/{database}"
    elif user:
        url = f"{driver}://{user}@{host}:{port}/{database}"
    else:
        url = f"{driver}://{host}:{port}/{database}"

    return url


def connect_database_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Connect to a database and verify connectivity.

    Args:
        params: Dictionary containing:
            - db_type (str, required): Database type (postgresql, mysql, sqlite, mssql, oracle)
            - host (str, optional): Database host (default: localhost)
            - port (int, optional): Database port
            - database (str, required): Database name or file path for SQLite
            - user (str, optional): Username
            - password (str, optional): Password
            - connection_string (str, optional): Full connection URL (overrides other params)

    Returns:
        Dictionary with success status and connection info
    """
    try:
        from sqlalchemy import text

        # Use provided connection string or build one
        conn_string = params.get('connection_string')

        if not conn_string:
            db_type = params.get('db_type')
            if not db_type:
                return {'success': False, 'error': 'db_type or connection_string required'}

            database = params.get('database')
            if not database:
                return {'success': False, 'error': 'database parameter required'}

            conn_string = _build_connection_string(
                db_type=db_type,
                host=params.get('host', 'localhost'),
                port=params.get('port'),
                database=database,
                user=params.get('user', ''),
                password=params.get('password', '')
            )

        # Get engine and test connection
        engine = _get_engine(conn_string)

        with engine.connect() as conn:
            # Test with simple query
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        # Store connection string for later use (masked)
        masked = conn_string.split('@')[-1] if '@' in conn_string else conn_string

        return {
            'success': True,
            'message': 'Connected successfully',
            'connection': masked,
            'pool_size': engine.pool.size()
        }

    except ImportError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def query_database_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a SQL query and return results.

    Args:
        params: Dictionary containing:
            - sql (str, required): SQL query to execute
            - params (dict, optional): Query parameters for safe substitution
            - connection_string (str, optional): Database connection URL
            - db_type, host, port, database, user, password: Connection params
            - limit (int, optional): Maximum rows to return (default: 100)

    Returns:
        Dictionary with query results
    """
    try:
        from sqlalchemy import text

        sql = params.get('sql')
        if not sql:
            return {'success': False, 'error': 'sql parameter required'}

        query_params = params.get('params', {})
        limit = params.get('limit', 100)

        # Get connection string
        conn_string = params.get('connection_string')
        if not conn_string:
            db_type = params.get('db_type')
            database = params.get('database')
            if not db_type or not database:
                return {'success': False, 'error': 'connection_string or (db_type + database) required'}

            conn_string = _build_connection_string(
                db_type=db_type,
                host=params.get('host', 'localhost'),
                port=params.get('port'),
                database=database,
                user=params.get('user', ''),
                password=params.get('password', '')
            )

        engine = _get_engine(conn_string)

        with engine.connect() as conn:
            result = conn.execute(text(sql), query_params)

            # Check if query returns rows
            if result.returns_rows:
                columns = list(result.keys())
                rows = []
                for i, row in enumerate(result):
                    if i >= limit:
                        break
                    rows.append(dict(zip(columns, row)))

                return {
                    'success': True,
                    'columns': columns,
                    'rows': rows,
                    'row_count': len(rows),
                    'truncated': len(rows) >= limit
                }
            else:
                # INSERT/UPDATE/DELETE
                conn.commit()
                return {
                    'success': True,
                    'affected_rows': result.rowcount,
                    'message': f'{result.rowcount} rows affected'
                }

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def list_tables_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    List all tables in the database.

    Args:
        params: Dictionary containing connection parameters

    Returns:
        Dictionary with list of tables
    """
    try:
        from sqlalchemy import inspect

        # Get connection string
        conn_string = params.get('connection_string')
        if not conn_string:
            db_type = params.get('db_type')
            database = params.get('database')
            if not db_type or not database:
                return {'success': False, 'error': 'connection_string or (db_type + database) required'}

            conn_string = _build_connection_string(
                db_type=db_type,
                host=params.get('host', 'localhost'),
                port=params.get('port'),
                database=database,
                user=params.get('user', ''),
                password=params.get('password', '')
            )

        engine = _get_engine(conn_string)
        inspector = inspect(engine)

        tables = inspector.get_table_names()
        views = inspector.get_view_names()

        return {
            'success': True,
            'tables': tables,
            'views': views,
            'total_tables': len(tables),
            'total_views': len(views)
        }

    except Exception as e:
        logger.error(f"List tables error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def describe_table_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed schema information for a table.

    Args:
        params: Dictionary containing:
            - table (str, required): Table name
            - Connection parameters

    Returns:
        Dictionary with table schema
    """
    try:
        from sqlalchemy import inspect

        table_name = params.get('table')
        if not table_name:
            return {'success': False, 'error': 'table parameter required'}

        # Get connection string
        conn_string = params.get('connection_string')
        if not conn_string:
            db_type = params.get('db_type')
            database = params.get('database')
            if not db_type or not database:
                return {'success': False, 'error': 'connection_string or (db_type + database) required'}

            conn_string = _build_connection_string(
                db_type=db_type,
                host=params.get('host', 'localhost'),
                port=params.get('port'),
                database=database,
                user=params.get('user', ''),
                password=params.get('password', '')
            )

        engine = _get_engine(conn_string)
        inspector = inspect(engine)

        # Get columns
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append({
                'name': col['name'],
                'type': str(col['type']),
                'nullable': col.get('nullable', True),
                'default': str(col.get('default', '')) if col.get('default') else None,
                'primary_key': col.get('primary_key', False)
            })

        # Get primary keys
        pk = inspector.get_pk_constraint(table_name)
        primary_keys = pk.get('constrained_columns', []) if pk else []

        # Get foreign keys
        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            foreign_keys.append({
                'columns': fk['constrained_columns'],
                'references_table': fk['referred_table'],
                'references_columns': fk['referred_columns']
            })

        # Get indexes
        indexes = []
        for idx in inspector.get_indexes(table_name):
            indexes.append({
                'name': idx['name'],
                'columns': idx['column_names'],
                'unique': idx.get('unique', False)
            })

        return {
            'success': True,
            'table': table_name,
            'columns': columns,
            'primary_keys': primary_keys,
            'foreign_keys': foreign_keys,
            'indexes': indexes,
            'column_count': len(columns)
        }

    except Exception as e:
        logger.error(f"Describe table error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def natural_language_query_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert natural language to SQL and execute.

    Args:
        params: Dictionary containing:
            - question (str, required): Natural language question
            - Connection parameters
            - execute (bool, optional): Execute the generated SQL (default: True)

    Returns:
        Dictionary with generated SQL and optional results
    """
    try:
        from Jotty.core.llm import generate

        question = params.get('question')
        if not question:
            return {'success': False, 'error': 'question parameter required'}

        execute = params.get('execute', True)

        # Get connection string
        conn_string = params.get('connection_string')
        if not conn_string:
            db_type = params.get('db_type')
            database = params.get('database')
            if not db_type or not database:
                return {'success': False, 'error': 'connection_string or (db_type + database) required'}

            conn_string = _build_connection_string(
                db_type=db_type,
                host=params.get('host', 'localhost'),
                port=params.get('port'),
                database=database,
                user=params.get('user', ''),
                password=params.get('password', '')
            )
        else:
            # Infer db_type from connection string
            if 'postgresql' in conn_string:
                db_type = 'postgresql'
            elif 'mysql' in conn_string:
                db_type = 'mysql'
            elif 'sqlite' in conn_string:
                db_type = 'sqlite'
            elif 'mssql' in conn_string:
                db_type = 'mssql'
            else:
                db_type = 'unknown'

        # Get schema for context
        tables_result = list_tables_tool({
            'connection_string': conn_string,
            'db_type': params.get('db_type'),
            'database': params.get('database'),
            'host': params.get('host'),
            'port': params.get('port'),
            'user': params.get('user'),
            'password': params.get('password')
        })

        if not tables_result.get('success'):
            return tables_result

        # Get schema for each table (limit to first 10 tables)
        schema_info = []
        for table in tables_result.get('tables', [])[:10]:
            desc = describe_table_tool({
                'table': table,
                'connection_string': conn_string,
                'db_type': params.get('db_type'),
                'database': params.get('database'),
                'host': params.get('host'),
                'port': params.get('port'),
                'user': params.get('user'),
                'password': params.get('password')
            })
            if desc.get('success'):
                cols = [f"{c['name']} ({c['type']})" for c in desc.get('columns', [])]
                schema_info.append(f"Table: {table}\n  Columns: {', '.join(cols)}")

        schema_context = '\n'.join(schema_info)

        # Generate SQL using LLM
        prompt = f"""Convert this natural language question to a SQL query for {db_type}:

Question: {question}

Database Schema:
{schema_context}

Rules:
- Return ONLY the SQL query, no explanation
- Use proper {db_type} syntax
- Include appropriate WHERE clauses
- Limit results to 100 rows unless specified
- Use safe practices (no DROP, DELETE without WHERE, etc.)

SQL:"""

        llm_response = generate(prompt, model='sonnet', timeout=60)

        if not llm_response.success:
            return {'success': False, 'error': f'LLM error: {llm_response.error}'}

        # Extract SQL from response
        sql = llm_response.text.strip()

        # Clean up SQL (remove markdown code blocks if present)
        if sql.startswith('```'):
            sql = sql.split('```')[1]
            if sql.startswith('sql'):
                sql = sql[3:]
        sql = sql.strip()

        result = {
            'success': True,
            'question': question,
            'generated_sql': sql,
            'db_type': db_type
        }

        # Execute if requested
        if execute:
            query_result = query_database_tool({
                'sql': sql,
                'connection_string': conn_string,
                'db_type': params.get('db_type'),
                'database': params.get('database'),
                'host': params.get('host'),
                'port': params.get('port'),
                'user': params.get('user'),
                'password': params.get('password'),
                'limit': params.get('limit', 100)
            })

            result['executed'] = True
            result['query_result'] = query_result
        else:
            result['executed'] = False

        return result

    except Exception as e:
        logger.error(f"Natural language query error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def insert_data_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely insert data into a table.

    Args:
        params: Dictionary containing:
            - table (str, required): Table name
            - data (dict or list, required): Data to insert (single row dict or list of dicts)
            - Connection parameters

    Returns:
        Dictionary with insert status
    """
    try:
        from sqlalchemy import text

        table = params.get('table')
        data = params.get('data')

        if not table:
            return {'success': False, 'error': 'table parameter required'}
        if not data:
            return {'success': False, 'error': 'data parameter required'}

        # Normalize to list
        if isinstance(data, dict):
            data = [data]

        if not data or not isinstance(data[0], dict):
            return {'success': False, 'error': 'data must be a dict or list of dicts'}

        # Get connection string
        conn_string = params.get('connection_string')
        if not conn_string:
            db_type = params.get('db_type')
            database = params.get('database')
            if not db_type or not database:
                return {'success': False, 'error': 'connection_string or (db_type + database) required'}

            conn_string = _build_connection_string(
                db_type=db_type,
                host=params.get('host', 'localhost'),
                port=params.get('port'),
                database=database,
                user=params.get('user', ''),
                password=params.get('password', '')
            )

        engine = _get_engine(conn_string)

        # Build parameterized INSERT
        columns = list(data[0].keys())
        placeholders = ', '.join([f':{col}' for col in columns])
        column_names = ', '.join(columns)

        sql = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"

        inserted = 0
        with engine.connect() as conn:
            for row in data:
                conn.execute(text(sql), row)
                inserted += 1
            conn.commit()

        return {
            'success': True,
            'table': table,
            'inserted_rows': inserted,
            'message': f'Inserted {inserted} rows into {table}'
        }

    except Exception as e:
        logger.error(f"Insert data error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


def get_database_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get database metadata and statistics.

    Args:
        params: Dictionary containing connection parameters

    Returns:
        Dictionary with database info
    """
    try:
        from sqlalchemy import inspect, text

        # Get connection string
        conn_string = params.get('connection_string')
        if not conn_string:
            db_type = params.get('db_type')
            database = params.get('database')
            if not db_type or not database:
                return {'success': False, 'error': 'connection_string or (db_type + database) required'}

            conn_string = _build_connection_string(
                db_type=db_type,
                host=params.get('host', 'localhost'),
                port=params.get('port'),
                database=database,
                user=params.get('user', ''),
                password=params.get('password', '')
            )

        engine = _get_engine(conn_string)
        inspector = inspect(engine)

        # Get basic info
        tables = inspector.get_table_names()
        views = inspector.get_view_names()

        # Get table sizes if possible
        table_info = []
        for table in tables[:20]:  # Limit to 20 tables
            try:
                cols = inspector.get_columns(table)
                table_info.append({
                    'name': table,
                    'column_count': len(cols)
                })
            except:
                table_info.append({'name': table, 'column_count': 'unknown'})

        return {
            'success': True,
            'dialect': engine.dialect.name,
            'driver': engine.driver,
            'database': engine.url.database,
            'tables': table_info,
            'table_count': len(tables),
            'view_count': len(views),
            'pool_size': engine.pool.size()
        }

    except Exception as e:
        logger.error(f"Get database info error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


__all__ = [
    'connect_database_tool',
    'query_database_tool',
    'list_tables_tool',
    'describe_table_tool',
    'natural_language_query_tool',
    'insert_data_tool',
    'get_database_info_tool',
]
