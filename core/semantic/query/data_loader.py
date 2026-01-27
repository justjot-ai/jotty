"""
High-Performance Data Loader

Uses ConnectorX for fast database-to-DataFrame loading.
Supports multiple output formats: Pandas, Polars, PyArrow.

Performance benefits over traditional methods:
- 13-21x faster than Pandas read_sql
- 3x less memory consumption
- Zero-copy architecture (Rust-based)
- Parallel query execution with partitioning

Supported databases:
- PostgreSQL, MySQL, MariaDB, SQLite
- SQL Server, Oracle, BigQuery
- Redshift, ClickHouse, Trino
"""
from typing import Dict, Any, Optional, List, Union, Literal
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported DataFrame output formats."""
    PANDAS = "pandas"
    POLARS = "polars"
    ARROW = "arrow"
    MODIN = "modin"
    DASK = "dask"


class DatabaseProtocol(Enum):
    """Database connection protocols for ConnectorX."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MARIADB = "mariadb"
    SQLITE = "sqlite"
    MSSQL = "mssql"
    ORACLE = "oracle"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    CLICKHOUSE = "clickhouse"
    TRINO = "trino"


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.

    Provides a unified interface for loading query results
    into various DataFrame formats.
    """

    @abstractmethod
    def load(
        self,
        query: str,
        output_format: OutputFormat = OutputFormat.PANDAS,
        **kwargs
    ) -> Any:
        """
        Load query results into a DataFrame.

        Args:
            query: SQL query to execute
            output_format: Desired output format
            **kwargs: Additional loader-specific options

        Returns:
            DataFrame in the requested format
        """
        pass

    @abstractmethod
    def get_connection_string(self) -> str:
        """Get the database connection string."""
        pass


class ConnectorXLoader(BaseDataLoader):
    """
    High-performance data loader using ConnectorX.

    Features:
    - Zero-copy data transfer from database to DataFrame
    - Parallel query execution with automatic partitioning
    - Support for multiple output formats
    - Connection string caching

    Example:
        loader = ConnectorXLoader(
            db_type='postgresql',
            host='localhost',
            database='mydb',
            user='postgres',
            password='secret'
        )

        # Fast DataFrame loading
        df = loader.load("SELECT * FROM large_table")

        # Parallel loading with partitioning
        df = loader.load(
            "SELECT * FROM large_table",
            partition_on="id",
            partition_num=4
        )

        # Load as Polars DataFrame (even faster)
        df = loader.load(
            "SELECT * FROM large_table",
            output_format=OutputFormat.POLARS
        )
    """

    # ConnectorX protocol mapping
    PROTOCOL_MAP = {
        'postgresql': 'postgresql',
        'postgres': 'postgresql',
        'pg': 'postgresql',
        'mysql': 'mysql',
        'mariadb': 'mariadb',
        'sqlite': 'sqlite',
        'mssql': 'mssql',
        'sqlserver': 'mssql',
        'oracle': 'oracle',
        'bigquery': 'bigquery',
        'redshift': 'redshift',
        'clickhouse': 'clickhouse',
        'trino': 'trino',
    }

    # Default ports for databases
    DEFAULT_PORTS = {
        'postgresql': 5432,
        'mysql': 3306,
        'mariadb': 3306,
        'mssql': 1433,
        'oracle': 1521,
        'clickhouse': 9000,
        'trino': 8080,
        'redshift': 5439,
    }

    def __init__(
        self,
        db_type: str = None,
        host: str = "localhost",
        port: int = None,
        database: str = "",
        user: str = "",
        password: str = "",
        connection_string: str = None,
        **kwargs
    ):
        """
        Initialize ConnectorX loader.

        Args:
            db_type: Database type (postgresql, mysql, sqlite, etc.)
            host: Database host
            port: Database port (uses default if not specified)
            database: Database name
            user: Username
            password: Password
            connection_string: Full connection URL (overrides other params)
            **kwargs: Additional connection parameters
        """
        self.db_type = self._normalize_db_type(db_type)
        self.host = host
        self.port = port or self.DEFAULT_PORTS.get(self.db_type)
        self.database = database
        self.user = user
        self.password = password
        self._connection_string = connection_string
        self._extra_params = kwargs

        # Validate ConnectorX is available
        self._validate_connectorx()

    def _normalize_db_type(self, db_type: str) -> str:
        """Normalize database type to ConnectorX protocol."""
        if not db_type:
            return 'postgresql'
        db_type = db_type.lower()
        return self.PROTOCOL_MAP.get(db_type, db_type)

    def _validate_connectorx(self):
        """Validate ConnectorX is installed."""
        try:
            import connectorx
            self._cx = connectorx
        except ImportError:
            raise ImportError(
                "ConnectorX is not installed. Install with: pip install connectorx"
            )

    def get_connection_string(self) -> str:
        """
        Build ConnectorX-compatible connection string.

        Returns:
            Connection string in format: protocol://user:pass@host:port/database
        """
        if self._connection_string:
            return self._connection_string

        # Build connection string based on database type
        if self.db_type == 'sqlite':
            return f"sqlite://{self.database}"

        # URL encode password for special characters
        from urllib.parse import quote_plus
        encoded_password = quote_plus(self.password) if self.password else ''

        # Build standard connection string
        auth = f"{self.user}:{encoded_password}@" if self.user else ""
        port_str = f":{self.port}" if self.port else ""

        conn_str = f"{self.db_type}://{auth}{self.host}{port_str}/{self.database}"

        # Add extra parameters as query string
        if self._extra_params:
            params = "&".join(f"{k}={v}" for k, v in self._extra_params.items())
            conn_str += f"?{params}"

        return conn_str

    def load(
        self,
        query: str,
        output_format: Union[OutputFormat, str] = OutputFormat.PANDAS,
        partition_on: str = None,
        partition_num: int = None,
        partition_range: tuple = None,
        protocol: str = None,
        return_type: str = None,
        **kwargs
    ) -> Any:
        """
        Load query results using ConnectorX.

        Args:
            query: SQL query to execute
            output_format: Output format (pandas, polars, arrow, modin, dask)
            partition_on: Column to partition on for parallel loading
            partition_num: Number of partitions (enables parallel loading)
            partition_range: Tuple of (min, max) for partition range
            protocol: Override connection protocol
            return_type: Deprecated, use output_format instead
            **kwargs: Additional ConnectorX options

        Returns:
            DataFrame in the requested format
        """
        # Handle string output format
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format.lower())

        # Backwards compatibility
        if return_type:
            output_format = OutputFormat(return_type.lower())

        conn_str = self.get_connection_string()

        # Build ConnectorX parameters
        cx_params = {
            'conn': conn_str,
            'query': query,
            'return_type': self._get_cx_return_type(output_format),
        }

        # Add partitioning options for parallel loading
        if partition_on:
            cx_params['partition_on'] = partition_on
        if partition_num:
            cx_params['partition_num'] = partition_num
        if partition_range:
            cx_params['partition_range'] = partition_range
        if protocol:
            cx_params['protocol'] = protocol

        # Add any extra kwargs
        cx_params.update(kwargs)

        try:
            logger.debug(f"ConnectorX loading with params: {list(cx_params.keys())}")
            result = self._cx.read_sql(**cx_params)
            logger.debug(f"Loaded {len(result) if hasattr(result, '__len__') else 'N/A'} rows")
            return result

        except Exception as e:
            logger.error(f"ConnectorX load failed: {e}")
            raise

    def _get_cx_return_type(self, output_format: OutputFormat) -> str:
        """Map OutputFormat to ConnectorX return_type."""
        mapping = {
            OutputFormat.PANDAS: 'pandas',
            OutputFormat.POLARS: 'polars',
            OutputFormat.ARROW: 'arrow',
            OutputFormat.MODIN: 'modin',
            OutputFormat.DASK: 'dask',
        }
        return mapping.get(output_format, 'pandas')

    def load_parallel(
        self,
        query: str,
        partition_column: str,
        num_partitions: int = 4,
        output_format: OutputFormat = OutputFormat.PANDAS,
        **kwargs
    ) -> Any:
        """
        Load query results with automatic parallel partitioning.

        Best for large tables with an integer primary key or indexed column.

        Args:
            query: SQL query to execute
            partition_column: Column to partition on (should be indexed)
            num_partitions: Number of parallel partitions (default: 4)
            output_format: Output format
            **kwargs: Additional options

        Returns:
            DataFrame with combined results from all partitions
        """
        return self.load(
            query=query,
            output_format=output_format,
            partition_on=partition_column,
            partition_num=num_partitions,
            **kwargs
        )

    def test_connection(self) -> Dict[str, Any]:
        """
        Test database connection.

        Returns:
            Dictionary with connection status and info
        """
        try:
            # Try a simple query
            result = self.load("SELECT 1 as test", output_format=OutputFormat.PANDAS)
            return {
                "success": True,
                "message": "Connection successful",
                "connection_string": self.get_connection_string().replace(
                    self.password, "***" if self.password else ""
                ),
                "db_type": self.db_type,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "db_type": self.db_type,
            }


class DataLoaderFactory:
    """
    Factory for creating appropriate data loader based on database type.
    """

    # Databases supported by ConnectorX
    CONNECTORX_SUPPORTED = {
        'postgresql', 'postgres', 'pg',
        'mysql', 'mariadb',
        'sqlite',
        'mssql', 'sqlserver',
        'oracle',
        'bigquery',
        'redshift',
        'clickhouse',
        'trino',
    }

    @classmethod
    def create(
        cls,
        db_type: str,
        use_connectorx: bool = True,
        **connection_params
    ) -> BaseDataLoader:
        """
        Create a data loader for the given database type.

        Args:
            db_type: Database type
            use_connectorx: Use ConnectorX if available (default: True)
            **connection_params: Database connection parameters

        Returns:
            Appropriate DataLoader instance
        """
        db_type_lower = db_type.lower() if db_type else 'postgresql'

        # Use ConnectorX for supported databases
        if use_connectorx and db_type_lower in cls.CONNECTORX_SUPPORTED:
            try:
                return ConnectorXLoader(db_type=db_type, **connection_params)
            except ImportError:
                logger.warning("ConnectorX not available, falling back to SQLAlchemy")

        # Fallback to SQLAlchemy-based loader
        return SQLAlchemyLoader(db_type=db_type, **connection_params)

    @classmethod
    def get_supported_databases(cls) -> List[str]:
        """Get list of databases with ConnectorX support."""
        return list(cls.CONNECTORX_SUPPORTED)


class SQLAlchemyLoader(BaseDataLoader):
    """
    Fallback data loader using SQLAlchemy and Pandas.

    Used when ConnectorX is not available or for unsupported databases.
    """

    DRIVER_MAP = {
        'postgresql': 'postgresql+psycopg2',
        'mysql': 'mysql+pymysql',
        'sqlite': 'sqlite',
        'mssql': 'mssql+pymssql',
        'oracle': 'oracle+oracledb',
    }

    def __init__(
        self,
        db_type: str = None,
        host: str = "localhost",
        port: int = None,
        database: str = "",
        user: str = "",
        password: str = "",
        connection_string: str = None,
        **kwargs
    ):
        """Initialize SQLAlchemy loader."""
        self.db_type = db_type or 'postgresql'
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._connection_string = connection_string
        self._extra_params = kwargs
        self._engine = None

    def get_connection_string(self) -> str:
        """Build SQLAlchemy connection string."""
        if self._connection_string:
            return self._connection_string

        from urllib.parse import quote_plus

        driver = self.DRIVER_MAP.get(self.db_type, self.db_type)

        if self.db_type == 'sqlite':
            return f"sqlite:///{self.database}"

        encoded_password = quote_plus(self.password) if self.password else ''
        port_str = f":{self.port}" if self.port else ""

        return f"{driver}://{self.user}:{encoded_password}@{self.host}{port_str}/{self.database}"

    @property
    def engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            from sqlalchemy import create_engine
            self._engine = create_engine(self.get_connection_string())
        return self._engine

    def load(
        self,
        query: str,
        output_format: Union[OutputFormat, str] = OutputFormat.PANDAS,
        **kwargs
    ) -> Any:
        """Load query results using Pandas read_sql."""
        import pandas as pd

        df = pd.read_sql(query, self.engine)

        # Convert to requested format
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format.lower())

        if output_format == OutputFormat.POLARS:
            import polars as pl
            return pl.from_pandas(df)
        elif output_format == OutputFormat.ARROW:
            import pyarrow as pa
            return pa.Table.from_pandas(df)

        return df

    def test_connection(self) -> Dict[str, Any]:
        """Test database connection."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"success": True, "message": "Connection successful"}
        except Exception as e:
            return {"success": False, "error": str(e)}


__all__ = [
    'BaseDataLoader',
    'ConnectorXLoader',
    'SQLAlchemyLoader',
    'DataLoaderFactory',
    'OutputFormat',
    'DatabaseProtocol',
]
