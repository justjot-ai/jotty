"""
Database Schema Extractor

Extracts schema from live database connections using SQLAlchemy.
Supports PostgreSQL, MySQL, SQLite, SQL Server, Oracle, and more.
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from ..models import Column, ForeignKey, Index
from .base import BaseExtractor

logger = logging.getLogger(__name__)


class DatabaseExtractor(BaseExtractor):
    """
    Extract schema from live database using SQLAlchemy.

    Supports any database supported by SQLAlchemy:
    - PostgreSQL
    - MySQL
    - SQLite
    - SQL Server (MSSQL)
    - Oracle
    - And more via dialects
    """

    # Default ports for common databases
    DEFAULT_PORTS = {
        "postgresql": 5432,
        "mysql": 3306,
        "mssql": 1433,
        "oracle": 1521,
    }

    # SQLAlchemy driver mappings
    DRIVERS = {
        "postgresql": "postgresql+psycopg2",
        "mysql": "mysql+pymysql",
        "sqlite": "sqlite",
        "mssql": "mssql+pymssql",
        "oracle": "oracle+oracledb",
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
        **kwargs: Any,
    ) -> None:
        """
        Initialize database extractor.

        Args:
            db_type: Database type (postgresql, mysql, sqlite, mssql, oracle)
            host: Database host
            port: Database port
            database: Database name
            user: Username
            password: Password
            connection_string: Full SQLAlchemy connection URL (overrides other params)
            **kwargs: Additional connection parameters
        """
        if connection_string:
            self.connection_string = connection_string
            db_type = self._infer_db_type(connection_string)
        else:
            if not db_type:
                raise ValueError("Either db_type or connection_string is required")
            self.connection_string = self._build_connection_string(
                db_type, host, port, database, user, password, **kwargs
            )

        super().__init__(database_type=db_type or "unknown")

        self._engine = None
        self._inspector = None

    def _infer_db_type(self, connection_string: str) -> str:
        """Infer database type from connection string."""
        conn_lower = connection_string.lower()
        for db_type in self.DRIVERS.keys():
            if db_type in conn_lower:
                return db_type
        return "unknown"

    def _build_connection_string(
        self,
        db_type: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        **kwargs: Any,
    ) -> str:
        """Build SQLAlchemy connection string."""
        driver = self.DRIVERS.get(db_type)
        if not driver:
            raise ValueError(f"Unsupported database type: {db_type}")

        # SQLite is special
        if db_type == "sqlite":
            return f"sqlite:///{database}"

        port = port or self.DEFAULT_PORTS.get(db_type)
        encoded_password = quote_plus(password) if password else ""

        # Handle Oracle service_name
        if db_type == "oracle" and kwargs.get("service_name"):
            return f"{driver}://{user}:{encoded_password}@{host}:{port}/?service_name={kwargs['service_name']}"

        if user and encoded_password:
            return f"{driver}://{user}:{encoded_password}@{host}:{port}/{database}"
        elif user:
            return f"{driver}://{user}@{host}:{port}/{database}"
        else:
            return f"{driver}://{host}:{port}/{database}"

    @property
    def engine(self) -> Any:
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            from sqlalchemy import create_engine
            from sqlalchemy.pool import QueuePool

            self._engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
            )
        return self._engine

    @property
    def inspector(self) -> Any:
        """Get or create SQLAlchemy inspector."""
        if self._inspector is None:
            from sqlalchemy import inspect

            self._inspector = inspect(self.engine)
        return self._inspector

    def _extract_tables(self) -> List[str]:
        """Extract list of table names."""
        try:
            return self.inspector.get_table_names()
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            return []

    def _extract_columns(self, table_name: str) -> List[Column]:
        """Extract columns for a table."""
        columns = []
        try:
            for col in self.inspector.get_columns(table_name):
                column = Column(
                    name=col["name"],
                    data_type=str(col.get("type", "unknown")),
                    nullable=col.get("nullable", True),
                    default=str(col.get("default", "")) if col.get("default") else None,
                )
                columns.append(column)
        except Exception as e:
            logger.error(f"Failed to get columns for {table_name}: {e}")

        return columns

    def _extract_primary_keys(self, table_name: str) -> List[str]:
        """Extract primary key column names."""
        try:
            pk = self.inspector.get_pk_constraint(table_name)
            return pk.get("constrained_columns", []) if pk else []
        except Exception as e:
            logger.error(f"Failed to get primary keys for {table_name}: {e}")
            return []

    def _extract_foreign_keys(self, table_name: str) -> List[ForeignKey]:
        """Extract foreign keys."""
        foreign_keys = []
        try:
            for fk in self.inspector.get_foreign_keys(table_name):
                foreign_keys.append(
                    ForeignKey(
                        columns=fk.get("constrained_columns", []),
                        referenced_table=fk.get("referred_table", ""),
                        referenced_columns=fk.get("referred_columns", []),
                        constraint_name=fk.get("name"),
                    )
                )
        except Exception as e:
            logger.error(f"Failed to get foreign keys for {table_name}: {e}")

        return foreign_keys

    def _extract_indexes(self, table_name: str) -> List[Index]:
        """Extract indexes."""
        indexes = []
        try:
            for idx in self.inspector.get_indexes(table_name):
                indexes.append(
                    Index(
                        name=idx.get("name", ""),
                        columns=idx.get("column_names", []),
                        unique=idx.get("unique", False),
                    )
                )
        except Exception as e:
            logger.error(f"Failed to get indexes for {table_name}: {e}")

        return indexes

    def test_connection(self) -> Dict[str, Any]:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return {"success": True, "message": "Connection successful"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._inspector = None
