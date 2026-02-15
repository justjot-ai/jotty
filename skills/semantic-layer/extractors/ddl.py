"""
DDL Schema Extractor

Extracts schema from DDL strings using sqlglot and simple-ddl-parser.
Supports parsing DDL from multiple database dialects.
"""
from typing import List, Optional, Dict, Any
import logging

from .base import BaseExtractor
from ..models import Column, ForeignKey, Index

logger = logging.getLogger(__name__)


class DDLExtractor(BaseExtractor):
    """
    Extract schema from DDL strings.

    Uses:
    - sqlglot: SQL parser supporting 30+ dialects with AST
    - simple-ddl-parser: DDL-specific parser with rich output

    Supports dialects:
    - PostgreSQL, MySQL, SQLite, SQL Server, Oracle
    - Snowflake, BigQuery, Redshift, Databricks
    - And many more via sqlglot
    """

    # Dialect mappings for sqlglot
    DIALECT_MAP = {
        "postgresql": "postgres",
        "postgres": "postgres",
        "mysql": "mysql",
        "sqlite": "sqlite",
        "mssql": "tsql",
        "sqlserver": "tsql",
        "tsql": "tsql",
        "oracle": "oracle",
        "snowflake": "snowflake",
        "bigquery": "bigquery",
        "redshift": "redshift",
        "databricks": "databricks",
        "spark": "spark",
        "hive": "hive",
        "duckdb": "duckdb",
    }

    def __init__(self, ddl: str, dialect: str = 'postgres') -> None:
        """
        Initialize DDL extractor.

        Args:
            ddl: DDL string to parse
            dialect: SQL dialect (postgresql, mysql, sqlite, mssql, oracle, etc.)
        """
        self.ddl = ddl
        self.dialect = self._normalize_dialect(dialect)

        super().__init__(database_type=dialect)

        self._parsed_tables: Dict[str, Dict] = {}
        self._parse_ddl()

    def _normalize_dialect(self, dialect: str) -> str:
        """Normalize dialect name for sqlglot."""
        return self.DIALECT_MAP.get(dialect.lower(), dialect.lower())

    def _parse_ddl(self) -> None:
        """Parse DDL using available parsers."""
        # Try simple-ddl-parser first (more DDL-specific)
        try:
            self._parse_with_simple_ddl_parser()
            if self._parsed_tables:
                logger.info(f"Parsed {len(self._parsed_tables)} tables with simple-ddl-parser")
                return
        except Exception as e:
            logger.debug(f"simple-ddl-parser failed: {e}")

        # Fall back to sqlglot
        try:
            self._parse_with_sqlglot()
            logger.info(f"Parsed {len(self._parsed_tables)} tables with sqlglot")
        except Exception as e:
            logger.error(f"All DDL parsers failed: {e}")

    def _parse_with_simple_ddl_parser(self) -> Any:
        """Parse DDL using simple-ddl-parser."""
        from simple_ddl_parser import DDLParser

        parser = DDLParser(self.ddl)
        result = parser.run(output_mode="python")

        for table_def in result:
            table_name = table_def.get('table_name', '')
            if not table_name:
                continue

            columns = []
            primary_keys = []
            foreign_keys = []

            for col in table_def.get('columns', []):
                col_name = col.get('name', '')
                col_type = col.get('type', 'unknown')

                # Handle type with size
                if col.get('size'):
                    col_type = f"{col_type}({col.get('size')})"

                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'nullable': not col.get('nullable', True) == False,
                    'default': col.get('default'),
                })

                if col.get('primary_key'):
                    primary_keys.append(col_name)

            # Extract primary key constraint
            pk_constraint = table_def.get('primary_key', [])
            if pk_constraint:
                primary_keys.extend([pk for pk in pk_constraint if pk not in primary_keys])

            # Extract foreign keys
            for fk in table_def.get('constraints', {}).get('foreign_keys', []):
                foreign_keys.append({
                    'columns': fk.get('columns', []),
                    'referenced_table': fk.get('reference', {}).get('table', ''),
                    'referenced_columns': fk.get('reference', {}).get('columns', []),
                })

            self._parsed_tables[table_name] = {
                'columns': columns,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys,
                'schema': table_def.get('schema'),
            }

    def _parse_with_sqlglot(self) -> Any:
        """Parse DDL using sqlglot."""
        import sqlglot
        from sqlglot import exp

        # Parse all statements
        statements = sqlglot.parse(self.ddl, dialect=self.dialect)

        for stmt in statements:
            if not isinstance(stmt, exp.Create):
                continue

            # Get table name
            table_expr = stmt.this
            if not isinstance(table_expr, exp.Schema):
                continue

            table_name = table_expr.this.name if table_expr.this else ""
            if not table_name:
                continue

            columns = []
            primary_keys = []
            foreign_keys = []

            # Extract columns
            for col_def in table_expr.expressions:
                if isinstance(col_def, exp.ColumnDef):
                    col_name = col_def.this.name if col_def.this else ""
                    col_type = col_def.kind.sql(dialect=self.dialect) if col_def.kind else "unknown"

                    # Check constraints
                    nullable = True
                    is_pk = False
                    default = None

                    for constraint in col_def.constraints or []:
                        if isinstance(constraint.kind, exp.NotNullColumnConstraint):
                            nullable = False
                        elif isinstance(constraint.kind, exp.PrimaryKeyColumnConstraint):
                            is_pk = True
                        elif isinstance(constraint.kind, exp.DefaultColumnConstraint):
                            default = constraint.kind.this.sql(dialect=self.dialect) if constraint.kind.this else None

                    columns.append({
                        'name': col_name,
                        'type': col_type,
                        'nullable': nullable,
                        'default': default,
                    })

                    if is_pk:
                        primary_keys.append(col_name)

                # Handle table-level constraints
                elif isinstance(col_def, exp.PrimaryKey):
                    for expr in col_def.expressions:
                        if hasattr(expr, 'name'):
                            primary_keys.append(expr.name)

                elif isinstance(col_def, exp.ForeignKey):
                    fk_cols = [e.name for e in col_def.expressions if hasattr(e, 'name')]
                    ref = col_def.args.get('reference')
                    if ref:
                        ref_table = ref.this.name if ref.this else ""
                        ref_cols = [e.name for e in ref.expressions if hasattr(e, 'name')]
                        foreign_keys.append({
                            'columns': fk_cols,
                            'referenced_table': ref_table,
                            'referenced_columns': ref_cols,
                        })

            self._parsed_tables[table_name] = {
                'columns': columns,
                'primary_keys': primary_keys,
                'foreign_keys': foreign_keys,
            }

    def _extract_tables(self) -> List[str]:
        """Extract list of table names."""
        return list(self._parsed_tables.keys())

    def _extract_columns(self, table_name: str) -> List[Column]:
        """Extract columns for a table."""
        table_data = self._parsed_tables.get(table_name, {})
        columns = []

        for col in table_data.get('columns', []):
            columns.append(Column(
                name=col['name'],
                data_type=col['type'],
                nullable=col.get('nullable', True),
                default=col.get('default'),
            ))

        return columns

    def _extract_primary_keys(self, table_name: str) -> List[str]:
        """Extract primary key column names."""
        table_data = self._parsed_tables.get(table_name, {})
        return table_data.get('primary_keys', [])

    def _extract_foreign_keys(self, table_name: str) -> List[ForeignKey]:
        """Extract foreign keys."""
        table_data = self._parsed_tables.get(table_name, {})
        foreign_keys = []

        for fk in table_data.get('foreign_keys', []):
            foreign_keys.append(ForeignKey(
                columns=fk['columns'],
                referenced_table=fk['referenced_table'],
                referenced_columns=fk['referenced_columns'],
            ))

        return foreign_keys

    @classmethod
    def from_file(cls, file_path: str, dialect: str = "postgres") -> "DDLExtractor":
        """Create extractor from DDL file."""
        with open(file_path, 'r') as f:
            ddl = f.read()
        return cls(ddl, dialect)

    @classmethod
    def transpile_ddl(cls, ddl: str, from_dialect: str, to_dialect: str) -> str:
        """
        Transpile DDL from one dialect to another using sqlglot.

        Args:
            ddl: DDL string
            from_dialect: Source dialect
            to_dialect: Target dialect

        Returns:
            Transpiled DDL string
        """
        import sqlglot

        from_dialect = cls.DIALECT_MAP.get(from_dialect.lower(), from_dialect.lower())
        to_dialect = cls.DIALECT_MAP.get(to_dialect.lower(), to_dialect.lower())

        return sqlglot.transpile(ddl, read=from_dialect, write=to_dialect, pretty=True)[0]
