"""
Core Semantic Models

Data models representing database schema, relationships, and semantic metadata.
These models are database-agnostic and serve as the foundation for LookML generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ColumnType(Enum):
    """Normalized column types across databases."""

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    TIMESTAMP = "timestamp"
    JSON = "json"
    BINARY = "binary"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Types of table relationships."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class MeasureType(Enum):
    """Types of measures for aggregations."""

    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"


@dataclass
class Column:
    """Represents a database column with semantic metadata."""

    name: str
    data_type: str  # Original database type
    normalized_type: ColumnType = ColumnType.UNKNOWN
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    default: Optional[str] = None
    description: Optional[str] = None

    # Semantic hints
    is_dimension: bool = True  # Dimensions are groupable
    is_measure: bool = False  # Measures are aggregatable
    measure_type: Optional[MeasureType] = None

    # LookML hints
    hidden: bool = False
    label: Optional[str] = None

    def __post_init__(self) -> None:
        """Auto-detect semantic properties based on type and name."""
        self._normalize_type()
        self._detect_semantics()

    def _normalize_type(self) -> None:
        """Normalize database-specific types to generic types."""
        # Skip if already explicitly set to a non-UNKNOWN value
        if self.normalized_type != ColumnType.UNKNOWN:
            return

        type_lower = self.data_type.lower()

        type_mapping = {
            ColumnType.STRING: [
                "varchar",
                "char",
                "text",
                "string",
                "nvarchar",
                "nchar",
                "clob",
                "nclob",
                "str",
                "objectid",
                "uuid",  # MongoDB/Python types
            ],
            ColumnType.INTEGER: [
                "int",
                "integer",
                "bigint",
                "smallint",
                "tinyint",
                "serial",
                "bigserial",
                "int32",
                "int64",
                "long",  # MongoDB types
            ],
            ColumnType.FLOAT: ["float", "double", "real", "float4", "float8"],
            ColumnType.DECIMAL: ["decimal", "numeric", "number", "money", "decimal128"],
            ColumnType.BOOLEAN: ["bool", "boolean", "bit"],
            ColumnType.DATE: ["date"],
            ColumnType.DATETIME: ["datetime", "datetime2"],
            ColumnType.TIMESTAMP: ["timestamp", "timestamptz"],
            ColumnType.TIME: ["time", "timetz"],
            ColumnType.JSON: ["json", "jsonb", "list", "dict", "array", "object"],  # MongoDB types
            ColumnType.BINARY: ["blob", "binary", "varbinary", "bytea", "raw", "bytes"],
        }

        for norm_type, patterns in type_mapping.items():
            for pattern in patterns:
                if pattern in type_lower:
                    self.normalized_type = norm_type
                    return

        # Default to STRING for unknown types (safer than UNKNOWN)
        self.normalized_type = ColumnType.STRING

    def _detect_semantics(self) -> None:
        """Auto-detect if column is dimension or measure."""
        name_lower = self.name.lower()

        # Numeric types that look like measures
        measure_patterns = [
            "amount",
            "total",
            "sum",
            "count",
            "qty",
            "quantity",
            "price",
            "cost",
            "revenue",
            "sales",
            "profit",
            "balance",
        ]

        if self.normalized_type in [ColumnType.FLOAT, ColumnType.DECIMAL, ColumnType.NUMBER]:
            for pattern in measure_patterns:
                if pattern in name_lower:
                    self.is_measure = True
                    self.is_dimension = False
                    self.measure_type = MeasureType.SUM
                    return

        # IDs and keys are dimensions
        if name_lower.endswith("_id") or name_lower == "id" or self.primary_key:
            self.is_dimension = True
            self.is_measure = False


@dataclass
class ForeignKey:
    """Represents a foreign key relationship."""

    columns: List[str]
    referenced_table: str
    referenced_columns: List[str]
    constraint_name: Optional[str] = None


@dataclass
class Index:
    """Represents a database index."""

    name: str
    columns: List[str]
    unique: bool = False


@dataclass
class Table:
    """Represents a database table with full schema."""

    name: str
    schema: Optional[str] = None
    columns: List[Column] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    indexes: List[Index] = field(default_factory=list)
    description: Optional[str] = None
    row_count: Optional[int] = None

    # LookML hints
    label: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Get fully qualified table name."""
        if self.schema:
            return f"{self.schema}.{self.name}"
        return self.name

    def get_column(self, name: str) -> Optional[Column]:
        """Get column by name."""
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None

    @property
    def dimensions(self) -> List[Column]:
        """Get all dimension columns."""
        return [c for c in self.columns if c.is_dimension]

    @property
    def measures(self) -> List[Column]:
        """Get all measure columns."""
        return [c for c in self.columns if c.is_measure]


@dataclass
class Relationship:
    """Represents a relationship between two tables."""

    from_table: str
    from_columns: List[str]
    to_table: str
    to_columns: List[str]
    relation_type: RelationType = RelationType.MANY_TO_ONE
    join_type: str = "left_outer"  # inner, left_outer, full_outer, cross

    @property
    def sql_on(self) -> str:
        """Generate SQL ON clause for join."""
        conditions = []
        for from_col, to_col in zip(self.from_columns, self.to_columns):
            conditions.append(f"{self.from_table}.{from_col} = {self.to_table}.{to_col}")
        return " AND ".join(conditions)


@dataclass
class Schema:
    """Represents a complete database schema with all tables and relationships."""

    name: str
    tables: List[Table] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    database_type: str = "unknown"

    # Metadata
    extracted_at: Optional[str] = None
    version: str = "1.0"

    def get_table(self, name: str) -> Optional[Table]:
        """Get table by name."""
        for table in self.tables:
            if table.name.lower() == name.lower():
                return table
            if table.full_name.lower() == name.lower():
                return table
        return None

    def infer_relationships(self) -> None:
        """Infer relationships from foreign keys."""
        for table in self.tables:
            for fk in table.foreign_keys:
                # Determine relationship type
                # If FK columns are also PK, it's likely one-to-one
                is_pk = all(col in table.primary_keys for col in fk.columns)
                rel_type = RelationType.ONE_TO_ONE if is_pk else RelationType.MANY_TO_ONE

                rel = Relationship(
                    from_table=table.name,
                    from_columns=fk.columns,
                    to_table=fk.referenced_table,
                    to_columns=fk.referenced_columns,
                    relation_type=rel_type,
                )

                # Avoid duplicates
                if not any(
                    r.from_table == rel.from_table
                    and r.to_table == rel.to_table
                    and r.from_columns == rel.from_columns
                    for r in self.relationships
                ):
                    self.relationships.append(rel)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "database_type": self.database_type,
            "tables": [
                {
                    "name": t.name,
                    "schema": t.schema,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.data_type,
                            "normalized_type": c.normalized_type.value,
                            "nullable": c.nullable,
                            "primary_key": c.primary_key,
                            "is_dimension": c.is_dimension,
                            "is_measure": c.is_measure,
                        }
                        for c in t.columns
                    ],
                    "primary_keys": t.primary_keys,
                    "foreign_keys": [
                        {
                            "columns": fk.columns,
                            "referenced_table": fk.referenced_table,
                            "referenced_columns": fk.referenced_columns,
                        }
                        for fk in t.foreign_keys
                    ],
                }
                for t in self.tables
            ],
            "relationships": [
                {
                    "from_table": r.from_table,
                    "from_columns": r.from_columns,
                    "to_table": r.to_table,
                    "to_columns": r.to_columns,
                    "type": r.relation_type.value,
                }
                for r in self.relationships
            ],
            "version": self.version,
        }
