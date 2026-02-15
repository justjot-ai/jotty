"""
Base Schema Extractor

Abstract base class for all schema extractors.
Implements the Template Method pattern for DRY schema extraction.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..models import Schema, Table, Column, ForeignKey, Index, Relationship

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Abstract base class for schema extractors.

    Subclasses must implement:
    - _extract_tables(): Get list of table names
    - _extract_columns(table_name): Get columns for a table
    - _extract_primary_keys(table_name): Get primary key columns
    - _extract_foreign_keys(table_name): Get foreign keys
    - _extract_indexes(table_name): Get indexes (optional)
    """

    def __init__(self, database_type: str = 'unknown') -> None:
        self.database_type = database_type
        self._schema_cache: Optional[Schema] = None

    def extract(self, schema_name: str = "default", use_cache: bool = True) -> Schema:
        """
        Extract complete database schema.

        This is the template method that orchestrates the extraction process.

        Args:
            schema_name: Name for the extracted schema
            use_cache: Whether to use cached schema if available

        Returns:
            Schema object with all tables and relationships
        """
        if use_cache and self._schema_cache:
            return self._schema_cache

        logger.info(f"Extracting schema: {schema_name}")

        schema = Schema(
            name=schema_name,
            database_type=self.database_type,
            extracted_at=datetime.now().isoformat()
        )

        # Extract all tables
        table_names = self._extract_tables()
        logger.info(f"Found {len(table_names)} tables")

        for table_name in table_names:
            try:
                table = self._build_table(table_name)
                if table:
                    schema.tables.append(table)
            except Exception as e:
                logger.warning(f"Failed to extract table {table_name}: {e}")

        # Infer relationships from foreign keys
        schema.infer_relationships()

        # Additional relationship inference (naming conventions)
        self._infer_additional_relationships(schema)

        self._schema_cache = schema
        return schema

    def _build_table(self, table_name: str) -> Optional[Table]:
        """Build a complete Table object."""
        columns = self._extract_columns(table_name)
        if not columns:
            return None

        primary_keys = self._extract_primary_keys(table_name)
        foreign_keys = self._extract_foreign_keys(table_name)
        indexes = self._extract_indexes(table_name)

        # Mark primary key columns
        for col in columns:
            if col.name in primary_keys:
                col.primary_key = True

        return Table(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            indexes=indexes
        )

    def _infer_additional_relationships(self, schema: Schema) -> Any:
        """
        Infer relationships from naming conventions.

        Common patterns:
        - user_id -> users.id
        - customer_id -> customers.id
        - order_id -> orders.id
        """
        table_names = {t.name.lower(): t.name for t in schema.tables}

        for table in schema.tables:
            for column in table.columns:
                # Skip if already has FK relationship
                if any(column.name in fk.columns for fk in table.foreign_keys):
                    continue

                # Check for _id pattern
                if column.name.lower().endswith('_id'):
                    potential_table = column.name[:-3]  # Remove _id

                    # Try singular and plural forms
                    for candidate in [potential_table, potential_table + 's',
                                     potential_table + 'es', potential_table.rstrip('s')]:
                        if candidate.lower() in table_names:
                            ref_table = table_names[candidate.lower()]
                            ref_table_obj = schema.get_table(ref_table)

                            if ref_table_obj and ref_table_obj.primary_keys:
                                rel = Relationship(
                                    from_table=table.name,
                                    from_columns=[column.name],
                                    to_table=ref_table,
                                    to_columns=ref_table_obj.primary_keys[:1],
                                    join_type="left_outer"
                                )

                                # Avoid duplicates
                                if not any(r.from_table == rel.from_table and
                                          r.to_table == rel.to_table and
                                          r.from_columns == rel.from_columns
                                          for r in schema.relationships):
                                    schema.relationships.append(rel)
                                break

    @abstractmethod
    def _extract_tables(self) -> List[str]:
        """Extract list of table names. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _extract_columns(self, table_name: str) -> List[Column]:
        """Extract columns for a table. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _extract_primary_keys(self, table_name: str) -> List[str]:
        """Extract primary key column names. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _extract_foreign_keys(self, table_name: str) -> List[ForeignKey]:
        """Extract foreign keys. Must be implemented by subclasses."""
        pass

    def _extract_indexes(self, table_name: str) -> List[Index]:
        """Extract indexes. Optional - default returns empty list."""
        return []

    def clear_cache(self) -> None:
        """Clear cached schema."""
        self._schema_cache = None
