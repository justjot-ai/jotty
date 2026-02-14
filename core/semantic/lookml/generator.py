"""
LookML Generator

Converts database schema to LookML semantic layer.
"""
from typing import List, Optional, Dict, Any
import logging
import re

from ..models import Schema, Table, Column, ColumnType, Relationship as SchemaRelationship, MeasureType as SchemaMeasureType
from .models import (
    LookMLModel, View, Explore, Dimension, Measure, Join,
    DimensionType, MeasureType, JoinType, Relationship
)

logger = logging.getLogger(__name__)


class LookMLGenerator:
    """
    Generate LookML from database schema.

    Converts:
    - Tables -> Views
    - Columns -> Dimensions/Measures
    - Relationships -> Explores with Joins
    """

    # Map semantic column types to LookML dimension types
    TYPE_MAP = {
        ColumnType.STRING: DimensionType.STRING,
        ColumnType.INTEGER: DimensionType.NUMBER,
        ColumnType.NUMBER: DimensionType.NUMBER,
        ColumnType.FLOAT: DimensionType.NUMBER,
        ColumnType.DECIMAL: DimensionType.NUMBER,
        ColumnType.BOOLEAN: DimensionType.YESNO,
        ColumnType.DATE: DimensionType.DATE,
        ColumnType.DATETIME: DimensionType.DATETIME,
        ColumnType.TIMESTAMP: DimensionType.DATETIME,
        ColumnType.TIME: DimensionType.TIME,
        ColumnType.JSON: DimensionType.STRING,
        ColumnType.BINARY: DimensionType.STRING,
        ColumnType.UNKNOWN: DimensionType.STRING,
    }

    # Map semantic measure types to LookML measure types
    MEASURE_TYPE_MAP = {
        SchemaMeasureType.COUNT: MeasureType.COUNT,
        SchemaMeasureType.COUNT_DISTINCT: MeasureType.COUNT_DISTINCT,
        SchemaMeasureType.SUM: MeasureType.SUM,
        SchemaMeasureType.AVERAGE: MeasureType.AVERAGE,
        SchemaMeasureType.MIN: MeasureType.MIN,
        SchemaMeasureType.MAX: MeasureType.MAX,
        SchemaMeasureType.MEDIAN: MeasureType.MEDIAN,
    }

    def __init__(self, schema: Schema) -> None:
        """
        Initialize generator with schema.

        Args:
            schema: Database schema to convert
        """
        self.schema = schema
        self._views: Dict[str, View] = {}
        self._explores: Dict[str, Explore] = {}

    def generate(self, model_name: str = None, connection: str = "") -> LookMLModel:
        """
        Generate complete LookML model.

        Args:
            model_name: Name for the LookML model
            connection: Database connection name

        Returns:
            LookMLModel with views and explores
        """
        model_name = model_name or self.schema.name

        # Generate views from tables
        for table in self.schema.tables:
            view = self._generate_view(table)
            self._views[view.name] = view

        # Generate explores from relationships
        self._generate_explores()

        return LookMLModel(
            name=model_name,
            connection=connection,
            views=list(self._views.values()),
            explores=list(self._explores.values())
        )

    def _generate_view(self, table: Table) -> View:
        """Generate LookML view from table."""
        view_name = self._to_snake_case(table.name)

        dimensions = []
        measures = []

        for column in table.columns:
            if column.is_measure and column.measure_type:
                measure = self._generate_measure(column, view_name)
                measures.append(measure)
            else:
                dimension = self._generate_dimension(column, table)
                dimensions.append(dimension)

        # Add count measure for every view
        measures.append(Measure(
            name="count",
            type=MeasureType.COUNT,
            description=f"Count of {table.name} records",
            drill_fields=[d.name for d in dimensions if d.primary_key or not d.hidden][:5]
        ))

        return View(
            name=view_name,
            sql_table_name=table.full_name,
            label=self._to_label(table.name),
            dimensions=dimensions,
            measures=measures
        )

    def _generate_dimension(self, column: Column, table: Table) -> Dimension:
        """Generate LookML dimension from column."""
        dim_type = self.TYPE_MAP.get(column.normalized_type, DimensionType.STRING)

        # Determine if this is a primary key
        is_pk = column.name in table.primary_keys or column.primary_key

        dimension = Dimension(
            name=self._to_snake_case(column.name),
            type=dim_type,
            sql="${TABLE}." + column.name,
            label=column.label or self._to_label(column.name),
            description=column.description,
            primary_key=is_pk,
            hidden=column.hidden or column.name.lower().endswith('_id') and not is_pk,
        )

        # Group date dimensions
        if dim_type in [DimensionType.DATE, DimensionType.DATETIME]:
            dimension.group_label = "Dates"

        return dimension

    def _generate_measure(self, column: Column, view_name: str) -> Measure:
        """Generate LookML measure from column."""
        measure_type = self.MEASURE_TYPE_MAP.get(column.measure_type, MeasureType.SUM)

        return Measure(
            name=f"total_{self._to_snake_case(column.name)}",
            type=measure_type,
            sql="${TABLE}." + column.name,
            label=f"Total {self._to_label(column.name)}",
            description=column.description,
        )

    def _generate_explores(self) -> Any:
        """Generate explores from relationships."""
        # Group relationships by base table
        base_tables: Dict[str, List[SchemaRelationship]] = {}

        for rel in self.schema.relationships:
            if rel.from_table not in base_tables:
                base_tables[rel.from_table] = []
            base_tables[rel.from_table].append(rel)

        # Create an explore for each base table
        for table in self.schema.tables:
            view_name = self._to_snake_case(table.name)

            if view_name not in self._views:
                continue

            joins = []

            # Add joins for all relationships from this table
            for rel in base_tables.get(table.name, []):
                to_view_name = self._to_snake_case(rel.to_table)

                if to_view_name not in self._views:
                    continue

                join = self._generate_join(rel, view_name)
                joins.append(join)

            # Only create explore if it has joins or is a significant table
            if joins or len(table.columns) > 3:
                explore = Explore(
                    name=view_name,
                    view_name=view_name,
                    label=self._to_label(table.name),
                    description=f"Explore {table.name} data",
                    joins=joins
                )
                self._explores[view_name] = explore

    def _generate_join(self, rel: SchemaRelationship, base_view: str) -> Join:
        """Generate LookML join from relationship."""
        to_view = self._to_snake_case(rel.to_table)

        # Build SQL ON clause
        on_conditions = []
        for from_col, to_col in zip(rel.from_columns, rel.to_columns):
            on_conditions.append(
                f"${{{base_view}.{self._to_snake_case(from_col)}}} = ${{{to_view}.{self._to_snake_case(to_col)}}}"
            )

        sql_on = " AND ".join(on_conditions)

        # Map relationship type
        rel_map = {
            "one_to_one": Relationship.ONE_TO_ONE,
            "one_to_many": Relationship.ONE_TO_MANY,
            "many_to_one": Relationship.MANY_TO_ONE,
            "many_to_many": Relationship.MANY_TO_MANY,
        }

        join_type_map = {
            "inner": JoinType.INNER,
            "left_outer": JoinType.LEFT_OUTER,
            "full_outer": JoinType.FULL_OUTER,
            "cross": JoinType.CROSS,
        }

        relationship = rel_map.get(rel.relation_type.value, Relationship.MANY_TO_ONE)
        join_type = join_type_map.get(rel.join_type, JoinType.LEFT_OUTER)

        return Join(
            name=to_view,
            type=join_type,
            relationship=relationship,
            sql_on=sql_on
        )

    def to_lookml_string(self, model: LookMLModel = None) -> str:
        """
        Generate LookML string output.

        Args:
            model: LookMLModel to serialize (uses generated if not provided)

        Returns:
            LookML formatted string
        """
        try:
            import lkml
            model = model or self.generate()
            return lkml.dump(model.to_dict())
        except Exception as e:
            logger.warning(f"lkml serialization failed, using simple format: {e}")
            return self._simple_lookml_format(model or self.generate())

    def _simple_lookml_format(self, model: LookMLModel) -> str:
        """Simple LookML string format without lkml library."""
        lines = []

        # Views
        for view in model.views:
            lines.append(f"view: {view.name} {{")
            if view.sql_table_name:
                lines.append(f'  sql_table_name: {view.sql_table_name} ;;')
            if view.label:
                lines.append(f'  label: "{view.label}"')

            for dim in view.dimensions:
                lines.append(f"\n  dimension: {dim.name} {{")
                lines.append(f"    type: {dim.type.value}")
                if dim.sql:
                    lines.append(f"    sql: {dim.sql} ;;")
                if dim.primary_key:
                    lines.append("    primary_key: yes")
                if dim.hidden:
                    lines.append("    hidden: yes")
                lines.append("  }")

            for measure in view.measures:
                lines.append(f"\n  measure: {measure.name} {{")
                lines.append(f"    type: {measure.type.value}")
                if measure.sql:
                    lines.append(f"    sql: {measure.sql} ;;")
                lines.append("  }")

            lines.append("}\n")

        # Explores
        for explore in model.explores:
            lines.append(f"explore: {explore.name} {{")
            if explore.label:
                lines.append(f'  label: "{explore.label}"')

            for join in explore.joins:
                lines.append(f"\n  join: {join.name} {{")
                lines.append(f"    type: {join.type.value}")
                lines.append(f"    relationship: {join.relationship.value}")
                lines.append(f"    sql_on: {join.sql_on} ;;")
                lines.append("  }")

            lines.append("}\n")

        return "\n".join(lines)

    def to_context_string(self, model: LookMLModel = None) -> str:
        """
        Generate a compact context string for LLM consumption.

        This is optimized for passing to LLMs as query context.

        Args:
            model: LookMLModel to format

        Returns:
            Compact semantic context string
        """
        model = model or self.generate()
        lines = []

        lines.append(f"# Database: {self.schema.name} ({self.schema.database_type})")
        lines.append("")

        # Views/Tables
        lines.append("## Tables and Columns")
        for view in model.views:
            lines.append(f"\n### {view.name}")
            if view.sql_table_name:
                lines.append(f"Table: {view.sql_table_name}")

            # Dimensions
            dims = [f"{d.name} ({d.type.value})" + (" [PK]" if d.primary_key else "")
                   for d in view.dimensions if not d.hidden]
            if dims:
                lines.append(f"Dimensions: {', '.join(dims)}")

            # Measures
            measures = [f"{m.name} ({m.type.value})" for m in view.measures]
            if measures:
                lines.append(f"Measures: {', '.join(measures)}")

        # Relationships
        if model.explores:
            lines.append("\n## Relationships")
            for explore in model.explores:
                if explore.joins:
                    for join in explore.joins:
                        lines.append(f"- {explore.name} -> {join.name} ({join.relationship.value})")
                        lines.append(f"  ON: {join.sql_on}")

        return "\n".join(lines)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert name to snake_case."""
        # Handle camelCase
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Handle consecutive caps
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        # Replace spaces and hyphens
        s3 = re.sub(r'[\s\-]+', '_', s2)
        return s3.lower()

    @staticmethod
    def _to_label(name: str) -> str:
        """Convert name to human-readable label."""
        # Split on underscores, hyphens, camelCase
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        words = re.sub(r'[_\-]+', ' ', words)
        return words.title()
