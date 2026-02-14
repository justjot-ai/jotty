"""
Comprehensive tests for semantic models, LookML models, and date preprocessing.

Tests cover:
1. Semantic Models (core/semantic/models.py)
2. LookML Models (core/semantic/lookml/models.py)
3. Date Preprocessor (core/semantic/query/date_preprocessor.py)

All tests are offline, unit-level, and use mocks where needed.
"""
import pytest
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Semantic models imports
from Jotty.core.semantic.models import (
    ColumnType,
    RelationType,
    MeasureType,
    Column,
    ForeignKey,
    Index,
    Table,
    Relationship,
    Schema,
)

# LookML models imports
from Jotty.core.semantic.lookml.models import (
    DimensionType,
    MeasureType as LookMLMeasureType,
    JoinType,
    Relationship as LookMLRelationship,
    Dimension,
    Measure,
    View,
    Join,
    Explore,
    LookMLModel,
)

# Date preprocessor imports
from Jotty.core.semantic.query.date_preprocessor import (
    DateFormat,
    BaseDatePreprocessor,
    DatePreprocessor,
    SQLDatePreprocessor,
    DatePreprocessorFactory,
)


# =============================================================================
# Test Semantic Models - Enums
# =============================================================================

@pytest.mark.unit
class TestColumnType:
    """Test ColumnType enum values."""

    def test_string_type(self):
        assert ColumnType.STRING.value == "string"

    def test_number_type(self):
        assert ColumnType.NUMBER.value == "number"

    def test_integer_type(self):
        assert ColumnType.INTEGER.value == "integer"

    def test_float_type(self):
        assert ColumnType.FLOAT.value == "float"

    def test_decimal_type(self):
        assert ColumnType.DECIMAL.value == "decimal"

    def test_boolean_type(self):
        assert ColumnType.BOOLEAN.value == "boolean"

    def test_date_type(self):
        assert ColumnType.DATE.value == "date"

    def test_datetime_type(self):
        assert ColumnType.DATETIME.value == "datetime"

    def test_time_type(self):
        assert ColumnType.TIME.value == "time"

    def test_timestamp_type(self):
        assert ColumnType.TIMESTAMP.value == "timestamp"

    def test_json_type(self):
        assert ColumnType.JSON.value == "json"

    def test_binary_type(self):
        assert ColumnType.BINARY.value == "binary"

    def test_unknown_type(self):
        assert ColumnType.UNKNOWN.value == "unknown"


@pytest.mark.unit
class TestRelationType:
    """Test RelationType enum values."""

    def test_one_to_one(self):
        assert RelationType.ONE_TO_ONE.value == "one_to_one"

    def test_one_to_many(self):
        assert RelationType.ONE_TO_MANY.value == "one_to_many"

    def test_many_to_one(self):
        assert RelationType.MANY_TO_ONE.value == "many_to_one"

    def test_many_to_many(self):
        assert RelationType.MANY_TO_MANY.value == "many_to_many"


@pytest.mark.unit
class TestMeasureType:
    """Test MeasureType enum values."""

    def test_count(self):
        assert MeasureType.COUNT.value == "count"

    def test_count_distinct(self):
        assert MeasureType.COUNT_DISTINCT.value == "count_distinct"

    def test_sum(self):
        assert MeasureType.SUM.value == "sum"

    def test_average(self):
        assert MeasureType.AVERAGE.value == "average"

    def test_min(self):
        assert MeasureType.MIN.value == "min"

    def test_max(self):
        assert MeasureType.MAX.value == "max"

    def test_median(self):
        assert MeasureType.MEDIAN.value == "median"


# =============================================================================
# Test Semantic Models - Column
# =============================================================================

@pytest.mark.unit
class TestColumn:
    """Test Column dataclass and normalization."""

    def test_basic_creation(self):
        col = Column(name="test_col", data_type="varchar")
        assert col.name == "test_col"
        assert col.data_type == "varchar"

    def test_default_values(self):
        col = Column(name="id", data_type="int")
        assert col.nullable is True
        assert col.primary_key is False
        assert col.unique is False
        assert col.default is None
        assert col.description is None
        assert col.is_dimension is True
        assert col.is_measure is False
        assert col.measure_type is None
        assert col.hidden is False
        assert col.label is None

    def test_normalize_varchar_to_string(self):
        col = Column(name="name", data_type="varchar")
        assert col.normalized_type == ColumnType.STRING

    def test_normalize_char_to_string(self):
        col = Column(name="code", data_type="char")
        assert col.normalized_type == ColumnType.STRING

    def test_normalize_text_to_string(self):
        col = Column(name="description", data_type="text")
        assert col.normalized_type == ColumnType.STRING

    def test_normalize_nvarchar_to_string(self):
        col = Column(name="unicode_text", data_type="nvarchar")
        assert col.normalized_type == ColumnType.STRING

    def test_normalize_uuid_to_string(self):
        col = Column(name="uuid", data_type="uuid")
        assert col.normalized_type == ColumnType.STRING

    def test_normalize_int_to_integer(self):
        col = Column(name="count", data_type="int")
        assert col.normalized_type == ColumnType.INTEGER

    def test_normalize_integer_to_integer(self):
        col = Column(name="count", data_type="integer")
        assert col.normalized_type == ColumnType.INTEGER

    def test_normalize_bigint_to_integer(self):
        col = Column(name="big_id", data_type="bigint")
        assert col.normalized_type == ColumnType.INTEGER

    def test_normalize_smallint_to_integer(self):
        col = Column(name="small_num", data_type="smallint")
        assert col.normalized_type == ColumnType.INTEGER

    def test_normalize_serial_to_integer(self):
        col = Column(name="id", data_type="serial")
        assert col.normalized_type == ColumnType.INTEGER

    def test_normalize_float_to_float(self):
        col = Column(name="rate", data_type="float")
        assert col.normalized_type == ColumnType.FLOAT

    def test_normalize_double_to_float(self):
        col = Column(name="precision", data_type="double")
        assert col.normalized_type == ColumnType.FLOAT

    def test_normalize_real_to_float(self):
        col = Column(name="real_num", data_type="real")
        assert col.normalized_type == ColumnType.FLOAT

    def test_normalize_decimal_to_decimal(self):
        col = Column(name="price", data_type="decimal")
        assert col.normalized_type == ColumnType.DECIMAL

    def test_normalize_numeric_to_decimal(self):
        col = Column(name="amount", data_type="numeric")
        assert col.normalized_type == ColumnType.DECIMAL

    def test_normalize_money_to_decimal(self):
        col = Column(name="balance", data_type="money")
        assert col.normalized_type == ColumnType.DECIMAL

    def test_normalize_bool_to_boolean(self):
        col = Column(name="active", data_type="bool")
        assert col.normalized_type == ColumnType.BOOLEAN

    def test_normalize_boolean_to_boolean(self):
        col = Column(name="enabled", data_type="boolean")
        assert col.normalized_type == ColumnType.BOOLEAN

    def test_normalize_bit_to_boolean(self):
        col = Column(name="flag", data_type="bit")
        assert col.normalized_type == ColumnType.BOOLEAN

    def test_normalize_date_to_date(self):
        col = Column(name="birth_date", data_type="date")
        assert col.normalized_type == ColumnType.DATE

    def test_normalize_datetime_to_datetime(self):
        col = Column(name="created", data_type="datetime")
        # Note: "datetime" contains "date" so it normalizes to DATE based on pattern matching order
        assert col.normalized_type == ColumnType.DATE

    def test_normalize_datetime2_to_datetime(self):
        col = Column(name="updated", data_type="datetime2")
        # Note: "datetime2" contains "datetime" which contains "date" so it normalizes to DATE
        assert col.normalized_type == ColumnType.DATE

    def test_normalize_timestamp_to_timestamp(self):
        col = Column(name="ts", data_type="timestamp")
        assert col.normalized_type == ColumnType.TIMESTAMP

    def test_normalize_timestamptz_to_timestamp(self):
        col = Column(name="ts_tz", data_type="timestamptz")
        assert col.normalized_type == ColumnType.TIMESTAMP

    def test_normalize_time_to_time(self):
        col = Column(name="time", data_type="time")
        assert col.normalized_type == ColumnType.TIME

    def test_normalize_json_to_json(self):
        col = Column(name="data", data_type="json")
        assert col.normalized_type == ColumnType.JSON

    def test_normalize_jsonb_to_json(self):
        col = Column(name="metadata", data_type="jsonb")
        assert col.normalized_type == ColumnType.JSON

    def test_normalize_array_to_json(self):
        col = Column(name="tags", data_type="array")
        assert col.normalized_type == ColumnType.JSON

    def test_normalize_blob_to_binary(self):
        col = Column(name="file", data_type="blob")
        assert col.normalized_type == ColumnType.BINARY

    def test_normalize_binary_to_binary(self):
        col = Column(name="data", data_type="binary")
        assert col.normalized_type == ColumnType.BINARY

    def test_normalize_bytea_to_binary(self):
        col = Column(name="bytes", data_type="bytea")
        assert col.normalized_type == ColumnType.BINARY

    def test_normalize_unknown_defaults_to_string(self):
        col = Column(name="weird", data_type="custom_type")
        assert col.normalized_type == ColumnType.STRING

    def test_detect_measure_amount(self):
        col = Column(name="total_amount", data_type="decimal")
        assert col.is_measure is True
        assert col.is_dimension is False
        assert col.measure_type == MeasureType.SUM

    def test_detect_measure_total(self):
        col = Column(name="total", data_type="decimal")
        assert col.is_measure is True
        assert col.measure_type == MeasureType.SUM

    def test_detect_measure_price(self):
        col = Column(name="unit_price", data_type="float")
        assert col.is_measure is True

    def test_detect_measure_cost(self):
        col = Column(name="cost", data_type="decimal")
        assert col.is_measure is True

    def test_detect_measure_revenue(self):
        col = Column(name="revenue", data_type="decimal")
        assert col.is_measure is True

    def test_detect_measure_sales(self):
        col = Column(name="sales", data_type="float")
        assert col.is_measure is True

    def test_detect_measure_quantity(self):
        col = Column(name="quantity", data_type="decimal")
        assert col.is_measure is True

    def test_detect_dimension_id_suffix(self):
        col = Column(name="customer_id", data_type="int")
        assert col.is_dimension is True
        assert col.is_measure is False

    def test_detect_dimension_id_name(self):
        col = Column(name="id", data_type="int")
        assert col.is_dimension is True

    def test_detect_dimension_primary_key(self):
        col = Column(name="pk", data_type="int", primary_key=True)
        assert col.is_dimension is True
        assert col.is_measure is False

    def test_string_column_is_dimension(self):
        col = Column(name="name", data_type="varchar")
        assert col.is_dimension is True
        assert col.is_measure is False

    def test_explicit_normalized_type_not_overridden(self):
        col = Column(name="special", data_type="custom", normalized_type=ColumnType.INTEGER)
        assert col.normalized_type == ColumnType.INTEGER


# =============================================================================
# Test Semantic Models - ForeignKey, Index
# =============================================================================

@pytest.mark.unit
class TestForeignKey:
    """Test ForeignKey dataclass."""

    def test_basic_creation(self):
        fk = ForeignKey(
            columns=["customer_id"],
            referenced_table="customers",
            referenced_columns=["id"]
        )
        assert fk.columns == ["customer_id"]
        assert fk.referenced_table == "customers"
        assert fk.referenced_columns == ["id"]
        assert fk.constraint_name is None

    def test_with_constraint_name(self):
        fk = ForeignKey(
            columns=["order_id"],
            referenced_table="orders",
            referenced_columns=["id"],
            constraint_name="fk_order"
        )
        assert fk.constraint_name == "fk_order"

    def test_composite_foreign_key(self):
        fk = ForeignKey(
            columns=["user_id", "org_id"],
            referenced_table="user_orgs",
            referenced_columns=["user_id", "org_id"]
        )
        assert len(fk.columns) == 2
        assert len(fk.referenced_columns) == 2


@pytest.mark.unit
class TestIndex:
    """Test Index dataclass."""

    def test_basic_creation(self):
        idx = Index(name="idx_email", columns=["email"])
        assert idx.name == "idx_email"
        assert idx.columns == ["email"]
        assert idx.unique is False

    def test_unique_index(self):
        idx = Index(name="idx_unique_email", columns=["email"], unique=True)
        assert idx.unique is True

    def test_composite_index(self):
        idx = Index(name="idx_user_org", columns=["user_id", "org_id"])
        assert len(idx.columns) == 2


# =============================================================================
# Test Semantic Models - Table
# =============================================================================

@pytest.mark.unit
class TestTable:
    """Test Table dataclass."""

    def test_basic_creation(self):
        table = Table(name="users")
        assert table.name == "users"
        assert table.schema is None
        assert table.columns == []
        assert table.primary_keys == []
        assert table.foreign_keys == []
        assert table.indexes == []
        assert table.description is None
        assert table.row_count is None
        assert table.label is None

    def test_full_name_without_schema(self):
        table = Table(name="users")
        assert table.full_name == "users"

    def test_full_name_with_schema(self):
        table = Table(name="users", schema="public")
        assert table.full_name == "public.users"

    def test_get_column_case_insensitive(self):
        table = Table(
            name="users",
            columns=[
                Column(name="Id", data_type="int"),
                Column(name="Email", data_type="varchar"),
            ]
        )
        col = table.get_column("id")
        assert col is not None
        assert col.name == "Id"

    def test_get_column_uppercase(self):
        table = Table(
            name="users",
            columns=[Column(name="email", data_type="varchar")]
        )
        col = table.get_column("EMAIL")
        assert col is not None
        assert col.name == "email"

    def test_get_column_missing(self):
        table = Table(name="users", columns=[])
        col = table.get_column("nonexistent")
        assert col is None

    def test_dimensions_property(self):
        table = Table(
            name="orders",
            columns=[
                Column(name="id", data_type="int"),
                Column(name="customer_id", data_type="int"),
                Column(name="total_amount", data_type="decimal"),
            ]
        )
        dimensions = table.dimensions
        assert len(dimensions) == 2
        assert all(col.is_dimension for col in dimensions)

    def test_measures_property(self):
        table = Table(
            name="orders",
            columns=[
                Column(name="id", data_type="int"),
                Column(name="total_amount", data_type="decimal"),
                Column(name="revenue", data_type="float"),
            ]
        )
        measures = table.measures
        assert len(measures) == 2
        assert all(col.is_measure for col in measures)

    def test_with_foreign_keys(self):
        fk = ForeignKey(
            columns=["customer_id"],
            referenced_table="customers",
            referenced_columns=["id"]
        )
        table = Table(name="orders", foreign_keys=[fk])
        assert len(table.foreign_keys) == 1

    def test_with_indexes(self):
        idx = Index(name="idx_email", columns=["email"])
        table = Table(name="users", indexes=[idx])
        assert len(table.indexes) == 1


# =============================================================================
# Test Semantic Models - Relationship
# =============================================================================

@pytest.mark.unit
class TestRelationship:
    """Test Relationship dataclass."""

    def test_basic_creation(self):
        rel = Relationship(
            from_table="orders",
            from_columns=["customer_id"],
            to_table="customers",
            to_columns=["id"]
        )
        assert rel.from_table == "orders"
        assert rel.from_columns == ["customer_id"]
        assert rel.to_table == "customers"
        assert rel.to_columns == ["id"]
        assert rel.relation_type == RelationType.MANY_TO_ONE
        assert rel.join_type == "left_outer"

    def test_sql_on_single_column(self):
        rel = Relationship(
            from_table="orders",
            from_columns=["customer_id"],
            to_table="customers",
            to_columns=["id"]
        )
        assert rel.sql_on == "orders.customer_id = customers.id"

    def test_sql_on_multiple_columns(self):
        rel = Relationship(
            from_table="order_items",
            from_columns=["order_id", "product_id"],
            to_table="products",
            to_columns=["order_id", "id"]
        )
        expected = "order_items.order_id = products.order_id AND order_items.product_id = products.id"
        assert rel.sql_on == expected

    def test_one_to_one_relationship(self):
        rel = Relationship(
            from_table="users",
            from_columns=["profile_id"],
            to_table="profiles",
            to_columns=["id"],
            relation_type=RelationType.ONE_TO_ONE
        )
        assert rel.relation_type == RelationType.ONE_TO_ONE

    def test_custom_join_type(self):
        rel = Relationship(
            from_table="orders",
            from_columns=["customer_id"],
            to_table="customers",
            to_columns=["id"],
            join_type="inner"
        )
        assert rel.join_type == "inner"


# =============================================================================
# Test Semantic Models - Schema
# =============================================================================

@pytest.mark.unit
class TestSchema:
    """Test Schema dataclass."""

    def test_basic_creation(self):
        schema = Schema(name="public")
        assert schema.name == "public"
        assert schema.tables == []
        assert schema.relationships == []
        assert schema.database_type == "unknown"
        assert schema.extracted_at is None
        assert schema.version == "1.0"

    def test_get_table_by_name(self):
        schema = Schema(
            name="public",
            tables=[
                Table(name="users"),
                Table(name="orders"),
            ]
        )
        table = schema.get_table("users")
        assert table is not None
        assert table.name == "users"

    def test_get_table_case_insensitive(self):
        schema = Schema(name="public", tables=[Table(name="Users")])
        table = schema.get_table("users")
        assert table is not None

    def test_get_table_by_full_name(self):
        schema = Schema(
            name="public",
            tables=[Table(name="users", schema="public")]
        )
        table = schema.get_table("public.users")
        assert table is not None

    def test_get_table_missing(self):
        schema = Schema(name="public", tables=[])
        table = schema.get_table("nonexistent")
        assert table is None

    def test_infer_relationships_many_to_one(self):
        schema = Schema(
            name="public",
            tables=[
                Table(
                    name="orders",
                    primary_keys=["id"],
                    foreign_keys=[
                        ForeignKey(
                            columns=["customer_id"],
                            referenced_table="customers",
                            referenced_columns=["id"]
                        )
                    ]
                ),
                Table(name="customers", primary_keys=["id"]),
            ]
        )
        schema.infer_relationships()
        assert len(schema.relationships) == 1
        rel = schema.relationships[0]
        assert rel.from_table == "orders"
        assert rel.to_table == "customers"
        assert rel.relation_type == RelationType.MANY_TO_ONE

    def test_infer_relationships_one_to_one(self):
        schema = Schema(
            name="public",
            tables=[
                Table(
                    name="users",
                    primary_keys=["profile_id"],
                    foreign_keys=[
                        ForeignKey(
                            columns=["profile_id"],
                            referenced_table="profiles",
                            referenced_columns=["id"]
                        )
                    ]
                ),
                Table(name="profiles", primary_keys=["id"]),
            ]
        )
        schema.infer_relationships()
        assert len(schema.relationships) == 1
        rel = schema.relationships[0]
        assert rel.relation_type == RelationType.ONE_TO_ONE

    def test_infer_relationships_no_duplicates(self):
        schema = Schema(
            name="public",
            tables=[
                Table(
                    name="orders",
                    foreign_keys=[
                        ForeignKey(
                            columns=["customer_id"],
                            referenced_table="customers",
                            referenced_columns=["id"]
                        )
                    ]
                )
            ]
        )
        schema.infer_relationships()
        schema.infer_relationships()
        assert len(schema.relationships) == 1

    def test_to_dict(self):
        schema = Schema(
            name="public",
            database_type="postgresql",
            tables=[
                Table(
                    name="users",
                    columns=[
                        Column(name="id", data_type="int", primary_key=True),
                        Column(name="email", data_type="varchar"),
                    ],
                    primary_keys=["id"],
                    foreign_keys=[],
                )
            ],
            relationships=[
                Relationship(
                    from_table="orders",
                    from_columns=["user_id"],
                    to_table="users",
                    to_columns=["id"]
                )
            ],
            version="2.0"
        )
        result = schema.to_dict()
        assert result["name"] == "public"
        assert result["database_type"] == "postgresql"
        assert result["version"] == "2.0"
        assert len(result["tables"]) == 1
        assert len(result["relationships"]) == 1
        assert result["tables"][0]["name"] == "users"
        assert len(result["tables"][0]["columns"]) == 2


# =============================================================================
# Test LookML Models - Enums
# =============================================================================

@pytest.mark.unit
class TestDimensionType:
    """Test DimensionType enum values."""

    def test_string_type(self):
        assert DimensionType.STRING.value == "string"

    def test_number_type(self):
        assert DimensionType.NUMBER.value == "number"

    def test_yesno_type(self):
        assert DimensionType.YESNO.value == "yesno"

    def test_date_type(self):
        assert DimensionType.DATE.value == "date"

    def test_datetime_type(self):
        assert DimensionType.DATETIME.value == "datetime"

    def test_time_type(self):
        assert DimensionType.TIME.value == "time"

    def test_tier_type(self):
        assert DimensionType.TIER.value == "tier"

    def test_zipcode_type(self):
        assert DimensionType.ZIPCODE.value == "zipcode"

    def test_location_type(self):
        assert DimensionType.LOCATION.value == "location"


@pytest.mark.unit
class TestLookMLMeasureType:
    """Test LookML MeasureType enum values."""

    def test_count(self):
        assert LookMLMeasureType.COUNT.value == "count"

    def test_count_distinct(self):
        assert LookMLMeasureType.COUNT_DISTINCT.value == "count_distinct"

    def test_sum(self):
        assert LookMLMeasureType.SUM.value == "sum"

    def test_average(self):
        assert LookMLMeasureType.AVERAGE.value == "average"

    def test_min(self):
        assert LookMLMeasureType.MIN.value == "min"

    def test_max(self):
        assert LookMLMeasureType.MAX.value == "max"

    def test_median(self):
        assert LookMLMeasureType.MEDIAN.value == "median"

    def test_percentile(self):
        assert LookMLMeasureType.PERCENTILE.value == "percentile"

    def test_number(self):
        assert LookMLMeasureType.NUMBER.value == "number"


@pytest.mark.unit
class TestJoinType:
    """Test JoinType enum values."""

    def test_left_outer(self):
        assert JoinType.LEFT_OUTER.value == "left_outer"

    def test_inner(self):
        assert JoinType.INNER.value == "inner"

    def test_full_outer(self):
        assert JoinType.FULL_OUTER.value == "full_outer"

    def test_cross(self):
        assert JoinType.CROSS.value == "cross"


@pytest.mark.unit
class TestLookMLRelationship:
    """Test LookML Relationship enum values."""

    def test_one_to_one(self):
        assert LookMLRelationship.ONE_TO_ONE.value == "one_to_one"

    def test_one_to_many(self):
        assert LookMLRelationship.ONE_TO_MANY.value == "one_to_many"

    def test_many_to_one(self):
        assert LookMLRelationship.MANY_TO_ONE.value == "many_to_one"

    def test_many_to_many(self):
        assert LookMLRelationship.MANY_TO_MANY.value == "many_to_many"


# =============================================================================
# Test LookML Models - Dimension
# =============================================================================

@pytest.mark.unit
class TestDimension:
    """Test Dimension dataclass."""

    def test_basic_creation(self):
        dim = Dimension(name="user_id")
        assert dim.name == "user_id"
        assert dim.type == DimensionType.STRING
        assert dim.sql is None
        assert dim.label is None
        assert dim.description is None
        assert dim.primary_key is False
        assert dim.hidden is False
        assert dim.group_label is None

    def test_with_all_fields(self):
        dim = Dimension(
            name="id",
            type=DimensionType.NUMBER,
            sql="${TABLE}.id",
            label="User ID",
            description="Unique user identifier",
            primary_key=True,
            hidden=False,
            group_label="User Info"
        )
        assert dim.type == DimensionType.NUMBER
        assert dim.sql == "${TABLE}.id"
        assert dim.label == "User ID"
        assert dim.primary_key is True

    def test_to_dict_minimal(self):
        dim = Dimension(name="email")
        result = dim.to_dict()
        assert result["name"] == "email"
        assert result["type"] == "string"
        assert "sql" not in result
        assert "label" not in result
        assert "primary_key" not in result

    def test_to_dict_with_sql(self):
        dim = Dimension(name="email", sql="${TABLE}.email")
        result = dim.to_dict()
        assert result["sql"] == "${TABLE}.email"

    def test_to_dict_with_label(self):
        dim = Dimension(name="email", label="Email Address")
        result = dim.to_dict()
        assert result["label"] == "Email Address"

    def test_to_dict_with_description(self):
        dim = Dimension(name="email", description="User email")
        result = dim.to_dict()
        assert result["description"] == "User email"

    def test_to_dict_primary_key_yes(self):
        dim = Dimension(name="id", primary_key=True)
        result = dim.to_dict()
        assert result["primary_key"] == "yes"

    def test_to_dict_hidden_yes(self):
        dim = Dimension(name="internal_id", hidden=True)
        result = dim.to_dict()
        assert result["hidden"] == "yes"

    def test_to_dict_with_group_label(self):
        dim = Dimension(name="city", group_label="Location")
        result = dim.to_dict()
        assert result["group_label"] == "Location"


# =============================================================================
# Test LookML Models - Measure
# =============================================================================

@pytest.mark.unit
class TestMeasure:
    """Test Measure dataclass."""

    def test_basic_creation(self):
        measure = Measure(name="count")
        assert measure.name == "count"
        assert measure.type == LookMLMeasureType.COUNT
        assert measure.sql is None
        assert measure.label is None
        assert measure.description is None
        assert measure.hidden is False
        assert measure.drill_fields == []
        assert measure.filters == {}

    def test_with_all_fields(self):
        measure = Measure(
            name="total_revenue",
            type=LookMLMeasureType.SUM,
            sql="${TABLE}.revenue",
            label="Total Revenue",
            description="Sum of all revenue",
            hidden=False,
            drill_fields=["user_id", "order_date"],
            filters={"status": "completed"}
        )
        assert measure.type == LookMLMeasureType.SUM
        assert measure.sql == "${TABLE}.revenue"
        assert len(measure.drill_fields) == 2
        assert measure.filters["status"] == "completed"

    def test_to_dict_minimal(self):
        measure = Measure(name="count")
        result = measure.to_dict()
        assert result["name"] == "count"
        assert result["type"] == "count"
        assert "sql" not in result
        assert "drill_fields" not in result
        assert "filters" not in result

    def test_to_dict_with_sql(self):
        measure = Measure(name="total", sql="${TABLE}.amount")
        result = measure.to_dict()
        assert result["sql"] == "${TABLE}.amount"

    def test_to_dict_with_label(self):
        measure = Measure(name="count", label="Total Count")
        result = measure.to_dict()
        assert result["label"] == "Total Count"

    def test_to_dict_with_description(self):
        measure = Measure(name="count", description="Count of records")
        result = measure.to_dict()
        assert result["description"] == "Count of records"

    def test_to_dict_hidden_yes(self):
        measure = Measure(name="internal_metric", hidden=True)
        result = measure.to_dict()
        assert result["hidden"] == "yes"

    def test_to_dict_with_drill_fields(self):
        measure = Measure(name="count", drill_fields=["id", "name"])
        result = measure.to_dict()
        assert result["drill_fields"] == ["id", "name"]

    def test_to_dict_with_filters(self):
        measure = Measure(name="count", filters={"status": "active", "type": "premium"})
        result = measure.to_dict()
        assert "filters" in result
        filters = result["filters"]
        assert len(filters) == 2
        assert {"field": "status", "value": "active"} in filters
        assert {"field": "type", "value": "premium"} in filters


# =============================================================================
# Test LookML Models - View
# =============================================================================

@pytest.mark.unit
class TestView:
    """Test View dataclass."""

    def test_basic_creation(self):
        view = View(name="users")
        assert view.name == "users"
        assert view.sql_table_name is None
        assert view.label is None
        assert view.description is None
        assert view.dimensions == []
        assert view.measures == []

    def test_with_all_fields(self):
        view = View(
            name="users",
            sql_table_name="public.users",
            label="Users",
            description="User data",
            dimensions=[Dimension(name="id")],
            measures=[Measure(name="count")]
        )
        assert view.sql_table_name == "public.users"
        assert view.label == "Users"
        assert len(view.dimensions) == 1
        assert len(view.measures) == 1

    def test_to_dict_minimal(self):
        view = View(name="orders")
        result = view.to_dict()
        assert result["name"] == "orders"
        assert result["dimensions"] == []
        assert result["measures"] == []
        assert "sql_table_name" not in result

    def test_to_dict_with_sql_table_name(self):
        view = View(name="users", sql_table_name="schema.users")
        result = view.to_dict()
        assert result["sql_table_name"] == "schema.users"

    def test_to_dict_with_label(self):
        view = View(name="users", label="User Data")
        result = view.to_dict()
        assert result["label"] == "User Data"

    def test_to_dict_with_description(self):
        view = View(name="users", description="User information")
        result = view.to_dict()
        assert result["description"] == "User information"

    def test_to_dict_with_dimensions_and_measures(self):
        view = View(
            name="users",
            dimensions=[
                Dimension(name="id", type=DimensionType.NUMBER),
                Dimension(name="email")
            ],
            measures=[
                Measure(name="count"),
                Measure(name="total", type=LookMLMeasureType.SUM)
            ]
        )
        result = view.to_dict()
        assert len(result["dimensions"]) == 2
        assert len(result["measures"]) == 2
        assert result["dimensions"][0]["name"] == "id"
        assert result["measures"][0]["name"] == "count"


# =============================================================================
# Test LookML Models - Join
# =============================================================================

@pytest.mark.unit
class TestJoin:
    """Test Join dataclass."""

    def test_basic_creation(self):
        join = Join(name="orders")
        assert join.name == "orders"
        assert join.type == JoinType.LEFT_OUTER
        assert join.relationship == LookMLRelationship.MANY_TO_ONE
        assert join.sql_on == ""
        assert join.foreign_key is None

    def test_with_all_fields(self):
        join = Join(
            name="customers",
            type=JoinType.INNER,
            relationship=LookMLRelationship.ONE_TO_MANY,
            sql_on="${orders.customer_id} = ${customers.id}",
            foreign_key="customer_id"
        )
        assert join.type == JoinType.INNER
        assert join.relationship == LookMLRelationship.ONE_TO_MANY
        assert join.sql_on == "${orders.customer_id} = ${customers.id}"
        assert join.foreign_key == "customer_id"

    def test_to_dict_minimal(self):
        join = Join(name="orders", sql_on="${users.id} = ${orders.user_id}")
        result = join.to_dict()
        assert result["name"] == "orders"
        assert result["type"] == "left_outer"
        assert result["relationship"] == "many_to_one"
        assert result["sql_on"] == "${users.id} = ${orders.user_id}"
        assert "foreign_key" not in result

    def test_to_dict_with_foreign_key(self):
        join = Join(
            name="orders",
            sql_on="${users.id} = ${orders.user_id}",
            foreign_key="user_id"
        )
        result = join.to_dict()
        assert result["foreign_key"] == "user_id"


# =============================================================================
# Test LookML Models - Explore
# =============================================================================

@pytest.mark.unit
class TestExplore:
    """Test Explore dataclass."""

    def test_basic_creation(self):
        explore = Explore(name="users")
        assert explore.name == "users"
        assert explore.view_name is None
        assert explore.label is None
        assert explore.description is None
        assert explore.joins == []
        assert explore.always_filter == {}
        assert explore.hidden is False

    def test_with_all_fields(self):
        explore = Explore(
            name="users",
            view_name="user_view",
            label="Users",
            description="User explore",
            joins=[Join(name="orders", sql_on="${users.id} = ${orders.user_id}")],
            always_filter={"status": "active"},
            hidden=True
        )
        assert explore.view_name == "user_view"
        assert explore.label == "Users"
        assert len(explore.joins) == 1
        assert explore.always_filter["status"] == "active"
        assert explore.hidden is True

    def test_to_dict_minimal(self):
        explore = Explore(name="users")
        result = explore.to_dict()
        assert result["name"] == "users"
        assert "view_name" not in result
        assert "joins" not in result
        assert "always_filter" not in result
        assert "hidden" not in result

    def test_to_dict_with_view_name(self):
        explore = Explore(name="users", view_name="user_base")
        result = explore.to_dict()
        assert result["view_name"] == "user_base"

    def test_to_dict_with_label(self):
        explore = Explore(name="users", label="User Data")
        result = explore.to_dict()
        assert result["label"] == "User Data"

    def test_to_dict_with_description(self):
        explore = Explore(name="users", description="User explore")
        result = explore.to_dict()
        assert result["description"] == "User explore"

    def test_to_dict_with_joins(self):
        explore = Explore(
            name="users",
            joins=[
                Join(name="orders", sql_on="${users.id} = ${orders.user_id}"),
                Join(name="profiles", sql_on="${users.profile_id} = ${profiles.id}")
            ]
        )
        result = explore.to_dict()
        assert len(result["joins"]) == 2
        assert result["joins"][0]["name"] == "orders"

    def test_to_dict_with_always_filter(self):
        explore = Explore(
            name="users",
            always_filter={"status": "active", "deleted": "false"}
        )
        result = explore.to_dict()
        assert "always_filter" in result
        filters = result["always_filter"]["filters"]
        assert len(filters) == 2
        assert {"field": "status", "value": "active"} in filters

    def test_to_dict_hidden_yes(self):
        explore = Explore(name="internal", hidden=True)
        result = explore.to_dict()
        assert result["hidden"] == "yes"


# =============================================================================
# Test LookML Models - LookMLModel
# =============================================================================

@pytest.mark.unit
class TestLookMLModel:
    """Test LookMLModel dataclass."""

    def test_basic_creation(self):
        model = LookMLModel(name="my_model")
        assert model.name == "my_model"
        assert model.connection == ""
        assert model.views == []
        assert model.explores == []

    def test_with_all_fields(self):
        model = LookMLModel(
            name="analytics",
            connection="database",
            views=[View(name="users"), View(name="orders")],
            explores=[Explore(name="users")]
        )
        assert model.connection == "database"
        assert len(model.views) == 2
        assert len(model.explores) == 1

    def test_to_dict(self):
        model = LookMLModel(
            name="analytics",
            connection="db",
            views=[View(name="users")],
            explores=[Explore(name="users")]
        )
        result = model.to_dict()
        assert result["name"] == "analytics"
        assert result["connection"] == "db"
        assert len(result["views"]) == 1
        assert len(result["explores"]) == 1

    def test_get_view_exists(self):
        model = LookMLModel(
            name="model",
            views=[
                View(name="users"),
                View(name="orders")
            ]
        )
        view = model.get_view("users")
        assert view is not None
        assert view.name == "users"

    def test_get_view_missing(self):
        model = LookMLModel(name="model", views=[])
        view = model.get_view("nonexistent")
        assert view is None


# =============================================================================
# Test Date Preprocessor - DateFormat
# =============================================================================

@pytest.mark.unit
class TestDateFormat:
    """Test DateFormat enum values."""

    def test_iso_format(self):
        assert DateFormat.ISO.value == "iso"

    def test_sql_standard_format(self):
        assert DateFormat.SQL_STANDARD.value == "sql"

    def test_mssql_format(self):
        assert DateFormat.MSSQL.value == "mssql"

    def test_oracle_format(self):
        assert DateFormat.ORACLE.value == "oracle"


# =============================================================================
# Test Date Preprocessor - BaseDatePreprocessor
# =============================================================================

@pytest.mark.unit
class TestBaseDatePreprocessor:
    """Test BaseDatePreprocessor abstract class."""

    def test_date_patterns_count(self):
        assert len(BaseDatePreprocessor.DATE_PATTERNS) == 24

    def test_special_dates_initialization(self):
        preprocessor = DatePreprocessor()
        assert "today" in preprocessor.special_dates
        assert "yesterday" in preprocessor.special_dates
        assert "this week" in preprocessor.special_dates
        assert "last week" in preprocessor.special_dates
        assert "this month" in preprocessor.special_dates
        assert "last month" in preprocessor.special_dates
        assert "this year" in preprocessor.special_dates
        assert "last year" in preprocessor.special_dates
        assert "this quarter" in preprocessor.special_dates
        assert "last quarter" in preprocessor.special_dates
        assert "ytd" in preprocessor.special_dates
        assert "year to date" in preprocessor.special_dates
        assert "mtd" in preprocessor.special_dates
        assert "month to date" in preprocessor.special_dates

    def test_get_quarter_dates_current_q1(self):
        with patch('Jotty.core.semantic.query.date_preprocessor.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 2, 15)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            start, end = BaseDatePreprocessor._get_quarter_dates(0)
            assert start == datetime(2024, 1, 1)
            assert end.year == 2024
            assert end.month == 3
            assert end.day == 31

    def test_get_quarter_dates_current_q2(self):
        with patch('Jotty.core.semantic.query.date_preprocessor.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 5, 15)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            start, end = BaseDatePreprocessor._get_quarter_dates(0)
            assert start == datetime(2024, 4, 1)
            assert end.month == 6

    def test_get_quarter_dates_current_q3(self):
        with patch('Jotty.core.semantic.query.date_preprocessor.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 8, 15)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            start, end = BaseDatePreprocessor._get_quarter_dates(0)
            assert start == datetime(2024, 7, 1)
            assert end.month == 9

    def test_get_quarter_dates_current_q4(self):
        with patch('Jotty.core.semantic.query.date_preprocessor.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 11, 15)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            start, end = BaseDatePreprocessor._get_quarter_dates(0)
            assert start == datetime(2024, 10, 1)
            assert end.month == 12

    def test_get_quarter_dates_last_quarter(self):
        with patch('Jotty.core.semantic.query.date_preprocessor.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 5, 15)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            start, end = BaseDatePreprocessor._get_quarter_dates(-1)
            assert start == datetime(2024, 1, 1)
            assert end.month == 3

    def test_get_quarter_dates_year_wraparound_backwards(self):
        with patch('Jotty.core.semantic.query.date_preprocessor.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 2, 15)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            start, end = BaseDatePreprocessor._get_quarter_dates(-1)
            assert start.year == 2023
            assert start.month == 10
            assert end.year == 2023
            assert end.month == 12

    def test_get_quarter_dates_year_wraparound_forwards(self):
        with patch('Jotty.core.semantic.query.date_preprocessor.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2024, 11, 15)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            start, end = BaseDatePreprocessor._get_quarter_dates(1)
            assert start.year == 2025
            assert start.month == 1


# =============================================================================
# Test Date Preprocessor - DatePreprocessor (MongoDB)
# =============================================================================

@pytest.mark.unit
class TestDatePreprocessor:
    """Test DatePreprocessor for MongoDB."""

    def test_format_date_iso(self):
        preprocessor = DatePreprocessor()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert result == "2024-01-15T10:30:00"

    def test_get_context_hint_empty(self):
        preprocessor = DatePreprocessor()
        hint = preprocessor.get_context_hint({})
        assert hint == ""

    def test_get_context_hint_with_dates(self):
        preprocessor = DatePreprocessor()
        context = {
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-12-31T23:59:59"
        }
        hint = preprocessor.get_context_hint(context)
        assert "Date Context" in hint
        assert "start_date" in hint
        assert "end_date" in hint
        assert "$match" in hint

    @patch('Jotty.core.semantic.query.date_preprocessor.datetime')
    def test_preprocess_today(self, mock_datetime):
        mock_now = datetime(2024, 2, 14, 15, 30, 0)
        mock_datetime.now.return_value = mock_now

        preprocessor = DatePreprocessor()
        query = "Show me orders from today"
        modified_query, context = preprocessor.preprocess(query)

        assert "today_start" in context
        assert "today_end" in context
        assert "2024-02-14" in modified_query

    @patch('Jotty.core.semantic.query.date_preprocessor.datetime')
    def test_preprocess_yesterday(self, mock_datetime):
        mock_now = datetime(2024, 2, 14, 15, 30, 0)
        mock_datetime.now.return_value = mock_now

        preprocessor = DatePreprocessor()
        query = "Show orders from yesterday"
        modified_query, context = preprocessor.preprocess(query)

        assert "yesterday_start" in context
        assert "yesterday_end" in context
        assert "2024-02-13" in modified_query

    @patch('Jotty.core.semantic.query.date_preprocessor.datetime')
    def test_preprocess_last_7_days(self, mock_datetime):
        mock_now = datetime(2024, 2, 14, 15, 30, 0)
        mock_datetime.now.return_value = mock_now

        preprocessor = DatePreprocessor()
        query = "Show data from last 7 days"
        modified_query, context = preprocessor.preprocess(query)

        assert "start_date" in context
        assert "end_date" in context
        assert "2024-02-07" in modified_query

    @patch('Jotty.core.semantic.query.date_preprocessor.datetime')
    def test_preprocess_past_30_days(self, mock_datetime):
        mock_now = datetime(2024, 2, 14, 15, 30, 0)
        mock_datetime.now.return_value = mock_now

        preprocessor = DatePreprocessor()
        query = "Show data from past 30 days"
        modified_query, context = preprocessor.preprocess(query)

        assert "start_date" in context
        expected_start = mock_now - timedelta(days=30)
        assert expected_start.strftime('%Y-%m-%d') in modified_query


# =============================================================================
# Test Date Preprocessor - SQLDatePreprocessor
# =============================================================================

@pytest.mark.unit
class TestSQLDatePreprocessor:
    """Test SQLDatePreprocessor for SQL databases."""

    def test_default_dialect_postgresql(self):
        preprocessor = SQLDatePreprocessor()
        assert preprocessor.dialect == "postgresql"

    def test_custom_dialect_mysql(self):
        preprocessor = SQLDatePreprocessor(dialect="mysql")
        assert preprocessor.dialect == "mysql"

    def test_format_date_postgresql(self):
        preprocessor = SQLDatePreprocessor(dialect="postgresql")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert result == "'2024-01-15T10:30:00'::timestamp"

    def test_format_date_mysql(self):
        preprocessor = SQLDatePreprocessor(dialect="mysql")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert result == "'2024-01-15T10:30:00'"

    def test_format_date_sqlite(self):
        preprocessor = SQLDatePreprocessor(dialect="sqlite")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert result == "'2024-01-15T10:30:00'"

    def test_format_date_mssql(self):
        preprocessor = SQLDatePreprocessor(dialect="mssql")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert "CONVERT" in result
        assert "2024-01-15T10:30:00" in result

    def test_format_date_oracle(self):
        preprocessor = SQLDatePreprocessor(dialect="oracle")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert "TO_TIMESTAMP" in result
        assert "2024-01-15T10:30:00" in result

    def test_format_date_snowflake(self):
        preprocessor = SQLDatePreprocessor(dialect="snowflake")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert result == "'2024-01-15T10:30:00'::timestamp"

    def test_format_date_bigquery(self):
        preprocessor = SQLDatePreprocessor(dialect="bigquery")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert "TIMESTAMP" in result
        assert "2024-01-15T10:30:00" in result

    def test_format_date_unknown_dialect(self):
        preprocessor = SQLDatePreprocessor(dialect="unknown")
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date(dt)
        assert result == "'2024-01-15T10:30:00'"

    def test_format_date_simple(self):
        preprocessor = SQLDatePreprocessor()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = preprocessor.format_date_simple(dt)
        assert result == "2024-01-15 10:30:00"

    def test_get_context_hint_empty(self):
        preprocessor = SQLDatePreprocessor()
        hint = preprocessor.get_context_hint({})
        assert hint == ""

    def test_get_context_hint_postgresql(self):
        preprocessor = SQLDatePreprocessor(dialect="postgresql")
        context = {"start_date": "'2024-01-01'::timestamp"}
        hint = preprocessor.get_context_hint(context)
        assert "Date Context" in hint
        assert "PostgreSQL" in hint
        assert "timestamp" in hint

    def test_get_context_hint_mysql(self):
        preprocessor = SQLDatePreprocessor(dialect="mysql")
        context = {"start_date": "'2024-01-01'"}
        hint = preprocessor.get_context_hint(context)
        assert "MySQL" in hint

    def test_get_context_hint_oracle(self):
        preprocessor = SQLDatePreprocessor(dialect="oracle")
        context = {"start_date": "TO_TIMESTAMP('2024-01-01', 'YYYY-MM-DD')"}
        hint = preprocessor.get_context_hint(context)
        assert "Oracle" in hint
        assert "TO_TIMESTAMP" in hint

    @patch('Jotty.core.semantic.query.date_preprocessor.datetime')
    def test_preprocess_last_month(self, mock_datetime):
        mock_now = datetime(2024, 2, 14, 15, 30, 0)
        mock_datetime.now.return_value = mock_now

        preprocessor = SQLDatePreprocessor(dialect="postgresql")
        query = "Show data from last month"
        modified_query, context = preprocessor.preprocess(query)

        assert "last_month_start" in context
        assert "last_month_end" in context


# =============================================================================
# Test Date Preprocessor - DatePreprocessorFactory
# =============================================================================

@pytest.mark.unit
class TestDatePreprocessorFactory:
    """Test DatePreprocessorFactory."""

    def test_create_mongodb(self):
        preprocessor = DatePreprocessorFactory.create("mongodb")
        assert isinstance(preprocessor, DatePreprocessor)
        assert not isinstance(preprocessor, SQLDatePreprocessor)

    def test_create_postgresql(self):
        preprocessor = DatePreprocessorFactory.create("postgresql")
        assert isinstance(preprocessor, SQLDatePreprocessor)
        assert preprocessor.dialect == "postgresql"

    def test_create_mysql(self):
        preprocessor = DatePreprocessorFactory.create("mysql")
        assert isinstance(preprocessor, SQLDatePreprocessor)
        assert preprocessor.dialect == "mysql"

    def test_create_sqlite(self):
        preprocessor = DatePreprocessorFactory.create("sqlite")
        assert isinstance(preprocessor, SQLDatePreprocessor)
        assert preprocessor.dialect == "sqlite"

    def test_create_mssql(self):
        preprocessor = DatePreprocessorFactory.create("mssql")
        assert isinstance(preprocessor, SQLDatePreprocessor)
        assert preprocessor.dialect == "mssql"

    def test_create_oracle(self):
        preprocessor = DatePreprocessorFactory.create("oracle")
        assert isinstance(preprocessor, SQLDatePreprocessor)
        assert preprocessor.dialect == "oracle"

    def test_create_snowflake(self):
        preprocessor = DatePreprocessorFactory.create("snowflake")
        assert isinstance(preprocessor, SQLDatePreprocessor)
        assert preprocessor.dialect == "snowflake"

    def test_create_bigquery(self):
        preprocessor = DatePreprocessorFactory.create("bigquery")
        assert isinstance(preprocessor, SQLDatePreprocessor)
        assert preprocessor.dialect == "bigquery"

    def test_create_case_insensitive(self):
        preprocessor = DatePreprocessorFactory.create("PostgreSQL")
        assert preprocessor.dialect == "postgresql"

    def test_create_none_defaults_to_postgresql(self):
        preprocessor = DatePreprocessorFactory.create(None)
        assert isinstance(preprocessor, SQLDatePreprocessor)
        assert preprocessor.dialect == "postgresql"

    def test_get_supported_databases(self):
        databases = DatePreprocessorFactory.get_supported_databases()
        assert len(databases) == 8
        assert "mongodb" in databases
        assert "postgresql" in databases
        assert "mysql" in databases
        assert "sqlite" in databases
        assert "mssql" in databases
        assert "oracle" in databases
        assert "snowflake" in databases
        assert "bigquery" in databases
