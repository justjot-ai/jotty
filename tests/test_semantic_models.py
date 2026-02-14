"""
Tests for core/semantic/models.py
==================================
Covers: ColumnType, RelationType, MeasureType enums,
        Column, ForeignKey, Index, Table, Relationship, Schema dataclasses.
"""
import pytest

from Jotty.core.semantic.models import (
    ColumnType, RelationType, MeasureType,
    Column, ForeignKey, Index, Table, Relationship, Schema,
)


# ===========================================================================
# Enum Tests
# ===========================================================================

@pytest.mark.unit
class TestColumnTypeEnum:
    """Tests for ColumnType enum."""

    def test_all_values(self):
        expected = {
            "string", "number", "integer", "float", "decimal",
            "boolean", "date", "datetime", "time", "timestamp",
            "json", "binary", "unknown",
        }
        actual = {ct.value for ct in ColumnType}
        assert expected == actual

    def test_lookup_by_value(self):
        assert ColumnType("string") is ColumnType.STRING
        assert ColumnType("unknown") is ColumnType.UNKNOWN


@pytest.mark.unit
class TestRelationTypeEnum:
    """Tests for RelationType enum."""

    def test_values(self):
        assert RelationType.ONE_TO_ONE.value == "one_to_one"
        assert RelationType.ONE_TO_MANY.value == "one_to_many"
        assert RelationType.MANY_TO_ONE.value == "many_to_one"
        assert RelationType.MANY_TO_MANY.value == "many_to_many"


@pytest.mark.unit
class TestMeasureTypeEnum:
    """Tests for MeasureType enum."""

    def test_values(self):
        assert MeasureType.COUNT.value == "count"
        assert MeasureType.SUM.value == "sum"
        assert MeasureType.AVERAGE.value == "average"
        assert MeasureType.MIN.value == "min"
        assert MeasureType.MAX.value == "max"
        assert MeasureType.MEDIAN.value == "median"
        assert MeasureType.COUNT_DISTINCT.value == "count_distinct"


# ===========================================================================
# Column Tests
# ===========================================================================

@pytest.mark.unit
class TestColumnNormalization:
    """Tests for Column type normalization."""

    def test_varchar_normalizes_to_string(self):
        col = Column(name="name", data_type="varchar(255)")
        assert col.normalized_type == ColumnType.STRING

    def test_int_normalizes_to_integer(self):
        col = Column(name="id", data_type="int")
        assert col.normalized_type == ColumnType.INTEGER

    def test_bigint_normalizes_to_integer(self):
        col = Column(name="big", data_type="bigint")
        assert col.normalized_type == ColumnType.INTEGER

    def test_float_normalizes_to_float(self):
        col = Column(name="rate", data_type="float")
        assert col.normalized_type == ColumnType.FLOAT

    def test_decimal_normalizes_to_decimal(self):
        col = Column(name="price", data_type="decimal(10,2)")
        assert col.normalized_type == ColumnType.DECIMAL

    def test_boolean_normalizes(self):
        col = Column(name="active", data_type="boolean")
        assert col.normalized_type == ColumnType.BOOLEAN

    def test_date_normalizes(self):
        col = Column(name="dob", data_type="date")
        assert col.normalized_type == ColumnType.DATE

    def test_datetime_normalizes(self):
        """'datetime' normalizes to DATE (pattern matches 'date' prefix first)."""
        col = Column(name="created", data_type="datetime")
        assert col.normalized_type == ColumnType.DATE

    def test_timestamp_normalizes(self):
        col = Column(name="ts", data_type="timestamp")
        assert col.normalized_type == ColumnType.TIMESTAMP

    def test_json_normalizes(self):
        col = Column(name="data", data_type="jsonb")
        assert col.normalized_type == ColumnType.JSON

    def test_binary_normalizes(self):
        col = Column(name="img", data_type="blob")
        assert col.normalized_type == ColumnType.BINARY

    def test_unknown_type_defaults_to_string(self):
        col = Column(name="x", data_type="somecustomtype")
        assert col.normalized_type == ColumnType.STRING

    def test_explicit_type_not_overridden(self):
        """Explicit normalized_type is preserved."""
        col = Column(name="x", data_type="varchar", normalized_type=ColumnType.JSON)
        assert col.normalized_type == ColumnType.JSON

    def test_mongodb_types(self):
        """MongoDB-specific types normalize correctly."""
        col = Column(name="oid", data_type="objectid")
        assert col.normalized_type == ColumnType.STRING

        col2 = Column(name="arr", data_type="array")
        assert col2.normalized_type == ColumnType.JSON


@pytest.mark.unit
class TestColumnSemantics:
    """Tests for Column semantic detection."""

    def test_amount_detected_as_measure(self):
        col = Column(name="total_amount", data_type="decimal(10,2)")
        assert col.is_measure is True
        assert col.is_dimension is False
        assert col.measure_type == MeasureType.SUM

    def test_price_detected_as_measure(self):
        col = Column(name="unit_price", data_type="float")
        assert col.is_measure is True

    def test_revenue_detected_as_measure(self):
        col = Column(name="monthly_revenue", data_type="decimal(12,2)")
        assert col.is_measure is True

    def test_id_column_is_dimension(self):
        col = Column(name="user_id", data_type="int")
        assert col.is_dimension is True
        assert col.is_measure is False

    def test_primary_key_is_dimension(self):
        col = Column(name="id", data_type="int", primary_key=True)
        assert col.is_dimension is True
        assert col.is_measure is False

    def test_string_column_is_dimension(self):
        col = Column(name="status", data_type="varchar(50)")
        assert col.is_dimension is True
        assert col.is_measure is False

    def test_integer_non_measure_is_dimension(self):
        """Integer column without measure pattern stays as dimension."""
        col = Column(name="version", data_type="int")
        assert col.is_dimension is True

    def test_defaults(self):
        """Column defaults are sensible."""
        col = Column(name="test", data_type="varchar")
        assert col.nullable is True
        assert col.primary_key is False
        assert col.unique is False
        assert col.default is None
        assert col.description is None
        assert col.hidden is False
        assert col.label is None


# ===========================================================================
# ForeignKey / Index Tests
# ===========================================================================

@pytest.mark.unit
class TestForeignKeyAndIndex:
    """Tests for ForeignKey and Index dataclasses."""

    def test_foreign_key_creation(self):
        fk = ForeignKey(
            columns=["user_id"],
            referenced_table="users",
            referenced_columns=["id"],
            constraint_name="fk_users",
        )
        assert fk.columns == ["user_id"]
        assert fk.referenced_table == "users"
        assert fk.constraint_name == "fk_users"

    def test_foreign_key_no_constraint_name(self):
        fk = ForeignKey(columns=["id"], referenced_table="other", referenced_columns=["id"])
        assert fk.constraint_name is None

    def test_index_creation(self):
        idx = Index(name="idx_users_email", columns=["email"], unique=True)
        assert idx.name == "idx_users_email"
        assert idx.columns == ["email"]
        assert idx.unique is True

    def test_index_defaults(self):
        idx = Index(name="idx_name", columns=["name"])
        assert idx.unique is False


# ===========================================================================
# Table Tests
# ===========================================================================

@pytest.mark.unit
class TestTable:
    """Tests for Table dataclass."""

    def test_full_name_with_schema(self):
        t = Table(name="users", schema="public")
        assert t.full_name == "public.users"

    def test_full_name_without_schema(self):
        t = Table(name="users")
        assert t.full_name == "users"

    def test_get_column_found(self):
        cols = [Column(name="id", data_type="int"), Column(name="name", data_type="varchar")]
        t = Table(name="users", columns=cols)
        col = t.get_column("name")
        assert col is not None
        assert col.name == "name"

    def test_get_column_case_insensitive(self):
        cols = [Column(name="Email", data_type="varchar")]
        t = Table(name="users", columns=cols)
        col = t.get_column("email")
        assert col is not None
        assert col.name == "Email"

    def test_get_column_not_found(self):
        t = Table(name="users", columns=[])
        assert t.get_column("nonexistent") is None

    def test_dimensions_property(self):
        cols = [
            Column(name="id", data_type="int"),
            Column(name="total_amount", data_type="decimal(10,2)"),
            Column(name="status", data_type="varchar"),
        ]
        t = Table(name="orders", columns=cols)
        dims = t.dimensions
        dim_names = [d.name for d in dims]
        assert "id" in dim_names
        assert "status" in dim_names

    def test_measures_property(self):
        cols = [
            Column(name="id", data_type="int"),
            Column(name="total_amount", data_type="decimal(10,2)"),
        ]
        t = Table(name="orders", columns=cols)
        measures = t.measures
        assert len(measures) >= 1
        assert any(m.name == "total_amount" for m in measures)

    def test_defaults(self):
        t = Table(name="test")
        assert t.schema is None
        assert t.columns == []
        assert t.primary_keys == []
        assert t.foreign_keys == []
        assert t.indexes == []
        assert t.description is None
        assert t.row_count is None
        assert t.label is None


# ===========================================================================
# Relationship Tests
# ===========================================================================

@pytest.mark.unit
class TestRelationship:
    """Tests for Relationship dataclass."""

    def test_sql_on_single_column(self):
        r = Relationship(
            from_table="orders",
            from_columns=["user_id"],
            to_table="users",
            to_columns=["id"],
        )
        assert r.sql_on == "orders.user_id = users.id"

    def test_sql_on_composite_key(self):
        r = Relationship(
            from_table="order_items",
            from_columns=["order_id", "product_id"],
            to_table="products",
            to_columns=["order_id", "id"],
        )
        assert "order_items.order_id = products.order_id" in r.sql_on
        assert "order_items.product_id = products.id" in r.sql_on
        assert " AND " in r.sql_on

    def test_defaults(self):
        r = Relationship(
            from_table="a", from_columns=["id"],
            to_table="b", to_columns=["a_id"],
        )
        assert r.relation_type == RelationType.MANY_TO_ONE
        assert r.join_type == "left_outer"


# ===========================================================================
# Schema Tests
# ===========================================================================

@pytest.mark.unit
class TestSchema:
    """Tests for Schema dataclass."""

    def _make_schema(self):
        users = Table(
            name="users",
            columns=[Column(name="id", data_type="int", primary_key=True)],
            primary_keys=["id"],
        )
        orders = Table(
            name="orders",
            columns=[
                Column(name="id", data_type="int", primary_key=True),
                Column(name="user_id", data_type="int"),
            ],
            primary_keys=["id"],
            foreign_keys=[ForeignKey(columns=["user_id"], referenced_table="users", referenced_columns=["id"])],
        )
        return Schema(name="test_db", tables=[users, orders], database_type="postgres")

    def test_get_table_found(self):
        s = self._make_schema()
        t = s.get_table("users")
        assert t is not None
        assert t.name == "users"

    def test_get_table_case_insensitive(self):
        s = self._make_schema()
        t = s.get_table("USERS")
        assert t is not None

    def test_get_table_not_found(self):
        s = self._make_schema()
        assert s.get_table("nonexistent") is None

    def test_get_table_by_full_name(self):
        t = Table(name="users", schema="public")
        s = Schema(name="db", tables=[t])
        found = s.get_table("public.users")
        assert found is not None

    def test_infer_relationships(self):
        """infer_relationships creates relationships from foreign keys."""
        s = self._make_schema()
        assert len(s.relationships) == 0
        s.infer_relationships()
        assert len(s.relationships) == 1
        r = s.relationships[0]
        assert r.from_table == "orders"
        assert r.to_table == "users"
        assert r.relation_type == RelationType.MANY_TO_ONE

    def test_infer_relationships_no_duplicates(self):
        """infer_relationships doesn't create duplicates on repeat call."""
        s = self._make_schema()
        s.infer_relationships()
        s.infer_relationships()
        assert len(s.relationships) == 1

    def test_infer_one_to_one(self):
        """FK on PK columns creates ONE_TO_ONE relationship."""
        t1 = Table(name="users", primary_keys=["id"],
                   columns=[Column(name="id", data_type="int", primary_key=True)])
        t2 = Table(name="profiles", primary_keys=["user_id"],
                   columns=[Column(name="user_id", data_type="int", primary_key=True)],
                   foreign_keys=[ForeignKey(columns=["user_id"], referenced_table="users", referenced_columns=["id"])])
        s = Schema(name="db", tables=[t1, t2])
        s.infer_relationships()
        assert s.relationships[0].relation_type == RelationType.ONE_TO_ONE

    def test_to_dict(self):
        """to_dict returns structured dict."""
        s = self._make_schema()
        s.infer_relationships()
        d = s.to_dict()
        assert d["name"] == "test_db"
        assert d["database_type"] == "postgres"
        assert len(d["tables"]) == 2
        assert len(d["relationships"]) == 1
        assert d["version"] == "1.0"

    def test_to_dict_table_columns(self):
        """to_dict includes column details."""
        s = self._make_schema()
        d = s.to_dict()
        users_dict = d["tables"][0]
        assert users_dict["name"] == "users"
        assert len(users_dict["columns"]) == 1
        col = users_dict["columns"][0]
        assert col["name"] == "id"
        assert col["primary_key"] is True

    def test_defaults(self):
        s = Schema(name="db")
        assert s.tables == []
        assert s.relationships == []
        assert s.database_type == "unknown"
        assert s.extracted_at is None
        assert s.version == "1.0"
