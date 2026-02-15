"""
LookML Data Models

Represents LookML constructs: views, explores, dimensions, measures, joins.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DimensionType(Enum):
    """LookML dimension types."""

    STRING = "string"
    NUMBER = "number"
    YESNO = "yesno"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"
    TIER = "tier"
    ZIPCODE = "zipcode"
    LOCATION = "location"


class MeasureType(Enum):
    """LookML measure types."""

    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PERCENTILE = "percentile"
    NUMBER = "number"


class JoinType(Enum):
    """LookML join types."""

    LEFT_OUTER = "left_outer"
    INNER = "inner"
    FULL_OUTER = "full_outer"
    CROSS = "cross"


class Relationship(Enum):
    """LookML join relationships."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


@dataclass
class Dimension:
    """LookML dimension."""

    name: str
    type: DimensionType = DimensionType.STRING
    sql: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    primary_key: bool = False
    hidden: bool = False
    group_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LookML-compatible dictionary."""
        d = {
            "name": self.name,
            "type": self.type.value,
        }
        if self.sql:
            d["sql"] = self.sql
        if self.label:
            d["label"] = self.label
        if self.description:
            d["description"] = self.description
        if self.primary_key:
            d["primary_key"] = "yes"
        if self.hidden:
            d["hidden"] = "yes"
        if self.group_label:
            d["group_label"] = self.group_label
        return d


@dataclass
class Measure:
    """LookML measure."""

    name: str
    type: MeasureType = MeasureType.COUNT
    sql: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    hidden: bool = False
    drill_fields: List[str] = field(default_factory=list)
    filters: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LookML-compatible dictionary."""
        d = {
            "name": self.name,
            "type": self.type.value,
        }
        if self.sql:
            d["sql"] = self.sql
        if self.label:
            d["label"] = self.label
        if self.description:
            d["description"] = self.description
        if self.hidden:
            d["hidden"] = "yes"
        if self.drill_fields:
            d["drill_fields"] = self.drill_fields
        if self.filters:
            d["filters"] = [{"field": k, "value": v} for k, v in self.filters.items()]
        return d


@dataclass
class View:
    """LookML view representing a table."""

    name: str
    sql_table_name: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    dimensions: List[Dimension] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LookML-compatible dictionary."""
        d = {"name": self.name}
        if self.sql_table_name:
            d["sql_table_name"] = self.sql_table_name
        if self.label:
            d["label"] = self.label
        if self.description:
            d["description"] = self.description

        # Add dimensions and measures
        d["dimensions"] = [dim.to_dict() for dim in self.dimensions]
        d["measures"] = [m.to_dict() for m in self.measures]

        return d


@dataclass
class Join:
    """LookML join definition."""

    name: str
    type: JoinType = JoinType.LEFT_OUTER
    relationship: Relationship = Relationship.MANY_TO_ONE
    sql_on: str = ""
    foreign_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LookML-compatible dictionary."""
        d = {
            "name": self.name,
            "type": self.type.value,
            "relationship": self.relationship.value,
            "sql_on": self.sql_on,
        }
        if self.foreign_key:
            d["foreign_key"] = self.foreign_key
        return d


@dataclass
class Explore:
    """LookML explore representing a queryable dataset."""

    name: str
    view_name: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    joins: List[Join] = field(default_factory=list)
    always_filter: Dict[str, str] = field(default_factory=dict)
    hidden: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LookML-compatible dictionary."""
        d = {"name": self.name}
        if self.view_name:
            d["view_name"] = self.view_name
        if self.label:
            d["label"] = self.label
        if self.description:
            d["description"] = self.description
        if self.joins:
            d["joins"] = [j.to_dict() for j in self.joins]
        if self.always_filter:
            d["always_filter"] = {
                "filters": [{"field": k, "value": v} for k, v in self.always_filter.items()]
            }
        if self.hidden:
            d["hidden"] = "yes"
        return d


@dataclass
class LookMLModel:
    """Complete LookML model with views and explores."""

    name: str
    connection: str = ""
    views: List[View] = field(default_factory=list)
    explores: List[Explore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to LookML-compatible dictionary."""
        return {
            "name": self.name,
            "connection": self.connection,
            "views": [v.to_dict() for v in self.views],
            "explores": [e.to_dict() for e in self.explores],
        }

    def get_view(self, name: str) -> Optional[View]:
        """Get view by name."""
        for view in self.views:
            if view.name == name:
                return view
        return None
